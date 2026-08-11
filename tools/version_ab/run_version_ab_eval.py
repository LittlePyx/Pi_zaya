from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = ROOT / "docs" / "version_ab_eval_v1.json"
DEFAULT_OUT_ROOT = ROOT / "test_results" / "version_ab"


def _records(value: object) -> list[dict[str, Any]]:
    return [item for item in list(value or []) if isinstance(item, dict)]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def load_contract(path: Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    payload = _load_json(path)
    if int(payload.get("version") or 0) != 1:
        raise ValueError("version A/B fixture version must be 1")
    for side in ("baseline", "candidate"):
        version = dict(payload.get(side) or {})
        if len(str(version.get("sha") or "")) != 40 or not str(version.get("label") or "").strip():
            raise ValueError(f"{side} must define a label and full 40-character SHA")
    qa = dict(payload.get("qa") or {})
    suites = _records(qa.get("suites"))
    if [(item.get("suite"), int(item.get("expected_cases") or 0)) for item in suites] != [
        ("full_library_acceptance_v1", 29),
        ("live_smoke_v1", 5),
    ]:
        raise ValueError("version A/B fixture must preserve the fixed 29-case full suite and 5-case smoke suite")
    if float(qa.get("case_timeout_s") or 0) < 30:
        raise ValueError("case timeout must preserve at least the 30-second evidence-card quality budget")
    if int(qa.get("top_k") or 0) <= 0 or int(qa.get("max_tokens") or 0) <= 0:
        raise ValueError("QA retrieval and generation settings must remain positive")
    project = dict(payload.get("project_journeys") or {})
    if int(project.get("runs") or 0) != 3:
        raise ValueError("version A/B fixture must preserve three complete project journeys")
    if not list(project.get("required_paths") or []):
        raise ValueError("project journey capability paths must be explicit")
    return payload


def corpus_fingerprint(db_root: Path, *, exclude_top_level: set[str] | None = None) -> dict[str, Any]:
    root = db_root.resolve(strict=True)
    excluded = {str(item).casefold() for item in (exclude_top_level or set())}
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not (
            path.relative_to(root).parts
            and path.relative_to(root).parts[0].casefold() in excluded
        )
    ]
    files.sort(key=lambda item: item.relative_to(root).as_posix().casefold())
    digest = hashlib.sha256()
    total_bytes = 0
    identity_hashes: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                file_digest.update(chunk)
        if relative.casefold() in {"docs.json", "references_index.json"}:
            identity_hashes[relative] = file_digest.hexdigest()
    return {
        "root": str(root),
        "excluded_top_level": sorted(excluded),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "identity_hashes": identity_hashes,
    }


def _openapi_paths(base_url: str, timeout_s: float = 10.0) -> set[str]:
    with request.urlopen(f"{base_url.rstrip('/')}/openapi.json", timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {str(item) for item in dict(payload.get("paths") or {})}


def _git_state(repo: Path) -> dict[str, Any]:
    sha = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"],
        text=True,
        encoding="utf-8",
    ).splitlines()
    return {"repo": str(repo.resolve()), "sha": sha, "dirty_paths": status}


def _latest_file(root: Path, name: str) -> Path | None:
    matches = sorted(root.rglob(name), key=lambda item: item.stat().st_mtime)
    return matches[-1] if matches else None


def _qa_failure_summary(raw_path: Path | None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    failures: list[dict[str, Any]] = []
    buckets: Counter[str] = Counter()
    if raw_path is None or not raw_path.is_file():
        return failures, {}
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        if bool(quality.get("ok")):
            continue
        names = [
            str(item.get("name") or "unknown")
            for item in _records(quality.get("failures"))
        ]
        if not names:
            names = [str(row.get("error_type") or "unknown")]
        buckets.update(names)
        failures.append(
            {
                "id": str(row.get("id") or ""),
                "status": str(row.get("status") or ""),
                "error": str(row.get("error") or ""),
                "failure_names": names,
                "latency_ms": row.get("latency_ms"),
                "case_wall_ms": row.get("case_wall_ms"),
            }
        )
    return failures, dict(sorted(buckets.items()))


def _summary_count(summary: dict[str, Any], key: str, default: int) -> int:
    value = summary.get(key)
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _run_qa_suite(
    *,
    side: str,
    suite: dict[str, Any],
    qa: dict[str, Any],
    runner_repo: Path,
    base_url: str,
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    suite_root = output_root / side / "qa" / str(suite["name"])
    command = [
        sys.executable,
        str(runner_repo / "tools" / "research_qa" / "run_research_qa_eval.py"),
        "--fixture",
        str(runner_repo / str(qa["fixture"])),
        "--base-url",
        base_url,
        "--source-root",
        str(source_root),
        "--suite",
        str(suite["suite"]),
        "--timeout-s",
        str(qa["timeout_s"]),
        "--case-timeout-s",
        str(qa["case_timeout_s"]),
        "--top-k",
        str(qa["top_k"]),
        "--max-tokens",
        str(qa["max_tokens"]),
        "--out-dir",
        str(suite_root),
        "--fail-on-quality",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=runner_repo, check=False)
    elapsed_s = round(time.perf_counter() - started, 3)
    summary_path = _latest_file(suite_root, "summary.json")
    summary = _load_json(summary_path) if summary_path else {}
    raw_path = summary_path.parent / "raw_results.jsonl" if summary_path else None
    failures, buckets = _qa_failure_summary(raw_path)
    expected_cases = int(suite["expected_cases"])
    actual_cases = _summary_count(summary, "total", 0)
    failed = _summary_count(summary, "failed", expected_cases)
    return {
        "name": str(suite["name"]),
        "suite": str(suite["suite"]),
        "expected_cases": expected_cases,
        "actual_cases": actual_cases,
        "coverage_complete": actual_cases == expected_cases,
        "passed": _summary_count(summary, "passed", 0),
        "failed": failed,
        "quality_ok": actual_cases == expected_cases and failed == 0,
        "process_exit_code": int(completed.returncode),
        "process_elapsed_s": elapsed_s,
        "summary_path": str(summary_path) if summary_path else "",
        "timing": dict(summary.get("timing") or {}),
        "failure_buckets": buckets,
        "failure_cases": failures,
    }


def _unsupported_project_runs(
    *,
    count: int,
    missing_paths: list[str],
    runner_missing: bool,
) -> list[dict[str, Any]]:
    reasons = []
    if missing_paths:
        reasons.append("missing API capabilities: " + ", ".join(missing_paths))
    if runner_missing:
        reasons.append("project journey evaluator is absent")
    reason = "; ".join(reasons) or "project journey is unsupported"
    return [
        {"run": index, "supported": False, "passed": False, "reason": reason}
        for index in range(1, count + 1)
    ]


def _run_project_journeys(
    *,
    side: str,
    project: dict[str, Any],
    repo: Path,
    source_root: Path,
    output_root: Path,
    missing_paths: list[str],
) -> list[dict[str, Any]]:
    count = int(project["runs"])
    runner = repo / str(project["runner"])
    if missing_paths or not runner.is_file():
        return _unsupported_project_runs(
            count=count,
            missing_paths=missing_paths,
            runner_missing=not runner.is_file(),
        )
    rows: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        run_root = output_root / side / "project_journeys" / f"run_{index}"
        command = [
            sys.executable,
            str(runner),
            "--fixture",
            str(repo / str(project["fixture"])),
            "--db-root",
            str(source_root),
            "--max-tokens",
            str(project["max_tokens"]),
            "--out-root",
            str(run_root),
        ]
        env = os.environ.copy()
        env["KB_DB_DIR"] = str(source_root)
        env["KB_CROSSREF_BUDGET_S"] = "0"
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=repo, env=env, check=False)
        elapsed_s = round(time.perf_counter() - started, 3)
        report_path = _latest_file(run_root, "report.json")
        report = _load_json(report_path) if report_path else {}
        summary = dict(report.get("summary") or {})
        failed_checks = [str(item) for item in list(summary.get("failed") or [])]
        rows.append(
            {
                "run": index,
                "supported": True,
                "passed": bool(report_path) and not failed_checks and completed.returncode == 0,
                "process_exit_code": int(completed.returncode),
                "process_elapsed_s": elapsed_s,
                "report_path": str(report_path) if report_path else "",
                "checks_passed": int(summary.get("passed") or 0),
                "checks_total": int(summary.get("total") or 0),
                "failed_checks": failed_checks,
                "journey_elapsed_ms": report.get("total_elapsed_ms"),
            }
        )
    return rows


def _timing_value(suite: dict[str, Any], metric: str, percentile: str) -> float | None:
    raw = dict(dict(suite.get("timing") or {}).get(metric) or {}).get(percentile)
    return float(raw) if isinstance(raw, (int, float)) else None


def _build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Historical Version A/B Evaluation",
        "",
        f"Created: `{report['created_at']}`",
        "",
        f"- Corpus identical: **{str(report['corpus']['identical']).lower()}**",
        f"- Comparison complete: **{str(report['comparison']['complete']).lower()}**",
        f"- Candidate release contract: **{str(report['comparison']['candidate_release_ok']).lower()}**",
        f"- Candidate materially better: **{str(report['comparison']['candidate_materially_better']).lower()}**",
        "",
        "## QA",
        "",
        "| Version | Suite | Passed | Failed | Coverage | First-answer p95 | UI p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for side in ("baseline", "candidate"):
        for suite in report["versions"][side]["qa"]:
            first_p95 = _timing_value(suite, "first_answer_ms", "p95")
            ui_p95 = _timing_value(suite, "latency_ms", "p95")
            lines.append(
                f"| {side} | {suite['suite']} | {suite['passed']} | {suite['failed']} | "
                f"{suite['actual_cases']}/{suite['expected_cases']} | "
                f"{first_p95 if first_p95 is not None else 'n/a'} | "
                f"{ui_p95 if ui_p95 is not None else 'n/a'} |"
            )
    lines.extend(["", "## Project journeys", ""])
    for side in ("baseline", "candidate"):
        rows = report["versions"][side]["project_journeys"]
        passed = sum(1 for item in rows if item.get("passed"))
        unsupported = sum(1 for item in rows if not item.get("supported"))
        lines.append(f"- **{side}:** {passed}/{len(rows)} passed; {unsupported} unsupported.")
    lines.extend(["", "## Visible failures", ""])
    any_failures = False
    for side in ("baseline", "candidate"):
        for suite in report["versions"][side]["qa"]:
            for case in suite["failure_cases"]:
                any_failures = True
                detail = ", ".join(case["failure_names"])
                if case.get("error"):
                    detail += f"; {case['error']}"
                lines.append(f"- `{side}/{suite['name']}/{case['id']}`: {detail}")
        for row in report["versions"][side]["project_journeys"]:
            if not row.get("passed"):
                any_failures = True
                detail = row.get("reason") or ", ".join(row.get("failed_checks") or []) or "journey failed"
                lines.append(f"- `{side}/project/run_{row['run']}`: {detail}")
    if not any_failures:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def _comparison_summary(
    versions: dict[str, Any],
    *,
    project_runs: int,
) -> dict[str, Any]:
    baseline_qa_passed = sum(item["passed"] for item in versions["baseline"]["qa"])
    candidate_qa_passed = sum(item["passed"] for item in versions["candidate"]["qa"])
    candidate_release_ok = all(
        item["quality_ok"] for item in versions["candidate"]["qa"]
    ) and all(item["passed"] for item in versions["candidate"]["project_journeys"])
    complete = all(
        item["coverage_complete"]
        for side in ("baseline", "candidate")
        for item in versions[side]["qa"]
    ) and all(
        len(versions[side]["project_journeys"]) == int(project_runs)
        for side in ("baseline", "candidate")
    )
    baseline_project_passed = sum(
        1 for item in versions["baseline"]["project_journeys"] if item["passed"]
    )
    candidate_project_passed = sum(
        1 for item in versions["candidate"]["project_journeys"] if item["passed"]
    )
    return {
        "complete": complete,
        "qa_pass_delta": candidate_qa_passed - baseline_qa_passed,
        "project_pass_delta": candidate_project_passed - baseline_project_passed,
        "candidate_release_ok": candidate_release_ok,
        "candidate_materially_better": bool(
            candidate_release_ok
            and candidate_qa_passed > baseline_qa_passed
            and candidate_project_passed > baseline_project_passed
        ),
    }


def rebuild_report(report_path: Path) -> dict[str, Any]:
    """Recompute aggregate gates from immutable suite and journey artifacts."""
    report = _load_json(report_path.resolve(strict=True))
    versions = dict(report["versions"])
    for side in ("baseline", "candidate"):
        for row in versions[side]["qa"]:
            summary_path_text = str(row.get("summary_path") or "").strip()
            if not summary_path_text:
                continue
            summary_path = Path(summary_path_text).resolve(strict=True)
            summary = _load_json(summary_path)
            raw_path = summary_path.parent / "raw_results.jsonl"
            failures, buckets = _qa_failure_summary(raw_path)
            expected_cases = int(row["expected_cases"])
            actual_cases = _summary_count(summary, "total", 0)
            failed = _summary_count(summary, "failed", expected_cases)
            row.update(
                {
                    "actual_cases": actual_cases,
                    "coverage_complete": actual_cases == expected_cases,
                    "passed": _summary_count(summary, "passed", 0),
                    "failed": failed,
                    "quality_ok": actual_cases == expected_cases and failed == 0,
                    "timing": dict(summary.get("timing") or {}),
                    "failure_buckets": buckets,
                    "failure_cases": failures,
                }
            )
    project_runs = int(
        dict(dict(report.get("settings") or {}).get("project_journeys") or {}).get("runs")
        or 0
    )
    report["versions"] = versions
    report["comparison"] = _comparison_summary(
        versions,
        project_runs=project_runs,
    )
    report["recomputed_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.with_name("report.md").write_text(
        _build_markdown(report),
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a strict same-corpus historical QA and project-journey A/B evaluation."
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--baseline-url", default="")
    parser.add_argument("--candidate-url", default="")
    parser.add_argument("--baseline-repo", type=Path)
    parser.add_argument("--candidate-repo", type=Path, default=ROOT)
    parser.add_argument("--baseline-source-root", type=Path)
    parser.add_argument("--candidate-source-root", type=Path)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--rebuild-report",
        type=Path,
        help="Recompute aggregate gates from an existing report and its raw artifacts.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    contract_path = args.fixture.resolve(strict=True)
    contract = load_contract(contract_path)
    if args.rebuild_report:
        report = rebuild_report(args.rebuild_report)
        print(
            json.dumps(
                {"report": str(args.rebuild_report.resolve()), **report["comparison"]},
                ensure_ascii=False,
            )
        )
        return 0 if (
            report["comparison"]["complete"]
            and report["comparison"]["candidate_release_ok"]
        ) else 1
    if args.dry_run:
        print(
            json.dumps(
                {
                    "ok": True,
                    "fixture": str(contract_path),
                    "qa_suites": dict(contract["qa"])["suites"],
                    "qa_total_cases": sum(
                        int(item["expected_cases"])
                        for item in _records(dict(contract["qa"])["suites"])
                    ),
                    "project_journey_runs": int(dict(contract["project_journeys"])["runs"]),
                    "case_timeout_s": float(dict(contract["qa"])["case_timeout_s"]),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    required = {
        "baseline-url": args.baseline_url,
        "candidate-url": args.candidate_url,
        "baseline-repo": args.baseline_repo,
        "baseline-source-root": args.baseline_source_root,
        "candidate-source-root": args.candidate_source_root,
    }
    missing_args = [name for name, value in required.items() if not value]
    if missing_args:
        parser.error("missing live evaluation arguments: " + ", ".join(missing_args))

    repos = {
        "baseline": args.baseline_repo.resolve(strict=True),
        "candidate": args.candidate_repo.resolve(strict=True),
    }
    source_roots = {
        "baseline": args.baseline_source_root.resolve(strict=True),
        "candidate": args.candidate_source_root.resolve(strict=True),
    }
    urls = {"baseline": args.baseline_url.rstrip("/"), "candidate": args.candidate_url.rstrip("/")}
    git_states = {side: _git_state(repo) for side, repo in repos.items()}
    for side in ("baseline", "candidate"):
        expected_sha = str(dict(contract[side])["sha"])
        if git_states[side]["sha"] != expected_sha:
            raise RuntimeError(
                f"{side} repo SHA mismatch: expected {expected_sha}, got {git_states[side]['sha']}"
            )

    excluded = {str(item) for item in list(dict(contract.get("corpus") or {}).get("exclude_top_level") or [])}
    fingerprints = {
        side: corpus_fingerprint(source_root, exclude_top_level=excluded)
        for side, source_root in source_roots.items()
    }
    corpus_identical = fingerprints["baseline"]["sha256"] == fingerprints["candidate"]["sha256"]
    if not corpus_identical:
        raise RuntimeError("baseline and candidate active-corpus SHA-256 fingerprints differ")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.out_root.resolve(strict=False) / stamp
    output_root.mkdir(parents=True, exist_ok=False)
    qa = dict(contract["qa"])
    project = dict(contract["project_journeys"])
    versions: dict[str, Any] = {}
    for side in ("baseline", "candidate"):
        paths = _openapi_paths(urls[side])
        qa_missing = sorted(set(qa["required_paths"]) - paths)
        project_missing = sorted(set(project["required_paths"]) - paths)
        qa_rows: list[dict[str, Any]] = []
        if qa_missing:
            for suite in _records(qa["suites"]):
                expected_cases = int(suite["expected_cases"])
                qa_rows.append(
                    {
                        "name": suite["name"],
                        "suite": suite["suite"],
                        "expected_cases": expected_cases,
                        "actual_cases": 0,
                        "coverage_complete": False,
                        "passed": 0,
                        "failed": expected_cases,
                        "quality_ok": False,
                        "failure_buckets": {"missing_api_capability": expected_cases},
                        "failure_cases": [],
                        "missing_paths": qa_missing,
                        "timing": {},
                    }
                )
        else:
            for suite in _records(qa["suites"]):
                qa_rows.append(
                    _run_qa_suite(
                        side=side,
                        suite=suite,
                        qa=qa,
                        runner_repo=repos["candidate"],
                        base_url=urls[side],
                        source_root=source_roots[side],
                        output_root=output_root,
                    )
                )
        project_rows = _run_project_journeys(
            side=side,
            project=project,
            repo=repos[side],
            source_root=source_roots[side],
            output_root=output_root,
            missing_paths=project_missing,
        )
        versions[side] = {
            "label": dict(contract[side])["label"],
            "git": git_states[side],
            "url": urls[side],
            "capabilities": {
                "path_count": len(paths),
                "qa_missing": qa_missing,
                "project_missing": project_missing,
            },
            "qa": qa_rows,
            "project_journeys": project_rows,
        }

    comparison = _comparison_summary(
        versions,
        project_runs=int(project["runs"]),
    )
    report = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "fixture": str(contract_path),
        "output_root": str(output_root),
        "settings": {"qa": qa, "project_journeys": project},
        "corpus": {"identical": corpus_identical, "fingerprints": fingerprints},
        "versions": versions,
        "comparison": comparison,
    }
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "report.md").write_text(_build_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                **report["comparison"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if comparison["complete"] and comparison["candidate_release_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
