from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:8016"
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MANIFESTS = [
    "tools/manual_regression/manifests/paper_guide_baseline_p0_v1.json",
    "tools/manual_regression/manifests/paper_guide_smoke_scinerf2024_v1.json",
    "tools/manual_regression/manifests/paper_guide_smoke_general_v1.json",
    "tools/manual_regression/manifests/paper_guide_smoke_natcomm2023_v1.json",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _new_hit_counts() -> dict[str, int]:
    return {"exact": 0, "block": 0, "heading": 0, "none": 0}


def _new_scorecard_counts() -> dict[str, dict[str, int]]:
    return {
        "locate": {"cases": 0, "pass": 0, "first_click_ok": 0, "exact_first_click": 0, "heading_or_none": 0},
        "citation": {"cases": 0, "pass": 0},
        "structured_markers": {"cases": 0, "pass": 0, "raw_cite_leak_cases": 0},
        "quality": {"cases": 0, "pass": 0},
    }


def _merge_hit_counts(dst: dict[str, int], src: dict[str, Any] | None) -> None:
    rec = src if isinstance(src, dict) else {}
    for key in ("exact", "block", "heading", "none"):
        dst[key] = int(dst.get(key) or 0) + max(0, _to_int(rec.get(key)))


def _merge_scorecard_counts(dst: dict[str, dict[str, int]], src: dict[str, Any] | None) -> None:
    rec = src if isinstance(src, dict) else {}
    for section, keys in {
        "locate": ("cases", "pass", "first_click_ok", "exact_first_click", "heading_or_none"),
        "citation": ("cases", "pass"),
        "structured_markers": ("cases", "pass", "raw_cite_leak_cases"),
        "quality": ("cases", "pass"),
    }.items():
        dst_section = dst.setdefault(section, {})
        src_section = rec.get(section) if isinstance(rec.get(section), dict) else {}
        for key in keys:
            dst_section[key] = int(dst_section.get(key) or 0) + max(0, _to_int(src_section.get(key)))


def _scorecard_rates_from_counts(counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    def _rate(numerator: Any, denominator: Any) -> float:
        den = max(0, _to_int(denominator))
        if den <= 0:
            return 0.0
        return float(max(0, _to_int(numerator))) / float(den)

    locate = counts.get("locate") if isinstance(counts.get("locate"), dict) else {}
    citation = counts.get("citation") if isinstance(counts.get("citation"), dict) else {}
    structured = counts.get("structured_markers") if isinstance(counts.get("structured_markers"), dict) else {}
    quality = counts.get("quality") if isinstance(counts.get("quality"), dict) else {}
    return {
        "locate": {
            "pass_rate": _rate(locate.get("pass"), locate.get("cases")),
            "first_click_rate": _rate(locate.get("first_click_ok"), locate.get("cases")),
            "exact_first_click_rate": _rate(locate.get("exact_first_click"), locate.get("cases")),
            "heading_or_none_rate": _rate(locate.get("heading_or_none"), locate.get("cases")),
        },
        "citation": {
            "ref_num_accuracy": _rate(citation.get("pass"), citation.get("cases")),
        },
        "structured_markers": {
            "clean_rate": _rate(structured.get("pass"), structured.get("cases")),
            "raw_cite_leak_rate": _rate(structured.get("raw_cite_leak_cases"), structured.get("cases")),
        },
        "quality": {
            "minimum_ok_rate": _rate(quality.get("pass"), quality.get("cases")),
        },
    }


def _load_case_metrics(raw_json_path: str, *, summary_json_path: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {
        "case_total": 0,
        "case_pass": 0,
        "case_fail": 0,
        "cases_with_direct": 0,
        "cases_with_exact": 0,
        "primary_hit_levels": _new_hit_counts(),
        "hit_level_counts": _new_hit_counts(),
        "scorecard_counts": _new_scorecard_counts(),
    }
    summary_path = Path(str(summary_json_path or "").strip())
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
            out["case_total"] = max(0, _to_int(counts.get("total")))
            out["case_pass"] = max(0, _to_int(counts.get("pass")))
            out["case_fail"] = max(0, _to_int(counts.get("fail")))
            hit_levels = payload.get("hit_levels") if isinstance(payload.get("hit_levels"), dict) else {}
            primary = payload.get("primary_hit_levels") if isinstance(payload.get("primary_hit_levels"), dict) else {}
            out["cases_with_direct"] = max(0, _to_int(hit_levels.get("cases_with_direct")))
            out["cases_with_exact"] = max(0, _to_int(hit_levels.get("cases_with_exact")))
            _merge_hit_counts(out["hit_level_counts"], hit_levels.get("counts") if isinstance(hit_levels.get("counts"), dict) else {})
            _merge_hit_counts(out["primary_hit_levels"], primary.get("counts") if isinstance(primary.get("counts"), dict) else {})
            _merge_scorecard_counts(out["scorecard_counts"], payload.get("scorecard"))
            return out
    raw_path = Path(str(raw_json_path or "").strip())
    if not raw_path.exists():
        return out
    try:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    except Exception:
        return out
    if not isinstance(payload, list):
        return out
    for row in payload:
        if not isinstance(row, dict):
            continue
        evaluation = row.get("evaluation")
        if not isinstance(evaluation, dict):
            continue
        out["case_total"] = int(out["case_total"]) + 1
        status = str(evaluation.get("status") or "").strip().upper()
        if status == "PASS":
            out["case_pass"] = int(out["case_pass"]) + 1
        else:
            out["case_fail"] = int(out["case_fail"]) + 1
        if _to_int(evaluation.get("visible_direct_segment_count")) > 0:
            out["cases_with_direct"] = int(out["cases_with_direct"]) + 1
        hit_counts = evaluation.get("hit_level_counts")
        _merge_hit_counts(out["hit_level_counts"], hit_counts if isinstance(hit_counts, dict) else {})
        if _to_int((hit_counts or {}).get("exact")) > 0:
            out["cases_with_exact"] = int(out["cases_with_exact"]) + 1
        primary = str(evaluation.get("primary_hit_level") or "").strip().lower()
        if primary not in {"exact", "block", "heading", "none"}:
            primary = "none"
        out["primary_hit_levels"][primary] = int(out["primary_hit_levels"].get(primary) or 0) + 1
    return out


def _rates_from_hit_counts(counts: dict[str, int]) -> dict[str, float]:
    total = max(0, sum(max(0, _to_int(v)) for v in counts.values()))
    if total <= 0:
        return {
            "exact_share": 0.0,
            "exact_or_block_share": 0.0,
            "heading_or_none_share": 0.0,
        }
    exact = max(0, _to_int(counts.get("exact")))
    block = max(0, _to_int(counts.get("block")))
    heading = max(0, _to_int(counts.get("heading")))
    none = max(0, _to_int(counts.get("none")))
    return {
        "exact_share": float(exact) / float(total),
        "exact_or_block_share": float(exact + block) / float(total),
        "heading_or_none_share": float(heading + none) / float(total),
    }


def _run_one_manifest(*, base_url: str, timeout_s: float, manifest_path: Path) -> dict[str, object]:
    cmd = [
        sys.executable,
        "tools/manual_regression/paper_guide_benchmark.py",
        "--base-url",
        base_url,
        "--manifest",
        str(manifest_path),
        "--timeout-s",
        str(timeout_s),
    ]
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    proc = subprocess.run(
        cmd,
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
        check=False,
    )
    ended_at = dt.datetime.now().isoformat(timespec="seconds")
    stdout = str(proc.stdout or "").strip()
    stderr = str(proc.stderr or "").strip()
    report_path = ""
    raw_path = ""
    summary_json_path = ""
    for ln in stdout.splitlines():
        line = str(ln or "").strip()
        if line.startswith("Summary JSON written:"):
            summary_json_path = line.replace("Summary JSON written:", "", 1).strip()
        elif line.startswith("Report written:"):
            report_path = line.replace("Report written:", "", 1).strip()
        elif line.startswith("Raw JSON written:"):
            raw_path = line.replace("Raw JSON written:", "", 1).strip()
    return {
        "manifest": str(manifest_path),
        "ok": bool(proc.returncode == 0),
        "returncode": int(proc.returncode),
        "started_at": started_at,
        "ended_at": ended_at,
        "report_path": report_path,
        "raw_json_path": raw_path,
        "summary_json_path": summary_json_path,
        "stdout_tail": "\n".join(stdout.splitlines()[-30:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-30:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed paper-guide failure replay pool v1.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Backend base URL.")
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S, help="Per-manifest timeout passed to benchmark script.")
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Optional extra manifest path. Can be passed multiple times.",
    )
    args = parser.parse_args()

    base = _repo_root()
    manifests: list[Path] = []
    seen: set[str] = set()
    for raw in list(DEFAULT_MANIFESTS) + list(args.manifest or []):
        p = Path(str(raw or "").strip())
        if not p.is_absolute():
            p = (base / p).resolve()
        key = str(p).lower()
        if (not str(p)) or key in seen:
            continue
        seen.add(key)
        manifests.append(p)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (base / "test_results" / "paper_guide_failure_replay_v1" / ts).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for mf in manifests:
        if not mf.exists():
            results.append(
                {
                    "manifest": str(mf),
                    "ok": False,
                    "returncode": 2,
                    "started_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "ended_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "report_path": "",
                    "raw_json_path": "",
                    "stdout_tail": "",
                    "stderr_tail": f"manifest not found: {mf}",
                }
            )
            continue
        rec = _run_one_manifest(
            base_url=str(args.base_url or DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL,
            timeout_s=float(args.timeout_s or DEFAULT_TIMEOUT_S),
            manifest_path=mf,
        )
        rec["case_metrics"] = _load_case_metrics(
            str(rec.get("raw_json_path") or ""),
            summary_json_path=str(rec.get("summary_json_path") or ""),
        )
        results.append(rec)

    ok_count = sum(1 for r in results if bool(r.get("ok")))
    case_total = 0
    case_pass = 0
    case_fail = 0
    cases_with_direct = 0
    cases_with_exact = 0
    hit_level_counts = _new_hit_counts()
    primary_hit_levels = _new_hit_counts()
    scorecard_counts = _new_scorecard_counts()
    for rec in results:
        metrics = rec.get("case_metrics")
        if not isinstance(metrics, dict):
            continue
        case_total += max(0, _to_int(metrics.get("case_total")))
        case_pass += max(0, _to_int(metrics.get("case_pass")))
        case_fail += max(0, _to_int(metrics.get("case_fail")))
        cases_with_direct += max(0, _to_int(metrics.get("cases_with_direct")))
        cases_with_exact += max(0, _to_int(metrics.get("cases_with_exact")))
        _merge_hit_counts(hit_level_counts, metrics.get("hit_level_counts") if isinstance(metrics.get("hit_level_counts"), dict) else {})
        _merge_hit_counts(primary_hit_levels, metrics.get("primary_hit_levels") if isinstance(metrics.get("primary_hit_levels"), dict) else {})
        _merge_scorecard_counts(scorecard_counts, metrics.get("scorecard_counts") if isinstance(metrics.get("scorecard_counts"), dict) else {})

    summary = {
        "timestamp": ts,
        "base_url": str(args.base_url or DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL,
        "timeout_s": float(args.timeout_s or DEFAULT_TIMEOUT_S),
        "total": int(len(results)),
        "ok": int(ok_count),
        "failed": int(len(results) - ok_count),
        "case_total": int(case_total),
        "case_pass": int(case_pass),
        "case_fail": int(case_fail),
        "cases_with_direct": int(cases_with_direct),
        "cases_with_exact": int(cases_with_exact),
        "hit_level_counts": hit_level_counts,
        "hit_level_rates": _rates_from_hit_counts(hit_level_counts),
        "primary_hit_levels": primary_hit_levels,
        "scorecard_counts": scorecard_counts,
        "scorecard_rates": _scorecard_rates_from_counts(scorecard_counts),
        "results": results,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[failure-replay-v1] total={summary['total']} ok={summary['ok']} failed={summary['failed']}")
    for item in results:
        name = Path(str(item.get("manifest") or "")).name
        status = "PASS" if bool(item.get("ok")) else "FAIL"
        metrics = item.get("case_metrics") if isinstance(item.get("case_metrics"), dict) else {}
        hit_counts = metrics.get("hit_level_counts") if isinstance(metrics.get("hit_level_counts"), dict) else {}
        print(
            f"[{status}] {name} (rc={int(item.get('returncode') or 0)}) "
            f"cases={_to_int(metrics.get('case_pass'))}/{_to_int(metrics.get('case_total'))} "
            f"hit(exact/block/heading/none)="
            f"{_to_int(hit_counts.get('exact'))}/{_to_int(hit_counts.get('block'))}/"
            f"{_to_int(hit_counts.get('heading'))}/{_to_int(hit_counts.get('none'))}"
        )
    rates = summary.get("hit_level_rates") if isinstance(summary.get("hit_level_rates"), dict) else {}
    print(
        "[failure-replay-v1] hit-level-share "
        f"exact={float(rates.get('exact_share') or 0.0):.3f} "
        f"exact_or_block={float(rates.get('exact_or_block_share') or 0.0):.3f} "
        f"heading_or_none={float(rates.get('heading_or_none_share') or 0.0):.3f}"
    )
    score_rates = summary.get("scorecard_rates") if isinstance(summary.get("scorecard_rates"), dict) else {}
    locate_rates = score_rates.get("locate") if isinstance(score_rates.get("locate"), dict) else {}
    citation_rates = score_rates.get("citation") if isinstance(score_rates.get("citation"), dict) else {}
    structured_rates = score_rates.get("structured_markers") if isinstance(score_rates.get("structured_markers"), dict) else {}
    print(
        "[failure-replay-v1] scorecard "
        f"locate_first_click={float(locate_rates.get('first_click_rate') or 0.0):.3f} "
        f"locate_exact_first_click={float(locate_rates.get('exact_first_click_rate') or 0.0):.3f} "
        f"citation_ref_num_accuracy={float(citation_rates.get('ref_num_accuracy') or 0.0):.3f} "
        f"structured_clean_rate={float(structured_rates.get('clean_rate') or 0.0):.3f}"
    )
    print(f"Summary written: {summary_path}")

    return 0 if int(summary["failed"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
