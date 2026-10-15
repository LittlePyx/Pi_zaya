from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.product_query_policy import search_hits_with_fallback
from kb.retrieval_eval_fixture import sha256_file
from kb.retrieval_shadow import RetrievalShadowRunner, schedule_product_retrieval_shadow
from kb.retriever import BM25Retriever
from kb.store import load_all_chunks
from tools.retrieval_eval.run_user_journey_qwen_dense_diagnostic import (
    DEFAULT_QWEN_INDEX,
)
from tools.retrieval_eval.validate_product_shadow_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_user_journey_expansion_v1.json"
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_product_shadow_operational_v1.json"
DEFAULT_TELEMETRY = (
    REPO_ROOT / "test_results" / "retrieval_product_shadow_v1" / "observations.jsonl"
)
EXPECTED_RESULT_FINGERPRINT = (
    "48e54e620d73681f3fbaf8ad7d3e86982b51f5079d7ab0f1d66878113500461e"
)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(max(0.0, float(value)) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return round(ordered[index], 3)


def _tree_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    completed = [row for row in results if row.get("status") == "done"]
    warm = [row for row in completed if not bool(row.get("cold_start"))]
    unchanged = sum(bool(row.get("product_output_unchanged")) for row in completed)
    preserved = sum(bool(row.get("protected_prefix_preserved")) for row in completed)
    return {
        "attempted_query_count": total,
        "completed_query_count": len(completed),
        "completion_rate": round(len(completed) / total, 6) if total else 0.0,
        "cold_start_count": sum(bool(row.get("cold_start")) for row in completed),
        "warm_query_count": len(warm),
        "product_output_unchanged_rate": (
            round(unchanged / len(completed), 6) if completed else 0.0
        ),
        "protected_prefix_preserved_rate": (
            round(preserved / len(completed), 6) if completed else 0.0
        ),
        "source_escape_count": sum(
            max(0, int(row.get("source_escape_count") or 0)) for row in completed
        ),
        "dense_novelty_count": sum(
            max(0, int(row.get("dense_novelty_count") or 0)) for row in completed
        ),
        "warm_shadow_p50_ms": _percentile(
            [float(row.get("duration_ms") or 0.0) for row in warm], 0.50
        ),
        "warm_shadow_p95_ms": _percentile(
            [float(row.get("duration_ms") or 0.0) for row in warm], 0.95
        ),
        "foreground_submission_p50_ms": _percentile(
            [float(row.get("submission_ms") or 0.0) for row in results], 0.50
        ),
        "foreground_submission_p95_ms": _percentile(
            [float(row.get("submission_ms") or 0.0) for row in results], 0.95
        ),
        "status_counts": {
            status: count
            for status, count in {
                "done": sum(str(row.get("status") or "") == "done" for row in results),
                "error": sum(
                    str(row.get("status") or "") == "error" for row in results
                ),
                "busy": sum(str(row.get("status") or "") == "busy" for row in results),
                "other": sum(
                    str(row.get("status") or "") not in {"done", "error", "busy"}
                    for row in results
                ),
            }.items()
            if count
        },
    }


def evaluate_operational_gate(
    summary: dict[str, Any],
    gate: dict[str, Any],
    *,
    privacy_replay_passed: bool,
    fail_open_probe_passed: bool,
    index_unchanged: bool,
) -> dict[str, Any]:
    checks = {
        "minimum_completed_queries": int(summary.get("completed_query_count") or 0)
        >= int(gate["minimum_completed_queries"]),
        "completion_rate": float(summary.get("completion_rate") or 0.0)
        >= float(gate["required_completion_rate"]),
        "product_output_unchanged_rate": float(
            summary.get("product_output_unchanged_rate") or 0.0
        )
        >= float(gate["required_product_output_unchanged_rate"]),
        "protected_prefix_preserved_rate": float(
            summary.get("protected_prefix_preserved_rate") or 0.0
        )
        >= float(gate["required_protected_prefix_preserved_rate"]),
        "source_escape_count": int(summary.get("source_escape_count") or 0)
        <= int(gate["maximum_source_escape_count"]),
        "warm_shadow_p95_ms": float(summary.get("warm_shadow_p95_ms") or 0.0)
        <= float(gate["maximum_warm_shadow_p95_ms"]),
        "foreground_submission_p95_ms": float(
            summary.get("foreground_submission_p95_ms") or 0.0
        )
        <= float(gate["maximum_foreground_submission_p95_ms"]),
        "privacy_replay": bool(privacy_replay_passed),
        "fail_open_probe": bool(fail_open_probe_passed)
        and bool(gate.get("fail_open_tests_required")),
        "index_unchanged": bool(index_unchanged),
    }
    return {"checks": checks, "passes_all": all(checks.values())}


def _result_fingerprint(results: list[dict[str, Any]], gate: dict[str, Any]) -> str:
    projection = {
        "results": [
            {
                key: row.get(key)
                for key in (
                    "case_id",
                    "mode",
                    "status",
                    "product_output_unchanged",
                    "protected_prefix_preserved",
                    "source_escape_count",
                    "dense_novelty_count",
                    "overlap_count_at_5",
                    "overlap_count_at_10",
                    "changed_count_at_5",
                    "changed_count_at_10",
                )
            }
            for row in results
        ],
        "gate": gate,
    }
    canonical = json.dumps(projection, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _expected_case_rows(fixture: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (str(case["id"]), mode)
        for case in fixture["cases"]
        for mode in ("scoped", "full_library")
    ]


def validate_report(
    report: dict[str, Any],
    *,
    protocol: dict[str, Any],
    fixture: dict[str, Any],
    expected_result_fingerprint: str | None = EXPECTED_RESULT_FINGERPRINT,
) -> list[str]:
    errors: list[str] = []
    if (
        report.get("schema_version") != 1
        or report.get("report_kind") != "retrieval_product_shadow_operational_v1"
    ):
        errors.append("product shadow operational report identity changed")
    if report.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL) or report.get(
        "fixture_sha256"
    ) != sha256_file(DEFAULT_FIXTURE):
        errors.append("product shadow operational source identity changed")
    rows = list(report.get("results") or [])
    if [
        (str(row.get("case_id") or ""), str(row.get("mode") or "")) for row in rows
    ] != _expected_case_rows(fixture):
        errors.append("product shadow operational case membership changed")
    allowed_result_fields = {
        "case_id",
        "mode",
        "status",
        "submission_ms",
        "duration_ms",
        "product_output_unchanged",
        "cold_start",
        "index_cache_hit",
        "product_hit_count",
        "shadow_hit_count",
        "overlap_count_at_5",
        "overlap_count_at_10",
        "changed_count_at_5",
        "changed_count_at_10",
        "dense_novelty_count",
        "source_escape_count",
        "protected_prefix_preserved",
        "fallback_reason_id",
    }
    if any(set(row) - allowed_result_fields for row in rows):
        errors.append("product shadow report contains unapproved result fields")
    expected_summary = summarize_results(rows)
    if report.get("summary") != expected_summary:
        errors.append("product shadow operational summary is inconsistent")
    telemetry = report.get("telemetry") or {}
    privacy_passed = bool(telemetry.get("privacy_replay_passed"))
    fail_open_passed = bool(telemetry.get("fail_open_probe_passed"))
    index = report.get("index") or {}
    index_unchanged = bool(index.get("unchanged")) and bool(index.get("before_sha256"))
    if index.get("before_sha256") != index.get("after_sha256") or not index_unchanged:
        errors.append("product shadow changed the read-only index")
    expected_gate = evaluate_operational_gate(
        expected_summary,
        protocol["controlled_operational_gate"],
        privacy_replay_passed=privacy_passed,
        fail_open_probe_passed=fail_open_passed,
        index_unchanged=index_unchanged,
    )
    if report.get("operational_gate") != expected_gate:
        errors.append("product shadow operational gate is inconsistent")
    fingerprint = _result_fingerprint(rows, expected_gate)
    if report.get("result_fingerprint") != fingerprint:
        errors.append("product shadow result fingerprint is inconsistent")
    if expected_result_fingerprint and fingerprint != expected_result_fingerprint:
        errors.append("pinned product shadow result changed")
    expected_decision = {
        "status": (
            "passed_for_opt_in_design"
            if expected_gate["passes_all"]
            else "rejected_keep_shadow_disabled"
        ),
        "opt_in_design_allowed": bool(expected_gate["passes_all"]),
        "shadow_enabled_by_default": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }
    if report.get("decision") != expected_decision:
        errors.append("product shadow operational decision changed")
    if (
        telemetry.get("event_count") != len(rows) + 1
        or telemetry.get("content_free") is not True
    ):
        errors.append("product shadow telemetry replay is incomplete")
    return errors


def _scope_chunks(
    chunks: list[dict], case: dict[str, Any], fixture: dict[str, Any]
) -> list[dict]:
    logical_to_index = {
        str(row["doc_id"]): str(row["index_doc_id"])
        for row in fixture["corpus"]["documents"]
    }
    allowed = {
        logical_to_index[str(doc_id)] for doc_id in case.get("scope_doc_ids") or []
    }
    return [
        chunk
        for chunk in chunks
        if str(chunk.get("id") or "").split(":", 1)[0] in allowed
    ]


def _product_candidates(
    chunks: list[dict], query: str, *, whole_library: bool
) -> list[dict]:
    hits, _scores, _used_query, _translation, _variants = search_hits_with_fallback(
        query,
        BM25Retriever(chunks),
        4,
        SimpleNamespace(api_key=None, query_expansion_enabled=False),
        allow_translate=True,
        allow_expand=False,
        whole_library=whole_library,
    )
    return list(hits or [])


def _run_replay(
    *,
    db_root: Path,
    index_dir: Path,
    telemetry_path: Path,
    protocol: dict[str, Any],
    fixture: dict[str, Any],
) -> dict[str, Any]:
    if telemetry_path.exists():
        raise ValueError("shadow telemetry path already exists")
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["KB_DENSE_SHADOW_ENABLED"] = "1"
    os.environ["KB_DENSE_SHADOW_INDEX_DIR"] = str(index_dir)
    os.environ["KB_LOCAL_OBSERVABILITY_PATH"] = str(telemetry_path)
    chunks = load_all_chunks(db_root)
    index_before = _tree_fingerprint(index_dir)
    probe_hits = [{"id": "probe-doc:1", "text": "", "meta": {}}]

    def fail_probe(_db: Path, _index: Path):
        raise RuntimeError("expected fail-open probe")

    probe_submission = RetrievalShadowRunner(fail_probe).schedule(
        query="shadow-fail-open-probe",
        product_hits=probe_hits,
        db_dir=db_root,
        index_dir=index_dir,
        channel="controlled_replay_shadow",
        source_scope="library",
        scoped=False,
    )
    probe_result = (
        probe_submission.future.result(timeout=5.0)
        if probe_submission.future is not None
        else {"status": probe_submission.status}
    )
    fail_open_probe_passed = bool(
        probe_result.get("status") == "error"
        and probe_result.get("product_output_unchanged") is True
        and probe_result.get("fallback_reason_id") == "dense_runtime_unavailable"
        and probe_hits == [{"id": "probe-doc:1", "text": "", "meta": {}}]
    )
    results: list[dict[str, Any]] = []
    privacy_canaries: list[str] = []
    for case in fixture["cases"]:
        query = str(case["query"])
        privacy_canaries.append(query)
        for mode in ("scoped", "full_library"):
            corpus = (
                _scope_chunks(chunks, case, fixture) if mode == "scoped" else chunks
            )
            product_hits = _product_candidates(
                corpus,
                query,
                whole_library=mode == "full_library",
            )
            privacy_canaries.extend(
                str(hit.get("text") or "") for hit in product_hits[:1]
            )
            started = time.perf_counter()
            submission = schedule_product_retrieval_shadow(
                query=query,
                product_hits=product_hits,
                db_dir=db_root,
                source_scope="current_paper" if mode == "scoped" else "library",
            )
            submission_ms = (time.perf_counter() - started) * 1000.0
            if submission.future is None:
                result: dict[str, Any] = {"status": submission.status}
            else:
                result = submission.future.result(timeout=120.0)
            results.append(
                {
                    "case_id": str(case["id"]),
                    "mode": mode,
                    "submission_ms": round(submission_ms, 3),
                    **{
                        key: (round(value, 3) if key == "duration_ms" else value)
                        for key, value in result.items()
                    },
                }
            )
    telemetry_text = telemetry_path.read_text(encoding="utf-8")
    privacy_passed = not any(
        canary and canary in telemetry_text for canary in privacy_canaries
    )
    event_count = len([line for line in telemetry_text.splitlines() if line.strip()])
    index_after = _tree_fingerprint(index_dir)
    summary = summarize_results(results)
    gate = evaluate_operational_gate(
        summary,
        protocol["controlled_operational_gate"],
        privacy_replay_passed=privacy_passed,
        fail_open_probe_passed=fail_open_probe_passed,
        index_unchanged=index_before == index_after,
    )
    decision = {
        "status": (
            "passed_for_opt_in_design"
            if gate["passes_all"]
            else "rejected_keep_shadow_disabled"
        ),
        "opt_in_design_allowed": bool(gate["passes_all"]),
        "shadow_enabled_by_default": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }
    return {
        "schema_version": 1,
        "report_kind": "retrieval_product_shadow_operational_v1",
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "fixture_sha256": sha256_file(DEFAULT_FIXTURE),
        "index": {
            "relative_path": index_dir.relative_to(REPO_ROOT).as_posix(),
            "current_pointer_sha256": sha256_file(index_dir / "current.json"),
            "read_only": True,
            "before_sha256": index_before,
            "after_sha256": index_after,
            "unchanged": index_before == index_after,
        },
        "results": results,
        "summary": summary,
        "telemetry": {
            "event_count": event_count,
            "content_free": True,
            "privacy_replay_passed": privacy_passed,
            "fail_open_probe_passed": fail_open_probe_passed,
        },
        "operational_gate": gate,
        "result_fingerprint": _result_fingerprint(results, gate),
        "decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run or validate the frozen product shadow replay."
    )
    parser.add_argument("--db-root", type=Path, default=REPO_ROOT / "db")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_QWEN_INDEX)
    parser.add_argument("--telemetry", type=Path, default=DEFAULT_TELEMETRY)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    contract_errors = validate_protocol(protocol)
    if contract_errors:
        parser.error("invalid shadow protocol: " + "; ".join(contract_errors))
    if args.validate_only:
        report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
        errors = validate_report(report, protocol=protocol, fixture=fixture)
        if errors:
            for error in errors:
                print(f"[FAIL] {error}")
            return 1
        print(f"[OK] product shadow operational replay: {report['decision']['status']}")
        return 0
    if DEFAULT_REPORT.exists():
        parser.error(
            "product shadow operational report already exists; use --validate-only"
        )
    report = _run_replay(
        db_root=args.db_root.expanduser().resolve(),
        index_dir=args.index_dir.expanduser().resolve(),
        telemetry_path=args.telemetry.expanduser().resolve(),
        protocol=protocol,
        fixture=fixture,
    )
    errors = validate_report(
        report,
        protocol=protocol,
        fixture=fixture,
        expected_result_fingerprint=None,
    )
    if errors:
        raise ValueError("invalid product shadow report: " + "; ".join(errors))
    DEFAULT_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"summary": report["summary"], "decision": report["decision"]}, indent=2
        )
    )
    print(f"[OK] wrote {DEFAULT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
