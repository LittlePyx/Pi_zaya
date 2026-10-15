from __future__ import annotations

import json

from tools.retrieval_eval.run_product_shadow_operational_replay import (
    DEFAULT_FIXTURE,
    DEFAULT_PROTOCOL,
    _result_fingerprint,
    evaluate_operational_gate,
    summarize_results,
    validate_report,
)


def _rows(*, duration_ms: float = 80.0, submission_ms: float = 1.0) -> list[dict]:
    fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    rows = []
    for index, case in enumerate(fixture["cases"]):
        for mode in ("scoped", "full_library"):
            rows.append(
                {
                    "case_id": case["id"],
                    "mode": mode,
                    "submission_ms": submission_ms,
                    "status": "done",
                    "duration_ms": 1800.0
                    if index == 0 and mode == "scoped"
                    else duration_ms,
                    "product_output_unchanged": True,
                    "cold_start": index == 0 and mode == "scoped",
                    "index_cache_hit": not (index == 0 and mode == "scoped"),
                    "product_hit_count": 40,
                    "shadow_hit_count": 10,
                    "overlap_count_at_5": 5,
                    "overlap_count_at_10": 8,
                    "changed_count_at_5": 0,
                    "changed_count_at_10": 2,
                    "dense_novelty_count": 2,
                    "source_escape_count": 0,
                    "protected_prefix_preserved": True,
                }
            )
    return rows


def test_operational_summary_excludes_cold_start_from_warm_latency() -> None:
    summary = summarize_results(_rows())
    assert summary["attempted_query_count"] == 36
    assert summary["completed_query_count"] == 36
    assert summary["cold_start_count"] == 1
    assert summary["warm_query_count"] == 35
    assert summary["warm_shadow_p95_ms"] == 80.0


def test_operational_gate_rejects_foreground_latency_regression() -> None:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    summary = summarize_results(_rows(submission_ms=5.1))
    gate = evaluate_operational_gate(
        summary,
        protocol["controlled_operational_gate"],
        privacy_replay_passed=True,
        fail_open_probe_passed=True,
        index_unchanged=True,
    )
    assert gate["checks"]["foreground_submission_p95_ms"] is False
    assert gate["passes_all"] is False


def test_synthetic_passing_operational_report_validates() -> None:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    rows = _rows()
    summary = summarize_results(rows)
    gate = evaluate_operational_gate(
        summary,
        protocol["controlled_operational_gate"],
        privacy_replay_passed=True,
        fail_open_probe_passed=True,
        index_unchanged=True,
    )
    report = {
        "schema_version": 1,
        "report_kind": "retrieval_product_shadow_operational_v1",
        "protocol_sha256": __import__(
            "kb.retrieval_eval_fixture", fromlist=["sha256_file"]
        ).sha256_file(DEFAULT_PROTOCOL),
        "fixture_sha256": __import__(
            "kb.retrieval_eval_fixture", fromlist=["sha256_file"]
        ).sha256_file(DEFAULT_FIXTURE),
        "index": {
            "before_sha256": "a" * 64,
            "after_sha256": "a" * 64,
            "unchanged": True,
        },
        "results": rows,
        "summary": summary,
        "telemetry": {
            "event_count": 37,
            "content_free": True,
            "privacy_replay_passed": True,
            "fail_open_probe_passed": True,
        },
        "operational_gate": gate,
        "result_fingerprint": _result_fingerprint(rows, gate),
        "decision": {
            "status": "passed_for_opt_in_design",
            "opt_in_design_allowed": True,
            "shadow_enabled_by_default": False,
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
        },
    }
    assert (
        validate_report(
            report,
            protocol=protocol,
            fixture=fixture,
            expected_result_fingerprint=None,
        )
        == []
    )


def test_report_validator_rejects_raw_result_field() -> None:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    rows = _rows()
    rows[0]["query"] = "private"
    report = {
        "schema_version": 1,
        "report_kind": "retrieval_product_shadow_operational_v1",
        "protocol_sha256": "",
        "fixture_sha256": "",
        "results": rows,
        "summary": {},
        "telemetry": {},
        "operational_gate": {},
        "result_fingerprint": "",
        "decision": {},
    }
    assert "product shadow report contains unapproved result fields" in validate_report(
        report,
        protocol=protocol,
        fixture=fixture,
        expected_result_fingerprint=None,
    )
