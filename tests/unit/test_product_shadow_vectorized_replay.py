from __future__ import annotations

import copy
import json

from tools.retrieval_eval.run_product_shadow_operational_replay import (
    DEFAULT_FIXTURE,
    DEFAULT_PROTOCOL as DEFAULT_PARENT_PROTOCOL,
    _result_fingerprint,
    evaluate_operational_gate,
    summarize_results,
)
from tools.retrieval_eval.run_product_shadow_vectorized_replay import (
    DEFAULT_PARENT_REPORT,
    DEFAULT_PROTOCOL,
    build_child_report,
    validate_report,
)


def _passing_base() -> tuple[dict, dict, dict]:
    child = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    parent = json.loads(DEFAULT_PARENT_PROTOCOL.read_text(encoding="utf-8"))
    fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    base = json.loads(DEFAULT_PARENT_REPORT.read_text(encoding="utf-8"))
    rows = copy.deepcopy(base["results"])
    for row in rows:
        row["duration_ms"] = 60.0 if not row.get("cold_start") else 1500.0
        row["submission_ms"] = 1.0
    summary = summarize_results(rows)
    gate = evaluate_operational_gate(
        summary,
        parent["controlled_operational_gate"],
        privacy_replay_passed=True,
        fail_open_probe_passed=True,
        index_unchanged=True,
    )
    base.update(
        {
            "results": rows,
            "summary": summary,
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
    )
    return build_child_report(base, child), child, parent


def test_vectorized_child_adapter_validates_without_changing_parent_report() -> None:
    report, child, parent = _passing_base()
    fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))

    assert report["decision"]["status"] == "passed_for_explicit_opt_in_design"
    assert (
        validate_report(
            report,
            child_protocol=child,
            parent_protocol=parent,
            fixture=fixture,
            expected_result_fingerprint=None,
        )
        == []
    )


def test_vectorized_child_validator_rejects_candidate_hash_change() -> None:
    report, child, parent = _passing_base()
    fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    report["performance_candidate"]["dense_retriever_sha256"] = "0" * 64

    assert "vectorized shadow performance candidate changed" in validate_report(
        report,
        child_protocol=child,
        parent_protocol=parent,
        fixture=fixture,
        expected_result_fingerprint=None,
    )


def test_vectorized_child_validator_rejects_relabeled_failure() -> None:
    report, child, parent = _passing_base()
    fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    report["operational_gate"]["checks"]["warm_shadow_p95_ms"] = False
    report["operational_gate"]["passes_all"] = False
    report["result_fingerprint"] = _result_fingerprint(
        report["results"], report["operational_gate"]
    )

    errors = validate_report(
        report,
        child_protocol=child,
        parent_protocol=parent,
        fixture=fixture,
        expected_result_fingerprint=None,
    )
    assert "product shadow operational gate is inconsistent" in errors
    assert "vectorized shadow decision changed" in errors
