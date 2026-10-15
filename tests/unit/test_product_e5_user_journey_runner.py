from __future__ import annotations

import copy
import json

from tools.retrieval_eval.run_product_e5_selector_shadow import (
    CANDIDATE_CHANNEL,
    PRODUCT_CHANNEL,
)
from tools.retrieval_eval.run_product_e5_user_journey import (
    DEFAULT_REPORT,
    _decision,
    evaluate_quality_gate,
    validate_report,
)
from tools.retrieval_eval.validate_product_e5_user_journey_protocol import (
    DEFAULT_PROTOCOL,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def _mode(metrics: dict[str, float]) -> dict:
    return {
        "metrics": dict(metrics),
        "cases": [
            {
                "id": "case-1",
                "metrics": dict(metrics),
            }
        ],
    }


def _channels() -> dict:
    product = {
        "recall@5": 0.2,
        "recall@10": 0.2,
        "mrr": 0.1,
        "ndcg@10": 0.1,
        "contract_recall@10": 0.2,
        "wrong_source_rate@5": 0.3,
        "wrong_source_rate@10": 0.4,
    }
    candidate = {
        **product,
        "recall@10": 0.3,
        "mrr": 0.2,
        "ndcg@10": 0.2,
        "contract_recall@10": 0.3,
        "wrong_source_rate@10": 0.3,
    }
    return {
        PRODUCT_CHANNEL: {
            "modes": {
                mode: _mode(product) for mode in ("scoped", "full_library")
            }
        },
        CANDIDATE_CHANNEL: {
            "modes": {
                mode: _mode(candidate) for mode in ("scoped", "full_library")
            }
        },
    }


def test_product_e5_user_journey_quality_gate_passes_safe_gain() -> None:
    gate = evaluate_quality_gate(_channels(), _protocol())

    assert gate["passes_all"] is True
    assert gate["minimum_gain_in_at_least_one_mode"] is True


def test_product_e5_user_journey_quality_gate_rejects_case_recall_harm() -> None:
    channels = _channels()
    row = channels[CANDIDATE_CHANNEL]["modes"]["scoped"]["cases"][0]
    row["metrics"]["recall@10"] = 0.0

    gate = evaluate_quality_gate(channels, _protocol())

    assert gate["modes"]["scoped"]["case_regressions"]["recall@10"] == ["case-1"]
    assert gate["passes_all"] is False


def test_product_e5_user_journey_quality_gate_rejects_wrong_source_increase() -> None:
    channels = _channels()
    channels[CANDIDATE_CHANNEL]["modes"]["full_library"]["metrics"][
        "wrong_source_rate@10"
    ] = 0.5

    gate = evaluate_quality_gate(channels, _protocol())

    assert gate["modes"]["full_library"]["checks"][
        "wrong_source_rate@10_no_increase"
    ] is False
    assert gate["passes_all"] is False


def test_product_e5_user_journey_decision_never_tunes_or_enables() -> None:
    passed = {"passes_all": True}
    decision = _decision(_protocol(), passed, passed)
    rejected = _decision(_protocol(), passed, {"passes_all": False})

    assert decision["product_enabled"] is False
    assert decision["candidate_tuning_allowed"] is False
    assert rejected["status"] == "current_domain_quality_rejected_keep_bm25"


def test_product_e5_user_journey_quality_gate_rejects_case_membership_drift() -> None:
    channels = copy.deepcopy(_channels())
    channels[CANDIDATE_CHANNEL]["modes"]["scoped"]["cases"][0]["id"] = "other"

    try:
        evaluate_quality_gate(channels, _protocol())
    except ValueError as exc:
        assert str(exc) == "product/candidate case membership changed"
    else:
        raise AssertionError("case membership drift was accepted")


def test_product_e5_user_journey_rejection_report_is_pinned_and_valid() -> None:
    report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))

    assert validate_report(report, protocol=_protocol()) == []


def test_product_e5_user_journey_report_rejects_hidden_wrong_source_tamper() -> None:
    report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    report["quality_gate"]["passes_all"] = True

    errors = validate_report(report, protocol=_protocol())

    assert "product E5 user-journey quality gate is inconsistent" in errors
    assert "product E5 user-journey result fingerprint is inconsistent" in errors
