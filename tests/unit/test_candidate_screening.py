from __future__ import annotations

import json

from tools.retrieval_eval.run_candidate_screening import (
    CANDIDATE_CHANNEL,
    DEFAULT_NOMINATION_REPORT,
    PRODUCT_CHANNEL,
    SAFE_CHANNEL,
    _screening_fixture,
    _validate_nomination_report,
    evaluate_screening_gates,
    select_nominee,
    validate_screening_report,
)
from tools.retrieval_eval.validate_candidate_batch import DEFAULT_BATCH
from tools.retrieval_eval.validate_candidate_comparison_protocol import DEFAULT_PROTOCOL


def _channel(recall5, recall10, mrr, ndcg10, wrong5, p95):
    mode = {
        "metrics": {
            "recall@5": recall5,
            "recall@10": recall10,
            "mrr": mrr,
            "ndcg@10": ndcg10,
            "wrong_source_rate@5": wrong5,
        },
        "latency_ms": {"p95": p95},
    }
    return {"modes": {"scoped": mode, "full_library": mode}}


def test_screening_fixture_excludes_every_confirmation_case() -> None:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    fixture = {
        "id": "full",
        "cases": [
            {"id": case_id}
            for case_id in (
                protocol["inner_split"]["screening_case_ids"]
                + protocol["inner_split"]["confirmation_case_ids"]
            )
        ],
    }
    screened = _screening_fixture(fixture, protocol)
    observed = {case["id"] for case in screened["cases"]}
    assert observed == set(protocol["inner_split"]["screening_case_ids"])
    assert not observed & set(protocol["inner_split"]["confirmation_case_ids"])


def test_screening_gates_require_safe_and_resource_invariants() -> None:
    passing = evaluate_screening_gates(
        base_dense=_channel(0.4, 0.5, 0.2, 0.25, 0.3, 20),
        product_sparse=_channel(0.6, 0.62, 0.4, 0.45, 0.0, 2),
        candidate_dense=_channel(0.5, 0.6, 0.3, 0.35, 0.2, 50),
        safe_rrf=_channel(0.6, 0.7, 0.42, 0.48, 0.0, 55),
        index_size_ratio=1.3,
    )
    assert passing["passes_all"] is True
    failing = evaluate_screening_gates(
        base_dense=_channel(0.4, 0.5, 0.2, 0.25, 0.3, 20),
        product_sparse=_channel(0.6, 0.62, 0.4, 0.45, 0.0, 2),
        candidate_dense=_channel(0.5, 0.49, 0.3, 0.35, 0.2, 120),
        safe_rrf=_channel(0.6, 0.61, 0.39, 0.44, 0.0, 125),
        index_size_ratio=1.6,
    )
    assert failing["passes_all"] is False


def test_nomination_is_eligible_only_and_deterministic() -> None:
    product = _channel(0.6, 0.6, 0.4, 0.4, 0.1, 2)

    def report(candidate_id, model_id, gain, eligible):
        return {
            "candidate": {"candidate_id": candidate_id, "model_id": model_id},
            "channels": {
                PRODUCT_CHANNEL: product,
                SAFE_CHANNEL: _channel(0.6, 0.6 + gain, 0.4, 0.4 + gain, 0.1, 2),
                CANDIDATE_CHANNEL: _channel(0.4, 0.5, 0.2, 0.2, 0.2, 30),
            },
            "screening_gate": {"passes_all": eligible},
        }

    decision = select_nominee(
        [
            report("rejected", "a/model", 0.5, False),
            report("lower", "b/model", 0.02, True),
            report("winner", "c/model", 0.04, True),
        ]
    )
    assert decision["nominee_candidate_id"] == "winner"
    assert decision["eligible_candidate_ids"] == ["lower", "winner"]
    assert decision["confirmation_allowed"] is True


def test_candidate_screening_reports_are_pinned_and_rejected() -> None:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    batch = json.loads(DEFAULT_BATCH.read_text(encoding="utf-8"))
    report_names = (
        "retrieval_candidate_screening_granite_embedding_278m_multilingual_v1.json",
        "retrieval_candidate_screening_qwen3_embedding_0_6b_v1.json",
    )
    reports = [
        json.loads((DEFAULT_PROTOCOL.parent / name).read_text(encoding="utf-8"))
        for name in report_names
    ]
    for report in reports:
        assert validate_screening_report(report, protocol=protocol, batch=batch) == []
        assert report["decision"]["status"] == "rejected"
        assert report["decision"]["confirmation_accessed"] is False
    report = reports[1]
    report["channels"][CANDIDATE_CHANNEL]["modes"]["scoped"]["metrics"][
        "recall@10"
    ] = 1.0
    errors = validate_screening_report(report, protocol=protocol, batch=batch)
    assert any("measured recall@10 changed" in error for error in errors)
    assert any("screening gate arithmetic changed" in error for error in errors)


def test_nomination_report_pins_zero_nominee_and_candidate_reports() -> None:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    batch = json.loads(DEFAULT_BATCH.read_text(encoding="utf-8"))
    reports = []
    for candidate in batch["candidates"]:
        path = DEFAULT_PROTOCOL.parent / f"retrieval_candidate_screening_{candidate['candidate_id']}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        assert validate_screening_report(report, protocol=protocol, batch=batch) == []
        reports.append(report)
    nomination = json.loads(DEFAULT_NOMINATION_REPORT.read_text(encoding="utf-8"))
    assert _validate_nomination_report(nomination, reports) == []
    assert nomination["decision"] == {
        "eligible_candidate_ids": [],
        "nominee_candidate_id": "",
        "confirmation_allowed": False,
        "status": "no_eligible_candidate",
    }
    nomination["decision"]["confirmation_allowed"] = True
    errors = _validate_nomination_report(nomination, reports)
    assert any("nomination decision changed" in error for error in errors)
