from __future__ import annotations

import json

from kb.agent.runner import build_agent_trace_for_completed_answer
from kb.claim_verification_observability import record_production_claim_verification


def _events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_claim_verification_observation_is_aggregate_only(monkeypatch, tmp_path) -> None:
    path = tmp_path / "verification.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(path))

    assert record_production_claim_verification(
        evidence_status="needs_review",
        total_claim_count=3,
        supported_claim_count=2,
        unsupported_claim_count=1,
        duration_ms=4.25,
    )

    event = _events(path)[0]
    assert event["name"] == "pi_zaya.claim_verification"
    assert event["attributes"] == {
        "outcome_id": "claim_verification.needs_review",
        "status": "done",
        "duration_ms": 4.25,
        "verification_outcome_id": "needs_review",
        "total_claim_count": 3,
        "supported_claim_count": 2,
        "unsupported_claim_count": 1,
        "model_revision": "lexical_structural_v1",
    }


def test_existing_rag_trace_records_lexical_verifier_without_content(monkeypatch, tmp_path) -> None:
    path = tmp_path / "rag-verification.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(path))
    answer = "Private mechanism uses masked reconstruction [1]."
    hits = [
        {
            "text": "The private mechanism uses masked reconstruction.",
            "meta": {"source_path": "C:/private/library/paper.pdf"},
        }
    ]

    trace = build_agent_trace_for_completed_answer(
        "private question",
        answer,
        evidence_hits=hits,
        answer_mode="evidence_grounded",
    )

    claim_event = next(
        event for event in _events(path) if event["name"] == "pi_zaya.claim_verification"
    )
    assert trace["verification"]["total_claims"] == 1
    assert claim_event["attributes"]["model_revision"] == "lexical_structural_v1"
    stored_text = path.read_text(encoding="utf-8")
    assert "private question" not in stored_text
    assert "Private mechanism" not in stored_text
    assert "paper.pdf" not in stored_text


def test_non_local_answer_records_skipped_verification(monkeypatch, tmp_path) -> None:
    path = tmp_path / "skipped-verification.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(path))

    build_agent_trace_for_completed_answer(
        "general private question",
        "general private answer",
        evidence_hits=[],
        answer_mode="general_llm",
    )

    claim_event = next(
        event for event in _events(path) if event["name"] == "pi_zaya.claim_verification"
    )
    attributes = claim_event["attributes"]
    assert attributes["status"] == "skipped"
    assert attributes["verification_outcome_id"] == "not_applicable"
    assert attributes["fallback_reason_id"] == "non_local_answer"
    assert "model_revision" not in attributes
