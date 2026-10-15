from __future__ import annotations

import json

import pytest

from kb.answer_finalization_outcome import record_answer_finalization_outcome
from kb.local_observability import (
    MIN_LOCAL_OBSERVATION_MAX_BYTES,
    LocalJsonlObservationRecorder,
    emit_local_observation,
    local_observability_contract,
    sanitize_observation_attributes,
)


def test_observation_allowlist_rejects_content_and_paths() -> None:
    with pytest.raises(ValueError, match="unapproved"):
        sanitize_observation_attributes(
            "pi_zaya.retrieval",
            {"status": "done", "query": "private paper question"},
        )
    with pytest.raises(ValueError, match="stable redacted identifier"):
        sanitize_observation_attributes(
            "pi_zaya.retrieval",
            {"status": "C:\\private\\paper.pdf"},
        )


def test_local_jsonl_recorder_writes_otel_aligned_redacted_span(tmp_path) -> None:
    path = tmp_path / "observations.jsonl"
    event = LocalJsonlObservationRecorder(path).emit(
        "pi_zaya.retrieval",
        {
            "outcome_id": "retrieval.ready",
            "status": "done",
            "duration_ms": 12.3456,
            "retrieval_channel_id": "bm25_product",
            "source_scope_id": "current_paper",
            "retrieval_hit_count": 6,
            "cache_hit": True,
        },
    )
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == event
    assert len(stored["trace_id"]) == 32
    assert len(stored["span_id"]) == 16
    assert stored["attributes"]["duration_ms"] == 12.346
    assert "query" not in stored["attributes"]


def test_local_observation_contract_types_every_field_and_forbids_raw_content() -> None:
    contract = local_observability_contract()
    typed_fields = {
        field
        for fields in contract["attribute_types"].values()
        for field in fields
    }
    allowed_fields = {
        field for fields in contract["events"].values() for field in fields
    }
    assert allowed_fields <= typed_fields
    assert not (allowed_fields & set(contract["privacy"]["prohibited_attribute_names"]))
    assert contract["sink"]["default_enabled"] is False
    assert contract["sink"]["external_export_enabled"] is False
    assert contract["schema_version"] == 2
    assert "pi_zaya.local_semantic_component" in contract["events"]


def test_local_jsonl_recorder_rotates_at_bounded_size(tmp_path) -> None:
    path = tmp_path / "bounded.jsonl"
    path.write_text("x" * (MIN_LOCAL_OBSERVATION_MAX_BYTES - 10), encoding="utf-8")
    recorder = LocalJsonlObservationRecorder(
        path,
        max_bytes=MIN_LOCAL_OBSERVATION_MAX_BYTES,
        backup_count=1,
    )

    recorder.emit("pi_zaya.retrieval", {"status": "done"})

    assert path.is_file()
    assert path.with_name("bounded.jsonl.1").stat().st_size == (
        MIN_LOCAL_OBSERVATION_MAX_BYTES - 10
    )
    assert path.stat().st_size < MIN_LOCAL_OBSERVATION_MAX_BYTES


def test_observation_is_disabled_without_explicit_local_path(monkeypatch) -> None:
    monkeypatch.delenv("KB_LOCAL_OBSERVABILITY_PATH", raising=False)
    assert emit_local_observation("pi_zaya.retrieval", {"status": "done"}) is False


def test_model_call_allowlist_rejects_prompt_response_and_model_name() -> None:
    for field in ("prompt", "response", "model"):
        with pytest.raises(ValueError, match="unapproved"):
            sanitize_observation_attributes(
                "pi_zaya.model_call",
                {"status": "done", field: "private value"},
            )


def test_finalization_records_only_compact_outcome(monkeypatch, tmp_path) -> None:
    path = tmp_path / "finalization.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(path))
    outcome = record_answer_finalization_outcome(
        {
            "minimum_ok": True,
            "claim_evidence": {
                "final_gate_applied": True,
                "minimum_ok": True,
                "repaired_citations": 1,
                "pass_timings_ms": {"initial": 4.2, "recheck": 2.1},
                "repairs": [{"claim": "private claim", "source_path": "C:/paper.pdf"}],
            },
        }
    )
    stored = path.read_text(encoding="utf-8")
    assert outcome["duration_ms"] == 6.3
    assert "private claim" not in stored
    assert "paper.pdf" not in stored
    assert json.loads(stored)["attributes"]["repair_action_count"] == 1
