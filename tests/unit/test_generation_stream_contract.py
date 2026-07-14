from __future__ import annotations

import json

from api.contracts.generation import (
    GENERATION_STREAM_EVENT_FIELDS,
    GenerationStreamEvent,
)


def _schema_properties() -> dict:
    try:
        schema = GenerationStreamEvent.model_json_schema()
    except Exception:  # pragma: no cover - pydantic v1 compatibility
        schema = GenerationStreamEvent.schema()
    return dict(schema.get("properties") or {})


def test_generation_stream_event_schema_keeps_versioned_field_set():
    props = _schema_properties()

    assert tuple(props.keys()) == GENERATION_STREAM_EVENT_FIELDS
    version_schema = props["stream_schema_version"]
    assert version_schema.get("const", version_schema.get("enum", [None])[0]) == 2


def test_generation_stream_event_missing_task_uses_complete_error_shape():
    payload = GenerationStreamEvent.missing_task("Generation failed.").as_dict()

    assert tuple(payload.keys()) == GENERATION_STREAM_EVENT_FIELDS
    assert payload["stream_schema_version"] == 2
    assert payload["stage"] == "error"
    assert payload["status"] == "error"
    assert payload["done"] is True
    assert payload["error"] == "not_found"
    assert payload["partial"] == "Generation failed."
    assert payload["answer"] == "Generation failed."
    assert payload["char_count"] == len("Generation failed.")
    assert payload["agent_trace"] == {}
    assert payload["answer_contract"] == {}


def test_generation_stream_event_public_projection_hides_internal_and_non_agent_fields():
    payload = GenerationStreamEvent.from_task(
        {
            "stage": "done",
            "status": "done",
            "char_count": 99,
            "answer_quality": {
                "minimum_ok": True,
                "citation_plan": {
                    "slots": [
                        {
                            "source_path": r"F:\private\library\paper.en.md",
                            "instruction": "internal citation placement instruction",
                        }
                    ]
                },
            },
            "paper_guide_debug": {"retrieval_mode": "internal"},
            "research_trace": {"trace_id": "internal"},
            "agent_trace": {"mode": "research_agent"},
            "answer_contract": {"schema_version": 1},
        },
        partial="public",
        answer="public",
        visible_text="public",
        include_internal_debug=False,
    ).as_dict()

    assert payload["done"] is True
    assert payload["char_count"] == len("public")
    assert payload["answer_quality"] == {}
    assert payload["paper_guide_debug"] == {}
    assert payload["research_trace"] == {}
    assert payload["agent_trace"] == {}
    assert payload["answer_contract"] == {}
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "citation_plan" not in serialized
    assert "F:\\\\private" not in serialized
    assert "internal citation placement instruction" not in serialized


def test_generation_stream_event_internal_agent_projection_exposes_agent_contracts():
    payload = GenerationStreamEvent.from_task(
        {
            "stage": "synthesizing",
            "status": "running",
            "agent_mode": True,
            "answer_contract_v1": True,
            "answer_quality": {
                "minimum_ok": True,
                "citation_plan": {"slots": [{"instruction": "internal"}]},
            },
            "paper_guide_debug": {"retrieval_mode": "internal"},
            "research_trace": {"trace_id": "internal"},
            "agent_trace": {"mode": "research_agent"},
            "agent_source_summary": {"kind": "local_kb"},
            "answer_runtime_check": {"status": "passed"},
            "answer_contract": {"schema_version": 1},
        },
        partial="",
        answer="agent answer",
        visible_text="agent answer",
        include_internal_debug=True,
    ).as_dict()

    assert payload["done"] is False
    assert payload["answer_contract_v1"] is True
    assert payload["answer_quality"]["citation_plan"]["slots"][0]["instruction"] == "internal"
    assert payload["paper_guide_debug"] == {"retrieval_mode": "internal"}
    assert payload["research_trace"] == {"trace_id": "internal"}
    assert payload["agent_trace"] == {"mode": "research_agent"}
    assert payload["agent_source_summary"] == {"kind": "local_kb"}
    assert payload["answer_runtime_check"] == {"status": "passed"}
    assert payload["answer_contract"] == {"schema_version": 1}
