from __future__ import annotations

from kb.answer_contract import build_answer_contract_payload


def test_answer_contract_payload_merges_profile_policy_and_runtime_check():
    contract = build_answer_contract_payload(
        answer_quality={"answer_profile": "hybrid_synthesis", "minimum_ok": True},
        agent_source_summary={
            "kind": "local_plus_external",
            "label_key": "agent_trace_source_local_external",
            "label": "Local + external",
            "detail": "Local citations plus external context.",
            "source_blend": "hybrid_local_external",
            "should_show": True,
            "source_policy_payload": {
                "schema_version": 1,
                "kind": "local_plus_external",
                "source_blend": "hybrid_local_external",
                "answer_mode": "hybrid_local_external",
                "uses_local_knowledge_base": True,
                "uses_external_model": True,
                "requires_user_notice": True,
                "badge": {
                    "label_key": "agent_trace_source_local_external",
                    "label": "Local + external",
                    "detail": "Local citations plus external context.",
                    "should_show": True,
                },
            },
        },
        answer_runtime_check={
            "status": "passed",
            "summary": {
                "failed": [],
                "needs_review_count": 0,
                "profile": "hybrid_synthesis",
                "source_blend": "hybrid_local_external",
                "answer_mode": "hybrid_local_external",
            },
            "checks": {"large_internal_debug_payload": {"ok": True}},
        },
    )

    assert contract["schema_version"] == 1
    assert contract["answer_profile"] == "hybrid_synthesis"
    assert contract["source_policy_payload"]["kind"] == "local_plus_external"
    assert contract["source_summary"]["kind"] == "local_plus_external"
    assert contract["runtime_check"]["status"] == "passed"
    assert contract["runtime_check"]["answer_mode"] == "hybrid_local_external"
    assert contract["ui"]["source_badge"]["label_key"] == "agent_trace_source_local_external"
    assert "checks" not in contract["runtime_check"]


def test_answer_contract_payload_keeps_repair_summary_compact():
    contract = build_answer_contract_payload(
        answer_quality={"answer_profile": "general_api"},
        agent_source_summary={"kind": "general_api", "label": "Not from KB", "should_show": True},
        answer_runtime_check={
            "status": "needs_review",
            "summary": {"failed": ["main_answer_clutter"], "needs_review_count": 1},
            "repair": {
                "changed": True,
                "reasons": ["debug_content_removed"],
                "before": {"answer": "large hidden text"},
                "after": {"answer": "clean answer"},
            },
        },
    )

    assert contract["runtime_check"]["status"] == "needs_review"
    assert contract["runtime_check"]["failed"] == ["main_answer_clutter"]
    assert contract["runtime_check"]["repair"] == {
        "changed": True,
        "reasons": ["debug_content_removed"],
    }


def test_answer_contract_payload_returns_empty_for_empty_inputs():
    assert build_answer_contract_payload() == {}
