from __future__ import annotations

import kb.generation_agent_finalize_runtime as agent_finalize


def test_compact_agent_trace_limits_nested_rows_without_mutating_input() -> None:
    trace = {
        "verification": {"claims": [{"idx": idx} for idx in range(60)]},
        "steps": [
            {
                "tool": f"tool-{idx}",
                "output": {
                    "hits": [{"idx": n} for n in range(10)],
                    "references": [{"idx": n} for n in range(10)],
                },
            }
            for idx in range(12)
        ],
        "research_run": {
            "subtasks": [{"idx": idx} for idx in range(14)],
            "evidence_matrix": [{"idx": idx} for idx in range(14)],
        },
    }

    compact = agent_finalize._gen_compact_agent_trace(trace)

    assert len(compact["verification"]["claims"]) == 50
    assert len(compact["steps"]) == 10
    assert len(compact["steps"][0]["output"]["hits"]) == 8
    assert len(compact["steps"][0]["output"]["references"]) == 8
    assert len(compact["research_run"]["subtasks"]) == 12
    assert len(compact["research_run"]["evidence_matrix"]) == 12
    assert len(trace["verification"]["claims"]) == 60
    assert len(trace["steps"]) == 12
    assert len(trace["steps"][0]["output"]["hits"]) == 10


def test_build_agent_completion_payload_assembles_source_check_and_contract(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    def _summary(trace):
        calls.append(("summary", {"trace": trace}))
        return {"evidence_status": "grounded"}

    def _runtime_check(task, **kwargs):
        calls.append(("runtime", dict(kwargs)))
        return {
            "status": "passed",
            "source_summary": dict(kwargs.get("agent_source_summary") or {}),
        }

    def _contract(task, **kwargs):
        calls.append(("contract", dict(kwargs)))
        return {
            "source_summary": dict(kwargs.get("agent_source_summary") or {}),
            "runtime": dict(kwargs.get("answer_runtime_check") or {}),
        }

    monkeypatch.setattr(agent_finalize, "_gen_agent_source_summary", _summary)
    monkeypatch.setattr(agent_finalize, "_gen_answer_runtime_check", _runtime_check)
    monkeypatch.setattr(agent_finalize, "_gen_answer_contract", _contract)

    payload = agent_finalize._gen_build_agent_completion_payload(
        {"agent_mode": True},
        answer="Grounded answer.",
        answer_quality={"minimum_ok": True},
        agent_trace={"trace": "demo"},
        answer_mode="local_grounded",
        source_blend="local_grounded",
        runtime_repair={"changed": True, "reasons": ["debug_content_in_answer"]},
    )

    assert [name for name, _kwargs in calls] == ["summary", "runtime", "contract"]
    assert payload["agent_source_summary"] == {"evidence_status": "grounded"}
    assert payload["answer_runtime_check"]["status"] == "passed"
    assert payload["answer_contract"]["source_summary"] == {"evidence_status": "grounded"}
    assert payload["answer_contract"]["runtime"]["status"] == "passed"
    assert calls[1][1]["answer"] == "Grounded answer."
    assert calls[1][1]["runtime_repair"]["changed"] is True


def test_store_agent_trace_meta_compacts_trace_and_merges_source_summary() -> None:
    captured: dict[str, object] = {}

    class _FakeChatStore:
        def __init__(self, db_path):
            captured["db_path"] = str(db_path)

        def merge_message_meta(self, message_id: int, patch: dict) -> bool:
            captured["message_id"] = int(message_id)
            captured["patch"] = patch
            return True

    agent_finalize._gen_store_agent_trace_meta(
        {
            "agent_mode": True,
            "chat_db": "/tmp/chat.db",
            "assistant_msg_id": 42,
        },
        agent_trace={
            "verification": {"claims": [{"idx": idx} for idx in range(55)]},
            "steps": [{"output": {"hits": [{"idx": idx} for idx in range(9)]}}],
        },
        chat_store_cls=_FakeChatStore,
        agent_source_summary_builder=lambda trace: {
            "claim_count": len((trace.get("verification") or {}).get("claims") or []),
        },
    )

    patch = captured["patch"]
    trace = patch["agent_trace"]
    assert captured["message_id"] == 42
    assert patch["agent_mode"] == "research_agent"
    assert len(trace["verification"]["claims"]) == 50
    assert len(trace["steps"][0]["output"]["hits"]) == 8
    assert patch["agent_source_summary"] == {"claim_count": 50}
