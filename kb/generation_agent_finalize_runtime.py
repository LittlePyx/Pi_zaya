from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

from kb.agent.source_summary import build_agent_source_summary
from kb.answer_contract import build_answer_contract_payload
from kb.answer_runtime_check import build_answer_runtime_check, repair_answer_for_runtime_contract
from kb.chat_store import ChatStore


def _gen_compact_agent_trace(agent_trace: dict | None) -> dict:
    if not isinstance(agent_trace, dict):
        return {}
    trace = copy.deepcopy(agent_trace)
    verification = trace.get("verification")
    if isinstance(verification, dict) and isinstance(verification.get("claims"), list):
        verification["claims"] = verification["claims"][:50]
    steps = trace.get("steps")
    if isinstance(steps, list):
        for step in steps[:10]:
            if not isinstance(step, dict):
                continue
            output = step.get("output")
            if not isinstance(output, dict):
                continue
            for key in ("hits", "references", "guide", "comparisons", "claims"):
                if isinstance(output.get(key), list):
                    output[key] = output[key][:8]
        trace["steps"] = steps[:10]
    research_run = trace.get("research_run")
    if isinstance(research_run, dict):
        if isinstance(research_run.get("subtasks"), list):
            research_run["subtasks"] = research_run["subtasks"][:12]
        if isinstance(research_run.get("evidence_matrix"), list):
            research_run["evidence_matrix"] = research_run["evidence_matrix"][:12]
    return trace


def _gen_agent_source_summary(
    agent_trace: dict | None,
    *,
    source_summary_builder: Callable[[dict | None], dict] = build_agent_source_summary,
) -> dict:
    try:
        return source_summary_builder(agent_trace)
    except Exception:
        return {}


def _gen_answer_runtime_check(
    task: dict,
    *,
    answer: str,
    answer_quality: dict | None = None,
    agent_trace: dict | None = None,
    agent_source_summary: dict | None = None,
    answer_mode: str = "",
    source_blend: str = "",
    runtime_repair: dict | None = None,
    runtime_check_builder: Callable[..., dict] = build_answer_runtime_check,
) -> dict:
    if not bool(task.get("agent_mode")):
        return {}
    try:
        check = runtime_check_builder(
            answer=answer,
            answer_quality=answer_quality,
            agent_trace=agent_trace,
            agent_source_summary=agent_source_summary,
            answer_mode=answer_mode,
            source_blend=source_blend,
        )
        repair = dict(runtime_repair or {})
        if repair.get("changed") or repair.get("reasons"):
            check["repair"] = {
                "changed": bool(repair.get("changed")),
                "reasons": list(repair.get("reasons") or [])[:8],
                "before": dict(repair.get("before") or {}),
                "after": dict(repair.get("after") or {}),
            }
        return check
    except Exception:
        return {}


def _gen_answer_contract(
    task: dict,
    *,
    answer_quality: dict | None = None,
    agent_source_summary: dict | None = None,
    answer_runtime_check: dict | None = None,
    contract_builder: Callable[..., dict] = build_answer_contract_payload,
) -> dict:
    if not bool(task.get("agent_mode")):
        return {}
    try:
        return contract_builder(
            answer_quality=answer_quality,
            agent_source_summary=agent_source_summary,
            answer_runtime_check=answer_runtime_check,
        )
    except Exception:
        return {}


def _gen_repair_answer_runtime(
    task: dict,
    *,
    prompt: str,
    answer: str,
    answer_quality: dict | None = None,
    agent_trace: dict | None = None,
    agent_source_summary: dict | None = None,
    answer_mode: str = "",
    source_blend: str = "",
    repair_builder: Callable[..., dict] = repair_answer_for_runtime_contract,
) -> dict:
    if not bool(task.get("agent_mode")):
        return {"answer": str(answer or ""), "changed": False, "reasons": []}
    try:
        return repair_builder(
            answer=answer,
            query=prompt,
            answer_quality=answer_quality,
            agent_trace=agent_trace,
            agent_source_summary=agent_source_summary,
            answer_mode=answer_mode,
            source_blend=source_blend,
        )
    except Exception:
        return {"answer": str(answer or ""), "changed": False, "reasons": ["runtime_repair_error"]}


def _sync_runtime_repaired_answer_contracts(paper_guide_contracts: dict | None, *, answer: str) -> dict:
    contracts = dict(paper_guide_contracts or {})
    packet = contracts.get("render_packet") if isinstance(contracts.get("render_packet"), dict) else {}
    if not packet:
        return contracts
    packet = dict(packet)
    for key in ("answer_markdown", "rendered_body", "rendered_content", "copy_markdown", "copy_text"):
        if key in packet:
            packet[key] = str(answer or "").strip()
    contracts["render_packet"] = packet
    return contracts


def _gen_build_agent_completion_payload(
    task: dict,
    *,
    answer: str,
    answer_quality: dict | None = None,
    agent_trace: dict | None = None,
    answer_mode: str = "",
    source_blend: str = "",
    runtime_repair: dict | None = None,
) -> dict:
    agent_source_summary = _gen_agent_source_summary(agent_trace)
    answer_runtime_check = _gen_answer_runtime_check(
        task,
        answer=answer,
        answer_quality=answer_quality,
        agent_trace=agent_trace,
        agent_source_summary=agent_source_summary,
        answer_mode=answer_mode,
        source_blend=source_blend,
        runtime_repair=runtime_repair,
    )
    answer_contract = _gen_answer_contract(
        task,
        answer_quality=answer_quality,
        agent_source_summary=agent_source_summary,
        answer_runtime_check=answer_runtime_check,
    )
    return {
        "agent_source_summary": agent_source_summary,
        "answer_runtime_check": answer_runtime_check,
        "answer_contract": answer_contract,
    }


def _gen_store_agent_trace_meta(
    task: dict,
    *,
    agent_trace: dict | None,
    chat_store_cls=ChatStore,
    agent_source_summary_builder: Callable[[dict | None], dict] = _gen_agent_source_summary,
) -> None:
    if not bool(task.get("agent_mode")):
        return
    trace = _gen_compact_agent_trace(agent_trace)
    if not trace:
        return
    chat_db = Path(str(task.get("chat_db") or "")).expanduser()
    chat_store = chat_store_cls(chat_db)
    try:
        amid = int(task.get("assistant_msg_id") or 0)
    except Exception:
        amid = 0
    if amid <= 0:
        return
    try:
        meta = {"agent_trace": trace, "agent_mode": "research_agent"}
        source_summary = agent_source_summary_builder(trace)
        if source_summary:
            meta["agent_source_summary"] = source_summary
        chat_store.merge_message_meta(amid, meta)
    except Exception:
        pass
