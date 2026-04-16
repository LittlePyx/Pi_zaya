from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from kb import runtime_state as RUNTIME
from kb.chat_store import ChatStore
from kb.paper_guide_provenance import _build_paper_guide_answer_provenance


def _live_assistant_text(task_id: str, *, live_assistant_prefix: str) -> str:
    return f"{str(live_assistant_prefix or '')}{str(task_id or '').strip()}"


def _is_live_assistant_text(text: str, *, live_assistant_prefix: str) -> bool:
    return str(text or "").strip().startswith(str(live_assistant_prefix or ""))


def _live_assistant_task_id(text: str, *, live_assistant_prefix: str) -> str:
    prefix = str(live_assistant_prefix or "")
    s = str(text or "").strip()
    if not s.startswith(prefix):
        return ""
    return s[len(prefix) :].strip()


def _gen_get_task(session_id: str) -> dict | None:
    sid = (session_id or "").strip()
    if not sid:
        return None
    with RUNTIME.GEN_LOCK:
        task = RUNTIME.GEN_TASKS.get(sid)
        return dict(task) if isinstance(task, dict) else None


def _gen_update_task(session_id: str, task_id: str, *, time_module=time, **patch) -> None:
    sid = (session_id or "").strip()
    tid = (task_id or "").strip()
    if (not sid) or (not tid):
        return
    with RUNTIME.GEN_LOCK:
        cur = RUNTIME.GEN_TASKS.get(sid)
        if not isinstance(cur, dict):
            return
        if str(cur.get("id") or "") != tid:
            return
        nxt = dict(cur)
        nxt.update(patch)
        nxt["updated_at"] = time_module.time()
        RUNTIME.GEN_TASKS[sid] = nxt


def _gen_should_cancel(session_id: str, task_id: str) -> bool:
    sid = (session_id or "").strip()
    tid = (task_id or "").strip()
    if (not sid) or (not tid):
        return True
    with RUNTIME.GEN_LOCK:
        cur = RUNTIME.GEN_TASKS.get(sid)
        if not isinstance(cur, dict):
            return True
        if str(cur.get("id") or "") != tid:
            return True
        return bool(cur.get("cancel") or False)


def _gen_mark_cancel(session_id: str, task_id: str, *, time_module=time) -> bool:
    sid = (session_id or "").strip()
    tid = (task_id or "").strip()
    if (not sid) or (not tid):
        return False
    with RUNTIME.GEN_LOCK:
        cur = RUNTIME.GEN_TASKS.get(sid)
        if not isinstance(cur, dict):
            return False
        if str(cur.get("id") or "") != tid:
            return False
        if str(cur.get("status") or "") != "running":
            return False
        if bool(cur.get("answer_ready") or False):
            return False
        cur2 = dict(cur)
        cur2["cancel"] = True
        cur2["stage"] = "canceled"
        cur2["updated_at"] = time_module.time()
        RUNTIME.GEN_TASKS[sid] = cur2
        return True


def _gen_store_answer(task: dict, answer: str, *, chat_store_cls=ChatStore) -> None:
    conv_id = str(task.get("conv_id") or "")
    chat_db = Path(str(task.get("chat_db") or "")).expanduser()
    chat_store = chat_store_cls(chat_db)
    try:
        amid = int(task.get("assistant_msg_id") or 0)
    except Exception:
        amid = 0
    if amid > 0:
        ok = chat_store.update_message_content(amid, answer)
        if not ok:
            chat_store.append_message(conv_id, "assistant", answer)
    else:
        chat_store.append_message(conv_id, "assistant", answer)


def _gen_store_partial(task: dict, partial: str, *, chat_store_cls=ChatStore) -> None:
    chat_db = Path(str(task.get("chat_db") or "")).expanduser()
    chat_store = chat_store_cls(chat_db)
    try:
        amid = int(task.get("assistant_msg_id") or 0)
    except Exception:
        amid = 0
    if amid <= 0:
        return
    txt = str(partial or "").strip()
    if not txt:
        return
    try:
        chat_store.update_message_content(amid, txt)
    except Exception:
        pass


def _gen_store_answer_quality_meta(
    task: dict,
    *,
    answer_quality: dict | None,
    chat_store_cls=ChatStore,
) -> None:
    quality = dict(answer_quality or {})
    if not quality:
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
        chat_store.merge_message_meta(amid, {"answer_quality": quality})
    except Exception:
        pass


def _gen_store_paper_guide_contract_meta(
    task: dict,
    *,
    paper_guide_contracts: dict | None,
    chat_store_cls=ChatStore,
) -> None:
    contracts = dict(paper_guide_contracts or {})
    if not contracts:
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
        chat_store.merge_message_meta(amid, {"paper_guide_contracts": contracts})
    except Exception:
        pass


def _gen_store_answer_provenance(
    task: dict,
    *,
    answer: str,
    answer_hits: list[dict],
    support_resolution: list[dict] | None = None,
    primary_evidence: dict | None = None,
    chat_store_cls=ChatStore,
    build_answer_provenance=_build_paper_guide_answer_provenance,
) -> None:
    if not bool(task.get("paper_guide_mode")):
        return
    source_path = str(task.get("paper_guide_bound_source_path") or "").strip()
    if not source_path:
        return
    chat_db = Path(str(task.get("chat_db") or "")).expanduser()
    chat_store = chat_store_cls(chat_db)
    try:
        amid = int(task.get("assistant_msg_id") or 0)
    except Exception:
        amid = 0
    if amid <= 0:
        return
    provenance_kwargs = {
        "answer": answer,
        "answer_hits": list(answer_hits or []),
        "bound_source_path": source_path,
        "bound_source_name": str(task.get("paper_guide_bound_source_name") or "").strip(),
        "db_dir": task.get("db_dir"),
        "settings_obj": task.get("settings_obj"),
        "llm_rerank": bool(task.get("llm_rerank", True)),
        "support_resolution": list(support_resolution or []),
    }
    if isinstance(primary_evidence, dict) and primary_evidence:
        provenance_kwargs["primary_evidence"] = dict(primary_evidence)
    provenance = build_answer_provenance(
        **provenance_kwargs,
    )
    if not isinstance(provenance, dict):
        return
    try:
        chat_store.merge_message_meta(amid, {"provenance": provenance})
    except Exception:
        pass


def _gen_store_answer_provenance_fast(
    task: dict,
    *,
    answer: str,
    answer_hits: list[dict],
    support_resolution: list[dict] | None = None,
    primary_evidence: dict | None = None,
    store_answer_provenance=_gen_store_answer_provenance,
) -> None:
    task_copy = dict(task or {})
    # Inline LLM refinement is an optional optimization for paper-guide provenance mapping.
    # Keep it opt-in via env flag so unit tests and offline runs remain deterministic.
    try:
        inline_enabled = bool(int(str(os.environ.get("KB_PROVENANCE_INLINE_LLM", "0") or "0")))
    except Exception:
        inline_enabled = False
    settings_obj = task_copy.get("settings_obj")
    has_api_key = bool(settings_obj is not None and getattr(settings_obj, "api_key", None))
    task_copy["llm_rerank"] = bool(inline_enabled and has_api_key)
    store_kwargs = {
        "answer": answer,
        "answer_hits": answer_hits,
        "support_resolution": support_resolution,
    }
    if isinstance(primary_evidence, dict) and primary_evidence:
        store_kwargs["primary_evidence"] = dict(primary_evidence)
    store_answer_provenance(task_copy, **store_kwargs)


def _should_run_provenance_async_refine(task: dict, *, environ=None) -> bool:
    if not bool((task or {}).get("paper_guide_mode")):
        return False
    source_path = str((task or {}).get("paper_guide_bound_source_path") or "").strip()
    if not source_path:
        return False
    env = os.environ if environ is None else environ
    # If we've already enabled inline LLM refinement, do not schedule another LLM pass.
    try:
        inline_enabled = bool(int(str(env.get("KB_PROVENANCE_INLINE_LLM", "0") or "0")))
    except Exception:
        inline_enabled = False
    if inline_enabled:
        return False
    try:
        enabled = bool(int(str(env.get("KB_PROVENANCE_ASYNC_LLM", "1") or "1")))
    except Exception:
        enabled = True
    if not enabled:
        return False
    if not bool((task or {}).get("llm_rerank", True)):
        return False
    settings_obj = (task or {}).get("settings_obj")
    if settings_obj is None:
        return False
    return bool(getattr(settings_obj, "api_key", None))


def _gen_store_answer_provenance_async(
    task: dict,
    *,
    answer: str,
    answer_hits: list[dict],
    support_resolution: list[dict] | None = None,
    primary_evidence: dict | None = None,
    store_answer_provenance=_gen_store_answer_provenance,
    perf_log=None,
    threading_module=threading,
    time_module=time,
) -> None:
    task_copy = dict(task or {})
    task_copy["llm_rerank"] = True

    def _run() -> None:
        t0 = time_module.perf_counter()
        try:
            store_kwargs = {
                "answer": answer,
                "answer_hits": answer_hits,
                "support_resolution": support_resolution,
            }
            if isinstance(primary_evidence, dict) and primary_evidence:
                store_kwargs["primary_evidence"] = dict(primary_evidence)
            store_answer_provenance(task_copy, **store_kwargs)
            if callable(perf_log):
                perf_log("gen.provenance_async", elapsed=time_module.perf_counter() - t0, ok=1)
        except Exception as exc:
            if callable(perf_log):
                perf_log("gen.provenance_async", elapsed=time_module.perf_counter() - t0, ok=0, err=str(exc)[:120])

    try:
        threading_module.Thread(target=_run, daemon=True, name="kb_gen_provenance_async").start()
    except Exception:
        pass
