from __future__ import annotations

import hashlib
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

from kb import runtime_state as RUNTIME
from kb.bg_queue_state import (
    begin_next_task_or_idle as bg_begin_next_task_or_idle,
    cancel_all as bg_cancel_all,
    enqueue as bg_enqueue,
    finish_task as bg_finish_task,
    remove_queued_tasks_for_pdf as bg_remove_queued_tasks_for_pdf,
    should_cancel as bg_should_cancel,
    snapshot as bg_snapshot,
    update_page_progress as bg_update_page_progress,
)
from kb.chat_store import ChatStore
from kb.file_ops import _resolve_md_output_paths
from kb.llm import DeepSeekChat
from kb.pdf_tools import run_pdf_to_md
from kb.retrieval_engine import (
    _deep_read_md_for_context,
    _group_hits_by_doc_for_refs_fast,
    _search_hits_with_fallback,
    _top_heading,
)
from kb.retrieval_heuristics import _is_probably_bad_heading, _quick_answer_for_prompt
from kb.store import load_all_chunks
from kb.retriever import BM25Retriever
from ui.chat_widgets import _normalize_math_markdown
from ui.strings import S

_LIVE_ASSISTANT_PREFIX = "__KB_LIVE_TASK__:"
_CITE_SINGLE_BRACKET_RE = re.compile(
    r"(?<!\[)\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*(\d{1,4})\s*\](?!\])",
    re.IGNORECASE,
)
_CITE_SID_ONLY_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*\]\]",
    re.IGNORECASE,
)

# Backward-compat for long-lived Streamlit processes that loaded older runtime_state.
if not hasattr(RUNTIME, "BG_LOCK"):
    RUNTIME.BG_LOCK = threading.Lock()
if not hasattr(RUNTIME, "BG_STATE"):
    RUNTIME.BG_STATE = {
        "queue": [],
        "running": False,
        "done": 0,
        "total": 0,
        "current": "",
        "cur_page_done": 0,
        "cur_page_total": 0,
        "cur_page_msg": "",
        "cancel": False,
        "last": "",
    }

_BG_STATE = RUNTIME.BG_STATE
_BG_LOCK = RUNTIME.BG_LOCK


def _cite_source_id(source_path: str) -> str:
    s = str(source_path or "").strip()
    if not s:
        return "s0000000"
    return "s" + hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:8]

def _live_assistant_text(task_id: str) -> str:
    return f"{_LIVE_ASSISTANT_PREFIX}{str(task_id or '').strip()}"

def _is_live_assistant_text(text: str) -> bool:
    return str(text or "").strip().startswith(_LIVE_ASSISTANT_PREFIX)

def _live_assistant_task_id(text: str) -> str:
    s = str(text or "").strip()
    if not s.startswith(_LIVE_ASSISTANT_PREFIX):
        return ""
    return s[len(_LIVE_ASSISTANT_PREFIX) :].strip()

def _gen_get_task(session_id: str) -> dict | None:
    sid = (session_id or "").strip()
    if not sid:
        return None
    with RUNTIME.GEN_LOCK:
        t = RUNTIME.GEN_TASKS.get(sid)
        return dict(t) if isinstance(t, dict) else None

def _gen_update_task(session_id: str, task_id: str, **patch) -> None:
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
        nxt["updated_at"] = time.time()
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

def _gen_mark_cancel(session_id: str, task_id: str) -> bool:
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
        cur2 = dict(cur)
        cur2["cancel"] = True
        cur2["stage"] = "canceled"
        cur2["updated_at"] = time.time()
        RUNTIME.GEN_TASKS[sid] = cur2
        return True

def _gen_store_answer(task: dict, answer: str) -> None:
    conv_id = str(task.get("conv_id") or "")
    chat_db = Path(str(task.get("chat_db") or "")).expanduser()
    chat_store = ChatStore(chat_db)
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

def _gen_store_partial(task: dict, partial: str) -> None:
    chat_db = Path(str(task.get("chat_db") or "")).expanduser()
    chat_store = ChatStore(chat_db)
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

def _gen_worker(session_id: str, task_id: str) -> None:
    task = _gen_get_task(session_id) or {}
    if str(task.get("id") or "") != str(task_id or ""):
        return

    _gen_update_task(session_id, task_id, status="running", stage="starting", started_at=time.time())

    try:
        conv_id = str(task.get("conv_id") or "")
        prompt = str(task.get("prompt") or "").strip()
        chat_db = Path(str(task.get("chat_db") or "")).expanduser()
        db_dir = Path(str(task.get("db_dir") or "")).expanduser()
        top_k = int(task.get("top_k") or 6)
        temperature = float(task.get("temperature") or 0.15)
        max_tokens = int(task.get("max_tokens") or 1200)
        deep_read = bool(task.get("deep_read"))
        settings_obj = task.get("settings_obj")
        chat_store = ChatStore(chat_db)

        if (not conv_id) or (not prompt):
            raise RuntimeError("invalid task")
        if _gen_should_cancel(session_id, task_id):
            raise RuntimeError("canceled")

        quick_answer = _quick_answer_for_prompt(prompt)
        if quick_answer is not None:
            try:
                umid0 = int(task.get("user_msg_id") or 0)
            except Exception:
                umid0 = 0
            if umid0 > 0:
                try:
                    chat_store.upsert_message_refs(
                        user_msg_id=umid0,
                        conv_id=conv_id,
                        prompt=prompt,
                        prompt_sig=str(task.get("prompt_sig") or ""),
                        hits=[],
                        scores=[],
                        used_query="",
                        used_translation=False,
                    )
                except Exception:
                    pass
            _gen_store_answer(task, quick_answer)
            _gen_update_task(session_id, task_id, status="done", stage="done", answer=quick_answer, partial=quick_answer, char_count=len(quick_answer), finished_at=time.time())
            return

        chunks = load_all_chunks(db_dir)
        retriever = BM25Retriever(chunks)

        _gen_update_task(session_id, task_id, stage="retrieve")
        hits_raw, scores_raw, used_query, used_translation = _search_hits_with_fallback(
            prompt,
            retriever,
            top_k=top_k,
            settings=settings_obj,
        )
        hits = _group_hits_by_top_heading(hits_raw, top_k=top_k)

        grouped_docs: list[dict] = []
        _gen_update_task(session_id, task_id, stage="refs")
        if (not getattr(retriever, "is_empty", False)) and prompt:
            try:
                grouped_docs = _group_hits_by_doc_for_refs_fast(hits_raw, top_k_docs=top_k)
            except Exception:
                grouped_docs = []

        try:
            umid = int(task.get("user_msg_id") or 0)
        except Exception:
            umid = 0
        if umid > 0:
            try:
                chat_store.upsert_message_refs(
                    user_msg_id=umid,
                    conv_id=conv_id,
                    prompt=prompt,
                    prompt_sig=str(task.get("prompt_sig") or ""),
                    hits=list(grouped_docs or []),
                    scores=list(scores_raw or []),
                    used_query=str(used_query or ""),
                    used_translation=bool(used_translation),
                )
            except Exception:
                pass

        _gen_update_task(session_id, task_id, stage="context", used_query=str(used_query or ""), used_translation=bool(used_translation), refs_done=True)

        ctx_parts: list[str] = []
        doc_first_idx: dict[str, int] = {}
        # Keep prompt compact for fast first-token latency.
        answer_hits = list(hits[: max(1, min(int(top_k), 4))])
        for i, h in enumerate(answer_hits, start=1):
            meta = h.get("meta", {}) or {}
            src = (meta.get("source_path", "") or "").strip()
            if src and src not in doc_first_idx:
                doc_first_idx[src] = i
            src_name = Path(src).name if src else ""
            top = meta.get("top_heading") or _top_heading(meta.get("heading_path", ""))
            top = "" if _is_probably_bad_heading(top) else top
            sid = _cite_source_id(src)
            header = f"[{i}] [SID:{sid}] {src_name or 'unknown'}" + (f" | {top}" if top else "")
            body = h.get("text", "") or ""
            ctx_parts.append(header + "\n" + body)

        deep_added = 0
        deep_docs = 0
        if deep_read and answer_hits:
            deep_budget_s = 4.0
            deep_begin = time.monotonic()
            q_fine = (used_query or prompt or "").strip()
            items = list(doc_first_idx.items())[:1]
            total = len(items)
            for n, (src, idx0) in enumerate(items, start=1):
                if _gen_should_cancel(session_id, task_id):
                    raise RuntimeError("canceled")
                if (time.monotonic() - deep_begin) >= deep_budget_s:
                    _gen_update_task(session_id, task_id, stage="deep-read skipped (timeout)")
                    break
                _gen_update_task(session_id, task_id, stage=f"deep-read {n}/{total}")
                p = Path(src)
                extras: list[dict] = []
                if q_fine:
                    extras.extend(_deep_read_md_for_context(p, q_fine, max_snippets=1, snippet_chars=900))
                if not extras:
                    continue
                deep_docs += 1
                seen_snip = set()
                extras2: list[str] = []
                for ex in sorted(extras, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True):
                    t = (ex.get("text") or "").strip()
                    if not t:
                        continue
                    k = hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest()[:12]
                    if k in seen_snip:
                        continue
                    seen_snip.add(k)
                    extras2.append(t)
                    if len(extras2) >= 1:
                        break
                if not extras2:
                    continue
                try:
                    base = ctx_parts[idx0 - 1]
                except Exception:
                    continue
                for t in extras2:
                    if t in base:
                        continue
                    base += "\n\n（深读补充定位：来自原文）\n" + t
                    deep_added += 1
                ctx_parts[idx0 - 1] = base

        _gen_update_task(session_id, task_id, deep_read_docs=int(deep_docs), deep_read_added=int(deep_added), stage="answer")
        ctx = "\n\n---\n\n".join(ctx_parts)

        system = (
            "你的名字是 π-zaya。\n"
            "如果用户问‘你是谁/你叫什么/你是谁开发的’之类的问题，统一回答：我是 P&I Lab 开发的 π-zaya。\n"
            "你是我的个人知识库助手。优先基于我提供的检索片段回答问题。\n"
            "规则：\n"
            "1) 如果检索片段存在：优先基于片段回答；需要引用时，用 [1] [2] 这样的编号标注。\n"
            "2) 如果检索片段为空：也要给出可用的通用回答，但开头必须写明‘未命中知识库片段’。\n"
            "3) 不要编造不存在的论文、公式、数据或结论。\n"
            "4) 不要输出‘参考定位/Top-K/引用列表’之类的额外段落（我会在页面里单独展示）。\n"
            "5) 数学公式输出格式：短的变量/符号用 $...$（行内）；较长的等式/推导用 $$...$$（行间）。不要用反引号包裹公式。\n"
        )
        system += (
            "\nStructured citation protocol:\n"
            "- Context headers contain [SID:<sid>] identifiers.\n"
            "- When citing paper references, MUST use [[CITE:<sid>:<ref_num>]].\n"
            "- Example: [[CITE:s1a2b3c4:24]] or [[CITE:s1a2b3c4:24]][[CITE:s1a2b3c4:25]].\n"
            "- Do NOT output free-form numeric citations like [24] / [2][4].\n"
            "- NEVER output malformed markers like [[CITE:<sid>]] or [CITE:<sid>] (missing ref_num).\n"
        )
        user = f"问题：\n{prompt}\n\n检索片段（含深读补充定位）：\n{ctx if ctx else '(无)'}\n"
        history = chat_store.get_messages(conv_id)
        hist = [m for m in history if m.get("role") in ("user", "assistant")][-10:]
        messages = [{"role": "system", "content": system}, *hist, {"role": "user", "content": user}]

        ds = DeepSeekChat(settings_obj)
        partial = ""
        streamed = False
        last_store_ts = 0.0
        last_store_len = 0
        try:
            for piece in ds.chat_stream(messages=messages, temperature=temperature, max_tokens=max_tokens):
                if _gen_should_cancel(session_id, task_id):
                    raise RuntimeError("canceled")
                partial += piece
                streamed = True
                _gen_update_task(session_id, task_id, stage="answer", partial=partial, char_count=len(partial))
                now = time.monotonic()
                # Reduce sqlite write frequency while still keeping crash-recovery checkpoints.
                if (
                    ((now - last_store_ts) >= 0.9 and (len(partial) - last_store_len) >= 48)
                    or (("\n\n" in piece) and (len(partial) - last_store_len) >= 120)
                ):
                    _gen_store_partial(task, partial)
                    last_store_ts = now
                    last_store_len = len(partial)
        except Exception:
            if streamed:
                if _gen_should_cancel(session_id, task_id):
                    raise RuntimeError("canceled")
            else:
                resp = ds.chat(messages=messages, temperature=temperature, max_tokens=max_tokens)
                partial = str(resp or "")
                _gen_update_task(session_id, task_id, stage="answer", partial=partial, char_count=len(partial))

        if _gen_should_cancel(session_id, task_id):
            answer = (str(partial or "").strip() + "\n\n（已停止生成）").strip() or "（已停止生成）"
            _gen_store_answer(task, answer)
            _gen_update_task(session_id, task_id, status="canceled", stage="canceled", answer=answer, partial=answer, char_count=len(answer), finished_at=time.time())
            return

        answer = _normalize_math_markdown(_strip_model_ref_section(_sanitize_structured_cite_tokens(partial or ""))).strip() or "（未返回文本）"
        _gen_store_answer(task, answer)
        _gen_update_task(session_id, task_id, status="done", stage="done", answer=answer, partial=answer, char_count=len(answer), finished_at=time.time())

    except Exception as e:
        if str(e) == "canceled":
            snap = _gen_get_task(session_id) or {}
            partial = str(snap.get("partial") or "").strip()
            answer = (partial + "\n\n（已停止生成）").strip() or "（已停止生成）"
            try:
                _gen_store_answer(task, answer)
            except Exception:
                pass
            _gen_update_task(session_id, task_id, status="canceled", stage="canceled", answer=answer, partial=answer, char_count=len(answer), finished_at=time.time())
            return

        err = S["llm_fail"].format(err=str(e))
        try:
            _gen_store_answer(task, err)
        except Exception:
            pass
        _gen_update_task(session_id, task_id, status="error", stage="error", error=str(e), answer=err, partial=err, char_count=len(err), finished_at=time.time())

def _gen_start_task(task: dict) -> bool:
    sid = str(task.get("session_id") or "").strip()
    tid = str(task.get("id") or "").strip()
    if (not sid) or (not tid):
        return False
    with RUNTIME.GEN_LOCK:
        cur = RUNTIME.GEN_TASKS.get(sid)
        if isinstance(cur, dict) and str(cur.get("status") or "") == "running":
            return False
        item = dict(task)
        item.setdefault("status", "running")
        item.setdefault("stage", "starting")
        item.setdefault("partial", "")
        item.setdefault("char_count", 0)
        item.setdefault("cancel", False)
        item.setdefault("created_at", time.time())
        item.setdefault("updated_at", time.time())
        RUNTIME.GEN_TASKS[sid] = item
    try:
        threading.Thread(target=_gen_worker, args=(sid, tid), daemon=True).start()
    except Exception:
        with RUNTIME.GEN_LOCK:
            cur = RUNTIME.GEN_TASKS.get(sid)
            if isinstance(cur, dict) and str(cur.get("id") or "") == tid:
                cur2 = dict(cur)
                cur2["status"] = "error"
                cur2["stage"] = "error"
                cur2["answer"] = "线程启动失败"
                cur2["finished_at"] = time.time()
                RUNTIME.GEN_TASKS[sid] = cur2
        return False
    return True

def _bg_enqueue(task: dict) -> None:
    if "_tid" not in task:
        task = dict(task)
        task["_tid"] = uuid.uuid4().hex
    bg_enqueue(_BG_STATE, _BG_LOCK, task)
    _bg_ensure_started()

def _bg_remove_queued_tasks_for_pdf(pdf_path: Path) -> int:
    """
    Remove queued (not running) conversion tasks for a given PDF.
    Returns removed count.
    """
    return bg_remove_queued_tasks_for_pdf(_BG_STATE, _BG_LOCK, pdf_path)

def _bg_cancel_all() -> None:
    bg_cancel_all(_BG_STATE, _BG_LOCK, "正在停止当前转换…")

def _bg_snapshot() -> dict:
    return bg_snapshot(_BG_STATE, _BG_LOCK)

def _bg_worker_loop() -> None:
    while True:
        task = bg_begin_next_task_or_idle(_BG_STATE, _BG_LOCK)

        if task is None:
            time.sleep(0.35)
            continue

        pdf = Path(task["pdf"])
        out_root = Path(task["out_root"])
        db_dir = Path(task.get("db_dir") or "").expanduser() if task.get("db_dir") else None
        no_llm = bool(task.get("no_llm", False))
        # Equation image fallback should be a last resort.
        # - For full_llm (quality-first), prefer editable/searchable LaTeX over screenshots.
        # - In no-LLM degraded runs, `kb/pdf_tools.run_pdf_to_md` will force-enable it to preserve fidelity.
        eq_image_fallback = bool(task.get("eq_image_fallback", False))
        replace = bool(task.get("replace", False))
        speed_mode = str(task.get("speed_mode", "balanced"))
        if speed_mode == "ultra_fast":
            # Keep VL/LLM path in ultra_fast; converter itself handles speed/quality tradeoff.
            # Forcing no_llm here causes a dramatic quality drop that does not match UI semantics.
            eq_image_fallback = False
        task_id = str(task.get("_tid") or "")

        try:
            md_folder = out_root / pdf.stem
            if replace and md_folder.exists():
                # Safety: only delete inside out_root
                try:
                    md_root = out_root.resolve()
                    target = md_folder.resolve()
                    if str(target).lower().startswith(str(md_root).lower()):
                        import shutil

                        shutil.rmtree(md_folder, ignore_errors=True)
                except Exception:
                    pass

            def _on_progress(page_done: int, page_total: int, msg: str = "") -> None:
                try:
                    bg_update_page_progress(_BG_STATE, _BG_LOCK, page_done, page_total, msg, task_id=task_id)
                except Exception:
                    pass

            def _should_cancel() -> bool:
                return bg_should_cancel(_BG_STATE, _BG_LOCK)

            ok, out_folder = run_pdf_to_md(
                pdf_path=pdf,
                out_root=out_root,
                no_llm=no_llm,
                keep_debug=False,
                eq_image_fallback=eq_image_fallback,
                progress_cb=_on_progress,
                cancel_cb=_should_cancel,
                speed_mode=speed_mode,
            )
            if ok:
                msg = f"OK: {out_folder}"
            else:
                txt = str(out_folder or "").strip().lower()
                msg = "CANCELLED" if txt == "cancelled" else f"FAIL: {out_folder}"

            # Auto-ingest can add noticeable latency in the conversion UI.
            # Skip it in ultra_fast mode to keep end-to-end time near the 5s target.
            do_auto_ingest = ok and bool(db_dir) and (speed_mode != "ultra_fast")
            if do_auto_ingest and db_dir:
                try:
                    ingest_py = Path(__file__).resolve().parent / "ingest.py"
                    _, md_main, md_exists = _resolve_md_output_paths(out_root, pdf)
                    if ingest_py.exists() and md_exists:
                        subprocess.run(
                            [os.sys.executable, str(ingest_py), "--src", str(md_main), "--db", str(db_dir), "--incremental"],
                            check=False,
                            capture_output=True,
                            text=True,
                        )
                        msg = f"OK+INGEST: {out_folder}"
                except Exception:
                    # Not fatal; conversion still succeeded.
                    pass
        except Exception as e:
            msg = f"FAIL: {e}"

        bg_finish_task(_BG_STATE, _BG_LOCK, msg, task_id=task_id)

def _bg_ensure_started() -> None:
    worker_ver = "2026-02-12.bg.v4"
    t = getattr(RUNTIME, "BG_THREAD", None)
    running_ver = str(getattr(RUNTIME, "BG_WORKER_VERSION", "") or "")
    if t is not None and t.is_alive():
        # Never interrupt an active conversion thread on app rerun/hot-reload.
        # Otherwise users observe "sudden stop" in the middle of conversion.
        if running_ver != worker_ver:
            try:
                RUNTIME.BG_WORKER_VERSION = worker_ver
            except Exception:
                pass
        return
    t = threading.Thread(target=_bg_worker_loop, daemon=True)
    RUNTIME.BG_THREAD = t
    RUNTIME.BG_WORKER_VERSION = worker_ver
    t.start()

def _build_bg_task(
    *,
    pdf_path: Path,
    out_root: Path,
    db_dir: Path,
    no_llm: bool,
    replace: bool = False,
    speed_mode: str = "balanced",
) -> dict:
    pdf = Path(pdf_path)
    mode = str(speed_mode)
    return {
        "_tid": uuid.uuid4().hex,
        "pdf": str(pdf),
        "out_root": str(out_root),
        "db_dir": str(db_dir),
        # no_llm is controlled only by user-selected mode "no_llm".
        # ultra_fast should remain VL-based (lower quality, but still LLM).
        "no_llm": bool(no_llm),
        # Default OFF across all normal modes; enable only explicitly.
        # In no-LLM runs we still force-enable it inside `run_pdf_to_md` for fidelity.
        "eq_image_fallback": False,
        "replace": bool(replace),
        "speed_mode": mode,
        "name": pdf.name,
    }

def _group_hits_by_top_heading(hits: list[dict], top_k: int) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    grouped: list[dict] = []
    for h in hits:
        meta = h.get("meta", {}) or {}
        src = (meta.get("source_path") or "").strip()
        top = _top_heading(meta.get("heading_path", ""))
        key = (src, top)
        if key in seen:
            continue
        seen.add(key)
        gh = dict(h)
        gh_meta = dict(meta)
        gh_meta["top_heading"] = top
        gh["meta"] = gh_meta
        grouped.append(gh)
        if len(grouped) >= max(1, int(top_k)):
            break
    return grouped

def _strip_model_ref_section(answer: str) -> str:
    if not answer:
        return answer
    # Common markers we used in prompts previously.
    for marker in ("可参考定位", "参考定位"):
        idx = answer.find(marker)
        if idx > 0:
            return answer[:idx].rstrip()
    return answer


def _sanitize_structured_cite_tokens(answer: str) -> str:
    s = str(answer or "")
    if not s:
        return s
    # Normalize accidental single-bracket form to canonical form expected by renderer.
    s = _CITE_SINGLE_BRACKET_RE.sub(lambda m: f"[[CITE:{m.group(1)}:{m.group(2)}]]", s)
    # Drop malformed sid-only tokens; they have no ref number and cannot be resolved.
    s = _CITE_SID_ONLY_RE.sub("", s)
    return s

