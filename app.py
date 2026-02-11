# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import html
import base64
import mimetypes
import tkinter as tk
from tkinter import filedialog
import os
import subprocess
import shutil
import time
import threading
import uuid
from pathlib import Path
from typing import Optional
import re
from collections import Counter
import json
from dataclasses import replace

import streamlit as st
import streamlit.components.v1 as components

from kb.chat_store import ChatStore
from kb.chunking import chunk_markdown
from kb.config import load_settings
from kb.library_store import LibraryStore
from kb.llm import DeepSeekChat
from kb import runtime_state as RUNTIME
from kb.pdf_tools import PDF_META_EXTRACT_VERSION, PdfMetaSuggestion, build_base_name, ensure_dir, extract_pdf_meta_suggestion, open_in_explorer, run_pdf_to_md
from kb.prefs import load_prefs, save_prefs
from kb.retriever import BM25Retriever
from kb.store import load_all_chunks
from kb.tokenize import tokenize

# Backward-compat for old runtime_state modules in long-lived Streamlit processes.
if not hasattr(RUNTIME, "GEN_LOCK"):
    RUNTIME.GEN_LOCK = threading.Lock()
if not hasattr(RUNTIME, "GEN_TASKS"):
    RUNTIME.GEN_TASKS = {}


# Keep source ASCII-stable: use unicode escapes for UI strings to avoid Windows encoding issues.
S = {
    "title": "\u03c0-zaya \u00b7 \u4f60\u7684\u77e5\u8bc6\u5e93\u52a9\u7406",
    "settings": "\u8bbe\u7f6e",
    "db_path": "DB \u8def\u5f84",
    "top_k": "\u68c0\u7d22 Top-K",
    "temp": "\u6e29\u5ea6",
    "max_tokens": "\u6700\u5927\u8f93\u51fa tokens",
    "show_ctx": "\u663e\u793a\u7247\u6bb5\u5168\u6587",
    "deep_read": "\u6df1\u8bfb MD\uff08\u66f4\u51c6\uff0c\u7a0d\u6162\uff09",
    "llm_rerank": "LLM \u8bed\u4e49\u91cd\u6392\uff08\u66f4\u51c6\uff0c\u7a0d\u6162\uff09",
    "reload_db": "\u91cd\u65b0\u52a0\u8f7d DB",
    "clear_chat": "\u6e05\u7a7a\u5bf9\u8bdd",
    "history": "\u5bf9\u8bdd\u8bb0\u5f55",
    "new_chat": "\u65b0\u5efa\u5bf9\u8bdd",
    "pick_chat": "\u9009\u62e9\u5bf9\u8bdd",
    "del_chat": "\u5220\u9664\u5f53\u524d\u5bf9\u8bdd",
    "chat": "\u5bf9\u8bdd",
    "input": "\u8f93\u5165",
    "prompt_label": "\u95ee\u70b9\u4ec0\u4e48\u2026\uff08\u4f1a\u5148\u68c0\u7d22\u4f60\u7684 Markdown \u518d\u56de\u7b54\uff09",
    "send": "\u53d1\u9001",
    "thinking": "\u601d\u8003\u4e2d...",
    "refs": "\u53c2\u8003\u5b9a\u4f4d",
    "no_msgs": "\u8fd8\u6ca1\u6709\u6d88\u606f\uff0c\u4e0b\u9762\u8f93\u5165\u95ee\u9898\u5f00\u59cb\u5427\u3002",
    "kb_empty": "DB \u91cc\u8fd8\u6ca1\u6709 chunks\uff1a{db}\u3002\u8bf7\u5230\u300c\u6587\u732e\u7ba1\u7406\u300d\u9875\uff1a\u8bbe\u7f6e PDF \u76ee\u5f55 \u2192 \u8f6c\u6362\u6210 MD \u2192 \u70b9\u51fb\u300c\u66f4\u65b0\u77e5\u8bc6\u5e93\u300d\uff08\u6216\u8fd0\u884c ingest.py\uff09\u3002",
    "llm_fail": "\uff08\u8c03\u7528\u6a21\u578b\u5931\u8d25\uff1a{err}\uff09\n\n\u8bf7\u68c0\u67e5 DEEPSEEK_API_KEY / BASE_URL / MODEL \u662f\u5426\u6b63\u786e\u3002",
    "page": "\u9875\u9762",
    "page_chat": "\u5bf9\u8bdd",
    "page_library": "\u6587\u732e\u7ba1\u7406",
    # "page_convert" removed: conversion is integrated into the library page.
    "lib_root": "\u6587\u732e\u76ee\u5f55\uff08PDF\uff09",
    "md_root": "\u8f93\u51fa\u76ee\u5f55\uff08Markdown\uff09",
    "open_dir": "\u6253\u5f00\u76ee\u5f55",
    "upload_pdf": "\u4e0a\u4f20 PDF",
    "save_pdf": "\u4fdd\u5b58 PDF",
    "convert_now": "\u7acb\u5373\u8f6c\u6362",
    "reindex_now": "\u66f4\u65b0\u77e5\u8bc6\u5e93",
    "cleanup": "\u6e05\u7406\u4e34\u65f6\u6587\u4ef6",
    "convert_opts": "\u8f6c\u6362\u9009\u9879",
    "no_llm": "\u4e0d\u7528 LLM\uff08\u66f4\u5feb\uff09",
    "batch_upload": "\u652f\u6301\u6279\u91cf\u4e0a\u4f20",
    "dup_found": "\u68c0\u6d4b\u5230\u91cd\u590d\u6587\u4ef6\uff1a\u4f60\u4e0a\u4f20\u7684 PDF \u548c\u5df2\u6709\u6587\u732e\u5185\u5bb9\u76f8\u540c\u3002",
    "dup_path": "\u5df2\u5b58\u5728\uff1a",
    "dup_skip": "\u8df3\u8fc7\uff08\u4e0d\u4fdd\u5b58\uff09",
    "dup_force": "\u4ecd\u7136\u4fdd\u5b58\u4e3a\u65b0\u526f\u672c\uff08\u4e0d\u63a8\u8350\uff09",
    "name_rule": "\u6587\u4ef6\u547d\u540d\uff1a\u671f\u520a-\u5e74\u4efd-\u6587\u732e\u540d\uff08\u53ef\u7f16\u8f91\uff09",
    "venue": "\u671f\u520a/\u4f1a\u8bae",
    "year": "\u5e74\u4efd",
    "title_field": "\u6587\u732e\u6807\u9898",
    "base_name": "\u5efa\u8bae\u6587\u4ef6\u540d\u524d\u7f00",
    "saved_as": "\u5df2\u4fdd\u5b58\u4e3a",
    "run_ok": "\u5b8c\u6210",
    "run_fail": "\u5931\u8d25",
    "handled_skip": "\u5df2\u8df3\u8fc7\uff08\u672a\u4fdd\u5b58\uff09",
    "handled_saved": "\u5df2\u4fdd\u5b58",
    "handled_converted": "\u5df2\u8f6c\u6362",
    "kb_miss": "\u672c\u6b21\u672a\u547d\u4e2d\u77e5\u8bc6\u5e93\u7247\u6bb5\uff0c\u56de\u7b54\u5c06\u4e3b\u8981\u57fa\u4e8e\u6a21\u578b\u901a\u7528\u77e5\u8bc6\u3002",
}



# Background conversion queue state is kept in an imported module.
# This survives Streamlit reruns more reliably than script-level globals.
_BG_LOCK = RUNTIME.BG_LOCK
_BG_STATE = RUNTIME.BG_STATE
_CACHE_LOCK = RUNTIME.CACHE_LOCK
_CACHE = RUNTIME.CACHE


def _cache_get(bucket: str, key: str):
    with _CACHE_LOCK:
        d = _CACHE.get(bucket) or {}
        return d.get(key)


def _cache_set(bucket: str, key: str, val, *, max_items: int = 600) -> None:
    with _CACHE_LOCK:
        d = _CACHE.setdefault(bucket, {})
        d[key] = val
        # Simple bound to avoid unbounded growth.
        if len(d) > int(max_items):
            try:
                # Drop about half (oldest insertion order in Py>=3.7 dict).
                for k in list(d.keys())[: max(1, len(d) // 2)]:
                    d.pop(k, None)
            except Exception:
                d.clear()


_LIVE_ASSISTANT_PREFIX = "__KB_LIVE_ASSISTANT__:"


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
            header = f"[{i}] {src_name or 'unknown'}" + (f" | {top}" if top else "")
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

        answer = _normalize_math_markdown(_strip_model_ref_section(partial or "")).strip() or "（未返回文本）"
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
    with _BG_LOCK:
        _BG_STATE["queue"].append(task)
        _BG_STATE["total"] = int(_BG_STATE.get("done", 0)) + len(_BG_STATE["queue"])


def _bg_remove_queued_tasks_for_pdf(pdf_path: Path) -> int:
    """
    Remove queued (not running) conversion tasks for a given PDF.
    Returns removed count.
    """
    p = str(Path(pdf_path))
    removed = 0
    with _BG_LOCK:
        q = list(_BG_STATE.get("queue") or [])
        kept = []
        for t in q:
            try:
                if str(t.get("pdf") or "") == p:
                    removed += 1
                else:
                    kept.append(t)
            except Exception:
                kept.append(t)
        _BG_STATE["queue"] = kept
        _BG_STATE["total"] = int(_BG_STATE.get("done", 0)) + len(_BG_STATE["queue"])
    return removed


def _bg_cancel_all() -> None:
    with _BG_LOCK:
        _BG_STATE["cancel"] = True
        _BG_STATE["cur_page_msg"] = "正在停止当前转换…"


def _bg_snapshot() -> dict:
    with _BG_LOCK:
        snap = dict(_BG_STATE)
        try:
            snap["queue"] = list(_BG_STATE.get("queue") or [])
        except Exception:
            snap["queue"] = []
        return snap


def _bg_worker_loop() -> None:
    while True:
        task = None
        with _BG_LOCK:
            if _BG_STATE.get("cancel"):
                _BG_STATE["queue"].clear()
                _BG_STATE["running"] = False
                _BG_STATE["current"] = ""
                _BG_STATE["cancel"] = False
                _BG_STATE["total"] = int(_BG_STATE.get("done", 0))

            if _BG_STATE["queue"]:
                task = _BG_STATE["queue"].pop(0)
                _BG_STATE["running"] = True
                _BG_STATE["current"] = str(task.get("name") or "")
                _BG_STATE["cur_page_done"] = 0
                _BG_STATE["cur_page_total"] = 0
                _BG_STATE["cur_page_msg"] = ""
            else:
                _BG_STATE["running"] = False
                _BG_STATE["current"] = ""
                _BG_STATE["cur_page_done"] = 0
                _BG_STATE["cur_page_total"] = 0
                _BG_STATE["cur_page_msg"] = ""

        if task is None:
            time.sleep(0.35)
            continue

        pdf = Path(task["pdf"])
        out_root = Path(task["out_root"])
        db_dir = Path(task.get("db_dir") or "").expanduser() if task.get("db_dir") else None
        no_llm = bool(task.get("no_llm", False))
        eq_image_fallback = bool(task.get("eq_image_fallback", True))
        replace = bool(task.get("replace", False))

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
                with _BG_LOCK:
                    try:
                        _BG_STATE["cur_page_done"] = int(page_done or 0)
                        _BG_STATE["cur_page_total"] = int(page_total or 0)
                        _BG_STATE["cur_page_msg"] = str(msg or "")[:220]
                    except Exception:
                        pass

            def _should_cancel() -> bool:
                with _BG_LOCK:
                    return bool(_BG_STATE.get("cancel"))

            ok, out_folder = run_pdf_to_md(
                pdf_path=pdf,
                out_root=out_root,
                no_llm=no_llm,
                keep_debug=False,
                eq_image_fallback=eq_image_fallback,
                progress_cb=_on_progress,
                cancel_cb=_should_cancel,
            )
            if ok:
                msg = f"OK: {out_folder}"
            else:
                txt = str(out_folder or "").strip().lower()
                msg = "CANCELLED" if txt == "cancelled" else f"FAIL: {out_folder}"

            # Auto-ingest the generated markdown so chat can retrieve it after DB reload.
            if ok and db_dir:
                try:
                    ingest_py = Path(__file__).resolve().parent / "ingest.py"
                    md_main = md_folder / f"{pdf.stem}.en.md"
                    if not md_main.exists() and md_folder.exists():
                        any_md = next(iter(sorted(md_folder.glob("*.md"))), None)
                        if any_md:
                            md_main = any_md
                    if ingest_py.exists() and md_main.exists():
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

        with _BG_LOCK:
            _BG_STATE["done"] = int(_BG_STATE.get("done", 0)) + 1
            _BG_STATE["total"] = int(_BG_STATE.get("done", 0)) + len(_BG_STATE["queue"])
            _BG_STATE["last"] = msg


def _bg_ensure_started() -> None:
    t = getattr(RUNTIME, "BG_THREAD", None)
    if t is not None and t.is_alive():
        return
    t = threading.Thread(target=_bg_worker_loop, daemon=True)
    RUNTIME.BG_THREAD = t
    t.start()


def _init_theme_css(theme_mode: str = "dark") -> None:
    mode = "dark" if str(theme_mode or "").lower() == "dark" else "light"
    color_scheme = "dark" if mode == "dark" else "light"

    if mode == "dark":
        tokens = """
  --bg: #1f1f1f;
  --panel: #252526;
  --sidebar-bg: #181818;
  --line: rgba(168, 176, 189, 0.42);
  --muted: #d2d9e4;
  --text-main: #e7eaef;
  --text-soft: #e0e7f0;
  --sidebar-strong-text: #e9eff8;
  --sidebar-soft-text: #d5deea;
  --slider-tick-text: #d9e1ec;
  --accent: #4daafc;
  --blue-weak: rgba(77, 170, 252, 0.18);
  --blue-line: rgba(77, 170, 252, 0.58);
  --font-display: "LittleP", "Segoe UI", "Microsoft YaHei", "PingFang SC", system-ui, -apple-system, sans-serif;
  --font-body: "Segoe UI", "Microsoft YaHei", "PingFang SC", system-ui, -apple-system, sans-serif;
  --btn-bg: #2d2d30;
  --btn-border: #45494f;
  --btn-text: #e7eaef;
  --btn-hover: #37373d;
  --btn-active: #3f444c;
  --btn-shadow: 0 1px 0 rgba(0, 0, 0, 0.32), 0 12px 30px rgba(0, 0, 0, 0.42);
  --input-bg: #1f2632;
  --input-border: #505a6d;
  --msg-user-bg: rgba(77, 170, 252, 0.22);
  --msg-user-border: rgba(127, 196, 255, 0.45);
  --msg-user-text: #eef6ff;
  --msg-ai-bg: #222934;
  --msg-ai-border: #3f4b5f;
  --snip-bg: rgba(148, 163, 184, 0.14);
  --snip-border: rgba(148, 163, 184, 0.34);
  --snip-text: #d7deea;
  --snip-quote-bg: rgba(77, 170, 252, 0.16);
  --snip-quote-border: rgba(77, 170, 252, 0.50);
  --snip-mark-bg: rgba(250, 204, 21, 0.28);
  --snip-mark-text: #f8fafc;
  --notice-text: #fde68a;
  --notice-bg: rgba(245, 158, 11, 0.20);
  --notice-border: rgba(245, 158, 11, 0.38);
  --ref-accent: rgba(77, 170, 252, 0.52);
  --dock-bg: linear-gradient(180deg, rgba(31, 31, 31, 0.72) 0%, rgba(31, 31, 31, 0.94) 20%, rgba(31, 31, 31, 0.98) 100%);
  --dock-border: rgba(148, 163, 184, 0.30);
  --dock-shadow: 0 -10px 28px rgba(0, 0, 0, 0.45);
  --copy-btn-bg: rgba(45, 45, 48, 0.94);
  --copy-btn-border: rgba(148, 163, 184, 0.34);
  --copy-btn-text: #dbe4f0;
  --toast-bg: rgba(36, 39, 45, 0.96);
  --toast-border: rgba(148, 163, 184, 0.30);
  --toast-text: #ebf1f8;
  --hint-text: #d2d9e4;
  --code-bg: #171d28;
  --code-border: #3d4658;
  --code-text: #e6edf3;
  --code-inline-bg: rgba(77, 170, 252, 0.14);
"""
    else:
        tokens = """
  --bg: #fcfcfd;
  --panel: #ffffff;
  --sidebar-bg: #f7f8fa;
  --line: rgba(90, 98, 112, 0.24);
  --muted: rgba(55, 65, 81, 0.76);
  --text-main: #1f2329;
  --text-soft: #4b5563;
  --sidebar-strong-text: #1f2329;
  --sidebar-soft-text: #5a6472;
  --slider-tick-text: #5a6472;
  --accent: #0f6cbd;
  --blue-weak: rgba(15, 108, 189, 0.10);
  --blue-line: rgba(15, 108, 189, 0.40);
  --font-display: "LittleP", "Segoe UI", "Microsoft YaHei", "PingFang SC", system-ui, -apple-system, sans-serif;
  --font-body: "Segoe UI", "Microsoft YaHei", "PingFang SC", system-ui, -apple-system, sans-serif;
  --btn-bg: #ffffff;
  --btn-border: rgba(31, 35, 41, 0.16);
  --btn-text: #1f2329;
  --btn-hover: rgba(15, 108, 189, 0.08);
  --btn-active: rgba(15, 108, 189, 0.14);
  --btn-shadow: 0 1px 0 rgba(16, 24, 40, 0.04), 0 10px 24px rgba(16, 24, 40, 0.06);
  --input-bg: #ffffff;
  --input-border: rgba(31, 35, 41, 0.18);
  --msg-user-bg: #eaf2ff;
  --msg-user-border: rgba(15, 108, 189, 0.24);
  --msg-user-text: #1f2a37;
  --msg-ai-bg: #ffffff;
  --msg-ai-border: rgba(49, 51, 63, 0.12);
  --snip-bg: rgba(49, 51, 63, 0.04);
  --snip-border: rgba(49, 51, 63, 0.12);
  --snip-text: rgba(15, 23, 42, 0.90);
  --snip-quote-bg: rgba(15, 108, 189, 0.08);
  --snip-quote-border: rgba(15, 108, 189, 0.30);
  --snip-mark-bg: rgba(251, 191, 36, 0.36);
  --snip-mark-text: #0f172a;
  --notice-text: rgba(120, 53, 15, 0.95);
  --notice-bg: rgba(245, 158, 11, 0.10);
  --notice-border: rgba(245, 158, 11, 0.20);
  --ref-accent: rgba(15, 108, 189, 0.24);
  --dock-bg: linear-gradient(180deg, rgba(252, 252, 253, 0.76) 0%, rgba(252, 252, 253, 0.96) 18%, rgba(252, 252, 253, 0.99) 100%);
  --dock-border: rgba(49, 51, 63, 0.12);
  --dock-shadow: 0 -8px 26px rgba(16, 24, 40, 0.08);
  --copy-btn-bg: rgba(255, 255, 255, 0.88);
  --copy-btn-border: rgba(49, 51, 63, 0.16);
  --copy-btn-text: rgba(31, 42, 55, 0.90);
  --toast-bg: rgba(255, 255, 255, 0.95);
  --toast-border: rgba(49, 51, 63, 0.16);
  --toast-text: rgba(31, 42, 55, 0.88);
  --hint-text: rgba(75, 85, 99, 0.62);
  --code-bg: #f5f7fb;
  --code-border: rgba(31, 35, 41, 0.16);
  --code-text: #1f2329;
  --code-inline-bg: rgba(15, 108, 189, 0.10);
"""

    css = """
<style>
:root{
__TOKENS__
  --text-color: var(--text-main);
  --secondary-text-color: var(--text-soft);
  --body-text-color: var(--text-main);
  --content-max: 1220px;
}
html, body{
  background: var(--bg) !important;
  color: var(--text-main) !important;
  color-scheme: __SCHEME__;
  font-family: var(--font-body);
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"]{
  background: var(--bg) !important;
}
header[data-testid="stHeader"]{
  background: var(--bg) !important;
  border-bottom: 1px solid var(--line) !important;
}
div[data-testid="stToolbar"],
div[data-testid="stStatusWidget"]{
  background: transparent !important;
}
div[data-testid="stToolbar"] button,
div[data-testid="stStatusWidget"] *{
  color: var(--text-soft) !important;
}
[data-stale="true"]{
  opacity: 1 !important;
  filter: none !important;
  transition: none !important;
}
body.kb-resizing [data-testid="stAppViewContainer"],
body.kb-resizing [data-testid="stAppViewContainer"]{
  opacity: 1 !important;
  filter: none !important;
}
body.kb-resizing [data-testid="stAppViewContainer"] *{
  filter: none !important;
}
body.kb-resizing [data-stale="true"]{
  opacity: 1 !important;
  filter: none !important;
  transition: none !important;
}
body.kb-resizing section[data-testid="stSidebar"]{
  background: var(--sidebar-bg) !important;
}
body.kb-resizing section[data-testid="stSidebar"] > div,
body.kb-resizing section[data-testid="stSidebar"] > div > div{
  background: var(--sidebar-bg) !important;
}
body.kb-resizing section[data-testid="stSidebar"] div[style*="z-index"]{
  background: transparent !important;
  opacity: 0 !important;
}
.block-container{
  width: 100%;
  max-width: var(--content-max);
  margin-left: auto !important;
  margin-right: auto !important;
  padding-top: 1.6rem;
  padding-bottom: 12.2rem;
}
section[data-testid="stSidebar"] > div:first-child{
  background: var(--sidebar-bg) !important;
  border-right: 1px solid var(--line) !important;
}
section[data-testid="stSidebar"]{
  background: var(--sidebar-bg) !important;
  --text-color: var(--sidebar-strong-text) !important;
  --secondary-text-color: var(--sidebar-soft-text) !important;
  --body-text-color: var(--sidebar-strong-text) !important;
}
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div > div{
  background: var(--sidebar-bg) !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="Close"],
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="关闭"]{
  width: 34px !important;
  min-width: 34px !important;
  height: 34px !important;
  min-height: 34px !important;
  padding: 0 !important;
  border-radius: 10px !important;
  border: 1px solid var(--btn-border) !important;
  background: color-mix(in srgb, var(--sidebar-bg) 76%, var(--panel)) !important;
  box-shadow: none !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  position: relative !important;
  font-size: 0 !important;
  line-height: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:hover,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="Close"]:hover,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="关闭"]:hover{
  background: var(--btn-hover) !important;
  border-color: var(--blue-line) !important;
  transform: none !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:active,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="Close"]:active,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="关闭"]:active{
  background: var(--btn-active) !important;
  border-color: var(--blue-line) !important;
  transform: none !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button svg,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="Close"] svg,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="关闭"] svg,
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button [data-testid="stIcon"],
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="Close"] [data-testid="stIcon"],
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="关闭"] [data-testid="stIcon"]{
  display: none !important;
}
section[data-testid="stSidebar"] .kb-close-glyph{
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  font-size: 24px !important;
  line-height: 1 !important;
  font-weight: 500 !important;
  color: var(--text-main) !important;
  transform: translateY(-1px);
  pointer-events: none !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] *{
  color: var(--sidebar-strong-text) !important;
  opacity: 1 !important;
}
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] *{
  color: var(--sidebar-soft-text) !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label *,
section[data-testid="stSidebar"] div[data-testid="stRadio"] p,
section[data-testid="stSidebar"] div[data-testid="stRadio"] span,
section[data-testid="stSidebar"] [role="radiogroup"] label,
section[data-testid="stSidebar"] [role="radiogroup"] label *,
section[data-testid="stSidebar"] [role="radiogroup"] p,
section[data-testid="stSidebar"] [role="radiogroup"] span{
  color: var(--text-main) !important;
  fill: var(--text-main) !important;
  opacity: 1 !important;
}
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] label,
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] label *,
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] p,
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] span{
  color: var(--text-main) !important;
  opacity: 1 !important;
}
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stTickBarMax"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderTickBarMax"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderTickBar"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stThumbValue"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderValue"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-baseweb="slider"] *,
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid*="TickBar"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid*="tick"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [class*="stSlider"] *,
section[data-testid="stSidebar"] div[data-testid="stSlider"] [style*="color"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] div[style*="color"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] small,
section[data-testid="stSidebar"] div[data-testid="stSlider"] p,
section[data-testid="stSidebar"] div[data-testid="stSlider"] span{
  color: var(--sidebar-soft-text) !important;
  -webkit-text-fill-color: var(--sidebar-soft-text) !important;
  fill: var(--sidebar-soft-text) !important;
  opacity: 1 !important;
}
section[data-testid="stSidebar"] div[data-testid="stSlider"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] > div,
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid*="TickBar"]{
  opacity: 1 !important;
}
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid*="TickBar"]::before,
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid*="TickBar"]::after{
  color: var(--sidebar-soft-text) !important;
  -webkit-text-fill-color: var(--sidebar-soft-text) !important;
  opacity: 1 !important;
}
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"],
div[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
div[data-testid="stSlider"] [data-testid="stSliderTickBarMax"],
div[data-testid="stSlider"] [data-testid="stSliderTickBar"],
div[data-testid="stSlider"] [data-testid*="TickBarMin"],
div[data-testid="stSlider"] [data-testid*="TickBarMax"],
div[data-testid="stSlider"] [class*="TickBarMin"],
div[data-testid="stSlider"] [class*="TickBarMax"],
div[data-testid="stSlider"] [class*="tickBarMin"],
div[data-testid="stSlider"] [class*="tickBarMax"],
div[data-testid="stSlider"] .stSliderTickBar,
div[data-testid="stSlider"] .stSliderTickBar *,
div[data-testid="stSlider"] [data-testid*="TickBar"]{
  color: var(--slider-tick-text) !important;
  -webkit-text-fill-color: var(--slider-tick-text) !important;
  fill: var(--slider-tick-text) !important;
  stroke: var(--slider-tick-text) !important;
  opacity: 1 !important;
  filter: brightness(1.12) contrast(1.08) !important;
}
div[data-testid="stSlider"] [data-testid*="ThumbValue"],
div[data-testid="stSlider"] [data-testid="stSliderValue"],
div[data-testid="stSlider"] [class*="ThumbValue"]{
  color: var(--accent) !important;
  -webkit-text-fill-color: var(--accent) !important;
  fill: var(--accent) !important;
  opacity: 1 !important;
}
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stThumbValue"]{
  color: var(--accent) !important;
  -webkit-text-fill-color: var(--accent) !important;
  fill: var(--accent) !important;
  opacity: 1 !important;
}
.stMarkdown .katex,
.stMarkdown .katex *,
.stMarkdown .katex-display,
.stMarkdown .katex-display *,
.msg-ai .katex,
.msg-ai .katex *,
.msg-ai .katex-display,
.msg-ai .katex-display *{
  color: var(--text-main) !important;
  fill: var(--text-main) !important;
  opacity: 1 !important;
}
.stMarkdown mjx-container,
.stMarkdown mjx-container *,
.msg-ai mjx-container,
.msg-ai mjx-container *{
  color: var(--text-main) !important;
  fill: currentColor !important;
  stroke: currentColor !important;
  opacity: 1 !important;
}
.kb-sidebar-logo-wrap{
  display: flex;
  justify-content: center;
  align-items: center;
  margin: -4.2rem 0 0.28rem 0;
}
.kb-sidebar-logo-img{
  width: 220px;
  max-width: 86%;
  height: auto;
  display: block;
  object-fit: contain;
  image-rendering: -webkit-optimize-contrast;
  image-rendering: crisp-edges;
  transform: translateZ(0);
}
h1, h2, h3, h4, h5{
  color: var(--text-main) !important;
  letter-spacing: -0.01em;
}
h1{
  font-family: var(--font-display);
  font-weight: 800;
}
p, li, td, th{ color: var(--text-main) !important; }
small, .stCaption, .msg-meta, .refbox, .genbox, .chat-empty-state{ color: var(--muted) !important; }
.kb-hero-title{
  margin: 0.18rem 0 0.66rem 0 !important;
  color: var(--text-main) !important;
  letter-spacing: -0.012em !important;
  line-height: 1.04 !important;
  font-size: clamp(2.12rem, 3.2vw, 3.05rem) !important;
  font-family: var(--font-display) !important;
  font-weight: 820 !important;
}
.kb-title-caret{
  display: inline-block;
  color: var(--blue-line);
  margin-left: 0.08rem;
  animation: kb-title-caret-blink 0.72s step-end infinite;
  font-weight: 700;
}
@keyframes kb-title-caret-blink{ 0%,100%{opacity:1;} 50%{opacity:0;} }
@media (prefers-reduced-motion: reduce){ .kb-title-caret{ animation: none !important; } }

div.stButton > button,
button[kind]{
  background: var(--btn-bg) !important;
  border: 1px solid var(--btn-border) !important;
  color: var(--btn-text) !important;
  border-radius: 12px !important;
  padding: 0.44rem 0.88rem !important;
  font-weight: 620 !important;
  box-shadow: 0 1px 0 rgba(16, 24, 40, 0.03);
}
section[data-testid="stSidebar"] div.stButton > button{ width: 100%; }
div.stButton > button:hover,
button[kind]:hover{
  background: var(--btn-hover) !important;
  border-color: var(--blue-line) !important;
  box-shadow: var(--btn-shadow);
}
div.stButton > button:active,
button[kind]:active{
  background: var(--btn-active) !important;
  border-color: var(--blue-line) !important;
}
div.stButton > button:focus,
div.stButton > button:focus-visible,
button[kind]:focus,
button[kind]:focus-visible{
  outline: none !important;
  box-shadow: 0 0 0 2px var(--blue-weak) !important;
}

textarea,
input,
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] [data-baseweb="select"] > div{
  background: var(--input-bg) !important;
  color: var(--text-main) !important;
  border: 1px solid var(--input-border) !important;
  border-radius: 12px !important;
  box-shadow: none !important;
}
ul[data-testid="stSelectboxVirtualDropdown"],
div[role="listbox"]{
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
}
li[data-testid="stSelectboxVirtualDropdownOption"],
div[role="option"]{ color: var(--text-main) !important; }

button[data-baseweb="tab"]{
  background: var(--btn-bg) !important;
  border: 1px solid var(--btn-border) !important;
  color: var(--text-main) !important;
  border-radius: 10px 10px 0 0 !important;
}
button[data-baseweb="tab"][aria-selected="true"]{
  background: var(--blue-weak) !important;
  border-color: var(--blue-line) !important;
}

details[data-testid="stExpander"]{
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
}
details[data-testid="stExpander"] summary,
details[data-testid="stExpander"] summary *{ color: var(--text-main) !important; }

[data-testid="stFileUploaderDropzone"]{
  background: var(--panel) !important;
  border: 1px dashed var(--input-border) !important;
  border-radius: 14px !important;
}
[data-testid="stFileUploaderDropzone"] *{ color: var(--text-soft) !important; }

pre{
  position: relative;
  border-radius: 12px !important;
  background: var(--code-bg) !important;
  border: 1px solid var(--code-border) !important;
  color: var(--code-text) !important;
  overflow: auto !important;
  box-shadow: none !important;
}
pre code{
  background: transparent !important;
  color: var(--code-text) !important;
  text-decoration: none !important;
  border: 0 !important;
  box-shadow: none !important;
}
pre span{
  background: transparent !important;
  text-decoration: none !important;
  border: 0 !important;
  box-shadow: none !important;
}
pre, pre *{
  text-decoration: none !important;
  background-image: none !important;
}
div[data-testid="stCodeBlock"],
div[data-testid="stCode"],
.stCodeBlock{
  background: var(--code-bg) !important;
  border: 1px solid var(--code-border) !important;
  border-radius: 12px !important;
  overflow: hidden !important;
}
div[data-testid="stCodeBlock"] > div,
div[data-testid="stCodeBlock"] pre,
div[data-testid="stCode"] > div,
div[data-testid="stCode"] pre,
.stCodeBlock > div,
.stCodeBlock pre{
  background: transparent !important;
  border: 0 !important;
  border-radius: 12px !important;
  color: var(--code-text) !important;
  box-shadow: none !important;
}
div[data-testid="stCodeBlock"] code,
div[data-testid="stCodeBlock"] pre code,
div[data-testid="stCode"] code,
div[data-testid="stCode"] pre code,
.stCodeBlock code,
.stCodeBlock pre code,
.stMarkdown div[data-testid="stMarkdownContainer"] pre code,
.stMarkdown pre code,
.msg-ai pre code{
  background: transparent !important;
  color: var(--code-text) !important;
}
div[data-testid="stCodeBlock"] span,
div[data-testid="stCode"] span,
.stCodeBlock span,
.stMarkdown div[data-testid="stMarkdownContainer"] pre span,
.stMarkdown pre span,
.msg-ai pre span{
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  text-decoration: none !important;
}
.stMarkdown :not(pre) > code,
.msg-ai :not(pre) > code{
  background: var(--code-inline-bg) !important;
  color: var(--code-text) !important;
  border: 1px solid var(--code-border) !important;
  border-radius: 6px !important;
  padding: 0.08em 0.32em !important;
}
.refbox code, .meta-kv{ color: var(--text-soft) !important; }
.hr{ height: 1px; background: var(--line); margin: 1rem 0; }
.pill{
  display: inline-flex;
  align-items: center;
  padding: 0.12rem 0.52rem;
  border-radius: 999px;
  font-size: 0.76rem;
  font-weight: 650;
  border: 1px solid transparent;
}
.pill.ok{
  background: rgba(34, 197, 94, 0.14);
  color: #22c55e !important;
  border-color: rgba(34, 197, 94, 0.34);
}
.pill.warn{
  background: rgba(245, 158, 11, 0.16);
  color: #f59e0b !important;
  border-color: rgba(245, 158, 11, 0.36);
}
.pill.run{
  background: var(--blue-weak);
  color: var(--accent) !important;
  border-color: var(--blue-line);
}

.msg-user{
  background: var(--msg-user-bg);
  border: 1px solid var(--msg-user-border);
  color: var(--msg-user-text) !important;
  border-radius: 16px;
  padding: 10px 14px;
  width: fit-content;
  max-width: min(820px, 88%);
  margin-left: auto;
}
.msg-ai{ background: transparent; border: none; max-width: min(900px, 94%); }
.msg-ai-stream{
  background: var(--msg-ai-bg);
  border: 1px solid var(--msg-ai-border);
  border-radius: 14px;
  padding: 12px 14px;
}
.msg-refs{
  margin: 0.35rem 0 0.80rem 0;
  padding: 0.06rem 0 0.80rem 0;
  border-left: none !important;
  outline: none !important;
  box-shadow: none !important;
}
.msg-refs::before,
.msg-refs::after{
  display: none !important;
  content: none !important;
}
.msg-refs details[data-testid="stExpander"]{
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
}
.msg-refs details[data-testid="stExpander"] summary,
.msg-refs details[data-testid="stExpander"] summary *,
.msg-refs details[data-testid="stExpander"] summary p,
.msg-refs details[data-testid="stExpander"] summary span{
  color: var(--text-main) !important;
  opacity: 1 !important;
  -webkit-text-fill-color: var(--text-main) !important;
}
.msg-refs [data-testid="stMarkdownContainer"] *,
.msg-refs .refbox,
.msg-refs .refbox *{
  color: var(--text-soft) !important;
  opacity: 1 !important;
}
.msg-refs details[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] *,
.msg-refs details[data-testid="stExpander"] summary p,
.msg-refs details[data-testid="stExpander"] summary span,
.msg-refs details[data-testid="stExpander"] summary div{
  color: var(--text-main) !important;
  -webkit-text-fill-color: var(--text-main) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] .msg-refs details[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] *,
html[data-theme="dark"] .msg-refs details[data-testid="stExpander"] summary p,
html[data-theme="dark"] .msg-refs details[data-testid="stExpander"] summary span,
html[data-theme="dark"] .msg-refs details[data-testid="stExpander"] summary div,
body[data-theme="dark"] .msg-refs details[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] *,
body[data-theme="dark"] .msg-refs details[data-testid="stExpander"] summary p,
body[data-theme="dark"] .msg-refs details[data-testid="stExpander"] summary span,
body[data-theme="dark"] .msg-refs details[data-testid="stExpander"] summary div{
  color: #e7eaef !important;
  -webkit-text-fill-color: #e7eaef !important;
  opacity: 1 !important;
}
html[data-theme="dark"] .msg-refs details[data-testid="stExpander"] summary,
html[data-theme="dark"] .msg-refs details[data-testid="stExpander"] summary *,
html[data-theme="dark"] .msg-refs details[data-testid="stExpander"] summary svg,
html[data-theme="dark"] .msg-refs details[data-testid="stExpander"] summary path,
body[data-theme="dark"] .msg-refs details[data-testid="stExpander"] summary,
body[data-theme="dark"] .msg-refs details[data-testid="stExpander"] summary *,
body[data-theme="dark"] .msg-refs details[data-testid="stExpander"] summary svg,
body[data-theme="dark"] .msg-refs details[data-testid="stExpander"] summary path{
  color: #e7eaef !important;
  -webkit-text-fill-color: #e7eaef !important;
  fill: #e7eaef !important;
  stroke: #e7eaef !important;
}
html[data-theme="dark"] .msg-refs details[data-testid="stExpander"] [data-testid="stMarkdownContainer"] *,
html[data-theme="dark"] .msg-refs .refbox,
html[data-theme="dark"] .msg-refs .refbox *,
body[data-theme="dark"] .msg-refs details[data-testid="stExpander"] [data-testid="stMarkdownContainer"] *,
body[data-theme="dark"] .msg-refs .refbox,
body[data-theme="dark"] .msg-refs .refbox *{
  color: #dbe4f0 !important;
  -webkit-text-fill-color: #dbe4f0 !important;
}
html[data-theme="light"] .msg-refs details[data-testid="stExpander"] summary,
html[data-theme="light"] .msg-refs details[data-testid="stExpander"] summary *,
body[data-theme="light"] .msg-refs details[data-testid="stExpander"] summary,
body[data-theme="light"] .msg-refs details[data-testid="stExpander"] summary *{
  color: #1f2329 !important;
  -webkit-text-fill-color: #1f2329 !important;
}
html[data-theme="light"] .msg-refs details[data-testid="stExpander"] [data-testid="stMarkdownContainer"] *,
html[data-theme="light"] .msg-refs .refbox,
html[data-theme="light"] .msg-refs .refbox *,
body[data-theme="light"] .msg-refs details[data-testid="stExpander"] [data-testid="stMarkdownContainer"] *,
body[data-theme="light"] .msg-refs .refbox,
body[data-theme="light"] .msg-refs .refbox *{
  color: #4b5563 !important;
  -webkit-text-fill-color: #4b5563 !important;
}

.snipbox{
  background: var(--snip-bg);
  border: 1px solid var(--snip-border);
  border-radius: 12px;
  padding: 10px 12px;
  margin: 0.35rem 0 0.55rem 0;
}
.snipbox pre{
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--snip-text) !important;
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
}
.snipquote{ border-left: 3px solid var(--snip-quote-border); background: var(--snip-quote-bg); border-radius: 10px; padding: 9px 11px; margin: 0.15rem 0 0.45rem 0; }
.snipquote .snipquote-title{ font-size: 0.78rem; color: var(--muted) !important; margin: 0 0 0.25rem 0; }
.snipquote .snipquote-body{ font-size: 0.88rem; line-height: 1.42; color: var(--text-main) !important; }
.snipquote mark{ background: var(--snip-mark-bg); color: var(--snip-mark-text); }

.kb-notice{
  font-size: 0.84rem;
  color: var(--notice-text) !important;
  background: var(--notice-bg);
  border: 1px solid var(--notice-border);
  border-radius: 10px;
  padding: 0.35rem 0.55rem;
  margin: 0 0 0.55rem 0;
}

.stTextArea{ position: relative; }
.stTextArea::after,
div[data-testid="stTextArea"]::after{
  content: "Ctrl+Enter 发送";
  position: absolute;
  right: 14px;
  bottom: 10px;
  font-size: 12px;
  color: var(--hint-text);
  pointer-events: none;
}

.kb-input-dock{
  position: fixed !important;
  bottom: max(10px, env(safe-area-inset-bottom, 0px));
  z-index: 40;
  left: 50%;
  transform: translateX(-50%);
  width: min(var(--content-max), calc(100vw - 1.6rem));
  box-sizing: border-box;
  background: var(--dock-bg) !important;
  border: 1px solid var(--dock-border) !important;
  border-radius: 20px;
  padding: 0.48rem 0.56rem 0.18rem 0.56rem;
  box-shadow: var(--dock-shadow);
  backdrop-filter: blur(5px);
  margin: 0 !important;
}
.kb-input-dock,
.kb-input-dock > div,
.kb-input-dock div[data-testid="stForm"]{
  background: var(--dock-bg) !important;
  border-color: var(--dock-border) !important;
}
.kb-input-dock textarea,
.kb-input-dock div[data-testid="stTextArea"] textarea{
  border-color: var(--input-border) !important;
}
.kb-input-dock textarea:focus,
.kb-input-dock div[data-testid="stTextArea"] textarea:focus{
  border-color: var(--blue-line) !important;
  box-shadow: 0 0 0 1px var(--blue-weak) !important;
}
html[data-theme="dark"] .kb-input-dock,
body[data-theme="dark"] .kb-input-dock{
  background: var(--dock-bg) !important;
  border-color: var(--dock-border) !important;
}
body.kb-resizing .kb-input-dock,
body.kb-resizing .kb-input-dock *{
  border-color: var(--dock-border) !important;
}
.kb-input-dock.kb-dock-positioned{ max-width: none !important; }
.kb-input-dock div[data-testid="stForm"]{ margin-bottom: 0 !important; }
.kb-input-dock::before{
  content: "问点什么...（会先检索你的 Markdown 再回答）";
  display: block;
  font-size: 0.75rem;
  color: var(--muted);
  margin: 0 0 0.32rem 0.16rem;
}
.kb-input-dock div[data-testid="stTextArea"] label{ display: none !important; }
.kb-input-dock textarea{ min-height: 92px !important; border-radius: 14px !important; }
.kb-input-dock div[data-testid="stFormSubmitButton"]{ display: flex !important; justify-content: flex-end !important; margin-top: 0.38rem !important; }
.kb-input-dock div[data-testid="stFormSubmitButton"] > button{
  width: 40px !important;
  min-width: 40px !important;
  height: 40px !important;
  min-height: 40px !important;
  border-radius: 999px !important;
  padding: 0 !important;
}

.kb-copybar{ display: flex; gap: 10px; align-items: center; margin: 6px 0 10px 0; }
.kb-copybtn{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 6px 10px;
  border-radius: 10px;
  border: 1px solid var(--copy-btn-border);
  background: var(--copy-btn-bg);
  color: var(--copy-btn-text);
  font-weight: 600;
  font-size: 12px;
  cursor: pointer;
}
.kb-copybtn:hover{ background: var(--btn-hover); border-color: var(--blue-line); }
.kb-copybtn:active{ background: var(--btn-active); border-color: var(--blue-line); }
.kb-codecopy{
  position: absolute;
  top: 10px;
  right: 10px;
  padding: 4px 8px;
  border-radius: 10px;
  border: 1px solid var(--copy-btn-border);
  background: var(--copy-btn-bg);
  color: var(--copy-btn-text);
  font-weight: 600;
  font-size: 12px;
  cursor: pointer;
  z-index: 2;
}
.kb-codecopy:hover{ background: var(--btn-hover); border-color: var(--blue-line); }
div[data-testid="stCodeBlock"] .kb-codecopy,
div[data-testid="stCode"] .kb-codecopy,
.stCodeBlock .kb-codecopy{
  display: none !important;
}
.kb-toast{
  position: fixed;
  right: 18px;
  bottom: 18px;
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid var(--toast-border);
  background: var(--toast-bg);
  color: var(--toast-text);
  font-weight: 600;
  font-size: 12px;
  opacity: 0;
  transform: translateY(6px);
  transition: opacity 120ms ease, transform 120ms ease;
  z-index: 999999;
  pointer-events: none;
}
.kb-toast.show{ opacity: 1; transform: translateY(0); }
html[data-theme="dark"] small,
html[data-theme="dark"] .stCaption,
html[data-theme="dark"] div[data-testid="stCaptionContainer"] *,
body[data-theme="dark"] small,
body[data-theme="dark"] .stCaption,
body[data-theme="dark"] div[data-testid="stCaptionContainer"] *{
  color: var(--text-soft) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] div[data-testid="stWidgetLabel"],
html[data-theme="dark"] div[data-testid="stWidgetLabel"] *,
body[data-theme="dark"] div[data-testid="stWidgetLabel"],
body[data-theme="dark"] div[data-testid="stWidgetLabel"] *{
  color: var(--text-main) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] *,
html[data-theme="dark"] section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
html[data-theme="dark"] section[data-testid="stSidebar"] p,
html[data-theme="dark"] section[data-testid="stSidebar"] span,
body[data-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] *,
body[data-theme="dark"] section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
body[data-theme="dark"] section[data-testid="stSidebar"] p,
body[data-theme="dark"] section[data-testid="stSidebar"] span{
  color: var(--text-main) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] div[data-testid="stRadio"] label,
html[data-theme="dark"] div[data-testid="stRadio"] label *,
html[data-theme="dark"] div[data-testid="stCheckbox"] label,
html[data-theme="dark"] div[data-testid="stCheckbox"] label *,
body[data-theme="dark"] div[data-testid="stRadio"] label,
body[data-theme="dark"] div[data-testid="stRadio"] label *,
body[data-theme="dark"] div[data-testid="stCheckbox"] label,
body[data-theme="dark"] div[data-testid="stCheckbox"] label *{
  color: var(--text-main) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] div[data-testid="stSlider"] label,
html[data-theme="dark"] div[data-testid="stSlider"] label *,
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stTickBarMin"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stTickBarMax"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderTickBarMax"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderTickBar"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stThumbValue"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderValue"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-baseweb="slider"] *,
body[data-theme="dark"] div[data-testid="stSlider"] label,
body[data-theme="dark"] div[data-testid="stSlider"] label *,
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stTickBarMin"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stTickBarMax"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderTickBarMax"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderTickBar"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stThumbValue"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderValue"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-baseweb="slider"] *{
  color: var(--text-soft) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] details[data-testid="stExpander"] summary,
html[data-theme="dark"] details[data-testid="stExpander"] summary *,
body[data-theme="dark"] details[data-testid="stExpander"] summary,
body[data-theme="dark"] details[data-testid="stExpander"] summary *{
  color: var(--text-main) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] [data-testid="stMarkdownContainer"] p,
html[data-theme="dark"] [data-testid="stMarkdownContainer"] li,
html[data-theme="dark"] [data-testid="stMarkdownContainer"] span,
body[data-theme="dark"] [data-testid="stMarkdownContainer"] p,
body[data-theme="dark"] [data-testid="stMarkdownContainer"] li,
body[data-theme="dark"] [data-testid="stMarkdownContainer"] span{
  color: var(--text-soft) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] .msg-meta,
html[data-theme="dark"] .refbox,
html[data-theme="dark"] .genbox,
html[data-theme="dark"] .chat-empty-state,
body[data-theme="dark"] .msg-meta,
body[data-theme="dark"] .refbox,
body[data-theme="dark"] .genbox,
body[data-theme="dark"] .chat-empty-state{
  color: var(--text-soft) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stSlider"] *,
body[data-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stSlider"] *{
  color: var(--text-soft) !important;
  fill: var(--text-soft) !important;
  stroke: var(--text-soft) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stSlider"] [style*="color"],
body[data-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stSlider"] [style*="color"]{
  color: var(--text-soft) !important;
  opacity: 1 !important;
}
</style>
<script>
(function () {
  const host = window.parent || window;
  const doc = host.document || document;
  const mode = "__MODE__";
  const FIX_TIMER_NS = "__kbSidebarSliderTextFixTimer";
  try {
    if (host[FIX_TIMER_NS]) {
      host.clearInterval(host[FIX_TIMER_NS]);
      host[FIX_TIMER_NS] = 0;
    }
  } catch (e) {}
  try {
    doc.documentElement.setAttribute("data-theme", mode);
    if (doc.body) doc.body.setAttribute("data-theme", mode);
  } catch (e) {}

  function paintNode(node, color) {
    if (!node || !node.style) return;
    try {
      node.style.setProperty("color", color, "important");
      node.style.setProperty("-webkit-text-fill-color", color, "important");
      node.style.setProperty("fill", color, "important");
      node.style.setProperty("stroke", color, "important");
      node.style.setProperty("opacity", "1", "important");
    } catch (e) {}
  }

  function applySidebarSliderTextFix() {
    try {
      if (mode !== "dark") return;
      const rootStyle = host.getComputedStyle(doc.documentElement);
      const soft = (rootStyle.getPropertyValue("--sidebar-soft-text") || "#d5deea").trim();
      const accent = (rootStyle.getPropertyValue("--accent") || "#4daafc").trim();
      const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
      if (!sidebar) return;
      const sliders = sidebar.querySelectorAll('div[data-testid="stSlider"]');
      for (const slider of sliders) {
        paintNode(slider, soft);
        const nodes = slider.querySelectorAll("*");
        for (const n of nodes) {
          const tid = String((n.getAttribute && n.getAttribute("data-testid")) || "");
          const t = String(n.textContent || "").trim();
          const isNumeric = /^-?\d+(\.\d+)?$/.test(t);
          const isTick =
            tid.toLowerCase().includes("tick") ||
            tid.toLowerCase().includes("min") ||
            tid.toLowerCase().includes("max");
          if (tid.toLowerCase().includes("thumbvalue")) {
            paintNode(n, accent);
          } else if (isTick || isNumeric) {
            paintNode(n, soft);
          }
        }
      }
    } catch (e) {}
  }

  applySidebarSliderTextFix();
  try {
    host[FIX_TIMER_NS] = host.setInterval(applySidebarSliderTextFix, 220);
  } catch (e) {}
})();
</script>
"""
    st.markdown(
        css.replace("__TOKENS__", tokens).replace("__SCHEME__", color_scheme).replace("__MODE__", mode),
        unsafe_allow_html=True,
    )

def _top_heading(heading_path: str) -> str:
    hp = (heading_path or "").strip()
    if not hp:
        return ""
    return hp.split(" / ", 1)[0].strip()


def _resolve_sidebar_logo_path() -> Path | None:
    base = Path(__file__).resolve().parent
    env_logo = (os.environ.get("KB_SIDEBAR_LOGO") or "").strip().strip("'\"")
    candidates: list[Path] = []
    if env_logo:
        p = Path(env_logo).expanduser()
        if not p.is_absolute():
            candidates.append((base / p).resolve())
        candidates.append(p.resolve())
    candidates.extend(
        [
            base / "team_logo.png",
            base / "assets" / "team_logo.png",
            base / "assets" / "team_logo.jpg",
            base / "assets" / "team_logo.jpeg",
            base / "assets" / "team_logo.webp",
        ]
    )
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue
    return None


def _sidebar_logo_data_uri(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
        if not raw:
            return None
        mime, _ = mimetypes.guess_type(str(path))
        if not mime:
            mime = "image/png"
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _render_app_title() -> None:
    title = str(S.get("title") or "").strip()
    if not title:
        return
    safe_title = html.escape(title)
    if bool(st.session_state.get("_hero_title_typed_once")):
        st.markdown(f"<h1 class='kb-hero-title'>{safe_title}</h1>", unsafe_allow_html=True)
        return

    holder = st.empty()
    acc: list[str] = []
    for ch in title:
        acc.append(ch)
        live = html.escape("".join(acc))
        holder.markdown(
            f"<h1 class='kb-hero-title'>{live}<span class='kb-title-caret'>▌</span></h1>",
            unsafe_allow_html=True,
        )
        time.sleep(0.020 if ord(ch) < 128 else 0.028)

    holder.markdown(f"<h1 class='kb-hero-title'>{safe_title}</h1>", unsafe_allow_html=True)
    st.session_state["_hero_title_typed_once"] = True


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


def _load_retriever(db_dir: Path) -> BM25Retriever:
    chunks = load_all_chunks(db_dir)
    return BM25Retriever(chunks)


def _render_kb_empty_hint(*, compact: bool = False) -> None:
    msg = "知识库还没有建好（DB 为空）。请到「文献管理」页：设置 PDF 目录 → 转换成 MD → 点击「更新知识库」。"
    if compact:
        st.caption(f"（{msg}）")
    else:
        c = st.columns([1.6, 10.4])
        with c[0]:
            if st.button("去文献管理", key="kb_empty_go_library"):
                st.session_state["page_radio"] = S["page_library"]
                st.experimental_rerun()
        with c[1]:
            st.info(msg)


def _normalize_math_markdown(text: str) -> str:
    """
    Make math rendering more stable in Streamlit markdown.

    Goals:
    - Inline math: $...$
    - Display math: $$...$$
    - Avoid code spans wrapping math (backticks break KaTeX/MathJax).
    """
    if not text:
        return text

    import re

    s = text

    # Prefer $...$ and $$...$$ over \( \) and \[ \]
    s = s.replace("\\(", "$").replace("\\)", "$")
    s = s.replace("\\[", "$$").replace("\\]", "$$")

    # Unwrap math that was mistakenly put in code spans.
    s = re.sub(r"`(\$[^`]+?\$)`", r"\1", s)
    s = re.sub(r"`(\$\$[\s\S]+?\$\$)`", r"\1", s)

    return s


def _md_to_plain_text(md: str) -> str:
    """
    Best-effort Markdown -> plain text for clipboard copy.
    Keeps formulas as their LaTeX source ($$...$$ / $...$).
    """
    if not md:
        return ""

    s = md
    # Remove code fences but keep their content.
    s = re.sub(r"```[^\n]*\n", "", s)
    s = s.replace("```", "")
    # Remove inline code backticks.
    s = re.sub(r"`([^`]+)`", r"\1", s)
    # Links: [text](url) -> text
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    # Images: ![alt](url) -> alt
    s = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", s)
    # Basic emphasis markers
    s = s.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
    # Headings/list markers
    s = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", s)
    s = re.sub(r"(?m)^\s*[-*+]\s+", "", s)
    s = re.sub(r"(?m)^\s*\d+\.\s+", "", s)
    # Collapse extra blank lines
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def _inject_copy_js() -> None:
    """
    Attach clipboard behaviors to:
    - Answer-level copy buttons (text / markdown)
    - Per-code-block copy buttons
    - Click-to-copy for LaTeX formulas rendered by KaTeX/MathJax (best effort)
    """
    components.html(
        r"""
<script>
(function () {
  const root = window.parent.document;
  const TOAST_ID = "kb_toast";

  function ensureToast() {
    let t = root.getElementById(TOAST_ID);
    if (!t) {
      t = root.createElement("div");
      t.id = TOAST_ID;
      t.className = "kb-toast";
      t.textContent = "已复制";
      root.body.appendChild(t);
    }
    return t;
  }

  function toast(msg) {
    const t = ensureToast();
    t.textContent = msg || "已复制";
    t.classList.add("show");
    clearTimeout(t._kbTimer);
    t._kbTimer = setTimeout(() => t.classList.remove("show"), 900);
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      toast("已复制");
      return true;
    } catch (e) {
      // Fallback: execCommand
      try {
        const ta = root.createElement("textarea");
        ta.value = text;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.left = "-9999px";
        root.body.appendChild(ta);
        ta.select();
        root.execCommand("copy");
        root.body.removeChild(ta);
        toast("已复制");
        return true;
      } catch (e2) {
        toast("复制失败");
        return false;
      }
    }
  }

  function hookCopyButtons() {
    const btns = root.querySelectorAll("button.kb-copybtn");
    for (const b of btns) {
      if (b.dataset.kbHooked === "1") continue;
      b.dataset.kbHooked = "1";
      b.addEventListener("click", async (e) => {
        e.preventDefault();
        const targetId = b.getAttribute("data-target");
        if (!targetId) return;
        const ta = root.getElementById(targetId);
        if (!ta) return;
        await copyText(ta.value || "");
      });
    }
  }

  function hookCodeBlocks() {
    function hasNativeCopy(pre) {
      if (!pre) return false;
      try {
        if (pre.closest('div[data-testid="stCodeBlock"], div[data-testid="stCode"], .stCodeBlock')) {
          return true;
        }
        const host = pre.parentElement;
        if (!host) return false;
        const nativeBtn = host.querySelector(
          'button[aria-label*="copy" i], button[title*="copy" i], button[aria-label*="复制"], button[title*="复制"], [data-testid*="copy" i]'
        );
        return !!nativeBtn;
      } catch (e) {
        return false;
      }
    }

    const pres = root.querySelectorAll("pre");
    for (const pre of pres) {
      if (hasNativeCopy(pre)) {
        const oldBtn = pre.querySelector(".kb-codecopy");
        if (oldBtn) {
          try { oldBtn.remove(); } catch (e) {}
        }
        pre.dataset.kbCodeHooked = "1";
        continue;
      }
      if (pre.dataset.kbCodeHooked === "1") continue;
      const code = pre.querySelector("code");
      if (!code) continue;
      pre.dataset.kbCodeHooked = "1";
      const btn = root.createElement("button");
      btn.className = "kb-codecopy";
      btn.type = "button";
      btn.textContent = "复制代码";
      btn.addEventListener("click", async (e) => {
        e.preventDefault();
        e.stopPropagation();
        await copyText(code.innerText || "");
      });
      pre.appendChild(btn);
    }
  }

  function extractTexFromKaTeX(node) {
    try {
      const ann = node.querySelector('annotation[encoding="application/x-tex"]');
      if (ann && ann.textContent) return ann.textContent;
    } catch (e) {}
    return null;
  }

  function hookMathClickToCopy() {
    const mathNodes = root.querySelectorAll(".katex, .MathJax, mjx-container");
    for (const n of mathNodes) {
      if (n.dataset && n.dataset.kbMathHooked === "1") continue;
      if (n.dataset) n.dataset.kbMathHooked = "1";
      n.style.cursor = "copy";
      n.addEventListener("click", async (e) => {
        // Prefer KaTeX annotation if available.
        const tex = extractTexFromKaTeX(n) || (n.innerText || "").trim();
        if (!tex) return;
        await copyText(tex);
        toast("已复制 LaTeX");
      });
    }
  }

  function tick() {
    hookCopyButtons();
    hookCodeBlocks();
    hookMathClickToCopy();
  }

  tick();
  setInterval(tick, 900);
})();
</script>
        """,
        height=0,
    )


def _inject_runtime_ui_fixes(theme_mode: str) -> None:
    mode = "dark" if str(theme_mode or "").lower() == "dark" else "light"
    components.html(
        f"""
<script>
(function () {{
  const host = window.parent || window;
  const doc = host.document || document;
  const KEY = "__kbSliderTickRuntimeFixV1";
  const mode = "{mode}";
  try {{
    if (host[KEY]) {{
      host.clearInterval(host[KEY]);
      host[KEY] = 0;
    }}
  }} catch (e) {{}}

  function paint(el, color) {{
    if (!el || !el.style) return;
    try {{
      el.style.setProperty("color", color, "important");
      el.style.setProperty("-webkit-text-fill-color", color, "important");
      el.style.setProperty("fill", color, "important");
      el.style.setProperty("stroke", color, "important");
      el.style.setProperty("opacity", "1", "important");
      el.style.setProperty("filter", "none", "important");
    }} catch (e) {{}}
  }}

  function apply() {{
    try {{
      const soft = mode === "dark" ? "#dbe4f0" : "#5a6472";
      const accent = mode === "dark" ? "#4daafc" : "#0f6cbd";
      const mainText = mode === "dark" ? "#e7eaef" : "#1f2329";
      const refsNodes = doc.querySelectorAll(
        '.msg-refs details[data-testid="stExpander"] summary, .msg-refs details[data-testid="stExpander"] summary *, .msg-refs [data-testid="stMarkdownContainer"], .msg-refs [data-testid="stMarkdownContainer"] *'
      );
      for (const n of refsNodes) {{
        if (!n || !n.style) continue;
        try {{
          n.style.removeProperty("color");
          n.style.removeProperty("-webkit-text-fill-color");
          n.style.removeProperty("fill");
          n.style.removeProperty("stroke");
          n.style.removeProperty("opacity");
          n.style.removeProperty("filter");
        }} catch (e) {{}}
      }}

      const sidebars2 = doc.querySelectorAll('section[data-testid="stSidebar"]');
      for (const sidebar of sidebars2) {{
        const sbRect = sidebar.getBoundingClientRect ? sidebar.getBoundingClientRect() : null;
        const btns = sidebar.querySelectorAll("button");
        for (const b of btns) {{
          const aria = String(b.getAttribute("aria-label") || "").toLowerCase();
          const rect = b.getBoundingClientRect ? b.getBoundingClientRect() : null;
          const nearTopRight = !!(sbRect && rect && rect.top <= (sbRect.top + 90) && rect.left >= (sbRect.right - 100));
          const maybeClose = aria.includes("close") || aria.includes("collapse") || aria.includes("关闭") || nearTopRight;
          if (!maybeClose) continue;

          try {{
            b.style.setProperty("width", "34px", "important");
            b.style.setProperty("height", "34px", "important");
            b.style.setProperty("min-width", "34px", "important");
            b.style.setProperty("min-height", "34px", "important");
            b.style.setProperty("padding", "0", "important");
            b.style.setProperty("font-size", "0", "important");
            b.style.setProperty("line-height", "0", "important");
            b.style.setProperty("display", "inline-flex", "important");
            b.style.setProperty("align-items", "center", "important");
            b.style.setProperty("justify-content", "center", "important");
          }} catch (e) {{}}

          const svgs = b.querySelectorAll("svg");
          for (const s of svgs) {{
            try {{ s.style.setProperty("display", "none", "important"); }} catch (e) {{}}
          }}
          const icons = b.querySelectorAll('[data-testid="stIcon"]');
          for (const ic of icons) {{
            try {{ ic.style.setProperty("display", "none", "important"); }} catch (e) {{}}
          }}

          let glyph = b.querySelector(".kb-close-glyph");
          if (!glyph) {{
            glyph = doc.createElement("span");
            glyph.className = "kb-close-glyph";
            glyph.textContent = "×";
            glyph.setAttribute("aria-hidden", "true");
            b.appendChild(glyph);
          }}
          paint(glyph, mainText);
        }}
      }}
    }} catch (e) {{}}
  }}

  apply();
  try {{
    host[KEY] = host.setInterval(apply, 420);
  }} catch (e) {{}}
}})();
</script>
        """,
        height=0,
    )


def _render_answer_copy_bar(answer_md: str, *, key_ns: str) -> None:
    md = _normalize_math_markdown(answer_md or "")
    txt = _md_to_plain_text(md)
    md_id = f"{key_ns}_md"
    txt_id = f"{key_ns}_txt"

    # Hidden payloads (large text lives here, buttons reference them by id).
    st.markdown(
        f"""
<textarea id="{html.escape(md_id)}" style="display:none">{html.escape(md)}</textarea>
<textarea id="{html.escape(txt_id)}" style="display:none">{html.escape(txt)}</textarea>
<div class="kb-copybar">
  <button class="kb-copybtn" type="button" data-target="{html.escape(txt_id)}">复制文本</button>
  <button class="kb-copybtn" type="button" data-target="{html.escape(md_id)}">复制 Markdown</button>
</div>
        """,
        unsafe_allow_html=True,
    )


def _lookup_pdf_by_stem(pdf_root: Path, stem: str) -> Path | None:
    """
    Best-effort mapping from a paper stem -> pdf path under pdf_root.

    Notes:
    - Avoid building a full index for very large folders: do an on-demand rglob
      and cache per stem in session_state.
    """
    stem = (stem or "").strip()
    if not stem:
        return None

    try:
        root = Path(pdf_root).expanduser().resolve()
    except Exception:
        root = Path(pdf_root)

    meta_key = "pdf_lookup_meta"
    cache_key = "pdf_lookup_cache"
    meta = st.session_state.get(meta_key) or {}
    if meta.get("root") != str(root):
        st.session_state[meta_key] = {"root": str(root)}
        st.session_state[cache_key] = {}

    cache = st.session_state.setdefault(cache_key, {})
    k = stem.lower()
    if k in cache:
        v = (cache.get(k) or "").strip()
        return Path(v) if v else None

    def _shorten(s: str, n: int) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else s[:n].rstrip()

    # Fast path: direct child (exact)
    try:
        direct = root / f"{stem}.pdf"
        if direct.exists() and direct.is_file():
            cache[k] = str(direct)
            return direct
    except Exception:
        pass

    # Fast path: direct child (prefix, handles "-2.pdf" etc.)
    try:
        direct_hits = sorted([p for p in root.glob(f"{stem}*.pdf") if p.is_file()])
        if direct_hits:
            cache[k] = str(direct_hits[0])
            return direct_hits[0]
    except Exception:
        pass

    # Slow path: recursive search (stop at first match) - exact then prefix.
    found: Path | None = None
    try:
        for p in root.rglob(f"{stem}.pdf"):
            try:
                if p.is_file():
                    found = p
                    break
            except Exception:
                continue
    except Exception:
        found = None

    if not found:
        try:
            pat = f"{stem}*.pdf"
            for p in root.rglob(pat):
                try:
                    if p.is_file():
                        found = p
                        break
                except Exception:
                    continue
        except Exception:
            found = None

    # Last resort: if stem is very long, search by a shorter prefix (Windows filenames often get a suffix like "-2").
    if not found and len(stem) > 80:
        short = _shorten(stem, 80)
        try:
            for p in root.rglob(f"{short}*.pdf"):
                try:
                    if p.is_file():
                        found = p
                        break
                except Exception:
                    continue
        except Exception:
            found = None

    cache[k] = str(found) if found else ""
    return found


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in (text or ""))


def _has_latin(text: str) -> bool:
    return any(("a" <= ch.lower() <= "z") for ch in (text or ""))


def _extract_keywords_for_desc(text: str, *, max_n: int = 4) -> list[str]:
    """
    Human-readable keywords for the relevance one-liner.
    Prefer CJK phrases (>=2) if present; otherwise use latin tokens.
    """
    t = (text or "").strip()
    if not t:
        return []
    # CJK phrases (clean question suffixes to keep it readable)
    cjk_phrases = re.findall(r"[\u4e00-\u9fff]{2,}", t)
    cleaned: list[str] = []
    for p in cjk_phrases:
        p = (p or "").strip()
        if not p:
            continue
        for suf in ("是什么", "是啥", "是什么意思", "怎么做", "怎么", "如何", "为什么", "是否", "可以吗", "可以", "以及", "原理", "原因"):
            if p.endswith(suf) and len(p) > len(suf):
                p = p[: -len(suf)].strip()
        # Split by common glue words to get shorter, clearer phrases
        parts = re.split(r"[的与和及以及并而或，。；、]", p)
        parts = [x.strip() for x in parts if x and len(x.strip()) >= 2]
        cleaned.extend(parts if parts else [p])
    cjk_phrases = cleaned
    if cjk_phrases:
        out: list[str] = []
        seen = set()
        for p in cjk_phrases:
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
            if len(out) >= max_n:
                break
        return out

    # Latin tokens
    toks = tokenize(t)
    out2: list[str] = []
    seen2 = set()
    for w in toks:
        w = (w or "").strip()
        if not w or len(w) <= 2:
            continue
        if w in seen2:
            continue
        seen2.add(w)
        out2.append(w)
        if len(out2) >= max_n:
            break
    return out2


def _quick_answer_for_prompt(prompt_text: str) -> str | None:
    """
    Fast local answers for trivial identity prompts.
    This avoids unnecessary queue stalls for basic "who are you" questions.
    """
    q = (prompt_text or "").strip().lower()
    if not q:
        return None
    keys = [
        "你是谁",
        "你叫什么",
        "你是谁开发的",
        "你是哪家开发的",
        "介绍一下你自己",
        "自我介绍",
        "who are you",
        "what is your name",
        "who made you",
        "who developed you",
    ]
    if any(k in q for k in keys):
        return "我是 P&I Lab 开发的 π-zaya。"
    return None


def _translate_query_for_search(settings, prompt_text: str) -> str | None:
    """
    Translate a CJK-only query to a compact English search query (keywords),
    so BM25 can match English papers.
    """
    q = (prompt_text or "").strip()
    if not q:
        return None
    if not _has_cjk(q) or _has_latin(q):
        return None
    if not getattr(settings, "api_key", None):
        return None

    key = hashlib.sha1((str(getattr(settings, "api_key", None)) + "|" + q).encode("utf-8", "ignore")).hexdigest()[:16]
    cached = _cache_get("trans", key)
    if isinstance(cached, str) and cached.strip():
        return cached.strip()

    # Fast dictionary-based fallback for common KB terms.
    # This keeps retrieval responsive even when translation LLM is slow/unavailable.
    terms: list[str] = []
    mapping = [
        ("单像素", "single-pixel"),
        ("单光子", "single-photon"),
        ("单曝光", "single-shot"),
        ("单次曝光", "single-shot"),
        ("压缩成像", "compressive imaging"),
        ("压缩感知", "compressed sensing"),
        ("小波", "wavelet"),
        ("成像", "imaging"),
        ("重建", "reconstruction"),
        ("采样", "sampling"),
        ("掩膜", "mask pattern"),
        ("编码", "coding"),
        ("光谱", "spectral"),
        ("超材料", "metamaterial"),
        ("超表面", "metasurface"),
        ("复用", "multiplexing"),
        ("频分", "frequency-division"),
    ]
    for zh, en_term in mapping:
        if zh in q:
            terms.append(en_term)
    if terms:
        uniq: list[str] = []
        seen = set()
        for t in terms:
            if t in seen:
                continue
            seen.add(t)
            uniq.append(t)
        heuristic = " ".join(uniq[:8]).strip()
        if heuristic:
            _cache_set("trans", key, heuristic, max_items=500)
            return heuristic

    try:
        settings_fast = replace(
            settings,
            timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), 8.0),
            max_retries=0,
        )
    except Exception:
        settings_fast = settings
    ds = DeepSeekChat(settings_fast)
    system = (
        "You translate Chinese research questions into an English search query for academic retrieval.\n"
        "Rules:\n"
        "- Output ONLY the English search query (no quotes, no explanations).\n"
        "- Prefer keywords and key phrases; include useful synonyms.\n"
        "- Keep it compact (8-18 tokens).\n"
        "- DO NOT confuse terms:\n"
        "  - 单曝光/单次曝光 -> single-shot, single exposure, snapshot\n"
        "  - 单像素 -> single-pixel\n"
        "  - 压缩成像 -> compressive imaging\n"
        "  - 光谱成像 -> spectral imaging\n"
        "- If the user didn't mention 单像素, avoid adding single-pixel.\n"
    )
    user = f"中文问题：{q}\n\n英文检索关键词："
    try:
        out = (ds.chat(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.0, max_tokens=80) or "").strip()
    except Exception:
        out = ""
    out = " ".join(out.split())
    _cache_set("trans", key, out, max_items=500)
    return out or None


def _norm_text_for_match(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[\s_]+", " ", s)
    return s


def _query_term_profile(prompt_text: str, used_query: str) -> dict[str, bool]:
    """
    Identify key intent terms to help rerank documents.
    """
    zh = (prompt_text or "")
    en = _norm_text_for_match(used_query or "")
    p = {
        "wants_single_shot": ("单曝光" in zh) or ("单次曝光" in zh) or ("single-shot" in en) or ("single shot" in en) or ("single exposure" in en) or ("snapshot" in en),
        "wants_single_pixel": ("单像素" in zh) or ("single-pixel" in en) or ("single pixel" in en),
        "wants_single_photon": ("单光子" in zh) or ("single photon" in en) or ("spad" in en) or ("sns" in en) or ("nanowire" in en),
        "wants_compressive": ("压缩" in zh) or ("compressive" in en),
        "wants_spectral": ("光谱" in zh) or ("spectral" in en),
    }
    return p


def _doc_term_bonus(profile: dict[str, bool], doc_name: str, snippets: list[str]) -> float:
    """
    Small but decisive boosts/penalties for term mismatches (e.g. single-shot vs single-pixel).
    """
    hay = _norm_text_for_match(doc_name or "") + "\n" + _norm_text_for_match("\n".join(snippets or []))
    has_single_shot = any(k in hay for k in ["single-shot", "single shot", "single exposure", "snapshot"])
    has_single_pixel = any(k in hay for k in ["single-pixel", "single pixel"])
    has_single_photon = any(k in hay for k in ["single-photon", "single photon", "spad", "sns", "snspd", "nanowire"])
    has_spectral = "spectral" in hay
    has_compressive = "compressive" in hay

    bonus = 0.0
    if profile.get("wants_single_shot"):
        if has_single_shot:
            bonus += 2.4
        if has_single_pixel and (not has_single_shot):
            bonus -= 2.8
    if profile.get("wants_single_pixel"):
        if has_single_pixel:
            bonus += 2.2
        if has_single_shot and (not has_single_pixel):
            bonus -= 1.6
        if has_single_photon and (not has_single_pixel):
            bonus -= 2.0
    if profile.get("wants_single_photon"):
        if has_single_photon:
            bonus += 2.0
        if has_single_pixel and (not has_single_photon):
            bonus -= 1.6
    if profile.get("wants_spectral"):
        bonus += 0.9 if has_spectral else -0.6
    if profile.get("wants_compressive"):
        bonus += 0.6 if has_compressive else -0.3
    return bonus


def _llm_semantic_rerank_score(settings, *, question: str, doc_headings: list[str], snippets: list[str]) -> tuple[float, str]:
    """
    LLM-based semantic relevance scoring, grounded on snippets (not filenames).

    Returns: (score_0_100, short_reason)
    """
    if not settings or (not getattr(settings, "api_key", None)):
        return 0.0, ""

    q = (question or "").strip()
    if not q:
        return 0.0, ""

    hs = [h for h in (doc_headings or []) if isinstance(h, str)]
    hs = [h for h in hs if h and (not _is_probably_bad_heading(h))][:25]

    sn = []
    for s in (snippets or [])[:4]:
        s2 = " ".join((s or "").strip().split())
        if len(s2) > 420:
            s2 = s2[:420].rstrip() + "…"
        if s2:
            sn.append(s2)
    if not sn:
        return 0.0, ""

    cache_key = hashlib.sha1(("\n".join(sn) + "|" + q).encode("utf-8", "ignore")).hexdigest()[:16]
    v0 = _cache_get("rerank", cache_key)
    if isinstance(v0, dict):
        try:
            return float(v0.get("score", 0.0) or 0.0), str(v0.get("why", "") or "")
        except Exception:
            return 0.0, ""

    ds = DeepSeekChat(settings)
    en = _has_latin(q) and (not _has_cjk(q))
    if en:
        sys = (
            "You are a strict academic retriever reranker.\n"
            "Output JSON ONLY: {\"score\":number,\"why\":string}.\n"
            "Rules:\n"
            "- score is 0..100 and MUST reflect how directly the provided snippets answer the question.\n"
            "- Penalize false friends (e.g., single-shot vs single-pixel) when mismatched.\n"
            "- Use only the snippets/headings; do NOT use filenames.\n"
            "- why: <= 18 words.\n"
        )
    else:
        sys = (
            "你是严格的学术检索重排器。\n"
            "只能输出 JSON：{\"score\":number,\"why\":string}。\n"
            "规则：\n"
            "- score 为 0..100，必须反映“这些片段是否直接回答用户问题”。\n"
            "- 遇到术语假朋友要扣分（如 single-shot vs single-pixel）。\n"
            "- 只能根据 snippets/headings 判断，不能根据文件名判断。\n"
            "- why：<= 18 个字，写清楚为什么。\n"
        )

    user = (
        f"问题：{q}\n\n"
        f"可用章节（仅供理解结构）：\n- " + "\n- ".join(hs) + "\n\n"
        f"片段：\n- " + "\n- ".join(sn) + "\n"
    )
    try:
        out = (ds.chat(messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}], temperature=0.0, max_tokens=160) or "").strip()
    except Exception:
        out = ""

    if out.startswith("```"):
        out = out.strip().strip("`")
        out = out.replace("json", "", 1).strip()

    try:
        data = json.loads(out)
    except Exception:
        data = None
    if not isinstance(data, dict):
        return 0.0, ""

    try:
        score = float(data.get("score", 0.0) or 0.0)
    except Exception:
        score = 0.0
    score = max(0.0, min(100.0, score))
    why = str(data.get("why") or "").strip()
    _cache_set("rerank", cache_key, {"score": score, "why": why}, max_items=600)
    return score, why


def _search_hits_with_fallback(
    prompt_text: str,
    retriever: BM25Retriever,
    top_k: int,
    settings,
    *,
    allow_translate: bool = True,
) -> tuple[list[dict], list[float], str, bool]:
    """
    Returns: (hits_raw, scores, used_query, used_translation)
    """
    q1 = (prompt_text or "").strip()
    hits1 = retriever.search(q1, top_k=max(10, top_k * 6)) if q1 else []
    scores1 = [float(h.get("score", 0.0) or 0.0) for h in hits1]
    best1 = float(max(scores1) if scores1 else 0.0)

    # If the query is CJK-only, BM25 over English corpora can return all-zeros (but still returns arbitrary docs).
    # Try translating to English to get meaningful retrieval.
    used_trans = False
    q2 = _translate_query_for_search(settings, q1) if bool(allow_translate) else None
    if q2:
        hits2 = retriever.search(q2, top_k=max(10, top_k * 6))
        scores2 = [float(h.get("score", 0.0) or 0.0) for h in hits2]
        best2 = float(max(scores2) if scores2 else 0.0)
        # Prefer translated retrieval when it yields a more meaningful signal.
        if (not hits1 and hits2) or (best1 <= 0.0 and best2 > 0.0) or (best2 > (best1 * 1.25 + 0.10)):
            return hits2, scores2, q2, True

    return hits1, scores1, q1, used_trans


def _is_probably_bad_heading(h: str) -> bool:
    s = " ".join((h or "").strip().split())
    if not s:
        return True
    if len(s) > 90:
        return True
    low = s.lower()
    # Metadata / copyright / pricing lines often leak into markdown as headings.
    bad_sub = (
        "received",
        "revised",
        "accepted",
        "publication date",
        "©",
        "copyright",
        "all rights reserved",
        "usd",
        "$",
        "doi:",
        "issn",
        "arxiv:",
        "download",
    )
    if any(x in low for x in bad_sub):
        return True
    # Too many digits usually means page labels / ids.
    digits = sum(ch.isdigit() for ch in s)
    if digits >= 10:
        return True
    return False


def _normalize_heading(h: str) -> str:
    s = " ".join((h or "").strip().split())
    if not s:
        return ""
    # Common "1 INTRODUCTION" -> keep as-is (it’s useful for users).
    return s


def _is_noise_snippet_text(t: str) -> bool:
    s = " ".join((t or "").strip().split())
    if not s:
        return True
    low = s.lower()
    # Boilerplate / metadata / copyright / submission timelines
    bad = (
        "received",
        "revised",
        "accepted",
        "publication date",
        "copyright",
        "all rights reserved",
        "creativecommons",
        "license",
        "doi:",
        "issn",
        "arxiv:",
        "$",
        "usd",
    )
    if any(x in low for x in bad):
        return True
    # Affiliations / author lists (comma-heavy lines)
    if s.count(",") >= 6 and len(s) > 120:
        return True
    if re.search(r"university|institute|department|laboratory|school of|college of", low) and len(s) > 80:
        return True
    return False


def _clean_snippet_for_display(t: str, *, max_chars: int = 900) -> str:
    """
    Display as plain text (no markdown rendering); keep line breaks but trim.
    """
    s = (t or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    # Drop pure image lines (common in converted markdown).
    try:
        s = "\n".join([ln for ln in s.split("\n") if not re.match(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$", ln)])
    except Exception:
        pass
    # Remove very long runs of whitespace
    s = re.sub(r"[ \t]{3,}", "  ", s)
    if len(s) > max_chars:
        s = s[:max_chars].rstrip() + "…"
    return s


def _preferred_section_keys(prompt_text: str) -> list[str]:
    t = (prompt_text or "").lower()
    # Prefer a stable set of section names to match.
    if re.search(r"(参考文献|引用|cite|citation|reference)", prompt_text or "", flags=re.I):
        return ["references", "bibliography"]
    if re.search(r"(是什么|定义|含义|概念|what is|definition)", prompt_text or "", flags=re.I):
        return ["abstract", "introduction", "overview", "background"]
    if re.search(r"(编码|掩膜|码|pattern|mask|sampling|coded|coding|hadamard|fourier|dmd|metamaterial|multiplex)", prompt_text or "", flags=re.I):
        return ["method", "approach", "model", "coding", "encoding", "pattern", "mask", "sampling", "implementation"]
    if re.search(r"(公式|推导|证明|算法|方法|实现|derive|equation|method|approach|model)", prompt_text or "", flags=re.I):
        return ["method", "approach", "model", "algorithm", "theory"]
    if re.search(r"(实验|结果|指标|性能|对比|消融|experiment|result|evaluation|ablation|baseline|compare)", prompt_text or "", flags=re.I):
        return ["experiment", "results", "evaluation", "implementation", "setup"]
    return ["method", "approach", "model", "experiment", "results", "introduction"]


def _pick_best_heading_for_doc(headings: list[tuple[float, str]], prompt_text: str) -> str:
    """
    Choose a high-confidence heading that is most likely where the user can find the answer.
    headings: (score, heading_text)
    """
    prefs = _preferred_section_keys(prompt_text)
    wants_refs = bool(re.search(r"(参考文献|引用|cite|citation|reference)", prompt_text or "", flags=re.I))
    best: tuple[float, str] | None = None
    for score, h in headings:
        hh = _normalize_heading(h)
        if _is_probably_bad_heading(hh):
            continue
        low = hh.lower()
        if ("references" in low or "bibliography" in low) and (not wants_refs):
            # Don't point users to References unless they explicitly asked for citations.
            continue
        bonus = 0.0
        for i, k in enumerate(prefs):
            if k in low:
                bonus += 1.5 - (i * 0.15)
                break
        # Slightly prefer shorter headings.
        bonus += max(0.0, (40 - len(hh)) / 200.0)
        v = float(score) + bonus
        if best is None or v > best[0]:
            best = (v, hh)
    return best[1] if best else ""


def _aspects_from_snippets(snippets: list[str], prompt_text: str) -> list[str]:
    """
    Extract a few concrete 'what you can find here' aspects based on matched text (not hallucinated).
    """
    text = "\n".join([s for s in snippets if s]).lower()
    out: list[str] = []

    def add(label: str, *keys: str) -> None:
        if label in out:
            return
        if any(k in text for k in keys):
            out.append(label)

    q = (prompt_text or "")
    wants_encoding = bool(re.search(r"(编码|掩膜|pattern|mask|sampling|hadamard|fourier|dmd|metamaterial|multiplex|coded|coding)", q, flags=re.I))
    wants_detector = bool(re.search(r"(单光子|探测器|spad|snspd|nanowire|dark count|jitter|quantum efficiency|detector)", q, flags=re.I))
    wants_compare = bool(re.search(r"(对比|比较|baseline|compare|ablation|消融)", q, flags=re.I))
    wants_metrics = bool(re.search(r"(指标|性能|效率|灵敏度|暗计数|jitter|psnr|ssim|efficiency|dcr)", q, flags=re.I))

    # If user clearly asks about detectors, prioritize detector aspects.
    if wants_detector:
        add("探测原理/工作机制", "spad", "avalanche", "geiger", "nanowire", "snspd", "superconduct", "transition edge", "tes")
        add("器件结构/材料体系", "semiconductor", "bulk", "low-dimensional", "perovskite", "superconduct", "nanowire", "material")
        add("关键指标与权衡", "dark count", "dcr", "time jitter", "jitter", "efficiency", "quantum efficiency", "snr", "time resolution")
        add("应用场景", "application", "lidar", "imaging", "communication", "quantum", "bio")
        if wants_compare:
            add("对比/优缺点总结", "compare", "comparison", "advantage", "disadvantage", "trade-off", "baseline")
        # Add a couple of anchor terms (makes the hint less template-like).
        anchors = []
        for k, lab in [
            ("spad", "SPAD"),
            ("snspd", "SNSPD"),
            ("nanowire", "纳米线"),
            ("transition edge", "TES"),
            ("perovskite", "钙钛矿"),
        ]:
            if k in text and lab not in anchors:
                anchors.append(lab)
        if anchors and out:
            out[0] = out[0] + "（" + "、".join(anchors[:2]) + "）"
        return out[:4]

    # Imaging / compressive sensing / single-pixel etc.
    if wants_encoding:
        add("编码/采样策略", "hadamard", "fourier", "pattern", "mask", "dmd", "coded", "coding", "sampling", "multiplex", "frequency-division", "metamaterial", "metasurface", "speckle")

    add("测量/前向模型", "forward model", "measurement model", "measurement", "sensing", "compressive", "snapshot", "ghost imaging", "single-pixel", "single pixel", "spi")
    add("光学/硬件架构", "hardware", "optical", "disperser", "camera", "mask", "sensor", "system", "metamaterial", "metasurface")
    add("重建/反演算法", "reconstruction", "inverse", "recover", "optimization", "solver", "iter", "algorithm", "unrolling", "tv", "l1")
    add("训练设置/实现细节", "implementation", "training", "hyperparameter", "batch", "lr", "learning rate")
    add("损失/正则项", "loss", "objective", "regular", "prior", "constraint")
    add("实验设置/数据集", "dataset", "benchmark", "setup", "scene", "real data", "simulated")
    add("评价指标/对比结果", "psnr", "ssim", "metric", "performance", "compare", "baseline", "ablation", "result")
    add("局限/失败案例", "limitation", "failure", "future work", "discussion")
    add("可追溯参考文献", "references", "bibliography")

    # Make aspects more query-driven: if user asked for metrics/compare, prioritize those.
    if wants_metrics and ("评价指标/对比结果" in out):
        out.remove("评价指标/对比结果")
        out.insert(0, "评价指标/对比结果")
    if wants_compare and ("评价指标/对比结果" in out) and out[0] != "评价指标/对比结果":
        try:
            out.remove("评价指标/对比结果")
        except Exception:
            pass
        out.insert(0, "评价指标/对比结果")

    # Add anchor terms to the encoding aspect for better specificity.
    if any(x.startswith("编码/采样策略") for x in out):
        anchors = []
        for k, lab in [
            ("hadamard", "Hadamard"),
            ("fourier", "Fourier"),
            ("dmd", "DMD"),
            ("frequency-division", "频分复用"),
            ("multiplex", "复用"),
            ("metamaterial", "超材料"),
            ("metasurface", "超表面"),
            ("speckle", "散斑"),
        ]:
            if k in text and lab not in anchors:
                anchors.append(lab)
        if anchors:
            for i, v in enumerate(out):
                if v.startswith("编码/采样策略"):
                    out[i] = "编码/采样策略（" + "、".join(anchors[:2]) + "）"
                    break

    # If user asks definition-like questions, ensure '概念定义' appears when possible.
    if re.search(r"(是什么|定义|含义|概念|what is|definition)", prompt_text or "", flags=re.I):
        if "概念定义/问题设定" not in out:
            out.insert(0, "概念定义/问题设定")

    return out[:4]


def _group_hits_by_doc_for_refs(
    hits_raw: list[dict],
    prompt_text: str,
    top_k_docs: int,
    *,
    deep_query: str = "",
    deep_read: bool = False,
    llm_rerank: bool = False,
    settings=None,
) -> list[dict]:
    """
    Merge hits from the same markdown doc into a single ref entry.
    """
    by_doc: dict[str, list[dict]] = {}
    for h in hits_raw or []:
        meta = h.get("meta", {}) or {}
        src = (meta.get("source_path") or "").strip()
        if not src:
            continue
        by_doc.setdefault(src, []).append(h)

    # Pre-sort docs by best hit score so we only deep-read a few.
    doc_order: list[tuple[float, str]] = []
    for src, hs in by_doc.items():
        try:
            best_score = float(max(float(h.get("score", 0.0) or 0.0) for h in hs))
        except Exception:
            best_score = 0.0
        doc_order.append((best_score, src))
    doc_order.sort(key=lambda x: x[0], reverse=True)

    docs: list[dict] = []
    profile = _query_term_profile(prompt_text, deep_query or "")
    # Keep it fast: do NOT do full-doc deep scoring for every hit.
    # Deep-read expansion (reading full md) is only applied to a couple of top docs.
    deep_expand_docs = 2 if deep_read else 0
    # Bound work: only consider a limited number of candidate docs.
    max_docs_consider = max(int(top_k_docs) * 2, 12)
    for _best, src in doc_order[:max_docs_consider]:
        hs = by_doc.get(src) or []
        hs2 = sorted(hs, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        best_score = float(hs2[0].get("score", 0.0) or 0.0) if hs2 else 0.0
        # Candidate headings: (score, top_heading)
        cand: list[tuple[float, str]] = []
        snippets: list[str] = []
        locs: list[tuple[float, str]] = []
        for h in hs2[:6]:
            meta = h.get("meta", {}) or {}
            top = (meta.get("top_heading") or _top_heading(meta.get("heading_path", "")) or "").strip()
            if top:
                cand.append((float(h.get("score", 0.0) or 0.0), top))
                if not _is_probably_bad_heading(top):
                    locs.append((float(h.get("score", 0.0) or 0.0), top))
            t = (h.get("text") or "").strip()
            if t:
                snippets.append(t)

        # Optional deep-read for better section targeting + aspects + ranking.
        deep_best = 0.0
        if deep_read and deep_query and (len(docs) < deep_expand_docs):
            try:
                extra = _deep_read_md_for_context(Path(src), deep_query, max_snippets=2, snippet_chars=1400)
            except Exception:
                extra = []
            for ex in extra or []:
                meta_ex = ex.get("meta", {}) or {}
                top2 = _top_heading(meta_ex.get("heading_path", "") or "")
                if top2:
                    cand.append((float(ex.get("score", 0.0) or 0.0) + 0.2, top2))
                tx = (ex.get("text") or "").strip()
                if tx:
                    snippets.append(tx)
                try:
                    deep_best = max(deep_best, float(ex.get("score", 0.0) or 0.0))
                except Exception:
                    pass

        # Final heading MUST be a real heading from this doc.
        # Prefer headings already attached to hits (grounded), but also read real md headings for navigation.
        best_heading = _pick_best_heading_for_doc(cand, prompt_text)
        headings_for_pack: list[str] = []
        try:
            prefer = _preferred_section_keys(prompt_text)
            picked = _pick_heading_from_md(Path(src), deep_query or prompt_text, prefer=prefer)
            if picked:
                best_heading = picked
        except Exception:
            pass
        try:
            headings_for_pack = _extract_md_headings(Path(src), max_n=40)
        except Exception:
            headings_for_pack = []
        if not headings_for_pack:
            # Minimal heading set: only what we have seen from hits/deep-read.
            seen_h2: list[str] = []
            for _sc2, hh in sorted(cand, key=lambda x: x[0], reverse=True):
                hh2 = _normalize_heading(hh)
                if not hh2 or _is_probably_bad_heading(hh2) or hh2 in seen_h2:
                    continue
                seen_h2.append(hh2)
                if len(seen_h2) >= 22:
                    break
            headings_for_pack = seen_h2
        aspects = _aspects_from_snippets(snippets[:3], prompt_text)

        # Build display snippets: pick the most relevant, non-noise snippets.
        q_for_pick = (deep_query or prompt_text or "").strip()
        q_tokens = [t for t in tokenize(q_for_pick) if len(t) >= 3]
        scored_snips: list[tuple[float, str]] = []
        for s in snippets:
            s2 = (s or "").strip()
            if not s2:
                continue
            if _is_noise_snippet_text(s2):
                continue
            try:
                sc = _score_tokens(s2, q_tokens) if q_tokens else 0.0
            except Exception:
                sc = 0.0
            # Prefer snippets that literally contain key phrases for single-shot/single-pixel disambiguation.
            low = _norm_text_for_match(s2)
            if profile.get("wants_single_shot") and any(k in low for k in ["single-shot", "single shot", "single exposure", "snapshot"]):
                sc += 3.0
            if profile.get("wants_single_shot") and any(k in low for k in ["single-pixel", "single pixel"]):
                sc -= 3.0
            scored_snips.append((float(sc), s2))
        scored_snips.sort(key=lambda x: x[0], reverse=True)
        show_snips = [_clean_snippet_for_display(s, max_chars=900) for _, s in scored_snips[:2]]

        # Best location candidates (real headings)
        locs.sort(key=lambda x: x[0], reverse=True)
        locs2 = []
        seen_h = set()
        for sc, hh in locs:
            hh2 = _normalize_heading(hh)
            if not hh2 or _is_probably_bad_heading(hh2) or hh2 in seen_h:
                continue
            seen_h.add(hh2)
            locs2.append({"heading": hh2, "score": float(sc)})
            if len(locs2) >= 3:
                break

        # Heuristic base score: BM25 + small deep-read signal + term mismatch penalties.
        doc_name = Path(src).name
        term_bonus = _doc_term_bonus(profile, doc_name, snippets[:3])
        deep_scaled = 1.6 * (deep_best ** 0.6) if deep_best > 0 else 0.0
        combined = (0.75 * best_score) + (0.25 * deep_scaled) + term_bonus

        meta_out = {"source_path": src}
        if best_heading:
            meta_out["top_heading"] = best_heading
        meta_out["ref_aspects"] = aspects
        meta_out["ref_snippets"] = snippets[:2]
        meta_out["ref_show_snippets"] = show_snips
        meta_out["ref_locs"] = locs2
        meta_out["ref_headings"] = headings_for_pack
        meta_out["ref_rank"] = {"bm25": best_score, "deep": deep_best, "term_bonus": term_bonus, "llm": 0.0, "why": "", "score": combined}

        docs.append(
            {
                "score": float(combined),
                "id": f"doc:{hashlib.sha1(src.encode('utf-8','ignore')).hexdigest()[:12]}",
                "text": snippets[0] if snippets else "",
                "meta": meta_out,
            }
        )

    # Optional LLM pack: one-shot semantic rerank + strong directional one-liner pieces (grounded on snippets/headings).
    if llm_rerank and settings and docs:
        pack = _llm_refs_pack(settings, question=(prompt_text or deep_query or ""), docs=docs)
        if isinstance(pack, dict) and pack:
            for i, d in enumerate(docs, start=1):
                meta = d.get("meta", {}) or {}
                pr = pack.get(i) or {}
                if not isinstance(pr, dict):
                    continue
                try:
                    llm_score = float(pr.get("score", 0.0) or 0.0)
                except Exception:
                    llm_score = 0.0
                llm_score = max(0.0, min(100.0, llm_score))
                llm_why = str(pr.get("why") or "").strip()

                meta["ref_pack"] = {
                    "score": llm_score,
                    "why": llm_why,
                    "what": str(pr.get("what") or "").strip(),
                    "find": [str(x).strip() for x in (pr.get("find") or []) if str(x).strip()][:4] if isinstance(pr.get("find"), list) else [],
                    "section": str(pr.get("section") or "").strip(),
                }

                # Use the packed section as the final "top_heading" when present (it is forced to be a real heading).
                sec = str(pr.get("section") or "").strip()
                if sec and (not _is_probably_bad_heading(sec)):
                    meta["top_heading"] = sec

                # Recompute combined score with semantic signal.
                r = meta.get("ref_rank") or {}
                try:
                    bm25 = float(r.get("bm25", 0.0) or 0.0)
                except Exception:
                    bm25 = 0.0
                try:
                    deep_best = float(r.get("deep", 0.0) or 0.0)
                except Exception:
                    deep_best = 0.0
                try:
                    term_bonus = float(r.get("term_bonus", 0.0) or 0.0)
                except Exception:
                    term_bonus = 0.0
                deep_scaled = 1.6 * (deep_best ** 0.6) if deep_best > 0 else 0.0
                combined2 = (llm_score / 10.0) + (0.25 * deep_scaled) + (0.10 * bm25) + (0.50 * term_bonus)
                d["score"] = float(combined2)
                meta["ref_rank"] = {"bm25": bm25, "deep": deep_best, "term_bonus": term_bonus, "llm": llm_score, "why": llm_why, "score": combined2}
                d["meta"] = meta

    docs.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
    return docs[: max(1, int(top_k_docs))]


def _group_hits_by_doc_for_refs_fast(hits_raw: list[dict], top_k_docs: int) -> list[dict]:
    """
    Fast fallback for background QA worker:
    - no full-md deep read
    - no LLM rerank
    - no heading extraction from file
    """
    by_doc: dict[str, dict] = {}
    for h in hits_raw or []:
        meta = h.get("meta", {}) or {}
        src = (meta.get("source_path") or "").strip()
        if not src:
            continue
        cur = by_doc.get(src)
        sc = float(h.get("score", 0.0) or 0.0)
        top = (meta.get("top_heading") or _top_heading(meta.get("heading_path", "")) or "").strip()
        txt = (h.get("text") or "").strip()
        if (cur is None) or (sc > float(cur.get("score", 0.0) or 0.0)):
            by_doc[src] = {
                "score": sc,
                "id": f"doc:{hashlib.sha1(src.encode('utf-8','ignore')).hexdigest()[:12]}",
                "text": txt,
                "meta": {
                    "source_path": src,
                    "top_heading": ("" if _is_probably_bad_heading(top) else top),
                    "ref_snippets": [txt] if txt else [],
                    "ref_show_snippets": [_clean_snippet_for_display(txt, max_chars=900)] if txt else [],
                    "ref_locs": ([{"heading": top, "score": sc}] if top and (not _is_probably_bad_heading(top)) else []),
                    "ref_headings": ([top] if top and (not _is_probably_bad_heading(top)) else []),
                    "ref_aspects": [],
                    "ref_rank": {"bm25": sc, "deep": 0.0, "term_bonus": 0.0, "llm": 0.0, "why": "", "score": sc},
                },
            }
        elif txt:
            m = by_doc[src].get("meta") or {}
            arr = list(m.get("ref_snippets") or [])
            if txt not in arr and len(arr) < 2:
                arr.append(txt)
                m["ref_snippets"] = arr
                m["ref_show_snippets"] = [_clean_snippet_for_display(x, max_chars=900) for x in arr]
                by_doc[src]["meta"] = m
    docs = list(by_doc.values())
    docs.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
    return docs[: max(1, int(top_k_docs))]


def _read_text_cached(path: Path) -> str:
    p = Path(path)
    try:
        mtime = float(p.stat().st_mtime)
    except Exception:
        mtime = 0.0
    key = f"{str(p)}|{mtime}"
    v0 = _cache_get("file_text", key)
    if isinstance(v0, str):
        return v0
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        text = ""
    _cache_set("file_text", key, text, max_items=220)
    return text


def _extract_md_headings(md_path: Path, *, max_n: int = 80) -> list[str]:
    """
    Extract real headings from the markdown file (ground truth for navigation).
    Returns plain heading titles (without leading #), preserving numbering if present.
    """
    md_path = Path(md_path)
    if not md_path.exists():
        return []
    text = _read_text_cached(md_path)
    if not text:
        return []
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("#"):
            continue
        level = len(s) - len(s.lstrip("#"))
        if level <= 0 or level > 4:
            continue
        title = s[level:].strip()
        title = _normalize_heading(title)
        if not title or _is_probably_bad_heading(title):
            continue
        if title not in out:
            out.append(title)
        if len(out) >= max_n:
            break
    return out


def _pick_heading_from_md(md_path: Path, query: str, *, prefer: list[str]) -> str:
    """
    Pick a heading that most likely contains the answer, from real md headings only.
    """
    hs = _extract_md_headings(md_path)
    if not hs:
        return ""
    q = (query or "").strip()
    if not q:
        return hs[0]
    q_toks = [t for t in tokenize(q) if len(t) >= 3]

    def score(h: str) -> float:
        low = h.lower()
        # Keyword overlap
        base = 0.0
        if q_toks:
            ht = tokenize(h)
            ct = Counter(ht)
            base += float(sum(ct.get(t, 0) for t in q_toks))
        # Preference boost (method/results/intro etc)
        bonus = 0.0
        for i, k in enumerate(prefer):
            if k in low:
                bonus += 3.0 - i * 0.25
                break
        # Slightly prefer medium-length headings
        bonus += max(0.0, (50 - abs(len(h) - 38)) / 200.0)
        return base + bonus

    best = max(hs, key=score)
    # Avoid pointing to REFERENCES unless asked.
    wants_refs = bool(re.search(r"(参考文献|引用|cite|citation|reference)", q, flags=re.I))
    if ("references" in best.lower() or "bibliography" in best.lower()) and not wants_refs:
        for h in hs:
            if ("references" in h.lower() or "bibliography" in h.lower()):
                continue
            if not _is_probably_bad_heading(h):
                return h
        return ""
    return best


def _score_tokens(text: str, query_tokens: list[str]) -> float:
    toks = tokenize(text or "")
    if not toks:
        return 0.0
    if not query_tokens:
        return 0.0
    ct = Counter(toks)
    return float(sum(ct.get(t, 0) for t in query_tokens))


def _deep_read_md_for_context(md_path: Path, query: str, *, max_snippets: int = 3, snippet_chars: int = 1400) -> list[dict]:
    """
    Read the full .md and extract the most relevant snippets (by token overlap),
    then return in the same dict shape as retriever hits.
    """
    md_path = Path(md_path)
    if not md_path.exists():
        return []
    text = _read_text_cached(md_path)
    if not text.strip():
        return []

    q_tokens = [t for t in tokenize(query or "") if len(t) >= 3]
    if not q_tokens:
        return []

    # Cache per (file mtime, query) to avoid repeated full-doc scans across reruns / background tasks.
    try:
        mtime = float(md_path.stat().st_mtime)
    except Exception:
        mtime = 0.0
    cache_key = hashlib.sha1((str(md_path) + "|" + str(mtime) + "|" + (query or "")).encode("utf-8", "ignore")).hexdigest()[:16]
    v0 = _cache_get("deep_read", cache_key)
    if isinstance(v0, list):
        try:
            return list(v0)
        except Exception:
            return []

    chunks = chunk_markdown(text, source_path=str(md_path), chunk_size=900, overlap=0)
    scored: list[tuple[float, dict]] = []
    for c in chunks:
        body = (c.get("text") or "").strip()
        if len(body) < 80:
            continue
        s = _score_tokens(body, q_tokens)
        if s <= 0.0:
            continue
        scored.append((s, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict] = []
    for rank, (s, c) in enumerate(scored[: max(1, int(max_snippets))], start=1):
        meta = dict((c.get("meta") or {}))
        meta.setdefault("source_path", str(md_path))
        meta["deep_read"] = True
        body = (c.get("text") or "").strip()
        if len(body) > snippet_chars:
            body = body[:snippet_chars].rstrip() + "…"
        out.append({"score": float(s), "id": f"deep:{hashlib.sha1((str(md_path)+'|'+str(rank)).encode('utf-8','ignore')).hexdigest()[:12]}", "text": body, "meta": meta})
    _cache_set("deep_read", cache_key, out, max_items=320)
    return out


def _llm_refs_pack(settings, *, question: str, docs: list[dict]) -> dict[int, dict]:
    """
    One-shot LLM pack for refs:
    - semantic relevance score (0..100) for reranking
    - a strong directional one-liner pieces (what/find/section) grounded on snippets

    Returns: {idx -> {"score":float, "why":str, "what":str, "find":[str], "section":str}}
    """
    if not settings or (not getattr(settings, "api_key", None)):
        return {}
    q = (question or "").strip()
    if not q or not docs:
        return {}

    items = []
    for i, d in enumerate(docs, start=1):
        meta = d.get("meta", {}) or {}
        headings = [h for h in (meta.get("ref_headings") or []) if isinstance(h, str)]
        headings = [h for h in headings if h and (not _is_probably_bad_heading(h))][:20]
        snippets = []
        for s in (meta.get("ref_show_snippets") or [])[:2]:
            s2 = " ".join(str(s).strip().split())
            if len(s2) > 360:
                s2 = s2[:360].rstrip() + "…"
            if s2:
                snippets.append(s2)
        if not snippets:
            # fallback to existing text field
            s = " ".join((d.get("text") or "").strip().split())
            if len(s) > 360:
                s = s[:360].rstrip() + "…"
            if s:
                snippets.append(s)
        items.append({"i": i, "headings": headings, "snippets": snippets})

    try:
        # Include mtimes so updates invalidate cache.
        sig_parts = [q]
        for d in docs:
            src = str((d.get("meta", {}) or {}).get("source_path") or "")
            try:
                mtime = float(Path(src).stat().st_mtime) if src else 0.0
            except Exception:
                mtime = 0.0
            sig_parts.append(src + "|" + str(mtime))
        sig = "|".join(sig_parts)
        cache_key = hashlib.sha1(sig.encode("utf-8", "ignore")).hexdigest()[:16]
    except Exception:
        cache_key = hashlib.sha1(q.encode("utf-8", "ignore")).hexdigest()[:16]

    v0 = _cache_get("refs_pack", cache_key)
    if isinstance(v0, dict):
        return v0

    ds = DeepSeekChat(settings)
    en = _has_latin(q) and (not _has_cjk(q))
    if en:
        sys = (
            "You are a strict academic retriever reranker and paper navigator.\n"
            "Output JSON ONLY: {\"items\":[{\"i\":int,\"score\":number,\"why\":string,\"what\":string,\"find\":[string],\"section\":string}]}.\n"
            "Rules:\n"
            "- score: 0..100, how directly snippets answer the question.\n"
            "- Penalize false friends (single-shot vs single-pixel) when mismatched.\n"
            "- Use ONLY snippets/headings; DO NOT use filenames.\n"
            "- section MUST be chosen from provided headings; otherwise empty string.\n"
            "- what: <= 18 words. why: <= 14 words. find: 2-4 short phrases.\n"
        )
    else:
        sys = (
            "你是严格的学术检索重排器 + 文献导航器。\n"
            "只能输出 JSON：{\"items\":[{\"i\":int,\"score\":number,\"why\":string,\"what\":string,\"find\":[string],\"section\":string}]}。\n"
            "规则：\n"
            "- score: 0..100，表示这些片段与问题的直接相关程度。\n"
            "- 遇到术语假朋友要扣分（如 single-shot vs single-pixel）。\n"
            "- 只能根据 snippets/headings 判断，不能根据文件名判断。\n"
            "- section 必须从给定 headings 中选一个；选不出来就空字符串。\n"
            "- what：<= 22字；why：<= 16字；find：2-4 个短语（具体能找到什么）。\n"
        )

    # Keep payload small
    payload = {"question": q, "docs": items}
    user = json.dumps(payload, ensure_ascii=False)
    try:
        out = (ds.chat(messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}], temperature=0.0, max_tokens=520) or "").strip()
    except Exception:
        out = ""
    if out.startswith("```"):
        out = out.strip().strip("`")
        out = out.replace("json", "", 1).strip()

    try:
        data = json.loads(out)
    except Exception:
        data = None
    if not isinstance(data, dict):
        return {}
    arr = data.get("items") or []
    if not isinstance(arr, list):
        return {}

    result: dict[int, dict] = {}
    for it in arr:
        if not isinstance(it, dict):
            continue
        try:
            i = int(it.get("i"))
        except Exception:
            continue
        try:
            sc = float(it.get("score", 0.0) or 0.0)
        except Exception:
            sc = 0.0
        sc = max(0.0, min(100.0, sc))
        result[i] = {
            "score": sc,
            "why": str(it.get("why") or "").strip(),
            "what": str(it.get("what") or "").strip(),
            "find": [str(x).strip() for x in (it.get("find") or []) if str(x).strip()][:4] if isinstance(it.get("find"), list) else [],
            "section": str(it.get("section") or "").strip(),
        }

    _cache_set("refs_pack", cache_key, result, max_items=260)
    return result


def _open_pdf(pdf_path: Path) -> tuple[bool, str]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return False, f"PDF 不存在：{pdf_path}"
    try:
        os.startfile(str(pdf_path))  # type: ignore[attr-defined]
        return True, f"已打开：{pdf_path}"
    except Exception:
        pass
    try:
        # Another Windows-friendly fallback.
        subprocess.Popen(["cmd", "/c", "start", "", str(pdf_path)], shell=False)
        return True, f"已打开：{pdf_path}"
    except Exception:
        pass
    try:
        # Fallback that also works for some Windows policy constraints.
        subprocess.Popen(["powershell", "-NoProfile", "-Command", "Start-Process", "-FilePath", str(pdf_path)])
        return True, f"已打开：{pdf_path}"
    except Exception:
        pass
    try:
        open_in_explorer(pdf_path)
        return True, f"已在资源管理器定位：{pdf_path}"
    except Exception as e:
        return False, f"打开失败：{e}"


def _file_url_for_pdf(path: Path, *, page: int | None = None) -> str:
    """
    Build a file:// URL for a local PDF, optionally with a #page=N fragment.
    Many PDF viewers (and browsers) support opening at a specific page.
    """
    from urllib.parse import quote

    p = Path(path).resolve()
    # Windows path -> file:///C:/... form
    s = p.as_posix()
    if re.match(r"^[A-Za-z]:/", s):
        url = "file:///" + quote(s)
    else:
        url = "file://" + quote(s)
    if page and int(page) > 0:
        url += f"#page={int(page)}"
    return url


def _open_pdf_at(pdf_path: Path, *, page: int | None = None) -> tuple[bool, str]:
    """
    Best-effort open a PDF and jump to a specific page.
    Falls back to opening the file if the viewer doesn't support fragments.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return False, f"PDF 不存在：{pdf_path}"
    if page and int(page) > 0:
        url = _file_url_for_pdf(pdf_path, page=int(page))
        try:
            subprocess.Popen(["cmd", "/c", "start", "", url], shell=False)
            return True, f"已打开：{pdf_path}（第 {int(page)} 页）"
        except Exception:
            pass
    return _open_pdf(pdf_path)


def _render_refs(
    hits: list[dict],
    *,
    prompt: str = "",
    show_heading: bool = True,
    key_ns: str = "refs",
    settings=None,
) -> None:
    # Keep UI state per prompt, otherwise clicking "更多" can affect other prompts/reruns.
    p_sig = hashlib.sha1((prompt or "").strip().encode("utf-8", "ignore")).hexdigest()[:8] if (prompt or "").strip() else "noprompt"
    more_key = f"{key_ns}_more_{p_sig}"
    open_key = f"{key_ns}_refs_open_{p_sig}"

    def _parse_paper_name(source_path: str) -> tuple[str, str, str]:
        """
        Try to parse 'venue-year-title' from the markdown filename.
        Fallback to the stem if it doesn't match.
        """
        stem = (Path(source_path).stem or "").strip()
        if not stem:
            return ("", "", "")

        # Find a 4-digit year and split around it.
        import re

        m = re.search(r"(19\\d{2}|20\\d{2})", stem)
        if not m:
            return ("", "", stem)

        year = m.group(1)
        left = stem[: m.start()].strip("- _")
        right = stem[m.end() :].strip("- _")
        venue = left.strip()
        title = right.strip()
        return (venue, year, title)

    def _wants_english(text: str) -> bool:
        t = (text or "").lower()
        if "用英文" in t or "英文" in t or "english" in t:
            return True
        # If it's mostly latin letters and spaces, assume English.
        latin = sum(1 for ch in (text or "") if ("a" <= ch.lower() <= "z"))
        cjk = sum(1 for ch in (text or "") if ("\u4e00" <= ch <= "\u9fff"))
        return latin > 0 and latin >= (cjk * 2)

    def _short_query(text: str, max_chars: int = 18) -> str:
        t = " ".join((text or "").strip().split())
        if not t:
            return ""
        return (t[:max_chars] + "…") if len(t) > max_chars else t

    def _rel_desc(prompt_text: str, top_heading: str, aspects: list[str]) -> str:
        """
        One-sentence relevance to the user's question (NOT quoting the paper content).
        """
        en = _wants_english(prompt_text)
        top = (top_heading or "").strip()
        q = _short_query(prompt_text, 22 if en else 16)

        role = ""
        top_u = top.upper()
        if "INTRODUCTION" in top_u or "ABSTRACT" in top_u or "OVERVIEW" in top_u:
            role = "background / overview" if en else "背景/概述（问题设定/定义）"
        elif "RELATED WORK" in top_u:
            role = "related work / comparison" if en else "相关工作/对比（和谁比、差异点）"
        elif "METHOD" in top_u or "APPROACH" in top_u or "MODEL" in top_u:
            role = "method / model details" if en else "方法/模型（测量模型、重建/训练流程）"
        elif "EXPERIMENT" in top_u or "RESULT" in top_u or "EVALUATION" in top_u:
            role = "results / evaluation" if en else "实验结果/评估（指标、对比、消融）"
        elif "REFERENCES" in top_u:
            role = "references to follow up" if en else "可追溯参考文献"
        else:
            # Do NOT echo the paper's heading/title in the parentheses area.
            role = "relevant section" if en else "方法/实验等最相关章节"

        def _intent_desc(text: str) -> str:
            t = (text or "").lower()
            if en:
                if any(x in t for x in ["what is", "meaning", "define", "definition"]):
                    return "definition / problem setup"
                if any(x in t for x in ["coding", "coded", "pattern", "mask", "sampling", "hadamard", "fourier", "dmd", "metamaterial", "multiplex"]):
                    return "coding / sampling patterns"
                if any(x in t for x in ["how", "algorithm", "method", "derive", "equation", "formula", "implementation"]):
                    return "method / key equations"
                if any(x in t for x in ["compare", "difference", "vs", "baseline", "ablation"]):
                    return "comparison / related work"
                if any(x in t for x in ["result", "experiment", "metric", "performance"]):
                    return "results / evaluation"
                return "relevant details"
            else:
                if re.search(r"(是什么|定义|含义|概念)", text or ""):
                    return "概念定义/问题设定"
                if re.search(r"(编码|掩膜|pattern|mask|采样|hadamard|fourier|dmd|超材料|复用|coded|coding)", text or "", flags=re.I):
                    return "编码/采样方案"
                if re.search(r"(怎么|如何|步骤|实现|算法|公式|推导)", text or ""):
                    return "方法/关键公式/实现细节"
                if re.search(r"(对比|区别|优缺点|比较|消融)", text or ""):
                    return "对比/差异/相关工作"
                if re.search(r"(结果|实验|性能|指标)", text or ""):
                    return "实验结果/指标对比"
                return "相关细节"

        kws = _extract_keywords_for_desc(prompt_text, max_n=3)
        intent = _intent_desc(prompt_text)
        kw_s = (", ".join(kws) if en else "、".join(kws)) if kws else ""
        if en:
            if kw_s:
                return f"In this paper you can find {intent} about {kw_s}; focus: {role}."
            if q:
                return f"In this paper you can find {intent} for “{q}”; focus: {role}."
            return f"In this paper you can find {intent}; focus: {role}."
        else:
            def _fallback_section_hint() -> str:
                keys = _preferred_section_keys(prompt_text)
                # Map to readable Chinese section hints (without inventing an exact heading).
                if any(k in keys for k in ["abstract", "introduction", "overview", "background"]):
                    return "建议看：摘要/引言"
                if any(k in keys for k in ["method", "approach", "model", "algorithm", "theory"]):
                    return "建议看：方法/模型"
                if any(k in keys for k in ["experiment", "results", "evaluation", "implementation", "setup"]):
                    return "建议看：实验/结果"
                return "建议看：方法/实验"

            parts = []
            if aspects:
                parts.append("、".join(aspects[:4]))
            else:
                if kw_s:
                    parts.append(f"围绕「{kw_s}」的{intent}")
                else:
                    parts.append(intent)
            sec = role
            if top and not _is_probably_bad_heading(top):
                sec = f"建议看：{top}"
            elif not top:
                sec = _fallback_section_hint()
            return f"可在本文找到：{'；'.join(parts)}；{sec}。"

    def _llm_ref_one_liner(settings_obj, prompt_text: str, meta: dict) -> str | None:
        """
        Generate a strongly directional one-liner:
        - Must not quote long paper content
        - Must select section from real headings
        - Must state: what this paper contains + where to look
        """
        if not settings_obj or (not getattr(settings_obj, "api_key", None)):
            return None
        src = (meta.get("source_path") or "").strip()
        if not src:
            return None
        md_path = Path(src)
        try:
            mtime = float(md_path.stat().st_mtime) if md_path.exists() else 0.0
        except Exception:
            mtime = 0.0
        q = (prompt_text or "").strip()
        if not q:
            return None

        headings = [h for h in (meta.get("ref_headings") or []) if isinstance(h, str)]
        headings = [h for h in headings if h and (not _is_probably_bad_heading(h))]
        # Limit to keep prompt small
        headings = headings[:30]

        snippets = [s for s in (meta.get("ref_snippets") or []) if isinstance(s, str)]
        # Clip snippets to avoid dumping content
        clipped: list[str] = []
        for s in snippets[:2]:
            s2 = " ".join((s or "").strip().split())
            if len(s2) > 280:
                s2 = s2[:280].rstrip() + "…"
            if s2:
                clipped.append(s2)

        cache = st.session_state.setdefault("ref_llm_cache", {})
        key = hashlib.sha1((src + "|" + str(mtime) + "|" + q).encode("utf-8", "ignore")).hexdigest()[:16]
        if key in cache:
            v = (cache.get(key) or "").strip()
            return v or None

        en = _wants_english(prompt_text)
        ds = DeepSeekChat(settings_obj)
        if en:
            sys = (
                "You are a paper navigator.\n"
                "Output JSON ONLY: {\"what\":string,\"find\":[string],\"section\":string}.\n"
                "Constraints:\n"
                "1) what: 1 sentence (<=22 words), describe what the paper is about (no copying).\n"
                "2) find: 2-4 short phrases: what the user can find in the paper (specific).\n"
                "3) section: MUST be chosen from provided headings; otherwise empty string.\n"
                "4) Grounded only on provided info; no hallucination.\n"
            )
        else:
            sys = (
                "你是“文献导航器”，帮用户快速定位论文里最可能有答案的地方。\n"
                "你必须输出且只输出 JSON：{\"what\":string,\"find\":[string],\"section\":string}。\n"
                "约束：\n"
                "1) what：一句话（<=45字），说明这篇论文“讲什么/贡献是什么”，不要抄原文句子。\n"
                "2) find：2-4 个短语，写“用户能在文中找到什么信息”（比如测量模型/重建算法/实验指标/对比结果等），要具体。\n"
                "3) section：必须从给定 headings 中选一个最可能的位置；如果无法判断就输出空字符串。\n"
                "4) 只能基于提供的信息推断，不要编造不存在的章节或内容。\n"
            )
        user = (
            f"用户问题：{q}\n\n"
            f"论文文件：{Path(src).name}\n\n"
            f"headings（可选章节列表）：\n- " + "\n- ".join(headings) + "\n\n"
            f"命中片段（仅供判断方向）：\n- " + "\n- ".join(clipped) + "\n"
        )
        try:
            out = (ds.chat(messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}], temperature=0.0, max_tokens=220) or "").strip()
        except Exception:
            out = ""

        # Strip fenced blocks if any
        if out.startswith("```"):
            out = out.strip().strip("`")
            out = out.replace("json", "", 1).strip()

        data = None
        try:
            data = json.loads(out)
        except Exception:
            data = None
        if not isinstance(data, dict):
            return None

        what = str(data.get("what") or "").strip()
        find = data.get("find") or []
        if not isinstance(find, list):
            find = []
        find = [str(x).strip() for x in find if str(x).strip()]
        find = find[:4]
        section = str(data.get("section") or "").strip()
        if section and headings and section not in headings:
            section = ""
        if not what:
            return None

        # Build final one-liner
        if en:
            find_s = ", ".join(find) if find else ""
            if section:
                line = f"What: {what}; Find: {find_s or 'definitions/method/experiments'}; See: {section}."
            else:
                line = f"What: {what}; Find: {find_s or 'definitions/method/experiments'}."
        else:
            find_s = "、".join(find) if find else ""
            if section:
                line = f"讲：{what}；可找：{find_s or '关键定义/方法/实验'}；建议看：{section}。"
            else:
                line = f"讲：{what}；可找：{find_s or '关键定义/方法/实验'}。"
        cache[key] = line
        return line

    def _ref_one_liner_no_llm(meta: dict, prompt_text: str, *, fallback_heading: str = "") -> str:
        """
        Build a strong, doc-specific one-liner without calling LLM.
        Uses only snippets/headings signals (NOT filenames).
        """
        en = _wants_english(prompt_text)
        profile = _query_term_profile(prompt_text, "")

        snips = meta.get("ref_snippets") or []
        if not isinstance(snips, list):
            snips = []
        locs = meta.get("ref_locs") or []
        if not isinstance(locs, list):
            locs = []

        hay = _norm_text_for_match("\n".join([str(s) for s in snips if str(s).strip()][:4]) + "\n" + str(fallback_heading or ""))
        has_single_shot = any(k in hay for k in ["single-shot", "single shot", "single exposure", "snapshot"])
        has_single_pixel = any(k in hay for k in ["single-pixel", "single pixel", "spi", "ghost imaging"])
        has_single_photon = any(k in hay for k in ["single-photon", "single photon", "spad", "snspd", "nanowire", "avalanche"])

        wants_refs = bool(re.search(r"(参考文献|引用|cite|citation|reference)", prompt_text or "", flags=re.I))

        # Recommended section
        sec = (meta.get("top_heading") or fallback_heading or "").strip()
        if _is_probably_bad_heading(sec):
            sec = ""
        if sec and (("references" in sec.lower()) or ("bibliography" in sec.lower())) and (not wants_refs):
            sec = ""
        if not sec:
            for it in locs:
                if not isinstance(it, dict):
                    continue
                hh = str(it.get("heading") or "").strip()
                if not hh or _is_probably_bad_heading(hh):
                    continue
                if (("references" in hh.lower()) or ("bibliography" in hh.lower())) and (not wants_refs):
                    continue
                sec = hh
                break

        # Domain-specific "find" phrases (grounded on terms seen in snippets).
        find: list[str] = []
        if profile.get("wants_single_photon"):
            if ("spad" in hay) or ("avalanche" in hay):
                find.append("SPAD/雪崩二极管原理")
            if ("snspd" in hay) or ("nanowire" in hay):
                find.append("SNSPD/纳米线原理")
            if ("dark count" in hay) or ("dcr" in hay):
                find.append("暗计数(DCR)")
            if "jitter" in hay:
                find.append("时间抖动(jitter)")
            if ("quantum efficiency" in hay) or ("detection efficiency" in hay) or ("efficiency" in hay):
                find.append("探测效率/量子效率")
            if not find:
                find = ["工作机制", "关键指标", "应用场景"]
        else:
            if "frequency-division" in hay or "frequency division" in hay:
                find.append("频分复用编码")
            if "metamaterial" in hay or "metasurface" in hay:
                find.append("超材料/超表面编码")
            if "hadamard" in hay:
                find.append("Hadamard 编码")
            if "fourier" in hay:
                find.append("Fourier 编码")
            if "dmd" in hay:
                find.append("DMD 实现细节")
            if "foveated" in hay or "supersampling" in hay:
                find.append("注视/自适应采样策略")
            if any(k in hay for k in ["reconstruction", "inverse", "recover", "optimization", "solver", "unrolling"]):
                find.append("重建/反演算法")
            if any(k in hay for k in ["learning rate", "hyperparameter", "batch", "epoch", "pytorch", "cuda"]):
                find.append("训练/实现参数")
            if not find:
                aspects = meta.get("ref_aspects") or []
                if isinstance(aspects, list):
                    find = [str(x).strip() for x in aspects if str(x).strip()]
            if not find:
                find = ["关键定义/方法/实验"]

        # Tag: review vs method vs principle (best-effort).
        tag = ""
        if any(k in hay for k in ["review", "survey", "prospect", "perspective", "tutorial"]):
            tag = "综述"
        elif any(k in hay for k in ["we propose", "propose a", "our method", "framework", "approach"]):
            tag = "方法"
        elif any(k in hay for k in ["principle", "mechanism", "avalanche", "nanowire"]):
            tag = "原理/机制"

        # Mismatch warnings (helps users trust ranking).
        warn = ""
        if profile.get("wants_single_photon") and (not has_single_photon) and has_single_pixel:
            warn = "(more imaging, fewer detector details)" if en else "（更偏成像方法，探测器机制较少）"
        if profile.get("wants_single_shot") and (not has_single_shot) and has_single_pixel:
            warn = "(single-pixel, not single-shot)" if en else "（偏单像素成像，非单曝光/快照）"

        find_s = (", ".join(find[:3]) if en else "、".join(find[:3]))
        if en:
            tag_en = {"综述": "review", "方法": "method", "原理/机制": "mechanism"}.get(tag, "")
            core = f"Find: {find_s or 'key definitions/method/experiments'}"
            if tag_en:
                core += f" ({tag_en})"
            if sec:
                core += f"; check: {sec}"
            if warn:
                core += f" {warn}"
            return core + "."

        core = f"本文可找到：{find_s or '关键定义/方法/实验'}"
        if tag:
            core += f"（类型：{tag}）"
        if sec:
            core += f"；优先看：{sec}"
        if warn:
            core += warn
        return core + "。"

    def _brief_ref_line(i: int, h: dict, prompt_text: str) -> tuple[str, str, str, str]:
        meta = h.get("meta", {}) or {}
        src = (meta.get("source_path") or "").strip()
        top = (meta.get("top_heading") or _top_heading(meta.get("heading_path", "")) or "").strip()
        if _is_probably_bad_heading(top):
            top = ""
        src_name = Path(src).name or src or "unknown"
        # Display-friendly: keep it short in the UI, but preserve src for lookups.
        main = src_name + (f" | {top}" if top else "")
        aspects = list(meta.get("ref_aspects") or [])
        what = ""
        pack = meta.get("ref_pack") or {}
        if isinstance(pack, dict):
            en = _wants_english(prompt_text)
            w = str(pack.get("what") or "").strip()
            sec = str(pack.get("section") or "").strip()
            find = pack.get("find") or []
            if not isinstance(find, list):
                find = []
            find = [str(x).strip() for x in find if str(x).strip()][:4]
            if w:
                if en:
                    find_s = ", ".join(find) if find else ""
                    what = f"This paper explains {w}; you can find {find_s or 'definitions/method/experiments'}"
                    if sec:
                        what += f"; check: {sec}"
                    what += "."
                else:
                    find_s = "、".join(find) if find else ""
                    what = f"这篇主要讲：{w}；文中可找到：{find_s or '关键定义/方法/实验'}"
                    if sec:
                        what += f"；优先看：{sec}"
                    what += "。"

        if not what:
            what = _ref_one_liner_no_llm(meta, prompt_text, fallback_heading=top)
        return (src, main, what, src_name)

    if show_heading:
        st.markdown(f"### {S['refs']}")
    if not hits:
        st.markdown(f"<div class='refbox'>{S['kb_miss']}</div>", unsafe_allow_html=True)
        return

    # Allow opening PDFs from refs (best-effort mapping by stem).
    pdf_root_str = (st.session_state.get("pdf_dir") or "").strip()
    pdf_root = Path(pdf_root_str) if pdf_root_str else None

    def _find_pdf_for_source(source_path: str) -> Path | None:
        if not pdf_root:
            return None
        stem = (Path(source_path).stem or "").strip()
        if stem.endswith(".en"):
            stem = stem[: -3]
        if not stem:
            return None
        candidates = [stem]
        if stem.startswith("__upload__"):
            candidates.append(stem[len("__upload__") :].lstrip("_- "))
        for c in candidates:
            if not c:
                continue
            p = _lookup_pdf_by_stem(pdf_root, c)
            if p and p.exists():
                return p
        return None

    def _split_snippet_sentences(text: str) -> list[str]:
        raw = _clean_snippet_for_display(text or "", max_chars=1400)
        if not raw:
            return []
        # Keep punctuation while splitting Chinese/English sentences.
        parts = re.split(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+(?=[A-Z0-9])|\n+", raw)
        out: list[str] = []
        for p in parts:
            s = " ".join((p or "").strip().split())
            if not s:
                continue
            if _is_noise_snippet_text(s):
                continue
            has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in s)
            if (not has_cjk) and len(s) < 16:
                continue
            if has_cjk and len(s) < 8:
                continue
            out.append(s)
        if out:
            return out[:28]
        fallback = " ".join(raw.split())
        return [fallback] if fallback else []

    def _pick_best_support_sentence(prompt_text: str, meta_h: dict, hit_text: str) -> tuple[str, list[str]]:
        q = (prompt_text or "").strip()
        # Highlight terms: query phrases + useful tokens.
        terms: list[str] = []
        seen_terms: set[str] = set()
        for x in (_extract_keywords_for_desc(q, max_n=6) + tokenize(q)):
            t = " ".join(str(x or "").strip().split())
            if not t:
                continue
            has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in t)
            if has_cjk:
                if len(t) < 2:
                    continue
            else:
                if len(t) < 3:
                    continue
            k = _norm_text_for_match(t)
            if not k or k in seen_terms:
                continue
            seen_terms.add(k)
            terms.append(t)
            if len(terms) >= 10:
                break

        q_tokens = [t for t in tokenize(q) if (len(t) >= 3) or any("\u4e00" <= ch <= "\u9fff" for ch in t)]
        snippets: list[str] = []
        for key in ("ref_show_snippets", "ref_snippets"):
            arr = meta_h.get(key) or []
            if isinstance(arr, list):
                for item in arr[:3]:
                    s = str(item or "").strip()
                    if s:
                        snippets.append(s)
        t0 = (hit_text or "").strip()
        if t0:
            snippets.append(t0)

        cands: list[str] = []
        seen_sent: set[str] = set()
        for s in snippets:
            for sent in _split_snippet_sentences(s):
                if sent in seen_sent:
                    continue
                seen_sent.add(sent)
                cands.append(sent)
                if len(cands) >= 36:
                    break
            if len(cands) >= 36:
                break
        if not cands:
            return ("", terms)

        term_norms = [_norm_text_for_match(t) for t in terms if t]
        best_score = -1e9
        best_sent = cands[0]
        for idx, sent in enumerate(cands):
            sc = _score_tokens(sent, q_tokens) if q_tokens else 0.0
            low = _norm_text_for_match(sent)
            for kk in term_norms:
                if kk and (kk in low):
                    sc += 1.6
            if 32 <= len(sent) <= 220:
                sc += 0.8
            elif len(sent) > 320:
                sc -= 0.8
            elif len(sent) < 14:
                sc -= 1.2
            sc += max(0.0, 0.22 - (0.015 * idx))
            if sc > best_score:
                best_score = sc
                best_sent = sent
        return (best_sent, terms)

    def _highlight_sentence_html(sentence: str, terms: list[str]) -> str:
        txt = (sentence or "").strip()
        if not txt:
            return ""
        spans: list[tuple[int, int]] = []
        for term in terms or []:
            t = " ".join(str(term or "").strip().split())
            if not t:
                continue
            has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in t)
            if (not has_cjk and len(t) < 3) or (has_cjk and len(t) < 2):
                continue
            flags = re.I if re.search(r"[A-Za-z]", t) else 0
            try:
                for m in re.finditer(re.escape(t), txt, flags=flags):
                    spans.append((m.start(), m.end()))
            except Exception:
                continue
        if not spans:
            return html.escape(txt)

        spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        merged: list[list[int]] = []
        for s, e in spans:
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            elif e > merged[-1][1]:
                merged[-1][1] = e

        out: list[str] = []
        last = 0
        for s, e in merged:
            out.append(html.escape(txt[last:s]))
            out.append(f"<mark>{html.escape(txt[s:e])}</mark>")
            last = e
        out.append(html.escape(txt[last:]))
        return "".join(out)

    # Default: show 3, and allow expanding for the rest.
    head = hits[:3]
    tail = hits[3:]
    st.markdown("<div class='refslist'>", unsafe_allow_html=True)
    def _render_one(i: int, h: dict) -> None:
        src, main, what, _src_name = _brief_ref_line(i, h, prompt)
        meta_h = h.get("meta", {}) or {}
        pdf_path = _find_pdf_for_source(src)
        uid = hashlib.md5(src.encode("utf-8", "ignore")).hexdigest()[:10]
        # Optional page range (from converter markers -> chunking -> retriever).
        p0 = meta_h.get("page_start", None)
        p1 = meta_h.get("page_end", None)
        page_label = ""
        try:
            if p0 is not None:
                if p1 is not None and int(p1) != int(p0):
                    page_label = f"p.{int(p0)}-{int(p1)}"
                else:
                    page_label = f"p.{int(p0)}"
        except Exception:
            page_label = ""

        cols = st.columns([12, 1.6])
        with cols[0]:
            if page_label:
                st.markdown(
                    f"<div class='refbox'>- [{i}] {html.escape(main)} <code>{html.escape(page_label)}</code>（{html.escape(what)}）</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"<div class='refbox'>- [{i}] {html.escape(main)}（{html.escape(what)}）</div>", unsafe_allow_html=True)
            # Optional: show plain-text snippets (no markdown layout) for precise location.
            try:
                if bool(st.session_state.get("show_context")):
                    locs = meta_h.get("ref_locs") or []
                    if isinstance(locs, list) and locs:
                        wants_refs = bool(re.search(r"(参考文献|引用|cite|citation|reference)", prompt or "", flags=re.I))
                        hs = [str(x.get("heading") or "").strip() for x in locs if isinstance(x, dict)]
                        hs = [x for x in hs if x]
                        if not wants_refs:
                            hs = [x for x in hs if ("references" not in x.lower()) and ("bibliography" not in x.lower())]
                        if hs:
                            st.caption("定位建议：" + " | ".join(hs[:3]))
                    show_key = f"{key_ns}_show_snip_{p_sig}_{i}_{uid}"
                    if st.checkbox("显示命中片段（纯文本）", value=False, key=show_key):
                        best_sent, hl_terms = _pick_best_support_sentence(prompt, meta_h, str(h.get("text") or ""))
                        if best_sent:
                            st.markdown(
                                "<div class='snipquote'>"
                                "<div class='snipquote-title'>本次回答最相关原句（命中你问题关键词）</div>"
                                f"<div class='snipquote-body'>{_highlight_sentence_html(best_sent, hl_terms)}</div>"
                                "</div>",
                                unsafe_allow_html=True,
                            )
                            st.caption("下面是该命中片段全文，便于核对上下文。")
                        snips = meta_h.get("ref_show_snippets") or []
                        if isinstance(snips, list) and snips:
                            for s in snips[:2]:
                                s2 = _clean_snippet_for_display(str(s), max_chars=900)
                                if s2:
                                    st.markdown(f"<div class='snipbox'><pre>{html.escape(s2)}</pre></div>", unsafe_allow_html=True)
                        else:
                            t = (h.get("text") or "").strip()
                            if t and (not _is_noise_snippet_text(t)):
                                s2 = _clean_snippet_for_display(t, max_chars=900)
                                if s2:
                                    st.markdown(f"<div class='snipbox'><pre>{html.escape(s2)}</pre></div>", unsafe_allow_html=True)
            except Exception:
                pass
        with cols[1]:
            if pdf_path and pdf_path.exists():
                k = hashlib.md5((src + "|" + str(pdf_path)).encode("utf-8", "ignore")).hexdigest()[:10]
                if st.button("打开PDF", key=f"{key_ns}_openpdf_{i}_{k}", help=str(pdf_path)):
                    page_to_open = int(p0) if (p0 is not None) else None
                    ok, _msg = _open_pdf_at(pdf_path, page=page_to_open)
                    if ok:
                        st.caption(f"已打开（p.{page_to_open}）" if page_to_open else "已打开")
                    else:
                        st.warning("打开失败（可尝试在资源管理器中手动打开）")

    for i, h in enumerate(head, start=1):
        _render_one(i, h)

    if tail:
        # Avoid nested expanders; also avoid showing an extra "展开更多..." line after expanded.
        if more_key not in st.session_state:
            st.session_state[more_key] = False

        if not st.session_state.get(more_key):
            c_more = st.columns([1, 14])
            with c_more[0]:
                if st.button("…", key=f"{more_key}_btn", help="显示更多参考定位"):
                    # Keep the expander open after rerun, otherwise it looks like refs "disappeared".
                    st.session_state[more_key] = True
                    st.session_state[open_key] = True
                    st.experimental_rerun()
            with c_more[1]:
                st.caption(f"共 {len(hits)} 条")

        if st.session_state.get(more_key):
            for j, h in enumerate(tail, start=4):
                _render_one(j, h)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_refs_inline(prompt: str, retriever: BM25Retriever, top_k: int, settings, *, key_ns: str) -> None:
    """
    Render a compact reference box for a specific prompt.
    Cached to avoid recomputing for long chats.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return

    def _is_smalltalk_query(text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return True
        # Very short identity / greeting / thanks / filler queries shouldn't force citations.
        smalltalk_phrases = (
            "你是谁",
            "你叫什么",
            "你是啥",
            "你好",
            "在吗",
            "谢谢",
            "谢了",
            "哈哈",
            "hh",
            "ok",
            "okay",
            "hi",
            "hello",
            "bye",
            "再见",
        )
        if any(p in t for p in smalltalk_phrases):
            return True

        toks = tokenize(text)
        if not toks:
            return True

        # If it's only common function words (BM25 will match everything), treat as smalltalk.
        stop = set(
            [
                "\u4f60",
                "\u6211",
                "\u4ed6",
                "\u5979",
                "\u5b83",
                "\u4eec",
                "\u662f",
                "\u4e0d",
                "\u6709",
                "\u6ca1",
                "\u5728",
                "\u5417",
                "\u5462",
                "\u554a",
                "\u54e6",
                "\u5466",
                "\u5427",
                "\u7684",
                "\u4e86",
                "\u548c",
                "\u4e0e",
                "\u6216",
                "\u5c31",
                "\u90fd",
                "\u8fd9",
                "\u90a3",
                "\u8c01",
                "\u600e",
                "\u4e48",
                "\u600e\u4e48",
            ]
        )
        if len(toks) <= 5 and all(x in stop for x in toks):
            return True
        return False

    def _should_show_refs_for(prompt_text: str, scores: list[float]) -> bool:
        if _is_smalltalk_query(prompt_text):
            return False
        if not scores:
            return False
        top = float(scores[0])
        # "Concentration" helps reject generic queries where many chunks score similarly.
        sum10 = float(sum(scores[:10]))
        conc = top / (sum10 + 1e-9)
        # Tuned to hide noisy refs for generic queries while keeping strong matches.
        if top < 1.0:
            return False
        if conc < 0.18 and top < 3.0:
            return False
        return True

    cache = st.session_state.setdefault("refs_cache", {})
    try:
        db_mtime = st.session_state.get("db_mtime") or 0
    except Exception:
        db_mtime = 0
    REFS_CACHE_VER = "v7"
    trans_flag = "1" if bool(getattr(settings, "api_key", None)) else "0"
    deep_flag = "1" if bool(st.session_state.get("deep_read")) else "0"
    llm_flag = "1" if bool(st.session_state.get("llm_rerank")) else "0"
    cache_key = f"{key_ns}:{hashlib.sha1((REFS_CACHE_VER + '|' + prompt + '|' + str(db_mtime) + '|' + trans_flag + '|' + deep_flag + '|' + llm_flag).encode('utf-8', 'ignore')).hexdigest()[:16]}"
    cache_val = cache.get(cache_key)
    if cache_val is None:
        hits_raw, scores, used_query, used_translation = _search_hits_with_fallback(prompt, retriever, top_k=top_k, settings=settings)
        grouped_docs = _group_hits_by_doc_for_refs(
            hits_raw,
            prompt_text=prompt,
            top_k_docs=top_k,
            deep_query=str(used_query or ""),
            deep_read=bool(st.session_state.get("deep_read")),
            llm_rerank=bool(st.session_state.get("llm_rerank")),
            settings=settings,
        )
        cache_val = {
            "hits": grouped_docs,
            "scores": scores,
            "used_query": used_query,
            "used_translation": bool(used_translation),
        }
        cache[cache_key] = cache_val
    else:
        # Backward-compat with older cache formats.
        if isinstance(cache_val, list):
            cache_val = {"hits": cache_val, "scores": [float(h.get("score", 0.0) or 0.0) for h in cache_val]}

    hits = (cache_val or {}).get("hits") or []
    scores = (cache_val or {}).get("scores") or []
    used_translation = bool((cache_val or {}).get("used_translation") or False)
    used_query = str((cache_val or {}).get("used_query") or "").strip()

    if not _should_show_refs_for(prompt, scores) and not used_translation:
        # Still show refs (user preference), but don't claim it's "weak" — scores can be misleading for cross-lingual queries.
        pass

    # Small, collapsed by default; keep the UI clean.
    # Always show it (except for smalltalk/empty cases handled above), so the user doesn't feel refs "disappeared".
        with st.expander(S["refs"], expanded=False):
            if used_translation:
                st.caption("（已将中文问题转换为英文检索关键词后再检索。）")
        _render_refs(hits, prompt=prompt, show_heading=False, key_ns=f"{key_ns}_refs", settings=settings)


def _refs_cache2_key(prompt: str, *, top_k: int, settings) -> str:
    p = (prompt or "").strip()
    try:
        db_mtime = st.session_state.get("db_mtime") or 0
    except Exception:
        db_mtime = 0
    llm_flag = "1" if bool(st.session_state.get("llm_rerank")) else "0"
    trans_flag = "1" if bool(getattr(settings, "api_key", None)) else "0"
    # Cache key should be stable across reruns. Don't depend on UI toggles that may flip during generation.
    # (We may still *compute* refs using deep-read, but the key must stay stable so refs won't "disappear".)
    REFS_CACHE_VER = "v10"
    sig = REFS_CACHE_VER + "|" + p + "|" + str(int(top_k)) + "|" + str(db_mtime) + "|" + llm_flag + "|" + trans_flag
    return hashlib.sha1(sig.encode("utf-8", "ignore")).hexdigest()[:16]


def _render_refs_panel(prompt: str | None, retriever: BM25Retriever, top_k: int, settings, *, key_ns: str) -> None:
    """
    Render a stable "参考定位" expander near the input box.
    IMPORTANT: keep it non-blocking by default (use cached data when available).
    """

    prompt = (prompt or "").strip()
    p_sig = hashlib.sha1(prompt.encode("utf-8", "ignore")).hexdigest()[:8] if prompt else "noprompt"
    open_key = f"{key_ns}_refs_open_{p_sig}"
    more_key = f"{key_ns}_more_{p_sig}"
    expanded = bool(st.session_state.get(open_key)) or bool(st.session_state.get(more_key))
    with st.expander(S["refs"], expanded=expanded):
        if not prompt:
            st.caption("（暂无可定位的问题）")
            return

        if bool(getattr(retriever, "is_empty", False)):
            _render_kb_empty_hint(compact=True)
            return

        cache = st.session_state.setdefault("refs_cache2", {})
        ck = _refs_cache2_key(prompt, top_k=top_k, settings=settings)
        data = cache.get(ck) or {}
        done = bool(data.get("done") or False)
        hits = data.get("hits") or []
        used_translation = bool(data.get("used_translation") or False)
        used_query = str(data.get("used_query") or "").strip()

        if not done:
            if hits:
                st.caption("（参考定位生成中：先展示当前候选，完成后会自动更新排序/说明。）")
                _render_refs(hits, prompt=prompt, show_heading=False, key_ns=key_ns, settings=settings)
                return
            st.caption("（参考定位生成中…如果你刚发送问题，这里会在检索/深读后自动更新。）")
            return
        if not hits:
            st.caption("（未命中知识库片段：本次问题在知识库中没有找到可定位的原文位置。）")
            return

        if used_translation:
            st.caption("（已将中文问题转换为英文检索关键词后再检索。）")
        _render_refs(hits, prompt=prompt, show_heading=False, key_ns=key_ns, settings=settings)


def _strip_model_ref_section(answer: str) -> str:
    if not answer:
        return answer
    # Common markers we used in prompts previously.
    for marker in ("可参考定位", "参考定位"):
        idx = answer.find(marker)
        if idx > 0:
            return answer[:idx].rstrip()
    return answer


def _split_kb_miss_notice(text: str) -> tuple[str, str]:
    """
    If the assistant starts with '未命中知识库片段...' (a UI hint, not main content),
    split it out so we can render it with a distinct style.
    """
    if not text:
        return ("", "")
    s = text.lstrip()
    prefix = "未命中知识库片段"
    if not s.startswith(prefix):
        return ("", text)

    # Prefer treating the first line as the notice if present.
    nl = s.find("\n")
    if nl != -1:
        notice = s[:nl].strip()
        rest = s[nl + 1 :].lstrip("\n")
        return (notice, rest)

    # Otherwise, treat a short first sentence as notice.
    for sep in ("。", ".", "！", "!", "？", "?", "；", ";"):
        idx = s.find(sep)
        if 0 <= idx <= 80:
            notice = s[: idx + 1].strip()
            rest = s[idx + 1 :].lstrip()
            return (notice, rest)

    # Fallback: only style the prefix.
    rest = s[len(prefix) :].lstrip("：: \t")
    return (prefix, rest)


def _write_tmp_upload(pdf_dir: Path, filename: str, data: bytes) -> Path:
    stem = (Path(filename).stem or "upload").strip() or "upload"
    tmp = pdf_dir / f"__upload__{stem}.pdf"
    tmp.write_bytes(data)
    return tmp


def _sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _pick_directory_dialog(initial_dir: str) -> Optional[str]:
    """
    Open a native folder picker on the local machine.
    """
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        sel = filedialog.askdirectory(initialdir=initial_dir or None, title="选择目录")
        try:
            root.destroy()
        except Exception:
            pass
        sel = (sel or "").strip()
        return sel or None
    except Exception:
        return None


def _cleanup_tmp_uploads(pdf_dir: Path) -> int:
    n = 0
    try:
        for p in Path(pdf_dir).glob("__upload__*.pdf"):
            try:
                p.unlink()
                n += 1
            except Exception:
                pass
    except Exception:
        pass
    return n


def _sanitize_filename_component(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r'[<>:"/\\\\|?*]+', "-", s)
    s = s.replace("\u0000", "").strip()
    s = s.strip(" .-_")
    return s


def _is_normalized_pdf_stem(stem: str) -> bool:
    """
    Heuristic: venue-year-title (at least has a 4-digit year between dashes).
    """
    t = (stem or "").strip()
    if not t:
        return False
    if len(t) < 12:
        return False
    return bool(re.search(r"-.?(19\d{2}|20\d{2}).?-", t))


def _unique_pdf_path(pdf_dir: Path, base: str) -> Path:
    base = _sanitize_filename_component(base) or "paper"
    dest = Path(pdf_dir) / f"{base}.pdf"
    if not dest.exists():
        return dest
    k = 2
    while (Path(pdf_dir) / f"{base}-{k}.pdf").exists() and k < 999:
        k += 1
    return Path(pdf_dir) / f"{base}-{k}.pdf"


def _list_pdf_paths_fast(pdf_dir: Path) -> list[Path]:
    """
    Fast non-recursive PDF listing.
    Avoids per-file Path.stat() calls (important on large folders / slow disks).
    """
    pdf_dir = Path(pdf_dir)
    out: list[Path] = []
    try:
        with os.scandir(pdf_dir) as it:
            for e in it:
                try:
                    if not e.is_file():
                        continue
                    if not e.name.lower().endswith(".pdf"):
                        continue
                    out.append(Path(e.path))
                except Exception:
                    continue
    except Exception:
        # Fallback for unusual FS errors.
        try:
            out = [x for x in pdf_dir.glob("*.pdf") if x.is_file()]
        except Exception:
            out = []
    return out


def _select_recent_pdf_paths(pdf_dir: Path, n: int) -> list[Path]:
    """
    Select N most-recently modified PDFs (non-recursive).
    This still needs to read mtimes, so keep it off the hot path; call only on explicit user actions.
    """
    import heapq

    n = int(n or 0)
    if n <= 0:
        return []

    heap: list[tuple[float, str]] = []
    try:
        with os.scandir(Path(pdf_dir)) as it:
            for e in it:
                try:
                    if not e.is_file():
                        continue
                    if not e.name.lower().endswith(".pdf"):
                        continue
                    mt = float(e.stat().st_mtime)
                    heapq.heappush(heap, (mt, e.path))
                    if len(heap) > n:
                        heapq.heappop(heap)
                except Exception:
                    continue
    except Exception:
        return []

    heap.sort(reverse=True)
    return [Path(p) for _, p in heap]


def _page_chat(
    settings,
    chat_store: ChatStore,
    retriever: BM25Retriever,
    db_dir: Path,
    top_k: int,
    temperature: float,
    max_tokens: int,
    show_context: bool,
    deep_read: bool,
) -> None:
    _inject_copy_js()

    retriever_err = str(st.session_state.get("retriever_load_error") or "").strip()
    if retriever_err:
        st.error(f"\u77e5\u8bc6\u5e93\u52a0\u8f7d\u5931\u8d25\uff1a{retriever_err}")

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex[:10]
    session_id = str(st.session_state.get("session_id") or "").strip()

    conv_id = str(st.session_state.get("conv_id") or "").strip()
    st.session_state["show_context"] = bool(show_context)
    st.session_state["deep_read"] = bool(deep_read)

    if bool(getattr(retriever, "is_empty", False)):
        _render_kb_empty_hint()

    cur_task = _gen_get_task(session_id)
    running_for_conv = bool(
        isinstance(cur_task, dict)
        and str(cur_task.get("status") or "") == "running"
        and str(cur_task.get("conv_id") or "") == conv_id
    )

    st.session_state["pending_prompt"] = ""

    # During streaming reruns, avoid repeated full DB reads to keep UI smooth.
    msgs_cache_conv = str(st.session_state.get("_chat_msgs_cache_conv") or "")
    need_refresh_msgs = (
        (not running_for_conv)
        or (msgs_cache_conv != conv_id)
        or (not isinstance(st.session_state.get("messages"), list))
    )
    if need_refresh_msgs:
        try:
            st.session_state["messages"] = chat_store.get_messages(conv_id)
        except Exception:
            st.session_state["messages"] = []
        st.session_state["_chat_msgs_cache_conv"] = conv_id
    msgs = list(st.session_state.get("messages") or [])

    refs_cache_conv = str(st.session_state.get("_chat_refs_cache_conv") or "")
    need_refresh_refs = (
        (not running_for_conv)
        or (refs_cache_conv != conv_id)
        or (not isinstance(st.session_state.get("_chat_refs_cache"), dict))
    )
    if need_refresh_refs:
        try:
            st.session_state["_chat_refs_cache"] = chat_store.list_message_refs(conv_id) or {}
        except Exception:
            st.session_state["_chat_refs_cache"] = {}
        st.session_state["_chat_refs_cache_conv"] = conv_id
    refs_by_user = st.session_state.get("_chat_refs_cache") or {}
    if not isinstance(refs_by_user, dict):
        refs_by_user = {}

    def _render_refs_for_user(user_msg_id: int, prompt_text: str, *, pending: bool = False) -> None:
        ref_pack = refs_by_user.get(user_msg_id) if isinstance(refs_by_user, dict) else None
        hits_hist: list[dict] = []
        prompt_hist = str(prompt_text or "").strip()
        if isinstance(ref_pack, dict):
            hits_hist = list(ref_pack.get("hits") or [])
            p2 = str(ref_pack.get("prompt") or "").strip()
            if p2:
                prompt_hist = p2

        with st.container():
            st.markdown("<div class='msg-refs'>", unsafe_allow_html=True)
            p_sig_hist = hashlib.sha1(prompt_hist.encode("utf-8", "ignore")).hexdigest()[:8] if prompt_hist else f"msg{user_msg_id}"
            open_key_hist = f"hist_{conv_id}_{user_msg_id}_refs_open_{p_sig_hist}"
            with st.expander(S["refs"], expanded=bool(st.session_state.get(open_key_hist) or False)):
                if hits_hist:
                    _render_refs(
                        hits_hist,
                        prompt=prompt_hist,
                        show_heading=False,
                        key_ns=f"hist_{conv_id}_{user_msg_id}",
                        settings=settings,
                    )
                elif pending:
                    st.caption("（参考定位生成中…）")
                else:
                    st.caption("（未命中知识库片段：这条回答没有可定位的参考位置。）")
            st.markdown("</div>", unsafe_allow_html=True)

    if (not msgs) and (not running_for_conv):
        st.markdown(f"<div class='chat-empty-state'>{html.escape(S['no_msgs'])}</div>", unsafe_allow_html=True)
    else:
        render_msgs = list(msgs)
        hidden_msgs = 0
        if running_for_conv:
            live_window = 20
            if len(render_msgs) > live_window:
                hidden_msgs = len(render_msgs) - live_window
                render_msgs = render_msgs[-live_window:]
            if hidden_msgs > 0:
                st.caption(f"（为保证流式输出流畅，已折叠更早的 {hidden_msgs} 条消息）")

        last_user_for_refs = None
        last_user_msg_id = 0
        shown_refs_user_ids: set[int] = set()
        idx_offset = max(0, len(msgs) - len(render_msgs))
        for idx, m in enumerate(render_msgs, start=idx_offset):
            role = m.get("role", "user")
            content = m.get("content", "") or ""
            try:
                msg_id = int(m.get("id") or 0)
            except Exception:
                msg_id = 0

            if role == "user":
                st.markdown("<div class='msg-meta'>你</div>", unsafe_allow_html=True)
                safe = html.escape(content).replace("\n", "<br/>")
                st.markdown(f"<div class='msg-user'>{safe}</div>", unsafe_allow_html=True)
                last_user_for_refs = content
                if msg_id > 0:
                    last_user_msg_id = msg_id
            else:
                pending = False
                if _is_live_assistant_text(content):
                    pending_tid = _live_assistant_task_id(content)
                    t0 = _gen_get_task(session_id)
                    if isinstance(t0, dict) and str(t0.get("id") or "") == pending_tid:
                        pending = str(t0.get("status") or "") == "running"
                        stage = str(t0.get("stage") or "-")
                        if pending:
                            st.markdown("<div class='msg-meta'>AI（生成中）</div>", unsafe_allow_html=True)
                            partial = str(t0.get("partial") or "").strip()
                            notice, body = _split_kb_miss_notice(partial)
                            if notice:
                                st.markdown(f"<div class='kb-notice'>{html.escape(notice)}</div>", unsafe_allow_html=True)
                            if (body or "").strip():
                                st.markdown(_normalize_math_markdown(body))
                            else:
                                st.caption("（生成中…）")
                            st.caption(f"阶段：{stage}")
                        else:
                            ans0 = str(t0.get("answer") or "").strip()
                            if ans0:
                                st.markdown("<div class='msg-meta'>AI</div>", unsafe_allow_html=True)
                                _render_answer_copy_bar(ans0, key_ns=f"copy_{idx}_done")
                                notice, body = _split_kb_miss_notice(ans0)
                                if notice:
                                    st.markdown(f"<div class='kb-notice'>{html.escape(notice)}</div>", unsafe_allow_html=True)
                                if (body or "").strip():
                                    st.markdown(_normalize_math_markdown(body))
                            else:
                                st.markdown("<div class='msg-meta'>AI（处理中）</div>", unsafe_allow_html=True)
                                st.caption("处理中…")
                    else:
                        st.markdown("<div class='msg-meta'>AI（处理中）</div>", unsafe_allow_html=True)
                        st.caption("处理中…")
                else:
                    st.markdown("<div class='msg-meta'>AI</div>", unsafe_allow_html=True)
                    msg_key = hashlib.md5((st.session_state.get("conv_id", "") + "|" + str(idx)).encode("utf-8", "ignore")).hexdigest()[:10]
                    _render_answer_copy_bar(content, key_ns=f"copy_{msg_key}")
                    notice, body = _split_kb_miss_notice(content or "")
                    if notice:
                        st.markdown(f"<div class='kb-notice'>{html.escape(notice)}</div>", unsafe_allow_html=True)
                    if (body or "").strip():
                        st.markdown(_normalize_math_markdown(body))

                if (last_user_msg_id > 0) and (last_user_msg_id not in shown_refs_user_ids):
                    _render_refs_for_user(last_user_msg_id, str(last_user_for_refs or "").strip(), pending=pending)
                    shown_refs_user_ids.add(last_user_msg_id)
            st.markdown("")

    st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)

    with st.form(key="prompt_form", clear_on_submit=True):
        prompt_val = st.text_area(" ", height=96, key="prompt_text")
        action_cols = st.columns([8.4, 1.2, 1.2])
        with action_cols[1]:
            if running_for_conv:
                stop_clicked = st.form_submit_button("■", help="停止输出")
            else:
                stop_clicked = False
                st.markdown("<div style='height:2.25rem;'></div>", unsafe_allow_html=True)
        with action_cols[2]:
            submitted = st.form_submit_button("↑")

    components.html(
        """
<script>
(function () {
  const host = window.parent || window;
  const root = host.document;
  if (!root || !root.body) return;

  const NS = "__kbDockManagerStableV3";
  if (host[NS] && typeof host[NS].destroy === "function") {
    try { host[NS].destroy(); } catch (e) {}
  }

  const RESIZE_CLASS = "kb-resizing";
  const DOCK_SIDE_GAP = 35;
  const DOCK_RIGHT_GAP = 35;
  const MIN_WIDTH = 320;
  const state = {
    raf: 0,
    timer: 0,
    ro: null,
    mo: null,
    dragging: false,
    form: null,
    ta: null,
    onMouseDown: null,
    onPointerDown: null,
    onTouchStart: null,
    onMouseMove: null,
    onPointerMove: null,
    onMouseUp: null,
    onPointerUp: null,
    onPointerCancel: null,
    onTouchEnd: null,
    onBlur: null,
    onResize: null,
  };

  function isInsideStaleNode(el) {
    try { return !!(el && el.closest && el.closest('[data-stale="true"]')); } catch (e) { return false; }
  }
  function pickFresh(nodes) {
    for (const n of (nodes || [])) {
      if (!n) continue;
      if (!isInsideStaleNode(n)) return n;
    }
    return (nodes && nodes.length) ? nodes[0] : null;
  }
  function findMainRegion() {
    return pickFresh(root.querySelectorAll('section.main'));
  }
  function findMainContainer() {
    return pickFresh([
      ...root.querySelectorAll('section.main .block-container'),
      ...root.querySelectorAll('[data-testid="stMainBlockContainer"]'),
      ...root.querySelectorAll('.block-container')
    ]);
  }
  function findSidebar() {
    return pickFresh(root.querySelectorAll('section[data-testid="stSidebar"]'));
  }
  function isSendBtnText(t) {
    const s = String(t || "").trim();
    return s === "发送" || s === "↑" || s === "■" || s.toLowerCase() === "send";
  }
  function hasSendButton(form) {
    if (!form) return false;
    const btns = form.querySelectorAll('button');
    for (const b of btns) {
      if (isSendBtnText(b.innerText || b.textContent || "")) return true;
    }
    return false;
  }
  function findPromptFormAndTextarea() {
    const forms = root.querySelectorAll('div[data-testid="stForm"], form');
    for (const form of forms) {
      if (isInsideStaleNode(form)) continue;
      const ta =
        form.querySelector('div[data-testid="stTextArea"] textarea') ||
        form.querySelector('.stTextArea textarea') ||
        form.querySelector('textarea');
      if (!ta || isInsideStaleNode(ta)) continue;
      if (hasSendButton(form)) return { form, ta };
    }
    return { form: null, ta: null };
  }
  function setResizing(on) {
    if (on) root.body.classList.add(RESIZE_CLASS);
    else root.body.classList.remove(RESIZE_CLASS);
  }
  function placeDock(form) {
    if (!form) return;
    const mainContainer = findMainContainer();
    const mainRegion = findMainRegion();
    const sidebar = findSidebar();
    const anchor = mainContainer || mainRegion;
    if (!anchor && !sidebar) return;

    const viewportW = Math.max(0, (host.innerWidth || root.documentElement.clientWidth || 0));
    const anchorRect = (anchor && anchor.getBoundingClientRect) ? anchor.getBoundingClientRect() : null;
    const sidebarRect = (sidebar && sidebar.getBoundingClientRect) ? sidebar.getBoundingClientRect() : null;

    let leftBound = DOCK_SIDE_GAP;
    let rightBound = Math.max(leftBound + MIN_WIDTH, viewportW - DOCK_RIGHT_GAP);

    if (anchorRect && isFinite(anchorRect.left) && isFinite(anchorRect.right) && anchorRect.width > 10) {
      leftBound = Math.max(leftBound, Math.floor(anchorRect.left) + DOCK_SIDE_GAP);
      rightBound = Math.min(rightBound, Math.floor(anchorRect.right) - DOCK_SIDE_GAP);
    }
    if (sidebarRect && isFinite(sidebarRect.right) && sidebarRect.width > 10) {
      leftBound = Math.max(leftBound, Math.floor(sidebarRect.right) + DOCK_SIDE_GAP);
    }
    if (!isFinite(leftBound) || !isFinite(rightBound)) return;

    rightBound = Math.min(rightBound, viewportW - DOCK_RIGHT_GAP);
    if (rightBound - leftBound < MIN_WIDTH) rightBound = leftBound + MIN_WIDTH;
    if (rightBound > viewportW - DOCK_RIGHT_GAP) {
      rightBound = viewportW - DOCK_RIGHT_GAP;
      leftBound = Math.max(DOCK_SIDE_GAP, rightBound - MIN_WIDTH);
    }

    const dockLeft = Math.max(DOCK_SIDE_GAP, Math.floor(leftBound));
    const dockWidth = Math.max(MIN_WIDTH, Math.floor(rightBound - dockLeft));

    form.classList.add('kb-input-dock', 'kb-dock-positioned');
    form.style.left = dockLeft + 'px';
    form.style.right = 'auto';
    form.style.width = dockWidth + 'px';
    form.style.transform = 'none';
  }
  function bindCtrlEnter(ta, form) {
    if (!ta || ta.dataset.kbCtrlEnterHooked === "1") return;
    ta.dataset.kbCtrlEnterHooked = "1";
    ta.addEventListener("keydown", function (e) {
      const isCtrlEnter = (e.ctrlKey || e.metaKey) && e.key === "Enter";
      if (!isCtrlEnter) return;
      e.preventDefault();
      const btns = (form || root).querySelectorAll("button");
      for (const b of btns) {
        if ((b.innerText || b.textContent || "").trim() === "↑") {
          b.click();
          break;
        }
      }
    }, { capture: true });
  }
  function hook() {
    const hit = findPromptFormAndTextarea();
    if (!hit.form || !hit.ta) return;
    state.form = hit.form;
    state.ta = hit.ta;
    state.form.classList.add("kb-input-dock");
    placeDock(state.form);
    bindCtrlEnter(state.ta, state.form);
  }
  function scheduleHook() {
    if (state.raf) return;
    state.raf = host.requestAnimationFrame(function () {
      state.raf = 0;
      hook();
    });
  }
  function startDragIfNearSidebarEdge(e) {
    const sidebar = findSidebar();
    if (!sidebar || !e) return;
    const clientX = Number(e.clientX);
    if (!isFinite(clientX)) return;
    const rect = sidebar.getBoundingClientRect();
    if (!rect || !isFinite(rect.right)) return;
    const nearEdge = Math.abs(rect.right - clientX) <= 24;
    if (!nearEdge) return;
    state.dragging = true;
    setResizing(true);
    scheduleHook();
  }
  function onDragMove() {
    if (!state.dragging) return;
    scheduleHook();
  }
  function stopDrag() {
    if (!state.dragging) return;
    state.dragging = false;
    setResizing(false);
    scheduleHook();
  }
  function installListeners() {
    state.onMouseDown = startDragIfNearSidebarEdge;
    state.onPointerDown = startDragIfNearSidebarEdge;
    state.onTouchStart = function (e) {
      const t = (e.touches && e.touches[0]) ? e.touches[0] : null;
      if (t) startDragIfNearSidebarEdge(t);
    };
    state.onMouseMove = onDragMove;
    state.onPointerMove = onDragMove;
    state.onMouseUp = stopDrag;
    state.onPointerUp = stopDrag;
    state.onPointerCancel = stopDrag;
    state.onTouchEnd = stopDrag;
    state.onBlur = stopDrag;
    state.onResize = scheduleHook;

    root.addEventListener("mousedown", state.onMouseDown, true);
    root.addEventListener("pointerdown", state.onPointerDown, true);
    root.addEventListener("touchstart", state.onTouchStart, true);
    root.addEventListener("mousemove", state.onMouseMove, true);
    root.addEventListener("pointermove", state.onPointerMove, true);
    root.addEventListener("mouseup", state.onMouseUp, true);
    root.addEventListener("pointerup", state.onPointerUp, true);
    root.addEventListener("pointercancel", state.onPointerCancel, true);
    root.addEventListener("touchend", state.onTouchEnd, true);
    host.addEventListener("blur", state.onBlur, true);
    host.addEventListener("resize", state.onResize, { passive: true });
  }
  function installObservers() {
    if (typeof ResizeObserver !== "undefined") {
      try {
        state.ro = new ResizeObserver(function () { scheduleHook(); });
        const candidates = [root.documentElement, root.body, findSidebar(), findMainContainer(), findMainRegion()];
        for (const c of candidates) {
          if (c) state.ro.observe(c);
        }
      } catch (e) {}
    }
    if (typeof MutationObserver !== "undefined") {
      try {
        state.mo = new MutationObserver(function () { scheduleHook(); });
        state.mo.observe(root.body, { childList: true, subtree: true, attributes: true });
      } catch (e) {}
    }
  }
  function destroy() {
    try { if (state.timer) host.clearInterval(state.timer); } catch (e) {}
    try { if (state.raf) host.cancelAnimationFrame(state.raf); } catch (e) {}
    try { if (state.ro) state.ro.disconnect(); } catch (e) {}
    try { if (state.mo) state.mo.disconnect(); } catch (e) {}
    try { root.removeEventListener("mousedown", state.onMouseDown, true); } catch (e) {}
    try { root.removeEventListener("pointerdown", state.onPointerDown, true); } catch (e) {}
    try { root.removeEventListener("touchstart", state.onTouchStart, true); } catch (e) {}
    try { root.removeEventListener("mousemove", state.onMouseMove, true); } catch (e) {}
    try { root.removeEventListener("pointermove", state.onPointerMove, true); } catch (e) {}
    try { root.removeEventListener("mouseup", state.onMouseUp, true); } catch (e) {}
    try { root.removeEventListener("pointerup", state.onPointerUp, true); } catch (e) {}
    try { root.removeEventListener("pointercancel", state.onPointerCancel, true); } catch (e) {}
    try { root.removeEventListener("touchend", state.onTouchEnd, true); } catch (e) {}
    try { host.removeEventListener("blur", state.onBlur, true); } catch (e) {}
    try { host.removeEventListener("resize", state.onResize, false); } catch (e) {}
    setResizing(false);
  }

  host[NS] = { destroy, schedule: scheduleHook };
  installListeners();
  installObservers();
  state.timer = host.setInterval(scheduleHook, 120);
  scheduleHook();
})();
</script>
        """,
        height=0,
    )

    if stop_clicked:
        t0 = _gen_get_task(session_id)
        if isinstance(t0, dict):
            _gen_mark_cancel(session_id, str(t0.get("id") or ""))
        st.experimental_rerun()

    if submitted:
        txt = (prompt_val or "").strip()
        if txt:
            t0 = _gen_get_task(session_id)
            if isinstance(t0, dict) and str(t0.get("status") or "") == "running":
                st.warning("上一条回答还在生成中，请先停止或等待完成。")
                return

            task_id = uuid.uuid4().hex[:12]
            try:
                user_msg_id = int(chat_store.append_message(conv_id, "user", txt) or 0)
            except Exception:
                user_msg_id = 0
            try:
                assistant_msg_id = int(chat_store.append_message(conv_id, "assistant", _live_assistant_text(task_id)) or 0)
            except Exception:
                assistant_msg_id = 0
            chat_store.set_title_if_default(conv_id, txt)
            st.session_state["messages"] = chat_store.get_messages(conv_id)

            ok = _gen_start_task(
                {
                    "id": task_id,
                    "session_id": session_id,
                    "conv_id": conv_id,
                    "prompt": txt,
                    "prompt_sig": hashlib.sha1(txt.encode("utf-8", "ignore")).hexdigest()[:12],
                    "created_at": time.time(),
                    "chat_db": str(getattr(settings, "chat_db_path", "") or ""),
                    "db_dir": str(db_dir),
                    "top_k": int(top_k),
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                    "deep_read": bool(deep_read),
                    "settings_obj": settings,
                    "user_msg_id": int(user_msg_id),
                    "assistant_msg_id": int(assistant_msg_id),
                }
            )
            if not ok:
                chat_store.update_message_content(assistant_msg_id, "（启动生成失败）")
            st.experimental_rerun()

    # Adaptive server-side polling:
    # - fast when new tokens arrive
    # - slower when model is still retrieving/deep-reading (to reduce UI jank)
    if running_for_conv and (not submitted) and (not stop_clicked):
        t_live = _gen_get_task(session_id) or {}
        poll_key = "_gen_poll_state"
        prev = st.session_state.get(poll_key) or {}
        prev_tid = str(prev.get("tid") or "")
        prev_chars = int(prev.get("chars") or 0)
        prev_stage = str(prev.get("stage") or "")

        cur_tid = str(t_live.get("id") or "")
        try:
            cur_chars = int(t_live.get("char_count") or 0)
        except Exception:
            cur_chars = 0
        cur_stage = str(t_live.get("stage") or "")

        if cur_tid != prev_tid:
            delay_s = 0.14
        elif cur_chars > prev_chars:
            delay_s = 0.16
        elif cur_stage != prev_stage:
            delay_s = 0.22
        else:
            delay_s = 0.42

        st.session_state[poll_key] = {"tid": cur_tid, "chars": cur_chars, "stage": cur_stage}
        time.sleep(delay_s)
        st.experimental_rerun()
    else:
        st.session_state.pop("_gen_poll_state", None)

def _page_library(settings, lib_store: LibraryStore, db_dir: Path, prefs_path: Path, prefs: dict, retriever_reload_flag: dict) -> None:
    _bg_ensure_started()
    bg_auto_rerun_issued = False

    retriever_err = str(st.session_state.get("retriever_load_error") or "").strip()
    if retriever_err:
        st.error(f"\u77e5\u8bc6\u5e93\u52a0\u8f7d\u5931\u8d25\uff1a{retriever_err}")
        st.caption("\u4f60\u4ecd\u7136\u53ef\u4ee5\u5728\u8fd9\u4e2a\u9875\u9762\u8f6c\u6362/\u66f4\u65b0\u77e5\u8bc6\u5e93\uff0c\u4fee\u590d\u540e\u56de\u5230\u300c\u5bf9\u8bdd\u300d\u9875\u5373\u53ef\u6b63\u5e38\u68c0\u7d22\u3002")

    try:
        r = st.session_state.get("retriever")
        if bool(getattr(r, "is_empty", False)):
            st.info("\u5f53\u524d DB \u8fd8\u6ca1\u6709\u4efb\u4f55 chunks\u3002\u7b2c\u4e00\u6b21\u4f7f\u7528\u8bf7\uff1a\u9009\u62e9 PDF \u76ee\u5f55 \u2192 \u8f6c\u6362\u6210 MD \u2192 \u70b9\u51fb\u300c\u66f4\u65b0\u77e5\u8bc6\u5e93\u300d\u3002")
    except Exception:
        pass

    def _guess_default_pdf_dir() -> Path:
        env = (os.environ.get("KB_PDF_DIR") or "").strip().strip("'\"")
        if env:
            try:
                return Path(env).expanduser().resolve()
            except Exception:
                return Path(env).expanduser()

        # Prefer common locations if they exist; otherwise use a dedicated folder under home.
        home = Path.home()
        candidates = [
            home / "research-papers",
            home / "ResearchPapers",
            home / "Papers",
            home / "Downloads",
        ]
        for c in candidates:
            try:
                if c.exists():
                    return c.resolve()
            except Exception:
                continue
        return home / "Pi_zaya_pdfs"

    default_pdf_dir = _guess_default_pdf_dir()
    default_md_dir = Path(os.environ.get("KB_MD_DIR", str(db_dir))).expanduser().resolve()

    pdf_default = st.session_state.get("pdf_dir") or prefs.get("pdf_dir") or str(default_pdf_dir)
    md_default = st.session_state.get("md_dir") or prefs.get("md_dir") or str(default_md_dir)

    # Directory pickers + manual edit (best of both)
    if "pdf_dir_input" not in st.session_state:
        st.session_state["pdf_dir_input"] = str(pdf_default)
    if "md_dir_input" not in st.session_state:
        st.session_state["md_dir_input"] = str(md_default)

    def _pick_pdf_dir() -> None:
        picked = _pick_directory_dialog(str(st.session_state.get("pdf_dir_input") or pdf_default))
        if picked:
            st.session_state["pdf_dir_input"] = picked
            prefs2 = dict(prefs)
            prefs2["pdf_dir"] = str(Path(picked).expanduser().resolve())
            save_prefs(prefs_path, prefs2)

    def _pick_md_dir() -> None:
        picked = _pick_directory_dialog(str(st.session_state.get("md_dir_input") or md_default))
        if picked:
            st.session_state["md_dir_input"] = picked
            prefs2 = dict(prefs)
            prefs2["md_dir"] = str(Path(picked).expanduser().resolve())
            save_prefs(prefs_path, prefs2)

    row1 = st.columns([12, 2])
    with row1[1]:
        # Align with text_input which has a label line above the input.
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        st.button("...", key="pick_pdf_dir", on_click=_pick_pdf_dir, help="选择 PDF 目录")
    with row1[0]:
        st.text_input(S["lib_root"], key="pdf_dir_input")

    row2 = st.columns([12, 2])
    with row2[1]:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        st.button("...", key="pick_md_dir", on_click=_pick_md_dir, help="选择 Markdown 输出目录")
    with row2[0]:
        st.text_input(S["md_root"], key="md_dir_input")

    pdf_dir = Path((st.session_state.get("pdf_dir_input") or str(pdf_default)).strip()).expanduser()
    md_out_root = Path((st.session_state.get("md_dir_input") or str(md_default)).strip()).expanduser()

    st.session_state["pdf_dir"] = str(pdf_dir)
    st.session_state["md_dir"] = str(md_out_root)

    # Persist across restarts.
    next_prefs = dict(prefs)
    next_prefs["pdf_dir"] = str(Path(pdf_dir).expanduser().resolve())
    next_prefs["md_dir"] = str(Path(md_out_root).expanduser().resolve())
    if next_prefs != prefs:
        save_prefs(prefs_path, next_prefs)
        prefs.update(next_prefs)

    ensure_dir(pdf_dir)
    ensure_dir(md_out_root)

    # List PDFs early (used by the rename manager + listing below).
    # Keep it fast: avoid stat() for sorting here.
    try:
        pdfs = _list_pdf_paths_fast(pdf_dir)
        pdfs.sort(key=lambda p: p.name.lower())
    except Exception:
        pdfs = []

    # Background status: keep it non-blocking and compact here.
    # Detailed progress bar is rendered under the conversion section ("已有文献") for better UX when scrolling.
    bg = _bg_snapshot()
    if bg.get("running") or (bg.get("total", 0) and bg.get("done", 0) < bg.get("total", 0)):
        st.markdown(
            "<div class='refbox'>\u540e\u53f0\u8f6c\u6362\u4efb\u52a1\u8fd0\u884c\u4e2d\uff1a\u8bf7\u5728\u4e0b\u65b9\u201c\u5df2\u6709\u6587\u732e\u201d\u533a\u57df\u67e5\u770b\u8fdb\u5ea6\u6761\u3002</div>",
            unsafe_allow_html=True,
        )

    # Prompt once when user selects a directory that likely needs naming cleanup.
    try:
        dir_sig = str(Path(pdf_dir).expanduser().resolve())
    except Exception:
        dir_sig = str(pdf_dir)
    dismissed = st.session_state.setdefault("rename_prompt_dismissed_dirs", set())
    if isinstance(dismissed, set) and (dir_sig not in dismissed) and pdfs:
        # Show prompt only when we see at least a few "non-normalized" names.
        non_norm = 0
        for p in pdfs[:40]:
            if p.name.lower().startswith("__upload__"):
                continue
            if not _is_normalized_pdf_stem(p.stem):
                non_norm += 1
            if non_norm >= 3:
                break
        if non_norm >= 3:
            st.markdown(
                "<div class='kb-notice'>检测到你设置了新的 PDF 目录：其中有些文件名可能不是「期刊-年份-标题」。要不要我根据 PDF 内容识别信息并给出重命名建议？</div>",
                unsafe_allow_html=True,
            )
            c_p = st.columns([1.1, 1.0, 10])
            with c_p[0]:
                if st.button("查看建议", key="rename_prompt_open"):
                    st.session_state["rename_mgr_open"] = True
                    st.session_state["rename_scan_trigger"] = False
                    st.session_state["rename_scan_scope"] = "最近 30 个"
                    st.session_state["rename_scan_use_llm"] = False
                    st.experimental_rerun()
            with c_p[1]:
                if st.button("以后再说", key="rename_prompt_dismiss"):
                    try:
                        dismissed.add(dir_sig)
                    except Exception:
                        pass
                    st.experimental_rerun()

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    cols = st.columns([1, 1, 1, 1])
    with cols[0]:
        if st.button(S["open_dir"], key="open_pdf_dir"):
            open_in_explorer(pdf_dir)
    with cols[1]:
        if st.button(S["reindex_now"], key="reindex_btn"):
            ingest_py = Path(__file__).resolve().parent / "ingest.py"
            if ingest_py.exists():
                subprocess.run(
                    [os.sys.executable, str(ingest_py), "--src", str(md_out_root), "--db", str(db_dir), "--incremental", "--prune"],
                    check=False,
                )
            st.info(S["run_ok"])
            retriever_reload_flag["reload"] = True
    with cols[2]:
        if st.button(S["cleanup"], key="cleanup_btn"):
            n = _cleanup_tmp_uploads(pdf_dir)
            st.info(f"{S['run_ok']}: cleaned {n} tmp uploads")
    with cols[3]:
        if st.button("文件名管理", key="rename_mgr_btn", help="根据 PDF 内容识别「期刊-年份-标题」并建议重命名"):
            st.session_state["rename_mgr_open"] = True
            st.session_state["rename_scan_trigger"] = False
            st.experimental_rerun()

    # "Small window" rename manager (modal-like overlay).
    if "rename_mgr_open" not in st.session_state:
        st.session_state["rename_mgr_open"] = False
    if "rename_scan_trigger" not in st.session_state:
        st.session_state["rename_scan_trigger"] = False
    if "rename_scan_scope" not in st.session_state:
        st.session_state["rename_scan_scope"] = "最近 30 个"
    if "rename_scan_use_llm" not in st.session_state:
        st.session_state["rename_scan_use_llm"] = False

    if bool(st.session_state.get("rename_mgr_open")):
        # If meta extraction heuristics changed, invalidate any cached suggestions/results in this session.
        expected_ver = str(PDF_META_EXTRACT_VERSION)
        if str(st.session_state.get("rename_mgr_cache_ver") or "") != expected_ver:
            st.session_state["rename_mgr_cache_ver"] = expected_ver
            st.session_state.pop("rename_pdf_meta_cache", None)
            st.session_state.pop("rename_scan_results_cache", None)

        # NOTE: Streamlit markdown cannot truly "wrap" widgets into an HTML modal.
        # A fixed overlay can block clicks and make users feel "stuck".
        # Use an in-page expander panel instead (fast, reliable, closable).
        with st.expander("PDF 文件名管理（期刊-年份-标题）", expanded=True):
            c_top = st.columns([1.0, 1.8, 1.6, 1.6, 6.0])
            with c_top[0]:
                if st.button("关闭", key="rename_mgr_close"):
                    st.session_state["rename_mgr_open"] = False
                    st.session_state["rename_scan_trigger"] = False
                    st.experimental_rerun()
            with c_top[1]:
                _scope_opts = ["最近 30 个", "最近 50 个", "最近 100 个", "全部"]
                _cur_scope = str(st.session_state.get("rename_scan_scope") or "最近 30 个")
                _scope_idx = _scope_opts.index(_cur_scope) if _cur_scope in _scope_opts else 0
                scope = st.selectbox(
                    "扫描范围",
                    options=_scope_opts,
                    index=_scope_idx,
                    key="rename_scan_scope",
                )
            with c_top[2]:
                st.checkbox("只显示需改名", value=True, key="rename_only_diff")
            with c_top[3]:
                st.checkbox("同时改名 MD", value=False, key="rename_also_md", help="如果已生成 Markdown，会把对应文件夹/主 md 尝试一起改名")

            use_llm = bool(st.session_state.get("rename_scan_use_llm"))
            with c_top[4]:
                st.checkbox("识别用 LLM（更准）", value=use_llm, key="rename_scan_use_llm", help="只在关键信息缺失时少量调用 LLM 补全（更准但更慢）")

            # Scan button
            c_scan = st.columns([1.2, 8.8])
            with c_scan[0]:
                clicked_scan = st.button("开始识别/刷新", key="rename_scan_btn")
            with c_scan[1]:
                st.markdown(
                    "<div class='refbox'>我会读取 PDF 内容（不是看文件名）来识别「期刊/会议、年份、标题」，然后给出建议文件名。你可以手动编辑后再应用。</div>",
                    unsafe_allow_html=True,
                )

        # Cache key for scan results (include extractor version to avoid stale meta after upgrades).
        scope_s = str(scope or "")
        llm_flag = "llm1" if bool(st.session_state.get("rename_scan_use_llm")) else "llm0"
        scan_key = hashlib.sha1((dir_sig + "|" + scope_s + "|" + llm_flag + "|ver:" + expected_ver).encode("utf-8", "ignore")).hexdigest()[:16]
        results_cache = st.session_state.setdefault("rename_scan_results_cache", {})
        results = results_cache.get(scan_key)

        # IMPORTANT: only scan when the user explicitly clicks the button.
        # Auto-scanning can block the UI and make the panel feel "stuck".
        trigger = bool(clicked_scan)
        if trigger:
            meta_cache = st.session_state.setdefault("rename_pdf_meta_cache", {})
            out_rows: list[dict] = []
            # Only build the scan list when the user explicitly requests a scan.
            if scope_s.startswith("最近"):
                try:
                    n_scope = int(re.sub(r"\D+", "", scope_s) or "50")
                except Exception:
                    n_scope = 50
                scan_pdfs = _select_recent_pdf_paths(pdf_dir, max(1, n_scope))
            else:
                scan_pdfs = list(pdfs)

            total = len(scan_pdfs)
            prog = st.progress(0.0) if total > 0 else None
            llm_budget = 8
            llm_used = 0
            with st.spinner(f"正在识别 PDF 信息（{total} 个）..."):
                for i, pdf in enumerate(scan_pdfs, start=1):
                    if pdf.name.lower().startswith("__upload__"):
                        continue
                    try:
                        st_m = float(pdf.stat().st_mtime)
                        st_s = int(pdf.stat().st_size)
                    except Exception:
                        st_m, st_s = 0.0, 0
                    cache_k = hashlib.sha1(
                        (str(pdf) + "|" + str(st_m) + "|" + str(st_s) + "|" + llm_flag + "|ver:" + expected_ver).encode("utf-8", "ignore")
                    ).hexdigest()[:16]
                    sugg = meta_cache.get(cache_k)
                    if not isinstance(sugg, PdfMetaSuggestion):
                        try:
                            # Fast pass first (heuristics only).
                            sugg = extract_pdf_meta_suggestion(pdf, settings=None)
                            # Optional refinement with a small budget to avoid UI "freezing".
                            if bool(st.session_state.get("rename_scan_use_llm")) and settings:
                                needs = (not (sugg.title or "").strip()) or (not (sugg.year or "").strip()) or (not (sugg.venue or "").strip())
                                if needs and llm_used < llm_budget:
                                    sugg2 = extract_pdf_meta_suggestion(pdf, settings=settings)
                                    if sugg2:
                                        sugg = sugg2
                                    llm_used += 1
                        except Exception:
                            sugg = PdfMetaSuggestion()
                        meta_cache[cache_k] = sugg
                    base = build_base_name(venue=sugg.venue, year=sugg.year, title=sugg.title)
                    base = (base or "").strip()
                    diff = bool(base) and (base != pdf.stem)
                    out_rows.append(
                        {
                            "path": str(pdf),
                            "old": pdf.name,
                            "old_stem": pdf.stem,
                            "suggest": base,
                            "diff": diff,
                            "meta": {"venue": sugg.venue, "year": sugg.year, "title": sugg.title},
                        }
                    )
                    if prog is not None:
                        prog.progress(min(1.0, i / max(1, total)))
            if prog is not None:
                prog.empty()
            results_cache[scan_key] = out_rows
            results = out_rows
        elif results is None:
            st.caption("点击「开始识别/刷新」后才会读取 PDF 进行识别（避免打开窗口就卡住）。")

        # Render results
        rows = list(results or [])
        only_diff = bool(st.session_state.get("rename_only_diff"))
        if only_diff:
            rows = [r for r in rows if bool(r.get("diff"))]

        if not rows:
            st.caption("（没有需要改名的文件，或识别不到可用信息）")
        else:
            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
            st.caption(f"共 {len(rows)} 条建议（只会在你点击“应用重命名”后真正改名）")

            # Bulk select
            c_bulk = st.columns([1.4, 1.4, 7.2])
            with c_bulk[0]:
                if st.button("全选", key="rename_sel_all"):
                    for r in rows:
                        uid = hashlib.md5(str(r.get("path") or "").encode("utf-8", "ignore")).hexdigest()[:10]
                        st.session_state[f"rename_sel_{uid}"] = True
                    st.experimental_rerun()
            with c_bulk[1]:
                if st.button("全不选", key="rename_sel_none"):
                    for r in rows:
                        uid = hashlib.md5(str(r.get("path") or "").encode("utf-8", "ignore")).hexdigest()[:10]
                        st.session_state[f"rename_sel_{uid}"] = False
                    st.experimental_rerun()
            with c_bulk[2]:
                st.caption("建议格式：期刊-年份-标题（可编辑）")

            for r in rows[:400]:
                p = str(r.get("path") or "")
                uid = hashlib.md5(p.encode("utf-8", "ignore")).hexdigest()[:10]
                old = str(r.get("old") or "")
                suggest = str(r.get("suggest") or "")
                meta = r.get("meta") or {}
                venue = str((meta or {}).get("venue") or "")
                year = str((meta or {}).get("year") or "")
                title = str((meta or {}).get("title") or "")

                st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
                c_row = st.columns([0.9, 4.9, 5.6, 1.6])
                with c_row[0]:
                    st.checkbox("选择", value=bool(st.session_state.get(f"rename_sel_{uid}", False)), key=f"rename_sel_{uid}")
                with c_row[1]:
                    st.markdown(f"<div class='meta-kv'><b>当前</b>：{html.escape(old)}</div>", unsafe_allow_html=True)
                    meta_line = " | ".join([x for x in [venue, year, title[:80] + ("…" if len(title) > 80 else "")] if x])
                    if meta_line:
                        st.caption(f"识别：{meta_line}")
                with c_row[3]:
                    pdf_path = Path(p)
                    if st.button("打开", key=f"rename_open_{uid}", help="打开 PDF 查看"):
                        try:
                            os.startfile(str(pdf_path))  # type: ignore[attr-defined]
                        except Exception:
                            try:
                                open_in_explorer(pdf_path)
                            except Exception as e:
                                st.warning(f"打开失败：{e}")
                    if st.button("定位", key=f"rename_loc_{uid}", help="在资源管理器中定位"):
                        try:
                            open_in_explorer(pdf_path)
                        except Exception as e:
                            st.warning(f"定位失败：{e}")
                with c_row[2]:
                    if not suggest:
                        st.caption("（识别不到可用信息，跳过）")
                    else:
                        st.text_input("建议新文件名（不含 .pdf）", value=suggest, key=f"rename_new_{uid}")

            # Apply rename
            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
            c_apply = st.columns([1.8, 8.2])
            with c_apply[0]:
                if st.button("应用重命名", key="rename_apply_btn"):
                    ops = []
                    also_md = bool(st.session_state.get("rename_also_md"))
                    for r in rows:
                        p = str(r.get("path") or "")
                        uid = hashlib.md5(p.encode("utf-8", "ignore")).hexdigest()[:10]
                        if not bool(st.session_state.get(f"rename_sel_{uid}", False)):
                            continue
                        src = Path(p)
                        if not src.exists():
                            ops.append(("fail", f"不存在：{src}"))
                            continue
                        new_base = str(st.session_state.get(f"rename_new_{uid}") or "").strip()
                        new_base = _sanitize_filename_component(new_base)
                        if not new_base:
                            ops.append(("skip", f"跳过（空名字）：{src.name}"))
                            continue
                        if new_base == src.stem:
                            ops.append(("skip", f"跳过（未变化）：{src.name}"))
                            continue
                        dest = _unique_pdf_path(pdf_dir, new_base)
                        try:
                            src.rename(dest)
                            try:
                                lib_store.update_path(src, dest)
                            except Exception:
                                pass
                            ops.append(("ok", f"{src.name} → {dest.name}"))
                        except Exception as e:
                            ops.append(("fail", f"{src.name} 重命名失败：{e}"))
                            continue

                        if also_md:
                            try:
                                old_folder = Path(md_out_root) / src.stem
                                new_folder = Path(md_out_root) / dest.stem
                                if old_folder.exists() and (not new_folder.exists()):
                                    old_folder.rename(new_folder)
                                    old_main = new_folder / f"{src.stem}.en.md"
                                    if old_main.exists():
                                        new_main = new_folder / f"{dest.stem}.en.md"
                                        try:
                                            old_main.rename(new_main)
                                        except Exception:
                                            pass
                            except Exception:
                                pass

                    # Clear scan cache for this dir so the list refreshes next time.
                    try:
                        results_cache.pop(scan_key, None)
                    except Exception:
                        pass
                    # Do not auto-rescan; let the user click "开始识别/刷新" to refresh suggestions.

                    ok_n = sum(1 for s, _ in ops if s == "ok")
                    fail_n = sum(1 for s, _ in ops if s == "fail")
                    if ok_n:
                        st.success(f"已重命名：{ok_n} 个文件")
                    if fail_n:
                        st.warning(f"失败：{fail_n} 个文件（可能是文件正被占用）")
                    if ops:
                        with st.expander("查看详情", expanded=False):
                            for stt, msg in ops[:200]:
                                (st.caption if stt in ("ok", "skip") else st.warning)(msg)

                    # After applying, dismiss the prompt for this dir.
                    try:
                        dismissed.add(dir_sig)
                    except Exception:
                        pass
                    st.experimental_rerun()
            with c_apply[1]:
                st.markdown(
                    "<div class='refbox'>提示：如果你已经把某些 PDF 转换成 MD 并且已经建库，重命名后建议点一次「更新知识库」重新索引，以免旧路径残留。</div>",
                    unsafe_allow_html=True,
                )

            # end expander

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader(S["convert_opts"])
    no_llm = st.checkbox(S["no_llm"], value=False, key="lib_no_llm")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("\u5df2\u6709\u6587\u732e")
    st.caption("\u672a\u8f6c\u6362\u7684\u4f1a\u663e\u793a\u5728\u201c\u672a\u8f6c\u6362\u201d\uff0c\u53ef\u4e00\u952e\u6279\u91cf\u8f6c\u6362\u3002\u8f6c\u6362\u4efb\u52a1\u4f1a\u5728\u540e\u53f0\u8fd0\u884c\uff0c\u4e0d\u4f1a\u5361\u4f4f\u9875\u9762\u3002")

    if not pdfs:
        st.caption("\uff08\u8fd8\u6ca1\u6709\u627e\u5230 PDF\uff09")
    else:
        # Avoid blocking the UI when the folder is huge: show a fast subset by default.
        scope_opt = st.selectbox(
            "\u5217\u8868\u8303\u56f4",
            options=["200\uff08\u66f4\u5feb\uff09", "500", "\u5168\u90e8\uff08\u53ef\u80fd\u8f83\u6162\uff09"],
            index=0,
            key="lib_pdf_list_scope",
        )
        if scope_opt.startswith("200"):
            pdfs_view = pdfs[:200]
        elif scope_opt.startswith("500"):
            pdfs_view = pdfs[:500]
        else:
            pdfs_view = pdfs
        if len(pdfs_view) < len(pdfs):
            st.caption(f"\u4e3a\u4fdd\u8bc1\u901f\u5ea6\uff0c\u5f53\u524d\u4ec5\u5c55\u793a {len(pdfs_view)}/{len(pdfs)} \u4e2a PDF\u3002")

        converted = []
        pending = []
        for pdf in pdfs_view:
            md_folder = Path(md_out_root) / pdf.stem
            md_main = md_folder / f"{pdf.stem}.en.md"
            md_exists = md_main.exists()
            if not md_exists and md_folder.exists():
                try:
                    any_md = next(iter(sorted(md_folder.glob('*.md'))), None)
                    if any_md:
                        md_main = any_md
                        md_exists = True
                except Exception:
                    pass
            item = {"pdf": pdf, "md_folder": md_folder, "md_main": md_main, "md_exists": md_exists}
            (converted if md_exists else pending).append(item)

        tabs = st.tabs([
            f"\u672a\u8f6c\u6362 ({len(pending)})",
            f"\u5df2\u8f6c\u6362 ({len(converted)})",
            f"\u5f53\u524d\u5217\u8868 ({len(pdfs_view)})",
        ])

        def render_bg_progress_under_tasks(*, key_ns: str) -> None:
            """
            Render background conversion progress where users are looking: under conversion tasks.
            """
            nonlocal bg_auto_rerun_issued
            bg2 = _bg_snapshot()
            running = bool(bg2.get("running")) or (bg2.get("total", 0) and bg2.get("done", 0) < bg2.get("total", 0))
            if not running:
                return
            done = int(bg2.get("done", 0) or 0)
            total = int(bg2.get("total", 0) or 0)
            cur = str(bg2.get("current") or "")
            last = str(bg2.get("last") or "").strip()
            p_done = int(bg2.get("cur_page_done", 0) or 0)
            p_total = int(bg2.get("cur_page_total", 0) or 0)
            p_msg = str(bg2.get("cur_page_msg") or "").strip()
            st.markdown("<div class='refbox'>\u540e\u53f0\u8f6c\u6362\u8fdb\u5ea6</div>", unsafe_allow_html=True)
            st.caption(f"{done}/{total}{(' | ' + cur) if cur else ''}")
            if total > 0:
                st.progress(min(1.0, done / max(1, total)))
            if p_total > 0:
                st.caption(f"\u5f53\u524d\u6587\u4ef6\u9875\u8fdb\u5ea6\uff1a{p_done}/{p_total}")
                st.progress(min(1.0, p_done / max(1, p_total)))
            elif bool(bg2.get("running")):
                st.caption("\u5f53\u524d\u6587\u4ef6\u5904\u7406\u4e2d\u2026")
            if p_msg and bool(bg2.get("running")):
                st.caption(p_msg)

            c_bg = st.columns([1.0, 1.0, 6.0])
            with c_bg[0]:
                if st.button("\u5237\u65b0", key=f"bg_refresh_under_{key_ns}"):
                    st.experimental_rerun()
            with c_bg[1]:
                if st.button("\u505c\u6b62", key=f"bg_cancel_under_{key_ns}"):
                    _bg_cancel_all()
                    st.experimental_rerun()
            with c_bg[2]:
                if last:
                    st.caption(f"\u6700\u8fd1\u4e00\u6761\uff1a{last}")

            # Auto refresh while running so the progress bar updates without manual clicks.
            auto_key = f"bg_auto_refresh_{key_ns}"
            if auto_key not in st.session_state:
                st.session_state[auto_key] = True
            auto = st.checkbox("\u81ea\u52a8\u5237\u65b0\u8fdb\u5ea6", value=bool(st.session_state.get(auto_key)), key=auto_key)
            if auto and running:
                components.html(
                    """
<script>
(function () {
  try {
    const root = window.parent;
    if (!root) return;
    if (root._kbAutoRefreshTimer) return;
    root._kbAutoRefreshTimer = setTimeout(function () {
      try {
        root._kbAutoRefreshTimer = null;
        root.postMessage({ isStreamlitMessage: true, type: "streamlit:rerunScript" }, "*");
      } catch (e) {}
    }, 1500);
  } catch (e) {}
})();
</script>
                    """,
                    height=0,
                )
                # Reliable fallback: some Streamlit versions ignore JS postMessage reruns.
                # Keep progress moving with a lightweight server-side polling rerun.
                if not bg_auto_rerun_issued:
                    bg_auto_rerun_issued = True
                    time.sleep(0.9)
                    st.experimental_rerun()

        def render_items(items: list[dict], *, show_missing_badge: bool, key_ns: str) -> None:
            from typing import Optional

            bg2 = _bg_snapshot()
            queue_tasks = list(bg2.get("queue") or [])
            current_name = str(bg2.get("current") or "")
            running_any = bool(bg2.get("running"))
            done = int(bg2.get("done", 0) or 0)
            total = int(bg2.get("total", 0) or 0)

            def _queue_pos(pdf_path: Path) -> Optional[int]:
                p = str(pdf_path)
                for i, t in enumerate(queue_tasks, start=1):
                    try:
                        if str(t.get("pdf") or "") == p:
                            return i
                    except Exception:
                        continue
                return None

            for idx, it in enumerate(items, start=1):
                pdf = it["pdf"]
                md_main = it["md_main"]
                md_exists = bool(it["md_exists"])
                uid = hashlib.md5(str(pdf).encode("utf-8", "ignore")).hexdigest()[:10]

                badge = "\u3010\u672a\u8f6c\u6362\u3011" if (show_missing_badge and not md_exists) else "\u3010\u5df2\u8f6c\u6362\u3011"
                title = f"{badge} {pdf.name}"

                with st.expander(title, expanded=False):
                    queued_pos = _queue_pos(pdf)
                    running_this = running_any and (current_name == pdf.name)
                    del_key = f"{key_ns}_del_state_{uid}"
                    if del_key not in st.session_state:
                        st.session_state[del_key] = False

                    if running_this:
                        st.markdown("<span class='pill run'>\u8f6c\u6362\u4e2d</span>", unsafe_allow_html=True)
                        # Show per-page progress for the current file when available.
                        p_done = int(bg2.get("cur_page_done", 0) or 0)
                        p_total = int(bg2.get("cur_page_total", 0) or 0)
                        if p_total > 0:
                            st.progress(min(1.0, p_done / max(1, p_total)))
                            st.caption(f"\u9875\u8fdb\u5ea6\uff1a{p_done}/{p_total}")
                        else:
                            st.progress(0.0)
                            st.caption("\u5904\u7406\u4e2d\u2026")
                    elif queued_pos is not None:
                        st.markdown("<span class='pill warn'>\u6392\u961f\u4e2d</span>", unsafe_allow_html=True)
                        st.caption(f"\u961f\u5217\u4f4d\u7f6e\uff1a{queued_pos}/{len(queue_tasks)}  |  \u540e\u53f0\u53ef\u7ee7\u7eed\u5207\u6362\u9875\u9762\u4f7f\u7528")
                    else:
                        if md_exists:
                            st.markdown("<span class='pill ok'>\u5df2\u8f6c\u6362</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("<span class='pill warn'>\u672a\u8f6c\u6362</span>", unsafe_allow_html=True)

                    c_top = st.columns([1.05, 1.05, 0.9, 1.2, 1.0, 2.4])
                    with c_top[0]:
                        if st.button("\u6253\u5f00 PDF", key=f"{key_ns}_open_pdf_{uid}"):
                            try:
                                os.startfile(str(pdf))  # type: ignore[attr-defined]
                            except Exception:
                                open_in_explorer(pdf.parent)
                    with c_top[1]:
                        if md_exists:
                            if st.button("\u6253\u5f00 MD", key=f"{key_ns}_open_md_{uid}"):
                                try:
                                    os.startfile(str(md_main))  # type: ignore[attr-defined]
                                except Exception:
                                    open_in_explorer(md_main.parent)
                        else:
                            st.write("")

                    with c_top[2]:
                        if md_exists:
                            replace = st.checkbox("\u66ff\u6362", value=False, key=f"{key_ns}_replace_md_{uid}")
                            st.caption("\u66ff\u6362\u5df2\u6709 MD")
                        else:
                            replace = False
                            st.write("")

                    with c_top[3]:
                        btn_label = "\u91cd\u65b0\u8f6c\u6362" if md_exists else "\u8f6c\u6362"
                        if st.button(btn_label, key=f"{key_ns}_convert_btn_{uid}"):
                            if md_exists and not replace:
                                st.info("\u68c0\u6d4b\u5230\u5df2\u6709 Markdown\uff1a\u5982\u9700\u91cd\u65b0\u751f\u6210\uff0c\u8bf7\u5148\u52fe\u9009\u201c\u66ff\u6362\u201d\u3002")
                            elif running_this or (queued_pos is not None):
                                st.info("\u6b64\u6587\u4ef6\u5df2\u5728\u540e\u53f0\u961f\u5217\u4e2d\u3002")
                            else:
                                _bg_enqueue(
                                    {
                                        "pdf": str(pdf),
                                        "out_root": str(md_out_root),
                                        "db_dir": str(db_dir),
                                        "no_llm": bool(no_llm),
                                        "eq_image_fallback": True,
                                        "replace": bool(replace),
                                        "name": pdf.name,
                                    }
                                )
                                st.info("\u5df2\u52a0\u5165\u540e\u53f0\u961f\u5217\uff08\u4e0d\u4f1a\u5361\u4f4f\u9875\u9762\uff09\uff0c\u4f60\u53ef\u4ee5\u5207\u6362\u53bb\u201c\u5bf9\u8bdd\u201d\u9875\u7ee7\u7eed\u4f7f\u7528\u3002")
                                st.experimental_rerun()

                    with c_top[4]:
                        # Delete entry point (confirmation UI is rendered below the button row
                        # to avoid nesting columns inside columns, which Streamlit forbids).
                        del_open = bool(st.session_state.get(del_key))
                        if running_this:
                            st.button("\u5220\u9664", key=f"{key_ns}_del_btn_{uid}", disabled=True)
                            st.caption("\u8f6c\u6362\u4e2d")
                        else:
                            if del_open:
                                if st.button("\u53d6\u6d88", key=f"{key_ns}_del_btn_{uid}"):
                                    st.session_state[del_key] = False
                                    st.experimental_rerun()
                            else:
                                if st.button("\u5220\u9664", key=f"{key_ns}_del_btn_{uid}", help="\u5220\u9664 PDF \u6587\u4ef6\uff08\u4e0d\u53ef\u6062\u590d\uff09"):
                                    st.session_state[del_key] = True
                                    st.experimental_rerun()

                    with c_top[5]:
                        md_name = md_main.name if md_exists else "\u2014"
                        st.markdown(
                            f"""<div class='meta-kv'>
<b>PDF</b>\uff1a{pdf.name}<br/>
<b>MD</b>\uff1a{md_name}
</div>""",
                            unsafe_allow_html=True,
                        )

                    # Delete confirmation panel (outside the button columns).
                    if (not running_this) and bool(st.session_state.get(del_key)):
                        st.warning("\u786e\u8ba4\u5220\u9664\u8fd9\u4e2a PDF\uff1f\u6b64\u64cd\u4f5c\u4e0d\u53ef\u6062\u590d\u3002")
                        also_md = st.checkbox(
                            "\u540c\u65f6\u5220\u9664\u5bf9\u5e94 MD \u6587\u4ef6\u5939",
                            value=True,
                            key=f"{key_ns}_del_also_md_{uid}",
                        )
                        c_del2 = st.columns([1.1, 1.1, 8.0])
                        with c_del2[0]:
                            if st.button("\u786e\u8ba4\u5220\u9664", key=f"{key_ns}_del_confirm_{uid}"):
                                # Remove queued tasks first (if any)
                                if queued_pos is not None:
                                    _bg_remove_queued_tasks_for_pdf(pdf)

                                # Delete PDF on disk
                                ok_pdf = False
                                try:
                                    pdf.unlink()
                                    ok_pdf = True
                                except Exception:
                                    ok_pdf = False

                                # Delete MD folder if requested
                                ok_md = True
                                if also_md:
                                    try:
                                        md_root = Path(md_out_root).resolve()
                                        target = (Path(md_out_root) / pdf.stem).resolve()
                                        if str(target).lower().startswith(str(md_root).lower()) and target.exists():
                                            shutil.rmtree(target, ignore_errors=True)
                                    except Exception:
                                        ok_md = False

                                # Best-effort remove from library index
                                try:
                                    lib_store.delete_by_path(pdf)
                                except Exception:
                                    pass

                                st.session_state[del_key] = False
                                if ok_pdf:
                                    st.success("\u5df2\u5220\u9664 PDF\u3002")
                                else:
                                    st.warning("\u5220\u9664 PDF \u5931\u8d25\uff08\u53ef\u80fd\u88ab\u5360\u7528\uff09\u3002")
                                if also_md and (not ok_md):
                                    st.warning("\u5220\u9664 MD \u6587\u4ef6\u5939\u5931\u8d25\u3002")
                                st.info("\u5982\u679c\u4f60\u5df2\u7ecf\u5efa\u5e93\uff0c\u5220\u9664\u540e\u5efa\u8bae\u70b9\u4e00\u6b21\u300c\u66f4\u65b0\u77e5\u8bc6\u5e93\u300d\u4ee5\u6e05\u7406\u65e7\u7d22\u5f15\u3002")
                                st.experimental_rerun()
                        with c_del2[1]:
                            if st.button("\u53d6\u6d88\u5220\u9664", key=f"{key_ns}_del_cancel_{uid}"):
                                st.session_state[del_key] = False
                                st.experimental_rerun()

        with tabs[0]:
            if pending:
                c_bulk = st.columns([1, 2])
                with c_bulk[0]:
                    if st.button("\u6279\u91cf\u8f6c\u6362\u5168\u90e8\u672a\u8f6c\u6362\uff08\u540e\u53f0\uff09", key="bulk_convert_pending"):
                        for it in pending:
                            pdf = it["pdf"]
                            _bg_enqueue({
                                "pdf": str(pdf),
                                "out_root": str(md_out_root),
                                "db_dir": str(db_dir),
                                "no_llm": bool(no_llm),
                                "eq_image_fallback": True,
                                "replace": False,
                                "name": pdf.name,
                            })
                        st.info(f"\u5df2\u628a {len(pending)} \u4e2a\u6587\u4ef6\u52a0\u5165\u540e\u53f0\u961f\u5217\u3002")
                        st.experimental_rerun()
                with c_bulk[1]:
                    st.markdown("<div class='refbox'>\u8f6c\u6362\u4f1a\u5728\u540e\u53f0\u8fd0\u884c\uff0c\u4f60\u53ef\u4ee5\u76f4\u63a5\u5207\u6362\u5230\u201c\u5bf9\u8bdd\u201d\u9875\u3002</div>", unsafe_allow_html=True)
                render_items(pending, show_missing_badge=True, key_ns="tab_pending")
            else:
                st.caption("\u5168\u90e8\u90fd\u5df2\u8f6c\u6362\u3002")
            # Always render progress *under* the list (user expects it here).
            render_bg_progress_under_tasks(key_ns="tab_pending_bottom")

        with tabs[1]:
            render_bg_progress_under_tasks(key_ns="tab_done")
            render_items(converted, show_missing_badge=False, key_ns="tab_done")

        with tabs[2]:
            render_bg_progress_under_tasks(key_ns="tab_all")
            render_items(pending + converted, show_missing_badge=True, key_ns="tab_all")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader(S["upload_pdf"])
    st.caption(S["batch_upload"])

    handled: dict = st.session_state.setdefault("upload_handled", {})
    ups = st.file_uploader("PDF", type=["pdf"], accept_multiple_files=True)

    if ups:
        for n, up in enumerate(ups, start=1):
            data = bytes(up.getbuffer())
            file_sha1 = hashlib.sha1(data).hexdigest()

            with st.expander(f"{up.name} ({n}/{len(ups)})", expanded=(n == 1)):
                if file_sha1 in handled:
                    st.info(handled[file_sha1].get("msg", ""))
                    continue

                exist = lib_store.get_by_sha1(file_sha1)
                if exist:
                    st.warning(S["dup_found"])
                    st.caption(f"{S['dup_path']} {exist.get('path','')}")
                    force = st.checkbox(S["dup_force"], value=False, key=f"dup_force_{n}")
                    if not force:
                        if st.button(S["dup_skip"], key=f"dup_skip_{n}"):
                            handled[file_sha1] = {"action": "skipped", "msg": S["handled_skip"], "ts": time.time()}
                            st.info(S["handled_skip"])
                            st.experimental_rerun()
                        else:
                            st.caption("\u5982\u679c\u4f60\u4e0d\u60f3\u91cd\u590d\u4fdd\u5b58\uff0c\u70b9\u51fb\u201c\u8df3\u8fc7\u201d\u5373\u53ef\u3002")
                        continue

                tmp_path = _write_tmp_upload(pdf_dir, up.name, data)
                sug: PdfMetaSuggestion = extract_pdf_meta_suggestion(tmp_path, settings=settings)

                st.caption(S["name_rule"])
                c1, c2, c3 = st.columns([2, 1, 3])
                with c1:
                    venue = st.text_input(S["venue"], value=sug.venue, key=f"venue_{n}")
                with c2:
                    year = st.text_input(S["year"], value=sug.year, key=f"year_{n}")
                with c3:
                    title = st.text_input(S["title_field"], value=sug.title, key=f"title_{n}")

                base = build_base_name(venue=venue, year=year, title=title)
                st.text_input(S["base_name"], value=base, disabled=True, key=f"base_{n}")

                dest_pdf = pdf_dir / f"{base}.pdf"
                if dest_pdf.exists():
                    k = 2
                    while (pdf_dir / f"{base}-{k}.pdf").exists() and k < 100:
                        k += 1
                    dest_pdf = pdf_dir / f"{base}-{k}.pdf"

                action_cols = st.columns(2)
                with action_cols[0]:
                    if st.button(S["save_pdf"], key=f"save_pdf_{n}"):
                        try:
                            if tmp_path.exists() and tmp_path.resolve() != dest_pdf.resolve():
                                tmp_path.replace(dest_pdf)
                        except Exception:
                            dest_pdf.write_bytes(data)
                        lib_store.upsert(file_sha1, dest_pdf)
                        handled[file_sha1] = {"action": "saved", "msg": S["handled_saved"], "ts": time.time()}
                        st.info(f"{S['saved_as']}: {dest_pdf}")

                with action_cols[1]:
                    if st.button(S["convert_now"], key=f"convert_now_{n}"):
                        try:
                            if tmp_path.exists() and tmp_path.resolve() != dest_pdf.resolve():
                                tmp_path.replace(dest_pdf)
                        except Exception:
                            dest_pdf.write_bytes(data)
                        lib_store.upsert(file_sha1, dest_pdf)
                        handled[file_sha1] = {"action": "converted", "msg": S["handled_converted"], "ts": time.time()}
                        _bg_enqueue({
                            "pdf": str(dest_pdf),
                            "out_root": str(md_out_root),
                            "db_dir": str(db_dir),
                            "no_llm": bool(no_llm),
                            "eq_image_fallback": True,
                            "replace": False,
                            "name": dest_pdf.name,
                        })
                        st.info("\u5df2\u52a0\u5165\u540e\u53f0\u961f\u5217\uff0c\u4f60\u53ef\u4ee5\u5207\u6362\u9875\u9762\u7ee7\u7eed\u4f7f\u7528\u3002")
                        st.experimental_rerun()

def main() -> None:
    st.set_page_config(page_title=S["title"], layout="wide")
    if "ui_theme" not in st.session_state:
        st.session_state["ui_theme"] = "light"
    _init_theme_css(st.session_state["ui_theme"])
    _inject_runtime_ui_fixes(st.session_state["ui_theme"])
    _render_app_title()

    settings = load_settings()
    chat_store = ChatStore(settings.chat_db_path)
    lib_store = LibraryStore(settings.library_db_path)

    prefs_path = Path(__file__).resolve().parent / "user_prefs.json"
    prefs = load_prefs(prefs_path)

    # Provide a default pdf_dir even if the user never opened the "文献管理" page
    # (refs can then still open PDFs).
    if ("pdf_dir" not in st.session_state) or (not str(st.session_state.get("pdf_dir") or "").strip()):
        env_pdf = (os.environ.get("KB_PDF_DIR") or "").strip().strip("'\"")
        if env_pdf:
            try:
                default_pdf_dir = Path(env_pdf).expanduser().resolve()
            except Exception:
                default_pdf_dir = Path(env_pdf).expanduser()
        else:
            home = Path.home()
            try:
                default_pdf_dir = (home / "research-papers").resolve() if (home / "research-papers").exists() else (home / "Pi_zaya_pdfs")
            except Exception:
                default_pdf_dir = home / "Pi_zaya_pdfs"
        pdf_dir_pref = (prefs.get("pdf_dir") or "").strip()
        try:
            st.session_state["pdf_dir"] = str(Path(pdf_dir_pref).expanduser().resolve()) if pdf_dir_pref else str(default_pdf_dir)
        except Exception:
            st.session_state["pdf_dir"] = str(default_pdf_dir)

    if "conv_id" not in st.session_state:
        recent = chat_store.list_conversations(limit=1)
        if recent:
            st.session_state["conv_id"] = recent[0]["id"]
        else:
            st.session_state["conv_id"] = chat_store.create_conversation()

    if "llm_rerank" not in st.session_state:
        st.session_state["llm_rerank"] = True
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = st.session_state.get("page_radio") or S["page_chat"]

    # Sidebar
    retriever_reload_flag: dict[str, bool] = {"reload": False}

    with st.sidebar:
        sidebar_logo = _resolve_sidebar_logo_path()
        if sidebar_logo is not None:
            logo_uri = _sidebar_logo_data_uri(sidebar_logo)
            if logo_uri:
                st.markdown(
                    f"<div class='kb-sidebar-logo-wrap'><img class='kb-sidebar-logo-img' src='{logo_uri}' alt='P&I Lab logo' /></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.image(str(sidebar_logo), width=220)

        c_mode = st.columns([3, 1])
        with c_mode[0]:
            st.subheader(S["settings"])
        with c_mode[1]:
            btn_icon = "🌞" if st.session_state["ui_theme"] == "dark" else "🌙"
            if st.button(btn_icon, key="theme_toggle_btn"):
                cur_page = st.session_state.get("page_radio") or st.session_state.get("active_page") or S["page_chat"]
                st.session_state["active_page"] = cur_page
                st.session_state["page_radio"] = cur_page
                st.session_state["ui_theme"] = "light" if st.session_state["ui_theme"] == "dark" else "dark"
                st.experimental_rerun()

        # Background conversion status (shown on every page)
        _bg_ensure_started()
        bg = _bg_snapshot()
        if bg.get("running") or (bg.get("total", 0) and bg.get("done", 0) < bg.get("total", 0)):
            done = int(bg.get("done", 0) or 0)
            total = int(bg.get("total", 0) or 0)
            cur = bg.get("current", "")
            st.caption(f"后台转换：{done}/{total}{(' | ' + cur) if cur else ''}")
            if total > 0:
                st.progress(min(1.0, done / max(1, total)))
            cbg = st.columns(2)
            with cbg[0]:
                if st.button("刷新", key="bg_refresh_sidebar"):
                    st.experimental_rerun()
            with cbg[1]:
                if st.button("停止", key="bg_cancel_sidebar"):
                    _bg_cancel_all()
                    st.experimental_rerun()
            st.markdown("---")

        page_options = [S["page_chat"], S["page_library"]]
        cur_active_page = st.session_state.get("active_page") or st.session_state.get("page_radio") or S["page_chat"]
        if cur_active_page not in page_options:
            cur_active_page = S["page_chat"]
            st.session_state["active_page"] = cur_active_page
            st.session_state["page_radio"] = cur_active_page
        page = st.radio(
            S["page"],
            options=page_options,
            index=page_options.index(cur_active_page),
            key="page_radio",
        )
        st.session_state["active_page"] = page

        db_default = prefs.get("db_path") or str(settings.db_dir)
        db_path = st.text_input(S["db_path"], value=str(db_default))
        db_path = (db_path or "").strip().strip("'\"")
        db_dir = Path(db_path).expanduser().resolve()
        if str(db_dir) != str(prefs.get("db_path") or ""):
            prefs2 = dict(prefs)
            prefs2["db_path"] = str(db_dir)
            save_prefs(prefs_path, prefs2)
            prefs.update(prefs2)

        top_k = st.slider(S["top_k"], min_value=2, max_value=20, value=int(prefs.get("top_k") or 6), step=1)
        temperature = st.slider(S["temp"], min_value=0.0, max_value=1.0, value=float(prefs.get("temperature") or 0.2), step=0.05)
        max_tokens_pref = int(prefs.get("max_tokens") or 1216)
        max_tokens_pref = max(256, min(4096, max_tokens_pref))
        max_tokens_pref = 256 + int(round((max_tokens_pref - 256) / 64.0)) * 64
        max_tokens_pref = max(256, min(4096, max_tokens_pref))
        max_tokens = st.slider(S["max_tokens"], min_value=256, max_value=4096, value=int(max_tokens_pref), step=64)
        show_context = st.checkbox(S["show_ctx"], value=bool(prefs.get("show_context") or False))
        deep_read = st.checkbox(S["deep_read"], value=bool(prefs.get("deep_read") if ("deep_read" in prefs) else True))
        llm_rerank = True
        st.session_state["llm_rerank"] = True

        # Persist simple knobs
        prefs_knobs = dict(prefs)
        prefs_knobs.update(
            {
                "top_k": int(top_k),
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "show_context": bool(show_context),
                "deep_read": bool(deep_read),
                "llm_rerank": bool(llm_rerank),
            }
        )
        if prefs_knobs != prefs:
            save_prefs(prefs_path, prefs_knobs)
            prefs.update(prefs_knobs)

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.subheader("模型/Key")
        try:
            st.caption(f"Base URL: {getattr(settings, 'base_url', '')}")
            st.caption(f"Model: {getattr(settings, 'model', '')}")
            st.caption(f"Timeout: {float(getattr(settings, 'timeout_s', 0) or 0):.0f}s | Retries: {int(getattr(settings, 'max_retries', 0) or 0)}")
        except Exception:
            pass
        if getattr(settings, "api_key", None):
            st.caption("API Key：已设置")
        else:
            st.warning("API Key：未设置（需要 DEEPSEEK_API_KEY 或 OPENAI_API_KEY）。设置后需重启 Streamlit。")
        if st.button("测试模型连接", key="test_llm_btn"):
            try:
                with st.spinner("测试中…"):
                    ds = DeepSeekChat(settings)
                    out = ds.chat(messages=[{"role": "user", "content": "ping"}], temperature=0.0, max_tokens=8)
                st.success(f"OK：{out or '(空)'}")
            except Exception as e:
                st.error(f"测试失败：{e}")

        col_a, col_b = st.columns(2)
        with col_a:
            reload_btn = st.button(S["reload_db"], key="reload_db")
        with col_b:
            clear_btn = st.button(S["clear_chat"], key="clear_chat")

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.subheader(S["history"])
        if st.button(S["new_chat"], key="new_chat"):
            st.session_state["conv_id"] = chat_store.create_conversation()

        convs = chat_store.list_conversations(limit=50)
        conv_ids = [c["id"] for c in convs] if convs else [st.session_state["conv_id"]]
        conv_labels = {}
        for c in convs:
            ts = time.strftime("%m-%d %H:%M", time.localtime(float(c.get("updated_at", 0) or 0)))
            conv_labels[c["id"]] = f"{ts} | {c.get('title','')}"

        selected_conv_id = st.selectbox(
            S["pick_chat"],
            options=conv_ids,
            format_func=lambda x: conv_labels.get(x, x),
            index=0,
            key="conv_select",
        )
        if selected_conv_id and selected_conv_id != st.session_state["conv_id"]:
            st.session_state["conv_id"] = selected_conv_id

        if st.button(S["del_chat"], key="delete_chat") and st.session_state.get("conv_id"):
            chat_store.delete_conversation(st.session_state["conv_id"])
            remaining = chat_store.list_conversations(limit=1)
            if remaining:
                st.session_state["conv_id"] = remaining[0]["id"]
            else:
                st.session_state["conv_id"] = chat_store.create_conversation()

            if "conv_select" in st.session_state:
                del st.session_state["conv_select"]

            st.experimental_rerun()

    # Retriever (auto-reload when DB changes)
    docs_json = db_dir / "docs.json"
    cur_mtime = docs_json.stat().st_mtime if docs_json.exists() else None
    prev_mtime = st.session_state.get("db_mtime")
    need_reload = ("retriever" not in st.session_state) or bool(reload_btn) or (cur_mtime and prev_mtime and cur_mtime > prev_mtime + 1e-6)
    if need_reload:
        with st.spinner("\u52a0\u8f7d\u77e5\u8bc6\u5e93..."):
            st.session_state.pop("retriever_load_error", None)
            try:
                st.session_state["retriever"] = _load_retriever(db_dir)
            except Exception as e:
                # Never crash the UI for new users / empty DB / corrupt chunks; fall back to an empty retriever.
                st.session_state["retriever"] = BM25Retriever([])
                st.session_state["retriever_load_error"] = f"{type(e).__name__}: {e}"
            st.session_state["db_mtime"] = cur_mtime or time.time()

    if clear_btn:
        st.session_state["conv_id"] = chat_store.create_conversation()

    st.session_state["messages"] = chat_store.get_messages(st.session_state["conv_id"])

    # Pages
    if page == S["page_chat"]:
        _page_chat(
            settings,
            chat_store,
            st.session_state["retriever"],
            db_dir,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens,
            show_context=show_context,
            deep_read=deep_read,
        )

    elif page == S["page_library"]:
        _page_library(settings, lib_store, db_dir, prefs_path, prefs, retriever_reload_flag)

    # No separate PDF->Markdown page: conversion lives in the library page.


if __name__ == "__main__":
    main()

