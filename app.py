# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import html
import inspect
import os
import subprocess
import shutil
import time
import threading
import uuid
from pathlib import Path

import streamlit as st
from ui.runtime_patches import _init_theme_css, _inject_auto_rerun_once, _inject_chat_dock_runtime, _inject_copy_js, _inject_runtime_ui_fixes, _set_live_streaming_mode, _teardown_chat_dock_runtime
from ui.strings import S
from ui.chat_widgets import _normalize_math_markdown, _render_ai_live_header, _render_answer_copy_bar, _render_app_title, _resolve_sidebar_logo_path, _sidebar_logo_data_uri
from ui.refs_renderer import _render_refs

from kb.chat_store import ChatStore
from kb.bg_queue_state import is_running_snapshot as bg_is_running_snapshot
from kb.config import load_settings
from kb.library_store import LibraryStore
from kb.llm import DeepSeekChat
from kb import runtime_state as RUNTIME
from kb.pdf_tools import PdfMetaSuggestion, build_base_name, ensure_dir, extract_pdf_meta_suggestion, open_in_explorer
from kb.prefs import load_prefs, save_prefs
from kb.file_ops import _cleanup_tmp_uploads, _list_pdf_paths_fast, _next_pdf_dest_path, _persist_upload_pdf, _pick_directory_dialog, _resolve_md_output_paths, _write_tmp_upload
from kb.rename_manager import ensure_state_defaults as ensure_rename_manager_state
from kb.rename_manager import render_panel as render_rename_manager_panel
from kb.rename_manager import render_prompt as render_rename_prompt
from kb.retriever import BM25Retriever
from kb.store import load_all_chunks
from kb.retrieval_engine import configure_cache as configure_retrieval_cache
from kb.task_runtime import _bg_cancel_all, _bg_enqueue, _bg_ensure_started, _bg_remove_queued_tasks_for_pdf, _bg_snapshot, _build_bg_task, _gen_get_task, _gen_mark_cancel, _gen_start_task, _is_live_assistant_text, _live_assistant_task_id, _live_assistant_text


def _patch_streamlit_label_visibility_compat() -> None:
    """
    Streamlit <=1.12 does not support `label_visibility` on some widgets.
    Keep backward compatibility so stale hot-reload modules do not crash.
    """
    try:
        sig_cb = inspect.signature(st.checkbox)
        cb_supports = "label_visibility" in sig_cb.parameters
    except Exception:
        cb_supports = True
    if not cb_supports:
        orig_cb = st.checkbox

        def _checkbox_compat(*args, **kwargs):
            kwargs.pop("label_visibility", None)
            return orig_cb(*args, **kwargs)

        st.checkbox = _checkbox_compat  # type: ignore[assignment]

    try:
        sig_ti = inspect.signature(st.text_input)
        ti_supports = "label_visibility" in sig_ti.parameters
    except Exception:
        ti_supports = True
    if not ti_supports:
        orig_ti = st.text_input

        def _text_input_compat(*args, **kwargs):
            kwargs.pop("label_visibility", None)
            return orig_ti(*args, **kwargs)

        st.text_input = _text_input_compat  # type: ignore[assignment]


_patch_streamlit_label_visibility_compat()

# Force converter script to this workspace copy, avoiding accidental fallback
# to an older sibling-repo script when multiple app processes are running.
_LOCAL_CONVERTER = Path(__file__).resolve().parent / "test2.py"
if _LOCAL_CONVERTER.exists():
    os.environ["KB_PDF_CONVERTER"] = str(_LOCAL_CONVERTER)

# Backward-compat for old runtime_state modules in long-lived Streamlit processes.
if not hasattr(RUNTIME, "GEN_LOCK"):
    RUNTIME.GEN_LOCK = threading.Lock()
if not hasattr(RUNTIME, "GEN_TASKS"):
    RUNTIME.GEN_TASKS = {}


# Keep source ASCII-stable: use unicode escapes for UI strings to avoid Windows encoding issues.



# Background conversion queue state is kept in an imported module.
# This survives Streamlit reruns more reliably than script-level globals.
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



configure_retrieval_cache(_cache_get, _cache_set)

























































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
    _set_live_streaming_mode(running_for_conv)

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
                safe = html.escape(content).replace("\n", "<br/>")
                st.markdown(
                    f"<div class='msg-user-wrap'><div class='msg-user'>{safe}</div><div class='msg-meta msg-meta-user'>你</div></div>",
                    unsafe_allow_html=True,
                )
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
                            _render_ai_live_header(stage=stage)
                            partial = str(t0.get("partial") or "").strip()
                            notice, body = _split_kb_miss_notice(partial)
                            if notice:
                                st.markdown(f"<div class='kb-notice'>{html.escape(notice)}</div>", unsafe_allow_html=True)
                            if (body or "").strip():
                                st.markdown(_normalize_math_markdown(body))
                            else:
                                st.markdown("<div class='kb-ai-live-dots'>...</div>", unsafe_allow_html=True)
                        else:
                            ans0 = str(t0.get("answer") or "").strip()
                            if ans0:
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

    _inject_chat_dock_runtime()

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
    if bg_is_running_snapshot(bg):
        st.markdown(
            "<div class='refbox'>\u540e\u53f0\u8f6c\u6362\u4efb\u52a1\u8fd0\u884c\u4e2d\uff1a\u8bf7\u5728\u4e0b\u65b9\u201c\u5df2\u6709\u6587\u732e\u201d\u533a\u57df\u67e5\u770b\u8fdb\u5ea6\u6761\u3002</div>",
            unsafe_allow_html=True,
        )

    try:
        dir_sig = str(Path(pdf_dir).expanduser().resolve())
    except Exception:
        dir_sig = str(pdf_dir)
    dismissed = st.session_state.setdefault("rename_prompt_dismissed_dirs", set())
    render_rename_prompt(pdfs=pdfs, dir_sig=dir_sig, dismissed_dirs=dismissed)

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

    ensure_rename_manager_state()
    render_rename_manager_panel(
        pdf_dir=pdf_dir,
        md_out_root=md_out_root,
        pdfs=pdfs,
        dir_sig=dir_sig,
        dismissed_dirs=dismissed,
        settings=settings,
        lib_store=lib_store,
    )

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
            md_folder, md_main, md_exists = _resolve_md_output_paths(md_out_root, pdf)
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
            bg2 = _bg_snapshot()
            q2 = list(bg2.get("queue") or [])
            if (not bool(bg2.get("running"))) and q2:
                _bg_ensure_started()
                bg2 = _bg_snapshot()
            running = bg_is_running_snapshot(bg2)
            if not running:
                return
            done = int(bg2.get("done", 0) or 0)
            total = int(bg2.get("total", 0) or 0)
            cur = str(bg2.get("current") or "")
            last = str(bg2.get("last") or "").strip()
            p_done = int(bg2.get("cur_page_done", 0) or 0)
            p_total = int(bg2.get("cur_page_total", 0) or 0)
            p_msg = str(bg2.get("cur_page_msg") or "").strip()
            p_profile = str(bg2.get("cur_profile") or "").strip()
            p_llm = str(bg2.get("cur_llm_profile") or "").strip()
            p_tail = list(bg2.get("cur_log_tail") or [])
            st.markdown("<div class='refbox'>\u540e\u53f0\u8f6c\u6362\u8fdb\u5ea6</div>", unsafe_allow_html=True)
            st.caption(f"{done}/{total}{(' | ' + cur) if cur else ''}")
            if p_profile:
                st.caption(p_profile)
            if p_llm:
                st.caption(p_llm)
            if total > 0:
                st.progress(min(1.0, done / max(1, total)))
            if p_total > 0:
                st.caption(f"\u5f53\u524d\u6587\u4ef6\u9875\u8fdb\u5ea6\uff1a{p_done}/{p_total}")
                st.progress(min(1.0, p_done / max(1, p_total)))
            elif bool(bg2.get("running")):
                st.caption("\u5f53\u524d\u6587\u4ef6\u5904\u7406\u4e2d\u2026")
            if p_msg and bool(bg2.get("running")) and (p_msg not in {p_profile, p_llm}):
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

            auto_key = f"bg_auto_refresh_{key_ns}"
            if auto_key not in st.session_state:
                st.session_state[auto_key] = True
            auto = st.checkbox("\u81ea\u52a8\u5237\u65b0\u8fdb\u5ea6", key=auto_key)
            if auto and running:
                _inject_auto_rerun_once(delay_ms=3500)

            if p_tail:
                with st.expander("\u8fdb\u5ea6\u65e5\u5fd7", expanded=False):
                    for ln in p_tail[-12:]:
                        st.caption(ln)

        def render_items(items: list[dict], *, show_missing_badge: bool, key_ns: str) -> None:
            from typing import Optional

            bg2 = _bg_snapshot()
            queue_tasks = list(bg2.get("queue") or [])
            current_name = str(bg2.get("current") or "")
            running_any = bool(bg2.get("running"))

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
                            replace = st.checkbox("\u66ff\u6362", value=True, key=f"{key_ns}_replace_md_{uid}")
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
                                    _build_bg_task(
                                        pdf_path=pdf,
                                        out_root=md_out_root,
                                        db_dir=db_dir,
                                        no_llm=bool(no_llm),
                                        replace=bool(replace),
                                    )
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
                            else:
                                if st.button("\u5220\u9664", key=f"{key_ns}_del_btn_{uid}", help="\u5220\u9664 PDF \u6587\u4ef6\uff08\u4e0d\u53ef\u6062\u590d\uff09"):
                                    st.session_state[del_key] = True

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
                            _bg_enqueue(
                                _build_bg_task(
                                    pdf_path=pdf,
                                    out_root=md_out_root,
                                    db_dir=db_dir,
                                    no_llm=bool(no_llm),
                                    replace=False,
                                )
                            )
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

                dest_pdf = _next_pdf_dest_path(pdf_dir, base)

                action_cols = st.columns(2)
                with action_cols[0]:
                    if st.button(S["save_pdf"], key=f"save_pdf_{n}"):
                        _persist_upload_pdf(tmp_path, dest_pdf, data)
                        lib_store.upsert(file_sha1, dest_pdf)
                        handled[file_sha1] = {"action": "saved", "msg": S["handled_saved"], "ts": time.time()}
                        st.info(f"{S['saved_as']}: {dest_pdf}")

                with action_cols[1]:
                    if st.button(S["convert_now"], key=f"convert_now_{n}"):
                        _persist_upload_pdf(tmp_path, dest_pdf, data)
                        lib_store.upsert(file_sha1, dest_pdf)
                        handled[file_sha1] = {"action": "converted", "msg": S["handled_converted"], "ts": time.time()}
                        _bg_enqueue(
                            _build_bg_task(
                                pdf_path=dest_pdf,
                                out_root=md_out_root,
                                db_dir=db_dir,
                                no_llm=bool(no_llm),
                                replace=False,
                            )
                        )
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
        if bg_is_running_snapshot(bg):
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

    if page != S["page_chat"]:
        _teardown_chat_dock_runtime()

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
