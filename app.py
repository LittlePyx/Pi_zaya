# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import html
import inspect
import os
import re
import subprocess
import shutil
import time
import threading
import uuid
from pathlib import Path

import streamlit as st
from ui.runtime_patches import (
    _init_theme_css,
    _inject_auto_rerun_once,
    _inject_chat_dock_runtime,
    _inject_copy_js,
    _inject_runtime_ui_fixes,
    _remember_scroll_for_next_rerun,
    _restore_scroll_after_rerun_if_needed,
    _sync_theme_with_browser_preference,
    _set_live_streaming_mode,
    _teardown_chat_dock_runtime,
)
from ui.strings import S
from ui.chat_widgets import _normalize_math_markdown, _render_ai_live_header, _render_answer_copy_bar, _render_app_title, _resolve_sidebar_logo_path, _sidebar_logo_data_uri
from ui.refs_renderer import (
    _annotate_equation_tags_with_sources,
    _annotate_inpaper_citations_with_hover_meta,
    _render_inpaper_citation_details,
    _render_refs,
)

from kb.chat_store import ChatStore
from kb.bg_queue_state import is_running_snapshot as bg_is_running_snapshot
from kb.config import load_settings
from kb.file_naming import (
    build_display_pdf_filename,
    build_storage_base_name,
    citation_meta_display_pdf_name,
    merge_citation_meta_file_labels,
    merge_citation_meta_name_fields,
)
from kb.library_store import LibraryStore
from kb.llm import DeepSeekChat
from kb import runtime_state as RUNTIME
from kb.pdf_tools import PdfMetaSuggestion, ensure_dir, extract_pdf_meta_suggestion, open_in_explorer
from kb.prefs import load_prefs, save_prefs
from kb.file_ops import (
    _cleanup_tmp_md_artifacts,
    _cleanup_tmp_uploads,
    _find_md_main_name_mismatches,
    _list_orphan_md_dirs,
    _list_pdf_paths_fast,
    _next_pdf_dest_path,
    _path_exists,
    _path_is_dir,
    _persist_upload_pdf,
    _pick_directory_dialog,
    _resolve_md_output_paths,
    _stash_orphan_md_dirs,
    _sync_md_main_filenames,
    _to_os_path,
    _write_tmp_upload,
)
from kb.rename_manager import ensure_state_defaults as ensure_rename_manager_state
from kb.rename_manager import render_panel as render_rename_manager_panel
from kb.rename_manager import render_prompt as render_rename_prompt
from kb.reference_sync import (
    is_running_snapshot as refsync_is_running_snapshot,
    snapshot as refsync_snapshot,
    start_reference_sync,
)
from kb.retriever import BM25Retriever
from kb.store import load_all_chunks
from kb.retrieval_engine import configure_cache as configure_retrieval_cache
from kb.task_runtime import _bg_cancel_all, _bg_enqueue, _bg_ensure_started, _bg_remove_queued_tasks_for_pdf, _bg_snapshot, _build_bg_task, _gen_get_task, _gen_mark_cancel, _gen_start_task, _is_live_assistant_text, _live_assistant_task_id, _live_assistant_text


def _patch_streamlit_label_visibility_compat() -> None:
    """
    Streamlit <=1.12 does not support `label_visibility` on some widgets.
    Keep backward compatibility so stale hot-reload modules do not crash.
    """
    def _wrap_drop_label_visibility(fn, *, tag: str):
        """
        Make the patch idempotent across Streamlit reruns.
        Streamlit reruns re-exec this file, and naive wrapping will create recursion.
        """
        try:
            if getattr(fn, "__kb_drop_label_visibility__", False):
                return fn
        except Exception:
            pass
        try:
            orig = getattr(fn, "__kb_drop_label_visibility_orig__", None) or fn
        except Exception:
            orig = fn

        def _compat(*args, **kwargs):
            kwargs.pop("label_visibility", None)
            return orig(*args, **kwargs)

        try:
            _compat.__kb_drop_label_visibility__ = True  # type: ignore[attr-defined]
            _compat.__kb_drop_label_visibility_orig__ = orig  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            _compat.__name__ = f"{getattr(fn, '__name__', tag)}__kb_compat"
        except Exception:
            pass
        return _compat

    try:
        sig_cb = inspect.signature(st.checkbox)
        cb_supports = "label_visibility" in sig_cb.parameters
    except Exception:
        cb_supports = True
    if not cb_supports:
        st.checkbox = _wrap_drop_label_visibility(st.checkbox, tag="checkbox")  # type: ignore[assignment]

    try:
        sig_ti = inspect.signature(st.text_input)
        ti_supports = "label_visibility" in sig_ti.parameters
    except Exception:
        ti_supports = True
    if not ti_supports:
        st.text_input = _wrap_drop_label_visibility(st.text_input, tag="text_input")  # type: ignore[assignment]


_patch_streamlit_label_visibility_compat()


def _patch_streamlit_rerun_compat() -> None:
    """
    Streamlit >=1.50 removed `st.experimental_rerun` in favor of `st.rerun`.
    This project still calls the experimental name in multiple modules.
    """
    try:
        if (not hasattr(st, "experimental_rerun")) and hasattr(st, "rerun"):
            st.experimental_rerun = st.rerun  # type: ignore[attr-defined]
    except Exception:
        pass


_patch_streamlit_rerun_compat()


def _safe_delete_file(path_obj: Path) -> tuple[bool, str]:
    p = Path(path_obj)
    try:
        if not _path_exists(p):
            return True, "not found"
    except Exception:
        pass
    err = ""
    try:
        os.remove(_to_os_path(p))
    except Exception as e:
        err = str(e)
        try:
            p.unlink()
            err = ""
        except Exception as e2:
            err = str(e2) or err
    try:
        if _path_exists(p):
            return False, err or "still exists after delete"
    except Exception:
        pass
    return True, ""


def _safe_delete_tree(path_obj: Path) -> tuple[bool, str]:
    p = Path(path_obj)
    try:
        if not _path_exists(p):
            return True, "not found"
        if not _path_is_dir(p):
            return False, "target is not a directory"
    except Exception:
        pass
    err = ""
    try:
        shutil.rmtree(_to_os_path(p), ignore_errors=False)
    except Exception as e:
        err = str(e)
        try:
            shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass
    try:
        if _path_exists(p):
            return False, err or "directory still exists after delete"
    except Exception:
        pass
    return True, ""

# Force converter script to this workspace copy, avoiding accidental fallback
# to an older sibling-repo script when multiple app processes are running.
_LOCAL_CONVERTER = Path(__file__).resolve().parent / "pdf_to_md.py"
if _LOCAL_CONVERTER.exists():
    os.environ["KB_PDF_CONVERTER"] = str(_LOCAL_CONVERTER)

# Backward-compat for old runtime_state modules in long-lived Streamlit processes.
if not hasattr(RUNTIME, "GEN_LOCK"):
    RUNTIME.GEN_LOCK = threading.Lock()
if not hasattr(RUNTIME, "GEN_TASKS"):
    RUNTIME.GEN_TASKS = {}
if not hasattr(RUNTIME, "CITATION_LOCK"):
    RUNTIME.CITATION_LOCK = threading.Lock()
if not hasattr(RUNTIME, "CITATION_TASKS"):
    RUNTIME.CITATION_TASKS = {}


# Keep source ASCII-stable: use unicode escapes for UI strings to avoid Windows encoding issues.



# Background conversion queue state is kept in an imported module.
# This survives Streamlit reruns more reliably than script-level globals.
_CACHE_LOCK = RUNTIME.CACHE_LOCK
_CACHE = RUNTIME.CACHE


_APP_STRUCT_CITE_CANON_RE = re.compile(r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*(\d{1,4})\s*\]\]", re.IGNORECASE)
_APP_STRUCT_CITE_SINGLE_RE = re.compile(r"(?<!\[)\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})(?:\s*:\s*(\d{1,4}))?\s*\](?!\])", re.IGNORECASE)
_APP_STRUCT_CITE_SID_ONLY_RE = re.compile(r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*\]\]", re.IGNORECASE)
_APP_STRUCT_CITE_GARBAGE_RE = re.compile(r"\[\[?\s*CITE\s*:[^\]\n]*\]?\]", re.IGNORECASE)


def _strip_structured_cite_tokens_for_display(md: str) -> str:
    s = str(md or "")
    if (not s) or ("CITE" not in s.upper()):
        return s
    out = _APP_STRUCT_CITE_CANON_RE.sub(lambda m: f"[{int(m.group(2))}]", s)
    out = _APP_STRUCT_CITE_SINGLE_RE.sub(
        lambda m: f"[{int(m.group(2))}]" if str(m.group(2) or "").strip() else "",
        out,
    )
    out = _APP_STRUCT_CITE_SID_ONLY_RE.sub("", out)
    out = _APP_STRUCT_CITE_GARBAGE_RE.sub("", out)
    return out


def _normalize_chat_markdown_for_display(md: str) -> str:
    return _normalize_math_markdown(_strip_structured_cite_tokens_for_display(md))


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



def _library_pdf_display_name(lib_store: LibraryStore, pdf_path: Path) -> str:
    try:
        meta = lib_store.get_citation_meta(pdf_path)
        full_name = citation_meta_display_pdf_name(meta)
        if full_name:
            return full_name
    except Exception:
        pass
    try:
        return Path(pdf_path).name
    except Exception:
        return str(pdf_path)


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
    lib_store: LibraryStore,
    retriever: BM25Retriever,
    db_dir: Path,
    pdf_dir: Path,
    md_out_root: Path,
    top_k: int,
    temperature: float,
    max_tokens: int,
    show_context: bool,
    deep_read: bool,
) -> None:
    disable_hooks = str(os.environ.get("KB_DISABLE_FRONTEND_HOOKS") or "").strip() in {"1", "true", "TRUE", "yes", "YES"}
    if not disable_hooks:
        _inject_copy_js()

    retriever_err = str(st.session_state.get("retriever_load_error") or "").strip()
    if retriever_err:
        st.error(f"\u77e5\u8bc6\u5e93\u52a0\u8f7d\u5931\u8d25\uff1a{retriever_err}")

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex[:10]
    session_id = str(st.session_state.get("session_id") or "").strip()

    conv_id = str(st.session_state.get("conv_id") or "").strip()
    # "Show full snippet" feature removed; force disabled for all sessions.
    st.session_state["show_context"] = False
    st.session_state["deep_read"] = bool(deep_read)

    if bool(getattr(retriever, "is_empty", False)):
        _render_kb_empty_hint()

    cur_task = _gen_get_task(session_id)
    running_for_conv = bool(
        isinstance(cur_task, dict)
        and str(cur_task.get("status") or "") == "running"
        and str(cur_task.get("conv_id") or "") == conv_id
    )
    _set_live_streaming_mode(running_for_conv, hide_stale=False)
    _restore_scroll_after_rerun_if_needed()

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

    def _get_refs_pack_for_user(user_msg_id: int) -> dict | None:
        nonlocal refs_by_user
        try:
            uid = int(user_msg_id or 0)
        except Exception:
            uid = 0
        if uid <= 0:
            return None
        pack0 = refs_by_user.get(uid) if isinstance(refs_by_user, dict) else None
        if isinstance(pack0, dict):
            return pack0
        try:
            latest = chat_store.list_message_refs(conv_id) or {}
        except Exception:
            latest = {}
        if not isinstance(latest, dict):
            latest = {}
        refs_by_user = latest
        try:
            st.session_state["_chat_refs_cache"] = latest
            st.session_state["_chat_refs_cache_conv"] = conv_id
        except Exception:
            pass
        pack1 = latest.get(uid)
        return pack1 if isinstance(pack1, dict) else None

    def _render_refs_for_user(user_msg_id: int, prompt_text: str, *, pending: bool = False) -> None:
        ref_pack = _get_refs_pack_for_user(user_msg_id)
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
                        refs_open_key=open_key_hist,
                        settings=settings,
                    )
                elif pending:
                    st.caption("（参考定位生成中…）")
                else:
                    st.caption("（未命中知识库片段：这条回答没有可定位的参考位置。）")
            st.markdown("</div>", unsafe_allow_html=True)

    chat_fragment_refresh_ok = bool(callable(getattr(st, "fragment", None)))

    def _render_chat_messages_area(*, live_running: bool, compact_live: bool = False) -> None:
            if (not msgs) and (not live_running):
                st.markdown(f"<div class='chat-empty-state'>{html.escape(S['no_msgs'])}</div>", unsafe_allow_html=True)
            else:
                render_msgs = list(msgs)
                hidden_msgs = 0
                lite_live = bool(compact_live and live_running)
                if live_running:
                    live_window = 8 if lite_live else 20
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
                                        hits_for_anno = []
                                        try:
                                            pack = _get_refs_pack_for_user(last_user_msg_id)
                                            if isinstance(pack, dict):
                                                hits_for_anno = list(pack.get("hits") or [])
                                        except Exception:
                                            hits_for_anno = []
                                        body2 = _annotate_equation_tags_with_sources(body, hits_for_anno)
                                        body3, cite_details = _annotate_inpaper_citations_with_hover_meta(
                                            body2,
                                            hits_for_anno,
                                            anchor_ns=f"{conv_id}:{idx}:{msg_id}:live",
                                        )
                                        st.markdown(_normalize_chat_markdown_for_display(body3))
                                        if cite_details:
                                            _render_inpaper_citation_details(
                                                cite_details,
                                                key_ns=f"{conv_id}_{idx}_{msg_id}_live",
                                            )
                                    else:
                                        st.markdown("<div class='kb-ai-live-dots'>...</div>", unsafe_allow_html=True)
                                else:
                                    ans0 = str(t0.get("answer") or "").strip()
                                    if ans0:
                                        copy_done_rendered = False
                                        notice, body = _split_kb_miss_notice(ans0)
                                        if notice:
                                            st.markdown(f"<div class='kb-notice'>{html.escape(notice)}</div>", unsafe_allow_html=True)
                                        if (body or "").strip():
                                            if lite_live:
                                                st.markdown(_normalize_chat_markdown_for_display(body))
                                                copy_done_rendered = True
                                            else:
                                                hits_for_anno = []
                                                try:
                                                    pack = _get_refs_pack_for_user(last_user_msg_id)
                                                    if isinstance(pack, dict):
                                                        hits_for_anno = list(pack.get("hits") or [])
                                                except Exception:
                                                    hits_for_anno = []
                                                body2 = _annotate_equation_tags_with_sources(body, hits_for_anno)
                                                body3, cite_details = _annotate_inpaper_citations_with_hover_meta(
                                                    body2,
                                                    hits_for_anno,
                                                    anchor_ns=f"{conv_id}:{idx}:{msg_id}:done",
                                                )
                                                copy_md_done = f"{notice}\n\n{body3}" if notice else body3
                                                _render_answer_copy_bar(
                                                    copy_md_done,
                                                    key_ns=f"copy_{idx}_done",
                                                    cite_details=cite_details,
                                                )
                                                copy_done_rendered = True
                                                st.markdown(_normalize_chat_markdown_for_display(body3))
                                                _render_inpaper_citation_details(
                                                    cite_details,
                                                    key_ns=f"{conv_id}_{idx}_{msg_id}_done",
                                                )
                                        if not copy_done_rendered:
                                            if lite_live:
                                                st.markdown(_normalize_chat_markdown_for_display(ans0))
                                            else:
                                                _render_answer_copy_bar(ans0, key_ns=f"copy_{idx}_done")
                                    else:
                                        st.markdown("<div class='msg-meta'>AI (processing)</div>", unsafe_allow_html=True)
                                        st.caption("Processing...")
                            else:
                                st.markdown("<div class='msg-meta'>AI（处理中）</div>", unsafe_allow_html=True)
                                st.caption("处理中…")
                        else:
                            msg_key = hashlib.md5((st.session_state.get("conv_id", "") + "|" + str(idx)).encode("utf-8", "ignore")).hexdigest()[:10]
                            copy_hist_rendered = False
                            notice, body = _split_kb_miss_notice(content or "")
                            if notice:
                                st.markdown(f"<div class='kb-notice'>{html.escape(notice)}</div>", unsafe_allow_html=True)
                            if (body or "").strip():
                                if lite_live:
                                    st.markdown(_normalize_chat_markdown_for_display(body))
                                    copy_hist_rendered = True
                                else:
                                    hits_for_anno = []
                                    try:
                                        pack = _get_refs_pack_for_user(last_user_msg_id)
                                        if isinstance(pack, dict):
                                            hits_for_anno = list(pack.get("hits") or [])
                                    except Exception:
                                        hits_for_anno = []
                                    body2 = _annotate_equation_tags_with_sources(body, hits_for_anno)
                                    body3, cite_details = _annotate_inpaper_citations_with_hover_meta(
                                        body2,
                                        hits_for_anno,
                                        anchor_ns=f"{conv_id}:{idx}:{msg_id}:hist",
                                    )
                                    copy_md_hist = f"{notice}\n\n{body3}" if notice else body3
                                    _render_answer_copy_bar(
                                        copy_md_hist,
                                        key_ns=f"copy_{msg_key}",
                                        cite_details=cite_details,
                                    )
                                    copy_hist_rendered = True
                                    st.markdown(_normalize_chat_markdown_for_display(body3))
                                    _render_inpaper_citation_details(
                                        cite_details,
                                        key_ns=f"{conv_id}_{idx}_{msg_id}_hist",
                                    )
                            if not copy_hist_rendered:
                                if lite_live:
                                    st.markdown(_normalize_chat_markdown_for_display(content))
                                else:
                                    _render_answer_copy_bar(content, key_ns=f"copy_{msg_key}")

                        if (not lite_live) and (last_user_msg_id > 0) and (last_user_msg_id not in shown_refs_user_ids):
                            _render_refs_for_user(
                                last_user_msg_id,
                                str(last_user_for_refs or "").strip(),
                                pending=pending,
                            )
                            shown_refs_user_ids.add(last_user_msg_id)
                    st.markdown("")

            st.markdown("<div id='kb-chat-tail-anchor' style='height:0.35rem;'></div>", unsafe_allow_html=True)

    if chat_fragment_refresh_ok and running_for_conv:
        @st.fragment(run_every=0.12)
        def _kb_chat_live_messages_fragment() -> None:
            t_now = _gen_get_task(session_id) or {}
            running_now = bool(
                isinstance(t_now, dict)
                and str(t_now.get("status") or "") == "running"
                and str(t_now.get("conv_id") or "") == conv_id
            )
            # Keep body runtime classes in sync even without an app-wide rerun.
            _set_live_streaming_mode(running_now, hide_stale=False)
            _render_chat_messages_area(live_running=running_now, compact_live=True)
            st.session_state["_kb_chat_fragment_prev_running"] = running_now
            pending_finish_rerun = bool(st.session_state.get("_kb_chat_finish_app_rerun_pending", False))
            if running_now and pending_finish_rerun:
                st.session_state.pop("_kb_chat_finish_app_rerun_pending", None)
                st.session_state.pop("_kb_chat_finish_app_rerun_after_ts", None)
                pending_finish_rerun = False
            if (not running_now) and (not pending_finish_rerun):
                # Trigger one app rerun on completion so chat fragments stop polling.
                # Preserve scroll to avoid a visible jump when switching back to non-fragment mode.
                st.session_state["_kb_chat_finish_app_rerun_pending"] = True
                st.session_state["_kb_chat_finish_app_rerun_after_ts"] = time.time() + 0.18
            if (not running_now) and bool(st.session_state.get("_kb_chat_finish_app_rerun_pending", False)):
                after_ts = float(st.session_state.get("_kb_chat_finish_app_rerun_after_ts", 0.0) or 0.0)
                if (after_ts <= 0.0) or (time.time() >= after_ts):
                    try:
                        _remember_scroll_for_next_rerun(
                            nonce=f"{session_id}:{conv_id}:{int(time.time() * 1000)}",
                            anchor_id="kb-chat-tail-anchor",
                        )
                    except Exception:
                        pass
                    st.session_state.pop("_kb_chat_finish_app_rerun_pending", None)
                    st.session_state.pop("_kb_chat_finish_app_rerun_after_ts", None)
                    try:
                        st.rerun(scope="app")
                    except Exception:
                        try:
                            st.rerun()
                        except Exception:
                            st.experimental_rerun()

        _kb_chat_live_messages_fragment()
    else:
        st.session_state.pop("_kb_chat_fragment_prev_running", None)
        st.session_state.pop("_kb_chat_finish_app_rerun_pending", None)
        st.session_state.pop("_kb_chat_finish_app_rerun_after_ts", None)
        _render_chat_messages_area(live_running=running_for_conv, compact_live=False)

    def _quick_chat_upload_to_library(selected_files) -> None:
        if not selected_files:
            st.session_state["_chat_pending_image_attachments"] = []
            return
        handled = st.session_state.setdefault("_chat_quick_upload_handled", {})
        if not isinstance(handled, dict):
            handled = {}

        pdf_uploaded_cnt = 0
        image_uploaded_cnt = 0
        dup_cnt = 0
        unsupported_cnt = 0
        err_cnt = 0
        current_image_attachments: list[dict] = []
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
        image_mime_to_ext = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
        }

        def _safe_upload_stem(name: str) -> str:
            s0 = str(name or "").strip()
            out_chars: list[str] = []
            for ch in s0:
                try:
                    if ch.isalnum():
                        out_chars.append(ch)
                    elif ch in ("-", "_", "."):
                        out_chars.append(ch)
                    elif ch.isspace():
                        out_chars.append("-")
                    else:
                        out_chars.append("_")
                except Exception:
                    out_chars.append("_")
            out = "".join(out_chars).strip(" ._-")
            return (out or "upload")[:72]

        for up in list(selected_files or []):
            try:
                data = bytes(up.getbuffer())
            except Exception:
                data = b""
            if not data:
                continue

            file_sha1 = hashlib.sha1(data).hexdigest()
            raw_name = str(getattr(up, "name", "") or "").strip()
            raw_mime = str(getattr(up, "type", "") or "").strip().lower()
            if file_sha1 in handled:
                prev_info = handled.get(file_sha1)
                if (
                    isinstance(prev_info, dict)
                    and str(prev_info.get("action") or "") in {"saved_image", "dup"}
                    and str(prev_info.get("path") or "").strip()
                ):
                    current_image_attachments.append(
                        {
                            "sha1": file_sha1,
                            "path": str(prev_info.get("path") or "").strip(),
                            "name": raw_name or (Path(str(prev_info.get("path") or "")).name or "image"),
                            "mime": raw_mime or "image/*",
                        }
                    )
                continue

            suffix = Path(raw_name).suffix.lower()
            is_pdf = bool((suffix == ".pdf") or (raw_mime == "application/pdf") or data.startswith(b"%PDF"))
            is_image = bool(raw_mime.startswith("image/") or (suffix in image_exts))

            if (not is_pdf) and (not is_image):
                handled[file_sha1] = {"action": "unsupported", "ts": time.time(), "name": raw_name, "mime": raw_mime}
                unsupported_cnt += 1
                continue

            if is_image and (not is_pdf):
                try:
                    chat_img_dir = Path(db_dir) / "_chat_uploads" / "images"
                    ensure_dir(chat_img_dir)
                    ext = suffix if suffix in image_exts else image_mime_to_ext.get(raw_mime, "")
                    if not ext:
                        if data.startswith(b"\x89PNG\r\n\x1a\n"):
                            ext = ".png"
                        elif data[:3] == b"\xff\xd8\xff":
                            ext = ".jpg"
                        elif data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
                            ext = ".gif"
                        elif (len(data) >= 12) and (data[:4] == b"RIFF") and (data[8:12] == b"WEBP"):
                            ext = ".webp"
                        elif data.startswith(b"BM"):
                            ext = ".bmp"
                    if not ext:
                        ext = ".png"
                    stem_seed = Path(raw_name).stem or f"pasted-{int(time.time())}"
                    safe_stem = _safe_upload_stem(stem_seed)
                    dest_img = chat_img_dir / f"{safe_stem}-{file_sha1[:10]}{ext}"
                    if dest_img.exists():
                        handled[file_sha1] = {"action": "dup", "ts": time.time(), "path": str(dest_img)}
                        current_image_attachments.append(
                            {"sha1": file_sha1, "path": str(dest_img), "name": raw_name or dest_img.name, "mime": raw_mime or "image/*"}
                        )
                        dup_cnt += 1
                        continue
                    dest_img.write_bytes(data)
                    handled[file_sha1] = {"action": "saved_image", "ts": time.time(), "path": str(dest_img)}
                    current_image_attachments.append(
                        {"sha1": file_sha1, "path": str(dest_img), "name": raw_name or dest_img.name, "mime": raw_mime or "image/*"}
                    )
                    image_uploaded_cnt += 1
                except Exception:
                    handled[file_sha1] = {"action": "error", "ts": time.time(), "kind": "image"}
                    err_cnt += 1
                continue

            try:
                exist = lib_store.get_by_sha1(file_sha1)
            except Exception:
                exist = None
            if exist:
                handled[file_sha1] = {"action": "dup", "ts": time.time()}
                dup_cnt += 1
                continue

            try:
                raw_name_pdf = str(raw_name or "upload.pdf")
                tmp_path = _write_tmp_upload(pdf_dir, raw_name_pdf, data)
                try:
                    sug = extract_pdf_meta_suggestion(tmp_path, settings=settings)
                except Exception:
                    sug = PdfMetaSuggestion()

                venue = str(getattr(sug, "venue", "") or "").strip()
                year = str(getattr(sug, "year", "") or "").strip()
                title = str(getattr(sug, "title", "") or "").strip() or (Path(raw_name_pdf).stem or "Untitled")

                base = build_storage_base_name(
                    venue=venue,
                    year=year,
                    title=title,
                    pdf_dir=pdf_dir,
                    md_out_root=md_out_root,
                )
                display_full_name = build_display_pdf_filename(
                    venue=venue,
                    year=year,
                    title=title,
                    fallback_name=raw_name_pdf,
                )
                dest_pdf = _next_pdf_dest_path(pdf_dir, base)
                upload_citation_meta = merge_citation_meta_file_labels(
                    sug.crossref_meta if isinstance(getattr(sug, "crossref_meta", None), dict) else None,
                    display_full_name=display_full_name,
                    storage_filename=dest_pdf.name,
                )
                upload_citation_meta = merge_citation_meta_name_fields(
                    upload_citation_meta,
                    venue=venue,
                    year=year,
                    title=title,
                )

                _persist_upload_pdf(tmp_path, dest_pdf, data)
                lib_store.upsert(file_sha1, dest_pdf, citation_meta=upload_citation_meta)
                handled[file_sha1] = {"action": "saved", "ts": time.time(), "path": str(dest_pdf)}
                pdf_uploaded_cnt += 1
            except Exception:
                handled[file_sha1] = {"action": "error", "ts": time.time(), "kind": "pdf"}
                err_cnt += 1

        # Keep only currently selected image attachments for the next submit.
        uniq_imgs: list[dict] = []
        seen_img = set()
        for it in current_image_attachments:
            if not isinstance(it, dict):
                continue
            p0 = str(it.get("path") or "").strip()
            h0 = str(it.get("sha1") or "").strip().lower()
            if (not p0) or (not _path_exists(Path(p0))):
                continue
            key0 = h0 or p0.lower()
            if key0 in seen_img:
                continue
            seen_img.add(key0)
            uniq_imgs.append(
                {
                    "sha1": h0,
                    "path": p0,
                    "name": str(it.get("name") or Path(p0).name),
                    "mime": str(it.get("mime") or "image/*"),
                }
            )

        st.session_state["_chat_quick_upload_handled"] = handled
        st.session_state["_chat_pending_image_attachments"] = uniq_imgs
        if pdf_uploaded_cnt or image_uploaded_cnt or dup_cnt or unsupported_cnt or err_cnt:
            msg_parts: list[str] = []
            if pdf_uploaded_cnt:
                msg_parts.append(f"Added {pdf_uploaded_cnt} PDF(s)")
            if image_uploaded_cnt:
                msg_parts.append(f"Added {image_uploaded_cnt} image(s)")
            if dup_cnt:
                msg_parts.append(f"Skipped duplicate {dup_cnt}")
            if unsupported_cnt:
                msg_parts.append(f"Unsupported {unsupported_cnt}")
            if err_cnt:
                msg_parts.append(f"Failed {err_cnt}")
            st.session_state["_chat_quick_upload_flash"] = " | ".join(msg_parts)

    def _render_chat_prompt_form_ui(*, live_running: bool, form_key: str) -> tuple[str, bool, bool]:
        with st.form(key=form_key, clear_on_submit=True):
            prompt_val_local = st.text_area(" ", height=96, key="prompt_text")
            uploader_nonce = int(st.session_state.get("chat_input_uploader_nonce", 0) or 0)
            uploader_key = f"chat_input_pdf_uploader_{uploader_nonce}"
            try:
                chat_uploads_local = st.file_uploader(
                    "Add files",
                    type=["pdf", "png", "jpg", "jpeg", "webp", "gif", "bmp"],
                    accept_multiple_files=True,
                    key=uploader_key,
                    label_visibility="collapsed",
                )
            except TypeError:
                chat_uploads_local = st.file_uploader(
                    "Add files",
                    type=["pdf", "png", "jpg", "jpeg", "webp", "gif", "bmp"],
                    accept_multiple_files=True,
                    key=uploader_key,
                )
            _quick_chat_upload_to_library(chat_uploads_local)
            if live_running:
                # Keep the form DOM shape stable while streaming: render one action button (stop only),
                # instead of switching to two submit buttons (send + stop), which causes the dock runtime
                # to reclassify/misplace wrappers after the first submit on some Streamlit builds.
                stop_clicked_local = st.form_submit_button("■", help="Stop generation")
                submitted_local = False
            else:
                submitted_local = st.form_submit_button("↑", help="Send (Ctrl+Enter)")
                stop_clicked_local = False
        return str(prompt_val_local or ""), bool(stop_clicked_local), bool(submitted_local)

    def _handle_chat_form_actions(*, prompt_val_in: str, stop_clicked_in: bool, submitted_in: bool, rerun_on_stop: bool, rerun_on_submit: bool) -> None:
        if stop_clicked_in:
            t0 = _gen_get_task(session_id)
            if isinstance(t0, dict):
                _gen_mark_cancel(session_id, str(t0.get("id") or ""))
            if rerun_on_stop:
                try:
                    st.rerun(scope="app")
                except Exception:
                    st.experimental_rerun()
            return

        if submitted_in:
            txt = (prompt_val_in or "").strip()
            raw_img_atts = st.session_state.get("_chat_pending_image_attachments") or []
            img_atts: list[dict] = []
            if isinstance(raw_img_atts, list):
                for it in raw_img_atts:
                    if not isinstance(it, dict):
                        continue
                    p0 = str(it.get("path") or "").strip()
                    if (not p0) or (not _path_exists(Path(p0))):
                        continue
                    img_atts.append(
                        {
                            "sha1": str(it.get("sha1") or "").strip().lower(),
                            "path": p0,
                            "name": str(it.get("name") or Path(p0).name),
                            "mime": str(it.get("mime") or "image/*"),
                        }
                    )

            if txt or img_atts:
                t0 = _gen_get_task(session_id)
                if isinstance(t0, dict) and str(t0.get("status") or "") == "running":
                    st.warning("Previous answer is still generating. Please stop or wait.")
                    return

                task_id = uuid.uuid4().hex[:12]
                user_store_text = txt if txt else f"[Image attachment x{len(img_atts)}]"
                try:
                    user_msg_id = int(chat_store.append_message(conv_id, "user", user_store_text) or 0)
                except Exception:
                    user_msg_id = 0
                try:
                    assistant_msg_id = int(chat_store.append_message(conv_id, "assistant", _live_assistant_text(task_id)) or 0)
                except Exception:
                    assistant_msg_id = 0
                chat_store.set_title_if_default(conv_id, txt or user_store_text)
                st.session_state["messages"] = chat_store.get_messages(conv_id)

                ok = _gen_start_task(
                    {
                        "id": task_id,
                        "session_id": session_id,
                        "conv_id": conv_id,
                        "prompt": txt,
                        "image_attachments": list(img_atts),
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
                    chat_store.update_message_content(assistant_msg_id, "(failed to start generation)")
                else:
                    st.session_state["_chat_pending_image_attachments"] = []
                    try:
                        st.session_state["chat_input_uploader_nonce"] = int(st.session_state.get("chat_input_uploader_nonce", 0) or 0) + 1
                    except Exception:
                        st.session_state["chat_input_uploader_nonce"] = 1
                if rerun_on_submit:
                    try:
                        st.rerun(scope="app")
                    except Exception:
                        st.experimental_rerun()

    prompt_val = ""
    stop_clicked = False
    submitted = False

    if chat_fragment_refresh_ok and running_for_conv:
        @st.fragment(run_every=0.28)
        def _kb_chat_input_fragment() -> None:
            running_now = False
            try:
                t_now = _gen_get_task(session_id) or {}
                running_now = bool(
                    isinstance(t_now, dict)
                    and str(t_now.get("status") or "") == "running"
                    and str(t_now.get("conv_id") or "") == conv_id
                )
            except Exception:
                running_now = False
            # Sync body classes here as well in case this fragment is the first one to observe completion.
            _set_live_streaming_mode(running_now, hide_stale=False)
            pv, sc, sb = _render_chat_prompt_form_ui(live_running=running_now, form_key="prompt_form_frag")
            _handle_chat_form_actions(
                prompt_val_in=pv,
                stop_clicked_in=sc,
                submitted_in=sb,
                rerun_on_stop=False,
                rerun_on_submit=True,
            )

        _kb_chat_input_fragment()
    else:
        prompt_val, stop_clicked, submitted = _render_chat_prompt_form_ui(
            live_running=running_for_conv,
            form_key="prompt_form",
        )

    if not disable_hooks:
        _inject_chat_dock_runtime()

    if (not chat_fragment_refresh_ok) or (not running_for_conv):
        _handle_chat_form_actions(
            prompt_val_in=prompt_val,
            stop_clicked_in=stop_clicked,
            submitted_in=submitted,
            rerun_on_stop=True,
            rerun_on_submit=True,
        )

    upload_flash = str(st.session_state.pop("_chat_quick_upload_flash", "") or "").strip()
    if upload_flash:
        st.caption(upload_flash)

    # Adaptive polling for chat streaming:
    # - Prefer browser-side one-shot rerun pulses (lighter visual feel than server sleep+rerun)
    # - Fall back to server-side rerun if frontend hooks are disabled/broken
    # - Poll faster when new tokens arrive, slower while retrieving/deep-reading
    if (not chat_fragment_refresh_ok) and running_for_conv and (not submitted) and (not stop_clicked):
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

        client_poll_ok = False
        if not disable_hooks:
            pulse_label = "__KB_CHAT_POLL_PULSE__"
            try:
                if st.button(pulse_label, key="_kb_chat_poll_pulse_btn"):
                    st.session_state["_kb_chat_client_poll_ts"] = time.time()
            except Exception:
                pass
            try:
                if not bool(st.session_state.get("_kb_chat_poll_started_ts")):
                    st.session_state["_kb_chat_poll_started_ts"] = time.time()
                _inject_auto_rerun_once(
                    delay_ms=int(max(120, delay_s * 1000)),
                    pulse_button_label=pulse_label,
                    nonce=f"chat:{session_id}:{cur_tid}:{cur_chars}:{time.time():.6f}",
                )
                client_poll_ok = True
            except Exception:
                client_poll_ok = False

        if client_poll_ok:
            now_ts = time.time()
            started_ts = float(st.session_state.get("_kb_chat_poll_started_ts", 0.0) or 0.0)
            client_ts = float(st.session_state.get("_kb_chat_client_poll_ts", 0.0) or 0.0)
            grace_s = max(1.1, delay_s * 3.2)
            stale_limit_s = max(2.2, delay_s * 4.5)
            client_poll_ok = not (
                (started_ts > 0.0)
                and ((now_ts - started_ts) > grace_s)
                and ((client_ts <= 0.0) or ((now_ts - client_ts) > stale_limit_s))
            )

        if not client_poll_ok:
            try:
                time.sleep(delay_s)
            except Exception:
                pass
            st.experimental_rerun()
    else:
        st.session_state.pop("_gen_poll_state", None)
        st.session_state.pop("_kb_chat_poll_started_ts", None)
        st.session_state.pop("_kb_chat_client_poll_ts", None)


def _dir_signature(path_obj: Path) -> str:
    try:
        return str(Path(path_obj).expanduser().resolve())
    except Exception:
        return str(path_obj)


def _maybe_auto_cleanup_tmp_artifacts(pdf_dir: Path, md_out_root: Path, *, interval_s: float = 45.0) -> dict[str, int]:
    dir_sig = f"{_dir_signature(pdf_dir)}|{_dir_signature(md_out_root)}"
    state = st.session_state.setdefault("_tmp_cleanup_state", {})
    if not isinstance(state, dict):
        state = {}

    now = time.time()
    rec = state.get(dir_sig) if isinstance(state.get(dir_sig), dict) else {}
    last_ts = float(rec.get("ts", 0.0) or 0.0)
    if (now - last_ts) < float(interval_s):
        return {"pdf": 0, "md": 0}

    cleaned_pdf = int(_cleanup_tmp_uploads(pdf_dir) or 0)
    try:
        cleaned_md, _ = _cleanup_tmp_md_artifacts(md_out_root)
        cleaned_md = int(cleaned_md or 0)
    except Exception:
        cleaned_md = 0
    state[dir_sig] = {"ts": now, "cleaned_pdf": cleaned_pdf, "cleaned_md": cleaned_md}
    st.session_state["_tmp_cleanup_state"] = state
    return {"pdf": cleaned_pdf, "md": cleaned_md}


def _has_markdown_newer_than(md_out_root: Path, *, docs_mtime=None) -> bool:
    if not Path(md_out_root).exists():
        return False

    skip_dirs = {"temp", "__pycache__", ".git"}
    for root, dirs, files in os.walk(md_out_root):
        dirs[:] = [d for d in dirs if str(d).lower() not in skip_dirs]
        for name in files:
            lower = str(name).lower()
            if (not lower.endswith(".md")) or (lower == "assets_manifest.md"):
                continue
            if docs_mtime is None:
                return True
            p = Path(root) / name
            try:
                if float(p.stat().st_mtime) > float(docs_mtime) + 1e-6:
                    return True
            except Exception:
                continue
    return False


def _kb_reindex_hint(md_out_root: Path, db_dir: Path, pdf_dir: Path) -> tuple[bool, str]:
    force_dirty = bool(st.session_state.get("kb_reindex_pending"))
    sig = (
        f"{_dir_signature(md_out_root)}|{_dir_signature(db_dir)}|{_dir_signature(pdf_dir)}|dirty:{int(force_dirty)}"
    )
    now = time.time()

    cache = st.session_state.get("_kb_reindex_hint_cache")
    if isinstance(cache, dict):
        if str(cache.get("sig") or "") == sig:
            cache_ts = float(cache.get("ts", 0.0) or 0.0)
            if (now - cache_ts) < 8.0:
                return bool(cache.get("need")), str(cache.get("reason") or "")

    need = False
    reason = ""
    if force_dirty:
        need = True
        reason = "检测到重命名变更，需更新知识库。"
    else:
        reasons: list[str] = []
        try:
            orphan_dirs = _list_orphan_md_dirs(md_out_root, pdf_dir)
        except Exception:
            orphan_dirs = []
        if orphan_dirs:
            need = True
            reasons.append(f"检测到 {len(orphan_dirs)} 个旧 MD 目录与当前 PDF 不匹配")

        try:
            bad_main = _find_md_main_name_mismatches(md_out_root, pdf_dir)
        except Exception:
            bad_main = []
        if bad_main:
            need = True
            reasons.append(f"检测到 {len(bad_main)} 个 MD 主文件名未与 PDF 同步")

        if need:
            reason = "；".join(reasons) + "，建议更新知识库。"
            st.session_state["_kb_reindex_hint_cache"] = {"sig": sig, "ts": now, "need": need, "reason": reason}
            return need, reason

        docs_json = Path(db_dir) / "docs.json"
        try:
            docs_mtime = docs_json.stat().st_mtime if docs_json.exists() else None
        except Exception:
            docs_mtime = None

        try:
            need = _has_markdown_newer_than(md_out_root, docs_mtime=docs_mtime)
        except Exception:
            need = False

        if need:
            if docs_mtime is None:
                reason = "检测到 Markdown 文档，尚未建立知识库索引。"
            else:
                reason = "检测到新增或修改的 Markdown，建议更新知识库。"

    st.session_state["_kb_reindex_hint_cache"] = {"sig": sig, "ts": now, "need": need, "reason": reason}
    return need, reason


def _page_library(settings, lib_store: LibraryStore, db_dir: Path, prefs_path: Path, prefs: dict, retriever_reload_flag: dict) -> None:
    _bg_ensure_started()
    lib_fragment_refresh_ok = bool(callable(getattr(st, "fragment", None)))
    try:
        st.session_state["_kb_library_progress_fragments_active"] = bool(lib_fragment_refresh_ok)
    except Exception:
        pass
    try:
        _set_live_streaming_mode(
            bool(bg_is_running_snapshot(_bg_snapshot()))
            or bool(refsync_is_running_snapshot(refsync_snapshot())),
            hide_stale=True,
        )
    except Exception:
        pass

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
    auto_cleaned = _maybe_auto_cleanup_tmp_artifacts(pdf_dir, md_out_root)
    if int(auto_cleaned.get("md", 0) or 0) > 0:
        st.session_state["kb_reindex_pending"] = True
        st.session_state.pop("_kb_reindex_hint_cache", None)

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

    dir_sig = _dir_signature(pdf_dir)
    dismissed = st.session_state.setdefault("rename_prompt_dismissed_dirs", set())
    render_rename_prompt(pdfs=pdfs, dir_sig=dir_sig, dismissed_dirs=dismissed)

    need_reindex, reindex_reason = _kb_reindex_hint(md_out_root, db_dir, pdf_dir)

    reindex_info_msgs: list[str] = []
    reindex_warn_msgs: list[str] = []
    reindex_err_msg = ""
    reindex_ok = False

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    cols = st.columns([1.0, 1.25, 1.1, 6.65])
    with cols[0]:
        if st.button(S["open_dir"], key="open_pdf_dir"):
            open_in_explorer(pdf_dir)
    with cols[1]:
        do_reindex = st.button(S["reindex_now"], key="reindex_btn")
        if do_reindex:
            ingest_py = Path(__file__).resolve().parent / "ingest.py"
            if ingest_py.exists():
                with st.spinner("正在更新知识库..."):
                    sync_n, sync_msgs = _sync_md_main_filenames(md_out_root, pdf_dir)
                    moved_n, moved_dirs = _stash_orphan_md_dirs(md_out_root, pdf_dir)
                    proc = subprocess.run(
                        [os.sys.executable, str(ingest_py), "--src", str(md_out_root), "--db", str(db_dir), "--incremental", "--prune"],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                if int(proc.returncode or 0) == 0:
                    try:
                        ref_budget_s = float(os.environ.get("KB_CROSSREF_BUDGET_S", "45") or 45.0)
                    except Exception:
                        ref_budget_s = 45.0
                    ref_launch: dict[str, object] = {}
                    ref_launch_err = ""
                    try:
                        ref_launch = start_reference_sync(
                            src_root=md_out_root,
                            db_dir=db_dir,
                            incremental=True,
                            enable_title_lookup=True,
                            crossref_time_budget_s=float(max(5.0, ref_budget_s)),
                            pdf_root=pdf_dir,
                            library_db_path=settings.library_db_path,
                        )
                    except Exception as e:
                        ref_launch_err = str(e)

                    if sync_n > 0:
                        preview_sync = ", ".join(sync_msgs[:2])
                        suffix_sync = "..." if sync_n > 2 else ""
                        reindex_info_msgs.append(f"已同步 {sync_n} 个 MD 主文件名：{preview_sync}{suffix_sync}")
                    if moved_n > 0:
                        preview = ", ".join(moved_dirs[:3])
                        suffix = "..." if moved_n > 3 else ""
                        reindex_info_msgs.append(f"已归档 {moved_n} 个旧 MD 目录：{preview}{suffix}")
                    if ref_launch_err:
                        reindex_warn_msgs.append(f"参考文献索引后台同步启动失败：{ref_launch_err}")
                    elif bool(ref_launch.get("started")):
                        reindex_info_msgs.append("参考文献索引已切换为后台同步，页面可继续正常使用。")
                    else:
                        reindex_info_msgs.append("参考文献索引任务已在后台运行中。")
                    reindex_ok = True
                    st.session_state["kb_reindex_pending"] = False
                    st.session_state.pop("_kb_reindex_hint_cache", None)
                    retriever_reload_flag["reload"] = True
                else:
                    err = (proc.stderr or proc.stdout or "").strip()
                    first_line = err.splitlines()[0] if err else "ingest.py 执行失败"
                    reindex_err_msg = f"更新失败：{first_line}"
            else:
                reindex_err_msg = "未找到 ingest.py，无法更新知识库。"
    with cols[2]:
        if st.button("文件名管理", key="rename_mgr_btn", help="根据 PDF 内容识别「期刊-年份-标题」并建议重命名"):
            st.session_state["rename_mgr_open"] = True
            st.session_state["rename_scan_trigger"] = False
    with cols[3]:
        hints: list[str] = []
        cleaned_pdf = int(auto_cleaned.get("pdf", 0) or 0)
        cleaned_md = int(auto_cleaned.get("md", 0) or 0)
        if (cleaned_pdf + cleaned_md) > 0:
            hints.append(f"已自动清理临时文件 PDF:{cleaned_pdf} MD:{cleaned_md}")
        if need_reindex and reindex_reason:
            hints.append(reindex_reason)
        if hints:
            st.caption(" | ".join(hints))

    if reindex_err_msg:
        st.error(reindex_err_msg)
    for _msg in reindex_warn_msgs:
        st.warning(_msg)
    for _msg in reindex_info_msgs:
        st.info(_msg)
    if reindex_ok:
        st.success(S["run_ok"])

    ref_snap = refsync_snapshot()
    ref_run_id = int(ref_snap.get("run_id", 0) or 0)
    ref_status = str(ref_snap.get("status") or "").strip().lower()
    ref_seen_id = int(st.session_state.get("_refsync_seen_run_id", 0) or 0)
    if ref_run_id > 0 and ref_run_id != ref_seen_id:
        if ref_status == "done":
            done_msg = str(ref_snap.get("message") or "").strip()
            st.success(done_msg or "参考文献索引后台同步完成。")
            st.session_state["_refsync_seen_run_id"] = ref_run_id
            st.session_state.pop("_kb_ref_index_cache_v1", None)
        elif ref_status == "error":
            err_msg = str(ref_snap.get("error") or "").strip() or str(ref_snap.get("message") or "").strip()
            st.warning(f"参考文献索引后台同步失败：{err_msg}")
            st.session_state["_refsync_seen_run_id"] = ref_run_id

    if lib_fragment_refresh_ok:
        @st.fragment(run_every=1.0)
        def _kb_library_run_monitor_fragment() -> None:
            try:
                bg_now = _bg_snapshot()
                ref_now = refsync_snapshot()
                running_now = bool(bg_is_running_snapshot(bg_now) or refsync_is_running_snapshot(ref_now))
                _set_live_streaming_mode(running_now, hide_stale=True)
                prev_running = bool(st.session_state.get("_kb_lib_monitor_prev_running", False))
                st.session_state["_kb_lib_monitor_prev_running"] = running_now
                if prev_running and (not running_now):
                    try:
                        st.rerun(scope="app")
                    except Exception:
                        st.rerun()
                st.markdown("<div style='display:none' aria-hidden='true'></div>", unsafe_allow_html=True)
            except Exception:
                pass

        _kb_library_run_monitor_fragment()

    if (not lib_fragment_refresh_ok) and refsync_is_running_snapshot(ref_snap):
        docs_done = int(ref_snap.get("docs_done", 0) or 0)
        docs_total = int(ref_snap.get("docs_total", 0) or 0)
        current_doc = str(ref_snap.get("current") or "").strip()
        stage = str(ref_snap.get("stage") or "").strip()
        started_at = float(ref_snap.get("started_at", 0.0) or 0.0)
        elapsed_s = max(0, int(time.time() - started_at)) if started_at > 0 else 0
        if docs_total > 0:
            progress = min(0.99, max(0.01, docs_done / max(1, docs_total)))
            status_line = f"{docs_done}/{docs_total} 文档，阶段: {stage or 'running'}"
        else:
            progress = 0.03
            status_line = f"阶段: {stage or 'starting'}"
        st.markdown("<div class='refbox'><strong>参考文献索引后台同步中</strong></div>", unsafe_allow_html=True)
        st.progress(progress)
        if current_doc:
            st.caption(f"{status_line} | 当前: {current_doc} | 已运行 {elapsed_s}s")
        else:
            st.caption(f"{status_line} | 已运行 {elapsed_s}s")

    if lib_fragment_refresh_ok:
        @st.fragment(run_every=1.0)
        def _kb_refsync_progress_fragment() -> None:
            snap = refsync_snapshot()
            if not refsync_is_running_snapshot(snap):
                _set_live_streaming_mode(bool(bg_is_running_snapshot(_bg_snapshot())), hide_stale=True)
                return
            _set_live_streaming_mode(True, hide_stale=True)
            docs_done = int(snap.get("docs_done", 0) or 0)
            docs_total = int(snap.get("docs_total", 0) or 0)
            current_doc = str(snap.get("current") or "").strip()
            stage = str(snap.get("stage") or "").strip()
            started_at = float(snap.get("started_at", 0.0) or 0.0)
            elapsed_s = max(0, int(time.time() - started_at)) if started_at > 0 else 0
            if docs_total > 0:
                progress = min(0.99, max(0.01, docs_done / max(1, docs_total)))
                status_line = f"{docs_done}/{docs_total} documents | stage {stage or 'running'}"
            else:
                progress = 0.03
                status_line = f"stage: {stage or 'starting'}"
            st.markdown("<div class='refbox'><strong>Reference Index Sync (Background)</strong></div>", unsafe_allow_html=True)
            st.progress(progress)
            if current_doc:
                st.caption(f"{status_line} | current: {current_doc} | elapsed {elapsed_s}s")
            else:
                st.caption(f"{status_line} | elapsed {elapsed_s}s")

        _kb_refsync_progress_fragment()

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
    
    # Speed mode selection - only three modes: normal, ultra_fast, no_llm
    speed_mode_options = {
        "normal": "普通模式 - 截图识别，最大并行度，质量最佳",
        "ultra_fast": "超快模式 - 截图识别，降低质量换取速度",
        "no_llm": "无LLM模式 - 基础文本提取，不使用多模态模型"
    }
    
    # Get current speed mode from session state or default
    current_speed_mode = st.session_state.get("lib_speed_mode", "normal")
    if current_speed_mode not in speed_mode_options:
        current_speed_mode = "normal"
    
    # Create a nice UI for speed mode selection
    col1, col2 = st.columns([2, 1])
    with col1:
        speed_mode = st.selectbox(
            "转换模式",
            options=list(speed_mode_options.keys()),
            format_func=lambda x: speed_mode_options[x],
            index=list(speed_mode_options.keys()).index(current_speed_mode) if current_speed_mode in speed_mode_options else 0,
            key="lib_speed_mode",
            help="选择转换模式。普通模式使用截图识别+最大并行度，质量最佳；超快模式降低质量换取速度；无LLM模式不使用多模态模型。"
        )
    
    with col2:
        # Show mode icon/indicator
        mode_icons = {
            "normal": "⭐",
            "ultra_fast": "💨",
            "no_llm": "📄"
        }
        st.markdown(f"### {mode_icons.get(speed_mode, '⭐')}")
    
    # Show mode description with styling
    mode_colors = {
        "normal": "🔵",
        "ultra_fast": "🟠",
        "no_llm": "⚪"
    }
    st.info(f"{mode_colors.get(speed_mode, '🔵')} **当前模式：{speed_mode_options[speed_mode]}**")
    
    # Set no_llm flag based on speed_mode
    no_llm = (speed_mode == "no_llm")
    
    # Vision-direct mode is always enabled for normal and ultra_fast modes
    # (no_llm mode doesn't use LLM, so vision-direct is not applicable)
    try:
        if speed_mode in ["normal", "ultra_fast"]:
            os.environ["KB_PDF_VISION_MODE"] = "1"
        else:
            os.environ.pop("KB_PDF_VISION_MODE", None)
    except Exception:
        pass

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

        # Get background task state to check for re-conversion tasks
        bg2 = _bg_snapshot()
        queue_tasks = list(bg2.get("queue") or [])
        current_name = str(bg2.get("current") or "")
        running_any = bool(bg2.get("running"))
        
        # Build a map of PDF -> task info (replace flag, queued/running status)
        # Use normalized paths for comparison (resolve to absolute paths)
        # Also build a name-based map as fallback
        pdf_task_map = {}
        pdf_name_map = {}  # Map by filename for fallback matching
        for task in queue_tasks:
            pdf_str = str(task.get("pdf") or "")
            task_name = str(task.get("name") or "")
            if pdf_str:
                try:
                    # Normalize path for comparison
                    pdf_path_normalized = str(Path(pdf_str).resolve())
                    task_info = {
                        "replace": bool(task.get("replace", False)),
                        "queued": True,
                        "running": False,
                        "original_path": pdf_str,
                    }
                    pdf_task_map[pdf_path_normalized] = task_info
                    # Also index by filename for fallback
                    if task_name:
                        pdf_name_map[task_name] = task_info
                except Exception:
                    # Fallback to string comparison if path resolution fails
                    task_info = {
                        "replace": bool(task.get("replace", False)),
                        "queued": True,
                        "running": False,
                        "original_path": pdf_str,
                    }
                    pdf_task_map[pdf_str] = task_info
                    if task_name:
                        pdf_name_map[task_name] = task_info
        if running_any and current_name:
            # Check if current task matches any PDF
            cur_task_replace = bool(bg2.get("cur_task_replace", False))
            for pdf in pdfs_view:
                if pdf.name == current_name:
                    try:
                        pdf_str_normalized = str(pdf.resolve())
                    except Exception:
                        pdf_str_normalized = str(pdf)
                    pdf_str = str(pdf)
                    
                    # Check both normalized and original paths
                    found = False
                    if pdf_str_normalized in pdf_task_map:
                        pdf_task_map[pdf_str_normalized]["running"] = True
                        pdf_task_map[pdf_str_normalized]["replace"] = cur_task_replace
                        found = True
                    else:
                        # Also check by original path string
                        for key, info in pdf_task_map.items():
                            if info.get("original_path") == pdf_str or key == pdf_str:
                                pdf_task_map[key]["running"] = True
                                pdf_task_map[key]["replace"] = cur_task_replace
                                found = True
                                break
                        # Fallback: check by filename
                        if not found and pdf.name in pdf_name_map:
                            pdf_name_map[pdf.name]["running"] = True
                            pdf_name_map[pdf.name]["replace"] = cur_task_replace
                            found = True
                    
                    if not found:
                        # Current task might not be in queue anymore (already started)
                        pdf_task_map[pdf_str_normalized] = {
                            "replace": cur_task_replace,
                            "queued": False,
                            "running": True,
                            "original_path": pdf_str,
                        }
                        pdf_name_map[pdf.name] = pdf_task_map[pdf_str_normalized]
        
        converted = []
        pending = []
        for pdf in pdfs_view:
            md_folder, md_main, md_exists = _resolve_md_output_paths(md_out_root, pdf)
            # Try to find task info using normalized path
            try:
                pdf_str_normalized = str(pdf.resolve())
            except Exception:
                pdf_str_normalized = str(pdf)
            pdf_str = str(pdf)
            
            # Look up task info - try normalized path first, then original path, then filename
            task_info = pdf_task_map.get(pdf_str_normalized, {})
            if not task_info:
                # Try to find by original path string
                for key, info in pdf_task_map.items():
                    if info.get("original_path") == pdf_str or key == pdf_str:
                        task_info = info
                        break
            # Fallback: check by filename
            if not task_info and pdf.name in pdf_name_map:
                task_info = pdf_name_map[pdf.name]
            
            # If file is being re-converted (replace=True), treat it as pending
            is_reconverting = task_info.get("replace", False) and (task_info.get("queued", False) or task_info.get("running", False))
            # Also check if file is queued or running (even without replace, it should show in pending if not converted yet)
            is_queued_or_running = task_info.get("queued", False) or task_info.get("running", False)
            
            # Classify: if MD exists AND not being re-converted AND not queued/running, it's converted; otherwise pending
            item = {
                "pdf": pdf,
                "md_folder": md_folder,
                "md_main": md_main,
                "md_exists": md_exists,
                "task_info": task_info,
            }
            if md_exists and not is_reconverting and not is_queued_or_running:
                converted.append(item)
            else:
                pending.append(item)

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
                # Reset warm-up UI state when background worker is idle.
                st.session_state.pop(f"_{key_ns}_overall_warm_start_ts", None)
                st.session_state.pop(f"_{key_ns}_file_warm_start_ts", None)
                st.session_state.pop(f"_{key_ns}_progress_meta", None)
                return

            done = int(bg2.get("done", 0) or 0)
            total = int(bg2.get("total", 0) or 0)
            cur = str(bg2.get("current") or "")
            cur_tid = str(bg2.get("cur_task_id") or "")
            last = str(bg2.get("last") or "").strip()
            p_done = int(bg2.get("cur_page_done", 0) or 0)
            p_total = int(bg2.get("cur_page_total", 0) or 0)
            p_msg = str(bg2.get("cur_page_msg") or "").strip()
            p_profile = str(bg2.get("cur_profile") or "").strip()
            p_llm = str(bg2.get("cur_llm_profile") or "").strip()
            p_tail = list(bg2.get("cur_log_tail") or [])

            # Progress card
            st.markdown("<div class='refbox'><strong>\u540e\u53f0\u8f6c\u6362\u8fdb\u5ea6</strong></div>", unsafe_allow_html=True)

            # Monotonic tiny warm-up before first-page progress becomes available.
            now_ts = time.time()
            meta_key = f"_{key_ns}_progress_meta"
            prev_meta = st.session_state.get(meta_key, {}) or {}
            prev_done = int(prev_meta.get("done", -1) or -1)
            prev_total = int(prev_meta.get("total", total) or total)
            prev_cur = str(prev_meta.get("cur") or "")
            prev_overall_display = float(prev_meta.get("overall_display", 0.0) or 0.0)
            prev_file_display = float(prev_meta.get("file_display", 0.0) or 0.0)
            prev_file_key = str(prev_meta.get("file_key") or "")
            prev_tid = str(prev_meta.get("cur_tid") or "")
            prev_p_done = int(prev_meta.get("p_done", 0) or 0)
            prev_p_total = int(prev_meta.get("p_total", 0) or 0)

            # New run: clear cached display values to avoid carrying old bars.
            if (
                (done < prev_done)
                or ((done == 0) and (prev_done > 0))
                or (total < prev_total)
                or (cur_tid and prev_tid and (cur_tid != prev_tid))
                or ((cur == prev_cur) and (p_done < prev_p_done) and (p_total <= max(prev_p_total, p_total)))
            ):
                prev_overall_display = 0.0
                prev_file_display = 0.0
                prev_file_key = ""
                st.session_state.pop(f"_{key_ns}_overall_warm_start_ts", None)
                st.session_state.pop(f"_{key_ns}_file_warm_start_ts", None)

            if cur != prev_cur:
                st.session_state.pop(f"_{key_ns}_file_warm_start_ts", None)

            def _warmup_value(start_key: str, base: float, cap: float, speed: float) -> float:
                if start_key not in st.session_state:
                    st.session_state[start_key] = now_ts
                elapsed = max(0.0, now_ts - float(st.session_state.get(start_key) or now_ts))
                return min(cap, base + elapsed * speed)

            # Tiny warm-up: keep visible movement but avoid first-page rollback.
            tiny_file_warm = _warmup_value(
                f"_{key_ns}_file_warm_start_ts",
                base=0.001,
                cap=0.010,
                speed=0.002,
            )

            # Include task_id so reconverting the same file starts from 0 visually.
            file_key = f"{done}|{cur}|{cur_tid}" if cur else ""
            file_display = 0.0
            if cur:
                if p_total > 0:
                    file_real = p_done / max(1, p_total)
                    if p_done <= 0:
                        file_target = max(file_real, tiny_file_warm)
                    else:
                        file_target = file_real
                else:
                    file_target = tiny_file_warm

                if file_key and (file_key == prev_file_key):
                    file_display = max(file_target, prev_file_display)
                else:
                    file_display = file_target

            file_display = max(0.0, min(1.0, file_display))
            if p_done > 0:
                st.session_state.pop(f"_{key_ns}_file_warm_start_ts", None)

            queued_n = max(0, len(q2))

            # Overall progress: include current-file fractional progress for smoother updates.
            overall_display = 0.0
            if total > 0:
                base_done_ratio = done / max(1, total)
                if cur:
                    cur_fraction = min(file_display, 0.999)
                else:
                    cur_fraction = _warmup_value(
                        f"_{key_ns}_overall_warm_start_ts",
                        base=0.001,
                        cap=0.010,
                        speed=0.0015,
                    )
                overall_target = max(base_done_ratio, min(1.0, (done + cur_fraction) / max(1, total)))
                overall_display = max(overall_target, prev_overall_display)
                overall_display = max(0.0, min(1.0, overall_display))

                st.markdown(f"**\u603b\u8fdb\u5ea6** {done}/{total} \u4e2a\u6587\u4ef6 ({overall_display * 100:.1f}%)")
                st.progress(overall_display)
                if running and done <= 0 and p_done <= 0:
                    st.caption(f"\u6b63\u5728\u542f\u52a8\u8f6c\u6362\u6d41\u7a0b\uff0c\u9996\u6279\u9875\u9762\u5904\u7406\u4e2d\uff08\u961f\u5217 {queued_n} \u4e2a\u6587\u4ef6\uff09...")
                if done > 0:
                    st.session_state.pop(f"_{key_ns}_overall_warm_start_ts", None)
            else:
                st.caption("\u6b63\u5728\u51c6\u5907\u4efb\u52a1\u961f\u5217...")

            # Current file progress
            if cur:
                st.markdown("---")
                st.markdown(f"**\u5f53\u524d\u6587\u4ef6** `{cur}`")

                if p_total > 0:
                    st.markdown(f"**\u9875\u8fdb\u5ea6** {p_done}/{p_total} \u9875 ({file_display * 100:.1f}%)")
                else:
                    st.markdown(f"**\u9875\u8fdb\u5ea6** \u9996\u6279\u9875\u9762\u9884\u5904\u7406\u4e2d ({file_display * 100:.1f}%)")
                st.progress(file_display)

                # Human-readable status line (filter internal profile noise).
                status_line = ""
                if p_msg and (p_msg not in {p_profile, p_llm}):
                    if (not p_msg.startswith("converter profile:")) and (not p_msg.startswith("LLM concurrency:")):
                        status_line = p_msg
                if (not status_line) and p_total > 0 and p_done <= 0:
                    status_line = "\u6b63\u5728\u8bfb\u53d6\u9996\u6279\u9875\u9762\u5185\u5bb9..."
                if (not status_line) and p_total <= 0:
                    status_line = "\u6b63\u5728\u521d\u59cb\u5316\u9875\u9762\u8ba1\u6570\u4e0e\u8bc6\u522b..."
                if status_line:
                    st.caption(f"\u72b6\u6001\uff1a{status_line}")
                    # Lightweight stall diagnosis for user-facing clarity.
                    # Typical status examples:
                    # - "Processing page 15/15 ... (alive 42s)"
                    # - "Post-processing after pages 15/15 ... (alive 18s)"
                    low = status_line.lower()
                    alive_s = -1
                    try:
                        k = low.rfind("alive ")
                        if k >= 0:
                            tail = low[k + len("alive "):]
                            num = ""
                            for ch in tail:
                                if ch.isdigit():
                                    num += ch
                                else:
                                    break
                            if num:
                                alive_s = int(num)
                    except Exception:
                        alive_s = -1
                    if alive_s >= 20:
                        if low.startswith("processing page "):
                            page_hint = ""
                            try:
                                part = low[len("processing page "):]
                                lhs = part.split("/", 1)[0].strip()
                                if lhs.isdigit():
                                    page_hint = lhs
                            except Exception:
                                page_hint = ""
                            if page_hint:
                                st.caption(f"\u8bca\u65ad\uff1a\u5f53\u524d\u5361\u5728\u7b2c {page_hint} \u9875\u7684 VL/LLM \u8bc6\u522b\u9636\u6bb5\uff08\u5df2\u7b49\u5f85 {alive_s}s\uff09")
                            else:
                                st.caption(f"\u8bca\u65ad\uff1a\u5f53\u524d\u5361\u5728\u5355\u9875 VL/LLM \u8bc6\u522b\u9636\u6bb5\uff08\u5df2\u7b49\u5f85 {alive_s}s\uff09")
                        elif low.startswith("post-processing after pages"):
                            st.caption(f"\u8bca\u65ad\uff1a\u9875\u9762\u5df2\u8f6c\u5b8c\uff0c\u5f53\u524d\u5361\u5728\u540e\u5904\u7406\u9636\u6bb5\uff08\u5df2\u7b49\u5f85 {alive_s}s\uff09")
            elif done > 0:
                st.markdown("---")
                st.success(f"\u6700\u65b0\u5b8c\u6210\uff1a{last}" if last else f"\u5df2\u5b8c\u6210 {done} \u4e2a\u6587\u4ef6")

            # Control buttons
            c_bg = st.columns([1.0, 1.0, 1.0])
            with c_bg[0]:
                # Refresh button - just rerun, don't affect background tasks
                if st.button("\u5237\u65b0", key=f"{key_ns}_refresh"):
                    st.experimental_rerun()
            with c_bg[1]:
                if st.button("\u505c\u6b62", key=f"{key_ns}_stop"):
                    _bg_cancel_all()
                    st.experimental_rerun()
            with c_bg[2]:
                # Auto-refresh is scheduled globally in the sidebar status area.
                st.caption(f"\u81ea\u52a8\u5237\u65b0\u4e2d \u00b7 \u5fc3\u8df3 {time.strftime('%H:%M:%S')}")

            # Optional detailed log: dedup/filter noisy internal lines.
            if p_tail:
                clean_tail: list[str] = []
                prev = ""
                for ln in p_tail[-24:]:
                    s = str(ln or "").strip()
                    if not s:
                        continue
                    if s.startswith("converter profile:") or s.startswith("LLM concurrency:"):
                        continue
                    if s == prev:
                        continue
                    clean_tail.append(s)
                    prev = s
                if clean_tail:
                    with st.expander("\u8be6\u7ec6\u8fdb\u5ea6\uff08\u53ef\u9009\uff09", expanded=False):
                        for ln in clean_tail[-10:]:
                            st.caption(f"- {ln}")

            st.session_state[meta_key] = {
                "done": done,
                "total": total,
                "cur": cur,
                "cur_tid": cur_tid,
                "p_done": p_done,
                "p_total": p_total,
                "overall_display": overall_display,
                "file_display": file_display,
                "file_key": file_key,
            }

            # NOTE:
            # Do not call server-side rerun here (inside tab blocks), it may interrupt
            # rendering of sibling tabs. Global server-side fallback rerun is handled
            # once at the end of main().

        if lib_fragment_refresh_ok:
            @st.fragment(run_every=0.9)
            def render_bg_progress_under_tasks_live(*, key_ns: str) -> None:
                render_bg_progress_under_tasks(key_ns=key_ns)
        else:
            def render_bg_progress_under_tasks_live(*, key_ns: str) -> None:
                render_bg_progress_under_tasks(key_ns=key_ns)

        def render_items(items: list[dict], *, show_missing_badge: bool, key_ns: str) -> None:
            from typing import Optional

            bg2 = _bg_snapshot()
            queue_tasks = list(bg2.get("queue") or [])
            current_name = str(bg2.get("current") or "")
            running_any = bool(bg2.get("running"))

            def _queue_pos(pdf_path: Path) -> Optional[int]:
                # Normalize paths for comparison
                try:
                    p_normalized = str(pdf_path.resolve())
                    p_original = str(pdf_path)
                except Exception:
                    p_normalized = str(pdf_path)
                    p_original = str(pdf_path)
                
                for i, t in enumerate(queue_tasks, start=1):
                    try:
                        task_pdf = str(t.get("pdf") or "")
                        if not task_pdf:
                            continue
                        try:
                            task_pdf_normalized = str(Path(task_pdf).resolve())
                        except Exception:
                            task_pdf_normalized = task_pdf
                        # Compare both normalized and original paths
                        if task_pdf_normalized == p_normalized or task_pdf == p_original or task_pdf_normalized == p_original or task_pdf == p_normalized:
                            return i
                    except Exception:
                        continue
                return None

            for idx, it in enumerate(items, start=1):
                pdf = it["pdf"]
                md_main = it["md_main"]
                md_exists = bool(it["md_exists"])
                task_info = it.get("task_info", {})
                uid = hashlib.md5(str(pdf).encode("utf-8", "ignore")).hexdigest()[:10]

                queued_pos = _queue_pos(pdf)
                running_this = running_any and (current_name == pdf.name)
                is_reconverting = task_info.get("replace", False) and (task_info.get("queued", False) or task_info.get("running", False))
                
                # Determine badge based on status
                if running_this:
                    badge = "\u3010\u8f6c\u6362\u4e2d\u3011"
                elif queued_pos is not None:
                    if task_info.get("replace", False):
                        badge = "\u3010\u91cd\u65b0\u8f6c\u6362\u961f\u5217\u4e2d\u3011"
                    else:
                        badge = "\u3010\u8f6c\u6362\u961f\u5217\u4e2d\u3011"
                elif is_reconverting:
                    badge = "\u3010\u91cd\u65b0\u8f6c\u6362\u4e2d\u3011"
                elif show_missing_badge and not md_exists:
                    badge = "\u3010\u672a\u8f6c\u6362\u3011"
                else:
                    badge = "\u3010\u5df2\u8f6c\u6362\u3011"

                pdf_display_name = _library_pdf_display_name(lib_store, pdf)
                title = f"{badge} {pdf_display_name}"

                with st.expander(title, expanded=False):
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
                                        speed_mode=str(speed_mode),
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
<b>PDF</b>\uff1a{html.escape(pdf_display_name)}<br/>
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
                                # Remove queued tasks first (best-effort, even if queued_pos UI missed it).
                                try:
                                    _bg_remove_queued_tasks_for_pdf(pdf)
                                except Exception:
                                    pass

                                # Delete PDF on disk
                                ok_pdf, pdf_del_msg = _safe_delete_file(pdf)

                                # Delete MD folder if requested
                                ok_md = True
                                md_del_msg = ""
                                if also_md:
                                    try:
                                        md_root = Path(md_out_root).resolve()
                                        target = (Path(md_out_root) / pdf.stem).resolve()
                                        if str(target).lower().startswith(str(md_root).lower()) and _path_exists(target):
                                            ok_md, md_del_msg = _safe_delete_tree(target)
                                    except Exception as e:
                                        ok_md = False
                                        md_del_msg = str(e)

                                # Best-effort remove from library index
                                try:
                                    lib_store.delete_by_path(pdf)
                                except Exception:
                                    pass
                                if ok_pdf:
                                    try:
                                        st.session_state["kb_reindex_pending"] = True
                                        st.session_state.pop("_kb_reindex_hint_cache", None)
                                    except Exception:
                                        pass

                                st.session_state[del_key] = False
                                if ok_pdf:
                                    st.success("\u5df2\u5220\u9664 PDF\u3002")
                                else:
                                    detail = f"（{pdf_del_msg}）" if str(pdf_del_msg or "").strip() else ""
                                    st.warning(f"\u5220\u9664 PDF \u5931\u8d25{detail}\uff08\u53ef\u80fd\u88ab\u5360\u7528/\u8def\u5f84\u8fc7\u957f\uff09\u3002")
                                if also_md and (not ok_md):
                                    detail = f"（{md_del_msg}）" if str(md_del_msg or "").strip() else ""
                                    st.warning(f"\u5220\u9664 MD \u6587\u4ef6\u5939\u5931\u8d25{detail}\u3002")
                                st.info("\u5982\u679c\u4f60\u5df2\u7ecf\u5efa\u5e93\uff0c\u5220\u9664\u540e\u5efa\u8bae\u70b9\u4e00\u6b21\u300c\u66f4\u65b0\u77e5\u8bc6\u5e93\u300d\u4ee5\u6e05\u7406\u65e7\u7d22\u5f15\u3002")
                                st.experimental_rerun()
                        with c_del2[1]:
                            if st.button("\u53d6\u6d88\u5220\u9664", key=f"{key_ns}_del_cancel_{uid}"):
                                st.session_state[del_key] = False
                                st.experimental_rerun()

        # During conversion, auto-refresh relies on frequent Streamlit reruns.
        # Rendering hundreds of expanders/buttons every second makes reruns slow, so the progress
        # bar *appears* not to refresh. We default to a compact view while tasks are running.
        bg_view = _bg_snapshot()
        running_view = bg_is_running_snapshot(bg_view)
        show_full_list = True
        if running_view:
            show_full_list = st.checkbox(
                "\u8f6c\u6362\u8fdb\u884c\u4e2d\u4e5f\u663e\u793a\u5b8c\u6574\u6587\u4ef6\u5217\u8868\uff08\u53ef\u80fd\u5361\u987f\uff09",
                value=False,
                key="lib_show_full_list_during_run",
                help="\u4e3a\u4fdd\u8bc1\u8fdb\u5ea6\u6761\u80fd\u6bcf\u79d2\u5237\u65b0\uff0c\u8f6c\u6362\u65f6\u9ed8\u8ba4\u4e0d\u6e32\u67d3\u5927\u5217\u8868\u3002",
            )

        if not show_full_list:
            render_bg_progress_under_tasks_live(key_ns="lib_compact_progress")
            q = list(bg_view.get("queue") or [])
            if q:
                names: list[str] = []
                for t in q[:12]:
                    nm = str(t.get("name") or "").strip()
                    if not nm:
                        try:
                            nm = Path(str(t.get("pdf") or "")).name
                        except Exception:
                            nm = str(t.get("pdf") or "").strip()
                    if nm:
                        names.append(nm)
                if names:
                    st.caption("\u961f\u5217\uff08\u524d 12\uff09\uff1a" + "\u3001".join(names) + ("\u2026" if len(q) > 12 else ""))
            st.caption(f"\u5217\u8868\u7edf\u8ba1\uff1a\u672a\u8f6c\u6362 {len(pending)} | \u5df2\u8f6c\u6362 {len(converted)}\u3002\u9700\u67e5\u770b\u6587\u4ef6\u64cd\u4f5c\uff0c\u8bf7\u52fe\u9009\u4e0a\u65b9\u201c\u663e\u793a\u5b8c\u6574\u5217\u8868\u201d\u3002")
        else:
            tabs = st.tabs([
                f"\u672a\u8f6c\u6362 ({len(pending)})",
                f"\u5df2\u8f6c\u6362 ({len(converted)})",
                f"\u5f53\u524d\u5217\u8868 ({len(pdfs_view)})",
            ])

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
                                        speed_mode=str(speed_mode),
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
                render_bg_progress_under_tasks_live(key_ns="tab_pending_bottom")

            with tabs[1]:
                render_bg_progress_under_tasks_live(key_ns="tab_done")
                render_items(converted, show_missing_badge=False, key_ns="tab_done")

            with tabs[2]:
                render_bg_progress_under_tasks_live(key_ns="tab_all")
                render_items(pending + converted, show_missing_badge=True, key_ns="tab_all")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader(S["upload_pdf"])
    st.caption(S["batch_upload"])

    # In old Streamlit builds, frequent heartbeat reruns may duplicate file_uploader DOM.
    # Lock upload controls while any background task with auto-rerun is active.
    bg_upload = _bg_snapshot()
    ref_upload = refsync_snapshot()
    conv_running = bg_is_running_snapshot(bg_upload)
    ref_running = refsync_is_running_snapshot(ref_upload)
    upload_locked = bool(conv_running or ref_running)

    prev_upload_locked = bool(st.session_state.get("_upload_locked_prev", False))
    if prev_upload_locked and not upload_locked:
        # Force-remount uploader once when background tasks stop, clearing stale DOM.
        st.session_state["pdf_uploader_nonce"] = int(st.session_state.get("pdf_uploader_nonce", 0) or 0) + 1
    st.session_state["_upload_locked_prev"] = upload_locked

    if upload_locked:
        if conv_running:
            st.info("\u540e\u53f0\u8f6c\u6362\u6b63\u5728\u8fd0\u884c\uff0c\u4e0a\u4f20\u533a\u6682\u65f6\u9501\u5b9a\uff0c\u4efb\u52a1\u7ed3\u675f\u540e\u4f1a\u81ea\u52a8\u6062\u590d\u3002")
            st.caption("\u5982\u9700\u7acb\u5373\u4e0a\u4f20\uff0c\u8bf7\u5148\u70b9\u51fb\u300c\u505c\u6b62\u300d\u3002")
        else:
            st.info("\u53c2\u8003\u6587\u732e\u7d22\u5f15\u540e\u53f0\u540c\u6b65\u4e2d\uff0c\u4e0a\u4f20\u533a\u6682\u65f6\u9501\u5b9a\uff0c\u540c\u6b65\u5b8c\u6210\u540e\u4f1a\u81ea\u52a8\u6062\u590d\u3002")
    else:
        handled: dict = st.session_state.setdefault("upload_handled", {})
        uploader_nonce = int(st.session_state.get("pdf_uploader_nonce", 0) or 0)
        uploader_key = f"pdf_uploader_main_{uploader_nonce}"
        ups = st.file_uploader("PDF", type=["pdf"], accept_multiple_files=True, key=uploader_key)

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

                    base = build_storage_base_name(
                        venue=venue,
                        year=year,
                        title=title,
                        pdf_dir=pdf_dir,
                        md_out_root=md_out_root,
                    )
                    display_full_name = build_display_pdf_filename(
                        venue=venue,
                        year=year,
                        title=title,
                        fallback_name=up.name,
                    )
                    st.text_input(S["base_name"], value=base, disabled=True, key=f"base_{n}")
                    st.text_input(
                        "完整显示名（用于引用/文献篮/参考定位）",
                        value=display_full_name,
                        disabled=True,
                        key=f"display_name_{n}",
                    )

                    dest_pdf = _next_pdf_dest_path(pdf_dir, base)
                    upload_citation_meta = merge_citation_meta_file_labels(
                        sug.crossref_meta if isinstance(sug.crossref_meta, dict) else None,
                        display_full_name=display_full_name,
                        storage_filename=dest_pdf.name,
                    )
                    upload_citation_meta = merge_citation_meta_name_fields(
                        upload_citation_meta,
                        venue=venue,
                        year=year,
                        title=title,
                    )

                    action_cols = st.columns(2)
                    with action_cols[0]:
                        if st.button(S["save_pdf"], key=f"save_pdf_{n}"):
                            _persist_upload_pdf(tmp_path, dest_pdf, data)
                            lib_store.upsert(file_sha1, dest_pdf, citation_meta=upload_citation_meta)
                            handled[file_sha1] = {"action": "saved", "msg": S["handled_saved"], "ts": time.time()}
                            st.info(f"{S['saved_as']}: {dest_pdf}")

                    with action_cols[1]:
                        if st.button(S["convert_now"], key=f"convert_now_{n}"):
                            _persist_upload_pdf(tmp_path, dest_pdf, data)
                            lib_store.upsert(file_sha1, dest_pdf, citation_meta=upload_citation_meta)
                            handled[file_sha1] = {"action": "converted", "msg": S["handled_converted"], "ts": time.time()}
                            _bg_enqueue(
                                _build_bg_task(
                                    pdf_path=dest_pdf,
                                    out_root=md_out_root,
                                    db_dir=db_dir,
                                    no_llm=bool(no_llm),
                                    replace=False,
                                    speed_mode=str(speed_mode),
                                )
                            )
                            st.info("\u5df2\u52a0\u5165\u540e\u53f0\u961f\u5217\uff0c\u4f60\u53ef\u4ee5\u5207\u6362\u9875\u9762\u7ee7\u7eed\u4f7f\u7528\u3002")
                            st.experimental_rerun()






def main() -> None:
    st.set_page_config(page_title=S["title"], layout="wide")
    # Diagnostic heartbeat: increments on every Streamlit script run.
    st.session_state["_run_tick"] = int(st.session_state.get("_run_tick", 0) or 0) + 1
    st.session_state["_run_tick_ts"] = time.strftime("%H:%M:%S")
    # Base theme CSS is light; browser preference overrides are applied in frontend JS.
    _init_theme_css("light")
    _sync_theme_with_browser_preference()
    _render_app_title()

    settings = load_settings()
    chat_store = ChatStore(settings.chat_db_path)
    lib_store = LibraryStore(settings.library_db_path)
    # Store lib_store in session_state for refs_renderer to access
    st.session_state["lib_store"] = lib_store

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

        st.subheader(S["settings"])
        st.caption("主题跟随浏览器/系统设置")

        # Background conversion status (shown on every page)
        _bg_ensure_started()
        bg = _bg_snapshot()
        if bg_is_running_snapshot(bg):
            done = int(bg.get("done", 0) or 0)
            total = int(bg.get("total", 0) or 0)
            cur = bg.get("current", "")
            run_tick = int(st.session_state.get("_run_tick", 0) or 0)
            run_tick_ts = str(st.session_state.get("_run_tick_ts") or "")
            st.caption(
                f"后台转换：{done}/{total}{(' | ' + cur) if cur else ''} | "
                f"run#{run_tick} @{run_tick_ts}"
            )
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

        # Keep DB path from config/prefs for retrieval internals.
        db_default = prefs.get("db_path") or str(settings.db_dir)
        db_path = (str(db_default) or "").strip().strip("'\"")
        db_dir = Path(db_path).expanduser().resolve()
        st.session_state["db_dir"] = str(db_dir)

        top_k = st.slider(S["top_k"], min_value=2, max_value=20, value=int(prefs.get("top_k") or 6), step=1)
        temperature = st.slider(S["temp"], min_value=0.0, max_value=1.0, value=float(prefs.get("temperature") or 0.2), step=0.05)
        max_tokens_pref = int(prefs.get("max_tokens") or 1216)
        max_tokens_pref = max(256, min(4096, max_tokens_pref))
        max_tokens_pref = 256 + int(round((max_tokens_pref - 256) / 64.0)) * 64
        max_tokens_pref = max(256, min(4096, max_tokens_pref))
        max_tokens = st.slider(S["max_tokens"], min_value=256, max_value=4096, value=int(max_tokens_pref), step=64)
        # "Show full snippet" feature removed; keep snippets collapsed-only behavior.
        show_context = False
        deep_read = st.checkbox(S["deep_read"], value=bool(prefs.get("deep_read") if ("deep_read" in prefs) else True))
        llm_rerank = True
        st.session_state["llm_rerank"] = True

        # Persist simple knobs
        prefs_knobs = dict(prefs)
        prefs_knobs.pop("show_context", None)
        prefs_knobs.update(
            {
                "top_k": int(top_k),
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "deep_read": bool(deep_read),
                "llm_rerank": bool(llm_rerank),
            }
        )
        if prefs_knobs != prefs:
            save_prefs(prefs_path, prefs_knobs)
            prefs.update(prefs_knobs)

        history_slot = st.container()

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.subheader("模型/Key")
        try:
            st.caption(f"Base URL: {getattr(settings, 'base_url', '')}")
            st.caption(f"Model: {getattr(settings, 'model', '')}")
            try:
                m0 = str(getattr(settings, "model", "") or "").strip().lower()
                is_vl = ("vl" in m0) or ("vision" in m0)
            except Exception:
                is_vl = False
            st.caption(f"VL(多模态) 判定: {'是' if is_vl else '否/未知'}")
            try:
                st.caption(f"KB_PDF_LLM_VISION_MATH: {str(os.environ.get('KB_PDF_LLM_VISION_MATH','(unset)'))}")
            except Exception:
                pass
            st.caption(f"Timeout: {float(getattr(settings, 'timeout_s', 0) or 0):.0f}s | Retries: {int(getattr(settings, 'max_retries', 0) or 0)}")
        except Exception:
            pass
        if getattr(settings, "api_key", None):
            st.caption("API Key：已设置")
        else:
            st.warning("API Key：未设置（需要 QWEN_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY）。设置后需重启 Streamlit。")
        if st.button("测试模型连接", key="test_llm_btn"):
            try:
                with st.spinner("测试中…"):
                    ds = DeepSeekChat(settings)
                    out = ds.chat(messages=[{"role": "user", "content": "ping"}], temperature=0.0, max_tokens=8)
                st.success(f"OK：{out or '(空)'}")
            except Exception as e:
                st.error(f"测试失败：{e}")

        # Sidebar actions trimmed: keep this area quieter and rely on auto reload / conversation controls.
        reload_btn = False
        clear_btn = False

        history_slot.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        history_slot.subheader(S["history"])
        def _history_new_chat_click() -> None:
            try:
                new_id = str(chat_store.create_conversation() or "").strip()
            except Exception:
                new_id = ""
            if new_id:
                st.session_state["conv_id"] = new_id

        def _history_pick_chat_click(target_id: str) -> None:
            cid = str(target_id or "").strip()
            if not cid:
                return
            st.session_state["conv_id"] = cid

        def _history_delete_chat_by_id(target_id: str) -> None:
            target = str(target_id or "").strip()
            if not target:
                return
            cur_before = str(st.session_state.get("conv_id") or "").strip()
            try:
                chat_store.delete_conversation(target)
            except Exception:
                pass
            try:
                remaining = chat_store.list_conversations(limit=50) or []
            except Exception:
                remaining = []
            remaining_ids = [str(x.get("id") or "").strip() for x in remaining if str(x.get("id") or "").strip()]

            next_id = ""
            if cur_before and (cur_before != target) and (cur_before in remaining_ids):
                next_id = cur_before
            elif remaining_ids:
                next_id = remaining_ids[0]
            else:
                try:
                    next_id = str(chat_store.create_conversation() or "").strip()
                except Exception:
                    next_id = ""

            if next_id:
                st.session_state["conv_id"] = next_id
            st.session_state.pop("conv_select", None)

        def _history_delete_chat_click() -> None:
            cur_id = str(st.session_state.get("conv_id") or "").strip()
            if not cur_id:
                return
            _history_delete_chat_by_id(cur_id)

        hist_action_cols = history_slot.columns([1, 1], gap="small")
        with hist_action_cols[0]:
            st.button(S["new_chat"], key="new_chat", on_click=_history_new_chat_click, use_container_width=True)
        with hist_action_cols[1]:
            st.button(
                "\u5220\u9664\u672c\u4f1a\u8bdd",
                key="delete_chat_inline",
                on_click=_history_delete_chat_click,
                disabled=not bool(str(st.session_state.get("conv_id") or "").strip()),
                use_container_width=True,
            )

        cur_conv_id_hint = str(st.session_state.get("conv_id") or "").strip()
        try:
            convs = chat_store.list_conversations(limit=200) or []
        except Exception:
            convs = []

        # Hard cap conversation history to 30 records in storage.
        if len(convs) > 30:
            ordered_ids = [str(c.get("id") or "").strip() for c in convs if str(c.get("id") or "").strip()]
            keep_ids = ordered_ids[:30]
            if cur_conv_id_hint and (cur_conv_id_hint in ordered_ids) and (cur_conv_id_hint not in keep_ids):
                # Preserve current conversation if it falls outside the top 30.
                replaced = False
                for i in range(len(keep_ids) - 1, -1, -1):
                    if keep_ids[i] != cur_conv_id_hint:
                        keep_ids[i] = cur_conv_id_hint
                        replaced = True
                        break
                if not replaced:
                    keep_ids = [cur_conv_id_hint]
            keep_set = set(keep_ids)
            for cid in ordered_ids:
                if cid in keep_set:
                    continue
                try:
                    chat_store.delete_conversation(cid)
                except Exception:
                    pass
            try:
                convs = chat_store.list_conversations(limit=50) or []
            except Exception:
                convs = [c for c in convs if str(c.get("id") or "").strip() in keep_set][:30]
        conv_labels: dict[str, str] = {}
        conv_updated_ts: dict[str, float] = {}
        conv_ids: list[str] = []
        for c in convs:
            cid = str(c.get("id") or "").strip()
            if not cid:
                continue
            conv_ids.append(cid)
            ts_raw = float(c.get("updated_at", 0) or 0)
            conv_updated_ts[cid] = ts_raw
            ts = time.strftime("%m-%d %H:%M", time.localtime(ts_raw))
            conv_labels[cid] = f"{ts} | {c.get('title','')}"

        cur_conv_id = str(st.session_state.get("conv_id") or "").strip()
        if not cur_conv_id:
            if conv_ids:
                cur_conv_id = conv_ids[0]
                st.session_state["conv_id"] = cur_conv_id
            else:
                cur_conv_id = str(chat_store.create_conversation() or "").strip()
                st.session_state["conv_id"] = cur_conv_id
                if cur_conv_id:
                    conv_ids = [cur_conv_id]

        if cur_conv_id:
            # Keep the active conversation at the top so the sidebar behaves like a chat app list.
            conv_ids = [cur_conv_id] + [x for x in conv_ids if x != cur_conv_id]

        recent_cutoff_ts = time.time() - (7 * 24 * 60 * 60)
        recent_conv_ids: list[str] = []
        older_conv_ids: list[str] = []
        for cid in conv_ids:
            ts0 = float(conv_updated_ts.get(cid, 0.0) or 0.0)
            # Always keep the active conversation visible, even if it is older than 7 days.
            if (cid == cur_conv_id) or (ts0 >= recent_cutoff_ts):
                recent_conv_ids.append(cid)
            else:
                older_conv_ids.append(cid)

        if "history_show_older_convs" not in st.session_state:
            st.session_state["history_show_older_convs"] = False

        def _toggle_history_older_click() -> None:
            st.session_state["history_show_older_convs"] = not bool(st.session_state.get("history_show_older_convs"))

        def _render_history_rows(row_ids: list[str], *, slot=None) -> None:
            host = slot or history_slot
            for cid in row_ids:
                row_label = conv_labels.get(cid, cid)
                row_cols = host.columns([9.45, 0.55], gap="small")
                with row_cols[0]:
                    st.button(
                        row_label,
                        key=f"conv_pick_{cid}",
                        on_click=_history_pick_chat_click,
                        args=(cid,),
                        use_container_width=True,
                    )
                with row_cols[1]:
                    st.button(
                        "\U0001F5D1",
                        key=f"conv_del_{cid}",
                        on_click=_history_delete_chat_by_id,
                        args=(cid,),
                        help="Delete this conversation",
                    )

        history_slot.caption(S["pick_chat"])
        if not conv_ids:
            history_slot.caption("(no conversations)")
        else:
            _render_history_rows(recent_conv_ids)
            if older_conv_ids:
                show_older = bool(st.session_state.get("history_show_older_convs"))
                toggle_label = (
                    "\u25b8 \u5c55\u5f00\u66f4\u65e9\u4f1a\u8bdd"
                    if not show_older
                    else "\u25be \u6536\u8d77\u66f4\u65e9\u4f1a\u8bdd"
                )
                history_slot.button(
                    toggle_label,
                    key="history_toggle_older_convs",
                    on_click=_toggle_history_older_click,
                    use_container_width=True,
                )
                if show_older:
                    history_slot.caption("\u66f4\u65e9\u4f1a\u8bdd")
                    try:
                        older_scroll_slot = history_slot.container(height=280)
                    except TypeError:
                        older_scroll_slot = history_slot.container()
                    _render_history_rows(older_conv_ids, slot=older_scroll_slot)

    _inject_runtime_ui_fixes("auto", st.session_state.get("conv_id", ""))

    # Resolve chat-page upload directories even if the user never opened the Library page.
    # (Library page normally initializes pdf_dir_input/md_dir_input.)
    pdf_dir_raw = str(
        st.session_state.get("pdf_dir_input")
        or st.session_state.get("pdf_dir")
        or prefs.get("pdf_dir")
        or ""
    ).strip()
    if not pdf_dir_raw:
        pdf_dir_raw = str(st.session_state.get("pdf_dir") or "").strip()
    try:
        pdf_dir = Path(pdf_dir_raw).expanduser().resolve() if pdf_dir_raw else Path.home().resolve()
    except Exception:
        pdf_dir = Path(pdf_dir_raw).expanduser() if pdf_dir_raw else Path.home()

    md_dir_default_raw = str(
        st.session_state.get("md_dir")
        or prefs.get("md_dir")
        or os.environ.get("KB_MD_DIR")
        or str(db_dir)
    ).strip()
    md_dir_raw = str(
        st.session_state.get("md_dir_input")
        or md_dir_default_raw
        or str(db_dir)
    ).strip()
    try:
        md_out_root = Path(md_dir_raw).expanduser().resolve()
    except Exception:
        md_out_root = Path(md_dir_raw).expanduser()

    # Retriever (auto-reload when DB changes)
    docs_json = db_dir / "docs.json"
    cur_mtime = docs_json.stat().st_mtime if docs_json.exists() else None
    prev_mtime = st.session_state.get("db_mtime")
    mtime_changed = bool(
        (cur_mtime is not None)
        and ((prev_mtime is None) or (float(cur_mtime) > float(prev_mtime) + 1e-6))
    )
    need_reload = (
        ("retriever" not in st.session_state)
        or bool(reload_btn)
        or bool(retriever_reload_flag.get("reload"))
        or mtime_changed
    )
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
            retriever_reload_flag["reload"] = False

    st.session_state["messages"] = chat_store.get_messages(st.session_state["conv_id"])

    if page != S["page_chat"]:
        _teardown_chat_dock_runtime()

    # Pages
    if page == S["page_chat"]:
        _page_chat(
            settings,
            chat_store,
            lib_store,
            st.session_state["retriever"],
            db_dir,
            pdf_dir,
            md_out_root,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens,
            show_context=show_context,
            deep_read=deep_read,
        )

    elif page == S["page_library"]:
        _page_library(settings, lib_store, db_dir, prefs_path, prefs, retriever_reload_flag)

    # No separate PDF->Markdown page: conversion lives in the library page.

    # Global auto-refresh heartbeat while background work is running.
    # Prefer browser-side one-shot rerun (lighter feel than server-side sleep+rerun).
    # Keep server-side heartbeat only as an opt-in fallback for very old/broken frontends.
    bg_end = _bg_snapshot()
    ref_end = refsync_snapshot()
    ref_running_end = (page == S["page_library"]) and refsync_is_running_snapshot(ref_end)
    is_running_end = bg_is_running_snapshot(bg_end) or bool(ref_running_end)
    was_running = bool(st.session_state.get("_bg_was_running", False))
    lib_local_fragment_refresh = bool(
        (page == S["page_library"])
        and callable(getattr(st, "fragment", None))
        and st.session_state.get("_kb_library_progress_fragments_active", False)
    )
    if is_running_end:
        st.session_state["_bg_was_running"] = True
        if lib_local_fragment_refresh:
            # Library page progress runs via st.fragment local reruns; avoid app-wide heartbeat flicker.
            st.session_state["_global_auto_rerun_ts"] = 0.0
            st.session_state["_global_server_rerun_ts"] = 0.0
            st.session_state["_kb_auto_hb_started_ts"] = 0.0
            st.session_state["_kb_client_auto_rerun_ts"] = 0.0
        else:
            if not bool(st.session_state.get("_kb_auto_hb_started_ts")):
                st.session_state["_kb_auto_hb_started_ts"] = time.time()
            # Slightly slower heartbeat reduces visible flicker on the library page
            # while keeping progress responsive enough for long conversions.
            interval_s = 1.4 if page == S["page_library"] else 1.0
            st.session_state["_global_auto_rerun_ts"] = time.time()
            pulse_label = "__KB_AUTO_RERUN_PULSE__"
            try:
                if st.button(pulse_label, key="_kb_auto_rerun_pulse_btn"):
                    st.session_state["_kb_client_auto_rerun_ts"] = time.time()
            except Exception:
                pass
            try:
                _inject_auto_rerun_once(
                    delay_ms=int(max(300, interval_s * 1000)),
                    pulse_button_label=pulse_label,
                    nonce=f"{time.time():.6f}",
                )
            except Exception:
                pass

            # Server-side fallback watchdog:
            # - opt-in env var forces old behavior
            # - otherwise only trigger if client heartbeat appears unhealthy for a while
            force_server_hb = str(os.environ.get("KB_FORCE_SERVER_HEARTBEAT", "") or "").strip().lower() in {"1", "true", "yes", "on"}
            now_ts = time.time()
            hb_started_ts = float(st.session_state.get("_kb_auto_hb_started_ts", 0.0) or 0.0)
            client_hb_ts = float(st.session_state.get("_kb_client_auto_rerun_ts", 0.0) or 0.0)
            grace_s = max(2.5, interval_s * 2.2)
            client_hb_stale = (hb_started_ts > 0) and ((now_ts - hb_started_ts) > grace_s) and (
                (client_hb_ts <= 0.0) or ((now_ts - client_hb_ts) > max(3.5, interval_s * 2.8))
            )
            if force_server_hb or client_hb_stale:
                now_ts = time.time()
                last_ts = float(st.session_state.get("_global_server_rerun_ts", 0.0) or 0.0)
                wait_s = max(0.0, interval_s - (now_ts - last_ts))
                if wait_s > 0:
                    time.sleep(min(wait_s, 1.0))
                st.session_state["_global_server_rerun_ts"] = time.time()
                st.experimental_rerun()
    else:
        st.session_state["_global_auto_rerun_ts"] = 0.0
        st.session_state["_global_server_rerun_ts"] = 0.0
        st.session_state["_kb_auto_hb_started_ts"] = 0.0
        st.session_state["_kb_client_auto_rerun_ts"] = 0.0
        # One extra rerun on running->stopped transition to flush stale disabled widgets.
        if was_running:
            st.session_state["_bg_was_running"] = False
            st.experimental_rerun()
        else:
            st.session_state["_bg_was_running"] = False


if __name__ == "__main__":
    main()
