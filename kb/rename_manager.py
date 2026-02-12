from __future__ import annotations

import hashlib
import html
import os
import re
from pathlib import Path
from typing import Any

import streamlit as st

from kb.library_store import LibraryStore
from kb.pdf_tools import (
    PDF_META_EXTRACT_VERSION,
    PdfMetaSuggestion,
    build_base_name,
    extract_pdf_meta_suggestion,
    open_in_explorer,
)


def _sanitize_filename_component(text: str) -> str:
    normalized = (text or "").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r'[<>:"/\\\\|?*]+', "-", normalized)
    normalized = normalized.replace("\u0000", "").strip()
    normalized = normalized.strip(" .-_")
    return normalized


def _is_normalized_pdf_stem(stem: str) -> bool:
    value = (stem or "").strip()
    if not value or len(value) < 12:
        return False
    return bool(re.search(r"-.?(19\d{2}|20\d{2}).?-", value))


def _unique_pdf_path(pdf_dir: Path, base: str) -> Path:
    base_name = _sanitize_filename_component(base) or "paper"
    dest = Path(pdf_dir) / f"{base_name}.pdf"
    if not dest.exists():
        return dest
    suffix = 2
    while (Path(pdf_dir) / f"{base_name}-{suffix}.pdf").exists() and suffix < 999:
        suffix += 1
    return Path(pdf_dir) / f"{base_name}-{suffix}.pdf"


def _select_recent_pdf_paths(pdf_dir: Path, n: int) -> list[Path]:
    import heapq

    limit = int(n or 0)
    if limit <= 0:
        return []

    heap: list[tuple[float, str]] = []
    try:
        with os.scandir(Path(pdf_dir)) as it:
            for entry in it:
                try:
                    if (not entry.is_file()) or (not entry.name.lower().endswith(".pdf")):
                        continue
                    modified = float(entry.stat().st_mtime)
                    heapq.heappush(heap, (modified, entry.path))
                    if len(heap) > limit:
                        heapq.heappop(heap)
                except Exception:
                    continue
    except Exception:
        return []

    heap.sort(reverse=True)
    return [Path(path) for _, path in heap]


def _scan_cache_key(*, dir_sig: str, scope: str, use_llm: bool, expected_ver: str) -> str:
    llm_flag = "llm1" if use_llm else "llm0"
    return hashlib.sha1((f"{dir_sig}|{scope}|{llm_flag}|ver:{expected_ver}").encode("utf-8", "ignore")).hexdigest()[:16]


def ensure_state_defaults() -> None:
    if "rename_mgr_open" not in st.session_state:
        st.session_state["rename_mgr_open"] = False
    if "rename_scan_scope" not in st.session_state:
        st.session_state["rename_scan_scope"] = "\u6700\u8fd1 30 \u7bc7"
    if "rename_scan_use_llm" not in st.session_state:
        st.session_state["rename_scan_use_llm"] = False
    if "rename_only_diff" not in st.session_state:
        st.session_state["rename_only_diff"] = True
    if "rename_also_md" not in st.session_state:
        st.session_state["rename_also_md"] = False


def render_prompt(*, pdfs: list[Path], dir_sig: str, dismissed_dirs: set[str]) -> None:
    if (not isinstance(dismissed_dirs, set)) or (dir_sig in dismissed_dirs) or (not pdfs):
        return

    non_normalized = 0
    for pdf in pdfs[:40]:
        if pdf.name.lower().startswith("__upload__"):
            continue
        if not _is_normalized_pdf_stem(pdf.stem):
            non_normalized += 1
        if non_normalized >= 3:
            break

    if non_normalized < 3:
        return

    st.markdown(
        "<div class='kb-notice'>"
        "\u68c0\u6d4b\u5230\u4f60\u8bbe\u7f6e\u4e86\u65b0\u7684 PDF \u76ee\u5f55\uff1a"
        "\u5176\u4e2d\u6709\u4e9b\u6587\u4ef6\u540d\u53ef\u80fd\u4e0d\u662f\u300c\u671f\u520a-\u5e74\u4efd-\u6807\u9898\u300d\u3002"
        "\u8981\u4e0d\u8981\u6211\u6839\u636e PDF \u5185\u5bb9\u8bc6\u522b\u4fe1\u606f\u5e76\u7ed9\u51fa\u91cd\u547d\u540d\u5efa\u8bae\uff1f"
        "</div>",
        unsafe_allow_html=True,
    )
    prompt_cols = st.columns([1.1, 1.0, 10.0])
    with prompt_cols[0]:
        if st.button("\u67e5\u770b\u5efa\u8bae", key="rename_prompt_open"):
            st.session_state["rename_mgr_open"] = True
            st.session_state["rename_scan_scope"] = "\u6700\u8fd1 30 \u7bc7"
            st.session_state["rename_scan_use_llm"] = False
    with prompt_cols[1]:
        if st.button("\u4ee5\u540e\u518d\u8bf4", key="rename_prompt_dismiss"):
            try:
                dismissed_dirs.add(dir_sig)
            except Exception:
                pass


def _scan_rows(
    *,
    pdf_dir: Path,
    pdfs: list[Path],
    dir_sig: str,
    scope: str,
    use_llm: bool,
    settings: Any,
    expected_ver: str,
) -> list[dict[str, Any]]:
    scan_key = _scan_cache_key(dir_sig=dir_sig, scope=scope, use_llm=use_llm, expected_ver=expected_ver)
    results_cache = st.session_state.setdefault("rename_scan_results_cache", {})

    cached_rows = results_cache.get(scan_key)
    if isinstance(cached_rows, list):
        return cached_rows

    meta_cache = st.session_state.setdefault("rename_pdf_meta_cache", {})
    rows: list[dict[str, Any]] = []

    if scope.startswith("\u6700\u8fd1"):
        try:
            n_scope = int(re.sub(r"\D+", "", scope) or "50")
        except Exception:
            n_scope = 50
        scan_pdfs = _select_recent_pdf_paths(pdf_dir, max(1, n_scope))
    else:
        scan_pdfs = list(pdfs)

    total = len(scan_pdfs)
    progress = st.progress(0.0) if total > 0 else None
    llm_budget = 8
    llm_used = 0

    with st.spinner(f"\u6b63\u5728\u8bc6\u522b PDF \u4fe1\u606f\uff08{total} \u7bc7\uff09..."):
        for i, pdf in enumerate(scan_pdfs, start=1):
            if pdf.name.lower().startswith("__upload__"):
                continue

            try:
                stat = pdf.stat()
                modified = float(stat.st_mtime)
                size = int(stat.st_size)
            except Exception:
                modified, size = 0.0, 0
            llm_flag = "llm1" if use_llm else "llm0"

            cache_key = hashlib.sha1(
                (f"{pdf}|{modified}|{size}|{llm_flag}|ver:{expected_ver}").encode("utf-8", "ignore")
            ).hexdigest()[:16]
            suggestion = meta_cache.get(cache_key)
            if not isinstance(suggestion, PdfMetaSuggestion):
                try:
                    suggestion = extract_pdf_meta_suggestion(pdf, settings=None)
                    if use_llm and settings:
                        needs_llm = not (suggestion.title or "").strip() or not (suggestion.year or "").strip() or not (
                            suggestion.venue or ""
                        ).strip()
                        if needs_llm and llm_used < llm_budget:
                            suggestion_llm = extract_pdf_meta_suggestion(pdf, settings=settings)
                            if suggestion_llm:
                                suggestion = suggestion_llm
                            llm_used += 1
                except Exception:
                    suggestion = PdfMetaSuggestion()
                meta_cache[cache_key] = suggestion

            base = build_base_name(venue=suggestion.venue, year=suggestion.year, title=suggestion.title).strip()
            rows.append(
                {
                    "path": str(pdf),
                    "old": pdf.name,
                    "old_stem": pdf.stem,
                    "suggest": base,
                    "diff": bool(base) and (base != pdf.stem),
                    "meta": {"venue": suggestion.venue, "year": suggestion.year, "title": suggestion.title},
                }
            )

            if progress is not None:
                progress.progress(min(1.0, i / max(1, total)))

    if progress is not None:
        progress.empty()

    results_cache[scan_key] = rows
    return rows


def _render_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if bool(st.session_state.get("rename_only_diff")):
        rows = [row for row in rows if bool(row.get("diff"))]

    if not rows:
        st.caption("\uff08\u6ca1\u6709\u9700\u8981\u6539\u540d\u7684\u6587\u4ef6\uff0c\u6216\u8bc6\u522b\u4e0d\u5230\u53ef\u7528\u4fe1\u606f\uff09")
        return []

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.caption(f"\u5171 {len(rows)} \u6761\u5efa\u8bae\uff08\u53ea\u4f1a\u5728\u4f60\u70b9\u51fb\u300c\u5e94\u7528\u91cd\u547d\u540d\u300d\u540e\u771f\u6b63\u6539\u540d\uff09")

    bulk_cols = st.columns([1.4, 1.4, 7.2])
    with bulk_cols[0]:
        if st.button("\u5168\u9009", key="rename_sel_all"):
            for row in rows:
                uid = hashlib.md5(str(row.get("path") or "").encode("utf-8", "ignore")).hexdigest()[:10]
                st.session_state[f"rename_sel_{uid}"] = True
    with bulk_cols[1]:
        if st.button("\u5168\u4e0d\u9009", key="rename_sel_none"):
            for row in rows:
                uid = hashlib.md5(str(row.get("path") or "").encode("utf-8", "ignore")).hexdigest()[:10]
                st.session_state[f"rename_sel_{uid}"] = False
    with bulk_cols[2]:
        st.caption("\u5efa\u8bae\u683c\u5f0f\uff1a\u671f\u520a-\u5e74\u4efd-\u6807\u9898\uff08\u53ef\u7f16\u8f91\uff09")

    for row in rows[:400]:
        pdf_path_str = str(row.get("path") or "")
        uid = hashlib.md5(pdf_path_str.encode("utf-8", "ignore")).hexdigest()[:10]
        old_name = str(row.get("old") or "")
        suggestion = str(row.get("suggest") or "")
        meta = row.get("meta") or {}
        venue = str(meta.get("venue") or "")
        year = str(meta.get("year") or "")
        title = str(meta.get("title") or "")

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        row_cols = st.columns([0.9, 4.9, 5.6, 1.6])

        with row_cols[0]:
            st.checkbox("\u9009\u62e9", value=bool(st.session_state.get(f"rename_sel_{uid}", False)), key=f"rename_sel_{uid}")

        with row_cols[1]:
            st.markdown(f"<div class='meta-kv'><b>\u5f53\u524d</b>\uff1a{html.escape(old_name)}</div>", unsafe_allow_html=True)
            title_short = title[:80] + ("\u2026" if len(title) > 80 else "")
            meta_line = " | ".join([value for value in [venue, year, title_short] if value])
            if meta_line:
                st.caption(f"\u8bc6\u522b\uff1a{meta_line}")

        with row_cols[2]:
            if not suggestion:
                st.caption("\uff08\u8bc6\u522b\u4e0d\u5230\u53ef\u7528\u4fe1\u606f\uff0c\u8df3\u8fc7\uff09")
            else:
                st.text_input("\u5efa\u8bae\u65b0\u6587\u4ef6\u540d\uff08\u4e0d\u542b .pdf\uff09", value=suggestion, key=f"rename_new_{uid}")

        with row_cols[3]:
            pdf_path = Path(pdf_path_str)
            if st.button("\u6253\u5f00", key=f"rename_open_{uid}", help="\u6253\u5f00 PDF \u67e5\u770b"):
                try:
                    os.startfile(str(pdf_path))  # type: ignore[attr-defined]
                except Exception:
                    try:
                        open_in_explorer(pdf_path)
                    except Exception as e:
                        st.warning(f"\u6253\u5f00\u5931\u8d25\uff1a{e}")
            if st.button("\u5b9a\u4f4d", key=f"rename_loc_{uid}", help="\u5728\u8d44\u6e90\u7ba1\u7406\u5668\u4e2d\u5b9a\u4f4d"):
                try:
                    open_in_explorer(pdf_path)
                except Exception as e:
                    st.warning(f"\u5b9a\u4f4d\u5931\u8d25\uff1a{e}")

    return rows


def _apply_renames(
    *,
    rows: list[dict[str, Any]],
    pdf_dir: Path,
    md_out_root: Path,
    lib_store: LibraryStore,
    dir_sig: str,
    dismissed_dirs: set[str],
    scan_key: str,
) -> None:
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    apply_cols = st.columns([1.8, 8.2])

    with apply_cols[0]:
        if not st.button("\u5e94\u7528\u91cd\u547d\u540d", key="rename_apply_btn"):
            return

        operations: list[tuple[str, str]] = []
        also_md = bool(st.session_state.get("rename_also_md"))

        for row in rows:
            pdf_path_str = str(row.get("path") or "")
            uid = hashlib.md5(pdf_path_str.encode("utf-8", "ignore")).hexdigest()[:10]
            if not bool(st.session_state.get(f"rename_sel_{uid}", False)):
                continue

            src = Path(pdf_path_str)
            if not src.exists():
                operations.append(("fail", f"\u4e0d\u5b58\u5728\uff1a{src}"))
                continue

            new_base = str(st.session_state.get(f"rename_new_{uid}") or "").strip()
            new_base = _sanitize_filename_component(new_base)
            if not new_base:
                operations.append(("skip", f"\u8df3\u8fc7\uff08\u7a7a\u540d\u5b57\uff09\uff1a{src.name}"))
                continue
            if new_base == src.stem:
                operations.append(("skip", f"\u8df3\u8fc7\uff08\u672a\u53d8\u5316\uff09\uff1a{src.name}"))
                continue

            dest = _unique_pdf_path(pdf_dir, new_base)
            try:
                src.rename(dest)
                try:
                    lib_store.update_path(src, dest)
                except Exception:
                    pass
                operations.append(("ok", f"{src.name} \u2192 {dest.name}"))
            except Exception as e:
                operations.append(("fail", f"{src.name} \u91cd\u547d\u540d\u5931\u8d25\uff1a{e}"))
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

        try:
            results_cache = st.session_state.setdefault("rename_scan_results_cache", {})
            results_cache.pop(scan_key, None)
        except Exception:
            pass

        ok_n = sum(1 for status, _ in operations if status == "ok")
        fail_n = sum(1 for status, _ in operations if status == "fail")
        if ok_n:
            st.success(f"\u5df2\u91cd\u547d\u540d\uff1a{ok_n} \u4e2a\u6587\u4ef6")
        if fail_n:
            st.warning(f"\u5931\u8d25\uff1a{fail_n} \u4e2a\u6587\u4ef6\uff08\u53ef\u80fd\u662f\u6587\u4ef6\u6b63\u88ab\u5360\u7528\uff09")
        if operations:
            st.markdown("---")
            st.markdown("**处理详情：**")
            with st.container():
                for status, message in operations[:200]:
                    (st.caption if status in ("ok", "skip") else st.warning)(message)

        try:
            dismissed_dirs.add(dir_sig)
        except Exception:
            pass

    with apply_cols[1]:
        st.markdown(
            "<div class='refbox'>"
            "\u63d0\u793a\uff1a\u5982\u679c\u4f60\u5df2\u7ecf\u628a\u67d0\u4e9b PDF \u8f6c\u6362\u6210 MD \u5e76\u4e14\u5df2\u5efa\u5e93\uff0c"
            "\u91cd\u547d\u540d\u540e\u5efa\u8bae\u70b9\u4e00\u6b21\u300c\u66f4\u65b0\u77e5\u8bc6\u5e93\u300d\u91cd\u65b0\u7d22\u5f15\uff0c"
            "\u4ee5\u514d\u65e7\u8def\u5f84\u6b8b\u7559\u3002"
            "</div>",
            unsafe_allow_html=True,
        )


def render_panel(
    *,
    pdf_dir: Path,
    md_out_root: Path,
    pdfs: list[Path],
    dir_sig: str,
    dismissed_dirs: set[str],
    settings: Any,
    lib_store: LibraryStore,
) -> None:
    ensure_state_defaults()
    if not bool(st.session_state.get("rename_mgr_open")):
        return

    expected_ver = str(PDF_META_EXTRACT_VERSION)
    if str(st.session_state.get("rename_mgr_cache_ver") or "") != expected_ver:
        st.session_state["rename_mgr_cache_ver"] = expected_ver
        st.session_state.pop("rename_pdf_meta_cache", None)
        st.session_state.pop("rename_scan_results_cache", None)

    with st.expander("\u6587\u4ef6\u540d\u7ba1\u7406\uff08\u671f\u520a-\u5e74\u4efd-\u6807\u9898\uff09", expanded=True):
        top_cols = st.columns([1.0, 1.8, 1.6, 1.6, 6.0])
        with top_cols[0]:
            if st.button("\u5173\u95ed", key="rename_mgr_close"):
                st.session_state["rename_mgr_open"] = False
                st.experimental_rerun()
        with top_cols[1]:
            options = ["\u6700\u8fd1 30 \u7bc7", "\u6700\u8fd1 50 \u7bc7", "\u6700\u8fd1 100 \u7bc7", "\u5168\u90e8"]
            current_scope = str(st.session_state.get("rename_scan_scope") or "\u6700\u8fd1 30 \u7bc7")
            index = options.index(current_scope) if current_scope in options else 0
            scope = st.selectbox("\u626b\u63cf\u8303\u56f4", options=options, index=index, key="rename_scan_scope")
        with top_cols[2]:
            st.checkbox("\u53ea\u663e\u793a\u9700\u6539\u540d", value=bool(st.session_state.get("rename_only_diff", True)), key="rename_only_diff")
        with top_cols[3]:
            st.checkbox(
                "\u540c\u65f6\u6539\u540d MD",
                value=bool(st.session_state.get("rename_also_md", False)),
                key="rename_also_md",
                help="\u5982\u679c\u5df2\u751f\u6210 Markdown\uff0c\u4f1a\u5c1d\u8bd5\u628a\u5bf9\u5e94\u6587\u4ef6\u5939/\u4e3b md \u4e00\u8d77\u6539\u540d\u3002",
            )
        with top_cols[4]:
            st.checkbox(
                "\u8bc6\u522b\u7528 LLM\uff08\u66f4\u51c6\uff09",
                value=bool(st.session_state.get("rename_scan_use_llm", False)),
                key="rename_scan_use_llm",
                help="\u53ea\u5728\u5173\u952e\u4fe1\u606f\u7f3a\u5931\u65f6\u5c11\u91cf\u8c03\u7528 LLM \u8865\u5168\uff08\u66f4\u51c6\u4f46\u66f4\u6162\uff09\u3002",
            )
        use_llm = bool(st.session_state.get("rename_scan_use_llm"))
        scan_key = _scan_cache_key(dir_sig=dir_sig, scope=str(scope), use_llm=use_llm, expected_ver=expected_ver)
        results_cache = st.session_state.setdefault("rename_scan_results_cache", {})
        cached_rows = results_cache.get(scan_key) if isinstance(results_cache.get(scan_key), list) else None

        action_cols = st.columns([1.2, 8.8])
        with action_cols[0]:
            scan_now = st.button("\u5f00\u59cb\u8bc6\u522b/\u5237\u65b0", key="rename_scan_btn")
        with action_cols[1]:
            st.markdown(
                "<div class='refbox'>"
                "\u8bfb\u53d6 PDF \u5185\u5bb9\uff08\u4e0d\u662f\u770b\u6587\u4ef6\u540d\uff09\u8bc6\u522b\u300c\u671f\u520a/\u4f1a\u8bae-\u5e74\u4efd-\u6807\u9898\u300d\uff0c"
                "\u7136\u540e\u7ed9\u51fa\u5efa\u8bae\u6587\u4ef6\u540d\u3002\u4f60\u53ef\u4ee5\u624b\u52a8\u7f16\u8f91\u540e\u518d\u5e94\u7528\u3002"
                "</div>",
                unsafe_allow_html=True,
            )

        if scan_now:
            rows = _scan_rows(
                pdf_dir=pdf_dir,
                pdfs=pdfs,
                dir_sig=dir_sig,
                scope=str(scope),
                use_llm=use_llm,
                settings=settings,
                expected_ver=expected_ver,
            )
        else:
            rows = list(cached_rows or [])

        if not rows:
            st.caption("\u70b9\u51fb\u300c\u5f00\u59cb\u8bc6\u522b/\u5237\u65b0\u300d\u540e\u624d\u4f1a\u8bfb\u53d6 PDF \u8fdb\u884c\u8bc6\u522b\uff08\u907f\u514d\u6253\u5f00\u7a97\u53e3\u5c31\u5361\u4f4f\uff09\u3002")
            return

        rows = _render_rows(rows)
        if rows:
            _apply_renames(
                rows=rows,
                pdf_dir=pdf_dir,
                md_out_root=md_out_root,
                lib_store=lib_store,
                dir_sig=dir_sig,
                dismissed_dirs=dismissed_dirs,
                scan_key=scan_key,
            )
