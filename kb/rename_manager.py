from __future__ import annotations

import hashlib
import html
import os
import re
import shutil
from pathlib import Path
from typing import Any

import streamlit as st

from kb.library_store import LibraryStore
from kb.pdf_tools import (
    PDF_META_EXTRACT_VERSION,
    PdfMetaSuggestion,
    abbreviate_venue,
    build_base_name,
    extract_pdf_meta_suggestion,
    open_in_explorer,
)

_RENAME_BASE_MAX = 88
_MD_IGNORED_MAIN_NAMES = {"assets_manifest.md", "quality_report.md"}


def _sanitize_filename_component(text: str) -> str:
    normalized = (text or "").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r'[<>:"/\\\\|?*]+', "-", normalized)
    normalized = normalized.replace("\u0000", "").strip()
    normalized = normalized.strip(" .-_")
    if len(normalized) > _RENAME_BASE_MAX:
        digest = hashlib.sha1(normalized.encode("utf-8", "ignore")).hexdigest()[:8]
        keep = max(12, _RENAME_BASE_MAX - len(digest) - 1)
        normalized = normalized[:keep].rstrip(" .-_") + "-" + digest
    return normalized


def _to_os_path(path_like: Path | str) -> str:
    p = Path(path_like).expanduser()
    try:
        s = str(p.resolve(strict=False))
    except Exception:
        s = str(p)
    if os.name != "nt":
        return s
    if s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):
        return "\\\\?\\UNC\\" + s.lstrip("\\")
    return "\\\\?\\" + s


def _path_exists(path_like: Path | str) -> bool:
    try:
        return bool(os.path.exists(_to_os_path(path_like)))
    except Exception:
        try:
            return Path(path_like).exists()
        except Exception:
            return False


def _path_is_dir(path_like: Path | str) -> bool:
    try:
        return bool(os.path.isdir(_to_os_path(path_like)))
    except Exception:
        try:
            return Path(path_like).is_dir()
        except Exception:
            return False


def _path_is_file(path_like: Path | str) -> bool:
    try:
        return bool(os.path.isfile(_to_os_path(path_like)))
    except Exception:
        try:
            return Path(path_like).is_file()
        except Exception:
            return False


def _safe_move_path(src: Path, dst: Path) -> bool:
    src_os = _to_os_path(src)
    dst_os = _to_os_path(dst)
    try:
        os.makedirs(os.path.dirname(dst_os), exist_ok=True)
    except Exception:
        pass
    try:
        os.replace(src_os, dst_os)
        return True
    except Exception:
        pass
    try:
        src.rename(dst)
        return True
    except Exception:
        pass
    try:
        shutil.move(src_os, dst_os)
        return True
    except Exception:
        return False


def _safe_move_file(src: Path, dst: Path) -> bool:
    if src == dst:
        return True
    src_os = _to_os_path(src)
    dst_os = _to_os_path(dst)
    try:
        os.makedirs(os.path.dirname(dst_os), exist_ok=True)
    except Exception:
        pass
    try:
        if os.path.exists(dst_os):
            os.remove(dst_os)
    except Exception:
        pass
    try:
        os.replace(src_os, dst_os)
        return True
    except Exception:
        pass
    try:
        with open(src_os, "rb") as fr, open(dst_os, "wb") as fw:
            shutil.copyfileobj(fr, fw, length=1024 * 1024)
        try:
            os.remove(src_os)
        except Exception:
            pass
        return True
    except Exception:
        return False


def _copy_file_no_overwrite(src: str, dst: str) -> str:
    if os.path.exists(dst):
        return dst
    return shutil.copy2(src, dst)


def _merge_dir_no_overwrite(src_dir: Path, dst_dir: Path) -> bool:
    src_os = _to_os_path(src_dir)
    dst_os = _to_os_path(dst_dir)
    try:
        shutil.copytree(src_os, dst_os, dirs_exist_ok=True, copy_function=_copy_file_no_overwrite)
    except Exception:
        return False
    try:
        shutil.rmtree(src_os, ignore_errors=True)
    except Exception:
        pass
    return True


def _pick_md_main_candidate(folder: Path, old_stem: str, new_stem: str) -> Path | None:
    if not _path_is_dir(folder):
        return None
    preferred = [
        folder / f"{old_stem}.en.md",
        folder / "output.md",
    ]
    for p in preferred:
        if _path_is_file(p):
            return p
    en_cands: list[Path] = []
    any_cands: list[Path] = []
    try:
        with os.scandir(_to_os_path(folder)) as it:
            for e in it:
                try:
                    if not e.is_file():
                        continue
                except Exception:
                    continue
                name = str(e.name or "")
                low = name.lower()
                if not low.endswith(".md"):
                    continue
                if low in _MD_IGNORED_MAIN_NAMES:
                    continue
                if low == f"{new_stem}.en.md".lower():
                    continue
                p = Path(folder) / name
                any_cands.append(p)
                if low.endswith(".en.md"):
                    en_cands.append(p)
    except Exception:
        return None
    if en_cands:
        return sorted(en_cands, key=lambda x: x.name.lower())[0]
    if any_cands:
        return sorted(any_cands, key=lambda x: x.name.lower())[0]
    return None


def _sync_md_after_pdf_rename(*, src_pdf: Path, dest_pdf: Path, md_out_root: Path) -> tuple[bool, str]:
    old_folder = Path(md_out_root) / src_pdf.stem
    new_folder = Path(md_out_root) / dest_pdf.stem
    target_folder: Path | None = None

    try:
        old_exists = _path_is_dir(old_folder)
        new_exists = _path_is_dir(new_folder)

        if old_exists and (not new_exists):
            moved = _safe_move_path(old_folder, new_folder)
            if not moved:
                copied = _merge_dir_no_overwrite(old_folder, new_folder)
                if not copied:
                    return False, f"MD folder move failed: {old_folder.name} -> {new_folder.name}"
            target_folder = new_folder
        elif old_exists and new_exists:
            merged = _merge_dir_no_overwrite(old_folder, new_folder)
            if not merged:
                return False, f"MD folder merge failed: {old_folder.name} -> {new_folder.name}"
            target_folder = new_folder
        elif new_exists:
            target_folder = new_folder
        else:
            # No generated MD folder yet; not an error for rename flow.
            return True, "No MD folder found, skipped"

        if (target_folder is None) or (not _path_is_dir(target_folder)):
            return False, "MD target folder missing after sync"

        new_main = target_folder / f"{dest_pdf.stem}.en.md"
        if _path_is_file(new_main):
            return True, f"MD synced: {target_folder.name}"

        cand = _pick_md_main_candidate(target_folder, src_pdf.stem, dest_pdf.stem)
        if cand is not None:
            ok = _safe_move_file(cand, new_main)
            if ok:
                return True, f"MD main renamed: {cand.name} -> {new_main.name}"
            return False, f"MD main rename failed: {cand.name} -> {new_main.name}"

        return True, f"MD folder synced: {target_folder.name}"
    except Exception as e:
        return False, f"MD sync exception: {e}"


def _muted(text: str) -> None:
    safe = html.escape(str(text or "").strip())
    st.markdown(
        f"<div style='font-size:0.76rem; color: var(--muted); opacity:0.76; line-height:1.34;'>{safe}</div>",
        unsafe_allow_html=True,
    )


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
    while True:
        tail = f"-{suffix}"
        stem = base_name
        if len(stem) + len(tail) > _RENAME_BASE_MAX:
            stem = stem[: max(8, _RENAME_BASE_MAX - len(tail))].rstrip(" .-_")
        cand = Path(pdf_dir) / f"{stem}{tail}.pdf"
        if (not cand.exists()) or (suffix >= 999):
            return cand
        suffix += 1


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
        st.session_state["rename_scan_use_llm"] = True
    if "rename_only_diff" not in st.session_state:
        st.session_state["rename_only_diff"] = True
    if "rename_also_md" not in st.session_state:
        st.session_state["rename_also_md"] = True


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
            st.session_state["rename_scan_use_llm"] = True
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
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    scan_key = _scan_cache_key(dir_sig=dir_sig, scope=scope, use_llm=use_llm, expected_ver=expected_ver)
    results_cache = st.session_state.setdefault("rename_scan_results_cache", {})
    if force_refresh:
        try:
            results_cache.pop(scan_key, None)
        except Exception:
            pass

    cached_rows = results_cache.get(scan_key)
    if (not force_refresh) and isinstance(cached_rows, list):
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
            suggestion = None if force_refresh else meta_cache.get(cache_key)
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

            md_out_root = None
            try:
                maybe_root = getattr(settings, "md_out_root", None) if settings is not None else None
                if maybe_root:
                    md_out_root = Path(maybe_root)
            except Exception:
                md_out_root = None
            base = build_base_name(
                venue=suggestion.venue,
                year=suggestion.year,
                title=suggestion.title,
                pdf_dir=pdf_dir,
                md_out_root=md_out_root,
            ).strip()
            rows.append(
                {
                    "path": str(pdf),
                    "old": pdf.name,
                    "old_stem": pdf.stem,
                    "suggest": base,
                    "diff": bool(base) and (base != pdf.stem),
                    "meta": {"venue": suggestion.venue, "year": suggestion.year, "title": suggestion.title},
                    "crossref_meta": suggestion.crossref_meta,  # Store for later use
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
        _muted("\u6ca1\u6709\u9700\u8981\u6539\u540d\u7684\u6587\u4ef6\u3002")
        return []

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    _muted(f"\u5171 {len(rows)} \u6761\u5efa\u8bae")

    uids = [hashlib.md5(str(row.get("path") or "").encode("utf-8", "ignore")).hexdigest()[:10] for row in rows]
    selected_n = sum(1 for uid in uids if bool(st.session_state.get(f"rename_sel_{uid}", False)))
    all_selected = (len(uids) > 0) and (selected_n == len(uids))

    bulk_cols = st.columns([1.5, 8.5])
    with bulk_cols[0]:
        toggle_label = "\u5168\u4e0d\u9009" if all_selected else "\u5168\u9009"
        if st.button(toggle_label, key="rename_sel_toggle"):
            target = not all_selected
            for uid in uids:
                st.session_state[f"rename_sel_{uid}"] = target
            st.experimental_rerun()
    with bulk_cols[1]:
        _muted(f"\u5df2\u9009 {selected_n}/{len(uids)}")

    for row in rows[:400]:
        pdf_path_str = str(row.get("path") or "")
        uid = hashlib.md5(pdf_path_str.encode("utf-8", "ignore")).hexdigest()[:10]
        sel_key = f"rename_sel_{uid}"
        new_key = f"rename_new_{uid}"
        old_name = str(row.get("old") or "")
        suggestion = str(row.get("suggest") or "")
        meta = row.get("meta") or {}
        venue = str(meta.get("venue") or "")
        year = str(meta.get("year") or "")
        title = str(meta.get("title") or "")

        if sel_key not in st.session_state:
            st.session_state[sel_key] = False
        if new_key not in st.session_state:
            st.session_state[new_key] = suggestion

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        row_cols = st.columns([0.5, 4.7, 6.0, 1.5])

        with row_cols[0]:
            st.checkbox(" ", key=sel_key)

        with row_cols[1]:
            st.markdown(f"<div class='meta-kv'><b>\u6587\u4ef6</b>\uff1a{html.escape(old_name)}</div>", unsafe_allow_html=True)
            title_short = title[:80] + ("\u2026" if len(title) > 80 else "")
            venue_short = abbreviate_venue(venue)
            if venue and venue_short and (venue_short != venue):
                venue_label = f"{venue} ({venue_short})"
            else:
                venue_label = venue
            meta_line = " | ".join([value for value in [venue_label, year, title_short] if value])
            if meta_line:
                _muted(meta_line)

        with row_cols[2]:
            if not suggestion:
                _muted("\u672a\u8bc6\u522b\u51fa\u53ef\u7528\u5efa\u8bae")
            else:
                st.text_input(" ", key=new_key)

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
        apply_now = st.button("\u5e94\u7528\u91cd\u547d\u540d", key="rename_apply_btn")
    with apply_cols[1]:
        _muted("重命名后会提示更新知识库。")

    if not apply_now:
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
            moved_pdf = _safe_move_path(src, dest)
            if not moved_pdf:
                raise RuntimeError("filesystem rename failed")
            try:
                lib_store.update_path(src, dest)
                # Store Crossref metadata if available and trusted
                crossref_meta = row.get("crossref_meta")
                if isinstance(crossref_meta, dict):
                    lib_store.set_citation_meta(dest, crossref_meta)
            except Exception:
                pass
            operations.append(("ok", f"{src.name} \u2192 {dest.name}"))
        except Exception as e:
            operations.append(("fail", f"{src.name} \u91cd\u547d\u540d\u5931\u8d25\uff1a{e}"))
            continue

        if also_md:
            md_ok, md_msg = _sync_md_after_pdf_rename(src_pdf=src, dest_pdf=dest, md_out_root=md_out_root)
            if not md_ok:
                operations.append(("fail", f"MD 同步失败（{dest.name}）：{md_msg}"))

    try:
        results_cache = st.session_state.setdefault("rename_scan_results_cache", {})
        results_cache.pop(scan_key, None)
    except Exception:
        pass

    ok_n = sum(1 for status, _ in operations if status == "ok")
    skip_n = sum(1 for status, _ in operations if status == "skip")
    fail_n = sum(1 for status, _ in operations if status == "fail")
    if ok_n > 0:
        st.session_state["kb_reindex_pending"] = True
        st.session_state.pop("_kb_reindex_hint_cache", None)

    st.markdown(
        (
            "<div style='display:flex;gap:0.45rem;flex-wrap:wrap;margin:0.30rem 0 0.56rem 0;'>"
            f"<span style='font-size:0.78rem;padding:0.16rem 0.50rem;border-radius:999px;border:1px solid rgba(34,197,94,0.35);background:rgba(34,197,94,0.14);color:#22c55e;'>已重命名 {ok_n}</span>"
            f"<span style='font-size:0.78rem;padding:0.16rem 0.50rem;border-radius:999px;border:1px solid rgba(148,163,184,0.34);background:rgba(148,163,184,0.12);color:var(--text-soft);'>跳过 {skip_n}</span>"
            f"<span style='font-size:0.78rem;padding:0.16rem 0.50rem;border-radius:999px;border:1px solid rgba(248,113,113,0.34);background:rgba(248,113,113,0.12);color:#f87171;'>失败 {fail_n}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    if fail_n:
        _muted("\u6709\u5931\u8d25\u9879\uff0c\u8bf7\u5728\u300c\u5904\u7406\u8be6\u60c5\u300d\u91cc\u67e5\u770b\u539f\u56e0\u3002")

    if operations:
        detail_lines: list[str] = []
        for status, message in operations[:250]:
            tag = "OK" if status == "ok" else ("SKIP" if status == "skip" else "FAIL")
            one_line = " ".join(str(message or "").replace("\r", " ").replace("\n", " ").split())
            detail_lines.append(f"[{tag}] {one_line}")
        sel_key = f"rename_detail_sel_{scan_key}"
        if sel_key not in st.session_state:
            st.session_state[sel_key] = detail_lines[0]
        st.selectbox("处理详情", options=detail_lines, key=sel_key)

    try:
        dismissed_dirs.add(dir_sig)
    except Exception:
        pass


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
        # Fixed defaults (not user-configurable)
        st.session_state["rename_only_diff"] = True
        st.session_state["rename_also_md"] = True
        st.session_state["rename_scan_use_llm"] = True

        top_cols = st.columns([2.6, 1.7, 0.45])
        with top_cols[0]:
            options = ["\u6700\u8fd1 30 \u7bc7", "\u6700\u8fd1 50 \u7bc7", "\u6700\u8fd1 100 \u7bc7", "\u5168\u90e8"]
            current_scope = str(st.session_state.get("rename_scan_scope") or "\u6700\u8fd1 30 \u7bc7")
            if current_scope not in options:
                st.session_state["rename_scan_scope"] = options[0]
            scope = st.selectbox("\u626b\u63cf\u8303\u56f4", options=options, key="rename_scan_scope")
        with top_cols[1]:
            st.markdown("<div style='height:1.82rem;'></div>", unsafe_allow_html=True)
            scan_now = st.button("\u5f00\u59cb\u8bc6\u522b/\u5237\u65b0", key="rename_scan_btn")
        with top_cols[2]:
            st.markdown("<div style='height:1.82rem;'></div>", unsafe_allow_html=True)
            if st.button("\u00d7", key="rename_mgr_close_x", help="\u5173\u95ed\u6587\u4ef6\u540d\u7ba1\u7406"):
                st.session_state["rename_mgr_open"] = False
                st.experimental_rerun()
        use_llm = bool(st.session_state.get("rename_scan_use_llm"))
        scan_key = _scan_cache_key(dir_sig=dir_sig, scope=str(scope), use_llm=use_llm, expected_ver=expected_ver)
        results_cache = st.session_state.setdefault("rename_scan_results_cache", {})
        cached_rows = results_cache.get(scan_key) if isinstance(results_cache.get(scan_key), list) else None

        if scan_now:
            rows = _scan_rows(
                pdf_dir=pdf_dir,
                pdfs=pdfs,
                dir_sig=dir_sig,
                scope=str(scope),
                use_llm=use_llm,
                settings=settings,
                expected_ver=expected_ver,
                force_refresh=True,
            )
        else:
            rows = list(cached_rows or [])

        if not rows:
            _muted("\u8fd8\u6ca1\u6709\u53ef\u7528\u5efa\u8bae\uff0c\u70b9\u201c\u5f00\u59cb\u8bc6\u522b/\u5237\u65b0\u201d\u3002")
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
