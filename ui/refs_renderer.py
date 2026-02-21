from __future__ import annotations

import hashlib
import html
import os
import re
import subprocess
import time
import difflib
from pathlib import Path
from urllib.parse import quote

import streamlit as st

from kb.citation_meta import extract_first_doi, fetch_best_crossref_meta
from kb.reference_index import (
    extract_references_map_from_md as _extract_references_map_from_md_index,
    load_reference_index as _load_reference_index_file,
    resolve_reference_entry as _resolve_reference_entry_from_index,
)
from kb.pdf_tools import open_in_explorer
from ui.strings import S
import json


def _trim_middle(text: str, *, max_len: int) -> str:
    s = (text or "").strip()
    if len(s) <= max_len:
        return s
    if max_len <= 8:
        return s[:max_len]
    keep = max_len - 3
    left = keep // 2
    right = keep - left
    return s[:left].rstrip() + "..." + s[-right:].lstrip()


def _top_heading(heading_path: str) -> str:
    hp = (heading_path or "").strip()
    if not hp:
        return ""
    return hp.split(" / ", 1)[0].strip()


def _display_source_name(source_path: str) -> str:
    name = Path(source_path).name or source_path or "unknown"
    for suf in (".en.md", ".md"):
        if name.lower().endswith(suf):
            name = name[: -len(suf)]
            break
    return _trim_middle(name, max_len=78)


def _is_temp_source_path(source_path: str) -> bool:
    s = (source_path or "").strip()
    if not s:
        return True
    p = Path(s)
    parts = [str(x).strip().lower() for x in p.parts]
    name = p.name.lower()
    stem = p.stem.lower()
    if any(x in {"temp", "__pycache__"} for x in parts):
        return True
    if any(x.startswith("__upload__") or x.startswith("_tmp_") or x.startswith("tmp_") for x in parts):
        return True
    if name.startswith("__upload__") or stem.startswith("__upload__"):
        return True
    if name.startswith("_tmp_") or stem.startswith("_tmp_"):
        return True
    if name.startswith("tmp_") or stem.startswith("tmp_"):
        return True
    return False


def _lookup_pdf_by_stem(pdf_root: Path, stem: str) -> Path | None:
    stem = (stem or "").strip()
    if not stem:
        return None
    if stem.endswith(".en"):
        stem = stem[: -3]

    direct = [
        pdf_root / f"{stem}.pdf",
        pdf_root / f"{stem}.PDF",
    ]
    for p in direct:
        if p.exists():
            return p

    # Fallback: scan by stem match.
    try:
        target = stem.lower()
        for p in pdf_root.glob("*.pdf"):
            if p.stem.lower() == target:
                return p
    except Exception:
        pass

    # Robust fallback: match by normalized title/year in filename.
    src_year, src_title_key = _parse_name_year_title_key(stem)
    if not src_title_key:
        return None

    best_path: Path | None = None
    best_score = -1.0
    try:
        for p in pdf_root.glob("*.pdf"):
            cand_year, cand_title_key = _parse_name_year_title_key(p.stem)
            if not cand_title_key:
                continue

            score = 0.0
            if src_title_key == cand_title_key:
                score = 6.0
            elif (src_title_key in cand_title_key) or (cand_title_key in src_title_key):
                score = 5.0
            else:
                a = set(src_title_key.split())
                b = set(cand_title_key.split())
                if a and b:
                    jacc = float(len(a & b)) / float(max(1, len(a | b)))
                    if jacc >= 0.74:
                        score = 3.0 + jacc

            if score <= 0.0:
                continue

            if src_year and cand_year:
                try:
                    dy = abs(int(src_year) - int(cand_year))
                except Exception:
                    dy = 99
                if dy == 0:
                    score += 2.0
                elif dy == 1:
                    score += 1.0

            if score > best_score:
                best_score = score
                best_path = p
    except Exception:
        return None

    return best_path


def _open_pdf(pdf_path: Path) -> tuple[bool, str]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return False, f"PDF not found: {pdf_path}"
    try:
        os.startfile(str(pdf_path))  # type: ignore[attr-defined]
        return True, f"Opened: {pdf_path}"
    except Exception:
        pass
    try:
        subprocess.Popen(["cmd", "/c", "start", "", str(pdf_path)], shell=False)
        return True, f"Opened: {pdf_path}"
    except Exception:
        pass
    try:
        subprocess.Popen(["powershell", "-NoProfile", "-Command", "Start-Process", "-FilePath", str(pdf_path)])
        return True, f"Opened: {pdf_path}"
    except Exception:
        pass
    try:
        open_in_explorer(pdf_path)
        return True, f"Revealed in Explorer: {pdf_path}"
    except Exception as e:
        return False, f"Open failed: {e}"


def _file_url_for_pdf(path: Path, *, page: int | None = None) -> str:
    p = Path(path).resolve()
    url = p.as_uri()
    if page and int(page) > 0:
        url += f"#page={int(page)}"
    if url.startswith("file:///"):
        prefix = "file:///"
        tail = url[len(prefix):]
        return prefix + quote(tail, safe="/:#?&=%")
    return quote(url, safe=":/#?&=%")


def _open_pdf_at(pdf_path: Path, *, page: int | None = None) -> tuple[bool, str]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return False, f"PDF not found: {pdf_path}"
    if page and int(page) > 0:
        url = _file_url_for_pdf(pdf_path, page=int(page))
        try:
            subprocess.Popen(["cmd", "/c", "start", "", url], shell=False)
            return True, f"Opened: {pdf_path} (page {int(page)})"
        except Exception:
            pass
    return _open_pdf(pdf_path)


def _safe_page(meta: dict) -> int | None:
    for x in [meta.get("page"), meta.get("page_num"), meta.get("page_idx")]:
        try:
            p = int(x)
        except Exception:
            continue
        if p > 0:
            return p
    return None


def _score_tier(score: float) -> str:
    if score >= 8.0:
        return "hi"
    if score >= 4.0:
        return "mid"
    return "low"


def _normalize_name_key(text: str) -> str:
    s = html.unescape((text or "").strip()).lower()
    s = s.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _parse_name_year_title_key(stem_like: str) -> tuple[str, str]:
    s = (stem_like or "").strip()
    if not s:
        return "", ""
    if s.lower().endswith(".en"):
        s = s[:-3]
    m = re.search(r"(19\d{2}|20\d{2})", s)
    if not m:
        return "", _normalize_name_key(s)
    year = m.group(1)
    title = s[m.end() :].lstrip(" -_.")
    if not title:
        title = s
    return year, _normalize_name_key(title)


def _snippet(text: str, *, heading: str = "", max_chars: int = 260) -> str:
    h_low = (heading or "").strip().lower()
    if ("references" in h_low) or ("bibliography" in h_low):
        return "References list (snippet omitted)."

    s = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not s:
        return ""

    cleaned_lines: list[str] = []
    for ln in s.split("\n"):
        ln = (ln or "").strip()
        if not ln:
            continue
        if re.match(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$", ln):
            continue
        ln = re.sub(r"!\[[^\]]*\]\([^)]+\)", "[image]", ln)
        ln = re.sub(r"^#{1,6}\s*", "", ln)
        ln = re.sub(r"^[-*+]\s+", "", ln)
        if len(ln) <= 1:
            continue
        cleaned_lines.append(ln)
        if len(cleaned_lines) >= 3:
            break

    s = " ".join(cleaned_lines) if cleaned_lines else ""
    s = re.sub(r"\s{2,}", " ", s).strip()
    if len(s) > max_chars:
        s = s[:max_chars].rstrip() + "..."
    return s


def _resolve_pdf_for_source(pdf_root: Path | None, source_path: str) -> Path | None:
    if not pdf_root:
        return None
    stem = (Path(source_path).stem or "").strip()
    if not stem:
        return None
    return _lookup_pdf_by_stem(pdf_root, stem)


# --- Citation Utilities ---

def _expand_venue_abbr(abbr: str) -> str:
    """
    Try to expand venue abbreviation to full name for better Crossref matching.
    Returns the original abbr if no expansion found.
    """
    if not abbr or len(abbr) < 2:
        return abbr
    
    abbr_lower = abbr.lower().strip()
    
    # Load venue map (reverse lookup: abbr -> full name)
    try:
        venue_map_path = Path(__file__).resolve().parent.parent / "kb" / "venue_abbr_map.json"
        if venue_map_path.exists():
            with open(venue_map_path, "r", encoding="utf-8") as f:
                venue_map = json.load(f)
                # Reverse lookup: find full name by abbreviation
                for full_name, mapped_abbr in venue_map.items():
                    if mapped_abbr.lower() == abbr_lower:
                        return full_name
    except Exception:
        pass
    
    return abbr


def _resolve_source_doc_path(source_path: str) -> Path | None:
    raw = (source_path or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if p.exists():
        return p
    if p.is_absolute():
        return None

    md_root_str = str(st.session_state.get("md_dir") or "").strip()
    if not md_root_str:
        return None
    md_root = Path(md_root_str)

    c1 = md_root / p
    if c1.exists():
        return c1
    c2 = md_root / p.name
    if c2.exists():
        return c2

    try:
        for hit in md_root.rglob(p.name):
            return hit
    except Exception:
        return None
    return None


def _load_source_preview_text(source_path: str, *, max_chars: int = 12000) -> str:
    p = _resolve_source_doc_path(source_path)
    if not p:
        return ""
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if not txt:
            return ""
        return txt[: max(1000, int(max_chars))]
    except Exception:
        return ""


def _infer_title_from_source_text(source_path: str, fallback_title: str) -> str:
    txt = _load_source_preview_text(source_path, max_chars=9000)
    if not txt:
        return fallback_title

    for raw_line in txt.splitlines()[:120]:
        line = (raw_line or "").strip()
        if not line:
            continue
        line = re.sub(r"^#{1,6}\s*", "", line).strip()
        if len(line) < 12:
            continue
        low = line.lower()
        if low.startswith(("abstract", "references", "bibliography")):
            continue
        if re.search(r"^(keywords?|introduction)\b", low):
            continue
        return line
    return fallback_title


def fetch_crossref_meta(title: str, *, source_path: str = "", expected_venue: str = "", expected_year: str = "") -> dict | None:
    """
    Synchronous fetch with strict confidence gate.
    Return None when not reliable enough.
    """
    q = (title or "").strip()
    if not q or len(q) < 5:
        return None

    doi_hint = extract_first_doi(source_path)
    if not doi_hint:
        doi_hint = extract_first_doi(_load_source_preview_text(source_path))
    venue = (expected_venue or "").strip()
    # Try to expand venue abbreviation to full name for better matching
    if venue:
        venue_expanded = _expand_venue_abbr(venue)
        # Use both original and expanded for better matching
        venues_to_try = [venue_expanded] if venue_expanded != venue else [venue]
    else:
        venues_to_try = [venue]
    year = (expected_year or "").strip()

    def _try(query_title: str, *, y: str, v: str, min_score: float, allow_title_only: bool = False) -> dict | None:
        return fetch_best_crossref_meta(
            query_title=query_title,
            expected_year=y,
            expected_venue=v,
            doi_hint=doi_hint,
            min_score=min_score,
            allow_title_only=allow_title_only,
        )

    # Try with each venue variant (original and expanded)
    for v_try in venues_to_try:
        # 1) Strict: year + venue (or what we currently know).
        out = _try(q, y=year, v=v_try, min_score=0.90)
        if isinstance(out, dict):
            return out

        # 2) Safe fallback for citation rendering:
        #    keep venue constraint, relax year (Crossref often stores online/print year differently).
        if year:
            out = _try(q, y="", v=v_try, min_score=0.90)
            if isinstance(out, dict):
                return out

    # 3) Relaxed title-only fallback (lower threshold for better recall).
    if len(q) >= 20:
        out = _try(q, y="", v="", min_score=0.92, allow_title_only=True)
        if isinstance(out, dict):
            return out

    # 4) Retry once with filename title when extracted first-line title is noisy.
    _, _, file_title = _parse_filename_meta(source_path)
    file_q = (file_title or "").strip()
    if file_q and file_q != q:
        for v_try in venues_to_try:
            out = _try(file_q, y=year, v=v_try, min_score=0.90)
            if isinstance(out, dict):
                return out
            if year:
                out = _try(file_q, y="", v=v_try, min_score=0.90)
                if isinstance(out, dict):
                    return out
        if len(file_q) >= 20:
            out = _try(file_q, y="", v="", min_score=0.92, allow_title_only=True)
            if isinstance(out, dict):
                return out

    return None


def _parse_filename_meta(path_str: str) -> tuple[str, str, str]:
    name = Path(path_str).stem
    if name.lower().endswith(".en"):
        name = name[:-3]
    m = re.match(r"^([^-]+)\s*-\s*(19\d{2}|20\d{2})\s*-\s*(.+)$", name)
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    return "", "", name


# --- Callback Function (THE FIX) ---
# --- Async Citation Worker ---

def _bg_citation_worker(task_id: str, net_key: str, source_path: str, venue_hint: str, year_hint: str):
    from kb import runtime_state as RUNTIME
    
    # 0. Try to load stored citation metadata from library_store first
    found = None
    try:
        # Get library_store from session_state (set by app.py)
        lib_store = st.session_state.get("lib_store")
        if lib_store:
            # Try to resolve PDF path from source_path
            pdf_root_str = str(st.session_state.get("pdf_dir") or "").strip()
            if pdf_root_str:
                pdf_root = Path(pdf_root_str)
                pdf_path = _resolve_pdf_for_source(pdf_root, source_path)
                if pdf_path and pdf_path.exists():
                    stored_meta = lib_store.get_citation_meta(pdf_path)
                    if stored_meta and isinstance(stored_meta, dict):
                        found = stored_meta
    except Exception:
        pass  # Fallback to fetching if stored metadata not available
    
    # 1. If no stored metadata, fetch from Crossref
    if not found:
        l_title_hint = os.path.basename(source_path)
        try:
            if l_title_hint.lower().endswith(".pdf"):
                l_title_hint = l_title_hint[:-4]
            search_title = _infer_title_from_source_text(source_path, l_title_hint)
        except Exception:
            search_title = l_title_hint

        # 2. Fetch (Blocking I/O)
        found = fetch_crossref_meta(
            search_title,
            source_path=source_path,
            expected_venue=venue_hint,
            expected_year=year_hint,
        )

    # 3. Update State
    with RUNTIME.CITATION_LOCK:
        tasks = RUNTIME.CITATION_TASKS
        if task_id in tasks:
            t = tasks[task_id]
            t["done"] = True
            t["result"] = found
            # Also update session_state copy if possible? 
            # No, stream lit session state is thread-local usually. 
            # We rely on the UI poll to pick it up from RUNTIME.CITATION_TASKS.


# --- Callback Function (Async) ---
def _on_cite_click(cite_key: str, net_key: str, source_path: str, refs_open_key: str = ""):
    """
    Triggers background fetch if data missing, returns immediately.
    """
    if refs_open_key:
        st.session_state[refs_open_key] = True

    # 1. Toggle visibility
    new_state = not st.session_state.get(cite_key, False)
    st.session_state[cite_key] = new_state

    # 2. If opening, start background task if needed
    if new_state:
        # Check if we already have data
        if st.session_state.get(net_key):
            return
        if st.session_state.get(f"{net_key}_failed"):
            return

        from kb import runtime_state as RUNTIME
        import threading
        import uuid

        # Check if task already running
        task_id = f"cite_task_{net_key}"
        with RUNTIME.CITATION_LOCK:
            if task_id in RUNTIME.CITATION_TASKS:
                return # Already running
            
            # Start new task
            RUNTIME.CITATION_TASKS[task_id] = {
                "created_at": time.time(),
                "done": False,
                "result": None,
                "net_key": net_key,
            }

        # Prepare hints
        l_venue, l_year, _ = _parse_filename_meta(source_path)
        
        # Fire thread
        t = threading.Thread(
            target=_bg_citation_worker,
            args=(task_id, net_key, source_path, l_venue, l_year),
            daemon=True
        )
        t.start()


def _render_refs(
        hits: list[dict],
        *,
        prompt: str = "",
        show_heading: bool = True,
        key_ns: str = "refs",
        refs_open_key: str = "",
        settings=None,
) -> None:
    del prompt, settings  # backward compatibility

    filtered_hits: list[dict] = []
    for h in hits or []:
        meta = h.get("meta", {}) or {}
        src = str(meta.get("source_path") or "").strip()
        if _is_temp_source_path(src):
            continue
        filtered_hits.append(h)

    if show_heading:
        st.markdown(f"### {S['refs']}")
    if not filtered_hits:
        st.markdown(f"<div class='refbox'>{S['kb_miss']}</div>", unsafe_allow_html=True)
        return

    pdf_root_str = str(st.session_state.get("pdf_dir") or "").strip()
    pdf_root = Path(pdf_root_str) if pdf_root_str else None
    show_context = bool(st.session_state.get("show_context") or False)

    for i, h in enumerate(filtered_hits, start=1):
        meta = h.get("meta", {}) or {}
        source_path = str(meta.get("source_path") or "").strip()
        heading = str(meta.get("top_heading") or _top_heading(str(meta.get("heading_path") or "")) or "").strip()
        page = _safe_page(meta)
        score = float(h.get("score", 0.0) or 0.0)

        # Basic Info
        source_label = _display_source_name(source_path)
        heading_label = heading if heading else ""  # Use full heading, not truncated
        score_s = f"{score:.2f}" if score > 0 else "-"
        score_tier = _score_tier(score)

        source_attr = html.escape(source_label, quote=True)
        heading_attr = html.escape(heading, quote=True) if heading else ""
        source_html = html.escape(source_label)
        heading_html = html.escape(heading) if heading else ""
        page_chip = f"<span class='ref-chip'>p.{int(page)}</span>" if page else ""

        pdf_path = _resolve_pdf_for_source(pdf_root, source_path)
        has_pdf = bool(pdf_path)
        uid = hashlib.sha1(str(source_path).encode("utf-8", "ignore")).hexdigest()[:10]
        cite_key = f"{key_ns}_cite_visible_{uid}"
        net_key = f"{key_ns}_net_meta_v5_{uid}"
        is_cite_open = st.session_state.get(cite_key, False)

        # Compact layout: rank, filename, buttons, score on one line
        # Tighter spacing: buttons moved right and closer together
        header_cols = st.columns([0.28, 3.2, 0.55, 0.55, 0.55, 0.4, 0.5])
        
        with header_cols[0]:
            st.markdown(
                f"<div style='display:flex; align-items:center; height:100%;'><span class='ref-rank'>#{i}</span></div>",
                unsafe_allow_html=True
            )
        
        with header_cols[1]:
            st.markdown(
                f"<div style='display:flex; align-items:center; height:100%;'><span class='ref-source-compact' title='{source_attr}'>{source_html}</span></div>",
                unsafe_allow_html=True
            )
        
        with header_cols[2]:
            if st.button("Open", key=f"{key_ns}_open_pdf_{uid}", help="Open PDF", disabled=(not has_pdf)):
                if refs_open_key:
                    st.session_state[refs_open_key] = True
                ok, msg = _open_pdf(pdf_path)
                if not ok:
                    st.warning(msg)
        
        with header_cols[3]:
            disabled_page = (page is None) or (not has_pdf)
            if st.button("Page", key=f"{key_ns}_open_page_{uid}", disabled=disabled_page,
                         help=f"Go to page {page}" if page else "Page unknown"):
                if refs_open_key:
                    st.session_state[refs_open_key] = True
                ok, msg = _open_pdf_at(pdf_path, page=page)
                if not ok:
                    st.warning(msg)
        
        with header_cols[4]:
            btn_label = "Close" if is_cite_open else "Cite"
            st.button(
                btn_label,
                key=f"{key_ns}_cite_btn_{uid}",
                help="Fetch citation",
                on_click=_on_cite_click,
                args=(cite_key, net_key, source_path, refs_open_key)
            )
        
        with header_cols[5]:
            if page_chip:
                st.markdown(f"<div style='display:flex; align-items:center; height:100%;'>{page_chip}</div>", unsafe_allow_html=True)
        
        with header_cols[6]:
            st.markdown(
                f"<div style='display:flex; align-items:center; height:100%;'><span class='ref-score ref-score-{score_tier}'>score {score_s}</span></div>",
                unsafe_allow_html=True
            )
        
        # Subtitle (heading) if available
        if heading_label:
            st.markdown(
                f"<div class='ref-item-sub-compact' title='{heading_attr}'>{heading_html}</div>",
                unsafe_allow_html=True
            )

        # Render Snippet (only if enabled and has text)
        text = _snippet(str(h.get("text") or ""), heading=heading)
        if show_context and text:
            st.markdown(
                f"<div class='snipbox-compact'><pre>{html.escape(text)}</pre></div>",
                unsafe_allow_html=True,
            )

        # Render Citation UI (if open)
        if st.session_state.get(cite_key, False):
            _render_citation_ui(uid, source_path, key_ns)

        st.markdown("<div class='ref-item-gap-compact'></div>", unsafe_allow_html=True)


# --- In-paper citation number resolver (e.g., "[45]" in body text) ---

_INPAPER_CITE_RE = re.compile(r"\[(\d{1,4})\]")
_INPAPER_CITE_GROUP_RE = re.compile(r"\[(\d{1,4}(?:\s*(?:-|–|—|,)\s*\d{1,4})+)\]")
_INPAPER_CITE_ANY_RE = re.compile(r"\[(\d{1,4}(?:\s*(?:-|–|—|,)\s*\d{1,4})*)\]")
_STRUCT_CITE_RE = re.compile(r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*(\d{1,4})\s*\]\]", re.IGNORECASE)
# Fallbacks for malformed model outputs like "[CITE:sid:24]" / "[[CITE:sid]]".
_STRUCT_CITE_SINGLE_RE = re.compile(r"(?<!\[)\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})(?:\s*:\s*(\d{1,4}))?\s*\](?!\])", re.IGNORECASE)
_STRUCT_CITE_SID_ONLY_RE = re.compile(r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*\]\]", re.IGNORECASE)
_STRUCT_CITE_GARBAGE_RE = re.compile(r"\[\[?\s*CITE\s*:[^\]\n]*\]?\]", re.IGNORECASE)
_CODE_FENCE_LINE_RE = re.compile(r"^\s*```")
_INLINE_CODE_RE = re.compile(r"(`[^`]*`)")
_INLINE_MATH_RE = re.compile(r"(\$[^$\n]+\$)")


_EQ_TAG_RE = re.compile(r"\\tag\{(\d{1,4})\}")
_REF_LEAD_LABEL_RE = re.compile(r"^\s*(?:\[\s*\d{1,4}\s*\]\s*){1,3}|^\s*\d{1,4}\s*[.)]\s*")


def _collect_source_paths_from_hits(hits: list[dict], *, max_docs: int = 16) -> list[str]:
    out: list[str] = []
    for h in hits or []:
        meta = h.get("meta", {}) or {}
        sp = str(meta.get("source_path") or "").strip()
        if not sp or _is_temp_source_path(sp):
            continue
        if sp in out:
            continue
        out.append(sp)
        if len(out) >= int(max_docs):
            break
    return out


def _load_reference_index_cached() -> dict:
    db_dir_str = str(st.session_state.get("db_dir") or "").strip()
    if not db_dir_str:
        return {}
    db_dir = Path(db_dir_str)
    idx_path = db_dir / "references_index.json"
    if not idx_path.exists():
        return {}

    try:
        idx_sig = f"{str(idx_path.resolve())}|{int(idx_path.stat().st_mtime)}|{int(idx_path.stat().st_size)}"
    except Exception:
        idx_sig = str(idx_path)

    cache_key = "_kb_ref_index_cache_v1"
    cache = st.session_state.get(cache_key)
    if isinstance(cache, dict) and str(cache.get("sig") or "") == idx_sig and isinstance(cache.get("data"), dict):
        return cache.get("data") or {}

    data = _load_reference_index_file(db_dir)
    st.session_state[cache_key] = {"sig": idx_sig, "data": data}
    return data if isinstance(data, dict) else {}


def _citation_hover_title(source_name: str, ref_num: int, ref_rec: dict) -> str:
    src = str(source_name or "").strip()
    title = str(ref_rec.get("title") or "").strip()
    doi = str(ref_rec.get("doi") or "").strip()
    parts = [f"source: {src}", f"ref [{int(ref_num)}]"]
    if title:
        parts.append(title)
    if doi:
        parts.append(f"DOI: {doi}")
    txt = " | ".join(parts)
    txt = txt.replace('"', "'").replace("\n", " ").strip()
    if len(txt) > 260:
        txt = txt[:257].rstrip() + "..."
    return txt


def _looks_noisy_reference_title(title: str) -> bool:
    t = str(title or "").strip()
    if not t:
        return True
    if len(t) >= 200:
        return True
    words = [w for w in re.split(r"\s+", t) if w]
    if len(words) >= 28:
        return True
    low = t.lower()
    # Long prose-like title is usually OCR contamination, not real paper title.
    if len(words) >= 18 and re.search(r"\b(the|and|with|through|because|therefore|contains|introduced)\b", low):
        return True
    return False


def _strip_reference_lead_label(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    t = s
    # Remove duplicated leading labels like "[50] [50] ...".
    for _ in range(3):
        t2 = _REF_LEAD_LABEL_RE.sub("", t).strip()
        if t2 == t:
            break
        t = t2
    return t


def _format_reference_cite_line(ref_rec: dict) -> str:
    if not isinstance(ref_rec, dict):
        return ""
    authors = _strip_reference_lead_label(str(ref_rec.get("authors") or "").strip())
    title = _strip_reference_lead_label(str(ref_rec.get("title") or "").strip())
    venue = _strip_reference_lead_label(str(ref_rec.get("venue") or "").strip())
    year = str(ref_rec.get("year") or "").strip()
    volume = str(ref_rec.get("volume") or "").strip()
    issue = str(ref_rec.get("issue") or "").strip()
    pages = str(ref_rec.get("pages") or "").strip()

    seg0 = []
    if authors:
        seg0.append(authors.rstrip(" ."))
    if title:
        seg0.append(title.rstrip(" ."))

    venue_seg = str(venue or "").strip()
    if volume:
        venue_seg += (", " if venue_seg else "") + volume
        if issue:
            venue_seg += f"({issue})"
    if pages:
        if volume:
            venue_seg += f":{pages}"
        else:
            venue_seg += (", " if venue_seg else "") + pages
    if year:
        venue_seg += f" ({year})" if venue_seg else year
    if venue_seg:
        seg0.append(venue_seg.rstrip(" ."))

    cite = ". ".join([x for x in seg0 if x]).strip()
    if cite and (not cite.endswith(".")):
        cite += "."
    return cite


def _normalize_reference_for_popup(ref_rec: dict) -> dict:
    if not isinstance(ref_rec, dict):
        return {}
    out = dict(ref_rec)
    doi = str(out.get("doi") or "").strip()
    title = _strip_reference_lead_label(str(out.get("title") or "").strip())
    authors = _strip_reference_lead_label(str(out.get("authors") or "").strip())
    venue = _strip_reference_lead_label(str(out.get("venue") or "").strip())
    year = str(out.get("year") or "").strip()
    volume = str(out.get("volume") or "").strip()
    issue = str(out.get("issue") or "").strip()
    pages = str(out.get("pages") or "").strip()
    raw = _strip_reference_lead_label(str(out.get("raw") or "").strip())

    # Avoid network calls in render path. Rendering should stay local/non-blocking.
    # DOI enrichment is handled during reference-index build/update.
    if (not doi) and raw:
        doi = str(extract_first_doi(raw) or "").strip()

    out["title"] = title
    out["authors"] = authors
    out["venue"] = venue
    out["year"] = year
    out["volume"] = volume
    out["issue"] = issue
    out["pages"] = pages
    out["doi"] = doi
    out["raw"] = raw
    out["cite_fmt"] = _format_reference_cite_line(out)
    return out


def _anchor_token(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return "global"
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:10]


def _build_inpaper_anchor(anchor_ns: str, ref_num: int, source_name: str = "") -> str:
    base = f"{str(anchor_ns or '').strip()}|{int(ref_num)}|{str(source_name or '').strip().lower()}"
    sig = _anchor_token(base)
    return f"kb-cite-{sig}-{int(ref_num)}"


def _source_cite_id(source_path: str) -> str:
    s = str(source_path or "").strip()
    if not s:
        return "s0000000"
    return "s" + hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:8]


def _ref_doi_url(ref_rec: dict) -> str:
    if not isinstance(ref_rec, dict):
        return ""
    u = str(ref_rec.get("doi_url") or "").strip()
    if u:
        return u
    d = str(ref_rec.get("doi") or "").strip()
    if d:
        return f"https://doi.org/{d}"
    return ""


def _annotate_inpaper_citations_with_hover_meta(
    md: str,
    hits: list[dict],
    *,
    anchor_ns: str = "",
) -> tuple[str, list[dict]]:
    s = (md or "")
    if not s or "[" not in s:
        return s, []

    def _strip_unresolved_structured_tokens(text: str) -> str:
        if not text or "CITE" not in text.upper():
            return text
        out = _STRUCT_CITE_RE.sub(lambda m: f"[{int(m.group(2))}]", text)
        out = _STRUCT_CITE_SINGLE_RE.sub(lambda m: f"[{int(m.group(2))}]" if str(m.group(2) or "").strip() else "", out)
        out = _STRUCT_CITE_SID_ONLY_RE.sub("", out)
        return _STRUCT_CITE_GARBAGE_RE.sub("", out)

    srcs = _collect_source_paths_from_hits(hits or [], max_docs=16)
    if not srcs:
        return _strip_unresolved_structured_tokens(s), []
    sid_to_source: dict[str, str] = {}
    for sp in srcs:
        sid = _source_cite_id(sp).lower()
        sid_to_source[sid] = sp

    index_data = _load_reference_index_cached()
    if not isinstance(index_data, dict):
        index_data = {}

    resolved_cache: dict[tuple[int, str], tuple[str, dict] | None] = {}
    detail_by_key: dict[str, dict] = {}

    def _resolve_num(n: int, preferred_sp: str = "") -> tuple[str, dict] | None:
        pref = str(preferred_sp or "").strip()
        ckey = (int(n), pref.lower())
        if ckey in resolved_cache:
            return resolved_cache[ckey]
        matches: list[tuple[str, dict]] = []
        ordered_srcs = list(srcs)
        if pref and pref in ordered_srcs:
            ordered_srcs = [pref] + [x for x in ordered_srcs if x != pref]
        for sp in ordered_srcs:
            got = _resolve_reference_entry_from_index(index_data, sp, n)
            if isinstance(got, dict):
                ref = got.get("ref")
                if isinstance(ref, dict):
                    src_name = str(got.get("source_name") or _display_source_name(sp)).strip()
                    matches.append((src_name, ref))

        picked: tuple[str, dict] | None = None
        if matches:
            if pref:
                # Preferred source is already first due reordering.
                picked = matches[0]
            elif len(matches) == 1:
                picked = matches[0]
            else:
                # Ambiguous across multiple source docs: do not force-pick the first one.
                picked = None
        resolved_cache[ckey] = picked
        return picked

    def _remember_detail(n: int, source_name: str, ref: dict) -> dict:
        skey = f"{int(n)}|{str(source_name or '').strip().lower()}"
        rec = detail_by_key.get(skey)
        if isinstance(rec, dict):
            return rec
        ref2 = _normalize_reference_for_popup(ref or {})
        raw_text = str(ref2.get("raw") or "").strip()
        doi_text = str(ref2.get("doi") or "").strip()
        if (not doi_text) and raw_text:
            doi_text = str(extract_first_doi(raw_text) or "").strip()
        doi_url = str(ref2.get("doi_url") or "").strip()
        if (not doi_url) and doi_text:
            doi_url = f"https://doi.org/{doi_text}"
        anchor = _build_inpaper_anchor(anchor_ns, int(n), source_name=source_name)
        rec = {
            "num": int(n),
            "anchor": anchor,
            "source_name": str(source_name or "").strip(),
            "raw": raw_text,
            "title": str(ref2.get("title") or "").strip(),
            "authors": str(ref2.get("authors") or "").strip(),
            "venue": str(ref2.get("venue") or "").strip(),
            "year": str(ref2.get("year") or "").strip(),
            "volume": str(ref2.get("volume") or "").strip(),
            "issue": str(ref2.get("issue") or "").strip(),
            "pages": str(ref2.get("pages") or "").strip(),
            "doi": doi_text,
            "doi_url": doi_url,
            "cite_fmt": str(ref2.get("cite_fmt") or "").strip(),
        }
        detail_by_key[skey] = rec
        return rec

    def _replace_text_segment(seg: str) -> str:
        structured_seen = False

        def _preferred_source_by_context(pos: int) -> str:
            try:
                left = seg[max(0, int(pos) - 160) : int(pos)]
            except Exception:
                left = seg
            # Heuristic: nearest low-number marker like [1]/[2] often denotes KB source id.
            markers = list(re.finditer(r"\[(\d{1,2})\]", left))
            if not markers:
                return ""
            for mm in reversed(markers):
                try:
                    k = int(mm.group(1))
                except Exception:
                    continue
                if 1 <= k <= len(srcs):
                    return str(srcs[k - 1])
            return ""

        def _mk_cite_link_md(n: int, detail: dict, title_attr: str) -> str:
            anchor = str(detail.get("anchor") or "").strip()
            t_attr = str(title_attr or "").replace('"', "'").replace("\n", " ").strip()
            return f"[{int(n)}](#{anchor} \"{t_attr}\")"

        def _resolve_struct_token(sid_raw: str, n_raw: str) -> str:
            nonlocal structured_seen
            sid = str(sid_raw or "").strip().lower()
            try:
                n = int(n_raw)
            except Exception:
                return ""
            structured_seen = True
            sp = sid_to_source.get(sid) or sid_to_source.get(sid.lower())
            if not sp:
                # Keep visible reference number even if sid cannot be mapped.
                return f"[{int(n)}]"
            got = _resolve_reference_entry_from_index(index_data, sp, int(n))
            if not isinstance(got, dict):
                return f"[{int(n)}]"
            ref = got.get("ref")
            if not isinstance(ref, dict):
                return f"[{int(n)}]"
            src_name = str(got.get("source_name") or _display_source_name(sp)).strip()
            detail = _remember_detail(int(n), src_name, ref)
            title_attr = _citation_hover_title(src_name, int(n), ref)
            return _mk_cite_link_md(int(n), detail, title_attr)

        def _repl_struct(m: re.Match) -> str:
            return _resolve_struct_token(str(m.group(1) or ""), str(m.group(2) or ""))

        def _repl_struct_single(m: re.Match) -> str:
            sid = str(m.group(1) or "")
            n_txt = str(m.group(2) or "").strip()
            if not n_txt:
                # Malformed form like [CITE:sid] -> hide raw token.
                return ""
            return _resolve_struct_token(sid, n_txt)

        def _repl_struct_sid_only(_: re.Match) -> str:
            # Malformed form like [[CITE:sid]] -> hide raw token.
            return ""

        def _repl_any(m: re.Match) -> str:
            raw = str(m.group(0) or "")
            spec = str(m.group(1) or "").strip()
            nums = _parse_int_set(spec)[:40]
            if not nums:
                return raw
            pref_sp = _preferred_source_by_context(int(m.start()))
            items: list[str] = []
            changed = False
            for n in nums:
                picked = _resolve_num(int(n), preferred_sp=pref_sp)
                if not picked:
                    items.append(f"[{int(n)}]")
                    continue
                src_name, ref = picked
                detail = _remember_detail(int(n), src_name, ref)
                title_attr = _citation_hover_title(src_name, int(n), ref)
                items.append(_mk_cite_link_md(int(n), detail, title_attr))
                changed = True
            if not changed:
                return raw
            return "".join(items)

        seg2 = _STRUCT_CITE_RE.sub(_repl_struct, seg)
        seg2 = _STRUCT_CITE_SINGLE_RE.sub(_repl_struct_single, seg2)
        seg2 = _STRUCT_CITE_SID_ONLY_RE.sub(_repl_struct_sid_only, seg2)
        # Final safety-net: never leak raw CITE tokens to UI.
        seg2 = _STRUCT_CITE_GARBAGE_RE.sub("", seg2)
        if structured_seen:
            return seg2
        return _INPAPER_CITE_ANY_RE.sub(_repl_any, seg2)

    out_lines: list[str] = []
    in_fence = False
    in_display_math = False
    for ln in s.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if _CODE_FENCE_LINE_RE.match(ln):
            in_fence = not in_fence
            out_lines.append(ln)
            continue
        if in_fence:
            out_lines.append(ln)
            continue
        if ln.strip() == "$$":
            in_display_math = not in_display_math
            out_lines.append(ln)
            continue
        if in_display_math:
            out_lines.append(ln)
            continue

        st_ln = (ln or "").strip()
        is_table_row = (st_ln.startswith("|") and st_ln.count("|") >= 2)
        is_table_sep = bool(re.match(r"^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*:?-{2,}:?\s*\|?\s*$", st_ln))
        if is_table_row or is_table_sep:
            out_lines.append(ln)
            continue

        code_parts = _INLINE_CODE_RE.split(ln)
        rebuilt_code: list[str] = []
        for i, cp in enumerate(code_parts):
            if i % 2 == 1:
                rebuilt_code.append(cp)
                continue
            math_parts = _INLINE_MATH_RE.split(cp)
            rebuilt_math: list[str] = []
            for j, mp in enumerate(math_parts):
                if j % 2 == 1:
                    rebuilt_math.append(mp)
                else:
                    rebuilt_math.append(_replace_text_segment(mp))
            rebuilt_code.append("".join(rebuilt_math))
        out_lines.append("".join(rebuilt_code))

    details = sorted(detail_by_key.values(), key=lambda x: (int(x.get("num") or 0), str(x.get("source_name") or "")))
    return "\n".join(out_lines), details


def _annotate_inpaper_citations_with_hover(md: str, hits: list[dict]) -> str:
    out, _ = _annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="")
    return out


def _render_inpaper_citation_details(
    cite_details: list[dict],
    *,
    key_ns: str,
    max_items: int = 24,
) -> None:
    del key_ns
    if not isinstance(cite_details, list) or not cite_details:
        return

    shown = [x for x in cite_details if isinstance(x, dict)]
    if not shown:
        return
    shown = sorted(shown, key=lambda x: int(x.get("num") or 0))[: int(max(1, max_items))]
    html_parts: list[str] = ["<div class='kb-cite-data-wrap' style='display:none'>"]
    for rec in shown:
        n = int(rec.get("num") or 0)
        if n <= 0:
            continue
        anchor = str(rec.get("anchor") or "").strip()
        if not anchor:
            continue
        payload = {
            "num": int(n),
            "source_name": str(rec.get("source_name") or "").strip(),
            "raw": str(rec.get("raw") or "").strip(),
            "cite_fmt": str(rec.get("cite_fmt") or "").strip(),
            "title": str(rec.get("title") or "").strip(),
            "authors": str(rec.get("authors") or "").strip(),
            "venue": str(rec.get("venue") or "").strip(),
            "year": str(rec.get("year") or "").strip(),
            "volume": str(rec.get("volume") or "").strip(),
            "issue": str(rec.get("issue") or "").strip(),
            "pages": str(rec.get("pages") or "").strip(),
            "doi": str(rec.get("doi") or "").strip(),
            "doi_url": str(rec.get("doi_url") or "").strip(),
        }
        payload_s = html.escape(json.dumps(payload, ensure_ascii=False), quote=True)
        html_parts.append(
            "<div class='kb-cite-data' "
            f"data-kb-cite='{html.escape(anchor, quote=True)}' "
            f"data-kb-payload=\"{payload_s}\"></div>"
        )
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def _iter_display_math_blocks(md: str) -> list[tuple[int, int, str]]:
    """
    Return list of (start_line_idx, end_line_idx_exclusive, inner_text) for $$...$$ blocks.
    """
    s = (md or "").replace("\r\n", "\n").replace("\r", "\n")
    if not s.strip():
        return []
    lines = s.split("\n")
    out: list[tuple[int, int, str]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != "$$":
            i += 1
            continue
        j = i + 1
        buf: list[str] = []
        while j < len(lines) and lines[j].strip() != "$$":
            buf.append(lines[j])
            j += 1
        if j < len(lines) and lines[j].strip() == "$$":
            inner = "\n".join(buf).strip()
            out.append((i, j + 1, inner))
            i = j + 1
            continue
        # Unclosed $$, stop scanning
        break
    return out


def _norm_eq_for_match(eq: str) -> str:
    t = str(eq or "")
    if not t:
        return ""
    # Drop tags and comments, normalize whitespace.
    t = _EQ_TAG_RE.sub("", t)
    t = re.sub(r"(?m)%.*$", "", t)
    t = t.replace("\\left", "").replace("\\right", "")
    t = re.sub(r"\s+", "", t)
    return t.strip()


def _best_eq_source_for_tag(
    eq_inner: str,
    tag_n: int,
    hits: list[dict],
) -> tuple[int, str] | None:
    """
    Infer which ref entry (1-based index into hits) this equation likely comes from,
    by matching equation content against snippets in hits.
    Returns (ref_rank, source_label) or None.
    """
    target = _norm_eq_for_match(eq_inner)
    if not target or not hits:
        return None

    best_i = 0
    best_label = ""
    best_score = -1.0

    for i, h in enumerate(hits or [], start=1):
        meta = h.get("meta", {}) or {}
        src = str(meta.get("source_path") or "").strip()
        if not src:
            continue
        label = _display_source_name(src)

        # Candidate snippet texts (small, fast): primary snippet + extra snippets if present
        snips: list[str] = []
        t0 = str(h.get("text") or "").strip()
        if t0:
            snips.append(t0)
        rs = meta.get("ref_snippets")
        if isinstance(rs, list):
            for x in rs[:3]:
                s2 = str(x or "").strip()
                if s2 and s2 not in snips:
                    snips.append(s2)

        # Scan snippets for equations with the same tag number
        for sn in snips:
            for _si, _sj, inner in _iter_display_math_blocks(sn):
                m = _EQ_TAG_RE.search(inner or "")
                if not m:
                    continue
                try:
                    n2 = int(m.group(1))
                except Exception:
                    continue
                if int(n2) != int(tag_n):
                    continue
                cand = _norm_eq_for_match(inner)
                if not cand:
                    continue
                if cand == target:
                    return i, label
                try:
                    sc = difflib.SequenceMatcher(None, target, cand).ratio()
                except Exception:
                    sc = 0.0
                if sc > best_score:
                    best_score = sc
                    best_i = i
                    best_label = label

    if best_i > 0 and best_score >= 0.72:
        return best_i, best_label

    # Fallback: if only one source, assume it's from there.
    if len(hits or []) == 1:
        meta0 = (hits[0] or {}).get("meta", {}) or {}
        src0 = str(meta0.get("source_path") or "").strip()
        if src0:
            return 1, _display_source_name(src0)
    return None


def _annotate_equation_tags_with_sources(md: str, hits: list[dict]) -> str:
    """
    Add a small note under display equations with \\tag{n}:
    '式(n) 来自参考定位 #k: filename'
    """
    s = (md or "").replace("\r\n", "\n").replace("\r", "\n")
    if "$$" not in s or "\\tag{" not in s:
        return md
    lines = s.split("\n")
    blocks = _iter_display_math_blocks(s)
    if not blocks:
        return md

    # Mark block boundaries for quick lookup
    block_by_start: dict[int, tuple[int, str]] = {}
    for si, sj, inner in blocks:
        m = _EQ_TAG_RE.search(inner or "")
        if not m:
            continue
        try:
            n = int(m.group(1))
        except Exception:
            continue
        block_by_start[si] = (n, inner)

    if not block_by_start:
        return md

    out: list[str] = []
    i = 0
    while i < len(lines):
        if i not in block_by_start:
            out.append(lines[i])
            i += 1
            continue

        # Copy the whole $$...$$ block as-is
        # Find its end
        j = i + 1
        out.append(lines[i])
        while j < len(lines):
            out.append(lines[j])
            if lines[j].strip() == "$$":
                break
            j += 1
        # Annotate under it
        tag_n, inner = block_by_start[i]
        picked = _best_eq_source_for_tag(inner, tag_n, hits or [])
        if picked:
            ref_rank, label = picked
            out.append(f"*（式({int(tag_n)}) 来自参考定位 #{int(ref_rank)}：`{label}`，可在下方参考定位点 Open/Page）*")
        out.append("")
        i = j + 1

    return "\n".join(out)


def _extract_inpaper_cite_numbers(text: str, *, min_n: int = 1, max_n: int = 9999) -> list[int]:
    s = str(text or "")
    if not s:
        return []
    out: set[int] = set()
    for m in _INPAPER_CITE_RE.finditer(s):
        try:
            n = int(m.group(1))
        except Exception:
            continue
        if n < int(min_n) or n > int(max_n):
            continue
        out.add(n)
        if len(out) >= 60:
            break
    return sorted(out)


def _parse_int_set(spec: str) -> list[int]:
    """
    Parse: "45,46-49  52" -> [45,46,47,48,49,52]
    """
    s = (spec or "").strip()
    if not s:
        return []
    s = s.replace("，", ",").replace("；", ",").replace(";", ",")
    parts = re.split(r"[,\s]+", s)
    out: set[int] = set()
    for p in parts:
        t = (p or "").strip()
        if not t:
            continue
        t = t.replace("—", "-").replace("–", "-")
        if "-" in t:
            a, b = t.split("-", 1)
            a = a.strip()
            b = b.strip()
            try:
                x = int(a)
                y = int(b)
            except Exception:
                continue
            if x <= 0 or y <= 0:
                continue
            if x > y:
                x, y = y, x
            # keep bounded to avoid accidental huge ranges
            if (y - x) > 500:
                continue
            for k in range(x, y + 1):
                out.add(k)
        else:
            try:
                out.add(int(t))
            except Exception:
                continue
    return sorted(n for n in out if n > 0)


def _read_text_tail(path: Path, *, max_bytes: int = 1_200_000) -> str:
    """
    Read the tail of a text file (references usually at the end).
    Returns UTF-8 decoded text with errors ignored.
    """
    p = Path(path)
    if not p.exists():
        return ""
    try:
        size = int(p.stat().st_size)
    except Exception:
        size = 0
    try:
        if size <= int(max_bytes) or size <= 0:
            return p.read_text(encoding="utf-8", errors="ignore")
        with open(p, "rb") as f:
            f.seek(max(0, size - int(max_bytes)))
            raw = f.read(int(max_bytes))
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _extract_references_map_from_md(md_text: str) -> dict[int, str]:
    return _extract_references_map_from_md_index(md_text)


def _load_references_map_for_source(source_path: str) -> dict[int, str]:
    p = _resolve_source_doc_path(source_path)
    if not p:
        return {}
    # Prefer tail read (references at end), but still works for small docs.
    txt = _read_text_tail(p, max_bytes=1_200_000)
    return _extract_references_map_from_md(txt)


def _render_inpaper_citation_resolver(
    hits: list[dict],
    *,
    assistant_text: str,
    key_ns: str,
) -> None:
    """
    UI helper: resolve in-paper citation numbers like [45] to actual reference entries,
    by reading the converted Markdown source doc and parsing its References section.
    """
    # Candidate sources from retrieved docs
    srcs: list[str] = []
    for h in hits or []:
        meta = h.get("meta", {}) or {}
        sp = str(meta.get("source_path") or "").strip()
        if not sp or _is_temp_source_path(sp):
            continue
        if sp not in srcs:
            srcs.append(sp)
        if len(srcs) >= 20:
            break
    if not srcs:
        return

    cited = _extract_inpaper_cite_numbers(assistant_text or "", min_n=1)
    if not cited:
        return

    st.markdown("---")
    st.markdown("#### 文内引用编号解析（例如 [45]）")
    open_key = f"{key_ns}_inpaper_open"
    is_open = st.checkbox(
        "展开解析（不会影响回答里的 [1][2] 知识库来源编号）",
        value=bool(st.session_state.get(open_key) or False),
        key=open_key,
    )
    if not is_open:
        return

    st.caption("提示：这里解析的是原文里的 [n] 引用编号（例如 [45]），用于反查 References 列表。")

    # Select a source doc to resolve against
    labels = [_display_source_name(s) for s in srcs]
    opts = list(range(len(srcs)))
    default_idx = 0
    sel = st.selectbox(
        "选择要解析的来源文档",
        options=opts,
        format_func=lambda i: labels[int(i)],
        index=default_idx,
        key=f"{key_ns}_inpaper_src_sel",
    )
    try:
        source_path = srcs[int(sel)]
    except Exception:
        source_path = srcs[0]

    default_spec = ",".join(str(x) for x in cited[:20])
    spec = st.text_input(
        "要解析的编号（支持 45,46-49）",
        value=default_spec,
        key=f"{key_ns}_inpaper_spec",
    )
    want_nums = _parse_int_set(spec)[:80]

    use_crossref = st.checkbox(
        "尝试用 Crossref 补全（需要联网，可能较慢）",
        value=False,
        key=f"{key_ns}_inpaper_crossref",
    )

    # Cache by (source_path, mtime) in session_state to avoid repeated file reads across reruns
    cache_key = f"{key_ns}_inpaper_refmap_cache"
    cache = st.session_state.get(cache_key)
    if not isinstance(cache, dict):
        cache = {}
        st.session_state[cache_key] = cache

    mtime = ""
    p = _resolve_source_doc_path(source_path)
    if p:
        try:
            mtime = str(int(p.stat().st_mtime))
        except Exception:
            mtime = ""
    map_key = f"{source_path}|{mtime}"

    refmap = cache.get(map_key)
    if not isinstance(refmap, dict):
        refmap = _load_references_map_for_source(source_path)
        cache[map_key] = refmap

    if not refmap:
        st.warning("未在该文档中检测到 `References/Bibliography` 段落或可解析的 `[n]` 条目。可能该文档还没转换完，或原文未包含参考文献。")
        return

    missing: list[int] = []
    shown = 0
    for n in want_nums:
        entry = refmap.get(int(n))
        if not entry:
            missing.append(int(n))
            continue

        shown += 1
        if not use_crossref:
            st.markdown(f"**[{int(n)}]** {entry}")
            continue

        doi = extract_first_doi(entry)
        if doi:
            meta = fetch_best_crossref_meta(query_title="", doi_hint=doi, allow_title_only=False)
        else:
            meta = None
        if isinstance(meta, dict) and str(meta.get("title") or "").strip():
            title = str(meta.get("title") or "").strip()
            year = str(meta.get("year") or "").strip()
            venue = str(meta.get("venue") or "").strip()
            doi2 = str(meta.get("doi") or doi or "").strip()
            extra = " | ".join(x for x in [year, venue, (f"DOI: {doi2}" if doi2 else "")] if x)
            st.markdown(f"**[{int(n)}]** {title}" + (f"  \n{extra}" if extra else ""))
            st.caption(entry)
        else:
            st.markdown(f"**[{int(n)}]** {entry}")

        if shown >= 40:
            st.caption("（条目较多，已截断显示）")
            break

    if missing:
        miss_s = ", ".join(f"[{x}]" for x in missing[:20])
        st.caption(f"未找到这些编号：{miss_s}" + (" …" if len(missing) > 20 else ""))


def _render_citation_ui(uid: str, source_path: str, key_ns: str) -> None:
    net_key = f"{key_ns}_net_meta_v5_{uid}"
    net_data = st.session_state.get(net_key)
    fetch_failed = bool(st.session_state.get(f"{net_key}_failed", False))

    # Async Check
    if (not net_data) and (not fetch_failed):
        from kb import runtime_state as RUNTIME
        task_id = f"cite_task_{net_key}"
        with RUNTIME.CITATION_LOCK:
            task = RUNTIME.CITATION_TASKS.get(task_id)
        
        if task:
            if task.get("done"):
                # Task finished, sync to session
                res = task.get("result")
                if res:
                    st.session_state[net_key] = res
                    net_data = res
                else:
                    st.session_state[f"{net_key}_failed"] = True
                    fetch_failed = True
                
                # Cleanup task (optional, or keep generic cleaner)
                # with RUNTIME.CITATION_LOCK:
                #    RUNTIME.CITATION_TASKS.pop(task_id, None)
                st.experimental_rerun()
            else:
                # Still running - show minimal loading indicator
                st.markdown(
                    "<div class='citation-loading'>检索中...</div>",
                    unsafe_allow_html=True,
                )
                time.sleep(0.5) 
                st.experimental_rerun()
                return
        else:
             # Should not happen if button clicked, but safety fallback
             pass

    if not isinstance(net_data, dict):
        # Silently fail - don't show error messages
        return

    d_title = str(net_data.get("title") or "").strip()
    d_authors = str(net_data.get("authors") or "").strip() or "[Unknown Authors]"
    d_venue = str(net_data.get("venue") or "").strip() or "Unknown Venue"
    d_year = str(net_data.get("year") or "").strip() or "20xx"
    d_vol = str(net_data.get("volume") or "").strip()
    d_issue = str(net_data.get("issue") or "").strip()
    d_pages = str(net_data.get("pages") or "").strip()
    d_doi = str(net_data.get("doi") or "").strip()
    match_method = str(net_data.get("match_method") or "").strip() or "title"
    match_score = float(net_data.get("match_score") or 0.0)

    if not d_title:
        return

    gbt_suffix = f", {d_year}"
    if d_vol: gbt_suffix += f", {d_vol}"
    if d_issue: gbt_suffix += f"({d_issue})"
    if d_pages: gbt_suffix += f": {d_pages}"
    gbt_str = f"{d_authors}. {d_title} [J]. {d_venue}{gbt_suffix}."

    bib_id = f"ref_{d_year}_{uid[:4]}"
    bib_extras = ""
    if d_vol: bib_extras += f"  volume={{{d_vol}}},\n"
    if d_pages: bib_extras += f"  pages={{{d_pages}}},\n"
    if d_doi: bib_extras += f"  doi={{{d_doi}}},\n"

    bib_str = f"""@article{{{bib_id},
  title={{{d_title}}},
  author={{{d_authors}}},
  journal={{{d_venue}}},
  year={{{d_year}}},
{bib_extras}}}"""

    # Compact citation UI - no extra container, just tabs
    t1, t2 = st.tabs(["GB/T 7714", "BibTeX"])
    with t1:
        st.code(gbt_str, language="text")
    with t2:
        st.code(bib_str, language="latex")
