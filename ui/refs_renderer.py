from __future__ import annotations

import hashlib
import html
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

import streamlit as st

from kb.citation_meta import extract_first_doi, fetch_best_crossref_meta
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
    
    # 1. Infer title
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
    if not show_context:
        st.markdown("<div class='ref-muted-note'>Snippet preview is off.</div>", unsafe_allow_html=True)

    for i, h in enumerate(filtered_hits, start=1):
        meta = h.get("meta", {}) or {}
        source_path = str(meta.get("source_path") or "").strip()
        heading = str(meta.get("top_heading") or _top_heading(str(meta.get("heading_path") or "")) or "").strip()
        page = _safe_page(meta)
        score = float(h.get("score", 0.0) or 0.0)

        # Basic Info
        source_label = _display_source_name(source_path)
        heading_label = _trim_middle(heading, max_len=60) if heading else "Section not tagged"
        score_s = f"{score:.2f}" if score > 0 else "-"
        score_tier = _score_tier(score)

        source_attr = html.escape(source_label, quote=True)
        heading_attr = html.escape(heading_label, quote=True)
        source_html = html.escape(source_label)
        heading_html = html.escape(heading_label)
        page_chip = f"<span class='ref-chip'>p.{int(page)}</span>" if page else ""

        # Render the Reference Card
        st.markdown(
            (
                "<div class='ref-item'>"
                "<div class='ref-item-top'>"
                f"<span class='ref-rank'>#{i}</span>"
                f"<span class='ref-source' title='{source_attr}'>{source_html}</span>"
                f"{page_chip}"
                f"<span class='ref-score ref-score-{score_tier}'>score {score_s}</span>"
                "</div>"
                f"<div class='ref-item-sub' title='{heading_attr}'>{heading_html}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        # Render Snippet
        text = _snippet(str(h.get("text") or ""), heading=heading)
        if show_context and text:
            st.markdown(
                f"<div class='snipbox'><pre>{html.escape(text)}</pre></div>",
                unsafe_allow_html=True,
            )

        pdf_path = _resolve_pdf_for_source(pdf_root, source_path)
        has_pdf = bool(pdf_path)
        # Stable uid: independent from ranking index, so Cite toggle won't require a second click.
        uid = hashlib.sha1(str(source_path).encode("utf-8", "ignore")).hexdigest()[:10]

        # UI: Open | Page | Cite
        cols = st.columns([0.85, 0.85, 0.85, 7.5])

        with cols[0]:
            if st.button("Open", key=f"{key_ns}_open_pdf_{uid}", help="Open PDF", disabled=(not has_pdf)):
                if refs_open_key:
                    st.session_state[refs_open_key] = True
                ok, msg = _open_pdf(pdf_path)
                if not ok:
                    st.warning(msg)

        with cols[1]:
            disabled_page = (page is None) or (not has_pdf)
            if st.button("Page", key=f"{key_ns}_open_page_{uid}", disabled=disabled_page,
                         help=f"Go to page {page}" if page else "Page unknown"):
                if refs_open_key:
                    st.session_state[refs_open_key] = True
                ok, msg = _open_pdf_at(pdf_path, page=page)
                if not ok:
                    st.warning(msg)

        with cols[2]:
            cite_key = f"{key_ns}_cite_visible_{uid}"
            net_key = f"{key_ns}_net_meta_v5_{uid}"  # v5 cache key
            is_cite_open = st.session_state.get(cite_key, False)
            btn_label = "Close" if is_cite_open else "Cite"

            # Use CALLBACK to avoid UI flicker/rerun issues
            st.button(
                btn_label,
                key=f"{key_ns}_cite_btn_{uid}",
                help="Fetch citation (Auto-fetch from Crossref)",
                on_click=_on_cite_click,
                args=(cite_key, net_key, source_path, refs_open_key)
            )

        # Render Citation UI (if open)
        if st.session_state.get(cite_key, False):
            _render_citation_ui(uid, source_path, key_ns)
        elif not has_pdf:
            st.caption("未定位到对应 PDF（可继续使用 Cite）。")

        st.markdown("<div class='ref-item-gap'></div>", unsafe_allow_html=True)


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
                # Still running
                with st.container():
                     st.markdown(
                        "<div style='background:rgba(128,128,128,0.06); padding:10px; border-radius:8px; margin-top:5px; margin-bottom:10px; border:1px solid rgba(128,128,128,0.15);'>"
                        "<div style='margin-bottom:8px; font-weight:600; font-size:0.9em; color:#666;'>Citation Export</div>"
                        "<div style='font-size:0.9em; color:#666;'>正在联网检索元数据 (Crossref)...</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                     # Poll again shortly
                     time.sleep(0.5) 
                     st.experimental_rerun()
                return
        else:
             # Should not happen if button clicked, but safety fallback
             pass

    if not isinstance(net_data, dict):
        with st.container():
            st.markdown(
                "<div style='background:rgba(128,128,128,0.06); padding:10px; border-radius:8px; margin-top:5px; margin-bottom:10px; border:1px solid rgba(128,128,128,0.15);'>"
                "<div style='margin-bottom:8px; font-weight:600; font-size:0.9em;'>Citation Export</div>",
                unsafe_allow_html=True,
            )
            if fetch_failed:
                st.info("未识别出可靠引用信息。为保证准确性，此条不自动生成 Cite。")
            else:
                st.info("点击 Cite 后未获得可用元数据。")
            st.markdown("</div>", unsafe_allow_html=True)
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
        with st.container():
            st.markdown(
                "<div style='background:rgba(128,128,128,0.06); padding:10px; border-radius:8px; margin-top:5px; margin-bottom:10px; border:1px solid rgba(128,128,128,0.15);'>"
                "<div style='margin-bottom:8px; font-weight:600; font-size:0.9em;'>Citation Export</div>",
                unsafe_allow_html=True,
            )
            st.info("未识别出可靠标题。为保证准确性，此条不自动生成 Cite。")
            st.markdown("</div>", unsafe_allow_html=True)
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

    with st.container():
        st.markdown(
            "<div style='background:rgba(128,128,128,0.06); padding:10px; border-radius:8px; margin-top:5px; margin-bottom:10px; border:1px solid rgba(128,128,128,0.15);'>"
            "<div style='margin-bottom:8px; font-weight:600; font-size:0.9em;'>Citation Export</div>",
            unsafe_allow_html=True
        )
        st.caption(f"Source: Crossref ({match_method}, confidence {match_score:.2f})")


        t1, t2 = st.tabs(["GB/T 7714", "BibTeX"])
        with t1:
            st.code(gbt_str, language="text")
        with t2:
            st.code(bib_str, language="latex")

        st.markdown("</div>", unsafe_allow_html=True)
