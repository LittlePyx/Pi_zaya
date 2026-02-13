from __future__ import annotations

import hashlib
import html
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

import streamlit as st

from kb.citation_meta import extract_first_doi, fetch_best_crossref_meta
from kb.pdf_tools import open_in_explorer
from ui.strings import S


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
    return None


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

    return fetch_best_crossref_meta(
        query_title=q,
        expected_year=expected_year,
        expected_venue=expected_venue,
        doi_hint=doi_hint,
        min_score=0.90,
    )


def _parse_filename_meta(path_str: str) -> tuple[str, str, str]:
    name = Path(path_str).stem
    if name.lower().endswith(".en"):
        name = name[:-3]
    m = re.match(r"^([^-]+)\s*-\s*(19\d{2}|20\d{2})\s*-\s*(.+)$", name)
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    return "", "", name


# --- Callback Function (THE FIX) ---
def _on_cite_click(cite_key: str, net_key: str, source_path: str):
    """
    This runs BEFORE the UI re-renders.
    It fetches data immediately, updates session_state,
    so the UI just wakes up with the data ready.
    """
    # 1. Toggle visibility
    new_state = not st.session_state.get(cite_key, False)
    st.session_state[cite_key] = new_state

    # 2. If opening, fetch data if missing
    if new_state:
        if net_key not in st.session_state:
            # Infer title
            l_venue, l_year, l_title = _parse_filename_meta(source_path)
            search_title = l_title if l_title else Path(source_path).stem
            search_title = _infer_title_from_source_text(source_path, search_title)

            # Fetch (This will block for ~1-2s, user feels a slight wait)
            found = fetch_crossref_meta(
                search_title,
                source_path=source_path,
                expected_venue=l_venue,
                expected_year=l_year,
            )
            if found:
                st.session_state[net_key] = found
                st.session_state.pop(f"{net_key}_failed", None)
            else:
                st.session_state.pop(net_key, None)
                st.session_state[f"{net_key}_failed"] = True


def _render_refs(
        hits: list[dict],
        *,
        prompt: str = "",
        show_heading: bool = True,
        key_ns: str = "refs",
        settings=None,
) -> None:
    del prompt, settings  # backward compatibility

    if show_heading:
        st.markdown(f"### {S['refs']}")
    if not hits:
        st.markdown(f"<div class='refbox'>{S['kb_miss']}</div>", unsafe_allow_html=True)
        return

    pdf_root_str = str(st.session_state.get("pdf_dir") or "").strip()
    pdf_root = Path(pdf_root_str) if pdf_root_str else None
    show_context = bool(st.session_state.get("show_context") or False)
    if not show_context:
        st.markdown("<div class='ref-muted-note'>Snippet preview is off.</div>", unsafe_allow_html=True)

    for i, h in enumerate(hits, start=1):
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
        uid = hashlib.sha1((str(pdf_path) + "|" + str(i)).encode("utf-8", "ignore")).hexdigest()[:10]

        if not pdf_path:
            st.markdown("<div class='ref-item-gap'></div>", unsafe_allow_html=True)
            continue

        # UI: Open | Page | Cite
        cols = st.columns([0.85, 0.85, 0.85, 7.5])

        with cols[0]:
            if st.button("Open", key=f"{key_ns}_open_pdf_{uid}", help="Open PDF"):
                ok, msg = _open_pdf(pdf_path)
                if not ok:
                    st.warning(msg)

        with cols[1]:
            disabled_page = (page is None)
            if st.button("Page", key=f"{key_ns}_open_page_{uid}", disabled=disabled_page,
                         help=f"Go to page {page}" if page else "Page unknown"):
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
                args=(cite_key, net_key, source_path)
            )

        # Render Citation UI (if open)
        if st.session_state.get(cite_key, False):
            _render_citation_ui(uid, source_path, key_ns)

        st.markdown("<div class='ref-item-gap'></div>", unsafe_allow_html=True)


def _render_citation_ui(uid: str, source_path: str, key_ns: str) -> None:
    net_key = f"{key_ns}_net_meta_v5_{uid}"
    net_data = st.session_state.get(net_key)
    fetch_failed = bool(st.session_state.get(f"{net_key}_failed", False))

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
