from __future__ import annotations

import hashlib
import html
import logging
import os
import re
import subprocess
import time
import difflib
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import streamlit as st
import requests

from kb.citation_meta import extract_first_doi, fetch_best_crossref_meta
from kb.config import load_settings
from kb.file_naming import citation_meta_display_pdf_name
from kb.inpaper_citation_grounding import (
    extract_citation_context_hints,
    has_explicit_reference_conflict,
    reference_alignment_score,
)
from kb.source_filters import is_excluded_source_path
from kb.reference_index import (
    extract_references_map_from_md as _extract_references_map_from_md_index,
    load_reference_index as _load_reference_index_file,
    resolve_reference_entry as _resolve_reference_entry_from_index,
)
from kb.pdf_tools import open_in_explorer
from kb.tokenize import tokenize
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
    src = str(source_path or "").strip()
    if not src:
        return "unknown"
    try:
        if bool(getattr(st, "_is_running_with_streamlit", False)):
            pdf_root_str = str(st.session_state.get("pdf_dir") or "").strip()
            pdf_root = Path(pdf_root_str) if pdf_root_str else None
            pdf_path = _resolve_pdf_for_source(pdf_root, src) if pdf_root else None
            lib_store = st.session_state.get("lib_store")
            if (pdf_path is not None) and hasattr(lib_store, "get_citation_meta"):
                meta = lib_store.get_citation_meta(pdf_path)  # type: ignore[attr-defined]
                full_name = citation_meta_display_pdf_name(meta)
                if full_name:
                    return full_name
    except Exception:
        pass
    name = Path(src).name or src
    low = name.lower()
    if low.endswith(".en.md"):
        name = name[:-6] + ".pdf"
    elif low.endswith(".md"):
        name = name[:-3] + ".pdf"
    return name or "unknown.pdf"


def _is_temp_source_path(source_path: str) -> bool:
    s = (source_path or "").strip()
    if not s:
        return True
    if is_excluded_source_path(s):
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
    for x in [meta.get("page"), meta.get("page_num"), meta.get("page_idx"), meta.get("page_start"), meta.get("page_end")]:
        try:
            p = int(x)
        except Exception:
            continue
        if p > 0:
            return p
    return None


def _safe_page_range(meta: dict) -> tuple[int | None, int | None]:
    def _to_pos_int(x) -> int | None:
        try:
            v = int(x)
        except Exception:
            return None
        return v if v > 0 else None

    p0 = _to_pos_int(meta.get("page_start")) or _safe_page(meta)
    p1 = _to_pos_int(meta.get("page_end")) or p0
    if (p0 is not None) and (p1 is not None) and p1 < p0:
        p0, p1 = p1, p0
    return p0, p1


def _score_tier(score: float) -> str:
    if score >= 8.0:
        return "hi"
    if score >= 4.0:
        return "mid"
    return "low"


def _split_section_subsection(heading_path: str) -> tuple[str, str]:
    hp = " / ".join([p.strip() for p in str(heading_path or "").split(" / ") if p.strip()])
    if not hp:
        return "", ""
    parts = [p.strip() for p in hp.split(" / ") if p.strip()]
    if not parts:
        return "", ""
    return parts[0], " / ".join(parts[1:]).strip()


_REF_HEADING_RE_UI = re.compile(
    r"\b(references?|bibliography|works?\s+cited|citation|acknowledg(e)?ments?|appendi(?:x|ces)|supplementary)\b",
    flags=re.I,
)
_VENUE_HEAD_TOKENS_UI = {
    "nature",
    "science",
    "ieee",
    "acm",
    "cvpr",
    "iccv",
    "eccv",
    "neurips",
    "icml",
    "ijcai",
    "aaai",
    "conference",
    "proceedings",
    "journal",
    "transactions",
    "letters",
    "communication",
    "communications",
    "photonics",
    "optics",
    "review",
    "advances",
    "arxiv",
}
_VENUE_JOIN_TOKENS_UI = {"of", "on", "for", "and", "the", "in", "&"}
_SECTION_WORDS_UI = {
    "abstract",
    "introduction",
    "background",
    "related",
    "work",
    "method",
    "methods",
    "approach",
    "model",
    "setup",
    "experiment",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "implementation",
    "evaluation",
    "analysis",
}
_METHOD_QUERY_RE_UI = re.compile(
    r"(鎬庝箞|濡備綍|鏂规硶|瀹炵幇|姝ラ|娴佺▼|鍘熺悊|鏈哄埗|绠楁硶|妯″瀷|鍏紡|鎺ㄥ|"
    r"\bhow\b|\bmethod\b|\bapproach\b|\bimplement(?:ation)?\b|\balgorithm\b|\bmodel\b|\bequation\b)",
    flags=re.I,
)
_LIMIT_QUERY_RE_UI = re.compile(
    r"(灞€闄恷闄愬埗|涓嶈冻|鏈潵宸ヤ綔|璁ㄨ|缁撹|"
    r"\blimitation\b|\bfuture\s+work\b|\bdiscussion\b|\bconclusion\b)",
    flags=re.I,
)
_DISCUSS_HEAD_RE_UI = re.compile(
    r"\b(discussion|conclusion|limitations?|future\s+work)\b|(璁ㄨ|缁撹|灞€闄恷鏈潵宸ヤ綔)",
    flags=re.I,
)


def _wants_reference_nav_ui(prompt: str) -> bool:
    q = str(prompt or "").strip()
    if not q:
        return False
    return bool(re.search(r"(鍙傝€冩枃鐚畖寮曠敤|cite|citation|reference|bibliography)", q, flags=re.I))


def _is_reference_heading_ui(h: str) -> bool:
    s = str(h or "").strip()
    return bool(_REF_HEADING_RE_UI.search(s))


def _is_venue_heading_ui(h: str) -> bool:
    s = " ".join(str(h or "").strip().split())
    if not s:
        return False
    low = s.lower()
    toks = re.findall(r"[a-z][a-z0-9.+-]*", low)
    if not toks:
        return False
    if any(t in _SECTION_WORDS_UI for t in toks):
        return False
    venue_hit = any(t in _VENUE_HEAD_TOKENS_UI for t in toks)
    if (len(toks) <= 6) and venue_hit and all((t in _VENUE_HEAD_TOKENS_UI or t in _VENUE_JOIN_TOKENS_UI) for t in toks):
        return True
    letters = re.sub(r"[^A-Za-z]", "", s)
    if letters and (letters == letters.upper()) and (len(toks) <= 5) and venue_hit:
        return True
    return False


def _looks_like_doc_title_heading_ui(h: str, source_path: str) -> bool:
    hh = " ".join(str(h or "").strip().split())
    src = str(source_path or "").strip()
    if (not hh) or (not src):
        return False
    low_h = re.sub(r"[^a-z0-9]+", " ", hh.lower()).strip()
    if len(low_h) < 24:
        return False
    stem = Path(src).stem
    stem = re.sub(r"(19|20)\d{2}", " ", stem)
    stem = re.sub(r"[_\-]+", " ", stem)
    low_s = re.sub(r"[^a-z0-9]+", " ", stem.lower()).strip()
    if not low_s:
        return False
    if low_h in low_s:
        return True
    h_toks = [t for t in low_h.split() if len(t) >= 3]
    s_toks = [t for t in low_s.split() if len(t) >= 3]
    if len(h_toks) < 3 or len(s_toks) < 3:
        return False
    hs = set(h_toks)
    ss = set(s_toks)
    inter = hs & ss
    if len(inter) < 3:
        return False
    return (len(inter) / max(1, len(hs))) >= 0.66


def _is_non_navigational_heading_ui(h: str, *, prompt: str, source_path: str) -> bool:
    s = " ".join(str(h or "").strip().split())
    if not s:
        return True
    if _is_venue_heading_ui(s):
        return True
    if (not _wants_reference_nav_ui(prompt)) and _is_reference_heading_ui(s):
        return True
    return False


def _should_avoid_discussion_ui(prompt: str) -> bool:
    q = str(prompt or "").strip()
    if not q:
        return True
    if _wants_reference_nav_ui(q):
        return False
    if _LIMIT_QUERY_RE_UI.search(q):
        return False
    return True


def _is_discussion_heading_ui(h: str) -> bool:
    s = " ".join(str(h or "").strip().split())
    if not s:
        return False
    return bool(_DISCUSS_HEAD_RE_UI.search(s))


def _looks_like_structured_section_heading_ui(h: str) -> bool:
    s = " ".join(str(h or "").strip().split())
    if not s:
        return False
    low = s.lower()
    if re.match(r"^\d+(\.\d+){0,3}\b", low):
        return True
    if re.match(r"^(section|sec\.?|chapter|part|appendix)\b", low):
        return True
    return bool(re.match(r"^[ivxlcdm]+\.\s+", low))


def _sanitize_heading_path_ui(hp: str, *, prompt: str, source_path: str) -> str:
    parts = [p.strip() for p in str(hp or "").split(" / ") if p.strip()]
    if not parts:
        return ""
    keep: list[str] = []
    for p in parts:
        p2 = " ".join(p.split())
        if _is_non_navigational_heading_ui(p2, prompt=prompt, source_path=source_path):
            continue
        if keep and keep[-1].lower() == p2.lower():
            continue
        keep.append(p2)
    if len(keep) >= 2:
        first = keep[0]
        second = keep[1]
        if (
            len(first) >= 36
            and _looks_like_structured_section_heading_ui(second)
            and (not _looks_like_structured_section_heading_ui(first))
        ):
            keep = keep[1:]
    if keep and _looks_like_doc_title_heading_ui(keep[0], source_path):
        keep = keep[1:] if len(keep) >= 2 else []
    return " / ".join(keep[:3]) if keep else ""


_GENERIC_HINT_PATTERNS_UI = (
    "this paper provides information related to the question",
    "directly relevant information points",
    "content relevant to the current question",
    "evidence useful for answering the question",
    "information related to the question",
    "directly relevant information points",
    "evidence for the current question",
)
_ANCHOR_STOPWORDS_UI = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "at",
    "by",
    "as",
    "is",
    "are",
    "be",
    "this",
    "that",
    "these",
    "those",
    "method",
    "methods",
    "approach",
    "approaches",
    "model",
    "models",
    "algorithm",
    "algorithms",
    "experiment",
    "experiments",
    "evaluation",
    "analysis",
    "result",
    "results",
    "problem",
    "problems",
    "challenge",
    "challenges",
    "constraint",
    "constraints",
    "bottleneck",
    "bottlenecks",
    "metric",
    "metrics",
    "performance",
    "paper",
    "study",
    "work",
    "section",
    "sections",
    "introduction",
    "background",
    "discussion",
    "conclusion",
    "conclusions",
    "data",
    "dataset",
    "datasets",
    "figure",
    "table",
    "supplementary",
    "appendix",
    "鏂囩尞",
    "璁烘枃",
    "鐮旂┒",
    "闂",
    "鎸戞垬",
    "鐡堕",
    "绾︽潫",
    "鐩稿叧",
    "淇℃伅",
    "鍐呭",
    "绔犺妭",
    "灏忚妭",
    "鏂规硶",
    "缁撴灉",
    "瀹為獙",
    "妯″瀷",
    "绠楁硶",
    "鏁版嵁",
    "with",
    "using",
    "use",
    "used",
    "based",
    "via",
    "from",
    "into",
    "over",
    "under",
    "through",
    "between",
    "across",
    "improve",
    "improves",
    "improved",
    "enable",
    "enables",
    "provide",
    "provides",
    "proposed",
    "propose",
}


def _looks_generic_guidance_ui(text: str) -> bool:
    s = " ".join(str(text or "").strip().split())
    if not s:
        return True
    low = s.lower()
    if any(k in low for k in _GENERIC_HINT_PATTERNS_UI):
        return True
    toks = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", low) if t not in _ANCHOR_STOPWORDS_UI]
    return len(set(toks)) <= 2 and len(s) <= 80


def _looks_keyword_list_ui(text: str) -> bool:
    s = " ".join(str(text or "").strip().split())
    if not s:
        return True
    if len(s) <= 90 and (s.count(",") + s.count("，") + s.count(";") + s.count("；")) >= 2:
        return True
    low = s.lower()
    verb_markers = (
        "鎻愬嚭",
        "閲囩敤",
        "閫氳繃",
        "瀹炵幇",
        "鎻愬崌",
        "楠岃瘉",
        "propose",
        "use",
        "introduce",
        "achieve",
        "improve",
        "show",
    )
    return not any(v in low for v in verb_markers)


def _contains_question_echo_ui(text: str, prompt: str) -> bool:
    t = " ".join(str(text or "").strip().split()).lower()
    q = " ".join(str(prompt or "").strip().split()).lower()
    if not t or not q:
        return False
    q_compact = re.sub(r"[\s`'\"鈥溾€濃€樷€欙紝銆傦紒锛?.?!:;锛涳細()锛堬級\-_/\\]+", "", q)
    t_compact = re.sub(r"[\s`'\"鈥溾€濃€樷€欙紝銆傦紒锛?.?!:;锛涳細()锛堬級\-_/\\]+", "", t)
    if len(q_compact) < 10:
        return False
    for n in (24, 18, 14):
        if len(q_compact) < n:
            continue
        max_start = min(len(q_compact) - n, 28)
        for s in range(max_start + 1):
            chunk = q_compact[s : s + n]
            if chunk and (chunk in t_compact):
                return True
    return False


def _too_similar_text_ui(a: str, b: str) -> bool:
    aa = " ".join(str(a or "").strip().split()).lower()
    bb = " ".join(str(b or "").strip().split()).lower()
    if not aa or not bb:
        return False
    if aa == bb:
        return True
    if (aa in bb or bb in aa) and min(len(aa), len(bb)) >= 18:
        return True
    try:
        return difflib.SequenceMatcher(None, aa, bb).ratio() >= 0.88
    except Exception:
        return False


def _looks_template_artifact_ui(text: str) -> bool:
    s = " ".join(str(text or "").strip().split())
    if not s:
        return False
    low = s.lower()
    # Common templated hint sentences that should not be used as anchors.
    if s.startswith("该文"):
        if ("方法中" in s) and ("实验中" in s):
            return True
    if s.startswith("可直接支撑提问的证据主要位于"):
        return True
    if ("目标任务" in s) or ("相关结果上有可核查提取" in s):
        return True
    if ("evidence is concentrated in" in low) and ("key points on" in low):
        return True
    return False


def _extract_anchor_terms_ui(meta: dict, *, prompt: str = "", max_n: int = 4) -> list[str]:
    if not isinstance(meta, dict):
        return []
    texts: list[str] = []
    for s in (meta.get("ref_show_snippets") or [])[:3]:
        s2 = " ".join(str(s or "").strip().split())
        if s2:
            texts.append(s2)
    for loc in (meta.get("ref_locs") or [])[:3]:
        if not isinstance(loc, dict):
            continue
        hp = str(loc.get("heading_path") or loc.get("heading") or "").strip()
        if hp:
            texts.append(hp)
    if not texts:
        s0 = " ".join(str(meta.get("text") or "").strip().split())
        if s0:
            texts.append(s0)
    all_text = "\n".join(texts)
    if not all_text:
        return []

    q_toks = set(tokenize(str(prompt or "").lower()))
    scores: dict[str, float] = {}

    def _bump(term: str, w: float) -> None:
        t = str(term or "").strip()
        if not t:
            return
        k = t.lower()
        if k in _ANCHOR_STOPWORDS_UI:
            return
        if len(k) <= 2:
            return
        if k in q_toks and len(k) <= 5:
            return
        ww = float(w)
        # Generic down-weighting for very short/common-looking terms.
        if len(k) <= 4 and re.fullmatch(r"[a-z]+", k):
            ww *= 0.65
        # Acronyms are useful but short all-caps tokens are often weak anchors.
        if re.fullmatch(r"[A-Z]{2,5}", t):
            ww *= 0.62
        if ww <= 0.0:
            return
        scores[t] = float(scores.get(t, 0.0) + ww)

    for ab in re.findall(r"\b[A-Z]{2,10}\b", all_text):
        _bump(ab, 3.0)

    for hy in re.findall(r"\b[A-Za-z]{3,}(?:-[A-Za-z0-9]{2,})+\b", all_text):
        _bump(hy, 2.0)

    for phr in re.findall(r"\b[A-Za-z][A-Za-z0-9\-]{2,}(?:\s+[A-Za-z][A-Za-z0-9\-]{2,}){1,4}\b", all_text):
        low = phr.lower()
        if len(phr) > 56:
            continue
        if any(
            bad in low
            for bad in (
                "quantitative",
                "comparison",
                "comparisons",
                "table",
                "fig",
                "result and analysis",
                "introduction",
                "conclusion",
                "datasets",
            )
        ):
            continue
        # Keep phrase candidates using generic shape cues instead of domain keywords.
        if re.search(r"[A-Z]{2,}", phr) or re.search(r"\d", phr) or ("-" in phr):
            _bump(phr, 2.6)
        elif len(phr) >= 18:
            _bump(phr, 1.4)

    for w in re.findall(r"\b[A-Za-z][A-Za-z0-9]{3,}\b", all_text):
        wl = w.lower()
        if wl in _ANCHOR_STOPWORDS_UI:
            continue
        if wl.endswith("tion") or wl.endswith("ing") or wl.endswith("ment"):
            _bump(w, 0.8)
        else:
            _bump(w, 1.0)

    for zh in re.findall(r"[\u4e00-\u9fff]{2,8}", all_text):
        if zh in {"杩欑瘒鏂囩尞", "褰撳墠闂", "鐩稿叧淇℃伅"}:
            continue
        _bump(zh, 1.4)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out: list[str] = []
    seen_low: set[str] = set()
    for t, _s in ranked:
        k = t.lower()
        if k in seen_low:
            continue
        if any((k in ex) or (ex in k) for ex in seen_low if len(ex) >= 4):
            continue
        seen_low.add(k)
        out.append(t)
        if len(out) >= int(max_n):
            break
    return out


def _has_cjk_text_ui(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def _looks_latin_heavy_ui(text: str) -> bool:
    s = str(text or "")
    if not s.strip():
        return False
    n_cjk = len(re.findall(r"[\u4e00-\u9fff]", s))
    n_lat = len(re.findall(r"[A-Za-z]", s))
    return (n_lat >= 18) and (n_lat >= (2 * n_cjk + 8))


def _clean_sentence_candidate_ui(text: str) -> str:
    s = " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())
    if not s:
        return ""
    s = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", s)
    s = re.sub(r"`{1,3}", "", s)
    s = re.sub(r"^#{1,6}\s*", "", s)
    s = s.replace("|", " ")
    s = re.sub(r"\$?\^\{\s*\[\s*\d[^]]{0,60}\]\s*\}\$?", " ", s)
    s = re.sub(r"\[\s*\d{1,4}(?:\s*[,;\-–—]\s*\d{1,4})*\s*\]", " ", s)
    s = s.replace("**", " ").replace("*", " ")
    s = re.sub(r"\s{2,}", " ", s).strip(" \t\r\n-–—，。；：")
    return s


def _looks_noisy_sentence_ui(text: str) -> bool:
    s = " ".join(str(text or "").strip().split())
    if not s:
        return True
    low = s.lower()
    if len(s) < 10:
        return True
    if ("http://" in low) or ("https://" in low):
        return True
    if re.search(r"(equal contribution|corresponding author|all rights reserved)", low):
        return True
    if s.count("|") >= 2:
        return True
    if re.fullmatch(r"[^\w\u4e00-\u9fff]{3,}", s):
        return True
    # Chunk-boundary fragments are common in OCR/MD conversion.
    if s.endswith("...") or s.endswith("…"):
        return True
    if re.match(r"^[a-z]{1,4}\b", s) and len(s) < 28:
        return True
    sym_n = len(re.findall(r"[^0-9A-Za-z\u4e00-\u9fff\s]", s))
    if sym_n > max(14, int(len(s) * 0.28)):
        return True
    return False


def _explode_find_terms_ui(text: str, *, max_n: int = 6) -> list[str]:
    raw = _clean_sentence_candidate_ui(text)
    if not raw:
        return []
    seeds = [raw]
    if raw.count("|") >= 2:
        cells = [c.strip() for c in raw.split("|") if c.strip()]
        if cells:
            seeds = cells
    out: list[str] = []
    seen: set[str] = set()
    for seg in seeds:
        seg2 = re.sub(r"\s+", " ", seg).strip(" ,，；。")
        if not seg2:
            continue
        parts = re.split(r"[，;；。]", seg2)
        for p in parts:
            t = _clean_sentence_candidate_ui(p)
            if not t:
                continue
            if len(t) <= 1:
                continue
            if len(t) > 56:
                continue
            k = t.lower()
            if k in seen:
                continue
            if k in _ANCHOR_STOPWORDS_UI:
                continue
            if re.fullmatch(r"\d+(?:\.\d+)?", k):
                continue
            if re.search(r"\b(table|figure|fig|supplementary|appendix)\b", k):
                continue
            if re.search(r"\b(quantitative|comparison|comparisons|result and analysis|introduction|conclusion|dataset|datasets)\b", k):
                continue
            seen.add(k)
            out.append(t)
            if len(out) >= int(max_n):
                return out
    return out


def _anchor_specificity_score_ui(term: str) -> float:
    t = " ".join(str(term or "").strip().split())
    if not t:
        return -1e9
    low = t.lower()
    score = 0.0
    if ("-" in t) or (" " in t):
        score += 2.3
    if re.search(r"\d", t):
        score += 1.7
    if re.search(r"[A-Z]{2,}", t):
        score += 1.4
    if len(t) >= 10:
        score += 1.0
    if re.fullmatch(r"[A-Z]{2,5}", t):
        score -= 1.4
    if len(t) > 40:
        score -= 1.2
    if re.search(r"\b(quantitative|comparison|comparisons|result and analysis|introduction|conclusion|dataset|datasets)\b", low):
        score -= 2.4
    if low in _ANCHOR_STOPWORDS_UI:
        score -= 2.6
    if re.fullmatch(r"[a-z]+", low) and len(low) <= 6:
        score -= 0.8
    return score


def _loc_phrase_ui(*, sec: str, meta: dict, cjk: bool) -> str:
    sec_s = str(sec or "").strip()
    p0, p1 = _safe_page_range(meta if isinstance(meta, dict) else {})
    if cjk:
        if p0 and p1 and p1 > p0:
            page_s = f"第{int(p0)}-{int(p1)}页"
        elif p0:
            page_s = f"第{int(p0)}页"
        else:
            page_s = ""
        if sec_s and page_s:
            return f"`{sec_s}`（{page_s}）"
        if sec_s:
            return f"`{sec_s}`"
        if page_s:
            return page_s
        return "正文命中段落"
    else:
        if p0 and p1 and p1 > p0:
            page_s = f"pp.{int(p0)}-{int(p1)}"
        elif p0:
            page_s = f"p.{int(p0)}"
        else:
            page_s = ""
        if sec_s and page_s:
            return f"`{sec_s}` ({page_s})"
        if sec_s:
            return f"`{sec_s}`"
        if page_s:
            return page_s
        return "the matched body paragraphs"


def _pick_term_from_sentence_ui(sentence: str, terms: list[str]) -> str:
    low = str(sentence or "").lower()
    for pat in (
        r"\b([A-Za-z]{3,}(?:-[A-Za-z0-9]{2,})+)\b",
        r"\b([A-Z]{2,}[A-Za-z0-9\-]{1,})\b",
        r"\b([A-Za-z][A-Za-z0-9]{3,}\s+[A-Za-z][A-Za-z0-9]{3,}(?:\s+[A-Za-z][A-Za-z0-9]{3,})?)\b",
    ):
        m0 = re.search(pat, low)
        if m0:
            return " ".join(m0.group(1).split())
    for t in terms or []:
        tt = str(t or "").strip()
        if not tt:
            continue
        if tt.lower() in low:
            return tt
    for t in terms or []:
        tt = str(t or "").strip()
        if not tt:
            continue
        if len(tt) <= 2:
            continue
        if re.fullmatch(r"[A-Z]{2,5}", tt):
            continue
        return tt
    return ""


def _pick_model_name_ui(sentence: str, terms: list[str]) -> str:
    stop = {
        "however",
        "therefore",
        "additionally",
        "results",
        "result",
        "method",
        "methods",
        "quantitative",
        "table",
        "figure",
        "analysis",
        "introduction",
        "conclusion",
        "discussion",
    }
    for t in terms or []:
        tt = str(t or "").strip()
        if not tt:
            continue
        if not re.search(r"[A-Z]{2,}", tt):
            continue
        if tt.lower() in stop:
            continue
        if re.fullmatch(r"[A-Z]{2,5}", tt):
            continue
        if " " in tt:
            toks = re.findall(r"\b[A-Z]{2,}[A-Za-z0-9\-]{1,}\b", tt)
            for tok in toks:
                if re.fullmatch(r"[A-Z]{2,5}", tok):
                    continue
                if tok.lower() not in stop:
                    return tok
            continue
        return tt
    for m in re.findall(r"\b[A-Z]{2,}[A-Za-z0-9\-]{1,}\b", str(sentence or "")):
        low = m.lower()
        if low in stop:
            continue
        if re.fullmatch(r"[A-Z]{2,5}", m):
            continue
        return m
    return ""


def _display_focus_term_ui(term: str) -> str:
    t = " ".join(str(term or "").strip().split())
    if not t:
        return ""
    low = t.lower()
    if low.startswith("the "):
        t = t[4:].strip()
        low = t.lower()
    if re.search(r"\b(table|figure|fig|section|chapter|appendix|supplementary)\b", low):
        return ""
    if len(t) > 36:
        t = t[:36].rstrip() + "..."
    return t


def _compress_evidence_clause_ui(
    sentence: str,
    *,
    cjk: bool,
    role: str,
    terms: list[str],
    max_chars: int,
) -> str:
    s_raw = _clean_sentence_candidate_ui(sentence)
    if not s_raw:
        return ""
    if (not cjk) or _has_cjk_text_ui(s_raw):
        return _trim_clause_ui(s_raw, max_chars=max_chars)
    if not _looks_latin_heavy_ui(s_raw):
        return _trim_clause_ui(s_raw, max_chars=max_chars)

    low = s_raw.lower()
    term = _pick_term_from_sentence_ui(s_raw, terms)
    term_disp = _display_focus_term_ui(term)
    model = _pick_model_name_ui(s_raw, terms)

    if role == "problem":
        if re.search(r"(struggle|limitation|limited|bottleneck|difficult|lack|challenge|suboptimal|incompetent|poor|not\s+outperform|did\s+not\s+outperform)", low):
            return _trim_clause_ui(f"指出现有方法在 {term_disp or '目标任务'} 上仍有明显局限", max_chars=max_chars)
        return _trim_clause_ui(f"围绕 {term_disp or '目标任务'} 提炼了待解决的关键问题", max_chars=max_chars)

    if role == "method":
        if re.search(r"(propos|introduc|develop|design|adopt|utiliz|construct|build)", low):
            if model and term and (model.lower() != term.lower()):
                return _trim_clause_ui(f"提出 {model}，并围绕 {term_disp or term} 给出实现路径", max_chars=max_chars)
            if model:
                return _trim_clause_ui(f"提出 {model} 并给出可复现的实现流程", max_chars=max_chars)
            if term:
                return _trim_clause_ui(f"提出并采用围绕 {term_disp or term} 的方法设计", max_chars=max_chars)
            return _trim_clause_ui("提出了具体的方法设计与实现流程", max_chars=max_chars)
        if term:
            return _trim_clause_ui(f"围绕 {term_disp or term} 给出实现细节", max_chars=max_chars)
        return _trim_clause_ui("给出具体方法与实现细节", max_chars=max_chars)

    if role == "result":
        if re.search(r"(outperform|improv|superior|better|achieve|show|demonstrate|experiment|results?)", low):
            if term:
                return _trim_clause_ui(f"实验显示在 {term_disp or term} 相关结果上有可核查提升", max_chars=max_chars)
            return _trim_clause_ui("实验结果显示相对现有方法有可核查提升", max_chars=max_chars)
        if term:
            return _trim_clause_ui(f"报告了围绕 {term_disp or term} 的可核查结果", max_chars=max_chars)
        return _trim_clause_ui("报告了可核查的实验结果", max_chars=max_chars)

    # relevance/evidence fallback
    if re.search(r"(did\s+not\s+outperform|not\s+outperform)", low):
        if model:
            return _trim_clause_ui(f"原文指出 {model} 在部分场景仍存在性能短板", max_chars=max_chars)
        return _trim_clause_ui("原文指出该方法在部分场景仍存在性能短板", max_chars=max_chars)
    if re.search(r"(outperform|superior|better)", low):
        if model:
            return _trim_clause_ui(f"原文报告 {model} 在对比实验中取得更优结果", max_chars=max_chars)
        return _trim_clause_ui("原文报告该方法在对比实验中取得更优结果", max_chars=max_chars)
    if re.search(r"(show|demonstrate|evidence|support|indicate|experiment|results?)", low):
        if term:
            return _trim_clause_ui(f"原文在 {term_disp or term} 相关内容上给出直接证据", max_chars=max_chars)
        return _trim_clause_ui("原文给出可直接用于回答提问的证据", max_chars=max_chars)
    if term:
        return _trim_clause_ui(f"围绕 {term_disp or term} 给出可核查描述", max_chars=max_chars)
    return _trim_clause_ui("给出可核查的实现与结果描述", max_chars=max_chars)


def _collect_ref_snippets_ui(meta: dict, *, max_n: int = 5) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        s2 = _clean_sentence_candidate_ui(str(s or ""))
        if not s2:
            return
        k = s2.lower()
        if k in seen:
            return
        seen.add(k)
        out.append(s2)

    if isinstance(meta, dict):
        for s in (meta.get("ref_snippets") or [])[:3]:
            _add(str(s or ""))
        for s in (meta.get("ref_show_snippets") or [])[:4]:
            _add(str(s or ""))
        for loc in (meta.get("ref_locs") or [])[:3]:
            if not isinstance(loc, dict):
                continue
            _add(str(loc.get("snippet") or ""))
    if isinstance(meta, dict) and (not out):
        _add(str(meta.get("text") or ""))
    return out[: max(1, int(max_n))]


def _split_sentences_ui(text: str, *, max_n: int = 24) -> list[str]:
    s = _clean_sentence_candidate_ui(text)
    if not s:
        return []
    parts = re.split(r"(?<=[銆傦紒锛?!?;锛沒)\s+|[銆傦紒锛燂紱]", s)
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        p2 = _clean_sentence_candidate_ui(p)
        if len(p2) < 10:
            continue
        if _looks_noisy_sentence_ui(p2):
            continue
        k2 = p2.lower()
        if k2 in seen:
            continue
        seen.add(k2)
        out.append(p2)
        if len(out) >= int(max_n):
            break
    return out


def _trim_clause_ui(text: str, *, max_chars: int = 110) -> str:
    s = " ".join(str(text or "").strip().split())
    if not s:
        return ""
    s = re.sub(r"^[,;:锛岋紱锛歕-]+", "", s).strip()
    s = re.sub(r"[銆傦紒锛?!?;锛沒+$", "", s).strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _find_sentence_by_pat_ui(sents: list[str], pat: re.Pattern, *, max_chars: int, anchors: list[str] | None = None) -> str:
    if not sents:
        return ""
    anchors_l = [str(x or "").strip().lower() for x in (anchors or []) if str(x or "").strip()]
    best = ""
    best_score = -1.0
    for s in sents:
        ss = str(s or "")
        if not pat.search(ss):
            continue
        if _looks_noisy_sentence_ui(ss):
            continue
        low = ss.lower()
        score = 1.0
        if anchors_l:
            score += 1.8 * sum(1 for a in anchors_l if a and (a in low))
        if re.search(r"\b(table|fig|figure)\b", low):
            score -= 1.3
        if re.search(r"\b(result and analysis|supplementary|appendix)\b", low):
            score -= 0.8
        if len(ss) >= 36:
            score += 0.3
        if score > best_score:
            best_score = score
            best = ss
    if best:
        return _trim_clause_ui(best, max_chars=max_chars)
    for s in sents:
        ss = str(s or "")
        if pat.search(ss):
            return _trim_clause_ui(ss, max_chars=max_chars)
    return ""


def _pick_role_sentence_ui(sents: list[str], *, role: str, anchors: list[str]) -> str:
    role_key = str(role or "").strip().lower()
    if role_key == "problem":
        pat = re.compile(
            r"(闂|鎸戞垬|鐡堕|鍙楅檺|闅句互|鍥伴毦|problem|challenge|bottleneck|limitation|difficult|lack|struggle)",
            flags=re.I,
        )
    elif role_key == "method":
        pat = re.compile(
            r"(鎻愬嚭|閲囩敤|璁捐|鏋勫缓|寮曞叆|瀹炵幇|propose|introduce|design|develop|variant)",
            flags=re.I,
        )
    else:
        pat = re.compile(
            r"(缁撴灉|鏄剧ず|琛ㄦ槑|鎻愬崌|鎻愰珮|浼樹簬|楠岃瘉|鎬ц兘|鎸囨爣|result|results|show|demonstrate|improv|outperform|achieve|experiment)",
            flags=re.I,
        )

    best = ""
    best_sc = -1e9
    anchors_l = [str(a or "").strip().lower() for a in (anchors or []) if str(a or "").strip()]
    for s in sents or []:
        ss = str(s or "")
        if not ss or (not pat.search(ss)):
            continue
        if _looks_noisy_sentence_ui(ss):
            continue
        low = ss.lower()
        sc = 1.0
        if anchors_l:
            hit_n = sum(1 for a in anchors_l if a and (a in low))
            if role_key == "problem":
                sc += 0.6 * hit_n
            elif role_key == "method":
                sc += 0.9 * hit_n
            else:
                sc += 1.2 * hit_n
        if re.search(r"\b(table|fig|figure)\b", low):
            sc -= 1.2
        if role_key == "problem":
            if re.search(r"(challenge|limitations?|struggle|bottleneck|鍙楅檺|鎸戞垬|鐡堕|鍥伴毦)", low):
                sc += 3.0
            if re.search(r"(did\s+not\s+outperform|not\s+outperform)", low):
                sc -= 2.2
            if re.search(r"(airplants|hotdog|noteworthy|second-best|underlined|bold)", low):
                sc -= 1.0
        elif role_key == "method":
            if re.search(r"(we\s+propose|propose|introduce|develop|design|variant)", low):
                sc += 2.3
            if re.search(r"(compared?|comparison|baseline)", low):
                sc -= 1.1
        else:
            if re.search(r"(results?|show|demonstrate|outperform|improv|achieve|瀹為獙|缁撴灉)", low):
                sc += 2.0
            if re.search(r"(second-best|underlined|bold)", low):
                sc -= 0.8
        if len(ss) >= 36:
            sc += 0.2
        if sc > best_sc:
            best_sc = sc
            best = ss
    return " ".join(best.split())


def _pick_specific_terms_ui(cands: list[str], *, max_n: int = 3) -> list[str]:
    ranked = sorted(
        [t for x in (cands or []) for t in _explode_find_terms_ui(str(x or ""), max_n=6)],
        key=_anchor_specificity_score_ui,
        reverse=True,
    )
    out: list[str] = []
    seen: set[str] = set()
    for t in ranked:
        k = t.lower()
        if k in seen:
            continue
        if any((k in s) or (s in k) for s in seen if len(s) >= 5):
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= int(max_n):
            break
    return out


def _build_ref_navigation(meta: dict, *, prompt: str, heading_fallback: str = "") -> dict:
    pack = meta.get("ref_pack") if isinstance(meta.get("ref_pack"), dict) else {}
    pack = pack if isinstance(pack, dict) else {}
    pack_state = str(meta.get("ref_pack_state") or "").strip().lower()
    pack_pending = pack_state == "pending"
    pack_ready = pack_state == "ready"
    prompt_is_cjk = _has_cjk_text_ui(prompt)

    source_path = str(meta.get("source_path") or "").strip()
    heading_path = _sanitize_heading_path_ui(str(meta.get("ref_best_heading_path") or "").strip(), prompt=prompt, source_path=source_path)
    sec = str(meta.get("ref_section") or "").strip()
    sub = str(meta.get("ref_subsection") or "").strip()
    if sec and _is_non_navigational_heading_ui(sec, prompt=prompt, source_path=source_path):
        sec = ""
    if sec and _looks_like_doc_title_heading_ui(sec, source_path):
        sec = ""
    if sub and _is_non_navigational_heading_ui(sub, prompt=prompt, source_path=source_path):
        sub = ""
    if (not sec) and heading_path:
        sec, sub = _split_section_subsection(heading_path)
    if not sec:
        sec_pack = str(pack.get("section") or "").strip()
        if sec_pack and (not _is_non_navigational_heading_ui(sec_pack, prompt=prompt, source_path=source_path)):
            if not _looks_like_doc_title_heading_ui(sec_pack, source_path):
                sec = sec_pack
    if not sec:
        sec_meta = str(meta.get("top_heading") or "").strip() or str(heading_fallback or "").strip()
        if sec_meta and (not _is_non_navigational_heading_ui(sec_meta, prompt=prompt, source_path=source_path)):
            if not _looks_like_doc_title_heading_ui(sec_meta, source_path):
                sec = sec_meta
    if (not heading_path) and sec:
        heading_path = sec + (f" / {sub}" if sub else "")
    if _should_avoid_discussion_ui(prompt):
        if sec and _is_discussion_heading_ui(sec):
            sec = ""
            sub = ""
            heading_path = ""
        elif heading_path and _is_discussion_heading_ui(heading_path):
            heading_path = ""

    what = _clean_sentence_candidate_ui(str(pack.get("what") or "").strip())
    why = _clean_sentence_candidate_ui(str(pack.get("why") or "").strip())
    if pack_ready:
        if _looks_template_artifact_ui(what):
            what = ""
        if _looks_template_artifact_ui(why):
            why = ""
    start_s = str(pack.get("start") or "").strip()
    gain_s = _clean_sentence_candidate_ui(str(pack.get("gain") or "").strip())
    find_list: list[str] = []
    raw_find = pack.get("find")
    if isinstance(raw_find, list):
        for x in raw_find:
            for item in _explode_find_terms_ui(str(x or ""), max_n=6):
                find_list.append(item)
    anchors = _extract_anchor_terms_ui(meta, prompt=prompt, max_n=5)
    if find_list:
        dedup_find: list[str] = []
        seen_find: set[str] = set()
        for f in find_list:
            f2 = " ".join(str(f or "").strip().split())
            if not f2:
                continue
            k2 = f2.lower()
            if k2 in seen_find:
                continue
            seen_find.add(k2)
            if _looks_generic_guidance_ui(f2):
                continue
            if _should_avoid_discussion_ui(prompt) and _is_discussion_heading_ui(f2):
                continue
            dedup_find.append(f2)
        find_list = dedup_find[:4]
    if (not find_list) and (not pack_pending) and (not pack_ready):
        raw_aspects = meta.get("ref_aspects")
        if isinstance(raw_aspects, list):
            for x in raw_aspects[:4]:
                for item in _explode_find_terms_ui(str(x or ""), max_n=4):
                    find_list.append(item)
    if find_list:
        dedup2: list[str] = []
        seen2: set[str] = set()
        for f in find_list:
            f2 = " ".join(str(f or "").strip().split())
            if not f2:
                continue
            k2 = f2.lower()
            if k2 in seen2:
                continue
            seen2.add(k2)
            if _looks_generic_guidance_ui(f2):
                continue
            if _should_avoid_discussion_ui(prompt) and _is_discussion_heading_ui(f2):
                continue
            dedup2.append(f2)
        find_list = dedup2[:4]
    if (not find_list) and anchors and (not pack_pending) and (not pack_ready):
        find_list = anchors[:4]

    try:
        sem_score = float(pack.get("score", 0.0) or 0.0)
    except Exception:
        sem_score = 0.0

    start_from = start_s
    if start_from and (not _wants_reference_nav_ui(prompt)) and _REF_HEADING_RE_UI.search(start_from):
        start_from = ""
    if start_from:
        m = re.search(r"`([^`]{2,180})`", start_from)
        if m:
            hp_m = _sanitize_heading_path_ui(m.group(1), prompt=prompt, source_path=source_path)
            if hp_m:
                start_from = start_from[: m.start()] + f"`{hp_m}`" + start_from[m.end() :]
            else:
                start_from = (start_from[: m.start()] + start_from[m.end() :]).strip(" ，;；。")
    if start_from:
        compact_start = re.sub(r"[\s`|,;:锛岋紱銆傦細路\-_/\\(){}\[\]]+", "", start_from)
        if len(compact_start) < 6:
            start_from = ""
        elif re.search(r"(鍏堜粠\s*寮€濮媩start\s+with\s*$)", start_from, flags=re.I):
            start_from = ""
    if start_from and _is_venue_heading_ui(start_from):
        start_from = ""
    if _should_avoid_discussion_ui(prompt) and _is_discussion_heading_ui(start_from):
        start_from = ""
    if start_from and _looks_generic_guidance_ui(start_from):
        start_from = ""
    if (not start_from) and (not pack_pending) and (not pack_ready):
        if heading_path:
            if anchors:
                start_from = f"先从 `{heading_path}` 开始，优先定位 {anchors[0]}，再核对相关定义、设置与结果。"
            else:
                start_from = f"先从 `{heading_path}` 开始，优先看与当前问题直接相关的定义、设置和关键结果。"
        elif sec:
            if anchors:
                start_from = f"先从 `{sec}` 开始，先定位 {anchors[0]} 和相关图表，再看支撑结论的段落。"
            else:
                start_from = f"先从 `{sec}` 开始，先定位与问题关键词直接匹配的段落和图表。"
        elif find_list:
            start_from = f"先在方法/实验相关段落中定位：{'、'.join(find_list[:2])}。"
        elif anchors:
            start_from = f"先在正文中搜索 {anchors[0]}，再顺着相关段落追踪其方法与结果证据。"
    if (not start_from) and prompt and (not pack_pending) and (not pack_ready):
        start_from = "先从方法/实验设置相关小节读起，优先找与问题关键词直接匹配的定义、设置和结果描述。"

    gain = gain_s
    if (not gain) and (not pack_pending) and (not pack_ready):
        gain = "、".join(find_list[:4]).strip()
    if (not gain) and what and (not pack_pending):
        gain = what
    if gain and _looks_generic_guidance_ui(gain) and anchors and (not pack_pending) and (not pack_ready):
        gain = f"可直接提取 {'、'.join(anchors[:3])} 等与提问强相关的证据。"

    summary_line = what
    if pack_ready:
        summary_line = summary_line.replace("...", " ").strip()
        why = why.replace("...", " ").strip()
        if prompt_is_cjk:
            if (not _has_cjk_text_ui(summary_line)) or _looks_latin_heavy_ui(summary_line):
                summary_line = ""
            if (not _has_cjk_text_ui(why)) or _looks_latin_heavy_ui(why):
                why = ""
        if _looks_generic_guidance_ui(summary_line) or _contains_question_echo_ui(summary_line, prompt):
            summary_line = ""
        if _looks_generic_guidance_ui(why):
            why = ""
    else:
        summary_line = ""
        why = ""

    return {
        "what": what,
        "summary_line": summary_line,
        "why": why,
        "start_from": start_from,
        "gain": gain,
        "sem_score": sem_score,
        "section": sec,
        "subsection": sub,
        "find": find_list[:4],
        "pack_pending": pack_pending,
    }


def _fallback_why_line_ui(
    *,
    prompt: str,
    heading_label: str = "",
    section_label: str = "",
    subsection_label: str = "",
    find_terms: list[str] | None = None,
) -> str:
    q = " ".join(str(prompt or "").strip().split())
    if not q:
        return ""
    if len(q) > 30:
        q = q[:30].rstrip() + "..."

    loc = str(subsection_label or "").strip() or str(section_label or "").strip() or str(heading_label or "").strip()
    terms: list[str] = []
    for t in (find_terms or []):
        tt = " ".join(str(t or "").strip().split())
        if tt and (tt not in terms):
            terms.append(tt)
        if len(terms) >= 2:
            break

    if loc and terms:
        return f"该文在“{loc}”处直接讨论了“{'、'.join(terms)}”，与“{q}”的关注点直接对应。"
    if loc:
        return f"该文在“{loc}”给出了与“{q}”直接相关的定义、方法或结果信息。"
    if terms:
        return f"该文对“{'、'.join(terms)}”有直接论述，可作为回答“{q}”的关键证据来源。"
    return f"该文内容与“{q}”主题一致，可作为当前问题的直接参考依据。"


def _normalize_name_key(text: str) -> str:
    s = html.unescape((text or "").strip()).lower()
    s = s.replace("–", "-").replace("—", "-")
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


def _resolve_source_doc_path(source_path: str, *, md_root_hint: str = "") -> Path | None:
    raw = (source_path or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if p.exists():
        return p
    if p.is_absolute():
        return None

    md_root_str = str(md_root_hint or "").strip()
    if not md_root_str:
        try:
            md_root_str = str(st.session_state.get("md_dir") or "").strip()
        except Exception:
            md_root_str = ""
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


def _load_source_preview_text(source_path: str, *, max_chars: int = 12000, md_root_hint: str = "") -> str:
    p = _resolve_source_doc_path(source_path, md_root_hint=md_root_hint)
    if not p:
        return ""
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if not txt:
            return ""
        return txt[: max(1000, int(max_chars))]
    except Exception:
        return ""


def _infer_title_from_source_text(source_path: str, fallback_title: str, *, md_root_hint: str = "") -> str:
    txt = _load_source_preview_text(source_path, max_chars=9000, md_root_hint=md_root_hint)
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


def fetch_crossref_meta(
    title: str,
    *,
    source_path: str = "",
    expected_venue: str = "",
    expected_year: str = "",
    md_root_hint: str = "",
) -> dict | None:
    """
    Synchronous fetch with strict confidence gate.
    Return None when not reliable enough.
    """
    q = (title or "").strip()
    doi_hint = extract_first_doi(source_path)
    if not doi_hint:
        doi_hint = extract_first_doi(_load_source_preview_text(source_path, md_root_hint=md_root_hint))
    if (not q or len(q) < 5) and (not doi_hint):
        return None
    if not q or len(q) < 5:
        q = ""
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

    # If DOI is available from source text/path, trust DOI-first resolution.
    # This avoids title-noise failures (e.g., OCR author lines as "title").
    if doi_hint:
        out = fetch_best_crossref_meta(
            query_title="",
            expected_year=year,
            expected_venue=venue,
            doi_hint=doi_hint,
            min_score=0.90,
            allow_title_only=False,
        )
        if isinstance(out, dict):
            return out

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


def _norm_name_for_match(text: str) -> str:
    s = " ".join(str(text or "").strip().split()).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _text_sim(a: str, b: str) -> float:
    aa = _norm_name_for_match(a)
    bb = _norm_name_for_match(b)
    if not aa or not bb:
        return 0.0
    if aa == bb:
        return 1.0
    try:
        seq = difflib.SequenceMatcher(None, aa, bb).ratio()
    except Exception:
        seq = 0.0
    ta = set(aa.split())
    tb = set(bb.split())
    jac = (len(ta & tb) / max(1, len(ta | tb))) if ta and tb else 0.0
    return float(min(1.0, 0.68 * seq + 0.32 * jac))


def _normalize_issn(issn: str) -> str:
    s = re.sub(r"[^0-9Xx]", "", str(issn or "").strip())
    if len(s) != 8:
        return ""
    return f"{s[:4]}-{s[4:]}"


def _normalize_issn_list(items) -> set[str]:
    out: set[str] = set()
    if isinstance(items, (list, tuple, set)):
        for x in items:
            n = _normalize_issn(str(x or ""))
            if n:
                out.add(n)
    else:
        n = _normalize_issn(str(items or ""))
        if n:
            out.add(n)
    return out


def _infer_venue_kind(meta: dict) -> str:
    t = str((meta or {}).get("type") or "").strip().lower()
    venue = str((meta or {}).get("venue") or "").strip().lower()
    if "proceedings" in t or "conference" in t:
        return "conference"
    if "journal" in t or t in {"article", "journal-article"}:
        return "journal"
    if any(k in venue for k in ["conference", "symposium", "workshop", "proceedings", "congress"]):
        return "conference"
    return "journal"


def _openalex_work_by_doi(doi: str) -> dict | None:
    d = str(doi or "").strip().lower()
    if not d:
        return None
    url = "https://api.openalex.org/works/https://doi.org/" + quote(d, safe="")
    try:
        r = requests.get(url, timeout=6.0, headers={"User-Agent": "Pi-zaya-KB/1.0"})
        if r.status_code != 200:
            return None
        out = r.json()
        return out if isinstance(out, dict) else None
    except Exception:
        return None


def _lookup_journal_if(meta: dict) -> dict | None:
    try:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    except Exception:
        pass
    venue = str((meta or {}).get("venue") or "").strip()
    ox_venue = str((meta or {}).get("openalex_venue") or "").strip()
    issn = _normalize_issn(str((meta or {}).get("issn") or ""))
    eissn = _normalize_issn(str((meta or {}).get("eissn") or ""))
    ox_issn_l = _normalize_issn(str((meta or {}).get("openalex_issn_l") or ""))
    ox_issn_all = set(_normalize_issn_list((meta or {}).get("openalex_issn_set") or []))
    issn_candidates = [x for x in {issn, eissn, ox_issn_l, *ox_issn_all} if x]
    venue_candidates = [x for x in [venue, ox_venue] if x]
    if not venue_candidates and not issn_candidates:
        return None
    try:
        from impact_factor.core import Factor  # type: ignore
    except Exception:
        return None
    try:
        fa = Factor()
    except Exception:
        return None

    recs = []
    for cand in issn_candidates:
        if recs:
            break
        try:
            recs = fa.search(cand, key="issn") or []
        except Exception:
            recs = []
        if recs:
            break
        try:
            recs = fa.search(cand, key="eissn") or []
        except Exception:
            recs = []
    if not recs:
        for cand in venue_candidates:
            try:
                recs = fa.search(cand, key="journal") or []
            except Exception:
                recs = []
            if recs:
                break
    if not recs:
        return None

    best = None
    best_sc = -1.0
    for r in recs:
        if not isinstance(r, dict):
            continue
        jname = str(r.get("journal") or "").strip()
        r_issn = _normalize_issn(str(r.get("issn") or ""))
        r_eissn = _normalize_issn(str(r.get("eissn") or ""))
        sc = 0.0
        for cand in venue_candidates:
            if cand:
                sc = max(sc, _text_sim(cand, jname))
        if r_issn and (r_issn in issn_candidates):
            sc += 1.6
        if r_eissn and (r_eissn in issn_candidates):
            sc += 1.3
        if sc > best_sc:
            best_sc = sc
            best = r
    if not isinstance(best, dict):
        return None
    if best_sc < 0.80:
        return None

    try:
        factor = float(best.get("factor"))
    except Exception:
        return None
    if factor <= 0:
        return None
    return {
        "journal_if": round(factor, 3),
        "journal_quartile": str(best.get("jcr") or "").strip(),
        "journal_if_source": "JCR dataset (impact_factor)",
        "journal_if_matched_journal": str(best.get("journal") or "").strip(),
    }


_CONF_ACR_STOP = {
    "IEEE",
    "ACM",
    "CVF",
    "IAPR",
    "IET",
    "SPIE",
    "OSA",
    "IFIP",
}


def _clean_conf_query_text(venue: str) -> str:
    v = " ".join(str(venue or "").strip().split())
    if not v:
        return ""
    v = re.sub(r"\b(19|20)\d{2}\b", " ", v)
    v = re.sub(r"[()/,:;]+", " ", v)
    v = re.sub(
        r"\b(proceedings|proc|conference|international|annual|ieee|acm|cvf|symposium|workshop)\b",
        " ",
        v,
        flags=re.I,
    )
    v = re.sub(r"\s+", " ", v).strip()
    return v


def _guess_conf_acronym(venue: str) -> str:
    v = " ".join(str(venue or "").strip().split())
    if not v:
        return ""
    # Prefer acronym inside parentheses: "... (CVPR)".
    m_paren = re.search(r"\(([A-Z][A-Z0-9]{2,12})\)", v)
    if m_paren:
        cand = str(m_paren.group(1) or "").strip().upper()
        if cand and cand not in _CONF_ACR_STOP:
            return cand
    # Next, choose uppercase token but skip publisher/organization tokens.
    cands = [str(x or "").strip().upper() for x in re.findall(r"\b([A-Z][A-Z0-9]{2,12})\b", v)]
    for cand in cands:
        if cand and cand not in _CONF_ACR_STOP:
            return cand
    # e.g., "International Conference on ..."
    toks = [w for w in re.findall(r"[A-Za-z]+", v) if w]
    initials = "".join(w[0].upper() for w in toks if w and w[0].isalpha())
    if 3 <= len(initials) <= 10:
        return initials
    return ""


def _core_parse_rows(html_text: str) -> list[dict]:
    s = str(html_text or "")
    rows: list[dict] = []
    for m in re.finditer(
        r"<tr[^>]*onclick=\"navigate\('[^']+'\)\"[^>]*>\s*"
        r"<td>\s*(.*?)\s*</td>\s*"
        r"<td[^>]*>\s*(.*?)\s*</td>\s*"
        r"<td[^>]*>\s*(.*?)\s*</td>\s*"
        r"<td[^>]*>\s*([A*BC]+)\s*</td>",
        s,
        flags=re.I | re.S,
    ):
        title = re.sub(r"<[^>]+>", " ", str(m.group(1) or ""))
        acr = re.sub(r"<[^>]+>", " ", str(m.group(2) or ""))
        source = re.sub(r"<[^>]+>", " ", str(m.group(3) or ""))
        rank = re.sub(r"<[^>]+>", " ", str(m.group(4) or ""))
        rows.append(
            {
                "title": " ".join(title.split()).strip(),
                "acronym": " ".join(acr.split()).strip(),
                "source": " ".join(source.split()).strip(),
                "rank": " ".join(rank.split()).strip(),
            }
        )
        if len(rows) >= 24:
            break
    return rows


@lru_cache(maxsize=256)
def _lookup_core_tier(venue: str) -> dict | None:
    v = " ".join(str(venue or "").strip().split())
    if not v:
        return None
    acr = _guess_conf_acronym(v)
    v_clean = _clean_conf_query_text(v)
    queries: list[str] = []
    for q in [acr, v_clean, v]:
        qn = " ".join(str(q or "").split()).strip()
        if qn and qn not in queries:
            queries.append(qn)
    # Keep query budget bounded to avoid long pending states in UI workers.
    sources = ["ICORE2026", "CORE2023", "CORE2021", "CORE2020"]
    start_ts = time.monotonic()
    budget_s = 9.0
    best = None
    best_sc = -1.0
    best_name_sim = 0.0
    for q in queries:
        if (time.monotonic() - start_ts) > budget_s:
            break
        if not q:
            continue
        for src in sources:
            if (time.monotonic() - start_ts) > budget_s:
                break
            url = "https://portal.core.edu.au/conf-ranks/"
            params = {"search": q, "by": "all", "source": src, "sort": "atitle", "page": "1"}
            try:
                r = requests.get(url, params=params, timeout=3.2, headers={"User-Agent": "Pi-zaya-KB/1.0"})
                if r.status_code != 200:
                    continue
                rows = _core_parse_rows(r.text)
            except Exception:
                continue
            for row in rows:
                title = str(row.get("title") or "")
                acronym = str(row.get("acronym") or "")
                rank = str(row.get("rank") or "").strip()
                if not rank:
                    continue
                sc_main = _text_sim(v, title)
                sc_clean = _text_sim(v_clean, title) if v_clean else 0.0
                sc = max(sc_main, sc_clean)
                if acr and acronym and (acronym.upper() == acr.upper()):
                    sc += 1.2
                if q and title and (_text_sim(q, title) >= 0.92):
                    sc += 0.3
                if sc > best_sc:
                    best_sc = sc
                    best_name_sim = max(sc_main, sc_clean)
                    best = {
                        "conference_tier": rank,
                        "conference_rank_source": src,
                        "conference_name": title,
                        "conference_acronym": acronym,
                        "conference_match_confidence": round(float(best_name_sim), 3),
                    }
            if best_sc >= 1.3:
                break
        if best_sc >= 1.3:
            break
    if not isinstance(best, dict):
        return None
    if best_sc < 0.88:
        return None
    return best


def _core_tier_to_ccf(tier: str) -> str:
    t = str(tier or "").strip().upper()
    if not t:
        return ""
    if t.startswith("A"):
        return "A"
    if t.startswith("B"):
        return "B"
    if t.startswith("C"):
        return "C"
    return ""


def _enrich_bibliometrics(meta: dict | None) -> dict | None:
    if not isinstance(meta, dict):
        return None
    out = dict(meta)
    doi = str(out.get("doi") or "").strip()
    venue = str(out.get("venue") or "").strip()
    issn = _normalize_issn(str(out.get("issn") or ""))
    eissn = _normalize_issn(str(out.get("eissn") or ""))
    venue_kind = _infer_venue_kind(out)
    out["venue_kind"] = venue_kind
    venue_verified = False

    if doi:
        ox = _openalex_work_by_doi(doi)
        if isinstance(ox, dict):
            try:
                out["citation_count"] = int(ox.get("cited_by_count") or 0)
            except Exception:
                pass
            out["citation_source"] = "OpenAlex"
            src0 = ((ox.get("primary_location") or {}).get("source") or {})
            ox_venue = str(src0.get("display_name") or "").strip()
            ox_issn_l = _normalize_issn(str(src0.get("issn_l") or ""))
            ox_issn_all = _normalize_issn_list(src0.get("issn"))
            if ox_venue:
                out["openalex_venue"] = ox_venue
                out["venue_match_confidence"] = round(_text_sim(venue, ox_venue), 3)
            if ox_issn_l:
                out["openalex_issn_l"] = ox_issn_l
            if ox_issn_all:
                out["openalex_issn_set"] = sorted(ox_issn_all)

            crossref_issn_set = {x for x in {issn, eissn} if x}
            issn_hit = bool(crossref_issn_set & ({ox_issn_l} if ox_issn_l else set())) or bool(
                crossref_issn_set & ox_issn_all
            )
            name_hit = bool(ox_venue and (_text_sim(venue, ox_venue) >= 0.78))
            if issn_hit or name_hit:
                venue_verified = True
                out["venue_verified_by"] = "OpenAlex DOI source"
                if ox_venue and (not venue or _text_sim(ox_venue, venue) > 0.90):
                    out["venue"] = ox_venue
    if ("citation_count" not in out) and isinstance(out.get("crossref_cited_by_count"), int):
        out["citation_count"] = int(out.get("crossref_cited_by_count") or 0)
        out["citation_source"] = "Crossref"

    # DOI-resolved Crossref metadata is already high confidence for venue mapping.
    if doi and (not venue_verified):
        venue_verified = True
        out["venue_verified_by"] = str(out.get("venue_verified_by") or "Crossref DOI")
    out["venue_verified"] = venue_verified

    if venue_kind == "journal":
        # Only expose IF when journal mapping is verified (DOI/OpenAlex) to avoid wrong-journal IF.
        if venue_verified:
            jif_meta = _lookup_journal_if(out)
            if isinstance(jif_meta, dict):
                out.update(jif_meta)
    else:
        tier_meta = _lookup_core_tier(venue)
        if isinstance(tier_meta, dict):
            out.update(tier_meta)
            ccf_tier = _core_tier_to_ccf(str(tier_meta.get("conference_tier") or ""))
            if ccf_tier:
                out["conference_ccf"] = ccf_tier
                out["conference_ccf_source"] = "CORE tier proxy"

    out["bibliometrics_checked"] = True
    return out


def _metrics_html(meta: dict) -> str:
    if not isinstance(meta, dict):
        return ""

    parts: list[str] = []

    cnum = meta.get("citation_count")
    if isinstance(cnum, int) and cnum >= 0:
        csrc = str(meta.get("citation_source") or "").strip()
        if csrc:
            parts.append(
                f"\u88ab\u5f15<strong>{int(cnum)}</strong> "
                f"<span class='kb-ref-metric-src'>({html.escape(csrc)})</span>"
            )
        else:
            parts.append(f"\u88ab\u5f15<strong>{int(cnum)}</strong>")
    else:
        parts.append("\u88ab\u5f15<span class='kb-ref-metric-na'>N/A</span>")

    year = str(meta.get("year") or "").strip()
    if re.fullmatch(r"(19|20)\d{2}", year):
        parts.append(f"\u5e74\u4efd<strong>{html.escape(year)}</strong>")

    doi = str(meta.get("doi") or "").strip()
    if doi:
        doi_url = doi
        if not re.match(r"^https?://", doi_url, flags=re.I):
            doi_url = "https://doi.org/" + quote(doi_url, safe="/:;._-()")
        parts.append(
            "DOI"
            f"<a class='kb-ref-doi-link' href='{html.escape(doi_url, quote=True)}' "
            "target='_blank' rel='noopener noreferrer'>"
            f"{html.escape(doi)}</a>"
        )

    kind = str(meta.get("venue_kind") or "").strip().lower()
    if kind == "conference":
        conf_acr = str(meta.get("conference_acronym") or "").strip()
        conf_name = str(meta.get("conference_name") or meta.get("venue") or "").strip()
        conf_label = conf_acr if conf_acr else conf_name
        if conf_label:
            parts.append(f"\u4f1a\u8bae<strong>{html.escape(conf_label)}</strong>")

        tier = str(meta.get("conference_tier") or "").strip()
        src = str(meta.get("conference_rank_source") or "").strip()
        if tier:
            txt = f"CORE<strong>{html.escape(tier)}</strong>"
            if src:
                txt += f" <span class='kb-ref-metric-src'>({html.escape(src)})</span>"
            parts.append(txt)
        else:
            parts.append("CORE<span class='kb-ref-metric-na'>N/A</span>")

        ccf = str(meta.get("conference_ccf") or "").strip().upper()
        ccf_src = str(meta.get("conference_ccf_source") or "").strip()
        if ccf:
            txt = f"CCF<strong>{html.escape(ccf)}</strong>"
            if ccf_src:
                txt += f" <span class='kb-ref-metric-src'>({html.escape(ccf_src)})</span>"
            parts.append(txt)
        else:
            parts.append("CCF<span class='kb-ref-metric-na'>N/A</span>")

        parts.append("IF<span class='kb-ref-metric-na'>N/A (\u4f1a\u8bae)</span>")

    else:
        venue = str(meta.get("venue") or "").strip()
        if venue:
            parts.append(f"\u671f\u520a<strong>{html.escape(venue)}</strong>")

        jif = meta.get("journal_if")
        jq = str(meta.get("journal_quartile") or "").strip()
        jsrc = str(meta.get("journal_if_source") or "").strip()
        if isinstance(jif, (int, float)) and float(jif) > 0:
            jif_s = f"{float(jif):.3f}".rstrip("0").rstrip(".")
            txt = f"IF<strong>{html.escape(jif_s)}</strong>"
            if jq:
                txt += f" <span class='kb-ref-metric-tag'>{html.escape(jq)}</span>"
            if jsrc:
                txt += f" <span class='kb-ref-metric-src'>({html.escape(jsrc)})</span>"
            parts.append(txt)
        else:
            parts.append("IF<span class='kb-ref-metric-na'>N/A</span>")

    if not parts:
        return ""
    return "<div class='kb-ref-metrics-row'>" + " | ".join(parts) + "</div>"


def _parse_filename_meta(path_str: str) -> tuple[str, str, str]:
    raw = str(path_str or "").strip()
    parts = re.split(r"[\\/]+", raw) if raw else []
    name = str(parts[-1] or "").strip() if parts else raw
    low = name.lower()
    if low.endswith(".md"):
        name = name[:-3]
        low = name.lower()
    if low.endswith(".en"):
        name = name[:-3]
    m = re.match(r"^([^-]+)\s*-\s*(19\d{2}|20\d{2})\s*-\s*(.+)$", name)
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    return "", "", name


# --- Async Citation Worker ---

def _has_metrics_payload(meta: dict | None) -> bool:
    if not isinstance(meta, dict):
        return False
    if bool(meta.get("bibliometrics_checked")):
        return True
    if isinstance(meta.get("citation_count"), int):
        return True
    if isinstance(meta.get("journal_if"), (int, float)):
        return True
    if str(meta.get("conference_tier") or "").strip():
        return True
    return False


_CITATION_TASK_TIMEOUT_S = 90.0
_CITATION_RETRY_COOLDOWN_S = 8.0
_CITATION_MAX_RETRIES = 2
_CITATION_FAIL_BACKOFF_S = 120.0


def _sync_citation_task_state(net_key: str) -> tuple[dict | None, bool, bool, bool]:
    """
    Returns: (net_data, failed, pending, changed)
    """
    net_data = st.session_state.get(net_key)
    failed = bool(st.session_state.get(f"{net_key}_failed", False))
    pending = False
    changed = False

    if isinstance(net_data, dict) and _has_metrics_payload(net_data):
        return net_data, failed, pending, changed
    if failed:
        return (net_data if isinstance(net_data, dict) else None), failed, pending, changed

    from kb import runtime_state as RUNTIME

    task_id = f"cite_task_{net_key}"
    with RUNTIME.CITATION_LOCK:
        task = RUNTIME.CITATION_TASKS.get(task_id)

    if not task:
        return (net_data if isinstance(net_data, dict) else None), failed, pending, changed

    if task.get("done"):
        res = task.get("result")
        err_msg = str(task.get("error") or "").strip()
        changed = True
        if isinstance(res, dict):
            st.session_state[net_key] = res
            net_data = res
            st.session_state.pop(f"{net_key}_failed", None)
            st.session_state.pop(f"{net_key}_failed_ts", None)
            st.session_state.pop(f"{net_key}_failed_reason", None)
            st.session_state[f"{net_key}_retry_n"] = 0
        else:
            st.session_state[f"{net_key}_failed"] = True
            st.session_state[f"{net_key}_failed_ts"] = float(time.time())
            st.session_state[f"{net_key}_failed_reason"] = (err_msg or "no_result")[:180]
            failed = True
        with RUNTIME.CITATION_LOCK:
            RUNTIME.CITATION_TASKS.pop(task_id, None)
    else:
        try:
            created_at = float(task.get("created_at") or 0.0)
        except Exception:
            created_at = 0.0
        # Guard against a stuck background thread: never keep "pending" forever.
        if created_at > 0 and (time.time() - created_at) > _CITATION_TASK_TIMEOUT_S:
            with RUNTIME.CITATION_LOCK:
                t2 = RUNTIME.CITATION_TASKS.get(task_id)
                if isinstance(t2, dict):
                    t2["done"] = True
                    t2["result"] = None
                    t2["error"] = str(t2.get("error") or "timeout")
            st.session_state[f"{net_key}_failed"] = True
            st.session_state[f"{net_key}_failed_ts"] = float(time.time())
            st.session_state[f"{net_key}_failed_reason"] = "timeout"
            failed = True
            changed = True
        else:
            pending = True

    return (net_data if isinstance(net_data, dict) else None), failed, pending, changed


def _ensure_citation_task(net_key: str, source_path: str) -> None:
    existing = st.session_state.get(net_key)
    if isinstance(existing, dict) and _has_metrics_payload(existing):
        return
    if st.session_state.get(f"{net_key}_failed"):
        now_ts = float(time.time())
        try:
            failed_ts = float(st.session_state.get(f"{net_key}_failed_ts") or 0.0)
        except Exception:
            failed_ts = 0.0
        try:
            retry_n = int(st.session_state.get(f"{net_key}_retry_n") or 0)
        except Exception:
            retry_n = 0
        if (now_ts - failed_ts) < _CITATION_RETRY_COOLDOWN_S:
            return
        if (retry_n >= _CITATION_MAX_RETRIES) and ((now_ts - failed_ts) < _CITATION_FAIL_BACKOFF_S):
            return
        st.session_state.pop(f"{net_key}_failed", None)
        st.session_state.pop(f"{net_key}_failed_reason", None)

    from kb import runtime_state as RUNTIME
    import threading

    task_id = f"cite_task_{net_key}"
    with RUNTIME.CITATION_LOCK:
        if task_id in RUNTIME.CITATION_TASKS:
            return
        RUNTIME.CITATION_TASKS[task_id] = {
            "created_at": time.time(),
            "done": False,
            "result": None,
            "net_key": net_key,
        }
    try:
        st.session_state[f"{net_key}_retry_n"] = int(st.session_state.get(f"{net_key}_retry_n") or 0) + 1
    except Exception:
        st.session_state[f"{net_key}_retry_n"] = 1

    l_venue, l_year, _ = _parse_filename_meta(source_path)
    try:
        pdf_root_hint = str(st.session_state.get("pdf_dir") or "").strip()
    except Exception:
        pdf_root_hint = ""
    try:
        md_root_hint = str(st.session_state.get("md_dir") or "").strip()
    except Exception:
        md_root_hint = ""
    try:
        lib_store_obj = st.session_state.get("lib_store")
    except Exception:
        lib_store_obj = None
    t = threading.Thread(
        target=_bg_citation_worker,
        args=(task_id, net_key, source_path, l_venue, l_year, pdf_root_hint, md_root_hint, lib_store_obj),
        daemon=True,
    )
    t.start()


def _bg_citation_worker(
    task_id: str,
    net_key: str,
    source_path: str,
    venue_hint: str,
    year_hint: str,
    pdf_root_hint: str = "",
    md_root_hint: str = "",
    lib_store_obj=None,
):
    from kb import runtime_state as RUNTIME

    found = None
    pdf_path = None
    lib_store = lib_store_obj
    worker_error = ""
    try:
        if lib_store and pdf_root_hint:
            pdf_root = Path(pdf_root_hint)
            pdf_path = _resolve_pdf_for_source(pdf_root, source_path)
            if pdf_path and pdf_path.exists():
                stored_meta = lib_store.get_citation_meta(pdf_path)
                if stored_meta and isinstance(stored_meta, dict):
                    found = dict(stored_meta)

        need_fetch = not isinstance(found, dict)
        if isinstance(found, dict):
            has_title = bool(str(found.get("title") or "").strip())
            has_doi = bool(str(found.get("doi") or "").strip())
            has_venue = bool(str(found.get("venue") or "").strip())
            need_fetch = not (has_title and has_venue and has_doi)

        if need_fetch:
            l_title_hint = os.path.basename(source_path)
            try:
                if l_title_hint.lower().endswith(".pdf"):
                    l_title_hint = l_title_hint[:-4]
                search_title = _infer_title_from_source_text(
                    source_path,
                    l_title_hint,
                    md_root_hint=md_root_hint,
                )
            except Exception:
                search_title = l_title_hint

            fetched = fetch_crossref_meta(
                search_title,
                source_path=source_path,
                expected_venue=venue_hint,
                expected_year=year_hint,
                md_root_hint=md_root_hint,
            )
            if isinstance(fetched, dict):
                if isinstance(found, dict):
                    merged = dict(found)
                    merged.update({k: v for k, v in fetched.items() if v not in (None, "", [], {})})
                    found = merged
                else:
                    found = fetched

        if isinstance(found, dict) and (not bool(found.get("bibliometrics_checked"))):
            try:
                enriched = _enrich_bibliometrics(found)
                if isinstance(enriched, dict):
                    found = enriched
                else:
                    found["bibliometrics_checked"] = True
            except Exception:
                found["bibliometrics_checked"] = True

        try:
            if isinstance(found, dict) and lib_store and pdf_path and pdf_path.exists() and hasattr(lib_store, "set_citation_meta"):
                lib_store.set_citation_meta(pdf_path, found)  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception as exc:
        worker_error = str(exc or "").strip()[:260]
        found = None
    finally:
        with RUNTIME.CITATION_LOCK:
            tasks = RUNTIME.CITATION_TASKS
            if task_id in tasks:
                t = tasks[task_id]
                t["done"] = True
                t["result"] = found
                if worker_error:
                    t["error"] = worker_error


def _on_cite_click(cite_key: str, net_key: str, source_path: str, refs_open_key: str = ""):
    """
    Toggle cite detail panel and trigger async fetch if needed.
    """
    if refs_open_key:
        st.session_state[refs_open_key] = True

    new_state = not st.session_state.get(cite_key, False)
    st.session_state[cite_key] = new_state
    if new_state:
        st.session_state.pop(f"{net_key}_failed", None)
        st.session_state.pop(f"{net_key}_failed_ts", None)
        st.session_state.pop(f"{net_key}_failed_reason", None)
        st.session_state[f"{net_key}_retry_n"] = 0
        _ensure_citation_task(net_key, source_path)


def _render_refs(
        hits: list[dict],
        *,
        prompt: str = "",
        show_heading: bool = True,
        key_ns: str = "refs",
        refs_open_key: str = "",
        settings=None,
) -> None:
    settings_obj = settings
    refs_panel_open = True
    if refs_open_key:
        # Expander state is not reliably mirrored into session_state when user manually toggles.
        # Default to visible once rendering starts, so metric tasks are not starved.
        refs_panel_open = bool(st.session_state.get(refs_open_key, True))

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

    def _norm_text(s: str) -> str:
        return re.sub(r"\s+", " ", str(s or "").strip()).strip()

    def _loc_chip_html(label: str, value: str) -> str:
        v = str(value or "").strip()
        if not v:
            return ""
        return (
            "<span class='kb-ref-loc-chip'>"
            f"<span class='kb-ref-loc-chip-label'>{html.escape(label)}</span>"
            f"<span class='kb-ref-loc-chip-value'>{html.escape(v)}</span>"
            "</span>"
        )

    def _insight_card_html(tag: str, title: str, text: str) -> str:
        body = _norm_text(text)
        if not body:
            return ""
        return (
            "<div class='kb-ref-insight-card'>"
            "<div class='kb-ref-insight-head'>"
            f"<span class='kb-ref-insight-tag'>{html.escape(tag)}</span>"
            f"<span class='kb-ref-guide-label kb-ref-inline-label'>{html.escape(title)}</span>"
            "</div>"
            f"<div class='kb-ref-insight-text'>{html.escape(body)}</div>"
            "</div>"
        )

    any_metric_changed = False
    any_metric_pending = False
    any_pack_pending = False

    for i, h in enumerate(filtered_hits, start=1):
        meta = h.get("meta", {}) or {}
        source_path = str(meta.get("source_path") or "").strip()
        heading_path = _sanitize_heading_path_ui(
            str(meta.get("ref_best_heading_path") or meta.get("heading_path") or "").strip(),
            prompt=prompt,
            source_path=source_path,
        )
        heading = str(meta.get("top_heading") or _top_heading(heading_path) or _top_heading(str(meta.get("heading_path") or "")) or "").strip()
        if heading and _is_non_navigational_heading_ui(heading, prompt=prompt, source_path=source_path):
            heading = ""
        if heading and _looks_like_doc_title_heading_ui(heading, source_path):
            heading = ""
        section_label = str(meta.get("ref_section") or "").strip()
        subsection_label = str(meta.get("ref_subsection") or "").strip()
        if section_label and _is_non_navigational_heading_ui(section_label, prompt=prompt, source_path=source_path):
            section_label = ""
        if subsection_label and _is_non_navigational_heading_ui(subsection_label, prompt=prompt, source_path=source_path):
            subsection_label = ""
        if (not section_label) and heading_path:
            section_label, subsection_label = _split_section_subsection(heading_path)
        if section_label and _looks_like_doc_title_heading_ui(section_label, source_path):
            # Hide title-line pseudo sections; show page and semantic guidance instead.
            section_label = ""
            subsection_label = ""
        p0, p1 = _safe_page_range(meta)
        score = float(h.get("score", 0.0) or 0.0)

        source_label = _display_source_name(source_path)
        heading_label = (heading_path or heading or "").strip()
        # Avoid duplicated location text: when heading path is shown, hide section chips.
        section_chip_label = section_label
        subsection_chip_label = subsection_label
        if heading_label:
            section_chip_label = ""
            subsection_chip_label = ""
        score_s = f"{score:.2f}" if score > 0 else "-"
        score_tier = _score_tier(score)

        source_attr = html.escape(source_label, quote=True)
        heading_attr = html.escape(heading_label, quote=True) if heading_label else ""
        source_html = html.escape(source_label)

        pdf_path = _resolve_pdf_for_source(pdf_root, source_path)
        has_pdf = bool(pdf_path)
        uid = hashlib.sha1(str(source_path).encode("utf-8", "ignore")).hexdigest()[:10]
        cite_key = f"{key_ns}_cite_visible_{uid}"
        net_key = f"{key_ns}_net_meta_v6_{uid}"
        is_cite_open = st.session_state.get(cite_key, False)

        net_meta = st.session_state.get(net_key)
        metric_failed = bool(st.session_state.get(f"{net_key}_failed", False))
        metric_pending = False
        metric_changed = False
        _ensure_citation_task(net_key, source_path)
        net_meta, metric_failed, metric_pending, metric_changed = _sync_citation_task_state(net_key)
        if metric_changed:
            any_metric_changed = True
        if metric_pending:
            any_metric_pending = True

        nav = _build_ref_navigation(meta, prompt=prompt, heading_fallback=heading)
        summary_line = str(nav.get("summary_line") or nav.get("what") or "").strip()
        why_line = str(nav.get("why") or "").strip()
        if summary_line and (not why_line):
            why_line = _fallback_why_line_ui(
                prompt=prompt,
                heading_label=heading_label,
                section_label=section_label,
                subsection_label=subsection_label,
                find_terms=list(nav.get("find") or []),
            )
        pack_pending = bool(nav.get("pack_pending"))
        pack_state_local = str(meta.get("ref_pack_state") or "").strip().lower()
        pack_ready_local = pack_state_local == "ready"
        if pack_pending:
            any_pack_pending = True

        loc_chip_parts: list[str] = []
        if section_chip_label:
            loc_chip_parts.append(_loc_chip_html("\u7ae0\u8282", section_chip_label))
        if subsection_chip_label:
            loc_chip_parts.append(_loc_chip_html("\u5c0f\u8282", subsection_chip_label))
        if p0 and p1 and p1 > p0:
            loc_chip_parts.append(_loc_chip_html("\u9875\u7801", f"{int(p0)}-{int(p1)}"))
        elif p0:
            loc_chip_parts.append(_loc_chip_html("\u9875\u7801", f"{int(p0)}"))

        status_html = f"<span class='ref-score ref-score-{score_tier}'>\u5339\u914d\u5206 {score_s}</span>"

        insight_cards: list[str] = []
        if summary_line:
            insight_cards.append(_insight_card_html("\u6458\u8981", "\u8fd9\u7bc7\u6587\u732e\u8bb2\u4ec0\u4e48 / \u63d0\u4f9b\u4ec0\u4e48", summary_line))
        if why_line:
            insight_cards.append(_insight_card_html("\u76f8\u5173", "\u4e3a\u4ec0\u4e48\u4e0e\u5f53\u524d\u95ee\u9898\u5f3a\u76f8\u5173", why_line))
        insights_html = (
            f"<div class='kb-ref-insight-grid'>{''.join(insight_cards)}</div>"
            if insight_cards
            else ""
        )
        metrics_html = _metrics_html(net_meta) if isinstance(net_meta, dict) else ""

        with st.container():
            header_cols = st.columns([0.50, 6.2, 1.0, 1.0], gap="small")

            with header_cols[0]:
                st.markdown(
                    "<div class='kb-ref-rank-wrap'><span class='ref-rank'>"
                    f"#{i}"
                    "</span></div>",
                    unsafe_allow_html=True,
                )

            with header_cols[1]:
                title_block = (
                    "<div class='kb-ref-header-block'>"
                    "<div class='kb-ref-title-row'>"
                    "<div class='kb-ref-title-stack'>"
                    f"<div class='kb-ref-title' title='{source_attr}'>{source_html}</div>"
                )
                title_block += "<div class='kb-ref-heading-meta-row'>"
                if heading_label:
                    title_block += (
                        f"<div class='kb-ref-heading-path' title='{heading_attr}'>{html.escape(heading_label)}</div>"
                    )
                title_block += f"<div class='kb-ref-heading-score-wrap'>{status_html}</div>"
                title_block += "</div>"
                title_block += "</div></div></div>"
                st.markdown(title_block, unsafe_allow_html=True)

            with header_cols[2]:
                if st.button(
                    "Open",
                    key=f"{key_ns}_open_pdf_{uid}",
                    help="Open PDF",
                    disabled=(not has_pdf),
                ):
                    if refs_open_key:
                        st.session_state[refs_open_key] = True
                    ok, msg = _open_pdf(pdf_path)
                    if not ok:
                        st.warning(msg)

            with header_cols[3]:
                btn_label = "Close" if is_cite_open else "Cite"
                st.button(
                    btn_label,
                    key=f"{key_ns}_cite_btn_{uid}",
                    help="Fetch citation",
                    on_click=_on_cite_click,
                    args=(cite_key, net_key, source_path, refs_open_key),
                )

            if loc_chip_parts:
                st.markdown(
                    f"<div class='kb-ref-loc-row'>{''.join(loc_chip_parts)}</div>",
                    unsafe_allow_html=True,
                )

            if insights_html:
                st.markdown(insights_html, unsafe_allow_html=True)
            elif pack_pending:
                st.caption("摘要与相关性正在生成中...")
            elif pack_ready_local:
                st.caption("LLM 摘要暂不可用")
            elif pack_state_local == "none":
                st.caption("摘要与相关性生成失败或超时（本次未产出）")
            elif settings_obj and getattr(settings_obj, "api_key", None):
                st.caption("摘要与相关性待 LLM 生成")
            else:
                st.caption("未配置 LLM，摘要与相关性不可用")

            if metrics_html:
                st.markdown(metrics_html, unsafe_allow_html=True)
            elif metric_pending and (refs_panel_open or is_cite_open):
                st.caption("文献指标检索中...")
            elif metric_failed:
                fail_reason = str(st.session_state.get(f"{net_key}_failed_reason") or "").strip()
                if fail_reason:
                    st.caption(f"鏂囩尞鎸囨爣妫€绱㈠け璐ワ紙{fail_reason}锛夛紝鍙偣鍑?Cite 閲嶈瘯")
                else:
                    st.caption("鏂囩尞鎸囨爣妫€绱㈠け璐ワ紝鍙偣鍑?Cite 閲嶈瘯")

            if st.session_state.get(cite_key, False):
                _render_citation_ui(uid, source_path, key_ns)

        if i < len(filtered_hits):
            st.markdown("<div class='kb-ref-item-gap'></div>", unsafe_allow_html=True)

    if any_metric_changed:
        st.experimental_rerun()
    elif (any_metric_pending or any_pack_pending) and refs_panel_open:
        # Light polling so async metrics resolve without requiring extra user actions.
        poll_key = f"{key_ns}_refs_poll_ts"
        now_ts = float(time.time())
        try:
            last_ts = float(st.session_state.get(poll_key) or 0.0)
        except Exception:
            last_ts = 0.0
        interval_s = 0.55 if any_pack_pending else 0.9
        if (now_ts - last_ts) >= interval_s:
            st.session_state[poll_key] = now_ts
            time.sleep(0.10 if any_pack_pending else 0.12)
            st.experimental_rerun()


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
# Policy:
# If a citation number is shown, it must be clickable. For ambiguous/unresolved
# references, hide the marker instead of showing non-clickable plain text.
_STRICT_STRUCTURED_CITATION_LINKING = True


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


@lru_cache(maxsize=8)
def _load_reference_index_file_cached(sig: str, db_dir_str: str) -> dict:
    del sig
    try:
        return _load_reference_index_file(Path(db_dir_str))
    except Exception:
        return {}


def _load_reference_index_cached() -> dict:
    use_streamlit_state = bool(getattr(st, "_is_running_with_streamlit", False))
    db_dir_str = ""
    if use_streamlit_state:
        db_dir_str = str(st.session_state.get("db_dir") or "").strip()
    if not db_dir_str:
        try:
            db_dir_str = str(load_settings().db_dir or "").strip()
        except Exception:
            db_dir_str = ""
    if not db_dir_str:
        return {}
    db_dir = Path(db_dir_str).expanduser().resolve()
    idx_path = db_dir / "references_index.json"
    if not idx_path.exists():
        return {}

    try:
        idx_sig = f"{str(idx_path.resolve())}|{int(idx_path.stat().st_mtime)}|{int(idx_path.stat().st_size)}"
    except Exception:
        idx_sig = str(idx_path)

    if not use_streamlit_state:
        return _load_reference_index_file_cached(idx_sig, str(db_dir))

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
    if not title:
        try:
            title = str((_fallback_fill_reference_meta_from_raw(ref_rec) or {}).get("title") or "").strip()
        except Exception:
            title = ""
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


_VENUE_SPLIT_ABBR_CONTINUATIONS: dict[str, set[str]] = {
    # Common optics/journal abbreviations that are often split at ". " boundary.
    "opt": {"express", "lett", "letters"},
    "biomed": {"optics", "express"},
    "nat": {"commun", "communications"},
    "appl": {"optics", "phys"},
}

_VENUE_TOKEN_HINTS: set[str] = {
    "ieee",
    "acm",
    "journal",
    "transactions",
    "trans",
    "proceedings",
    "proc",
    "conference",
    "symposium",
    "workshop",
    "letters",
    "lett",
    "express",
    "communications",
    "commun",
    "review",
    "rev",
    "opt",
    "optics",
    "phys",
    "physics",
    "medical",
    "med",
    "imaging",
    "pattern",
    "analysis",
    "intelligence",
    "biomed",
    "appl",
    "applied",
    "nature",
    "science",
    "photonics",
}


def _looks_like_venue_phrase(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    low = s.lower()
    if len(re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", low)) < 2:
        return False
    if re.search(r"\b(cvpr|iccv|eccv|neurips|icml|iclr|aaai|ijcai|kdd|siggraph)\b", low):
        return True
    hint_hits = 0
    for tok in _VENUE_TOKEN_HINTS:
        if re.search(rf"\b{re.escape(tok)}\b", low):
            hint_hits += 1
            if hint_hits >= 1:
                break
    if hint_hits <= 0:
        return False
    # Keep this conservative; extremely long phrases are more likely title fragments.
    if len(low.split()) > 18:
        return False
    return True


def _merge_venue_head(tail: str, venue: str) -> str:
    tail_s = str(tail or "").strip(" .;,:")
    venue_s = str(venue or "").strip()
    if not tail_s:
        return venue_s
    if not venue_s:
        return tail_s
    tail_tokens = re.findall(r"[A-Za-z]{1,16}", tail_s)
    last_tok = str(tail_tokens[-1] or "").lower() if tail_tokens else ""
    if last_tok in {"opt", "nat", "appl", "biomed", "trans", "rev", "lett", "commun", "proc"}:
        if not tail_s.endswith("."):
            tail_s += "."
    return f"{tail_s} {venue_s}".strip()


def _repair_split_title_venue(title: str, venue: str) -> tuple[str, str]:
    t0 = str(title or "").strip()
    v0 = str(venue or "").strip()
    if (not t0) or (not v0):
        return t0, v0

    # First, try strict abbreviation continuations.
    m_tail = re.match(r"^(?P<base>.+?)\.\s*(?P<abbr>[A-Za-z]{2,10})$", t0)
    m_head = re.match(r"^(?P<head>[A-Za-z]{2,24})\b", v0) if m_tail else None
    if m_tail and m_head:
        abbr = str(m_tail.group("abbr") or "").strip()
        head = str(m_head.group("head") or "").strip()
        base = str(m_tail.group("base") or "").strip(" .;,:")
        if abbr and head and base:
            if head.lower() in _VENUE_SPLIT_ABBR_CONTINUATIONS.get(abbr.lower(), set()):
                return base, f"{abbr}. {v0}"

    # Generic repair for other venues:
    # "... <title>. IEEE Trans" + "Med. Imaging ..."
    # "... <title>. Nat" + "Commun ..."
    # "... <title>. J" + "Biomed. Opt. ..."
    m_tail2 = re.match(r"^(?P<base>.+?)\.\s*(?P<trail>[A-Za-z][A-Za-z.\- ]{0,42})$", t0)
    if not m_tail2:
        return t0, v0
    base2 = str(m_tail2.group("base") or "").strip(" .;,:")
    tail2 = str(m_tail2.group("trail") or "").strip(" .;,:")
    if not base2 or not tail2:
        return t0, v0
    base_words = [w for w in re.split(r"\s+", base2) if w]
    if len(base_words) < 3:
        return t0, v0

    tail_tokens = re.findall(r"[A-Za-z]{1,16}", tail2)
    if (not tail_tokens) or (len(tail_tokens) > 3):
        return t0, v0
    if not all((len(tok) <= 10) or (tok.lower() in _VENUE_TOKEN_HINTS) for tok in tail_tokens):
        return t0, v0

    merged_v = _merge_venue_head(tail2, v0)
    if _looks_like_venue_phrase(merged_v):
        return base2, merged_v
    return t0, v0


def _fallback_fill_reference_meta_from_raw(ref_rec: dict) -> dict:
    """Best-effort local parse of authors/title/venue from raw numbered references.

    This is a render-time fallback only, used when the reference index stores
    sparse metadata (e.g. venue/year/doi present but title empty).
    """
    if not isinstance(ref_rec, dict):
        return {}
    raw0 = _strip_reference_lead_label(str(ref_rec.get("raw") or "").strip())
    if not raw0:
        return {}

    # Normalize spacing but keep punctuation shape; parser relies on ". " splits.
    raw = re.sub(r"\s+", " ", raw0).strip()
    venue_hint = _strip_reference_lead_label(str(ref_rec.get("venue") or "").strip())
    pageish_re = r"(?:\d+(?:\s*[–-]\s*\d+)?|[A-Za-z]{0,8}\d[\w.-]*)"

    tail = ""
    prefix_core = ""
    tail_year = ""

    if venue_hint:
        # Prefer the last venue occurrence (safer when title contains venue-like tokens).
        try:
            venue_matches = list(re.finditer(re.escape(venue_hint), raw, flags=re.IGNORECASE))
        except Exception:
            venue_matches = []
        if venue_matches:
            m_venue = venue_matches[-1]
            prefix = raw[: m_venue.start()].rstrip()
            tail = raw[m_venue.start() :].strip()
            prefix_core = re.sub(r"[.;,:]\s*$", "", prefix).strip()

    if (not prefix_core) and raw:
        m_year = re.search(r"\((?:[^()]*,\s*)?(?P<year>(?:19|20)\d{2})\)\.?\s*$", raw)
        if not m_year:
            m_year = re.search(r"\b(?P<year>(?:19|20)\d{2})\b\.?\s*$", raw)
        if m_year:
            tail_year = str(m_year.group("year") or "").strip()
            split_at = raw.rfind(". ", 0, m_year.start())
            if split_at >= 0:
                prefix = raw[:split_at].rstrip()
                tail = raw[split_at + 2 :].strip()
                prefix_core = prefix.strip(" .;,:")

    if not prefix_core:
        prefix_core = raw.strip(" .;,:")
    if not prefix_core:
        return {}

    def _authors_like(s: str) -> bool:
        t = str(s or "").strip()
        if len(t) < 3:
            return False
        if not (("," in t) or (" et al" in t.lower()) or (" & " in t) or re.search(r"\band\b", t, flags=re.I)):
            return False
        if not re.search(r"\b[A-Z]\.", t):
            return False
        # If prose stopwords appear in the "authors" block, boundary is probably too late.
        if len(t) >= 32 and re.search(r"\b(using|via|with|through|for|from|into|onto|under|over|between|within)\b", t, flags=re.I):
            return False
        return True

    def _title_ok(s: str) -> bool:
        t = str(s or "").strip().strip(" .;,:")
        if len(t) < 4:
            return False
        if t.startswith(("&", ",")):
            return False
        if re.match(r"^(?:et al\.?|and)\b", t, flags=re.I):
            return False
        # Author-list continuation like "L. Video ..." or "A. B. Title ..."
        if re.match(r"^[A-Z]\.\s", t):
            return False
        if re.match(r"^(?:[A-Z]\.\s+){2,}", t):
            return False
        if venue_hint and t.lower() == venue_hint.lower():
            return False
        if _looks_noisy_reference_title(t):
            return False
        return True

    best: dict[str, str] = {}
    for m in re.finditer(r"\.\s+", prefix_core):
        a = prefix_core[: m.start()].strip().strip(" .;,:")
        t = prefix_core[m.end():].strip().strip(" .;,:")
        if _authors_like(a) and _title_ok(t):
            best = {"authors": a, "title": t}
            break

    if not best:
        # Last-chance fallback: recover at least title as tail before venue.
        m2 = re.search(r"\.\s+(.+)$", prefix_core)
        if m2:
            t2 = str(m2.group(1) or "").strip().strip(" .;,:")
            if _title_ok(t2):
                best = {"title": t2}

    tail_core = str(tail or "").strip().strip(" .;,:")
    if tail_core and str(best.get("title") or "").strip():
        try:
            title_fixed, tail_fixed = _repair_split_title_venue(str(best.get("title") or ""), tail_core)
        except Exception:
            title_fixed, tail_fixed = str(best.get("title") or "").strip(), tail_core
        if title_fixed and (title_fixed != str(best.get("title") or "").strip()):
            best["title"] = title_fixed
        if tail_fixed:
            tail_core = tail_fixed

    if tail_core:
        year_m = re.search(r"\((?:[^()]*,\s*)?(?P<year>(?:19|20)\d{2})\)\s*$", tail_core)
        if year_m:
            best.setdefault("year", str(year_m.group("year") or "").strip())
            tail_no_year = tail_core[: year_m.start()].rstrip(" ,.;:")
        else:
            if tail_year:
                best.setdefault("year", tail_year)
                tail_no_year = re.sub(r"\b(?:19|20)\d{2}\b\.?\s*$", "", tail_core).rstrip(" ,.;:")
            else:
                tail_no_year = tail_core

        if tail_no_year.lower().startswith("in "):
            m_in_pages = re.match(rf"^(?P<venue>In .+?)\s+(?P<pages>{pageish_re})$", tail_no_year, flags=re.I)
            if m_in_pages:
                venue_p = str(m_in_pages.group("venue") or "").strip(" .;,:")
                pages_p = str(m_in_pages.group("pages") or "").strip(" .;,:")
                if venue_p:
                    best.setdefault("venue", venue_p)
                if pages_p:
                    best.setdefault("pages", pages_p)
            else:
                venue_p = tail_no_year.strip(" .;,:")
                if venue_p:
                    best.setdefault("venue", venue_p)
        else:
            m_venue = re.match(
                rf"^(?P<venue>.+?)(?:\s+(?P<volume>\d+[A-Za-z]?))?(?:\s*,\s*(?P<pages>{pageish_re}))?$",
                tail_no_year,
            )
            if m_venue:
                venue_p = str(m_venue.group("venue") or "").strip(" .;,:")
                volume_p = str(m_venue.group("volume") or "").strip(" .;,:")
                pages_p = str(m_venue.group("pages") or "").strip(" .;,:")
                if venue_p:
                    best.setdefault("venue", venue_p)
                if volume_p:
                    best.setdefault("volume", volume_p)
                if pages_p:
                    best.setdefault("pages", pages_p)

    return best


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

    if raw and ((not title) or (not authors)):
        try:
            parsed = _fallback_fill_reference_meta_from_raw(
                {
                    "raw": raw,
                    "venue": venue,
                    "title": title,
                    "authors": authors,
                }
            )
        except Exception:
            parsed = {}
        if not title:
            title_p = _strip_reference_lead_label(str((parsed or {}).get("title") or "").strip())
            if title_p and (not _looks_noisy_reference_title(title_p)):
                title = title_p
        if not authors:
            authors_p = _strip_reference_lead_label(str((parsed or {}).get("authors") or "").strip())
            if authors_p:
                authors = authors_p
        if not venue:
            venue_p = _strip_reference_lead_label(str((parsed or {}).get("venue") or "").strip())
            if venue_p:
                venue = venue_p
        if not year:
            year_p = str((parsed or {}).get("year") or "").strip()
            if year_p:
                year = year_p
        if not volume:
            volume_p = str((parsed or {}).get("volume") or "").strip()
            if volume_p:
                volume = volume_p
        if not pages:
            pages_p = str((parsed or {}).get("pages") or "").strip()
            if pages_p:
                pages = pages_p

    out["title"] = title
    out["authors"] = authors
    out["venue"] = venue
    out["year"] = year
    out["volume"] = volume
    out["issue"] = issue
    out["pages"] = pages
    out["doi"] = doi
    out["raw"] = raw
    if title and venue:
        try:
            title_fix, venue_fix = _repair_split_title_venue(title, venue)
        except Exception:
            title_fix, venue_fix = title, venue
        if title_fix:
            out["title"] = title_fix
        if venue_fix:
            out["venue"] = venue_fix
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
        out = _STRUCT_CITE_RE.sub("", text)
        out = _STRUCT_CITE_SINGLE_RE.sub("", out)
        out = _STRUCT_CITE_SID_ONLY_RE.sub("", out)
        return _STRUCT_CITE_GARBAGE_RE.sub("", out)

    srcs = _collect_source_paths_from_hits(hits or [], max_docs=16)
    if not srcs:
        return _strip_unresolved_structured_tokens(s), []
    source_hint_by_path: dict[str, dict] = {}
    source_hit_weight_by_path: dict[str, float] = {}
    for h in hits or []:
        meta_h = (h or {}).get("meta", {}) or {}
        sp_h = str(meta_h.get("source_path") or "").strip()
        if not sp_h or _is_temp_source_path(sp_h):
            continue
        source_hit_weight_by_path[sp_h] = float(source_hit_weight_by_path.get(sp_h, 0.0) or 0.0) + 1.0
        rec = source_hint_by_path.get(sp_h)
        if not isinstance(rec, dict):
            rec = {}
            source_hint_by_path[sp_h] = rec
        sha1_h = str(meta_h.get("source_sha1") or "").strip().lower()
        if sha1_h and (not str(rec.get("source_sha1") or "").strip()):
            rec["source_sha1"] = sha1_h
    sid_to_source: dict[str, str] = {}
    for sp in srcs:
        sid = _source_cite_id(sp).lower()
        sid_to_source[sid] = sp
    dominant_source_path = ""
    if len(srcs) == 1:
        dominant_source_path = str(srcs[0])
    elif source_hit_weight_by_path:
        ranked_sources = sorted(
            source_hit_weight_by_path.items(),
            key=lambda kv: float(kv[1] or 0.0),
            reverse=True,
        )
        if ranked_sources:
            top_sp, top_w = ranked_sources[0]
            sec_w = float(ranked_sources[1][1]) if len(ranked_sources) > 1 else 0.0
            if float(top_w) >= max(2.0, sec_w * 1.35):
                dominant_source_path = str(top_sp)

    index_data = _load_reference_index_cached()
    if not isinstance(index_data, dict):
        index_data = {}

    resolved_cache: dict[tuple[int, str], tuple[str, str, dict] | None] = {}
    candidate_cache: dict[tuple[int, str], list[tuple[str, str, dict]]] = {}
    detail_by_key: dict[str, dict] = {}

    def _resolve_num_candidates(n: int, preferred_sp: str = "") -> list[tuple[str, str, dict]]:
        pref = str(preferred_sp or "").strip()
        ckey = (int(n), pref.lower())
        cached = candidate_cache.get(ckey)
        if isinstance(cached, list):
            return list(cached)
        matches: list[tuple[str, str, dict]] = []
        ordered_srcs = list(srcs)
        if pref and pref in ordered_srcs:
            ordered_srcs = [pref] + [x for x in ordered_srcs if x != pref]
        for sp in ordered_srcs:
            hint = source_hint_by_path.get(sp) or {}
            got = _resolve_reference_entry_from_index(
                index_data,
                sp,
                int(n),
                source_sha1=str(hint.get("source_sha1") or "").strip().lower(),
            )
            if isinstance(got, dict):
                ref = got.get("ref")
                if isinstance(ref, dict):
                    matches.append((sp, _display_source_name(sp), ref))
        candidate_cache[ckey] = list(matches)
        return list(matches)

    def _resolve_num(n: int, preferred_sp: str = "") -> tuple[str, str, dict] | None:
        pref = str(preferred_sp or "").strip()
        ckey = (int(n), pref.lower())
        if ckey in resolved_cache:
            return resolved_cache[ckey]
        matches = _resolve_num_candidates(int(n), preferred_sp=pref)

        picked: tuple[str, str, dict] | None = None
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

    def _remember_detail(n: int, source_path: str, source_name: str, ref: dict) -> dict:
        skey = f"{int(n)}|{str(source_path or '').strip().lower()}"
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
            "source_path": str(source_path or "").strip(),
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

    def _replace_text_segment(seg: str, *, table_mode: bool = False) -> str:
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
            if table_mode:
                # Avoid markdown-table column splits caused by "|" inside title text.
                return f"[{int(n)}](#{anchor})"
            t_attr = str(title_attr or "").replace('"', "'").replace("\n", " ").strip()
            return f"[{int(n)}](#{anchor} \"{t_attr}\")"

        def _pick_grounded_numeric_candidate(
            n: int,
            *,
            pos: int,
            target_sp: str,
        ) -> tuple[str, str, dict] | None:
            pref_sp = str(target_sp or "").strip()
            matches = _resolve_num_candidates(int(n), preferred_sp=pref_sp)
            if not matches:
                return None
            if not pref_sp and len(matches) == 1:
                return matches[0]

            hints = extract_citation_context_hints(seg, token_start=int(pos), token_end=int(pos) + max(1, len(f"[{int(n)}]")))
            doi_hint = str(hints.get("doi") or "").strip()
            if not doi_hint:
                # Preserve legacy behavior: for free-form numeric citations like [50],
                # do not hard-gate by author/year text (it is often not reliable),
                # but allow DOI to disambiguate or drop on explicit conflict.
                return _resolve_num(int(n), preferred_sp=pref_sp)

            best: tuple[str, str, dict] | None = None
            best_score = float("-inf")
            for cand in matches:
                ref = cand[2]
                score = float(reference_alignment_score(ref, hints))
                if str(cand[0] or "").strip() == pref_sp:
                    score += 0.1
                if score > best_score:
                    best_score = score
                    best = cand
            if not best:
                return None
            if has_explicit_reference_conflict(best[2], hints):
                return None
            # DOI is treated as a hard identity signal.
            return best if best_score >= 6.0 else None

        def _resolve_struct_token(sid_raw: str, n_raw: str, *, pos: int = -1) -> str:
            nonlocal structured_seen
            sid = str(sid_raw or "").strip().lower()
            try:
                n = int(n_raw)
            except Exception:
                return ""
            structured_seen = True
            sp = sid_to_source.get(sid) or sid_to_source.get(sid.lower())
            if not sp:
                return ""
            hint = source_hint_by_path.get(sp) or {}
            got = _resolve_reference_entry_from_index(
                index_data,
                sp,
                int(n),
                source_sha1=str(hint.get("source_sha1") or "").strip().lower(),
            )
            if not isinstance(got, dict):
                return ""
            ref = got.get("ref")
            if not isinstance(ref, dict):
                return ""
            src_name = _display_source_name(sp)
            detail = _remember_detail(int(n), sp, src_name, ref)
            title_attr = _citation_hover_title(src_name, int(n), ref)
            return _mk_cite_link_md(int(n), detail, title_attr)

        def _repl_struct(m: re.Match) -> str:
            return _resolve_struct_token(str(m.group(1) or ""), str(m.group(2) or ""), pos=int(m.start()))

        def _repl_struct_single(m: re.Match) -> str:
            sid = str(m.group(1) or "")
            n_txt = str(m.group(2) or "").strip()
            if not n_txt:
                # Malformed form like [CITE:sid] -> hide raw token.
                return ""
            return _resolve_struct_token(sid, n_txt, pos=int(m.start()))

        def _repl_struct_sid_only(_: re.Match) -> str:
            # Malformed form like [[CITE:sid]] -> hide raw token.
            return ""

        def _repl_any(m: re.Match) -> str:
            raw = str(m.group(0) or "")
            spec = str(m.group(1) or "").strip()
            nums = _parse_int_set(spec)[:40]
            if not nums:
                return raw
            target_sp = str(dominant_source_path or "").strip()
            if not target_sp:
                pref_sp = _preferred_source_by_context(int(m.start()))
                if pref_sp:
                    target_sp = pref_sp
            items: list[str] = []
            changed = False
            for n in nums:
                picked = _pick_grounded_numeric_candidate(
                    int(n),
                    pos=int(m.start()),
                    target_sp=target_sp,
                )
                if not picked:
                    if not _STRICT_STRUCTURED_CITATION_LINKING:
                        items.append(f"[{int(n)}]")
                    continue
                sp_picked, src_name, ref = picked
                detail = _remember_detail(int(n), sp_picked, src_name, ref)
                title_attr = _citation_hover_title(src_name, int(n), ref)
                items.append(_mk_cite_link_md(int(n), detail, title_attr))
                changed = True
            if not changed:
                return "" if _STRICT_STRUCTURED_CITATION_LINKING else raw
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
        if is_table_sep:
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
                    rebuilt_math.append(_replace_text_segment(mp, table_mode=is_table_row))
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
            "source_path": str(rec.get("source_path") or "").strip(),
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
    '（式(n) 对应命中的库内文献：filename.pdf）'
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
            _ref_rank, label = picked
            safe_label = str(label or "").strip()
            if safe_label:
                out.append(f"*（式({int(tag_n)}) 对应命中的库内文献：`{safe_label}`）*")
        out.append("")
        i = j + 1

    return "\n".join(out)


def _parse_int_set(spec: str) -> list[int]:
    """
    Parse: "45,46-49  52" -> [45,46,47,48,49,52]
    """
    s = (spec or "").strip()
    if not s:
        return []
    s = s.replace("，", ",").replace("、", ",").replace(";", ",")
    parts = re.split(r"[,\s]+", s)
    out: set[int] = set()
    for p in parts:
        t = (p or "").strip()
        if not t:
            continue
        t = t.replace("–", "-").replace("—", "-")
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


def _render_citation_ui(uid: str, source_path: str, key_ns: str) -> None:
    net_key = f"{key_ns}_net_meta_v6_{uid}"
    net_data, fetch_failed, pending, changed = _sync_citation_task_state(net_key)
    if (not net_data) and (not fetch_failed):
        _ensure_citation_task(net_key, source_path)
        net_data, fetch_failed, pending, changed2 = _sync_citation_task_state(net_key)
        changed = changed or changed2

    if changed:
        st.experimental_rerun()
        return

    if (not net_data) and pending:
        st.markdown(
            "<div class='citation-loading'>妫€绱腑...</div>",
            unsafe_allow_html=True,
        )
        time.sleep(0.5)
        st.experimental_rerun()
        return

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

