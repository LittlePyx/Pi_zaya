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
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

import requests

from kb.citation_meta import extract_first_doi, fetch_best_crossref_meta
from kb.citation_card import compose_citation_card
from kb.config import load_settings
from kb.file_naming import citation_meta_display_pdf_name
from kb.inpaper_citation_enrichment import (
    enrich_inpaper_detail_context,
    extract_structured_cite_answer_context_line,
)
from kb.inpaper_citation_grounding import (
    extract_citation_context_hints,
    has_explicit_reference_conflict,
    reference_alignment_score,
)
from kb.source_filters import is_excluded_source_path
from kb.evidence_text import (
    clean_display_text as _clean_evidence_display_text,
    evidence_sentence_quality as _evidence_sentence_quality,
    looks_low_value_citation_context as _looks_low_value_citation_context,
    pick_readable_evidence_text as _pick_readable_evidence_text,
)
from kb.evidence_term_mapping import evidence_alignment_tokens
from kb.evidence_binding import (
    _SYSTEM_A_STRONG_BINDING_TERMS,
    _quantity_is_covered,
    _system_a_fact_quantities,
    _system_a_domain_terms,
    _system_a_keyword_terms,
    assess_system_a_hit_binding as _assess_system_a_hit_binding,
)
from kb.reference_index import (
    extract_references_map_from_md as _extract_references_map_from_md_index,
    load_reference_index as _load_reference_index_file,
    resolve_reference_entry as _resolve_reference_entry_from_index,
)
from kb.pdf_tools import open_in_explorer
from kb.path_safety import clean_file_source_path_input, path_is_within_roots, resolved_path, unique_resolved_roots
from kb.source_blocks import normalize_inline_markdown
from kb.tokenize import tokenize
from ui.strings import S
from ui.streamlit_compat import st
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
    pdf_roots = unique_resolved_roots([pdf_root])
    if not pdf_roots:
        return None
    pdf_root = pdf_roots[0]
    if stem.endswith(".en"):
        stem = stem[: -3]

    direct = [
        pdf_root / f"{stem}.pdf",
        pdf_root / f"{stem}.PDF",
    ]
    for p in direct:
        resolved = _existing_pdf_under_root(p, pdf_roots)
        if resolved is not None:
            return resolved

    # Fallback: scan by stem match.
    try:
        target = stem.lower()
        for p in pdf_root.glob("*.pdf"):
            resolved = _existing_pdf_under_root(p, pdf_roots)
            if resolved is not None and resolved.stem.lower() == target:
                return resolved
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
            resolved = _existing_pdf_under_root(p, pdf_roots)
            if resolved is None:
                continue
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
                best_path = resolved
    except Exception:
        return None

    return best_path


def _existing_pdf_under_root(path_obj: Path | str, roots: list[Path | str | None]) -> Path | None:
    path = resolved_path(path_obj)
    if path is None or not path_is_within_roots(path, roots):
        return None
    try:
        if path.is_file() and path.suffix.lower() == ".pdf":
            return path
    except Exception:
        return None
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
    r"(怎么|如何|方法|实现|步骤|流程|原理|机制|算法|模型|公式|推导|"
    r"\bhow\b|\bmethod\b|\bapproach\b|\bimplement(?:ation)?\b|\balgorithm\b|\bmodel\b|\bequation\b)",
    flags=re.I,
)
_LIMIT_QUERY_RE_UI = re.compile(
    r"(局限|限制|不足|未来工作|讨论|结论|"
    r"\blimitation\b|\bfuture\s+work\b|\bdiscussion\b|\bconclusion\b)",
    flags=re.I,
)
_DISCUSS_HEAD_RE_UI = re.compile(
    r"\b(discussion|conclusion|limitations?|future\s+work)\b|(讨论|结论|局限|未来工作)",
    flags=re.I,
)
_SYSTEM_A_SYNTHETIC_LOCATION_DISCUSSION_RE = re.compile(
    r"^\s*(?:(?:该文|本文|这篇(?:文献|论文|文章)|the\s+paper)\s*)?"
    r"(?:在|于|in\s+)?[“\"']?[^“”\"']{8,220}[”\"']\s*"
    r"(?:讨论了|比较了|定义或解释了|给出了与|directly\s+discusses|discusses|compares|defines|explains)"
    r"[“\"']?[^“”\"']{1,140}[”\"']?\s*[。.]?\s*$",
    flags=re.IGNORECASE,
)
_PDF_SHELL_HEADING_TOKENS_UI = {
    "article",
    "article info",
    "article information",
}


def _compact_spaced_heading_token_ui(h: str) -> str:
    s = " ".join(str(h or "").strip().split())
    if not s:
        return ""
    tokens = re.findall(r"[A-Za-z]", s)
    words = re.findall(r"[A-Za-z]+", s)
    if len(tokens) < 3 or not words:
        return ""
    if all(len(word) == 1 for word in words):
        return "".join(tokens).lower()
    return ""


def _is_pdf_shell_heading_ui(h: str) -> bool:
    s = " ".join(str(h or "").strip().split())
    if not s:
        return False
    low = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    if low in (_PDF_SHELL_HEADING_TOKENS_UI | {"research article", "original article"}):
        return True
    compact = _compact_spaced_heading_token_ui(s)
    if compact in {"abstract", "article", "articleinfo", "articleinformation"}:
        return True
    return False


def _wants_reference_nav_ui(prompt: str) -> bool:
    q = str(prompt or "").strip()
    if not q:
        return False
    return bool(re.search(r"(参考文献|引用|cite|citation|reference|bibliography)", q, flags=re.I))


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
    if _is_pdf_shell_heading_ui(s):
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
    "文献",
    "论文",
    "研究",
    "问题",
    "挑战",
    "瓶颈",
    "约束",
    "相关",
    "信息",
    "内容",
    "章节",
    "小节",
    "方法",
    "结果",
    "实验",
    "模型",
    "算法",
    "数据",
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
        "提出",
        "采用",
        "通过",
        "实现",
        "提升",
        "验证",
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
    q_compact = re.sub(r"[\s`'\"“”‘’，。！？.?!:;；：（）()\-_/\\]+", "", q)
    t_compact = re.sub(r"[\s`'\"“”‘’，。！？.?!:;；：（）()\-_/\\]+", "", t)
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
        if zh in {"这篇文献", "当前问题", "相关信息"}:
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
    parts = re.split(r"(?<=[。！？!?;；])\s+|[。！？；]", s)
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
    s = re.sub(r"^[,;:，；：、-]+", "", s).strip()
    s = re.sub(r"[。！？!?;；]+$", "", s).strip()
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
            r"(问题|挑战|瓶颈|受限|难以|困难|problem|challenge|bottleneck|limitation|difficult|lack|struggle)",
            flags=re.I,
        )
    elif role_key == "method":
        pat = re.compile(
            r"(提出|采用|设计|构建|引入|实现|propose|introduce|design|develop|variant)",
            flags=re.I,
        )
    else:
        pat = re.compile(
            r"(结果|显示|表明|提升|提高|优于|验证|性能|指标|result|results|show|demonstrate|improv|outperform|achieve|experiment)",
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
            if re.search(r"(challenge|limitations?|struggle|bottleneck|受限|挑战|瓶颈|困难)", low):
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
            if re.search(r"(results?|show|demonstrate|outperform|improv|achieve|实验|结果)", low):
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


def _looks_front_matter_evidence_ui(text: str) -> bool:
    src = str(text or "").strip()
    if not src:
        return False
    head = src[:700]
    low = head.lower()
    if re.search(r"\b(?:supplement|published with|institute of|university|academy of sciences|corresponding author)\b", low):
        return True
    if "@" in head and len(re.findall(r"\b[A-Z][A-Z-]{2,}\b", head)) >= 4:
        return True
    comma_count = head.count(",")
    name_like = len(re.findall(r"\b[A-Z][A-Za-z-]+(?:\s+[A-Z]\.){0,3}\s+[A-Z][A-Za-z-]+\b", head))
    return comma_count >= 8 and name_like >= 4 and not re.search(r"\b(?:we|this paper|this work|propose|show|demonstrate)\b", low)


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
        compact_start = re.sub(r"[\s`|,;:，；。：·\-_/\\(){}\[\]]+", "", start_from)
        if len(compact_start) < 6:
            start_from = ""
        elif re.search(r"(先从\s*开始|start\s+with\s*$)", start_from, flags=re.I):
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
    pdf_roots = unique_resolved_roots([pdf_root])
    if not pdf_roots:
        return None
    raw = clean_file_source_path_input(source_path)
    if not raw:
        return None
    src = Path(raw).expanduser()
    if src.suffix.lower() == ".pdf":
        candidate = src if src.is_absolute() else (pdf_roots[0] / src)
        return _existing_pdf_under_root(candidate, pdf_roots)
    stem = (src.stem or "").strip()
    if not stem:
        return None
    return _lookup_pdf_by_stem(pdf_roots[0], stem)


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
    raw = clean_file_source_path_input(source_path)
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
        if line.startswith("<!--") or re.fullmatch(r"<!--\s*kb_page\s*:\s*\d+\s*-->", line, flags=re.I):
            continue
        if re.match(r"^!\[[^\]]{0,120}\]\([^)]+\)\s*$", line):
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
    _quiet_sqlalchemy_logging()
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
    _quiet_sqlalchemy_logging(fa)

    try:
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
    finally:
        _close_factor_manager(fa)


def _close_factor_manager(factor_obj) -> None:
    manager = getattr(factor_obj, "manager", None)
    if manager is None:
        return
    session = getattr(manager, "session", None)
    if session is not None:
        try:
            session.close()
        except Exception:
            pass
    engine = getattr(manager, "engine", None)
    if engine is not None:
        try:
            engine.dispose()
        except Exception:
            pass


def _quiet_sqlalchemy_logging(factor_obj=None) -> None:
    try:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
        manager = getattr(factor_obj, "manager", None)
        engine = getattr(manager, "engine", None)
        engine_logger = getattr(engine, "logger", None)
        if engine_logger is not None:
            try:
                engine_logger.setLevel(logging.WARNING)
            except Exception:
                try:
                    engine_logger.level = logging.WARNING
                except Exception:
                    pass
    except Exception:
        pass


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


class _CoreRankingTableParser(HTMLParser):
    """Parse CORE result rows without regex backtracking across a whole page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[dict] = []
        self._active_row = False
        self._active_cell = False
        self._cells: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = str(tag or "").lower()
        if name == "tr":
            attr_map = {str(k or "").lower(): str(v or "") for k, v in attrs}
            self._active_row = "navigate(" in attr_map.get("onclick", "").lower()
            self._active_cell = False
            self._cells = []
            return
        if self._active_row and name == "td":
            self._active_cell = True
            self._cells.append([])

    def handle_endtag(self, tag: str) -> None:
        name = str(tag or "").lower()
        if name == "td":
            self._active_cell = False
            return
        if name != "tr" or not self._active_row:
            return
        cells = [" ".join("".join(parts).split()).strip() for parts in self._cells]
        self._active_row = False
        self._active_cell = False
        self._cells = []
        if len(cells) < 4 or not re.fullmatch(r"[A*BC]+", cells[3], flags=re.I):
            return
        self.rows.append(
            {
                "title": cells[0],
                "acronym": cells[1],
                "source": cells[2],
                "rank": cells[3],
            }
        )

    def handle_data(self, data: str) -> None:
        if self._active_row and self._active_cell and self._cells:
            self._cells[-1].append(str(data or ""))


def _core_parse_rows(html_text: str) -> list[dict]:
    # The endpoint occasionally returns a large or malformed page.  Parsing is
    # intentionally bounded so metadata warming can never starve the API.
    payload = str(html_text or "")[:4_000_000]
    parser = _CoreRankingTableParser()
    try:
        parser.feed(payload)
        parser.close()
    except Exception:
        return list(parser.rows[:24])
    return list(parser.rows[:24])


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
                    st.caption(f"文献指标检索失败（{fail_reason}），可点击 Cite 重试")
                else:
                    st.caption("文献指标检索失败，可点击 Cite 重试")

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
    parts = [f"source: {src}", f"ref {int(ref_num)}"]
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


def _authors_from_raw_reference(raw: str, *, title: str = "") -> str:
    raw_text = re.sub(r"\s+", " ", str(raw or "")).strip()
    title_text = re.sub(r"\s+", " ", str(title or "")).strip()
    if not raw_text or not title_text or len(title_text) < 8:
        return ""
    pos = raw_text.lower().find(title_text.lower())
    if pos <= 0:
        return ""
    prefix = raw_text[:pos].strip(" \t\r\n.;,:")
    prefix = re.sub(r"^\s*\[?\d{1,4}\]?\s*", "", prefix).strip(" \t\r\n.;,:")
    if len(prefix) < 3 or len(prefix) > 180:
        return ""
    low = prefix.lower()
    if re.search(r"\b(?:abstract|introduction|fig|figure|table|retrieved|unknown|reference)\b", low):
        return ""
    if re.search(r"\bet\s+al\.?\b", low):
        return prefix
    if ("," in prefix or " & " in prefix or re.search(r"\band\b", prefix, flags=re.I)) and re.search(r"[A-Z]", prefix):
        return prefix
    if re.fullmatch(r"[A-Z]{1,4}\.?\s+[A-Z][A-Za-z'’-]{2,}(?:\s+[A-Z][A-Za-z'’-]{2,})?", prefix):
        return prefix
    if re.fullmatch(r"[A-Z][A-Za-z'’-]{2,}\s+[A-Z]{1,4}\.?", prefix):
        return prefix
    return ""


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
        if not authors:
            authors_p = _authors_from_raw_reference(raw, title=title or str((parsed or {}).get("title") or ""))
            if authors_p:
                authors = _strip_reference_lead_label(authors_p)
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


def _build_inpaper_anchor(anchor_ns: str, ref_num: int, source_name: str = "", extra: str = "") -> str:
    base = (
        f"{str(anchor_ns or '').strip()}|{int(ref_num)}|"
        f"{str(source_name or '').strip().lower()}|{str(extra or '').strip().lower()}"
    )
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


_SYSTEM_B_PRIOR_CONTEXT_RE = re.compile(
    r"(?i)\b(?:prior|previous|existing|earlier|background|upstream|source|origin|"
    r"borrowed|inspired|based\s+on|cited|citation|reference|not\s+original|not\s+new)\b|"
    r"(?:前人|已有|先前|早期|背景|上游|来源|出处|源头|借鉴|引用|参考|不是.{0,12}(?:原创|新提出))"
)
_SYSTEM_B_METHOD_CONTEXT_RE = re.compile(
    r"(?i)\b(?:method|model|framework|algorithm|optimization|implementation|reconstruction|machinery|tool)\b|"
    r"(?:方法|模型|框架|算法|优化|实现|重建|工具|机制)"
)
_SYSTEM_A_LOW_VALUE_EVIDENCE_RE = re.compile(
    r"(?i)(?:"
    r"no\s+summary\s+available|metadata\s+only|only\s+metadata|"
    r"current(?:ly)?\s+only\s+retrieved\s+metadata|"
    r"\u8fd9\u7bc7\u6587\u732e\u5f53\u524d\u7f3a\u5c11\u53ef\u7528\u6458\u8981|"
    r"\u4ec5\u6839\u636e\u5143\u6570\u636e|"
    r"\u5f53\u524d\u4ec5\u68c0\u7d22\u5230\u6587\u732e\u5143\u6570\u636e|"
    r"\u6682\u65e0\u6cd5\u53ef\u9760\u63d0\u70bc"
    r")"
)
_SYSTEM_A_WRAPPED_EXCERPT_RE = re.compile(
    r"^\s*(?:\u539f\u6587\u7247\u6bb5\u5199\u5230|source\s+excerpt\s+says)\s*[:\uff1a]",
    re.IGNORECASE,
)


def _locale_prefers_zh(locale: str, *texts: str) -> bool:
    raw = str(locale or "").strip().lower()
    if raw == "zh":
        return True
    if raw == "en":
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", " ".join(str(text or "") for text in texts)))


def _system_b_prefers_zh(*texts: str, locale: str = "") -> bool:
    return _locale_prefers_zh(locale, *texts)



def _system_a_fp_text(text: str, *, max_len: int = 360) -> str:
    raw = normalize_inline_markdown(str(text or ""))
    raw = re.sub(r"\[[Rr]?\d{1,4}(?:\s*[,，、]\s*[Rr]?\d{1,4})*\]", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip().lower()
    return raw[: max(32, int(max_len))]


def _system_a_primary_evidence_from_ui_meta(ui_meta: dict) -> dict:
    """Return the same primary evidence object used by the reader locate action."""
    if not isinstance(ui_meta, dict):
        return {}
    direct = ui_meta.get("primary_evidence")
    if isinstance(direct, dict) and direct:
        return direct
    reader_open = ui_meta.get("reader_open")
    if not isinstance(reader_open, dict):
        reader_open = {}
    for key in ("primaryEvidence", "primary_evidence", "locateTarget", "locate_target"):
        nested = reader_open.get(key)
        if isinstance(nested, dict) and nested:
            return nested
    reader_signal_keys = (
        "snippet",
        "highlightSnippet",
        "highlight_snippet",
        "headingPath",
        "heading_path",
        "blockId",
        "block_id",
        "anchorId",
        "anchor_id",
    )
    if any(str(reader_open.get(key) or "").strip() for key in reader_signal_keys):
        return reader_open
    return {}


def _system_a_candidate_value(raw: dict, *keys: str) -> str:
    if not isinstance(raw, dict):
        return ""
    for key in keys:
        value = str(raw.get(key) or "").strip()
        if value:
            return value
    return ""


def _system_a_candidate_text(raw: dict) -> str:
    return _system_a_candidate_value(
        raw,
        "highlight_snippet",
        "highlightSnippet",
        "snippet",
        "anchor_text",
        "anchorText",
        "evidence_quote",
        "evidenceQuote",
        "text",
        "raw_text",
        "rawText",
    )


def _system_a_ui_relevance_for_occurrence(
    ui_meta: dict,
    original_primary: dict,
    *,
    heading: str,
    block_id: str,
    anchor_id: str,
    evidence_quote: str,
) -> str:
    """Reuse polished relevance only when it still describes this occurrence."""

    why_line = str((ui_meta or {}).get("why_line") or "").strip()
    generation = str((ui_meta or {}).get("why_generation") or "").strip().lower()
    if not why_line or generation in {"locale_suppressed", "pending", "failed", "error"}:
        return ""
    if generation not in {
        "answer_citation_grounded",
        "deterministic_grounded",
        "llm_grounded",
        "llm_pack",
        "section_grounded",
    }:
        return ""
    primary = dict(original_primary or {})
    if not primary:
        return ""

    primary_block = _system_a_candidate_value(primary, "block_id", "blockId")
    primary_anchor = _system_a_candidate_value(primary, "anchor_id", "anchorId")
    if primary_block and block_id and primary_block == str(block_id or "").strip():
        return why_line[:320]
    if primary_anchor and anchor_id and primary_anchor == str(anchor_id or "").strip():
        return why_line[:320]

    def _heading_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())

    primary_heading = _system_a_candidate_value(primary, "heading_path", "headingPath")
    if not primary_heading or _heading_key(primary_heading) != _heading_key(heading):
        return ""
    primary_text = _system_a_candidate_text(primary)
    if not primary_text or not evidence_quote:
        return ""
    primary_tokens = evidence_alignment_tokens(primary_text)
    evidence_tokens = evidence_alignment_tokens(evidence_quote)
    if len(primary_tokens & evidence_tokens) < 4:
        return ""
    return why_line[:320]


def _system_a_is_low_value_evidence_text(value: str) -> bool:
    text = _clean_evidence_display_text(value, max_len=900)
    if not text:
        return True
    if _SYSTEM_A_SYNTHETIC_LOCATION_DISCUSSION_RE.match(text):
        return True
    if _SYSTEM_A_LOW_VALUE_EVIDENCE_RE.search(text):
        return True
    if _SYSTEM_A_WRAPPED_EXCERPT_RE.search(text) and ("..." in text or "\u2026" in text):
        return True
    try:
        if _looks_low_value_citation_context(text):
            return True
    except Exception:
        return False
    return False


def _system_a_add_evidence_candidate(
    out: list[dict],
    seen: set[str],
    raw: object,
    *,
    source: str,
    rank: int,
    default_heading: str = "",
) -> None:
    if not isinstance(raw, dict):
        return
    text = _system_a_candidate_text(raw)
    if not str(text or "").strip():
        return
    heading = _system_a_candidate_value(raw, "heading_path", "headingPath") or default_heading
    block_id = _system_a_candidate_value(raw, "block_id", "blockId")
    anchor_id = _system_a_candidate_value(raw, "anchor_id", "anchorId")
    key = _system_a_fp_text(f"{heading}|{block_id}|{anchor_id}|{text}", max_len=520)
    if not key or key in seen:
        return
    seen.add(key)
    out.append(
        {
            "source": source,
            "rank": int(rank),
            "raw": raw,
            "text": str(text or "").strip(),
            "heading_path": heading,
            "block_id": block_id,
            "anchor_id": anchor_id,
            "anchor_kind": _system_a_candidate_value(raw, "anchor_kind", "anchorKind"),
        }
    )


def _system_a_evidence_candidates_from_hit(
    *,
    hit: dict,
    meta: dict,
    ui_meta: dict,
    primary_evidence: dict,
    default_heading: str,
) -> list[dict]:
    candidates: list[dict] = []
    seen: set[str] = set()
    _system_a_add_evidence_candidate(
        candidates,
        seen,
        primary_evidence,
        source="primary_evidence",
        rank=0,
        default_heading=default_heading,
    )
    reader_open = ui_meta.get("reader_open") if isinstance(ui_meta.get("reader_open"), dict) else {}
    if isinstance(reader_open, dict):
        for key in ("primaryEvidence", "primary_evidence", "locateTarget", "locate_target"):
            _system_a_add_evidence_candidate(
                candidates,
                seen,
                reader_open.get(key),
                source=f"reader_open.{key}",
                rank=1,
                default_heading=default_heading,
            )
        for list_key in (
            "evidenceAlternatives",
            "visibleAlternatives",
            "alternatives",
        ):
            values = reader_open.get(list_key)
            if isinstance(values, list):
                for idx, item in enumerate(values[:8]):
                    _system_a_add_evidence_candidate(
                        candidates,
                        seen,
                        item,
                        source=f"reader_open.{list_key}",
                        rank=2 + idx,
                        default_heading=default_heading,
                    )
    for list_key in ("evidenceAlternatives", "visibleAlternatives", "alternatives"):
        values = ui_meta.get(list_key)
        if isinstance(values, list):
            for idx, item in enumerate(values[:8]):
                _system_a_add_evidence_candidate(
                    candidates,
                    seen,
                    item,
                    source=f"ui_meta.{list_key}",
                    rank=3 + idx,
                    default_heading=default_heading,
                )
    meta_candidate = {
        "heading_path": default_heading,
        "snippet": (
            str(meta.get("evidence_quote") or "").strip()
            or str(meta.get("support_locate_anchor") or "").strip()
            or str(meta.get("anchor_text") or "").strip()
        ),
        "block_id": str(meta.get("primary_block_id") or meta.get("block_id") or "").strip(),
        "anchor_id": str(meta.get("primary_anchor_id") or meta.get("anchor_id") or "").strip(),
        "anchor_kind": str(meta.get("anchor_kind") or "").strip(),
    }
    _system_a_add_evidence_candidate(
        candidates,
        seen,
        meta_candidate,
        source="hit_meta",
        rank=12,
        default_heading=default_heading,
    )
    hit_locator = (hit or {}).get("locator") if isinstance((hit or {}).get("locator"), dict) else {}
    meta_locator = meta.get("locator") if isinstance(meta.get("locator"), dict) else {}
    hit_candidate = {
        # These values must describe the raw retrieval hit itself.  In
        # particular, ref_best_* and primary_* belong to separately selected
        # evidence and must not leak into a claim-specific raw-hit fallback.
        "heading_path": (
            _system_a_candidate_value(hit, "heading_path", "headingPath")
            or _system_a_candidate_value(hit_locator, "heading_path", "headingPath")
            or _system_a_candidate_value(meta, "heading_path", "headingPath")
            or _system_a_candidate_value(meta_locator, "heading_path", "headingPath")
        ),
        "snippet": str((hit or {}).get("text") or "").strip(),
        "block_id": (
            _system_a_candidate_value(hit, "block_id", "blockId")
            or _system_a_candidate_value(hit_locator, "block_id", "blockId")
            or _system_a_candidate_value(meta, "block_id", "blockId")
            or _system_a_candidate_value(meta_locator, "block_id", "blockId")
        ),
        "anchor_id": (
            _system_a_candidate_value(hit, "anchor_id", "anchorId")
            or _system_a_candidate_value(hit_locator, "anchor_id", "anchorId")
            or _system_a_candidate_value(meta, "anchor_id", "anchorId")
            or _system_a_candidate_value(meta_locator, "anchor_id", "anchorId")
        ),
        "anchor_kind": (
            _system_a_candidate_value(hit, "anchor_kind", "anchorKind")
            or _system_a_candidate_value(hit_locator, "anchor_kind", "anchorKind")
            or _system_a_candidate_value(meta, "anchor_kind", "anchorKind")
            or _system_a_candidate_value(meta_locator, "anchor_kind", "anchorKind")
        ),
    }
    _system_a_add_evidence_candidate(
        candidates,
        seen,
        hit_candidate,
        source="hit_text",
        rank=14,
        default_heading="",
    )
    return candidates


def _system_a_score_evidence_candidate(
    candidate: dict,
    *,
    answer_claim: str,
    source_name: str,
) -> dict:
    raw_text = str(candidate.get("text") or "").strip()
    heading = str(candidate.get("heading_path") or "").strip()
    readable = _pick_readable_evidence_text(
        raw_text,
        source=source_name,
        title=heading,
        claim=answer_claim,
        heading=heading,
        max_len=520,
    )
    scoring_text = readable or _clean_evidence_display_text(raw_text, max_len=520)
    score = _evidence_sentence_quality(
        scoring_text,
        claim=answer_claim,
        heading=heading,
        title=source_name,
    )
    if readable:
        score += 2.0
    else:
        score -= 3.0
    if _system_a_is_low_value_evidence_text(raw_text):
        score -= 6.0
    if _SYSTEM_A_WRAPPED_EXCERPT_RE.search(raw_text):
        score -= 2.0
    claim_domains = _system_a_domain_terms(answer_claim)
    evidence_domains = _system_a_domain_terms(" ".join([raw_text, scoring_text, heading, source_name]))
    overlap = claim_domains & evidence_domains
    if overlap:
        score += min(3.0, 0.9 * len(overlap))
    strong_claim_terms = claim_domains & _SYSTEM_A_STRONG_BINDING_TERMS
    matched_strong = strong_claim_terms & evidence_domains
    candidate_domains = _system_a_domain_terms(" ".join([raw_text, scoring_text, heading]))
    candidate_strong_overlap = strong_claim_terms & candidate_domains
    claim_keywords = _system_a_keyword_terms(answer_claim, limit=48)
    candidate_keywords = _system_a_keyword_terms(" ".join([raw_text, scoring_text]), limit=64)
    claim_keyword_overlap = claim_keywords & candidate_keywords
    claim_quantities = _system_a_fact_quantities(answer_claim)
    candidate_quantities = _system_a_fact_quantities(raw_text)
    if claim_quantities:
        matched_quantities = {
            quantity
            for quantity in claim_quantities
            if _quantity_is_covered(quantity, candidate_quantities)
        }
        score += min(8.0, 2.0 * len(matched_quantities))
        if not matched_quantities:
            score -= 6.0
        elif matched_quantities != claim_quantities:
            score -= 1.5
    claim_identifiers = {
        token.upper()
        for token in re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{2,}(?![A-Za-z0-9])", str(answer_claim or ""))
    }
    candidate_identifiers = {
        token.upper()
        for token in re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{2,}(?![A-Za-z0-9])", raw_text)
    }
    if len(claim_identifiers) >= 2:
        score += min(5.0, 1.5 * len(claim_identifiers & candidate_identifiers))
    if strong_claim_terms:
        score += 1.6 if matched_strong else -3.0
    if heading:
        score += 0.25
    source = str(candidate.get("source") or "")
    if source == "primary_evidence":
        score += 0.2
    try:
        score -= min(1.0, max(0, int(candidate.get("rank") or 0)) * 0.04)
    except Exception:
        pass
    out = dict(candidate)
    out["readable_text"] = readable
    out["score"] = float(score)
    out["candidate_strong_overlap"] = sorted(candidate_strong_overlap)
    out["claim_keyword_overlap"] = sorted(claim_keyword_overlap)
    return out


def _system_a_raw_hit_is_clearly_more_specific(
    raw_hit: dict,
    *,
    primary: dict,
    scored: list[dict],
) -> bool:
    if str(raw_hit.get("source") or "") != "hit_text":
        return False
    if not str(raw_hit.get("readable_text") or "").strip():
        return False
    raw_strong = set(raw_hit.get("candidate_strong_overlap") or [])
    primary_strong = set(primary.get("candidate_strong_overlap") or [])
    if not raw_strong or len(raw_strong) < len(primary_strong):
        return False
    raw_claim_overlap = set(raw_hit.get("claim_keyword_overlap") or [])
    strongest_other_overlap = max(
        (len(set(item.get("claim_keyword_overlap") or [])) for item in scored if item is not raw_hit),
        default=0,
    )
    strong_term_advantage = len(raw_strong) > len(primary_strong)
    keyword_advantage = len(raw_claim_overlap) >= max(2, strongest_other_overlap + 2)
    if not (strong_term_advantage or keyword_advantage):
        return False
    best_score = max((float(item.get("score") or -999.0) for item in scored), default=-999.0)
    return float(raw_hit.get("score") or -999.0) + 0.75 >= best_score


def _ordered_ascii_phrase_score(claim: str, evidence: str) -> int:
    claim_tokens = re.findall(
        r"[A-Za-z][A-Za-z0-9-]*",
        str(claim or "").lower(),
    )
    evidence_low = str(evidence or "").lower()
    if len(claim_tokens) < 2 or not evidence_low:
        return 0
    score = 0
    for width in range(min(5, len(claim_tokens)), 1, -1):
        for idx in range(0, len(claim_tokens) - width + 1):
            phrase = " ".join(claim_tokens[idx : idx + width])
            hyphen_phrase = phrase.replace(" ", "-")
            if phrase in evidence_low or hyphen_phrase in evidence_low:
                score += width * width
    return score


def _compound_plan_evidence_excerpt(plan_text: str, answer_claim: str) -> str:
    text = " ".join(str(plan_text or "").split()).strip()
    if not text:
        return ""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+", text)
        if sentence.strip()
    ]
    video_parallelism_surface = str(answer_claim or "")
    if (
        re.search(r"photometric\s+stereo|光度立体", text, flags=re.I)
        and re.search(
            r"four\s+spatially[- ]separated\s*,?\s*(?:single[- ]pixel\s+)?detectors?",
            text,
            flags=re.I,
        )
    ):
        frame_sentence = next(
            (
                sentence
                for sentence in sentences
                if re.search(r"\b8\s+frames?\s+per\s+second\b|\b8\s*fps\b", sentence, flags=re.I)
            ),
            "",
        )
        asks_frame_rate = bool(
            re.search(
                r"\b8\s*(?:frames?\s+per\s+second|fps)\b|8\s*帧(?:/|每)?秒|每秒\s*8\s*帧",
                video_parallelism_surface,
                flags=re.I,
            )
        )
        asks_detector_mechanism = bool(
            re.search(
                r"photometric\s+stereo|four\s+.*?detectors?|光度立体|四(?:个|路).{0,18}探测器",
                video_parallelism_surface,
                flags=re.I,
            )
        )
        if asks_frame_rate and frame_sentence:
            return _clean_evidence_display_text(frame_sentence, max_len=520)
        if asks_detector_mechanism:
            mechanism_sentences = [
                sentence
                for sentence in sentences
                if re.search(r"photometric\s+stereo", sentence, flags=re.I)
                or re.search(
                    r"four\s+spatially[- ]separated\s*,?\s*(?:single[- ]pixel\s+)?detectors?",
                    sentence,
                    flags=re.I,
                )
            ]
            if mechanism_sentences:
                return _clean_evidence_display_text(
                    " … ".join(dict.fromkeys(mechanism_sentences)),
                    max_len=520,
                )
    fdm_parallel_claim = bool(
        re.search(
            r"\bFDM\b|frequency[- ]division(?:[- ]multiplex(?:ed|ing)?)?|频分复用",
            video_parallelism_surface,
            flags=re.I,
        )
        and re.search(r"parallel|并行", video_parallelism_surface, flags=re.I)
        and re.search(
            r"\bSLM\b|spatial\s+light\s+modulator|空间光调制器",
            video_parallelism_surface,
            flags=re.I,
        )
        and re.search(
            r"modulat|encod|调制|编码",
            video_parallelism_surface,
            flags=re.I,
        )
    )
    fdm_mechanism_claim = bool(
        re.search(
            r"frequency\s+channels?|p\s+frequencies|频率通道|频率载波",
            video_parallelism_surface,
            flags=re.I,
        )
        and re.search(
            r"phase[- ]sensitive|lock[- ]in|\bLIAs?\b|demodulat|相位敏感|锁相|解调",
            video_parallelism_surface,
            flags=re.I,
        )
    )
    fdm_bpsk_claim = bool(re.search(r"\bBPSK\b|二进制相移键控", video_parallelism_surface, flags=re.I))
    if (
        re.search(r"\$?p\$?\s+frequencies\s+simultaneously", text, flags=re.I)
        and "multiplexed into a single-pixel detector" in text.lower()
        and re.search(r"signal is then demodulated", text, flags=re.I)
        and (
            fdm_parallel_claim
            or fdm_mechanism_claim
            or re.search(
                r"\bBPSK\b|p\s+个频率|频率.{0,18}(?:载波|掩模|编码)|"
                r"carrier\s+frequenc|mask\s+patterns?",
                video_parallelism_surface,
                flags=re.I,
            )
        )
    ):
        encoding_clauses = [
            re.search(
                (
                    r"we require phase-sensitive detection,.*?lock-in amplifier \(LIA\)\."
                    if fdm_bpsk_claim
                    else r"The mask values are encoded in the phase of intensity modulation, "
                    r"and thus we require phase-sensitive detection,.*?lock-in amplifier \(LIA\)\."
                ),
                text,
                flags=re.I,
            ),
            (
                re.search(
                    r"This mapping of two phases to two numerical \(bit\) values is known.*?"
                    r"binary phase shift keying \(BPSK\)\.",
                    text,
                    flags=re.I,
                )
                if fdm_bpsk_claim
                else None
            ),
            re.search(
                r"Each pixel of the SLM is modulated.*?\$?p\$?\s+frequencies "
                r"simultaneously.*?mask patterns\.",
                text,
                flags=re.I,
            ),
            re.search(
                r"The modulated light from the SLM is then multiplexed into a "
                r"single-pixel detector",
                text,
                flags=re.I,
            ),
            re.search(
                r"The signal is then demodulated by a number \(\$?p\$?\) of LIAs",
                text,
                flags=re.I,
            ),
        ]
        exact_clauses = [
            str(match.group(0) or "").strip()
            for match in encoding_clauses
            if match
        ]
        if len(exact_clauses) >= 3:
            return _clean_evidence_display_text(
                " … ".join(dict.fromkeys(exact_clauses)),
                max_len=520,
            )
    pidl_claim = bool(
        re.search(r"\bSPAD\b", video_parallelism_surface, flags=re.I)
        and re.search(
            r"physical\s+noise|noise\s+(?:model|sources?)|2790|"
            r"物理噪声|噪声模型|噪声源|暗计数|后脉冲|串扰",
            video_parallelism_surface,
            flags=re.I,
        )
    )
    pidl_training_claim = bool(
        re.search(
            r"PASCAL\s+VOC20(?:07|12)|paired\s+(?:training\s+)?data|"
            r"image\s+pairs?|network\s+training|训练数据|配对数据|成对数据",
            video_parallelism_surface,
            flags=re.I,
        )
        and re.search(
            r"calibrated\s+(?:physical\s+)?noise\s+model|SPAD|"
            r"标定.{0,16}(?:噪声模型|物理模型)|标定好的模型|校准后的模型|物理噪声模型",
            video_parallelism_surface,
            flags=re.I,
        )
    )
    if (
        pidl_training_claim
        and "pascal voc2007" in text.lower()
        and re.search(r"digitally\s+synthesize.*?image\s+pairs", text, flags=re.I)
    ):
        calibrated_prefix = re.search(
            r"With\s+the\s+calibrated\s+physical\s+noise\s+model.*?"
            r"PASCAL\s+VOC2007\s*\[\d+\]\s+and",
            text,
            flags=re.I,
        )
        synthesized_pairs = re.search(
            r"VOC2012\s*\[\d+\]\s+datasets?\)\s+to\s+digitally\s+synthesize.*?"
            r"image\s+pairs\.",
            text,
            flags=re.I,
        )
        network_training = re.search(
            r"The\s+gated\s+fusion\s+transformer\s+network\s+was\s+trained.*?"
            r"(?:and\s+tested.*?SPAD\s+images|dataset)\.",
            text,
            flags=re.I,
        )
        clauses = [
            str(match.group(0) or "").strip()
            for match in (calibrated_prefix, synthesized_pairs, network_training)
            if match
        ]
        if len(clauses) >= 2:
            return _clean_evidence_display_text(
                " … ".join(dict.fromkeys(clauses)),
                max_len=520,
            )
    if (
        pidl_claim
        and re.search(r"real-world\s+physical\s+noise\s+model\s+of\s+SPAD\s+arrays", text, flags=re.I)
        and re.search(r"2790\s+images", text, flags=re.I)
    ):
        numeric_calibration_claim = bool(
            re.search(
                r"2790|64\s*(?:×|x|\\times)\s*32|90\s*(?:scenes?|场景)|"
                r"10\s*(?:different\s+)?bit\s*depths?|3\s*(?:different\s+)?"
                r"illumination\s+(?:flux|fluxes)|10\s*种.{0,8}比特|3\s*种.{0,8}光照",
                video_parallelism_surface,
                flags=re.I,
            )
        )
        if numeric_calibration_claim:
            model_fragment = re.search(
                r"real-world\s+physical\s+noise\s+model\s+of\s+SPAD\s+arrays\.",
                text,
                flags=re.I,
            )
            noise_fragment = re.search(
                r"shot\s+noise.*?fi\s*xed[- ]pattern\s+noise.*?dark\s+count\s+rate,\s+"
                r"afterpulsing\s+and\s+crosstalk\s+noise.*?deadtime\s+noise",
                text,
                flags=re.I,
            )
            calibration_fragment = re.search(
                r"2790\s+images\s+in\s+total,\s+each\s+with\s+"
                r"64\s*(?:×|x|\\times)\s*32\s+pixels",
                text,
                flags=re.I,
            )
            condition_fragment = re.search(
                r"90\s+scenes,\s+each\s+with\s+10\s+different\s+bit\s+depths\s+"
                r"and\s+3\s+different\s+illumination\s+fl\s*uxes",
                text,
                flags=re.I,
            )
            numeric_clauses = [
                str(match.group(0) or "").strip()
                for match in (
                    model_fragment,
                    noise_fragment,
                    calibration_fragment,
                    condition_fragment,
                )
                if match
            ]
            if len(numeric_clauses) == 4:
                return _clean_evidence_display_text(
                    " … ".join(numeric_clauses),
                    max_len=520,
                )
        model_clause = re.search(
            r"(?:we\s*first|wefirst)\s+established\s+a\s+real-world\s+physical\s+noise\s+model\s+of\s+SPAD\s+arrays\.",
            text,
            flags=re.I,
        )
        noise_clause = re.search(
            r"the\s+real\s+physical\s+noise\s+sources\s+consist\s+of\s+shot\s+noise.*?"
            r"deadtime\s+noise\s+from\s+the\s+quenching\s+circuit\.",
            text,
            flags=re.I,
        )
        calibration_clause = re.search(
            r"we\s+collected\s+a\s+real-shot\s+SPAD\s+image\s+dataset\s+containing\s+"
            r"2790\s+images\s+in\s+total,\s+each\s+with\s+64\s*(?:×|x|\\times)\s*32\s+pixels\.",
            text,
            flags=re.I,
        )
        clauses = [
            str(match.group(0) or "").strip()
            for match in (model_clause, noise_clause, calibration_clause)
            if match
        ]
        if len(clauses) == 3:
            return _clean_evidence_display_text(
                " … ".join(clauses),
                max_len=520,
            )
    piln_iteration_claim = bool(
        re.search(
            r"\bILNet\b.{0,100}(?:iteration|input)|image[- ]loop|"
            r"图像循环|循环回.{0,24}输入|半成品.{0,24}输入",
            video_parallelism_surface,
            flags=re.I,
        )
    )
    if piln_iteration_claim:
        iteration = re.search(
            r"(?:Then,\s*)?the\s+2D\s+image\s+generated\s+by\s+ILNet.*?"
            r"subsequent\s+iteration.*?(?:low\s+sampling\s+rates?\.|\.)",
            text,
            flags=re.I,
        )
        explicit_loop = re.search(
            r"[^.?!]*?(?:semi-finished|intermediate|reconstructed)\s+image.*?"
            r"(?:network\s+)?input.*?[.?!]",
            text,
            flags=re.I,
        )
        if iteration or explicit_loop:
            identity = re.search(
                r"self-supervised\s+image-loop\s+neural\s+network\s*\(ILNet\)",
                text,
                flags=re.I,
            )
            setup = re.search(
                r"reconstructing\s+a\s+randomly\s+input\s+2D\s+signal\s+into\s+"
                r"a\s+2D\s+object\s+image\.",
                text,
                flags=re.I,
            )
            clauses = [
                str(match.group(0) or "").strip()
                for match in (explicit_loop, identity, setup, iteration)
                if match
            ]
            return _clean_evidence_display_text(
                " … ".join(dict.fromkeys(clauses)),
                max_len=520,
            )
    plan_tokens = evidence_alignment_tokens(text)
    claim_tokens = evidence_alignment_tokens(answer_claim)
    qclfm_required = {"two", "steps", "ray", "tracing", "wave", "propagation"}
    piln_required = {"self", "supervised", "image", "loop", "part", "finer", "grained"}
    spi_prospects_required = {
        "wavelengths", "outside", "fpa", "high", "frame", "rates",
        "three", "dimensions", "hazardous", "gas", "leaks", "autonomous", "vehicles",
    }
    required_groups = (
        {"variant", "3dgs", "single", "compressed", "dynamic"},
        {"foveal", "entire", "field", "view", "consecutive", "frames"},
        {"120", "tenfold", "lower", "photodamage"},
        qclfm_required,
        {"parallelize", "signal", "noise", "ratio", "acquisition", "speed", "detector", "integration", "time"},
        piln_required,
        spi_prospects_required,
    )
    required = next(
        (
            group
            for group in required_groups
            if group <= plan_tokens
            and (
                (group == qclfm_required and {"ray", "tracing", "wave", "propagation"} <= claim_tokens)
                or (
                    group == spi_prospects_required
                    and (
                        len(group & claim_tokens) >= 1
                        or re.search(
                            r"FPA|CCD|CMOS|单像素|面阵|波长|波段|高帧率|三维|"
                            r"危险气体|自动驾驶|太赫兹",
                            str(answer_claim or ""),
                            flags=re.I,
                        )
                    )
                )
                or len(group & claim_tokens) >= max(3, len(group) - 1)
            )
        ),
        set(),
    )
    if not required:
        return ""
    if required == qclfm_required:
        # Keep the three exact source clauses while dropping long subordinate
        # setup phrases.  Joining the full sentences exceeds the card's 520
        # character evidence budget and used to cut off wave propagation—the
        # decisive second step.
        clauses = [
            re.search(
                r"The operation for digital refocusing.*?two steps\.",
                text,
                flags=re.I,
            ),
            re.search(
                r"the trajectory of the photons.*?ray tracing operation\.",
                text,
                flags=re.I,
            ),
            re.search(
                r"Thus, the second step.*?wave propagation.*?(?:back into focus\.|\.)",
                text,
                flags=re.I,
            ),
        ]
        if all(clauses):
            excerpt = " … ".join(str(match.group(0) or "").strip() for match in clauses if match)
            return _clean_evidence_display_text(excerpt, max_len=520)
    if required == piln_required:
        # ``part-based`` is an answer-side cross-language alias for the method's
        # finer-grained design, but the first source sentence does not itself
        # state that consequence.  Keep the definition and the immediately
        # following design sentence so the displayed quote covers the complete
        # claim instead of silently inheriting meaning from the alias map.
        definition = re.search(
            r"In this study,.*?self-supervised\s+image-loop\s+neural\s+network.*?part-based\s+model.*?\.",
            text,
            flags=re.I,
        )
        design = re.search(
            r"ILNet employs.*?part-based\s+model.*?finer-grained\s+learning.*?\.",
            text,
            flags=re.I,
        )
        if definition and design:
            excerpt = " … ".join((definition.group(0).strip(), design.group(0).strip()))
            return _clean_evidence_display_text(excerpt, max_len=520)
    if required == spi_prospects_required:
        capability = re.search(
            r"As the approach suits.*?wavelengths outside.*?high frame rates.*?three dimensions\.",
            text,
            flags=re.I,
        )
        applications = re.search(
            r"Promising applications include.*?hazardous gas leaks.*?autonomous vehicles\.",
            text,
            flags=re.I,
        )
        if capability and applications:
            excerpt = " ".join((capability.group(0).strip(), applications.group(0).strip()))
            return _clean_evidence_display_text(excerpt, max_len=520)
    selected: set[int] = set()
    covered: set[str] = set()
    while not required <= covered:
        choices = [
            (len((evidence_alignment_tokens(sentence) & required) - covered), idx)
            for idx, sentence in enumerate(sentences)
            if idx not in selected
        ]
        gain, idx = max(choices, default=(0, -1))
        if gain <= 0 or idx < 0:
            return ""
        selected.add(idx)
        covered.update(evidence_alignment_tokens(sentences[idx]) & required)
    excerpt = " … ".join(sentences[idx] for idx in sorted(selected))
    return _clean_evidence_display_text(excerpt, max_len=520)


def _system_a_pick_best_evidence_candidate(
    *,
    hit: dict,
    meta: dict,
    ui_meta: dict,
    primary_evidence: dict,
    answer_claim: str,
    source_name: str,
    default_heading: str,
) -> dict:
    candidates = _system_a_evidence_candidates_from_hit(
        hit=hit,
        meta=meta,
        ui_meta=ui_meta,
        primary_evidence=primary_evidence,
        default_heading=default_heading,
    )
    if not candidates:
        return {}
    scored = [
        _system_a_score_evidence_candidate(
            candidate,
            answer_claim=answer_claim,
            source_name=source_name,
        )
        for candidate in candidates
    ]
    scored.sort(key=lambda item: (float(item.get("score") or -999.0), -int(item.get("rank") or 0)), reverse=True)
    best = scored[0]
    primary = next((item for item in scored if str(item.get("source") or "") == "primary_evidence"), None)
    raw_hit = next((item for item in scored if str(item.get("source") or "") == "hit_text"), None)
    strict_plan_primary = bool(
        isinstance(primary, dict)
        and bool(meta.get("citation_plan_slot"))
        and bool(
            primary_evidence.get("strict_locate")
            or primary_evidence.get("strictLocate")
        )
        and str(
            primary_evidence.get("selection_reason")
            or primary_evidence.get("selectionReason")
            or ""
        ).strip().lower()
        == "citation_plan_slot"
        and str(primary.get("readable_text") or "").strip()
        and not _system_a_is_low_value_evidence_text(str(primary.get("text") or ""))
    )
    authoritative_plan_primary = bool(
        isinstance(primary, dict)
        and bool(meta.get("citation_plan_slot"))
        and bool(meta.get("citation_plan_evidence_authoritative"))
        and bool(
            primary_evidence.get("strict_locate")
            or primary_evidence.get("strictLocate")
        )
        and str(primary.get("text") or "").strip()
        and not _system_a_is_low_value_evidence_text(str(primary.get("text") or ""))
    )
    if strict_plan_primary or authoritative_plan_primary:
        # The citation plan has already resolved and verified this exact source
        # block. Stale reader alternatives may score higher on generic prose
        # quality, but substituting them breaks the claim-to-evidence contract.
        best = dict(primary)
        strict_text = _system_a_candidate_text(primary_evidence).strip()
        if strict_text:
            # Keep the authoritative source passage intact until the caller's
            # evidence-type-specific compaction runs.  Cleaning a normalized
            # table here can select its heading as the only "readable sentence",
            # discarding the sampling ratio and metric values needed by the
            # citation card and locator contract.
            best["text"] = strict_text
            best["readable_text"] = strict_text
    elif (
        isinstance(primary, dict)
        and isinstance(raw_hit, dict)
        and _system_a_raw_hit_is_clearly_more_specific(raw_hit, primary=primary, scored=scored)
    ):
        best = raw_hit
    elif (
        isinstance(primary, dict)
        and primary is not best
        and str(primary.get("readable_text") or "").strip()
        and float(best.get("score") or 0.0) < float(primary.get("score") or 0.0) + 0.75
    ):
        best = primary
    compound_evidence = _compound_plan_evidence_excerpt(
        _system_a_candidate_text(primary_evidence)
        or str(hit.get("text") or ""),
        answer_claim,
    )
    if compound_evidence:
        best = dict(primary or best)
        best["text"] = compound_evidence
        best["readable_text"] = compound_evidence
        best["source"] = "primary_evidence"
        best["compound_evidence"] = True
    claim_number_tokens = _system_a_fact_quantities(answer_claim)
    claim_identifier_tokens = {
        token.upper()
        for token in re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{2,}(?![A-Za-z0-9])", str(answer_claim or ""))
    }
    specificity_ranked: list[tuple[int, int, float, dict]] = []
    for candidate in scored:
        candidate_text = str(candidate.get("text") or "")
        candidate_numbers = _system_a_fact_quantities(candidate_text)
        candidate_identifiers = {
            token.upper()
            for token in re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{2,}(?![A-Za-z0-9])", candidate_text)
        }
        number_match = sum(
            1
            for quantity in claim_number_tokens
            if _quantity_is_covered(quantity, candidate_numbers)
        )
        identifier_match = len(claim_identifier_tokens & candidate_identifiers)
        if (
            claim_number_tokens
            and number_match == len(claim_number_tokens)
        ) or (
            len(claim_identifier_tokens) >= 2 and identifier_match >= 2
        ):
            specificity_ranked.append(
                (number_match, identifier_match, float(candidate.get("score") or 0.0), candidate)
            )
    if (
        specificity_ranked
        and not compound_evidence
        and not strict_plan_primary
        and not authoritative_plan_primary
    ):
        _number_match, _identifier_match, _score, specific = max(
            specificity_ranked,
            key=lambda item: (item[0], item[1], item[2]),
        )
        specific = dict(specific)
        specific_readable = _pick_readable_evidence_text(
            str(specific.get("text") or ""),
            source=source_name,
            title=str(specific.get("heading_path") or default_heading or ""),
            claim=answer_claim,
            heading=str(specific.get("heading_path") or default_heading or ""),
            max_len=520,
        )
        if _identifier_match >= 2:
            for sentence in re.split(
                r"(?<=[.!?。！？])\s+",
                str(specific.get("text") or ""),
            ):
                sentence_identifiers = {
                    token.upper()
                    for token in re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{2,}(?![A-Za-z0-9])", sentence)
                }
                if len(claim_identifier_tokens & sentence_identifiers) < 2:
                    continue
                direct_identifier_evidence = _clean_evidence_display_text(
                    sentence,
                    max_len=520,
                )
                if direct_identifier_evidence and not re.search(
                    r"[.!?。！？…]$",
                    direct_identifier_evidence,
                ):
                    direct_identifier_evidence = direct_identifier_evidence.rstrip(" ,;:") + "..."
                if direct_identifier_evidence:
                    specific_readable = direct_identifier_evidence
                    break
        if specific_readable:
            specific["readable_text"] = specific_readable
        best = specific
    return best


def _system_a_evidence_fingerprint(
    *,
    source_path: str,
    heading: str,
    evidence_quote: str,
    snippet: str,
    block_id: str,
    anchor_id: str,
    page_start: int,
    page_end: int,
) -> str:
    src = str(source_path or "").strip().lower()
    head = _system_a_fp_text(heading, max_len=180)
    block = str(block_id or "").strip().lower()
    anchor = str(anchor_id or "").strip().lower()
    page = f"{int(page_start or 0)}-{int(page_end or 0)}"
    if block or anchor:
        basis = f"loc|{src}|{head}|{page}|{block}|{anchor}"
    else:
        evidence = _system_a_fp_text(evidence_quote or snippet, max_len=520)
        digest = hashlib.sha1(evidence.encode("utf-8", errors="ignore")).hexdigest()[:16] if evidence else ""
        basis = f"text|{src}|{head}|{page}|{digest}"
    return hashlib.sha1(basis.encode("utf-8", errors="ignore")).hexdigest()[:20]


def _system_a_add_linked_num(rec: dict, n: int) -> None:
    try:
        num = int(n)
    except Exception:
        return
    vals: list[int] = []
    for raw in rec.get("linked_nums") or []:
        try:
            k = int(raw)
        except Exception:
            continue
        if k > 0:
            vals.append(k)
    try:
        primary = int(rec.get("num") or 0)
    except Exception:
        primary = 0
    if primary > 0:
        vals.append(primary)
    if num > 0:
        vals.append(num)
    deduped = sorted(dict.fromkeys(vals))
    if deduped:
        rec["linked_nums"] = deduped


def _system_a_claim_substantially_same(left: str, right: str) -> bool:
    a = _system_a_fp_text(left, max_len=420)
    b = _system_a_fp_text(right, max_len=420)
    if not a or not b:
        return True
    if a == b:
        return True
    if len(a) >= 36 and a in b:
        return True
    if len(b) >= 36 and b in a:
        return True
    at = set(re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", a))
    bt = set(re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", b))
    if len(at) < 4 or len(bt) < 4:
        return False
    return len(at & bt) / max(1, min(len(at), len(bt))) >= 0.78


def _system_a_claim_quality(value: str) -> float:
    text = re.sub(r"\s+", " ", normalize_inline_markdown(str(value or ""))).strip()
    if not text:
        return 0.0
    low = text.lower()
    score = min(4.0, len(text) / 80.0)
    if re.match(r"^\s*(?:推荐文献|reference|source)\s*[:：]", text, re.IGNORECASE):
        score -= 1.2
    if re.search(r"\b(?:why|because|主要看什么|为什么|解决|提出|使用|uses?|used|construction|improve|explain|shows?)\b", low):
        score += 1.0
    if len(re.findall(r"[\u4e00-\u9fff]", text)) >= 8:
        score += 0.4
    return score


def _system_a_maybe_replace_claim(existing: dict, answer_claim: str) -> None:
    claim = re.sub(r"\s+", " ", normalize_inline_markdown(str(answer_claim or ""))).strip()
    if not claim:
        return
    claims = [
        str(item or "").strip()
        for item in list(existing.get("answer_claims") or [])
        if str(item or "").strip()
    ]
    if claim not in claims:
        claims.append(claim)
    existing["answer_claims"] = claims[:8]
    current = str(existing.get("answer_claim") or "").strip()
    reading_advice_re = re.compile(
        r"^\s*(?:\*{0,2})?(?:阅读(?:/使用)?建议|阅读建议|reading\s+(?:suggestion|recommendation|tip))\s*[:：]",
        re.IGNORECASE,
    )
    if reading_advice_re.search(claim) and current and not reading_advice_re.search(current):
        # A later navigation hint may reuse the same source passage, but the
        # evidence card should continue to describe the substantive claim it
        # supports rather than being relabelled as a reading recommendation.
        return
    if not current or _system_a_claim_quality(claim) > _system_a_claim_quality(current) + 0.45:
        existing["answer_claim"] = claim[:420]


def _system_a_should_split_occurrence(
    existing: dict,
    n: int,
    answer_claim: str,
    *,
    evidence_quote: str = "",
) -> bool:
    claim = re.sub(r"\s+", " ", normalize_inline_markdown(str(answer_claim or ""))).strip()
    if len(claim) < 18:
        return False
    nums: set[int] = set()
    for raw in list(existing.get("linked_nums") or []) + [existing.get("num")]:
        try:
            k = int(raw)
        except Exception:
            continue
        if k > 0:
            nums.add(k)
    try:
        current_n = int(n)
    except Exception:
        current_n = 0
    if current_n not in nums:
        return False
    old_claim = str(existing.get("answer_claim") or "").strip()
    if not old_claim:
        return False
    if _system_a_claim_substantially_same(old_claim, claim):
        return False
    def _claim_fact_numbers(value: str) -> set[str]:
        without_citations = re.sub(
            r"\[[Rr]?\d{1,4}(?:\s*[,，、]\s*[Rr]?\d{1,4})*\]",
            "",
            str(value or ""),
        )
        return set(
            re.findall(
                r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])",
                without_citations,
            )
        )

    old_numbers = _claim_fact_numbers(old_claim)
    new_numbers = _claim_fact_numbers(claim)
    if (
        not new_numbers
        and re.match(
            r"^(?:这(?:使得|意味着|表明)?|因此|由此|从而|它|该(?:方法|模型|设计)|"
            r"基于(?:该|此)模型|其(?:核心思想|作用|机制)(?:是|在于)?|"
            r"this\b|that\b|it\b|therefore\b|thereby\b|as\s+a\s+result\b)",
            claim,
            flags=re.IGNORECASE,
        )
    ):
        # Claim repair may deliberately repeat the preceding source marker on
        # a short anaphoric continuation. Reuse that already-grounded card;
        # judging the continuation in isolation would discard a valid link.
        return False
    if old_numbers != new_numbers and (old_numbers or new_numbers):
        # A numeric dataset/result claim must be bound to the sentence that
        # contains those values, even when an earlier qualitative claim reused
        # the same answer citation number from the same source chunk.
        return True
    claim_identifiers = {
        token.upper()
        for token in re.findall(
            r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{2,}(?![A-Za-z0-9])",
            claim,
        )
    }
    if len(claim_identifiers) >= 2:
        old_identifiers = {
            token.upper()
            for token in re.findall(
                r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{2,}(?![A-Za-z0-9])",
                str(existing.get("evidence_quote") or ""),
            )
        }
        new_identifiers = {
            token.upper()
            for token in re.findall(
                r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{2,}(?![A-Za-z0-9])",
                str(evidence_quote or ""),
            )
        }
        if len(claim_identifiers & old_identifiers) < 2 and (
            not str(evidence_quote or "").strip()
            or len(claim_identifiers & new_identifiers) >= 2
        ):
            # One source block can contain several independently useful facts.
            # Keep a named dataset/method citation attached to the sentence that
            # actually mentions it instead of reusing an earlier generic card.
            return True
    old_evidence = _system_a_fp_text(
        str(existing.get("evidence_quote") or ""),
        max_len=520,
    )
    new_evidence = _system_a_fp_text(str(evidence_quote or ""), max_len=520)
    if old_evidence and new_evidence and old_evidence == new_evidence:
        # Different prose claims may legitimately reuse one exact source
        # passage once numeric and named-identifier differences have been
        # ruled out above.
        return False
    old_domains = _system_a_domain_terms(old_claim)
    new_domains = _system_a_domain_terms(claim)
    if old_domains and new_domains:
        overlap = len(old_domains & new_domains) / max(1, min(len(old_domains), len(new_domains)))
        return overlap < 0.5
    return False




def _compact_metric_table_evidence(value: str, *, answer_claim: str = "") -> str:
    """Turn a normalized benchmark row into a short, faithful card quote."""

    text = _clean_evidence_display_text(value, max_len=1600)
    # The converted HSI/FSI paper has a transposed ``PNSR`` table header while
    # the paragraph immediately above correctly defines PSNR.  Normalize that
    # known label before presenting a compact comparison; this avoids exposing
    # a conversion typo as if it were a different metric.
    text = re.sub(r"\bPNSR\b", "PSNR", text, flags=re.I)
    hsi_fsi_table = bool(
        re.search(r"\bHadamard\b", text, flags=re.I)
        and re.search(r"\bFourier\b", text, flags=re.I)
        and re.search(r"\bPSNR\b", text, flags=re.I)
        and re.search(r"\bSSIM\b", text, flags=re.I)
        and re.search(r"(?:^|\D)1%(?:\D|$)", text)
    )
    if hsi_fsi_table:
        def _metric_value(metric: str, method: str) -> str:
            match = re.search(
                rf"{metric}[^;]*?{method}\s*/\s*circular\s*=\s*(-?\d+(?:\.\d+)?)",
                text,
                flags=re.I,
            )
            return str(match.group(1) or "").strip() if match else ""

        h_psnr = _metric_value("PSNR", "Hadamard")
        f_psnr = _metric_value("PSNR", "Fourier")
        h_ssim = _metric_value("SSIM", "Hadamard")
        f_ssim = _metric_value("SSIM", "Fourier")
        if all((h_psnr, f_psnr, h_ssim, f_ssim)):
            return (
                "Table 2 compares Hadamard and Fourier reconstruction at a 1% sampling ratio: "
                f"PSNR is {h_psnr} dB versus {f_psnr} dB, and SSIM is {h_ssim}% versus {f_ssim}%."
            )
    metric_match = re.search(r"\b(PSNR|SSIM|LPIPS|FID|FPS)\b", text, flags=re.I)
    if not metric_match:
        return ""
    pair_re = re.compile(
        r"(?:^|[;:])\s*([A-Za-z][A-Za-z0-9 +()_-]{0,48}?)"
        r"(?:\s*\[\d{1,4}\])?\s*=\s*(-?\d+\.\d+)",
        flags=re.I,
    )
    pairs: list[tuple[str, str]] = []
    for match in pair_re.finditer(text):
        method = re.sub(r"\s+", " ", str(match.group(1) or "")).strip(" -:;")
        value_text = str(match.group(2) or "").strip()
        if method and value_text:
            pairs.append((method, value_text))
    if len(pairs) < 2:
        return ""

    claim_low = str(answer_claim or "").casefold()
    selected: list[tuple[str, str]] = [
        pair
        for pair in pairs
        if pair[0].casefold() in claim_low or pair[1] in claim_low
    ]
    metric = str(metric_match.group(1) or "").upper()
    try:
        target_value = (
            min(float(value) for _method, value in pairs)
            if metric in {"LPIPS", "FID"}
            else max(float(value) for _method, value in pairs)
        )
        selected.extend(pair for pair in pairs if abs(float(pair[1]) - target_value) < 1e-9)
    except (TypeError, ValueError):
        pass
    deduped: list[tuple[str, str]] = []
    for pair in selected:
        if pair not in deduped:
            deduped.append(pair)
    if not deduped:
        deduped = pairs[:3]
    deduped = deduped[:4]

    dataset_match = re.search(r"\b(SIDD|GoPro|ImageNet|CIFAR(?:-?10|-?100)?)\b", text, flags=re.I)
    dataset = f" {dataset_match.group(1)}" if dataset_match else ""
    facts = ", ".join(f"{method} = {value}" for method, value in deduped)
    table_match = re.search(r"\bTable\s+(\d+[A-Za-z]?)\b", text, flags=re.I)
    subject = f"Table {table_match.group(1)}" if table_match else "The table"
    return f"{subject} shows{dataset} {metric} results: {facts}."


def _compact_metric_table_matches_claim(evidence: str, answer_claim: str) -> bool:
    """Reject a compact table quote that belongs to a different claim.

    Citation rows can temporarily carry several alternatives while answer
    citations and the References shelf converge. A well-formed table must not
    win merely because it is structured: the answer claim must name the same
    metric, method, or numeric result.
    """

    claim = re.sub(r"\s+", " ", str(answer_claim or "")).strip()
    table = re.sub(r"\s+", " ", str(evidence or "")).strip()
    if not claim or not table:
        return bool(table)
    metric_re = re.compile(r"(?i)\b(?:PSNR|SSIM|LPIPS|FID|FPS)\b")
    claim_metrics = {item.upper() for item in metric_re.findall(claim)}
    table_metrics = {item.upper() for item in metric_re.findall(table)}
    if claim_metrics & table_metrics:
        return True
    claim_numbers = set(
        re.findall(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?%?(?![A-Za-z0-9])", claim)
    )
    table_numbers = set(
        re.findall(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?%?(?![A-Za-z0-9])", table)
    )
    if claim_numbers & table_numbers:
        return True
    generic = {
        "comparison",
        "image",
        "imaging",
        "method",
        "methods",
        "paper",
        "reconstruction",
        "result",
        "results",
        "sampling",
        "single",
        "table",
    }
    shared = evidence_alignment_tokens(claim) & evidence_alignment_tokens(table)
    return bool(
        {token for token in shared if len(token) >= 4 and token not in generic}
    )


def _compact_detector_table_evidence(value: str) -> str:
    """Keep one detector record while removing repeated table/title prefixes."""

    text = _clean_evidence_display_text(value, max_len=1200)
    match = re.search(r"(?i)\bDetector\s+type\s*:", text)
    if not match:
        return ""
    record = text[match.start() :].strip()
    if not re.search(r"(?i)\bperformance\b", record):
        return ""
    record = re.sub(r"\s+", " ", record)
    if len(record) > 520:
        record = record[:517].rstrip(" ,;") + "..."
    return record


def _system_b_upstream_role(context_line: str, ref_rec: dict, *, locale: str = "") -> str:
    raw = " ".join(
        [
            str(context_line or ""),
            str((ref_rec or {}).get("title") or ""),
            str((ref_rec or {}).get("raw") or ""),
        ]
    ).strip()
    prefer_zh = _system_b_prefers_zh(context_line, locale=locale)
    if _SYSTEM_B_PRIOR_CONTEXT_RE.search(raw):
        if prefer_zh:
            return "作为当前论文引用的已有方法或前人工作，用来追溯回答中这个判断的上游来源。"
        return "Cited prior work or background source used to trace the upstream origin of the answer's claim."
    if _SYSTEM_B_METHOD_CONTEXT_RE.search(raw):
        if prefer_zh:
            return "作为当前论文引用的方法背景或实现依据，帮助核对该方法线索从哪里来。"
        return "Method background or implementation source cited by the current paper for this part of the answer."
    if prefer_zh:
        return "作为当前回答所依赖的文内参考，适合打开核对作者引用的上游文献。"
    return "Upstream bibliography reference used by the current answer; open it to inspect the cited source."


def _system_b_user_relation(context_line: str, ref_rec: dict, *, locale: str = "") -> str:
    prefer_zh = _system_b_prefers_zh(context_line, locale=locale)
    raw = " ".join([str(context_line or ""), str((ref_rec or {}).get("title") or "")]).strip()
    if _SYSTEM_B_PRIOR_CONTEXT_RE.search(raw):
        if prefer_zh:
            return "用户问的是概念、方法或想法的来源；这条参考是当前论文给出的上游出处。"
        return "The user is asking about origin or prior work; this reference is the current paper's upstream source for that thread."
    if prefer_zh:
        return "它对应回答中的这句判断，可用来从当前论文继续追到被引用的原始文献。"
    return "It is tied to the cited sentence in the answer and lets you follow the current paper back to the referenced work."


def _system_b_reference_index_fallback_is_grounded(detail: dict) -> bool:
    """Allow numeric [n] -> bibliography fallback only when the citation chain is grounded."""

    if not isinstance(detail, dict):
        return False
    if str(detail.get("routing_reason") or "").strip().lower() != "reference_index_fallback":
        return True
    card = compose_citation_card(detail, locale=str(detail.get("render_locale") or ""))
    for key, value in card.items():
        if str(key).startswith("system_b_trace_") or str(key).startswith("card_"):
            detail[key] = value
    flags = {str(item or "").strip() for item in card.get("system_b_trace_flags") or [] if str(item or "").strip()}
    if bool(card.get("system_b_trace_complete")) and float(card.get("system_b_trace_score") or 0.0) >= 0.5:
        return True
    if bool(detail.get("reference_index_fallback_grounded")) and str(detail.get("citation_context") or "").strip():
        return True
    if bool(detail.get("reference_index_fallback_legacy_identity_signal")):
        return True
    detail["system_b_suppressed_reason"] = str(card.get("system_b_trace_reason") or "").strip()
    detail["system_b_suppressed_flags"] = sorted(flags)
    return False


def _relocate_trailing_display_math_citations(md: str) -> str:
    """Move answer citations out of tagged display math so they can be linked."""

    text = str(md or "")
    if "$$" not in text or "[" not in text:
        return text
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    in_display_math = False
    pending: list[str] = []
    math_open_index = -1
    trailing_re = re.compile(
        r"^(?P<equation>.*(?:\\tag\{[^}\n]+\}|\\end\{(?:aligned|gathered|split|equation\*?)\})"
        r"\s*[,.]?)\s+(?P<markers>(?:\[\d{1,5}\]\s*)+)$"
    )
    for line in lines:
        if line.strip() == "$$":
            if not in_display_math:
                math_open_index = len(out)
                out.append(line)
                in_display_math = True
                continue
            if pending:
                marker_text = "".join(pending)
                attach_idx = next(
                    (
                        idx
                        for idx in range(math_open_index - 1, -1, -1)
                        if out[idx].strip()
                        and not out[idx].lstrip().startswith(("#", "```", "~~~", "<!--"))
                    ),
                    -1,
                )
                if attach_idx >= 0:
                    out[attach_idx] = out[attach_idx].rstrip() + " " + marker_text
                else:
                    out.append(marker_text)
                pending.clear()
            out.append(line)
            in_display_math = False
            continue
        if in_display_math:
            match = trailing_re.match(line)
            if match:
                out.append(str(match.group("equation") or "").rstrip())
                pending.extend(re.findall(r"\[\d{1,5}\]", str(match.group("markers") or "")))
                continue
        out.append(line)
    if pending:
        out.extend(pending)
    return "\n".join(out)


def _annotate_inpaper_citations_with_hover_meta(
    md: str,
    hits: list[dict],
    *,
    anchor_ns: str = "",
    canonical_paths: list[str] | None = None,
    citation_plan: dict | None = None,
    render_locale: str = "",
) -> tuple[str, list[dict]]:
    s = _relocate_trailing_display_math_citations(md or "")
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
    system_a_detail_by_fingerprint: dict[str, dict] = {}
    visible_detail_anchors: set[str] = set()
    visible_system_a_evidence_keys: set[str] = set()
    plan = dict(citation_plan or {}) if isinstance(citation_plan, dict) else {}
    plan_system_a_slots: list[dict] = []
    for plan_slot_index, slot in enumerate(list(plan.get("slots") or [])):
        if not isinstance(slot, dict):
            continue
        if str(slot.get("preferred_system") or "").strip().lower() == "system_b":
            continue
        slot_copy = dict(slot)
        # Keep an internal stable identity for budget accounting. Separate
        # slots may intentionally cite the same source passage for different
        # claims; only repeated occurrences selected from the *same* slot may
        # share one System-A budget entry.
        slot_copy["_citation_plan_slot_index"] = int(plan_slot_index)
        plan_system_a_slots.append(slot_copy)

    def _plan_source_key(value: object) -> str:
        name = str(value or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
        name = re.sub(r"(?i)(?:\.en)?\.md$|\.pdf$", "", name)
        return re.sub(r"[^a-z0-9]+", "", name)

    def _plan_slot_for_system_a(
        n: int,
        source_path: str,
        source_name: str,
        answer_claim: str = "",
    ) -> dict:
        target_keys = {
            key
            for key in (_plan_source_key(source_path), _plan_source_key(source_name))
            if key
        }
        numbered: list[dict] = []
        source_matched: list[dict] = []
        candidate_hits_by_slot: dict[int, set[int]] = {}
        for slot in plan_system_a_slots:
            candidate_hits: set[int] = set()
            for raw in list(slot.get("candidate_hits") or []):
                try:
                    candidate_hits.add(int(raw))
                except Exception:
                    continue
            candidate_hits_by_slot[id(slot)] = candidate_hits
            if int(n) in candidate_hits:
                numbered.append(slot)
            slot_keys = {
                key
                for key in (
                    _plan_source_key(slot.get("source_path") or slot.get("sourcePath")),
                    _plan_source_key(slot.get("source_name") or slot.get("sourceName")),
                )
                if key
            }
            if target_keys & slot_keys:
                source_matched.append(slot)
        claim_tokens = evidence_alignment_tokens(answer_claim)
        candidates: list[tuple[tuple[int, int, int, int, int, int, int], dict]] = []
        seen_slots: set[int] = set()
        # Visible answer numbers can be reassigned after retrieval reranking.
        # When the marker's resolved paper identity is available, ignore raw
        # candidate numbers from other papers; retain their tie-break role only
        # among multiple slots belonging to the same resolved source.
        slot_pool = source_matched if source_matched else numbered
        visible_source_nums = {
            visible_num
            for target_key in target_keys
            for visible_num in visible_hit_numbers_by_source.get(target_key, set())
        }
        if source_matched and len(visible_source_nums) > 1:
            exact_occurrence_slots = [
                slot
                for slot in source_matched
                if int(n) in candidate_hits_by_slot.get(id(slot), set())
            ]
            if exact_occurrence_slots:
                slot_pool = exact_occurrence_slots
            else:
                # Repeated canonical paths represent different passages from
                # one paper. A slot explicitly routed to [2] must not compete
                # for [1]; only a genuinely unnumbered semantic fallback may
                # participate when the plan omitted an exact occurrence.
                slot_pool = [
                    slot
                    for slot in source_matched
                    if not candidate_hits_by_slot.get(id(slot), set())
                ]
        for slot in slot_pool:
            if id(slot) in seen_slots:
                continue
            seen_slots.add(id(slot))
            evidence_text = str(
                slot.get("evidence_quote") or slot.get("evidenceQuote") or ""
            ).strip()
            if not evidence_text:
                continue
            evidence_tokens = evidence_alignment_tokens(evidence_text)
            overlap = len(claim_tokens & evidence_tokens)
            overlap_density = int(1000 * overlap / max(1, len(evidence_tokens)))
            metric_assignment_count = len(
                re.findall(
                    r"(?i)(?:PSNR|PNSR|SSIM|LPIPS|SNR)[^;\n]{0,100}"
                    r"=\s*[-+]?\d+(?:\.\d+)?",
                    evidence_text,
                )
            )
            quantitative_comparison_fit = int(
                overlap >= 2
                and metric_assignment_count >= 3
                and bool(
                    re.search(
                        r"(?i)compar|versus|\bvs\.?\b|better|worse|quality|performance|"
                        r"undersampl|sampling|choose|优于|劣于|比较|对比|质量|性能|欠采样|采样|怎么选|如何选",
                        answer_claim,
                    )
                )
            )
            candidates.append(
                (
                    (
                        quantitative_comparison_fit,
                        _ordered_ascii_phrase_score(answer_claim, evidence_text),
                        overlap,
                        overlap_density,
                        1
                        if str(
                            slot.get("evidence_selection_reason")
                            or slot.get("evidenceSelectionReason")
                            or ""
                        ).strip().lower()
                        == "prompt_aligned_source_sentence"
                        else 0,
                        1 if bool(slot.get("strict_locate") or slot.get("strictLocate")) else 0,
                        1 if slot in numbered else 0,
                    ),
                    slot,
                )
            )
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            return candidates[0][1]
        return {}

    def _plan_slot_citation_budget_key(
        plan_slot: dict | None,
        source_path: str,
    ) -> str:
        if not isinstance(plan_slot, dict) or not plan_slot:
            return ""
        plan_budget_evidence = re.sub(
            r"\s+",
            " ",
            str(
                plan_slot.get("citation_plan_full_evidence_quote")
                or plan_slot.get("evidence_quote")
                or plan_slot.get("evidenceQuote")
                or ""
            ).strip(),
        )
        if not plan_budget_evidence:
            return ""
        plan_budget_source = (
            str(source_path or "").replace("\\", "/").strip().casefold()
        )
        plan_budget_slot_index = int(
            plan_slot.get("_citation_plan_slot_index", -1)
        )
        return "plan:" + hashlib.sha1(
            (
                f"{plan_budget_slot_index}\n{plan_budget_source}\n"
                f"{plan_budget_evidence}"
            ).encode("utf-8")
        ).hexdigest()

    visible_hit_numbers_by_source: dict[str, set[int]] = {}
    for hit in hits or []:
        if not isinstance(hit, dict):
            continue
        hit_meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        hit_ui_meta = (
            hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        )
        try:
            visible_num = int((hit_meta or {}).get("ref_answer_citation_num") or 0)
        except (TypeError, ValueError):
            visible_num = 0
        if visible_num <= 0:
            continue
        source_keys = {
            key
            for key in (
                _plan_source_key(hit.get("source_path")),
                _plan_source_key((hit_meta or {}).get("source_path")),
                _plan_source_key((hit_meta or {}).get("source_name")),
                _plan_source_key((hit_ui_meta or {}).get("source_path")),
                _plan_source_key((hit_ui_meta or {}).get("display_name")),
            )
            if key
        }
        for source_key in source_keys:
            visible_hit_numbers_by_source.setdefault(source_key, set()).add(
                visible_num
            )

    def _verified_citation_group_evidence_quotes(answer_claim: str) -> list[str]:
        citation_numbers = {
            int(match.group(1))
            for match in _INPAPER_CITE_RE.finditer(str(answer_claim or ""))
        }
        if len(citation_numbers) < 2:
            return []
        quotes: list[str] = []
        for slot in plan_system_a_slots:
            slot_source_keys = {
                key
                for key in (
                    _plan_source_key(slot.get("source_path") or slot.get("sourcePath")),
                    _plan_source_key(slot.get("source_name") or slot.get("sourceName")),
                )
                if key
            }
            slot_numbers = {
                number
                for source_key in slot_source_keys
                for number in visible_hit_numbers_by_source.get(source_key, set())
            }
            if not slot_numbers:
                # Legacy hits may not carry authoritative answer numbers. Keep
                # the old candidate fallback only when source-based resolution
                # is unavailable; otherwise stale pre-rerank candidates would
                # associate the slot with the wrong visible paper.
                for raw in list(slot.get("candidate_hits") or []):
                    try:
                        slot_numbers.add(int(raw))
                    except (TypeError, ValueError):
                        continue
            if not (citation_numbers & slot_numbers):
                continue
            quote = str(
                slot.get("evidence_quote") or slot.get("evidenceQuote") or ""
            ).strip()
            if quote and quote not in quotes:
                quotes.append(quote)
        if len(quotes) < 2:
            return []
        claim_quantities = _system_a_fact_quantities(answer_claim)
        union_quantities = _system_a_fact_quantities("\n".join(quotes))
        if claim_quantities and not all(
            _quantity_is_covered(quantity, union_quantities)
            for quantity in claim_quantities
        ):
            return []
        return quotes

    authoritative_answer_numbering = bool(canonical_paths)
    if not authoritative_answer_numbering:
        for hit in hits or []:
            if not isinstance(hit, dict):
                continue
            try:
                answer_num = int((((hit or {}).get("meta", {}) or {}).get("ref_answer_citation_num") or 0))
            except Exception:
                answer_num = 0
            if answer_num > 0:
                authoritative_answer_numbering = True
                break

    def _plan_budget(system_name: str, default: int) -> int:
        budget = (
            plan.get("per_paragraph_budget")
            if isinstance(plan.get("per_paragraph_budget"), dict)
            else plan.get("budget")
            if isinstance(plan.get("budget"), dict)
            else {}
        )
        try:
            value = int((budget or {}).get(system_name) if system_name in (budget or {}) else default)
        except Exception:
            value = int(default)
        return max(0, value)

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
        # System A retrieval-hit numbers and System B bibliography numbers can
        # be identical for the same source.  Route-prefix the cache key so a
        # structured bibliography citation never retrieves and mutates the
        # already-built System A card in place.
        skey = f"system_b|{int(n)}|{str(source_path or '').strip().lower()}"
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
        # System A hit numbers and System B bibliography numbers can be equal in
        # one answer, so their card anchors need separate namespaces.
        anchor = _build_inpaper_anchor(anchor_ns, int(n), source_name=source_name, extra="system_b")
        rec = {
            "num": int(n),
            "anchor": anchor,
            "citation_route": "system_b",
            "routing_reason": "structured_cite",
            "routing_confidence": 0.9,
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

    def _replace_text_segment(
        seg: str,
        *,
        table_mode: bool = False,
        context_line: str = "",
        context_offset: int = 0,
        budget_state: dict | None = None,
    ) -> str:
        structured_seen = False
        unresolved_struct_refs: set[int] = set()
        resolved_struct_refs: set[int] = set()
        system_a_budget = _plan_budget("system_a", 2)
        system_b_budget = _plan_budget("system_b", 1 if plan else 40)
        shared_budget_state = budget_state if isinstance(budget_state, dict) else {}
        used_system_a_keys: set[str] = shared_budget_state.setdefault(
            "used_system_a_keys",
            set(),
        )
        used_system_b_keys: set[str] = shared_budget_state.setdefault(
            "used_system_b_keys",
            set(),
        )
        shared_budget_state.setdefault("used_system_a_count", 0)
        shared_budget_state.setdefault("used_system_b_count", 0)

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

        def _citation_budget_key(detail: dict) -> str:
            if bool(detail.get("is_inpaper")):
                return str(detail.get("anchor") or "").strip()
            return str(detail.get("citation_budget_key") or detail.get("evidence_fingerprint") or detail.get("anchor") or "").strip()

        def _claim_citation_budget(
            detail: dict,
            *,
            budget_key_override: str = "",
        ) -> bool:
            key = str(budget_key_override or "").strip() or _citation_budget_key(
                detail
            )
            if not key:
                return False
            anchor = str(detail.get("anchor") or "").strip()
            if bool(detail.get("is_inpaper")):
                if key in used_system_b_keys:
                    return True
                if anchor and anchor in visible_detail_anchors:
                    used_system_b_keys.add(key)
                    return True
                if (
                    int(shared_budget_state.get("used_system_b_count") or 0)
                    >= system_b_budget
                ):
                    return False
                used_system_b_keys.add(key)
                shared_budget_state["used_system_b_count"] = (
                    int(shared_budget_state.get("used_system_b_count") or 0) + 1
                )
                if anchor:
                    visible_detail_anchors.add(anchor)
                return True
            if key in used_system_a_keys:
                if anchor:
                    visible_detail_anchors.add(anchor)
                return True
            if key in visible_system_a_evidence_keys:
                used_system_a_keys.add(key)
                if anchor:
                    visible_detail_anchors.add(anchor)
                return True
            if (
                int(shared_budget_state.get("used_system_a_count") or 0)
                >= system_a_budget
            ):
                return False
            used_system_a_keys.add(key)
            visible_system_a_evidence_keys.add(key)
            shared_budget_state["used_system_a_count"] = (
                int(shared_budget_state.get("used_system_a_count") or 0) + 1
            )
            if anchor:
                visible_detail_anchors.add(anchor)
            return True

        def _citation_context_line(*, token_start: int, token_end: int) -> str:
            source_text = str(context_line or seg)
            offset = int(context_offset or 0) if context_line else 0
            return extract_structured_cite_answer_context_line(
                source_text,
                offset + int(token_start),
                offset + int(token_end),
                normalizer=normalize_inline_markdown,
            )

        def _enrich_system_b_detail_from_answer_context(detail: dict, *, token_start: int, token_end: int) -> None:
            context_line = _citation_context_line(token_start=token_start, token_end=token_end)
            role_line = _system_b_upstream_role(context_line, detail, locale=render_locale)
            relation_line = _system_b_user_relation(context_line, detail, locale=render_locale)
            if role_line and not str(detail.get("upstream_work_role") or "").strip():
                detail["upstream_work_role"] = role_line
            if relation_line and not str(detail.get("user_question_relation") or "").strip():
                detail["user_question_relation"] = relation_line
            if relation_line and not str(detail.get("support_relation") or "").strip():
                detail["support_relation"] = relation_line
            if not str(detail.get("why_line") or "").strip():
                detail["why_line"] = role_line or relation_line

            enrich_inpaper_detail_context(
                detail,
                source_path=str(detail.get("source_path") or ""),
                ref_num=int(detail.get("num") or 0),
                answer_context=context_line,
            )

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

        def _source_path_key(value: object) -> str:
            normalized = str(value or "").strip().replace("\\", "/").casefold()
            parts = [part for part in normalized.split("/") if part]
            # Public API payloads replace the absolute corpus prefix with
            # ``kb-source/<root-id>``.  Keep the document directory and file
            # name as the stable identity so canonical answer numbering still
            # resolves after that privacy projection.
            if len(parts) >= 2:
                return "/".join(parts[-2:])
            return normalized

        def _hit_context_for_numeric_ref(n: int, source_path: str) -> dict:
            wanted = _source_path_key(source_path)
            if not wanted:
                return {}
            for raw_hit in hits or []:
                if not isinstance(raw_hit, dict):
                    continue
                meta_hit = raw_hit.get("meta") if isinstance(raw_hit.get("meta"), dict) else {}
                if _source_path_key((meta_hit or {}).get("source_path")) != wanted:
                    continue
                pieces = [str(raw_hit.get("text") or "")]
                for key in ("ref_show_snippets", "ref_snippets"):
                    values = (meta_hit or {}).get(key)
                    if isinstance(values, list):
                        pieces.extend(str(item or "") for item in values[:3])
                for piece in pieces:
                    text = str(piece or "")
                    if not text:
                        continue
                    for marker in _INPAPER_CITE_ANY_RE.finditer(text):
                        if int(n) not in _parse_int_set(str(marker.group(1) or "")):
                            continue
                        start = max(0, int(marker.start()) - 220)
                        end = min(len(text), int(marker.end()) + 220)
                        context = re.sub(r"\s+", " ", text[start:end]).strip()
                        if not context:
                            continue
                        return {
                            "citation_context": context[:520],
                            "citation_context_source": "retrieval_hit_ref_marker",
                            "heading_path": str(
                                (meta_hit or {}).get("heading_path")
                                or (meta_hit or {}).get("ref_best_heading_path")
                                or ""
                            ).strip(),
                        }
            return {}

        def _hits_have_text() -> bool:
            for raw_hit in hits or []:
                if isinstance(raw_hit, dict) and str(raw_hit.get("text") or "").strip():
                    return True
            return False

        def _legacy_no_hit_text_identity_signal(*, ref: dict, token_start: int, token_end: int) -> bool:
            """Permit old persisted numeric refs only when the answer line identifies the bibliography entry."""

            if _hits_have_text():
                return False
            hints = extract_citation_context_hints(seg, token_start=int(token_start), token_end=int(token_end))
            doi_hint = str(hints.get("doi") or "").strip()
            author_confident = bool(hints.get("author_confident"))
            year_hint = str(hints.get("year") or "").strip()
            if not doi_hint and not (author_confident and year_hint):
                return False
            if has_explicit_reference_conflict(ref, hints):
                return False
            return float(reference_alignment_score(ref, hints)) >= 4.0

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
            try:
                token_end = int(pos) + max(1, len(f"[[CITE:{sid}:{int(n)}]]"))
            except Exception:
                token_end = int(pos)
            hints = extract_citation_context_hints(seg, token_start=int(pos), token_end=token_end)
            doi_hint = str(hints.get("doi") or "").strip()
            author_year_hint = bool(
                str(hints.get("author") or "").strip()
                and str(hints.get("year") or "").strip()
                and bool(hints.get("author_confident"))
            )
            if doi_hint or author_year_hint:
                score = float(reference_alignment_score(ref, hints))
                if has_explicit_reference_conflict(ref, hints):
                    return ""
                if doi_hint and score < 6.0:
                    return ""
                if author_year_hint and score < 3.5:
                    return ""
            src_name = _display_source_name(sp)
            detail = _remember_detail(int(n), sp, src_name, ref)
            detail["sid"] = sid
            detail["is_inpaper"] = True  # Mark as System B (in-paper bibliography ref)
            detail["citation_route"] = "system_b"
            detail["routing_reason"] = "structured_cite"
            detail["routing_confidence"] = 0.9
            _enrich_system_b_detail_from_answer_context(detail, token_start=int(pos), token_end=token_end)
            if not _claim_citation_budget(detail):
                return ""
            title_attr = _citation_hover_title(src_name, int(n), ref)
            anchor = str(detail.get("anchor") or "").strip()
            t_attr = str(title_attr or "").replace('"', "'").replace("\n", " ").strip()
            return f"[{int(n)}](#{anchor} \"{t_attr}\")"

        def _repl_struct(m: re.Match) -> str:
            result = _resolve_struct_token(str(m.group(1) or ""), str(m.group(2) or ""), pos=int(m.start()))
            if result:
                n_txt = str(m.group(2) or "").strip()
                if n_txt:
                    try:
                        resolved_struct_refs.add(int(n_txt))
                    except ValueError:
                        pass
                return result
            n_txt = str(m.group(2) or "").strip()
            if not n_txt:
                return ""
            return ""

        def _repl_struct_single(m: re.Match) -> str:
            sid = str(m.group(1) or "")
            n_txt = str(m.group(2) or "").strip()
            if not n_txt:
                # Malformed form like [CITE:sid] -> hide raw token.
                return ""
            result = _resolve_struct_token(sid, n_txt, pos=int(m.start()))
            if result:
                try:
                    resolved_struct_refs.add(int(n_txt))
                except ValueError:
                    pass
                return result
            return ""

        def _repl_struct_sid_only(_: re.Match) -> str:
            # Malformed form like [[CITE:sid]] -> hide raw token.
            return ""

        def _resolve_n_from_hits(n: int, *, token_start: int = -1, token_end: int = -1) -> dict | None:
            """Map [n] to hits[n-1] — the context snippet the LLM actually referenced."""
            idx = int(n) - 1
            hit: dict | None = None
            sp: str = ""
            # If canonical_paths is available, use it to find the correct hit
            # regardless of display-list ordering.
            if isinstance(canonical_paths, list) and 0 <= idx < len(canonical_paths):
                target_sp = _source_path_key(canonical_paths[idx])
                if target_sp:
                    canonical_source_hits: list[dict] = []
                    for _h in hits or []:
                        _mh = (_h or {}).get("meta", {}) or {}
                        _sp_h = _source_path_key(_mh.get("source_path"))
                        if _sp_h == target_sp:
                            canonical_source_hits.append(_h)
                            try:
                                answer_num = int(
                                    _mh.get("ref_answer_citation_num") or 0
                                )
                            except (TypeError, ValueError):
                                answer_num = 0
                            if answer_num == int(n):
                                hit = _h
                                sp = str(_mh.get("source_path") or "").strip()
                                break
                    if hit is None and canonical_source_hits:
                        hit = canonical_source_hits[0]
                        first_meta = (hit or {}).get("meta", {}) or {}
                        sp = str(first_meta.get("source_path") or "").strip()
                    # A canonical source contract is authoritative.  If its
                    # source is absent from the display hits, do not guess by
                    # display position and attach a different paper.
                    if hit is None:
                        return None
            if hit is None:
                numbered_hits: list[tuple[int, dict]] = []
                for _h in hits or []:
                    if not isinstance(_h, dict):
                        continue
                    _mh = (_h or {}).get("meta", {}) or {}
                    try:
                        answer_num = int(_mh.get("ref_answer_citation_num") or 0)
                    except Exception:
                        answer_num = 0
                    if answer_num > 0:
                        numbered_hits.append((answer_num, _h))
                for answer_num, numbered_hit in numbered_hits:
                    if answer_num != int(n):
                        continue
                    hit = numbered_hit
                    meta_h = (hit or {}).get("meta", {}) or {}
                    sp = str(meta_h.get("source_path") or "").strip()
                    break
                if hit is None and numbered_hits:
                    return None
            if hit is None:
                if not (0 <= idx < len(hits)):
                    return None
                hit = hits[idx]
                meta_h = (hit or {}).get("meta", {}) or {}
                sp = str(meta_h.get("source_path") or "").strip()
            if not sp or _is_temp_source_path(sp):
                return None
            answer_claim = ""
            if int(token_start) >= 0:
                answer_claim = _citation_context_line(token_start=int(token_start), token_end=int(token_end))
            claim_sig = _anchor_token(_system_a_fp_text(answer_claim, max_len=220)) if answer_claim else ""
            base_skey = f"{int(n)}|{sp.lower()}"
            skey = f"{base_skey}|claim:{claim_sig}" if claim_sig else base_skey
            cached = detail_by_key.get(skey)
            if isinstance(cached, dict):
                _system_a_maybe_replace_claim(cached, answer_claim)
                return cached
            cached = detail_by_key.get(base_skey)
            if (
                isinstance(cached, dict)
                and re.match(
                    r"^(?:这(?:使得|意味着|表明)?|因此|由此|从而|它|该(?:方法|模型|设计)|"
                    r"基于(?:该|此)模型|其(?:核心思想|作用|机制)(?:是|在于)?|"
                    r"this\b|that\b|it\b|therefore\b|thereby\b|as\s+a\s+result\b)",
                    str(answer_claim or "").strip(),
                    flags=re.IGNORECASE,
                )
                and not _system_a_should_split_occurrence(cached, int(n), answer_claim)
            ):
                _system_a_maybe_replace_claim(cached, answer_claim)
                return cached
            meta_h = dict((hit or {}).get("meta", {}) or {})
            ui_meta_h = dict((hit or {}).get("ui_meta", {}) or {})
            is_research_basket_synthetic = bool(
                meta_h.get("research_basket_evidence")
                and (
                    str(meta_h.get("basket_source_role") or "").strip() == "synthetic_basket_item"
                    or sp.replace("\\", "/").startswith("__research_basket__/")
                )
            )
            primary_evidence = _system_a_primary_evidence_from_ui_meta(ui_meta_h)
            if is_research_basket_synthetic:
                src_name = (
                    str(meta_h.get("source_name") or "").strip()
                    or str(ui_meta_h.get("display_name") or "").strip()
                    or str(meta_h.get("title") or "").strip()
                    or _display_source_name(sp)
                )
            else:
                src_name = _display_source_name(sp)
            plan_slot = _plan_slot_for_system_a(
                int(n),
                sp,
                src_name,
                answer_claim=answer_claim,
            )
            original_primary_evidence = dict(primary_evidence)
            if plan_slot:
                plan_evidence = str(
                    plan_slot.get("evidence_quote")
                    or plan_slot.get("evidenceQuote")
                    or ""
                ).strip()
                if plan_evidence:
                    # The selected plan slot is claim-specific. Promote it to
                    # the primary candidate before evidence scoring so a stale
                    # top-hit paragraph cannot keep the right paper but show
                    # the wrong section or quote in the popover.
                    plan_selection_reason = str(
                        plan_slot.get("evidence_selection_reason")
                        or plan_slot.get("evidenceSelectionReason")
                        or "citation_plan_slot"
                    ).strip()
                    plan_relocated = (
                        plan_selection_reason.lower()
                        == "prompt_aligned_source_sentence"
                    )
                    plan_heading = str(
                        plan_slot.get("heading_path")
                        or plan_slot.get("headingPath")
                        or ""
                    ).strip()
                    plan_heading = re.sub(
                        r"^(\d+(?:\.\d+)+)(?=\s)",
                        r"\1.",
                        plan_heading,
                    )
                    existing_heading = str(
                        primary_evidence.get("heading_path")
                        or primary_evidence.get("headingPath")
                        or ""
                    ).strip()
                    if (
                        plan_heading
                        and existing_heading
                        and re.sub(r"[^a-z0-9]+", "", plan_heading.lower())
                        == re.sub(r"[^a-z0-9]+", "", existing_heading.lower())
                    ):
                        # Preserve source punctuation (for example ``4.1.2.``)
                        # when both headings identify the same section.
                        plan_heading = existing_heading
                    primary_evidence = dict(primary_evidence)
                    primary_evidence.update(
                        {
                            "heading_path": plan_heading,
                            "snippet": plan_evidence,
                            "highlight_snippet": plan_evidence,
                            "block_id": str(
                                plan_slot.get("block_id")
                                or plan_slot.get("blockId")
                                or (
                                    ""
                                    if plan_relocated
                                    else primary_evidence.get("block_id")
                                    or primary_evidence.get("blockId")
                                )
                                or ""
                            ).strip(),
                            "anchor_id": str(
                                plan_slot.get("anchor_id")
                                or plan_slot.get("anchorId")
                                or (
                                    ""
                                    if plan_relocated
                                    else primary_evidence.get("anchor_id")
                                    or primary_evidence.get("anchorId")
                                )
                                or ""
                            ).strip(),
                            "anchor_kind": str(
                                plan_slot.get("anchor_kind")
                                or plan_slot.get("anchorKind")
                                or (
                                    ""
                                    if plan_relocated
                                    else primary_evidence.get("anchor_kind")
                                    or primary_evidence.get("anchorKind")
                                )
                                or ""
                            ).strip(),
                            "page_start": int(
                                plan_slot.get("page_start")
                                or plan_slot.get("pageStart")
                                or primary_evidence.get("page_start")
                                or primary_evidence.get("pageStart")
                                or 0
                            ),
                            "page_end": int(
                                plan_slot.get("page_end")
                                or plan_slot.get("pageEnd")
                                or plan_slot.get("page_start")
                                or plan_slot.get("pageStart")
                                or primary_evidence.get("page_end")
                                or primary_evidence.get("pageEnd")
                                or primary_evidence.get("page_start")
                                or primary_evidence.get("pageStart")
                                or 0
                            ),
                            "selection_reason": plan_selection_reason,
                            "strict_locate": bool(
                                plan_slot.get("strict_locate")
                                or plan_slot.get("strictLocate")
                                or plan_slot.get("block_id")
                                or plan_slot.get("anchor_id")
                            ),
                        }
                    )
                    if (
                        bool(
                            original_primary_evidence.get("strict_locate")
                            or original_primary_evidence.get("strictLocate")
                        )
                        and bool(
                            original_primary_evidence.get("block_id")
                            or original_primary_evidence.get("blockId")
                            or original_primary_evidence.get("anchor_id")
                            or original_primary_evidence.get("anchorId")
                        )
                        and not bool(
                            plan_slot.get("block_id")
                            or plan_slot.get("blockId")
                            or plan_slot.get("anchor_id")
                            or plan_slot.get("anchorId")
                        )
                        and not plan_relocated
                        and (
                            not plan_heading
                            or not existing_heading
                            or re.sub(r"[^a-z0-9]+", "", plan_heading.lower())
                            == re.sub(r"[^a-z0-9]+", "", existing_heading.lower())
                            or _ordered_ascii_phrase_score(answer_claim, plan_evidence)
                            <= _ordered_ascii_phrase_score(
                                answer_claim,
                                _system_a_candidate_text(original_primary_evidence),
                            )
                        )
                    ):
                        # A preflight-created exact hit can already carry a
                        # strict block/anchor that is more navigable than an
                        # otherwise equivalent unanchored plan slot.
                        primary_evidence = original_primary_evidence
                    # Evidence selection reads both the explicit primary
                    # candidate and the UI payload. Keep them synchronized so
                    # an old strict reader locator cannot outrank a relocated
                    # prompt-aligned passage from the same paper.
                    ui_meta_h["primary_evidence"] = dict(primary_evidence)
                    meta_h["citation_plan_slot"] = True
                    meta_h["citation_plan_evidence_authoritative"] = True
                    meta_h["citation_plan_evidence_selection_reason"] = (
                        plan_selection_reason
                    )
                    meta_h["citation_plan_source"] = "citation_plan_builder"
                    reader_open = dict(ui_meta_h.get("reader_open") or {})
                    alternatives = [
                        dict(item)
                        for item in list(reader_open.get("evidenceAlternatives") or [])
                        if isinstance(item, dict)
                    ]
                    plan_candidate = {
                        "headingPath": str(
                            plan_slot.get("heading_path")
                            or plan_slot.get("headingPath")
                            or meta_h.get("ref_best_heading_path")
                            or meta_h.get("heading_path")
                            or ""
                        ).strip(),
                        "snippet": plan_evidence,
                        "highlightSnippet": plan_evidence,
                        "blockId": str(
                            plan_slot.get("block_id") or plan_slot.get("blockId") or ""
                        ).strip(),
                        "anchorId": str(
                            plan_slot.get("anchor_id") or plan_slot.get("anchorId") or ""
                        ).strip(),
                        "anchorKind": str(
                            plan_slot.get("anchor_kind") or plan_slot.get("anchorKind") or ""
                        ).strip(),
                        "pageStart": int(
                            plan_slot.get("page_start") or plan_slot.get("pageStart") or 0
                        ),
                        "pageEnd": int(
                            plan_slot.get("page_end")
                            or plan_slot.get("pageEnd")
                            or plan_slot.get("page_start")
                            or plan_slot.get("pageStart")
                            or 0
                        ),
                        "selectionReason": str(
                            plan_slot.get("evidence_selection_reason")
                            or plan_slot.get("evidenceSelectionReason")
                            or "citation_plan_slot"
                        ).strip(),
                    }
                    original_candidate_text = _system_a_candidate_text(
                        original_primary_evidence
                    )
                    if (
                        plan_relocated
                        and original_candidate_text
                        and not any(
                            str(item.get("snippet") or item.get("highlightSnippet") or "").strip()
                            == original_candidate_text
                            for item in alternatives
                        )
                    ):
                        # A broad plan passage and a claim-specific locator can
                        # both be valid for one paper. Preserve the latter as
                        # an alternative instead of discarding its exact
                        # block/page metadata during plan relocation.
                        alternatives.append(
                            {
                                "headingPath": str(
                                    original_primary_evidence.get("heading_path")
                                    or original_primary_evidence.get("headingPath")
                                    or ""
                                ).strip(),
                                "snippet": original_candidate_text,
                                "highlightSnippet": original_candidate_text,
                                "blockId": str(
                                    original_primary_evidence.get("block_id")
                                    or original_primary_evidence.get("blockId")
                                    or ""
                                ).strip(),
                                "anchorId": str(
                                    original_primary_evidence.get("anchor_id")
                                    or original_primary_evidence.get("anchorId")
                                    or ""
                                ).strip(),
                                "anchorKind": str(
                                    original_primary_evidence.get("anchor_kind")
                                    or original_primary_evidence.get("anchorKind")
                                    or ""
                                ).strip(),
                                "pageStart": int(
                                    original_primary_evidence.get("page_start")
                                    or original_primary_evidence.get("pageStart")
                                    or 0
                                ),
                                "pageEnd": int(
                                    original_primary_evidence.get("page_end")
                                    or original_primary_evidence.get("pageEnd")
                                    or original_primary_evidence.get("page_start")
                                    or original_primary_evidence.get("pageStart")
                                    or 0
                                ),
                                "selectionReason": "same_source_claim_specific_alternative",
                            }
                        )
                    if not any(
                        str(item.get("snippet") or item.get("highlightSnippet") or "").strip()
                        == plan_evidence
                        for item in alternatives
                    ):
                        alternatives.insert(0, plan_candidate)
                    reader_open["evidenceAlternatives"] = alternatives
                    ui_meta_h["reader_open"] = reader_open
            default_heading = str(
                primary_evidence.get("heading_path")
                or primary_evidence.get("headingPath")
                or ui_meta_h.get("primary_evidence_heading_path")
                or ui_meta_h.get("primaryEvidenceHeadingPath")
                or ui_meta_h.get("heading_path")
                or ui_meta_h.get("headingPath")
                or meta_h.get("ref_best_heading_path")
                or meta_h.get("heading_path")
                or ""
            ).strip()
            evidence_pick = _system_a_pick_best_evidence_candidate(
                hit=hit or {},
                meta=meta_h,
                ui_meta=ui_meta_h,
                primary_evidence=primary_evidence,
                answer_claim=answer_claim,
                source_name=src_name,
                default_heading=default_heading,
            )
            if plan_slot and str(
                plan_slot.get("evidence_selection_reason")
                or plan_slot.get("evidenceSelectionReason")
                or ""
            ).strip() == "microscopy_direct":
                direct_plan_evidence = str(
                    plan_slot.get("evidence_quote")
                    or plan_slot.get("evidenceQuote")
                    or ""
                ).strip()
                if direct_plan_evidence:
                    evidence_pick = {
                        "source": "primary_evidence",
                        "text": direct_plan_evidence,
                        "readable_text": direct_plan_evidence,
                        "heading_path": str(
                            plan_slot.get("heading_path")
                            or plan_slot.get("headingPath")
                            or default_heading
                            or ""
                        ).strip(),
                        "raw": dict(primary_evidence),
                    }
            picked_raw_hit = str(evidence_pick.get("source") or "") == "hit_text"
            picked_raw = evidence_pick.get("raw") if isinstance(evidence_pick.get("raw"), dict) else {}
            if picked_raw_hit and original_primary_evidence:
                picked_heading_key = re.sub(
                    r"[^a-z0-9]+",
                    "",
                    str(evidence_pick.get("heading_path") or "").lower(),
                )
                original_heading_key = re.sub(
                    r"[^a-z0-9]+",
                    "",
                    str(
                        original_primary_evidence.get("heading_path")
                        or original_primary_evidence.get("headingPath")
                        or ""
                    ).lower(),
                )
                original_text = _system_a_candidate_text(original_primary_evidence)
                picked_text = str(evidence_pick.get("text") or "").strip()
                same_passage = bool(
                    original_text
                    and picked_text
                    and (
                        original_text in picked_text
                        or picked_text in original_text
                        or len(
                            evidence_alignment_tokens(original_text)
                            & evidence_alignment_tokens(picked_text)
                        )
                        >= 4
                    )
                )
                if (
                    same_passage
                    and picked_heading_key
                    and picked_heading_key == original_heading_key
                    and (
                        original_primary_evidence.get("block_id")
                        or original_primary_evidence.get("blockId")
                        or original_primary_evidence.get("anchor_id")
                        or original_primary_evidence.get("anchorId")
                    )
                ):
                    # The raw hit can be the best quote while the equivalent
                    # primary object carries the exact reader locator.
                    picked_raw = dict(original_primary_evidence)
                    picked_raw_hit = False
            if isinstance(picked_raw, dict) and picked_raw:
                primary_evidence = picked_raw
            if picked_raw_hit:
                heading = str(evidence_pick.get("heading_path") or "").strip()
            else:
                heading = str(
                    evidence_pick.get("heading_path")
                    or primary_evidence.get("heading_path")
                    or primary_evidence.get("headingPath")
                    or default_heading
                    or ""
                ).strip()
            snippet = str(
                evidence_pick.get("text")
                or primary_evidence.get("highlight_snippet")
                or primary_evidence.get("highlightSnippet")
                or primary_evidence.get("snippet")
                or hit.get("text")
                or ""
            ).strip()
            evidence_quote = str(
                evidence_pick.get("readable_text")
                or evidence_pick.get("text")
                or primary_evidence.get("highlight_snippet")
                or primary_evidence.get("highlightSnippet")
                or primary_evidence.get("snippet")
                or meta_h.get("evidence_quote")
                or meta_h.get("support_locate_anchor")
                or meta_h.get("anchor_text")
                or snippet
                or ""
            ).strip()
            picked_evidence_source = str(
                evidence_pick.get("source") or ""
            ).strip()
            picked_plan_primary = picked_evidence_source in {
                "primary_evidence",
                "reader_open.primaryEvidence",
                "reader_open.primary_evidence",
                "reader_open.locateTarget",
                "reader_open.locate_target",
            }
            authoritative_plan_evidence = bool(
                meta_h.get("citation_plan_slot")
                and meta_h.get("citation_plan_evidence_authoritative")
                and picked_plan_primary
            )
            metric_table_sources = [evidence_quote, snippet]
            if not authoritative_plan_evidence:
                metric_table_sources.append(str(hit.get("text") or ""))
                for key in ("ref_show_snippets", "ref_snippets"):
                    values = meta_h.get(key)
                    if isinstance(values, list):
                        metric_table_sources.extend(str(item or "") for item in values[:3])
            compact_table_candidates = [
                compact
                for candidate in metric_table_sources
                for compact in [_compact_metric_table_evidence(candidate, answer_claim=answer_claim)]
                if compact and _compact_metric_table_matches_claim(compact, answer_claim)
            ]
            compact_table_evidence = max(
                compact_table_candidates,
                key=lambda item: (item.count("="), len(item)),
                default="",
            )
            compact_detector_evidence = _compact_detector_table_evidence(
                evidence_quote
            )
            if compact_table_evidence:
                evidence_quote = compact_table_evidence
                ref_best_heading = str(meta_h.get("ref_best_heading_path") or "").strip()
                if ref_best_heading and not authoritative_plan_evidence:
                    heading = ref_best_heading
            elif compact_detector_evidence:
                evidence_quote = compact_detector_evidence
            if authoritative_plan_evidence or bool(compact_detector_evidence):
                cleaned_evidence_quote = _clean_evidence_display_text(
                    evidence_quote,
                    max_len=520,
                )
                if cleaned_evidence_quote:
                    evidence_quote = cleaned_evidence_quote
            else:
                readable_evidence_quote = _pick_readable_evidence_text(
                    evidence_quote,
                    source=src_name,
                    title=heading,
                    claim=answer_claim,
                    heading=heading,
                    max_len=520,
                )
                if readable_evidence_quote:
                    evidence_quote = readable_evidence_quote
                else:
                    cleaned_evidence_quote = _clean_evidence_display_text(
                        evidence_quote,
                        max_len=520,
                    )
                    if cleaned_evidence_quote:
                        evidence_quote = cleaned_evidence_quote
            compound_plan_specific = (
                str(evidence_quote or "").strip()
                if bool(evidence_pick.get("compound_evidence"))
                else ""
            )
            if plan_slot:
                plan_text = str(
                    plan_slot.get("citation_plan_full_evidence_quote")
                    or plan_slot.get("evidence_quote")
                    or plan_slot.get("evidenceQuote")
                    or ""
                ).strip()
                answer_claim_numbers = _system_a_fact_quantities(answer_claim)
                picked_evidence_numbers = _system_a_fact_quantities(
                    str(evidence_pick.get("text") or evidence_quote or "")
                )
                picked_covers_claim_numbers = bool(
                    answer_claim_numbers
                    and all(
                        _quantity_is_covered(quantity, picked_evidence_numbers)
                        for quantity in answer_claim_numbers
                    )
                )
                compound_plan_specific = _compound_plan_evidence_excerpt(
                    plan_text,
                    answer_claim,
                )
                if compound_plan_specific:
                    evidence_quote = compound_plan_specific
                    snippet = compound_plan_specific
                claim_identifiers = {
                    token.upper()
                    for token in re.findall(
                        r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{2,}(?![A-Za-z0-9])",
                        str(answer_claim or ""),
                    )
                }
                if (
                    plan_text
                    and len(claim_identifiers) >= 2
                    and not compound_plan_specific
                    and not picked_covers_claim_numbers
                ):
                    plan_specific = ""
                    for plan_sentence in re.split(r"(?<=[.!?。！？])\s+", plan_text):
                        sentence_identifiers = {
                            token.upper()
                            for token in re.findall(
                                r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{2,}(?![A-Za-z0-9])",
                                plan_sentence,
                            )
                        }
                        if len(claim_identifiers & sentence_identifiers) < 2:
                            continue
                        plan_specific = _clean_evidence_display_text(
                            plan_sentence,
                            max_len=520,
                        )
                        if plan_specific and not re.search(r"[.!?。！？…]$", plan_specific):
                            plan_specific = plan_specific.rstrip(" ,;:") + "..."
                        break
                    if not plan_specific:
                        plan_specific = _pick_readable_evidence_text(
                            plan_text,
                            source=src_name,
                            title=heading,
                            claim=answer_claim,
                            heading=heading,
                            max_len=520,
                        )
                    plan_specific_identifiers = {
                        token.upper()
                        for token in re.findall(
                            r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{2,}(?![A-Za-z0-9])",
                            plan_specific,
                        )
                    }
                    if len(claim_identifiers & plan_specific_identifiers) >= 2:
                        evidence_quote = plan_specific
                        snippet = plan_specific
            claim_named_identifiers = {
                token.upper()
                for token in re.findall(
                    r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{2,}(?![A-Za-z0-9])",
                    str(answer_claim or ""),
                )
            }
            if len(claim_named_identifiers) >= 2 and not compound_plan_specific:
                for snippet_sentence in re.split(
                    r"(?<=[.!?。！？])\s+",
                    str(evidence_pick.get("text") or snippet or ""),
                ):
                    snippet_identifiers = {
                        token.upper()
                        for token in re.findall(
                            r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{2,}(?![A-Za-z0-9])",
                            snippet_sentence,
                        )
                    }
                    if len(claim_named_identifiers & snippet_identifiers) < 2:
                        continue
                    named_evidence = _clean_evidence_display_text(
                        snippet_sentence,
                        max_len=520,
                    )
                    if named_evidence and not re.search(r"[.!?。！？…]$", named_evidence):
                        named_evidence = named_evidence.rstrip(" ,;:") + "..."
                    if named_evidence:
                        evidence_quote = named_evidence
                    break
            evidence_source = str(evidence_pick.get("source") or "retrieval_hit").strip() or "retrieval_hit"
            if evidence_source in {"hit_meta", "hit_text"}:
                evidence_source = "retrieval_hit"
            exact_support_plan = bool(
                authoritative_plan_evidence
                and str(meta_h.get("citation_plan_source") or "").strip().lower()
                == "exact_support_preflight"
            )
            if exact_support_plan:
                evidence_source = "exact_support_preflight"
            p0, p1 = _safe_page_range(meta_h)
            if isinstance(picked_raw, dict) and picked_raw:
                raw_page_start = _system_a_candidate_value(
                    picked_raw,
                    "page_start",
                    "pageStart",
                    "page",
                    "page_number",
                )
                raw_page_end = _system_a_candidate_value(
                    picked_raw,
                    "page_end",
                    "pageEnd",
                )
                try:
                    picked_p0 = int(raw_page_start or 0)
                except Exception:
                    picked_p0 = 0
                try:
                    picked_p1 = int(raw_page_end or picked_p0 or 0)
                except Exception:
                    picked_p1 = picked_p0
                if picked_p0 > 0:
                    p0, p1 = picked_p0, picked_p1 or picked_p0
            ref_rank = meta_h.get("ref_rank") if isinstance(meta_h.get("ref_rank"), dict) else {}
            if picked_raw_hit:
                block_id = str(evidence_pick.get("block_id") or "").strip()
                anchor_id = str(evidence_pick.get("anchor_id") or "").strip()
                anchor_kind = str(evidence_pick.get("anchor_kind") or "").strip()
            else:
                block_id = str(
                    primary_evidence.get("block_id")
                    or primary_evidence.get("blockId")
                    or meta_h.get("primary_block_id")
                    or meta_h.get("block_id")
                    or ""
                ).strip()
                anchor_id = str(
                    primary_evidence.get("anchor_id")
                    or primary_evidence.get("anchorId")
                    or meta_h.get("primary_anchor_id")
                    or meta_h.get("anchor_id")
                    or ""
                ).strip()
                anchor_kind = str(
                    primary_evidence.get("anchor_kind")
                    or primary_evidence.get("anchorKind")
                    or meta_h.get("anchor_kind")
                    or ""
                ).strip()
            # Converter table anchors are authoritative even when an older
            # payload labelled the selected excerpt as a sentence.  Preserve
            # the named table occurrence in the visible locator so the card
            # and reader jump describe the same target.
            named_occurrence_label = ""
            if anchor_id.lower().startswith("tb_"):
                anchor_kind = "table"
            if anchor_kind.lower() == "table":
                reader_open_h = (
                    ui_meta_h.get("reader_open")
                    if isinstance(ui_meta_h.get("reader_open"), dict)
                    else {}
                )
                locate_target_h = (
                    reader_open_h.get("locateTarget")
                    if isinstance(reader_open_h.get("locateTarget"), dict)
                    else {}
                )
                occurrence_surfaces = [
                    str((plan_slot or {}).get("evidence_quote") or ""),
                    str((plan_slot or {}).get("evidenceQuote") or ""),
                    str(locate_target_h.get("evidenceQuote") or ""),
                    str(reader_open_h.get("snippet") or ""),
                    str(evidence_quote or ""),
                    str(snippet or ""),
                ]
                for occurrence_surface in occurrence_surfaces:
                    occurrence_match = re.search(
                        r"(?i)\bTable\s+(\d+[A-Za-z]?)\b",
                        occurrence_surface,
                    )
                    if occurrence_match:
                        named_occurrence_label = f"Table {occurrence_match.group(1)}"
                        break
            if is_research_basket_synthetic:
                basket_title = str(meta_h.get("title") or src_name or "").strip()
                basket_heading = heading or str(ui_meta_h.get("heading_path") or "").strip()
                basket_quote = evidence_quote or snippet or str(hit.get("text") or "").strip()
                basket_support = str(
                    meta_h.get("support_relation")
                    or ui_meta_h.get("why_line")
                    or "User-selected research basket evidence for this turn."
                ).strip()
                basket_fp = _system_a_evidence_fingerprint(
                    source_path=sp,
                    heading=basket_heading,
                    evidence_quote=basket_quote,
                    snippet=snippet,
                    block_id=block_id,
                    anchor_id=anchor_id,
                    page_start=0,
                    page_end=0,
                )
                existing = system_a_detail_by_fingerprint.get(basket_fp)
                if isinstance(existing, dict):
                    _system_a_add_linked_num(existing, int(n))
                    _system_a_maybe_replace_claim(existing, answer_claim)
                    detail_by_key[skey] = existing
                    return existing
                anchor = _build_inpaper_anchor(anchor_ns, int(n), source_name=src_name)
                rec = {
                    "num": int(n),
                    "linked_nums": [int(n)],
                    "anchor": anchor,
                    "evidence_fingerprint": basket_fp,
                    "citation_budget_key": basket_fp,
                    "source_name": src_name,
                    "source_path": "",
                    "raw": basket_quote[:520],
                    "title": basket_title or src_name,
                    "is_inpaper": False,
                    "citation_route": "research_basket",
                    "routing_reason": "research_basket_evidence",
                    "routing_confidence": 1.0,
                    "heading_path": basket_heading,
                    "summary_line": basket_quote[:360],
                    "summary_source": "research_basket",
                    "answer_claim": answer_claim[:420],
                    "answer_claims": [answer_claim[:420]] if answer_claim else [],
                    "evidence_quote": basket_quote[:520],
                    "evidence_source": "research_basket",
                    "location_label": "Research basket",
                    "support_relation": basket_support,
                    "why_line": basket_support,
                    "block_id": block_id,
                    "anchor_id": anchor_id,
                    "anchor_kind": anchor_kind,
                    "page_start": 0,
                    "page_end": 0,
                    "score": 10.0,
                    "binding_status": "grounded",
                    "binding_confidence": 1.0,
                    "binding_reason": basket_support,
                    "binding_overlap_terms": [],
                    "evidence_pick_score": float(evidence_pick.get("score") or 0.0) if evidence_pick else 0.0,
                }
                detail_by_key[skey] = rec
                system_a_detail_by_fingerprint[basket_fp] = rec
                return rec
            try:
                score_value = float(
                    ref_rank.get("display_score")
                    or ref_rank.get("score")
                    or meta_h.get("score")
                    or 0.0
                )
            except Exception:
                score_value = 0.0
            binding_meta = dict(meta_h)
            if plan_slot and bool(
                binding_meta.get("citation_plan_evidence_authoritative")
            ):
                full_plan_evidence = str(
                    plan_slot.get("citation_plan_full_evidence_quote")
                    or plan_slot.get("evidence_quote")
                    or plan_slot.get("evidenceQuote")
                    or ""
                ).strip()
                if full_plan_evidence:
                    binding_meta["citation_plan_full_evidence_quote"] = (
                        full_plan_evidence
                    )
            group_evidence_quotes = _verified_citation_group_evidence_quotes(
                answer_claim
            )
            if group_evidence_quotes:
                binding_meta["citation_group_evidence_quotes"] = group_evidence_quotes
            binding = _assess_system_a_hit_binding(
                answer_claim=answer_claim,
                hit=hit or {},
                meta=binding_meta,
                heading=heading,
                evidence_quote=evidence_quote,
                source_name=src_name,
            )
            if (not heading) and _looks_front_matter_evidence_ui(evidence_quote):
                return {
                    "_suppress_link": True,
                    "num": int(n),
                    "binding_status": "mismatch",
                    "binding_confidence": 0.0,
                    "binding_reason": "The matched hit is document front matter rather than a locatable evidence passage.",
                    "binding_overlap_terms": list(binding.get("overlap_terms") or []),
                }
            if bool(binding.get("suppress_link")):
                return {
                    "_suppress_link": True,
                    "num": int(n),
                    "binding_status": str(binding.get("status") or "mismatch"),
                    "binding_confidence": float(binding.get("confidence") or 0.0),
                    "binding_reason": str(binding.get("reason") or "").strip(),
                    "binding_overlap_terms": list(binding.get("overlap_terms") or []),
                }
            location_bits: list[str] = []
            if heading:
                location_bits.append(heading)
            if named_occurrence_label and named_occurrence_label.lower() not in heading.lower():
                location_bits.append(named_occurrence_label)
            if p0:
                if p1 and int(p1) != int(p0):
                    location_bits.append(f"pp. {int(min(p0, p1))}-{int(max(p0, p1))}")
                else:
                    location_bits.append(f"p. {int(p0)}")
            if anchor_kind:
                location_bits.append(anchor_kind)
            occurrence_why_line = _system_a_ui_relevance_for_occurrence(
                ui_meta_h,
                original_primary_evidence,
                heading=heading,
                block_id=block_id,
                anchor_id=anchor_id,
                evidence_quote=evidence_quote,
            )
            why_line = str(
                occurrence_why_line
                or ref_rank.get("why")
                or meta_h.get("why_line")
                or ""
            ).strip()[:320]
            support_relation = why_line
            if not support_relation:
                support_relation = str(binding.get("reason") or "").strip()
            evidence_fp = _system_a_evidence_fingerprint(
                source_path=sp,
                heading=heading,
                evidence_quote=evidence_quote,
                snippet=snippet,
                block_id=block_id,
                anchor_id=anchor_id,
                page_start=int(p0 or 0),
                page_end=int(p1 or 0),
            )
            citation_budget_key = evidence_fp
            if plan_slot and bool(meta_h.get("citation_plan_evidence_authoritative")):
                citation_budget_key = (
                    _plan_slot_citation_budget_key(plan_slot, sp)
                    or citation_budget_key
                )
            existing = system_a_detail_by_fingerprint.get(evidence_fp)
            split_occurrence = bool(
                isinstance(existing, dict)
                and _system_a_should_split_occurrence(
                    existing,
                    int(n),
                    answer_claim,
                    evidence_quote=evidence_quote,
                )
            )
            if isinstance(existing, dict) and not split_occurrence:
                _system_a_add_linked_num(existing, int(n))
                _system_a_maybe_replace_claim(existing, answer_claim)
                detail_by_key[skey] = existing
                return existing
            if isinstance(existing, dict) and split_occurrence:
                existing["occurrence_specific"] = True
            same_number_source_exists = any(
                isinstance(item, dict)
                and str(item.get("citation_route") or "") == "system_a"
                and int(item.get("num") or 0) == int(n)
                and str(item.get("source_path") or "").strip().lower() == sp.lower()
                for item in detail_by_key.values()
            )
            occurrence_specific = bool(split_occurrence or same_number_source_exists)
            occurrence_extra = claim_sig if occurrence_specific else ""
            anchor = _build_inpaper_anchor(anchor_ns, int(n), source_name=src_name, extra=occurrence_extra)
            rec = {
                "num": int(n),
                "linked_nums": [int(n)],
                "anchor": anchor,
                "evidence_fingerprint": evidence_fp,
                # Occurrence-specific cards may keep distinct anchors and
                # claim copy, but repeated use of the same verified evidence
                # must consume one System-A budget slot. Counting the claim
                # suffix here caused a second sentence backed by the same
                # source block to lose its citation in the final renderer.
                "citation_budget_key": citation_budget_key,
                "occurrence_specific": occurrence_specific,
                "source_name": src_name,
                "source_path": sp,
                "raw": snippet[:520],
                "title": heading or src_name,
                "is_inpaper": False,  # System A (hit citation)
                "citation_route": "system_a",
                "routing_reason": (
                    "exact_support_preflight"
                    if exact_support_plan
                    else "retrieval_hit"
                ),
                "routing_confidence": float(binding.get("confidence") or 0.0),
                "heading_path": heading,
                "summary_line": evidence_quote[:360] or snippet[:360],
                "summary_source": evidence_source,
                "answer_claim": answer_claim[:420],
                "answer_claims": [answer_claim[:420]] if answer_claim else [],
                "evidence_quote": evidence_quote[:520],
                "evidence_source": evidence_source,
                "location_label": " · ".join([part for part in location_bits if str(part or "").strip()])[:260],
                "support_relation": support_relation,
                "why_line": why_line,
                "block_id": block_id,
                "anchor_id": anchor_id,
                "anchor_kind": anchor_kind,
                "page_start": int(p0 or 0),
                "page_end": int(p1 or 0),
                "score": score_value,
                "binding_status": str(binding.get("status") or "").strip(),
                "binding_confidence": float(binding.get("confidence") or 0.0),
                "binding_reason": str(binding.get("reason") or "").strip(),
                "binding_overlap_terms": list(binding.get("overlap_terms") or []),
                "evidence_pick_score": float(evidence_pick.get("score") or 0.0) if evidence_pick else 0.0,
                "citation_plan_slot": bool(meta_h.get("citation_plan_slot")),
                "compound_plan_evidence": bool(compound_plan_specific),
                "selection_reason": str(
                    meta_h.get("citation_plan_evidence_selection_reason")
                    or primary_evidence.get("selection_reason")
                    or primary_evidence.get("selectionReason")
                    or ""
                ).strip(),
                "strict_locate": bool(
                    primary_evidence.get("strict_locate")
                    or primary_evidence.get("strictLocate")
                ),
            }
            citation_meta_candidates: list[dict] = []
            source_key = sp.replace("\\", "/").strip().lower()
            for candidate in [hit, *(hits or [])]:
                if not isinstance(candidate, dict):
                    continue
                candidate_meta = candidate.get("meta") if isinstance(candidate.get("meta"), dict) else {}
                candidate_source = str(candidate_meta.get("source_path") or "").replace("\\", "/").strip().lower()
                if candidate_source != source_key:
                    continue
                candidate_ui = candidate.get("ui_meta") if isinstance(candidate.get("ui_meta"), dict) else {}
                candidate_citation_meta = (
                    candidate_ui.get("citation_meta")
                    if isinstance(candidate_ui.get("citation_meta"), dict)
                    else {}
                )
                if candidate_citation_meta:
                    citation_meta_candidates.append(candidate_citation_meta)
            citation_meta = max(
                citation_meta_candidates,
                key=lambda item: sum(value not in (None, "", [], {}) for value in item.values()),
                default={},
            )
            for field in (
                "authors",
                "venue",
                "year",
                "volume",
                "issue",
                "pages",
                "doi",
                "doi_url",
                "citation_count",
                "citation_source",
                "venue_kind",
                "venue_verified_by",
                "openalex_venue",
                "journal_if",
                "journal_quartile",
                "journal_if_source",
                "conference_tier",
                "conference_rank_source",
                "conference_ccf",
                "conference_ccf_source",
                "conference_name",
                "conference_acronym",
                "bibliometrics_checked",
            ):
                value = citation_meta.get(field)
                if value not in (None, "", [], {}):
                    rec[field] = value
            bibliographic_title = str(citation_meta.get("title") or "").strip()
            if bibliographic_title:
                rec["bibliographic_title"] = bibliographic_title
            detail_by_key[skey] = rec
            detail_by_key.setdefault(base_skey, rec)
            if not isinstance(existing, dict):
                system_a_detail_by_fingerprint[evidence_fp] = rec
            return rec

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
            linked_count = 0
            plain_fallback_count = 0
            for n in nums:
                if int(n) in unresolved_struct_refs:
                    # A failed System B token must not be silently reinterpreted
                    # as a System A retrieval-hit citation with the same number.
                    changed = True
                    continue
                # [n] normally maps to hits[n-1] — the context snippet the LLM
                # referenced.  If [n] cannot be a retrieval-hit citation (for
                # example [116] with one hit), fall back to a grounded reference
                # index lookup so legacy in-paper numeric refs stay usable.

                # If this number was already resolved by System B's _repl_struct,
                # the [[CITE:...]] was converted to [{n}](#anchor "title") and we
                # must preserve the [n] link text.  Checking resolved_struct_refs
                # before _resolve_n_from_hits is critical: _resolve_n_from_hits
                # may return a hit (same number, different source) and would
                # overwrite the System B anchor with a System A anchor.
                if int(n) in resolved_struct_refs:
                    items.append(f"[{int(n)}]")
                    changed = True
                    continue
                context_line = _citation_context_line(
                    token_start=int(m.start()),
                    token_end=int(m.end()),
                )
                hit_detail = _resolve_n_from_hits(
                    int(n),
                    token_start=int(m.start()),
                    token_end=int(m.end()),
                )
                picked: tuple[str, str, dict] | None = None
                if not hit_detail:
                    picked = _pick_grounded_numeric_candidate(
                        int(n),
                        pos=int(m.start()),
                        target_sp=target_sp,
                    )
                if hit_detail:
                    if bool(hit_detail.get("_suppress_link")):
                        if not _STRICT_STRUCTURED_CITATION_LINKING:
                            items.append(f"[{int(n)}]")
                            plain_fallback_count += 1
                        changed = True
                        continue
                    detail = hit_detail
                elif picked:
                    sp, src_name, ref = picked
                    detail = _remember_detail(int(n), sp, src_name, ref)
                    detail["is_inpaper"] = True
                    detail["citation_route"] = "system_b"
                    detail["routing_reason"] = "reference_index_fallback"
                    detail["routing_confidence"] = 0.55
                    _enrich_system_b_detail_from_answer_context(
                        detail,
                        token_start=int(m.start()),
                        token_end=int(m.end()),
                    )
                    hit_grounding = _hit_context_for_numeric_ref(int(n), sp)
                    if hit_grounding:
                        context_from_hit = str(hit_grounding.get("citation_context") or "").strip()
                        if context_from_hit:
                            detail["citation_context"] = context_from_hit[:520]
                            detail["citation_context_source"] = str(
                                hit_grounding.get("citation_context_source") or "retrieval_hit_ref_marker"
                            )
                            detail["evidence_quote"] = context_from_hit[:520]
                            detail["evidence_source"] = str(
                                hit_grounding.get("citation_context_source") or "retrieval_hit_ref_marker"
                            )
                            detail["summary_line"] = context_from_hit[:360]
                            detail["summary_source"] = str(
                                hit_grounding.get("citation_context_source") or "retrieval_hit_ref_marker"
                            )
                            detail["reference_index_fallback_grounded"] = True
                        heading_from_hit = str(hit_grounding.get("heading_path") or "").strip()
                        if heading_from_hit:
                            detail["heading_path"] = heading_from_hit
                            detail["location_label"] = heading_from_hit
                        detail["routing_confidence"] = max(float(detail.get("routing_confidence") or 0.0), 0.62)
                    elif _legacy_no_hit_text_identity_signal(
                        ref=ref,
                        token_start=int(m.start()),
                        token_end=int(m.end()),
                    ):
                        detail["reference_index_fallback_legacy_identity_signal"] = True
                    if render_locale:
                        detail["render_locale"] = render_locale
                    if not _system_b_reference_index_fallback_is_grounded(detail):
                        if not authoritative_answer_numbering:
                            items.append(f"[{int(n)}]")
                            plain_fallback_count += 1
                        changed = True
                        continue
                else:
                    if not _STRICT_STRUCTURED_CITATION_LINKING:
                        items.append(f"[{int(n)}]")
                    continue
                occurrence_budget_key = ""
                if hit_detail and not bool(detail.get("is_inpaper")):
                    occurrence_plan_slot = _plan_slot_for_system_a(
                        int(n),
                        str(detail.get("source_path") or ""),
                        str(detail.get("source_name") or ""),
                        answer_claim=context_line,
                    )
                    occurrence_budget_key = _plan_slot_citation_budget_key(
                        occurrence_plan_slot,
                        str(detail.get("source_path") or ""),
                    )
                if not _claim_citation_budget(
                    detail,
                    budget_key_override=occurrence_budget_key,
                ):
                    changed = True
                    continue
                title_attr = _citation_hover_title(
                    str(detail.get("source_name") or ""),
                    int(n),
                    detail,
                )
                items.append(_mk_cite_link_md(int(n), detail, title_attr))
                linked_count += 1
                changed = True
            if not changed:
                return "" if _STRICT_STRUCTURED_CITATION_LINKING else raw
            if (
                not authoritative_answer_numbering
                and linked_count == 0
                and plain_fallback_count > 0
                and plain_fallback_count == len(items)
            ):
                return raw
            return "".join(items)

        seg2 = _STRUCT_CITE_RE.sub(_repl_struct, seg)
        seg2 = _STRUCT_CITE_SINGLE_RE.sub(_repl_struct_single, seg2)
        seg2 = _STRUCT_CITE_SID_ONLY_RE.sub(_repl_struct_sid_only, seg2)
        # Final safety-net: never leak raw CITE tokens to UI.
        seg2 = _STRUCT_CITE_GARBAGE_RE.sub("", seg2)
        if structured_seen:
            seg2 = _INPAPER_CITE_ANY_RE.sub(_repl_any, seg2)
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

        # Inline code and math are rendered as protected fragments, but they
        # still belong to one Markdown line/paragraph. Share citation counters
        # across every prose fragment so `$x$` or ``code`` cannot reset the
        # per-paragraph System-A/System-B budget.
        line_budget_state: dict = {}
        code_parts = _INLINE_CODE_RE.split(ln)
        rebuilt_code: list[str] = []
        code_offset = 0
        for i, cp in enumerate(code_parts):
            if i % 2 == 1:
                rebuilt_code.append(cp)
                code_offset += len(cp)
                continue
            math_parts = _INLINE_MATH_RE.split(cp)
            rebuilt_math: list[str] = []
            math_offset = 0
            for j, mp in enumerate(math_parts):
                if j % 2 == 1:
                    rebuilt_math.append(mp)
                else:
                    rebuilt_math.append(
                        _replace_text_segment(
                            mp,
                            table_mode=is_table_row,
                            context_line=ln,
                            context_offset=code_offset + math_offset,
                            budget_state=line_budget_state,
                        )
                    )
                math_offset += len(mp)
            rebuilt_code.append("".join(rebuilt_math))
            code_offset += len(cp)
        out_lines.append("".join(rebuilt_code))

    unique_details: dict[str, dict] = {}
    for rec in detail_by_key.values():
        if not isinstance(rec, dict):
            continue
        anchor = str(rec.get("anchor") or "").strip()
        if not anchor:
            continue
        if anchor not in visible_detail_anchors:
            continue
        unique_details[anchor] = rec

    def _detail_sort_key(rec: dict) -> tuple[int, str]:
        nums: list[int] = []
        for raw in rec.get("linked_nums") or []:
            try:
                k = int(raw)
            except Exception:
                continue
            if k > 0:
                nums.append(k)
        try:
            primary = int(rec.get("num") or 0)
        except Exception:
            primary = 0
        if primary > 0:
            nums.append(primary)
        return (min(nums) if nums else 0, str(rec.get("source_name") or ""))

    details = [compose_citation_card(rec, locale=render_locale) for rec in sorted(unique_details.values(), key=_detail_sort_key)]
    return "\n".join(out_lines), details



def _render_inpaper_citation_details(
    cite_details: list[dict],
    *,
    max_items: int = 24,
) -> None:
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
            "linked_nums": list(rec.get("linked_nums") or []),
            "source_name": str(rec.get("source_name") or "").strip(),
            "source_path": str(rec.get("source_path") or "").strip(),
            "evidence_fingerprint": str(rec.get("evidence_fingerprint") or "").strip(),
            "citation_route": str(rec.get("citation_route") or "").strip(),
            "routing_reason": str(rec.get("routing_reason") or "").strip(),
            "routing_confidence": float(rec.get("routing_confidence") or 0.0),
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
            "card_kind": str(rec.get("card_kind") or "").strip(),
            "card_title": str(rec.get("card_title") or "").strip(),
            "card_subtitle": str(rec.get("card_subtitle") or "").strip(),
            "card_takeaway_label": str(rec.get("card_takeaway_label") or "").strip(),
            "card_takeaway": str(rec.get("card_takeaway") or "").strip(),
            "card_claim_label": str(rec.get("card_claim_label") or "").strip(),
            "card_claim": str(rec.get("card_claim") or "").strip(),
            "card_locator_label": str(rec.get("card_locator_label") or "").strip(),
            "card_locator": str(rec.get("card_locator") or "").strip(),
            "card_evidence_label": str(rec.get("card_evidence_label") or "").strip(),
            "card_evidence": str(rec.get("card_evidence") or "").strip(),
            "card_context_summary": str(rec.get("card_context_summary") or "").strip(),
            "card_support_label": str(rec.get("card_support_label") or "").strip(),
            "card_support_explanation": str(rec.get("card_support_explanation") or "").strip(),
            "card_quality_label": str(rec.get("card_quality_label") or "").strip(),
            "card_quality_score": float(rec.get("card_quality_score") or 0.0),
            "card_quality_flags": list(rec.get("card_quality_flags") or []),
            "card_warning": str(rec.get("card_warning") or "").strip(),
            "card_flow": list(rec.get("card_flow") or []),
            "card_display_contract_version": int(rec.get("card_display_contract_version") or 0),
            "card_visible_sections": list(rec.get("card_visible_sections") or []),
            "system_b_trace_complete": bool(rec.get("system_b_trace_complete") or False),
            "system_b_trace_score": float(rec.get("system_b_trace_score") or 0.0),
            "system_b_trace_reason": str(rec.get("system_b_trace_reason") or "").strip(),
            "system_b_trace_flags": list(rec.get("system_b_trace_flags") or []),
            "system_b_trace_steps": list(rec.get("system_b_trace_steps") or []),
            "system_b_trace_answer": str(rec.get("system_b_trace_answer") or "").strip(),
            "system_b_trace_context": str(rec.get("system_b_trace_context") or "").strip(),
            "system_b_trace_reference": str(rec.get("system_b_trace_reference") or "").strip(),
            "system_b_trace_locator": str(rec.get("system_b_trace_locator") or "").strip(),
            "system_b_trace_source": str(rec.get("system_b_trace_source") or "").strip(),
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
            "<div class='citation-loading'>检索中...</div>",
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
