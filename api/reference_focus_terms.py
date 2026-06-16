from __future__ import annotations

from functools import lru_cache
import re

from api.reference_source_identity import _normalize_title_identity
from kb.reference_query_family import (
    extract_multi_paper_topic as _shared_extract_multi_paper_topic,
    prompt_requests_reference_compare as _shared_prompt_requests_reference_compare,
    prompt_targets_sci_topic as _shared_prompt_targets_sci_topic,
)


def _prompt_requests_compare(prompt: str) -> bool:
    if _shared_prompt_requests_reference_compare(prompt):
        return True
    text = str(prompt or "").strip()
    if not text:
        return False
    compare_tokens = (
        "\u6bd4\u8f83",
        "\u5bf9\u6bd4",
        "\u6743\u8861",
        "\u77db\u76fe",
        "\u4e0d\u540c",
        "\u533a\u522b",
        "\u5dee\u5f02",
        "\u76f8\u6bd4",
        "\u76f8\u8f83",
        "\u53d6\u820d",
    )
    return any(token in text for token in compare_tokens)


_PROMPT_FOCUS_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "using", "about", "where", "which", "what",
    "that", "this", "these", "those", "paper", "papers", "library", "source", "sources",
    "section", "please", "point", "directly", "most", "does", "do", "did", "discuss", "discusses",
    "mentioned", "mention", "other", "besides", "find", "show", "explain",
}

_PROMPT_FOCUS_GENERIC_MODIFIERS = {
    "dynamic", "compressive", "physics", "physical", "single", "high", "low",
    "based", "guided", "driven", "general", "specific", "direct", "directly",
}

_PROMPT_FOCUS_PHRASE_PATTERNS = (
    re.compile(
        r"\bwhere\s+(?:in\s+the\s+[^?.!,]{1,80}\s+)?is\s+(.+?)\s+(?:discussed|mentioned|defined|introduced)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:which|what)\s+(?:other\s+)?papers?[^?.!]{0,120}?\b(?:discuss(?:es|ed)?|mention(?:s|ed)?|cover(?:s|ed)?|address(?:es|ed)?|describe(?:s|d)?|use(?:s|d)?|introduce(?:s|d)?|define(?:s|d)?|compare(?:s|d)?)\s+(.+?)(?:[?.!]|$)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bbesides\s+this\s+paper[^?.!]{0,120}?\b(?:discuss(?:es|ed)?|mention(?:s|ed)?|cover(?:s|ed)?|address(?:es|ed)?|describe(?:s|d)?|use(?:s|d)?|introduce(?:s|d)?|define(?:s|d)?|compare(?:s|d)?)\s+(.+?)(?:[?.!]|$)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:which|what)\s+papers?[^?.!]{0,120}?\b(?:directly\s+|most\s+directly\s+)?(?:compare(?:s|d)?|define(?:s|d)?)\s+(.+?)(?:[?.!]|$)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bbesides\s+this\s+paper[^?.!]{0,120}?\b(?:directly\s+|most\s+directly\s+)?(?:compare(?:s|d)?|define(?:s|d)?)\s+(.+?)(?:[?.!]|$)",
        flags=re.IGNORECASE,
    ),
)


_ZH_PROMPT_FOCUS_ALIASES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("深度学习", "神经网络", "神经网路"), ("deep learning", "neural network")),
    (("单像素成像", "单像素", "鬼成像"), ("single-pixel imaging", "single pixel imaging", "computational ghost imaging")),
    (("硬件", "实验装置", "实验设置", "装置", "部件"), ("experimental setup", "setup", "hardware", "camera", "lens", "DMD")),
    (("结构化探测", "结构化检测"), ("structured detection", "structured detector")),
    (("激光扫描显微", "扫描显微"), ("laser scanning microscopy", "scanning microscopy")),
    (("图像扫描显微",), ("image scanning microscopy", "ISM")),
    (("共聚焦",), ("confocal", "confocal microscopy")),
    (("权衡", "矛盾", "折中"), ("trade-off", "tradeoff")),
    (("挑战", "局限"), ("challenge", "limitation")),
)


def _refs_prompt_focus_alias_terms(prompt: str) -> tuple[str, ...]:
    text = str(prompt or "").strip()
    if not text:
        return ()
    out: list[str] = []
    seen: set[str] = set()

    def _push(raw: str) -> None:
        norm = _normalize_title_identity(raw)
        if len(norm) < 3 or norm in seen:
            return
        seen.add(norm)
        out.append(norm)

    for triggers, aliases in _ZH_PROMPT_FOCUS_ALIASES:
        if any(trigger and trigger in text for trigger in triggers):
            for alias in aliases:
                _push(alias)
    if re.search(r"(?<![A-Za-z0-9])ISM(?![A-Za-z0-9])", text):
        _push("image scanning microscopy")
        _push("ISM")
    return tuple(out)


def _clean_refs_focus_phrase(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    text = re.sub(
        r"\b(?:please\s+point\s+me(?:\s+to)?|point\s+me(?:\s+to)?|show\s+me|source\s+section(?:s)?|those\s+sources|source\s+too)\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.IGNORECASE)
    text = text.strip(" \t\r\n\"'“”‘’.,;:!?()[]{}")
    return text


def _looks_informative_focus_phrase(raw: str) -> bool:
    text = str(raw or "").strip()
    if not text:
        return False
    tokens = [tok for tok in _normalize_title_identity(text).split() if tok and tok not in _PROMPT_FOCUS_STOPWORDS]
    if not tokens:
        return False
    if len(tokens) >= 2:
        return True
    token = tokens[0]
    return bool(len(token) >= 4 and (any(ch.isdigit() for ch in token) or "-" in token or token.isupper()))


def _extract_prompt_focus_phrases(prompt: str) -> tuple[str, ...]:
    text = str(prompt or "").strip()
    if not text:
        return ()
    out: list[str] = []
    seen: set[str] = set()

    def _push(raw: str) -> None:
        cleaned = _clean_refs_focus_phrase(raw)
        if not _looks_informative_focus_phrase(cleaned):
            return
        norm = _normalize_title_identity(cleaned)
        if len(norm) < 3 or norm in seen:
            return
        seen.add(norm)
        out.append(norm)

    for pattern in _PROMPT_FOCUS_PHRASE_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        raw = str(m.group(1) or "")
        _push(raw)
        if _prompt_requests_compare(text):
            for part in re.split(r"\b(?:and|vs\.?|versus)\b", raw, flags=re.IGNORECASE):
                _push(part)
    for m in re.finditer(
        r"(?:比较|对比)(?:了)?\s*([^？?。.!]{2,140}?)(?:\s*(?:的)?(?:权衡|取舍|差异|区别|不同)|[？?。.!]|$)",
        text,
    ):
        raw = re.sub(r"^(?:哪些|哪几篇|哪几篇文献|哪些文献|文献|论文)\s*", "", str(m.group(1) or "").strip())
        _push(raw)
        for part in re.split(r"\s*(?:和|与|及|以及|、|/|\bvs\.?\b|\bversus\b|\band\b)\s*", raw, flags=re.IGNORECASE):
            _push(part)
    return tuple(out[:4])


def _prune_redundant_focus_terms(terms: list[str]) -> tuple[str, ...]:
    items = [str(term or "").strip() for term in terms if str(term or "").strip()]
    out: list[str] = []
    for term in items:
        if any(
            term != other
            and len(other) > len(term)
            and term in other
            and (not re.search(r"(?:\b(?:and|vs\.?|versus)\b|和|与|及|以及|、|/)", other, flags=re.IGNORECASE))
            for other in items
        ):
            continue
        out.append(term)
    return tuple(out[:8])


def _surface_has_focus_token_sequence(surface_tokens: list[str], term_tokens: list[str]) -> bool:
    if (not surface_tokens) or (not term_tokens) or (len(term_tokens) > len(surface_tokens)):
        return False
    width = len(term_tokens)
    for idx in range(len(surface_tokens) - width + 1):
        if surface_tokens[idx : idx + width] == term_tokens:
            return True
    return False


def _focus_term_adjacent_bigram_hits(surface: str, term_tokens: list[str]) -> int:
    if (not surface) or len(term_tokens) < 2:
        return 0
    hits = 0
    for idx in range(len(term_tokens) - 1):
        phrase = f"{term_tokens[idx]} {term_tokens[idx + 1]}".strip()
        if phrase and re.search(rf"\b{re.escape(phrase)}\b", surface, flags=re.I):
            hits += 1
    return hits


def _focus_term_single_distinctive_token_fallback(term_tokens: list[str], surface_tokens: set[str]) -> bool:
    if len(term_tokens) != 2 or (not surface_tokens):
        return False
    overlap = [tok for tok in term_tokens if tok in surface_tokens]
    if len(overlap) != 1:
        return False
    matched = overlap[0]
    unmatched = term_tokens[0] if matched == term_tokens[1] else term_tokens[1]
    if len(matched) < 10:
        return False
    if matched in _PROMPT_FOCUS_GENERIC_MODIFIERS:
        return False
    return unmatched in _PROMPT_FOCUS_GENERIC_MODIFIERS


def _focus_term_matches_surface(term: str, surface_text: str) -> bool:
    norm_term = _normalize_title_identity(term)
    surface = _normalize_title_identity(surface_text)
    if not norm_term or not surface:
        return False
    if re.search(rf"\b{re.escape(norm_term)}\b", surface, flags=re.I):
        return True
    term_tokens = [
        tok for tok in norm_term.split()
        if tok and tok not in _PROMPT_FOCUS_STOPWORDS and len(tok) >= 4
    ]
    if not term_tokens:
        return False
    surface_tokens = [tok for tok in surface.split() if tok]
    if not surface_tokens:
        return False
    surface_token_set = set(surface_tokens)
    if len(term_tokens) == 1:
        return bool(term_tokens[0] in surface_token_set)
    if len(term_tokens) == 2:
        if _surface_has_focus_token_sequence(surface_tokens, term_tokens):
            return True
        return _focus_term_single_distinctive_token_fallback(term_tokens, surface_token_set)
    if not all(tok in surface_token_set for tok in term_tokens):
        return False
    if _surface_has_focus_token_sequence(surface_tokens, term_tokens):
        return True
    return _focus_term_adjacent_bigram_hits(surface, term_tokens) > 0


def _refs_exact_focus_match_count(prompt: str, surface_text: str) -> int:
    surface = _normalize_title_identity(surface_text)
    if not surface:
        return 0
    count = 0
    for term in _refs_prompt_focus_terms(prompt):
        norm_term = _normalize_title_identity(term)
        if norm_term and re.search(rf"\b{re.escape(norm_term)}\b", surface, flags=re.I):
            count += 1
    return count


@lru_cache(maxsize=512)
def _refs_prompt_focus_terms(prompt: str) -> tuple[str, ...]:
    text = str(prompt or "").strip()
    if not text:
        return ()
    out: list[str] = []
    seen: set[str] = set()

    def _push(raw: str) -> None:
        cleaned = _clean_refs_focus_phrase(raw)
        if not cleaned:
            return
        norm = _normalize_title_identity(cleaned)
        if len(norm) < 3 or norm in seen:
            return
        seen.add(norm)
        out.append(norm)

    prompt_targets_sci = bool(_shared_prompt_targets_sci_topic(text))
    if prompt_targets_sci:
        _push("Snapshot Compressive Imaging")
        _push("SCI")
    for alias_term in _refs_prompt_focus_alias_terms(text):
        _push(alias_term)
    topic = _shared_extract_multi_paper_topic(text)
    if topic and (not prompt_targets_sci):
        _push(topic)

    for quoted in re.findall(r"[\"']([^\"']{2,80})[\"']", text):
        _push(quoted)
    for token in re.findall(r"(?<![A-Za-z0-9_-])[A-Za-z][A-Za-z0-9_-]{1,40}(?![A-Za-z0-9_-])", text):
        raw = str(token or "").strip()
        low = raw.lower()
        if low in _PROMPT_FOCUS_STOPWORDS:
            continue
        has_case_signal = any(ch.isupper() for ch in raw[1:]) or raw.isupper() or any(ch.isdigit() for ch in raw) or ("-" in raw)
        if not has_case_signal:
            continue
        _push(raw)
    for phrase in _extract_prompt_focus_phrases(text):
        _push(phrase)
    return _prune_redundant_focus_terms(out)
