from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from kb.source_blocks import normalize_inline_markdown


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\u3002\uff01\uff1f\uff1b;!?\.])\s+")
_LEAD_STRIP_RE = re.compile(r"^[\s,.;:\u3002\uff0c\uff1b\uff1a]+")
_TRAIL_STRIP_RE = re.compile(r"[\s,.;:\u3002\uff0c\uff1b\uff1a]+$")
_EVIDENCE_TRAIL_STRIP_RE = re.compile(r"[\s,;:\uff0c\uff1b\uff1a]+$")
_TERMINAL_PUNCT_RE = re.compile(r"[\u3002\uff01\uff1f.!?]$")
_LAST_TERMINAL_PUNCT_RE = re.compile(r"[\u3002\uff01\uff1f.!?]")
_FRAGMENT_LEAD_OK_RE = re.compile(
    r"^(?:a|an|the|this|these|those|most|many|some|several|existing|previous|prior|traditional|we|our|in|on|for|by|with|when|where|while|because|however|therefore|thus|as|if|to)\b",
    re.IGNORECASE,
)
_CONTENT_SENTENCE_START_RE = re.compile(
    r"\b(?:"
    r"single[-\s]?pixel imaging\s+(?:is|can|uses?|technology|systems?)|"
    r"deep learning\s+(?:models?|methods?|can|is|has|enables?)|"
    r"snapshot compressive imaging\s+(?:is|can|uses?|recovers?)|"
    r"compressive imaging\s+(?:is|can|uses?|recovers?)|"
    r"neural radiance\s+(?:field|fields|representation)|"
    r"a\s+DMD\s+can|"
    r"this (?:paper|work|study|method|system|microscope|approach)|"
    r"in this (?:paper|work|study)|"
    r"we\s+|however,?|recent(?:ly)?|the proposed|our\s+"
    r")\b",
    re.IGNORECASE,
)
_LOW_VALUE_EVIDENCE_SENTENCE_RE = re.compile(
    r"\b(?:implementation details are discussed later|details are discussed later|additional details are provided|"
    r"more details can be found|this section describes|the rest of this paper|future work)\b",
    re.IGNORECASE,
)
_TEX_INLINE_CITATION_RE = re.compile(
    r"(?:\$\s*)?\^\{\s*\[[\d,\-\s;]+\]\s*\}(?:\s*\$)?|"
    r"\\textsuperscript\{\s*\[[^\]\n]{1,80}\]\s*\}|"
    r"\\(?:cite|citep|citet|citealp|upcite)\s*\{[^}\n]{1,200}\}",
    re.IGNORECASE,
)
_INLINE_MATH_WRAPPER_RE = re.compile(r"\$([^$\n]{1,160})\$")
_STRUCTURED_CITE_TOKEN_RE = re.compile(
    r"\[\[?\s*CITE\s*:[^\]\n]{1,160}\]?\]?",
    re.IGNORECASE,
)
_BRACKET_REFERENCE_MARKER_RE = re.compile(r"\[\s*\d{1,4}(?:\s*[-,;]\s*\d{1,4})*\s*\]")
_WRAPPED_SOURCE_EXCERPT_RE = re.compile(
    r"^\s*(?:\u539f\u6587\u7247\u6bb5\u5199\u5230|\u539f\u6587\u5199\u5230|source\s+excerpt\s+says)\s*[:\uff1a]\s*[\u201c\"']?(?P<body>.+?)[\u201d\"']?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_CONTENT_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|can|could|may|might|will|would|uses?|used|shows?|"
    r"shown|presents?|presented|proposes?|proposed|demonstrates?|develops?|developed|introduces?|introduced|"
    r"improves?|improved|captures?|captured|reconstructs?|reconstructed|enables?|enabled|"
    r"adopts?|adopted|adopting|offers?|offering|collects?|collecting|employs?|employed|employing|"
    r"解决|提出|说明|表明|用于|能够|可以|实现|采用|提升|降低)\b",
    re.IGNORECASE,
)
_CONNECTOR_CONTINUATION_RE = re.compile(
    r"^(?:to|of|and|or|from|into|onto|within|without|using|allowing|which|that|where|while|for|by|with|at)\b",
    re.IGNORECASE,
)
_INCOMPLETE_RIGHT_EDGE_RE = re.compile(
    r"(?:\b(?:and|or|of|to|with|by|from|into|onto|at|for|using|allowing)|"
    r"\bat\s+[±+\-]?\??\d+(?:\.\d+)?\??|"
    r"[,;:，；：])$",
    re.IGNORECASE,
)


def _strip_leading_markdown_heading_lines(value: str) -> str:
    text = str(value or "").lstrip()
    if not text:
        return ""
    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        line = str(lines[idx] or "").strip()
        if not line:
            idx += 1
            continue
        if re.match(r"^#{1,6}\s+\S", line):
            idx += 1
            continue
        break
    return "\n".join(lines[idx:]).lstrip() if idx else text


def clean_display_text(value: Any, *, max_len: int = 520) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    raw = re.sub(r"<!--[\s\S]*?-->", " ", raw)
    raw = _TEX_INLINE_CITATION_RE.sub(" ", raw)
    raw = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", raw)
    raw = re.sub(r"(?m)^\s{0,3}>\s?", "", raw)
    raw = re.sub(r"(?m)^\s{0,3}[-*+]\s+", "", raw)
    raw = re.sub(r"(?m)^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*$", " ", raw)
    raw = re.sub(r"(?m)^\s*\|", "", raw)
    raw = re.sub(r"(?m)\|\s*$", "", raw)
    raw = re.sub(r"\s*\|\s*", " ", raw)
    raw = _INLINE_MATH_WRAPPER_RE.sub(r"\1", raw)
    text = normalize_inline_markdown(raw)
    text = _TEX_INLINE_CITATION_RE.sub(" ", text)
    text = _STRUCTURED_CITE_TOKEN_RE.sub(" ", text)
    text = re.sub(r"\\(?=\s|[,;])", " ", text)
    text = re.sub(r"(^|\s)#{1,6}\s+", " ", text)
    text = re.sub(r"\s*\|\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(?:\.{2,}|…)+\s*", "", text)
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 1)].rstrip() + "..."


def finish_evidence_text(value: Any, *, max_len: int = 520) -> str:
    text = clean_display_text(value, max_len=max_len)
    if not text:
        return ""
    text = _trim_dangling_bracket_tail(text)
    text = _trim_incomplete_sentence_tail(text)
    return clean_display_text(text, max_len=max_len)


def _trim_dangling_bracket_tail(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    last_open = text.rfind("[")
    last_close = text.rfind("]")
    if last_open > last_close and last_open >= 0:
        tail = text[last_open:]
        # OCR/Markdown chunks often end in explanatory bracket clauses such as
        # "[known as structured illumination (1)" after a semicolon split.
        if len(tail) <= 180 and len(loose_tokens(tail)) >= 3:
            text = text[:last_open].rstrip(" ,;:")
    return text.strip()


def _trim_incomplete_sentence_tail(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return text
    if text.endswith(("...", "\u2026")):
        ellipsis = "\u2026" if text.endswith("\u2026") else "..."
        stem = text[: -len(ellipsis)].rstrip()
        words = list(re.finditer(r"[A-Za-z]{2,}$", stem))
        if words:
            last_word = words[-1].group(0)
            if 2 <= len(last_word) <= 5 and last_word.lower() not in {"image", "model"}:
                stem = stem[: words[-1].start()].rstrip(" ,;:")
                return stem + ellipsis if stem else text
        return text
    if _TERMINAL_PUNCT_RE.search(text):
        return text

    terminals = list(_LAST_TERMINAL_PUNCT_RE.finditer(text))
    if terminals:
        last = terminals[-1]
        head = text[: last.end()].strip()
        tail = text[last.end() :].strip()
        tail_tokens = loose_tokens(tail)
        if head and tail and len(tail_tokens) <= 18:
            return head

    tokens = loose_tokens(text)
    if len(tokens) < 8:
        return text

    # If the source chunk itself ended mid-sentence, make that visible with an
    # ellipsis and avoid exposing a half word as if it were valid evidence.
    words = list(re.finditer(r"[A-Za-z]{2,}$", text))
    if words:
        last_word = words[-1].group(0)
        if 2 <= len(last_word) <= 8 and last_word.lower() not in {"method", "system", "image", "learning"}:
            text = text[: words[-1].start()].rstrip(" ,;:")
    return text.rstrip(" ,;:") + "..."


def loose_tokens(value: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", str(value or ""))]


def source_title_candidate(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    name = Path(text).name or text
    name = re.sub(r"\.(?:pdf|md)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\.en$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^[A-Za-z]{2,12}-\d{4}-", "", name)
    name = re.sub(r"[_-]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def _strip_token_prefix(text: str, candidate: str) -> str:
    tokens = loose_tokens(candidate)
    if len(tokens) < 4:
        return text
    matches = list(re.finditer(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text))
    if len(matches) < len(tokens):
        return text
    matched = 0
    for idx in range(min(len(tokens), len(matches))):
        if matches[idx].group(0).lower() != tokens[idx]:
            break
        matched += 1
    if matched < min(8, len(tokens)):
        return text
    return _LEAD_STRIP_RE.sub("", text[matches[matched - 1].end() :])


def looks_author_metadata_prefix(value: str) -> bool:
    text = str(value or "").strip()
    if len(text) < 16:
        return False
    comma_count = text.count(",") + text.count("\uff0c")
    name_pairs = len(re.findall(r"\b[A-Z][a-zA-Z'`-]+\s+[A-Z][a-zA-Z'`-]+\b", text))
    tokens = loose_tokens(text)
    if comma_count >= 2 or name_pairs >= 2:
        return True
    return len(tokens) >= 8 and bool(re.search(r"[*\\]", text))


def looks_broken_leading_prefix(value: str) -> bool:
    text = str(value or "").strip()
    if not text or len(text) > 360:
        return False
    if looks_author_metadata_prefix(text):
        return True
    first_token = re.match(r"^[A-Za-z]{2,}\b", text)
    if first_token and not _FRAGMENT_LEAD_OK_RE.match(text):
        return True
    if re.match(r"^(?:and|or|of|that|which|from|into|onto|within|without|using|used|measured|allowing)\b", text, re.IGNORECASE):
        return True
    return False


def looks_author_list_context(value: str) -> bool:
    text = clean_display_text(value, max_len=1400)
    if len(text) < 24:
        return False
    marker_count = len(_BRACKET_REFERENCE_MARKER_RE.findall(text))
    comma_count = text.count(",") + text.count("\uff0c")
    name_pairs = len(re.findall(r"\b[A-Z][a-zA-Z'`-]+\s+[A-Z][a-zA-Z'`-]+\b", text))
    if marker_count >= 3 and (name_pairs >= 3 or comma_count >= 4):
        return True
    if name_pairs >= 4 and comma_count >= 3 and not _CONTENT_VERB_RE.search(text):
        return True
    return False


def looks_bibliography_entry_context(value: str) -> bool:
    text = clean_display_text(value, max_len=1400)
    if len(text) < 30:
        return False
    text = re.sub(r"^\s*(?:\[\s*\d{1,4}\s*\]|\d{1,4}\s*[.)])\s*", "", text)
    has_year = bool(re.search(r"\b(?:18|19|20)\d{2}\b", text))
    if not has_year:
        return False
    starts_like_authors = bool(
        re.match(
            r"^(?:[A-Z][A-Za-z'`-]+,\s*(?:[A-Z]\.?\s*){1,4}|[A-Z][a-zA-Z'`-]+\s+[A-Z](?:\.|\b))",
            text,
        )
    )
    venue_like = bool(
        re.search(
            r"\b(?:IEEE|ACM|Springer|Elsevier|Nature|Science|Nat\.?|Opt\.?|Phys\.?|"
            r"Journal|Proceedings|Trans\.?|Conf\.?|CVPR|ICCV|ICML|NeurIPS|arXiv)\b",
            text,
            re.IGNORECASE,
        )
    )
    volume_pages = bool(re.search(r"\b\d{1,4}\s*,\s*\d{1,6}(?:[-–]\d{1,6})?\.?$", text))
    return starts_like_authors and venue_like and (volume_pages or text.count(",") >= 3)


def looks_low_value_citation_context(value: str) -> bool:
    text = clean_display_text(value, max_len=1400)
    if not text:
        return True
    if looks_author_list_context(text) or looks_bibliography_entry_context(text):
        return True
    tokens = loose_tokens(text)
    if len(tokens) < 5:
        return True
    first_chunk = text[:320]
    if looks_author_metadata_prefix(first_chunk) and not _CONTENT_VERB_RE.search(first_chunk):
        return True
    marker_count = len(_BRACKET_REFERENCE_MARKER_RE.findall(text))
    if marker_count >= 4 and marker_count >= max(2, len(tokens) // 8) and not _CONTENT_VERB_RE.search(text):
        return True
    return False


def _looks_heading_like_prefix(value: str) -> bool:
    text = str(value or "").strip(" \t\r\n,.;:\u3002\uff0c\uff1b\uff1a")
    if not text:
        return False
    if len(text) > 110:
        return False
    if re.search(r"[\u3002\uff01\uff1f.!?;]", text):
        return False
    tokens = loose_tokens(text)
    if len(tokens) <= 1 or len(tokens) > 12:
        return False
    if re.search(r"\b(?:is|are|was|were|be|been|being|can|could|will|would|uses?|used|shows?|shown|proposes?|proposed|demonstrates?)\b", text, re.IGNORECASE):
        return False
    return True


def strip_evidence_metadata_prefix(
    value: str,
    *,
    source: str = "",
    title: str = "",
) -> str:
    text = clean_display_text(_strip_leading_markdown_heading_lines(value), max_len=1600)
    if not text:
        return ""
    wrapped = _WRAPPED_SOURCE_EXCERPT_RE.match(text)
    if wrapped:
        text = str(wrapped.group("body") or "").strip()

    for raw_candidate in (source, title):
        candidate = source_title_candidate(raw_candidate)
        if len(candidate) < 18:
            continue
        stripped = _strip_token_prefix(text, candidate)
        if stripped != text:
            text = stripped
            break

    for match in _CONTENT_SENTENCE_START_RE.finditer(text):
        idx = match.start()
        if idx <= 0:
            break
        if idx > 360:
            break
        prefix = text[:idx]
        if looks_author_metadata_prefix(prefix) or _looks_heading_like_prefix(prefix) or looks_broken_leading_prefix(prefix):
            text = _LEAD_STRIP_RE.sub("", text[idx:])
        break

    return re.sub(r"\s+", " ", text).strip()


def split_evidence_sentences(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]


def looks_fragmentary_sentence(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    if re.match(r"^[a-z]{2,}\b", text) and not _FRAGMENT_LEAD_OK_RE.match(text):
        return True
    if re.match(r"^(?:and|or|of|that|which|from|into|onto|within|without|using|used|measured|allowing)\b", text, re.IGNORECASE):
        return True
    if len(text) > 80 and re.search(r"\b(?:and|or|of|to|with|by|from|into|onto)$", text, re.IGNORECASE):
        return True
    if len(text) > 120 and not re.search(r"[\u3002\uff01\uff1f;!?\.]$", text):
        return True
    return False


def looks_caption_heading_sentence(value: str) -> bool:
    text = str(value or "").strip()
    if re.match(r"^(?:fig(?:ure)?|table)\s*\d+[.:]?\s*$", text, re.IGNORECASE):
        return True
    if re.match(r"^[a-z]\s*,\s*", text, re.IGNORECASE):
        return True
    tokens = loose_tokens(text)
    return len(tokens) <= 5 and bool(re.search(r"\b(?:configuration|configurations|overview|pipeline|results?|figure)\b", text, re.IGNORECASE))


def usable_evidence_sentence(value: str) -> bool:
    text = str(value or "").strip()
    if looks_low_value_citation_context(text):
        return False
    if _LOW_VALUE_EVIDENCE_SENTENCE_RE.search(text):
        return False
    if looks_fragmentary_sentence(text) or looks_caption_heading_sentence(text):
        return False
    return len(loose_tokens(text)) >= 5


def needs_right_continuation(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if _INCOMPLETE_RIGHT_EDGE_RE.search(text.rstrip(" .!?。！？")):
        return True
    if len(text) > 90 and not _TERMINAL_PUNCT_RE.search(text):
        return True
    return False


def usable_evidence_continuation(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    tokens = loose_tokens(text)
    if len(tokens) > 16:
        return False
    return bool(_CONNECTOR_CONTINUATION_RE.match(text))


def evidence_sentence_quality(value: str, *, claim: str = "", heading: str = "", title: str = "") -> float:
    text = str(value or "").strip()
    if not text:
        return -10.0
    tokens = loose_tokens(text)
    score = 0.0
    if looks_fragmentary_sentence(text):
        score -= 5.0
    if looks_caption_heading_sentence(text):
        score -= 2.0
    if _LOW_VALUE_EVIDENCE_SENTENCE_RE.search(text):
        score -= 4.0
    if looks_low_value_citation_context(text):
        score -= 6.0
    if 8 <= len(tokens) <= 90:
        score += 2.0
    elif len(tokens) < 5:
        score -= 2.0
    if looks_author_metadata_prefix(text[:180]):
        score -= 3.0
    if re.search(r"\b(?:is|are|can|uses?|proposes?|shows?|demonstrates?|improves?|captures?|reconstructs?|enables?|introduces?|presents?)\b", text, re.IGNORECASE):
        score += 1.0
    context_tokens = set(loose_tokens(f"{claim} {heading} {title}"))
    if context_tokens:
        overlap = len(set(tokens) & context_tokens)
        score += min(2.0, overlap * 0.3)
    if re.search(r"\b(?:single[-\s]?pixel|imaging|deep learning|compressive|neural|reconstruction|sampling|dmd|admm|network|resolution|sectioning)\b", text, re.IGNORECASE):
        score += 1.0
    return score


def join_evidence_window(
    sentences: list[str],
    *,
    center_idx: int,
    claim: str = "",
    heading: str = "",
    title: str = "",
    max_len: int = 460,
) -> str:
    if center_idx < 0 or center_idx >= len(sentences):
        return ""
    if not usable_evidence_sentence(sentences[center_idx]):
        return ""
    chosen = [center_idx]
    center_score = evidence_sentence_quality(sentences[center_idx], claim=claim, heading=heading, title=title)

    prev_idx = center_idx - 1
    if prev_idx >= 0 and usable_evidence_sentence(sentences[prev_idx]):
        prev_score = evidence_sentence_quality(sentences[prev_idx], claim=claim, heading=heading, title=title)
        if prev_score >= 1.0 or center_score < 2.5:
            chosen.insert(0, prev_idx)

    for next_idx in range(center_idx + 1, min(len(sentences), center_idx + 3)):
        if len(chosen) >= 3:
            break
        previous_text = sentences[chosen[-1]]
        is_continuation = needs_right_continuation(previous_text) and usable_evidence_continuation(sentences[next_idx])
        if not usable_evidence_sentence(sentences[next_idx]) and not is_continuation:
            continue
        next_score = evidence_sentence_quality(sentences[next_idx], claim=claim, heading=heading, title=title)
        if next_score < 0.5 and len(chosen) > 1 and not is_continuation:
            continue
        chosen.append(next_idx)

    out: list[str] = []
    for idx in sorted(set(chosen)):
        candidate = " ".join([*out, sentences[idx]]).strip()
        if out and len(candidate) > max_len:
            continue
        out.append(sentences[idx])
    return " ".join(out).strip()


def pick_readable_evidence_text(
    value: Any,
    *,
    source: str = "",
    title: str = "",
    claim: str = "",
    heading: str = "",
    max_len: int = 460,
) -> str:
    text = strip_evidence_metadata_prefix(str(value or ""), source=source, title=title)
    if not text:
        return ""
    if looks_low_value_citation_context(text):
        return ""
    sentences = split_evidence_sentences(text)
    while sentences and not usable_evidence_sentence(sentences[0]):
        sentences.pop(0)
    if not sentences:
        return ""
    usable = [idx for idx, sentence in enumerate(sentences[:10]) if usable_evidence_sentence(sentence)]
    if usable:
        first_idx = usable[0]
        scored = [
            (
                evidence_sentence_quality(sentences[idx], claim=claim, heading=heading, title=title),
                idx,
            )
            for idx in usable
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_idx = scored[0]
        first_score = evidence_sentence_quality(sentences[first_idx], claim=claim, heading=heading, title=title)
        center_idx = best_idx if best_idx > first_idx and best_score >= first_score + 1.0 else first_idx
        window = join_evidence_window(
            sentences,
            center_idx=center_idx,
            claim=claim,
            heading=heading,
            title=title,
            max_len=max_len,
        )
        if window:
            text = window
    return finish_evidence_text(_EVIDENCE_TRAIL_STRIP_RE.sub("", text), max_len=max_len)
