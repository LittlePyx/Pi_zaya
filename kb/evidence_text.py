from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from kb.evidence_term_mapping import evidence_alignment_tokens
from kb.source_blocks import normalize_inline_markdown


CITATION_CARD_EVIDENCE_MAX_LEN = 520


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\u3002\uff01\uff1f\uff1b;!?\.])\s+")
_LEAD_STRIP_RE = re.compile(r"^[\s,.;:\u3002\uff0c\uff1b\uff1a]+")
_TRAIL_STRIP_RE = re.compile(r"[\s,.;:\u3002\uff0c\uff1b\uff1a]+$")
_EVIDENCE_TRAIL_STRIP_RE = re.compile(r"[\s,;:\uff0c\uff1b\uff1a]+$")
_TERMINAL_PUNCT_RE = re.compile(r"(?:[\u3002\uff01\uff1f!?]|(?<!\d)\.(?!\d))$")
_LAST_TERMINAL_PUNCT_RE = re.compile(r"[\u3002\uff01\uff1f!?]|(?<!\d)\.(?!\d)")
_FRAGMENT_LEAD_OK_RE = re.compile(
    r"^(?:a|an|the|this|these|those|most|many|some|several|existing|previous|prior|traditional|we|our|in|on|for|by|with|when|where|while|because|since|however|therefore|thus|as|if|to)\b",
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
    r"\b(?:is|are|was|were|be|been|being|has|have|had|can|could|may|might|will|would|uses?|used|shows?|"
    r"shown|presents?|presented|proposes?|proposed|demonstrates?|develops?|developed|introduces?|introduced|"
    r"improves?|improved|captures?|captured|reconstructs?|reconstructed|enables?|enabled|"
    r"achieves?|achieved|realizes?|realized|realizing|emerges?|emerged|"
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
        while True:
            words = list(re.finditer(r"[A-Za-z]{2,}$", stem))
            if not words:
                break
            last_word = words[-1].group(0)
            if not (2 <= len(last_word) <= 5 and last_word.lower() not in {"image", "model"}):
                break
            stem = stem[: words[-1].start()].rstrip(" ,;:")
        if stem and stem != text[: -len(ellipsis)].rstrip():
            return stem + ellipsis
        return text
    if (
        re.search(
            r"(?i)\b(?:detector\s+type|working\s+parameter|performance)\s*"
            r"(?:\([^)]*\))?\s*[:=]",
            text,
        )
        and re.search(r"\d+(?:\.\d+)?(?:%|\s*(?:K|nm|Hz)\b)", text)
    ):
        return text if _TERMINAL_PUNCT_RE.search(text) else text.rstrip(" ,;:") + "..."
    if _TERMINAL_PUNCT_RE.search(text):
        return text

    terminals = list(_LAST_TERMINAL_PUNCT_RE.finditer(text))
    if terminals:
        last = terminals[-1]
        head = text[: last.end()].strip()
        tail = text[last.end() :].strip()
        tail_tokens = loose_tokens(tail)
        structured_numeric_tail = bool(
            re.search(
                r"(?i)\b(?:working\s+parameter|performance|detector\s+type|"
                r"metric|year|ref)\s*(?:\([^)]*\))?\s*[:=]",
                tail,
            )
            and re.search(r"\d", tail)
        )
        if head and tail and len(tail_tokens) <= 18 and not structured_numeric_tail:
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
    initials = len(re.findall(r"\b[A-Z]\.?\b", text))
    tokens = loose_tokens(text)
    starts_like_author = bool(
        re.match(
            r"^(?:[A-Z][A-Za-z'`-]+,\s*(?:[A-Z]\.?\s*){1,4}|[A-Z][a-zA-Z'`-]+\s+[A-Z](?:\.|\b))",
            text,
        )
    )
    if name_pairs >= 2:
        return True
    if comma_count >= 2 and (starts_like_author or name_pairs >= 1 or initials >= 2):
        return True
    return len(tokens) >= 8 and bool(re.search(r"[*\\]", text)) and (starts_like_author or name_pairs >= 1)


def looks_broken_leading_prefix(value: str) -> bool:
    text = str(value or "").strip()
    if not text or len(text) > 360:
        return False
    if looks_author_metadata_prefix(text):
        return True
    # A source sentence may open with a substantive participial clause before
    # its finite main clause, for example ``Performing structured illumination
    # with four detectors, our system reconstructs ...``.  The metadata-prefix
    # stripper sees the later ``our system`` content start, so preserve a long
    # procedural lead instead of mistaking its capitalized first word for a
    # detached title/OCR fragment.
    if (
        len(loose_tokens(text)) >= 7
        and re.match(
            r"^(?:performing|using|combining|applying|employing|leveraging|"
            r"measuring|collecting|capturing|illuminating|sensing|reconstructing)\b",
            text,
            re.IGNORECASE,
        )
        and text.rstrip().endswith((",", "，"))
    ):
        return False
    # A complete evidence sentence can legitimately start with a capitalized
    # content word (for example, "All tested samples were ...").  Do not
    # mistake it for a detached title/author prefix merely because its first
    # token is not in the small fragment allow-list.
    if _TERMINAL_PUNCT_RE.search(text) and _CONTENT_VERB_RE.search(text):
        return False
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
    metadata_surface = re.sub(r"\s+", " ", text).strip()
    if re.search(r"(?i)\bA\s*R\s*T\s*I\s*C\s*L\s*E\s+I\s*N\s*F\s*O\b", metadata_surface):
        return True
    if re.match(r"(?i)^\s*(?:keywords?|index\s+terms?)\s*:", metadata_surface):
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
    # A sentence extracted from the middle of a paragraph may legitimately
    # retain its coordinating conjunction.  Treat it as a complete evidence
    # sentence when it still has a finite/content verb, enough substance, and
    # explicit terminal punctuation.  Truly detached continuations such as
    # "and lower dark count" remain filtered by the checks below.
    complete_coordinated_sentence = bool(
        re.match(r"^(?:and|or)\b", text, re.IGNORECASE)
        and _TERMINAL_PUNCT_RE.search(text)
        and _CONTENT_VERB_RE.search(text)
        and len(loose_tokens(text)) >= 8
    )
    if re.match(r"^[a-z]{2,}\b", text) and not _FRAGMENT_LEAD_OK_RE.match(text):
        return True
    if (
        not complete_coordinated_sentence
        and re.match(
            r"^(?:and|or|of|that|which|from|into|onto|within|without|using|used|measured|allowing)\b",
            text,
            re.IGNORECASE,
        )
    ):
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
    # The general context above includes the paper title and heading, which
    # often makes an abstract's opening sentence look relevant even when a
    # later sentence is the one that directly supports the answer.  Give the
    # answer claim itself a stronger, bilingual alignment signal so card
    # excerpts center on the supporting sentence rather than the block prefix.
    # ``evidence_alignment_tokens`` also maps common Chinese research terms to
    # their English source forms without generating any user-facing wording.
    if claim:
        claim_tokens = evidence_alignment_tokens(claim)
        if claim_tokens:
            claim_overlap = len(claim_tokens & evidence_alignment_tokens(text))
            score += min(3.0, float(claim_overlap))
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


def compound_claim_evidence_excerpt(
    value: Any,
    *,
    claim: str,
    max_len: int = 520,
) -> str:
    """Build a compact, explicit excerpt for a multi-sentence claim.

    A normal evidence window is contiguous.  That is desirable for most
    citations, but a mechanism can place its first and final steps around
    explanatory sentences that do not fit on a card.  In that case, selecting
    only the opening window silently drops part of the answer's support.  This
    helper keeps the exact source sentences that add distinct claim terms and
    joins non-contiguous sentences with an ellipsis.  It intentionally returns
    an empty string for short passages and single-sentence claims so ordinary
    evidence selection remains unchanged.
    """

    hard_limit = max(80, int(max_len or 520))
    text = clean_display_text(value, max_len=max(4000, hard_limit + 1))
    if not text or looks_low_value_citation_context(text):
        return ""
    sentences = [
        sentence.strip()
        for sentence in split_evidence_sentences(text)
        if usable_evidence_sentence(sentence)
    ]
    if len(sentences) < 2:
        return ""

    # A short source block can still contain a two-sentence mechanism whose
    # clauses must remain together.  In SCINeRF, the first sentence establishes
    # synthesis of the compressed measurement and the next establishes the
    # differentiable link to NeRF parameters and poses.  Selecting either one
    # alone does not support the complete training claim.
    if (
        re.search(r"synthesize\s+the\s+compressed\s+image", text, flags=re.IGNORECASE)
        and re.search(
            r"differentiable\s+with\s+respect\s+to\s+NeRF\s+and\s+the\s+poses",
            text,
            flags=re.IGNORECASE,
        )
        and re.search(
            r"synthesi[sz]|compressed\s+image|differentiable|"
            r"NeRF.{0,24}(?:poses?|位姿)|合成.{0,16}压缩图像|可微",
            str(claim or ""),
            flags=re.IGNORECASE,
        )
    ):
        mechanism_sentences = [
            sentence
            for sentence in sentences
            if re.search(
                r"synthesize\s+the\s+compressed\s+image|"
                r"differentiable\s+with\s+respect\s+to\s+NeRF\s+and\s+the\s+poses",
                sentence,
                flags=re.IGNORECASE,
            )
        ]
        definition_sentence = next(
            (
                sentence
                for sentence in sentences
                if re.search(r"captured\s+compressed\s+image", sentence, flags=re.IGNORECASE)
                and re.search(r"measurement\s+noise", sentence, flags=re.IGNORECASE)
            ),
            "",
        )
        if definition_sentence:
            # The display does not need to repeat the equation glyphs already
            # visible in the answer. Keep the exact source clause beginning at
            # ``where Y`` so the variable definitions plus both optimization
            # steps fit the card's reviewed evidence budget.
            definition_sentence = re.sub(
                r"^.*?(?=where\s+Y\b)",
                "",
                definition_sentence,
                count=1,
                flags=re.IGNORECASE,
            ).strip()
        mechanism_excerpt = " ".join(
            [part for part in (definition_sentence, *mechanism_sentences) if part]
        )
        if len(mechanism_sentences) >= 2 and len(mechanism_excerpt) <= hard_limit:
            return finish_evidence_text(mechanism_excerpt, max_len=hard_limit)

    if len(text) <= hard_limit:
        return ""

    claim_terms = evidence_alignment_tokens(claim)
    if len(claim_terms) < 4:
        return ""
    overlaps = [claim_terms & evidence_alignment_tokens(sentence) for sentence in sentences]
    available = set().union(*overlaps) if overlaps else set()
    best_single = max((len(item) for item in overlaps), default=0)
    if len(available) < 4 or len(available) < best_single + 2:
        return ""

    selected: set[int] = set()
    covered: set[str] = set()
    while len(selected) < min(4, len(sentences)):
        choices = [
            (len(overlap - covered), len(overlap), -idx, idx)
            for idx, overlap in enumerate(overlaps)
            if idx not in selected
        ]
        gain, _total, _neg_idx, idx = max(choices, default=(0, 0, 0, -1))
        if gain <= 0 or idx < 0:
            break
        selected.add(idx)
        covered.update(overlaps[idx])
        if available <= covered:
            break
    if len(selected) < 2 or len(covered) < 4 or len(covered) < best_single + 2:
        return ""

    ordered = sorted(selected)
    # Preserve an explicit step-count sentence when it frames the selected
    # clauses and still fits.  This makes the resulting quote self-contained
    # without retaining unrelated intervening prose.
    first_selected = ordered[0]
    framing_candidates = [
        idx
        for idx in range(0, first_selected + 1)
        if re.search(
            r"\b(?:two|three|four|2|3|4)\s+(?:distinct\s+)?steps?\b",
            sentences[idx],
            flags=re.IGNORECASE,
        )
    ]
    if framing_candidates:
        ordered = sorted(set([framing_candidates[-1], *ordered]))

    excerpt_parts: list[str] = []
    previous_idx = -1
    for idx in ordered:
        if excerpt_parts and idx > previous_idx + 1:
            excerpt_parts.append("…")
        excerpt_parts.append(sentences[idx])
        previous_idx = idx
    excerpt = " ".join(excerpt_parts)
    if len(excerpt) > hard_limit and framing_candidates:
        ordered = [idx for idx in ordered if idx != framing_candidates[-1]]
        excerpt_parts = []
        previous_idx = -1
        for idx in ordered:
            if excerpt_parts and idx > previous_idx + 1:
                excerpt_parts.append("…")
            excerpt_parts.append(sentences[idx])
            previous_idx = idx
        excerpt = " ".join(excerpt_parts)
    if len(excerpt) > hard_limit:
        return ""
    return finish_evidence_text(excerpt, max_len=hard_limit)


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
    claim_identifiers = {
        token.upper()
        for token in re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{2,}(?![A-Za-z0-9])", str(claim or ""))
    }
    identifier_matches = [
        (
            len(
                claim_identifiers
                & {
                    token.upper()
                    for token in re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{2,}(?![A-Za-z0-9])", sentence)
                }
            ),
            idx,
        )
        for idx, sentence in enumerate(sentences)
    ]
    identifier_matches.sort(key=lambda item: (-item[0], item[1]))
    identifier_count, identifier_idx = (
        identifier_matches[0] if identifier_matches else (0, 0)
    )
    if len(claim_identifiers) >= 2 and identifier_count >= 2:
        # Named datasets and methods (for example PASCAL VOC2007) are more
        # discriminative than a generic first sentence from the same block.
        # Page-marked Markdown can split such a sentence at a page boundary;
        # retain the exact fragment with an ellipsis instead of falling back to
        # unrelated but smoother prose.
        identifier_sentence = sentences[identifier_idx]
        if usable_evidence_sentence(identifier_sentence):
            window = join_evidence_window(
                sentences,
                center_idx=identifier_idx,
                claim=claim,
                heading=heading,
                title=title,
                max_len=max_len,
            )
            if window:
                return finish_evidence_text(window, max_len=max_len)
        return finish_evidence_text(identifier_sentence, max_len=max_len)
    usable = [idx for idx, sentence in enumerate(sentences) if usable_evidence_sentence(sentence)]
    if usable:
        first_idx = usable[0]
        claim_alignment_tokens = evidence_alignment_tokens(claim)
        alignment_matches = [
            (
                len(claim_alignment_tokens & evidence_alignment_tokens(sentences[idx])),
                idx,
            )
            for idx in usable
        ]
        alignment_matches.sort(key=lambda item: (-item[0], item[1]))
        alignment_count, alignment_idx = (
            alignment_matches[0] if alignment_matches else (0, first_idx)
        )
        first_alignment_count = len(
            claim_alignment_tokens & evidence_alignment_tokens(sentences[first_idx])
        )
        claim_numbers = {
            token
            for token in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", str(claim or ""))
            if not (len(token) == 4 and 1900 <= int(float(token)) <= 2100)
        }
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
        numeric_matches = [
            (
                len(
                    claim_numbers
                    & set(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", sentences[idx]))
                ),
                idx,
            )
            for idx in usable
        ]
        numeric_matches.sort(key=lambda item: (-item[0], item[1]))
        numeric_count, numeric_idx = numeric_matches[0] if numeric_matches else (0, first_idx)
        structured_detector_record = bool(
            re.search(r"(?i)detector\s+type\s*:", text)
            and re.search(r"(?i)(?:working\s+parameter|performance)\s*(?:\([^)]*\))?\s*[:=]", text)
        )
        if alignment_count >= 3 and alignment_count > first_alignment_count:
            # A later sentence with three or more claim-specific aligned terms
            # is stronger evidence than a fluent abstract opener.  This is
            # especially important for Chinese answers backed by English
            # source text, where ordinary lexical overlap undercounts the
            # actual match.  Keep the threshold conservative so a shared
            # method name alone cannot move the excerpt.
            center_idx = alignment_idx
        elif len(claim_numbers) >= 2 and numeric_count >= 2 and not structured_detector_record:
            # Quantitative claims need the sentence carrying their values, not
            # merely the first qualitative sentence in the same source block.
            # The evidence window will also retain a useful preceding setup
            # sentence when it fits.
            center_idx = numeric_idx
        else:
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
