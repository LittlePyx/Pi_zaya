from __future__ import annotations

import re
from typing import Any, Callable


Tokenize = Callable[..., dict[str, Any]]


def analyze_chunk_representation(
    *,
    text: str,
    evidence_terms: list[str],
    tokenize: Tokenize,
    max_length: int,
    passage_prefix: str,
) -> dict[str, Any]:
    """Describe whether labelled evidence survives the encoder's exact window."""

    combined = f"{passage_prefix}{text}"
    full = tokenize(
        combined,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
        verbose=False,
    )
    limited = tokenize(
        combined,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        verbose=False,
    )
    full_offsets = [tuple(item) for item in full.get("offset_mapping") or []]
    limited_offsets = [tuple(item) for item in limited.get("offset_mapping") or []]
    lowered = combined.lower()
    terms: list[dict[str, Any]] = []
    for raw_term in evidence_terms:
        term = str(raw_term or "")
        start, end = _find_term_span(combined, lowered, term)
        full_tokens = _overlapping_tokens(full_offsets, start, end)
        limited_tokens = _overlapping_tokens(limited_offsets, start, end)
        fully_visible = bool(limited_tokens) and _covered_end(limited_offsets, limited_tokens) >= end
        terms.append(
            {
                "term": term,
                "char_start": start,
                "char_end": end,
                "first_token": min(full_tokens) if full_tokens else None,
                "last_token": max(full_tokens) if full_tokens else None,
                "resolved": start >= 0 and bool(full_tokens),
                "fully_inside_window": fully_visible,
            }
        )
    total_tokens = len(list(full.get("input_ids") or []))
    return {
        "char_count": len(text),
        "token_count_with_prefix": total_tokens,
        "would_truncate": total_tokens > max_length,
        "evidence_term_count": len(terms),
        "resolved_term_count": sum(1 for item in terms if item["resolved"]),
        "all_evidence_inside_window": bool(terms)
        and all(item["fully_inside_window"] for item in terms),
        "terms": terms,
    }


def _overlapping_tokens(
    offsets: list[tuple[int, int]],
    start: int,
    end: int,
) -> list[int]:
    if start < 0 or end <= start:
        return []
    return [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_start < end and token_end > start
    ]


def _covered_end(offsets: list[tuple[int, int]], token_ids: list[int]) -> int:
    return max((offsets[index][1] for index in token_ids), default=-1)


def _find_term_span(text: str, lowered: str, term: str) -> tuple[int, int]:
    if not term:
        return -1, -1
    start = lowered.find(term.lower())
    if start >= 0:
        return start, start + len(term)
    pieces = re.findall(r"\w+", term, flags=re.UNICODE)
    if not pieces:
        return -1, -1
    flexible = re.compile(r"[\W_]*".join(re.escape(piece) for piece in pieces), re.IGNORECASE)
    match = flexible.search(text)
    return match.span() if match else (-1, -1)
