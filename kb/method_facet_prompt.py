from __future__ import annotations

import re

from kb.n2n_target_condition import _verified_block


def _compact(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def method_facet_prompt_quote(slot: dict, *, max_len: int = 720) -> str:
    """Expose independently verified method clauses within the old slot budget."""
    primary = _compact(slot.get("evidence_quote"))
    fallback = primary if len(primary) <= max_len else primary[:max_len-3].rstrip() + "..."
    path = str(slot.get("source_path") or "")
    if not slot.get("direct_method_facets") or not _verified_block(path, slot):
        return fallback
    pieces = [primary]
    for row in slot.get("direct_source_evidence") or []:
        if not isinstance(row, dict) or not all(row.get(k) for k in (
            "source_path", "block_id", "anchor_id", "heading_path", "page_start", "page_end",
        )):
            continue
        block = _verified_block(path, row)
        quote = _compact(row.get("text"))
        if not block or not quote or any(quote.casefold() in p.casefold() for p in pieces):
            continue
        addition = f"Additional same-source passage (p. {block['page_start']}): {quote}"
        if len(" ".join([*pieces, addition])) <= max_len:
            pieces.append(addition)
    result = " ".join(pieces)
    return result if len(result) <= max_len else fallback


def method_facet_prompt_instructions(slots: list[dict]) -> list[str]:
    if not any(slot.get('direct_method_facets') and 'Additional same-source passage (p.' in method_facet_prompt_quote(slot)
               for slot in slots):
        return []
    return [
        '- Each included Additional same-source passage supplies a requested method premise or operation. '
        'Explicitly address each included passage before comparing the methods; preserve its named variables, '
        'complete conditions and mathematical abbreviations, and cite the relation immediately. '
        'Distinguish independence between random variables or samples from independence of components within one variable; '
        'do not restrict a shared distribution to only one of its samples. '
        'Do not call a relation absent when an additional passage states it.'
    ]
