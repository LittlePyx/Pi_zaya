from __future__ import annotations

from pathlib import Path
from typing import Any

from kb.source_blocks import (
    SourceBlock,
    load_source_blocks,
    match_source_blocks,
    normalize_match_text,
)

_TEXT_LIMIT = 2_400
_CONTEXT_RADIUS = 1
_CANDIDATE_LIMIT = 3


def _clean_text(value: Any, *, limit: int = _TEXT_LIMIT) -> str:
    return str(value or "").replace("\x00", "").strip()[:limit]


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _serialize_block(block: SourceBlock) -> dict[str, Any]:
    return {
        "block_id": _clean_text(block.get("block_id"), limit=160),
        "anchor_id": _clean_text(block.get("anchor_id"), limit=160),
        "kind": _clean_text(block.get("kind"), limit=40),
        "heading_path": _clean_text(block.get("heading_path"), limit=500),
        "page_start": _positive_int(block.get("page_start")),
        "page_end": _positive_int(block.get("page_end")),
        "line_start": _positive_int(block.get("line_start")),
        "line_end": _positive_int(block.get("line_end")),
        "text": _clean_text(block.get("text")),
        "markdown": _clean_text(block.get("raw_text") or block.get("text")),
    }


def _reader_open(
    block: SourceBlock,
    *,
    source_path: str,
    source_name: str,
    evidence_quote: str,
) -> dict[str, Any]:
    block_text = _clean_text(block.get("text"))
    snippet = _clean_text(evidence_quote) or block_text
    heading_path = _clean_text(block.get("heading_path"), limit=500)
    block_id = _clean_text(block.get("block_id"), limit=160)
    anchor_id = _clean_text(block.get("anchor_id"), limit=160)
    anchor_kind = _clean_text(block.get("kind"), limit=40)
    return {
        "sourcePath": source_path,
        "sourceName": source_name,
        "headingPath": heading_path,
        "snippet": snippet,
        "highlightSnippet": snippet,
        "blockId": block_id,
        "anchorId": anchor_id,
        "anchorKind": anchor_kind,
        "strictLocate": True,
        "locateTarget": {
            "headingPath": heading_path,
            "snippet": snippet,
            "highlightSnippet": snippet,
            "evidenceQuote": snippet,
            "blockId": block_id,
            "anchorId": anchor_id,
            "anchorKind": anchor_kind,
        },
    }


def _context_neighbors(
    blocks: list[SourceBlock],
    target_index: int,
    *,
    direction: int,
    radius: int = _CONTEXT_RADIUS,
) -> list[dict[str, Any]]:
    collected: list[SourceBlock] = []
    cursor = target_index + direction
    while 0 <= cursor < len(blocks) and len(collected) < max(1, radius):
        block = blocks[cursor]
        if _clean_text(block.get("kind"), limit=40) == "heading":
            break
        if _clean_text(block.get("text")):
            collected.append(block)
        cursor += direction
    if direction < 0:
        collected.reverse()
    return [_serialize_block(block) for block in collected]


def _candidate_payload(
    row: dict[str, Any],
    *,
    source_path: str,
    source_name: str,
) -> dict[str, Any] | None:
    block = row.get("block")
    if not isinstance(block, dict):
        return None
    payload = _serialize_block(block)
    payload["score"] = round(float(row.get("score") or 0.0), 4)
    payload["reader_open"] = _reader_open(
        block,
        source_path=source_path,
        source_name=source_name,
        evidence_quote=_clean_text(block.get("text")),
    )
    return payload


def build_research_note_evidence_context(
    *,
    md_path: Path | str,
    source_path: str,
    source_name: str = "",
    heading_path: str = "",
    evidence_quote: str = "",
    block_id: str = "",
    anchor_id: str = "",
    anchor_kind: str = "",
    page_start: int | None = None,
    page_end: int | None = None,
) -> dict[str, Any]:
    """Verify a saved note citation against converted source blocks.

    Fuzzy matches are deliberately returned as review candidates and are never
    promoted to an exact Reader target.
    """
    source_path_clean = _clean_text(source_path, limit=1_200)
    source_name_clean = _clean_text(source_name, limit=500) or Path(source_path_clean).name
    requested_heading = _clean_text(heading_path, limit=500)
    quote = _clean_text(evidence_quote)
    requested_block_id = _clean_text(block_id, limit=160)
    requested_anchor_id = _clean_text(anchor_id, limit=160)
    requested_kind = _clean_text(anchor_kind, limit=40).lower()
    quote_norm = normalize_match_text(quote)

    blocks = load_source_blocks(md_path)
    if not blocks:
        return {
            "status": "unavailable",
            "reason": "no_source_blocks",
            "source_name": source_name_clean,
            "heading_path": requested_heading,
            "page_start": _positive_int(page_start),
            "page_end": _positive_int(page_end),
            "before": [],
            "target": None,
            "after": [],
            "candidates": [],
            "can_open_exact": False,
            "reader_open": None,
        }

    target_index: int | None = None
    exact_reason = ""
    if requested_block_id:
        identity_index = next(
            (index for index, block in enumerate(blocks) if _clean_text(block.get("block_id"), limit=160) == requested_block_id),
            None,
        )
        if identity_index is not None and (
            not quote_norm
            or quote_norm in normalize_match_text(str(blocks[identity_index].get("text") or ""))
        ):
            target_index = identity_index
            exact_reason = "exact_block"
    if target_index is None and requested_anchor_id:
        identity_index = next(
            (index for index, block in enumerate(blocks) if _clean_text(block.get("anchor_id"), limit=160) == requested_anchor_id),
            None,
        )
        if identity_index is not None and (
            not quote_norm
            or quote_norm in normalize_match_text(str(blocks[identity_index].get("text") or ""))
        ):
            target_index = identity_index
            exact_reason = "exact_anchor"

    quote_matches: list[int] = []
    if target_index is None and len(quote_norm) >= 16:
        quote_matches = [
            index
            for index, block in enumerate(blocks)
            if quote_norm in normalize_match_text(str(block.get("text") or ""))
        ]
        if len(quote_matches) == 1:
            target_index = quote_matches[0]
            exact_reason = "quote_relocated"

    if target_index is not None:
        target = blocks[target_index]
        serialized = _serialize_block(target)
        return {
            "status": "exact",
            "reason": exact_reason,
            "source_name": source_name_clean,
            "heading_path": serialized.get("heading_path") or requested_heading,
            "page_start": serialized.get("page_start") or _positive_int(page_start),
            "page_end": serialized.get("page_end") or _positive_int(page_end),
            "before": _context_neighbors(blocks, target_index, direction=-1),
            "target": serialized,
            "after": _context_neighbors(blocks, target_index, direction=1),
            "candidates": [],
            "can_open_exact": True,
            "reader_open": _reader_open(
                target,
                source_path=source_path_clean,
                source_name=source_name_clean,
                evidence_quote=quote,
            ),
        }

    ranked = match_source_blocks(
        blocks,
        snippet=quote,
        heading_path=requested_heading,
        prefer_kind=requested_kind,
        limit=_CANDIDATE_LIMIT,
    )
    candidates = [
        candidate
        for candidate in (
            _candidate_payload(row, source_path=source_path_clean, source_name=source_name_clean)
            for row in ranked
        )
        if candidate is not None
    ]
    if len(quote_matches) > 1:
        exact_rows = [
            {"score": 1.0, "block": blocks[index]}
            for index in quote_matches[:_CANDIDATE_LIMIT]
        ]
        candidates = [
            candidate
            for candidate in (
                _candidate_payload(row, source_path=source_path_clean, source_name=source_name_clean)
                for row in exact_rows
            )
            if candidate is not None
        ]

    return {
        "status": "candidate" if candidates else "unavailable",
        "reason": "ambiguous_quote" if len(quote_matches) > 1 else ("saved_target_missing" if candidates else "no_match"),
        "source_name": source_name_clean,
        "heading_path": requested_heading,
        "page_start": _positive_int(page_start),
        "page_end": _positive_int(page_end),
        "before": [],
        "target": None,
        "after": [],
        "candidates": candidates,
        "can_open_exact": False,
        "reader_open": None,
    }
