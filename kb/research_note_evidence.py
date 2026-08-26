from __future__ import annotations

import hashlib
import time
from copy import deepcopy
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
_REPAIR_HISTORY_LIMIT = 12
_REVIEW_HISTORY_LIMIT = 12


def _clean_text(value: Any, *, limit: int = _TEXT_LIMIT) -> str:
    return str(value or "").replace("\x00", "").strip()[:limit]


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def source_content_fingerprint(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _source_identity(value: Any) -> str:
    return _clean_text(value, limit=1_200).replace("\\", "/").casefold()


def _locator_snapshot(link: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_id": _clean_text(link.get("block_id"), limit=160),
        "anchor_id": _clean_text(link.get("anchor_id"), limit=160),
        "anchor_kind": _clean_text(link.get("anchor_kind"), limit=40),
        "heading_path": _clean_text(link.get("heading_path"), limit=500),
        "page_start": _positive_int(link.get("page_start")),
        "page_end": _positive_int(link.get("page_end")),
        "line_start": _positive_int(link.get("line_start")),
        "line_end": _positive_int(link.get("line_end")),
        "evidence_quote": _clean_text(
            link.get("evidence_quote") or link.get("text"),
            limit=12_000,
        ),
    }


def repair_research_note_source_state(
    source_state: dict[str, Any],
    *,
    source_index: int,
    source_path: str,
    capture_id: str = "",
    old_block_id: str = "",
    old_anchor_id: str = "",
    old_evidence_quote: str = "",
    target: dict[str, Any],
    source_fingerprint: str,
    repaired_at: float | None = None,
) -> dict[str, Any]:
    """Return a copy with one explicitly identified source locator repaired.

    The saved link must still match the locator the user started from. This
    prevents a delayed Reader confirmation from overwriting a newer edit.
    """
    next_state = deepcopy(source_state) if isinstance(source_state, dict) else {}
    links = next_state.get("links")
    if not isinstance(links, list) or not 0 <= int(source_index) < len(links):
        raise ValueError("research note source link not found")
    link = links[int(source_index)]
    if not isinstance(link, dict):
        raise ValueError("research note source link not found")
    if _source_identity(link.get("source_path")) != _source_identity(source_path):
        raise ValueError("research note source changed")

    expected_identity = {
        "capture_id": _clean_text(capture_id, limit=160),
        "block_id": _clean_text(old_block_id, limit=160),
        "anchor_id": _clean_text(old_anchor_id, limit=160),
        "evidence_quote": _clean_text(old_evidence_quote, limit=12_000),
    }
    for key, expected in expected_identity.items():
        if expected and _clean_text(link.get(key), limit=12_000) != expected:
            raise ValueError("research note source changed")

    target_snapshot = _locator_snapshot(target)
    if not target_snapshot["block_id"] and not target_snapshot["anchor_id"]:
        raise ValueError("verified evidence target has no stable locator")
    if not target_snapshot["evidence_quote"]:
        raise ValueError("verified evidence target has no text")

    previous = _locator_snapshot(link)
    for key in (
        "block_id",
        "anchor_id",
        "anchor_kind",
        "heading_path",
        "page_start",
        "page_end",
        "line_start",
        "line_end",
        "evidence_quote",
    ):
        value = target_snapshot[key]
        if value is None or value == "":
            link.pop(key, None)
        else:
            link[key] = value
    link.pop("location_label", None)
    for key in (
        "start_offset",
        "end_offset",
        "occurrence",
        "readable_index",
        "document_occurrence",
        "start_readable_index",
        "end_readable_index",
    ):
        link.pop(key, None)

    confirmed_at = float(repaired_at if repaired_at is not None else time.time())
    fingerprint = _clean_text(source_fingerprint, limit=96)
    link["evidence_verification"] = {
        "status": "exact",
        "reason": "confirmed_candidate",
        "verified_at": confirmed_at,
        "source_fingerprint": fingerprint,
    }
    raw_history = link.get("locator_repair_history")
    history = [deepcopy(row) for row in raw_history if isinstance(row, dict)] if isinstance(raw_history, list) else []
    history.append(
        {
            "version": 1,
            "repaired_at": confirmed_at,
            "source_fingerprint": fingerprint,
            "from": previous,
            "to": target_snapshot,
        }
    )
    link["locator_repair_history"] = history[-_REPAIR_HISTORY_LIMIT:]
    return next_state


def review_research_note_source_state(
    source_state: dict[str, Any],
    *,
    source_index: int,
    source_path: str,
    decision: str,
    capture_id: str = "",
    old_block_id: str = "",
    old_anchor_id: str = "",
    old_evidence_quote: str = "",
    expected_source_fingerprint: str,
    target: dict[str, Any],
    source_fingerprint: str,
    reviewed_at: float | None = None,
) -> dict[str, Any]:
    """Record an explicit review of unchanged evidence after source drift.

    This never changes the saved quote or locator. A changed locator must use
    the separate candidate-repair path. Both confirmation and invalidation
    retain the previous verification and locator in a bounded audit trail.
    """
    normalized_decision = _clean_text(decision, limit=24).lower()
    if normalized_decision not in {"confirmed", "invalid"}:
        raise ValueError("unsupported evidence review decision")

    next_state = deepcopy(source_state) if isinstance(source_state, dict) else {}
    links = next_state.get("links")
    if not isinstance(links, list) or not 0 <= int(source_index) < len(links):
        raise ValueError("research note source link not found")
    link = links[int(source_index)]
    if not isinstance(link, dict):
        raise ValueError("research note source link not found")
    if _source_identity(link.get("source_path")) != _source_identity(source_path):
        raise ValueError("research note source changed")

    expected_identity = {
        "capture_id": _clean_text(capture_id, limit=160),
        "block_id": _clean_text(old_block_id, limit=160),
        "anchor_id": _clean_text(old_anchor_id, limit=160),
        "evidence_quote": _clean_text(old_evidence_quote, limit=12_000),
    }
    for key, expected in expected_identity.items():
        if expected and _clean_text(link.get(key), limit=12_000) != expected:
            raise ValueError("research note source changed")

    previous_verification = link.get("evidence_verification")
    previous_verification = deepcopy(previous_verification) if isinstance(previous_verification, dict) else {}
    saved_fingerprint = _clean_text(previous_verification.get("source_fingerprint"), limit=96)
    expected_fingerprint = _clean_text(expected_source_fingerprint, limit=96)
    if not expected_fingerprint or saved_fingerprint != expected_fingerprint:
        raise ValueError("research note evidence verification changed")

    target_snapshot = _locator_snapshot(target)
    if not target_snapshot["block_id"] and not target_snapshot["anchor_id"]:
        raise ValueError("verified evidence target has no stable locator")
    if not target_snapshot["evidence_quote"]:
        raise ValueError("verified evidence target has no text")
    if expected_identity["block_id"] and target_snapshot["block_id"] != expected_identity["block_id"]:
        raise ValueError("evidence location changed; repair required")
    if expected_identity["anchor_id"] and target_snapshot["anchor_id"] != expected_identity["anchor_id"]:
        raise ValueError("evidence location changed; repair required")

    fingerprint = _clean_text(source_fingerprint, limit=96)
    if not fingerprint.startswith("sha256:"):
        raise ValueError("verified source fingerprint required")
    confirmed_at = float(reviewed_at if reviewed_at is not None else time.time())
    raw_history = link.get("evidence_review_history")
    history = [deepcopy(row) for row in raw_history if isinstance(row, dict)] if isinstance(raw_history, list) else []
    history.append(
        {
            "version": 1,
            "reviewed_at": confirmed_at,
            "decision": normalized_decision,
            "source_fingerprint_before": saved_fingerprint,
            "source_fingerprint_after": fingerprint,
            "previous_verification": previous_verification,
            "saved_locator": _locator_snapshot(link),
            "reviewed_target": target_snapshot,
        }
    )
    link["evidence_review_history"] = history[-_REVIEW_HISTORY_LIMIT:]
    link["evidence_verification"] = {
        "status": "exact" if normalized_decision == "confirmed" else "invalid",
        "reason": "confirmed_source_change" if normalized_decision == "confirmed" else "marked_invalid",
        "verified_at": confirmed_at,
        "source_fingerprint": fingerprint,
    }
    return next_state


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
    source_fingerprint: str = "",
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
    current_fingerprint = source_content_fingerprint(md_path)
    saved_fingerprint = _clean_text(source_fingerprint, limit=96)
    source_changed = bool(saved_fingerprint and saved_fingerprint != current_fingerprint)

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
            "source_fingerprint": current_fingerprint,
            "source_changed": source_changed,
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
            "source_fingerprint": current_fingerprint,
            "source_changed": source_changed,
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
        "source_fingerprint": current_fingerprint,
        "source_changed": source_changed,
    }
