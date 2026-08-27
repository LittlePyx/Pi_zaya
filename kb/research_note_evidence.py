from __future__ import annotations

import hashlib
import math
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
_CONTEXT_HISTORY_LIMIT = 12
_BASELINE_CHECK_HISTORY_LIMIT = 12
_SNAPSHOT_NEIGHBOR_TEXT_LIMIT = 600
_SNAPSHOT_TARGET_TEXT_LIMIT = 1_200


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


def _context_snapshot_block(value: Any, *, target: bool = False) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    text = _clean_text(
        value.get("text") or value.get("evidence_quote"),
        limit=_SNAPSHOT_TARGET_TEXT_LIMIT if target else _SNAPSHOT_NEIGHBOR_TEXT_LIMIT,
    )
    if not text:
        return None
    return {
        "block_id": _clean_text(value.get("block_id"), limit=160),
        "anchor_id": _clean_text(value.get("anchor_id"), limit=160),
        "kind": _clean_text(value.get("kind") or value.get("anchor_kind"), limit=40),
        "heading_path": _clean_text(value.get("heading_path"), limit=500),
        "page_start": _positive_int(value.get("page_start")),
        "page_end": _positive_int(value.get("page_end")),
        "line_start": _positive_int(value.get("line_start")),
        "line_end": _positive_int(value.get("line_end")),
        "text": text,
    }


def _normalize_context_snapshot(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    fingerprint = _clean_text(value.get("source_fingerprint"), limit=96)
    target = _context_snapshot_block(value.get("target"), target=True)
    if not fingerprint.startswith("sha256:") or target is None:
        return None
    try:
        captured_at = float(value.get("captured_at") or 0)
    except (TypeError, ValueError):
        captured_at = 0.0
    raw_before = value.get("before")
    raw_after = value.get("after")
    before_values = raw_before if isinstance(raw_before, list) else []
    after_values = raw_after if isinstance(raw_after, list) else []
    before = [
        block
        for block in (_context_snapshot_block(row) for row in before_values[-1:])
        if block is not None
    ]
    after = [
        block
        for block in (_context_snapshot_block(row) for row in after_values[:1])
        if block is not None
    ]
    return {
        "version": 1,
        "captured_at": captured_at if math.isfinite(captured_at) and captured_at > 0 else time.time(),
        "source_fingerprint": fingerprint,
        "reason": _clean_text(value.get("reason"), limit=80),
        "before": before,
        "target": target,
        "after": after,
    }


def research_note_evidence_context_snapshot(
    context: dict[str, Any],
    *,
    captured_at: float | None = None,
) -> dict[str, Any] | None:
    """Return a compact, versioned before/target/after evidence snapshot."""
    if not isinstance(context, dict) or _clean_text(context.get("status"), limit=24) != "exact":
        return None
    return _normalize_context_snapshot(
        {
            "version": 1,
            "captured_at": float(captured_at if captured_at is not None else time.time()),
            "source_fingerprint": context.get("source_fingerprint"),
            "reason": context.get("reason"),
            "before": context.get("before"),
            "target": context.get("target"),
            "after": context.get("after"),
        }
    )


def _verified_context_snapshot(
    value: Any,
    *,
    target: dict[str, Any],
    source_fingerprint: str,
) -> dict[str, Any] | None:
    snapshot = _normalize_context_snapshot(value)
    if snapshot is None:
        return None
    fingerprint = _clean_text(source_fingerprint, limit=96)
    if snapshot["source_fingerprint"] != fingerprint:
        raise ValueError("evidence context snapshot source changed")
    snapshot_target = snapshot["target"]
    target_locator = _locator_snapshot(target)
    if target_locator["block_id"] and snapshot_target["block_id"] != target_locator["block_id"]:
        raise ValueError("evidence context snapshot block changed")
    if target_locator["anchor_id"] and snapshot_target["anchor_id"] != target_locator["anchor_id"]:
        raise ValueError("evidence context snapshot anchor changed")
    return snapshot


def _compact_history_contexts(history: list[dict[str, Any]]) -> None:
    for row in history:
        row.pop("previous_context_snapshot", None)
        row.pop("reviewed_context_snapshot", None)


def _verification_snapshot(value: Any) -> dict[str, Any]:
    record = value if isinstance(value, dict) else {}
    try:
        verified_at = float(record.get("verified_at") or 0)
    except (TypeError, ValueError):
        verified_at = 0.0
    return {
        "status": _clean_text(record.get("status"), limit=24),
        "reason": _clean_text(record.get("reason"), limit=80),
        "verified_at": verified_at if math.isfinite(verified_at) and verified_at > 0 else None,
        "source_fingerprint": _clean_text(record.get("source_fingerprint"), limit=96),
    }


def _baseline_check_locator(link: dict[str, Any]) -> dict[str, Any]:
    locator = _locator_snapshot(link)
    return {
        key: locator[key]
        for key in (
            "block_id",
            "anchor_id",
            "anchor_kind",
            "heading_path",
            "page_start",
            "page_end",
            "line_start",
            "line_end",
        )
    }


def record_research_note_source_baseline_check(
    source_state: dict[str, Any],
    *,
    source_index: int,
    status: str,
    reason: str,
    source_fingerprint: str = "",
    checked_at: float | None = None,
) -> dict[str, Any]:
    """Persist one unresolved baseline-check outcome without rewriting evidence.

    Repeating the same outcome for the same locator and source fingerprint is
    idempotent. A changed outcome is retained in a bounded audit history.
    """
    normalized_status = _clean_text(status, limit=24).lower()
    normalized_reason = _clean_text(reason, limit=80)
    if normalized_status not in {"skipped", "failed"} or not normalized_reason:
        raise ValueError("unsupported evidence baseline check outcome")

    next_state = deepcopy(source_state) if isinstance(source_state, dict) else {}
    links = next_state.get("links")
    if not isinstance(links, list) or not 0 <= int(source_index) < len(links):
        raise ValueError("research note source link not found")
    link = links[int(source_index)]
    if not isinstance(link, dict):
        raise ValueError("research note source link not found")

    fingerprint = _clean_text(source_fingerprint, limit=96)
    if fingerprint and not fingerprint.startswith("sha256:"):
        fingerprint = ""
    locator = _baseline_check_locator(link)
    previous = link.get("evidence_baseline_check")
    try:
        previous_checked_at = float(previous.get("checked_at") or 0) if isinstance(previous, dict) else 0.0
    except (TypeError, ValueError):
        previous_checked_at = 0.0
    if (
        isinstance(previous, dict)
        and previous.get("version") == 1
        and _clean_text(previous.get("status"), limit=24).lower() == normalized_status
        and _clean_text(previous.get("reason"), limit=80) == normalized_reason
        and _clean_text(previous.get("source_fingerprint"), limit=96) == fingerprint
        and previous.get("locator") == locator
        and math.isfinite(previous_checked_at)
        and previous_checked_at > 0
    ):
        return next_state

    timestamp = float(checked_at if checked_at is not None else time.time())
    if not math.isfinite(timestamp) or timestamp <= 0:
        timestamp = time.time()
    record = {
        "version": 1,
        "checked_at": timestamp,
        "status": normalized_status,
        "reason": normalized_reason,
        "source_fingerprint": fingerprint,
        "locator": locator,
    }
    link["evidence_baseline_check"] = record
    raw_history = link.get("evidence_baseline_check_history")
    history = [deepcopy(row) for row in raw_history if isinstance(row, dict)] if isinstance(raw_history, list) else []
    history.append(deepcopy(record))
    link["evidence_baseline_check_history"] = history[-_BASELINE_CHECK_HISTORY_LIMIT:]
    return next_state


def resolve_research_note_source_baseline_check(
    source_state: dict[str, Any],
    *,
    source_index: int,
    reason: str,
    source_fingerprint: str = "",
    resolved_at: float | None = None,
) -> dict[str, Any]:
    """Clear a current baseline-check issue while retaining its audit trail."""
    next_state = deepcopy(source_state) if isinstance(source_state, dict) else {}
    links = next_state.get("links")
    if not isinstance(links, list) or not 0 <= int(source_index) < len(links):
        raise ValueError("research note source link not found")
    link = links[int(source_index)]
    if not isinstance(link, dict):
        raise ValueError("research note source link not found")
    previous = link.get("evidence_baseline_check")
    if not isinstance(previous, dict):
        link.pop("evidence_baseline_check", None)
        return next_state

    timestamp = float(resolved_at if resolved_at is not None else time.time())
    if not math.isfinite(timestamp) or timestamp <= 0:
        timestamp = time.time()
    fingerprint = _clean_text(source_fingerprint, limit=96)
    if fingerprint and not fingerprint.startswith("sha256:"):
        fingerprint = ""
    raw_history = link.get("evidence_baseline_check_history")
    history = [deepcopy(row) for row in raw_history if isinstance(row, dict)] if isinstance(raw_history, list) else []
    history.append(
        {
            "version": 1,
            "resolved_at": timestamp,
            "status": "resolved",
            "reason": _clean_text(reason, limit=80),
            "source_fingerprint": fingerprint,
            "previous_status": _clean_text(previous.get("status"), limit=24),
            "previous_reason": _clean_text(previous.get("reason"), limit=80),
            "locator": deepcopy(previous.get("locator")) if isinstance(previous.get("locator"), dict) else {},
        }
    )
    link["evidence_baseline_check_history"] = history[-_BASELINE_CHECK_HISTORY_LIMIT:]
    link.pop("evidence_baseline_check", None)
    return next_state


def capture_research_note_source_context_baseline(
    source_state: dict[str, Any],
    *,
    source_index: int,
    source_path: str,
    target: dict[str, Any],
    source_fingerprint: str,
    context_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Capture a current context baseline without changing saved evidence.

    This is intentionally stricter than locator repair. The stored source must
    still have the same stable locator, any prior fingerprint must still match,
    and an existing context snapshot is never replaced by this path.
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
    if link.get("evidence_context_snapshot") is not None:
        raise ValueError("research note evidence context already exists")

    previous_verification = _verification_snapshot(link.get("evidence_verification"))
    if previous_verification["status"] == "invalid":
        raise ValueError("invalid research note evidence cannot establish a baseline")
    fingerprint = _clean_text(source_fingerprint, limit=96)
    if not fingerprint.startswith("sha256:"):
        raise ValueError("verified source fingerprint required")
    saved_fingerprint = previous_verification["source_fingerprint"]
    if saved_fingerprint and saved_fingerprint != fingerprint:
        raise ValueError("research note evidence source changed")

    saved_locator = _locator_snapshot(link)
    target_locator = _locator_snapshot(target)
    if not saved_locator["block_id"] and not saved_locator["anchor_id"]:
        raise ValueError("research note evidence has no stable locator")
    if saved_locator["block_id"] and target_locator["block_id"] != saved_locator["block_id"]:
        raise ValueError("research note evidence block changed")
    if saved_locator["anchor_id"] and target_locator["anchor_id"] != saved_locator["anchor_id"]:
        raise ValueError("research note evidence anchor changed")
    if not target_locator["evidence_quote"]:
        raise ValueError("verified evidence target has no text")

    verified_snapshot = _verified_context_snapshot(
        context_snapshot,
        target=target,
        source_fingerprint=fingerprint,
    )
    if verified_snapshot is None:
        raise ValueError("verified evidence context required")

    captured_at = float(verified_snapshot["captured_at"])
    link["evidence_context_snapshot"] = verified_snapshot
    link["evidence_verification"] = {
        "status": "exact",
        "reason": "baseline_backfilled",
        "verified_at": captured_at,
        "source_fingerprint": fingerprint,
    }
    raw_history = link.get("evidence_context_history")
    history = [deepcopy(row) for row in raw_history if isinstance(row, dict)] if isinstance(raw_history, list) else []
    history.append(
        {
            "version": 1,
            "action": "baseline_backfilled",
            "captured_at": captured_at,
            "source_fingerprint": fingerprint,
            "locator": {
                key: saved_locator[key]
                for key in (
                    "block_id",
                    "anchor_id",
                    "anchor_kind",
                    "heading_path",
                    "page_start",
                    "page_end",
                    "line_start",
                    "line_end",
                )
            },
            "previous_verification": previous_verification,
        }
    )
    link["evidence_context_history"] = history[-_CONTEXT_HISTORY_LIMIT:]
    return resolve_research_note_source_baseline_check(
        next_state,
        source_index=source_index,
        reason="baseline_created",
        source_fingerprint=fingerprint,
        resolved_at=captured_at,
    )


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
    context_snapshot: dict[str, Any] | None = None,
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
    previous_context_snapshot = _normalize_context_snapshot(link.get("evidence_context_snapshot"))
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
    verified_snapshot = _verified_context_snapshot(
        context_snapshot,
        target=target,
        source_fingerprint=fingerprint,
    )
    link["evidence_verification"] = {
        "status": "exact",
        "reason": "confirmed_candidate",
        "verified_at": confirmed_at,
        "source_fingerprint": fingerprint,
    }
    raw_history = link.get("locator_repair_history")
    history = [deepcopy(row) for row in raw_history if isinstance(row, dict)] if isinstance(raw_history, list) else []
    _compact_history_contexts(history)
    history.append(
        {
            "version": 1,
            "repaired_at": confirmed_at,
            "source_fingerprint": fingerprint,
            "from": previous,
            "to": target_snapshot,
            "previous_context_snapshot": previous_context_snapshot,
            "reviewed_context_snapshot": verified_snapshot,
        }
    )
    link["locator_repair_history"] = history[-_REPAIR_HISTORY_LIMIT:]
    if verified_snapshot is not None:
        link["evidence_context_snapshot"] = verified_snapshot
    else:
        link.pop("evidence_context_snapshot", None)
    return resolve_research_note_source_baseline_check(
        next_state,
        source_index=source_index,
        reason="candidate_repaired",
        source_fingerprint=fingerprint,
        resolved_at=confirmed_at,
    )


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
    context_snapshot: dict[str, Any] | None = None,
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
    verified_snapshot = _verified_context_snapshot(
        context_snapshot,
        target=target,
        source_fingerprint=fingerprint,
    )
    previous_context_snapshot = _normalize_context_snapshot(link.get("evidence_context_snapshot"))
    confirmed_at = float(reviewed_at if reviewed_at is not None else time.time())
    raw_history = link.get("evidence_review_history")
    history = [deepcopy(row) for row in raw_history if isinstance(row, dict)] if isinstance(raw_history, list) else []
    _compact_history_contexts(history)
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
            "previous_context_snapshot": previous_context_snapshot,
            "reviewed_context_snapshot": verified_snapshot,
        }
    )
    link["evidence_review_history"] = history[-_REVIEW_HISTORY_LIMIT:]
    link["evidence_verification"] = {
        "status": "exact" if normalized_decision == "confirmed" else "invalid",
        "reason": "confirmed_source_change" if normalized_decision == "confirmed" else "marked_invalid",
        "verified_at": confirmed_at,
        "source_fingerprint": fingerprint,
    }
    if normalized_decision == "confirmed" and verified_snapshot is not None:
        link["evidence_context_snapshot"] = verified_snapshot
    elif normalized_decision == "confirmed":
        link.pop("evidence_context_snapshot", None)
    return resolve_research_note_source_baseline_check(
        next_state,
        source_index=source_index,
        reason=f"source_review_{normalized_decision}",
        source_fingerprint=fingerprint,
        resolved_at=confirmed_at,
    )


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
            "context_snapshot": None,
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
        result = {
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
        result["context_snapshot"] = research_note_evidence_context_snapshot(result)
        return result

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
        "context_snapshot": None,
    }
