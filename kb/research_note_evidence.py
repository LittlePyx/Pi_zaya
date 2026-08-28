from __future__ import annotations

import hashlib
import math
import time
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from kb.path_safety import clean_file_source_path_input
from kb.source_blocks import (
    SourceBlock,
    load_source_blocks,
    match_source_blocks,
    normalize_match_text,
)

_TEXT_LIMIT = 2_400
_CONTEXT_RADIUS = 1
_CANDIDATE_LIMIT = 3
_CANDIDATE_SCAN_LIMIT = 12
_REPAIR_HISTORY_LIMIT = 12
_REVIEW_HISTORY_LIMIT = 12
_CONTEXT_HISTORY_LIMIT = 12
_AUTO_RESOLUTION_HISTORY_LIMIT = 12
_BASELINE_CHECK_HISTORY_LIMIT = 12
_SOURCE_PATH_MIGRATION_HISTORY_LIMIT = 12
_SNAPSHOT_NEIGHBOR_TEXT_LIMIT = 600
_SNAPSHOT_TARGET_TEXT_LIMIT = 1_200
_CONTEXT_SCORE_MARGIN = 0.10
_CONTEXT_FIELD_WEIGHTS = {
    "target": 0.20,
    "before": 0.25,
    "after": 0.25,
    "heading": 0.18,
    "page": 0.12,
}
RESEARCH_NOTE_EVIDENCE_MAINTENANCE_VERSION = 3
_SYSTEM_RETRY_REASONS = {"unavailable", "verification_failed", "source_changed_during_capture"}
_USER_DECISION_REASONS = {"candidate", "source_changed", "location_changed"}


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


def research_note_source_identity(value: Any) -> str:
    return _clean_text(value, limit=1_200).replace("\\", "/").casefold()


def research_note_source_path_matches(value: Any, source_paths: Any) -> bool:
    """Match a saved source path to equivalent recovery-event path forms.

    The basename fallback admits root-relative library paths into the bounded
    candidate scan. The verifier still resolves and checks the real source
    before it can update evidence state.
    """
    identity = research_note_source_identity(value)
    if not identity:
        return False
    filename = identity.rsplit("/", 1)[-1]
    for raw_path in source_paths or ():
        candidate = research_note_source_identity(raw_path)
        if candidate and (candidate == identity or candidate.rsplit("/", 1)[-1] == filename):
            return True
    return False


def research_note_source_state_matches_paths(
    source_state: Any,
    source_paths: Any = None,
) -> bool:
    if not isinstance(source_state, dict):
        return False
    links = source_state.get("links")
    if not isinstance(links, list):
        return False
    scan_all = source_paths is None
    return any(
        isinstance(link, dict)
        and str(link.get("kind") or "").strip() != "answer"
        and bool(research_note_source_identity(link.get("source_path")))
        and (scan_all or research_note_source_path_matches(link.get("source_path"), source_paths))
        for link in links
    )


def _source_migration_identity(value: Any) -> str:
    cleaned = clean_file_source_path_input(value)
    return research_note_source_identity(cleaned).rstrip("/")


def _source_path_leaf(value: Any) -> str:
    return clean_file_source_path_input(value).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def research_note_source_state_dependency_count(
    source_state: Any,
    *,
    exact_source_paths: list[str] | tuple[str, ...] | None = None,
    prefix_source_paths: list[str] | tuple[str, ...] | None = None,
) -> int:
    """Count non-answer links that exactly depend on an explicit source set."""
    if not isinstance(source_state, dict):
        return 0
    links = source_state.get("links")
    if not isinstance(links, list):
        return 0
    exact = {
        identity
        for value in exact_source_paths or ()
        if (identity := _source_migration_identity(value))
    }
    prefixes = sorted(
        {
            identity
            for value in prefix_source_paths or ()
            if (identity := _source_migration_identity(value))
        },
        key=len,
        reverse=True,
    )
    if not exact and not prefixes:
        return 0
    count = 0
    for link in links:
        if not isinstance(link, dict) or str(link.get("kind") or "").strip() == "answer":
            continue
        identity = _source_migration_identity(link.get("source_path"))
        if not identity:
            continue
        if identity in exact or any(identity == prefix or identity.startswith(f"{prefix}/") for prefix in prefixes):
            count += 1
    return count


def migrate_research_note_source_state_paths(
    source_state: Any,
    *,
    exact_path_migrations: dict[str, str] | None = None,
    prefix_path_migrations: list[tuple[str, str]] | tuple[tuple[str, str], ...] | None = None,
    reason: str = "library_source_renamed",
    migrated_at: float | None = None,
    update_source_name: bool = True,
) -> tuple[dict[str, Any], int]:
    """Rewrite exact library source identities while retaining a bounded audit.

    Prefix migrations are intentionally explicit. Basename guessing is unsafe
    here because two library roots may contain files with the same leaf name.
    """
    next_state = deepcopy(source_state) if isinstance(source_state, dict) else {}
    links = next_state.get("links")
    if not isinstance(links, list):
        return next_state, 0

    exact: dict[str, str] = {}
    for old_path, new_path in dict(exact_path_migrations or {}).items():
        old_identity = _source_migration_identity(old_path)
        new_clean = _clean_text(clean_file_source_path_input(new_path), limit=1_200)
        if old_identity and new_clean and old_identity != _source_migration_identity(new_clean):
            exact[old_identity] = new_clean

    prefixes: list[tuple[str, str, int]] = []
    for row in prefix_path_migrations or ():
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            continue
        old_identity = _source_migration_identity(row[0])
        new_clean = _clean_text(clean_file_source_path_input(row[1]), limit=1_200).rstrip("/\\")
        if old_identity and new_clean and old_identity != _source_migration_identity(new_clean):
            prefixes.append((old_identity, new_clean, len(old_identity.split("/"))))
    prefixes.sort(key=lambda item: len(item[0]), reverse=True)

    if not exact and not prefixes:
        return next_state, 0
    timestamp = float(migrated_at if migrated_at is not None else time.time())
    if not math.isfinite(timestamp) or timestamp <= 0:
        timestamp = time.time()
    migration_reason = _clean_text(reason, limit=80) or "library_source_renamed"
    migrated = 0

    for link in links:
        if not isinstance(link, dict) or str(link.get("kind") or "").strip() == "answer":
            continue
        old_path = _clean_text(link.get("source_path"), limit=1_200)
        old_identity = _source_migration_identity(old_path)
        if not old_identity:
            continue
        new_path = exact.get(old_identity, "")
        if not new_path:
            for old_prefix, new_prefix, old_part_count in prefixes:
                if old_identity == old_prefix:
                    new_path = new_prefix
                    break
                marker = f"{old_prefix}/"
                if old_identity.startswith(marker):
                    cleaned_old = clean_file_source_path_input(old_path).replace("\\", "/")
                    suffix = "/".join(cleaned_old.split("/")[old_part_count:]).lstrip("/")
                    new_path = f"{new_prefix}/{suffix}" if suffix else new_prefix
                    break
        new_path = _clean_text(new_path, limit=1_200)
        if not new_path or _source_migration_identity(new_path) == old_identity:
            continue

        raw_history = link.get("source_path_migration_history")
        history = [deepcopy(row) for row in raw_history if isinstance(row, dict)] if isinstance(raw_history, list) else []
        history.append(
            {
                "version": 1,
                "migrated_at": timestamp,
                "reason": migration_reason,
                "from": old_path,
                "to": new_path,
            }
        )
        link["source_path_migration_history"] = history[-_SOURCE_PATH_MIGRATION_HISTORY_LIMIT:]
        link["source_path"] = new_path
        old_leaf = _source_path_leaf(old_path)
        source_name = _clean_text(link.get("source_name"), limit=500)
        if update_source_name and source_name and source_name.casefold() == old_leaf.casefold():
            link["source_name"] = _source_path_leaf(new_path)
        migrated += 1

    return next_state, migrated


def _source_identity(value: Any) -> str:
    return research_note_source_identity(value)


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
        row.pop("resolved_context_snapshot", None)


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


def research_note_evidence_maintenance_due(
    source_state: dict[str, Any] | None,
    *,
    now: float | None = None,
) -> bool:
    """Return whether any local note evidence needs a system maintenance pass."""
    state = source_state if isinstance(source_state, dict) else {}
    links = state.get("links")
    if not isinstance(links, list):
        return False
    timestamp = float(now if now is not None else time.time())
    if not math.isfinite(timestamp) or timestamp <= 0:
        timestamp = time.time()
    for link in links:
        if not isinstance(link, dict) or _clean_text(link.get("kind"), limit=24) == "answer":
            continue
        if not _clean_text(link.get("source_path"), limit=1_200):
            continue
        verification = link.get("evidence_verification")
        if isinstance(verification, dict) and _clean_text(verification.get("status"), limit=24) == "invalid":
            continue
        check = link.get("evidence_baseline_check")
        if link.get("evidence_context_snapshot") is not None:
            if not isinstance(check, dict):
                continue
            saved_reason = _clean_text(check.get("reason"), limit=80)
            saved_status = _clean_text(check.get("status"), limit=24)
            saved_version = _positive_int(check.get("maintenance_version")) or 0
            if (
                not (
                    saved_reason in _USER_DECISION_REASONS
                    and saved_version < RESEARCH_NOTE_EVIDENCE_MAINTENANCE_VERSION
                )
                and saved_status != "failed"
                and saved_reason not in _SYSTEM_RETRY_REASONS
            ):
                continue
        if not isinstance(check, dict):
            return True
        reason = _clean_text(check.get("reason"), limit=80)
        status = _clean_text(check.get("status"), limit=24)
        maintenance_version = _positive_int(check.get("maintenance_version")) or 0
        if reason in _USER_DECISION_REASONS:
            if maintenance_version < RESEARCH_NOTE_EVIDENCE_MAINTENANCE_VERSION:
                return True
            continue
        if status == "failed" or reason in _SYSTEM_RETRY_REASONS:
            try:
                retry_after = float(check.get("retry_after") or 0)
            except (TypeError, ValueError):
                retry_after = 0.0
            if not math.isfinite(retry_after) or retry_after <= timestamp:
                return True
            continue
        if maintenance_version < RESEARCH_NOTE_EVIDENCE_MAINTENANCE_VERSION:
            return True
    return False


def record_research_note_source_baseline_check(
    source_state: dict[str, Any],
    *,
    source_index: int,
    status: str,
    reason: str,
    source_fingerprint: str = "",
    checked_at: float | None = None,
    maintenance_version: int | None = None,
    retry_count: int | None = None,
    retry_after: float | None = None,
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
    normalized_maintenance_version = _positive_int(maintenance_version)
    normalized_retry_count = _positive_int(retry_count)
    try:
        normalized_retry_after = float(retry_after or 0)
    except (TypeError, ValueError):
        normalized_retry_after = 0.0
    if not math.isfinite(normalized_retry_after) or normalized_retry_after <= 0:
        normalized_retry_after = 0.0
    previous = link.get("evidence_baseline_check")
    try:
        previous_checked_at = float(previous.get("checked_at") or 0) if isinstance(previous, dict) else 0.0
    except (TypeError, ValueError):
        previous_checked_at = 0.0
    try:
        previous_retry_after = float(previous.get("retry_after") or 0) if isinstance(previous, dict) else 0.0
    except (TypeError, ValueError):
        previous_retry_after = 0.0
    if (
        isinstance(previous, dict)
        and previous.get("version") == 1
        and _clean_text(previous.get("status"), limit=24).lower() == normalized_status
        and _clean_text(previous.get("reason"), limit=80) == normalized_reason
        and _clean_text(previous.get("source_fingerprint"), limit=96) == fingerprint
        and previous.get("locator") == locator
        and (_positive_int(previous.get("maintenance_version")) == normalized_maintenance_version)
        and (_positive_int(previous.get("retry_count")) == normalized_retry_count)
        and previous_retry_after == normalized_retry_after
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
    if normalized_maintenance_version is not None:
        record["maintenance_version"] = normalized_maintenance_version
    if normalized_retry_count is not None:
        record["retry_count"] = normalized_retry_count
    if normalized_retry_after > 0:
        record["retry_after"] = normalized_retry_after
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


def auto_resolve_research_note_source_state(
    source_state: dict[str, Any],
    *,
    source_index: int,
    source_path: str,
    target: dict[str, Any],
    source_fingerprint: str,
    context_snapshot: dict[str, Any],
    match_reason: str,
    action: str,
    resolution_evidence: dict[str, Any] | None = None,
    resolved_at: float | None = None,
) -> dict[str, Any]:
    """Resolve only evidence matches that are deterministic without user judgment.

    Stable block/anchor matches may refresh a changed source fingerprint. A
    relocated quote is accepted only after the verifier has proved that the
    complete saved quote occurs at one exact source block. The saved quote is
    never rewritten by this automatic path.
    """
    normalized_match_reason = _clean_text(match_reason, limit=80)
    normalized_action = _clean_text(action, limit=80)
    allowed_actions = {
        "source_change_auto_verified",
        "exact_quote_auto_relocated",
        "stable_locator_auto_aligned",
        "saved_context_auto_disambiguated",
    }
    if normalized_match_reason not in {
        "exact_block",
        "exact_anchor",
        "quote_relocated",
        "context_disambiguated",
    }:
        raise ValueError("evidence match requires user review")
    if normalized_action not in allowed_actions:
        raise ValueError("unsupported automatic evidence resolution")

    next_state = deepcopy(source_state) if isinstance(source_state, dict) else {}
    links = next_state.get("links")
    if not isinstance(links, list) or not 0 <= int(source_index) < len(links):
        raise ValueError("research note source link not found")
    link = links[int(source_index)]
    if not isinstance(link, dict):
        raise ValueError("research note source link not found")
    if _source_identity(link.get("source_path")) != _source_identity(source_path):
        raise ValueError("research note source changed")

    saved_locator = _locator_snapshot(link)
    target_for_locator = dict(target)
    target_for_locator["anchor_kind"] = target.get("anchor_kind") or target.get("kind")
    target_locator = _locator_snapshot(target_for_locator)
    if not target_locator["block_id"] and not target_locator["anchor_id"]:
        raise ValueError("verified evidence target has no stable locator")
    if not target_locator["evidence_quote"]:
        raise ValueError("verified evidence target has no text")

    saved_quote = normalize_match_text(saved_locator["evidence_quote"])
    target_text = normalize_match_text(target_locator["evidence_quote"])
    if len(saved_quote) < 16 or saved_quote not in target_text:
        raise ValueError("saved evidence quote is not an exact current-source match")
    if normalized_match_reason == "exact_block" and (
        not saved_locator["block_id"] or saved_locator["block_id"] != target_locator["block_id"]
    ):
        raise ValueError("saved evidence block is no longer exact")
    if normalized_match_reason == "exact_anchor" and (
        not saved_locator["anchor_id"] or saved_locator["anchor_id"] != target_locator["anchor_id"]
    ):
        raise ValueError("saved evidence anchor is no longer exact")

    fingerprint = _clean_text(source_fingerprint, limit=96)
    if not fingerprint.startswith("sha256:"):
        raise ValueError("verified source fingerprint required")
    verified_snapshot = _verified_context_snapshot(
        context_snapshot,
        target=target_for_locator,
        source_fingerprint=fingerprint,
    )
    if verified_snapshot is None:
        raise ValueError("verified evidence context required")

    previous_verification = _verification_snapshot(link.get("evidence_verification"))
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
    ):
        value = target_locator[key]
        if value is None or value == "":
            link.pop(key, None)
        else:
            link[key] = value
    link.pop("location_label", None)
    if normalized_action in {
        "exact_quote_auto_relocated",
        "stable_locator_auto_aligned",
        "saved_context_auto_disambiguated",
    }:
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

    timestamp = float(resolved_at if resolved_at is not None else time.time())
    if not math.isfinite(timestamp) or timestamp <= 0:
        timestamp = time.time()
    link["evidence_verification"] = {
        "status": "exact",
        "reason": normalized_action,
        "verified_at": timestamp,
        "source_fingerprint": fingerprint,
    }
    link["evidence_context_snapshot"] = verified_snapshot
    raw_history = link.get("evidence_auto_resolution_history")
    history = [deepcopy(row) for row in raw_history if isinstance(row, dict)] if isinstance(raw_history, list) else []
    _compact_history_contexts(history)
    audit_row = {
        "version": 1,
        "resolved_at": timestamp,
        "action": normalized_action,
        "match_reason": normalized_match_reason,
        "source_fingerprint": fingerprint,
        "from": saved_locator,
        "to": target_locator,
        "previous_verification": previous_verification,
        "previous_context_snapshot": previous_context_snapshot,
        "resolved_context_snapshot": verified_snapshot,
    }
    if normalized_match_reason == "context_disambiguated" and isinstance(resolution_evidence, dict):
        fields = ("mode", "score_version", "candidate_count", "score", "score_gap")
        cleaned_evidence = {
            field: deepcopy(resolution_evidence[field])
            for field in fields
            if field in resolution_evidence
        }
        for field in ("available_fields", "matched_fields"):
            values = resolution_evidence.get(field)
            if isinstance(values, list):
                cleaned_evidence[field] = [
                    _clean_text(value, limit=40)
                    for value in values
                    if _clean_text(value, limit=40)
                ]
        field_scores = resolution_evidence.get("field_scores")
        if isinstance(field_scores, dict):
            cleaned_evidence["field_scores"] = {
                _clean_text(field, limit=40): score
                for field, score in field_scores.items()
                if _clean_text(field, limit=40) and isinstance(score, (int, float))
            }
        if cleaned_evidence:
            audit_row["resolution_evidence"] = cleaned_evidence
    history.append(audit_row)
    link["evidence_auto_resolution_history"] = history[-_AUTO_RESOLUTION_HISTORY_LIMIT:]
    return resolve_research_note_source_baseline_check(
        next_state,
        source_index=source_index,
        reason=normalized_action,
        source_fingerprint=fingerprint,
        resolved_at=timestamp,
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
    """Record an explicit evidence decision without rewriting the saved excerpt.

    Confirmation is limited to the saved locator; a changed locator must use
    the separate candidate-repair path. Invalidation may record a re-verified
    candidate at a changed locator because it does not adopt that candidate as
    evidence. Both decisions retain the saved locator and reviewed target in a
    bounded audit trail.
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
    previous_context_snapshot = _normalize_context_snapshot(link.get("evidence_context_snapshot"))
    saved_fingerprint = _clean_text(previous_verification.get("source_fingerprint"), limit=96)
    if not saved_fingerprint and previous_context_snapshot is not None:
        saved_fingerprint = _clean_text(previous_context_snapshot.get("source_fingerprint"), limit=96)
    expected_fingerprint = _clean_text(expected_source_fingerprint, limit=96)
    fingerprint = _clean_text(source_fingerprint, limit=96)
    if not fingerprint.startswith("sha256:"):
        raise ValueError("verified source fingerprint required")
    if (
        not expected_fingerprint
        or (saved_fingerprint and saved_fingerprint != expected_fingerprint)
        or (
            not saved_fingerprint
            and (normalized_decision != "invalid" or expected_fingerprint != fingerprint)
        )
    ):
        raise ValueError("research note evidence verification changed")

    target_snapshot = _locator_snapshot(target)
    if not target_snapshot["block_id"] and not target_snapshot["anchor_id"]:
        raise ValueError("verified evidence target has no stable locator")
    if not target_snapshot["evidence_quote"]:
        raise ValueError("verified evidence target has no text")
    if (
        normalized_decision == "confirmed"
        and expected_identity["block_id"]
        and target_snapshot["block_id"] != expected_identity["block_id"]
    ):
        raise ValueError("evidence location changed; repair required")
    if (
        normalized_decision == "confirmed"
        and expected_identity["anchor_id"]
        and target_snapshot["anchor_id"] != expected_identity["anchor_id"]
    ):
        raise ValueError("evidence location changed; repair required")

    verified_snapshot = _verified_context_snapshot(
        context_snapshot,
        target=target,
        source_fingerprint=fingerprint,
    )
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
    blocks: list[SourceBlock],
    source_path: str,
    source_name: str,
    saved_context_snapshot: Any = None,
) -> dict[str, Any] | None:
    block = row.get("block")
    if not isinstance(block, dict):
        return None
    try:
        block_index = int(row.get("index"))
    except (TypeError, ValueError):
        block_index = -1
    if not 0 <= block_index < len(blocks) or blocks[block_index] is not block:
        block_index = next(
            (index for index, candidate in enumerate(blocks) if candidate is block),
            -1,
        )
    before = _context_neighbors(blocks, block_index, direction=-1) if block_index >= 0 else []
    after = _context_neighbors(blocks, block_index, direction=1) if block_index >= 0 else []
    payload = _serialize_block(block)
    payload["score"] = round(float(row.get("score") or 0.0), 4)
    payload["before"] = before
    payload["after"] = after
    context_match = _saved_context_candidate_match(
        saved_context_snapshot,
        block=block,
        before=before,
        after=after,
    )
    if context_match["available_fields"]:
        payload["context_match"] = context_match
    payload["reader_open"] = _reader_open(
        block,
        source_path=source_path,
        source_name=source_name,
        evidence_quote=_clean_text(block.get("text")),
    )
    return payload


def _normalized_text_similarity(saved_text: Any, current_text: Any, *, limit: int) -> float:
    saved = normalize_match_text(_clean_text(saved_text, limit=limit))
    current = normalize_match_text(_clean_text(current_text, limit=limit))
    if not saved or not current:
        return 0.0
    if saved == current:
        return 1.0
    if min(len(saved), len(current)) < 24:
        return 0.0
    prefix_score = 0.0
    if len(saved) >= 80 and current.startswith(saved):
        prefix_score = len(saved) / max(len(saved), len(current))
    elif len(current) >= 80 and saved.startswith(current):
        prefix_score = len(current) / max(len(saved), len(current))
    sequence_score = SequenceMatcher(None, saved, current, autojunk=False).ratio()
    return round(max(prefix_score, sequence_score), 4)


def _saved_context_text_similarity(saved_block: Any, current_block: Any) -> float:
    if not isinstance(saved_block, dict) or not isinstance(current_block, dict):
        return 0.0
    return _normalized_text_similarity(
        saved_block.get("text"),
        current_block.get("text"),
        limit=_SNAPSHOT_NEIGHBOR_TEXT_LIMIT,
    )


def _saved_page_similarity(saved_target: dict[str, Any], current_block: SourceBlock) -> float:
    saved_start = _positive_int(saved_target.get("page_start"))
    if saved_start is None:
        return 0.0
    saved_end = _positive_int(saved_target.get("page_end")) or saved_start
    current_start = _positive_int(current_block.get("page_start"))
    if current_start is None:
        return 0.0
    current_end = _positive_int(current_block.get("page_end")) or current_start
    if saved_start == current_start and saved_end == current_end:
        return 1.0
    if max(saved_start, current_start) <= min(saved_end, current_end):
        return 0.85
    distance = min(abs(saved_start - current_start), abs(saved_end - current_end))
    return 0.5 if distance == 1 else 0.0


def _saved_context_candidate_match(
    saved_context_snapshot: Any,
    *,
    block: SourceBlock,
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> dict[str, Any]:
    snapshot = _normalize_context_snapshot(saved_context_snapshot)
    if snapshot is None:
        return {"available_fields": [], "matched_fields": []}
    target = snapshot["target"]
    available_fields: list[str] = []
    matched_fields: list[str] = []
    field_scores: dict[str, float] = {}

    target_similarity = _saved_context_text_similarity(target, block)
    available_fields.append("target")
    field_scores["target"] = target_similarity
    if target_similarity >= 0.92:
        matched_fields.append("target")

    saved_before = snapshot["before"][-1] if snapshot["before"] else None
    if saved_before is not None:
        available_fields.append("before")
        current_before = before[-1] if before else None
        similarity = _saved_context_text_similarity(saved_before, current_before)
        field_scores["before"] = similarity
        if similarity >= 0.88:
            matched_fields.append("before")

    saved_after = snapshot["after"][0] if snapshot["after"] else None
    if saved_after is not None:
        available_fields.append("after")
        current_after = after[0] if after else None
        similarity = _saved_context_text_similarity(saved_after, current_after)
        field_scores["after"] = similarity
        if similarity >= 0.88:
            matched_fields.append("after")

    saved_heading = normalize_match_text(_clean_text(target.get("heading_path"), limit=500))
    if saved_heading:
        available_fields.append("heading")
        heading_similarity = _normalized_text_similarity(
            target.get("heading_path"),
            block.get("heading_path"),
            limit=500,
        )
        field_scores["heading"] = heading_similarity
        if heading_similarity >= 0.90:
            matched_fields.append("heading")

    if _positive_int(target.get("page_start")) is not None:
        available_fields.append("page")
        page_similarity = _saved_page_similarity(target, block)
        field_scores["page"] = page_similarity
        if page_similarity >= 0.99:
            matched_fields.append("page")

    evidence_score = sum(
        _CONTEXT_FIELD_WEIGHTS[field] * field_scores.get(field, 0.0)
        for field in available_fields
    )

    return {
        "available_fields": available_fields,
        "matched_fields": matched_fields,
        "field_scores": {field: round(field_scores.get(field, 0.0), 4) for field in available_fields},
        "score": round(evidence_score, 4),
    }


def _context_match_qualifies(match: dict[str, Any]) -> bool:
    available = set(match.get("available_fields") or [])
    matched = set(match.get("matched_fields") or [])
    supporting = matched.intersection({"before", "after", "heading", "page"})
    try:
        score = float(match.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    if available.intersection({"before", "after"}):
        return bool(supporting.intersection({"before", "after"})) and len(supporting) >= 2 and score >= 0.50
    return {"heading", "page"}.issubset(supporting) and score >= 0.44


def _rank_candidate_payloads(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed_candidates = list(enumerate(candidates))
    indexed_candidates.sort(
        key=lambda row: (
            -float((row[1].get("context_match") or {}).get("score") or 0.0),
            -float(row[1].get("score") or 0.0),
            row[0],
        )
    )
    ranked = [candidate for _index, candidate in indexed_candidates]
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
    if not ranked:
        return ranked
    top_match = ranked[0].get("context_match")
    if not isinstance(top_match, dict) or not _context_match_qualifies(top_match):
        return ranked
    top_score = float(top_match.get("score") or 0.0)
    second_score = (
        float((ranked[1].get("context_match") or {}).get("score") or 0.0)
        if len(ranked) > 1
        else 0.0
    )
    score_gap = top_score - second_score
    if score_gap + 1e-9 < _CONTEXT_SCORE_MARGIN:
        return ranked
    top_match["best_match"] = True
    top_match["score_gap"] = round(score_gap, 4)
    return ranked


def _disambiguate_exact_quote_with_saved_context(
    blocks: list[SourceBlock],
    quote_matches: list[int],
    saved_context_snapshot: Any,
) -> tuple[int | None, dict[str, Any] | None]:
    snapshot = _normalize_context_snapshot(saved_context_snapshot)
    if snapshot is None or len(quote_matches) <= 1:
        return None, None
    rows: list[tuple[int, dict[str, Any]]] = []
    for index in quote_matches:
        match = _saved_context_candidate_match(
            snapshot,
            block=blocks[index],
            before=_context_neighbors(blocks, index, direction=-1),
            after=_context_neighbors(blocks, index, direction=1),
        )
        rows.append((index, match))

    qualified = [(index, match) for index, match in rows if _context_match_qualifies(match)]
    if not qualified:
        return None, None
    ranked = sorted(
        qualified,
        key=lambda row: (
            float(row[1].get("score") or 0.0),
            len(set(row[1]["matched_fields"]).intersection({"before", "after"})),
            len(row[1]["matched_fields"]),
        ),
        reverse=True,
    )
    top_score = float(ranked[0][1].get("score") or 0.0)
    second_score = max(
        (
            float(match.get("score") or 0.0)
            for index, match in rows
            if index != ranked[0][0]
        ),
        default=0.0,
    )
    score_gap = top_score - second_score
    if score_gap + 1e-9 < _CONTEXT_SCORE_MARGIN:
        return None, None
    index, match = ranked[0]
    return index, {
        "mode": "weighted_saved_context",
        "score_version": 1,
        "candidate_count": len(quote_matches),
        "score": round(top_score, 4),
        "score_gap": round(score_gap, 4),
        "matched_fields": list(match["matched_fields"]),
        "available_fields": list(match["available_fields"]),
        "field_scores": dict(match.get("field_scores") or {}),
    }


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
    saved_context_snapshot: Any = None,
) -> dict[str, Any]:
    """Verify a saved note citation against converted source blocks.

    Fuzzy matches are deliberately returned as review candidates and are never
    promoted to an exact Reader target. Multiple exact-quote locations may be
    resolved only when saved-context scoring produces a qualified, clearly
    leading candidate.
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
    context_disambiguation: dict[str, Any] | None = None
    if target_index is None and len(quote_norm) >= 16:
        quote_matches = [
            index
            for index, block in enumerate(blocks)
            if quote_norm in normalize_match_text(str(block.get("text") or ""))
        ]
        if len(quote_matches) == 1:
            target_index = quote_matches[0]
            exact_reason = "quote_relocated"
        elif len(quote_matches) > 1:
            target_index, context_disambiguation = _disambiguate_exact_quote_with_saved_context(
                blocks,
                quote_matches,
                saved_context_snapshot,
            )
            if target_index is not None:
                exact_reason = "context_disambiguated"

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
        if context_disambiguation is not None:
            result["context_disambiguation"] = context_disambiguation
        return result

    ranked = match_source_blocks(
        blocks,
        snippet=quote,
        heading_path=requested_heading,
        prefer_kind=requested_kind,
        limit=_CANDIDATE_SCAN_LIMIT,
    )
    candidates = _rank_candidate_payloads([
        candidate
        for candidate in (
            _candidate_payload(
                row,
                blocks=blocks,
                source_path=source_path_clean,
                source_name=source_name_clean,
                saved_context_snapshot=saved_context_snapshot,
            )
            for row in ranked
        )
        if candidate is not None
    ])[:_CANDIDATE_LIMIT]
    if len(quote_matches) > 1:
        exact_rows = [
            {"score": 1.0, "block": blocks[index], "index": index}
            for index in quote_matches
        ]
        candidates = _rank_candidate_payloads([
            candidate
            for candidate in (
                _candidate_payload(
                    row,
                    blocks=blocks,
                    source_path=source_path_clean,
                    source_name=source_name_clean,
                    saved_context_snapshot=saved_context_snapshot,
                )
                for row in exact_rows
            )
            if candidate is not None
        ])[:_CANDIDATE_LIMIT]

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
