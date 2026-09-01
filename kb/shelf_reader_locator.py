from __future__ import annotations

from typing import Any

from kb.source_blocks import SourceBlock, match_source_blocks, normalize_match_text


_PRECISE_KINDS = {
    "blockquote",
    "code",
    "equation",
    "figure",
    "inline_formula",
    "list_item",
    "paragraph",
    "table",
}


def _text(value: Any, *, limit: int = 4_000) -> str:
    return str(value or "").strip()[:limit]


def _positive_int(value: Any) -> int:
    try:
        number = int(value or 0)
    except Exception:
        return 0
    return number if number > 0 else 0


def _evidence_query(item: dict[str, Any]) -> str:
    for key in (
        "reader_evidence_quote",
        "evidence_quote",
        "shelf_excerpt",
        "card_evidence",
        "raw",
    ):
        value = _text(item.get(key))
        if value:
            return value
    return ""


def _block_locator(block: SourceBlock) -> dict[str, Any]:
    return {
        "block_id": _text(block.get("block_id"), limit=240),
        "anchor_id": _text(block.get("anchor_id"), limit=240),
        "anchor_kind": _text(block.get("kind"), limit=80),
        "heading_path": _text(block.get("heading_path"), limit=1_200),
        "page_start": _positive_int(block.get("page_start")),
        "page_end": _positive_int(block.get("page_end")),
    }


def _find_identity_block(blocks: list[SourceBlock], *, block_id: str, anchor_id: str) -> SourceBlock | None:
    if block_id:
        for block in blocks:
            if _text(block.get("block_id"), limit=240) == block_id:
                return block
    if anchor_id:
        for block in blocks:
            if _text(block.get("anchor_id"), limit=240) == anchor_id:
                return block
    return None


def _contains_evidence(block: SourceBlock, query_norm: str) -> bool:
    if not query_norm:
        return False
    block_norm = normalize_match_text(_text(block.get("text")))
    if not block_norm:
        return False
    if query_norm in block_norm:
        return True
    return len(block_norm) >= 24 and block_norm in query_norm


def _ranked_rows(
    blocks: list[SourceBlock],
    *,
    query: str,
    heading_path: str,
    anchor_kind: str,
) -> list[dict[str, Any]]:
    return match_source_blocks(
        blocks,
        snippet=query,
        heading_path=heading_path,
        prefer_kind=anchor_kind,
        limit=3,
        score_floor=0.0,
    )


def _identity_is_supported(
    block: SourceBlock,
    *,
    query: str,
    query_norm: str,
    heading_path: str,
    anchor_kind: str,
) -> bool:
    if _contains_evidence(block, query_norm):
        return True
    if not query and not heading_path:
        return True
    rows = _ranked_rows(
        [block],
        query=query,
        # When evidence text exists it must independently support the saved
        # identity. A broad-but-still-valid heading and matching block kind can
        # otherwise keep a stale sequence id after reconversion.
        heading_path="" if query else heading_path,
        anchor_kind="" if query else anchor_kind,
    )
    score = float(rows[0].get("score") or 0.0) if rows else 0.0
    if query:
        return score >= 0.52
    return score >= 0.28


def _choose_recovered_block(
    blocks: list[SourceBlock],
    *,
    query: str,
    query_norm: str,
    heading_path: str,
    anchor_kind: str,
) -> tuple[SourceBlock | None, str, float]:
    candidates = [
        block
        for block in blocks
        if _text(block.get("kind"), limit=80).lower() in _PRECISE_KINDS
    ]
    if not candidates:
        return None, "no_precise_blocks", 0.0

    exact = [block for block in candidates if _contains_evidence(block, query_norm)] if query_norm else []
    if len(exact) == 1:
        return exact[0], "unique_evidence_quote", 1.0

    pool = exact if exact else candidates
    ranked = _ranked_rows(
        pool,
        query=query,
        heading_path=heading_path,
        anchor_kind=anchor_kind,
    )
    if not ranked:
        return None, "no_match", 0.0
    best = ranked[0]
    best_block = best.get("block")
    if not isinstance(best_block, dict):
        return None, "no_match", 0.0
    best_score = float(best.get("score") or 0.0)
    second_score = float(ranked[1].get("score") or 0.0) if len(ranked) > 1 else 0.0
    margin = best_score - second_score

    if exact:
        if best_score >= 0.28 and margin >= 0.08:
            return best_block, "context_disambiguated_quote", best_score
        return None, "ambiguous_quote", best_score

    if best_score >= 0.64 and (len(ranked) == 1 or margin >= 0.12):
        return best_block, "high_confidence_evidence_match", best_score
    return None, "ambiguous_or_weak_match", best_score


def maintain_shelf_reader_locator(
    blocks: list[SourceBlock],
    item: dict[str, Any],
) -> dict[str, Any]:
    """Validate or recover one persisted shelf locator without model calls.

    Automatic persistence is intentionally conservative: an existing identity is
    retained only when its block still supports the saved evidence/context, and a
    replacement is returned only for a unique or clearly separated match.
    """

    key = _text(item.get("key"), limit=300)
    block_id = _text(item.get("block_id"), limit=240)
    anchor_id = _text(item.get("anchor_id"), limit=240)
    anchor_kind = _text(item.get("anchor_kind"), limit=80).lower()
    heading_path = _text(item.get("heading_path"), limit=1_200)
    query = _evidence_query(item)
    query_norm = normalize_match_text(query)
    if not key:
        return {"key": "", "ok": False, "changed": False, "status": "invalid", "reason": "missing_key"}
    if not blocks:
        return {"key": key, "ok": False, "changed": False, "status": "unresolved", "reason": "no_blocks"}
    if not (block_id or anchor_id or query or heading_path):
        return {"key": key, "ok": False, "changed": False, "status": "skipped", "reason": "no_locator_evidence"}

    direct = _find_identity_block(blocks, block_id=block_id, anchor_id=anchor_id)
    reason = ""
    score = 0.0
    target: SourceBlock | None = None
    if direct is not None and _identity_is_supported(
        direct,
        query=query,
        query_norm=query_norm,
        heading_path=heading_path,
        anchor_kind=anchor_kind,
    ):
        target = direct
        reason = "identity_valid"
        score = 1.0
    else:
        target, reason, score = _choose_recovered_block(
            blocks,
            query=query,
            query_norm=query_norm,
            heading_path=heading_path,
            anchor_kind=anchor_kind,
        )

    if target is None:
        return {
            "key": key,
            "ok": False,
            "changed": False,
            "status": "unresolved",
            "reason": reason or "no_match",
            "score": round(score, 6),
        }

    locator = _block_locator(target)
    changed = any(
        _text(item.get(field), limit=1_200) != _text(locator.get(field), limit=1_200)
        for field in ("block_id", "anchor_id", "anchor_kind", "heading_path")
    ) or any(
        _positive_int(item.get(field)) != _positive_int(locator.get(field))
        for field in ("page_start", "page_end")
    )
    return {
        "key": key,
        "ok": True,
        "changed": changed,
        "status": "recovered" if changed else "ready",
        "reason": reason,
        "score": round(score, 6),
        "locator": locator,
    }
