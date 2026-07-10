from __future__ import annotations

from pathlib import Path
import re


_NUMERIC_SOURCE_CITE_RE = re.compile(
    r"(?<!\[)\[(\d{1,3}(?:\s*(?:,|-)\s*\d{1,3})*)\](?![\]\(])"
)


def _text(value: object) -> str:
    return str(value or "").strip()


def _record(value: object) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def _source_path(value: object) -> str:
    if isinstance(value, dict):
        for key in ("source_path", "sourcePath", "path"):
            candidate = _text(value.get(key))
            if candidate:
                return candidate
    return _text(value)


def _append_source(out: list[str], seen: set[str], value: object, *, limit: int) -> None:
    source = _source_path(value)
    if not source:
        return
    key = source.replace("/", "\\").lower()
    if key in seen:
        return
    seen.add(key)
    out.append(source)
    if len(out) > limit:
        del out[limit:]


def _answer_source_indices(answer: object, *, max_index: int) -> list[int]:
    out: list[int] = []
    for match in _NUMERIC_SOURCE_CITE_RE.finditer(_text(answer)):
        raw = match.group(1)
        values: list[int] = []
        if "-" in raw and "," not in raw:
            left, _, right = raw.partition("-")
            try:
                start, end = int(left.strip()), int(right.strip())
            except ValueError:
                continue
            if 0 < start <= end <= max_index and end - start <= 12:
                values = list(range(start, end + 1))
        else:
            for item in raw.split(","):
                try:
                    values.append(int(item.strip()))
                except ValueError:
                    continue
        for value in values:
            if 0 < value <= max_index and value not in out:
                out.append(value)
    return out


def previous_assistant_source_hints(
    messages: list[dict] | None,
    *,
    before_message_id: int,
    limit: int = 8,
) -> list[str]:
    max_items = max(1, min(12, int(limit or 8)))
    for message in reversed(list(messages or [])):
        if not isinstance(message, dict) or _text(message.get("role")).lower() != "assistant":
            continue
        try:
            message_id = int(message.get("id") or 0)
        except Exception:
            message_id = 0
        if message_id <= 0 or message_id >= int(before_message_id or 0):
            continue

        meta = _record(message.get("meta"))
        out: list[str] = []
        seen: set[str] = set()
        canonical_paths = [
            _source_path(value)
            for value in list(meta.get("canonical_hit_paths") or [])
            if _source_path(value)
        ]
        for source_index in _answer_source_indices(
            message.get("content"),
            max_index=len(canonical_paths),
        ):
            _append_source(out, seen, canonical_paths[source_index - 1], limit=max_items)
        if out:
            return out[:max_items]

        contracts = _record(meta.get("paper_guide_contracts"))
        for value in list(contracts.get("doc_list") or []):
            _append_source(out, seen, value, limit=max_items)
        packet = _record(contracts.get("render_packet"))
        for value in list(packet.get("cite_details") or []):
            _append_source(out, seen, value, limit=max_items)

        render_cache = _record(meta.get("render_cache"))
        for value in list(render_cache.get("cite_details") or []):
            _append_source(out, seen, value, limit=max_items)
        if out:
            return out[:max_items]

        for value in canonical_paths:
            _append_source(out, seen, value, limit=max_items)
        if out:
            return out[:max_items]
    return []


def previous_assistant_reference_hits(
    messages: list[dict] | None,
    refs_by_user: dict[int, dict] | None,
    *,
    before_message_id: int,
    source_hints: list[str] | None = None,
) -> list[dict]:
    rows = [message for message in list(messages or []) if isinstance(message, dict)]
    assistant_index = -1
    for idx in range(len(rows) - 1, -1, -1):
        message = rows[idx]
        try:
            message_id = int(message.get("id") or 0)
        except Exception:
            message_id = 0
        if (
            _text(message.get("role")).lower() == "assistant"
            and 0 < message_id < int(before_message_id or 0)
        ):
            assistant_index = idx
            break
    if assistant_index < 0:
        return []

    user_message_id = 0
    for idx in range(assistant_index - 1, -1, -1):
        message = rows[idx]
        if _text(message.get("role")).lower() != "user":
            continue
        try:
            user_message_id = int(message.get("id") or 0)
        except Exception:
            user_message_id = 0
        if user_message_id > 0:
            break
    if user_message_id <= 0:
        return []

    refs_map = dict(refs_by_user or {})
    pack = _record(refs_map.get(user_message_id) or refs_map.get(str(user_message_id)))
    hits = [dict(hit) for hit in list(pack.get("hits") or []) if isinstance(hit, dict)]
    if not source_hints:
        return hits
    filtered, _ = filter_hits_to_source_hints(
        hits,
        [],
        source_hints,
        fallback_to_original=False,
    )
    return filtered


def build_answer_audit_scope_block(source_hints: list[str] | None) -> str:
    labels = [Path(_text(source)).name for source in list(source_hints or []) if _text(source)]
    source_lines = "\n".join(f"- {label}" for label in labels[:8])
    return (
        "FOLLOW-UP TASK: Audit the previous assistant answer.\n"
        "- Evaluate the previous answer as written; do not replace it with a new reading route or literature list.\n"
        "- Check requested counts, paper-title/evidence binding, relevance, and citation support.\n"
        "- State each mismatch directly and preserve correct parts of the previous answer.\n"
        "- Treat the previous answer's visible [n] links as renderer-owned source links. Do not audit internal "
        "citation offsets, marker syntax, DOC labels, or bibliography-number formatting unless the user explicitly asks.\n"
        "- In the user-facing audit, refer to each paper by title and cite its matching retrieved source; never expose "
        "DOC-k labels, offset markers, source paths, or retrieval diagnostics.\n"
        "- The authoritative source set below contains the papers actually selected by the previous answer. "
        "Treat every listed source as present in the library, do not add unlisted retrieval candidates to the count, "
        "and do not count a paper and its evidence hit as separate papers."
        + (
            f"\n- Authoritative previous-answer source count: {len(labels[:8])}."
            f"\n- Restrict evidence checks to these sources:\n{source_lines}"
            if source_lines
            else ""
        )
    )


def order_hits_by_source_hints(
    hits: list[dict] | None,
    source_hints: list[str] | None,
) -> list[dict]:
    candidates = [dict(hit) for hit in list(hits or []) if isinstance(hit, dict)]
    ordered: list[dict] = []
    used_sources: set[str] = set()
    for source_hint in list(source_hints or []):
        source_key = _text(source_hint).replace("/", "\\").lower()
        if not source_key or source_key in used_sources:
            continue
        matched = next(
            (
                hit
                for hit in candidates
                if _text(_record(hit.get("meta")).get("source_path"))
                .replace("/", "\\")
                .lower()
                == source_key
            ),
            None,
        )
        if not isinstance(matched, dict):
            continue
        used_sources.add(source_key)
        ordered.append(dict(matched))
    return ordered


def filter_hits_to_source_hints(
    hits: list[dict] | None,
    scores: list[float] | None,
    source_hints: list[str] | None,
    *,
    fallback_to_original: bool = True,
) -> tuple[list[dict], list[float]]:
    normalized_hints = {
        _text(value).replace("/", "\\").lower()
        for value in list(source_hints or [])
        if _text(value)
    }
    rows = list(hits or [])
    score_rows = list(scores or [])
    if not normalized_hints:
        return rows, score_rows

    kept_hits: list[dict] = []
    kept_scores: list[float] = []
    for idx, hit in enumerate(rows):
        meta = _record(hit.get("meta")) if isinstance(hit, dict) else {}
        source = _text(meta.get("source_path")).replace("/", "\\").lower()
        if source not in normalized_hints:
            continue
        kept_hits.append(dict(hit))
        if idx < len(score_rows):
            kept_scores.append(float(score_rows[idx]))
    if kept_hits:
        return kept_hits, kept_scores
    return (rows, score_rows) if fallback_to_original else ([], [])
