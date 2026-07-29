from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
import re
from typing import Any

from kb.evidence_term_mapping import evidence_alignment_tokens


_ANSWER_CITATION_LINK_RE = re.compile(
    r'\[(\d{1,4})\]\(\#([^\s)]+)(?:\s+"[^"]*")?\)'
)


def _positive_int(value: Any) -> int:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


@lru_cache(maxsize=4096)
def _canonical_source_path_identity(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    normalized = raw.replace("\\", "/").casefold()
    try:
        # Public API paths use ``kb-source/<root-id>/...`` while generation
        # uses an absolute local path. Resolve the public form when possible so
        # both representations keep one identity without falling back to an
        # ambiguous two-segment filename tail.
        from api.reference_ui import _resolve_source_md_path

        resolved = _resolve_source_md_path(raw)
    except Exception:
        resolved = None
    if isinstance(resolved, Path):
        try:
            normalized = str(resolved.resolve(strict=False)).replace("\\", "/").casefold()
        except Exception:
            normalized = str(resolved).replace("\\", "/").casefold()
    elif not normalized.startswith("kb-source/"):
        try:
            normalized = str(Path(raw).expanduser().resolve(strict=False)).replace(
                "\\", "/"
            ).casefold()
        except Exception:
            pass
    return re.sub(r"/+", "/", normalized).rstrip("/")


def system_a_source_key(detail: dict | None) -> str:
    """Return a stable document key for one rendered System-A citation."""

    row = detail if isinstance(detail, dict) else {}
    source_path = str(row.get("source_path") or row.get("sourcePath") or "").strip()
    if source_path:
        identity = _canonical_source_path_identity(source_path)
        if identity:
            digest = hashlib.sha1(identity.encode("utf-8", "ignore")).hexdigest()[:20]
            return f"path:{digest}"
    source_name = str(row.get("source_name") or row.get("sourceName") or "").strip().casefold()
    if not source_name:
        return ""
    name = Path(source_name).name or source_name
    return re.sub(r"(?i)(?:\.en)?\.md$|\.pdf$", "", name).strip()


def _system_a_detail(detail: dict | None) -> bool:
    row = detail if isinstance(detail, dict) else {}
    return (
        str(row.get("citation_route") or "").strip().lower() == "system_a"
        and bool(system_a_source_key(row))
    )


def _link_position(markdown: str, anchor: str) -> int:
    anchor_text = str(anchor or "").strip()
    if not anchor_text:
        return -1
    return str(markdown or "").find(f"](#{anchor_text}")


def _detail_original_numbers(detail: dict) -> list[int]:
    traced_values = [
        detail.get("answer_hit_num"),
        detail.get("original_num"),
        *list(detail.get("answer_hit_linked_nums") or []),
    ]
    fallback_values = [
        detail.get("num"),
        *list(detail.get("linked_nums") or []),
    ]
    values = traced_values if any(_positive_int(value) for value in traced_values) else fallback_values
    out: list[int] = []
    for value in values:
        number = _positive_int(value)
        if number > 0 and number not in out:
            out.append(number)
    return out


def _linked_answer_claims_by_source(markdown: str, rows: list[dict]) -> dict[str, list[str]]:
    anchor_sources = {
        str(row.get("anchor") or "").strip(): system_a_source_key(row)
        for row in rows
        if _system_a_detail(row) and str(row.get("anchor") or "").strip()
    }
    token_sources: dict[str, str] = {}

    def _placeholder(match: re.Match[str]) -> str:
        source_key = anchor_sources.get(str(match.group(2) or "").strip(), "")
        if not source_key:
            return " "
        token = f"KB_CITATION_OCCURRENCE_{len(token_sources)}"
        token_sources[token] = source_key
        return f" {token} "

    surface = _ANSWER_CITATION_LINK_RE.sub(_placeholder, str(markdown or ""))
    surface = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", surface)
    surface = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", surface)
    parts = re.split(
        r"(?<=[。！？!?；;\n])\s*|(?<=\.)\s+(?=(?:\*{1,2})?[A-Z\u4e00-\u9fff])",
        surface,
    )
    out: dict[str, list[str]] = {}
    for part in parts:
        tokens = [token for token in token_sources if token in part]
        if not tokens:
            continue
        claim = part
        for token in tokens:
            claim = claim.replace(token, " ")
        claim = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)、]\s*)", "", claim)
        claim = re.sub(r"[*_`>#]", " ", claim)
        claim = re.sub(r"\s+", " ", claim).strip()
        claim = re.sub(r"\s+([,.;:!?，。；：！？])", r"\1", claim)
        if len(claim) < 8:
            continue
        for source_key in dict.fromkeys(token_sources[token] for token in tokens):
            claims = out.setdefault(source_key, [])
            if claim not in claims:
                claims.append(claim)
    return out


def _citation_claim_context(markdown: str, start: int, end: int) -> str:
    text = str(markdown or "")
    left = max((text.rfind(mark, 0, start) for mark in ("。", "！", "？", "!", "?", ";", "；", "\n", ".")), default=-1)
    right_candidates = [
        position
        for mark in ("。", "！", "？", "!", "?", ";", "；", "\n", ".")
        if (position := text.find(mark, end)) >= 0
    ]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    claim = text[left + 1 : right]
    claim = _ANSWER_CITATION_LINK_RE.sub(" ", claim)
    claim = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", claim)
    claim = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", claim)
    claim = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)、]\s*)", "", claim)
    claim = re.sub(r"[*_`>#]", " ", claim)
    return re.sub(r"\s+", " ", claim).strip()


def _detail_evidence_text(detail: dict, *, focused: bool) -> str:
    fields = (
        ("evidence_quote", "card_evidence", "summary_line")
        if focused
        else (
            "evidence_quote",
            "card_evidence",
            "raw",
            "summary_line",
            "heading_path",
        )
    )
    return " ".join(str(detail.get(field) or "").strip() for field in fields).strip()


def _rebind_system_a_occurrence_anchors(markdown: str, rows: list[dict]) -> str:
    """Bind each repeated same-paper citation to its best matching passage."""

    text = str(markdown or "")
    anchor_rows = {
        str(row.get("anchor") or "").strip(): row
        for row in rows
        if _system_a_detail(row) and str(row.get("anchor") or "").strip()
    }
    candidates_by_source: dict[str, list[dict]] = {}
    for row in anchor_rows.values():
        candidates_by_source.setdefault(system_a_source_key(row), []).append(row)

    matches = list(_ANSWER_CITATION_LINK_RE.finditer(text))
    if not matches:
        return text
    pieces: list[str] = []
    cursor = 0
    for match in matches:
        pieces.append(text[cursor : match.start()])
        surface = match.group(0)
        current_anchor = str(match.group(2) or "").strip()
        current = anchor_rows.get(current_anchor)
        if current is None:
            pieces.append(surface)
            cursor = match.end()
            continue
        candidates = candidates_by_source.get(system_a_source_key(current), [])
        visible_num = _positive_int(match.group(1))
        candidates = [
            row
            for row in candidates
            if _positive_int(row.get("num")) == visible_num
            and str(row.get("anchor") or "").strip()
        ]
        if len(candidates) <= 1:
            pieces.append(surface)
            cursor = match.end()
            continue
        claim_tokens = evidence_alignment_tokens(
            _citation_claim_context(text, match.start(), match.end())
        )
        if not claim_tokens:
            pieces.append(surface)
            cursor = match.end()
            continue

        def _score(row: dict) -> tuple[int, int, int, int]:
            focused_tokens = evidence_alignment_tokens(_detail_evidence_text(row, focused=True))
            broad_tokens = evidence_alignment_tokens(_detail_evidence_text(row, focused=False))
            anchor = str(row.get("anchor") or "").strip()
            return (
                len(claim_tokens & focused_tokens),
                len(claim_tokens & broad_tokens),
                1 if anchor == current_anchor else 0,
                -len(focused_tokens),
            )

        best = max(candidates, key=_score)
        best_anchor = str(best.get("anchor") or "").strip()
        if best_anchor and best_anchor != current_anchor:
            surface = surface.replace(f"#{current_anchor}", f"#{best_anchor}", 1)
        pieces.append(surface)
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces)


def remap_system_a_citations_for_display(
    markdown: str,
    cite_details: list[dict] | None,
) -> tuple[str, list[dict], list[dict]]:
    """Make visible System-A numbers contiguous by cited document.

    Retrieval-hit numbers are an internal generation coordinate. Reference
    cards are grouped by document, so exposing those raw coordinates can leave
    an answer with ``[4]`` while the matching card is labelled ``#1``. This
    function runs only after every link has already been grounded. It changes
    the visible label and public detail number while preserving the original
    hit number and exact anchor for traceability. System-B bibliography numbers
    are deliberately left untouched.
    """

    text = str(markdown or "")
    rows = [dict(item) for item in list(cite_details or []) if isinstance(item, dict)]
    eligible: list[tuple[int, int, str, dict]] = []
    for index, row in enumerate(rows):
        if not _system_a_detail(row):
            continue
        anchor = str(row.get("anchor") or "").strip()
        position = _link_position(text, anchor)
        eligible.append((position, index, system_a_source_key(row), row))
    if not eligible:
        return text, rows, []

    eligible.sort(
        key=lambda item: (
            item[0] if item[0] >= 0 else 10**12,
            _positive_int(item[3].get("num")) or 10**9,
            item[1],
        )
    )
    # System-B numbers are bibliography coordinates and must stay unchanged.
    # Reserve them before compacting System-A retrieval-hit numbers; otherwise
    # a System-A source can become visible ``[1]`` beside a different System-B
    # reference that is already ``[1]``.  Besides confusing the reader, that
    # violates the packet's one-number/one-source invariant and forces every
    # subsequent poll to rebuild the same packet.
    reserved_display_numbers: set[int] = set()
    for row in rows:
        if _system_a_detail(row):
            continue
        for value in [row.get("num"), *list(row.get("linked_nums") or [])]:
            number = _positive_int(value)
            if number > 0:
                reserved_display_numbers.add(number)

    display_by_source: dict[str, int] = {}
    registry_by_source: dict[str, dict] = {}
    next_display_num = 1
    for _position, _index, source_key, row in eligible:
        if source_key not in display_by_source:
            while next_display_num in reserved_display_numbers:
                next_display_num += 1
            display_by_source[source_key] = next_display_num
            next_display_num += 1
            registry_by_source[source_key] = {
                "display_num": display_by_source[source_key],
                "source_key": source_key,
                "source_path": str(row.get("source_path") or row.get("sourcePath") or "").strip(),
                "source_name": str(row.get("source_name") or row.get("sourceName") or "").strip(),
                "original_nums": [],
            }
        original_numbers = _detail_original_numbers(row)
        registry_numbers = registry_by_source[source_key]["original_nums"]
        for number in original_numbers:
            if number not in registry_numbers:
                registry_numbers.append(number)

    remapped: list[dict] = []
    for row in rows:
        if not _system_a_detail(row):
            remapped.append(row)
            continue
        source_key = system_a_source_key(row)
        display_num = int(display_by_source[source_key])
        original_numbers = _detail_original_numbers(row)
        original_num = _positive_int(row.get("answer_hit_num")) or _positive_int(row.get("original_num"))
        if original_num <= 0:
            original_num = _positive_int(row.get("num"))
        next_row = dict(row)
        if original_num > 0:
            next_row.setdefault("answer_hit_num", original_num)
            next_row.setdefault("original_num", original_num)
        if original_numbers:
            next_row["answer_hit_linked_nums"] = original_numbers
        next_row["display_num"] = display_num
        next_row["num"] = display_num
        next_row["linked_nums"] = [display_num]
        anchor = str(next_row.get("anchor") or "").strip()
        if anchor:
            text = re.sub(
                rf"\[\d{{1,4}}\](?=\(\#{re.escape(anchor)}(?:\s|\)))",
                f"[{display_num}]",
                text,
            )
        remapped.append(next_row)

    known_anchors = {
        str(row.get("anchor") or "").strip()
        for row in remapped
        if str(row.get("anchor") or "").strip()
    }
    # Citation rendering can deduplicate detail rows by document while the
    # final Markdown still contains occurrence-level anchors.  Collapse those
    # links onto the canonical card anchor.  A render packet then has one
    # source/number/anchor identity, while every occurrence remains clickable.
    for match in re.finditer(
        r'\[(\d{1,4})\]\(\#([^\s)]+)(?:\s+"([^"]*)")?\)',
        text,
    ):
        linked_num = _positive_int(match.group(1))
        anchor = str(match.group(2) or "").strip()
        title = str(match.group(3) or "").strip().lower()
        if (
            not anchor
            or anchor in known_anchors
            or anchor.startswith("kb-cite-reader-")
            or "source:" not in title
        ):
            continue
        original_number_candidates = [
            row
            for row in remapped
            if _system_a_detail(row)
            and linked_num in _detail_original_numbers(row)
        ]
        candidates = original_number_candidates or [
            row
            for row in remapped
            if _system_a_detail(row)
            and _positive_int(row.get("num")) == linked_num
        ]
        source_keys = {system_a_source_key(row) for row in candidates if system_a_source_key(row)}
        if len(source_keys) != 1 or not candidates:
            continue
        display_num = _positive_int(candidates[0].get("num"))
        canonical_anchor = str(candidates[0].get("anchor") or "").strip()
        if display_num <= 0 or not canonical_anchor:
            continue
        text = re.sub(
            rf'\[{linked_num}\]\(\#{re.escape(anchor)}(?P<tail>(?:\s+"[^"]*")?\))',
            lambda occurrence: (
                f"[{display_num}](#{canonical_anchor}{occurrence.group('tail')}"
            ),
            text,
            count=1,
        )

    text = _rebind_system_a_occurrence_anchors(text, remapped)

    # Keep per-occurrence evidence, but give every card for the same paper the
    # complete set of answer claims attributed to that paper.  This prevents a
    # source-level claim/evidence contract from being split accidentally across
    # two anchors while preserving the more precise passage shown on click.
    claims_by_source = _linked_answer_claims_by_source(text, remapped)
    for row in remapped:
        if not _system_a_detail(row):
            continue
        source_claims = claims_by_source.setdefault(system_a_source_key(row), [])
        for raw_claim in [row.get("answer_claim"), *list(row.get("answer_claims") or [])]:
            claim = str(raw_claim or "").strip()
            if claim and claim not in source_claims:
                source_claims.append(claim)
    for row in remapped:
        if not _system_a_detail(row):
            continue
        source_claims = claims_by_source.get(system_a_source_key(row)) or []
        if source_claims:
            row["answer_claims"] = list(source_claims)

    registry = sorted(
        registry_by_source.values(),
        key=lambda item: int(item.get("display_num") or 0),
    )
    return text, remapped, registry


__all__ = [
    "remap_system_a_citations_for_display",
    "system_a_source_key",
]
