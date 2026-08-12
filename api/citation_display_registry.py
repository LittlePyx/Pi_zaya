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
    out: dict[str, list[str]] = {}

    # A citation at the end of a paragraph conventionally supports the whole
    # paragraph. Preserve that scope only when every citation in the paragraph
    # resolves to one source; mixed-source comparisons continue to use the
    # sentence-local path below.
    for paragraph in re.split(r"\n\s*\n", surface):
        tokens = [token for token in token_sources if token in paragraph]
        source_keys = {
            token_sources[token]
            for token in tokens
            if token_sources.get(token)
        }
        if not tokens or len(source_keys) != 1:
            continue
        last_token = max(tokens, key=paragraph.rfind)
        token_end = paragraph.rfind(last_token) + len(last_token)
        suffix = paragraph[token_end:]
        if not re.fullmatch(
            r"[\s\]\[(){}*_`.,;:!?\u3002\uff01\uff1f\uff0c\uff1b\uff1a-]*",
            suffix,
        ):
            continue
        paragraph_claim = paragraph
        for token in tokens:
            paragraph_claim = paragraph_claim.replace(token, " ")
        paragraph_claim = re.sub(
            r"(?m)^\s*#{1,6}\s+[^\n]*(?:\n+|$)",
            "",
            paragraph_claim,
        )
        paragraph_claim = re.sub(
            r"^\s*(?:[-*+]\s+|\d+[.)]\s*)",
            "",
            paragraph_claim,
        )
        paragraph_claim = re.sub(r"[*_`>#]", " ", paragraph_claim)
        paragraph_claim = re.sub(r"\s+", " ", paragraph_claim).strip()
        paragraph_claim = re.sub(
            r"\s+([,.;:!?\u3002\uff0c\uff1b\uff1a\uff01\uff1f])",
            r"\1",
            paragraph_claim,
        )
        if 8 <= len(paragraph_claim) <= 900:
            claims = out.setdefault(next(iter(source_keys)), [])
            if paragraph_claim not in claims:
                claims.append(paragraph_claim)
    parts = re.split(
        r"(?<=[。！？!?；;\n])\s*|(?<=\.)\s+(?=(?:\*{1,2})?[A-Z\u4e00-\u9fff])",
        surface,
    )
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
        current_originals = set(_detail_original_numbers(current))
        if current_originals:
            same_occurrence = [
                row
                for row in candidates
                if current_originals & set(_detail_original_numbers(row))
            ]
            if same_occurrence:
                # Display compaction intentionally gives every passage from
                # one paper the same visible number.  Preserve the internal
                # answer-hit coordinate before using semantic similarity, or
                # a weak/filtered quote for occurrence 3 can be rebound to the
                # stronger quote belonging to occurrence 2.  Multiple rows
                # with the same original coordinate remain eligible for the
                # existing claim-aware passage selection.
                candidates = same_occurrence
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


def _exact_system_a_evidence_identity(detail: dict) -> tuple[str, str] | None:
    """Identify repeated cards for the exact same passage of one document.

    A paper may legitimately contribute several different passages, so source
    identity alone is deliberately insufficient here.  Renderer-produced
    evidence fingerprints (or their equivalent citation-budget key) are the
    narrow signal that two rows describe the same evidence occurrence.
    """

    if not _system_a_detail(detail):
        return None
    budget_key = str(detail.get("citation_budget_key") or "").strip().casefold()
    evidence_fingerprint = str(detail.get("evidence_fingerprint") or "").strip().casefold()
    evidence_surface = re.sub(
        r"\s+",
        " ",
        _detail_evidence_text(detail, focused=True),
    ).strip().casefold()

    # A plan-level budget key is shared by every passage assigned to that
    # citation slot.  It is therefore safe for deduplication only together
    # with the actual evidence text; otherwise a paper's benefit and risk
    # passages can incorrectly collapse into one display card.
    if budget_key and evidence_surface:
        evidence_digest = hashlib.sha1(
            evidence_surface.encode("utf-8", "ignore")
        ).hexdigest()[:20]
        fingerprint = f"{budget_key}|evidence:{evidence_digest}"
    else:
        fingerprint = evidence_fingerprint or budget_key
    source_key = system_a_source_key(detail)
    if not source_key or len(fingerprint) < 8:
        return None
    return source_key, fingerprint


def _dedupe_exact_system_a_evidence_cards(
    markdown: str,
    rows: list[dict],
) -> tuple[str, list[dict]]:
    """Collapse only exact-evidence duplicates and preserve every answer link."""

    text = str(markdown or "")
    groups: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        identity = _exact_system_a_evidence_identity(row)
        if identity is not None:
            groups.setdefault(identity, []).append(index)

    replacements: dict[int, dict] = {}
    skipped: set[int] = set()
    for indexes in groups.values():
        if len(indexes) < 2:
            continue

        def _quality(index: int) -> tuple[int, int, float, int, int]:
            row = rows[index]
            claim = str(row.get("answer_claim") or "").strip()
            evidence = _detail_evidence_text(row, focused=True)
            try:
                confidence = float(row.get("binding_confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            anchor = str(row.get("anchor") or "").strip()
            position = _link_position(text, anchor)
            return (
                1 if claim else 0,
                1 if str(row.get("card_evidence") or "").strip() else 0,
                confidence,
                len(evidence),
                -(position if position >= 0 else 10**12),
            )

        canonical_index = max(indexes, key=_quality)
        canonical = dict(rows[canonical_index])
        canonical_anchor = str(canonical.get("anchor") or "").strip()

        merged_claims: list[str] = []
        merged_original_numbers: list[int] = []
        for index in indexes:
            row = rows[index]
            for raw_claim in [row.get("answer_claim"), *list(row.get("answer_claims") or [])]:
                claim = str(raw_claim or "").strip()
                if claim and claim not in merged_claims:
                    merged_claims.append(claim)
            for number in _detail_original_numbers(row):
                if number not in merged_original_numbers:
                    merged_original_numbers.append(number)
            alias = str(row.get("anchor") or "").strip()
            if canonical_anchor and alias and alias != canonical_anchor:
                text = re.sub(
                    rf"\]\(#{re.escape(alias)}(?=(?:\s|\)))",
                    f"](#{canonical_anchor}",
                    text,
                )

        if merged_claims:
            canonical["answer_claims"] = merged_claims
            if not str(canonical.get("answer_claim") or "").strip():
                canonical["answer_claim"] = merged_claims[0]
        if merged_original_numbers:
            canonical["answer_hit_linked_nums"] = merged_original_numbers

        first_index = min(indexes)
        replacements[first_index] = canonical
        skipped.update(indexes)
        skipped.discard(first_index)

    if not replacements:
        return text, rows
    deduped: list[dict] = []
    for index, row in enumerate(rows):
        if index in replacements:
            deduped.append(replacements[index])
        elif index not in skipped:
            deduped.append(row)
    return text, deduped


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

    text, remapped = _dedupe_exact_system_a_evidence_cards(text, remapped)

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
