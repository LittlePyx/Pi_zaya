from __future__ import annotations

import re

from kb.citation_marker_context import numeric_marker_is_content
from api.citation_display_registry import _positive_int


_MARKER = re.compile(r'\[(\d{1,4})\](?:\(#(kb-cite-[^\s)]+)(?:\s+"[^"]*")?\))?')


def has_bound_local_citation(quality: dict, contract: dict) -> bool:
    plan = quality.get("citation_plan")
    return (
        not contract
        and quality.get("has_hits") is True
        and isinstance(plan, dict)
        and any(
            isinstance(slot, dict)
            and slot.get("source_path")
            and len(str(slot.get("evidence_quote") or "")) >= 40
            and (slot.get("block_id") or slot.get("anchor_id"))
            for slot in plan.get("slots", [])
        )
    )


def reconcile_citation_markers(
    body: str, details: list[dict], *, source_numbers: set[int]
) -> str:
    """Remove unbound citation decoration; never invent evidence or alter math."""
    anchors = {
        str(row.get("anchor") or ""): _positive_int(row.get("num"))
        for row in details
        if isinstance(row, dict) and row.get("anchor")
    }

    def replace(match: re.Match[str]) -> str:
        if numeric_marker_is_content(body, match.start()):
            return match.group(0)
        number = int(match.group(1))
        anchor = match.group(2)
        if anchor:
            return match.group(0) if anchors.get(anchor) == number else ""
        if body[match.end() :].startswith("("):
            return match.group(0)  # External URLs/ordinary Markdown links.
        # Only generation's document numbers are protocol. Arrays, years and
        # unknown numeric content are not inferred to be library citations.
        return "" if number in source_numbers else match.group(0)

    return _MARKER.sub(replace, body)


def reconcile_blocked_message(message: dict) -> dict:
    """Keep blocked display/copy/cache projections free of dead local links."""
    meta = message.get("meta") or {}
    numbers = set(range(1, 1000)) if meta.get("canonical_hit_paths") else set()
    packet = (meta.get("paper_guide_contracts") or {}).get("render_packet") or {}
    for record in (message, packet):
        for key in (
            "content",
            "answer_markdown",
            "rendered_body",
            "rendered_content",
            "copy_markdown",
            "copy_text",
        ):
            if isinstance(record.get(key), str):
                record[key] = reconcile_citation_markers(
                    record[key], [], source_numbers=numbers
                )
    return message


def reconcile_render_projections(
    body: str,
    rendered: str,
    copy_md: str,
    copy_text: str,
    details: list[dict],
    *,
    source_numbers: set[int],
) -> tuple[str, str, str, str]:
    # Copy uses plain numeric citations; retain numbers that have real cards.
    copy_numbers = source_numbers - {
        _positive_int(row.get("num")) for row in details if isinstance(row, dict)
    }
    return (
        reconcile_citation_markers(body, details, source_numbers=source_numbers),
        reconcile_citation_markers(rendered, details, source_numbers=source_numbers),
        reconcile_citation_markers(copy_md, details, source_numbers=copy_numbers),
        reconcile_citation_markers(copy_text, details, source_numbers=copy_numbers),
    )


def reconcile_render_packet(packet: dict, *, local_answer: bool) -> dict:
    if not local_answer:
        return packet
    result = dict(packet)
    fields = ("rendered_body", "rendered_content", "copy_markdown", "copy_text")
    values = reconcile_render_projections(
        *(str(result.get(key) or "") for key in fields),
        list(result.get("cite_details") or []),
        source_numbers=set(range(1, 1000)),
    )
    result.update(zip(fields, values))
    return result
