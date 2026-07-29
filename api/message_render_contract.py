from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from api.citation_display_registry import _canonical_source_path_identity

_VISIBLE_SINGLE_NUMERIC_CITE_RE = re.compile(r"(?<!\[)\[(\d{1,4})\](?![\]\(])")
_LINKED_NUMERIC_CITE_RE = re.compile(
    r"(?<![!\\])\[(\d{1,4})\]\(\#([^\s)]+)(?:\s+\"[^\"\r\n]*\")?\)"
)
_STRUCTURED_CITE_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*\d{1,4}\s*\]\]",
    re.IGNORECASE,
)

LEGACY_RENDER_FIELDS = (
    "notice",
    "rendered_body",
    "rendered_content",
    "copy_markdown",
    "copy_text",
)


def _dict_list(raw: object) -> list[dict]:
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _dict_or_empty(raw: object) -> dict:
    return dict(raw) if isinstance(raw, dict) else {}


def transform_markdown_outside_code(text: str, transform) -> str:
    """Apply ``transform`` only to ordinary Markdown prose.

    Citation syntax is never meaningful inside fenced or inline code.  Keeping
    those spans byte-for-byte intact also prevents a renderer from silently
    turning an array literal such as ``[1, 2]`` into citation UI.
    """

    raw = str(text or "")
    if not raw:
        return raw

    def _transform_inline(line: str) -> str:
        pieces: list[str] = []
        cursor = 0
        while cursor < len(line):
            tick = line.find("`", cursor)
            if tick < 0:
                pieces.append(str(transform(line[cursor:])))
                break
            pieces.append(str(transform(line[cursor:tick])))
            run_end = tick
            while run_end < len(line) and line[run_end] == "`":
                run_end += 1
            marker = line[tick:run_end]
            close = line.find(marker, run_end)
            if close < 0:
                # An unmatched delimiter is still code-like user content.  It
                # is safer to preserve it than to interpret brackets inside it.
                pieces.append(line[tick:])
                break
            close_end = close + len(marker)
            pieces.append(line[tick:close_end])
            cursor = close_end
        return "".join(pieces)

    out: list[str] = []
    fence_char = ""
    fence_len = 0
    for line in raw.splitlines(keepends=True):
        fence_match = re.match(r"^[ \t]{0,3}(`{3,}|~{3,})", line)
        if fence_char:
            out.append(line)
            if fence_match:
                marker = str(fence_match.group(1) or "")
                if marker.startswith(fence_char) and len(marker) >= fence_len:
                    fence_char = ""
                    fence_len = 0
            continue
        if fence_match:
            marker = str(fence_match.group(1) or "")
            fence_char = marker[:1]
            fence_len = len(marker)
            out.append(line)
            continue
        out.append(_transform_inline(line))
    return "".join(out)


def markdown_without_code(text: str) -> str:
    return _mask_markdown_code(text)


def _mask_markdown_code(text: str) -> str:
    """Return prose while replacing code spans with whitespace of equal shape."""

    raw = str(text or "")
    protected: list[str] = []

    def _capture_code_preserving_newlines(value: str) -> str:
        return "".join("\n" if char == "\n" else " " for char in value)

    # Reuse the scanner by transforming prose into itself, then explicitly
    # capture code through a small mirrored scan.  This keeps citation regexes
    # from observing any brackets inside code while preserving line offsets.
    cursor = 0
    fence_char = ""
    fence_len = 0
    for line in raw.splitlines(keepends=True):
        fence_match = re.match(r"^[ \t]{0,3}(`{3,}|~{3,})", line)
        if fence_char:
            protected.append(_capture_code_preserving_newlines(line))
            if fence_match:
                marker = str(fence_match.group(1) or "")
                if marker.startswith(fence_char) and len(marker) >= fence_len:
                    fence_char = ""
                    fence_len = 0
            cursor += len(line)
            continue
        if fence_match:
            marker = str(fence_match.group(1) or "")
            fence_char = marker[:1]
            fence_len = len(marker)
            protected.append(_capture_code_preserving_newlines(line))
            cursor += len(line)
            continue

        line_out: list[str] = []
        line_cursor = 0
        while line_cursor < len(line):
            tick = line.find("`", line_cursor)
            if tick < 0:
                line_out.append(line[line_cursor:])
                break
            line_out.append(line[line_cursor:tick])
            run_end = tick
            while run_end < len(line) and line[run_end] == "`":
                run_end += 1
            marker = line[tick:run_end]
            close = line.find(marker, run_end)
            if close < 0:
                line_out.append(_capture_code_preserving_newlines(line[tick:]))
                break
            close_end = close + len(marker)
            line_out.append(_capture_code_preserving_newlines(line[tick:close_end]))
            line_cursor = close_end
        protected.append("".join(line_out))
        cursor += len(line)
    if cursor < len(raw):
        protected.append(raw[cursor:])
    return "".join(protected)


@dataclass(frozen=True)
class MessageRenderPayload:
    notice: str = ""
    rendered_body: str = ""
    rendered_content: str = ""
    copy_markdown: str = ""
    copy_text: str = ""
    cite_details: list[dict] = field(default_factory=list)
    refs_user_msg_id: int = 0
    render_packet: dict = field(default_factory=dict)

    @classmethod
    def from_cache(cls, cache: dict | None) -> MessageRenderPayload | None:
        if not isinstance(cache, dict):
            return None
        render_packet = _dict_or_empty(cache.get("render_packet"))
        authoritative = render_packet if render_packet else cache
        return cls(
            notice=str(authoritative.get("notice") or ""),
            rendered_body=str(authoritative.get("rendered_body") or ""),
            rendered_content=str(authoritative.get("rendered_content") or ""),
            copy_markdown=str(authoritative.get("copy_markdown") or ""),
            copy_text=str(authoritative.get("copy_text") or ""),
            cite_details=_dict_list(authoritative.get("cite_details")),
            refs_user_msg_id=int(
                authoritative.get("refs_user_msg_id")
                or cache.get("refs_user_msg_id")
                or 0
            ),
            render_packet=render_packet,
        )

    @classmethod
    def from_record(cls, rec: dict, *, render_packet: dict | None = None) -> MessageRenderPayload:
        packet = _dict_or_empty(render_packet)
        authoritative = packet if packet else rec
        return cls(
            notice=str(authoritative.get("notice") or ""),
            rendered_body=str(authoritative.get("rendered_body") or ""),
            rendered_content=str(authoritative.get("rendered_content") or ""),
            copy_markdown=str(authoritative.get("copy_markdown") or ""),
            copy_text=str(authoritative.get("copy_text") or ""),
            cite_details=_dict_list(authoritative.get("cite_details")),
            refs_user_msg_id=int(
                authoritative.get("refs_user_msg_id")
                or rec.get("refs_user_msg_id")
                or 0
            ),
            render_packet=packet,
        )

    @classmethod
    def from_render_packet(cls, render_packet: dict | None) -> MessageRenderPayload | None:
        packet = _dict_or_empty(render_packet)
        if not packet:
            return None
        return cls(
            notice=str(packet.get("notice") or ""),
            rendered_body=str(packet.get("rendered_body") or ""),
            rendered_content=str(packet.get("rendered_content") or ""),
            copy_markdown=str(packet.get("copy_markdown") or ""),
            copy_text=str(packet.get("copy_text") or ""),
            cite_details=_dict_list(packet.get("cite_details")),
            render_packet=packet,
        )

    def as_dict(self) -> dict:
        return {
            "notice": self.notice,
            "rendered_body": self.rendered_body,
            "rendered_content": self.rendered_content,
            "copy_markdown": self.copy_markdown,
            "copy_text": self.copy_text,
            "cite_details": _dict_list(self.cite_details),
            "refs_user_msg_id": int(self.refs_user_msg_id or 0),
            "render_packet": _dict_or_empty(self.render_packet),
        }

    def as_legacy_projection(self) -> dict:
        return {
            "notice": self.notice,
            "rendered_body": self.rendered_body,
            "rendered_content": self.rendered_content,
            "copy_markdown": self.copy_markdown,
            "copy_text": self.copy_text,
            "cite_details": _dict_list(self.cite_details),
        }

    def as_cache_payload(
        self,
        *,
        schema: int,
        cache_key: str,
        answer_sig: str = "",
        input_ref_sig: str = "",
        citation_plan_sig: str = "",
        locale: str = "",
    ) -> dict:
        packet = _dict_or_empty(self.render_packet)
        if not packet:
            packet = {
                "notice": self.notice,
                "rendered_body": self.rendered_body,
                "rendered_content": self.rendered_content,
                "copy_markdown": self.copy_markdown,
                "copy_text": self.copy_text,
                "cite_details": _dict_list(self.cite_details),
            }
        packet["schema"] = int(schema or 0)
        packet["answer_sig"] = str(answer_sig or packet.get("answer_sig") or "")
        packet["input_ref_sig"] = str(
            input_ref_sig or packet.get("input_ref_sig") or ""
        )
        packet["citation_plan_sig"] = str(
            citation_plan_sig or packet.get("citation_plan_sig") or ""
        )
        packet["locale"] = str(locale or packet.get("locale") or "").strip().lower()
        authoritative = MessageRenderPayload.from_render_packet(packet) or self
        return {
            "schema": int(schema or 0),
            "cache_key": str(cache_key or ""),
            "answer_sig": str(packet.get("answer_sig") or ""),
            "input_ref_sig": str(packet.get("input_ref_sig") or ""),
            "citation_plan_sig": str(packet.get("citation_plan_sig") or ""),
            "locale": str(packet.get("locale") or ""),
            "notice": authoritative.notice,
            "rendered_body": authoritative.rendered_body,
            "rendered_content": authoritative.rendered_content,
            "copy_markdown": authoritative.copy_markdown,
            "copy_text": authoritative.copy_text,
            "cite_details": _dict_list(authoritative.cite_details),
            "refs_user_msg_id": int(self.refs_user_msg_id or 0),
            "render_packet": packet,
        }


def iter_numeric_citation_numbers(text: str) -> list[int]:
    nums: list[int] = []
    prose = markdown_without_code(str(text or ""))
    for match in _VISIBLE_SINGLE_NUMERIC_CITE_RE.finditer(prose):
        try:
            n = int(match.group(1) or 0)
        except (TypeError, ValueError):
            continue
        # A bracketed publication year is content, never an answer-reference
        # marker.  Multi-value arrays/ranges do not match the single-marker
        # expression above.
        if n > 0 and not 1800 <= n <= 2100:
            nums.append(n)
    return nums


def count_linkable_source_hits(hits: list[dict] | None) -> int:
    count = 0
    for hit in list(hits or []):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        if str(meta.get("source_path") or "").strip():
            count += 1
    return count


def _source_cite_id(source_path: str) -> str:
    raw = str(source_path or "").strip()
    if not raw:
        return ""
    return "s" + hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:8]


def _linkable_source_sids(hits: list[dict] | None) -> set[str]:
    sids: set[str] = set()
    for hit in list(hits or []):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        sid = _source_cite_id(str(meta.get("source_path") or "").strip()).lower()
        if sid:
            sids.add(sid)
    return sids


def _linkable_hit_numbers(hits: list[dict] | None) -> set[int]:
    numbers: set[int] = set()
    ordinal = 0
    for hit in list(hits or []):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        if not str(meta.get("source_path") or "").strip():
            continue
        ordinal += 1
        try:
            explicit = int(meta.get("ref_answer_citation_num") or 0)
        except (TypeError, ValueError):
            explicit = 0
        numbers.add(explicit if explicit > 0 else ordinal)
    return numbers


def content_has_linkable_answer_citations(content: str, hits: list[dict] | None) -> bool:
    raw = markdown_without_code(str(content or ""))
    if not raw or "[" not in raw:
        return False
    linkable_numbers = _linkable_hit_numbers(hits)
    if not linkable_numbers:
        return False
    structured_sids = {str(match.group(1) or "").strip().lower() for match in _STRUCTURED_CITE_RE.finditer(raw)}
    if structured_sids and structured_sids.intersection(_linkable_source_sids(hits)):
        return True
    return any(int(n) in linkable_numbers for n in iter_numeric_citation_numbers(raw))


def _normalized_source_path(value: object) -> str:
    raw = str(value or "").strip().replace("\\", "/").casefold()
    return re.sub(r"/+", "/", raw)


def _source_tail(value: object) -> str:
    path = _normalized_source_path(value)
    parts = [part for part in path.split("/") if part]
    return "/".join(parts[-2:]) if len(parts) >= 2 else path


def _system_b_explicit_source_sid(record: dict) -> str:
    for key in ("sid", "source_sid", "sourceSid", "source_id", "sourceId"):
        value = str(record.get(key) or "").strip().casefold()
        if value:
            return value
    discovered: set[str] = set()
    for example in list(record.get("candidate_cite_examples") or []):
        discovered.update(
            str(match.group(1) or "").strip().casefold()
            for match in re.finditer(
                r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*\d{1,4}\s*\]\]",
                str(example or ""),
                flags=re.IGNORECASE,
            )
            if str(match.group(1) or "").strip()
        )
    return next(iter(discovered)) if len(discovered) == 1 else ""


def _system_b_source_path(record: dict) -> str:
    return _normalized_source_path(
        record.get("source_path")
        or record.get("sourcePath")
        or record.get("source_name")
        or record.get("sourceName")
    )


def _source_path_is_absolute(path: str) -> bool:
    value = str(path or "")
    return bool(value.startswith("/") or re.match(r"^[a-z]:/", value))


def _system_b_source_records_match(left: dict, right: dict) -> bool:
    left_sid = _system_b_explicit_source_sid(left)
    right_sid = _system_b_explicit_source_sid(right)
    if left_sid and right_sid:
        return left_sid == right_sid

    left_path = _system_b_source_path(left)
    right_path = _system_b_source_path(right)
    if not left_path or not right_path:
        return False
    if left_path == right_path:
        return True

    left_public = left_path.startswith("kb-source/")
    right_public = right_path.startswith("kb-source/")
    if left_public or right_public:
        left_canonical = _canonical_source_path_identity(left_path)
        right_canonical = _canonical_source_path_identity(right_path)
        return bool(
            left_canonical
            and right_canonical
            and left_canonical == right_canonical
        )

    left_absolute = _source_path_is_absolute(left_path)
    right_absolute = _source_path_is_absolute(right_path)
    if left_absolute != right_absolute:
        absolute_path = left_path if left_absolute else right_path
        relative_path = right_path if left_absolute else left_path
        relative_parts = [part for part in relative_path.split("/") if part]
        if len(relative_parts) >= 2 and absolute_path.endswith(f"/{relative_path}"):
            return True

    return False


def _system_b_source_coordinate(record: dict) -> str:
    sid = _system_b_explicit_source_sid(record)
    if sid:
        return f"sid:{sid}"
    path = _system_b_source_path(record)
    return f"path:{path}" if path else ""


def _citation_detail_source_identity(detail: dict) -> str:
    source_path = _normalized_source_path(detail.get("source_path"))
    route = str(detail.get("citation_route") or "").strip().casefold()
    is_system_b = bool(detail.get("is_inpaper")) or route == "system_b"
    if is_system_b:
        doi = str(detail.get("doi") or "").strip().casefold()
        if doi:
            return f"doi:{doi}"
        reference_identity = " ".join(
            str(detail.get(key) or "").strip().casefold()
            for key in ("title", "raw", "cite_fmt")
            if str(detail.get(key) or "").strip()
        )
        try:
            ref_num = int(detail.get("inpaper_ref_num") or detail.get("ref_num") or 0)
        except (TypeError, ValueError):
            ref_num = 0
        if source_path and (reference_identity or ref_num > 0):
            return f"system_b:{source_path}:{ref_num}:{reference_identity}"
    return f"source:{source_path}" if source_path else ""


def _detail_matches_hits(detail: dict, hits: list[dict] | None) -> bool:
    if hits is None:
        return True
    detail_tail = _source_tail(detail.get("source_path"))
    if not detail_tail:
        return False
    hit_tails = {
        _source_tail(
            (hit.get("meta") if isinstance(hit.get("meta"), dict) else {}).get(
                "source_path"
            )
        )
        for hit in list(hits or [])
        if isinstance(hit, dict)
    }
    return detail_tail in hit_tails


def render_payload_has_citation_links(
    payload: MessageRenderPayload | dict | None,
    *,
    hits: list[dict] | None = None,
) -> bool:
    normalized = payload
    if isinstance(payload, dict):
        normalized = MessageRenderPayload.from_cache(payload)
    if not isinstance(normalized, MessageRenderPayload):
        return False
    details = _dict_list(normalized.cite_details)
    surface = str(normalized.rendered_body or normalized.rendered_content or "")
    links = [
        (int(match.group(1)), str(match.group(2) or "").strip().lstrip("#"))
        for match in _LINKED_NUMERIC_CITE_RE.finditer(markdown_without_code(surface))
        if str(match.group(2) or "").strip().lower().startswith("kb-cite-")
    ]
    if not details or not links:
        return False

    details_by_anchor: dict[str, dict] = {}
    num_to_identity: dict[int, str] = {}
    identity_to_num: dict[str, int] = {}
    for detail in details:
        anchor = str(detail.get("anchor") or "").strip().lstrip("#")
        try:
            num = int(detail.get("num") or 0)
        except (TypeError, ValueError):
            return False
        identity = _citation_detail_source_identity(detail)
        if not anchor or num <= 0 or not identity or not _detail_matches_hits(detail, hits):
            return False
        if anchor in details_by_anchor:
            return False
        if num in num_to_identity and num_to_identity[num] != identity:
            return False
        if identity in identity_to_num and identity_to_num[identity] != num:
            return False
        details_by_anchor[anchor] = detail
        num_to_identity[num] = identity
        identity_to_num[identity] = num

    linked_anchors: set[str] = set()
    for label_num, anchor in links:
        detail = details_by_anchor.get(anchor)
        if not isinstance(detail, dict):
            return False
        try:
            detail_num = int(detail.get("num") or 0)
        except (TypeError, ValueError):
            return False
        if label_num <= 0 or detail_num != label_num:
            return False
        linked_anchors.add(anchor)

    # No empty cards and no orphan anchors in either direction.
    return linked_anchors == set(details_by_anchor)


def render_payload_is_degraded_for_citations(
    payload: MessageRenderPayload | dict | None,
    *,
    raw_content: str,
    hits: list[dict] | None,
) -> bool:
    if not content_has_linkable_answer_citations(raw_content, hits):
        return False
    return not render_payload_has_citation_links(payload, hits=hits)


def render_payload_is_missing_planned_system_a(
    payload: MessageRenderPayload | dict | None,
    *,
    citation_plan: dict | None,
) -> bool:
    """Reject a cached render that dropped an authorized System-A plan.

    The raw model answer may legitimately contain no numeric marker because the
    renderer binds typed System-A slots after generation.  The older degraded
    check only considered markers already present in raw text, so a prematurely
    cached marker-free packet could bypass that binding forever.
    """

    plan = _dict_or_empty(citation_plan)
    budget = _dict_or_empty(plan.get("budget"))
    try:
        system_a_budget = int(budget.get("system_a") or 0)
    except (TypeError, ValueError):
        system_a_budget = 0
    system_a_slots = [
        item
        for item in list(plan.get("slots") or [])
        if isinstance(item, dict)
        and str(item.get("preferred_system") or "").strip().lower() != "system_b"
        and str(item.get("source_path") or item.get("sourcePath") or "").strip()
        and str(item.get("evidence_quote") or "").strip()
    ]
    # Plans may retain ranked fallback candidates after the number of
    # System-A citations the answer is actually allowed to use.  Treating
    # every fallback as mandatory makes a complete cached packet look stale
    # forever (and forces the full evidence renderer to run on every message
    # poll).  Slot order is the plan's ranking, so only the authorized budget
    # is part of the cache-completeness contract.
    if system_a_budget > 0:
        system_a_slots = system_a_slots[:system_a_budget]
    has_system_a_slot = bool(system_a_slots)
    if system_a_budget <= 0 or not has_system_a_slot:
        return False
    normalized = payload
    if isinstance(payload, dict):
        normalized = MessageRenderPayload.from_cache(payload)
    if not isinstance(normalized, MessageRenderPayload):
        return True
    details = list(normalized.cite_details or [])
    details.extend(_dict_list(_dict_or_empty(normalized.render_packet).get("cite_details")))
    system_a_details = [
        item
        for item in details
        if isinstance(item, dict)
        and str(item.get("citation_route") or "").strip().lower() == "system_a"
    ]
    if not system_a_details:
        return True

    def _source_key(value: object) -> str:
        normalized_path = str(value or "").strip().replace("\\", "/").casefold()
        parts = [part for part in normalized_path.split("/") if part]
        return "/".join(parts[-2:]) if len(parts) >= 2 else normalized_path

    def _slot_identity(slot: dict) -> tuple[str, str]:
        return (
            _source_key(slot.get("source_path") or slot.get("sourcePath")),
            str(slot.get("heading_path") or slot.get("headingPath") or "").strip().casefold(),
        )

    def _has_structured_benchmark_pairs(slot: dict) -> bool:
        evidence = str(slot.get("evidence_quote") or "")
        return len(
            re.findall(
                r"(?:^|[:,;])\s*[A-Za-z][A-Za-z0-9 +()_-]{0,48}?"
                r"(?:\s*\[\d{1,4}\])?\s*=\s*-?\d+\.\d+",
                evidence,
                flags=re.I,
            )
        ) >= 2

    structured_identities = {
        _slot_identity(slot)
        for slot in system_a_slots
        if _has_structured_benchmark_pairs(slot)
    }
    if structured_identities:
        system_a_slots = [
            slot
            for slot in system_a_slots
            if _slot_identity(slot) not in structured_identities
            or _has_structured_benchmark_pairs(slot)
        ]

    def _terms(value: object) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9+.-]{2,}|[\u4e00-\u9fff]{2,8}", str(value or ""))
            if token.casefold() not in {"the", "and", "for", "with", "from", "table", "paper"}
        }

    def _matches_planned_evidence(detail: dict, slot: dict) -> bool:
        if _source_key(detail.get("source_path")) != _source_key(
            slot.get("source_path") or slot.get("sourcePath")
        ):
            return False
        detail_heading = str(
            detail.get("heading_path") or detail.get("card_locator") or detail.get("location_label") or ""
        ).strip().casefold()
        plan_heading = str(slot.get("heading_path") or slot.get("headingPath") or "").strip().casefold()
        detail_evidence = str(
            detail.get("evidence_quote")
            or detail.get("card_evidence")
            or detail.get("summary_line")
            or detail.get("raw")
            or ""
        ).strip()
        if not detail_heading and not detail_evidence:
            # Preserve compatibility with old, source-only cache records. They
            # cannot be audited for passage quality but are not known-bad.
            return True
        heading_match = bool(
            plan_heading
            and detail_heading
            and (plan_heading in detail_heading or detail_heading in plan_heading)
        )
        if plan_heading and detail_heading and not heading_match:
            return False
        plan_evidence = str(slot.get("evidence_quote") or "")
        detail_terms = _terms(detail_evidence)
        plan_terms = _terms(plan_evidence)
        metric_match = re.search(r"\b(PSNR|SSIM|LPIPS|FID|FPS)\b", plan_evidence, flags=re.I)
        benchmark_pairs = [
            (str(method or "").strip(), float(value))
            for method, value in re.findall(
                r"(?:^|[:,;])\s*([A-Za-z][A-Za-z0-9 +()_-]{0,48}?)"
                r"(?:\s*\[\d{1,4}\])?\s*=\s*(-?\d+\.\d+)",
                plan_evidence,
                flags=re.I,
            )
        ]
        if metric_match and len(benchmark_pairs) >= 2:
            metric_name = str(metric_match.group(1) or "").upper()
            extreme = (
                min(value for _method, value in benchmark_pairs)
                if metric_name in {"LPIPS", "FID"}
                else max(value for _method, value in benchmark_pairs)
            )
            tied_methods = [
                method
                for method, value in benchmark_pairs
                if abs(value - extreme) < 1e-9
            ]
            for method in tied_methods:
                method_terms = _terms(method).difference({"ours", "method", "model"})
                if method_terms and not method_terms.intersection(detail_terms):
                    return False
        plan_numbers = set(re.findall(r"(?<![\w.])-?\d+\.\d+(?![\w.])", plan_evidence))
        detail_numbers = set(re.findall(r"(?<![\w.])-?\d+\.\d+(?![\w.])", detail_evidence))
        numeric_match = bool(plan_numbers and detail_numbers and plan_numbers.intersection(detail_numbers))
        term_overlap = plan_terms.intersection(detail_terms)
        evidence_match = bool(numeric_match and term_overlap) if plan_numbers else len(term_overlap) >= 2
        answer_claim = str(detail.get("answer_claim") or detail.get("card_claim") or "")
        shared_claim_terms = _terms(answer_claim).intersection(plan_terms)
        shared_claim_terms.difference_update(
            {
                "simple", "baselines", "image", "imaging", "restoration", "paper",
                "results", "result", "table", "experiments", "applications",
            }
        )
        if shared_claim_terms:
            claim_coverage = len(shared_claim_terms.intersection(detail_terms)) / len(shared_claim_terms)
            if claim_coverage < 0.85:
                return False
        return heading_match or evidence_match

    slots_by_source: dict[str, list[dict]] = {}
    for slot in system_a_slots:
        source_key = _source_key(slot.get("source_path") or slot.get("sourcePath"))
        if source_key:
            slots_by_source.setdefault(source_key, []).append(slot)
    # A single paper can legitimately contribute several passages.  Do not let
    # a weak cached occurrence survive merely because a different claim from
    # the same paper happens to match another plan slot.  When a citation claim
    # strongly identifies one passage, that occurrence must match its best
    # claim-aligned slot before the whole packet can be reused.
    for detail in system_a_details:
        source_slots = slots_by_source.get(_source_key(detail.get("source_path"))) or []
        claim_terms = _terms(detail.get("answer_claim") or detail.get("card_claim") or "")
        if not source_slots or not claim_terms:
            continue
        ranked_slots = sorted(
            (
                (len(claim_terms.intersection(_terms(slot.get("evidence_quote") or ""))), slot)
                for slot in source_slots
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        best_overlap, best_slot = ranked_slots[0]
        if best_overlap >= 3 and not _matches_planned_evidence(detail, best_slot):
            return True
    if len(slots_by_source) >= 2:
        # A multi-paper route is complete only when every selected paper keeps
        # an answer-relevant passage.  The former any-match check let one good
        # card preserve an otherwise stale cache containing weak passages for
        # the remaining papers.
        return any(
            not any(
                _matches_planned_evidence(detail, slot)
                for detail in system_a_details
                for slot in source_slots
            )
            for source_slots in slots_by_source.values()
        )
    return not any(
        _matches_planned_evidence(detail, slot)
        for detail in system_a_details
        for slot in system_a_slots
    )


def render_payload_is_missing_planned_system_b(
    payload: MessageRenderPayload | dict | None,
    *,
    citation_plan: dict | None,
) -> bool:
    """Reject a cached render that dropped an authorized System-B slot.

    System-B markers are resolved after answer generation, so the raw answer
    signature alone cannot prove that a cached packet still contains the
    planned upstream-reference card.  Count only the authorized slots and
    accept legacy details that identify the route through ``is_inpaper``.
    """

    plan = _dict_or_empty(citation_plan)
    budget = _dict_or_empty(plan.get("budget"))
    try:
        system_b_budget = int(budget.get("system_b") or 0)
    except (TypeError, ValueError):
        system_b_budget = 0
    raw_system_b_slots = [
        item
        for item in list(plan.get("slots") or [])
        if isinstance(item, dict)
        and str(item.get("preferred_system") or "").strip().lower() == "system_b"
        and (
            list(item.get("candidate_refs") or [])
            or list(item.get("candidate_cite_examples") or [])
            or str(item.get("evidence_quote") or "").strip()
        )
    ]

    def _planned_ref_numbers(slot: dict) -> set[int]:
        values = [slot.get("ref_num"), *list(slot.get("candidate_refs") or [])]
        for example in list(slot.get("candidate_cite_examples") or []):
            values.extend(
                match.group(1)
                for match in re.finditer(
                    r"\[\[\s*CITE\s*:\s*[A-Za-z0-9_-]{4,24}\s*:\s*(\d{1,4})\s*\]\]",
                    str(example or ""),
                    flags=re.IGNORECASE,
                )
            )
        numbers: set[int] = set()
        for value in values:
            try:
                number = int(value or 0)
            except (TypeError, ValueError):
                continue
            if number > 0:
                numbers.add(number)
        return numbers

    # The builder already deduplicates System-B opportunities, but historical
    # plans may contain the same source/reference coordinate more than once.
    # Treat that as one cache obligation rather than forcing an impossible
    # duplicate-card count on every message poll.
    system_b_slots: list[dict] = []
    seen_slot_coordinates: set[tuple[str, tuple[int, ...]]] = set()
    for slot in raw_system_b_slots:
        source_coordinate = _system_b_source_coordinate(slot)
        ref_numbers = tuple(sorted(_planned_ref_numbers(slot)))
        coordinate = (source_coordinate, ref_numbers)
        if coordinate in seen_slot_coordinates:
            continue
        seen_slot_coordinates.add(coordinate)
        system_b_slots.append(slot)

    system_b_slots = system_b_slots[: max(0, system_b_budget)]
    required_count = len(system_b_slots)
    if required_count <= 0:
        return False

    normalized = payload
    if isinstance(payload, dict):
        normalized = MessageRenderPayload.from_cache(payload)
    if not isinstance(normalized, MessageRenderPayload):
        return True
    details = list(normalized.cite_details or [])

    system_b_details: list[dict] = []
    for detail in details:
        if not isinstance(detail, dict):
            continue
        route = str(detail.get("citation_route") or "").strip().lower()
        if route != "system_b" and detail.get("is_inpaper") is not True:
            continue
        system_b_details.append(detail)

    unmatched_detail_indexes = set(range(len(system_b_details)))
    for slot in system_b_slots:
        planned_source = _system_b_source_coordinate(slot)
        planned_refs = _planned_ref_numbers(slot)
        if not planned_source or not planned_refs:
            return True
        matched_index = -1
        for index in sorted(unmatched_detail_indexes):
            detail = system_b_details[index]
            try:
                detail_ref = int(
                    detail.get("inpaper_ref_num")
                    or detail.get("ref_num")
                    or detail.get("num")
                    or 0
                )
            except (TypeError, ValueError):
                detail_ref = 0
            if (
                _system_b_source_records_match(slot, detail)
                and detail_ref in planned_refs
            ):
                matched_index = index
                break
        if matched_index < 0:
            return True
        unmatched_detail_indexes.discard(matched_index)
    return False


def normalize_render_cache_payload(
    cache: dict | None,
    *,
    schema: int,
    expected_key: str,
) -> MessageRenderPayload | None:
    if not isinstance(cache, dict):
        return None
    if int(cache.get("schema") or 0) != int(schema or 0):
        return None
    if str(cache.get("cache_key") or "").strip() != str(expected_key or "").strip():
        return None
    return MessageRenderPayload.from_cache(cache)


def build_render_cache_payload(
    *,
    schema: int,
    cache_key: str,
    notice: str,
    rendered_body: str,
    rendered_content: str,
    copy_markdown: str,
    copy_text: str,
    cite_details: list[dict],
    refs_user_msg_id: int,
    render_packet: dict | None = None,
    answer_sig: str = "",
    input_ref_sig: str = "",
    citation_plan_sig: str = "",
    locale: str = "",
) -> dict:
    return MessageRenderPayload(
        notice=str(notice or ""),
        rendered_body=str(rendered_body or ""),
        rendered_content=str(rendered_content or ""),
        copy_markdown=str(copy_markdown or ""),
        copy_text=str(copy_text or ""),
        cite_details=_dict_list(cite_details),
        refs_user_msg_id=int(refs_user_msg_id or 0),
        render_packet=_dict_or_empty(render_packet),
    ).as_cache_payload(
        schema=schema,
        cache_key=cache_key,
        answer_sig=answer_sig,
        input_ref_sig=input_ref_sig,
        citation_plan_sig=citation_plan_sig,
        locale=locale,
    )


def project_render_packet_to_record(rec: dict, render_packet: dict | None) -> bool:
    payload = MessageRenderPayload.from_render_packet(render_packet)
    if payload is None:
        return False
    rec.update(payload.as_legacy_projection())
    return True


def strip_legacy_render_fields(rec: dict) -> None:
    for key in LEGACY_RENDER_FIELDS:
        rec.pop(key, None)
