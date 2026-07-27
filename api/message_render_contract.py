from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_VISIBLE_NUMERIC_CITE_RE = re.compile(r"\[\d{1,4}(?:\s*(?:-|\u2013|\u2014|,)\s*\d{1,4})*\]")
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
        cite_details = _dict_list(cache.get("cite_details"))
        packet_cite_details = _dict_list(render_packet.get("cite_details"))
        if not cite_details:
            cite_details = packet_cite_details
        return cls(
            notice=str(cache.get("notice") or render_packet.get("notice") or ""),
            rendered_body=str(cache.get("rendered_body") or render_packet.get("rendered_body") or ""),
            rendered_content=str(cache.get("rendered_content") or render_packet.get("rendered_content") or ""),
            copy_markdown=str(cache.get("copy_markdown") or render_packet.get("copy_markdown") or ""),
            copy_text=str(cache.get("copy_text") or render_packet.get("copy_text") or ""),
            cite_details=cite_details,
            refs_user_msg_id=int(cache.get("refs_user_msg_id") or 0),
            render_packet=render_packet,
        )

    @classmethod
    def from_record(cls, rec: dict, *, render_packet: dict | None = None) -> MessageRenderPayload:
        packet = _dict_or_empty(render_packet)
        return cls(
            notice=str(rec.get("notice") or packet.get("notice") or ""),
            rendered_body=str(rec.get("rendered_body") or packet.get("rendered_body") or ""),
            rendered_content=str(rec.get("rendered_content") or packet.get("rendered_content") or ""),
            copy_markdown=str(rec.get("copy_markdown") or packet.get("copy_markdown") or ""),
            copy_text=str(rec.get("copy_text") or packet.get("copy_text") or ""),
            cite_details=_dict_list(rec.get("cite_details")) or _dict_list(packet.get("cite_details")),
            refs_user_msg_id=int(rec.get("refs_user_msg_id") or 0),
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

    def as_cache_payload(self, *, schema: int, cache_key: str) -> dict:
        return {
            "schema": int(schema or 0),
            "cache_key": str(cache_key or ""),
            "notice": self.notice,
            "rendered_body": self.rendered_body,
            "rendered_content": self.rendered_content,
            "copy_markdown": self.copy_markdown,
            "copy_text": self.copy_text,
            "cite_details": _dict_list(self.cite_details),
            "refs_user_msg_id": int(self.refs_user_msg_id or 0),
            "render_packet": _dict_or_empty(self.render_packet),
        }


def iter_numeric_citation_numbers(text: str) -> list[int]:
    nums: list[int] = []
    for match in _VISIBLE_NUMERIC_CITE_RE.finditer(str(text or "")):
        for raw in re.findall(r"\d{1,4}", str(match.group(0) or "")):
            try:
                n = int(raw)
            except Exception:
                continue
            if n > 0:
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


def content_has_linkable_answer_citations(content: str, hits: list[dict] | None) -> bool:
    raw = str(content or "")
    if not raw or "[" not in raw:
        return False
    hit_count = count_linkable_source_hits(hits)
    if hit_count <= 0:
        return False
    structured_sids = {str(match.group(1) or "").strip().lower() for match in _STRUCTURED_CITE_RE.finditer(raw)}
    if structured_sids and structured_sids.intersection(_linkable_source_sids(hits)):
        return True
    return any(1 <= int(n) <= hit_count for n in iter_numeric_citation_numbers(raw))


def render_payload_has_citation_links(payload: MessageRenderPayload | dict | None) -> bool:
    normalized = payload
    if isinstance(payload, dict):
        normalized = MessageRenderPayload.from_cache(payload)
    if not isinstance(normalized, MessageRenderPayload):
        return False
    if any(isinstance(item, dict) for item in normalized.cite_details):
        return True
    render_packet = _dict_or_empty(normalized.render_packet)
    if any(isinstance(item, dict) for item in _dict_list(render_packet.get("cite_details"))):
        return True
    for key in ("rendered_content", "rendered_body", "copy_markdown"):
        if "#kb-cite-" in str(getattr(normalized, key) or ""):
            return True
        if "#kb-cite-" in str(render_packet.get(key) or ""):
            return True
    return False


def render_payload_is_degraded_for_citations(
    payload: MessageRenderPayload | dict | None,
    *,
    raw_content: str,
    hits: list[dict] | None,
) -> bool:
    if not content_has_linkable_answer_citations(raw_content, hits):
        return False
    return not render_payload_has_citation_links(payload)


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
    ).as_cache_payload(schema=schema, cache_key=cache_key)


def project_render_packet_to_record(rec: dict, render_packet: dict | None) -> bool:
    payload = MessageRenderPayload.from_render_packet(render_packet)
    if payload is None:
        return False
    rec.update(payload.as_legacy_projection())
    return True


def strip_legacy_render_fields(rec: dict) -> None:
    for key in LEGACY_RENDER_FIELDS:
        rec.pop(key, None)
