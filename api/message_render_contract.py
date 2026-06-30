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
