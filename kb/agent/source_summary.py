from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit


SOURCE_SUMMARY_KINDS = {
    "local_kb",
    "local_plus_external",
    "external_not_kb",
    "general_api",
    "unknown",
}

SOURCE_POLICY_PAYLOAD_SCHEMA_VERSION = 2


def _record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _ratio(value: Any) -> float:
    try:
        n = float(value or 0.0)
    except Exception:
        return 0.0
    return round(max(0.0, min(1.0, n)), 4)


def _confidence(
    *,
    kind: str,
    evidence_status: str,
    retrieval_confidence: str,
    support_ratio: float,
    evidence_hit_count: int,
    unsupported_claims: int,
) -> str:
    if kind in {"external_not_kb", "general_api"}:
        return "external"
    if evidence_status in {"insufficient", "not_applicable"}:
        return "low"
    if retrieval_confidence == "low" or unsupported_claims > 0:
        return "low"
    if evidence_status == "grounded" and support_ratio >= 0.8 and evidence_hit_count > 0:
        return "high"
    return "medium"


def _kind(*, source_blend: str, source_policy: str, answer_mode: str, evidence_hit_count: int) -> str:
    if source_blend == "hybrid_local_external" or source_policy == "local_plus_external_background":
        return "local_plus_external"
    if source_blend == "external_academic" or answer_mode == "external_academic_llm":
        return "external_not_kb"
    if source_blend == "general_llm" or answer_mode == "general_llm":
        return "general_api"
    if source_blend == "local_grounded" or source_policy in {"local_only", "trusted_sites_only"} or evidence_hit_count > 0:
        return "local_kb"
    return "unknown"


def _policy_flags(kind: str, *, external_source_kind: str = "") -> dict[str, bool]:
    uses_local = kind in {"local_kb", "local_plus_external"}
    uses_external = kind in {"local_plus_external", "external_not_kb", "general_api"}
    uses_external_web = bool(uses_external and external_source_kind == "web")
    uses_external_model = bool(uses_external and not uses_external_web)
    requires_notice = kind in {"local_plus_external", "external_not_kb"}
    return {
        "uses_local_knowledge_base": uses_local,
        "uses_external_model": uses_external_model,
        "uses_external_web": uses_external_web,
        "requires_user_notice": requires_notice,
    }


def _generation_output(trace: dict[str, Any]) -> dict[str, Any]:
    for raw_step in list(trace.get("steps") or []):
        step = _record(raw_step)
        if str(step.get("tool") or "").strip() != "generate_grounded_answer":
            continue
        output = _record(step.get("output"))
        if output:
            return output
    return {}


def _safe_web_url(value: object) -> str:
    raw = str(value or "").replace("\x00", "").strip()
    if not raw or len(raw) > 2400:
        return ""
    try:
        parsed = urlsplit(raw)
    except Exception:
        return ""
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    if parsed.username or parsed.password:
        return ""
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, parsed.path, parsed.query, ""))


def _web_sources(value: object) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in list(value or []) if isinstance(value, list) else []:
        item = _record(raw)
        url = _safe_web_url(item.get("url"))
        if not url or url in seen:
            continue
        seen.add(url)
        title = str(item.get("title") or "").replace("\x00", " ").strip()[:500]
        try:
            domain = str(urlsplit(url).hostname or "").lower()
        except Exception:
            domain = ""
        out.append({"title": title, "url": url, "domain": domain})
        if len(out) >= 8:
            break
    return out


def _notice_kind(kind: str) -> str:
    if kind == "local_plus_external":
        return "local_plus_external"
    if kind == "external_not_kb":
        return "external_not_kb"
    return "none"


def _citation_policy(kind: str) -> str:
    if kind in {"local_kb", "local_plus_external"}:
        return "local_citations_required"
    return "not_applicable"


def _build_source_policy_payload(
    *,
    kind: str,
    label_key: str,
    label: str,
    detail: str,
    confidence: str,
    source_blend: str,
    source_policy: str,
    answer_mode: str,
    evidence_status: str,
    retrieval_confidence: str,
    support_ratio: float,
    evidence_hit_count: int,
    unsupported_claims: int,
    source_notice_count: int,
    quality_gate_status: str,
    should_show: bool,
    external_supplement_mode: str = "disabled",
    external_source_kind: str = "",
    web_search_used: bool = False,
    web_sources: list[dict[str, str]] | None = None,
    external_supplement_unavailable: str = "",
) -> dict[str, Any]:
    flags = _policy_flags(kind, external_source_kind=external_source_kind)
    sources = list(web_sources or [])[:8]
    return {
        "schema_version": SOURCE_POLICY_PAYLOAD_SCHEMA_VERSION,
        "kind": kind,
        "source_blend": source_blend,
        "source_policy": source_policy,
        "answer_mode": answer_mode,
        "evidence_status": evidence_status,
        "retrieval_confidence": retrieval_confidence,
        "confidence": confidence,
        "uses_local_knowledge_base": flags["uses_local_knowledge_base"],
        "uses_external_model": flags["uses_external_model"],
        "uses_external_web": flags["uses_external_web"],
        "requires_user_notice": flags["requires_user_notice"],
        "external_supplement_mode": external_supplement_mode,
        "external_source_kind": external_source_kind,
        "web_search_used": bool(web_search_used),
        "web_source_count": len(sources),
        "web_sources": sources,
        "external_supplement_unavailable": external_supplement_unavailable,
        "notice_kind": _notice_kind(kind),
        "citation_policy": _citation_policy(kind),
        "badge": {
            "label_key": label_key,
            "label": label,
            "detail": detail,
            "should_show": should_show,
        },
        "support": {
            "support_ratio": support_ratio,
            "evidence_hit_count": evidence_hit_count,
            "unsupported_claims": unsupported_claims,
            "source_notice_count": source_notice_count,
            "quality_gate_status": quality_gate_status,
        },
    }


def build_agent_source_summary(agent_trace: dict | None) -> dict[str, Any]:
    """Build a compact source badge payload for normal chat UI.

    The full agent trace remains available for audit/debug surfaces. This
    summary intentionally contains only source class and lightweight support
    signals so the main answer stays uncluttered.
    """

    trace = _record(agent_trace)
    if not trace:
        return {}
    summary = _record(trace.get("summary"))
    context = _record(trace.get("context"))
    verification = _record(trace.get("verification"))
    research_run = _record(trace.get("research_run"))
    generation_output = _generation_output(trace)

    source_policy = _text(summary.get("source_policy"), research_run.get("source_policy"), context.get("source_policy"))
    evidence_status = _text(summary.get("evidence_status"), verification.get("evidence_status"))
    source_blend = _text(summary.get("answer_source_blend"), context.get("answer_source_blend"))
    answer_mode = _text(summary.get("answer_mode"), context.get("answer_mode"))
    retrieval_confidence = _text(summary.get("retrieval_confidence"), context.get("retrieval_confidence"))
    support_ratio = _ratio(summary.get("support_ratio") if "support_ratio" in summary else verification.get("support_ratio"))
    evidence_hit_count = _int(
        summary.get("evidence_hit_count")
        if "evidence_hit_count" in summary
        else verification.get("evidence_hit_count")
    )
    unsupported_claims = _int(
        summary.get("unsupported_claims")
        if "unsupported_claims" in summary
        else verification.get("unsupported_claims")
    )
    source_notice_count = _int(
        summary.get("source_notice_count")
        if "source_notice_count" in summary
        else verification.get("source_notice_count")
    )
    quality_gate_status = _text(summary.get("quality_gate_status"))
    external_supplement_mode = _text(
        context.get("external_supplement_mode"),
        _record(_record(trace.get("context")).get("evidence_gate")).get("external_supplement_mode"),
        "disabled",
    ).lower()
    if external_supplement_mode not in {"disabled", "model", "web"}:
        external_supplement_mode = "disabled"
    web_search_used = bool(generation_output.get("web_search_used") or context.get("web_search_used"))
    web_sources = _web_sources(generation_output.get("web_citations")) if web_search_used else []
    external_supplement_unavailable = _text(
        generation_output.get("external_supplement_unavailable"),
        context.get("external_supplement_unavailable"),
    )

    kind = _kind(
        source_blend=source_blend,
        source_policy=source_policy,
        answer_mode=answer_mode,
        evidence_hit_count=evidence_hit_count,
    )
    if kind not in SOURCE_SUMMARY_KINDS:
        kind = "unknown"
    external_source_kind = ""
    if kind in {"local_plus_external", "external_not_kb", "general_api"}:
        if web_search_used:
            external_source_kind = "web"
        elif external_supplement_mode == "model":
            external_source_kind = "model"
    labels = {
        "local_kb": ("agent_trace_source_local_only", "Local KB"),
        "local_plus_external": ("agent_trace_source_local_external", "Local + external"),
        "external_not_kb": ("agent_trace_evidence_not_from_kb", "Not from KB"),
        "general_api": ("agent_trace_evidence_not_from_kb", "Not from KB"),
        "unknown": ("agent_trace_source_fallback", "Source"),
    }
    details = {
        "local_kb": "Answer uses local knowledge-base evidence.",
        "local_plus_external": "Local citations come from the knowledge base; uncited background may use external model context.",
        "external_not_kb": "No matching local knowledge-base evidence was used; answer comes from an external model/API.",
        "general_api": "This answer does not use local knowledge-base evidence.",
        "unknown": "",
    }
    label_key, label = labels[kind]
    detail = details[kind]
    if external_source_kind == "web":
        if kind == "local_plus_external":
            label_key, label = "agent_trace_source_local_web", "Local + web"
            detail = "Local citations come from the knowledge base; linked web sources provide separately identified background."
        else:
            label_key, label = "agent_trace_source_web", "Web sources"
            detail = "No matching local evidence was used; this answer comes from linked web sources."
    elif external_source_kind == "model":
        if kind == "local_plus_external":
            label_key, label = "agent_trace_source_local_model", "Local + model background"
            detail = "Local citations come from the knowledge base; uncited background comes from model knowledge and is not verifiable evidence."
        else:
            label_key, label = "agent_trace_source_model", "Model background"
            detail = "No matching local evidence was used; this is model background, not verifiable library evidence."
    confidence = _confidence(
        kind=kind,
        evidence_status=evidence_status,
        retrieval_confidence=retrieval_confidence,
        support_ratio=support_ratio,
        evidence_hit_count=evidence_hit_count,
        unsupported_claims=unsupported_claims,
    )
    should_show = kind != "unknown"
    source_policy_payload = _build_source_policy_payload(
        kind=kind,
        label_key=label_key,
        label=label,
        detail=detail,
        confidence=confidence,
        source_blend=source_blend,
        source_policy=source_policy,
        answer_mode=answer_mode,
        evidence_status=evidence_status,
        retrieval_confidence=retrieval_confidence,
        support_ratio=support_ratio,
        evidence_hit_count=evidence_hit_count,
        unsupported_claims=unsupported_claims,
        source_notice_count=source_notice_count,
        quality_gate_status=quality_gate_status,
        should_show=should_show,
        external_supplement_mode=external_supplement_mode,
        external_source_kind=external_source_kind,
        web_search_used=web_search_used,
        web_sources=web_sources,
        external_supplement_unavailable=external_supplement_unavailable,
    )

    return {
        "kind": kind,
        "label_key": label_key,
        "label": label,
        "detail": detail,
        "confidence": confidence,
        "source_blend": source_blend,
        "source_policy": source_policy,
        "source_policy_payload": source_policy_payload,
        "evidence_status": evidence_status,
        "retrieval_confidence": retrieval_confidence,
        "support_ratio": support_ratio,
        "evidence_hit_count": evidence_hit_count,
        "unsupported_claims": unsupported_claims,
        "source_notice_count": source_notice_count,
        "quality_gate_status": quality_gate_status,
        "should_show": should_show,
        "external_supplement_mode": external_supplement_mode,
        "external_source_kind": external_source_kind,
        "web_search_used": web_search_used,
        "web_sources": web_sources,
        "external_supplement_unavailable": external_supplement_unavailable,
    }
