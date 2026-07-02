from __future__ import annotations

from typing import Any


SOURCE_SUMMARY_KINDS = {
    "local_kb",
    "local_plus_external",
    "external_not_kb",
    "general_api",
    "unknown",
}


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

    kind = _kind(
        source_blend=source_blend,
        source_policy=source_policy,
        answer_mode=answer_mode,
        evidence_hit_count=evidence_hit_count,
    )
    if kind not in SOURCE_SUMMARY_KINDS:
        kind = "unknown"
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
    confidence = _confidence(
        kind=kind,
        evidence_status=evidence_status,
        retrieval_confidence=retrieval_confidence,
        support_ratio=support_ratio,
        evidence_hit_count=evidence_hit_count,
        unsupported_claims=unsupported_claims,
    )

    return {
        "kind": kind,
        "label_key": label_key,
        "label": label,
        "detail": details[kind],
        "confidence": confidence,
        "source_blend": source_blend,
        "source_policy": source_policy,
        "evidence_status": evidence_status,
        "retrieval_confidence": retrieval_confidence,
        "support_ratio": support_ratio,
        "evidence_hit_count": evidence_hit_count,
        "unsupported_claims": unsupported_claims,
        "source_notice_count": source_notice_count,
        "quality_gate_status": quality_gate_status,
        "should_show": kind != "unknown",
    }
