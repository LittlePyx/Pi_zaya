from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SYSTEM_B_AUDIT_CONTRACT_VERSION = 1


def _text(value: Any) -> str:
    return str(value or "").strip()


def _norm(value: Any) -> str:
    return _text(value).lower()


def _floatish(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    if out != out:
        return 0.0
    return out


def _boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _norm(value)
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [_text(item) for item in value if _text(item)]
    text = _text(value)
    return [text] if text else []


def _route(detail: Mapping[str, Any]) -> str:
    explicit = _norm(detail.get("citation_route") or detail.get("route"))
    if explicit in {"system_b", "system-b", "b"}:
        return "system_b"
    if explicit in {"system_a", "system-a", "a"}:
        return "system_a"
    return "system_b" if bool(detail.get("is_inpaper")) else "system_a"


def _inc(counter: dict[str, int], key: str) -> None:
    k = _text(key) or "unknown"
    counter[k] = int(counter.get(k) or 0) + 1


def _example(detail: Mapping[str, Any], *, flags: list[str], reason: str) -> dict[str, Any]:
    try:
        num = int(detail.get("num") or detail.get("ref_num") or 0)
    except Exception:
        num = 0
    title = _text(detail.get("card_title") or detail.get("title") or detail.get("raw") or detail.get("cite_fmt"))
    out: dict[str, Any] = {
        "num": num,
        "routing_reason": _text(detail.get("routing_reason")),
        "trace_score": round(_floatish(detail.get("system_b_trace_score")), 3),
        "flags": flags[:6],
    }
    if title:
        out["title"] = title[:180]
    if reason:
        out["reason"] = reason[:220]
    return out


def summarize_system_b_citation_audit(
    details: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] | None,
    *,
    max_examples: int = 5,
) -> dict[str, Any]:
    """Summarize visible System B citation-chain quality for diagnostics."""

    visible_count = 0
    system_b_total = 0
    trace_complete_count = 0
    trace_incomplete_count = 0
    needs_review_count = 0
    answer_context_only_count = 0
    source_markdown_count = 0
    reference_index_fallback_count = 0
    structured_cite_count = 0
    weak_context_count = 0
    low_trace_score_count = 0
    by_routing_reason: dict[str, int] = {}
    by_context_source: dict[str, int] = {}
    by_trace_source: dict[str, int] = {}
    by_trace_flag: dict[str, int] = {}
    examples: list[dict[str, Any]] = []

    for raw in list(details or []):
        if not isinstance(raw, Mapping):
            continue
        visible_count += 1
        if _route(raw) != "system_b":
            continue
        system_b_total += 1
        routing_reason = _text(raw.get("routing_reason"))
        context_source = _text(raw.get("citation_context_source") or raw.get("evidence_source"))
        trace_source = _text(raw.get("system_b_trace_source"))
        trace_score = _floatish(raw.get("system_b_trace_score"))
        trace_complete = _boolish(raw.get("system_b_trace_complete"))
        flag_set = set(_string_list(raw.get("system_b_trace_flags")) + _string_list(raw.get("card_quality_flags")))
        flags = sorted(flag_set)
        reason = _text(raw.get("system_b_trace_reason") or raw.get("card_warning"))

        _inc(by_routing_reason, routing_reason)
        _inc(by_context_source, context_source)
        _inc(by_trace_source, trace_source or context_source or routing_reason)
        for flag in flags:
            _inc(by_trace_flag, flag)

        if routing_reason == "structured_cite":
            structured_cite_count += 1
        if routing_reason == "reference_index_fallback":
            reference_index_fallback_count += 1
        if context_source == "answer_context" or trace_source == "answer_context" or "answer_context_only" in flags:
            answer_context_only_count += 1
        if context_source == "source_markdown" or trace_source == "source_markdown":
            source_markdown_count += 1
        if flag_set & {"weak_citation_context", "missing_citation_context", "reference_entry_only"}:
            weak_context_count += 1

        if trace_complete is True:
            trace_complete_count += 1
        else:
            trace_incomplete_count += 1
        if trace_score and trace_score < 0.5:
            low_trace_score_count += 1
        if trace_complete is not True or (trace_score and trace_score < 0.5):
            needs_review_count += 1
            if len(examples) < max(0, int(max_examples)):
                examples.append(_example(raw, flags=flags, reason=reason))

    complete_rate = round(trace_complete_count / system_b_total, 3) if system_b_total else 1.0
    return {
        "audit_contract_version": SYSTEM_B_AUDIT_CONTRACT_VERSION,
        "ok": needs_review_count == 0,
        "visible_count": visible_count,
        "system_b_total": system_b_total,
        "trace_complete_count": trace_complete_count,
        "trace_incomplete_count": trace_incomplete_count,
        "needs_review_count": needs_review_count,
        "complete_rate": complete_rate,
        "structured_cite_count": structured_cite_count,
        "reference_index_fallback_count": reference_index_fallback_count,
        "answer_context_only_count": answer_context_only_count,
        "source_markdown_count": source_markdown_count,
        "weak_context_count": weak_context_count,
        "low_trace_score_count": low_trace_score_count,
        "by_routing_reason": by_routing_reason,
        "by_context_source": by_context_source,
        "by_trace_source": by_trace_source,
        "by_trace_flag": by_trace_flag,
        "review_examples": examples,
    }
