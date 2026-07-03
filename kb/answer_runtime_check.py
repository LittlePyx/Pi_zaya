from __future__ import annotations

import re
from typing import Any

from kb.answer_presentation import clean_assistant_answer_presentation_text


ANSWER_RUNTIME_CHECK_SCHEMA_VERSION = 1

_SOURCE_BLENDS = {
    "local_grounded",
    "hybrid_local_external",
    "external_academic",
    "general_llm",
}

_ANSWER_MODE_TO_SOURCE_BLEND = {
    "evidence_grounded": "local_grounded",
    "hybrid_local_external": "hybrid_local_external",
    "external_academic_llm": "external_academic",
    "general_llm": "general_llm",
}

_PROFILE_CONTRACTS = {
    "local_evidence_grounded": {
        "source_blends": {"local_grounded"},
        "summary_kind": "local_kb",
        "max_notice_lines": 0,
        "requires_notice": False,
    },
    "hybrid_synthesis": {
        "source_blends": {"hybrid_local_external"},
        "summary_kind": "local_plus_external",
        "max_notice_lines": 1,
        "requires_notice": True,
    },
    "external_academic": {
        "source_blends": {"external_academic"},
        "summary_kind": "external_not_kb",
        "max_notice_lines": 1,
        "requires_notice": True,
    },
    "general_api": {
        "source_blends": {"general_llm"},
        "summary_kind": "general_api",
        "max_notice_lines": 0,
        "requires_notice": False,
    },
    "insufficient_local_evidence": {
        "source_blends": {"external_academic", "hybrid_local_external", "local_grounded"},
        "summary_kind": "external_not_kb",
        "max_notice_lines": 1,
        "requires_notice": True,
    },
}

_SOURCE_NOTICE_RE = re.compile(
    r"^\s*(?:source note|source)\s*:\s*"
    r"|^\s*(?:note|notice)\s*:\s*(?:local citations|no matching local knowledge-base evidence)"
    r"|no matching local knowledge-base evidence"
    r"|not (?:a )?knowledge-base-grounded answer"
    r"|local citations\s*\[n\]\s*come from the knowledge base"
    r"|\u672a\u547d\u4e2d\u77e5\u8bc6\u5e93\u7247\u6bb5"
    r"|\u4e0d\u662f\u57fa\u4e8e\u77e5\u8bc6\u5e93"
    r"|\u6ca1\u6709\u68c0\u7d22\u5230\u53ef\u76f4\u63a5\u5f15\u7528\u7684\u5e93\u5185\u7247\u6bb5"
    r"|\u5f53\u524d(?:\u6ca1\u6709|\u672a).*?(?:\u5e93\u5185|\u77e5\u8bc6\u5e93).*?(?:\u8bc1\u636e|\u7247\u6bb5|\u547d\u4e2d)",
    flags=re.IGNORECASE,
)

_EN_EXTERNAL_NOTICE_PREFIX_RE = re.compile(
    r"^\s*(?:(?:note|notice)\s*:\s*)?"
    r"no matching local knowledge-base evidence was found"
    r"(?:[;,]\s*[^.\n]*)?\.?\s*",
    flags=re.IGNORECASE,
)
_EN_HYBRID_NOTICE_PREFIX_RE = re.compile(
    r"^\s*(?:(?:note|notice)\s*:\s*)?"
    r"local citations\s*\[n\]\s*come from the knowledge base"
    r"(?:[;,]\s*[^.\n]*)?\.?\s*",
    flags=re.IGNORECASE,
)
_ZH_SOURCE_NOTICE_PREFIX_RE = re.compile(
    r"^\s*(?:(?:\u6ce8\u610f|\u6ce8)\s*[:\uff1a]\s*)?"
    r"(?:"
    r"\u672c\u5730\u77e5\u8bc6\u5e93\u6ca1\u6709\u547d\u4e2d\u76f8\u5173\u8bc1\u636e"
    r"|\u5e26\s*\[n\]\s*\u7684\u5185\u5bb9\u6765\u81ea\u672c\u5730\u77e5\u8bc6\u5e93"
    r"|\u672a\u547d\u4e2d\u77e5\u8bc6\u5e93\u7247\u6bb5"
    r"|\u6ca1\u6709\u68c0\u7d22\u5230\u53ef\u76f4\u63a5\u5f15\u7528\u7684\u5e93\u5185\u7247\u6bb5"
    r")"
    r"[^。\n]{0,220}。?\s*",
    flags=re.IGNORECASE,
)

_CLUTTER_PATTERNS = {
    "agent_trace_leak": re.compile(r"\bagent_trace\b", flags=re.IGNORECASE),
    "trace_panel_leak": re.compile(r"\bresearch agent trace\b", flags=re.IGNORECASE),
    "trace_json_leak": re.compile(r'"mode"\s*:\s*"research_agent"', flags=re.IGNORECASE),
    "plan_steps_leak": re.compile(r"^\s*plan steps?\s*:", flags=re.IGNORECASE | re.MULTILINE),
    "tool_calls_leak": re.compile(r"^\s*tool calls?\s*:", flags=re.IGNORECASE | re.MULTILINE),
    "verification_stats_leak": re.compile(
        r"^\s*verification (?:summary|statistics)\s*:",
        flags=re.IGNORECASE | re.MULTILINE,
    ),
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


def _has_cjk(*values: Any) -> bool:
    return any(re.search(r"[\u4e00-\u9fff]", str(value or "")) for value in values)


def _trace_summary(agent_trace: dict | None) -> dict[str, Any]:
    trace = _record(agent_trace)
    return _record(trace.get("summary"))


def _trace_context(agent_trace: dict | None) -> dict[str, Any]:
    trace = _record(agent_trace)
    return _record(trace.get("context"))


def _source_blend(
    *,
    answer_quality: dict[str, Any],
    agent_source_summary: dict[str, Any],
    agent_trace: dict | None,
    answer_mode: str,
    source_blend: str,
) -> str:
    summary = _trace_summary(agent_trace)
    context = _trace_context(agent_trace)
    raw = _text(
        source_blend,
        answer_quality.get("answer_source_blend"),
        answer_quality.get("source_blend"),
        agent_source_summary.get("source_blend"),
        summary.get("answer_source_blend"),
        context.get("answer_source_blend"),
    )
    if raw in _SOURCE_BLENDS:
        return raw
    mapped = _ANSWER_MODE_TO_SOURCE_BLEND.get(_text(answer_mode, summary.get("answer_mode"), context.get("answer_mode")))
    return mapped or ""


def _answer_profile(answer_quality: dict[str, Any], *, source_blend: str, answer_mode: str) -> str:
    raw = _text(answer_quality.get("answer_profile"), answer_quality.get("profile"))
    if raw in _PROFILE_CONTRACTS:
        return raw
    if source_blend == "hybrid_local_external" or answer_mode == "hybrid_local_external":
        return "hybrid_synthesis"
    if source_blend == "external_academic" or answer_mode == "external_academic_llm":
        return "external_academic"
    if source_blend == "general_llm" or answer_mode == "general_llm":
        return "general_api"
    if source_blend == "local_grounded" or answer_mode == "evidence_grounded":
        return "local_evidence_grounded"
    return "unknown"


def _source_notice_lines(answer: str) -> int:
    seen: set[str] = set()
    for raw in str(answer or "").splitlines():
        line = re.sub(r"\s+", " ", str(raw or "").strip())
        if not line:
            continue
        if _SOURCE_NOTICE_RE.search(line):
            seen.add(line.lower())
    return len(seen)


def _main_answer_clutter(answer: str) -> list[str]:
    text = str(answer or "")
    return [reason for reason, pattern in _CLUTTER_PATTERNS.items() if pattern.search(text)]


def _compact_answer_text(text: str) -> str:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _strip_notice_prefix_from_line(line: str) -> str:
    text = str(line or "")
    for pattern in (_EN_EXTERNAL_NOTICE_PREFIX_RE, _EN_HYBRID_NOTICE_PREFIX_RE, _ZH_SOURCE_NOTICE_PREFIX_RE):
        stripped = pattern.sub("", text, count=1).strip()
        if stripped != text.strip():
            return stripped
    return text.strip()


def _remove_source_notices(answer: str) -> tuple[str, int]:
    removed = 0
    lines: list[str] = []
    for raw in str(answer or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = str(raw or "").rstrip()
        stripped = line.strip()
        remainder = _strip_notice_prefix_from_line(line)
        if remainder != stripped:
            removed += 1
            if remainder and not _SOURCE_NOTICE_RE.search(remainder):
                lines.append(remainder)
            continue
        if _SOURCE_NOTICE_RE.match(stripped):
            removed += 1
            continue
        lines.append(line)
    return _compact_answer_text("\n".join(lines)), removed


def _desired_notice(*, profile: str, source_blend: str, answer_mode: str, query: str, answer: str) -> str:
    use_zh = _has_cjk(query, answer)
    is_hybrid = profile == "hybrid_synthesis" or source_blend == "hybrid_local_external" or answer_mode == "hybrid_local_external"
    is_external = (
        profile in {"external_academic", "insufficient_local_evidence"}
        or source_blend == "external_academic"
        or answer_mode == "external_academic_llm"
    )
    if is_hybrid:
        if use_zh:
            return (
                "\u6ce8\u610f\uff1a\u5e26 [n] \u7684\u5185\u5bb9\u6765\u81ea\u672c\u5730\u77e5\u8bc6\u5e93\uff1b"
                "\u672a\u5e26\u672c\u5730\u5f15\u7528\u7684\u80cc\u666f\u89e3\u91ca\u53ef\u80fd\u6765\u81ea\u5916\u90e8\u6a21\u578b\u8865\u5145\u3002"
            )
        return "Note: local citations [n] come from the knowledge base; uncited background may use external model context."
    if is_external:
        if use_zh:
            return (
                "\u6ce8\u610f\uff1a\u672c\u5730\u77e5\u8bc6\u5e93\u6ca1\u6709\u547d\u4e2d\u76f8\u5173\u8bc1\u636e\uff0c"
                "\u4ee5\u4e0b\u662f\u5916\u90e8\u6a21\u578b\u7684\u901a\u7528\u56de\u7b54\uff0c"
                "\u4e0d\u4ee3\u8868\u5f53\u524d\u77e5\u8bc6\u5e93\u7ed3\u8bba\u3002"
            )
        return "Note: no matching local knowledge-base evidence was found; this is an external model answer, not a knowledge-base-grounded answer."
    return ""


def _summary_for_check(check: dict[str, Any]) -> dict[str, Any]:
    summary = _record(check.get("summary"))
    return {
        "status": str(check.get("status") or ""),
        "failed": list(summary.get("failed") or []),
        "needs_review_count": _int(summary.get("needs_review_count")),
        "profile": str(summary.get("profile") or ""),
        "source_blend": str(summary.get("source_blend") or ""),
        "answer_mode": str(summary.get("answer_mode") or ""),
    }


def _source_summary_for_repaired_check(source_summary: dict[str, Any], *, repaired_answer: str) -> dict[str, Any]:
    out = dict(source_summary or {})
    if out:
        out["source_notice_count"] = _source_notice_lines(repaired_answer)
    return out


def _check_answer_profile(*, profile: str, source_blend: str, contract: dict[str, Any]) -> dict[str, Any]:
    if not contract:
        return {
            "ok": None,
            "profile": profile,
            "source_blend": source_blend,
            "expected_source_blends": [],
            "reasons": [],
        }
    expected = sorted(str(item) for item in set(contract.get("source_blends") or set()) if str(item or "").strip())
    reasons: list[str] = []
    if not source_blend:
        reasons.append("missing_source_blend")
    elif expected and source_blend not in expected:
        reasons.append("source_blend_mismatch")
    return {
        "ok": not reasons,
        "profile": profile,
        "source_blend": source_blend,
        "expected_source_blends": expected,
        "reasons": reasons,
    }


def _check_source_summary(*, source_summary: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    expected_kind = str(contract.get("summary_kind") or "").strip() if contract else ""
    if not expected_kind:
        return {
            "ok": None,
            "kind": str(source_summary.get("kind") or "").strip(),
            "expected_kind": "",
            "should_show": source_summary.get("should_show"),
            "reasons": [],
        }
    reasons: list[str] = []
    kind = str(source_summary.get("kind") or "").strip()
    if not source_summary:
        reasons.append("missing_source_summary")
    elif kind != expected_kind:
        reasons.append("source_summary_kind_mismatch")
    if source_summary and source_summary.get("should_show") is False:
        reasons.append("source_summary_hidden")
    if source_summary and not _text(source_summary.get("label"), source_summary.get("label_key")):
        reasons.append("missing_source_summary_label")
    if source_summary and len(str(source_summary.get("detail") or "")) > 220:
        reasons.append("source_summary_detail_too_long")
    forbidden = sorted(
        key
        for key in ("agent_trace", "plan", "steps", "claims", "research_run")
        if key in source_summary
    )
    if forbidden:
        reasons.append("source_summary_contains_trace_details")
    return {
        "ok": not reasons,
        "kind": kind,
        "expected_kind": expected_kind,
        "should_show": source_summary.get("should_show"),
        "forbidden_keys": forbidden,
        "reasons": reasons,
    }


def _check_notice_shape(*, answer: str, source_summary: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    notice_lines = _source_notice_lines(answer)
    trace_notice_count = _int(source_summary.get("source_notice_count"))
    max_notice_lines = _int(contract.get("max_notice_lines")) if contract else 1
    requires_notice = bool(contract.get("requires_notice")) if contract else False
    reasons: list[str] = []
    if notice_lines > max_notice_lines:
        reasons.append("too_many_source_notices")
    if requires_notice and notice_lines <= 0:
        reasons.append("missing_source_notice")
    if not requires_notice and max_notice_lines <= 0 and notice_lines > 0:
        reasons.append("unnecessary_source_notice")
    if trace_notice_count > max_notice_lines:
        reasons.append("trace_notice_count_too_high")
    return {
        "ok": not reasons,
        "notice_lines": notice_lines,
        "trace_notice_count": trace_notice_count,
        "max_notice_lines": max_notice_lines,
        "requires_notice": requires_notice,
        "reasons": reasons,
    }


def _check_main_answer_clutter(answer: str) -> dict[str, Any]:
    reasons = _main_answer_clutter(answer)
    return {"ok": not reasons, "reasons": reasons}


def build_answer_runtime_check(
    *,
    answer: str,
    answer_quality: dict | None = None,
    agent_source_summary: dict | None = None,
    agent_trace: dict | None = None,
    answer_mode: str = "",
    source_blend: str = "",
) -> dict[str, Any]:
    """Build a compact meta-only guardrail result for generated answers.

    The payload intentionally avoids answer snippets, trace logs, claim rows,
    and tool outputs. It is for runtime diagnostics, not user-facing prose.
    """

    quality = _record(answer_quality)
    source_summary = _record(agent_source_summary)
    summary = _trace_summary(agent_trace)
    context = _trace_context(agent_trace)
    resolved_answer_mode = _text(answer_mode, quality.get("answer_mode"), summary.get("answer_mode"), context.get("answer_mode"))
    resolved_source_blend = _source_blend(
        answer_quality=quality,
        agent_source_summary=source_summary,
        agent_trace=agent_trace,
        answer_mode=resolved_answer_mode,
        source_blend=source_blend,
    )
    profile = _answer_profile(quality, source_blend=resolved_source_blend, answer_mode=resolved_answer_mode)
    contract = dict(_PROFILE_CONTRACTS.get(profile) or {})
    checks = {
        "answer_profile": _check_answer_profile(
            profile=profile,
            source_blend=resolved_source_blend,
            contract=contract,
        ),
        "source_summary": _check_source_summary(source_summary=source_summary, contract=contract),
        "notice_shape": _check_notice_shape(answer=answer, source_summary=source_summary, contract=contract),
        "main_answer_clutter": _check_main_answer_clutter(answer),
    }
    failed = sorted(name for name, result in checks.items() if _record(result).get("ok") is False)
    return {
        "schema_version": ANSWER_RUNTIME_CHECK_SCHEMA_VERSION,
        "status": "needs_review" if failed else "passed",
        "checks": checks,
        "summary": {
            "failed": failed,
            "needs_review_count": len(failed),
            "profile": profile,
            "source_blend": resolved_source_blend,
            "answer_mode": resolved_answer_mode,
        },
    }


def repair_answer_for_runtime_contract(
    *,
    answer: str,
    query: str = "",
    answer_quality: dict | None = None,
    agent_source_summary: dict | None = None,
    agent_trace: dict | None = None,
    answer_mode: str = "",
    source_blend: str = "",
) -> dict[str, Any]:
    """Apply deterministic user-facing cleanup before answer storage.

    This is intentionally limited to presentation/source-disclosure fixes:
    debug suffix removal, source-notice dedupe/removal, and adding the one
    required source notice for hybrid/external answers. It does not rewrite
    factual content or call an LLM.
    """

    original = str(answer or "")
    quality = _record(answer_quality)
    source_summary = _record(agent_source_summary)
    before = build_answer_runtime_check(
        answer=original,
        answer_quality=quality,
        agent_source_summary=source_summary,
        agent_trace=agent_trace,
        answer_mode=answer_mode,
        source_blend=source_blend,
    )
    summary = _record(before.get("summary"))
    profile = str(summary.get("profile") or "")
    resolved_source_blend = str(summary.get("source_blend") or "")
    resolved_answer_mode = str(summary.get("answer_mode") or answer_mode or "")
    desired_notice = _desired_notice(
        profile=profile,
        source_blend=resolved_source_blend,
        answer_mode=resolved_answer_mode,
        query=query,
        answer=original,
    )

    reasons: list[str] = []
    cleaned = clean_assistant_answer_presentation_text(original).strip()
    if original.strip() and cleaned != original.strip():
        reasons.append("debug_content_removed")

    body, removed_notices = _remove_source_notices(cleaned)
    if removed_notices:
        if desired_notice:
            reasons.append("source_notice_normalized")
        else:
            reasons.append("unnecessary_source_notice_removed")

    repaired = body
    if desired_notice:
        if removed_notices <= 0:
            reasons.append("missing_source_notice_added")
        repaired = f"{desired_notice}\n\n{body}".strip() if body else desired_notice
    repaired = _compact_answer_text(repaired)
    if original.strip() and not repaired:
        repaired = original.strip()
        reasons.append("empty_repair_skipped")

    changed = repaired != original.strip()
    after_source_summary = _source_summary_for_repaired_check(source_summary, repaired_answer=repaired)
    after = build_answer_runtime_check(
        answer=repaired,
        answer_quality=quality,
        agent_source_summary=after_source_summary,
        agent_trace=agent_trace,
        answer_mode=resolved_answer_mode,
        source_blend=resolved_source_blend,
    )
    return {
        "answer": repaired,
        "changed": bool(changed),
        "reasons": list(dict.fromkeys(reasons)),
        "before": _summary_for_check(before),
        "after": _summary_for_check(after),
    }
