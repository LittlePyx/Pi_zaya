from __future__ import annotations

import re
from typing import Callable

from api.reference_summary_text import _clean_summary_line, _summary_excerpt


def _has_summary_action_signal(text: str) -> bool:
    s = str(text or "")
    return bool(re.search(r"(提出|设计|构建|采用|引入|实现|develop|propose|introduce|present)", s, flags=re.I))


def _has_summary_result_signal(text: str) -> bool:
    s = str(text or "")
    return bool(re.search(r"(结果|显示|提升|降低|加速|优于|有效|性能|实验|result|show|improv|outperform|achiev)", s, flags=re.I))


def _looks_low_value_shelf_summary(text: str) -> bool:
    s = _clean_summary_line(text)
    if not s:
        return False
    low = s.lower()
    patterns = (
        r"\u5e2e\u52a9\u6838\u5bf9",
        r"\u7ebf\u7d22\u4ece\u54ea\u91cc\u6765",
        r"\u65b9\u6cd5\u80cc\u666f|\u5b9e\u73b0\u4f9d\u636e",
        r"\u4f5c\u4e3a\u5f53\u524d\u8bba\u6587\u5f15\u7528",
        r"\u5f53\u524d\u8bba\u6587\u5f15\u7528\u7684\u65b9\u6cd5",
        r"\u5f15\u7528\u7684\u65b9\u6cd5\u80cc\u666f",
        r"\u6765\u6e90\u7ebf\u7d22",
        r"\bhelps?\s+(?:verify|check|trace)\b",
        r"\bmethod\s+background\b",
        r"\bcited\s+(?:prior\s+)?work\b",
    )
    return any(re.search(pattern, s) or re.search(pattern, low) for pattern in patterns)


def _looks_metadata_only_summary(text: str) -> bool:
    s = _clean_summary_line(text)
    if not s:
        return False
    return bool(
        re.search(
            r"\u4ec5\u68c0\u7d22\u5230|\u6682\u65e0\u53ef\u7528\u6458\u8981|\u7f3a\u5c11\u53ef\u7528\u6458\u8981|\u5efa\u8bae.*DOI|metadata only|no abstract",
            s,
            flags=re.I,
        )
    )


def _is_summary_quality_ok(
    text: str,
    *,
    looks_fragmentary_ref_summary: Callable[[str], bool],
    looks_why_like_ref_summary: Callable[[str], bool],
) -> bool:
    s = _clean_summary_line(text)
    if not s:
        return False
    if _looks_low_value_shelf_summary(s):
        return False
    if looks_fragmentary_ref_summary(s):
        return False
    if looks_why_like_ref_summary(s):
        return False
    if len(s) < 50:
        return False
    if not re.search(
        r"(提出|设计|构建|采用|引入|实现|比较|分析|评估|develop|propose|introduce|present|compare|analy[sz]e|evaluat)",
        s,
        flags=re.I,
    ):
        return False
    if not re.search(
        r"(结果|显示|提升|降低|差异|优劣|加速|优于|有效|性能|实验|result|show|improv|outperform|achiev|difference|trade-?off|advantage|limitation)",
        s,
        flags=re.I,
    ):
        return False
    return True


def _summary_quality_contract(
    meta: dict,
    *,
    is_summary_quality_ok: Callable[[str], bool],
    looks_like_title_echo: Callable[[str, str], bool],
) -> dict:
    data = dict(meta or {})
    summary = _summary_excerpt(str(data.get("summary_line") or ""), max_sentences=3, max_len=360)
    source = str(data.get("summary_source") or "").strip().lower()
    provider = str(data.get("summary_provider") or "").strip().lower()
    generation = str(data.get("summary_generation") or "").strip().lower()
    title = str(data.get("title") or "").strip()
    issues: list[dict[str, str]] = []

    if not summary:
        issues.append({"code": "missing_summary", "severity": "error", "field": "summary_line"})
    elif _looks_low_value_shelf_summary(summary):
        issues.append({"code": "low_value_summary", "severity": "error", "field": "summary_line"})
    elif source == "metadata" and _looks_metadata_only_summary(summary):
        issues.append({"code": "metadata_only_summary", "severity": "warning", "field": "summary_line"})
    elif not is_summary_quality_ok(summary):
        issues.append({"code": "weak_summary", "severity": "warning", "field": "summary_line"})
    if summary and title and looks_like_title_echo(summary, title):
        issues.append({"code": "title_echo", "severity": "warning", "field": "summary_line"})
    if source == "metadata":
        issues.append({"code": "metadata_only", "severity": "warning", "field": "summary_source"})
    elif summary and not source:
        issues.append({"code": "missing_summary_source", "severity": "warning", "field": "summary_source"})

    trusted_sources = {
        "abstract",
        "fulltext",
        "citation_context",
        "reference_primary_evidence",
        "navigation",
        "exact_anchor",
        "section_intent_rescue",
        "doc_list_seed",
        "doc_list_prompt_aligned",
    }
    error_count = sum(1 for item in issues if item.get("severity") == "error")
    warning_count = sum(1 for item in issues if item.get("severity") == "warning")
    trusted = bool(
        summary
        and source in trusted_sources
        and error_count == 0
        and not any(item.get("code") == "title_echo" for item in issues)
    )
    if error_count:
        status = "error"
    elif warning_count:
        status = "fallback" if source == "metadata" else "warning"
    elif trusted:
        status = "grounded"
    else:
        status = "fallback"
    score = max(0, 100 - error_count * 45 - warning_count * 14)
    if trusted:
        score = max(score, 92)
    elif source == "metadata":
        score = min(score, 68)
    elif source in {"citation_card", "citation_card_view"}:
        score = min(max(score, 74), 86)

    return {
        "contract_version": 1,
        "ok": bool(trusted and error_count == 0),
        "status": status,
        "score": int(score),
        "source": source,
        "provider": provider,
        "generation": generation,
        "issues": issues,
        "export_ready": bool(summary and source != "metadata" and error_count == 0),
    }
