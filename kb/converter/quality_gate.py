from __future__ import annotations

from pathlib import Path
from typing import Any

from .quality_repair import (
    conversion_quality_result_path,
    conversion_quality_report_is_stale,
    load_conversion_quality_result,
    plan_conversion_quality_repair,
    repair_markdown_quality,
    write_conversion_quality_result,
)


_BLOCKING_ACTIONS = {"reconvert", "review"}
_CRITICAL_AUTOFIX_ISSUES = {
    "missing_source_pages",
    "page_marker_gaps",
    "source_page_marker_alignment",
    "reference_index_truncated",
}


def _report_is_stale(md_path: Path, report: dict[str, Any]) -> bool:
    return conversion_quality_report_is_stale(md_path, report)


def _issue_codes_from_report(report: dict[str, Any]) -> list[str]:
    plan = report.get("repair_plan") if isinstance(report.get("repair_plan"), dict) else {}
    codes = plan.get("issue_codes") if isinstance(plan, dict) else None
    if isinstance(codes, list):
        return [str(item or "").strip().lower() for item in codes if str(item or "").strip()]
    repair = report.get("auto_repair") if isinstance(report.get("auto_repair"), dict) else {}
    codes = repair.get("remaining_issue_codes") if isinstance(repair, dict) else None
    if isinstance(codes, list):
        return [str(item or "").strip().lower() for item in codes if str(item or "").strip()]
    return []


def _compact_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": str((plan or {}).get("action") or "review"),
        "scope": str((plan or {}).get("scope") or ""),
        "speed_mode": str((plan or {}).get("speed_mode") or ""),
        "reason": str((plan or {}).get("reason") or ""),
        "issue_codes": [str(item) for item in list((plan or {}).get("issue_codes") or []) if str(item or "").strip()][:30],
        "reconvert_issue_codes": [
            str(item) for item in list((plan or {}).get("reconvert_issue_codes") or []) if str(item or "").strip()
        ][:30],
        "autofix_issue_codes": [
            str(item) for item in list((plan or {}).get("autofix_issue_codes") or []) if str(item or "").strip()
        ][:30],
        "review_issue_codes": [
            str(item) for item in list((plan or {}).get("review_issue_codes") or []) if str(item or "").strip()
        ][:30],
        "retry_pages": [
            int(item)
            for item in list((plan or {}).get("retry_pages") or [])
            if str(item or "").isdigit() and int(item) > 0
        ][:500],
    }


def _unreliable_pages_from_report(report: dict[str, Any]) -> list[int]:
    source_quality = report.get("source_quality") if isinstance(report.get("source_quality"), dict) else {}
    return sorted(
        {
            int(item)
            for item in list((source_quality or {}).get("evidence_unreliable_pages") or [])
            if str(item or "").isdigit() and int(item) > 0
        }
    )[:500]


def load_or_write_conversion_quality_result(
    md_path: Path | str,
    *,
    refresh_stale: bool = True,
    source_pdf_path: Path | str | None = None,
) -> dict[str, Any]:
    path = Path(md_path).expanduser()
    report = load_conversion_quality_result(path)
    if refresh_stale and _report_is_stale(path, report):
        try:
            report = write_conversion_quality_result(path, source_pdf_path=source_pdf_path)
        except Exception:
            report = {}
    return report if isinstance(report, dict) else {}


def assess_markdown_index_quality(
    md_path: Path | str,
    *,
    quality_result: dict[str, Any] | None = None,
    refresh_stale: bool = True,
    allow_blocked: bool = False,
    source_pdf_path: Path | str | None = None,
) -> dict[str, Any]:
    path = Path(md_path).expanduser()
    if not path.exists() or not path.is_file():
        plan = plan_conversion_quality_repair(["missing_markdown"])
        compact = _compact_plan(plan)
        return {
            "ok": bool(allow_blocked),
            "indexable": bool(allow_blocked),
            "status": "degraded" if allow_blocked else "blocked",
            "action": "reconvert",
            "reason": str(plan.get("reason") or "Markdown output is missing."),
            "issue_codes": ["missing_markdown"],
            "blocking_issue_codes": ["missing_markdown"],
            "repair_plan": compact,
            "report_path": str(conversion_quality_result_path(path)),
        }

    report = dict(quality_result or {})
    if not report:
        report = load_or_write_conversion_quality_result(path, refresh_stale=refresh_stale, source_pdf_path=source_pdf_path)
    elif refresh_stale and _report_is_stale(path, report):
        report = load_or_write_conversion_quality_result(path, refresh_stale=True, source_pdf_path=source_pdf_path)
    if not report:
        plan = plan_conversion_quality_repair(["quality_scan_failed"])
        compact = _compact_plan(plan)
        return {
            "ok": bool(allow_blocked),
            "indexable": bool(allow_blocked),
            "status": "degraded" if allow_blocked else "blocked",
            "action": "review",
            "reason": "Conversion quality scan failed before indexing.",
            "issue_codes": ["quality_scan_failed"],
            "blocking_issue_codes": ["quality_scan_failed"],
            "repair_plan": compact,
            "report_path": str(conversion_quality_result_path(path)),
        }

    plan = report.get("repair_plan") if isinstance(report.get("repair_plan"), dict) else {}
    if not isinstance(plan, dict) or not plan:
        plan = plan_conversion_quality_repair(_issue_codes_from_report(report), metrics=report.get("metrics") if isinstance(report.get("metrics"), dict) else {})

    compact = _compact_plan(plan)
    action = str(compact.get("action") or "review").strip().lower() or "review"
    issue_codes = [str(item) for item in list(compact.get("issue_codes") or []) if str(item or "").strip()]
    critical_autofix_codes = [code for code in issue_codes if code in _CRITICAL_AUTOFIX_ISSUES]
    blocking_codes = (
        list(compact.get("reconvert_issue_codes") or []) + list(compact.get("review_issue_codes") or [])
        if action in _BLOCKING_ACTIONS
        else []
    )
    if critical_autofix_codes:
        blocking_codes = list(blocking_codes) + critical_autofix_codes
    blocked = action in _BLOCKING_ACTIONS or bool(critical_autofix_codes)
    status = "blocked" if blocked else ("ready" if action == "none" else "degraded")
    indexable = (not blocked) or bool(allow_blocked)
    if blocked and allow_blocked:
        status = "degraded"

    return {
        "ok": bool(indexable),
        "indexable": bool(indexable),
        "status": status,
        "action": action,
        "reason": str(compact.get("reason") or ""),
        "issue_codes": issue_codes[:30],
        "blocking_issue_codes": [str(item) for item in blocking_codes if str(item or "").strip()][:30],
        "repair_plan": compact,
        "report_path": str(conversion_quality_result_path(path)),
        "quality_result": report,
        "evidence_unreliable_pages": _unreliable_pages_from_report(report),
    }


def prepare_markdown_for_index(
    md_path: Path | str,
    *,
    auto_repair: bool = True,
    allow_blocked: bool = False,
    source_pdf_path: Path | str | None = None,
) -> dict[str, Any]:
    path = Path(md_path).expanduser()
    assessment = assess_markdown_index_quality(path, allow_blocked=allow_blocked, source_pdf_path=source_pdf_path)
    repair_result: dict[str, Any] = {}
    action_before_repair = str(assessment.get("action") or "").strip().lower()
    issue_codes_before_repair = [
        str(item or "").strip().lower()
        for item in list(assessment.get("issue_codes") or [])
        if str(item or "").strip()
    ]
    should_attempt_repair = action_before_repair == "autofix" or (
        action_before_repair == "reconvert"
        and bool({"missing_images", "source_page_text_corruption"}.intersection(issue_codes_before_repair))
    )
    if bool(auto_repair) and should_attempt_repair:
        try:
            repair_result = repair_markdown_quality(path, issue_codes=issue_codes_before_repair, source_pdf_path=source_pdf_path)
            report = write_conversion_quality_result(path, auto_repair_result=repair_result, source_pdf_path=source_pdf_path)
            assessment = assess_markdown_index_quality(
                path,
                quality_result=report,
                refresh_stale=False,
                allow_blocked=allow_blocked,
                source_pdf_path=source_pdf_path,
            )
        except Exception as exc:
            assessment = {
                **assessment,
                "ok": False,
                "indexable": False,
                "status": "blocked",
                "action": "review",
                "reason": f"Markdown auto-repair failed before indexing: {exc}",
                "blocking_issue_codes": list(assessment.get("issue_codes") or []),
            }
    assessment["auto_repair"] = {
        "attempted": bool(auto_repair and should_attempt_repair),
        "changed": bool(repair_result.get("changed")),
        "unsafe": bool(repair_result.get("unsafe")),
        "applied": [str(item) for item in list(repair_result.get("applied") or []) if str(item or "").strip()][:20],
    }
    return assessment


def index_quality_document_fields(assessment: dict[str, Any]) -> dict[str, Any]:
    gate = {
        "status": str((assessment or {}).get("status") or ""),
        "indexable": bool((assessment or {}).get("indexable")),
        "action": str((assessment or {}).get("action") or ""),
        "reason": str((assessment or {}).get("reason") or ""),
        "issue_codes": [str(item) for item in list((assessment or {}).get("issue_codes") or []) if str(item or "").strip()][:30],
        "blocking_issue_codes": [
            str(item) for item in list((assessment or {}).get("blocking_issue_codes") or []) if str(item or "").strip()
        ][:30],
        "report_path": str((assessment or {}).get("report_path") or ""),
    }
    return {
        "index_status": "ready" if gate["status"] == "ready" else f"quality_{gate['status'] or 'unknown'}",
        "quality_gate": gate,
    }
