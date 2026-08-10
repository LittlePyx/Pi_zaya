from __future__ import annotations

from collections import Counter
from typing import Any

from kb.evidence_watch import source_identity


PROJECT_RESEARCH_STATUS_CONTRACT_VERSION = 1

_ACTIVE_GAP_STATUSES = {"open", "in_progress"}
_CURRENT_BRIEF_LINEAGE = {"current", "current_equivalent"}
_STALE_BRIEF_LINEAGE = {"matrix_updated", "matrix_updated_unverified"}


def _text(value: object) -> str:
    return str(value or "").strip()


def _records(value: object) -> list[dict[str, Any]]:
    return [item for item in list(value or []) if isinstance(item, dict)]


def _latest(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    return max(
        records,
        key=lambda item: (
            float(item.get("updated_at") or 0.0),
            float(item.get("created_at") or 0.0),
            _text(item.get("id")),
        ),
        default=None,
    )


def _source_path(item: dict[str, Any]) -> str:
    return _text(
        item.get("libraryMatchPath")
        or item.get("library_match_path")
        or item.get("sourcePath")
        or item.get("source_path")
    )


def _unique_source_count(items: list[dict[str, Any]]) -> int:
    identities = {
        source_identity(path)
        for item in items
        if (path := _source_path(item))
    }
    identities.discard("")
    return len(identities)


def _matrix_source_items(matrices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for matrix in matrices:
        for row in _records(matrix.get("rows")):
            if _text(row.get("source_status") or "active") == "active":
                items.append(row)
        items.extend(_records(matrix.get("source_items")))
    return items


def _resource_ref(record: dict[str, Any] | None, *, kind: str) -> dict[str, Any]:
    record = record or {}
    return {
        f"{kind}_id": _text(record.get("id")),
        f"{kind}_title": _text(record.get("title")),
        f"{kind}_revision": int(record.get("revision") or 0),
    }


def _action(
    code: str,
    target: str,
    *,
    priority: int,
    reason: str,
    matrix: dict[str, Any] | None = None,
    brief: dict[str, Any] | None = None,
    gap_count: int = 0,
    candidate_count: int = 0,
    workspace_tab: str = "",
) -> dict[str, Any]:
    return {
        "code": code,
        "target": target,
        "priority": int(priority),
        "reason": reason,
        **_resource_ref(matrix, kind="matrix"),
        **_resource_ref(brief, kind="brief"),
        "gap_count": int(gap_count),
        "candidate_count": int(candidate_count),
        "workspace_tab": workspace_tab,
    }


def build_project_research_status(
    *,
    project: dict[str, Any],
    citation_shelf: dict[str, Any] | None,
    matrices: list[dict[str, Any]],
    briefs: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    comparison_scan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one deterministic project recommendation from audited product state.

    The builder never asks a model to judge readiness. Its order is fixed:
    changed sources, evidence defects, comparison boundaries/candidates, stale
    briefs, missing workflow artifacts, then export readiness.
    """

    matrices = _records(matrices)
    briefs = _records(briefs)
    active_gaps = [
        item
        for item in _records(gaps)
        if _text(item.get("status")) in _ACTIVE_GAP_STATUSES
    ]
    shelf_items = _records((citation_shelf or {}).get("items"))
    matrix_source_items = _matrix_source_items(matrices)
    shelf_source_count = _unique_source_count(shelf_items)
    matrix_source_count = _unique_source_count(matrix_source_items)
    project_source_count = _unique_source_count([*shelf_items, *matrix_source_items])

    verified_matrices = [item for item in matrices if _text(item.get("quality_status")) == "verified"]
    review_matrices = [item for item in matrices if _text(item.get("quality_status")) != "verified"]
    latest_matrix = _latest(matrices)
    latest_verified_matrix = _latest(verified_matrices)

    verified_briefs = [item for item in briefs if _text(item.get("quality_status")) == "verified"]
    current_briefs = [
        item
        for item in verified_briefs
        if _text((item.get("lineage") or {}).get("status")) in _CURRENT_BRIEF_LINEAGE
    ]
    stale_briefs = [
        item
        for item in briefs
        if _text((item.get("lineage") or {}).get("status")) in _STALE_BRIEF_LINEAGE
    ]
    blocked_briefs = [
        item
        for item in briefs
        if _text((item.get("lineage") or {}).get("status"))
        and _text((item.get("lineage") or {}).get("status"))
        not in {*_CURRENT_BRIEF_LINEAGE, *_STALE_BRIEF_LINEAGE}
    ]
    latest_brief = _latest(briefs)
    latest_current_brief = _latest(current_briefs)

    gap_counts = Counter(_text(item.get("kind")) for item in active_gaps)
    source_change_count = gap_counts["source_change"]
    unsupported_count = gap_counts["unsupported_cell"]
    matrix_review_count = gap_counts["matrix_needs_review"]
    missing_count = gap_counts["missing_cell"]
    comparison_boundary_count = gap_counts["comparison_not_comparable"]
    stale_gap_count = gap_counts["brief_stale"]

    comparison_scan = comparison_scan if isinstance(comparison_scan, dict) else {}
    comparison_scan_complete = bool(comparison_scan.get("scan_complete"))
    pending_candidate_count = int(comparison_scan.get("candidate_count") or 0)
    eligible_comparison_matrices = int(comparison_scan.get("eligible_matrix_count") or 0)
    scanned_comparison_matrices = int(comparison_scan.get("scanned_matrix_count") or 0)
    skipped_stale_matrices = int(comparison_scan.get("skipped_stale_matrix_count") or 0)
    source_block_count = source_change_count + skipped_stale_matrices
    verified_comparison_count = sum(
        1
        for matrix in matrices
        for audit in _records(matrix.get("comparison_audits"))
        if _text(audit.get("status")) == "verified"
    )
    not_comparable_count = sum(
        1
        for matrix in matrices
        for audit in _records(matrix.get("comparison_audits"))
        if _text(audit.get("status")) == "not_comparable"
    )

    source_gap = next((item for item in active_gaps if _text(item.get("kind")) == "source_change"), None)
    evidence_gap = next(
        (
            item
            for kind in ("unsupported_cell", "matrix_needs_review")
            for item in active_gaps
            if _text(item.get("kind")) == kind
        ),
        None,
    )
    missing_gap = next((item for item in active_gaps if _text(item.get("kind")) == "missing_cell"), None)
    comparison_gap = next(
        (item for item in active_gaps if _text(item.get("kind")) == "comparison_not_comparable"),
        None,
    )
    stale_gap = next((item for item in active_gaps if _text(item.get("kind")) == "brief_stale"), None)

    matrix_by_id = {_text(item.get("id")): item for item in matrices if _text(item.get("id"))}
    brief_by_id = {_text(item.get("id")): item for item in briefs if _text(item.get("id"))}

    if source_block_count:
        matrix = (
            matrix_by_id.get(_text((source_gap or {}).get("matrix_id")))
            or matrix_by_id.get(_text(comparison_scan.get("first_stale_matrix_id")))
            or latest_matrix
        )
        recommended = _action(
            "refresh_changed_sources",
            "evidence_matrix",
            priority=100,
            reason="source_changes_require_reindex_and_evidence_refresh",
            matrix=matrix,
            gap_count=source_change_count,
        )
    elif unsupported_count or matrix_review_count:
        matrix = matrix_by_id.get(_text((evidence_gap or {}).get("matrix_id"))) or latest_matrix
        recommended = _action(
            "repair_evidence",
            "research_gaps",
            priority=90,
            reason="unsupported_or_unverified_matrix_evidence",
            matrix=matrix,
            gap_count=unsupported_count + matrix_review_count,
        )
    elif missing_count:
        matrix = matrix_by_id.get(_text((missing_gap or {}).get("matrix_id"))) or latest_matrix
        recommended = _action(
            "fill_evidence_gaps",
            "research_gaps",
            priority=85,
            reason="matrix_cells_are_missing_grounded_evidence",
            matrix=matrix,
            gap_count=missing_count,
        )
    elif comparison_boundary_count:
        matrix = matrix_by_id.get(_text((comparison_gap or {}).get("matrix_id"))) or latest_verified_matrix
        recommended = _action(
            "resolve_comparison_boundaries",
            "research_gaps",
            priority=80,
            reason="saved_comparisons_are_not_evidence_comparable",
            matrix=matrix,
            gap_count=comparison_boundary_count,
        )
    elif pending_candidate_count:
        candidate_matrix_id = _text(comparison_scan.get("first_candidate_matrix_id"))
        matrix = matrix_by_id.get(candidate_matrix_id) or latest_verified_matrix
        recommended = _action(
            "review_comparison_candidates",
            "evidence_matrix",
            priority=75,
            reason="evidence_bound_comparisons_await_human_confirmation",
            matrix=matrix,
            candidate_count=pending_candidate_count,
            workspace_tab="comparisons",
        )
    elif stale_gap_count or stale_briefs or blocked_briefs:
        brief = brief_by_id.get(_text((stale_gap or {}).get("brief_id"))) or _latest(stale_briefs) or _latest(blocked_briefs) or latest_brief
        quality = brief.get("quality") if isinstance((brief or {}).get("quality"), dict) else {}
        matrix = matrix_by_id.get(_text(quality.get("source_matrix_id"))) or latest_verified_matrix
        recommended = _action(
            "update_research_brief",
            "research_brief",
            priority=70,
            reason="research_brief_is_stale_or_lineage_blocked",
            matrix=matrix,
            brief=brief,
            gap_count=max(stale_gap_count, len(stale_briefs) + len(blocked_briefs)),
        )
    elif project_source_count < 2:
        recommended = _action(
            "add_project_sources",
            "citation_shelf",
            priority=60,
            reason="at_least_two_full_text_sources_are_needed",
        )
    elif not matrices:
        recommended = _action(
            "create_evidence_matrix",
            "evidence_matrix",
            priority=50,
            reason="project_sources_are_ready_for_structured_evidence",
        )
    elif not verified_matrices:
        recommended = _action(
            "review_evidence_matrix",
            "evidence_matrix",
            priority=45,
            reason="no_verified_evidence_matrix_is_available",
            matrix=latest_matrix,
        )
    elif not comparison_scan_complete:
        recommended = _action(
            "refresh_project_status",
            "project_status",
            priority=40,
            reason="comparison_candidate_coverage_has_not_been_refreshed",
            matrix=latest_verified_matrix,
        )
    elif not briefs:
        recommended = _action(
            "create_research_brief",
            "research_brief",
            priority=30,
            reason="verified_matrix_is_ready_for_audited_synthesis",
            matrix=latest_verified_matrix,
        )
    elif not current_briefs:
        recommended = _action(
            "review_research_brief",
            "research_brief",
            priority=25,
            reason="no_current_verified_research_brief_is_available",
            matrix=latest_verified_matrix,
            brief=latest_brief,
        )
    else:
        quality = latest_current_brief.get("quality") if isinstance((latest_current_brief or {}).get("quality"), dict) else {}
        matrix = matrix_by_id.get(_text(quality.get("source_matrix_id"))) or latest_verified_matrix
        recommended = _action(
            "export_current_brief",
            "research_brief",
            priority=0,
            reason="all_current_evidence_gates_are_clear",
            matrix=matrix,
            brief=latest_current_brief,
        )

    if source_block_count:
        readiness = "blocked"
    elif (
        active_gaps
        or pending_candidate_count
        or project_source_count < 2
        or not verified_matrices
        or not current_briefs
        or not comparison_scan_complete
    ):
        readiness = "needs_review"
    else:
        readiness = "ready"

    stages = {
        "sources": {
            "status": "blocked" if source_block_count else ("ready" if project_source_count >= 2 else "needs_input"),
            "project_source_count": project_source_count,
            "shelf_source_count": shelf_source_count,
            "matrix_source_count": matrix_source_count,
            "changed_source_count": source_change_count,
            "stale_index_matrix_count": skipped_stale_matrices,
        },
        "matrices": {
            "status": "needs_review" if review_matrices or unsupported_count or matrix_review_count else ("ready" if verified_matrices else "not_started"),
            "total": len(matrices),
            "verified": len(verified_matrices),
            "needs_review": len(review_matrices),
            **_resource_ref(latest_matrix, kind="latest_matrix"),
        },
        "evidence": {
            "status": "needs_review" if unsupported_count or matrix_review_count or missing_count else ("ready" if verified_matrices else "not_started"),
            "active_gap_count": unsupported_count + matrix_review_count + missing_count,
            "unsupported_count": unsupported_count,
            "missing_count": missing_count,
            "matrix_review_count": matrix_review_count,
        },
        "comparisons": {
            "status": "needs_review" if comparison_boundary_count or pending_candidate_count else ("ready" if comparison_scan_complete and verified_matrices else "not_started"),
            "verified_count": verified_comparison_count,
            "not_comparable_count": not_comparable_count,
            "boundary_gap_count": comparison_boundary_count,
            "pending_candidate_count": pending_candidate_count,
            "eligible_matrix_count": eligible_comparison_matrices,
            "scanned_matrix_count": scanned_comparison_matrices,
            "skipped_stale_matrix_count": skipped_stale_matrices,
            "scan_complete": comparison_scan_complete,
        },
        "briefs": {
            "status": "needs_review" if stale_briefs or blocked_briefs else ("ready" if current_briefs else "not_started"),
            "total": len(briefs),
            "verified": len(verified_briefs),
            "current": len(current_briefs),
            "stale": len(stale_briefs),
            "lineage_blocked": len(blocked_briefs),
            **_resource_ref(latest_brief, kind="latest_brief"),
        },
    }
    return {
        "contract_version": PROJECT_RESEARCH_STATUS_CONTRACT_VERSION,
        "project": {
            "id": _text(project.get("id")),
            "name": _text(project.get("name")),
        },
        "readiness": readiness,
        "stages": stages,
        "active_gap_count": len(active_gaps),
        "gap_counts": dict(sorted((key, value) for key, value in gap_counts.items() if key)),
        "recommended_action": recommended,
    }
