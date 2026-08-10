from __future__ import annotations

from copy import deepcopy

import pytest

from kb.project_status import build_project_research_status


def _matrix(*, status: str = "verified") -> dict:
    return {
        "id": "matrix-1",
        "title": "Comparison matrix",
        "revision": 3,
        "quality_status": status,
        "updated_at": 30,
        "rows": [
            {"id": "row-a", "source_path": "/papers/a.md", "source_status": "active"},
            {"id": "row-b", "source_path": "/papers/b.md", "source_status": "active"},
        ],
        "source_items": [],
        "comparison_audits": [],
    }


def _brief(*, lineage: str = "current", status: str = "verified") -> dict:
    return {
        "id": "brief-1",
        "title": "Current brief",
        "revision": 2,
        "quality_status": status,
        "quality": {"source_matrix_id": "matrix-1"},
        "lineage": {"status": lineage},
        "updated_at": 40,
    }


def _gap(kind: str, *, matrix_id: str = "matrix-1", brief_id: str = "") -> dict:
    return {
        "id": f"gap-{kind}",
        "kind": kind,
        "status": "open",
        "matrix_id": matrix_id,
        "brief_id": brief_id,
    }


def _scan(*, candidates: int = 0) -> dict:
    return {
        "candidate_count": candidates,
        "first_candidate_matrix_id": "matrix-1" if candidates else "",
        "eligible_matrix_count": 1,
        "scanned_matrix_count": 1,
        "skipped_stale_matrix_count": 0,
        "scan_complete": True,
    }


def _status(
    *,
    matrices: list[dict] | None = None,
    briefs: list[dict] | None = None,
    gaps: list[dict] | None = None,
    comparison_scan: dict | None = None,
    shelf_items: list[dict] | None = None,
) -> dict:
    return build_project_research_status(
        project={"id": "project-1", "name": "Living review"},
        citation_shelf={
            "items": shelf_items
            if shelf_items is not None
            else [
                {"key": "a", "sourcePath": "/papers/a.md"},
                {"key": "b", "libraryMatchPath": "/papers/b.md"},
            ]
        },
        matrices=deepcopy(matrices if matrices is not None else [_matrix()]),
        briefs=deepcopy(briefs if briefs is not None else [_brief()]),
        gaps=deepcopy(gaps or []),
        comparison_scan=deepcopy(comparison_scan if comparison_scan is not None else _scan()),
    )


@pytest.mark.parametrize(
    ("gaps", "candidates", "brief_lineage", "expected_code", "expected_target"),
    [
        (
            [_gap("source_change"), _gap("unsupported_cell"), _gap("missing_cell")],
            2,
            "matrix_updated",
            "refresh_changed_sources",
            "evidence_matrix",
        ),
        (
            [_gap("unsupported_cell"), _gap("missing_cell")],
            2,
            "matrix_updated",
            "repair_evidence",
            "research_gaps",
        ),
        (
            [_gap("missing_cell")],
            2,
            "matrix_updated",
            "fill_evidence_gaps",
            "research_gaps",
        ),
        (
            [_gap("comparison_not_comparable")],
            2,
            "matrix_updated",
            "resolve_comparison_boundaries",
            "research_gaps",
        ),
        ([], 2, "matrix_updated", "review_comparison_candidates", "evidence_matrix"),
        ([], 0, "matrix_updated", "update_research_brief", "research_brief"),
    ],
)
def test_project_status_uses_fixed_quality_first_priority(
    gaps: list[dict],
    candidates: int,
    brief_lineage: str,
    expected_code: str,
    expected_target: str,
) -> None:
    result = _status(
        gaps=gaps,
        comparison_scan=_scan(candidates=candidates),
        briefs=[_brief(lineage=brief_lineage)],
    )

    assert result["recommended_action"]["code"] == expected_code
    assert result["recommended_action"]["target"] == expected_target
    assert isinstance(result["recommended_action"], dict)


def test_project_status_requires_sources_before_creating_matrix() -> None:
    result = _status(
        matrices=[],
        briefs=[],
        comparison_scan={"scan_complete": True},
        shelf_items=[{"key": "a", "sourcePath": "/papers/a.md"}],
    )
    assert result["recommended_action"]["code"] == "add_project_sources"
    assert result["readiness"] == "needs_review"

    ready_sources = _status(
        matrices=[],
        briefs=[],
        comparison_scan={"scan_complete": True},
    )
    assert ready_sources["recommended_action"]["code"] == "create_evidence_matrix"


def test_project_status_requires_candidate_coverage_before_brief() -> None:
    result = _status(briefs=[], comparison_scan={"scan_complete": False, "eligible_matrix_count": 1})
    assert result["recommended_action"]["code"] == "refresh_project_status"

    refreshed = _status(briefs=[], comparison_scan=_scan())
    assert refreshed["recommended_action"]["code"] == "create_research_brief"
    assert refreshed["recommended_action"]["matrix_id"] == "matrix-1"


def test_project_status_is_ready_only_with_current_verified_brief_and_full_scan() -> None:
    result = _status()

    assert result["readiness"] == "ready"
    assert result["recommended_action"]["code"] == "export_current_brief"
    assert result["recommended_action"]["brief_id"] == "brief-1"
    assert result["stages"]["sources"]["project_source_count"] == 2
    assert result["stages"]["comparisons"]["scan_complete"] is True


def test_project_status_deduplicates_shelf_and_matrix_sources() -> None:
    result = _status()
    assert result["stages"]["sources"] == {
        "status": "ready",
        "project_source_count": 2,
        "shelf_source_count": 2,
        "matrix_source_count": 2,
        "changed_source_count": 0,
        "stale_index_matrix_count": 0,
    }


def test_project_status_blocks_when_comparison_scan_skips_a_stale_index() -> None:
    result = _status(
        comparison_scan={
            "candidate_count": 0,
            "eligible_matrix_count": 1,
            "scanned_matrix_count": 0,
            "skipped_stale_matrix_count": 1,
            "first_stale_matrix_id": "matrix-1",
            "scan_complete": False,
        }
    )

    assert result["readiness"] == "blocked"
    assert result["recommended_action"]["code"] == "refresh_changed_sources"
    assert result["recommended_action"]["matrix_id"] == "matrix-1"
