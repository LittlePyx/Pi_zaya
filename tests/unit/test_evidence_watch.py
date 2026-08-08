from __future__ import annotations

from pathlib import Path

from kb.evidence_watch import (
    build_evidence_watch_events,
    evidence_watch_scope_items,
    source_watch_snapshot,
)


def _item(path: Path, *, key: str, title: str) -> dict:
    return {
        "key": key,
        "title": title,
        "sourceName": title,
        "sourcePath": str(path),
        "authors": "A. Researcher",
        "year": "2026",
    }


def _matrix(path: Path) -> dict:
    return {
        "id": "matrix-1",
        "project_id": "project-1",
        "title": "Living evidence",
        "revision": 3,
        "rows": [
            {
                "id": "row-a",
                "source_path": str(path),
                "cells": {
                    "method": {
                        "value": "The method uses coded acquisition.",
                        "evidence_ids": ["ev-a"],
                    }
                },
            }
        ],
        "comparison_audits": [
            {"id": "cmp-1", "left_row_id": "row-a", "right_row_id": "row-b"}
        ],
    }


def test_content_change_is_actionable_and_reports_downstream_impact(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("original evidence", encoding="utf-8")
    item = _item(path, key="a", title="Paper A")
    baseline = source_watch_snapshot([item])
    path.write_text("revised evidence", encoding="utf-8")

    events = build_evidence_watch_events(
        _matrix(path),
        baseline=baseline,
        current=source_watch_snapshot([item]),
        briefs=[
            {
                "id": "brief-1",
                "title": "Brief",
                "revision": 2,
                "quality": {"source_matrix_id": "matrix-1"},
                "evidence": [{"source_path": str(path), "citation_number": 4}],
            }
        ],
    )

    assert [event["kind"] for event in events] == ["source_content_changed"]
    event = events[0]
    assert event["actionable"] is True
    assert event["impact"]["affected_row_ids"] == ["row-a"]
    assert event["impact"]["affected_fields"] == ["method"]
    assert event["impact"]["affected_comparison_ids"] == ["cmp-1"]
    assert event["impact"]["affected_briefs"][0]["citation_numbers"] == [4]


def test_metadata_only_change_is_visible_but_not_actionable(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("stable evidence", encoding="utf-8")
    baseline = source_watch_snapshot([_item(path, key="a", title="Old title")])
    current = source_watch_snapshot([_item(path, key="a", title="Corrected title")])

    events = build_evidence_watch_events(_matrix(path), baseline=baseline, current=current)

    assert [event["kind"] for event in events] == ["source_metadata_changed"]
    assert events[0]["actionable"] is False
    assert events[0]["severity"] == "info"


def test_source_addition_is_actionable_and_lists_candidate_fields(tmp_path: Path) -> None:
    first = tmp_path / "a.md"
    second = tmp_path / "b.md"
    first.write_text("A", encoding="utf-8")
    second.write_text("B", encoding="utf-8")
    first_item = _item(first, key="a", title="Paper A")
    second_item = _item(second, key="b", title="Paper B")

    events = build_evidence_watch_events(
        _matrix(first),
        baseline=source_watch_snapshot([first_item]),
        current=source_watch_snapshot([first_item, second_item]),
    )

    assert [event["kind"] for event in events] == ["source_added"]
    assert events[0]["actionable"] is True
    assert len(events[0]["impact"]["candidate_fields"]) == 5


def test_source_removal_is_actionable(tmp_path: Path) -> None:
    first = tmp_path / "a.md"
    second = tmp_path / "b.md"
    first.write_text("A", encoding="utf-8")
    second.write_text("B", encoding="utf-8")
    first_item = _item(first, key="a", title="Paper A")
    second_item = _item(second, key="b", title="Paper B")

    events = build_evidence_watch_events(
        _matrix(first),
        baseline=source_watch_snapshot([first_item, second_item]),
        current=source_watch_snapshot([first_item]),
    )

    assert [event["kind"] for event in events] == ["source_removed"]
    assert events[0]["source_name"] == "Paper B"
    assert events[0]["actionable"] is True


def test_missing_source_is_reported_separately_from_content_change(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("evidence", encoding="utf-8")
    item = _item(path, key="a", title="Paper A")
    baseline = source_watch_snapshot([item])
    path.unlink()

    events = build_evidence_watch_events(
        _matrix(path),
        baseline=baseline,
        current=source_watch_snapshot([item]),
    )

    assert [event["kind"] for event in events] == ["source_unavailable"]
    assert events[0]["severity"] == "error"
    assert events[0]["actionable"] is True


def test_watch_scope_keeps_tracked_source_when_basket_exceeds_matrix_limit(tmp_path: Path) -> None:
    items: list[dict] = []
    for index in range(9):
        path = tmp_path / f"paper-{index}.md"
        path.write_text(str(index), encoding="utf-8")
        items.append(_item(path, key=str(index), title=f"Paper {index}"))

    scoped = evidence_watch_scope_items(items, tracked_items=[items[-1]], limit=8)

    assert len(scoped) == 8
    assert scoped[0]["key"] == "8"
    assert {item["key"] for item in scoped} == {"0", "1", "2", "3", "4", "5", "6", "8"}
