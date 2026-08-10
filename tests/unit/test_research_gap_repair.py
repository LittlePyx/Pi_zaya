from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from kb.evidence_matrix import (
    apply_evidence_matrix_cell_repair,
    evidence_matrix_cell_repair_candidates,
    evidence_matrix_quality,
)


def _chunk(source: str, text: str, *, chunk_id: str, heading: str = "Discussion / Limitations") -> dict:
    return {
        "id": chunk_id,
        "text": text,
        "meta": {
            "source_path": source,
            "heading_path": heading,
            "page_start": 7,
            "page_end": 7,
            "block_id": f"block-{chunk_id}",
            "evidence_ready": True,
        },
    }


def _matrix(source: str) -> dict:
    method_quote = "We propose a coded optical reconstruction network for dynamic imaging."
    return {
        "id": "matrix-1",
        "project_id": "project-1",
        "title": "Dynamic imaging evidence",
        "objective": "Compare dynamic imaging reconstruction accuracy and limitations.",
        "revision": 3,
        "rows": [
            {
                "id": "row-a",
                "source_item_key": "paper-a",
                "paper": "Paper A",
                "source_name": "Paper A",
                "source_path": source,
                "source_status": "active",
                "cells": {
                    "method": {
                        "field": "method",
                        "value": method_quote,
                        "support_status": "grounded",
                        "evidence_ids": ["ev-method"],
                        "manual_override": False,
                    },
                    "limitation": {
                        "field": "limitation",
                        "value": "",
                        "support_status": "missing",
                        "evidence_ids": [],
                        "manual_override": False,
                    },
                },
            }
        ],
        "evidence": [
            {
                "id": "ev-method",
                "field": "method",
                "source_path": source,
                "evidence_quote": method_quote,
            }
        ],
        "source_items": [{"key": "paper-a", "title": "Paper A", "sourcePath": source}],
        "comparison_flags": [],
        "comparison_audits": [],
        "quality": {"source_watch_snapshot": {"contract_version": 1, "sources": []}},
    }


def _gap(source: str) -> dict:
    return {
        "id": "gap-1",
        "gap_key": "gap-key-1",
        "project_id": "project-1",
        "kind": "missing_cell",
        "matrix_id": "matrix-1",
        "matrix_revision": 3,
        "row_id": "row-a",
        "field": "limitation",
        "source_path": source,
    }


def test_same_source_cell_repair_recovers_exact_grounded_evidence(tmp_path: Path) -> None:
    source = str(tmp_path / "paper-a.md")
    other_source = str(tmp_path / "other" / "paper-a.md")
    exact = "However, our dynamic reconstruction remains limited by motion and calibration errors."
    chunks = [
        _chunk(other_source, exact, chunk_id="other:1"),
        _chunk(source, exact, chunk_id="paper-a:7"),
    ]
    matrix = _matrix(source)
    gap = _gap(source)

    repairs = evidence_matrix_cell_repair_candidates(
        matrix,
        gap,
        db_dir=tmp_path,
        chunks=chunks,
    )

    assert len(repairs) == 1
    repair = repairs[0]
    assert repair["source_path"] == source
    assert repair["value"] == exact
    assert repair["evidence_quote"] == exact
    assert repair["block_id"] == "block-paper-a:7"
    assert repair["same_source_verified"] is True

    payload = apply_evidence_matrix_cell_repair(matrix, gap, repair, db_dir=tmp_path)
    cell = payload["rows"][0]["cells"]["limitation"]
    assert cell == {
        "field": "limitation",
        "value": exact,
        "support_status": "grounded",
        "evidence_ids": [repair["evidence_id"]],
        "manual_override": False,
        "repair_confirmed": True,
    }
    repaired_evidence = next(item for item in payload["evidence"] if item["id"] == repair["evidence_id"])
    assert repaired_evidence["source_path"] == source
    assert repaired_evidence["evidence_quote"] == exact
    assert payload["quality_status"] == "verified"
    assert payload["quality"]["missing_cell_count"] == 3
    assert payload["quality"]["unsupported_cell_count"] == 0
    assert payload["quality"]["source_watch_snapshot"]["contract_version"] == 1

    status, quality = evidence_matrix_quality(
        rows=payload["rows"],
        evidence=payload["evidence"],
        selected_items=matrix["source_items"],
    )
    assert status == "verified"
    assert {(item["row_id"], item["field"]) for item in quality["missing_cells"]} == {
        ("row-a", "dataset_or_experiment"),
        ("row-a", "metric"),
        ("row-a", "key_result"),
    }


def test_cell_repair_keeps_honest_missing_state_and_rejects_cross_source(tmp_path: Path) -> None:
    source = str(tmp_path / "paper-a.md")
    other_source = str(tmp_path / "paper-b.md")
    matrix = _matrix(source)
    gap = _gap(source)
    chunks = [
        _chunk(
            source,
            "The proposed system uses a calibration module without reporting a limitation.",
            chunk_id="paper-a:2",
            heading="Method",
        )
    ]

    assert evidence_matrix_cell_repair_candidates(
        matrix,
        gap,
        db_dir=tmp_path,
        chunks=chunks,
    ) == []

    exact = "However, our dynamic reconstruction remains limited by motion errors."
    repair = evidence_matrix_cell_repair_candidates(
        matrix,
        gap,
        db_dir=tmp_path,
        chunks=[_chunk(source, exact, chunk_id="paper-a:8")],
    )[0]
    forged = deepcopy(repair)
    forged["source_path"] = other_source
    with pytest.raises(ValueError, match="source paper"):
        apply_evidence_matrix_cell_repair(matrix, gap, forged, db_dir=tmp_path)
