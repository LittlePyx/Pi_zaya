from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from kb.evidence_matrix import (
    apply_evidence_matrix_source_expansion,
    evidence_matrix_source_expansion_preview,
)


def _chunk(source: str, text: str, *, chunk_id: str) -> dict:
    return {
        "id": chunk_id,
        "text": text,
        "meta": {
            "source_path": source,
            "heading_path": "Method / Experiments / Discussion",
            "page_start": 4,
            "page_end": 4,
            "block_id": f"block-{chunk_id}",
            "evidence_ready": True,
        },
    }


def _matrix(source: str) -> dict:
    method = "We propose an optical reconstruction network."
    return {
        "id": "matrix-1",
        "project_id": "project-1",
        "title": "Imaging evidence",
        "objective": "Compare optical reconstruction methods, experiments, metrics, results, and limitations.",
        "revision": 3,
        "rows": [
            {
                "id": "row-a",
                "source_item_key": "paper-a",
                "paper": "Paper A",
                "source_name": "paper-a.md",
                "source_path": source,
                "source_status": "active",
                "cells": {
                    "method": {
                        "field": "method",
                        "value": method,
                        "support_status": "grounded",
                        "evidence_ids": ["ev-a-method"],
                        "manual_override": False,
                    }
                },
            }
        ],
        "evidence": [
            {
                "id": "ev-a-method",
                "field": "method",
                "source_item_key": "paper-a",
                "source_path": source,
                "source_name": "paper-a.md",
                "evidence_quote": method,
            }
        ],
        "source_items": [
            {
                "key": "paper-a",
                "title": "Paper A",
                "sourceName": "paper-a.md",
                "sourcePath": source,
            }
        ],
        "comparison_flags": [],
        "comparison_audits": [],
        "quality_status": "verified",
        "quality": {"contract_version": 2},
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


def test_cross_source_expansion_adds_a_new_grounded_row_without_filling_original_gap(
    tmp_path: Path,
) -> None:
    source_a = str(tmp_path / "paper-a.md")
    source_b = str(tmp_path / "paper-b.md")
    text = " ".join(
        (
            "We propose a coded aperture reconstruction method using a compact neural network.",
            "Experiments use a dynamic imaging benchmark with simulated and real measurements.",
            "We report PSNR and SSIM metrics for quantitative evaluation.",
            "The proposed method improves PSNR by 2.4 dB and outperforms the baseline.",
            "However, reconstruction remains limited by calibration errors and rapid motion.",
        )
    )
    chunks = [_chunk(source_b, text, chunk_id="paper-b:4")]
    matrix = _matrix(source_a)
    gap = _gap(source_a)
    candidate = {
        "id": "candidate-b",
        "gap_key": gap["gap_key"],
        "source_path": source_b,
        "source_name": "paper-b.md",
        "title": "Paper B",
        "chunk_id": "paper-b:4",
        "evidence_quote": text,
        "heading_path": "Method / Experiments / Discussion",
        "page_start": 4,
        "page_end": 4,
        "block_id": "block-paper-b:4",
        "anchor_id": "",
    }
    source_item = {
        "key": "research-gap:candidate-b",
        "title": "Paper B",
        "sourceName": "paper-b.md",
        "sourcePath": source_b,
        "shelfExcerpt": text,
        "headingPath": "Method / Experiments / Discussion",
        "pageStart": 4,
        "blockId": "block-paper-b:4",
    }
    original_rows = deepcopy(matrix["rows"])
    original_evidence = deepcopy(matrix["evidence"])

    preview = evidence_matrix_source_expansion_preview(
        matrix,
        gap,
        candidate,
        source_item,
        db_dir=tmp_path,
        chunks=chunks,
    )

    assert preview["matrix_revision"] == 3
    assert preview["row"]["source_path"] == source_b
    assert preview["quality_status"] == "verified"
    assert preview["grounded_fields"] == [
        "method",
        "dataset_or_experiment",
        "metric",
        "limitation",
    ]
    assert preview["missing_fields"] == ["key_result"]
    assert matrix["rows"] == original_rows
    assert matrix["evidence"] == original_evidence

    payload = apply_evidence_matrix_source_expansion(
        matrix,
        gap,
        preview,
        db_dir=tmp_path,
    )

    assert payload["quality_status"] == "verified"
    assert payload["rows"][0] == original_rows[0]
    assert payload["evidence"][0] == original_evidence[0]
    assert payload["rows"][1]["source_path"] == source_b
    assert payload["rows"][0]["cells"].get("limitation") is None
    assert payload["source_items"][1]["sourcePath"] == source_b
    assert payload["preserved_row_count"] == 1
    assert payload["quality"]["last_research_gap_expansion"]["new_row_id"] == payload["new_row_id"]


def test_cross_source_expansion_rejects_unconfirmed_or_duplicate_source(tmp_path: Path) -> None:
    source_a = str(tmp_path / "paper-a.md")
    source_b = str(tmp_path / "paper-b.md")
    text = "We propose a reconstruction method and report PSNR results on a benchmark."
    matrix = _matrix(source_a)
    gap = _gap(source_a)
    candidate = {
        "id": "candidate-b",
        "gap_key": gap["gap_key"],
        "source_path": source_b,
        "chunk_id": "paper-b:2",
        "evidence_quote": text,
        "heading_path": "Results",
        "page_start": 2,
    }
    wrong_source_item = {
        "key": "research-gap:candidate-b",
        "title": "Paper A",
        "sourceName": "paper-a.md",
        "sourcePath": source_a,
    }
    with pytest.raises(ValueError, match="does not match"):
        evidence_matrix_source_expansion_preview(
            matrix,
            gap,
            candidate,
            wrong_source_item,
            db_dir=tmp_path,
            chunks=[_chunk(source_b, text, chunk_id="paper-b:2")],
        )

    duplicate = dict(candidate)
    duplicate["source_path"] = source_a
    duplicate["chunk_id"] = "paper-a:2"
    duplicate_item = {
        "key": "research-gap:candidate-a",
        "title": "Paper A",
        "sourceName": "paper-a.md",
        "sourcePath": source_a,
    }
    with pytest.raises(ValueError, match="already a matrix source"):
        evidence_matrix_source_expansion_preview(
            matrix,
            gap,
            duplicate,
            duplicate_item,
            db_dir=tmp_path,
            chunks=[_chunk(source_a, text, chunk_id="paper-a:2")],
        )
