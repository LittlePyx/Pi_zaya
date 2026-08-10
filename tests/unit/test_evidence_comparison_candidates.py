from __future__ import annotations

from pathlib import Path

from kb.evidence_matrix import (
    audit_evidence_comparison,
    find_evidence_comparison_candidates,
)


def _row(row_id: str, paper: str, source_path: str) -> dict:
    return {
        "id": row_id,
        "source_item_key": row_id,
        "paper": paper,
        "source_name": paper,
        "source_path": source_path,
        "source_status": "active",
        "cells": {},
    }


def _metric_chunk(
    chunk_id: str,
    source_path: str,
    *,
    protocol: str,
    target: str,
    result: str,
    dataset: str = "Cozy2room",
    metric: str = "LPIPS",
) -> dict:
    return {
        "id": chunk_id,
        "text": (
            f"Table 1. Quantitative SCI image reconstruction comparisons on the {protocol}. "
            f"{dataset} {metric} (lower is better): baseline = .6031; {target} = {result}"
        ),
        "meta": {
            "source_path": source_path,
            "heading_path": "Experiments / Table 1",
            "page_start": 6,
            "block_id": f"block-{chunk_id}",
            "structured_kind": "table_metric",
            "table_metric": metric,
            "table_metric_direction": "lower",
        },
    }


def _candidate_spec(candidate: dict) -> dict:
    return {
        "mode": candidate["mode"],
        "left_row_id": candidate["left_row_id"],
        "right_row_id": candidate["right_row_id"],
        "dimensions": [
            {
                "dimension": item["dimension"],
                "left_value": item["left_value"],
                "right_value": item["right_value"],
                "mapping_confirmed": item["dimension"]
                in candidate["required_confirmations"],
            }
            for item in candidate["dimensions"]
        ],
        "left_target": candidate["left_target"],
        "right_target": candidate["right_target"],
        "left_result": candidate["left_result"],
        "right_result": candidate["right_result"],
    }


def test_comparison_candidates_prefill_only_exact_locatable_paired_evidence(
    tmp_path: Path,
) -> None:
    left_path = str(tmp_path / "SCIGS.md")
    right_path = str(tmp_path / "SCINeRF.md")
    rows = [
        _row("row-left", "SCIGS", left_path),
        _row("row-right", "SCINeRF", right_path),
    ]
    chunks = [
        _metric_chunk(
            "left-table",
            left_path,
            protocol="static datasets",
            target="SCIGS(ours)",
            result=".0423",
        ),
        _metric_chunk(
            "right-table",
            right_path,
            protocol="synthetic datasets",
            target="ours",
            result=".0445",
        ),
    ]
    matrix = {
        "id": "matrix-1",
        "revision": 4,
        "rows": rows,
        "comparison_audits": [],
        "quality": {"last_research_gap_expansion": {"new_row_id": "row-right"}},
    }

    result = find_evidence_comparison_candidates(
        matrix,
        db_dir=tmp_path,
        corpus_chunks=chunks,
    )

    assert result["examined_row_pairs"] == 1
    assert result["structured_observation_count"] == 2
    assert len(result["items"]) == 1
    candidate = result["items"][0]
    assert candidate["matrix_revision"] == 4
    assert candidate["left_target"] == "SCIGS(ours)"
    assert candidate["right_target"] == "ours"
    assert candidate["left_result"] == ".0423"
    assert candidate["right_result"] == ".0445"
    assert candidate["required_confirmations"] == ["evaluation_protocol"]
    assert [item["match_type"] for item in candidate["dimensions"]] == [
        "exact",
        "exact",
        "review_required",
        "controlled_alias",
    ]
    assert all(item["page_start"] == 6 for item in candidate["evidence"])
    assert all(item["evidence_quote"] in chunks[index]["text"] for index, item in enumerate(candidate["evidence"]))

    audit = audit_evidence_comparison(
        rows=rows,
        spec=_candidate_spec(candidate),
        db_dir=tmp_path,
        corpus_chunks=chunks,
    )

    assert audit["status"] == "verified"
    assert audit["relation"] == "left_more_favorable"
    assert audit["user_confirmed_mappings"] == ["evaluation_protocol"]


def test_comparison_candidates_reject_mismatched_dataset_and_hide_saved_pair(
    tmp_path: Path,
) -> None:
    left_path = str(tmp_path / "SCIGS.md")
    right_path = str(tmp_path / "SCINeRF.md")
    rows = [
        _row("row-left", "SCIGS", left_path),
        _row("row-right", "SCINeRF", right_path),
    ]
    left = _metric_chunk(
        "left-table",
        left_path,
        protocol="static datasets",
        target="SCIGS(ours)",
        result=".0423",
    )
    mismatch = _metric_chunk(
        "right-table",
        right_path,
        protocol="synthetic datasets",
        target="ours",
        result=".0310",
        dataset="Hotdog",
    )
    no_candidate = find_evidence_comparison_candidates(
        {"id": "matrix-1", "revision": 1, "rows": rows, "comparison_audits": []},
        db_dir=tmp_path,
        corpus_chunks=[left, mismatch],
    )
    assert no_candidate["items"] == []

    matching = _metric_chunk(
        "right-table",
        right_path,
        protocol="synthetic datasets",
        target="ours",
        result=".0445",
    )
    initial = find_evidence_comparison_candidates(
        {"id": "matrix-1", "revision": 1, "rows": rows, "comparison_audits": []},
        db_dir=tmp_path,
        corpus_chunks=[left, matching],
    )
    candidate = initial["items"][0]
    audit = audit_evidence_comparison(
        rows=rows,
        spec=_candidate_spec(candidate),
        db_dir=tmp_path,
        corpus_chunks=[left, matching],
    )
    after = find_evidence_comparison_candidates(
        {
            "id": "matrix-1",
            "revision": 2,
            "rows": rows,
            "comparison_audits": [audit],
        },
        db_dir=tmp_path,
        corpus_chunks=[left, matching],
    )
    assert after["items"] == []
