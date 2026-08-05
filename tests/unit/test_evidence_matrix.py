from io import BytesIO
from pathlib import Path

from openpyxl import load_workbook

from kb.evidence_matrix import (
    MATRIX_CELL_FIELDS,
    build_project_evidence_matrix,
    evidence_matrix_csv,
    evidence_matrix_hits,
    evidence_matrix_markdown,
    evidence_matrix_quality,
    evidence_matrix_xlsx,
)


def _chunk(source: str, heading: str, text: str, index: int) -> dict:
    return {
        "id": f"chunk-{index}",
        "text": text,
        "meta": {
            "source_path": source,
            "source_name": Path(source).stem,
            "heading_path": heading,
            "page": index,
            "block_id": f"block-{index}",
        },
    }


def test_build_matrix_is_source_balanced_grounded_and_preserves_notes(monkeypatch) -> None:
    sources = ["F:/papers/a.md", "F:/papers/b.md"]
    chunks = []
    for source_index, source in enumerate(sources, start=1):
        chunks.extend(
            [
                _chunk(source, "Method", f"The method uses a coded optical network for source {source_index}.", source_index * 10 + 1),
                _chunk(source, "Experiments", f"Experiments use the Dataset-{source_index} benchmark with 100 samples.", source_index * 10 + 2),
                _chunk(source, "Metrics", f"The evaluation metric is PSNR and reaches {30 + source_index} dB.", source_index * 10 + 3),
                _chunk(source, "Results", f"Results improve reconstruction performance by {source_index + 2} percent.", source_index * 10 + 4),
                _chunk(source, "Limitations", f"A limitation remains under low-light conditions for source {source_index}.", source_index * 10 + 5),
            ]
        )
    monkeypatch.setattr("kb.evidence_matrix.load_all_chunks", lambda _db_dir: chunks)
    selected = [
        {"key": "a", "title": "Paper A", "sourcePath": sources[0]},
        {"key": "b", "title": "Paper B", "sourcePath": sources[1]},
    ]
    previous = [{"source_path": sources[0], "notes": "Keep this manual note."}]

    rows, evidence, flags = build_project_evidence_matrix(
        selected,
        objective="Compare reconstruction quality.",
        db_dir="unused",
        existing_rows=previous,
    )

    assert len(rows) == 2
    assert rows[0]["notes"] == "Keep this manual note."
    assert all(set(row["cells"]) == set(MATRIX_CELL_FIELDS) for row in rows)
    assert all(row["cells"][field]["support_status"] == "grounded" for row in rows for field in MATRIX_CELL_FIELDS)
    assert {item["source_path"] for item in evidence} == set(sources)
    assert any(item["code"] == "experimental_conditions_differ" for item in flags)

    status, quality = evidence_matrix_quality(
        rows=rows,
        evidence=evidence,
        selected_items=selected,
        comparison_flags=flags,
    )
    assert status == "verified"
    assert quality["covered_source_count"] == 2
    assert quality["supported_cell_count"] == quality["populated_cell_count"] == 10
    assert quality["unexpected_sources"] == []

    record = {"rows": rows, "evidence": evidence}
    hits = evidence_matrix_hits(record)
    assert {_hit["meta"]["source_path"] for _hit in hits[:2]} == set(sources)


def test_matrix_quality_rejects_manual_or_cross_source_cell_support() -> None:
    selected = [{"key": "a", "title": "Paper A", "sourcePath": "F:/papers/a.md"}]
    rows = [
        {
            "id": "row-a",
            "source_path": "F:/papers/a.md",
            "source_status": "active",
            "cells": {
                "method": {
                    "value": "A manually rewritten method claim.",
                    "support_status": "needs_review",
                    "evidence_ids": ["ev-1"],
                    "manual_override": True,
                }
            },
        }
    ]
    evidence = [
        {
            "id": "ev-1",
            "source_path": "F:/papers/b.md",
            "evidence_quote": "A different paper reports a method.",
        }
    ]

    status, quality = evidence_matrix_quality(
        rows=rows,
        evidence=evidence,
        selected_items=selected,
    )

    assert status == "needs_review"
    assert "unsupported_cells" in quality["reasons"]
    assert "unexpected_sources" in quality["reasons"]


def test_build_matrix_leaves_ambiguous_reference_and_positive_contrast_cells_empty(monkeypatch) -> None:
    source = "F:/papers/conservative.md"
    chunks = [
        _chunk(source, "References", "Ichioka, Thin observation module: experimental verification with 100 samples.", 1),
        _chunk(source, "Introduction", "Conventional methods employ an iterative network for reconstruction.", 2),
        _chunk(source, "Results", "Although simplified, its performance is equal to or better than baseline.", 3),
        _chunk(source, "Experiments", "Figure 7. Experimental setup of the system.", 4),
        _chunk(source, "Results", "The results are computed from images produced by state-of-the-art methods.", 5),
        _chunk(source, "Results", "Figure 4C shows examples of difference map stacks.", 6),
        _chunk(source, "Results", "Previous winning solution on the Image Deblurring Challenge Track2.", 7),
    ]
    monkeypatch.setattr("kb.evidence_matrix.load_all_chunks", lambda _db_dir: chunks)

    rows, evidence, _flags = build_project_evidence_matrix(
        [{"key": "paper", "title": "Paper", "sourcePath": source}],
        objective="Compare the proposed method and evidence.",
        db_dir="unused",
    )

    assert rows[0]["cells"]["method"]["support_status"] == "missing"
    assert rows[0]["cells"]["dataset_or_experiment"]["support_status"] == "missing"
    assert rows[0]["cells"]["metric"]["support_status"] == "missing"
    assert rows[0]["cells"]["key_result"]["support_status"] == "grounded"
    assert rows[0]["cells"]["limitation"]["support_status"] == "missing"
    assert len(evidence) == 1
    assert evidence[0]["field"] == "key_result"


def test_matrix_exports_include_matrix_and_evidence() -> None:
    row = {
        "paper": "Paper A",
        "source_name": "Paper A",
        "source_path": "F:/papers/a.md",
        "notes": "Review Figure 2.",
        "cells": {
            field: {"value": f"{field} evidence", "support_status": "grounded", "evidence_ids": [f"ev-{field}"]}
            for field in MATRIX_CELL_FIELDS
        },
    }
    record = {
        "title": "Imaging evidence matrix",
        "objective": "Compare methods.",
        "quality_status": "verified",
        "revision": 2,
        "rows": [row],
        "evidence": [
            {
                "id": "ev-method",
                "source_name": "Paper A",
                "source_path": "F:/papers/a.md",
                "field": "method",
                "heading_path": "Method / Architecture",
                "page_start": 3,
                "evidence_quote": "The method uses a coded optical network.",
            }
        ],
    }

    markdown = evidence_matrix_markdown(record)
    assert "Imaging evidence matrix" in markdown
    assert "Evidence appendix" in markdown
    assert "Method / Architecture" in markdown
    csv_payload = evidence_matrix_csv(record).decode("utf-8-sig")
    assert "Paper A" in csv_payload
    assert "Review Figure 2" in csv_payload

    workbook = load_workbook(BytesIO(evidence_matrix_xlsx(record)))
    assert workbook.sheetnames == ["Evidence Matrix", "Evidence"]
    assert workbook["Evidence Matrix"]["A2"].value == "Paper A"
    assert workbook["Evidence"]["F2"].value == "The method uses a coded optical network."


def test_matrix_tabular_exports_escape_spreadsheet_formulas() -> None:
    row = {
        "paper": "Paper A",
        "source_path": "F:/papers/a.md",
        "notes": "=HYPERLINK(\"https://example.invalid\")",
        "cells": {field: {"value": ""} for field in MATRIX_CELL_FIELDS},
    }
    record = {"rows": [row], "evidence": []}

    assert "'=HYPERLINK" in evidence_matrix_csv(record).decode("utf-8-sig")
    workbook = load_workbook(BytesIO(evidence_matrix_xlsx(record)))
    assert workbook["Evidence Matrix"]["H2"].value.startswith("'=HYPERLINK")
