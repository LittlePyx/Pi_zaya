from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import fitz

from kb.converter.throughput_benchmark import ThroughputConfig, run_throughput_suite


def _make_pdf(path: Path) -> None:
    doc = fitz.open()
    doc.new_page()
    doc.save(path)
    doc.close()


def _write_quality(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "output.md").write_text("<!-- kb_page: 1 -->\n\nBody", encoding="utf-8")
    (output_dir / "conversion_quality_result.json").write_text(
        json.dumps(
            {
                "needs_reconvert": False,
                "recommended_action": "none",
                "metrics": {
                    "page_marker_count": 1,
                    "page_marker_gap_count": 0,
                    "missing_image_count": 0,
                    "table_literal_break_count": 0,
                    "collapsed_table_row_count": 0,
                    "ambiguous_table_break_row_count": 0,
                    "fragmented_table_column_count": 0,
                    "fragmented_table_duplicate_count": 0,
                    "unclosed_display_math_block_count": 0,
                    "prose_dominant_display_math_block_count": 0,
                    "display_math_markdown_link_count": 0,
                    "mojibake_count": 0,
                    "analyzer_error_count": 0,
                },
                "source_quality": {
                    "missing_source_page_count": 0,
                    "source_page_text_corruption_count": 0,
                    "source_page_prose_omission_count": 0,
                    "min_source_page_coverage": 0.9,
                },
                "repair_plan": {"issue_codes": []},
            }
        ),
        encoding="utf-8",
    )


def test_throughput_suite_pairs_modes_reverses_order_and_shares_fixed_budget(tmp_path: Path) -> None:
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    _make_pdf(pdf_a)
    _make_pdf(pdf_b)
    lock = threading.Lock()
    state = {"active": 0, "max_active": 0}
    calls: list[tuple[str, int, str, str, str]] = []

    def _fake_runner(**kwargs):
        pdf_path = Path(kwargs["pdf_path"])
        out_root = Path(kwargs["out_root"])
        max_active = int(kwargs["max_active_conversions"])
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            calls.append(
                (
                    pdf_path.name,
                    max_active,
                    str(os.environ.get("KB_LLM_MAX_INFLIGHT")),
                    str(os.environ.get("KB_PDF_WORKERS")),
                    str(os.environ.get("KB_PDF_LLM_WORKERS")),
                )
            )
        time.sleep(0.02)
        output_dir = out_root / pdf_path.stem
        _write_quality(output_dir)
        kwargs["progress_cb"](1, 1, "Finished page 1/1 (0.01s, 10 chars)")
        with lock:
            state["active"] -= 1
        return True, output_dir

    payload = run_throughput_suite(
        pdf_paths=[pdf_a, pdf_b],
        out_root=tmp_path / "results",
        config=ThroughputConfig(
            global_inflight=8,
            workers=4,
            llm_workers=3,
            repeat=2,
            min_throughput_improvement_pct=0.0,
            max_per_doc_p95_slowdown_pct=100.0,
        ),
        runner=_fake_runner,
        fail_fast=True,
    )

    assert [(row["mode"], row["repeat"]) for row in payload["experiments"]] == [
        ("serial", 1),
        ("parallel", 1),
        ("parallel", 2),
        ("serial", 2),
    ]
    assert len(payload["document_runs"]) == 8
    assert state["max_active"] == 2
    assert {item[2:] for item in calls} == {("8", "4", "3")}
    assert {item[1] for item in calls} == {1, 2}
    assert payload["summary"]["median_throughput_improvement_pct"] > 30.0
    assert payload["summary"]["gate_checks"]["all_runs_ok"] is True
    assert payload["summary"]["gate_checks"]["page_markers_exact"] is True
    assert (tmp_path / "results" / "throughput_results.json").exists()
    assert (tmp_path / "results" / "serial" / "run_01" / "doc_01").is_dir()
    assert (tmp_path / "results" / "serial" / "run_01" / "doc_02").is_dir()


def test_throughput_suite_requires_exactly_two_pdfs(tmp_path: Path) -> None:
    pdf_path = tmp_path / "one.pdf"
    _make_pdf(pdf_path)

    try:
        run_throughput_suite(
            pdf_paths=[pdf_path],
            out_root=tmp_path / "results",
            config=ThroughputConfig(repeat=1),
        )
    except ValueError as exc:
        assert "exactly two PDFs" in str(exc)
    else:
        raise AssertionError("expected two-PDF validation error")
