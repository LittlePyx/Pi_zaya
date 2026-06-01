from __future__ import annotations

import json
from pathlib import Path

from kb.converter.quality_acceptance import (
    evaluate_conversion_quality,
    load_quality_manifest,
    summarize_conversion_quality,
)


def _write_sample_markdown(path: Path, *, image_target: str = "assets/fig.png") -> None:
    path.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "",
                "# Demo Paper",
                "",
                "## Abstract",
                "",
                "This abstract cites prior work [1-2].",
                "",
                "## Method",
                "",
                f"![Figure 1]({image_target})",
                "",
                "**Figure 1.** A compact diagram.",
                "",
                "| A | B |",
                "| --- | --- |",
                "| 1 | 2 |",
                "",
                "$$",
                "x = y",
                "$$",
                "",
                "Inline math uses $a+b$.",
                "",
                "## References",
                "",
                "[1] Ada Lovelace. Example reference. Journal.",
                "[2] Grace Hopper. Another example reference. Journal.",
            ]
        ),
        encoding="utf-8",
    )


def test_summarize_conversion_quality_counts_research_paper_surfaces(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "fig.png").write_bytes(b"png")
    md_path = tmp_path / "paper.md"
    _write_sample_markdown(md_path)

    metrics = summarize_conversion_quality(md_path)

    assert metrics.has_abstract_heading is True
    assert metrics.page_marker_count == 1
    assert metrics.image_count == 1
    assert metrics.missing_image_count == 0
    assert metrics.caption_count == 1
    assert metrics.table_block_count == 1
    assert metrics.display_math_block_count == 1
    assert metrics.unclosed_display_math_block_count == 0
    assert metrics.inline_math_count == 1
    assert metrics.reference_line_count == 2
    assert metrics.extracted_reference_count == 2
    assert metrics.body_citation_expanded_index_count == 2


def test_page_marker_quality_allows_textless_pdf_page_skips(tmp_path):
    md_path = tmp_path / "paper.md"
    md_path.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "# Paper",
                "<!-- kb_page: 2 -->",
                "Text before an image-only source page.",
                "<!-- kb_page: 5 -->",
                "Text resumes after skipped source pages.",
            ]
        ),
        encoding="utf-8",
    )

    metrics = summarize_conversion_quality(md_path)

    assert metrics.page_marker_count == 3
    assert metrics.page_marker_gap_count == 0


def test_page_marker_quality_flags_duplicate_or_out_of_order_markers(tmp_path):
    md_path = tmp_path / "paper.md"
    md_path.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "# Paper",
                "<!-- kb_page: 3 -->",
                "Later text.",
                "<!-- kb_page: 3 -->",
                "Duplicate marker.",
                "<!-- kb_page: 2 -->",
                "Out of order marker.",
            ]
        ),
        encoding="utf-8",
    )

    metrics = summarize_conversion_quality(md_path)

    assert metrics.page_marker_gap_count == 2


def test_evaluate_conversion_quality_accepts_good_markdown(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "fig.png").write_bytes(b"png")
    md_path = tmp_path / "paper.md"
    _write_sample_markdown(md_path)

    result = evaluate_conversion_quality(
        md_path,
        checks={
            "min_chars": 100,
            "min_headings": 3,
            "require_abstract_heading": True,
            "min_images": 1,
            "min_captions": 1,
            "min_tables": 1,
            "min_display_math": 1,
            "min_inline_math": 1,
            "min_references": 2,
            "min_body_citations": 1,
            "max_missing_images": 0,
            "max_unclosed_display_math": 0,
            "max_mojibake": 0,
            "must_contain_text": ["Demo Paper"],
            "must_start_with": ["<!-- kb_page: 1 -->"],
            "must_not_start_with": ["# Method"],
            "ordered_text": ["# Demo Paper", "## Abstract", "## Method", "## References"],
        },
    )

    assert result["ok"] is True
    assert result["failures"] == []


def test_evaluate_conversion_quality_flags_broken_markdown(tmp_path):
    md_path = tmp_path / "broken.md"
    md_path.write_text(
        "\n".join(
            [
                "# Broken Paper",
                "",
                "![missing](assets/missing.png)",
                "",
                "$$",
                "x = y",
                "",
                "\u951b",
                "",
                "## References",
                "[1] Only reference.",
            ]
        ),
        encoding="utf-8",
    )

    result = evaluate_conversion_quality(
        md_path,
        checks={
            "require_abstract_heading": True,
            "min_images": 2,
            "min_references": 2,
            "max_missing_images": 0,
            "max_unclosed_display_math": 0,
            "max_mojibake": 0,
            "must_not_start_with": ["# Broken Paper"],
            "ordered_text": ["## Abstract", "## References"],
        },
    )

    assert result["ok"] is False
    assert "missing_abstract_heading" in result["failures"]
    assert "image_count:1<2" in result["failures"]
    assert "extracted_reference_count:1<2" in result["failures"]
    assert "missing_image_count:1>0" in result["failures"]
    assert "unclosed_display_math_block_count:1>0" in result["failures"]
    assert "mojibake_count:1>0" in result["failures"]
    assert "forbidden_prefix_present:# Broken Paper" in result["failures"]
    assert "ordered_text_missing:## Abstract" in result["failures"]


def test_load_quality_manifest_merges_defaults_and_resolves_paths(tmp_path):
    md_path = tmp_path / "paper.md"
    _write_sample_markdown(md_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_id": "demo_suite",
                "defaults": {"min_chars": 10, "max_missing_images": 0},
                "cases": [
                    {
                        "id": "demo",
                        "md_path": "paper.md",
                        "checks": {"min_headings": 3},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_quality_manifest(manifest_path, repo_root=tmp_path)
    case = manifest["cases"][0]

    assert manifest["suite_id"] == "demo_suite"
    assert case["_exists"] is True
    assert Path(case["_md_abspath"]) == md_path
    assert case["checks"]["min_chars"] == 10
    assert case["checks"]["min_headings"] == 3
    assert case["checks"]["max_missing_images"] == 0
