from __future__ import annotations

import json
from pathlib import Path

from kb.converter.quality_acceptance import summarize_conversion_quality
from kb.converter.quality_repair import conversion_repair_strategy_for_issue, repair_markdown_quality


def test_repair_markdown_quality_fixes_safe_source_level_issues(tmp_path: Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "page_1_fig_1.png").write_bytes(b"png")
    (assets / "figure_index.json").write_text(
        json.dumps(
            {
                "figures": [
                    {
                        "asset_name": "page_1_fig_1.png",
                        "caption": "Figure 1. Experimental setup with a DMD and single-pixel detector.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    md_path = tmp_path / "paper.en.md"
    md_path.write_text(
        "\n".join(
            [
                "# Demo Paper",
                "",
                "Ada Lovelace & Grace Hopper",
                "",
                (
                    "Single-pixel imaging reconstructs scenes from structured measurements and a compact detector. "
                    "The approach improves acquisition efficiency while preserving image quality in photon-limited "
                    "settings, and the converted paper should expose this opening summary as an abstract for retrieval. "
                    "This sentence keeps the paragraph long enough to be recognized as front-matter abstract text."
                ),
                "",
                "![Figure 1](assets/page_1_fig_1.png)",
                "",
                "$$",
                "x = y",
                "",
                "## Introduction",
                "",
                "The method section starts here.",
                "",
                "## References",
                "",
                "[1] Ada Lovelace. Example reference. Journal, 2024.",
            ]
        ),
        encoding="utf-8",
    )

    before = summarize_conversion_quality(md_path)
    assert before.page_marker_count == 0
    assert before.has_abstract_heading is False
    assert before.caption_count == 0
    assert before.unclosed_display_math_block_count == 1

    result = repair_markdown_quality(
        md_path,
        issue_codes=["missing_page_markers", "missing_abstract", "missing_captions", "unclosed_display_math"],
    )

    assert result["changed"] is True
    assert "ensure_page_anchor" in result["applied"]
    assert "figure_metadata_captions" in result["applied"]
    assert "postprocess_markdown" in result["applied"]
    after_text = md_path.read_text(encoding="utf-8")
    assert after_text.lstrip().startswith("<!-- kb_page: 1 -->")
    assert "## Abstract" in after_text
    assert "**Figure 1.** Experimental setup" in after_text
    after = summarize_conversion_quality(md_path)
    assert after.page_marker_count == 1
    assert after.has_abstract_heading is True
    assert after.caption_count == 1
    assert after.unclosed_display_math_block_count == 0
    assert (tmp_path / "paper.en.md.bak").exists()


def test_conversion_repair_strategy_marks_safe_known_issue():
    strategy = conversion_repair_strategy_for_issue("missing_captions")

    assert strategy["safe"] is True
    assert "figure_metadata_captions" in strategy["strategies"]
