from __future__ import annotations

import json
from pathlib import Path

from kb.converter.quality_acceptance import summarize_conversion_quality
from kb.converter.quality_repair import (
    append_conversion_repair_attempt,
    conversion_quality_result_path,
    conversion_repair_strategy_for_issue,
    load_conversion_quality_result,
    plan_conversion_quality_repair,
    repair_markdown_quality,
    repair_markdown_text,
    write_conversion_quality_result,
)


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


def test_plan_conversion_quality_repair_routes_source_issues_to_reconvert():
    plan = plan_conversion_quality_repair(["unclosed_display_math", "missing_references", "weak_structure"])

    assert plan["action"] == "reconvert"
    assert plan["scope"] == "document"
    assert plan["speed_mode"] == "normal"
    assert plan["md_autofix_first"] is True
    assert "missing_references" in plan["reconvert_issue_codes"]
    assert "unclosed_display_math" in plan["autofix_issue_codes"]


def test_plan_conversion_quality_repair_keeps_safe_issues_local():
    plan = plan_conversion_quality_repair(["missing_page_markers", "missing_captions"])

    assert plan["action"] == "autofix"
    assert plan["scope"] == "markdown"
    assert plan["replace"] is False


def test_repair_markdown_text_fixes_safe_issues_without_writing(tmp_path: Path):
    md_path = tmp_path / "paper.en.md"
    original = "\n".join(
        [
            "# Demo Paper",
            "",
            "## Abstract",
            "",
            "This converted paper has an abstract.",
            "",
            "$$",
            "x = y",
        ]
    )
    md_path.write_text(original, encoding="utf-8")

    result = repair_markdown_text(
        md_path,
        original,
        issue_codes=["missing_page_markers", "unclosed_display_math"],
    )

    repaired = str(result.get("repaired_text") or "")
    assert result["changed"] is True
    assert "ensure_page_anchor" in result["applied"]
    assert "balance_display_math" in result["applied"]
    assert repaired.lstrip().startswith("<!-- kb_page: 1 -->")
    assert repaired.rstrip().endswith("$$")
    assert md_path.read_text(encoding="utf-8") == original


def test_write_conversion_quality_result_records_repair_trace(tmp_path: Path):
    md_path = tmp_path / "paper.en.md"
    md_path.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "",
                "# Demo Paper",
                "",
                "## Abstract",
                "",
                "This paper cites prior work [1].",
                "",
                "## References",
                "",
                "[1] Ada Lovelace. Example reference. Journal.",
            ]
        ),
        encoding="utf-8",
    )

    payload = write_conversion_quality_result(
        md_path,
        auto_repair_result={
            "changed": True,
            "applied": ["ensure_page_anchor"],
            "issue_codes_before": ["missing_page_markers"],
            "issue_codes_after": [],
            "remaining_issue_codes": [],
        },
    )

    report_path = conversion_quality_result_path(md_path)
    loaded = load_conversion_quality_result(md_path)
    assert report_path.exists()
    assert payload["auto_repair"]["changed"] is True
    assert payload["auto_repair"]["applied"] == ["ensure_page_anchor"]
    assert payload["repair_plan"]["action"] == "none"
    assert payload["recommended_action"] == "none"
    assert loaded["md_size"] == md_path.stat().st_size


def test_conversion_quality_result_preserves_repair_attempt_history(tmp_path: Path):
    md_path = tmp_path / "paper.en.md"
    md_path.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "",
                "# Demo Paper",
                "",
                "## Abstract",
                "",
                "This paper cites prior work [1].",
                "",
                "## References",
                "",
                "[1] Ada Lovelace. Example reference. Journal.",
            ]
        ),
        encoding="utf-8",
    )

    write_conversion_quality_result(md_path)
    row = append_conversion_repair_attempt(
        md_path,
        event="reconvert_queued",
        status="queued",
        action="reconvert",
        scope="document",
        speed_mode="normal",
        issue_codes=["weak_structure"],
        task_id="task-1",
        source="test",
        reason="Need source repair.",
    )
    payload = write_conversion_quality_result(md_path)

    assert row["task_id"] == "task-1"
    assert payload["latest_repair_attempt"]["task_id"] == "task-1"
    assert payload["repair_attempts"][-1]["event"] == "reconvert_queued"
