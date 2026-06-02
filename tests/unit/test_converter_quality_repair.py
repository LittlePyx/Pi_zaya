from __future__ import annotations

import json
import re
from pathlib import Path

from kb.converter.quality_acceptance import summarize_conversion_quality
from kb.converter.quality_gate import prepare_markdown_for_index
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


def test_prepare_markdown_for_index_repairs_missing_image_link_before_reconvert(tmp_path: Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "page_2_fig_1.png").write_bytes(b"png")
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
                "This paper has a usable abstract and stable front matter.",
                "",
                "<!-- kb_page: 2 -->",
                "",
                "## Results",
                "",
                "The spectrum is shown in Figure 1.",
                "",
                "![Figure](./assets/image_auto_01_p002_r2.png)",
                "",
                "Figure 1. Reflectance spectra for two coatings.",
                "",
                "## References",
                "",
                "[1] Ada Lovelace. Example reference. Journal of Testing, 2024.",
            ]
        ),
        encoding="utf-8",
    )

    before = summarize_conversion_quality(md_path)
    assert before.missing_image_count == 1

    assessment = prepare_markdown_for_index(md_path, auto_repair=True)

    assert assessment["indexable"] is True
    assert assessment["auto_repair"]["attempted"] is True
    assert "repair_missing_image_links" in assessment["auto_repair"]["applied"]
    assert "page_2_fig_1.png" in md_path.read_text(encoding="utf-8")
    after = summarize_conversion_quality(md_path)
    assert after.missing_image_count == 0


def test_plan_conversion_quality_repair_routes_source_text_loss_to_reconvert():
    plan = plan_conversion_quality_repair(["source_text_loss", "missing_abstract"])

    assert plan["action"] == "reconvert"
    assert plan["scope"] == "document"
    assert plan["speed_mode"] == "normal"
    assert "source_text_loss" in plan["reconvert_issue_codes"]
    assert "missing_abstract" in plan["autofix_issue_codes"]


def test_plan_conversion_quality_repair_keeps_safe_issues_local():
    plan = plan_conversion_quality_repair(["missing_page_markers", "missing_captions"])

    assert plan["action"] == "autofix"
    assert plan["scope"] == "markdown"
    assert plan["replace"] is False


def test_repair_markdown_text_normalizes_duplicate_or_out_of_order_page_markers(tmp_path: Path):
    md_path = tmp_path / "paper.en.md"
    original = "\n".join(
        [
            "<!-- kb_page: 1 -->",
            "# Demo Paper",
            "<!-- kb_page: 5 -->",
            "Later page text.",
            "<!-- kb_page: 4 -->",
            "Out of order page text.",
            "<!-- kb_page: 4 -->",
            "Duplicate page text.",
            "## References",
            "[1] Ada Lovelace. Example reference. Journal, 2024.",
        ]
    )
    md_path.write_text(original, encoding="utf-8")

    result = repair_markdown_text(md_path, original, issue_codes=["page_marker_gaps"])

    assert result["changed"] is True
    assert "normalize_page_markers" in result["applied"]
    assert "<!-- kb_page: 6 -->" in result["repaired_text"]
    assert "<!-- kb_page: 7 -->" in result["repaired_text"]
    after = summarize_conversion_quality(md_path, result["repaired_text"])
    assert after.page_marker_gap_count == 0


def test_repair_markdown_text_uses_table_only_fallback_for_analyzer_errors(tmp_path: Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "page_1_fig_1.png").write_bytes(b"png")
    md_path = tmp_path / "table-error.en.md"
    original = "\n".join(
        [
            "<!-- kb_page: 1 -->",
            "",
            "# Demo Paper",
            "",
            "## Abstract",
            "",
            "This paper has enough text for a stable abstract section.",
            "",
            "![Figure 1](assets/page_1_fig_1.png)",
            "",
            "**Figure 1.** Demo caption.",
            "![Figure 1](assets/page_1_fig_1.png)",
            "",
            "**Figure 1.** Demo caption.",
            "",
            "| Metric | A | B |",
            "|---|---|---|",
            "| PSNR | 1 | 2 | 3 |",
            "",
            "## References",
            "",
            "[1] Ada Lovelace. Example reference. Journal, 2024.",
        ]
    )
    md_path.write_text(original, encoding="utf-8")

    before = summarize_conversion_quality(md_path, original)
    assert before.analyzer_error_count == 1
    assert before.image_count == 2

    result = repair_markdown_text(md_path, original, issue_codes=["analyzer_errors"])

    repaired = str(result.get("repaired_text") or "")
    after = summarize_conversion_quality(md_path, repaired)
    assert result["changed"] is True
    assert result["unsafe"] is False
    assert result["applied"] == ["normalize_markdown_tables"]
    assert after.analyzer_error_count == 0
    assert after.image_count == 2
    assert "| PSNR | 1 | 2 | 3 |" in repaired


def test_repair_markdown_text_normalizes_heading_level_jumps_narrowly(tmp_path: Path):
    md_path = tmp_path / "heading-jump.en.md"
    original = "\n".join(
        [
            "<!-- kb_page: 1 -->",
            "",
            "# Demo Paper",
            "",
            "## Abstract",
            "",
            "This paper has an abstract and stable text.",
            "",
            "#### Skipped Method Heading",
            "",
            "The method section remains readable.",
            "",
            "## References",
            "",
            "[1] Ada Lovelace. Example reference. Journal, 2024.",
        ]
    )
    md_path.write_text(original, encoding="utf-8")

    before = summarize_conversion_quality(md_path, original)
    assert before.heading_level_jump_count == 1

    result = repair_markdown_text(md_path, original, issue_codes=["heading_level_jumps"])

    repaired = str(result.get("repaired_text") or "")
    after = summarize_conversion_quality(md_path, repaired)
    assert result["changed"] is True
    assert result["applied"] == ["normalize_heading_levels"]
    assert "### Skipped Method Heading" in repaired
    assert "#### Skipped Method Heading" not in repaired
    assert after.heading_level_jump_count == 0


def test_repair_markdown_text_does_not_let_postprocess_restore_numeric_heading_jump(tmp_path: Path):
    md_path = tmp_path / "numeric-heading-jump.en.md"
    original = "\n".join(
        [
            "<!-- kb_page: 1 -->",
            "",
            "# Demo Paper",
            "",
            "## 1 Introduction",
            "",
            "This paper has a stable introduction.",
            "",
            "#### 2.1.2 SPAD based on conventional bulk semiconductors",
            "",
            "The subsection is missing its parent heading in the source conversion.",
            "",
            "## References",
            "",
            "[1] Ada Lovelace. Example reference. Journal, 2024.",
        ]
    )
    md_path.write_text(original, encoding="utf-8")

    result = repair_markdown_text(md_path, original, issue_codes=["heading_level_jumps"])

    repaired = str(result.get("repaired_text") or "")
    after = summarize_conversion_quality(md_path, repaired)
    assert result["changed"] is True
    assert result["applied"] == ["normalize_heading_levels"]
    assert "### 2.1.2 SPAD based on conventional bulk semiconductors" in repaired
    assert "#### 2.1.2 SPAD based on conventional bulk semiconductors" not in repaired
    assert after.heading_level_jump_count == 0


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


def test_repair_markdown_text_recovers_real_page_markers_from_pdf(tmp_path: Path):
    import fitz

    pdf_path = tmp_path / "Demo Paper.pdf"
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text(
        (50, 80),
        "Demo Paper Title\n"
        "The first page describes the optical setup and modulation strategy for a compact detector.",
        fontsize=11,
    )
    page2 = doc.new_page()
    page2.insert_text(
        (50, 80),
        "The second page reports calibration measurements and reconstruction accuracy under low light.",
        fontsize=11,
    )
    doc.save(str(pdf_path))
    doc.close()

    md_path = tmp_path / "Demo Paper.en.md"
    original = "\n".join(
        [
            "# Demo Paper Title",
            "",
            "The first page describes the optical setup and modulation strategy for a compact detector.",
            "",
            "The second page reports calibration measurements and reconstruction accuracy under low light.",
        ]
    )
    md_path.write_text(original, encoding="utf-8")

    result = repair_markdown_text(
        md_path,
        original,
        issue_codes=["missing_page_markers"],
        source_pdf_path=pdf_path,
    )

    repaired = str(result.get("repaired_text") or "")
    assert result["changed"] is True
    assert "recover_page_markers_from_pdf" in result["applied"]
    assert "<!-- kb_page: 1 -->" in repaired
    assert "<!-- kb_page: 2 -->" in repaired
    assert repaired.index("<!-- kb_page: 1 -->") < repaired.index("The first page describes")
    assert repaired.index("<!-- kb_page: 2 -->") < repaired.index("The second page reports")


def test_repair_markdown_text_recovers_missing_caption_from_pdf_text(tmp_path: Path):
    import fitz

    pdf_path = tmp_path / "Caption Paper.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 80), "Caption Paper", fontsize=14)
    page.insert_text(
        (50, 180),
        "Figure 3. Optical layout with a modulation mask and a single-pixel detector.",
        fontsize=10,
    )
    doc.save(str(pdf_path))
    doc.close()

    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "page_1_fig_1.png").write_bytes(b"png")
    md_path = tmp_path / "Caption Paper.en.md"
    original = "\n".join(
        [
            "<!-- kb_page: 1 -->",
            "",
            "# Caption Paper",
            "",
            "![Figure 3](assets/page_1_fig_1.png)",
        ]
    )
    md_path.write_text(original, encoding="utf-8")

    result = repair_markdown_text(
        md_path,
        original,
        issue_codes=["missing_captions"],
        source_pdf_path=pdf_path,
    )

    repaired = str(result.get("repaired_text") or "")
    assert result["changed"] is True
    assert "pdf_text_captions" in result["applied"]
    assert "**Figure 3.** Optical layout with a modulation mask" in repaired


def test_repair_markdown_text_inserts_abstract_heading_without_full_postprocess(tmp_path: Path):
    md_path = tmp_path / "Nature Style.en.md"
    original = "\n".join(
        [
            "# Nature Style Paper",
            "",
            "Ada Lovelace and Alan Turing",
            "",
            (
                "Solution-processed semiconductor lasers promise lightweight and scalable optoelectronic "
                "applications. This work demonstrates an integrated device architecture with stable emission "
                "and reports the measurements needed to evaluate its performance under electrical driving."
            ),
            "",
            "## Structure of the integrated device",
            "",
            "The structure is shown in Fig. 1.",
        ]
    )
    md_path.write_text(original, encoding="utf-8")

    result = repair_markdown_text(md_path, original, issue_codes=["missing_abstract"])

    repaired = str(result.get("repaired_text") or "")
    assert result["changed"] is True
    assert "abstract_heading_only" in result["applied"]
    assert "## Abstract" in repaired
    assert repaired.index("## Abstract") < repaired.index("Solution-processed semiconductor lasers")
    assert repaired.index("## Abstract") < repaired.index("## Structure of the integrated device")


def test_repair_markdown_text_moves_early_references_after_body_sections(tmp_path: Path):
    md_path = tmp_path / "Misordered References.en.md"
    original = "\n".join(
        [
            "<!-- kb_page: 1 -->",
            "",
            "# Misordered References",
            "",
            "## References",
            "",
            "[1] Ada Lovelace. Example Journal, 2024.",
            "[2] Alan Turing. Proceedings of Tests, 2025.",
            "[3] Grace Hopper. Computer Review, 2026.",
            "",
            "## 2. Comparison of theory",
            "",
            "The actual body section was recovered after the reference list.",
        ]
    )
    md_path.write_text(original, encoding="utf-8")

    result = repair_markdown_text(md_path, original, issue_codes=["references_before_body"])

    repaired = str(result.get("repaired_text") or "")
    assert result["changed"] is True
    assert "move_early_references_to_end" in result["applied"]
    assert repaired.index("## 2. Comparison of theory") < repaired.index("## References")
    assert "[1] Ada Lovelace" in repaired


def test_write_conversion_quality_result_marks_supplementary_abstract_not_applicable(tmp_path: Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "page_1_fig_1.png").write_bytes(b"png")
    md_path = tmp_path / "Demo supplement.en.md"
    md_path.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "",
                "# Demo Paper Supplementary Material",
                "",
                "Supplement DOI: https://doi.org/10.1234/example",
                "",
                "## 1. Fabrication of the mask",
                "",
                "The supplementary document describes fabrication details.",
                "",
                "![Figure](assets/page_1_fig_1.png)",
                "",
                "Fig. S1. Reflectance of the supplementary spinning mask coating.",
            ]
        ),
        encoding="utf-8",
    )

    payload = write_conversion_quality_result(md_path)

    assert payload["source_quality"]["document_type"] == "supplementary"
    assert payload["source_quality"]["abstract_not_applicable"] is True
    assert "missing_abstract" not in payload["repair_plan"]["issue_codes"]
    assert "missing_references" not in payload["repair_plan"]["issue_codes"]
    assert "missing_captions" not in payload["repair_plan"]["issue_codes"]


def test_write_conversion_quality_result_does_not_treat_review_journal_name_as_review(tmp_path: Path):
    md_path = tmp_path / "Psychological Review-1954-Some informational aspects of visual perception.en.md"
    md_path.write_text(
        "\n".join(
            [
                "# Some informational aspects of visual perception",
                "",
                "## Abstract",
                "This paper studies informational aspects of visual perception.",
                "",
                "## References",
                "[1] A. Author. Example reference. Journal, 1953.",
            ]
        ),
        encoding="utf-8",
    )

    payload = write_conversion_quality_result(md_path)

    assert payload["source_quality"]["document_type"] == "research_article"


def test_write_conversion_quality_result_marks_review_when_title_says_review(tmp_path: Path):
    md_path = tmp_path / "Visual Computing-2019-Brief review of computational imaging techniques.en.md"
    md_path.write_text(
        "\n".join(
            [
                "# Brief review of computational imaging techniques",
                "",
                "## Abstract",
                "This brief review summarizes computational imaging techniques.",
                "",
                "## References",
                "[1] A. Author. Example reference. Journal, 2018.",
            ]
        ),
        encoding="utf-8",
    )

    payload = write_conversion_quality_result(md_path)

    assert payload["source_quality"]["document_type"] == "review"


def test_write_conversion_quality_result_flags_source_text_loss_from_pdf(tmp_path: Path):
    import fitz

    pdf_path = tmp_path / "Long Paper.pdf"
    doc = fitz.open()
    for page_index in range(6):
        page = doc.new_page()
        page.insert_text(
            (50, 80),
            (
                f"Long Paper page {page_index + 1}. "
                "This page contains method details, experiments, measurements, and discussion. "
                "The source PDF has substantial body text that should appear in Markdown. "
            )
            * 8,
            fontsize=10,
        )
    doc.save(str(pdf_path))
    doc.close()

    md_path = tmp_path / "Long Paper.en.md"
    md_path.write_text(
        "\n".join(
            [
                "# Long Paper",
                "",
                "## References",
                "",
                "[1] Ada Lovelace. Example Journal, 2024.",
                "[2] Alan Turing. Proceedings of Tests, 2025.",
                "[3] Grace Hopper. Computer Review, 2026.",
                "[4] Katherine Johnson. Aerospace Notes, 2027.",
                "[5] Mary Jackson. Engineering Reports, 2028.",
            ]
        ),
        encoding="utf-8",
    )

    payload = write_conversion_quality_result(md_path, source_pdf_path=pdf_path)

    assert payload["source_quality"]["source_text_loss"] is True
    assert payload["repair_plan"]["action"] == "reconvert"
    assert "source_text_loss" in payload["repair_plan"]["reconvert_issue_codes"]


def test_write_conversion_quality_result_flags_missing_source_page_from_pdf(tmp_path: Path):
    import fitz

    pdf_path = tmp_path / "Paged Paper.pdf"
    doc = fitz.open()
    page_texts = [
        " ".join(f"alpha{i:02d}" for i in range(90)),
        " ".join(f"bravo{i:02d}" for i in range(90)),
        " ".join(f"charlie{i:02d}" for i in range(90)),
    ]
    for text in page_texts:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(40, 60, 560, 760), text, fontsize=10)
    doc.save(str(pdf_path))
    doc.close()

    md_path = tmp_path / "Paged Paper.en.md"
    md_text = "\n\n".join(
        [
            "<!-- kb_page: 1 -->",
            "# Paged Paper",
            page_texts[0],
            "<!-- kb_page: 3 -->",
            page_texts[2],
            "## References",
            "[1] Ada Lovelace. Example Journal, 2024.",
        ]
    )
    md_path.write_text(md_text, encoding="utf-8")

    payload = write_conversion_quality_result(md_path, source_pdf_path=pdf_path)

    assert payload["source_quality"]["missing_source_page_count"] == 1
    assert payload["source_quality"]["missing_source_pages"][0]["page"] == 2
    assert payload["repair_plan"]["action"] == "autofix"
    assert "missing_source_pages" in payload["repair_plan"]["autofix_issue_codes"]


def test_repair_markdown_text_recovers_missing_source_page_from_pdf(tmp_path: Path):
    import fitz

    pdf_path = tmp_path / "Paged Paper.pdf"
    doc = fitz.open()
    page_texts = [
        " ".join(f"alpha{i:02d}" for i in range(90)),
        " ".join(f"bravo{i:02d}" for i in range(90)),
        " ".join(f"charlie{i:02d}" for i in range(90)),
    ]
    for text in page_texts:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(40, 60, 560, 760), text, fontsize=10)
    doc.save(str(pdf_path))
    doc.close()

    md_path = tmp_path / "Paged Paper.en.md"
    original = "\n\n".join(
        [
            "<!-- kb_page: 1 -->",
            "# Paged Paper",
            page_texts[0],
            "<!-- kb_page: 3 -->",
            page_texts[2],
            "## References",
            "[1] Ada Lovelace. Example Journal, 2024.",
        ]
    )
    md_path.write_text(original, encoding="utf-8")

    result = repair_markdown_text(
        md_path,
        original,
        issue_codes=["missing_source_pages"],
        source_pdf_path=pdf_path,
    )

    repaired = str(result.get("repaired_text") or "")
    assert result["changed"] is True
    assert "recover_missing_source_pages" in result["applied"]
    assert "<!-- kb_page: 2 -->" in repaired
    assert "bravo00 bravo01 bravo02" in repaired
    assert repaired.index("<!-- kb_page: 2 -->") < repaired.index("<!-- kb_page: 3 -->")
    assert "missing_source_pages" not in result["remaining_issue_codes"]


def test_repair_markdown_text_accepts_source_backfill_with_minor_warning(tmp_path: Path):
    import fitz

    pdf_path = tmp_path / "Paged Paper.pdf"
    doc = fitz.open()
    page_texts = [
        " ".join(f"alpha{i:02d}" for i in range(90)),
        "ACM Trans. " + " ".join(f"bravo{i:02d}" for i in range(160)),
        " ".join(f"charlie{i:02d}" for i in range(90)),
    ]
    for text in page_texts:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(40, 60, 560, 760), text, fontsize=10)
    doc.save(str(pdf_path))
    doc.close()

    md_path = tmp_path / "Paged Paper.en.md"
    original = "\n\n".join(
        [
            "<!-- kb_page: 1 -->",
            "# Paged Paper",
            page_texts[0],
            "<!-- kb_page: 3 -->",
            page_texts[2],
            "## References",
            "[1] Ada Lovelace. Example Journal, 2024.",
        ]
    )
    md_path.write_text(original, encoding="utf-8")

    result = repair_markdown_text(
        md_path,
        original,
        issue_codes=["missing_source_pages"],
        source_pdf_path=pdf_path,
    )

    repaired = str(result.get("repaired_text") or "")
    assert result["changed"] is True
    assert result["regression_reasons"] == []
    assert "<!-- kb_page: 2 -->" in repaired
    assert "ACM Trans. bravo00 bravo01 bravo02" in repaired
    assert "missing_source_pages" not in result["remaining_issue_codes"]


def test_repair_markdown_text_inserts_post_reference_pages_after_reference_markers(tmp_path: Path):
    import fitz

    pdf_path = tmp_path / "Nature Layout Paper.pdf"
    page_texts = {
        page: f"page{page:02d} " + " ".join(f"token{page:02d}{idx:03d}" for idx in range(90))
        for page in range(1, 14)
    }
    doc = fitz.open()
    for page in range(1, 14):
        pdf_page = doc.new_page()
        pdf_page.insert_textbox(fitz.Rect(40, 60, 560, 760), page_texts[page], fontsize=10)
    doc.save(str(pdf_path))
    doc.close()

    md_path = tmp_path / "Nature Layout Paper.en.md"
    original_parts = []
    for page in range(1, 8):
        original_parts.extend([f"<!-- kb_page: {page} -->", page_texts[page]])
    original_parts.extend(
        [
            "## References",
            f"[1] Reference line one, Journal, 2024. {page_texts[9]} <!-- kb_page: 9 -->",
            f"[2] Reference line two, Journal, 2025. {page_texts[10]} <!-- kb_page: 10 -->",
            "<!-- kb_page: 13 -->",
            page_texts[13],
        ]
    )
    original = "\n\n".join(original_parts)
    md_path.write_text(original, encoding="utf-8")

    result = repair_markdown_text(
        md_path,
        original,
        issue_codes=["missing_source_pages"],
        source_pdf_path=pdf_path,
    )
    repaired = str(result.get("repaired_text") or "")
    markers = [int(match.group(1)) for match in re.finditer(r"<!--\s*kb_page:\s*(\d+)\s*-->", repaired)]

    assert result["changed"] is True
    assert markers == list(range(1, 14))
    assert "page08 token08000" in repaired
    assert "page11 token11000" in repaired
    assert "page12 token12000" in repaired
    assert "page_marker_gaps" not in result["remaining_issue_codes"]
    assert "missing_source_pages" not in result["remaining_issue_codes"]


def test_repair_markdown_text_backfills_truncated_references_from_pdf(tmp_path: Path):
    import fitz

    pdf_path = tmp_path / "Reference Paper.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(
        fitz.Rect(40, 60, 560, 760),
        "\n".join(
            [
                "Final body text should not be part of the reference section.",
                "REFERENCES",
                "1. ALPHA, A. First recovered reference. Journal, 1950, 1, 1-2.",
                "2. BETA, B. Second recovered reference. Journal, 1951, 2, 3-4.",
                "3. GAMMA, C. Third recovered reference. Journal, 1952, 3, 5-6.",
                "4. DELTA, D. Fourth recovered reference. Journal, 1953, 4, 7-8.",
                "5. EPSILON, E. Fifth recovered reference. Journal, 1954, 5, 9-10.",
            ]
        ),
        fontsize=10,
    )
    doc.save(str(pdf_path))
    doc.close()

    md_path = tmp_path / "Reference Paper.en.md"
    original = "\n".join(
        [
            "<!-- kb_page: 1 -->",
            "# Reference Paper",
            "## Abstract",
            "This paper has a stable abstract.",
            "## References",
            "[1] ALPHA, A. First broken reference. Journal, 1950, 1, 1-2.",
            "### RUNNING HEADER",
            "[2] This line should be replaced by the PDF reference backfill.",
        ]
    )
    md_path.write_text(original, encoding="utf-8")

    result = repair_markdown_text(
        md_path,
        original,
        issue_codes=["reference_index_truncated"],
        source_pdf_path=pdf_path,
    )

    repaired = str(result.get("repaired_text") or "")
    assert result["changed"] is True
    assert "pdf_reference_backfill" in result["applied"]
    assert "Final body text should not be part" not in repaired
    assert "RUNNING HEADER" not in repaired
    assert "[5] EPSILON" in repaired


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
