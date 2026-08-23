from __future__ import annotations

import json
from pathlib import Path

from kb.converter.quality_gate import (
    assess_markdown_index_quality,
    index_quality_document_fields,
    prepare_markdown_for_index,
)
from kb.converter.quality_repair import (
    CONVERSION_QUALITY_RULES_VERSION,
    conversion_quality_result_path,
    load_conversion_quality_result,
)


def _good_markdown() -> str:
    return "\n".join(
        [
            "<!-- kb_page: 1 -->",
            "",
            "# Demo Paper",
            "",
            "## Abstract",
            "",
            "This paper studies a compact imaging method and cites prior work [1].",
            "",
            "## Method",
            "",
            "The method section contains enough prose for retrieval.",
            "",
            "## References",
            "",
            "[1] Ada Lovelace. Example reference. Journal of Testing, 2024.",
        ]
    )


def test_quality_gate_accepts_ready_markdown(tmp_path: Path):
    md_path = tmp_path / "paper.en.md"
    md_path.write_text(_good_markdown(), encoding="utf-8")

    result = prepare_markdown_for_index(md_path)

    assert result["indexable"] is True
    assert result["status"] == "ready"
    assert result["action"] == "none"
    assert conversion_quality_result_path(md_path).exists()


def test_quality_gate_refreshes_same_stat_report_from_older_rules(tmp_path: Path):
    md_path = tmp_path / "paper.en.md"
    md_path.write_text(_good_markdown(), encoding="utf-8")
    prepare_markdown_for_index(md_path)

    report_path = conversion_quality_result_path(md_path)
    legacy = json.loads(report_path.read_text(encoding="utf-8"))
    legacy.pop("quality_rules_version", None)
    report_path.write_text(json.dumps(legacy), encoding="utf-8")

    result = assess_markdown_index_quality(md_path, refresh_stale=True)
    refreshed = load_conversion_quality_result(md_path)

    assert result["indexable"] is True
    assert refreshed["quality_rules_version"] == CONVERSION_QUALITY_RULES_VERSION


def test_quality_gate_autofixes_safe_issue_before_indexing(tmp_path: Path):
    md_path = tmp_path / "paper.en.md"
    md_path.write_text(_good_markdown().replace("<!-- kb_page: 1 -->\n\n", ""), encoding="utf-8")

    result = prepare_markdown_for_index(md_path, auto_repair=True)

    assert result["indexable"] is True
    assert result["status"] == "ready"
    assert result["auto_repair"]["changed"] is True
    assert md_path.read_text(encoding="utf-8").lstrip().startswith("<!-- kb_page: 1 -->")


def test_quality_gate_unwraps_prose_captured_as_display_math(tmp_path: Path):
    md_path = tmp_path / "prose-math.en.md"
    md_path.write_text(
        _good_markdown().replace(
            "The method section contains enough prose for retrieval.",
            "\n".join(
                [
                    "$$",
                    (
                        r"about 1.4 \text{ Airy units (AU, with } 1\,\text{AU} = 1.22\,\lambda/(2\,\text{NA})"
                        r"\text{), which for our parameters } \lambda = 445\,\text{nm},\ \text{NA} = 1.4"
                    ),
                    "$$",
                ]
            ),
        ),
        encoding="utf-8",
    )

    result = prepare_markdown_for_index(md_path, auto_repair=True)

    repaired = md_path.read_text(encoding="utf-8")
    assert result["indexable"] is True
    assert result["status"] == "ready"
    assert result["auto_repair"]["changed"] is True
    assert "about 1.4 Airy units" in repaired
    assert "$$" not in repaired


def test_quality_gate_recovers_missing_source_pages_before_indexing(tmp_path: Path):
    import fitz

    pdf_path = tmp_path / "paged.pdf"
    page_texts = [
        " ".join(f"alpha{i:02d}" for i in range(90)),
        " ".join(f"bravo{i:02d}" for i in range(90)),
        " ".join(f"charlie{i:02d}" for i in range(90)),
    ]
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(40, 60, 560, 760), text, fontsize=10)
    doc.save(str(pdf_path))
    doc.close()

    md_path = tmp_path / "paged.en.md"
    md_path.write_text(
        "\n\n".join(
            [
                "<!-- kb_page: 1 -->",
                "# Paged Paper",
                "## Abstract",
                "This paper studies source page recovery and cites prior work [1].",
                page_texts[0],
                "<!-- kb_page: 3 -->",
                "## Method",
                page_texts[2],
                "## References",
                "[1] Ada Lovelace. Example reference. Journal of Testing, 2024.",
            ]
        ),
        encoding="utf-8",
    )

    result = prepare_markdown_for_index(md_path, auto_repair=True, source_pdf_path=pdf_path)

    repaired = md_path.read_text(encoding="utf-8")
    assert result["indexable"] is True
    assert result["status"] == "ready"
    assert result["auto_repair"]["changed"] is True
    assert "<!-- kb_page: 2 -->" in repaired
    assert "bravo00 bravo01 bravo02" in repaired


def test_quality_gate_blocks_persistent_critical_autofix_issue(tmp_path: Path):
    md_path = tmp_path / "persistent.en.md"
    md_path.write_text(_good_markdown(), encoding="utf-8")

    result = assess_markdown_index_quality(
        md_path,
        quality_result={
            "repair_plan": {
                "action": "autofix",
                "scope": "markdown",
                "reason": "Missing source pages remain after deterministic repair.",
                "issue_codes": ["missing_source_pages"],
                "autofix_issue_codes": ["missing_source_pages"],
                "reconvert_issue_codes": [],
                "review_issue_codes": [],
            },
            "metrics": {},
        },
        refresh_stale=False,
    )

    assert result["indexable"] is False
    assert result["status"] == "blocked"
    assert result["action"] == "autofix"
    assert result["blocking_issue_codes"] == ["missing_source_pages"]


def test_quality_gate_keeps_document_indexable_when_corruption_is_page_local(tmp_path: Path):
    md_path = tmp_path / "partially-damaged.en.md"
    md_path.write_text(_good_markdown(), encoding="utf-8")

    result = assess_markdown_index_quality(
        md_path,
        quality_result={
            "repair_plan": {
                "action": "reconvert",
                "scope": "pages",
                "reason": "One converted page is unreliable.",
                "issue_codes": ["source_page_text_corruption"],
                "autofix_issue_codes": [],
                "reconvert_issue_codes": ["source_page_text_corruption"],
                "review_issue_codes": [],
            },
            "source_quality": {
                "pdf_page_count": 3,
                "evidence_unreliable_pages": [2],
            },
            "metrics": {},
        },
        refresh_stale=False,
    )

    assert result["indexable"] is True
    assert result["status"] == "degraded"
    assert result["action"] == "reconvert"
    assert result["blocking_issue_codes"] == []
    assert result["evidence_unreliable_pages"] == [2]


def test_quality_gate_still_blocks_document_when_every_source_page_is_corrupt(tmp_path: Path):
    md_path = tmp_path / "fully-damaged.en.md"
    md_path.write_text(_good_markdown(), encoding="utf-8")

    result = assess_markdown_index_quality(
        md_path,
        quality_result={
            "repair_plan": {
                "action": "reconvert",
                "scope": "pages",
                "reason": "Every converted page is unreliable.",
                "issue_codes": ["source_page_text_corruption"],
                "autofix_issue_codes": [],
                "reconvert_issue_codes": ["source_page_text_corruption"],
                "review_issue_codes": [],
            },
            "source_quality": {
                "pdf_page_count": 2,
                "evidence_unreliable_pages": [1, 2],
            },
            "metrics": {},
        },
        refresh_stale=False,
    )

    assert result["indexable"] is False
    assert result["status"] == "blocked"
    assert result["blocking_issue_codes"] == ["source_page_text_corruption"]


def test_quality_gate_records_explicit_override_without_hiding_blocking_issues(tmp_path: Path):
    md_path = tmp_path / "user-confirmed.en.md"
    md_path.write_text(_good_markdown(), encoding="utf-8")

    result = assess_markdown_index_quality(
        md_path,
        quality_result={
            "repair_plan": {
                "action": "reconvert",
                "scope": "document",
                "reason": "The automatic detector still recommends reconversion.",
                "issue_codes": ["source_text_loss"],
                "autofix_issue_codes": [],
                "reconvert_issue_codes": ["source_text_loss"],
                "review_issue_codes": [],
            },
            "metrics": {},
        },
        refresh_stale=False,
        allow_blocked=True,
    )
    fields = index_quality_document_fields(result)

    assert result["indexable"] is True
    assert result["status"] == "degraded"
    assert result["override_applied"] is True
    assert result["blocking_issue_codes"] == ["source_text_loss"]
    assert fields["index_status"] == "quality_degraded"
    assert fields["quality_gate"]["override_applied"] is True
    assert fields["quality_gate"]["blocking_issue_codes"] == ["source_text_loss"]


def test_quality_gate_blocks_unresolved_source_page_marker_alignment(tmp_path: Path):
    md_path = tmp_path / "misaligned.en.md"
    md_path.write_text(_good_markdown(), encoding="utf-8")

    result = assess_markdown_index_quality(
        md_path,
        quality_result={
            "repair_plan": {
                "action": "autofix",
                "scope": "markdown",
                "reason": "Source page anchors remain misaligned.",
                "issue_codes": ["source_page_marker_alignment"],
                "autofix_issue_codes": ["source_page_marker_alignment"],
                "reconvert_issue_codes": [],
                "review_issue_codes": [],
            },
            "metrics": {},
        },
        refresh_stale=False,
    )

    assert result["indexable"] is False
    assert result["status"] == "blocked"
    assert result["blocking_issue_codes"] == ["source_page_marker_alignment"]


def test_quality_gate_blocks_source_level_conversion_damage(tmp_path: Path):
    md_path = tmp_path / "broken.en.md"
    md_path.write_text(
        "\n".join(
            [
                "# Broken Paper",
                "",
                "![missing](assets/missing.png)",
                "",
                "\u951b",
            ]
        ),
        encoding="utf-8",
    )

    result = prepare_markdown_for_index(md_path)

    assert result["indexable"] is False
    assert result["status"] == "blocked"
    assert result["action"] == "reconvert"
    assert "missing_images" in result["blocking_issue_codes"]
    assert conversion_quality_result_path(md_path).exists()


def test_quality_gate_blocks_unresolved_conversion_retry_markers(tmp_path: Path):
    md_path = tmp_path / "retry.en.md"
    md_path.write_text(
        _good_markdown().replace(
            "The method section contains enough prose for retrieval.",
            "The method contains damaged math. <!-- kb:conversion_retry kind=math_text page=1 -->",
        ),
        encoding="utf-8",
    )

    result = prepare_markdown_for_index(md_path)

    assert result["indexable"] is False
    assert result["status"] == "blocked"
    assert result["action"] == "reconvert"
    assert "conversion_retry_math_text" in result["blocking_issue_codes"]
