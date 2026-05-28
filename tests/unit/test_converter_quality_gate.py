from __future__ import annotations

from pathlib import Path

from kb.converter.quality_gate import prepare_markdown_for_index
from kb.converter.quality_repair import conversion_quality_result_path


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


def test_quality_gate_autofixes_safe_issue_before_indexing(tmp_path: Path):
    md_path = tmp_path / "paper.en.md"
    md_path.write_text(_good_markdown().replace("<!-- kb_page: 1 -->\n\n", ""), encoding="utf-8")

    result = prepare_markdown_for_index(md_path, auto_repair=True)

    assert result["indexable"] is True
    assert result["status"] == "ready"
    assert result["auto_repair"]["changed"] is True
    assert md_path.read_text(encoding="utf-8").lstrip().startswith("<!-- kb_page: 1 -->")


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
