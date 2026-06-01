from __future__ import annotations

import json
from pathlib import Path

from kb.converter.quality_center import (
    discover_quality_markdown_files,
    quality_center_summary,
    repair_quality_targets,
    source_pdf_for_markdown,
)


def test_quality_center_summary_treats_supplementary_abstract_as_not_applicable():
    summary = quality_center_summary(
        {
            "repair_plan": {"action": "none", "issue_codes": []},
            "source_quality": {
                "document_type": "supplementary",
                "abstract_not_applicable": True,
                "source_pdf_available": True,
                "pdf_page_count": 4,
                "page_alignment_confidence": "high",
            },
        }
    )

    assert summary["status"] == "ready"
    assert summary["severity"] == "ok"
    assert "补充材料" in summary["badges"]
    assert "摘要不适用" in summary["badges"]
    assert "缺独立摘要不再计为质量问题" in summary["message"]


def test_quality_center_summary_routes_source_text_loss_to_reconvert():
    summary = quality_center_summary(
        {
            "repair_plan": {"action": "reconvert", "issue_codes": ["source_text_loss", "missing_abstract"]},
            "source_quality": {
                "document_type": "research_article",
                "source_text_loss": True,
                "source_pdf_available": True,
                "pdf_page_count": 8,
                "page_alignment_confidence": "missing",
            },
        }
    )

    assert summary["status"] == "reconvert"
    assert summary["severity"] == "error"
    assert summary["action_label"] == "需要重转"
    assert "正文疑似缺失" in summary["issue_labels"]
    assert "重新转换源 PDF" in summary["message"]


def test_repair_quality_targets_autofixes_markdown_and_rebuilds_indices(tmp_path: Path):
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
                "Ada Lovelace and Grace Hopper",
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

    result = repair_quality_targets([md_path], rebuild_indices=True)

    assert result["scanned"] == 1
    assert result["repaired"] == 1
    assert result["changed"] == 1
    assert result["rebuilt"] == 1
    assert result["ready"] == 1
    repaired_text = md_path.read_text(encoding="utf-8")
    assert repaired_text.lstrip().startswith("<!-- kb_page: 1 -->")
    assert "## Abstract" in repaired_text
    assert "**Figure 1.** Experimental setup" in repaired_text
    assert (assets / "anchor_index.json").exists()


def test_discover_quality_markdown_files_prefers_converted_outputs(tmp_path: Path):
    converted_dir = tmp_path / "Paper"
    converted_dir.mkdir()
    target = converted_dir / "Paper.en.md"
    target.write_text("# Paper\n", encoding="utf-8")
    (converted_dir / "notes.md").write_text("# Notes\n", encoding="utf-8")

    assert discover_quality_markdown_files(tmp_path) == [target]


def test_source_pdf_for_markdown_uses_pdf_root_and_md_folder_name(tmp_path: Path):
    pdf_root = tmp_path / "pdfs"
    md_root = tmp_path / "md"
    pdf_root.mkdir()
    md_folder = md_root / "Stored Paper"
    md_folder.mkdir(parents=True)
    pdf_path = pdf_root / "Stored Paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    md_path = md_folder / "Stored Paper.en.md"
    md_path.write_text("# Stored Paper\n", encoding="utf-8")

    assert source_pdf_for_markdown(md_path, pdf_root) == pdf_path
