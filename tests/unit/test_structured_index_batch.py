from __future__ import annotations

import json
import time
from pathlib import Path

from kb.converter.structured_index_batch import (
    rebuild_structured_indices_for_root,
    structured_indices_need_rebuild,
)
from kb.converter.structured_indices import STRUCTURED_INDEX_VERSION
from kb.paper_guide_structured_index_runtime import load_paper_guide_reference_index


def _write_current_empty_indices(md_path: Path) -> None:
    assets = md_path.parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name in ("anchor_index.json", "equation_index.json", "figure_index.json", "table_index.json"):
        (assets / name).write_text(
            json.dumps({"version": STRUCTURED_INDEX_VERSION}, ensure_ascii=False),
            encoding="utf-8",
        )
    (assets / "reference_index.json").write_text(
        json.dumps(
            {
                "version": STRUCTURED_INDEX_VERSION,
                "citation_mention_count": 0,
                "references": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_structured_index_batch_rebuilds_stale_assets_and_records_mentions(tmp_path: Path) -> None:
    paper_dir = tmp_path / "Paper"
    paper_dir.mkdir()
    md_path = paper_dir / "Paper.en.md"
    md_path.write_text(
        "\n".join(
            [
                "# Paper",
                "<!-- kb_page: 4 -->",
                "## Background",
                "The design follows an earlier compressed sensing camera [2].",
                "",
                "## References",
                "[2] A. Example. Earlier compressed sensing camera. 2020.",
            ]
        ),
        encoding="utf-8",
    )
    assets = paper_dir / "assets"
    assets.mkdir()
    for name in ("anchor_index.json", "equation_index.json", "figure_index.json", "reference_index.json", "table_index.json"):
        (assets / name).write_text(json.dumps({"version": 1}), encoding="utf-8")

    assert structured_indices_need_rebuild(md_path) is True

    stats = rebuild_structured_indices_for_root(tmp_path)

    assert stats["scanned"] == 1
    assert stats["rebuilt"] == 1
    assert stats["failed"] == 0
    assert stats["citation_mention_count"] == 1
    payload = json.loads((assets / "reference_index.json").read_text(encoding="utf-8"))
    assert payload["version"] == STRUCTURED_INDEX_VERSION
    ref = payload["references"][0]
    assert ref["mention_count"] == 1
    assert ref["citation_mentions"][0]["page_start"] == 4
    assert "compressed sensing camera [2]" in ref["citation_mentions"][0]["citation_context"]


def test_structured_index_batch_skips_current_assets_until_markdown_changes(tmp_path: Path) -> None:
    paper_dir = tmp_path / "Paper"
    paper_dir.mkdir()
    md_path = paper_dir / "Paper.en.md"
    md_path.write_text("# Paper\n\nNo citations.\n", encoding="utf-8")
    _write_current_empty_indices(md_path)

    assert structured_indices_need_rebuild(md_path) is False
    stats = rebuild_structured_indices_for_root(tmp_path)
    assert stats["rebuilt"] == 0
    assert stats["skipped"] == 1

    time.sleep(0.01)
    md_path.write_text("# Paper\n\nNow changed.\n", encoding="utf-8")
    assert structured_indices_need_rebuild(md_path) is True


def test_structured_index_batch_skips_nonpaper_markdown_artifacts(tmp_path: Path) -> None:
    paper_dir = tmp_path / "Paper"
    paper_dir.mkdir()
    md_path = paper_dir / "Paper.en.md"
    md_path.write_text("# Paper\n\nA valid paper source.\n", encoding="utf-8")
    (paper_dir / "quality_report.md").write_text("# Markdown Quality Analysis Report\n", encoding="utf-8")
    (paper_dir / "output.md").write_text("# Legacy duplicate\n", encoding="utf-8")
    recovery_dir = paper_dir / ".conversion_cache" / "table_recovery"
    recovery_dir.mkdir(parents=True)
    (recovery_dir / "page_1_original_tables.md").write_text("| broken | archive |\n", encoding="utf-8")

    stats = rebuild_structured_indices_for_root(tmp_path, force=True)

    assert stats["scanned"] == 1
    assert stats["rebuilt"] == 1


def test_reference_index_loader_repairs_arxiv_prefix_year(tmp_path: Path) -> None:
    paper_dir = tmp_path / "RAG"
    assets = paper_dir / "assets"
    assets.mkdir(parents=True)
    md_path = paper_dir / "rag.en.md"
    md_path.write_text("# RAG\n", encoding="utf-8")
    (assets / "reference_index.json").write_text(
        json.dumps(
            {
                "references": [
                    {
                        "ref_num": 26,
                        "year": "2004",
                        "text": (
                            "[26] Vladimir Karpukhin et al. Dense passage retrieval. "
                            "arXiv preprint arXiv:2004.04906, 2020."
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    row = load_paper_guide_reference_index(md_path)[0]

    assert row["year"] == "2020"
    assert row["year_repaired_from_text"] is True
