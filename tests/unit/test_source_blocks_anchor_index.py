from __future__ import annotations

import json

from kb.source_blocks import (
    build_anchor_index,
    build_source_blocks,
    doc_id_for_path,
    load_anchor_index_cached,
    load_source_blocks,
)


def _make_blocks(md_text: str, *, doc_dir) -> list[dict]:
    """Helper: write md and figure_index, then load source blocks."""
    assets_dir = doc_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    md_path = doc_dir / "paper.en.md"
    md_path.write_text(md_text, encoding="utf-8")
    return load_source_blocks(md_path)


def test_build_anchor_index_figures(tmp_path):
    md_text = "\n".join([
        "# Paper",
        "## Results",
        "![Figure 3](./assets/fig3.png)",
        "**Figure 3.** Experimental setup with CCD camera and DMD.",
        "Some body text.",
    ])
    # Need a figure_index.json so the figure gets paper_figure_number
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "figure_index.json").write_text(
        json.dumps({
            "figures": [{
                "page": 1, "index": 1,
                "asset_name": "fig3.png",
                "paper_figure_number": 3,
                "figure_id": "fig_003",
                "caption": "Figure 3. Experimental setup with CCD camera and DMD.",
            }]
        }),
        encoding="utf-8",
    )

    doc_id = doc_id_for_path(tmp_path / "paper.en.md")
    blocks = build_source_blocks(
        md_text, doc_id=doc_id,
        figure_meta_by_asset={"fig3.png": {
            "paper_figure_number": 3,
            "figure_id": "fig_003",
            "caption": "Figure 3. Experimental setup with CCD camera and DMD.",
        }},
    )
    index = build_anchor_index(blocks)
    figures = index.get("figures", [])
    assert len(figures) >= 1
    entry = figures[0]
    assert entry["number"] == 3
    assert "CCD camera" in entry["caption_text"]
    assert entry["kind"] == "figure"
    assert entry["block_id"]
    assert entry["heading_path"] == "Paper / Results"


def test_extended_data_figure_keeps_separate_identity_and_binds_caption_before_image(tmp_path):
    md_text = """# Paper

<!-- kb_page: 7 -->
![Figure 5](./assets/page_7_fig_1.png)
**Figure 5.** Main FLIM result.

<!-- kb_page: 18 -->
Extended Data Fig. 5 | Live-cell imaging of mitochondria at 25 seconds per frame.

![Figure 5](./assets/page_18_fig_1.png)
"""
    doc_id = doc_id_for_path(tmp_path / "paper.en.md")

    blocks = build_source_blocks(md_text, doc_id=doc_id)
    figures = [block for block in blocks if block.get("kind") == "figure"]
    ext_caption = next(
        block
        for block in blocks
        if block.get("figure_role") == "caption" and block.get("figure_key") == "extended_data:5"
    )
    anchors = build_anchor_index(blocks)["figures"]

    assert {block.get("figure_key") for block in figures} == {"main:5", "extended_data:5"}
    assert ext_caption["page_start"] == 18
    assert ext_caption["linked_figure_block_id"]
    assert {entry.get("figure_key") for entry in anchors if entry.get("number") == 5} == {
        "main:5",
        "extended_data:5",
    }
    assert len([entry for entry in anchors if entry.get("number") == 5]) == 2


def test_anchor_index_does_not_deduplicate_same_caption_across_figure_scopes():
    blocks = [
        {
            "kind": "figure",
            "paper_figure_number": 5,
            "figure_scope": "main",
            "figure_key": "main:5",
            "caption_text": "Shared caption text.",
            "block_id": "fig-main-5",
        },
        {
            "kind": "figure",
            "paper_figure_number": 5,
            "figure_scope": "extended_data",
            "figure_key": "extended_data:5",
            "caption_text": "Shared caption text.",
            "block_id": "fig-ext-5",
        },
    ]

    figures = build_anchor_index(blocks)["figures"]

    assert {entry["figure_key"] for entry in figures} == {"main:5", "extended_data:5"}


def test_build_anchor_index_tables(tmp_path):
    md_text = "\n".join([
        "# Paper",
        "## Comparison",
        "| Method | PSNR |",
        "|---|---|",
        "| A | 35.2 |",
        "| B | 32.1 |",
        "",
        "**Table 1.** PSNR comparison across methods.",
        "",
        "Some discussion.",
    ])
    blocks = _make_blocks(md_text, doc_dir=tmp_path)
    index = build_anchor_index(blocks)
    tables = index.get("tables", [])
    assert len(tables) >= 1
    entry = tables[0]
    assert entry["number"] == 1
    assert entry["kind"] == "table"
    assert "PSNR" in entry["caption_text"]
    assert entry["heading_path"] == "Paper / Comparison"


def test_build_anchor_index_equations(tmp_path):
    md_text = "\n".join([
        "# Paper",
        "## Method",
        "$$",
        "y = Ax + n \\tag{1}",
        "$$",
        "Equation (1) defines the forward model.",
    ])
    blocks = _make_blocks(md_text, doc_dir=tmp_path)
    index = build_anchor_index(blocks)
    equations = index.get("equations", [])
    assert len(equations) >= 1
    entry = equations[0]
    assert entry["number"] == 1
    assert entry["kind"] == "equation"


def test_build_anchor_index_empty_blocks():
    index = build_anchor_index([])
    assert index == {"figures": [], "tables": [], "equations": []}


def test_build_anchor_index_no_figures(tmp_path):
    md_text = "\n".join([
        "# Paper",
        "## Intro",
        "Plain paragraph without any figures.",
    ])
    blocks = _make_blocks(md_text, doc_dir=tmp_path)
    index = build_anchor_index(blocks)
    assert index["figures"] == []
    assert index["tables"] == []
    # Equations might also be empty
    assert isinstance(index["equations"], list)


def test_build_anchor_index_table_number_extraction(tmp_path):
    """Table number should be extracted from table buffer or next line."""
    md_text = "\n".join([
        "# Paper",
        "## Results",
        "| Col1 | Col2 |",
        "|---|---|",
        "| 1 | 2 |",
        "**Table 2.** Quantitative results.",
    ])
    blocks = _make_blocks(md_text, doc_dir=tmp_path)
    index = build_anchor_index(blocks)
    tables = index.get("tables", [])
    assert len(tables) >= 1
    entry = tables[0]
    assert entry["number"] == 2
    assert "Quantitative" in entry["caption_text"]


def test_build_anchor_index_table_number_in_buffer(tmp_path):
    """Table number should be extracted even when caption is in table buffer."""
    md_text = "\n".join([
        "# Paper",
        "## Results",
        "**Table 3: Ablation study.**",
        "| Config | Score |",
        "|---|---|",
        "| A | 90 |",
        "| B | 85 |",
    ])
    blocks = _make_blocks(md_text, doc_dir=tmp_path)
    index = build_anchor_index(blocks)
    tables = index.get("tables", [])
    assert len(tables) >= 1
    entry = tables[0]
    assert entry["number"] == 3


def test_build_anchor_index_deduplicates_figures(tmp_path):
    """Multiple blocks for same figure should not create duplicate index entries."""
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "figure_index.json").write_text(
        json.dumps({
            "figures": [{
                "page": 1, "index": 1,
                "asset_name": "fig3.png",
                "paper_figure_number": 3,
                "figure_id": "fig_003",
                "caption": "Figure 3. Caption.",
            }]
        }),
        encoding="utf-8",
    )

    md_text = "\n".join([
        "# Paper",
        "## Results",
        "![Fig 3](./assets/fig3.png)",
        "**Figure 3.** Caption.",
        "Caption continuation paragraph.",
    ])
    doc_id = doc_id_for_path(tmp_path / "paper.en.md")
    blocks = build_source_blocks(
        md_text, doc_id=doc_id,
        figure_meta_by_asset={"fig3.png": {
            "paper_figure_number": 3,
            "figure_id": "fig_003",
            "caption": "Figure 3. Caption.",
        }},
    )
    index = build_anchor_index(blocks)
    assert len(index["figures"]) == 1


def test_load_anchor_index_cached_populates(tmp_path):
    """load_anchor_index_cached should return a populated index after source blocks are loaded."""
    md_text = "\n".join([
        "# Paper",
        "## Results",
        "| A | B |",
        "|---|---|",
        "| 1 | 2 |",
        "**Table 4.** Comparison table.",
    ])
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    md_path = tmp_path / "paper.en.md"
    md_path.write_text(md_text, encoding="utf-8")

    # Load source blocks first (populates cache)
    load_source_blocks(md_path)
    # Then load anchor index from cache
    index = load_anchor_index_cached(md_path)
    assert isinstance(index, dict)
    assert "tables" in index
    assert len(index["tables"]) >= 1
    assert index["tables"][0]["number"] == 4


def test_load_anchor_index_cached_empty_for_missing_file(tmp_path):
    """load_anchor_index_cached should return empty structure for non-existent files."""
    index = load_anchor_index_cached(tmp_path / "nonexistent.md")
    assert index == {"figures": [], "tables": [], "equations": []}
