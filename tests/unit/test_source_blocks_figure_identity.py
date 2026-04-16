from __future__ import annotations

import json

from kb.source_blocks import load_source_blocks


def test_load_source_blocks_enriches_figure_and_caption_from_figure_index(tmp_path):
    doc_dir = tmp_path / "DemoPaper"
    assets_dir = doc_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    md_path = doc_dir / "DemoPaper.en.md"
    md_path.write_text(
        "\n".join(
            [
                "# Demo",
                "",
                "## Results",
                "![Figure](./assets/page_7_fig_1.png)",
                "",
                "Figure 4. Demo caption for the reconstructed image.",
                "",
                "A follow-up body paragraph.",
            ]
        ),
        encoding="utf-8",
    )
    (assets_dir / "figure_index.json").write_text(
        json.dumps(
            {
                "figures": [
                    {
                        "page": 7,
                        "index": 1,
                        "asset_name": "page_7_fig_1.png",
                        "asset_name_raw": "page_7_fig_1.png",
                        "asset_name_alias": "fig_4.png",
                        "figure_id": "fig_004",
                        "figure_ident": "4",
                        "paper_figure_number": 4,
                        "caption": "Figure 4. Demo caption for the reconstructed image.",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    blocks = load_source_blocks(md_path)

    figure_block = next(block for block in blocks if str(block.get("kind") or "") == "figure")
    caption_block = next(
        block for block in blocks
        if str(block.get("figure_role") or "") == "caption"
    )

    assert figure_block["number"] == 4
    assert figure_block["paper_figure_number"] == 4
    assert figure_block["figure_id"] == "fig_004"
    assert figure_block["asset_name_alias"] == "fig_4.png"
    assert figure_block["text"] == "Figure 4"
    assert figure_block["heading_path"] == "Demo / Results / Figure 4"

    assert caption_block["figure_id"] == "fig_004"
    assert caption_block["paper_figure_number"] == 4
    assert caption_block["linked_figure_block_id"] == figure_block["block_id"]
    assert caption_block["text"].startswith("Figure 4. Demo caption")
    assert caption_block["heading_path"] == "Demo / Results / Figure 4"


def test_load_source_blocks_preserves_box_heading_context(tmp_path):
    doc_dir = tmp_path / "BoxPaper"
    doc_dir.mkdir(parents=True, exist_ok=True)
    md_path = doc_dir / "BoxPaper.en.md"
    md_path.write_text(
        "\n".join(
            [
                "# Demo",
                "",
                "## Acquisition",
                "",
                "<!-- box:start id=1 -->",
                "",
                "**[Box 1 - The maths behind single-pixel imaging]**",
                "",
                "$$",
                "x = Ay",
                "$$",
                "",
                "Box 1 text explains the sensing model.",
                "",
                "<!-- box:end id=1 -->",
                "",
                "Outside the box paragraph.",
            ]
        ),
        encoding="utf-8",
    )

    blocks = load_source_blocks(md_path)

    box_equation = next(
        block for block in blocks
        if str(block.get("kind") or "") == "equation"
    )
    box_paragraph = next(
        block for block in blocks
        if str(block.get("text") or "").startswith("Box 1 text explains")
    )
    outside_paragraph = next(
        block for block in blocks
        if str(block.get("text") or "").startswith("Outside the box paragraph")
    )

    assert box_equation["heading_path"] == "Demo / Acquisition / Box 1"
    assert box_paragraph["heading_path"] == "Demo / Acquisition / Box 1"
    assert outside_paragraph["heading_path"] == "Demo / Acquisition"


def test_load_source_blocks_promotes_caption_continuation_paragraph_to_figure_heading(tmp_path):
    doc_dir = tmp_path / "FigurePaper"
    assets_dir = doc_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    md_path = doc_dir / "FigurePaper.en.md"
    md_path.write_text(
        "\n".join(
            [
                "# Demo",
                "",
                "## Results",
                "",
                "![Figure 6](./assets/page_10_fig_1.png)",
                "",
                "**Figure 6.** Illustration of the reported deep transformer network for high-fidelity large-scale single-photon imaging.",
                "",
                "a The workflow and structure of the reported network. b The enhancement comparison between CNN-based U-net network and the reported transformer-based network.",
                "",
                "The next paragraph returns to the main body discussion.",
            ]
        ),
        encoding="utf-8",
    )
    (assets_dir / "figure_index.json").write_text(
        json.dumps(
            {
                "figures": [
                    {
                        "page": 10,
                        "index": 1,
                        "asset_name": "page_10_fig_1.png",
                        "asset_name_raw": "page_10_fig_1.png",
                        "asset_name_alias": "fig_6.png",
                        "figure_id": "fig_006",
                        "figure_ident": "6",
                        "paper_figure_number": 6,
                        "caption": "Figure 6. Illustration of the reported deep transformer network for high-fidelity large-scale single-photon imaging.",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    blocks = load_source_blocks(md_path)

    continuation_block = next(
        block
        for block in blocks
        if str(block.get("figure_role") or "") == "caption_continuation"
    )
    body_block = next(
        block
        for block in blocks
        if str(block.get("text") or "").startswith("The next paragraph returns")
    )

    assert continuation_block["paper_figure_number"] == 6
    assert continuation_block["heading_path"] == "Demo / Results / Figure 6"
    assert continuation_block["linked_figure_block_id"]
    assert "enhancement comparison" in continuation_block["text"].lower()
    assert body_block["heading_path"] == "Demo / Results"
