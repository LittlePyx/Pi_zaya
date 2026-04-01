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

    assert caption_block["figure_id"] == "fig_004"
    assert caption_block["paper_figure_number"] == 4
    assert caption_block["linked_figure_block_id"] == figure_block["block_id"]
    assert caption_block["text"].startswith("Figure 4. Demo caption")
