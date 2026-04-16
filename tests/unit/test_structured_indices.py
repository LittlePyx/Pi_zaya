import json

from kb.converter.structured_indices import rebuild_structured_indices_for_markdown


def test_rebuild_structured_indices_writes_expected_assets_and_enriches_figure_index(tmp_path):
    md_path = tmp_path / "demo.en.md"
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    (assets_dir / "figure_index.json").write_text(
        json.dumps(
            {
                "figures": [
                    {
                        "page": 2,
                        "index": 1,
                        "asset_name": "page_2_fig_1.png",
                        "fig_no": 4,
                        "fig_ident": "4",
                        "caption": "Figure 4. Noise-robustness comparison using the Lena image.",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    md_text = "\n".join(
        [
            "# Results",
            "",
            "![Figure 4](./assets/page_2_fig_1.png)",
            "",
            "**Figure 4.** Noise-robustness comparison using the Lena image.",
            "",
            "## Method",
            "",
            "$$Y = \\\\sum_{i=1}^{N} X_i \\\\odot M_i + Z \\\\tag{3}$$",
            "",
            "where Y is the measured compressed image.",
            "",
            "## References",
            "",
            "[1] A. Demo. Stable grounding for figures. Journal, 2024. doi:10.1000/demo-grounding",
        ]
    )

    out = rebuild_structured_indices_for_markdown(md_path, md_text=md_text, assets_dir=assets_dir)

    anchor_payload = json.loads((assets_dir / "anchor_index.json").read_text(encoding="utf-8"))
    equation_payload = json.loads((assets_dir / "equation_index.json").read_text(encoding="utf-8"))
    reference_payload = json.loads((assets_dir / "reference_index.json").read_text(encoding="utf-8"))
    figure_payload = json.loads((assets_dir / "figure_index.json").read_text(encoding="utf-8"))

    assert out["anchor_index"]["anchor_count"] == anchor_payload["anchor_count"]
    assert any(item["kind"] == "figure" for item in anchor_payload["anchors"])
    assert any(item["kind"] == "equation" for item in anchor_payload["anchors"])

    assert equation_payload["equation_count"] == 1
    eq = equation_payload["equations"][0]
    assert eq["equation_number"] == 3
    assert "\\tag{3}" in eq["normalized_tex"]
    assert "measured compressed image" in eq["context_after"]

    assert reference_payload["ref_count"] == 1
    ref = reference_payload["references"][0]
    assert ref["ref_num"] == 1
    assert ref["doi"] == "10.1000/demo-grounding"
    assert ref["year"] == "2024"
    assert ref["title"] == "Stable grounding for figures"
    assert ref["authors"] == "A. Demo"
    assert ref["parse_confidence"] > 0.0

    fig = figure_payload["figures"][0]
    assert fig["figure_block_id"]
    assert fig["caption_block_id"]
    assert fig["anchor_id"]
    assert fig["page"] == 2
    assert "Results / Figure 4" in fig["heading_path"]
    assert "Noise-robustness comparison" in fig["locate_anchor"]
