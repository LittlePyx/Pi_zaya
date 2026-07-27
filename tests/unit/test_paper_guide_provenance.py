from pathlib import Path

import kb.paper_guide_provenance as provenance


def test_support_resolution_uses_exact_surface_when_segment_ordinals_differ() -> None:
    exact_claim = "每个衍射受限光斑的入射照明功率降低约十倍，从而减少光损伤。"
    segments = [
        {"raw_markdown": "iISM 同时改善分辨率与灵敏度。", "text": "iISM 同时改善分辨率与灵敏度。"},
        {"raw_markdown": exact_claim, "text": exact_claim},
    ]
    block = {
        "block_id": "blk_abstract",
        "anchor_id": "p_abstract",
        "heading_path": "Paper / Abstract",
        "text": "tenfold lower incident illumination power, significantly reducing photodamage",
        "kind": "paragraph",
    }

    out = provenance._annotate_segments_with_support_resolution(
        segments,
        support_resolution=[
            {
                # Grounding prose ordinals and provenance Markdown ordinals can
                # legitimately differ; the exact surface must win.
                "segment_index": 0,
                "segment_text": exact_claim,
                "doc_idx": 900,
                "support_id": "DOC-900",
                "source_path": "paper.en.md",
                "block_id": "blk_abstract",
                "anchor_id": "p_abstract",
                "heading_path": "Paper / Abstract",
                "locate_anchor": block["text"],
                "claim_type": "own_result",
                "cite_policy": "locate_only",
            }
        ],
        block_lookup={"blk_abstract": block},
    )

    assert not out[0].get("support_doc_k")
    assert out[1]["support_doc_k"] == 900
    assert out[1]["primary_block_id"] == "blk_abstract"
    assert out[1]["support_locate_anchor"] == block["text"]


def test_anchor_row_to_provenance_block_preserves_figure_identity():
    block = provenance._anchor_row_to_provenance_block(
        {
            "block_id": "extended-caption",
            "anchor_id": "p_00018",
            "kind": "paragraph",
            "text": "Extended Data Figure 5. Live-cell mitochondria.",
            "paper_figure_number": 5,
            "figure_scope": "extended_data",
            "figure_key": "extended_data:5",
            "figure_id": "extended_data_fig_005",
            "figure_role": "caption",
            "linked_figure_block_id": "extended-figure",
        }
    )

    assert block["paper_figure_number"] == 5
    assert block["figure_scope"] == "extended_data"
    assert block["figure_key"] == "extended_data:5"
    assert block["linked_figure_block_id"] == "extended-figure"


def test_select_figure_index_entry_uses_semantic_scope_before_richness_score():
    rows = [
        {
            "paper_figure_number": 5,
            "figure_scope": "main",
            "figure_key": "main:5",
            "caption_block_id": "main-caption",
            "figure_block_id": "main-figure",
            "caption": "Figure 5. Rich main caption.",
        },
        {
            "paper_figure_number": 5,
            "figure_scope": "extended_data",
            "figure_key": "extended_data:5",
            "caption_block_id": "extended-caption",
            "caption": "Extended Data Figure 5. Live-cell mitochondria.",
        },
    ]

    selected = provenance._select_figure_index_entry(
        rows,
        figure_number=5,
        figure_scope="extended_data",
    )

    assert selected["caption_block_id"] == "extended-caption"


def test_quote_exact_binding_preempts_weak_long_block_match_and_label_only_locates(monkeypatch, tmp_path):
    md_path = tmp_path / "paper.en.md"
    md_path.write_text("placeholder", encoding="utf-8")
    target_sentence = (
        "Beyond applications in imaging, compressive sensing has provided benefits to various other "
        "imaging applications such as microscopy 16 , 24 , including fluorescence and hyperspectral "
        "imaging 9 , remote sensing 46 and quantum state tomography 73 ."
    )
    blocks = [
        {
            "doc_id": "doc",
            "kind": "paragraph",
            "block_id": "blk_doc_00002",
            "anchor_id": "p_00002",
            "heading_path": "Paper / Authors",
            "text": "Gibson [1,2] and Miles J.",
            "raw_text": "Gibson [1,2] and Miles J.",
            "line_start": 2,
            "line_end": 2,
        },
        {
            "doc_id": "doc",
            "kind": "paragraph",
            "block_id": "blk_doc_00032",
            "anchor_id": "p_00032",
            "heading_path": "Paper / Acquisition and image reconstruction strategies",
            "text": target_sentence
            + " It is worth noting that various technical steps can be taken when implementing a binary sampling basis.",
            "raw_text": target_sentence
            + " It is worth noting that various technical steps can be taken when implementing a binary sampling basis.",
            "line_start": 63,
            "line_end": 63,
        },
    ]
    monkeypatch.setattr(provenance, "_resolve_paper_guide_md_path", lambda *_args, **_kwargs: md_path)
    monkeypatch.setattr(provenance, "load_source_blocks", lambda _path: blocks)
    monkeypatch.setattr(provenance, "load_paper_guide_anchor_index", lambda _path: [])
    monkeypatch.setattr(provenance, "load_paper_guide_equation_index", lambda _path: [])
    monkeypatch.setattr(provenance, "load_paper_guide_figure_index", lambda _path: [])

    out = provenance._build_paper_guide_answer_provenance(
        answer=(
            "是的，文章在原文中有一处明确的“同一句话提到多个相关工作”的地方：\n\n"
            "> Beyond applications in imaging, compressive sensing has provided benefits to various other "
            "imaging applications such as microscopy 16, 24, including fluorescence and hyperspectral "
            "imaging 9, remote sensing 46 and quantum state tomography 73.\n\n"
            "- 显微成像（microscopy）：\n"
            "- 荧光与高光谱成像："
        ),
        answer_hits=[],
        bound_source_path=str(tmp_path / "paper.pdf"),
        bound_source_name="paper.pdf",
        db_dir=Path(tmp_path),
        llm_rerank=False,
    )

    assert out is not None
    segments = out["segments"]
    quote_segments = [seg for seg in segments if seg.get("kind") == "blockquote"]
    assert quote_segments
    assert quote_segments[0]["evidence_mode"] == "direct"
    assert quote_segments[0]["primary_block_id"] == "blk_doc_00032"
    assert quote_segments[0]["mapping_source"] == "quote_exact"
    assert "Beyond applications in imaging" in quote_segments[0]["evidence_quote"]
    assert all(seg.get("primary_block_id") != "blk_doc_00002" for seg in segments)
