from pathlib import Path

import kb.paper_guide_provenance as provenance


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
