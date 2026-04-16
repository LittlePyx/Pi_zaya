from kb.paper_guide_contracts import (
    _build_paper_guide_intent_model,
    _build_paper_guide_render_packet_model,
    _build_paper_guide_retrieval_bundle_model,
    _build_paper_guide_support_pack_model,
    _build_paper_guide_support_resolution,
    _paper_guide_evidence_card_model_from_raw,
    _paper_guide_grounding_trace_segment_model_from_raw,
    _paper_guide_should_suppress_render_locate,
    _paper_guide_support_record_model_from_resolution,
    _paper_guide_support_record_model_from_slot,
    _normalize_paper_guide_support_slot,
)


def test_build_paper_guide_support_resolution_normalizes_lists():
    out = _build_paper_guide_support_resolution(
        doc_idx="2",
        support_id=" DOC-1 ",
        candidate_refs=["4", 4, "59", "bad", 0],
        ref_spans=[
            {"text": " Duarte et al. [4] ", "nums": ["4", 4], "scope": "same_clause"},
            {"text": "Duarte et al. [4]", "nums": [4], "scope": "same_clause"},
            {"text": "compressive sensing [59]", "nums": ["59"], "scope": "same_sentence"},
        ],
        figure_number="3",
        box_number="1",
        panel_letters=["F", "f", "g", ""],
        line_index="3",
    )

    assert out["doc_idx"] == 2
    assert out["support_id"] == "DOC-1"
    assert out["candidate_refs"] == [4, 59]
    assert out["figure_number"] == 3
    assert out["box_number"] == 1
    assert out["panel_letters"] == ["f", "g"]
    assert out["line_index"] == 3
    assert out["ref_spans"] == [
        {"text": "Duarte et al. [4]", "nums": [4], "scope": "same_clause"},
        {"text": "compressive sensing [59]", "nums": [59], "scope": "same_sentence"},
    ]


def test_normalize_paper_guide_support_slot_preserves_extra_keys():
    out = _normalize_paper_guide_support_slot(
        {
            "doc_idx": "1",
            "source_path": " demo.md ",
            "candidate_refs": ["35", 35, 0],
            "figure_number": "3",
            "box_number": "1",
            "panel_letters": ["F", "f"],
            "deepread_texts": ["  alpha  ", "", "beta"],
            "custom_flag": True,
        }
    )

    assert out["doc_idx"] == 1
    assert out["source_path"] == "demo.md"
    assert out["candidate_refs"] == [35]
    assert out["figure_number"] == 3
    assert out["box_number"] == 1
    assert out["panel_letters"] == ["f"]
    assert out["deepread_texts"] == ["alpha", "beta"]
    assert out["custom_flag"] is True


def test_build_paper_guide_intent_model_normalizes_targets():
    out = _build_paper_guide_intent_model(
        prompt=" Explain Figure 3 panel (F) simply ",
        family=" Figure_Walkthrough ",
        exact_support=1,
        beginner_mode="yes",
        target_figure="3",
        target_panels=["F", "f", ""],
        target_equation="0",
        target_scope={"prompt_family": "figure_walkthrough"},
    )

    assert out.prompt == "Explain Figure 3 panel (F) simply"
    assert out.family == "figure_walkthrough"
    assert out.exact_support is True
    assert out.beginner_mode is True
    assert out.target_figure == 3
    assert out.target_panels == ["f"]
    assert out.target_equation == 0
    assert out.target_scope == {"prompt_family": "figure_walkthrough"}


def test_support_record_model_adapters_preserve_normalized_fields():
    slot_model = _paper_guide_support_record_model_from_slot(
        {
            "source_path": " demo.md ",
            "candidate_refs": ["35", "35", 0],
            "panel_letters": ["B", "b"],
            "deepread_texts": [" alpha ", ""],
        }
    )
    resolution_model = _paper_guide_support_record_model_from_resolution(
        {
            "source_path": " demo.md ",
            "resolved_ref_num": "26",
            "figure_number": "2",
            "segment_index": "4",
            "panel_letters": ["C", "c"],
        }
    )

    assert slot_model.source_path == "demo.md"
    assert slot_model.candidate_refs == [35]
    assert slot_model.panel_letters == ["b"]
    assert slot_model.deepread_texts == ["alpha"]
    assert resolution_model.source_path == "demo.md"
    assert resolution_model.resolved_ref_num == 26
    assert resolution_model.figure_number == 2
    assert resolution_model.segment_index == 4
    assert resolution_model.panel_letters == ["c"]


def test_build_paper_guide_support_pack_model_accepts_slot_and_resolution_records():
    out = _build_paper_guide_support_pack_model(
        family=" Method ",
        answer_markdown="The paper states this explicitly.",
        support_records=[
            {"source_path": "paper.md", "support_id": "slot-1", "candidate_refs": ["8"]},
            {"source_path": "paper.md", "resolved_ref_num": "26", "segment_text": "NeRF is used."},
        ],
        needs_supplement=1,
    )

    assert out.family == "method"
    assert out.answer_markdown == "The paper states this explicitly."
    assert out.needs_supplement is True
    assert len(out.support_records) == 2
    assert out.support_records[0].support_id == "slot-1"
    assert out.support_records[0].candidate_refs == [8]
    assert out.support_records[1].resolved_ref_num == 26
    assert out.support_records[1].segment_text == "NeRF is used."


def test_grounding_trace_segment_model_from_raw_uses_primary_fields():
    out = _paper_guide_grounding_trace_segment_model_from_raw(
        {
            "segment_id": "seg_001",
            "text": "NeRF is used as the 3D representation.",
            "source_path": "paper.md",
            "primary_block_id": "b-1",
            "primary_anchor_id": "a-1",
            "primary_heading_path": "Methods / Figure 3",
            "anchor_kind": "figure",
            "support_slot_figure_number": "3",
            "claim_type": "figure_claim",
            "locate_policy": "required",
            "support_block_ids": ["b-1", "b-2", ""],
            "evidence_block_ids": ["b-1"],
        }
    )

    assert out.segment_id == "seg_001"
    assert out.primary_block_id == "b-1"
    assert out.primary_anchor_id == "a-1"
    assert out.heading_path == "Methods / Figure 3"
    assert out.anchor_kind == "figure"
    assert out.anchor_number == 3
    assert out.claim_type == "figure_claim"
    assert out.locate_policy == "required"
    assert out.support_block_ids == ["b-1", "b-2"]
    assert out.evidence_block_ids == ["b-1"]


def test_grounding_trace_segment_model_from_raw_falls_back_to_support_resolution_fields():
    out = _paper_guide_grounding_trace_segment_model_from_raw(
        {
            "support_id": "DOC-1-S1",
            "segment_text": "APR uses phase correlation for registration.",
            "source_path": "paper.md",
            "block_id": "b-7",
            "anchor_id": "a-7",
            "heading_path": "Methods / APR",
            "claim_type": "method_claim",
            "resolved_ref_num": "35",
        }
    )

    assert out.segment_id == "DOC-1-S1"
    assert out.text == "APR uses phase correlation for registration."
    assert out.source_path == "paper.md"
    assert out.primary_block_id == "b-7"
    assert out.primary_anchor_id == "a-7"
    assert out.heading_path == "Methods / APR"
    assert out.claim_type == "method_claim"


def test_paper_guide_evidence_card_model_from_raw_normalizes_fields():
    out = _paper_guide_evidence_card_model_from_raw(
        {
            "doc_idx": "2",
            "sid": " s123 ",
            "source_path": " demo.md ",
            "heading": " Methods ",
            "cue": " phase correlation [35] ",
            "snippet": " APR is grounded here. ",
            "candidate_refs": ["35", 35, 0],
            "deepread_texts": [" alpha ", "alpha", "", "beta"],
        }
    )

    assert out.doc_idx == 2
    assert out.sid == "s123"
    assert out.source_path == "demo.md"
    assert out.heading == "Methods"
    assert out.cue == "phase correlation [35]"
    assert out.snippet == "APR is grounded here."
    assert out.candidate_refs == [35]
    assert out.deepread_texts == ["alpha", "beta"]


def test_build_paper_guide_retrieval_bundle_model_normalizes_payload():
    out = _build_paper_guide_retrieval_bundle_model(
        prompt_family=" Method ",
        target_scope={"prompt_family": "method"},
        evidence_cards=[
            {
                "doc_idx": "1",
                "sid": "s123",
                "source_path": "demo.md",
                "heading": "Methods",
                "candidate_refs": ["35", 35],
                "deepread_texts": ["alpha", "alpha", "beta"],
            }
        ],
        candidate_refs_by_source={"demo.md": ["35", 35, 0]},
        direct_source_path=" direct.md ",
        focus_source_path=" focus.md ",
        bound_source_path=" bound.md ",
    )

    assert out.prompt_family == "method"
    assert out.target_scope == {"prompt_family": "method"}
    assert out.candidate_refs_by_source == {"demo.md": [35]}
    assert out.direct_source_path == "direct.md"
    assert out.focus_source_path == "focus.md"
    assert out.bound_source_path == "bound.md"
    assert len(out.evidence_cards) == 1
    assert out.evidence_cards[0].candidate_refs == [35]
    assert out.evidence_cards[0].deepread_texts == ["alpha", "beta"]


def test_build_paper_guide_render_packet_model_derives_primary_render_state():
    out = _build_paper_guide_render_packet_model(
        answer_markdown="Answer [[CITE:s1:3]]",
        notice="Low confidence notice.",
        rendered_body="Answer [3](#kb-cite-demo-3)",
        rendered_content="Low confidence notice.\n\nAnswer [3](#kb-cite-demo-3)",
        copy_markdown="Answer [3]",
        copy_text="Answer 3",
        cite_details=[
            {
                "num": "3",
                "anchor": " kb-cite-demo-3 ",
                "source_name": " demo.pdf ",
                "source_path": " demo.md ",
            }
        ],
        citation_validation={"kept": 1, "dropped": 0},
        provenance_segments=[
            {
                "segment_id": "seg-hidden",
                "locate_policy": "hidden",
            },
            {
                "segment_id": "seg-1",
                "locate_policy": "required",
                "locate_target": {
                    "segmentId": "seg-1",
                    "headingPath": "Methods / APR",
                    "blockId": "b-7",
                },
                "reader_open": {
                    "sourcePath": "demo.md",
                    "blockId": "b-7",
                },
            },
        ],
    )

    assert out.answer_markdown == "Answer [[CITE:s1:3]]"
    assert out.notice == "Low confidence notice."
    assert out.cite_details[0].num == 3
    assert out.cite_details[0].anchor == "kb-cite-demo-3"
    assert out.cite_details[0].source_name == "demo.pdf"
    assert out.locate_target == {
        "segmentId": "seg-1",
        "headingPath": "Methods / APR",
        "blockId": "b-7",
    }
    assert out.reader_open == {
        "sourcePath": "demo.md",
        "blockId": "b-7",
    }
    assert out.segment_ids == ["seg-hidden", "seg-1"]
    assert out.visible_segment_ids == ["seg-1"]
    assert out.provenance_segment_count == 2
    assert out.visible_segment_count == 1


def test_build_paper_guide_render_packet_model_prefers_grounded_support_segment_over_shell_sentence():
    out = _build_paper_guide_render_packet_model(
        answer_markdown="The paper cites [4] for this point.\n> most of the existing methods employ ADMM [4],",
        provenance_segments=[
            {
                "segment_id": "seg-shell",
                "locate_policy": "required",
                "hit_level": "exact",
                "claim_type": "critical_fact_claim",
                "text": "The paper cites [4] for this point.",
                "primary_heading_path": "Method / Wrong Block",
                "primary_block_id": "b-wrong",
                "primary_anchor_id": "a-wrong",
                "evidence_quote": "A generic method sentence unrelated to the citation prompt.",
                "locate_target": {
                    "segmentId": "seg-shell",
                    "headingPath": "Method / Wrong Block",
                    "snippet": "A generic method sentence unrelated to the citation prompt.",
                    "anchorText": "A generic method sentence unrelated to the citation prompt.",
                    "blockId": "b-wrong",
                    "anchorId": "a-wrong",
                    "anchorKind": "sentence",
                    "hitLevel": "exact",
                },
                "reader_open": {
                    "sourcePath": "demo.md",
                    "headingPath": "Method / Wrong Block",
                    "snippet": "A generic method sentence unrelated to the citation prompt.",
                    "blockId": "b-wrong",
                    "anchorId": "a-wrong",
                    "strictLocate": True,
                },
            },
            {
                "segment_id": "seg-cite",
                "locate_policy": "required",
                "hit_level": "exact",
                "mapping_source": "support_slot",
                "mapping_quality": 1.6,
                "evidence_confidence": 2.4,
                "claim_type": "prior_work",
                "support_slot_claim_type": "prior_work",
                "support_locate_anchor": "most of the existing methods employ alternating direction method of multipliers (ADMM) [4],",
                "resolved_ref_num": 4,
                "text": "most of the existing methods employ alternating direction method of multipliers (ADMM) [4],",
                "primary_heading_path": "Related Work / Snapshot Compressive Imaging",
                "primary_block_id": "b-right",
                "primary_anchor_id": "a-right",
                "locate_target": {
                    "segmentId": "seg-cite",
                    "headingPath": "Related Work / Snapshot Compressive Imaging",
                    "snippet": "most of the existing methods employ alternating direction method of multipliers (ADMM) [4],",
                    "anchorText": "most of the existing methods employ alternating direction method of multipliers (ADMM) [4],",
                    "blockId": "b-right",
                    "anchorId": "a-right",
                    "anchorKind": "blockquote",
                    "hitLevel": "exact",
                },
                "reader_open": {
                    "sourcePath": "demo.md",
                    "headingPath": "Related Work / Snapshot Compressive Imaging",
                    "snippet": "most of the existing methods employ alternating direction method of multipliers (ADMM) [4],",
                    "blockId": "b-right",
                    "anchorId": "a-right",
                    "strictLocate": True,
                },
            },
        ],
    )

    assert out.locate_target["segmentId"] == "seg-cite"
    assert out.locate_target["blockId"] == "b-right"
    assert "alternating direction method of multipliers" in out.locate_target["anchorText"].lower()
    assert out.reader_open["blockId"] == "b-right"


def test_build_paper_guide_render_packet_model_suppresses_negative_evidence_note_locate():
    out = _build_paper_guide_render_packet_model(
        answer_markdown="The paper does not mention ADMM in the retrieved context.",
        provenance_segments=[
            {
                "segment_id": "seg-neg",
                "locate_policy": "required",
                "locate_surface_policy": "primary",
                "hit_level": "exact",
                "claim_type": "evidence_note_claim",
                "text": "The paper does not mention ADMM in the retrieved context.",
                "primary_heading_path": "Discussion",
                "primary_block_id": "b-neg",
                "primary_anchor_id": "a-neg",
                "anchor_kind": "sentence",
                "evidence_quote": "The paper does not mention ADMM in the retrieved context.",
                "locate_target": {
                    "segmentId": "seg-neg",
                    "headingPath": "Discussion",
                    "snippet": "The paper does not mention ADMM in the retrieved context.",
                    "anchorText": "The paper does not mention ADMM in the retrieved context.",
                    "blockId": "b-neg",
                    "anchorId": "a-neg",
                    "anchorKind": "sentence",
                    "locatePolicy": "required",
                    "locateSurfacePolicy": "primary",
                },
                "reader_open": {
                    "sourcePath": "demo.md",
                    "headingPath": "Discussion",
                    "snippet": "The paper does not mention ADMM in the retrieved context.",
                    "blockId": "b-neg",
                    "anchorId": "a-neg",
                    "anchorKind": "sentence",
                    "strictLocate": True,
                },
            }
        ],
    )

    assert out.locate_target == {}
    assert out.reader_open == {}
    assert out.visible_segment_ids == ["seg-neg"]


def test_should_suppress_render_locate_keeps_formula_and_figure_claims():
    assert _paper_guide_should_suppress_render_locate(
        {"claim_type": "formula_claim", "text": "Equation (1) is not found in the appendix.", "anchor_kind": "equation"},
        locate_target={"anchorKind": "equation", "snippet": "Equation (1) is not found in the appendix."},
        reader_open={"anchorKind": "equation"},
    ) is False
    assert _paper_guide_should_suppress_render_locate(
        {"claim_type": "figure_claim", "text": "Figure 3 is not found in this excerpt.", "anchor_kind": "figure"},
        locate_target={"anchorKind": "figure", "snippet": "Figure 3 is not found in this excerpt."},
        reader_open={"anchorKind": "figure"},
    ) is False


def test_build_paper_guide_render_packet_model_prefers_positive_locate_over_negative_evidence_note():
    out = _build_paper_guide_render_packet_model(
        answer_markdown="The paper does not mention ADMM in one checked section, but it explicitly cites ADMM [4] elsewhere.",
        provenance_segments=[
            {
                "segment_id": "seg-neg",
                "locate_policy": "required",
                "locate_surface_policy": "primary",
                "hit_level": "exact",
                "claim_type": "evidence_note_claim",
                "text": "The paper does not mention ADMM in the retrieved context.",
                "primary_block_id": "b-neg",
                "primary_anchor_id": "a-neg",
                "anchor_kind": "sentence",
                "evidence_quote": "The paper does not mention ADMM in the retrieved context.",
                "locate_target": {
                    "segmentId": "seg-neg",
                    "headingPath": "Discussion",
                    "snippet": "The paper does not mention ADMM in the retrieved context.",
                    "anchorText": "The paper does not mention ADMM in the retrieved context.",
                    "blockId": "b-neg",
                    "anchorId": "a-neg",
                    "anchorKind": "sentence",
                    "locatePolicy": "required",
                    "locateSurfacePolicy": "primary",
                },
                "reader_open": {
                    "sourcePath": "demo.md",
                    "headingPath": "Discussion",
                    "snippet": "The paper does not mention ADMM in the retrieved context.",
                    "blockId": "b-neg",
                    "anchorId": "a-neg",
                    "anchorKind": "sentence",
                    "strictLocate": True,
                },
            },
            {
                "segment_id": "seg-pos",
                "locate_policy": "required",
                "locate_surface_policy": "primary",
                "hit_level": "exact",
                "mapping_source": "support_slot",
                "mapping_quality": 1.7,
                "evidence_confidence": 2.3,
                "claim_type": "prior_work",
                "support_slot_claim_type": "prior_work",
                "support_locate_anchor": "most of the existing methods employ alternating direction method of multipliers (ADMM) [4],",
                "resolved_ref_num": 4,
                "text": "most of the existing methods employ alternating direction method of multipliers (ADMM) [4],",
                "primary_block_id": "b-pos",
                "primary_anchor_id": "a-pos",
                "anchor_kind": "blockquote",
                "locate_target": {
                    "segmentId": "seg-pos",
                    "headingPath": "Related Work",
                    "snippet": "most of the existing methods employ alternating direction method of multipliers (ADMM) [4],",
                    "anchorText": "most of the existing methods employ alternating direction method of multipliers (ADMM) [4],",
                    "blockId": "b-pos",
                    "anchorId": "a-pos",
                    "anchorKind": "blockquote",
                    "locatePolicy": "required",
                    "locateSurfacePolicy": "primary",
                },
                "reader_open": {
                    "sourcePath": "demo.md",
                    "headingPath": "Related Work",
                    "snippet": "most of the existing methods employ alternating direction method of multipliers (ADMM) [4],",
                    "blockId": "b-pos",
                    "anchorId": "a-pos",
                    "anchorKind": "blockquote",
                    "strictLocate": True,
                },
            },
        ],
    )

    assert out.locate_target["segmentId"] == "seg-pos"
    assert out.reader_open["blockId"] == "b-pos"
