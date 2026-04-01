from kb.paper_guide_message_builder import _build_generation_prompt_bundle


def test_build_generation_prompt_bundle_adds_abstract_rule_for_citeless_family():
    out = _build_generation_prompt_bundle(
        prompt="把摘要原文给出并翻译",
        ctx="DOC-1 [SID:s12345678] demo\n# Abstract\nHere we introduce...",
        paper_guide_mode=True,
        paper_guide_bound_source_ready=True,
        paper_guide_prompt_family="abstract",
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        answer_contract_v1=False,
        has_answer_hits=True,
        locked_citation_source=None,
        image_first_prompt=False,
        anchor_grounded_answer=False,
        paper_guide_special_focus_block="",
        paper_guide_support_slots_block="",
        paper_guide_evidence_cards_block="",
        paper_guide_citation_grounding_block="",
        image_attachment_count=0,
    )

    assert "Paper-guide abstract rule:" in out["system"]
    assert "Question:\n把摘要原文给出并翻译" in out["user"]
    assert "Retrieved context (with deep-read supplements):" in out["user"]
    assert out["paper_guide_contract_enabled"] is False


def test_build_generation_prompt_bundle_adds_citation_lock_for_non_citeless_family():
    out = _build_generation_prompt_bundle(
        prompt="How is APR grounded?",
        ctx="DOC-1 [SID:s12345678] demo\nAPR was performed using image registration [35].",
        paper_guide_mode=True,
        paper_guide_bound_source_ready=True,
        paper_guide_prompt_family="method",
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        answer_contract_v1=False,
        has_answer_hits=True,
        locked_citation_source={"sid": "s12345678", "source_name": "demo.pdf"},
        image_first_prompt=False,
        anchor_grounded_answer=True,
        paper_guide_special_focus_block="FOCUS BLOCK",
        paper_guide_support_slots_block="SUPPORT BLOCK",
        paper_guide_evidence_cards_block="EVIDENCE BLOCK",
        paper_guide_citation_grounding_block="GROUNDING BLOCK",
        image_attachment_count=2,
    )

    assert "Citation source lock:" in out["system"]
    assert "[[CITE:s12345678:<ref_num>]]" in out["system"]
    assert "Paper-guide support-slot protocol:" in out["system"]
    assert "Anchor-grounded answer rule:" in out["system"]
    assert "FOCUS BLOCK" in out["user"]
    assert "SUPPORT BLOCK" in out["user"]
    assert "EVIDENCE BLOCK" in out["user"]
    assert "GROUNDING BLOCK" in out["user"]
    assert "Attached images: 2." in out["user"]


def test_build_generation_prompt_bundle_skips_citation_lock_for_citation_lookup_prompt():
    out = _build_generation_prompt_bundle(
        prompt="Which references are cited for RVT, and where is that stated exactly?",
        ctx="DOC-1 [SID:s12345678] demo\nRVT was proposed in [34].",
        paper_guide_mode=True,
        paper_guide_bound_source_ready=True,
        paper_guide_prompt_family="citation_lookup",
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        answer_contract_v1=False,
        has_answer_hits=True,
        locked_citation_source={"sid": "s12345678", "source_name": "demo.pdf"},
        image_first_prompt=False,
        anchor_grounded_answer=False,
        paper_guide_special_focus_block="",
        paper_guide_support_slots_block="",
        paper_guide_evidence_cards_block="",
        paper_guide_citation_grounding_block="",
        image_attachment_count=0,
    )

    assert "Citation source lock:" not in out["system"]
