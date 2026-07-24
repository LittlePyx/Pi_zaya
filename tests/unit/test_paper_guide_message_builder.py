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
    assert "Research answer plan:" not in out["system"]
    assert "Question:\n把摘要原文给出并翻译" in out["user"]
    assert "Retrieved context (with deep-read supplements):" in out["user"]
    assert out["paper_guide_contract_enabled"] is False
    assert out["research_answer_plan"] == ""


def test_normal_prompt_bundle_adds_user_facing_quality_protocol_with_hits():
    out = _build_generation_prompt_bundle(
        prompt="Compare the method trade-offs.",
        ctx="DOC-1 [SID:s12345678] demo\nThe method improves resolution.",
        paper_guide_mode=False,
        paper_guide_bound_source_ready=False,
        paper_guide_prompt_family="",
        answer_intent="compare",
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

    system = out["system"]
    assert out["research_answer_plan"] == "compare"
    assert "Research answer plan:" in system
    assert "Plan type: compare." in system
    assert "User-facing research answer quality protocol:" in system
    assert "Follow the Research answer plan above" in system
    assert "Every paper-specific claim based on retrieved snippets" in system
    assert "[10001]" in system
    assert "Do not use bare [1] [2] [3]" in system
    assert "Required citation reminder:" in out["user"]
    assert "Retrieved context (with deep-read supplements):" in out["user"]


def test_normal_prompt_bundle_adds_no_hit_quality_protocol():
    out = _build_generation_prompt_bundle(
        prompt="What should I read next?",
        ctx="",
        paper_guide_mode=False,
        paper_guide_bound_source_ready=False,
        paper_guide_prompt_family="",
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        answer_contract_v1=False,
        has_answer_hits=False,
        locked_citation_source=None,
        image_first_prompt=False,
        anchor_grounded_answer=False,
        paper_guide_special_focus_block="",
        paper_guide_support_slots_block="",
        paper_guide_evidence_cards_block="",
        paper_guide_citation_grounding_block="",
        image_attachment_count=0,
    )

    system = out["system"]
    assert out["research_answer_plan"] == "literature_positioning"
    assert "Plan type: literature_positioning." in system
    assert "User-facing research answer quality protocol:" in system
    assert "no matching library snippets were retrieved" in system
    assert "general guidance" in system
    assert "Retrieved context (with deep-read supplements):\n(none)" in out["user"]


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
        paper_guide_reference_opportunities_block="Paper-guide upstream reference opportunities:\n- label=ADMM | cite_example=[[CITE:s12345678:4]]",
        citation_plan_block="Citation plan (follow before adding citations):\n- intent=method",
        image_attachment_count=2,
    )

    assert "Citation source lock:" in out["system"]
    assert out["research_answer_plan"] == "method_explain"
    assert "Plan type: method_explain." in out["system"]
    assert "[[CITE:s12345678:<ref_num>]]" in out["system"]
    assert "copy a provided cite_example exactly" in out["system"]
    assert "Paper-guide support-slot protocol:" in out["system"]
    assert "Upstream-reference protocol:" in out["system"]
    assert "Answer the user's substantive question first" in out["system"]
    assert "Never begin the final answer with locator-only shells" in out["system"]
    assert "limit on distinct evidence cards" in out["system"]
    assert "a citation in an earlier paragraph does not support a later uncited restatement" in out["system"]
    assert "Do not collect the citations in an evidence preamble" in out["system"]
    assert "Remove unsupported specifics before finalizing" in out["system"]
    assert "Anchor-grounded answer rule:" in out["system"]
    assert "FOCUS BLOCK" in out["user"]
    assert "SUPPORT BLOCK" in out["user"]
    assert "Paper-guide upstream reference opportunities:" in out["user"]
    assert "EVIDENCE BLOCK" in out["user"]
    assert "GROUNDING BLOCK" in out["user"]
    assert "Attached images: 2." in out["user"]


def test_build_generation_prompt_bundle_does_not_force_ref_num_without_candidates():
    out = _build_generation_prompt_bundle(
        prompt="How is APR grounded?",
        ctx="DOC-1 [SID:s12345678] demo\nAPR was performed using image registration.",
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
        paper_guide_special_focus_block="",
        paper_guide_support_slots_block="",
        paper_guide_evidence_cards_block="",
        paper_guide_citation_grounding_block="",
        image_attachment_count=0,
    )

    assert "Citation source lock:" in out["system"]
    assert "Include at least one valid" not in out["system"]
    assert "[[CITE:s12345678:<ref_num>]]" not in out["system"]
    assert "do not invent a ref_num" in out["system"]


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
    assert out["research_answer_plan"] == "literature_positioning"
