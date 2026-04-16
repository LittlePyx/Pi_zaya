from kb.paper_guide_prompting import (
    _augment_paper_guide_retrieval_prompt,
    _build_paper_guide_citation_grounding_block,
    _build_paper_guide_evidence_cards_block,
    _looks_like_reference_list_snippet_local,
    _merge_paper_guide_deepread_context,
    _paper_guide_prompt_family,
    _paper_guide_requested_heading_hints,
    _paper_guide_text_matches_requested_targets,
    _requested_figure_number,
)


def test_paper_guide_prompt_family_detects_core_families():
    assert _paper_guide_prompt_family("What problem does this paper solve?") == "overview"
    assert _paper_guide_prompt_family("Walk me through what Figure 2 shows.") == "figure_walkthrough"
    assert (
        _paper_guide_prompt_family(
            "What does Equation (1) define in this paper, and where do the authors define the variables like t_n and t_f?"
        )
        == "equation"
    )
    assert _paper_guide_prompt_family("From Box 1 only, what condition is given for reconstruction?") == "box_only"
    assert _paper_guide_prompt_family("From the Discussion only, what future directions do the authors suggest?") == "discussion_only"
    assert (
        _paper_guide_prompt_family(
            "Which prior work is RVT attributed to in this paper, and what in-paper citation do they use?"
        )
        == "citation_lookup"
    )


def test_paper_guide_prompt_family_prefers_overview_for_beginner_problem_and_application_questions():
    assert (
        _paper_guide_prompt_family(
            "I am a beginner. What problem is this paper solving, and what is the basic idea of the method?"
        )
        == "overview"
    )
    assert (
        _paper_guide_prompt_family(
            "I am just getting started. Why is single-pixel imaging interesting, and what kinds of applications does this review emphasize?"
        )
        == "overview"
    )
    assert (
        _paper_guide_prompt_family(
            "I am new to this paper. What are RVT and APR doing here, in simple terms?"
        )
        == "overview"
    )


def test_paper_guide_prompt_family_prefers_method_for_training_summary_questions():
    assert (
        _paper_guide_prompt_family(
            "If I wanted a beginner-level summary of how they train the network, what should I pay attention to first?"
        )
        == "method"
    )


def test_paper_guide_prompt_family_prefers_strength_limits_for_section_tradeoff_prompt():
    assert (
        _paper_guide_prompt_family(
            "In the 'How a single-pixel camera works' section only, what trade-off do the authors describe between the advantages of single-pixel imaging and the detector dynamic range?"
        )
        == "strength_limits"
    )


def test_paper_guide_prompt_family_prefers_strength_limits_for_discussion_limitation_prompt():
    assert (
        _paper_guide_prompt_family(
            "From the Discussion section only, what limitation do the authors note about calibrating different SPAD arrays, and what follow-up do they suggest?"
        )
        == "strength_limits"
    )


def test_paper_guide_prompt_family_keeps_explicit_comparison_tradeoff_as_compare():
    assert (
        _paper_guide_prompt_family(
            "Compared with the open-pinhole condition, what trade-off do the authors report for iISM-APR in terms of CNR versus resolution?"
        )
        == "compare"
    )


def test_paper_guide_requested_heading_hints_include_figure_targets():
    hints = _paper_guide_requested_heading_hints("Explain Figure 3 panels (c) and (d).")
    assert "figure 3" in [item.lower() for item in hints]
    assert "caption" in [item.lower() for item in hints]
    assert "panel" in [item.lower() for item in hints]


def test_paper_guide_requested_heading_hints_include_literal_section_title():
    hints = _paper_guide_requested_heading_hints(
        "In the 'How a single-pixel camera works' section only, what trade-off do the authors describe?"
    )
    assert "how a single-pixel camera works" in [item.lower() for item in hints]


def test_augment_paper_guide_retrieval_prompt_avoids_reference_list_bias_for_citation_lookup():
    out = _augment_paper_guide_retrieval_prompt(
        "Which prior work is RVT attributed to in this paper, and what in-paper citation do they use when introducing it?",
        family="citation_lookup",
    )
    low = out.lower()
    assert "reference list" not in low
    assert "works cited" not in low
    assert "attributed to" in low


def test_augment_paper_guide_retrieval_prompt_keeps_reference_list_bias_for_explicit_request():
    out = _augment_paper_guide_retrieval_prompt(
        "From the reference list only, which entries correspond to the Richardson-Lucy method?",
        family="citation_lookup",
    )
    low = out.lower()
    assert "reference list" in low
    assert "works cited" in low


def test_paper_guide_text_matches_requested_targets_for_box_and_section():
    assert _paper_guide_text_matches_requested_targets(
        "Box 1. The transform-domain condition is M >= O(K log(N/K)).",
        prompt="From Box 1 only, what condition on M is given?",
    )
    assert _paper_guide_text_matches_requested_targets(
        "Discussion. The authors suggest future extensions to faster hardware.",
        prompt="From the Discussion only, what future directions do the authors suggest?",
    )


def test_paper_guide_text_matches_requested_targets_for_future_work_and_literal_section_title():
    assert _paper_guide_text_matches_requested_targets(
        "### Future Work\nThe authors suggest adaptive masking for dynamic scenes.",
        prompt="From the Future Work section only, what extension do the authors suggest next?",
    )
    assert _paper_guide_text_matches_requested_targets(
        "### How a single-pixel camera works\nThe trade-off is that the detector stays simple while dynamic range becomes the bottleneck.",
        prompt=(
            "In the 'How a single-pixel camera works' section only, what trade-off do the authors describe "
            "between the advantages of single-pixel imaging and the detector dynamic range?"
        ),
    )


def test_looks_like_reference_list_snippet_local_distinguishes_reference_entries():
    assert _looks_like_reference_list_snippet_local("[34] Smith et al. Precision single-particle localization using radial variance transform. 2020.")
    assert not _looks_like_reference_list_snippet_local(
        "We used radial variance transform [34] to estimate the shift vectors."
    )


def test_merge_paper_guide_deepread_context_replaces_title_shell_for_abstract():
    out = _merge_paper_guide_deepread_context(
        "Title only shell\nAuthor list",
        "# Abstract\nThe paper measures contrast with low power.",
        prompt_family="abstract",
    )
    assert out.startswith("Title only shell\nThe paper measures contrast with low power.")
    assert "Author list" not in out


def test_build_paper_guide_evidence_cards_block_formats_doc_scoped_cards():
    block = _build_paper_guide_evidence_cards_block(
        [
            {
                "doc_idx": 2,
                "sid": "s123",
                "heading": "Methods > APR",
                "candidate_refs": [34],
                "snippet": "We used RVT [34] to estimate shift vectors.",
                "deepread_texts": [],
                "cue": "RVT [34]",
            }
        ],
        prompt_family="citation_lookup",
    )
    assert "DOC-2" in block
    assert "sid=s123" in block
    assert "refs=34" in block
    assert "cite_example=[[CITE:s123:34]]" in block
    assert "RVT [34]" in block


def test_build_paper_guide_citation_grounding_block_formats_doc_scoped_candidates():
    hit = {"meta": {"source_path": "paper.md", "heading_path": "Methods > APR"}}
    block = _build_paper_guide_citation_grounding_block(
        [hit],
        hit_source_path=lambda item: str((item.get("meta") or {}).get("source_path") or ""),
        paper_guide_focus_heading=lambda item: str((item.get("meta") or {}).get("heading_path") or ""),
        cite_source_id=lambda _src: "s456",
        extract_candidate_ref_nums=lambda _hits, **_kwargs: [34, 35],
        extract_candidate_ref_cue_texts=lambda _hit, **_kwargs: ["RVT [34]"],
    )
    assert "Paper-guide citation grounding hints:" in block
    assert "DOC-1" in block
    assert "sid=s456" in block
    assert "refs=34, 35" in block
    assert "heading=Methods > APR" in block
    assert "cue=RVT [34]" in block


def test_requested_figure_number_falls_back_to_answer_hit_anchor_target():
    n = _requested_figure_number(
        "Walk me through the relevant panels.",
        [{"meta": {"anchor_target_kind": "figure", "anchor_target_number": 3}}],
    )
    assert n == 3
