import kb.paper_guide.skills as paper_guide_skills


def test_run_exact_equation_skill_returns_sanitized_skill_result():
    result = paper_guide_skills.run_exact_equation_skill(
        prompt_text="Explain Equation (1) and point me to the exact support.",
        prompt_family="equation",
        source_path="bound.md",
        db_dir="db",
        has_hits=False,
        prompt_requests_exact_support=lambda prompt: "equation" in prompt.lower(),
        resolve_exact_support=lambda source_path, **_kwargs: {
            "source_path": source_path,
            "equation_markdown": "$$ E = mc^2 $$",
        },
        build_exact_answer=lambda rec: (
            f"Equation support from {rec['source_path']}",
            [{"block_id": "blk_eq", "locate_anchor": rec["equation_markdown"]}],
        ),
        sanitize_answer=lambda answer, **_kwargs: f"{answer}\n[sanitized]",
    )

    assert result is not None
    assert result.answer_text.endswith("[sanitized]")
    assert result.support_resolution == [{"block_id": "blk_eq", "locate_anchor": "$$ E = mc^2 $$"}]


def test_run_exact_citation_lookup_skill_extracts_inline_refs_when_record_missing_ref_nums():
    result = paper_guide_skills.run_exact_citation_lookup_skill(
        prompt_text="Which reference do they cite here, and where is that stated exactly?",
        prompt_family="citation_lookup",
        source_path="bound.md",
        db_dir="db",
        has_hits=True,
        prompt_requests_exact_support=lambda _prompt: True,
        resolve_exact_support=lambda source_path, **_kwargs: {
            "source_path": source_path,
            "block_id": "blk_rel",
            "heading_path": "Related Work",
            "locate_anchor": "The method follows ADMM [4,7].",
        },
        extract_inline_reference_numbers=lambda anchor, **_kwargs: [4, 7] if anchor else [],
        sanitize_answer=lambda answer, **_kwargs: answer,
    )

    assert result is not None
    assert "The paper cites [4], [7] for this point." in result.answer_text
    assert result.support_resolution == [
        {
            "source_path": "bound.md",
            "block_id": "blk_rel",
            "heading_path": "Related Work",
            "locate_anchor": "The method follows ADMM [4,7].",
            "candidate_refs": [4, 7],
            "resolved_ref_num": 4,
        }
    ]


def test_run_exact_figure_panel_skill_surfaces_clause_references():
    result = paper_guide_skills.run_exact_figure_panel_skill(
        prompt_text="For Figure 3 panel (f), show me the exact caption clause.",
        prompt_family="figure_walkthrough",
        source_path="bound.md",
        db_dir="db",
        has_hits=False,
        prompt_requests_exact_support=lambda _prompt: True,
        resolve_exact_support=lambda source_path, **_kwargs: {
            "source_path": source_path,
            "block_id": "blk_cap",
            "heading_path": "Results / Figure 3",
            "locate_anchor": "(f) methane imaging using SPC$^{15}$",
            "figure_number": 3,
            "panel_letters": ["f"],
        },
        extract_clause_reference_numbers=lambda _anchor, **_kwargs: [15],
        sanitize_answer=lambda answer, **_kwargs: answer,
    )

    assert result is not None
    assert "Figure 3 caption for panel (f) states:" in result.answer_text
    assert "References in this clause: [15]" in result.answer_text
    assert result.support_resolution == [
        {
            "source_path": "bound.md",
            "block_id": "blk_cap",
            "heading_path": "Results / Figure 3",
            "locate_anchor": "(f) methane imaging using SPC$^{15}$",
            "figure_number": 3,
            "panel_letters": ["f"],
            "candidate_refs": [15],
            "resolved_ref_num": 15,
        }
    ]


def test_run_exact_method_skill_sets_segment_metadata():
    result = paper_guide_skills.run_exact_method_skill(
        prompt_text="Where exactly do they say the shift vectors are applied back?",
        prompt_family="method",
        source_path="bound.md",
        db_dir="db",
        has_hits=False,
        prompt_requests_exact_support=lambda _prompt: True,
        resolve_exact_support=lambda source_path, **_kwargs: {
            "source_path": source_path,
            "block_id": "blk_exact",
            "heading_path": "Results / APR",
            "locate_anchor": "Finally, these RVT-APR shift vectors were applied back to the original iISM dataset.",
        },
        sanitize_answer=lambda answer, **_kwargs: answer,
    )

    assert result is not None
    assert "Results / APR" in result.answer_text
    assert result.support_resolution == [
        {
            "source_path": "bound.md",
            "block_id": "blk_exact",
            "heading_path": "Results / APR",
            "locate_anchor": "Finally, these RVT-APR shift vectors were applied back to the original iISM dataset.",
            "segment_text": "Finally, these RVT-APR shift vectors were applied back to the original iISM dataset.",
            "segment_index": -1,
        }
    ]


def test_run_overview_component_role_skill_builds_broad_skill_answer():
    result = paper_guide_skills.run_overview_component_role_skill(
        prompt_text="I am new to this paper. What are RVT and APR doing here, in simple terms?",
        prompt_family="overview",
        source_path="bound.md",
        db_dir="db",
        has_hits=True,
        prompt_requests_component_role_explanation=lambda _prompt: True,
        extract_method_focus_terms=lambda _prompt: ["RVT", "APR"],
        extract_component_role_focus=lambda source_path, **_kwargs: (
            f"Focus excerpt from {source_path}: RVT converts each pinhole image into a radial-symmetry map. "
            "APR uses phase-correlation registration to estimate shift vectors."
        ),
        build_overview_role_lines=lambda _text, **_kwargs: [
            "RVT converts each pinhole image into a radial-symmetry map.",
            "APR uses phase-correlation registration to estimate shift vectors.",
        ],
        sanitize_answer=lambda answer, **_kwargs: answer,
    )

    assert result is not None
    assert "retrieved method evidence" in result.answer_text.lower()
    assert "radial-symmetry map" in result.answer_text
    assert result.support_resolution == [
        {
            "source_path": "bound.md",
            "locate_anchor": (
                "Focus excerpt from bound.md: RVT converts each pinhole image into a radial-symmetry map. "
                "APR uses phase-correlation registration to estimate shift vectors."
            ),
            "segment_text": (
                "Focus excerpt from bound.md: RVT converts each pinhole image into a radial-symmetry map. "
                "APR uses phase-correlation registration to estimate shift vectors."
            ),
            "claim_type": "overview_component_role",
            "segment_index": -1,
        }
    ]


def test_run_section_target_skill_builds_strength_limits_answer():
    result = paper_guide_skills.run_section_target_skill(
        prompt_text="From the Discussion section only, what limitation do the authors note about calibrating different SPAD arrays?",
        prompt_family="strength_limits",
        source_path="bound.md",
        db_dir="db",
        has_hits=False,
        has_non_ref_target=True,
        select_target_hit=lambda source_path, **_kwargs: {
            "text": "Different SPAD arrays may deviate from each other, so automatic calibration is worthy of further study.",
            "meta": {
                "source_path": source_path,
                "heading_path": "Discussion",
                "kind": "paragraph",
                "block_id": "blk_discuss",
            },
        },
        extract_discussion_future_snippet=lambda text: text,
        sanitize_answer=lambda answer, **_kwargs: answer,
    )

    assert result is not None
    low = result.answer_text.lower()
    assert "discussion section" in low
    assert "different spad arrays may deviate" in low
    assert result.support_resolution == [
        {
            "source_path": "bound.md",
            "heading_path": "Discussion",
            "kind": "paragraph",
            "block_id": "blk_discuss",
            "locate_anchor": "Different SPAD arrays may deviate from each other, so automatic calibration is worthy of further study.",
            "segment_text": "Different SPAD arrays may deviate from each other, so automatic calibration is worthy of further study.",
            "claim_type": "strength_limits",
            "segment_index": -1,
        }
    ]


def test_run_abstract_skill_returns_direct_answer_and_anchor_support():
    result = paper_guide_skills.run_abstract_skill(
        prompt_text="Please summarize the abstract and give me one exact anchor sentence.",
        prompt_family="abstract",
        source_path="direct.md",
        db_dir="db",
        llm="llm-client",
        build_direct_abstract_answer=lambda **kwargs: (
            "Abstract text:\nA compact abstract.\n\nAnchor sentence for locate jump:\n> A compact abstract."
            if kwargs["source_path"] == "direct.md" and kwargs["llm"] == "llm-client"
            else ""
        ),
        sanitize_answer=lambda answer, **_kwargs: answer,
    )

    assert result is not None
    assert "Abstract text:" in result.answer_text
    assert result.support_resolution == [
        {
            "source_path": "direct.md",
            "locate_anchor": "A compact abstract.",
            "segment_text": "A compact abstract.",
            "claim_type": "abstract",
            "segment_index": -1,
        }
    ]


def test_run_box_target_skill_builds_targeted_box_answer():
    result = paper_guide_skills.run_box_target_skill(
        prompt_text="From Box 1 only, what condition do the authors give for reconstructing the image in the transform domain?",
        prompt_family="box_only",
        source_path="bound.md",
        db_dir="db",
        has_hits=False,
        has_non_ref_target=True,
        select_target_hit=lambda source_path, **_kwargs: {
            "text": "In the transform domain, the image can be reconstructed even if M < N.",
            "meta": {
                "source_path": source_path,
                "heading_path": "Box 1",
                "kind": "paragraph",
                "block_id": "blk_box1",
            },
        },
        extract_box_numbers=lambda _prompt: [1],
        sanitize_answer=lambda answer, **_kwargs: answer,
    )

    assert result is not None
    assert "From Box 1" in result.answer_text
    assert "M < N" in result.answer_text
    assert result.support_resolution == [
        {
            "source_path": "bound.md",
            "heading_path": "Box 1",
            "kind": "paragraph",
            "block_id": "blk_box1",
            "locate_anchor": "In the transform domain, the image can be reconstructed even if M < N.",
            "segment_text": "In the transform domain, the image can be reconstructed even if M < N.",
            "claim_type": "box_only",
            "segment_index": -1,
        }
    ]
