import kb.paper_guide_router as legacy_router
from kb.paper_guide.router import (
    PaperGuideBroadSkillDeps,
    PaperGuideExactSkillDeps,
    _dispatch_paper_guide_broad_skill,
    _dispatch_paper_guide_exact_support_skill,
    _resolve_paper_guide_intent,
)


def test_resolve_paper_guide_intent_marks_beginner_overview_prompt():
    out = _resolve_paper_guide_intent("给初学者简单讲一下这篇论文主要解决什么问题，核心思路是什么？")

    assert out.family == "overview"
    assert out.beginner_mode is True
    assert out.exact_support is False
    assert out.target_figure == 0
    assert out.target_equation == 0


def test_resolve_paper_guide_intent_prefers_overview_for_beginner_role_prompt():
    out = _resolve_paper_guide_intent("I am new to this paper. What are RVT and APR doing here, in simple terms?")

    assert out.family == "overview"
    assert out.beginner_mode is True
    assert out.exact_support is False


def test_resolve_paper_guide_intent_extracts_exact_figure_panel_target():
    out = _resolve_paper_guide_intent("请给我 Figure 6 panel (b) 的 exact caption clause 原文")

    assert out.family == "figure_walkthrough"
    assert out.exact_support is True
    assert out.target_figure == 6
    assert out.target_panels == ["b"]
    assert out.target_scope["target_figure_num"] == 6
    assert out.target_scope["target_panel_letters"] == ["b"]


def test_resolve_paper_guide_intent_extracts_exact_equation_target():
    out = _resolve_paper_guide_intent("Where do the authors define the variables in equation (3)?")

    assert out.family == "equation"
    assert out.exact_support is True
    assert out.target_equation == 3


def test_resolve_paper_guide_intent_extracts_exact_citation_lookup_target():
    out = _resolve_paper_guide_intent("这句话文内引用编号是什么？原文哪里明确写到？")

    assert out.family == "citation_lookup"
    assert out.exact_support is True


def test_resolve_paper_guide_intent_marks_exact_method_support_prompt():
    out = _resolve_paper_guide_intent("Point me to the exact supporting sentence for the optimizer and batch size.")

    assert out.family == "method"
    assert out.exact_support is True


def test_resolve_paper_guide_intent_respects_explicit_family_override():
    out = _resolve_paper_guide_intent(
        "What problem does this paper solve?",
        prompt_family="method",
    )

    assert out.family == "method"


def test_legacy_router_shim_still_exposes_intent_resolution():
    out = legacy_router._resolve_paper_guide_intent("Where do the authors define the variables in equation (3)?")

    assert out.family == "equation"
    assert out.target_equation == 3


def test_dispatch_paper_guide_exact_support_skill_routes_equation_family():
    calls: list[str] = []

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("unexpected skill route")

    result = _dispatch_paper_guide_exact_support_skill(
        prompt_text="Where do the authors define the variables in equation (3)?",
        resolved_intent=_resolve_paper_guide_intent("Where do the authors define the variables in equation (3)?"),
        source_path="bound.md",
        db_dir="db",
        has_hits=False,
        deps=PaperGuideExactSkillDeps(
            resolve_exact_method_support=_unexpected,
            resolve_exact_equation_support=lambda source_path, **_kwargs: calls.append("equation") or {
                "source_path": source_path,
                "equation_markdown": "$$ x = y $$",
            },
            build_exact_equation_answer=lambda rec: (
                f"Equation from {rec['source_path']}",
                [{"block_id": "blk_eq"}],
            ),
            resolve_exact_citation_lookup_support=_unexpected,
            extract_inline_reference_numbers=lambda *_args, **_kwargs: [],
            resolve_exact_figure_panel_caption_support=_unexpected,
            extract_caption_clause_superscript_ref_nums=lambda *_args, **_kwargs: [],
            sanitize_answer=lambda answer, **_kwargs: answer,
        ),
    )

    assert calls == ["equation"]
    assert result is not None
    assert result.answer_text == "Equation from bound.md"
    assert result.support_resolution == [{"block_id": "blk_eq"}]


def test_dispatch_paper_guide_exact_support_skill_routes_citation_lookup_family():
    calls: list[str] = []

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("unexpected skill route")

    result = _dispatch_paper_guide_exact_support_skill(
        prompt_text="Which reference do they cite here, and where is that stated exactly?",
        resolved_intent=_resolve_paper_guide_intent(
            "Which reference do they cite here, and where is that stated exactly?",
            prompt_family="citation_lookup",
        ),
        source_path="bound.md",
        db_dir="db",
        has_hits=True,
        deps=PaperGuideExactSkillDeps(
            resolve_exact_method_support=_unexpected,
            resolve_exact_equation_support=_unexpected,
            build_exact_equation_answer=lambda _rec: ("", []),
            resolve_exact_citation_lookup_support=lambda source_path, **_kwargs: calls.append("citation") or {
                "source_path": source_path,
                "heading_path": "Related Work",
                "locate_anchor": "The method follows ADMM [4].",
                "ref_nums": [4],
            },
            extract_inline_reference_numbers=lambda *_args, **_kwargs: [],
            resolve_exact_figure_panel_caption_support=_unexpected,
            extract_caption_clause_superscript_ref_nums=lambda *_args, **_kwargs: [],
            sanitize_answer=lambda answer, **_kwargs: answer,
        ),
    )

    assert calls == ["citation"]
    assert result is not None
    assert "The paper cites [4] for this point." in result.answer_text
    assert result.support_resolution[0]["resolved_ref_num"] == 4


def test_dispatch_paper_guide_broad_skill_routes_component_role_overview():
    calls: list[str] = []

    result = _dispatch_paper_guide_broad_skill(
        prompt_text="I am new to this paper. What are RVT and APR doing here, in simple terms?",
        resolved_intent=_resolve_paper_guide_intent("I am new to this paper. What are RVT and APR doing here, in simple terms?"),
        source_path="bound.md",
        db_dir="db",
        has_hits=True,
        has_non_ref_target=False,
        deps=PaperGuideBroadSkillDeps(
            build_direct_abstract_answer=lambda **_kwargs: "",
            prompt_requests_component_role_explanation=lambda _prompt: True,
            extract_method_focus_terms=lambda _prompt: calls.append("terms") or ["RVT", "APR"],
            extract_component_role_focus=lambda source_path, **_kwargs: calls.append(f"focus:{source_path}") or "RVT ... APR ...",
            build_overview_role_lines=lambda _text, **_kwargs: calls.append("lines") or [
                "RVT converts each pinhole image into a radial-symmetry map.",
                "APR uses phase-correlation registration to estimate shift vectors.",
            ],
            select_section_target_hit=lambda *_args, **_kwargs: {},
            extract_discussion_future_snippet=lambda text: text,
            extract_box_numbers=lambda _prompt: [],
            sanitize_answer=lambda answer, **_kwargs: answer,
        ),
    )

    assert calls == ["terms", "focus:bound.md", "lines"]
    assert result is not None
    assert "retrieved method evidence" in result.answer_text.lower()
    assert result.support_resolution[0]["claim_type"] == "overview_component_role"


def test_dispatch_paper_guide_broad_skill_routes_section_target_family():
    calls: list[str] = []

    result = _dispatch_paper_guide_broad_skill(
        prompt_text="From the Discussion section only, what future directions do the authors suggest?",
        resolved_intent=_resolve_paper_guide_intent(
            "From the Discussion section only, what future directions do the authors suggest?",
            prompt_family="discussion_only",
        ),
        source_path="bound.md",
        db_dir="db",
        has_hits=False,
        has_non_ref_target=True,
        deps=PaperGuideBroadSkillDeps(
            build_direct_abstract_answer=lambda **_kwargs: "",
            prompt_requests_component_role_explanation=lambda _prompt: False,
            extract_method_focus_terms=lambda _prompt: [],
            extract_component_role_focus=lambda *_args, **_kwargs: "",
            build_overview_role_lines=lambda *_args, **_kwargs: [],
            select_section_target_hit=lambda source_path, **_kwargs: calls.append(f"section:{source_path}") or {
                "text": "Parallelized detection schemes could accelerate imaging.",
                "meta": {
                    "heading_path": "Discussion",
                    "kind": "paragraph",
                },
            },
            extract_discussion_future_snippet=lambda text: calls.append("snippet") or text,
            extract_box_numbers=lambda _prompt: [],
            sanitize_answer=lambda answer, **_kwargs: answer,
        ),
    )

    assert calls == ["section:bound.md", "snippet"]
    assert result is not None
    assert "parallelized detection schemes" in result.answer_text.lower()
    assert result.support_resolution[0]["claim_type"] == "discussion_only"


def test_dispatch_paper_guide_broad_skill_routes_abstract_family():
    calls: list[str] = []

    result = _dispatch_paper_guide_broad_skill(
        prompt_text="Summarize the abstract and give me an anchor sentence.",
        resolved_intent=_resolve_paper_guide_intent(
            "Summarize the abstract and give me an anchor sentence.",
            prompt_family="abstract",
        ),
        source_path="direct.md",
        db_dir="db",
        has_hits=False,
        has_non_ref_target=True,
        llm="llm-client",
        deps=PaperGuideBroadSkillDeps(
            build_direct_abstract_answer=lambda **kwargs: calls.append(f"abstract:{kwargs['source_path']}:{kwargs['llm']}") or (
                "Abstract text:\nA compact abstract.\n\nAnchor sentence for locate jump:\n> A compact abstract."
            ),
            prompt_requests_component_role_explanation=lambda _prompt: False,
            extract_method_focus_terms=lambda _prompt: [],
            extract_component_role_focus=lambda *_args, **_kwargs: "",
            build_overview_role_lines=lambda *_args, **_kwargs: [],
            select_section_target_hit=lambda *_args, **_kwargs: {},
            extract_discussion_future_snippet=lambda text: text,
            extract_box_numbers=lambda _prompt: [],
            sanitize_answer=lambda answer, **_kwargs: answer,
        ),
    )

    assert calls == ["abstract:direct.md:llm-client"]
    assert result is not None
    assert "Abstract text:" in result.answer_text
    assert result.support_resolution[0]["claim_type"] == "abstract"


def test_dispatch_paper_guide_broad_skill_routes_box_only_family():
    calls: list[str] = []

    result = _dispatch_paper_guide_broad_skill(
        prompt_text="From Box 1 only, what condition do the authors give for reconstructing the image in the transform domain?",
        resolved_intent=_resolve_paper_guide_intent(
            "From Box 1 only, what condition do the authors give for reconstructing the image in the transform domain?",
            prompt_family="box_only",
        ),
        source_path="bound.md",
        db_dir="db",
        has_hits=False,
        has_non_ref_target=True,
        deps=PaperGuideBroadSkillDeps(
            build_direct_abstract_answer=lambda **_kwargs: "",
            prompt_requests_component_role_explanation=lambda _prompt: False,
            extract_method_focus_terms=lambda _prompt: [],
            extract_component_role_focus=lambda *_args, **_kwargs: "",
            build_overview_role_lines=lambda *_args, **_kwargs: [],
            select_section_target_hit=lambda source_path, **_kwargs: calls.append(f"box:{source_path}") or {
                "text": "In the transform domain, the image can be reconstructed even if M < N.",
                "meta": {
                    "heading_path": "Box 1",
                    "kind": "paragraph",
                },
            },
            extract_discussion_future_snippet=lambda text: text,
            extract_box_numbers=lambda _prompt: [1],
            sanitize_answer=lambda answer, **_kwargs: answer,
        ),
    )

    assert calls == ["box:bound.md"]
    assert result is not None
    assert "From Box 1" in result.answer_text
    assert result.support_resolution[0]["claim_type"] == "box_only"
