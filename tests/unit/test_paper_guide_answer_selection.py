from kb.paper_guide_answer_selection import (
    _build_answer_hits_for_generation,
    _has_anchor_grounded_answer_hits,
    _paper_guide_focus_heading,
    _select_paper_guide_answer_hits,
    _stabilize_paper_guide_output_mode,
)


def test_stabilize_paper_guide_output_mode_prevents_generic_overview_from_drifting_critical():
    out = _stabilize_paper_guide_output_mode(
        "critical_review",
        prompt="What problem does this paper solve, and what are its core contributions?",
        intent="reading",
        explicit_hint="",
    )
    assert out == "reading_guide"


def test_paper_guide_focus_heading_prefers_specific_suffix_over_generic_prefix():
    hit = {
        "meta": {
            "heading_path": "Abstract / Materials and Methods / Adaptive pixel-reassignment (APR)",
        }
    }

    assert _paper_guide_focus_heading(hit) == "Materials and Methods / Adaptive pixel-reassignment (APR)"


def test_select_paper_guide_answer_hits_prefers_box_target_over_generic_sections():
    src = r"db\demo\paper.en.md"
    hits = [
        {
            "score": 18.0,
            "text": "## Introduction\nThis section motivates the problem.",
            "meta": {"source_path": src, "heading_path": "Introduction"},
        },
        {
            "score": 12.0,
            "text": "**[Box 1 - The maths behind single-pixel imaging]**\nWhen M >= O(K log(N/K)), the image can be reconstructed.",
            "meta": {
                "source_path": src,
                "heading_path": "Acquisition / Box 1",
                "paper_guide_targeted_block": True,
            },
        },
    ]

    out = _select_paper_guide_answer_hits(
        grouped_docs=[],
        heading_hits=hits,
        prompt="From Box 1 only, what condition on M is given for reconstructing the image in the transform domain?",
        top_n=1,
    )

    assert len(out) == 1
    assert "Box 1" in str(out[0].get("text") or "")


def test_select_paper_guide_answer_hits_prefers_intext_attribution_over_reference_list():
    src = r"db\demo\paper.en.md"
    hits = [
        {
            "score": 18.0,
            "text": "[33] Richardson, W. H. Bayesian-based iterative method of image restoration.",
            "meta": {
                "source_path": src,
                "heading_path": "References",
                "block_id": "blk_refs",
            },
        },
        {
            "score": 15.0,
            "text": (
                "We invert the ISM image-formation model using a maximum likelihood estimation technique "
                "akin to the Richardson-Lucy method [33,34]."
            ),
            "meta": {
                "source_path": src,
                "heading_path": "Abstract / Results",
                "block_id": "blk_intext",
            },
        },
    ]

    out = _select_paper_guide_answer_hits(
        grouped_docs=[],
        heading_hits=hits,
        prompt="Which references does the paper cite for the maximum-likelihood / Richardson-Lucy connection, and where is that stated exactly?",
        top_n=1,
    )

    assert len(out) == 1
    assert str((out[0].get("meta") or {}).get("block_id") or "") == "blk_intext"


def test_select_paper_guide_answer_hits_prefers_exact_equation_hit_over_generic_intro():
    src = r"db\demo\paper.en.md"
    hits = [
        {
            "score": 18.0,
            "text": "We build on neural radiance fields to recover the scene from a compressed image.",
            "meta": {
                "source_path": src,
                "heading_path": "1. Introduction",
                "block_id": "blk_intro",
            },
        },
        {
            "score": 10.0,
            "text": (
                "$$\n"
                "C(\\mathbf{r}) = \\int_{t_n}^{t_f} T(t)\\sigma(\\mathbf{r}(t))\\mathbf{c}(\\mathbf{r}(t),\\mathbf{d})dt, \\tag{1}\n"
                "$$\n"
                "where t_n and t_f are near and far bounds."
            ),
            "meta": {
                "source_path": src,
                "heading_path": "3. Method / 3.1. Background on NeRF",
                "block_id": "blk_eq1",
                "anchor_target_kind": "equation",
                "anchor_target_number": 1,
            },
        },
    ]

    out = _select_paper_guide_answer_hits(
        grouped_docs=[],
        heading_hits=hits,
        prompt=(
            "What does Equation (1) define in this paper, and where do the authors define the variables "
            "like t_n and t_f? Point me to the exact supporting part."
        ),
        top_n=1,
    )

    assert len(out) == 1
    assert str((out[0].get("meta") or {}).get("block_id") or "") == "blk_eq1"


def test_build_answer_hits_for_generation_keeps_one_hit_per_source_by_default():
    grouped_docs = [
        {"text": "A1", "meta": {"source_path": "db/a.md"}},
        {"text": "A2", "meta": {"source_path": "db/a.md"}},
    ]
    heading_hits = [
        {"text": "B1", "meta": {"source_path": "db/b.md"}},
    ]

    out = _build_answer_hits_for_generation(
        grouped_docs=grouped_docs,
        heading_hits=heading_hits,
        top_n=3,
    )

    assert [str(item.get("text") or "") for item in out] == ["A1", "B1"]


def test_has_anchor_grounded_answer_hits_detects_positive_anchor_match():
    hits = [
        {
            "meta": {
                "anchor_target_kind": "equation",
                "anchor_match_score": 14.5,
            }
        }
    ]

    assert _has_anchor_grounded_answer_hits(hits) is True
