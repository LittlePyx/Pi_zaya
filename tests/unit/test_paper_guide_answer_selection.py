from kb.paper_guide_answer_selection import (
    _build_answer_hits_for_generation,
    _has_anchor_grounded_answer_hits,
    _paper_guide_focus_heading,
    _rescue_multi_source_answer_hits,
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


def test_select_paper_guide_answer_hits_prefers_target_paragraph_over_heading_only_shell():
    src = r"db\demo\paper.en.md"
    hits = [
        {
            "score": 42.0,
            "text": "How a single-pixel camera works",
            "meta": {
                "source_path": src,
                "heading_path": "Abstract / How a single-pixel camera works",
                "block_id": "blk_heading",
                "kind": "heading",
                "paper_guide_targeted_block": True,
            },
        },
        {
            "score": 42.0,
            "text": (
                "A detailed comparison shows that there is a trade-off between the advantages of "
                "single-pixel imaging and the dynamic range of the detector and associated quantization electronics."
            ),
            "meta": {
                "source_path": src,
                "heading_path": "Abstract / How a single-pixel camera works",
                "block_id": "blk_para",
                "kind": "paragraph",
                "paper_guide_targeted_block": True,
            },
        },
    ]

    out = _select_paper_guide_answer_hits(
        grouped_docs=[],
        heading_hits=hits,
        prompt=(
            "In the 'How a single-pixel camera works' section only, what trade-off do the authors describe "
            "between the advantages of single-pixel imaging and the detector dynamic range?"
        ),
        top_n=1,
    )

    assert len(out) == 1
    assert str((out[0].get("meta") or {}).get("block_id") or "") == "blk_para"


def test_select_paper_guide_answer_hits_prefers_author_biographies_over_high_score_abstract():
    src = r"db\demo\paper.en.md"
    hits = [
        {
            "score": 272.0,
            "text": "## Abstract\nThis review surveys deep-learning single-pixel imaging.",
            "meta": {
                "source_path": src,
                "heading_path": "Abstract",
                "block_id": "blk_abstract",
                "paper_guide_targeted_block": True,
            },
        },
        {
            "score": 6.0,
            "text": (
                "Kai Song received his B.S. degree in 2019 and is currently pursuing his Ph.D. degree. "
                "His research interests include single-photon imaging and single-pixel imaging."
            ),
            "meta": {
                "source_path": src,
                "heading_path": "Author Biographies",
                "block_id": "blk_kai_song",
                "paper_guide_targeted_block": True,
            },
        },
    ]

    out = _select_paper_guide_answer_hits(
        grouped_docs=[],
        heading_hits=hits,
        prompt="请概括作者 Kai Song 的学历、当前职位和研究方向，并定位 Author Biographies。",
        top_n=1,
    )

    assert len(out) == 1
    assert str((out[0].get("meta") or {}).get("block_id") or "") == "blk_kai_song"


def test_select_paper_guide_answer_hits_keeps_multiple_targeted_blocks_in_same_heading():
    src = r"db\demo\nat.en.md"
    hits = [
        {
            "score": 99.0,
            "text": "Optimization algorithms may minimize the l1 norm or total variation; reconstruction time can exceed acquisition time.",
            "meta": {
                "source_path": src,
                "heading_path": "Abstract / Acquisition and image reconstruction strategies",
                "block_id": "blk_optimization",
                "paper_guide_targeted_block": True,
            },
        },
        {
            "score": 86.0,
            "text": "Hadamard, Fourier or wavelet basis patterns can be reconstructed with a computationally fast transform.",
            "meta": {
                "source_path": src,
                "heading_path": "Abstract / Acquisition and image reconstruction strategies",
                "block_id": "blk_fast_basis",
                "paper_guide_targeted_block": True,
            },
        },
    ]

    out = _select_paper_guide_answer_hits(
        grouped_docs=[],
        heading_hits=hits,
        prompt="文中提到的几类主流重建方法分别有什么优缺点，适用场景怎么选？",
        top_n=3,
    )

    assert [str((hit.get("meta") or {}).get("block_id") or "") for hit in out] == [
        "blk_optimization",
        "blk_fast_basis",
    ]


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


def test_multi_source_answer_rescue_replaces_low_value_representative_from_same_source():
    structured_source = r"db\structured-detection.md"
    grouped_docs = [
        {
            "score": 80.0,
            "text": "A useful interferometric microscopy explanation.",
            "meta": {"source_path": r"db\interferometric.md", "heading_path": "Abstract"},
        },
        {
            "score": 90.0,
            "text": (
                "Data from: structured detection for simultaneous super-resolution and optical sectioning. "
                "Zenodo https://doi.org/example. Acknowledgements We thank the sample providers."
            ),
            "meta": {"source_path": structured_source},
        },
    ]
    raw_hits = [
        {
            "score": 120.0,
            "text": "# Structured detection for laser scanning microscopy",
            "meta": {"source_path": structured_source},
        },
        {
            "score": 110.0,
            "text": "[1] A reference-list entry about structured illumination.",
            "meta": {"source_path": structured_source, "heading_path": "References"},
        },
        {
            "score": 65.0,
            "text": (
                "Imaging a thick three-dimensional sample requires optical sectioning to reject out-of-focus light. "
                "Structured detection provides simultaneous super-resolution and optical sectioning."
            ),
            "meta": {"source_path": structured_source, "heading_path": "Introduction", "block_id": "intro-1"},
        },
        {
            "score": 999.0,
            "text": "An unrelated deep-learning single-pixel imaging passage.",
            "meta": {"source_path": r"db\dl-spi.md", "heading_path": "Methods"},
        },
    ]

    out = _rescue_multi_source_answer_hits(
        grouped_docs=grouped_docs,
        raw_hits=raw_hits,
        prompt="What problems do structured detection and interferometric microscopy solve?",
    )

    assert [str((item.get("meta") or {}).get("source_path") or "") for item in out] == [
        r"db\interferometric.md",
        structured_source,
    ]
    assert str((out[1].get("meta") or {}).get("block_id") or "") == "intro-1"
    assert (out[1].get("meta") or {}).get("multi_source_representative_rescue") is True
    assert all("dl-spi" not in str((item.get("meta") or {}).get("source_path") or "") for item in out)


def test_multi_source_rescue_preserves_grouped_structured_table_evidence():
    source = r"db\Simple Baselines for Image Restoration.md"
    benchmark = (
        "Table 6. SIDD PSNR: MPRNet = 39.71; Restormer = 40.02; "
        "Baseline ours = 40.30; NAFNet ours = 40.30"
    )
    ablation = "Table 3. SIDD PSNR: 9 = 39.78; 18 = 39.90; 36 = 39.96; 72 = 39.95"
    grouped_docs = [
        {
            "score": 14.6,
            "text": benchmark,
            "meta": {
                "source_path": source,
                "heading_path": "5 Experiments / 5.2 Applications",
                "structured_kind": "table_metric",
                "table_number": 6,
                "table_metric_label": "SIDD PSNR",
                "table_subject_kind": "method",
            },
        }
    ]
    raw_hits = [
        {
            "score": 31.2,
            "text": benchmark,
            "meta": {
                "source_path": source,
                "heading_path": "5 Experiments / 5.2 Applications",
                "structured_kind": "table_metric",
                "table_number": 6,
            },
        },
        {
            "score": 99.0,
            "text": ablation,
            "meta": {
                "source_path": source,
                "heading_path": "5 Experiments / 5.1 Ablations",
                "structured_kind": "table_metric",
                "table_number": 3,
            },
        },
    ]

    out = _rescue_multi_source_answer_hits(
        grouped_docs=grouped_docs,
        raw_hits=raw_hits,
        prompt="Which model has the highest SIDD PSNR?",
    )

    assert len(out) == 1
    assert out[0]["text"] == benchmark
    assert out[0]["meta"]["table_number"] == 6
    assert ablation not in out[0]["text"]


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
