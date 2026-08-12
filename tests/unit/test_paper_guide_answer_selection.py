from kb.paper_guide_answer_selection import (
    _bundle_answer_hits_by_source,
    _build_answer_hits_for_generation,
    _has_anchor_grounded_answer_hits,
    _merge_same_source_answer_hits,
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


def test_select_paper_guide_answer_hits_preserves_update_and_initialization_aspects():
    src = r"db\learned-primal-dual\paper.en.md"
    hits = [
        {
            "score": 44.0,
            "text": (
                "An initial guess marginally decreased training time but did not improve final results. "
                "The pseudo-inverse adds complexity, so we report zero-initialization."
            ),
            "meta": {
                "source_path": src,
                "heading_path": "Learned Primal-Dual / Choice of starting point",
                "block_id": "blk_init",
                "paper_guide_targeted_block": True,
            },
        },
        {
            "score": 10.0,
            "text": (
                "Algorithm 2 replaces the primal proximal with learned proximal Gamma and "
                "the dual proximal with learned proximal Lambda."
            ),
            "meta": {
                "source_path": src,
                "heading_path": "Learned Primal-Dual / Learned PDHG",
                "block_id": "blk_updates",
                "paper_guide_targeted_block": True,
            },
        },
        {
            "score": 38.0,
            "text": "Generic background about inverse problems.",
            "meta": {
                "source_path": src,
                "heading_path": "Variational regularization",
                "block_id": "blk_background",
            },
        },
    ]

    out = _select_paper_guide_answer_hits(
        grouped_docs=[],
        heading_hits=hits,
        prompt=(
            "Learned Primal-Dual 怎样把 PDHG 的对偶更新和原始更新改造成可学习模块？"
            "为什么选择零初始化而不是 FBP？"
        ),
        top_n=2,
    )

    assert [str((item.get("meta") or {}).get("block_id") or "") for item in out] == [
        "blk_updates",
        "blk_init",
    ]


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


def test_multi_source_answer_rescue_replaces_unrequested_author_biography():
    source = r"db\dl-spi-review.md"
    out = _rescue_multi_source_answer_hits(
        grouped_docs=[
            {
                "score": 40.0,
                "text": "Kai Song received his degree and studies single-photon imaging.",
                "meta": {"source_path": source, "heading_path": "Author Biographies"},
            }
        ],
        raw_hits=[
            {
                "score": 22.0,
                "text": (
                    "Iterative reconstruction has limited image quality and lengthy computational times, "
                    "while deep-learning single-pixel imaging improves reconstruction speed."
                ),
                "meta": {"source_path": source, "heading_path": "Abstract", "block_id": "abstract-1"},
            }
        ],
        prompt="physics-informed deep learning 在单光子成像里到底帮了什么？",
    )

    assert len(out) == 1
    assert out[0]["meta"]["block_id"] == "abstract-1"


def test_multi_source_answer_rescue_keeps_requested_author_biography():
    source = r"db\dl-spi-review.md"
    biography = {
        "score": 40.0,
        "text": "Kai Song received his degree and studies single-photon imaging.",
        "meta": {"source_path": source, "heading_path": "Author Biographies"},
    }
    out = _rescue_multi_source_answer_hits(
        grouped_docs=[biography],
        raw_hits=[
            {
                "score": 50.0,
                "text": "This review discusses reconstruction algorithms.",
                "meta": {"source_path": source, "heading_path": "Abstract"},
            }
        ],
        prompt="请概括作者 Kai Song 的学历、当前职位和研究方向。",
    )

    assert len(out) == 1
    assert out[0]["text"] == biography["text"]


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


def test_merge_same_source_answer_hits_preserves_passage_locators_under_one_doc():
    source = r"db\demo\demo.en.md"
    hits = [
        {
            "score": 12.0,
            "text": "The contracting path captures context.",
            "meta": {
                "source_path": source,
                "heading_path": "2 Network Architecture",
                "page_start": 4,
                "block_id": "blk-contract",
            },
        },
        {
            "score": 10.0,
            "text": "The cropped feature map is concatenated with the upsampled output.",
            "meta": {
                "source_path": source,
                "heading_path": "2 Network Architecture",
                "page_start": 4,
                "block_id": "blk-concat",
            },
        },
    ]

    out = _merge_same_source_answer_hits(hits)

    assert len(out) == 1
    assert "Source passage 1" in out[0]["text"]
    assert "Source passage 2" in out[0]["text"]
    assert "p. 4" in out[0]["text"]
    assert out[0]["meta"]["source_passage_count"] == 2
    assert len(out[0]["meta"]["source_passages"]) == 2
    assert [
        passage["score"] for passage in out[0]["meta"]["source_passages"]
    ] == [12.0, 10.0]


def test_merge_same_source_answer_hits_leaves_multiple_papers_separate():
    hits = [
        {"text": "Alpha evidence", "meta": {"source_path": "alpha.md"}},
        {"text": "Beta evidence", "meta": {"source_path": "beta.md"}},
    ]

    assert _merge_same_source_answer_hits(hits) == hits


def test_bundle_answer_hits_by_source_keeps_one_citation_doc_per_paper():
    hits = [
        {"text": "Alpha mechanism", "meta": {"source_path": "alpha.md", "page_start": 1}},
        {"text": "Alpha limitation", "meta": {"source_path": "alpha.md", "page_start": 4}},
        {"text": "Beta mechanism", "meta": {"source_path": "beta.md", "page_start": 2}},
        {"text": "Beta deployment", "meta": {"source_path": "beta.md", "page_start": 5}},
    ]

    out = _bundle_answer_hits_by_source(hits)

    assert len(out) == 2
    assert "Alpha mechanism" in out[0]["text"]
    assert "Alpha limitation" in out[0]["text"]
    assert "Beta mechanism" in out[1]["text"]
    assert "Beta deployment" in out[1]["text"]


def test_named_mechanism_selection_reserves_distinct_source_blocks():
    source = "restormer.en.md"
    hits = [
        {
            "score": 150.0,
            "text": (
                "MDTA applies self-attention across channels. GDFN performs controlled "
                "feature transformation."
            ),
            "meta": {"source_path": source, "heading_path": "Conclusion", "page_start": 8},
        },
        {
            "score": 80.0,
            "text": (
                "MDTA computes cross-covariance across feature channels rather than the "
                "spatial dimension."
            ),
            "meta": {"source_path": source, "heading_path": "Introduction", "page_start": 2},
        },
        {
            "score": 78.0,
            "text": (
                "Overall, the GDFN controls the information flow and lets each level "
                "focus on fine details."
            ),
            "meta": {"source_path": source, "heading_path": "3.2 GDFN", "page_start": 4},
        },
    ]

    selected = _select_paper_guide_answer_hits(
        grouped_docs=[],
        heading_hits=hits,
        prompt=(
            "In Restormer, what does MDTA transpose about self-attention, and what "
            "distinct filtering role does GDFN play?"
        ),
        top_n=2,
    )

    assert [hit["meta"]["page_start"] for hit in selected] == [2, 4]


def test_dataset_identifier_does_not_displace_exact_quantitative_passage() -> None:
    source = "sam.en.md"
    hits = [
        {
            "score": 179.5,
            "text": "SA-1B contains 11M images and 1.1B masks.",
            "meta": {
                "source_path": source,
                "heading_path": "1 Introduction",
                "page_start": 3,
                "block_id": "intro",
                "paper_guide_targeted_block": True,
            },
        },
        {
            "score": 154.0,
            "text": (
                "Our data engine has three stages: assisted-manual, semi-automatic, "
                "and fully automatic."
            ),
            "meta": {
                "source_path": source,
                "heading_path": "4 Segment Anything Data Engine",
                "page_start": 2,
                "block_id": "engine",
                "paper_guide_targeted_block": True,
            },
        },
        {
            "score": 147.0,
            "text": (
                "Our data engine produced 1.1B masks, 99.1% of which were generated "
                "fully automatically."
            ),
            "meta": {
                "source_path": source,
                "heading_path": "5 Segment Anything Dataset / Masks",
                "page_start": 6,
                "block_id": "masks",
                "paper_guide_targeted_block": True,
            },
        },
        {
            "score": 128.0,
            "text": "The SA-1B data card lists caveats and intended use cases.",
            "meta": {
                "source_path": source,
                "heading_path": "F.2 Data Annotation Card",
                "page_start": 28,
                "block_id": "appendix",
                "paper_guide_targeted_block": True,
            },
        },
    ]

    selected = _select_paper_guide_answer_hits(
        grouped_docs=[],
        heading_hits=hits,
        prompt=(
            "SAM 的数据引擎分哪三个阶段？最终 SA-1B 有多少图像和掩码，"
            "其中多少比例是全自动生成的？"
        ),
        top_n=3,
    )

    assert {hit["meta"]["block_id"] for hit in selected} == {
        "intro",
        "engine",
        "masks",
    }


def test_multi_fact_selection_reserves_same_page_quantitative_companion() -> None:
    source = "sam.en.md"
    hits = [
        {
            "score": 180.0,
            "text": "SA-1B contains 11M images and 1.1B masks.",
            "meta": {"source_path": source, "heading_path": "Introduction", "page_start": 3, "block_id": "intro"},
        },
        {
            "score": 160.0,
            "text": "The data engine has assisted-manual, semi-automatic, and fully automatic stages.",
            "meta": {"source_path": source, "heading_path": "Data Engine", "page_start": 2, "block_id": "stages"},
        },
        {
            "score": 150.0,
            "text": "Our data engine produced 1.1B masks, 99.1% fully automatically.",
            "meta": {"source_path": source, "heading_path": "Dataset", "page_start": 6, "block_id": "masks"},
        },
        {
            "score": 140.0,
            "text": "A broad ablation studies model scaling.",
            "meta": {"source_path": source, "heading_path": "Ablations", "page_start": 12, "block_id": "ablation"},
        },
        {
            "score": 125.0,
            "text": "Our dataset consists of 11M images and 1.1B segmentation masks.",
            "meta": {"source_path": source, "heading_path": "Dataset", "page_start": 6, "block_id": "dataset-counts"},
        },
    ]

    selected = _select_paper_guide_answer_hits(
        grouped_docs=[],
        heading_hits=hits,
        prompt=(
            "What are the assisted-manual, semi-automatic, and fully automatic stages, "
            "and how many 11M images, 1.1B masks, and 99.1% automatic masks are in SA-1B?"
        ),
        top_n=4,
    )

    assert "masks" in {hit["meta"]["block_id"] for hit in selected}
    assert "dataset-counts" in {hit["meta"]["block_id"] for hit in selected}
    assert "ablation" not in {hit["meta"]["block_id"] for hit in selected}


def test_section_heading_page_is_preserved_for_next_page_equation() -> None:
    source = "ddpm.en.md"
    selected = _select_paper_guide_answer_hits(
        grouped_docs=[],
        heading_hits=[
            {
                "score": 90.0,
                "text": "3.4 Simplified training objective",
                "meta": {
                    "source_path": source,
                    "heading_path": "3.4 Simplified training objective",
                    "page_start": 4,
                    "kind": "heading",
                    "block_id": "heading",
                },
            },
            {
                "score": 130.0,
                "text": r"L_simple predicts epsilon noise.",
                "meta": {
                    "source_path": source,
                    "heading_path": "3.4 Simplified training objective",
                    "page_start": 5,
                    "kind": "equation",
                    "block_id": "formula",
                },
            },
        ],
        prompt="What does the simplified training objective L_simple predict?",
        top_n=1,
    )

    assert selected[0]["meta"]["block_id"] == "formula"
    assert selected[0]["meta"]["section_page_start"] == 4
    assert selected[0]["meta"]["section_heading_text"] == "3.4 Simplified training objective"
