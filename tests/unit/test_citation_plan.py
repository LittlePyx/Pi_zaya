from __future__ import annotations

from pathlib import Path

from kb.citation_plan import (
    _prompt_aligned_source_slot,
    _rank_system_a_answer_hits,
    build_citation_plan,
    build_citation_plan_prompt_block,
    citation_plan_prefers_system_b,
)


def test_prompt_aligned_abstract_keeps_complete_multi_sentence_claim(tmp_path: Path) -> None:
    prefix = "Background context. " * 35
    target = (
        "A high-resolution foveal region tracks motion, yet unlike a simple zoom, every frame "
        "delivers new spatial information from across the entire field of view. This strategy "
        "records fast-changing features while accumulating slowly evolving detail over several consecutive frames."
    )
    source = tmp_path / "foveated.en.md"
    source.write_text(f"# Paper\n\n## Abstract\n\n{prefix}{target}\n", encoding="utf-8")

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source),
            "evidence_atom_text": "Generic supersampling background.",
        },
        ranking_texts=["动态超采样和普通 zoom 有什么区别？运动区域如何处理？"],
    )

    assert "unlike a simple zoom" in slot["evidence_quote"]
    assert "entire field of view" in slot["evidence_quote"]
    assert "several consecutive frames" in slot["evidence_quote"]
    assert "Background context" not in slot["evidence_quote"]


def test_comparison_source_summary_replaces_front_matter_hit(tmp_path: Path) -> None:
    source = tmp_path / "3d-video.en.md"
    source.write_text(
        "# 3D single-pixel video\n\n"
        "To cite this article: Example et al. 2016.\n\n"
        "## Abstract\n\n"
        "Photometric stereo uses four spatially-separated single-pixel detectors. "
        "The system reconstructs continuous 3D video at 8 frames per second.\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source),
            "heading_path": "3D single-pixel video",
            "evidence_atom_text": "To cite this article: Example et al. 2016.",
        },
        ranking_texts=[
            "3D single-pixel video photometric stereo four detectors 8 frames per second"
        ],
        prefer_source_summary=True,
    )

    assert slot["heading_path"].endswith("Abstract")
    assert "four spatially-separated" in slot["evidence_quote"]
    assert "8 frames per second" in slot["evidence_quote"]


def test_chinese_fdm_query_ranks_exact_english_abstract_before_figure_caption() -> None:
    indexed_hits = [
        (
            1,
            {
                "text": "Figure 4. State characterization of the illumination patterns.",
                "meta": {
                    "source_path": "fdm.en.md",
                    "heading_path": "Experimental results / Figure 4",
                },
            },
        ),
        (
            2,
            {
                "text": (
                    "Frequency-division multiplexed single-pixel imaging parallelizes acquisition "
                    "and offers a trade-off between signal-to-noise ratio and acquisition speed "
                    "without altering detector integration time."
                ),
                "meta": {
                    "source_path": "fdm.en.md",
                    "heading_path": "Abstract",
                },
            },
        ),
    ]

    ranked = _rank_system_a_answer_hits(
        indexed_hits,
        ranking_texts=[
            "频分复用为什么能并行采集，又如何在不改变探测器积分时间时权衡信噪比和采集速度？"
        ],
    )

    assert ranked[0][0] == 2


def test_origin_question_builds_system_b_first_plan():
    plan = build_citation_plan(
        prompt="ADMM 是怎么来的？作者是不是借鉴了前人的方法？",
        prompt_family="overview",
        answer_hits=[
            {
                "text": "SCINeRF uses ADMM in the optimization loop.",
                "meta": {"source_path": "scinerf.en.md", "heading_path": "Related Work"},
            }
        ],
        support_slots=[],
        reference_opportunities=[
            {
                "sid": "s1234abcd",
                "ref_num": 4,
                "label": "ADMM",
                "source_path": "scinerf.en.md",
                "heading_path": "2. Related Work",
                "evidence_quote": "Most existing methods employ ADMM [4].",
            }
        ],
    )

    assert plan["intent"] == "origin_lookup"
    assert plan["budget"] == {"system_a": 1, "system_b": 1}
    assert plan["system_b_enabled"] is True
    assert plan["slots"][0]["preferred_system"] == "system_b"
    assert plan["slots"][0]["candidate_cite_examples"] == ["[[CITE:s1234abcd:4]]"]
    assert plan["slots"][0]["grounding_contract"]["context_marker_verified"] is True
    assert plan["route_policy"]["system_a"] == "retrieved_paper_text_only"
    assert citation_plan_prefers_system_b(plan, context="ADMM comes from prior work.", ref_num=4)

    block = build_citation_plan_prompt_block(plan)
    assert "Citation plan" in block
    assert "preferred_system=system_b" in block
    assert "[[CITE:s1234abcd:4]]" in block


def test_lineage_question_builds_grounded_system_b_plan():
    plan = build_citation_plan(
        prompt="SCI 这条线是怎么从光谱成像走到 3D 场景重建的？",
        prompt_family="overview",
        answer_hits=[
            {
                "text": "Snapshot compressive imaging evolved toward 3D scene reconstruction.",
                "meta": {"source_path": "scinerf.en.md", "heading_path": "Introduction"},
            }
        ],
        reference_opportunities=[
            {
                "sid": "s1234abcd",
                "ref_num": 50,
                "label": "Snapshot compressive imaging",
                "source_path": "scinerf.en.md",
                "heading_path": "Introduction",
                "evidence_quote": "video Snapshot Compressive Imaging (SCI) [50] system has emerged",
            }
        ],
    )

    assert plan["intent"] == "origin_lookup"
    assert plan["budget"] == {"system_a": 1, "system_b": 1}
    assert plan["system_b_enabled"] is True
    assert plan["slots"][0]["candidate_cite_examples"] == ["[[CITE:s1234abcd:50]]"]


def test_three_source_lineage_budgets_each_system_a_source() -> None:
    answer_hits = [
        {
            "text": "CASSI uses a dual-disperser architecture for snapshot spectral imaging.",
            "meta": {"source_path": "cassi.en.md", "heading_path": "Abstract"},
        },
        {
            "text": "SCINeRF extends SCI to neural 3D reconstruction.",
            "meta": {"source_path": "scinerf.en.md", "heading_path": "Abstract"},
        },
        {
            "text": "SCIGS reconstructs dynamic 3D scenes from one compressed image.",
            "meta": {"source_path": "scigs.en.md", "heading_path": "Abstract"},
        },
    ]

    plan = build_citation_plan(
        prompt="Trace the lineage from CASSI through SCINeRF to SCIGS.",
        prompt_family="overview",
        answer_hits=answer_hits,
    )

    assert plan["intent"] == "origin_lookup"
    assert plan["budget"] == {"system_a": 3, "system_b": 0}
    assert plan["system_b_enabled"] is False
    assert {
        slot["source_path"]
        for slot in plan["slots"]
        if slot["preferred_system"] == "system_a"
    } == {"cassi.en.md", "scinerf.en.md", "scigs.en.md"}


def test_support_slots_rank_prompt_aligned_passage_before_title_noise() -> None:
    source_path = "spi-prospects.en.md"
    plan = build_citation_plan(
        prompt="什么场景值得用单像素相机？",
        prompt_family="overview",
        support_slots=[
            {
                "source_path": source_path,
                "heading_path": "Abstract",
                "locate_anchor": (
                    "Modern digital cameras employ silicon focal plane array image sensors "
                    "featuring millions of pixels."
                ),
            },
            {
                "source_path": source_path,
                "heading_path": "Title",
                "locate_anchor": "Principles and prospects for single-pixel imaging",
            },
            {
                "source_path": source_path,
                "heading_path": "Authors",
                "locate_anchor": "Gibson and Miles Padgett",
            },
            {
                "source_path": source_path,
                "heading_path": "Abstract",
                "locate_anchor": (
                    "Images can be collected at wavelengths outside the reach of FPA "
                    "technology or at high frame rates or in three dimensions."
                ),
            },
        ],
        retrieval_queries=[
            (
                "single-pixel imaging applications wavelengths outside FPA technology "
                "high frame rates three dimensions"
            )
        ],
    )

    system_a = [
        slot
        for slot in plan["slots"]
        if slot["preferred_system"] == "system_a"
    ]
    assert "wavelengths outside the reach of FPA" in system_a[0]["evidence_quote"]
    assert "high frame rates" in system_a[0]["evidence_quote"]


def test_support_slot_source_alignment_replaces_stale_evidence_atom(
    tmp_path,
) -> None:
    source_path = tmp_path / "spi-prospects.en.md"
    source_path.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "",
                "# Principles and prospects for single-pixel imaging",
                "",
                "## Abstract",
                "",
                (
                    "Modern digital cameras employ silicon focal plane array (FPA) "
                    "image sensors featuring millions of pixels."
                ),
                "",
                (
                    "Although the focus of this Review has been on single-pixel "
                    "cameras for imaging, the sparsity principle applies to other "
                    "multidimensional sensing problems and spectral applications."
                ),
                "",
                (
                    "As the approach suits a wide variety of detector technologies, "
                    "images can be collected at wavelengths outside the reach of FPA "
                    "technology or at high frame rates or in three dimensions. "
                    "Promising applications include the visualization of hazardous gas "
                    "leaks and 3D situation awareness for autonomous vehicles."
                ),
            ]
        ),
        encoding="utf-8",
    )

    plan = build_citation_plan(
        prompt="什么场景值得用单像素相机？",
        prompt_family="overview",
        support_slots=[
            {
                "source_path": str(source_path),
                "heading_path": "Abstract",
                "evidence_atom_text": (
                    "Modern digital cameras employ silicon focal plane array (FPA) "
                    "image sensors featuring millions of pixels."
                ),
                "evidence_quote": "A stale fallback quote.",
            }
        ],
        retrieval_queries=[
            (
                "applications wavelengths outside FPA technology high frame rates "
                "three dimensions hazardous gas leaks autonomous vehicles"
            )
        ],
    )

    system_a = [
        slot
        for slot in plan["slots"]
        if slot["preferred_system"] == "system_a"
    ]
    assert system_a[0]["heading_path"].endswith("Abstract")
    assert system_a[0]["page_start"] == 1
    assert system_a[0]["page_end"] == 1
    assert "wavelengths outside the reach of FPA technology" in system_a[0]["evidence_quote"]
    assert "hazardous gas leaks" in system_a[0]["evidence_quote"]


def test_evolutionary_method_wording_does_not_trigger_lineage_route() -> None:
    plan = build_citation_plan(
        prompt="Explain the evolutionary optimization method and implementation.",
        prompt_family="overview",
        answer_hits=[
            {
                "text": "The evolutionary optimizer alternates mutation and selection steps.",
                "meta": {"source_path": "optimizer.en.md", "heading_path": "Methods"},
            }
        ],
        reference_opportunities=[
            {
                "sid": "s1234abcd",
                "ref_num": 7,
                "label": "Earlier optimizer",
                "source_path": "optimizer.en.md",
                "heading_path": "Related Work",
            }
        ],
    )

    assert plan["intent"] == "method_explain"
    assert plan["budget"] == {"system_a": 2, "system_b": 0}
    assert plan["system_b_enabled"] is False
    assert all(slot["preferred_system"] == "system_a" for slot in plan["slots"])


def test_bare_chinese_evolution_wording_does_not_trigger_lineage_route() -> None:
    plan = build_citation_plan(
        prompt="请解释演化优化方法的实现步骤。",
        prompt_family="overview",
        answer_hits=[
            {
                "text": "The optimizer applies mutation, selection, and replacement steps.",
                "meta": {"source_path": "optimizer.en.md", "heading_path": "Methods"},
            }
        ],
    )

    assert plan["intent"] == "method_explain"
    assert plan["budget"] == {"system_a": 2, "system_b": 0}
    assert plan["system_b_enabled"] is False


def test_explicit_evolution_history_and_from_to_contexts_keep_lineage_route() -> None:
    prompts = (
        "Summarize the development history of snapshot compressive imaging.",
        "Explain how SCI evolved from spectral imaging to 3D scene reconstruction.",
    )
    for prompt in prompts:
        plan = build_citation_plan(
            prompt=prompt,
            prompt_family="overview",
            answer_hits=[
                {
                    "text": "SCI evolved from spectral acquisition toward neural 3D reconstruction.",
                    "meta": {"source_path": "sci.en.md", "heading_path": "Introduction"},
                }
            ],
            reference_opportunities=[
                {
                    "sid": "s1234abcd",
                    "ref_num": 50,
                    "label": "Snapshot compressive imaging",
                    "source_path": "sci.en.md",
                    "heading_path": "Introduction",
                    "evidence_quote": "Snapshot compressive imaging builds on prior spectral systems [50].",
                }
            ],
        )

        assert plan["intent"] == "origin_lookup"
        assert plan["budget"] == {"system_a": 1, "system_b": 1}
        assert plan["system_b_enabled"] is True


def test_scope_question_with_named_review_budgets_method_and_review() -> None:
    plan = build_citation_plan(
        prompt="How does PILN relate to the deep-learning SPI review and my research line?",
        prompt_family="overview",
        answer_hits=[
            {
                "text": "The part-based image-loop network targets low-rate single-pixel imaging.",
                "meta": {"source_path": "piln.en.md", "heading_path": "Abstract"},
            },
            {
                "text": "This review maps deep-learning methods for single-pixel imaging.",
                "meta": {"source_path": "dl-spi-review.en.md", "heading_path": "Abstract"},
            },
        ],
    )

    assert plan["intent"] == "scope_boundary"
    assert plan["budget"] == {"system_a": 2, "system_b": 0}
    assert sum(slot["preferred_system"] == "system_a" for slot in plan["slots"]) == 2


def test_single_paper_technical_or_method_route_stays_method_plan():
    reference_opportunities = [
        {
            "sid": "s1234abcd",
            "ref_num": 7,
            "label": "Earlier method",
            "source_path": "paper.en.md",
            "heading_path": "Related Work",
            "evidence_quote": "The paper cites an earlier method [7].",
        }
    ]
    for prompt in (
        "这篇论文的技术路线是什么？",
        "这篇论文的方法路线是什么？",
    ):
        plan = build_citation_plan(
            prompt=prompt,
            prompt_family="overview",
            answer_hits=[
                {
                    "text": "The proposed pipeline contains acquisition and reconstruction stages.",
                    "meta": {"source_path": "paper.en.md", "heading_path": "Methods"},
                }
            ],
            reference_opportunities=reference_opportunities,
        )

        assert plan["intent"] == "method_explain"
        assert plan["budget"] == {"system_a": 2, "system_b": 0}
        assert plan["system_a_enabled"] is True
        assert plan["system_b_enabled"] is False
        assert all(slot["preferred_system"] == "system_a" for slot in plan["slots"])


def test_explicit_from_to_development_lineage_stays_origin_lookup():
    for lineage_label in ("发展路线", "沿革", "演化"):
        plan = build_citation_plan(
            prompt=f"请梳理从 CASSI 到 SCINeRF 的{lineage_label}。",
            prompt_family="overview",
            answer_hits=[
                {
                    "text": "CASSI acquisition was later extended to neural 3D reconstruction.",
                    "meta": {"source_path": "scinerf.en.md", "heading_path": "Introduction"},
                }
            ],
            reference_opportunities=[
                {
                    "sid": "s1234abcd",
                    "ref_num": 50,
                    "label": "Snapshot Compressive Imaging",
                    "source_path": "scinerf.en.md",
                    "heading_path": "Introduction",
                    "evidence_quote": "Snapshot Compressive Imaging (SCI) [50] system has emerged.",
                }
            ],
        )

        assert plan["intent"] == "origin_lookup"
        assert plan["budget"] == {"system_a": 1, "system_b": 1}
        assert plan["system_b_enabled"] is True
        assert plan["slots"][0]["preferred_system"] == "system_b"


def test_system_a_prompt_uses_offset_citation_example():
    plan = build_citation_plan(
        prompt="请解释这篇论文的方法。",
        prompt_family="method",
        answer_hits=[
            {
                "text": "Retrieved evidence.",
                "meta": {"source_path": "paper.en.md", "heading_path": "Methods"},
            }
        ],
    )

    block = build_citation_plan_prompt_block(plan)

    assert "retrieved_hit=1" in block
    assert "cite_example=[10001]" in block
    assert " | hit=1" not in block


def test_comparison_prompt_keeps_decisive_late_evidence_and_requests_compact_answer() -> None:
    evidence = (
        "Photometric stereo estimates surface shape from multiple illumination directions. "
        "Performing high-speed structured illumination and sensing reflected light with four "
        "spatially-separated single-pixel detectors reconstructs 3D video at 8 frames per second."
    )
    plan = build_citation_plan(
        prompt="Compare the two acceleration mechanisms.",
        prompt_family="compare",
        answer_hits=[
            {
                "text": evidence,
                "meta": {"source_path": "3d-video.en.md", "heading_path": "Abstract"},
            }
        ],
    )

    block = build_citation_plan_prompt_block(plan)

    assert "four spatially-separated single-pixel detectors" in block
    assert "8 frames per second" in block
    assert "one direct verdict" in block
    assert "Do not add a comparison table" in block

def test_comparison_plan_disables_system_b_budget():
    plan = build_citation_plan(
        prompt="这两篇方法有什么区别？",
        prompt_family="compare",
        answer_hits=[
            {
                "text": "Hadamard and Fourier patterns differ in sampling basis.",
                "meta": {"source_path": "oe2017.en.md", "heading_path": "Comparison"},
            }
        ],
        reference_opportunities=[
            {
                "sid": "s9999abcd",
                "ref_num": 2,
                "label": "Fourier basis",
                "source_path": "oe2017.en.md",
            }
        ],
    )

    assert plan["intent"] == "comparison"
    assert plan["budget"]["system_a"] == 2
    assert plan["per_paragraph_budget"]["system_a"] == 2
    assert plan["budget"]["system_b"] == 0
    assert plan["system_b_enabled"] is False
    assert all(slot["preferred_system"] != "system_b" for slot in plan["slots"])
    assert not citation_plan_prefers_system_b(plan, context="Fourier basis [2].", ref_num=2)


def test_scope_boundary_plan_uses_one_direct_paper_evidence_citation():
    plan = build_citation_plan(
        prompt="这篇 perovskite laser 和我的单像素成像主线关系大吗？值得一起读吗？",
        prompt_family="overview",
        answer_hits=[
            {
                "text": "We demonstrate lasing from a dual-cavity perovskite device.",
                "meta": {"source_path": "perovskite.en.md", "heading_path": "Abstract"},
            }
        ],
        reference_opportunities=[],
    )

    assert plan["intent"] == "scope_boundary"
    assert plan["budget"] == {"system_a": 1, "system_b": 0}


def test_s2ism_tradeoff_plan_prioritizes_direct_abstract_evidence():
    s2ism_path = "NatPhoton-Structured detection in laser scanning microscopy.en.md"
    plan = build_citation_plan(
        prompt="s2ISM 这篇说的 trade-off 是什么？为什么厚样本会麻烦？",
        prompt_family="compare",
        support_slots=[
            {
                "source_path": "iism.en.md",
                "heading_path": "Microscope setup",
                "evidence_quote": "The lateral resolution is measured from the iPSF.",
            },
            {
                "source_path": s2ism_path,
                "heading_path": "Results / Versatility of s2ISM",
                "evidence_quote": "s2ISM can be applied to any LSM equipped with a detector array.",
            },
        ],
        answer_hits=[
            {
                "text": "The lateral resolution is measured from the iPSF.",
                "meta": {"source_path": "iism.en.md", "heading_path": "Microscope setup"},
            },
            {
                "text": (
                    "Current image scanning microscopy approaches do not provide optical sectioning "
                    "and fail with thick samples unless detector size is limited, introducing a "
                    "trade-off between optical sectioning and signal-to-noise ratio. Fast detector "
                    "arrays overcome the trade-off between spatial resolution and signal-to-noise ratio."
                ),
                "meta": {"source_path": s2ism_path, "heading_path": "Abstract"},
            },
        ],
    )

    first = next(slot for slot in plan["slots"] if slot["preferred_system"] == "system_a")
    assert first["source_path"] == s2ism_path
    assert first["candidate_hits"] == [2]
    assert "thick samples" in first["evidence_quote"]
    assert "optical sectioning versus SNR" in first["support_example"]
    assert plan["budget"]["system_a"] == 1
    assert plan["per_paragraph_budget"]["system_a"] == 1
    assert [slot["source_path"] for slot in plan["slots"] if slot["preferred_system"] == "system_a"] == [
        s2ism_path
    ]


def test_s2ism_tradeoff_plan_keeps_iism_when_user_explicitly_compares_methods():
    s2ism_path = "NatPhoton-Structured detection in laser scanning microscopy.en.md"
    iism_path = "LSA-Interferometric image scanning microscopy.en.md"
    plan = build_citation_plan(
        prompt="s2ISM 和 iISM 的 trade-off 有什么区别？请直接对比。",
        prompt_family="compare",
        answer_hits=[
            {
                "text": (
                    "Current image scanning microscopy approaches fail with thick samples unless "
                    "detector size is limited, creating an optical-sectioning versus SNR trade-off."
                ),
                "meta": {"source_path": s2ism_path, "heading_path": "Abstract"},
            },
            {
                "text": (
                    "Interferometric image scanning microscopy combines interferometric detection "
                    "with image scanning microscopy for 120 nm lateral resolution in live cells."
                ),
                "meta": {"source_path": iism_path, "heading_path": "Abstract"},
            },
        ],
    )

    system_a_paths = [
        slot["source_path"]
        for slot in plan["slots"]
        if slot["preferred_system"] == "system_a"
    ]
    assert system_a_paths[0] == s2ism_path
    assert iism_path in system_a_paths
    assert plan["budget"]["system_a"] >= 2


def test_s2ism_tradeoff_plan_recovers_exact_abstract_before_refs_enrichment(
    tmp_path: Path,
):
    s2ism_path = tmp_path / "NatPhoton-Structured detection in laser scanning microscopy.en.md"
    s2ism_path.write_text(
        "# Structured detection for laser scanning microscopy\n\n"
        "## Abstract\n\n"
        "Fast detector arrays overcome the trade-off between spatial resolution and "
        "signal-to-noise ratio. However, current image scanning microscopy approaches "
        "do not provide optical sectioning and fail with thick samples unless the detector "
        "size is limited, introducing a trade-off between optical sectioning and "
        "signal-to-noise ratio.\n\n"
        "## Results\n\nThe method is versatile.\n",
        encoding="utf-8",
    )

    plan = build_citation_plan(
        prompt="s2ISM 的核心 trade-off 是什么？为什么厚样本会麻烦？",
        prompt_family="compare",
        answer_hits=[
            {
                "text": "The method can be applied to any LSM equipped with a detector array.",
                "meta": {
                    "source_path": str(s2ism_path),
                    "heading_path": "Results / Versatility of s2ISM",
                },
            }
        ],
    )

    first = next(slot for slot in plan["slots"] if slot["preferred_system"] == "system_a")
    assert first["source_path"] == str(s2ism_path)
    assert first["heading_path"] == "Abstract"
    assert "spatial resolution" in first["evidence_quote"]
    assert "thick samples" in first["evidence_quote"]
    assert "optical sectioning versus SNR" in first["support_example"]


def test_multi_source_route_budgets_one_system_a_citation_per_planned_source():
    roadmap = build_citation_plan(
        prompt="我刚开始看单像素成像，应该先读哪几篇？每篇主要看什么？",
        prompt_family="overview",
        answer_hits=[
            {
                "text": f"Evidence for paper {idx}",
                "meta": {"source_path": f"paper-{idx}.en.md", "heading_path": "Abstract"},
            }
            for idx in range(1, 4)
        ],
    )
    method_map = build_citation_plan(
        prompt="structured detection、interferometric、light-field 分别解决什么问题？",
        prompt_family="method",
        answer_hits=[
            {
                "text": f"Evidence for method {idx}",
                "meta": {"source_path": f"method-{idx}.en.md", "heading_path": "Abstract"},
            }
            for idx in range(1, 4)
        ],
    )

    assert roadmap["budget"]["system_a"] == 3
    assert method_map["budget"]["system_a"] == 3
    assert roadmap["per_paragraph_budget"]["system_a"] == 2
    assert method_map["per_paragraph_budget"]["system_a"] == 2
    assert sum(slot["preferred_system"] == "system_a" for slot in method_map["slots"]) == 3
    prompt_block = build_citation_plan_prompt_block(method_map)
    assert "per paragraph budget: SystemA=2" in prompt_block
    assert "whole answer coverage target: SystemA=3" in prompt_block
    assert "limits distinct evidence cards, not marker reuse" in prompt_block
    assert "Reuse the same marker after every later substantive sentence" in prompt_block
    assert "leave the detailed body uncited" in prompt_block
    assert "If no planned evidence slot directly supports it, omit it" in prompt_block


def test_hsi_fsi_plan_recovers_direct_comparison_evidence_from_source(tmp_path: Path):
    source_path = tmp_path / "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md"
    source_path.write_text(
        "# Hadamard single-pixel imaging versus Fourier single-pixel imaging\n\n"
        "Ghost imaging uses random patterns and can require many measurements. "
        "Hadamard single-pixel imaging (HSI) and Fourier single-pixel imaging (FSI) "
        "are two representative techniques that use a deterministic model. "
        "HSI uses Hadamard basis patterns for illumination while FSI uses Fourier basis patterns. "
        "In this paper, we theoretically and experimentally compare HSI and FSI in terms of "
        "principles, imaging efficiency, and noise robustness.\n",
        encoding="utf-8",
    )

    plan = build_citation_plan(
        prompt="我刚开始看单像素成像，应该先读哪几篇？每篇主要看什么，尤其 Hadamard 和 Fourier？",
        prompt_family="overview",
        answer_hits=[
            {
                "text": "Ghost imaging was initially considered a quantum effect.",
                "meta": {
                    "source_path": str(source_path),
                    "heading_path": "Comparison of theory",
                },
            },
            {
                "text": "A second roadmap paper provides broader context.",
                "meta": {
                    "source_path": str(tmp_path / "second-roadmap-paper.en.md"),
                    "heading_path": "Abstract",
                },
            },
        ],
    )

    slot = plan["slots"][0]
    assert slot["heading_path"].endswith("Introduction")
    assert "HSI uses Hadamard basis patterns" in slot["evidence_quote"]
    assert "FSI uses Fourier basis patterns" in slot["evidence_quote"]
    assert "noise robustness" in slot["evidence_quote"]


def test_single_paper_comparison_preserves_distinct_planned_evidence_slots(tmp_path: Path):
    source_path = tmp_path / "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md"
    source_path.write_text(
        "HSI uses Hadamard basis patterns while FSI uses Fourier basis patterns. "
        "We compare HSI and FSI experimentally in terms of noise robustness.\n",
        encoding="utf-8",
    )
    measurement_evidence = "At a 10% sampling ratio, HSI and FSI are compared using PSNR and SSIM."
    hardware_evidence = "Binary DMD patterns run faster than grayscale illumination patterns."

    plan = build_citation_plan(
        prompt="Hadamard 和 Fourier 怎么选？",
        prompt_family="compare",
        support_slots=[
            {
                "source_path": str(source_path),
                "heading_path": "3. Comparison of experiment",
                "evidence_quote": measurement_evidence,
                "claim_type": "compare_result",
            },
            {
                "source_path": str(source_path),
                "heading_path": "2. Comparison of theory",
                "evidence_quote": hardware_evidence,
                "claim_type": "compare_result",
            },
        ],
    )

    assert [slot["evidence_quote"] for slot in plan["slots"]] == [measurement_evidence, hardware_evidence]
    assert [slot["heading_path"] for slot in plan["slots"]] == [
        "3. Comparison of experiment",
        "2. Comparison of theory",
    ]


def test_single_paper_benefit_risk_question_preserves_risk_evidence(tmp_path: Path):
    source_path = tmp_path / "LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.en.md"
    source_path.write_text(
        "Deep learning offers exceptional reconstruction quality and fast reconstruction speed.\n",
        encoding="utf-8",
    )
    benefit = "Deep learning offers exceptional reconstruction quality and fast reconstruction speed."
    risk = "Data-driven strategies require prolonged training and have limited generalization across imaging scenes."

    plan = build_citation_plan(
        prompt="深度学习给单像素成像带来的好处和坑分别是什么？",
        prompt_family="strength_limits",
        answer_hits=[
            {"text": benefit, "meta": {"source_path": str(source_path), "heading_path": "Abstract"}},
            {
                "text": risk,
                "meta": {
                    "source_path": str(source_path),
                    "heading_path": "4. Strategy and Advantages",
                },
            },
        ],
    )

    assert any(slot["evidence_quote"] == benefit for slot in plan["slots"])
    assert any(slot["evidence_quote"] == risk for slot in plan["slots"])
    assert any(slot["heading_path"] == "4. Strategy and Advantages" for slot in plan["slots"])


def test_piln_method_question_promotes_exact_abstract_definition(tmp_path: Path):
    source_path = tmp_path / "Optics & Laser Technology-2024-Part-based image-loop network for single-pixel imaging.en.md"
    source_path.write_text(
        "<!-- kb_page: 2 -->\n\n## Abstract\n\n"
        "We proposed a self-supervised image-loop neural network (ILNet) with a part-based model. "
        "The part-based model divides image features into different parts to facilitate finer-grained learning.\n\n"
        "## 2. Method and experiment setup\n\n### 2.1. Methods\n\n"
        "The generated image is used as input for subsequent iterations.\n",
        encoding="utf-8",
    )

    plan = build_citation_plan(
        prompt="ILNet 为什么叫 image-loop？part-based 设计具体解决什么？",
        prompt_family="method",
        answer_hits=[
            {
                "text": "The generated image is used as input for subsequent iterations.",
                "meta": {
                    "source_path": str(source_path),
                    "heading_path": "2. Method and experiment setup / 2.1. Methods",
                },
            }
        ],
    )

    slot = plan["slots"][0]
    assert slot["heading_path"].endswith("Abstract")
    assert "self-supervised image-loop neural network" in slot["evidence_quote"]
    assert "finer-grained learning" in slot["evidence_quote"]
    assert slot["page_start"] == 2


def test_classical_denoising_question_promotes_two_family_taxonomy(tmp_path: Path):
    source_path = tmp_path / "Visual Computing-2019-Brief review of image denoising techniques.en.md"
    source_path.write_text(
        "<!-- kb_page: 2 -->\n\n"
        "Generally, image denoising methods can be roughly classified as: spatial "
        "domain methods, transform domain methods.\n\n"
        "## Classical denoising method\n\n"
        "Spatial domain methods aim to remove noise by calculating the gray value "
        "of each pixel based on the correlation between pixels/image patches in the "
        "original image.\n\n"
        "## Non-data adaptive transform\n\n"
        "Wavelet transform decomposes data into a scale-space representation.\n",
        encoding="utf-8",
    )

    plan = build_citation_plan(
        prompt="Map classical denoising into spatial domain and transform domain methods.",
        prompt_family="method",
        answer_hits=[
            {
                "text": "Wavelet transform decomposes data into a scale-space representation.",
                "meta": {
                    "source_path": str(source_path),
                    "heading_path": "Non-data adaptive transform",
                },
            }
        ],
    )

    slot = plan["slots"][0]
    assert slot["heading_path"].endswith("Classical denoising method")
    assert "spatial domain methods" in slot["evidence_quote"]
    assert "transform domain methods" in slot["evidence_quote"]
    assert "correlation between pixels/image patches" in slot["evidence_quote"]
    assert slot["candidate_hits"] == [1]
    assert slot["page_start"] == 2


def test_piln_classification_reserves_method_and_review_after_generic_hits(tmp_path: Path):
    review = tmp_path / "LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.en.md"
    review.write_text(
        "<!-- kb_page: 8 -->\n\n#### 4.1.2. Model-Driven Strategy\n\n"
        "Model-driven strategy is an unsupervised learning mode. This strategy integrates "
        "the physical process of SPI with neural networks and leverages the discrepancy "
        "between real and estimated measurements to guide network optimization.\n",
        encoding="utf-8",
    )
    piln = tmp_path / "Optics & Laser Technology-2024-Part-based image-loop network for single-pixel imaging.en.md"
    piln.write_text(
        "<!-- kb_page: 2 -->\n\n## Abstract\n\n"
        "We proposed a self-supervised image-loop neural network (ILNet) with a part-based model. "
        "ILNet employs a part-based model that divides image features to facilitate "
        "finer-grained learning.\n",
        encoding="utf-8",
    )
    generic_hits = [
        {
            "text": f"Generic evidence {index}.",
            "meta": {"source_path": str(tmp_path / f"generic-{index}.en.md"), "heading_path": "Abstract"},
        }
        for index in range(1, 4)
    ]

    plan = build_citation_plan(
        prompt="PILN should be placed in which DL-SPI strategy, and when is it suitable?",
        prompt_family="method",
        answer_hits=[
            *generic_hits,
            {
                "text": "The review categorizes deep-learning SPI strategies.",
                "meta": {"source_path": str(review), "heading_path": "Abstract"},
            },
            {
                "text": "The generated image is fed into the next ILNet iteration.",
                "meta": {"source_path": str(piln), "heading_path": "2.1. Methods"},
            },
        ],
    )

    assert plan["budget"]["system_a"] >= 2
    assert plan["slots"][0]["source_path"] == str(piln)
    assert plan["slots"][0]["candidate_hits"] == [5]
    assert "self-supervised image-loop neural network" in plan["slots"][0]["evidence_quote"]
    assert plan["slots"][1]["source_path"] == str(review)
    assert plan["slots"][1]["candidate_hits"] == [4]
    assert "model-driven strategy" in plan["slots"][1]["evidence_quote"].lower()
    assert "physical process of SPI" in plan["slots"][1]["evidence_quote"]


def test_beginner_roadmap_plan_uses_clean_source_evidence_for_each_paper(tmp_path: Path):
    deep_learning = tmp_path / "LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.en.md"
    deep_learning.write_text(
        "## Abstract\n\n"
        "However, the limited image quality and lengthy computational times for iterative reconstruction still hinder its practical application. "
        "Recently, single-pixel imaging based on deep learning has attracted attention due to its exceptional reconstruction quality and fast reconstruction speed.\n",
        encoding="utf-8",
    )
    principles = tmp_path / "NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md"
    principles.write_text(
        "### Acquisition and image reconstruction strategies.\n\n"
        "The original concept of the single-pixel imaging approach, demonstrated by Sen et al., was developed further in conjunction with compressive sensing and reported by Duarte et al. "
        "Their pioneering work laid the foundations for recovering images from a single-pixel camera when the number of measurements is fewer than the total number of unknown pixels, also known as under-sampling.\n",
        encoding="utf-8",
    )

    plan = build_citation_plan(
        prompt="我刚开始看单像素成像，想先建立主线，应该先读哪几篇？每篇主要看什么？",
        prompt_family="overview",
        answer_hits=[
            {
                "text": "![Figure 9](./assets/page_14_fig_2.png) **Figure 9.** Color SPI.",
                "meta": {"source_path": str(deep_learning), "heading_path": "5.5 Color Single-Pixel Imaging"},
            },
            {
                "text": "### Acquisition and image reconstruction strategies.",
                "meta": {"source_path": str(principles), "heading_path": "Abstract"},
            },
        ],
    )

    assert plan["budget"]["system_a"] == 2
    by_source = {slot["source_path"]: slot for slot in plan["slots"]}
    deep_slot = by_source[str(deep_learning)]
    principles_slot = by_source[str(principles)]
    assert deep_slot["heading_path"].endswith("Abstract")
    assert "fast reconstruction speed" in deep_slot["evidence_quote"]
    assert "![" not in deep_slot["evidence_quote"]
    assert principles_slot["heading_path"].endswith("Acquisition and image reconstruction strategies")
    assert "under-sampling" in principles_slot["evidence_quote"]
    assert "###" not in principles_slot["evidence_quote"]


def test_non_origin_reading_questions_do_not_budget_system_b() -> None:
    opportunity = {
        "sid": "s1234abcd",
        "ref_num": 17,
        "label": "single-pixel imaging background",
        "source_path": "spi-review.en.md",
    }
    beginner = build_citation_plan(
        prompt="我刚开始看单像素成像，想先建立主线，应该先读哪几篇？",
        prompt_family="overview",
        answer_hits=[
            {
                "text": "Single-pixel imaging principles and prospects.",
                "meta": {"source_path": "spi-review.en.md", "heading_path": "Abstract"},
            }
        ],
        reference_opportunities=[opportunity],
    )
    method = build_citation_plan(
        prompt="这个系统具体怎么实现重聚焦？",
        prompt_family="method",
        answer_hits=[
            {
                "text": "Digital refocusing uses ray tracing and wave propagation.",
                "meta": {"source_path": "qclfm.en.md", "heading_path": "Concept"},
            }
        ],
        reference_opportunities=[opportunity],
    )

    assert beginner["budget"]["system_b"] == 0
    assert beginner["system_b_enabled"] is False
    assert method["budget"]["system_b"] == 0
    assert method["system_b_enabled"] is False


def test_comparison_with_clickable_source_markers_stays_system_a_only():
    plan = build_citation_plan(
        prompt=(
            "只依据本文比较 HSI 与 FSI 的采样基和重建原理，"
            "每个结论都标出可点击回原文的来源。"
        ),
        prompt_family="compare",
        answer_hits=[
            {
                "text": "HSI uses Hadamard basis patterns while FSI uses Fourier basis patterns.",
                "meta": {"source_path": "oe2017.en.md", "heading_path": "Comparison"},
            }
        ],
        reference_opportunities=[
            {"sid": "s9999abcd", "ref_num": 18, "label": "HSI", "source_path": "oe2017.en.md"}
        ],
    )

    assert plan["intent"] == "comparison"
    assert plan["system_b_enabled"] is False
    assert all(slot["preferred_system"] != "system_b" for slot in plan["slots"])


def test_comparison_rejecting_upstream_recommendations_stays_system_a_only():
    plan = build_citation_plan(
        prompt=(
            "只依据本文，简洁比较 HSI 与 FSI 的采样基和重建原理；"
            "每个结论给可点击原文来源，不要推荐上游文献。"
        ),
        prompt_family="compare",
        answer_hits=[
            {
                "text": "HSI uses an inverse Hadamard transform; FSI uses an inverse Fourier transform.",
                "meta": {"source_path": "oe2017.en.md", "heading_path": "Principle of HSI and FSI"},
            }
        ],
        reference_opportunities=[
            {"sid": "s9999abcd", "ref_num": 7, "label": "Computational", "source_path": "oe2017.en.md"}
        ],
    )

    assert plan["intent"] == "comparison"
    assert plan["budget"]["system_b"] == 0
    assert plan["system_b_enabled"] is False
    assert all(slot["preferred_system"] != "system_b" for slot in plan["slots"])


def test_exact_reference_lookup_keeps_system_b_while_rejecting_extra_recommendations():
    plan = build_citation_plan(
        prompt=(
            "本文参考文献中的 A single-pixel terahertz imaging system based on compressed sensing "
            "是第几条？不要推荐其他上游文献。"
        ),
        prompt_family="citation_lookup",
        answer_hits=[],
        reference_opportunities=[
            {
                "sid": "s9999abcd",
                "ref_num": 9,
                "label": "terahertz imaging",
                "source_path": "oe2017.en.md",
                "heading_path": "Introduction",
                "evidence_quote": "The terahertz implementation follows the compressed-sensing system [9].",
            }
        ],
    )

    assert plan["intent"] == "origin_lookup"
    assert plan["system_b_enabled"] is True
    assert plan["slots"][0]["candidate_refs"] == [9]


def test_origin_intent_without_same_context_marker_disables_system_b() -> None:
    plan = build_citation_plan(
        prompt="Where did ADMM come from?",
        answer_hits=[
            {
                "text": "The current paper uses ADMM in its reconstruction loop.",
                "meta": {"source_path": "paper.en.md", "heading_path": "Method"},
            }
        ],
        reference_opportunities=[
            {
                "sid": "s1234abcd",
                "ref_num": 4,
                "label": "ADMM",
                "source_path": "paper.en.md",
                "heading_path": "Method",
                "evidence_quote": "The current paper uses ADMM in its reconstruction loop.",
            }
        ],
    )

    assert plan["intent"] == "origin_lookup"
    assert plan["budget"] == {"system_a": 1, "system_b": 0}
    assert plan["system_a_enabled"] is True
    assert plan["system_b_enabled"] is False
    assert all(slot["preferred_system"] == "system_a" for slot in plan["slots"])


def test_reference_list_marker_does_not_masquerade_as_system_b_context() -> None:
    plan = build_citation_plan(
        prompt="Who originally introduced this method?",
        answer_hits=[
            {
                "text": "The current paper applies the method.",
                "meta": {"source_path": "paper.en.md", "heading_path": "Method"},
            }
        ],
        reference_opportunities=[
            {
                "sid": "s1234abcd",
                "ref_num": 4,
                "label": "Upstream method",
                "source_path": "paper.en.md",
                "heading_path": "References",
                "evidence_quote": "[4] Author, Upstream method, 2019.",
                "context_marker_verified": True,
            }
        ],
    )

    assert plan["budget"]["system_b"] == 0
    assert plan["system_b_enabled"] is False


def test_empty_retrieval_shell_does_not_enable_system_a() -> None:
    plan = build_citation_plan(
        prompt="Explain the method.",
        answer_hits=[
            {"text": "", "meta": {"source_path": "paper.en.md", "heading_path": "Method"}}
        ],
    )

    assert plan["system_a_enabled"] is False
    assert plan["slots"] == []


def test_multi_paper_source_marker_request_keeps_system_a_slots_for_every_requested_paper():
    hits = [
        {
            "text": f"Evidence for paper {idx}",
            "meta": {
                "source_path": f"paper-{idx}.en.md",
                "heading_path": f"Paper {idx} / Results",
            },
        }
        for idx in range(1, 7)
    ]

    plan = build_citation_plan(
        prompt="请只用最相关的 4 篇论文做阅读路线，并用来源编号标出可点回原文的依据。",
        prompt_family="method",
        answer_hits=hits,
        reference_opportunities=[
            {
                "sid": "s1234abcd",
                "ref_num": 11,
                "label": "upstream work",
                "source_path": "paper-1.en.md",
            }
        ],
    )

    assert plan["intent"] == "method_explain"
    assert plan["budget"]["system_a"] == 4
    system_a_slots = [slot for slot in plan["slots"] if slot["preferred_system"] == "system_a"]
    assert [slot["candidate_hits"] for slot in system_a_slots] == [[1], [2], [3], [4]]


def test_explicit_multi_paper_plan_ranks_translated_query_facets_without_renumbering_hits():
    def hit(source: str, heading: str, text: str) -> dict:
        return {
            "text": text,
            "meta": {"source_path": source, "heading_path": heading},
        }

    hits = [
        hit("3D single-pixel video.en.md", "Custom system design", "single pixel imaging hardware"),
        hit("Part-based image-loop network.en.md", "Keywords", "single pixel imaging deep learning"),
        hit(
            "Principles and prospects for single-pixel imaging.en.md",
            "Acquisition and reconstruction strategies",
            "compressive sensing laid the foundations for single-pixel imaging",
        ),
        hit("single-pixel compressive holography.en.md", "Results", "single pixel imaging"),
        hit(
            "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.en.md",
            "Abstract",
            "deep learning reconstruction quality and speed review",
        ),
        hit(
            "Frequency-division-multiplexed single-pixel imaging with metamaterials.en.md",
            "Principle",
            "frequency division multiplexing metamaterials spatial light modulation",
        ),
    ]

    plan = build_citation_plan(
        prompt="这三篇之间的知识依赖是什么？正文只引用这三篇。",
        answer_hits=hits,
        retrieval_queries=[
            "single-pixel imaging review frequency division multiplexing metamaterials deep learning survey",
            "single-pixel imaging compressive sensing metamaterial spatial light modulation deep learning reconstruction",
        ],
    )

    system_a_slots = [slot for slot in plan["slots"] if slot["preferred_system"] == "system_a"]
    assert plan["budget"]["system_a"] == 3
    assert [slot["candidate_hits"][0] for slot in system_a_slots] == [3, 6, 5]


def test_explicit_two_paper_fixed_set_hard_limits_authoritative_slots():
    hits = [
        {
            "text": f"Evidence for paper {index}",
            "meta": {"source_path": f"paper-{index}.en.md", "heading_path": "Results"},
        }
        for index in range(1, 4)
    ]

    plan = build_citation_plan(
        prompt="正文只引用这两篇。",
        answer_hits=hits,
        retrieval_queries=["compare these two papers"],
    )

    system_a_slots = [slot for slot in plan["slots"] if slot["preferred_system"] == "system_a"]
    assert plan["budget"]["system_a"] == 2
    assert len(system_a_slots) == 2


def test_explicit_acronym_pair_reserves_exact_title_variants_over_generic_review():
    hits = [
        {
            "text": "A calibrated physical noise model supports SPAD reconstruction.",
            "meta": {
                "source_path": "NatCommun-2023-High-resolution single-photon imaging with physics-informed deep learning.en.md",
                "heading_path": "High-resolution single-photon imaging with physics-informed deep learning / Abstract",
            },
        },
        {
            "text": "The image-loop network uses bucket measurements and random speckle patterns.",
            "meta": {
                "source_path": "Optics-2024-Part-based image-loop network for single-pixel imaging.en.md",
                "heading_path": "Part-based image-loop network for single-pixel imaging / Methods",
            },
        },
        {
            "text": "A broad review of model-driven deep-learning reconstruction methods.",
            "meta": {
                "source_path": "LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.en.md",
                "heading_path": "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning / Abstract",
            },
        },
    ]

    plan = build_citation_plan(
        prompt="Compare PIDL and PILN, and cite only these two papers.",
        answer_hits=hits,
        retrieval_queries=[
            "physics-informed deep learning computational single-photon imaging physical prior data generator neural network loss inference",
            "part-based image-loop network single-pixel imaging ILNet physical model untrained neural network inference",
            "physics-informed deep learning training priors comparison",
            "physics-informed neural networks training inference input output",
        ],
    )

    system_a_slots = [slot for slot in plan["slots"] if slot["preferred_system"] == "system_a"]
    assert [slot["candidate_hits"] for slot in system_a_slots] == [[1], [2]]


def test_previous_answer_audit_uses_every_authoritative_source_without_system_b():
    hits = [
        {
            "text": f"Evidence for paper {idx}",
            "meta": {
                "source_path": f"paper-{idx}.en.md",
                "heading_path": f"Paper {idx} / Results",
            },
        }
        for idx in range(1, 5)
    ]

    plan = build_citation_plan(
        prompt="Audit the previous answer and verify that its four titles match their evidence.",
        prompt_family="overview",
        answer_hits=hits,
        reference_opportunities=[
            {
                "sid": "s1234abcd",
                "ref_num": 11,
                "label": "upstream work",
                "source_path": "paper-1.en.md",
            }
        ],
    )

    assert plan["intent"] == "answer_audit"
    assert plan["budget"] == {"system_a": 4, "system_b": 0}
    assert plan["system_b_enabled"] is False
    system_a_slots = [slot for slot in plan["slots"] if slot["preferred_system"] == "system_a"]
    assert [slot["candidate_hits"] for slot in system_a_slots] == [[1], [2], [3], [4]]


def test_answer_hit_slot_preserves_primary_evidence_locator() -> None:
    evidence = "The multi-source physical noise model of SPAD arrays includes crosstalk."
    plan = build_citation_plan(
        prompt="How does physics-informed deep learning help SPAD imaging?",
        answer_hits=[
            {
                "text": "A broader retrieved chunk.",
                "meta": {
                    "source_path": "pidl.en.md",
                    "heading_path": "Introduction",
                    "page_start": 1,
                    "primary_evidence": {
                        "heading_path": "Introduction / Figure 1a",
                        "snippet": evidence,
                        "block_id": "fig-1a",
                        "anchor_id": "caption-a",
                        "anchor_kind": "paragraph",
                        "page_start": 2,
                        "page_end": 2,
                        "selection_reason": "answer_aligned_block",
                        "strict_locate": True,
                    },
                },
            }
        ],
    )

    slot = next(item for item in plan["slots"] if item["preferred_system"] == "system_a")
    assert slot["evidence_quote"] == evidence
    assert slot["heading_path"] == "Introduction / Figure 1a"
    assert slot["block_id"] == "fig-1a"
    assert slot["anchor_id"] == "caption-a"
    assert slot["page_start"] == 2
    assert slot["page_end"] == 2
    assert slot["strict_locate"] is True


def test_answer_hit_slot_keeps_late_numeric_calibration_evidence() -> None:
    prefix = (
        "We first established a real-world physical noise model of SPAD arrays. "
        "The model contains shot noise, fixed-pattern noise, dark count rate, "
        "afterpulsing, crosstalk noise, and deadtime noise. "
    )
    evidence = prefix + ("Parameter discussion. " * 20) + (
        "We collected 2790 images from 90 scenes at 10 bit depths and 3 illumination fluxes."
    )
    assert evidence.index("2790") > 520

    plan = build_citation_plan(
        prompt="How was the SPAD physical noise model calibrated?",
        answer_hits=[
            {
                "text": evidence,
                "meta": {
                    "source_path": "pidl.en.md",
                    "heading_path": "Introduction",
                },
            }
        ],
    )

    slot = next(item for item in plan["slots"] if item["preferred_system"] == "system_a")
    assert "2790 images" in slot["evidence_quote"]
    assert "90 scenes" in slot["evidence_quote"]


def test_implicit_two_sided_comparison_keeps_both_exact_sources() -> None:
    hits = [
        {
            "text": "A broad review summarizes deep-learning reconstruction for hyperspectral imaging.",
            "meta": {"source_path": "generic-hsi-review.en.md", "heading_path": "Prospects"},
        },
        {
            "text": (
                "Frequency-division multiplexing parallelizes several spatial patterns in one "
                "detector integration window, trading signal-to-noise ratio for acquisition speed."
            ),
            "meta": {"source_path": "frequency-division-multiplexed-spi.en.md", "heading_path": "Abstract"},
        },
        {
            "text": "A general single-pixel imaging review lists many emerging applications.",
            "meta": {"source_path": "generic-spi-review.en.md", "heading_path": "Outlook"},
        },
        {
            "text": (
                "The 3D video system uses photometric stereo with four detectors and reconstructs "
                "three-dimensional video at eight frames per second."
            ),
            "meta": {"source_path": "real-time-3d-video-spi.en.md", "heading_path": "Results"},
        },
    ]

    plan = build_citation_plan(
        prompt=(
            "频分复用单像素成像和 3D single-pixel video 都强调速度："
            "它们分别把什么环节并行化，为什么后者需要多个探测器？"
        ),
        answer_hits=hits,
        retrieval_queries=[
            "frequency division multiplexed single-pixel imaging parallel acquisition",
            "real-time three-dimensional video single-pixel imaging photometric stereo four detectors",
        ],
    )

    system_a_slots = [slot for slot in plan["slots"] if slot["preferred_system"] == "system_a"]
    assert plan["intent"] == "comparison"
    assert len(system_a_slots) == 2
    selected = {slot["source_path"] for slot in system_a_slots}
    assert selected == {
        "frequency-division-multiplexed-spi.en.md",
        "real-time-3d-video-spi.en.md",
    }


def test_comparison_answer_hits_align_each_source_summary_and_keep_hit_locator(
    tmp_path: Path,
) -> None:
    fdm = tmp_path / "frequency-division-multiplexed-spi.en.md"
    fdm.write_text(
        "<!-- kb_page: 1 -->\n\n# Frequency-division multiplexed single-pixel imaging\n\n"
        "Publisher front matter and citation instructions.\n\n"
        "<!-- kb_page: 2 -->\n\n## Abstract\n\n"
        "Frequency-division multiplexing parallelizes the single-pixel imaging process "
        "by projecting multiple spatial patterns during one detector integration time. "
        "This trades signal-to-noise ratio for acquisition speed.\n\n"
        "## B. Encoding\n\n"
        "The mask values are encoded in the phase of intensity modulation, requiring "
        "phase-sensitive detection from a lock-in amplifier. Each SLM pixel is modulated "
        "on p frequencies simultaneously according to the mask patterns. The modulated "
        "light is multiplexed into a single-pixel detector. The signal is demodulated by "
        "a number p of LIAs, one for each modulation frequency.\n",
        encoding="utf-8",
    )
    video_3d = tmp_path / "real-time-3d-video-spi.en.md"
    video_3d.write_text(
        "<!-- kb_page: 1 -->\n\n# Real-time 3D video single-pixel imaging\n\n"
        "To cite this article: Example et al. 2016.\n\n"
        "<!-- kb_page: 3 -->\n\n## Abstract\n\n"
        "Photometric stereo uses four spatially separated single-pixel detectors to "
        "recover surface orientation in parallel. The system reconstructs continuous "
        "three-dimensional video at eight frames per second.\n",
        encoding="utf-8",
    )

    plan = build_citation_plan(
        prompt=(
            "频分复用单像素成像和 3D single-pixel video 分别把什么环节并行化，"
            "为什么后者需要 four detectors？"
        ),
        answer_hits=[
            {
                "text": "Publisher front matter and citation instructions.",
                "meta": {
                    "source_path": str(fdm),
                    "heading_path": "Frequency-division multiplexed single-pixel imaging",
                    "page_start": 1,
                },
            },
            {
                "text": "To cite this article: Example et al. 2016.",
                "meta": {
                    "source_path": str(video_3d),
                    "heading_path": "Real-time 3D video single-pixel imaging",
                    "page_start": 1,
                },
            },
        ],
        retrieval_queries=[
            "frequency division multiplexing parallel patterns detector integration time",
            "3D video photometric stereo four single-pixel detectors eight frames per second",
        ],
    )

    by_source = {
        slot["source_path"]: slot
        for slot in plan["slots"]
        if slot["preferred_system"] == "system_a"
    }
    assert by_source[str(fdm)]["candidate_hits"] == [1]
    assert by_source[str(fdm)]["heading_path"].endswith("B. Encoding")
    assert by_source[str(fdm)]["page_start"] == 2
    assert by_source[str(fdm)]["block_id"] == ""
    assert by_source[str(fdm)]["strict_locate"] is False
    assert "lock-in amplifier" in by_source[str(fdm)]["evidence_quote"]
    assert "frequencies simultaneously" in by_source[str(fdm)]["evidence_quote"]
    assert "demodulated" in by_source[str(fdm)]["evidence_quote"]
    assert by_source[str(video_3d)]["candidate_hits"] == [2]
    assert by_source[str(video_3d)]["heading_path"].endswith("Abstract")
    assert by_source[str(video_3d)]["page_start"] == 3
    assert "four spatially separated" in by_source[str(video_3d)]["evidence_quote"]
    assert "eight frames per second" in by_source[str(video_3d)]["evidence_quote"]


def test_basis_vs_foveated_answer_hit_uses_foveated_abstract(tmp_path: Path) -> None:
    basis = tmp_path / "hadamard-versus-fourier.en.md"
    basis.write_text(
        "<!-- kb_page: 2 -->\n\n## Introduction\n\n"
        "Hadamard single-pixel imaging uses binary Hadamard basis patterns, whereas "
        "Fourier single-pixel imaging samples sinusoidal Fourier coefficients.\n",
        encoding="utf-8",
    )
    foveated = tmp_path / "adaptive-foveated-spi.en.md"
    foveated.write_text(
        "<!-- kb_page: 1 -->\n\n## Abstract\n\n"
        "Adaptive foveated single-pixel imaging applies dynamic supersampling to a "
        "foveal region while every frame still delivers new information from the "
        "entire field of view.\n\n"
        "<!-- kb_page: 7 -->\n\n## Spatially variant digital supersampling\n\n"
        "A generic interpolation kernel generates the displayed pixels.\n",
        encoding="utf-8",
    )

    plan = build_citation_plan(
        prompt=(
            "Hadamard/Fourier basis choice 和 adaptive foveated dynamic supersampling "
            "分别决定什么，二者是同一层面的采样策略吗？"
        ),
        answer_hits=[
            {
                "text": (
                    "Hadamard single-pixel imaging uses binary Hadamard basis patterns, "
                    "whereas Fourier single-pixel imaging samples sinusoidal coefficients."
                ),
                "meta": {
                    "source_path": str(basis),
                    "heading_path": "Introduction",
                    "page_start": 2,
                },
            },
            {
                "text": "A generic interpolation kernel generates the displayed pixels.",
                "meta": {
                    "source_path": str(foveated),
                    "heading_path": "Spatially variant digital supersampling",
                    "page_start": 7,
                },
            },
        ],
        retrieval_queries=[
            "Hadamard Fourier basis pattern choice",
            "adaptive foveated dynamic supersampling entire field of view",
        ],
    )

    by_source = {
        slot["source_path"]: slot
        for slot in plan["slots"]
        if slot["preferred_system"] == "system_a"
    }
    assert by_source[str(basis)]["candidate_hits"] == [1]
    assert by_source[str(basis)]["heading_path"].endswith("Introduction")
    assert by_source[str(basis)]["page_start"] == 2
    assert by_source[str(foveated)]["candidate_hits"] == [2]
    assert by_source[str(foveated)]["heading_path"].endswith("Abstract")
    assert by_source[str(foveated)]["page_start"] == 1
    assert "entire field of view" in by_source[str(foveated)]["evidence_quote"]


def test_same_layer_question_is_routed_as_two_sided_comparison() -> None:
    plan = build_citation_plan(
        prompt=(
            "Hadamard/Fourier 的选择和 foveated dynamic supersampling 是同一层面的采样策略吗？"
            "设计系统时，这两类选择分别在决定什么？"
        ),
        answer_hits=[
            {
                "text": "Every frame delivers new information from the entire field of view.",
                "meta": {"source_path": "adaptive-foveated-spi.en.md", "heading_path": "Abstract"},
            },
            {
                "text": "HSI uses Hadamard basis patterns while FSI uses Fourier basis patterns.",
                "meta": {"source_path": "hadamard-versus-fourier.en.md", "heading_path": "Introduction"},
            },
            {
                "text": "A broad review of single-pixel imaging applications.",
                "meta": {"source_path": "generic-prospects.en.md", "heading_path": "Outlook"},
            },
        ],
        retrieval_queries=[
            "Hadamard Fourier basis pattern choice",
            "adaptive foveated dynamic supersampling entire field of view",
        ],
    )

    slots = [slot for slot in plan["slots"] if slot["preferred_system"] == "system_a"]
    assert plan["intent"] == "comparison"
    assert len(slots) == 2
    assert {slot["source_path"] for slot in slots} == {
        "adaptive-foveated-spi.en.md",
        "hadamard-versus-fourier.en.md",
    }


def test_pair_reading_question_does_not_add_a_generic_third_paper() -> None:
    plan = build_citation_plan(
        prompt="单光子成像里，探测器综述和 physics-informed deep learning 这篇应该怎么搭配读？",
        answer_hits=[
            {
                "text": "A detector review explains single-photon detector physics.",
                "meta": {"source_path": "detector-review.en.md", "heading_path": "Abstract"},
            },
            {
                "text": "Physics-informed deep learning models real SPAD noise.",
                "meta": {
                    "source_path": "physics-informed-deep-learning-spad.en.md",
                    "heading_path": "Abstract",
                },
            },
            {
                "text": "A generic deep-learning imaging survey.",
                "meta": {"source_path": "generic-dl-review.en.md", "heading_path": "Abstract"},
            },
        ],
        retrieval_queries=[
            "single photon detector review",
            "physics informed deep learning SPAD physical noise",
        ],
    )

    slots = [slot for slot in plan["slots"] if slot["preferred_system"] == "system_a"]
    assert plan["intent"] == "comparison"
    assert len(slots) == 2
    assert {slot["source_path"] for slot in slots} == {
        "detector-review.en.md",
        "physics-informed-deep-learning-spad.en.md",
    }
