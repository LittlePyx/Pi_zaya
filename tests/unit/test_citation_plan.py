from __future__ import annotations

from pathlib import Path

from kb.citation_plan import (
    build_citation_plan,
    build_citation_plan_prompt_block,
    citation_plan_prefers_system_b,
)


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
    assert plan["budget"] == {"system_a": 3, "system_b": 1}
    assert {
        slot["source_path"]
        for slot in plan["slots"]
        if slot["preferred_system"] == "system_a"
    } == {"cassi.en.md", "scinerf.en.md", "scigs.en.md"}


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
            {"sid": "s9999abcd", "ref_num": 9, "label": "terahertz imaging", "source_path": "oe2017.en.md"}
        ],
    )

    assert plan["intent"] == "origin_lookup"
    assert plan["system_b_enabled"] is True
    assert plan["slots"][0]["candidate_refs"] == [9]


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
