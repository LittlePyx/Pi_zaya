from __future__ import annotations

import re

from ui.refs_renderer import (
    _annotate_inpaper_citations_with_hover_meta,
    _assess_system_a_hit_binding,
    _compact_metric_table_evidence,
    _compact_metric_table_matches_claim,
    _compound_plan_evidence_excerpt,
    _system_a_is_low_value_evidence_text,
    _system_a_pick_best_evidence_candidate,
    _system_a_ui_relevance_for_occurrence,
)


def test_quantitative_parenthetical_example_is_not_mistaken_for_another_paper() -> None:
    evidence = (
        "Working parameter (wavelength, time jitter/tj) = 400-1000 nm; "
        "Performance = 50%-92% QE at 200-300 K."
    )

    binding = _assess_system_a_hit_binding(
        answer_claim=(
            "The detector review compares mainstream devices "
            "(for example, Si-SPAD at 400-1000 nm, QE 50%-92%, 200-300 K)."
        ),
        hit={"text": evidence},
        meta={"citation_plan_evidence_authoritative": True,
              "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence"},
        heading="Performance information of different single-photon detectors",
        evidence_quote=evidence,
        source_name="Emerging single-photon detection technique for high-performance photodetector.pdf",
    )

    assert binding["suppress_link"] is False
    assert binding["status"] == "grounded"


def test_compact_metric_table_must_match_the_bound_claim() -> None:
    compact = "Table 2 shows PSNR and SSIM results for Hadamard and Fourier sampling."

    assert not _compact_metric_table_matches_claim(
        compact,
        "The foveated method moves the high-resolution region and retains new information across the field of view.",
    )
    assert _compact_metric_table_matches_claim(
        compact,
        "At a 1% sampling ratio, Hadamard and Fourier have different PSNR and SSIM values.",
    )


def test_system_a_hides_marker_when_card_evidence_misses_claim_value() -> None:
    source_path = "db/hsi/hsi.en.md"
    evidence = "At 1% sampling, HSI reaches PSNR 30.2 dB."
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "At 1% sampling, HSI reaches PSNR 30.2 dB and SSIM 0.91 [1].",
        [
            {
                "text": evidence,
                "meta": {
                    "source_path": source_path,
                    "heading_path": "Results",
                    "citation_plan_evidence_authoritative": True,
                    "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
                },
            }
        ],
        canonical_paths=[source_path],
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "heading_path": "Results",
                    "evidence_quote": evidence,
                    "evidence_selection_reason": "prompt_aligned_source_sentence",
                }
            ],
        },
        anchor_ns="missing-value",
    )

    assert "[1]" not in rendered
    assert "[]" not in rendered
    assert details == []


def test_system_a_keeps_cross_language_multiplier_citation() -> None:
    source_path = "db/iism/iism.en.md"
    evidence = (
        "This technique achieves about 120 nm lateral resolution while operating at "
        "tenfold lower incident illumination power, significantly reducing photodamage."
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "在约 120 nm 横向分辨率下，入射照明功率可降低约 10 倍，从而显著减少光损伤 [1]。",
        [
            {
                "text": evidence,
                "meta": {
                    "source_path": source_path,
                    "heading_path": "Abstract",
                    "ref_answer_citation_num": 1,
                },
            }
        ],
        canonical_paths=[source_path],
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "heading_path": "Abstract",
                    "evidence_quote": evidence,
                    "evidence_selection_reason": "prompt_aligned_source_sentence",
                }
            ],
        },
        anchor_ns="cross-language-multiplier",
        render_locale="zh",
    )

    assert "](#kb-cite-" in rendered
    assert len(details) == 1
    assert details[0]["citation_route"] == "system_a"
    assert "tenfold lower" in details[0]["evidence_quote"]


def test_system_a_keeps_table_locator_when_metric_value_is_grounded() -> None:
    source_path = "db/simple-baselines/simple-baselines.en.md"
    evidence = "SIDD PSNR: Baseline = 40.30 dB; NAFNet = 40.30 dB."
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "在表 6 中，Baseline 与 NAFNet 的 SIDD PSNR 都达到 40.30 dB [1]。",
        [
            {
                "text": evidence,
                "meta": {
                    "source_path": source_path,
                    "heading_path": "Table 6 / SIDD",
                    "citation_plan_evidence_authoritative": True,
                    "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
                },
            }
        ],
        canonical_paths=[source_path],
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "heading_path": "Table 6 / SIDD",
                    "evidence_quote": evidence,
                    "evidence_selection_reason": "prompt_aligned_source_sentence",
                }
            ],
        },
        anchor_ns="table-locator",
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    assert details[0]["binding_status"] == "grounded"
    assert "40.30" in details[0]["card_evidence"]


def test_system_a_table_anchor_restores_named_occurrence_in_locator() -> None:
    source_path = "db/simple-baselines/simple-baselines.en.md"
    compact = "SIDD PSNR: Baseline ours = 40.30; NAFNet ours = 40.30."
    located = "Table 6. Image Denoising Results on SIDD. " + compact
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "SIDD PSNR 最高值为 40.30，Baseline ours 与 NAFNet ours 并列 [1]。",
        [
            {
                "text": compact,
                "meta": {
                    "source_path": source_path,
                    "heading_path": "5 Experiments / 5.2 Applications",
                    "ref_answer_citation_num": 1,
                },
                "ui_meta": {
                    "primary_evidence": {
                        "heading_path": "5 Experiments / 5.2 Applications",
                        "snippet": compact,
                        "block_id": "blk_table_6",
                        "anchor_id": "tb_00006",
                        "anchor_kind": "sentence",
                        "page_start": 13,
                        "page_end": 13,
                    }
                },
            }
        ],
        canonical_paths=[source_path],
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "heading_path": "5 Experiments / 5.2 Applications",
                    "evidence_quote": located,
                    "block_id": "blk_table_6",
                    "anchor_id": "tb_00006",
                    "anchor_kind": "sentence",
                    "page_start": 13,
                    "page_end": 13,
                    "evidence_selection_reason": "prompt_aligned_source_sentence",
                }
            ],
        },
        anchor_ns="named-table-locator",
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    assert details[0]["anchor_kind"] == "table"
    assert "Table 6" in details[0]["location_label"]
    assert "Table 6" in details[0]["card_locator"]


def test_system_a_hides_marker_when_card_names_a_different_method() -> None:
    source_path = "db/scinerf/scinerf.en.md"
    evidence = (
        "SCINeRF recovers a 3D scene from one compressed image by incorporating "
        "the physical SCI process into NeRF training."
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "SCIGS reconstructs an explicit dynamic 3D scene from one compressed image [1].",
        [
            {
                "text": evidence,
                "meta": {
                    "source_path": source_path,
                    "source_name": "SCINeRF.pdf",
                    "heading_path": "Abstract",
                    "citation_plan_evidence_authoritative": True,
                    "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
                },
            }
        ],
        canonical_paths=[source_path],
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "source_name": "SCINeRF.pdf",
                    "heading_path": "Abstract",
                    "evidence_quote": evidence,
                    "evidence_selection_reason": "prompt_aligned_source_sentence",
                }
            ],
        },
        anchor_ns="wrong-method",
    )

    assert "[1]" not in rendered
    assert "[]" not in rendered
    assert details == []


def test_multi_source_numeric_comparison_renders_only_with_complete_union() -> None:
    paths = ["db/scigs/scigs.en.md", "db/scinerf/scinerf.en.md"]
    evidence = [
        "SCIGS obtains 30.2 dB on the benchmark.",
        "SCINeRF obtains 31.5 dB on the benchmark.",
    ]
    hits = [
        {
            "text": evidence[index],
            "meta": {
                "source_path": paths[index],
                "source_name": f"method-{index + 1}.pdf",
                "heading_path": "Results",
                "citation_plan_evidence_authoritative": True,
                "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
            },
        }
        for index in range(2)
    ]
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [index + 1],
                "source_path": paths[index],
                "heading_path": "Results",
                "evidence_quote": evidence[index],
                "evidence_selection_reason": "prompt_aligned_source_sentence",
            }
            for index in range(2)
        ],
    }

    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "SCIGS obtains 30.2 dB [1], while SCINeRF obtains 31.5 dB [2].",
        hits,
        canonical_paths=paths,
        citation_plan=plan,
        anchor_ns="multi-source-comparison",
    )

    assert "[1](#kb-cite-" in rendered
    assert "[2](#kb-cite-" in rendered
    assert len(details) == 2
    assert {detail["card_evidence"] for detail in details} == set(evidence)


def test_multi_source_union_resolves_stale_plan_numbers_by_visible_source() -> None:
    beta_path = "kb-source/0/beta/beta.en.md"
    other_path = "kb-source/0/other/other.en.md"
    alpha_path = "kb-source/0/alpha/alpha.en.md"
    alpha_evidence = "Alpha obtains 30.2 dB on the benchmark."
    beta_evidence = "Beta obtains 31.5 dB on the benchmark."
    hits = [
        {
            "text": beta_evidence,
            "meta": {
                "source_path": beta_path,
                "source_name": "Beta.pdf",
                "heading_path": "Results",
                "ref_answer_citation_num": 1,
                "citation_plan_evidence_authoritative": True,
            },
        },
        {
            "text": "Other reports an unrelated result.",
            "meta": {
                "source_path": other_path,
                "source_name": "Other.pdf",
                "heading_path": "Results",
                "ref_answer_citation_num": 2,
            },
        },
        {
            "text": alpha_evidence,
            "meta": {
                "source_path": alpha_path,
                "source_name": "Alpha.pdf",
                "heading_path": "Results",
                "ref_answer_citation_num": 3,
                "citation_plan_evidence_authoritative": True,
            },
        },
    ]
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                # These candidates were assigned before visible-hit reranking.
                "candidate_hits": [1],
                "source_path": "F:/db/alpha/alpha.en.md",
                "source_name": "Alpha.pdf",
                "heading_path": "Results",
                "evidence_quote": alpha_evidence,
            },
            {
                "preferred_system": "system_a",
                "candidate_hits": [2],
                "source_path": "F:/db/beta/beta.en.md",
                "source_name": "Beta.pdf",
                "heading_path": "Results",
                "evidence_quote": beta_evidence,
            },
        ],
    }

    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "Alpha obtains 30.2 dB [3], while Beta obtains 31.5 dB [1].",
        hits,
        canonical_paths=[beta_path, other_path, alpha_path],
        citation_plan=plan,
        anchor_ns="stale-plan-union",
    )

    assert "[3](#kb-cite-" in rendered
    assert "[1](#kb-cite-" in rendered
    assert len(details) == 2
    assert {str(detail["source_name"]).casefold() for detail in details} == {
        "alpha.pdf",
        "beta.pdf",
    }
    assert {detail["card_evidence"] for detail in details} == {
        alpha_evidence,
        beta_evidence,
    }


def test_multi_source_numeric_comparison_hides_cards_when_union_is_incomplete() -> None:
    paths = ["db/scigs/scigs.en.md", "db/scinerf/scinerf.en.md"]
    evidence = [
        "SCIGS obtains 30.2 dB on the benchmark.",
        "SCINeRF is a related reconstruction method.",
    ]
    hits = [
        {
            "text": evidence[index],
            "meta": {
                "source_path": paths[index],
                "heading_path": "Results",
                "citation_plan_evidence_authoritative": True,
                "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
            },
        }
        for index in range(2)
    ]
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [index + 1],
                "source_path": paths[index],
                "heading_path": "Results",
                "evidence_quote": evidence[index],
                "evidence_selection_reason": "prompt_aligned_source_sentence",
            }
            for index in range(2)
        ],
    }

    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "SCIGS obtains 30.2 dB [1], while SCINeRF obtains 31.5 dB [2].",
        hits,
        canonical_paths=paths,
        citation_plan=plan,
        anchor_ns="incomplete-multi-source-comparison",
    )

    assert "#kb-cite-" not in rendered
    assert "[]" not in rendered
    assert details == []


def test_anaphoric_continuation_keeps_the_previous_grounded_link() -> None:
    source_path = "db/pidl/pidl.en.md"
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "The study collected 2790 real-shot SPAD images to calibrate its physical noise model [1]. "
            "This makes the model applicable across varied acquisition conditions [1]."
        ),
        [
            {
                "text": (
                    "We collected a real-shot SPAD dataset containing 2790 images "
                    "to calibrate the physical noise model under varied acquisition conditions."
                ),
                "meta": {
                    "source_path": source_path,
                    "heading_path": "Introduction",
                },
            }
        ],
        canonical_paths=[source_path],
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "heading_path": "Introduction",
                    "evidence_quote": (
                        "We collected a real-shot SPAD dataset containing 2790 images "
                        "to calibrate the physical noise model under varied acquisition conditions."
                    ),
                }
            ],
        },
        anchor_ns="anaphoric",
    )

    assert rendered.count("[1](#kb-cite-") == 2
    assert len(details) == 1


def test_chinese_anaphoric_continuation_reuses_compound_foveated_evidence() -> None:
    source_path = "db/foveated/foveated.en.md"
    evidence = (
        "A high-resolution foveal region tracks motion while every frame delivers new "
        "information across the entire field of view. This strategy records fast-changing "
        "detail and accumulates slowly evolving detail over consecutive frames."
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "系统用高分辨率中央凹区域追踪运动，并保留整个视场的新信息 [1]。"
            "其核心思想是对快速变化区域即时记录，对慢变区域跨连续多帧累积细节 [1]。"
        ),
        [{"text": evidence, "meta": {"source_path": source_path, "heading_path": "Abstract"}}],
        canonical_paths=[source_path],
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "heading_path": "Abstract",
                    "evidence_quote": evidence,
                    "evidence_selection_reason": "prompt_aligned_source_sentence",
                }
            ],
        },
        anchor_ns="foveated-anaphoric",
    )

    assert rendered.count("[1](#kb-cite-") == 2
    assert len(details) == 1
    assert len(details[0]["answer_claims"]) == 2


def test_chinese_foveated_mechanism_keeps_cross_language_source_link() -> None:
    claim = (
        "\u8bbe\u8ba1\u7cfb\u7edf\u65f6\uff0c\u8fd9\u4e00\u9009\u62e9\u51b3\u5b9a\u4e86\u5982\u4f55\u5229\u7528\u573a\u666f\u7684\u65f6\u7a7a\u5197\u4f59\uff1a"
        "\u5feb\u901f\u53d8\u5316\u7684\u7279\u5f81\u88ab\u5feb\u901f\u8bb0\u5f55\uff0c\u800c\u7f13\u6162\u6f14\u53d8\u7684\u533a\u57df\u5219\u901a\u8fc7\u591a\u5e27\u7d2f\u79ef\u7ec6\u8282\u3002"
    )
    evidence = (
        "This strategy rapidly records the detail of quickly changing features while "
        "simultaneously accumulating detail of more slowly evolving regions over several "
        "consecutive frames."
    )

    binding = _assess_system_a_hit_binding(
        answer_claim=claim,
        hit={"text": evidence},
        meta={},
        heading="Abstract",
        evidence_quote=evidence,
        source_name="foveated-single-pixel-imaging.pdf",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False
    assert {"rapidly", "changing", "accumulating", "slowly"} <= set(
        binding["overlap_terms"]
    )


def test_chinese_efficiency_and_noise_claim_keeps_exact_english_source_link() -> None:
    claim = "\u4e24\u79cd\u65b9\u6cd5\u5728\u6210\u50cf\u6548\u7387\u548c\u566a\u58f0\u9c81\u68d2\u6027\u65b9\u9762\u5b58\u5728\u5dee\u5f02\u3002"
    evidence = (
        "We compare HSI and FSI in terms of imaging efficiency and noise robustness."
    )

    binding = _assess_system_a_hit_binding(
        answer_claim=claim,
        hit={"text": evidence},
        meta={},
        heading="Introduction",
        evidence_quote=evidence,
        source_name="Hadamard versus Fourier single-pixel imaging.pdf",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False


def test_chinese_detector_challenges_keep_exact_english_source_link() -> None:
    claim = (
        "\u63a2\u6d4b\u5668\u7684\u5171\u540c\u6311\u6218\u662f\u5236\u9020\u590d\u6742\u3001\u6210\u672c\u9ad8\uff0c"
        "\u4e14\u9700\u8981\u4f4e\u6e29\u7b49\u7279\u6b8a\u5de5\u4f5c\u6761\u4ef6\u3002"
    )
    evidence = (
        "The complexity and high manufacturing cost, coupled with the requirement of "
        "special conditions like a low-temperature environment, pose significant challenges."
    )

    binding = _assess_system_a_hit_binding(
        answer_claim=claim,
        hit={"text": evidence},
        meta={},
        heading="Abstract",
        evidence_quote=evidence,
        source_name="single-photon detector review.pdf",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False


def test_chinese_model_based_continuation_reuses_physics_model_evidence() -> None:
    source_path = "db/pidl/pidl.en.md"
    evidence = (
        "The physical noise model was calibrated with real-shot SPAD images. Based on the calibrated model, "
        "the study synthesizes a large-scale dataset and enables super-resolution, bit-depth enhancement, "
        "and image quality improvement."
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "论文用真实 SPAD 数据标定物理噪声模型 [1]。"
            "基于该模型，论文合成大规模数据集并实现超分辨率和位深增强 [1]。"
        ),
        [{"text": evidence, "meta": {"source_path": source_path, "heading_path": "Abstract"}}],
        canonical_paths=[source_path],
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "heading_path": "Abstract",
                    "evidence_quote": evidence,
                    "evidence_selection_reason": "prompt_aligned_source_sentence",
                }
            ],
        },
        anchor_ns="model-anaphoric",
    )

    assert rendered.count("[1](#kb-cite-") == 2
    assert len(details) == 1
    assert len(details[0]["answer_claims"]) == 2


def test_system_a_citation_detail_carries_reader_card_fields() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "ADMM prior optimization machinery is explained in the retrieved paper [1].",
        [
            {
                "text": "Most existing methods employ alternating direction method of multipliers (ADMM) [4].",
                "meta": {
                    "source_path": "db/demo/paper.en.md",
                    "heading_path": "2. Related Work",
                    "evidence_quote": "Most existing methods employ alternating direction method of multipliers (ADMM) [4].",
                    "primary_block_id": "blk_001",
                    "primary_anchor_id": "p_001",
                    "anchor_kind": "sentence",
                    "page_start": 2,
                    "page_end": 3,
                    "ref_rank": {"display_score": 8.75, "why": "Related Work names ADMM as prior optimization machinery."},
                },
                "ui_meta": {
                    "citation_meta": {
                        "title": "A grounded ADMM paper",
                        "authors": "Jane Doe, John Smith",
                        "venue": "Optics Express",
                        "year": "2024",
                        "doi": "10.1364/OE.123456",
                        "citation_count": 42,
                        "citation_source": "OpenAlex",
                        "journal_if": 3.3,
                        "journal_quartile": "Q2",
                    }
                },
            }
        ],
        anchor_ns="test",
    )

    assert "#kb-cite-" in rendered
    assert len(details) == 1
    detail = details[0]
    assert detail["is_inpaper"] is False
    assert detail["heading_path"] == "2. Related Work"
    assert "ADMM prior optimization" in detail["answer_claim"]
    assert detail["evidence_quote"].startswith("Most existing methods employ")
    assert detail["evidence_source"] == "retrieval_hit"
    assert "2. Related Work" in detail["location_label"]
    assert "pp. 2-3" in detail["location_label"]
    assert "Related Work names ADMM" in detail["support_relation"]
    assert detail["summary_source"] == "retrieval_hit"
    assert detail["block_id"] == "blk_001"
    assert detail["anchor_id"] == "p_001"
    assert detail["anchor_kind"] == "sentence"
    assert detail["page_start"] == 2
    assert detail["page_end"] == 3
    assert detail["score"] == 8.75
    assert "ADMM" in detail["why_line"]
    assert detail["card_kind"] == "answer_evidence"
    assert detail["card_title"] == "paper.pdf"
    assert detail["card_subtitle"].startswith("2. Related Work")
    assert detail["card_locator"].startswith("2. Related Work")
    assert detail["card_evidence"].startswith("Most existing methods employ")
    assert detail["authors"] == "Jane Doe, John Smith"
    assert detail["venue"] == "Optics Express"
    assert detail["year"] == "2024"
    assert detail["doi"] == "10.1364/OE.123456"
    assert detail["citation_count"] == 42
    assert detail["journal_if"] == 3.3
    assert detail["journal_quartile"] == "Q2"
    assert detail["bibliographic_title"] == "A grounded ADMM paper"
    assert detail["card_quality_label"] in {"候选依据", "证据匹配"}


def test_system_a_uses_richest_metadata_from_duplicate_source_hits() -> None:
    source_path = "db/demo/paper.en.md"
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "The paper compares Hadamard and Fourier sampling [1].",
        [
            {
                "text": "The paper compares Hadamard and Fourier sampling.",
                "meta": {
                    "source_path": source_path,
                    "heading_path": "3. Comparison",
                    "citation_plan_slot": True,
                },
                "ui_meta": {"summary_line": "The paper compares Hadamard and Fourier sampling."},
            },
            {
                "text": "Hadamard sampling is more robust under the tested noise levels.",
                "meta": {"source_path": source_path, "heading_path": "3.1 Simulation"},
                "ui_meta": {
                    "citation_meta": {
                        "doi": "10.1364/OE.123456",
                        "citation_count": 42,
                        "journal_if": 3.3,
                        "journal_quartile": "Q2",
                    }
                },
            },
        ],
        anchor_ns="test-rich-meta",
        canonical_paths=[source_path],
    )

    assert "#kb-cite-" in rendered
    assert len(details) == 1
    assert details[0]["citation_plan_slot"] is True
    assert details[0]["doi"] == "10.1364/OE.123456"
    assert details[0]["citation_count"] == 42
    assert details[0]["journal_if"] == 3.3
    assert details[0]["journal_quartile"] == "Q2"


def test_authoritative_detector_table_slot_keeps_the_planned_record() -> None:
    source_path = "db/detector-review/detector-review.en.md"
    record = (
        "Table 1. Detector performance. Detector type: InGaAs/InAlAs-SPAD. "
        "Working parameter (wavelength = 1310 nm); Performance = 61.2% DE at "
        "200 K; Year = 2022; Ref. = [82]"
    )
    unrelated = (
        "A perovskite detector reaches 88% efficiency for 18 keV X-ray detection."
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "InGaAs/InAlAs-SPAD 在 1310 nm、200 K 下达到 61.2% 探测效率 [1]。",
        [
            {
                "text": record,
                "meta": {
                    "source_path": source_path,
                    "heading_path": "2.3 Superconducting",
                    "citation_plan_slot": True,
                    "citation_plan_evidence_authoritative": True,
                },
                "ui_meta": {
                    "primary_evidence": {
                        "heading_path": "2.3 Superconducting",
                        "snippet": record,
                        "highlight_snippet": record,
                        "selection_reason": "citation_plan_slot",
                    },
                    "reader_open": {
                        "evidenceAlternatives": [
                            {
                                "headingPath": "2.4 Perovskite",
                                "snippet": unrelated,
                            }
                        ]
                    },
                },
            }
        ],
        anchor_ns="detector-table",
        canonical_paths=[source_path],
    )

    assert "#kb-cite-" in rendered
    assert len(details) == 1
    assert details[0]["citation_plan_slot"] is True
    assert details[0]["evidence_quote"].startswith(
        "Detector type: InGaAs/InAlAs-SPAD"
    )
    assert "61.2% DE at 200 K" in details[0]["evidence_quote"]
    assert "perovskite" not in details[0]["evidence_quote"].lower()


def test_authoritative_exact_support_slot_keeps_verified_page_and_passage() -> None:
    source_path = "db/pidl/pidl.en.md"
    exact_evidence = (
        "The multi-source physical noise model of SPAD arrays includes shot noise, "
        "dark count rate, afterpulsing and crosstalk noise."
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "Poisson noise alone is insufficient because SPAD has crosstalk and dark count rate [1].",
        [
            {
                "text": exact_evidence,
                "meta": {
                    "source_path": source_path,
                    "heading_path": "Introduction / Figure 1a",
                    "page_start": 2,
                    "page_end": 2,
                    "citation_plan_slot": True,
                    "citation_plan_evidence_authoritative": True,
                    "citation_plan_source": "exact_support_preflight",
                    "citation_plan_evidence_selection_reason": "spad_noise_model_exact_source",
                },
                "ui_meta": {
                    "primary_evidence": {
                        "heading_path": "Introduction / Figure 1a",
                        "snippet": exact_evidence,
                        "highlight_snippet": exact_evidence,
                        "page_start": 2,
                        "page_end": 2,
                        "selection_reason": "spad_noise_model_exact_source",
                        "strict_locate": True,
                    },
                    "reader_open": {
                        "evidenceAlternatives": [
                            {
                                "headingPath": "Abstract",
                                "snippet": "A broad abstract sentence about SPAD imaging.",
                                "pageStart": 1,
                                "pageEnd": 1,
                            }
                        ]
                    },
                },
            }
        ],
        anchor_ns="exact-spad",
        canonical_paths=[source_path],
    )

    assert "#kb-cite-" in rendered
    assert len(details) == 1
    assert details[0]["heading_path"] == "Introduction / Figure 1a"
    assert details[0]["page_start"] == 2
    assert details[0]["evidence_quote"] == exact_evidence
    assert details[0]["routing_reason"] == "exact_support_preflight"
    assert details[0]["evidence_source"] == "exact_support_preflight"
    assert details[0]["selection_reason"] == "spad_noise_model_exact_source"
    assert details[0]["strict_locate"] is True


def test_authoritative_exact_support_card_keeps_long_multi_claim_passage() -> None:
    source_path = "db/pidl/pidl.en.md"
    exact_evidence = (
        "The underlying limitation originates from the employed single-source Poisson noise model, "
        "which deviates from complex real SPAD noise containing crosstalk and dark count rate. "
        "The multi-source physical noise model of SPAD arrays consists of shot noise, fixed-pattern "
        "noise, dark count rate, afterpulsing and crosstalk noise, and deadtime noise."
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "Poisson noise alone is insufficient; the model includes realistic SPAD noise [1].",
        [
            {
                "text": exact_evidence,
                "meta": {
                    "source_path": source_path,
                    "heading_path": "Introduction / Figure 1a",
                    "page_start": 2,
                    "page_end": 2,
                    "citation_plan_slot": True,
                    "citation_plan_evidence_authoritative": True,
                    "citation_plan_source": "exact_support_preflight",
                    "citation_plan_evidence_selection_reason": "spad_noise_model_exact_source",
                },
                "ui_meta": {
                    "primary_evidence": {
                        "heading_path": "Introduction / Figure 1a",
                        "snippet": exact_evidence,
                        "page_start": 2,
                        "page_end": 2,
                        "selection_reason": "spad_noise_model_exact_source",
                        "strict_locate": True,
                    }
                },
            }
        ],
        anchor_ns="exact-spad-long",
        canonical_paths=[source_path],
    )

    assert "#kb-cite-" in rendered
    assert len(details) == 1
    assert "single-source Poisson noise model" in details[0]["card_evidence"]
    assert "afterpulsing and crosstalk noise" in details[0]["card_evidence"]
    assert "evidence_quote_filtered" not in details[0]["card_quality_flags"]
    assert "missing_evidence_quote" not in details[0]["card_quality_flags"]


def test_system_a_canonical_path_matches_windows_and_posix_separators() -> None:
    canonical_path = "F:/library/scigs/scigs.en.md"
    raw_hit_path = r"F:\library\scigs\scigs.en.md"
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "SCIGS 从单张压缩图像重建动态 3D 场景 [1]。",
        [
            {
                "text": (
                    "The proposed SCIGS reconstructs a 3D explicit scene from a single compressed "
                    "image and extends the task to dynamic 3D scenes."
                ),
                "meta": {
                    "source_path": raw_hit_path,
                    "heading_path": "Abstract",
                    "block_id": "blk_abstract",
                    "anchor_id": "p_abstract",
                },
            },
            {
                "text": "Title: SCIGS: 3D Gaussians Splatting from a Snapshot Compressive Image",
                "meta": {
                    "source_path": canonical_path,
                    "citation_plan_slot": True,
                },
            },
        ],
        anchor_ns="test-cross-platform-path",
        canonical_paths=[canonical_path],
    )

    assert "#kb-cite-" in rendered
    assert len(details) == 1
    assert details[0]["citation_plan_slot"] is False
    assert details[0]["heading_path"] == "Abstract"
    assert "dynamic 3D scenes" in details[0]["evidence_quote"]
    assert details[0]["block_id"] == "blk_abstract"


def test_system_a_binds_chinese_color_spi_claim_to_english_acquisition_evidence() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "彩色 SPI 可使用频分复用、多探测器和空间-光谱采集 [1]。",
        [
            {
                "text": (
                    "Color SPI uses frequency-division multiplexing, a single-time measurement "
                    "with multiple detectors, and a spatial-spectral acquisition scheme."
                ),
                "meta": {"source_path": "F:/library/dl-spi-review.en.md", "heading_path": "Color SPI"},
            }
        ],
    )

    assert "#kb-cite-" in rendered
    assert len(details) == 1
    assert "frequency-division multiplexing" in details[0]["evidence_quote"]


def test_system_a_binds_chinese_basis_claim_to_english_hsi_fsi_evidence() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "该路线从随机模式转向确定性正交基，并追求完美重构 [1]。",
        [
            {
                "text": (
                    "Random patterns form a non-orthogonal set. Deterministic orthogonal basis "
                    "patterns used by HSI and FSI enable perfect reconstruction in principle."
                ),
                "meta": {"source_path": "F:/library/hsi-fsi.en.md", "heading_path": "Comparison of theory"},
            }
        ],
    )

    assert "#kb-cite-" in rendered
    assert len(details) == 1
    assert "perfect reconstruction" in details[0]["evidence_quote"]


def test_system_a_binds_chinese_spi_bottleneck_to_english_abstract_evidence() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "传统 SPI 的图像质量有限、迭代重建计算时间长，限制了实际应用 [1]。",
        [
            {
                "text": (
                    "The limited image quality and lengthy computational times for iterative "
                    "reconstruction still hinder practical application."
                ),
                "meta": {"source_path": "F:/library/dl-spi-review.en.md", "heading_path": "Abstract"},
            }
        ],
    )

    assert "#kb-cite-" in rendered
    assert len(details) == 1
    assert "image quality and reconstruction time" in details[0]["binding_overlap_terms"]


def test_system_a_does_not_bind_from_source_title_without_body_evidence() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "彩色 SPI 可以使用频分复用完成单次采集 [1]。",
        [
            {
                "text": "The detector array was calibrated before every experiment.",
                "meta": {
                    "source_path": "F:/library/frequency-multiplexing.en.md",
                    "source_name": "Frequency-Division Multiplexing for Color SPI.pdf",
                    "heading_path": "Calibration",
                },
            }
        ],
    )

    assert "#kb-cite-" not in rendered
    assert details == []


def test_system_a_cleans_markdown_heading_from_evidence_quote() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "Deep learning improves reconstruction speed and quality [1].",
        [
            {
                "text": "## Abstract Deep learning improves reconstruction speed and image quality for single-pixel imaging.",
                "meta": {
                    "source_path": "db/demo/deep-learning.en.md",
                    "heading_path": "Abstract",
                    "evidence_quote": "## Abstract Deep learning improves reconstruction speed and image quality for single-pixel imaging.",
                },
            }
        ],
        anchor_ns="test-markdown",
    )

    assert "#kb-cite-" in rendered
    assert len(details) == 1
    assert "Deep learning improves" in details[0]["evidence_quote"]
    assert "##" not in details[0]["evidence_quote"]
    flags = set((details[0].get("card_view") or {}).get("quality", {}).get("flags") or [])
    assert "missing_evidence_quote" not in flags


def test_system_a_treats_synthetic_section_discussion_as_low_value_evidence() -> None:
    assert _system_a_is_low_value_evidence_text(
        "该文在“Hadamard single-pixel imaging versus Fourier single-pixel imaging / 3. Comparison of experiment / 3.1 Numerical simulations”讨论了“single pixel imaging”。"
    )


def test_system_a_links_qclfm_refocusing_claim_across_chinese_and_english() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "\u8be5\u663e\u5fae\u955c\u901a\u8fc7\u5c04\u7ebf\u8ffd\u8e2a\u548c"
            "\u6ce2\u52a8\u5149\u5b66\u4f20\u64ad\u4e24\u6b65\u6570\u5b57"
            "\u91cd\u805a\u7126\uff0c\u5c06\u79bb\u7126\u6837\u54c1\u91cd\u65b0\u5bf9\u7126 [1]\u3002"
        ),
        [
            {
                "text": (
                    "The operation for digital refocusing of a sample placed out of focus "
                    "can be achieved using two steps. First, the trajectory of the photons "
                    "can be reconstructed through a ray tracing operation. For microscopic "
                    "samples, diffraction effects from wave optics must also be taken into account."
                ),
                "meta": {
                    "source_path": "db/demo/qclfm.en.md",
                    "heading_path": "B. Experimental Results / Digital Refocusing Procedure",
                    "evidence_quote": (
                        "The second step is to reverse this diffraction by applying a wave "
                        "propagation of distance -z to bring the sample back into focus."
                    ),
                },
            }
        ],
        anchor_ns="test",
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    assert details[0]["citation_route"] == "system_a"
    assert details[0]["binding_status"] == "grounded"
    assert "digital refocusing" in details[0]["binding_overlap_terms"]


def test_compound_qclfm_excerpt_keeps_both_refocusing_steps() -> None:
    excerpt = _compound_plan_evidence_excerpt(
        (
            "The operation for digital refocusing of a sample placed out of focus by a "
            "distance z can be achieved using two steps. First, using the position and "
            "angular information of each photon, the trajectory of the photons can be "
            "reconstructed through a ray tracing operation. For microscopic samples, "
            "diffraction effects from wave optics must also be taken into account. Thus, "
            "the second step is to reverse this diffraction by applying a wave propagation "
            "of distance -z to bring the sample back into focus."
        ),
        "Refocusing first uses ray tracing and then reverse wave propagation.",
    )

    assert "two steps" in excerpt
    assert "ray tracing operation" in excerpt
    assert "wave propagation" in excerpt


def test_compound_piln_excerpt_keeps_definition_and_finer_grained_design() -> None:
    excerpt = _compound_plan_evidence_excerpt(
        (
            "In this study, we proposed a self-supervised image-loop neural network "
            "(ILNet) with a part-based model for single-pixel imaging (SPI). ILNet employs "
            "a part-based model that divides image features into different parts to "
            "facilitate finer-grained learning, resulting in improved image details when "
            "reconstructing a randomly input 2D signal into a 2D object image."
        ),
        (
            "ILNet is a self-supervised image-loop neural network whose part-based model "
            "supports finer-grained learning."
        ),
    )

    assert "self-supervised image-loop neural network" in excerpt
    assert "part-based model" in excerpt
    assert "finer-grained learning" in excerpt


def test_compound_spi_prospects_excerpt_keeps_capabilities_and_applications() -> None:
    excerpt = _compound_plan_evidence_excerpt(
        (
            "As the approach suits a wide variety of detector technologies, images can "
            "be collected at wavelengths outside the reach of FPA technology or at high "
            "frame rates or in three dimensions. Promising applications include the "
            "visualization of hazardous gas leaks and 3D situation awareness for "
            "autonomous vehicles."
        ),
        "Single-pixel cameras support high frame rates and 3D situation awareness for autonomous vehicles.",
    )

    assert "wavelengths outside the reach of FPA technology" in excerpt
    assert "high frame rates" in excerpt
    assert "three dimensions" in excerpt
    assert "hazardous gas leaks" in excerpt
    assert "autonomous vehicles" in excerpt


def test_compound_3d_video_excerpt_selects_the_frame_rate_sentence() -> None:
    excerpt = _compound_plan_evidence_excerpt(
        (
            "Photometric stereo estimates surface normals from images captured under "
            "different illumination directions. Performing high-speed single-pixel "
            "imaging with four spatially-separated detectors enables continuous "
            "real-time 3D video at approximately 8 frames per second."
        ),
        "原文报告该三维视频系统的重建速度约为 8 帧/秒。",
    )

    assert "four spatially-separated detectors" in excerpt
    assert "8 frames per second" in excerpt
    assert not excerpt.startswith("Photometric stereo estimates")


def test_compound_fdm_excerpt_keeps_parallel_encoding_and_demodulation_chain() -> None:
    excerpt = _compound_plan_evidence_excerpt(
        (
            "The mask values are encoded in the phase of intensity modulation, and thus "
            "we require phase-sensitive detection, in this case provided by a lock-in "
            "amplifier (LIA). "
            "Each pixel of the SLM is modulated with either 0 or pi phase on $p$ "
            "frequencies simultaneously, according to the present mask patterns. "
            "The modulated light from the SLM is then multiplexed into a single-pixel "
            "detector. The signal is then demodulated by a number (p) of LIAs."
        ),
        "每个像素用 BPSK 将多个掩模编码到 p 个频率载波上。",
    )

    assert "p frequencies simultaneously" in excerpt
    assert "phase-sensitive detection" in excerpt
    assert "multiplexed into a single-pixel detector" in excerpt
    assert "demodulated" in excerpt


def test_compound_fdm_excerpt_accepts_chinese_parallel_slm_claim() -> None:
    plan_text = (
        "The mask values are encoded in the phase of intensity modulation, and thus "
        "we require phase-sensitive detection, in this case provided by a lock-in "
        "amplifier (LIA). "
        "Each pixel of the SLM is modulated with either 0 or pi phase on $p$ "
        "frequencies simultaneously, according to the present mask patterns. "
        "The modulated light from the SLM is then multiplexed into a single-pixel "
        "detector. The signal is then demodulated by a number (p) of LIAs."
    )

    excerpt = _compound_plan_evidence_excerpt(
        plan_text,
        "频分复用单像素成像将空间光调制器（SLM）的像素调制并行化。",
    )

    assert "p frequencies simultaneously" in excerpt
    assert "phase-sensitive detection" in excerpt
    assert "multiplexed into a single-pixel detector" in excerpt
    assert "demodulated" in excerpt
    assert (
        _compound_plan_evidence_excerpt(
            plan_text,
            "空间光调制器（SLM）的像素调制并行化。",
        )
        == ""
    )


def test_compound_fdm_excerpt_accepts_bound_mechanism_sentence() -> None:
    plan_text = (
        "The mask values are encoded in the phase of intensity modulation, and thus "
        "we require phase-sensitive detection, in this case provided by a lock-in "
        "amplifier (LIA). Each pixel of the SLM is modulated with either 0 or pi phase "
        "on p frequencies simultaneously, according to the present mask patterns. "
        "The modulated light from the SLM is then multiplexed into a single-pixel "
        "detector. The signal is then demodulated by a number (p) of LIAs."
    )

    excerpt = _compound_plan_evidence_excerpt(
        plan_text,
        "单个探测器接收多个频率通道，并由锁相放大器同时解调空间编码掩模。",
    )

    assert "phase-sensitive detection" in excerpt
    assert "p frequencies simultaneously" in excerpt
    assert "demodulated" in excerpt


def test_compound_pidl_excerpt_keeps_noise_chain_and_calibration_count() -> None:
    plan_text = (
        "With low bit depth and heavy noise, wefirst established a real-world physical "
        "noise model of SPAD arrays. The real physical noise sources consist of shot "
        "noise, fixed-pattern noise, dark count rate, afterpulsing and crosstalk noise, "
        "and deadtime noise from the quenching circuit. To calibrate the parameters, "
        "we collected a real-shot SPAD image dataset containing 2790 images in total, "
        "each with 64 × 32 pixels."
    )

    excerpt = _compound_plan_evidence_excerpt(
        plan_text,
        "SPAD 物理噪声模型包含暗计数、后脉冲和串扰，并由 2790 张图像标定。",
    )

    assert "physical noise model of SPAD arrays" in excerpt
    assert "dark count rate" in excerpt
    assert "deadtime noise" in excerpt
    assert "2790 images" in excerpt
    assert "64 × 32 pixels" in excerpt
    assert len(excerpt) <= 520


def test_compound_pidl_numeric_excerpt_keeps_all_calibration_dimensions() -> None:
    plan_text = (
        "With low bit depth and heavy noise, wefirst established a real-world physical "
        "noise model of SPAD arrays. The real physical noise sources consist of shot "
        "noise, fi xed-pattern noise from the SPAD array, dark count rate, afterpulsing "
        "and crosstalk noise from blind electron avalanche, and deadtime noise from the "
        "quenching circuit. We collected a real-shot SPAD image dataset containing 2790 "
        "images in total, each with 64 × 32 pixels. Among these images, there are 90 "
        "scenes, each with 10 different bit depths and 3 different illumination fl uxes."
    )

    excerpt = _compound_plan_evidence_excerpt(
        plan_text,
        (
            "The SPAD noise model was calibrated with 2790 images at 64 × 32 pixels, "
            "covering 90 scenes, 10 bit depths, and 3 illumination fluxes."
        ),
    )

    assert "physical noise model of SPAD arrays" in excerpt
    assert "dark count rate" in excerpt
    assert "2790 images" in excerpt
    assert "64 × 32 pixels" in excerpt
    assert "90 scenes" in excerpt
    assert "10 different bit depths" in excerpt
    assert "3 different illumination fl uxes" in excerpt
    assert len(excerpt) <= 520


def test_compound_pidl_training_excerpt_keeps_calibrated_model_and_image_pairs() -> None:
    plan_text = (
        "With the calibrated physical noise model under different illumination and "
        "acquisition settings, we further employed off-the-shelf public highresolution "
        "images (collected from the PASCAL VOC2007 [31] and … VOC2012 [32] datasets) "
        "to digitally synthesize a large-scale realistic singlephoton image dataset "
        "containing 2.6 million image pairs. The gated fusion transformer network was "
        "trained using the above large-scale singlephoton image dataset and tested on "
        "various SPAD images."
    )

    excerpt = _compound_plan_evidence_excerpt(
        plan_text,
        (
            "最后，利用标定好的模型和公开的高分辨率图像（如 PASCAL VOC2007）"
            "合成配对数据，用于训练深度学习网络。"
        ),
    )

    assert "calibrated physical noise model" in excerpt
    assert "PASCAL VOC2007" in excerpt
    assert "digitally synthesize" in excerpt
    assert "2.6 million image pairs" in excerpt
    assert "network was trained" in excerpt
    assert "tested on various SPAD images" in excerpt
    assert len(excerpt) <= 520


def test_system_a_ui_relevance_only_crosses_the_same_evidence_occurrence() -> None:
    evidence = (
        "The modulated light from the SLM is multiplexed into a single-pixel detector "
        "and retains phase and modulation frequency information."
    )
    ui_meta = {
        "why_line": "该段说明 SLM 调制信号如何进入单探测器并保留频率通道信息。",
        "why_generation": "answer_citation_grounded",
    }
    primary = {
        "heading_path": "B. Encoding",
        "block_id": "blk-encoding",
        "snippet": evidence,
    }

    assert _system_a_ui_relevance_for_occurrence(
        ui_meta,
        primary,
        heading="B. Encoding",
        block_id="blk-encoding",
        anchor_id="",
        evidence_quote=evidence,
    ) == ui_meta["why_line"]
    assert _system_a_ui_relevance_for_occurrence(
        ui_meta,
        primary,
        heading="C. Frequency Selection",
        block_id="blk-frequency",
        anchor_id="",
        evidence_quote="Carrier frequencies are selected to avoid crosstalk.",
    ) == ""


def test_compound_piln_excerpt_keeps_image_loop_iteration_chain() -> None:
    excerpt = _compound_plan_evidence_excerpt(
        (
            "In this study, we proposed a self-supervised image-loop neural network "
            "(ILNet) with a part-based model for single-pixel imaging (SPI). ILNet employs "
            "a part-based model that divides image features into different parts to "
            "facilitate finer-grained learning, resulting in improved image details when "
            "reconstructing a randomly input 2D signal into a 2D object image. Then, the "
            "2D image generated by ILNet can serve as input for the subsequent iteration "
            "to continuously incorporate prior information and ensure high-quality "
            "imaging at low sampling rates."
        ),
        "ILNet 的图像循环机制将半成品重建图像循环回网络输入，以逐步提升重建质量。",
    )

    assert "self-supervised image-loop neural network" in excerpt
    assert "randomly input 2D signal" in excerpt
    assert "subsequent iteration" in excerpt
    assert "low sampling rates" in excerpt


def test_system_a_keeps_piln_iteration_mechanism_link_after_binding() -> None:
    source_path = "db/PILN/PILN.en.md"
    plan_evidence = (
        "In this study, we proposed a self-supervised image-loop neural network "
        "(ILNet) with a part-based model for single-pixel imaging (SPI). ILNet employs "
        "a part-based model that divides image features into different parts to "
        "facilitate finer-grained learning, resulting in improved image details when "
        "reconstructing a randomly input 2D signal into a 2D object image. Then, the "
        "2D image generated by ILNet can serve as input for the subsequent iteration "
        "to continuously incorporate prior information and ensure high-quality "
        "imaging at low sampling rates."
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "ILNet 的图像循环机制将半成品重建图像循环回网络输入，"
            "替代原始随机信号，从而逐步提升重建质量 [1]。\n\n"
            "| 场景 | 原因 |\n|---|---|\n"
            "| 低采样率成像 | 图像循环机制支持高质量重建 [1] |"
        ),
        [
            {
                "text": plan_evidence,
                "meta": {
                    "source_path": source_path,
                    "source_name": "PILN",
                    "heading_path": "Abstract",
                    "ref_answer_citation_num": 1,
                },
            }
        ],
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "per_paragraph_budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "source_name": "PILN",
                    "heading_path": "Abstract",
                    "evidence_quote": plan_evidence,
                }
            ],
        },
        anchor_ns="piln-iteration",
        canonical_paths=[source_path],
        render_locale="zh",
    )

    assert rendered.count("[1](#kb-cite-") == 2
    assert any(
        "subsequent iteration" in str(detail.get("evidence_quote") or "")
        and "low sampling rates" in str(detail.get("evidence_quote") or "")
        for detail in details
    )


def test_system_a_selects_piln_abstract_slot_over_same_paper_methods_slot() -> None:
    source_path = "db/PILN/PILN.en.md"
    abstract_evidence = (
        "In this study, we proposed a self-supervised image-loop neural network "
        "(ILNet) with a part-based model for single-pixel imaging. ILNet employs "
        "a part-based model that divides image features to facilitate "
        "finer-grained learning."
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "ILNet is a self-supervised image-loop neural network whose part-based "
            "model supports finer-grained learning [1]."
        ),
        [
            {
                "text": "ILNet uses a semi-finished reconstructed image loop as its next input.",
                "meta": {
                    "source_path": source_path,
                    "source_name": "PILN",
                    "heading_path": "2.1. Methods",
                    "ref_answer_citation_num": 1,
                },
                "ui_meta": {
                    "primary_evidence": {
                        "heading_path": "2.1. Methods",
                        "snippet": "ILNet uses a semi-finished reconstructed image loop as its next input.",
                        "block_id": "methods-block",
                        "anchor_id": "methods-sentence",
                        "strict_locate": True,
                    }
                },
            }
        ],
        citation_plan={
            "budget": {"system_a": 2, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": source_path,
                    "source_name": "PILN",
                    "heading_path": "Abstract",
                    "page_start": 2,
                    "page_end": 2,
                    "evidence_quote": abstract_evidence,
                },
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "source_name": "PILN",
                    "heading_path": "2.1. Methods",
                    "page_start": 2,
                    "page_end": 2,
                    "strict_locate": True,
                    "block_id": "methods-block",
                    "evidence_quote": (
                        "ILNet uses a part-based model and a semi-finished reconstructed "
                        "image loop as its next input."
                    ),
                },
            ],
        },
        anchor_ns="piln-abstract",
        canonical_paths=[source_path],
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    assert details[0]["heading_path"] == "Abstract"
    assert "self-supervised image-loop neural network" in details[0]["evidence_quote"]
    assert "finer-grained learning" in details[0]["evidence_quote"]


def test_hsi_fsi_metric_table_compaction_restores_metric_and_sampling_label() -> None:
    compact = _compact_metric_table_evidence(
        (
            "Table 2. 1%: PNSR (dB) / Hadamard / circular = 8.01; "
            "PNSR (dB) / Fourier / circular = 8.08; "
            "SSIM (%) / Hadamard / circular = 10.0; "
            "SSIM (%) / Fourier / circular = 11.1"
        ),
        answer_claim="在 1% 采样率下比较 Hadamard 与 Fourier 的 PSNR 和 SSIM。",
    )

    assert "1% sampling ratio" in compact
    assert "PSNR is 8.01 dB versus 8.08 dB" in compact
    assert "SSIM is 10.0% versus 11.1%" in compact


def test_authoritative_metric_plan_survives_readability_cleanup_and_keeps_plan_locator() -> None:
    source_path = "db/hsi-fsi/hsi-fsi.en.md"
    plan_evidence = (
        "Table 2. Hadamard single-pixel imaging versus Fourier single-pixel imaging / "
        "3. Comparison of experiment / 3.1 Numerical simulations. "
        "Table 2. Quantitative comparison results for Siemens star. "
        "1%: PNSR (dB) / Hadamard / circular = 8.01; "
        "PNSR (dB) / Fourier / circular = 8.08; "
        "SSIM (%) / Hadamard / circular = 10.0; "
        "SSIM (%) / Fourier / circular = 11.1"
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "在 1% 采样率下，Hadamard 与 Fourier 的 PSNR 和 SSIM 很接近 [1]。",
        [
            {
                "text": plan_evidence,
                "meta": {
                    "source_path": source_path,
                    "source_name": "Hadamard versus Fourier",
                    "heading_path": "2. Comparison of theory / 2.1 Principle of HSI and FSI",
                    "ref_best_heading_path": "2. Comparison of theory / 2.1 Principle of HSI and FSI",
                    "ref_answer_citation_num": 1,
                },
            }
        ],
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "heading_path": "3. Comparison of experiment / 3.1 Numerical simulations",
                    "evidence_quote": plan_evidence,
                    "block_id": "table-block",
                    "anchor_id": "table-2",
                    "anchor_kind": "table",
                    "page_start": 3,
                    "page_end": 3,
                }
            ],
        },
        anchor_ns="authoritative-metric-plan",
        canonical_paths=[source_path],
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    detail = details[0]
    evidence = str(detail.get("card_evidence") or "")
    assert "1% sampling ratio" in evidence
    assert "PSNR is 8.01 dB versus 8.08 dB" in evidence
    assert "SSIM is 10.0% versus 11.1%" in evidence
    assert detail["heading_path"] == (
        "3. Comparison of experiment / 3.1 Numerical simulations"
    )
    assert detail["card_locator"].startswith("3. Comparison of experiment")
    visible_texts = [
        str(section.get("text") or "")
        for section in detail["card_view"]["sections"]
    ]
    assert len(visible_texts) == len(set(visible_texts))


def test_metric_table_plan_beats_same_source_generic_comparison_sentence() -> None:
    source_path = "db/hsi-fsi/hsi-fsi.en.md"
    generic_evidence = (
        "We evaluate reconstruction quality using PSNR and SSIM. Based on the "
        "comparison results, FSI has better performance than HSI under undersampling."
    )
    table_evidence = (
        "Table 2. Quantitative comparison results for Siemens star. "
        "1%: PNSR (dB) / Hadamard / circular = 8.01; "
        "PNSR (dB) / Fourier / circular = 8.08; "
        "SSIM (%) / Hadamard / circular = 10.0; "
        "SSIM (%) / Fourier / circular = 11.1"
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "欠采样时，Fourier 单像素成像（FSI）的重建质量优于 Hadamard（HSI） [1]。",
        [
            {
                "text": table_evidence,
                "meta": {
                    "source_path": source_path,
                    "source_name": (
                        "Hadamard single-pixel imaging versus Fourier single-pixel imaging"
                    ),
                    "heading_path": "3. Comparison / 3.1 Numerical simulations",
                    "ref_answer_citation_num": 1,
                },
            }
        ],
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": source_path,
                    "source_name": (
                        "Hadamard single-pixel imaging versus Fourier single-pixel imaging"
                    ),
                    "heading_path": "3. Comparison / 3.1 Numerical simulations",
                    "evidence_quote": generic_evidence,
                    "evidence_selection_reason": "prompt_aligned_source_sentence",
                },
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "source_name": (
                        "Hadamard single-pixel imaging versus Fourier single-pixel imaging"
                    ),
                    "heading_path": "3. Comparison / 3.1 Numerical simulations",
                    "evidence_quote": table_evidence,
                },
            ],
        },
        anchor_ns="same-source-metric-plan",
        canonical_paths=[source_path],
        render_locale="zh",
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    assert "1% sampling ratio" in details[0]["card_evidence"]
    assert "PSNR is 8.01 dB versus 8.08 dB" in details[0]["card_evidence"]
    assert "SSIM is 10.0% versus 11.1%" in details[0]["card_evidence"]


def test_system_a_links_training_generalization_claim_across_chinese_and_english() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "数据驱动策略训练时间长、泛化能力有限，难以适应多样化成像场景 [1]。",
        [
            {
                "text": (
                    "Data-driven strategies have prolonged training duration and limited "
                    "generalization across diverse imaging scenes."
                ),
                "meta": {
                    "source_path": "db/demo/dl-spi-review.en.md",
                    "heading_path": "4. Strategy and Advantages",
                    "evidence_quote": (
                        "Data-driven strategies have prolonged training duration and limited "
                        "generalization across diverse imaging scenes."
                    ),
                    "citation_plan_slot": True,
                },
            }
        ],
        anchor_ns="test",
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    assert details[0]["binding_status"] == "grounded"


def test_claim_specific_hit_beats_broader_prompt_aligned_plan_and_keeps_link() -> None:
    source_path = "db/fdm/fdm.en.md"
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "SLM 的每个像素同时加载多个载波频率，从而并行编码多个掩模图案 [1]。",
        [
            {
                "text": (
                    "All SLM pixels use BPSK at carrier frequencies f1 through f4, "
                    "with one mask encoded at each frequency."
                ),
                "meta": {"source_path": source_path, "heading_path": "State characterization"},
                "ui_meta": {
                    "primary_evidence": {
                        "heading_path": "State characterization",
                        "snippet": "All SLM pixels use BPSK at carrier frequencies f1 through f4.",
                        "block_id": "old-block",
                        "anchor_id": "old-anchor",
                        "page_start": 3,
                        "strict_locate": True,
                    }
                },
            }
        ],
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "heading_path": "Abstract",
                    "page_start": 1,
                    "page_end": 1,
                    "evidence_selection_reason": "prompt_aligned_source_sentence",
                    "evidence_quote": (
                        "Frequency-division methods parallelize the single-pixel imaging "
                        "process and trade signal-to-noise ratio for acquisition speed."
                    ),
                }
            ],
        },
        anchor_ns="prompt-aligned",
        canonical_paths=[source_path],
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    assert details[0]["binding_status"] == "grounded"
    assert details[0]["heading_path"] == "State characterization"
    assert details[0]["page_start"] == 3
    assert "BPSK" in details[0]["evidence_quote"]
    assert details[0]["block_id"] == "old-block"
    assert details[0]["anchor_id"] == "old-anchor"
    assert "SLM" in details[0]["binding_overlap_terms"]


def test_system_a_suppresses_weak_candidate_binding_instead_of_linking() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "For real-time or low-sampling imaging, Hadamard subsampling is worth comparing [1].",
        [
            {
                "text": (
                    "With a static object, the corresponding surface orientation can be "
                    "determined by analyzing the object images under different illumination directions."
                ),
                "meta": {
                    "source_path": "db/demo/3d-single-pixel-video.en.md",
                    "heading_path": "Methods / Photometric stereo",
                    "evidence_quote": (
                        "Photometric stereo allows the surface orientation of a static object "
                        "to be estimated from images under different illumination directions."
                    ),
                },
            }
        ],
        anchor_ns="test",
    )

    assert "#kb-cite-" not in rendered
    assert "[1]" not in rendered
    assert details == []


def test_system_a_links_perovskite_boundary_claim_across_chinese_and_english() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "\u8be5\u8bba\u6587\u7684\u6838\u5fc3\u662f\u7535\u9a71\u52a8"
            "\u9499\u949b\u77ff\u6fc0\u5149\u5668\u4ef6\u7684\u5668\u4ef6\u7269\u7406 [1]\u3002"
        ),
        [
            {
                "text": (
                    "We have demonstrated electrically driven lasing from a dual-cavity "
                    "perovskite device."
                ),
                "meta": {
                    "source_path": "db/demo/perovskite-laser.en.md",
                    "heading_path": "Conclusion",
                    "evidence_quote": (
                        "We have demonstrated electrically driven lasing from a dual-cavity "
                        "perovskite device."
                    ),
                },
            }
        ],
        anchor_ns="test",
    )

    assert "#kb-cite-" in rendered
    assert len(details) == 1
    assert details[0]["citation_route"] == "system_a"
    assert details[0]["binding_status"] == "grounded"


def test_system_a_requires_specific_strong_term_not_only_broad_domain_overlap() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "High-speed DMD modulation is a hardware route for real-time single-pixel imaging [1].",
        [
            {
                "text": (
                    "With a static object, photometric stereo determines surface orientation "
                    "from object images under different illumination directions."
                ),
                "meta": {
                    "source_path": "db/demo/3d-single-pixel-video.en.md",
                    "heading_path": "Methods / Photometric stereo",
                    "evidence_quote": (
                        "Photometric stereo estimates surface orientation from images under "
                        "different illumination directions."
                    ),
                    "source_name": "Journal of Optics-2016-3D single-pixel video.pdf",
                },
            }
        ],
        anchor_ns="test",
    )

    assert "#kb-cite-" not in rendered
    assert "[1]" not in rendered
    assert details == []


def test_system_a_suppresses_link_when_answer_claim_conflicts_with_hit_topic() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "Interferometric (iSCAT) microscopy detects unlabeled proteins through "
            "scattering contrast [2]."
        ),
        [
            {
                "text": "Adaptive foveated single-pixel imaging uses dynamic supersampling.",
                "meta": {
                    "source_path": "db/demo/foveated.en.md",
                    "heading_path": "INTRODUCTION / Foveated single-pixel imaging",
                },
            },
            {
                "text": (
                    "Structured detection for simultaneous super-resolution and optical "
                    "sectioning in laser scanning microscopy."
                ),
                "meta": {
                    "source_path": "db/demo/structured-detection.en.md",
                    "heading_path": "Abstract",
                    "evidence_quote": (
                        "This work proposes structured detection for optical sectioning "
                        "in laser scanning microscopy."
                    ),
                },
            },
        ],
        anchor_ns="test",
    )

    assert "[2](#kb-cite-" not in rendered
    assert "Interferometric (iSCAT) microscopy detects unlabeled proteins" in rendered
    assert "Structured detection" not in rendered
    assert details == []


def test_system_a_marks_grounded_binding_with_shared_domain_terms() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "Structured detection improves optical sectioning in laser scanning microscopy [1].",
        [
            {
                "text": (
                    "Structured detection enables simultaneous super-resolution and "
                    "optical sectioning in laser scanning microscopy."
                ),
                "meta": {
                    "source_path": "db/demo/structured-detection.en.md",
                    "heading_path": "Abstract",
                    "evidence_quote": (
                        "Structured detection enables simultaneous super-resolution "
                        "and optical sectioning in laser scanning microscopy."
                    ),
                },
            },
        ],
        anchor_ns="test",
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    detail = details[0]
    assert detail["binding_status"] == "grounded"
    assert detail["binding_confidence"] >= 0.8
    assert "structured detection" in detail["binding_overlap_terms"]
    assert detail["card_quality_label"] == "证据匹配"
    assert detail["card_warning"] == ""
    assert "答案句" in detail["support_relation"] or "answer sentence" in detail["support_relation"]


def test_system_a_binds_chinese_denoising_taxonomy_to_english_source_terms() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "这篇综述把经典去噪方法分为空间域方法和变换域方法两类 [1]。",
        [
            {
                "text": (
                    "Image denoising methods can be roughly classified as spatial domain "
                    "methods and transform domain methods."
                ),
                "meta": {
                    "source_path": "db/demo/denoising-review.en.md",
                    "source_name": "Brief review of image denoising techniques",
                    "heading_path": "Classical denoising method",
                },
            }
        ],
        anchor_ns="test",
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    assert details[0]["binding_status"] == "grounded"
    assert {
        "spatial domain denoising",
        "transform domain denoising",
    } <= set(details[0]["binding_overlap_terms"])


def test_system_a_reuses_one_card_for_duplicate_evidence_hits() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "Structured detection improves optical sectioning in laser scanning "
            "microscopy [1] and is the same evidence when mentioned again [2]."
        ),
        [
            {
                "text": "Structured detection improves optical sectioning in laser scanning microscopy.",
                "meta": {
                    "source_path": "db/demo/structured-detection.en.md",
                    "heading_path": "Abstract",
                    "evidence_quote": "Structured detection improves optical sectioning in laser scanning microscopy.",
                    "primary_block_id": "abstract-1",
                    "primary_anchor_id": "sent-1",
                    "anchor_kind": "sentence",
                },
            },
            {
                "text": "Structured detection improves optical sectioning in laser scanning microscopy.",
                "meta": {
                    "source_path": "db/demo/structured-detection.en.md",
                    "heading_path": "Abstract",
                    "evidence_quote": "Structured detection improves optical sectioning in laser scanning microscopy.",
                    "primary_block_id": "abstract-1",
                    "primary_anchor_id": "sent-1",
                    "anchor_kind": "sentence",
                },
            },
        ],
        anchor_ns="test",
    )

    anchors = re.findall(r"\[(?:1|2)\]\(#([^) \"\n]+)", rendered)
    assert len(anchors) == 2
    assert len(set(anchors)) == 1
    assert len(details) == 1
    assert details[0]["linked_nums"] == [1, 2]
    assert details[0]["evidence_fingerprint"]


def test_system_a_reuses_repeated_same_number_for_same_evidence() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "Foveated single-pixel imaging uses dynamic supersampling with a DMD [1].\n"
            "The same dynamic supersampling evidence is cited again here [1]."
        ),
        [
            {
                "text": (
                    "## Foveated single-pixel imaging\n"
                    "Single-pixel imaging can use dynamic supersampling with a DMD."
                ),
                "meta": {
                    "source_path": "db/demo/foveated.en.md",
                    "heading_path": "INTRODUCTION",
                    "evidence_quote": (
                        "## Foveated single-pixel imaging\n"
                        "Single-pixel imaging can use dynamic supersampling with a DMD."
                    ),
                    "primary_block_id": "intro-1",
                    "primary_anchor_id": "sent-1",
                    "anchor_kind": "paragraph",
                },
            },
        ],
        anchor_ns="test",
    )

    anchors = re.findall(r"\[1\]\(#([^) \"\n]+)", rendered)
    assert len(anchors) == 2
    assert len(set(anchors)) == 1
    assert len(details) == 1
    assert "occurrence_specific_claim" not in details[0]["card_quality_flags"]
    assert "Foveated single-pixel imaging uses dynamic supersampling" in details[0]["answer_claim"]
    assert len(details[0]["answer_claims"]) == 2
    assert any("cited again" in claim for claim in details[0]["answer_claims"])
    assert "##" not in details[0]["card_evidence"]


def test_system_a_prefers_primary_evidence_location_from_hit_ui_meta() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "Deep learning helps with difficult scattering cases [1].",
        [
            {
                "text": "# Paper title\nAuthor A\nSingle-pixel imaging based on deep learning has attrac",
                "meta": {
                    "source_path": "db/demo/lpr.en.md",
                    "ref_best_heading_path": "5. Single-Pixel Imaging Realizations with Deep Learning / 5.4. Optical Encryption",
                },
                "ui_meta": {
                    "heading_path": "5.2. Imaging Through Scattering Media",
                    "primary_evidence": {
                        "heading_path": "5.2. Imaging Through Scattering Media",
                        "snippet": (
                            "Turbulence-immune imaging is a classical challenge in the field of imaging "
                            "through scattering weak media. DL has exhibited remarkable efficacy in addressing this problem"
                        ),
                        "block_id": "blk_scattering",
                        "anchor_id": "p_42",
                        "anchor_kind": "paragraph",
                    },
                },
            }
        ],
        anchor_ns="test",
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    detail = details[0]
    assert detail["heading_path"] == "5.2. Imaging Through Scattering Media"
    assert detail["card_locator"].startswith("5.2. Imaging Through Scattering Media")
    assert detail["block_id"] == "blk_scattering"
    assert detail["anchor_id"] == "p_42"
    assert "Optical Encryption" not in detail["card_locator"]
    assert "attrac" not in detail["card_evidence"]


def test_system_a_prefers_claim_specific_raw_hit_over_stale_primary() -> None:
    claim = (
        "Waveguide integration raises single-photon detector efficiency by constraining "
        "transmission at the cut-off frequency [1]."
    )
    hit = {
        "text": (
            "### 4.2 Waveguide integration\n"
            "The waveguide serves to confine energy within the waveguide medium. "
            "The cut-off frequency constrains energy transmission. Waveguide integration "
            "is widely used to improve light absorption and increase detection efficiency."
        ),
        "meta": {
            "source_path": "db/demo/spd-review.en.md",
            "ref_best_heading_path": "4 Methods / 4.2 Waveguide",
        },
        "ui_meta": {
            "primary_evidence": {
                "heading_path": "Emerging single-photon detection technique / Abstract",
                "snippet": (
                    "Single-photon detectors are a highly sensitive light detection technique "
                    "capable of detecting individual photons at extremely low light intensity levels."
                ),
                "block_id": "abstract",
                "anchor_id": "p1",
            }
        },
    }
    primary_evidence = hit["ui_meta"]["primary_evidence"]

    picked = _system_a_pick_best_evidence_candidate(
        hit=hit,
        meta=hit["meta"],
        ui_meta=hit["ui_meta"],
        primary_evidence=primary_evidence,
        answer_claim=claim,
        source_name="Emerging single-photon detection technique for high-performance photodetector.pdf",
        default_heading=primary_evidence["heading_path"],
    )

    assert picked["source"] == "hit_text"
    assert "cut-off frequency" in picked["readable_text"]
    assert "detection efficiency" in picked["readable_text"]


def test_system_a_strict_plan_primary_beats_stale_reader_alternative() -> None:
    exact = (
        "A beat frequency realizes phase stepping naturally in time through "
        "heterodyne holography."
    )
    stale = "Single-pixel holography uses a single-pixel detector."
    hit = {
        "text": exact,
        "meta": {
            "source_path": "db/SPH/SPH.en.md",
            "heading_path": "Introduction",
            "citation_plan_slot": True,
        },
        "ui_meta": {
            "primary_evidence": {
                "heading_path": "Introduction",
                "snippet": exact,
                "selection_reason": "citation_plan_slot",
                "strict_locate": True,
                "block_id": "blk-intro",
                "anchor_id": "p-intro",
            },
            "reader_open": {
                "evidenceAlternatives": [
                    {
                        "headingPath": "Results / Figure 2",
                        "snippet": stale,
                        "blockId": "blk-stale",
                    }
                ]
            },
        },
    }

    picked = _system_a_pick_best_evidence_candidate(
        hit=hit,
        meta=hit["meta"],
        ui_meta=hit["ui_meta"],
        primary_evidence=hit["ui_meta"]["primary_evidence"],
        answer_claim="外差拍频让相移在时间上自然展开。",
        source_name="High-throughput single-pixel holography",
        default_heading="Introduction",
    )

    assert picked["source"] == "primary_evidence"
    assert picked["readable_text"] == exact


def test_system_a_raw_hit_uses_its_own_locator_instead_of_stale_primary() -> None:
    source_path = "db/demo/spd-review.en.md"
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "Waveguide cut-off frequency constrains transmission and improves detection efficiency [1].",
        [
            {
                "text": (
                    "The waveguide cut-off frequency constrains energy transmission. "
                    "Waveguide integration improves light absorption and detection efficiency."
                ),
                "meta": {
                    "source_path": source_path,
                    "heading_path": "4 Methods / 4.2 Waveguide",
                    "block_id": "waveguide-block",
                    "anchor_id": "waveguide-anchor",
                    "anchor_kind": "paragraph",
                },
                "ui_meta": {
                    "primary_evidence": {
                        "heading_path": "Abstract",
                        "snippet": "Single-photon detectors sense weak light.",
                        "block_id": "abstract-block",
                        "anchor_id": "abstract-anchor",
                        "anchor_kind": "sentence",
                    }
                },
            }
        ],
        canonical_paths=[source_path],
        citation_plan={"budget": {"system_a": 1, "system_b": 0}},
        anchor_ns="raw-own-locator",
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    detail = details[0]
    assert detail["heading_path"] == "4 Methods / 4.2 Waveguide"
    assert detail["block_id"] == "waveguide-block"
    assert detail["anchor_id"] == "waveguide-anchor"
    assert detail["anchor_kind"] == "paragraph"


def test_system_a_raw_hit_without_locator_clears_stale_primary_locator() -> None:
    source_path = "db/demo/spd-review.en.md"
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "Waveguide cut-off frequency constrains transmission and improves detection efficiency [1].",
        [
            {
                "text": (
                    "The waveguide cut-off frequency constrains energy transmission. "
                    "Waveguide integration improves light absorption and detection efficiency."
                ),
                "meta": {
                    "source_path": source_path,
                    "ref_best_heading_path": "Abstract",
                    "primary_block_id": "abstract-block",
                    "primary_anchor_id": "abstract-anchor",
                },
                "ui_meta": {
                    "primary_evidence": {
                        "heading_path": "Abstract",
                        "snippet": "Single-photon detectors sense weak light.",
                        "block_id": "abstract-block",
                        "anchor_id": "abstract-anchor",
                        "anchor_kind": "sentence",
                    }
                },
            }
        ],
        canonical_paths=[source_path],
        citation_plan={"budget": {"system_a": 1, "system_b": 0}},
        anchor_ns="raw-no-locator",
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    detail = details[0]
    assert detail["heading_path"] == ""
    assert detail["block_id"] == ""
    assert detail["anchor_id"] == ""
    assert detail["anchor_kind"] == ""


def test_system_a_prefers_reader_open_primary_evidence_when_direct_primary_missing() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "Light-field microscopy reconstructs three-dimensional information [1].",
        [
            {
                "text": (
                    "# Quantum correlation light-field microscope with extreme depth of field\n"
                    "Yingwen Zhang, Yuhang Qin, Wenhao Li"
                ),
                "meta": {
                    "source_path": "db/demo/qclfm.en.md",
                    "heading_path": "I. INTRODUCTION",
                },
                "ui_meta": {
                    "reader_open": {
                        "primaryEvidence": {
                            "headingPath": "I. INTRODUCTION / Light-field microscope",
                            "highlightSnippet": (
                                "Conventional light-field microscope designs typically make use "
                                "of a microlens array to record spatial and angular information."
                            ),
                            "blockId": "intro-light-field",
                            "anchorId": "sent-light-field",
                            "anchorKind": "sentence",
                        }
                    }
                },
            }
        ],
        anchor_ns="test",
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    detail = details[0]
    assert detail["heading_path"] == "I. INTRODUCTION / Light-field microscope"
    assert detail["card_locator"].startswith("I. INTRODUCTION / Light-field microscope")
    assert detail["block_id"] == "intro-light-field"
    assert detail["anchor_id"] == "sent-light-field"
    assert "microlens array" in detail["card_evidence"]
    assert "Yingwen Zhang" not in detail["card_evidence"]
    assert "##" not in detail["card_evidence"]


def test_system_a_replaces_truncated_wrapped_primary_with_readable_alternative() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "Adaptive foveated single-pixel imaging uses dynamic supersampling to spend more samples near important regions [1].",
        [
            {
                "text": "source excerpt says: \"For comparison, uniformly imaging the entire field of view at the higher resolu...\"",
                "meta": {
                    "source_path": "db/demo/foveated.en.md",
                    "heading_path": "INTRODUCTION / Linear constraints",
                },
                "ui_meta": {
                    "reader_open": {
                        "primaryEvidence": {
                            "headingPath": "INTRODUCTION / Linear constraints",
                            "highlightSnippet": (
                                "source excerpt says: \"For comparison, uniformly imaging the entire "
                                "field of view at the higher resolu...\""
                            ),
                            "blockId": "bad-wrapper",
                            "anchorId": "p_bad",
                            "anchorKind": "paragraph",
                        },
                        "evidenceAlternatives": [
                            {
                                "headingPath": "INTRODUCTION",
                                "highlightSnippet": (
                                    "Dynamic supersampling adaptively allocates high-resolution "
                                    "sampling to the fovea while using lower resolution elsewhere."
                                ),
                                "blockId": "good-fovea",
                                "anchorId": "p_good",
                                "anchorKind": "paragraph",
                            }
                        ],
                    }
                },
            }
        ],
        anchor_ns="test",
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    detail = details[0]
    assert detail["block_id"] == "good-fovea"
    assert detail["anchor_id"] == "p_good"
    assert detail["heading_path"] == "INTRODUCTION"
    assert detail["evidence_source"] == "reader_open.evidenceAlternatives"
    assert detail["card_evidence"].startswith("Dynamic supersampling")
    assert "source excerpt says" not in detail["card_evidence"]
    assert "higher resolu" not in detail["card_evidence"]


def test_system_a_does_not_route_to_system_b_from_reference_title_words() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "This paper is useful for learning deep-learning single-pixel imaging [1].",
        [
            {
                "text": "Deep learning improves single-pixel imaging reconstruction quality.",
                "meta": {
                    "source_path": "db/demo/deep-spi.en.md",
                    "heading_path": "Abstract",
                    "evidence_quote": "Deep learning improves single-pixel imaging reconstruction quality.",
                },
            }
        ],
        anchor_ns="test",
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    assert details[0]["citation_route"] == "system_a"
    assert details[0]["card_kind"] == "answer_evidence"


def test_system_a_keeps_distinct_cards_for_distinct_evidence_locations() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "Structured detection improves optical sectioning [1], while dynamic "
            "supersampling changes the sampling pattern [2]."
        ),
        [
            {
                "text": "Structured detection improves optical sectioning in laser scanning microscopy.",
                "meta": {
                    "source_path": "db/demo/microscopy.en.md",
                    "heading_path": "Abstract",
                    "evidence_quote": "Structured detection improves optical sectioning in laser scanning microscopy.",
                    "primary_block_id": "abstract-1",
                    "primary_anchor_id": "sent-1",
                    "anchor_kind": "sentence",
                },
            },
            {
                "text": "Dynamic supersampling allocates more samples to important image regions.",
                "meta": {
                    "source_path": "db/demo/microscopy.en.md",
                    "heading_path": "Method",
                    "evidence_quote": "Dynamic supersampling allocates more samples to important image regions.",
                    "primary_block_id": "method-2",
                    "primary_anchor_id": "sent-2",
                    "anchor_kind": "sentence",
                },
            },
        ],
        anchor_ns="test",
    )

    anchors = re.findall(r"\[(?:1|2)\]\(#([^) \"\n]+)", rendered)
    assert len(anchors) == 2
    assert len(set(anchors)) == 2
    assert len(details) == 2
    assert [d["linked_nums"] for d in details] == [[1], [2]]


def test_system_a_splits_reused_number_for_distinct_numeric_claims() -> None:
    source_path = "db/pidl/pidl.en.md"
    evidence = (
        "The multi-source physical noise model of SPAD arrays includes crosstalk and dark count rate. "
        "We collected 2790 real SPAD images covering 90 scenes, 10 bit depths, and 3 illumination fluxes."
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "The SPAD model includes crosstalk and dark count rate [1].\n\n"
            "The calibration dataset contains 2790 SPAD images from 90 scenes [1]."
        ),
        [
            {
                "text": evidence,
                "meta": {
                    "source_path": source_path,
                    "heading_path": "Introduction",
                    "page_start": 3,
                    "page_end": 3,
                },
            }
        ],
        citation_plan={"budget": {"system_a": 2, "system_b": 0}},
        anchor_ns="reused-numeric",
        canonical_paths=[source_path],
    )

    assert rendered.count("[1](#kb-cite-") == 2
    assert len(details) == 2
    assert any("2790" in str(detail.get("card_evidence") or "") for detail in details)
    assert all(detail["page_start"] == 3 for detail in details)


def test_system_a_suppresses_adjacent_topic_without_physical_noise_model_evidence() -> None:
    source_path = "db/dl-spi-review/dl-spi-review.en.md"
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "这篇单光子论文建议先为 SPAD 传感器建立并校准多源物理噪声模型 [1]。"
        ),
        [
            {
                "text": (
                    "Photon-level single-pixel imaging uses a single photon detector and deep "
                    "learning reconstruction for extreme or long-distance scenes."
                ),
                "meta": {
                    "source_path": source_path,
                    "heading_path": "Imaging at Photon-Level",
                    "page_start": 10,
                },
            }
        ],
        citation_plan={"budget": {"system_a": 1, "system_b": 0}},
        anchor_ns="physical-model-scope",
        canonical_paths=[source_path],
    )

    assert "[1]" not in rendered
    assert details == []


def test_system_a_uses_full_plan_evidence_after_saved_hit_was_compacted() -> None:
    source_path = "db/pidl/pidl.en.md"
    compact_hit = (
        "The multi-source physical noise model of SPAD arrays includes crosstalk and dark count rate."
    )
    full_plan_evidence = (
        compact_hit
        + " We collected 2790 real SPAD images covering 90 scenes, 10 bit depths, "
        "and 3 illumination fluxes."
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "The SPAD model includes crosstalk and dark count rate [1].\n\n"
            "The calibration dataset contains 2790 images from 90 scenes, 10 bit depths, "
            "and 3 illumination fluxes [1]."
        ),
        [
            {
                "text": compact_hit,
                "meta": {
                    "source_path": source_path,
                    "source_name": "PIDL",
                    "heading_path": "Introduction",
                    "ref_answer_citation_num": 1,
                },
            }
        ],
        citation_plan={
            "budget": {"system_a": 2, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "source_name": "PIDL",
                    "heading_path": "Introduction",
                    "page_start": 3,
                    "page_end": 3,
                    "evidence_quote": full_plan_evidence,
                }
            ],
        },
        anchor_ns="compacted-plan",
        canonical_paths=[source_path],
    )

    assert rendered.count("[1](#kb-cite-") == 2
    assert any("2790" in str(detail.get("card_evidence") or "") for detail in details)
    assert any("3 illumination fluxes" in str(detail.get("card_evidence") or "") for detail in details)


def test_microscopy_direct_plan_quote_overrides_broad_abstract_lead_in() -> None:
    source_path = "db/s2ism/s2ism.en.md"
    broad_lead = (
        "Fast detector arrays overcome the spatial-resolution and signal-to-noise trade-off. "
        "Current approaches do not provide optical sectioning in thick samples."
    )
    direct_evidence = (
        "From single-plane acquisition, we reconstruct an image with digital and optical "
        "super-resolution, high signal-to-noise ratio and enhanced optical sectioning."
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "s2ISM structured detection simultaneously provides super-resolution, high SNR, "
            "and optical sectioning [1]."
        ),
        [
            {
                "text": broad_lead,
                "meta": {
                    "source_path": source_path,
                    "source_name": "Structured detection for s2ISM",
                    "heading_path": "Abstract",
                    "ref_answer_citation_num": 1,
                },
            }
        ],
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "source_name": "Structured detection for s2ISM",
                    "heading_path": "Abstract",
                    "page_start": 1,
                    "page_end": 1,
                    "evidence_quote": direct_evidence,
                    "evidence_selection_reason": "microscopy_direct",
                }
            ],
        },
        anchor_ns="microscopy-direct",
        canonical_paths=[source_path],
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    rendered_evidence = str(details[0].get("evidence_quote") or "")
    assert details[0]["selection_reason"] == "microscopy_direct"
    assert "super-resolution" in rendered_evidence
    assert "high signal-to-noise ratio" in rendered_evidence
    assert "optical sectioning" in rendered_evidence
    assert "Current approaches do not provide" not in rendered_evidence


def test_system_a_keeps_cross_language_super_resolution_outcome_citation() -> None:
    source_path = "db/pidl/pidl.en.md"
    evidence = (
        "We introduce deep learning into SPAD, enabling super-resolution single-photon "
        "imaging with enhancement of bit depth and imaging quality."
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "最终效果是实现超分辨率、位深增强和成像质量提升 [1]。",
        [
            {
                "text": evidence,
                "meta": {
                    "source_path": source_path,
                    "source_name": "Physics-informed SPAD imaging",
                    "heading_path": "Abstract",
                    "ref_answer_citation_num": 1,
                },
            }
        ],
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "source_name": "Physics-informed SPAD imaging",
                    "heading_path": "Abstract",
                    "page_start": 1,
                    "page_end": 1,
                    "evidence_quote": evidence,
                }
            ],
        },
        anchor_ns="cross-language-outcome",
        canonical_paths=[source_path],
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    assert details[0]["binding_status"] == "grounded"


def test_system_a_recognizes_chinese_frequency_multiplexing_wording() -> None:
    source_path = "db/fdm/fdm.en.md"
    evidence = (
        "Each pixel of the SLM is modulated on p frequencies simultaneously. "
        "The modulated light is multiplexed into a single-pixel detector."
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "前者在单探测器上通过频率复用并行化空间编码 [1]。",
        [
            {
                "text": evidence,
                "meta": {
                    "source_path": source_path,
                    "source_name": "Frequency-division-multiplexed SPI",
                    "heading_path": "B. Encoding",
                    "ref_answer_citation_num": 1,
                },
            }
        ],
        citation_plan={"budget": {"system_a": 1, "system_b": 0}},
        anchor_ns="frequency-multiplexing-zh",
        canonical_paths=[source_path],
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    assert details[0]["binding_status"] == "grounded"


def test_system_a_splits_named_dataset_claim_within_same_source_block() -> None:
    source_path = "db/pidl/pidl.en.md"
    full_plan_evidence = (
        "We established and calibrated a physical noise model of SPAD arrays. "
        "We then used public high-resolution images collected from PASCAL VOC2007 "
        "to train the reconstruction network."
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "The method calibrates a physical noise model of SPAD arrays [1].\n\n"
            "Training also uses the PASCAL VOC2007 public image dataset [1]."
        ),
        [
            {
                "text": "We established and calibrated a physical noise model of SPAD arrays.",
                "meta": {
                    "source_path": source_path,
                    "source_name": "PIDL",
                    "heading_path": "Introduction",
                    "ref_answer_citation_num": 1,
                },
            }
        ],
        citation_plan={
            "budget": {"system_a": 2, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "source_name": "PIDL",
                    "heading_path": "Introduction",
                    "page_start": 3,
                    "page_end": 3,
                    "evidence_quote": full_plan_evidence,
                }
            ],
        },
        anchor_ns="named-dataset",
        canonical_paths=[source_path],
    )

    anchors = re.findall(r"\[1\]\(#([^) \"\n]+)", rendered)
    assert len(anchors) == 2
    assert len(set(anchors)) == 2
    assert len(details) == 2
    named_detail = next(
        detail for detail in details if "PASCAL VOC2007" in str(detail.get("answer_claim") or "")
    )
    assert "PASCAL VOC2007" in str(named_detail.get("card_evidence") or "")
    assert named_detail["page_start"] == 3


def test_system_a_keeps_real_pidl_calibration_and_training_sentence_links() -> None:
    source_path = (
        "db/NatCommun-2023-High-resolution single-photon imaging with physics-informed "
        "deep learning/NatCommun-2023-High-resolution single-photon imaging with physics-informed "
        "deep learning.en.md"
    )
    plan_evidence = (
        "with low bit depth, low resolution and heavy noise in photon-limited scenarios, "
        "wefirst established a real-world physical noise model of SPAD arrays. The real "
        "physical noise sources consist of shot noise from photon incidence, fi xed-pattern "
        "noise from SPAD array's photon absorption, dark count rate, afterpulsing and "
        "crosstalk noise from blind electron avalanche, and deadtime noise from the quenching "
        "circuit. We collected a real-shot SPAD image dataset containing 2790 images in total, "
        "each with 64 × 32 pixels. Among these images, there are 90 scenes, each with 10 "
        "different bit depths and 3 different illumination fl uxes. With the calibrated "
        "physical noise model under different illumination and acquisition settings, we "
        "further employed off-the-shelf public highresolution images (collected from the "
        "PASCAL VOC2007 [31] and … VOC2012 [32] datasets) to digitally synthesize a large-scale "
        "realistic singlephoton image dataset containing 2.6 million image pairs. The gated "
        "fusion transformer network was trained using the above large-scale singlephoton image "
        "dataset and tested on various SPAD images."
    )
    answer = (
        "该方法采集 2790 张真实 SPAD 图像（64×32 像素，涵盖 90 个场景、10 种比特深度"
        "和 3 种光照通量）来标定模型参数 [1]。然后，利用这个标定好的物理模型，结合"
        "PASCAL VOC2007/VOC2012 公共高分辨率图像数字合成大规模真实单光子图像数据集，"
        "并用该数据集训练网络 [1]。"
    )

    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        answer,
        [
            {
                "text": (
                    "wefirst established a real-world physical noise model of SPAD arrays. "
                    "We collected a real-shot SPAD image dataset containing 2790 images in total, "
                    "each with 64 × 32 pixels."
                ),
                "meta": {
                    "source_path": source_path,
                    "source_name": "High-resolution single-photon imaging with PIDL",
                    "heading_path": "Introduction",
                    "ref_answer_citation_num": 1,
                },
            }
        ],
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "per_paragraph_budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "source_name": "High-resolution single-photon imaging with PIDL",
                    "heading_path": "Introduction",
                    "page_start": 3,
                    "page_end": 3,
                    "evidence_quote": plan_evidence,
                }
            ],
        },
        anchor_ns="real-pidl-training",
        canonical_paths=[source_path],
        render_locale="zh",
    )

    anchors = re.findall(r"\[1\]\(#([^) \"\n]+)", rendered)
    assert len(anchors) == 2
    assert len(set(anchors)) == 2
    assert len(details) == 2
    training_detail = next(
        detail for detail in details if "PASCAL VOC2007" in str(detail.get("answer_claim") or "")
    )
    training_evidence = str(training_detail.get("card_evidence") or "")
    assert "calibrated physical noise model" in training_evidence
    assert "PASCAL VOC2007" in training_evidence
    assert "2.6 million image pairs" in training_evidence
    assert "network was trained" in training_evidence


def test_system_a_budget_keeps_distinct_plan_slots_separate() -> None:
    source_path = "db/paper/paper.en.md"
    evidence = (
        "AlphaNet reaches PSNR 30.0 dB on DatasetX. "
        "BetaNet reaches SSIM 0.9 on DatasetY."
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "AlphaNet reaches PSNR 30.0 dB on DatasetX [1]. "
            "BetaNet reaches SSIM 0.9 on DatasetY [2]."
        ),
        [
            {
                "text": evidence,
                "meta": {
                    "source_path": source_path,
                    "source_name": "Paper",
                    "heading_path": "Section",
                    "ref_answer_citation_num": number,
                },
            }
            for number in (1, 2)
        ],
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "per_paragraph_budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [number],
                    "source_path": source_path,
                    "heading_path": "Section",
                    "evidence_quote": evidence,
                }
                for number in (1, 2)
            ],
        },
        anchor_ns="distinct-plan-slots",
        canonical_paths=[source_path, source_path],
    )

    assert rendered.count("](#kb-cite-") == 1
    assert len(details) == 1


def test_system_a_budget_survives_inline_math_fragment_split() -> None:
    sources = ["db/alpha/alpha.en.md", "db/beta/beta.en.md"]
    evidence = [
        "AlphaNet reaches PSNR 30.0 dB on DatasetX.",
        "BetaNet reaches SSIM 0.9 on DatasetY.",
    ]
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "AlphaNet reaches PSNR 30.0 dB on DatasetX [1]. "
            "$x$ BetaNet reaches SSIM 0.9 on DatasetY [2]."
        ),
        [
            {
                "text": quote,
                "meta": {
                    "source_path": source_path,
                    "source_name": f"Paper {number}",
                    "heading_path": "Results",
                    "ref_answer_citation_num": number,
                },
            }
            for number, (source_path, quote) in enumerate(
                zip(sources, evidence),
                start=1,
            )
        ],
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "per_paragraph_budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [number],
                    "source_path": source_path,
                    "heading_path": "Results",
                    "evidence_quote": quote,
                }
                for number, (source_path, quote) in enumerate(
                    zip(sources, evidence),
                    start=1,
                )
            ],
        },
        anchor_ns="inline-math-budget",
        canonical_paths=sources,
    )

    assert rendered.count("](#kb-cite-") == 1
    assert len(details) == 1


def test_system_a_keeps_real_pidl_calibrated_model_training_sentence_link() -> None:
    source_path = (
        "db/NatCommun-2023-High-resolution single-photon imaging with physics-informed "
        "deep learning/NatCommun-2023-High-resolution single-photon imaging with physics-informed "
        "deep learning.en.md"
    )
    plan_evidence = (
        "With the calibrated physical noise model under different illumination and acquisition "
        "settings, we further employed off-the-shelf public highresolution images (collected from "
        "the PASCAL VOC2007 [31] and … VOC2012 [32] datasets) to digitally synthesize a large-scale "
        "realistic singlephoton image dataset containing 2.6 million image pairs. The gated fusion "
        "transformer network was trained using the above large-scale singlephoton image dataset "
        "and tested on various SPAD images."
    )
    answer = (
        "最后，利用校准后的模型和公开的高分辨率图像（如PASCAL VOC2007）合成配对数据，"
        "用于训练深度学习网络进行图像增强 [1]。"
    )
    heading_path = (
        "High-resolution single-photon imaging with physics-informed deep learning / "
        "Introduction"
    )

    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        answer,
        [
            {
                "text": (
                    "wefirst established a real-world physical noise model of SPAD arrays. "
                    "We collected a real-shot SPAD image dataset containing 2790 images in total."
                ),
                "meta": {
                    "source_path": source_path,
                    "source_name": "High-resolution single-photon imaging with PIDL",
                    "heading_path": heading_path,
                    "ref_answer_citation_num": 1,
                },
            }
        ],
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "source_name": "High-resolution single-photon imaging with PIDL",
                    "heading_path": heading_path,
                    "page_start": 3,
                    "page_end": 3,
                    "evidence_quote": plan_evidence,
                }
            ],
        },
        anchor_ns="real-pidl-calibrated-model",
        canonical_paths=[source_path],
        render_locale="zh",
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    evidence = str(details[0].get("card_evidence") or "")
    assert "calibrated physical noise model" in evidence
    assert "PASCAL VOC2007" in evidence
    assert "2.6 million image pairs" in evidence
    assert "network was trained" in evidence


def test_system_a_keeps_real_fdm_bpsk_and_demodulation_occurrence_cards() -> None:
    source_path = (
        "db/Optica-2016-Frequency-division-multiplexed single-pixel imaging with metamaterials/"
        "Optica-2016-Frequency-division-multiplexed single-pixel imaging with metamaterials.en.md"
    )
    plan_evidence = (
        "The mask values are encoded in the phase of intensity modulation, and thus we require "
        "phase-sensitive detection, in this case provided by a lock-in amplifier (LIA). This "
        "mapping of two phases to two numerical (bit) values is known in communications as "
        "binary phase shift keying (BPSK). Each pixel of the SLM is modulated with either 0 or "
        "pi phase on p frequencies simultaneously, according to the present mask patterns. "
        "The modulated light from the SLM is then multiplexed into a single-pixel detector. "
        "The signal is then demodulated by a number (p) of LIAs."
    )
    answer = (
        "它利用超材料 SLM，让每个像素同时对多个不同频率的载波进行二进制相移键控"
        "（BPSK）调制（相位 0 或 π）[1]。这样，多个 Hadamard 掩模的信息被编码到不同"
        "频率的载波上，通过一个单像素探测器接收后，再用锁相放大器（LIA）进行相位敏感"
        "解调，从而同时获取多个掩模对应的信号 [1]。"
    )

    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        answer,
        [
            {
                "text": plan_evidence,
                "meta": {
                    "source_path": source_path,
                    "source_name": "Frequency-division-multiplexed SPI",
                    "heading_path": "B. Encoding",
                    "ref_answer_citation_num": 1,
                },
            }
        ],
        citation_plan={
            "budget": {"system_a": 2, "system_b": 0},
            "per_paragraph_budget": {"system_a": 2, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "source_name": "Frequency-division-multiplexed SPI",
                    "heading_path": "B. Encoding",
                    "page_start": 2,
                    "page_end": 2,
                    "evidence_quote": plan_evidence,
                }
            ],
        },
        anchor_ns="real-fdm-bpsk",
        canonical_paths=[source_path],
        render_locale="zh",
    )

    anchors = re.findall(r"\[1\]\(#([^) \"\n]+)", rendered)
    assert len(anchors) == 2
    detail_anchors = {str(detail.get("anchor") or "") for detail in details}
    assert set(anchors) <= detail_anchors
    assert any("BPSK" in str(detail.get("answer_claim") or "") for detail in details)
    card_evidence = " ".join(str(detail.get("card_evidence") or "") for detail in details)
    assert "binary phase shift keying (BPSK)" in card_evidence
    assert "phase-sensitive detection" in card_evidence
    assert "demodulated by a number (p) of LIAs" in card_evidence


def test_system_a_reading_tip_does_not_replace_substantive_card_claim() -> None:
    source_path = "db/pidl/pidl.en.md"
    evidence = "Deep learning with a calibrated SPAD noise model improves reconstruction quality."
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "The calibrated SPAD noise model improves reconstruction quality [1].\n\n"
            "阅读建议：阅读这篇论文的噪声模型部分 [1]。"
        ),
        [
            {
                "text": evidence,
                "meta": {
                    "source_path": source_path,
                    "heading_path": "Introduction",
                    "evidence_quote": evidence,
                    "page_start": 3,
                },
            }
        ],
        citation_plan={"budget": {"system_a": 2, "system_b": 0}},
        anchor_ns="reading-tip",
        canonical_paths=[source_path],
    )

    assert rendered.count("[1](#kb-cite-") >= 1
    assert len(details) == 1
    assert "improves reconstruction quality" in str(details[0].get("answer_claim") or "")
    assert not str(details[0].get("answer_claim") or "").startswith("阅读建议")


def test_system_a_context_keeps_sentence_before_inline_math_split() -> None:
    source_path = "db/Frontiers-2024-single-photon-detection-review.en.md"
    answer = (
        "探测器综述解释了单光子探测器的波导集成机制，并给出了截止频率 "
        "$f_c$ 等关键参数 [1]。下一句再讨论深度学习方法。"
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        answer,
        [
            {
                "text": (
                    "The waveguide cut-off frequency f_c controls transmission. "
                    "Waveguide integration improves light absorption and detection efficiency."
                ),
                "meta": {
                    "source_path": source_path,
                    "heading_path": "4.2 Waveguide integration",
                },
                "ui_meta": {
                    "primary_evidence": {
                        "heading_path": "Abstract",
                        "snippet": (
                            "Single-photon detectors can detect individual photons at very low light levels."
                        ),
                    }
                },
            }
        ],
        canonical_paths=[source_path],
        citation_plan={"budget": {"system_a": 1, "system_b": 0}},
        anchor_ns="inline-math",
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    detail = details[0]
    assert detail["binding_status"] == "grounded"
    assert "截止频率" in detail["answer_claim"]
    assert "下一句" not in detail["answer_claim"]
    assert detail["evidence_quote"]


def test_system_a_keeps_real_piln_measurement_label_claim_link() -> None:
    source_path = "db/piln/part-based-image-loop-network.en.md"
    locator_snippet = (
        "In this study, we proposed a self-supervised image-loop neural network "
        "(ILNet) with a part-based model for single-pixel imaging (SPI). ILNet "
        "employs a part-based model that divides image features into different "
        "parts to facilitate finer-grained learning."
    )
    full_plan_evidence = (
        locator_snippet
        + " Then, the 2D image generated by ILNet can serve as input for the "
        "subsequent iteration to ensure high-quality imaging at low sampling rates. "
        "1D signals collected by the single-pixel detector are used as labels for "
        "adaptively optimizing and reconstructing the image."
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "1. **无需真实图像标签的自监督重建**：ILNet 不需要成对的"
            "高质量图像作为训练标签，而是利用物理采集的 1D 信号作为"
            "监督信号 [1]。"
        ),
        [
            {
                "text": locator_snippet,
                "meta": {
                    "source_path": source_path,
                    "source_name": "Part-based image-loop network for single-pixel imaging",
                    "heading_path": "Abstract",
                    "page_start": 2,
                    "ref_answer_citation_num": 1,
                },
            }
        ],
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "source_name": "Part-based image-loop network for single-pixel imaging",
                    "heading_path": "Abstract",
                    "page_start": 2,
                    "page_end": 2,
                    "evidence_quote": full_plan_evidence,
                    "evidence_selection_reason": "prompt_aligned_source_sentence",
                }
            ],
        },
        anchor_ns="real-piln-measurement-label",
        canonical_paths=[source_path],
        render_locale="zh",
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    evidence = str(details[0].get("card_evidence") or "")
    assert "1D signals collected by the single-pixel detector" in evidence
    assert "used as labels" in evidence
    assert details[0]["binding_status"] == "grounded"


def test_system_a_does_not_leak_tagged_display_math_citation() -> None:
    source_path = "db/piln/part-based-image-loop-network.en.md"
    evidence = "The ILNet loss compares the real and reconstructed network outputs."
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "损失函数为：\n$$\nL = \\|I(real)-I(out)\\|^2 \\tag{1} [1]\n$$",
        [
            {
                "text": evidence,
                "meta": {
                    "source_path": source_path,
                    "heading_path": "Method / Loss function",
                    "ref_answer_citation_num": 1,
                },
            }
        ],
        canonical_paths=[source_path],
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "heading_path": "Method / Loss function",
                    "evidence_quote": evidence,
                }
            ],
        },
        anchor_ns="display-math-citation",
    )

    assert r"\tag{1} [1]" not in rendered
    assert r"\tag{1}" in rendered
    assert "[1]" not in rendered
    assert details == []


def test_system_a_does_not_move_scientific_brackets_inside_display_math() -> None:
    source_path = "db/math/paper.en.md"
    rendered, _details = _annotate_inpaper_citations_with_hover_meta(
        "$$\nx \\in [0,1] \\tag{2}\n$$\nNo citation.",
        [{"text": "Math definition.", "meta": {"source_path": source_path}}],
        canonical_paths=[source_path],
        citation_plan={"budget": {"system_a": 0, "system_b": 0}},
    )

    assert r"x \in [0,1] \tag{2}" in rendered
