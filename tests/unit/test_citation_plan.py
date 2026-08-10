from __future__ import annotations

from pathlib import Path

import kb.citation_plan as citation_plan
from kb.citation_plan import (
    _prompt_aligned_source_slot,
    _rank_system_a_answer_hits,
    _system_a_slots,
    build_citation_plan,
    build_citation_plan_prompt_block,
    citation_plan_prefers_system_b,
)


def test_source_sentence_records_cache_reuses_and_invalidates_file_versions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "paper.en.md"
    source.write_text(
        "# Paper\n\n<!-- kb_page: 2 -->\n\n## Method\n\nFirst complete evidence sentence.\n",
        encoding="utf-8",
    )
    citation_plan._source_sentence_records_for_signature.cache_clear()
    citation_plan._source_text_for_signature.cache_clear()

    first = citation_plan._source_sentence_records(str(source))
    before = citation_plan._source_sentence_records_for_signature.cache_info()
    repeated = citation_plan._source_sentence_records(str(source))
    after = citation_plan._source_sentence_records_for_signature.cache_info()

    assert repeated == first
    assert after.hits == before.hits + 1

    source.write_text(
        "# Paper\n\n<!-- kb_page: 7 -->\n\n## Results\n\nSecond, longer evidence sentence after repair.\n",
        encoding="utf-8",
    )
    refreshed = citation_plan._source_sentence_records(str(source))

    assert refreshed != first
    assert any(page == 7 and "Second, longer" in text for _heading, text, page in refreshed)


def test_system_a_slot_keeps_complete_targeted_source_block_over_card_excerpt(
    tmp_path: Path,
) -> None:
    source = tmp_path / "qclfm.en.md"
    full_evidence = (
        "We use the inherent position and angular/momentum correlation of entangled "
        "photon pairs. Since each degree of freedom can be measured on separate "
        "cameras, position resolution need not be sacrificed for angular resolution. "
        "This allowed a DOF between 2–5 times larger at 5 μm resolution."
    )
    source.write_text(
        "# QCLFM\n\n<!-- kb_page: 3 -->\n\n## III. DISCUSSION\n\n"
        + full_evidence
        + "\n",
        encoding="utf-8",
    )
    hit = {
        "text": full_evidence,
        "meta": {
            "source_path": str(source),
            "heading_path": "QCLFM / III. DISCUSSION",
            "block_id": "blk-discussion",
            "anchor_id": "p-discussion",
            "page_start": 3,
            "paper_guide_targeted_block": True,
            "primary_evidence": {
                "snippet": full_evidence.split(". ", 1)[0] + ".",
                "block_id": "blk-discussion",
                "anchor_id": "p-discussion",
                "page_start": 3,
            },
        },
    }

    slots = _system_a_slots(
        support_slots=[],
        answer_hits=[hit],
        max_items=1,
        ranking_texts=[
            "QCLFM 为什么能同时保住位置和角度分辨率？论文实际报告的景深提升有多大？"
        ],
    )

    assert len(slots) == 1
    assert "separate cameras" in slots[0]["evidence_quote"]
    assert "2–5 times larger" in slots[0]["evidence_quote"]
    assert "5 μm" in slots[0]["evidence_quote"]
    assert slots[0]["candidate_hits"] == [1]


def test_prompt_alignment_expands_anchored_excerpt_missing_requested_result(
    tmp_path: Path,
) -> None:
    source = tmp_path / "qclfm.en.md"
    first_sentence = (
        "We use the inherent position and angular/momentum correlation of entangled "
        "photon pairs."
    )
    full_evidence = (
        first_sentence
        + " Since each degree of freedom can be measured on separate cameras, position "
        "resolution need not be sacrificed for angular resolution. This allowed a DOF "
        "between 2–5 times larger at 5 μm resolution."
    )
    source.write_text(
        "# QCLFM\n\n<!-- kb_page: 3 -->\n\n## III. DISCUSSION\n\n"
        + full_evidence
        + "\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source),
            "heading_path": "QCLFM / III. DISCUSSION",
            "evidence_quote": first_sentence,
            "block_id": "blk-discussion",
            "anchor_id": "p-discussion",
        },
        ranking_texts=[
            "QCLFM 为什么能同时保住位置和角度分辨率？论文实际报告的景深提升有多大？"
        ],
    )

    assert "separate cameras" in slot["evidence_quote"]
    assert "2–5 times larger" in slot["evidence_quote"]
    assert slot["page_start"] == 3


def test_prompt_alignment_ignores_scoped_source_path_title_noise(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "Quantum correlation light-field microscope with extreme depth of field"
    source_dir.mkdir()
    source = source_dir / "Quantum correlation light-field microscope with extreme depth of field.en.md"
    requested = (
        "We use the inherent position and angular/momentum correlation of entangled "
        "photon pairs. Since each degree of freedom can be measured on separate "
        "cameras, position resolution need not be sacrificed for angular resolution. "
        "This allowed a DOF between 2–5 times larger at 5 μm resolution."
    )
    distractor = (
        "A major limitation to quantum correlation light-field microscope imaging is "
        "slow acquisition speed for the event camera. Detection efficiency is 7%, "
        "timing resolution is 8 ns, and signal-to-noise ratio limits digital refocusing."
    )
    source.write_text(
        "# QCLFM\n\n<!-- kb_page: 3 -->\n\n## Discussion\n\n"
        + requested
        + "\n\n<!-- kb_page: 4 -->\n\n## Limitations\n\n"
        + distractor
        + "\n",
        encoding="utf-8",
    )
    question = "QCLFM 为什么能同时保住位置和角度分辨率？论文实际报告的景深提升有多大？"

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source),
            "heading_path": "QCLFM / Limitations",
            "evidence_quote": distractor,
            "block_id": "blk-limit",
            "anchor_id": "p-limit",
            "page_start": 4,
        },
        ranking_texts=[
            question,
            f"{source} {question}",
            (
                f"{question}\nQUERY SCOPE: Current paper. Use this paper only. "
                "structured detection interferometric detection light-field microscopy "
                "optical sectioning super-resolution signal-to-noise ratio SNR "
                "digital refocusing"
            ),
            f"{source} {question}",
            "motivation",
        ],
    )

    assert "separate cameras" in slot["evidence_quote"]
    assert "2–5 times larger" in slot["evidence_quote"]
    assert slot["page_start"] == 3


def test_prompt_alignment_replaces_metric_experiment_with_parameter_enumeration(
    tmp_path: Path,
) -> None:
    source = tmp_path / "detector-review.en.md"
    enumeration = (
        "The main parameters of single photon detectors are detection efficiency "
        "(DE), dark count, system dead time, time jitter, and so on."
    )
    distractor = (
        "A waveguide SPAD reached 85% detection efficiency at 78 K, and timing "
        "jitter varied with excess bias."
    )
    source.write_text(
        "# Review\n\n<!-- kb_page: 10 -->\n\n"
        "## 3 Single photon detection parameter\n\n"
        + enumeration
        + "\n\n<!-- kb_page: 14 -->\n\n## 4.2 Waveguide\n\n"
        + distractor
        + "\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source),
            "heading_path": "Review / 4.2 Waveguide",
            "evidence_quote": distractor,
            "block_id": "blk-waveguide",
            "anchor_id": "p-waveguide",
            "page_start": 14,
        },
        ranking_texts=[
            "评价单光子探测器时，除了探测效率还必须看哪些关键指标？请按原文列出。"
        ],
    )

    assert slot["evidence_quote"] == enumeration
    assert slot["page_start"] == 10


def test_hsi_fsi_source_focus_updates_stale_retrieval_page(tmp_path: Path) -> None:
    source = tmp_path / "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md"
    source.write_text(
        "# HSI versus FSI\n\n<!-- kb_page: 3 -->\n\n## Introduction\n\n"
        "HSI uses Hadamard basis patterns for illumination while FSI uses Fourier basis patterns. "
        "In this paper, we theoretically and experimentally compare HSI and FSI in terms of "
        "principles, imaging efficiency, and noise robustness.\n\n"
        "<!-- kb_page: 13 -->\n\n## Results\n\nLater measurements.\n",
        encoding="utf-8",
    )

    slots = _system_a_slots(
        support_slots=None,
        answer_hits=[
            {
                "text": "Later measurements.",
                "meta": {
                    "source_path": str(source),
                    "heading_path": "Results",
                    "page_start": 13,
                    "page_end": 13,
                },
            }
        ],
        max_items=1,
        focus_multi_source_evidence=True,
    )

    assert slots[0]["heading_path"].endswith("Introduction")
    assert slots[0]["page_start"] == 3
    assert slots[0]["page_end"] == 3


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


def test_prompt_aligned_source_prefers_complete_dense_sentence_over_equal_long_paragraph(
    tmp_path: Path,
) -> None:
    source = tmp_path / "structured-detection.en.md"
    broad_prefix = " ".join(
        "Background survey material discusses structured detection, optical sectioning, "
        "signal-to-noise ratio, and thick samples without stating the paper's result."
        for _ in range(16)
    )
    direct = (
        "The s2ISM structured-detection method simultaneously restores optical sectioning "
        "and signal-to-noise ratio for thick samples."
    )
    source.write_text(
        f"# Paper\n\n## Abstract\n\n{broad_prefix} {direct}\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source),
            "evidence_quote": "Generic detector-array background.",
        },
        ranking_texts=[
            "s2ISM structured detection optical sectioning signal-to-noise ratio thick samples"
        ],
        prefer_source_summary=True,
    )

    assert slot["evidence_quote"] == direct
    assert "Background survey material" not in slot["evidence_quote"]
    assert not slot["evidence_quote"].endswith("...")


def test_prompt_aligned_source_extracts_late_structured_detection_compound_claim(
    tmp_path: Path,
) -> None:
    filler = " ".join(
        "Background discussion covers detector arrays and conventional microscopy."
        for _ in range(24)
    )
    source = tmp_path / "s2ism.en.md"
    source.write_text(
        "# Structured detection\n\n<!-- kb_page: 1 -->\n\n## Abstract\n\n"
        "From single-plane acquisition, we reconstruct an image with digital and optical "
        "super-resolution, high signal-to-noise ratio and enhanced optical sectioning.\n\n"
        "<!-- kb_page: 2 -->\n\n"
        "Structured detection can leverage axial information for enhanced resolution and sectioning. "
        f"{filler} "
        "Since super-resolution and optical sectioning are achieved simultaneously, "
        "we named our technique s2ISM.\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source),
            "evidence_quote": "Generic detector-array background.",
        },
        ranking_texts=[
            "structured detection s2ISM super-resolution optical sectioning"
        ],
        prefer_source_summary=True,
    )

    assert "Structured detection can leverage" in slot["evidence_quote"]
    assert "super-resolution and optical sectioning are achieved simultaneously" in slot[
        "evidence_quote"
    ]
    assert "high signal-to-noise ratio" in slot["evidence_quote"]
    assert "s2ISM" in slot["evidence_quote"]
    assert "Background discussion" not in slot["evidence_quote"]
    assert slot["page_start"] == 1
    assert slot["page_end"] == 2


def test_prompt_aligned_source_keeps_complete_spad_geiger_quenching_chain(
    tmp_path: Path,
) -> None:
    source = tmp_path / "spad-review.en.md"
    source.write_text(
        "# Detector review\n\n<!-- kb_page: 2 -->\n\n"
        "## Principle of single photon detection avalanche diode\n\n"
        "Single photon avalanche diode (SPAD) is a p-n junction that operates in Geiger mode. "
        "The device operates with a bias voltage significantly higher than its reverse bias "
        "breakdown voltage. "
        + "Generic material-development background. " * 45
        + "When the SPAD operates in Geiger mode, excessive induced current will damage the "
        "device's performance. To optimize the avalanche diode, it must be supported by the "
        "quenching circuit.\n\n<!-- kb_page: 3 -->\n\n"
        "The quenching circuit extracts a digital pulse signal upon detecting avalanche current "
        "and subsequently quench the current by applying an extra reverse bias.\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source),
            "evidence_quote": "Generic SPAD background.",
        },
        ranking_texts=[
            "SPAD 为什么工作在 Geiger 模式，为什么高于击穿电压，并需要淬灭电路？"
        ],
    )

    evidence = slot["evidence_quote"]
    assert "operates in Geiger mode" in evidence
    assert "reverse bias breakdown voltage" in evidence
    assert "quenching circuit" in evidence
    assert "extra reverse bias" in evidence
    assert "Generic material-development background" not in evidence
    assert not evidence.endswith("...")


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
    assert "This comparison has 3 planned SystemA sources" in prompt_block
    assert "Cover all 3 explicitly and do not stop after a subset" in prompt_block
    assert "two-sided comparison" not in prompt_block
    assert "Do not introduce a third paper" not in prompt_block
    assert "Do not introduce unplanned papers" in prompt_block
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


def test_single_paper_improvements_and_limitations_prefers_richer_challenges(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / (
        "LPR-2025-Advances and Challenges of Single‐Pixel Imaging Based on Deep Learning.en.md"
    )
    source_path.write_text(
        "<!-- kb_page: 1 -->\n\n## Abstract\n\n"
        "Single-pixel imaging based on deep learning has exceptional reconstruction "
        "quality and fast reconstruction speed.\n\n"
        "<!-- kb_page: 16 -->\n\n## 6. Challenges and Outlooks\n\n"
        "The inherent limitations include reliance on extensive datasets, limited "
        "interpretability, susceptibility to overfitting, and limited generalization.\n",
        encoding="utf-8",
    )
    unrelated = tmp_path / "generic-single-pixel-review.en.md"
    unrelated.write_text("## Abstract\n\nA generic review.\n", encoding="utf-8")

    plan = build_citation_plan(
        prompt=(
            "What practical improvements does deep learning bring to single-pixel imaging, "
            "and what limitations should I keep in mind before using it?"
        ),
        prompt_family="strength_limits",
        answer_hits=[
            {
                "text": "A generic single-pixel imaging overview.",
                "meta": {"source_path": str(unrelated), "heading_path": "Abstract"},
            },
            {
                "text": "Single-pixel imaging based on deep learning has exceptional "
                "reconstruction quality and fast reconstruction speed.",
                "meta": {"source_path": str(source_path), "heading_path": "Abstract"},
            },
        ],
    )

    system_a = [
        slot for slot in plan["slots"] if slot["preferred_system"] == "system_a"
    ]
    assert plan["intent"] == "comparison"
    assert len(system_a) == 2
    assert {slot["source_path"] for slot in system_a} == {str(source_path)}
    assert any("reconstruction speed" in slot["evidence_quote"] for slot in system_a)
    challenge = next(
        slot for slot in system_a if "limited interpretability" in slot["evidence_quote"]
    )
    assert challenge["heading_path"].endswith("6. Challenges and Outlooks")
    assert challenge["page_start"] == 16


def test_scinerf_formula_question_pins_equation_roles_and_training_link(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "CVPR-2024-SCINeRF.en.md"
    source_path.write_text(
        "# SCINeRF\n\n<!-- kb_page: 1 -->\n\n## Abstract\n\n"
        "SCINeRF recovers a 3D scene from one compressed image.\n\n"
        "<!-- kb_page: 4 -->\n\n### 3.2. Image Formation Model of Video SCI\n\n"
        "$$\\mathbf{Y} = \\sum_{i=1}^{N} \\mathbf{X}_i \\odot \\mathbf{M}_i + "
        "\\mathbf{Z}.$$ \n\n"
        "Y is the captured compressed image, Xi is a virtual image, odot denotes "
        "element-wise multiplication, and Z is the measurement noise.\n\n"
        "We render Xi to synthesize the compressed image Y, which is differentiable "
        "with respect to NeRF and the poses.\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source_path),
            "evidence_quote": "SCINeRF recovers a 3D scene from one compressed image.",
            "heading_path": "Abstract",
            "page_start": 1,
        },
        ranking_texts=[
            "SCINeRF SCI forward image formation equation binary masks measurement noise "
            "differentiable NeRF poses"
        ],
    )

    assert slot["heading_path"].endswith("3.2. Image Formation Model of Video SCI")
    assert slot["page_start"] == 4
    assert "\\mathbf{Y}" in slot["evidence_quote"]
    assert "element-wise multiplication" in slot["evidence_quote"]
    assert "differentiable with respect to NeRF and the poses" in slot["evidence_quote"]

    full_evidence = slot["evidence_quote"]
    plan = build_citation_plan(
        prompt=(
            "SCINeRF 的 SCI 前向成像公式表达了什么？请解释二值掩模、噪声和"
            "可微联合优化。"
        ),
        prompt_family="method",
        support_slots=[
            {
                "source_path": str(source_path),
                "heading_path": "Abstract",
                "evidence_quote": "SCINeRF recovers a 3D scene from one compressed image.",
            }
        ],
        answer_hits=[
            {
                "text": "SCINeRF recovers a 3D scene from one compressed image.",
                "meta": {"source_path": str(source_path), "heading_path": "Abstract"},
            },
            {
                "text": "The captured image is modulated by N binary masks.",
                "meta": {
                    "source_path": str(source_path),
                    "heading_path": "3.2. Image Formation Model of Video SCI",
                },
            },
            {
                "text": full_evidence,
                "meta": {
                    "source_path": str(source_path),
                    "heading_path": "3.2. Image Formation Model of Video SCI",
                },
            },
        ],
    )
    formula_slot = next(
        item
        for item in plan["slots"]
        if "element-wise multiplication" in item["evidence_quote"]
    )
    assert formula_slot["candidate_hits"] == [3]


def test_both_each_method_prompt_is_a_two_source_comparison() -> None:
    hits = [
        {
            "text": (
                "Each SLM pixel is modulated on p frequencies simultaneously; the signal "
                "uses phase-sensitive detection and is demodulated by p lock-in amplifiers."
            ),
            "meta": {
                "source_path": "frequency-division-multiplexed-spi.en.md",
                "heading_path": "B. Encoding",
            },
        },
        {
            "text": (
                "Photometric stereo uses four spatially-separated detectors and reconstructs "
                "3D video at 8 frames per second."
            ),
            "meta": {
                "source_path": "3d-single-pixel-video.en.md",
                "heading_path": "Abstract",
            },
        },
        {
            "text": "HSI uses Hadamard patterns and FSI uses Fourier patterns.",
            "meta": {
                "source_path": "hadamard-versus-fourier.en.md",
                "heading_path": "Introduction",
            },
        },
    ]

    plan = build_citation_plan(
        prompt=(
            "Both frequency-division-multiplexed single-pixel imaging and 3D single-pixel "
            "video claim speedups. What does each method parallelize?"
        ),
        answer_hits=hits,
    )

    system_a = [
        slot for slot in plan["slots"] if slot["preferred_system"] == "system_a"
    ]
    assert plan["intent"] == "comparison"
    assert len(system_a) == 2
    assert {slot["source_path"] for slot in system_a} == {
        "frequency-division-multiplexed-spi.en.md",
        "3d-single-pixel-video.en.md",
    }


def test_prompt_alignment_skips_bibliography_without_references_heading(tmp_path: Path) -> None:
    source_path = tmp_path / "perovskite.en.md"
    abstract = (
        "We demonstrate electrically driven lasing from a dual-cavity perovskite device "
        "that integrates a low-threshold crystal microcavity with a high-power PeLED."
    )
    source_path.write_text(
        "# Paper\n\n## Abstract\n\n"
        f"{abstract}\n\n## Conclusion\n\n"
        "1. Smith, J., Jones, A. Electrically driven lasers. Nature 620, 100-110 (2023).\n\n"
        "2. Brown, T., Green, R. Dual cavity emitters. Science 381, 20-28 (2022).\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source_path),
            "evidence_quote": "A generic device overview.",
        },
        ranking_texts=["dual-cavity electrically driven perovskite lasing device"],
    )

    assert "dual-cavity perovskite device" in slot["evidence_quote"]
    assert "Smith" not in slot["evidence_quote"]
    assert slot["heading_path"] == "Paper / Abstract"


def test_perovskite_scope_alignment_prefers_abstract_over_result_metrics(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "Electrically driven lasing from a dual-cavity perovskite device.en.md"
    abstract = (
        "We demonstrate an electrically driven perovskite laser based on a dual-cavity "
        "perovskite device with a low lasing threshold."
    )
    conclusion = (
        "The dual-cavity perovskite device has a minimum lasing threshold of 92 A cm-2, "
        "lower than the integrated single-cavity device."
    )
    source_path.write_text(
        "<!-- kb_page: 1 -->\n\n## Abstract\n\n"
        + abstract
        + "\n\n<!-- kb_page: 5 -->\n\n## Conclusion\n\n"
        + conclusion
        + "\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source_path),
            "heading_path": "Conclusion",
            "evidence_quote": conclusion,
            "page_start": 5,
        },
        ranking_texts=["这篇 perovskite laser 和单像素成像主线关系大吗？"],
    )

    assert "electrically driven perovskite laser" in slot["evidence_quote"]
    assert slot["heading_path"].endswith("Abstract")
    assert slot["page_start"] == 1


def test_perovskite_scope_plan_keeps_abstract_hit_locator(tmp_path: Path) -> None:
    source_path = tmp_path / "Electrically driven lasing from a dual-cavity perovskite device.en.md"
    abstract = (
        "We demonstrate an electrically driven perovskite laser based on a dual-cavity "
        "perovskite device with a low lasing threshold."
    )
    conclusion = (
        "The dual-cavity perovskite device has a minimum lasing threshold of 92 A cm-2, "
        "lower than the integrated single-cavity device."
    )
    source_path.write_text(
        "<!-- kb_page: 1 -->\n\n## Abstract\n\n"
        + abstract
        + "\n\n<!-- kb_page: 5 -->\n\n## Conclusion\n\n"
        + conclusion
        + "\n",
        encoding="utf-8",
    )
    hits = [
        {
            "text": abstract,
            "meta": {
                "source_path": str(source_path),
                "heading_path": "Abstract",
                "block_id": "abstract-block",
                "anchor_id": "abstract-paragraph",
            },
        },
        {
            "text": conclusion,
            "meta": {
                "source_path": str(source_path),
                "heading_path": "Conclusion",
                "block_id": "conclusion-block",
                "anchor_id": "conclusion-paragraph",
            },
        },
    ]

    plan = build_citation_plan(
        prompt="这篇 perovskite laser 和我的单像素成像主线关系大吗？值得一起读吗？",
        prompt_family="overview",
        answer_hits=hits,
    )

    slot = next(item for item in plan["slots"] if item["preferred_system"] == "system_a")
    assert slot["candidate_hits"] == [1]
    assert slot["heading_path"].endswith("Abstract")
    assert "electrically driven perovskite laser" in slot["evidence_quote"]


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


def test_piln_physical_loop_question_keeps_method_and_transfer_bundle(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "Part-based image-loop network for single-pixel imaging.en.md"
    source_path.write_text(
        "<!-- kb_page: 2 -->\n\n## Abstract\n\n"
        "The generated image serves as input for the subsequent iteration for continuous incorporation of prior information. "
        "Signals collected by the single-pixel detector are used as labels. "
        "The method reconstructs unknown free-space and underwater experiments.\n\n"
        "## 2.1. Methods\n\n"
        "The part-based model divides image features into different parts. "
        "The difference between the $I_N(out)$ image signal and $I_N(real)$ captured by the SPD is used as a loss function. "
        "The generated image is used as input for the subsequent iterations. "
        "Providing prior information improves the final image.\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source_path),
            "heading_path": "2.1. Methods",
            "evidence_quote": "Broad ILNet description.",
        },
        ranking_texts=[
            "ILNet 如何用 part-based 特征、I_N(out) 与 I_N(real) 损失和迭代形成物理闭环，"
            "并迁移到自由空间和水下？"
        ],
    )

    evidence = slot["evidence_quote"]
    assert "divides image features into different parts" in evidence
    assert "I_N(out)" in evidence and "I_N(real)" in evidence
    assert "input for the subsequent iterations" in evidence
    assert "continuous incorporation of prior information" in evidence
    assert "unknown free-space and underwater" in evidence


def test_prompt_aligned_slot_keeps_daq_channel_budget_not_video_abstract(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "3D single-pixel video.en.md"
    source_path.write_text(
        "<!-- kb_page: 1 -->\n\n## Abstract\n\n"
        "Four detectors reconstruct real-time 3D video at eight frames per second.\n\n"
        "<!-- kb_page: 4 -->\n\n## Methods / Custom single-pixel system design\n\n"
        "The DAQ has a maximum acquisition rate of 250 kHz for all channels. "
        "As there are four channels employed, the sampling rate for each channel is set to 62.5 kHz. "
        "Given that each pattern is displayed for 50 μs, there are approximately three samples acquired for each pattern.\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source_path),
            "heading_path": "Abstract",
            "evidence_quote": "Four detectors reconstruct real-time video.",
        },
        ranking_texts=[
            "3D single-pixel video 的 DAQ 总采样率如何分给四个通道？"
            "50 μs 显示时间每个图案得到多少样本？"
        ],
    )

    assert "Custom single-pixel system design" in slot["heading_path"]
    assert "250 kHz for all channels" in slot["evidence_quote"]
    assert "approximately three samples" in slot["evidence_quote"]


def test_prompt_aligned_slot_bundles_fdm_detector_boundary(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "Frequency-division-multiplexed single-pixel imaging.en.md"
    source_path.write_text(
        "<!-- kb_page: 1 -->\n\n## Abstract\n\n"
        "FDM trades signal-to-noise ratio for acquisition speed without altering detector integration time.\n\n"
        "<!-- kb_page: 3 -->\n\n## 4. DISCUSSION\n\n"
        "If we assume that the primary source of noise is additive white Gaussian (AWG), the SNR is proportional to the square root of the integration time. "
        "Detector integration time cannot be reduced without bound. "
        "All detectors have inherent limits, typically characterized by the 3 dB down point. "
        "At frequencies greater than f_{3 dB}, the noise increases and it is no longer advantageous to trade off SNR for integration time. "
        "Our FDM scheme sacrifices some SNR for acquisition speed without lowering the integration time and does not suffer from the same fundamental limitation. "
        "If noise is not AWG, there may be a characteristic time for optimal SNR. "
        "FDM decreases acquisition time without deviation from such an optimal integration time.\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source_path),
            "heading_path": "Abstract",
            "evidence_quote": "FDM is faster.",
        },
        ranking_texts=[
            "FDM 与缩短积分时间在 AWG 下哪里相似，3 dB 边界为何失效，"
            "非 AWG 的 characteristic time for optimal SNR 有何影响？"
        ],
    )

    assert slot["heading_path"].endswith("4. DISCUSSION")
    assert "square root of the integration time" in slot["evidence_quote"]
    assert "3 dB down point" in slot["evidence_quote"]
    assert "noise is not AWG" in slot["evidence_quote"]
    assert "characteristic time for optimal SNR" in slot["evidence_quote"]
    assert "without lowering the integration time" in slot["evidence_quote"]
    assert "without deviation from such an optimal integration time" in slot["evidence_quote"]
    assert not slot["evidence_quote"].endswith("...")


def test_fdm_boundary_prefers_one_complete_anchored_duplicate_block(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "Frequency-division-multiplexed single-pixel imaging.en.md"
    boundary = (
        "If we assume that the primary source of noise is additive white Gaussian (AWG), "
        "the SNR is proportional to the square root of the integration time. "
        "Detector integration time cannot be reduced without bound. "
        "All detectors have inherent limits, typically characterized by the 3 dB down point. "
        "At frequencies greater than f 3 dB it is no longer advantageous to trade off SNR for integration time. "
        "Our FDM scheme sacrifices SNR without lowering the integration time and avoids that fundamental limitation. "
        "When noise is not AWG, there may be a characteristic time for optimal SNR. "
        "FDM decreases acquisition time without deviation from such an optimal integration time."
    )
    source_path.write_text(
        "<!-- kb_page: 3 -->\n\n## 4. DISCUSSION\n\n"
        "If we assume that the primary source of noise is additive white Gaussian (AWG), "
        "the SNR is proportional to the square root of the integration time.\n\n"
        "Detector integration time cannot be reduced without bound. All detectors have inherent limits, "
        "typically characterized by the 3 dB down point. At frequencies greater than f_{3 dB} it is no longer advantageous. "
        "Our FDM scheme works without lowering the integration time and avoids that fundamental limitation.\n\n"
        "When noise is not AWG, there may be a characteristic time for optimal SNR. "
        "FDM decreases acquisition time without deviation from such an optimal integration time.\n\n"
        "<!-- kb_page: 4 -->\n\n## 4. DISCUSSION\n\n"
        + boundary
        + "\n",
        encoding="utf-8",
    )

    plan = build_citation_plan(
        prompt="FDM 在 AWG 下的相似处、3 dB 边界和非 AWG 特征时间是什么？",
        answer_hits=[
            {
                "text": boundary,
                "meta": {
                    "source_path": str(source_path),
                    "heading_path": "4. DISCUSSION",
                    "block_id": "blk-fdm-page-4",
                    "anchor_id": "p-fdm-page-4",
                    "page_start": 4,
                    "page_end": 4,
                    "paper_guide_targeted_block": True,
                },
            }
        ],
    )

    slot = plan["slots"][0]
    assert slot["page_start"] == 4
    assert slot["block_id"] == "blk-fdm-page-4"
    assert slot["anchor_id"] == "p-fdm-page-4"
    assert "characteristic time for optimal SNR" in slot["evidence_quote"]


def test_prompt_aligned_slot_bundles_complete_sph_sampling_conditions(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "Imaging biological tissue with high-throughput single-pixel compressive holography.en.md"
    source_path.write_text(
        "<!-- kb_page: 10 -->\n\n## Methods / Principle of high-throughput SPH\n\n"
        "**Experimental setup.** The experimental setup is schematically shown in Fig. 7. "
        "Thus, the beat frequency of these two beams is 62,500 Hz, indicating a temporal period of 16 μs. "
        "The detector signal was digitized with a sampling rate of 1.25 Ms/s. "
        "Considering the 48-μs refresh time, three beating cycles last for each Hadamard pattern and 20 data points were acquired within one cycle. "
        "For the same number of data points, signal quality is insensitive to beat frequency provided the Nyquist sampling criterion was followed. "
        "An integer number of beating cycles for each displayed pattern is also desired.\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source_path),
            "heading_path": "Figure 7",
            "evidence_quote": "AOMs produce a beating frequency.",
        },
        ranking_texts=[
            "SPH 中 62.5 kHz 拍频、1.25 Ms/s 采样和 48 μs 图案周期怎样配合？"
        ],
    )

    evidence = slot["evidence_quote"]
    assert slot["page_start"] == 10
    assert slot["heading_path"].endswith("Experimental setup")
    assert "Experimental setup" in evidence
    assert "62,500 Hz" in evidence and "1.25 Ms/s" in evidence
    assert "three beating cycles" in evidence and "20 data points" in evidence
    assert "Nyquist sampling criterion" in evidence
    assert "integer number of beating cycles" in evidence

    plan = build_citation_plan(
        prompt=(
            "SPH 中 62.5 kHz 拍频、1.25 Ms/s 采样和 48 μs 图案周期怎样配合？"
        ),
        answer_hits=[
            {
                "text": evidence,
                "meta": {
                    "source_path": str(source_path),
                    "heading_path": "Methods / Principle of high-throughput SPH",
                    "block_id": "blk-sph-experiment",
                    "anchor_id": "p-sph-experiment",
                    "page_start": 10,
                    "paper_guide_targeted_block": True,
                },
            }
        ],
    )
    assert plan["slots"][0]["block_id"] == "blk-sph-experiment"
    assert plan["slots"][0]["anchor_id"] == "p-sph-experiment"
    assert plan["slots"][0]["strict_locate"] is True


def test_prompt_aligned_slot_bundles_complete_iism_depth_phase_relation(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "Interferometric Image Scanning Microscopy.en.md"
    source_path.write_text(
        "<!-- kb_page: 2 -->\n\n## Results / Principle of interferometric ISM (iISM)\n\n"
        "In a confocal geometry, the relative phase between reflected and scattered "
        "electric fields is:\n\n"
        "$$ \\Delta\\varphi = \\frac{4\\pi}{\\lambda} n z + "
        "\\varphi_{\\text{Gouy}} \\tag{2} $$\n\n"
        "with n the refractive index of the medium, z the axial position of the "
        "scatterer relative to the interface, lambda the illumination wavelength, "
        "and phi_Gouy the Gouy phase.\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source_path),
            "heading_path": "Abstract",
            "evidence_quote": "The phase carries depth.",
        },
        ranking_texts=[
            "iISM 的相位为何携带深度？z、n、lambda 与 Gouy phase 分别是什么？"
        ],
    )

    evidence = slot["evidence_quote"]
    assert slot["page_start"] == 2
    assert "relative phase between reflected and scattered electric fields" in evidence
    assert "refractive index of the medium" in evidence
    assert "axial position of the scatterer" in evidence
    assert "illumination wavelength" in evidence
    assert "Gouy phase" in evidence


def test_prompt_aligned_slot_prefers_iism_abstract_for_live_cell_benefit(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "Interferometric Image Scanning Microscopy.en.md"
    abstract = (
        "This technique combines interferometric detection with image scanning microscopy "
        "to achieve about 120 nm lateral resolution while operating at tenfold lower "
        "incident illumination power per diffraction limited spot, significantly reducing "
        "photodamage while enhancing signal-to-noise and contrast."
    )
    discussion = (
        "By combining interferometric detection with a modified APR algorithm based on RVT, "
        "we realized a lateral resolution of about 120 nm and maintained confocal resolution "
        "with 0.5 μW incident power, about 10 times lower than previously reported."
    )
    source_path.write_text(
        "<!-- kb_page: 1 -->\n\n## Abstract\n\n"
        + abstract
        + "\n\n<!-- kb_page: 7 -->\n\n## Results / Live cells\n\n"
        + discussion
        + "\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source_path),
            "heading_path": "Results / Live cells",
            "evidence_quote": discussion,
            "page_start": 7,
        },
        ranking_texts=[
            "iISM 在活细胞里同时改善了什么？120 nm 分辨率是用什么代价换来的？"
        ],
    )

    assert slot["page_start"] == 1
    assert slot["heading_path"].endswith("Abstract")
    assert slot["evidence_quote"] == abstract

    plan = build_citation_plan(
        prompt="iISM 在活细胞里同时改善了什么？120 nm 分辨率是用什么代价换来的？",
        answer_hits=[
            {
                "text": discussion,
                "meta": {
                    "source_path": str(source_path),
                    "heading_path": "Results / Live cells",
                    "page_start": 7,
                    "block_id": "discussion-block",
                    "paper_guide_targeted_block": True,
                },
            },
            {
                "text": abstract,
                "meta": {
                    "source_path": str(source_path),
                    "heading_path": "Abstract",
                    "page_start": 1,
                },
            },
        ],
    )
    system_a_slots = [
        item for item in plan["slots"] if item["preferred_system"] == "system_a"
    ]
    assert len(system_a_slots) == 1
    assert system_a_slots[0]["page_start"] == 1
    assert system_a_slots[0]["heading_path"].endswith("Abstract")


def test_prompt_aligned_slot_bundles_sequential_two_stage_main_result(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "Sequentially designed compressed sensing.en.md"
    source_path.write_text(
        "<!-- kb_page: 1 -->\n\n## Abstract\n\n"
        "Sequential adaptive compressed sensing recovers weaker sparse signals.\n\n"
        "## II. MAIN RESULT\n\n"
        "The algorithm consists of two stages. The first stage involves $\\log_2 \\log n$ steps. "
        "The measurements remove half of the zero components while all the non-zero components are retained. "
        "The expected number remaining is bounded by $n / \\log n + k$. "
        "The second stage faces a lower dimensional problem and uses only $k \\log n$ additional measurements. "
        "The support can be recovered exactly at much lower SNRs than non-adaptive compressed sensing.\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source_path),
            "heading_path": "Abstract",
            "evidence_quote": "Sequential adaptive compressed sensing recovers weaker sparse signals.",
        },
        ranking_texts=[
            "Sequential Compressed Sensing 的两阶段分别做什么？说明剩余维度、额外测量和 lower SNR。"
        ],
    )

    evidence = slot["evidence_quote"]
    assert slot["heading_path"].endswith("II. MAIN RESULT")
    assert "two stages" in evidence and "\\log_2 \\log n" in evidence
    assert "remove half of the zero components" in evidence
    assert "n / \\log n + k" in evidence
    assert "k \\log n" in evidence and "additional measurements" in evidence
    assert "much lower SNRs" in evidence


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


def test_physics_informed_single_photon_focus_uses_introduction_noise_model(
    tmp_path: Path,
) -> None:
    pidl = tmp_path / "High-resolution single-photon imaging with physics-informed deep learning.en.md"
    pidl.write_text(
        "<!-- kb_page: 2 -->\n\n# High-resolution single-photon imaging with physics-informed deep learning\n\n"
        "## Introduction\n\nBackground motivation.\n\n"
        "<!-- kb_page: 3 -->\n\n"
        "We first established a real-world physical noise model of SPAD arrays. "
        "The real physical noise sources include shot noise, dark count rate, afterpulsing, "
        "crosstalk noise, and deadtime noise. "
        "To calibrate the model, we collected 2790 images from 90 scenes, each with "
        "10 different bit depths and 3 different illumination fluxes.\n\n"
        "<!-- kb_page: 8 -->\n\n## Discussion\n\nThe technique adapts to several imaging modalities.\n",
        encoding="utf-8",
    )
    detector = tmp_path / "Emerging single-photon detection technique for high-performance photodetector.en.md"
    detector.write_text("# Detector review\n\n## Introduction\n\nA review of SPAD materials.\n", encoding="utf-8")

    plan = build_citation_plan(
        prompt="physics-informed deep learning 在单光子成像里到底帮了什么？",
        answer_hits=[
            {"text": "A review of SPAD materials.", "meta": {"source_path": str(detector)}},
            {"text": "The technique adapts to several modalities.", "meta": {"source_path": str(pidl)}},
        ],
    )

    system_a = [slot for slot in plan["slots"] if slot["preferred_system"] == "system_a"]
    assert len(system_a) == 1
    focus = next(slot for slot in system_a if slot["source_path"] == str(pidl))
    assert focus["heading_path"].endswith("Introduction")
    assert focus["page_start"] == 3
    assert "physical noise model of SPAD arrays" in focus["evidence_quote"]
    assert "2790 images from 90 scenes" in focus["evidence_quote"]


def test_physics_informed_role_plan_keeps_noise_and_training_evidence_in_one_source_slot(
    tmp_path: Path,
) -> None:
    pidl = tmp_path / "High-resolution single-photon imaging with physics-informed deep learning.en.md"
    pidl.write_text(
        "# High-resolution single-photon imaging with physics-informed deep learning\n\n"
        "## Introduction\n\n<!-- kb_page: 3 -->\n\n"
        "We first established a real-world physical noise model of SPAD arrays. "
        "The real physical noise sources consist of shot noise, fixed-pattern noise, "
        "dark count rate, afterpulsing and crosstalk noise, and deadtime noise. "
        "We collected a real-shot SPAD image dataset containing 2790 images in total, "
        "each with 64 x 32 pixels. Among these images, there are 90 scenes, each with "
        "10 different bit depths and 3 different illumination fluxes. "
        "With the calibrated physical noise model, we employed public highresolution "
        "images collected from the PASCAL VOC2007 [31] and\n\n"
        "*an intervening figure caption*\n\n"
        "VOC2012 [32] datasets) to digitally synthesize a realistic singlephoton image "
        "dataset containing 2.6 million image pairs. The gated fusion transformer network "
        "was trained using the above large-scale singlephoton image dataset.\n",
        encoding="utf-8",
    )

    plan = build_citation_plan(
        prompt="physics-informed deep learning 在单光子成像里到底帮了什么？",
        answer_hits=[
            {
                "text": "The technique supports SPAD reconstruction.",
                "meta": {"source_path": str(pidl)},
            }
        ],
    )

    system_a = [slot for slot in plan["slots"] if slot["preferred_system"] == "system_a"]
    assert len(system_a) == 1
    assert {slot["source_path"] for slot in system_a} == {str(pidl)}
    focus = system_a[0]
    assert focus["candidate_hits"] == [1]
    assert focus["page_start"] == 3
    assert "physical noise model of SPAD arrays" in focus["evidence_quote"]
    assert "PASCAL VOC2007" in focus["evidence_quote"]
    assert "digitally synthesize" in focus["evidence_quote"]
    assert "image pairs" in focus["evidence_quote"]
    assert "network was trained" in focus["evidence_quote"]
    prompt_block = build_citation_plan_prompt_block(plan)
    assert "PASCAL images to synthesize paired data" in prompt_block
    assert "Do not claim that it replaces a black box" in prompt_block
    assert "scene changes" in prompt_block


def test_single_photon_reading_pair_keeps_detector_review_and_pidl_introduction(
    tmp_path: Path,
) -> None:
    detector = tmp_path / "Emerging single-photon detection technique for high-performance photodetector.en.md"
    detector.write_text("# Detector review\n\n## Introduction\n\nSi-SPAD detectors have characteristic noise sources.\n", encoding="utf-8")
    pidl = tmp_path / "High-resolution single-photon imaging with physics-informed deep learning.en.md"
    pidl.write_text(
        "# High-resolution single-photon imaging with physics-informed deep learning\n\n"
        "## Introduction\n\n<!-- kb_page: 3 -->\n\n"
        "We first established a real-world physical noise model of SPAD arrays. "
        "The real physical noise sources include dark count rate and crosstalk noise. "
        "We collected 2790 images from 90 scenes at 10 different bit depths and "
        "3 different illumination fluxes.\n",
        encoding="utf-8",
    )

    plan = build_citation_plan(
        prompt="单光子成像里，探测器综述和 physics-informed deep learning 这篇应该怎么搭配读？",
        answer_hits=[
            {"text": "Si-SPAD detector review.", "meta": {"source_path": str(detector)}},
            {"text": "Physics-informed SPAD reconstruction.", "meta": {"source_path": str(pidl)}},
        ],
    )

    system_a = [slot for slot in plan["slots"] if slot["preferred_system"] == "system_a"]
    assert len(system_a) == 2
    assert [slot["source_path"] for slot in system_a] == [str(detector), str(pidl)]
    assert system_a[1]["page_start"] == 3
    assert "2790 images" in system_a[1]["evidence_quote"]


def test_physics_informed_spad_explicit_discussion_request_is_not_forced_to_introduction(
    tmp_path: Path,
) -> None:
    pidl = tmp_path / "High-resolution single-photon imaging with physics-informed deep learning.en.md"
    pidl.write_text(
        "# High-resolution single-photon imaging with physics-informed deep learning\n\n"
        "## Introduction\n\n<!-- kb_page: 3 -->\n\n"
        "We established a physical noise model of SPAD arrays. The physical noise sources "
        "include dark count and crosstalk. We collected 2790 images from 90 scenes at "
        "10 different bit depths and 3 illumination fluxes.\n\n"
        "## Discussion\n\n<!-- kb_page: 8 -->\n\n"
        "The reported physics-informed technique adapts to several single-photon imaging modalities.\n",
        encoding="utf-8",
    )

    plan = build_citation_plan(
        prompt="From the Discussion section only, what does physics-informed SPAD imaging enable?",
        answer_hits=[
            {
                "text": "The reported physics-informed technique adapts to several modalities.",
                "meta": {
                    "source_path": str(pidl),
                    "heading_path": "Discussion",
                    "page_start": 8,
                },
            }
        ],
    )

    focus = next(slot for slot in plan["slots"] if slot["source_path"] == str(pidl))
    assert focus["heading_path"] == "Discussion"
    assert focus["page_start"] == 8


def test_physics_informed_spad_comparison_preserves_the_other_method_and_budget(
    tmp_path: Path,
) -> None:
    pidl = tmp_path / "High-resolution single-photon imaging with physics-informed deep learning.en.md"
    pidl.write_text(
        "# High-resolution single-photon imaging with physics-informed deep learning\n\n"
        "## Introduction\n\n<!-- kb_page: 3 -->\n\n"
        "We established a physical noise model of SPAD arrays. The physical noise sources "
        "include dark count and crosstalk. We collected 2790 images from 90 scenes at "
        "10 different bit depths and 3 illumination fluxes.\n",
        encoding="utf-8",
    )
    other = tmp_path / "MethodX single-photon reconstruction.en.md"
    other.write_text("# MethodX\n\n## Abstract\n\nMethodX uses a calibrated statistical prior.\n", encoding="utf-8")

    plan = build_citation_plan(
        prompt="Compare physics-informed SPAD reconstruction with MethodX.",
        answer_hits=[
            {"text": "Physics-informed SPAD reconstruction.", "meta": {"source_path": str(pidl)}},
            {"text": "MethodX uses a calibrated statistical prior.", "meta": {"source_path": str(other)}},
        ],
    )

    system_a = [slot for slot in plan["slots"] if slot["preferred_system"] == "system_a"]
    assert {slot["source_path"] for slot in system_a} == {str(pidl), str(other)}
    assert plan["budget"]["system_a"] >= 2


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


def test_prompt_alignment_bundles_video_result_with_photometric_mechanism(
    tmp_path: Path,
) -> None:
    source = tmp_path / "real-time-3d-single-pixel-video.en.md"
    source.write_text(
        "<!-- kb_page: 2 -->\n\n## Abstract\n\n"
        "Four spatially-separated single-pixel detectors reconstruct continuous "
        "three-dimensional video at 8 frames per second with a resolution of "
        r"$64 \times 64$ pixels."
        "\n\n<!-- kb_page: 3 -->\n\n## Introduction\n\n"
        "Photometric stereo estimates surface orientation from images acquired under "
        "multiple lighting directions.\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source),
            "heading_path": "Introduction",
            "evidence_quote": "A generic real-time imaging overview.",
        },
        ranking_texts=[
            "How does real-time 3D single-pixel video use photometric stereo and four "
            "spatially separated detectors for parallel reconstruction, and what frame "
            "rate and 64 x 64 resolution does it report?"
        ],
    )

    assert "four spatially-separated" in slot["evidence_quote"].lower()
    assert "8 frames per second" in slot["evidence_quote"]
    assert r"64 \times 64" in slot["evidence_quote"]
    assert "Photometric stereo" in slot["evidence_quote"]
    assert "multiple lighting directions" in slot["evidence_quote"]
    assert slot["heading_path"].endswith("Abstract")
    assert slot["page_start"] == 2


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


def test_author_by_author_profile_request_budgets_one_locator_per_author() -> None:
    source = "author-review.en.md"
    plan = build_citation_plan(
        prompt=(
            "\u8bf7\u6839\u636e Author Biographies\uff0c\u5206\u522b\u6982\u62ec Kai Song\u3001Yaoxing Bian \u548c "
            "Liantuan Xiao \u7684\u6559\u80b2\u7ecf\u5386\u3001\u5f53\u524d\u804c\u4f4d\u548c\u7814\u7a76\u65b9\u5411\uff0c\u5e76\u9010\u4eba\u7ed9\u51fa\u539f\u6587\u8bc1\u636e\u3002"
        ),
        prompt_family="overview",
        answer_hits=[
            {
                "text": "Kai Song received his B.S. and M.S. degrees and is pursuing his Ph.D.",
                "meta": {"source_path": source, "heading_path": "Author Biographies"},
            },
            {
                "text": "Yaoxing Bian received his Ph.D. and is currently a lecturer.",
                "meta": {"source_path": source, "heading_path": "Author Biographies"},
            },
            {
                "text": "Liantuan Xiao is currently working as a Changjiang professor.",
                "meta": {"source_path": source, "heading_path": "Author Biographies"},
            },
        ],
        support_slots=[
            {
                "source_path": source,
                "heading_path": "Author Biographies",
                "evidence_quote": (
                    "Kai Song received his B.S. and M.S. degrees and is pursuing his Ph.D."
                ),
            }
        ],
    )

    biography_slots = [
        slot
        for slot in plan["slots"]
        if slot["preferred_system"] == "system_a"
        and "author biographies" in slot["heading_path"].lower()
    ]
    assert len(biography_slots) == 3
    assert plan["budget"]["system_a"] == 3
    assert plan["per_paragraph_budget"]["system_a"] == 3
    assert plan["coverage_mode"] == "per_entity"
    assert plan["coverage_entity_type"] == "author_profile"
    assert plan["coverage_target_count"] == 3
    assert plan["coverage_targets"] == ["Kai Song", "Yaoxing Bian", "Liantuan Xiao"]
    assert [slot["candidate_hits"] for slot in biography_slots] == [[1], [2], [3]]
    assert [slot["coverage_target"] for slot in biography_slots] == [
        "Kai Song",
        "Yaoxing Bian",
        "Liantuan Xiao",
    ]


def test_author_profile_heading_aliases_share_per_author_coverage() -> None:
    prompt = (
        "\u8bf7\u6839\u636e Author Biographies\uff0c\u5206\u522b\u6982\u62ec Kai Song\u3001Yaoxing Bian \u548c "
        "Liantuan Xiao \u7684\u6559\u80b2\u7ecf\u5386\u3001\u5f53\u524d\u804c\u4f4d\u548c\u7814\u7a76\u65b9\u5411\u3002"
    )
    evidence = (
        "Kai Song received his degrees. Yaoxing Bian is currently a lecturer. "
        "Liantuan Xiao is currently a professor."
    )

    for heading in ("Author Biography", "Author Biographies", "\u4f5c\u8005\u7b80\u4ecb"):
        plan = build_citation_plan(
            prompt=prompt,
            prompt_family="overview",
            answer_hits=[
                {
                    "text": evidence,
                    "meta": {
                        "source_path": "author-review.en.md",
                        "heading_path": heading,
                    },
                }
            ],
        )

        assert plan["budget"]["system_a"] == 3, heading
        assert plan["per_paragraph_budget"]["system_a"] == 3, heading
        assert plan["coverage_mode"] == "per_entity", heading
        assert plan["coverage_target_count"] == 3, heading
        assert plan["coverage_targets"] == ["Kai Song", "Yaoxing Bian", "Liantuan Xiao"], heading


def test_aggregated_biography_hit_still_budgets_each_named_author() -> None:
    plan = build_citation_plan(
        prompt=(
            "\u8bf7\u6839\u636e Author Biographies\uff0c\u5206\u522b\u6982\u62ec Kai Song\u3001Yaoxing Bian \u548c "
            "Liantuan Xiao \u7684\u6559\u80b2\u7ecf\u5386\u3001\u5f53\u524d\u804c\u4f4d\u548c\u7814\u7a76\u65b9\u5411\uff0c\u5e76\u9010\u4eba\u7ed9\u51fa\u539f\u6587\u8bc1\u636e\u3002"
        ),
        prompt_family="overview",
        answer_hits=[
            {
                "text": (
                    "Kai Song received his degrees. Yaoxing Bian is a lecturer. "
                    "Liantuan Xiao is a Changjiang professor."
                ),
                "meta": {
                    "source_path": "author-review.en.md",
                    "heading_path": "Author Biographies",
                },
            }
        ],
    )

    assert plan["budget"]["system_a"] == 3
    assert plan["per_paragraph_budget"]["system_a"] == 3
    assert plan["coverage_mode"] == "per_entity"
    assert plan["coverage_target_count"] == 3
    assert plan["coverage_targets"] == ["Kai Song", "Yaoxing Bian", "Liantuan Xiao"]


def test_author_profile_targets_exclude_title_case_field_labels() -> None:
    plan = build_citation_plan(
        prompt=(
            "Using Author Biographies, respectively summarize Kai Song and Yaoxing Bian: "
            "Current Position and Research Direction. Please Summarize with evidence."
        ),
        prompt_family="overview",
        answer_hits=[
            {
                "text": (
                    "Kai Song — Education Background: degrees; Research Direction: "
                    "single-pixel imaging. Yaoxing Bian — Current Position: lecturer; "
                    "Research Interests: random lasers."
                ),
                "meta": {
                    "source_path": "author-review.en.md",
                    "heading_path": "Author Biographies",
                },
            }
        ],
    )

    assert plan["coverage_target_count"] == 2
    assert plan["coverage_targets"] == ["Kai Song", "Yaoxing Bian"]
    assert plan["budget"]["system_a"] == 2


def test_author_profile_source_locator_keeps_system_a_overview_intent() -> None:
    plan = build_citation_plan(
        prompt=(
            "请依据论文的 Author Biographies，分别说明 Kai Song、Yaoxing Bian 和 "
            "Liantuan Xiao 的教育经历、当前职位与研究方向；每个人都要有可点击的原文出处。"
        ),
        prompt_family="overview",
        answer_hits=[
            {
                "text": (
                    "Kai Song received his degrees. Yaoxing Bian is a lecturer. "
                    "Liantuan Xiao is a professor."
                ),
                "meta": {
                    "source_path": "author-review.en.md",
                    "heading_path": "Author Biographies",
                },
            }
        ],
    )

    assert plan["intent"] == "beginner_overview"
    assert plan["budget"]["system_a"] == 3
    assert plan["budget"]["system_b"] == 0


def test_unnamed_chinese_author_count_does_not_enable_target_aware_rendering() -> None:
    plan = build_citation_plan(
        prompt="请根据 Author Biographies 分别概括三位作者的教育经历和研究方向。",
        prompt_family="overview",
        answer_hits=[
            {
                "text": "Kai Song, Yaoxing Bian, and Liantuan Xiao are the three authors.",
                "meta": {
                    "source_path": "author-review.en.md",
                    "heading_path": "Author Biographies",
                },
            }
        ],
    )

    assert plan["budget"]["system_a"] == 3
    assert "coverage_mode" not in plan
    assert "coverage_target_count" not in plan


def test_prompt_aligned_source_pins_fdm_speed_snr_abstract_bundle(tmp_path: Path) -> None:
    source = tmp_path / "frequency-division-multiplexed-spi.en.md"
    exact = (
        "Here, we implement frequency-division methods to parallelize the single-pixel "
        "imaging process and demonstrate a trade-off between signal-to-noise ratio and "
        "acquisition speed—without altering detector integration time."
    )
    source.write_text(
        f"# FDM\n\n## Abstract\n\n{exact}\n\n## Discussion\n\n"
        "Frequency-division imaging improves frame rate in several experiments.\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {"source_path": str(source), "evidence_quote": "Generic discussion."},
        ranking_texts=["频分复用单像素成像为什么能更快？代价是什么，是否需要改变探测器积分时间？"],
    )

    assert slot["evidence_quote"] == exact
    assert slot["heading_path"].endswith("Abstract")


def test_prompt_aligned_source_pins_sequential_support_abstract_bundle(tmp_path: Path) -> None:
    source = tmp_path / "sequential-adaptive-compressed-sensing.en.md"
    exact = (
        "A sequential adaptive compressed sensing procedure for signal support recovery is "
        "proposed and analyzed based on the principle of distilled sensing."
    )
    source.write_text(
        f"# Sequential CS\n\n## Abstract\n\n{exact}\n\n## Prior work\n\n"
        "Adaptive sensing procedures allocate measurements sequentially.\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {"source_path": str(source), "evidence_quote": "Related work."},
        ranking_texts=["顺序自适应压缩感知如何实现 signal support recovery？"],
    )

    assert slot["evidence_quote"] == exact
    assert slot["heading_path"].endswith("Abstract")


def test_prompt_aligned_source_pins_scinerf_abstract_for_scigs_comparison(
    tmp_path: Path,
) -> None:
    source = tmp_path / "CVPR-2024-SCINeRF.en.md"
    exact = (
        "Specifically, we formulate the physical imaging process of SCI as part of "
        "the training of NeRF, allowing the scene representation to be optimized."
    )
    source.write_text(
        "<!-- kb_page: 1 -->\n\n## Abstract\n\n"
        + exact
        + "\n\n<!-- kb_page: 5 -->\n\n## 4. Experiments\n\n"
        "SCINeRF compares GAP-TV, PnP-FFDNet, and EfficientSCI.\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {"source_path": str(source), "evidence_quote": "Experimental baselines."},
        ranking_texts=["What is the difference between SCIGS and SCINeRF?"],
    )

    assert slot["evidence_quote"] == exact
    assert slot["heading_path"].endswith("Abstract")
    assert slot["page_start"] == 1


def test_prompt_aligned_source_pins_cassi_dual_disperser_architecture(
    tmp_path: Path,
) -> None:
    source = tmp_path / (
        "OE-2007-Single-shot compressive spectral imaging with a "
        "dual-disperser architecture.en.md"
    )
    exact = (
        "The primary features of the system design are two dispersive elements, "
        "arranged in opposition and surrounding a binary-valued aperture code."
    )
    source.write_text(
        "<!-- kb_page: 1 -->\n\n## Abstract\n\n"
        + exact
        + "\n\n<!-- kb_page: 10 -->\n\n## Figure 8\n\n"
        "A citrus target is imaged under broadband illumination.\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {"source_path": str(source), "evidence_quote": "Figure 8 citrus target."},
        ranking_texts=[
            "CASSI \u7684\u53cc\u8272\u6563\u7ed3\u6784\u600e\u4e48\u6446\uff1f\u4e3a\u4ec0\u4e48\u4e2d\u95f4\u662f\u4e8c\u503c\u5b54\u5f84\uff1f"
        ],
    )

    assert slot["evidence_quote"] == exact
    assert slot["heading_path"].endswith("Abstract")
    assert slot["page_start"] == 1


def test_sequential_scope_plan_rebinds_algorithm_hit_to_abstract_hit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "SSP-2012-Sequentially designed compressed sensing.en.md"
    abstract = (
        "A sequential adaptive compressed sensing procedure for signal support recovery is "
        "proposed and analyzed. The procedure is based on the principle of distilled sensing, "
        "and makes use of sparse sensing matrices to perform sketching observations."
    )
    algorithm = (
        "The sequential compressed sensing procedure uses a multi-step acquisition process "
        "where each step involves measurement and refinement."
    )
    source.write_text(
        "<!-- kb_page: 1 -->\n\n## Abstract\n\n"
        + abstract
        + "\n\n## III. SEQUENTIAL COMPRESSED SENSING ALGORITHM\n\n"
        + algorithm
        + "\n",
        encoding="utf-8",
    )
    hits = [
        {
            "text": algorithm,
            "meta": {
                "source_path": str(source),
                "heading_path": "III. SEQUENTIAL COMPRESSED SENSING ALGORITHM",
                "block_id": "algorithm-block",
                "anchor_id": "algorithm-paragraph",
            },
        },
        {
            "text": abstract,
            "meta": {
                "source_path": str(source),
                "heading_path": "Abstract",
                "block_id": "abstract-block",
                "anchor_id": "abstract-paragraph",
            },
        },
    ]

    plan = build_citation_plan(
        prompt="Sequential compressed sensing 相比一次性随机测量多利用了什么信息？它主要保证恢复什么？",
        answer_hits=hits,
    )

    system_a = [slot for slot in plan["slots"] if slot["preferred_system"] == "system_a"]
    assert len(system_a) == 1
    assert system_a[0]["candidate_hits"] == [2]
    assert system_a[0]["heading_path"].endswith("Abstract")
    assert "signal support recovery" in system_a[0]["evidence_quote"]
    assert "distilled sensing" in system_a[0]["evidence_quote"]


def test_prompt_aligned_source_keeps_degradation_chain_and_global_propagation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "robust-imaging.en.md"
    chain = (
        "The degradation process is as follows: illumination blur occurs first, "
        "spatial downsampling follows, mechanical jitter causes misalignment, the "
        "detection path adds blur, and photon shot noise plus electronic noise affect detection."
    )
    propagation = (
        "In SPI, as the single-pixel detector integrates light from the entire scene, "
        "noise from each photodetector readout can propagate and spread to the entire "
        "image after reconstruction."
    )
    source.write_text(
        "# Robust imaging\n\n## Introduction\n\n"
        "A comprehensive degradation model improves robust reconstruction under global noise.\n\n"
        "## Results\n\n"
        f"{chain}\n\n{propagation}\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source),
            "evidence_quote": (
                "A comprehensive degradation model improves robust reconstruction "
                "under global noise."
            ),
        },
        ranking_texts=[
            "真实退化链有哪些环节？局部读出噪声为什么会传播成全局污染？"
        ],
    )

    assert chain in slot["evidence_quote"]
    assert propagation in slot["evidence_quote"]
    assert slot["heading_path"].endswith("Results")


def test_prompt_aligned_source_uses_unfolding_module_contract_not_abstract(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ista-net.en.md"
    r_module = (
        "$r^{(k)}$ Module: it corresponds to the data update, whose matrix term is "
        "the gradient of the data-fidelity term; the step size may vary by iteration."
    )
    x_module = (
        "$x^{(k)}$ Module: it computes the proximal mapping associated with the "
        "nonlinear transform."
    )
    parameters = (
        "Parameters in ISTA-Net: the learnable set includes the step size in the "
        "$r^{(k)}$ module, the parameters of the forward and backward transforms, "
        "and the shrinkage threshold in the $x^{(k)}$ module."
    )
    source.write_text(
        "# ISTA-Net\n\n## Abstract\n\nThe proximal mapping uses learned parameters.\n\n"
        "## 3.2 Framework\n\n"
        f"{r_module}\n\n{x_module}\n\n{parameters}\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source),
            "evidence_quote": "The proximal mapping uses learned parameters.",
        },
        ranking_texts=[
            "ISTA-Net unfolding 如何把一次 iteration 变成 phase？r 模块、x 模块和参数分别做什么？"
        ],
    )

    assert r_module in slot["evidence_quote"]
    assert x_module in slot["evidence_quote"]
    assert parameters in slot["evidence_quote"]
    assert slot["heading_path"].endswith("3.2 Framework")


def test_prompt_aligned_source_recognizes_formula_role_wording_for_unfolding(
    tmp_path: Path,
) -> None:
    source = tmp_path / "CVPR-2018-ISTA-Net.en.md"
    r_module = (
        "$r^{(k)}$ Module: it corresponds to the data update, whose matrix term is "
        "the gradient of the data-fidelity term; the step size may vary by iteration."
    )
    x_module = (
        "$x^{(k)}$ Module: it computes the proximal mapping associated with the "
        "nonlinear transform."
    )
    parameters = (
        "Parameters in ISTA-Net: the learnable set includes the step size in the "
        "$r^{(k)}$ module, the parameters of the forward and backward transforms, "
        "and the shrinkage threshold in the $x^{(k)}$ module."
    )
    source.write_text(
        "# ISTA-Net\n\n## Abstract\n\nThe proximal mapping uses learned parameters.\n\n"
        "## 3.2 Framework\n\n"
        f"{r_module}\n\n{x_module}\n\n{parameters}\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source),
            "evidence_quote": "The proximal mapping uses learned parameters.",
        },
        ranking_texts=[
            "《ISTA-Net》把一次 ISTA 迭代展开成网络时，r^(k)、x^(k) 和"
            "可学习参数分别承担什么作用？请给出原文章节证据。"
        ],
    )

    assert r_module in slot["evidence_quote"]
    assert x_module in slot["evidence_quote"]
    assert parameters in slot["evidence_quote"]
    assert slot["heading_path"].endswith("3.2 Framework")


def test_prompt_aligned_source_bundles_requested_table_rows_with_trailing_metrics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "CVPR-2018-ISTA-Net.en.md"
    source.write_text(
        "# ISTA-Net\n\n## 5.2 Comparison\n\n"
        "**Table 1.** Average PSNR on Set11.\n"
        "| Algorithm | CS Ratio | Time CPU/GPU | FPS CPU/GPU | | | | | | |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        "| | 50% | 40% | 30% | 25% | 10% | 4% | 1% | | |\n"
        "| ISTA-Net | 37.43 | 35.36 | 32.91 | 31.53 | 25.80 | 21.23 | 17.30 | 0.923s/0.039s | 1.08/25.6 |\n"
        "| ISTA-Net$^+$ | 38.07 | 36.06 | 33.82 | 32.57 | 26.64 | 21.31 | 17.34 | 1.375s/0.047s | 0.73/21.3 |\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {
            "source_path": str(source),
            "heading_path": "ISTA-Net / Abstract",
            "evidence_quote": "A broad table summary.",
        },
        ranking_texts=[
            "《ISTA-Net》表1里 Set11、25% CS ratio 时，ISTA-Net 与 ISTA-Net+ "
            "的 PSNR、CPU/GPU 时间和 FPS 分别是多少？"
        ],
    )

    assert "CS Ratio 25% = 31.53" in slot["evidence_quote"]
    assert "Time CPU/GPU = 0.923s/0.039s" in slot["evidence_quote"]
    assert "FPS CPU/GPU = 1.08/25.6" in slot["evidence_quote"]
    assert "CS Ratio 25% = 32.57" in slot["evidence_quote"]
    assert "Time CPU/GPU = 1.375s/0.047s" in slot["evidence_quote"]
    assert "FPS CPU/GPU = 0.73/21.3" in slot["evidence_quote"]
    assert slot["selection_reason"] == "prompt_aligned_table_rows"


def test_foveated_intent_promotes_exact_sciadv_source_into_system_a_top_three(
    tmp_path: Path,
) -> None:
    target = (
        tmp_path
        / "SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.en.md"
    )
    target_evidence = (
        "Adaptive foveated single-pixel imaging combines a foveal region with dynamic "
        "supersampling, while successive frames acquire complementary spatial information."
    )
    target.write_text(
        f"# Adaptive foveated single-pixel imaging\n\n## Abstract\n\n{target_evidence}\n",
        encoding="utf-8",
    )
    distractors = [
        {
            "text": f"Generic adaptive imaging discussion {index}.",
            "meta": {
                "source_path": str(tmp_path / f"generic-{index}.en.md"),
                "heading_path": "Abstract",
            },
        }
        for index in range(1, 4)
    ]
    target_hit = {
        "text": target_evidence,
        "meta": {
            "source_path": str(target),
            "heading_path": "Abstract",
            "page_start": 1,
        },
    }

    for prompt in (
        "dynamic supersampling 是不是就是只盯着画面重要的地方多拍一点？",
        "foveated 成像如何分配采样？",
        "是不是只盯着重要区域多拍一些？",
    ):
        plan = build_citation_plan(
            prompt=prompt,
            answer_hits=[*distractors, target_hit],
        )
        system_a = [
            slot
            for slot in plan["slots"]
            if slot["preferred_system"] == "system_a"
        ]

        assert len(system_a) <= 3
        assert system_a[0]["source_path"] == str(target)
        assert system_a[0]["candidate_hits"] == [4]


def test_sequential_exact_duplicate_slots_fold_and_bind_first_valid_hit() -> None:
    source_path = "SSP-2012-Sequentially designed compressed sensing.en.md"
    exact = (
        "A sequential adaptive compressed sensing procedure for signal support recovery is "
        "proposed and analyzed. The procedure is based on the principle of distilled sensing."
    )
    plan = build_citation_plan(
        prompt="顺序自适应压缩感知相比一次性随机测量有什么优势？",
        support_slots=[
            {
                "source_path": source_path,
                "heading_path": "Sequentially Designed Compressed Sensing / Abstract",
                "evidence_quote": exact,
            }
        ],
        answer_hits=[
            {
                "text": exact,
                "meta": {
                    "source_path": source_path,
                    "heading_path": "Sequentially Designed Compressed Sensing / Abstract",
                    "evidence_quote": exact,
                },
            },
            {
                "text": exact,
                "meta": {
                    "source_path": source_path,
                    "heading_path": "Abstract",
                    "evidence_quote": exact,
                },
            },
        ],
    )
    system_a = [
        slot for slot in plan["slots"] if slot["preferred_system"] == "system_a"
    ]

    assert len(system_a) == 1
    assert system_a[0]["source_path"] == source_path
    assert system_a[0]["candidate_hits"] == [1]


def test_prompt_aligned_source_pins_hsi_fsi_sampling_metric_comparison(tmp_path: Path) -> None:
    source = tmp_path / "hadamard-fourier-comparison.en.md"
    exact = (
        "We compare HSI and FSI under different sampling ratios using PSNR and SSIM, "
        "and FSI provides better reconstruction quality in the undersampling regime."
    )
    source.write_text(
        "# HSI and FSI\n\n## 3. Comparison\n\n"
        f"{exact}\n\nA later table lists isolated PSNR and SSIM values.\n",
        encoding="utf-8",
    )

    slot = _prompt_aligned_source_slot(
        {"source_path": str(source), "evidence_quote": "Table values."},
        ranking_texts=["Hadamard 和 Fourier 在不同采样率下怎么选？比较 PSNR 与 SSIM。"],
    )

    assert slot["evidence_quote"] == exact
    assert "Comparison" in slot["heading_path"]
