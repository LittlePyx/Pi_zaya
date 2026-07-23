from __future__ import annotations

import re

from ui.refs_renderer import (
    _annotate_inpaper_citations_with_hover_meta,
    _system_a_is_low_value_evidence_text,
    _system_a_pick_best_evidence_candidate,
)


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
    assert "training and generalization" in details[0]["binding_overlap_terms"]


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
    assert len(anchors) == 1
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
    assert len(anchors) == 1
    assert len(set(anchors)) == 1
    assert len(details) == 1
    assert "occurrence_specific_claim" not in details[0]["card_quality_flags"]
    assert "Foveated single-pixel imaging uses dynamic supersampling" in details[0]["answer_claim"]
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
