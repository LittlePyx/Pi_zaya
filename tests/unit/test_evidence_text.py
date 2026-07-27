from __future__ import annotations

from kb.evidence_text import (
    finish_evidence_text,
    looks_low_value_citation_context,
    pick_readable_evidence_text,
    strip_evidence_metadata_prefix,
)


def test_finish_evidence_text_does_not_treat_decimal_point_as_sentence_end() -> None:
    evidence = (
        "Detector type: InGaAs/InAlAs-SPAD. Performance = 61.2% DE at "
        "200 K; Year = 2022; Ref. = [82]"
    )

    assert finish_evidence_text(evidence) == f"{evidence}..."


def test_article_info_keywords_are_not_accepted_as_citation_evidence() -> None:
    assert looks_low_value_citation_context(
        "A R T I C L E I N F O Keywords: Single-pixel imaging Information extraction network Deep learning"
    )
    assert looks_low_value_citation_context(
        "Keywords: single-pixel imaging; deep learning; reconstruction"
    )


def test_pick_readable_evidence_text_skips_broken_leading_fragment() -> None:
    raw = (
        "rson can be described uniquely with a few targeted questions, which is a broken OCR tail. "
        "Computational imaging configurations. "
        "A DMD can be used to spatially filter light by selectively redirecting parts of an incident light beam at ?24? "
        "to the normal. "
        "An object is flood-illuminated and imaged onto a detector."
    )

    picked = pick_readable_evidence_text(
        raw,
        claim="DMD modulation is the sampling hardware behind this method.",
        heading="Understanding compressed sensing",
    )

    assert picked.startswith("A DMD can be used")
    assert "to the normal" in picked
    assert "rson can" not in picked
    assert "Computational imaging configurations" not in picked


def test_pick_readable_evidence_text_keeps_complete_supporting_sentence_over_caption() -> None:
    raw = (
        "Figure 1. Results overview. "
        "Deep learning models improve single-pixel reconstruction quality by learning a nonlinear mapping "
        "from compressed measurements to target images. "
        "Additional details are discussed later."
    )

    picked = pick_readable_evidence_text(
        raw,
        claim="深度学习用于提高单像素重建质量。",
        heading="Deep learning reconstruction",
    )

    assert picked.startswith("Deep learning models improve")
    assert "Figure 1" not in picked
    assert "details are discussed later" not in picked


def test_pick_readable_evidence_text_keeps_cited_sentence_with_commas() -> None:
    raw = (
        "[184] Compared to traditional reconstruction methods, the network achieved large advancements "
        "in both the image quality and reconstruction speed, successfully realizing hyperspectral SPI."
    )

    picked = pick_readable_evidence_text(
        raw,
        claim="PILN has a speed trade-off compared with hardware-accelerated SPI.",
        heading="Color Single-Pixel Imaging",
        title="Fast hyperspectral single-pixel imaging via frequency-division multiplexed illumination",
    )

    assert picked.startswith("[184] Compared to traditional reconstruction methods")
    assert "hyperspectral SPI" in picked


def test_pick_readable_evidence_text_keeps_complete_since_clause() -> None:
    evidence = (
        "Since super-resolution and optical sectioning are achieved simultaneously, "
        "we named our technique s2ISM."
    )

    picked = pick_readable_evidence_text(
        evidence,
        claim="s2ISM provides simultaneous super-resolution and optical sectioning.",
    )

    assert picked == evidence


def test_pick_readable_evidence_text_trims_short_tail_phrase_before_ellipsis() -> None:
    raw = (
        "This paper proposes a novel method for recovering dynamic 3D scene representations "
        "from a single snapshot compressive image, which is the first to introduce an dynamic "
        "explicit representation in this..."
    )

    picked = pick_readable_evidence_text(raw, claim="dynamic 3D scene representation")

    assert picked.endswith("explicit representation...")
    assert "in this..." not in picked


def test_pick_readable_evidence_text_prefers_sentence_with_claim_numbers() -> None:
    raw = (
        "We established a real-world physical noise model of SPAD arrays. "
        "The model includes shot noise, fixed-pattern noise, dark count rate, afterpulsing, "
        "crosstalk, and deadtime noise. "
        "To calibrate this model, we collected a real-shot SPAD dataset containing 2790 images. "
        "The dataset covers 90 scenes, 10 bit depths, and 3 illumination fluxes."
    )

    picked = pick_readable_evidence_text(
        raw,
        claim="该数据集包含2790张图像，覆盖90个场景、10种位深度和3种照明通量。",
        heading="Introduction",
    )

    assert "2790 images" in picked
    assert "90 scenes" in picked
    assert "10 bit depths" in picked
    assert "3 illumination fluxes" in picked


def test_pick_readable_evidence_text_prefers_named_dataset_fragment_at_page_break() -> None:
    raw = (
        "We established a real-world physical noise model of SPAD arrays. "
        "The model contains crosstalk and dark count noise. "
        "With the calibrated physical noise model, we employed public high-resolution "
        "images collected from the PASCAL VOC2007 dataset and"
    )

    picked = pick_readable_evidence_text(
        raw,
        claim=(
            "The SPAD network uses public high-resolution images from PASCAL VOC2007 "
            "as its training prior."
        ),
        heading="Introduction",
    )

    assert picked.startswith("With the calibrated physical noise model")
    assert "PASCAL VOC2007" in picked
    assert picked.endswith("...")


def test_metadata_prefix_strip_preserves_complete_capitalized_evidence_sentence() -> None:
    evidence = (
        "All tested samples were collected under realistic conditions involving mist, jitter, and sensor noise. "
        "The proposed method consistently achieves the lowest LPIPS scores across all samples."
    )

    assert strip_evidence_metadata_prefix(evidence) == evidence
