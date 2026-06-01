from __future__ import annotations

from kb.evidence_text import pick_readable_evidence_text


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


def test_pick_readable_evidence_text_trims_short_tail_phrase_before_ellipsis() -> None:
    raw = (
        "This paper proposes a novel method for recovering dynamic 3D scene representations "
        "from a single snapshot compressive image, which is the first to introduce an dynamic "
        "explicit representation in this..."
    )

    picked = pick_readable_evidence_text(raw, claim="dynamic 3D scene representation")

    assert picked.endswith("explicit representation...")
    assert "in this..." not in picked
