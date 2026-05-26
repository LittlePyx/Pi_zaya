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
