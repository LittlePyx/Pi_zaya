from __future__ import annotations

from kb.citation_context_summary import (
    accept_system_b_context_summary,
    build_system_b_context_summary,
    reject_system_b_context_summary,
)


def test_build_system_b_context_summary_distills_spi_detector_context() -> None:
    summary = build_system_b_context_summary(
        context=(
            "Unlike traditional focal plane array detectors, SPI only adopts a single-pixel "
            "detector to collect echo signals, offering significant advantages in detection "
            "sensitivity, spectral response range, and imaging cost."
        ),
        claim="SPI has advantages in sensitivity and spectral response.",
        title="Optical imaging by means of two-photon quantum entanglement",
        source="LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.pdf",
        reference_entry="Pittman T, Shih Y. Optical imaging by means of two-photon quantum entanglement.",
    )

    assert summary == "当前论文在说明单像素探测相对焦平面阵列的硬件或成本差异时引用它。"


def test_build_system_b_context_summary_rejects_reference_entry_like_context() -> None:
    summary = build_system_b_context_summary(
        context=(
            "Macias-Garza, F., Bovik, A. C., Diller, K. R. The missing cone problem "
            "and low-pass distortion in optical serial sectioning microscopy. IEEE Trans. "
            "Acoust., Speech, Signal Process. 2, 890-893 (1988)."
        ),
        title="Missing Cone Of Frequencies And Low-Pass Distortion In Three-Dimensional Microscopic Images",
    )

    assert summary == ""


def test_reject_system_b_context_summary_blocks_metadata_and_context_copies() -> None:
    context = "The current paper cites missing cone and low-pass distortion work when discussing limitations."

    assert reject_system_b_context_summary(
        "这篇发表于 Physical Review A 1995 的论文值得打开。",
        context=context,
        title="Optical imaging by means of two-photon quantum entanglement",
    ) == "metadata_repeated"
    assert reject_system_b_context_summary(context, context=context) == "duplicates_context"


def test_accept_system_b_context_summary_keeps_llm_research_note() -> None:
    accepted = accept_system_b_context_summary(
        "当前论文引用它时强调 SPI 用单点探测器替代传统焦平面阵列，线索落在探测结构和成本优势。",
        context="Unlike traditional focal plane array detectors, SPI only adopts a single-pixel detector to collect echo signals.",
        claim="SPI has advantages in sensitivity and spectral response.",
        title="Optical imaging by means of two-photon quantum entanglement",
    )

    assert accepted.startswith("当前论文引用它时强调 SPI")
