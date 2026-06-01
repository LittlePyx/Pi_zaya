from __future__ import annotations

import json

from kb.citation_context import extract_inpaper_reference_context


def test_extract_inpaper_reference_context_skips_reference_list(tmp_path) -> None:
    md = tmp_path / "paper.en.md"
    md.write_text(
        "\n".join(
            [
                "# Example Paper",
                "<!-- kb_page: 2 -->",
                "## 2. Related Work",
                "Most existing methods employ ADMM [4] for optimization in compressive imaging.",
                "",
                "## References",
                "[4] Boyd S. Distributed Optimization and Statistical Learning via ADMM. 2011.",
            ]
        ),
        encoding="utf-8",
    )

    out = extract_inpaper_reference_context(
        str(md),
        4,
        answer_context="ADMM is prior optimization machinery.",
    )

    assert out["citation_context_source"] == "source_markdown"
    assert "Most existing methods employ ADMM [4]" in out["citation_context"]
    assert "Boyd S" not in out["citation_context"]
    assert out["heading_path"].endswith("2. Related Work")
    assert out["page_start"] == 2
    assert "p. 2" in out["location_label"]


def test_extract_inpaper_reference_context_expands_ranges(tmp_path) -> None:
    md = tmp_path / "paper.en.md"
    md.write_text(
        "\n".join(
            [
                "# Example Paper",
                "## Method",
                "Deep reconstruction networks follow earlier compressive sensing systems [11-13].",
                "",
                "## References",
                "[12] Example reference entry that must not be selected.",
            ]
        ),
        encoding="utf-8",
    )

    out = extract_inpaper_reference_context(str(md), 12)

    assert out["citation_context_source"] == "source_markdown"
    assert "earlier compressive sensing systems [11-13]" in out["citation_context"]
    assert "Example reference entry" not in out["citation_context"]
    assert out["heading_path"].endswith("Method")


def test_extract_inpaper_reference_context_skips_author_affiliation_list(tmp_path) -> None:
    md = tmp_path / "paper.en.md"
    md.write_text(
        "\n".join(
            [
                "# Structured detection for simultaneous super-resolution and optical sectioning in laser scanning microscopy",
                "",
                "Alessandro Zunino [1,4], Giacomo Garre [1,2,4], Eleonora Perego [1,3], Sabrina Zappone [1,2], Mattia Donato [1], Nadine Vastenhouw [3] & Giuseppe Vicidomini [1]",
                "",
                "## Abstract",
                "",
                "Imaging a three-dimensional sample requires optical sectioning, namely, the capability to reject out-of-focus light and image a single plane [1,2]. Conventional wide-field microscopes cannot separate in-focus light from out-of-focus light, due to the well-known phenomenon of the missing cone [3,4].",
                "",
                "## References",
                "[3] Macias-Garza, F. The missing cone problem and low-pass distortion in optical serial sectioning microscopy. 1988.",
            ]
        ),
        encoding="utf-8",
    )

    out = extract_inpaper_reference_context(
        str(md),
        3,
        answer_context="missing cone and optical sectioning",
    )

    assert out["citation_context_source"] == "source_markdown"
    assert "missing cone [3,4]" in out["citation_context"]
    assert "Alessandro Zunino" not in out["citation_context"]
    assert out["heading_path"].endswith("Abstract")
    assert "Abstract" in out["location_label"]


def test_extract_inpaper_reference_context_skips_inline_notation_fragment(tmp_path) -> None:
    md = tmp_path / "paper.en.md"
    md.write_text(
        "\n".join(
            [
                "# Example Paper",
                "## Results",
                "The result of s [2]ISM is an image with enhanced resolution and optical sectioning.",
                "",
                "## Methods",
                "The implementation follows the detector-array method [2] when configuring the reconstruction.",
                "",
                "## References",
                "[2] Detector-array reference entry that must not be selected.",
            ]
        ),
        encoding="utf-8",
    )

    out = extract_inpaper_reference_context(str(md), 2)

    assert out["citation_context_source"] == "source_markdown"
    assert "detector-array method [2]" in out["citation_context"]
    assert "s [2]ISM" not in out["citation_context"]
    assert out["heading_path"].endswith("Methods")


def test_extract_inpaper_reference_context_does_not_treat_reference_named_section_as_bibliography(tmp_path) -> None:
    md = tmp_path / "paper.en.md"
    md.write_text(
        "\n".join(
            [
                "# Example Paper",
                "## Reference representation",
                "The model builds a reference representation from calibrated measurements [5].",
                "",
                "## References",
                "[5] Calibration reference entry.",
            ]
        ),
        encoding="utf-8",
    )

    out = extract_inpaper_reference_context(str(md), 5)

    assert out["citation_context_source"] == "source_markdown"
    assert "reference representation" in out["citation_context"].lower()
    assert out["heading_path"].endswith("Reference representation")


def test_extract_inpaper_reference_context_prefers_structured_reference_index(tmp_path) -> None:
    md = tmp_path / "paper.en.md"
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    md.write_text(
        "\n".join(
            [
                "# Example Paper",
                "This body intentionally has no usable inline marker.",
                "",
                "## References",
                "[7] Example upstream work.",
            ]
        ),
        encoding="utf-8",
    )
    (assets_dir / "reference_index.json").write_text(
        json.dumps(
            {
                "references": [
                    {
                        "ref_num": 7,
                        "citation_mentions": [
                            {
                                "citation_context": "The introduction briefly lists prior work [7].",
                                "heading_path": "Example Paper / Introduction",
                                "location_label": "Example Paper / Introduction / p. 1",
                                "page_start": 1,
                                "page_end": 1,
                                "line_start": 10,
                                "line_end": 10,
                                "anchor_kind": "paragraph",
                            },
                            {
                                "citation_context": "The method follows a calibrated detector-array design [7].",
                                "heading_path": "Example Paper / Methods",
                                "location_label": "Example Paper / Methods / p. 3",
                                "page_start": 3,
                                "page_end": 3,
                                "line_start": 42,
                                "line_end": 42,
                                "anchor_kind": "paragraph",
                                "block_id": "blk_demo_00042",
                                "anchor_id": "p_00042",
                            }
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    out = extract_inpaper_reference_context(
        str(md),
        7,
        answer_context="detector-array design",
    )

    assert out["citation_context_source"] == "structured_reference_index"
    assert "detector-array design [7]" in out["citation_context"]
    assert out["heading_path"].endswith("Methods")
    assert out["page_start"] == 3
    assert out["block_id"] == "blk_demo_00042"


def test_extract_inpaper_reference_context_ignores_marker_only_structured_context(tmp_path) -> None:
    md = tmp_path / "paper.en.md"
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    md.write_text(
        "\n".join(
            [
                "# Example Paper",
                "## Super-resolution",
                (
                    "Compared to traditional reconstruction methods, the network achieved large advancements "
                    "in both image quality and reconstruction speed [66]."
                ),
                "",
                "## References",
                "[66] Example upstream work.",
            ]
        ),
        encoding="utf-8",
    )
    (assets_dir / "reference_index.json").write_text(
        json.dumps(
            {
                "references": [
                    {
                        "ref_num": 66,
                        "citation_mentions": [
                            {
                                "citation_context": "...[66]",
                                "heading_path": "Example Paper / Figure caption",
                                "page_start": 5,
                            }
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    out = extract_inpaper_reference_context(
        str(md),
        66,
        answer_context="reconstruction speed and image quality",
    )

    assert out["citation_context_source"] == "source_markdown"
    assert "network achieved large advancements" in out["citation_context"]
