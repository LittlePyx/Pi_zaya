from pathlib import Path

from tools.manual_regression import implicit_section_regression as runner


def test_load_manifest_resolves_real_sample_sources():
    suite = runner.load_suite("tools/manual_regression/manifests/paper_guide_implicit_sections_v1.json")

    assert suite["suite_id"] == "paper_guide_implicit_sections_v1"
    case_ids = {case["id"] for case in suite["cases"]}
    assert case_ids == {
        "natcommun_2021_sph_implicit_abstract_intro",
        "nature_2025_multi_paragraph_abstract_only",
        "optics_laser_2024_spaced_abstract_heading",
        "optica_2016_implicit_abstract_before_intro",
        "arxiv_qclfm_implicit_abstract_with_roman_intro",
        "cvpr_2024_teaser_before_explicit_abstract",
        "jopt_2016_reader_service_block_before_abstract",
        "lpr_2025_implicit_abstract_with_numbered_intro",
    }
    assert all(Path(case["_source_abspath"]).exists() for case in suite["cases"])


def test_real_sample_suite_passes_current_converter_rules():
    suite = runner.load_suite("tools/manual_regression/manifests/paper_guide_implicit_sections_v1.json")

    summary = runner.evaluate_suite(suite)

    assert summary["case_count"] == 8
    assert summary["fail_count"] == 0
    assert summary["overall_status"] == "PASS"


def test_evaluate_case_detects_inserted_abstract_and_introduction(tmp_path):
    source = tmp_path / "demo.en.md"
    source.write_text(
        "\n".join(
            [
                "# ARTICLE",
                "",
                "## Imaging biological tissue with high-throughput single-pixel compressive holography",
                "",
                "Daixuan Wu$^{1,2}$, Jiawei Luo$^{1,2}$, Yuecheng Shen$^{1,2}$ & Zhaohui Li$^{1,2}$",
                "",
                "Single-pixel holography is capable of generating holographic images with rich spatial information by employing only a single-pixel detector. Thanks to the relatively low dark-noise production, high sensitivity, large bandwidth, and cheap price of single-pixel detectors in comparison to pixel-array detectors, this system becomes attractive for biophotonics and related applications.",
                "",
                "Pixel-array detectors, such as CCD and CMOS cameras, were commonly used in the traditional imaging scheme. However, these detectors are only cost-effective and maintain good performance within a certain spectrum range. In contrast, single-pixel detectors have lower dark-noise production, higher sensitivity, faster response time, and a much cheaper price, motivating the broader introduction to the field.",
                "",
                "## Results",
                "",
                "Body text.",
            ]
        ),
        encoding="utf-8",
    )

    result = runner.evaluate_case(
        {
            "id": "demo",
            "title": "demo",
            "source_path": str(source),
            "checks": {
                "must_contain": ["## Abstract", "## Introduction", "## Results"],
                "ordered": ["## Abstract", "## Introduction", "## Results"],
            },
        }
    )

    assert result["status"] == "PASS"
    assert "## Abstract" in result["after_headings"]
    assert "## Introduction" in result["after_headings"]


def test_evaluate_case_detects_forbidden_spaced_abstract_heading(tmp_path):
    source = tmp_path / "elsevier.en.md"
    source.write_text(
        "\n".join(
            [
                "# Optics and Laser Technology",
                "",
                "## A B S T R A C T",
                "",
                "In this study, we proposed a self-supervised image-loop neural network.",
                "",
                "## 1. Introduction",
                "",
                "Single-pixel imaging utilizes the second-order correlation of classical or quantum light.",
            ]
        ),
        encoding="utf-8",
    )

    result = runner.evaluate_case(
        {
            "id": "elsevier",
            "title": "elsevier",
            "source_path": str(source),
            "checks": {
                "must_contain": ["## Abstract", "## 1. Introduction"],
                "must_not_contain": ["## A B S T R A C T"],
                "ordered": ["## Abstract", "## 1. Introduction"],
            },
        }
    )

    assert result["status"] == "PASS"
    assert "## A B S T R A C T" not in result["after_headings"]


def test_evaluate_case_supports_ordered_text_checks(tmp_path):
    source = tmp_path / "cvpr.en.md"
    source.write_text(
        "\n".join(
            [
                "# SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
                "",
                "![Figure 1](./assets/page_1_fig_1.png)",
                "",
                "**Figure 1.** Given a single snapshot compressed image, our method is able to recover the underlying 3D scene representation.",
                "",
                "## Abstract",
                "",
                "In this paper, we explore the potential of Snapshot Compressive Imaging technique.",
                "",
                "## 1. Introduction",
            ]
        ),
        encoding="utf-8",
    )

    result = runner.evaluate_case(
        {
            "id": "cvpr",
            "title": "cvpr",
            "source_path": str(source),
            "checks": {
                "must_contain": ["## Abstract", "## 1. Introduction"],
                "must_contain_text": ["![Figure 1](./assets/page_1_fig_1.png)"],
                "ordered_text": [
                    "![Figure 1](./assets/page_1_fig_1.png)",
                    "## Abstract",
                    "## 1. Introduction",
                ],
            },
        }
    )

    assert result["status"] == "PASS"
    assert result["ordered_text_positions"]["## Abstract"] > result["ordered_text_positions"]["![Figure 1](./assets/page_1_fig_1.png)"]
