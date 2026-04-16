from pathlib import Path

from tools.manual_regression import frontmatter_pdf_regression as runner


def test_load_manifest_converter_frontmatter_pdf_v1_resolves_real_pdf_paths():
    suite = runner.load_suite("tools/manual_regression/manifests/converter_frontmatter_pdf_v1.json")

    assert suite["suite_id"] == "converter_frontmatter_pdf_v1"
    case_ids = {case["id"] for case in suite["cases"]}
    assert case_ids == {
        "SCIADV_2017_TITLE_ABSTRACT_RECOVERY",
        "NATCOMM_2021_IMPLICIT_ABSTRACT_INTRO",
        "JOPT_2016_READER_SERVICE_CLEANUP",
    }
    assert all(Path(case["_pdf_abspath"]).exists() for case in suite["cases"])


def test_evaluate_output_text_checks_prefix_and_author_leak_guards():
    case = {
        "id": "demo",
        "title": "demo",
        "_checks": {
            "must_contain": ["## Abstract"],
            "must_contain_text": ["Adaptive foveated single-pixel imaging with dynamic supersampling"],
            "must_not_contain_text": ["David B. Phillips"],
            "must_not_start_with": ["# INTRODUCTION"],
            "ordered_text": [
                "Adaptive foveated single-pixel imaging with dynamic supersampling",
                "## Abstract",
            ],
        },
    }

    output = "\n".join(
        [
            "# Adaptive foveated single-pixel imaging with dynamic supersampling",
            "",
            "## Abstract",
            "",
            "In contrast to conventional multipixel cameras...",
        ]
    )

    result = runner.evaluate_output_text(
        output,
        case=case,
        requested_speed_mode="auto",
        actual_speed_mode="no_llm",
        output_path="tmp/output.md",
    )

    assert result["status"] == "PASS"
    assert result["ordered_text_positions"]["## Abstract"] > result["ordered_text_positions"]["Adaptive foveated single-pixel imaging with dynamic supersampling"]


def test_run_case_auto_falls_back_to_no_llm_when_credentials_missing(tmp_path, monkeypatch):
    class _FakeSettings:
        api_key = None
        base_url = "https://example.com/v1"
        model = "qwen3-vl-plus"
        timeout_s = 60.0
        max_retries = 1

    class _FakeConverter:
        def __init__(self, cfg):
            self.cfg = cfg

        def convert(self, pdf_path: str, save_dir: str) -> None:
            out = Path(save_dir) / "output.md"
            out.write_text(
                "# 3D single-pixel video\n\n## Abstract\n\n## Introduction\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(runner, "load_settings", lambda: _FakeSettings())
    monkeypatch.setattr(runner, "PDFConverter", _FakeConverter)

    case = {
        "id": "demo_case",
        "title": "demo",
        "pdf_path": __file__,
        "_pdf_abspath": __file__,
        "_checks": {
            "must_contain": ["# 3D single-pixel video", "## Abstract", "## Introduction"],
            "ordered": ["# 3D single-pixel video", "## Abstract", "## Introduction"],
        },
    }

    result = runner.run_case(
        case,
        out_root=tmp_path,
        requested_speed_mode="auto",
    )

    assert result["status"] == "PASS"
    assert result["actual_speed_mode"] == "no_llm"
