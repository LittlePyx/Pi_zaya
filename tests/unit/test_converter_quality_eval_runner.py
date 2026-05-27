from __future__ import annotations

import json
from pathlib import Path

from kb.converter.quality_acceptance import load_quality_manifest
from tools.converter_quality import run_converter_quality_eval as runner


def _write_manifest(tmp_path: Path, *, md_path: str = "paper.md", min_chars: int = 10) -> Path:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_id": "demo_converter_quality",
                "defaults": {"max_missing_images": 0},
                "cases": [
                    {
                        "id": "demo",
                        "title": "Demo",
                        "md_path": md_path,
                        "checks": {
                            "min_chars": min_chars,
                            "min_headings": 1,
                            "must_contain_text": ["Demo Paper"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_markdown(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Demo Paper",
                "",
                "## Abstract",
                "",
                "Body cites [1].",
                "",
                "## References",
                "[1] A reference.",
            ]
        ),
        encoding="utf-8",
    )


def test_evaluate_manifest_returns_pass_for_matching_case(tmp_path):
    _write_markdown(tmp_path / "paper.md")
    manifest = load_quality_manifest(_write_manifest(tmp_path), repo_root=tmp_path)

    summary = runner.evaluate_manifest(manifest)

    assert summary["overall_status"] == "PASS"
    assert summary["pass_count"] == 1
    assert summary["fail_count"] == 0
    assert summary["missing_count"] == 0
    assert summary["results"][0]["status"] == "PASS"


def test_evaluate_manifest_marks_missing_markdown(tmp_path):
    manifest = load_quality_manifest(_write_manifest(tmp_path, md_path="missing.md"), repo_root=tmp_path)

    summary = runner.evaluate_manifest(manifest)

    assert summary["overall_status"] == "FAIL"
    assert summary["missing_count"] == 1
    assert summary["results"][0]["status"] == "MISSING"
    assert summary["results"][0]["failures"][0].startswith("missing_markdown:")


def test_build_report_contains_failures_and_metric_table(tmp_path):
    manifest = load_quality_manifest(_write_manifest(tmp_path, md_path="missing.md"), repo_root=tmp_path)
    summary = runner.evaluate_manifest(manifest)

    report = runner.build_report(summary, output_dir=tmp_path)

    assert "# Converter Markdown Quality Eval" in report
    assert "## Failures" in report
    assert "`demo` [MISSING]" in report
    assert "## Key Metrics" in report


def test_main_dry_run_loads_manifest_without_reading_markdown(tmp_path, capsys):
    manifest_path = _write_manifest(tmp_path, md_path="missing.md")

    exit_code = runner.main(["--manifest", str(manifest_path), "--dry-run"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "[OK] cases: 1" in captured.out
    assert "demo [missing]" in captured.out


def test_main_fail_on_quality_returns_one_for_failed_case(tmp_path):
    _write_markdown(tmp_path / "paper.md")
    manifest_path = _write_manifest(tmp_path, min_chars=1000)

    exit_code = runner.main(
        [
            "--manifest",
            str(manifest_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--fail-on-quality",
        ]
    )

    assert exit_code == 1
