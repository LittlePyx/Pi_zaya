from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.version_ab import run_version_ab_eval as ab_eval


def test_version_ab_contract_preserves_full_qa_and_project_coverage() -> None:
    contract = ab_eval.load_contract()

    assert [item["expected_cases"] for item in contract["qa"]["suites"]] == [29, 5]
    assert contract["project_journeys"]["runs"] == 3
    assert contract["qa"]["case_timeout_s"] >= 30


def test_version_ab_contract_rejects_reduced_coverage(tmp_path: Path) -> None:
    contract = ab_eval.load_contract()
    contract["qa"]["suites"][0]["expected_cases"] = 28
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="29-case"):
        ab_eval.load_contract(path)


def test_corpus_fingerprint_excludes_only_declared_top_level(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root in (left, right):
        (root / "paper").mkdir(parents=True)
        (root / "paper" / "source.md").write_text("same evidence", encoding="utf-8")
        (root / "docs.json").write_text("{}", encoding="utf-8")
        (root / "temp").mkdir()
    (left / "temp" / "ignored.txt").write_text("left", encoding="utf-8")
    (right / "temp" / "ignored.txt").write_text("right", encoding="utf-8")

    left_fp = ab_eval.corpus_fingerprint(left, exclude_top_level={"temp"})
    right_fp = ab_eval.corpus_fingerprint(right, exclude_top_level={"temp"})

    assert left_fp["sha256"] == right_fp["sha256"]
    (right / "paper" / "source.md").write_text("changed evidence", encoding="utf-8")
    assert left_fp["sha256"] != ab_eval.corpus_fingerprint(
        right,
        exclude_top_level={"temp"},
    )["sha256"]


def test_failure_summary_keeps_timeout_and_quality_buckets(tmp_path: Path) -> None:
    path = tmp_path / "raw_results.jsonl"
    rows = [
        {
            "id": "timeout-case",
            "status": "error",
            "error": "deadline",
            "error_type": "TimeoutError",
            "quality": {"ok": False, "failures": [{"name": "runner_error"}]},
        },
        {
            "id": "quality-case",
            "status": "done",
            "quality": {
                "ok": False,
                "failures": [
                    {"name": "citations_include_required_docs"},
                    {"name": "refs_card_copy_quality"},
                ],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(item) for item in rows), encoding="utf-8")

    failures, buckets = ab_eval._qa_failure_summary(path)

    assert [item["id"] for item in failures] == ["timeout-case", "quality-case"]
    assert buckets == {
        "citations_include_required_docs": 1,
        "refs_card_copy_quality": 1,
        "runner_error": 1,
    }


def test_summary_count_preserves_zero_failures() -> None:
    summary = {"total": 29, "passed": 29, "failed": 0}

    assert ab_eval._summary_count(summary, "failed", 29) == 0
    assert ab_eval._summary_count(summary, "missing", 29) == 29


def test_rebuild_report_recomputes_zero_failure_candidate(tmp_path: Path) -> None:
    versions: dict[str, dict] = {}
    for side, passed, failed in (("baseline", 0, 1), ("candidate", 1, 0)):
        suite_dir = tmp_path / side
        suite_dir.mkdir()
        summary_path = suite_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "total": 1,
                    "passed": passed,
                    "failed": failed,
                    "timing": {},
                }
            ),
            encoding="utf-8",
        )
        failure_rows = []
        if failed:
            failure_rows.append(
                {
                    "id": "old-timeout",
                    "status": "error",
                    "quality": {
                        "ok": False,
                        "failures": [{"name": "runner_error"}],
                    },
                }
            )
        (suite_dir / "raw_results.jsonl").write_text(
            "\n".join(json.dumps(item) for item in failure_rows),
            encoding="utf-8",
        )
        versions[side] = {
            "label": side,
            "qa": [
                {
                    "name": "full_library",
                    "suite": "full_library_acceptance_v1",
                    "expected_cases": 1,
                    "summary_path": str(summary_path),
                    "actual_cases": 1,
                    "passed": passed,
                    "failed": 1,
                    "coverage_complete": True,
                    "quality_ok": False,
                    "timing": {},
                    "failure_cases": [],
                }
            ],
            "project_journeys": [
                {
                    "run": 1,
                    "supported": side == "candidate",
                    "passed": side == "candidate",
                    "reason": "unsupported" if side == "baseline" else "",
                }
            ],
        }
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "created_at": "2026-08-12T00:00:00+08:00",
                "corpus": {"identical": True},
                "settings": {"project_journeys": {"runs": 1}},
                "versions": versions,
                "comparison": {},
            }
        ),
        encoding="utf-8",
    )

    rebuilt = ab_eval.rebuild_report(report_path)

    assert rebuilt["versions"]["candidate"]["qa"][0]["failed"] == 0
    assert rebuilt["versions"]["candidate"]["qa"][0]["quality_ok"] is True
    assert rebuilt["comparison"]["candidate_release_ok"] is True
    assert rebuilt["comparison"]["candidate_materially_better"] is True
