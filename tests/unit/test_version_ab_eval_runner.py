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
