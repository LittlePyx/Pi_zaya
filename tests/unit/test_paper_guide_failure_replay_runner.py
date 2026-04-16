import json

from tools.manual_regression import run_paper_guide_failure_replay_v1 as runner


def test_load_case_metrics_prefers_summary_json_with_scorecard(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "counts": {"total": 3, "pass": 2, "fail": 1},
                "hit_levels": {
                    "cases_with_direct": 2,
                    "cases_with_exact": 1,
                    "counts": {"exact": 1, "block": 1, "heading": 1, "none": 0},
                },
                "primary_hit_levels": {
                    "counts": {"exact": 1, "block": 0, "heading": 1, "none": 1},
                },
                "scorecard": {
                    "locate": {
                        "cases": 2,
                        "pass": 1,
                        "first_click_ok": 1,
                        "exact_first_click": 1,
                        "heading_or_none": 1,
                    },
                    "citation": {"cases": 1, "pass": 1},
                    "structured_markers": {"cases": 3, "pass": 2, "raw_cite_leak_cases": 1},
                    "quality": {"cases": 3, "pass": 3},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics = runner._load_case_metrics("", summary_json_path=str(summary_path))

    assert metrics["case_total"] == 3
    assert metrics["case_pass"] == 2
    assert metrics["case_fail"] == 1
    assert metrics["cases_with_direct"] == 2
    assert metrics["cases_with_exact"] == 1
    assert metrics["hit_level_counts"] == {"exact": 1, "block": 1, "heading": 1, "none": 0}
    assert metrics["primary_hit_levels"] == {"exact": 1, "block": 0, "heading": 1, "none": 1}
    assert metrics["scorecard_counts"]["locate"]["first_click_ok"] == 1
    assert metrics["scorecard_counts"]["citation"]["pass"] == 1
    assert metrics["scorecard_counts"]["structured_markers"]["raw_cite_leak_cases"] == 1


def test_scorecard_rates_from_counts_computes_key_quality_rates():
    rates = runner._scorecard_rates_from_counts(
        {
            "locate": {"cases": 4, "pass": 3, "first_click_ok": 3, "exact_first_click": 2, "heading_or_none": 1},
            "citation": {"cases": 2, "pass": 1},
            "structured_markers": {"cases": 5, "pass": 4, "raw_cite_leak_cases": 1},
            "quality": {"cases": 5, "pass": 5},
        }
    )

    assert rates["locate"]["first_click_rate"] == 0.75
    assert rates["locate"]["exact_first_click_rate"] == 0.5
    assert rates["locate"]["heading_or_none_rate"] == 0.25
    assert rates["citation"]["ref_num_accuracy"] == 0.5
    assert rates["structured_markers"]["clean_rate"] == 0.8
    assert rates["quality"]["minimum_ok_rate"] == 1.0
