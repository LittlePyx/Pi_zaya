from __future__ import annotations

import copy
import json

from tools.retrieval_eval.build_user_journey_bm25_outcome import build_outcome
from tools.retrieval_eval.validate_user_journey_bm25_baseline import (
    DEFAULT_BASELINE,
    DEFAULT_FIXTURE,
    validate_baseline,
)


def _inputs() -> tuple[dict, dict]:
    fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    baseline = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))
    return fixture, baseline


def test_user_journey_bm25_baseline_is_frozen_and_arithmetic_is_valid() -> None:
    fixture, baseline = _inputs()
    assert validate_baseline(baseline, fixture) == []


def test_user_journey_bm25_baseline_rejects_metric_drift() -> None:
    fixture, baseline = _inputs()
    baseline["channels"]["bm25_product_queries"]["modes"]["scoped"]["metrics"][
        "recall@10"
    ] = 1.0
    assert any(
        "metric arithmetic" in error for error in validate_baseline(baseline, fixture)
    )


def test_user_journey_bm25_baseline_rejects_case_membership_drift() -> None:
    fixture, baseline = _inputs()
    baseline["channels"]["bm25_raw"]["modes"]["full_library"]["cases"].pop()
    assert any(
        "case membership" in error for error in validate_baseline(baseline, fixture)
    )


def test_user_journey_bm25_outcome_keeps_holdout_to_aggregates() -> None:
    fixture, baseline = _inputs()
    outcome = build_outcome(fixture, baseline)
    serialized_holdout = json.dumps(outcome["split_summary"]["holdout"])
    assert all(
        case_id not in serialized_holdout
        for case_id in fixture["split"]["holdout_case_ids"]
    )
    assert (
        outcome["decision"]["holdout_case_details_may_be_used_for_candidate_design"]
        is False
    )


def test_user_journey_bm25_outcome_changes_when_case_metric_changes() -> None:
    fixture, baseline = _inputs()
    original = build_outcome(fixture, baseline)
    changed = copy.deepcopy(baseline)
    row = changed["channels"]["bm25_raw"]["modes"]["scoped"]["cases"][0]
    row["metrics"]["recall@10"] = 1.0 - float(row["metrics"]["recall@10"])
    changed["channels"]["bm25_raw"]["modes"]["scoped"]["metrics"]["recall@10"] = round(
        sum(
            float(item["metrics"]["recall@10"])
            for item in changed["channels"]["bm25_raw"]["modes"]["scoped"]["cases"]
        )
        / 18,
        6,
    )
    assert (
        build_outcome(fixture, changed)["baseline"]["result_fingerprint"]
        != original["baseline"]["result_fingerprint"]
    )
