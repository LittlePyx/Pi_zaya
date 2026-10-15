from __future__ import annotations

import copy
import json

from tools.retrieval_eval.analyze_e5_small_window_census import (
    DEFAULT_REPORT,
    EXPECTED_RESULT_FINGERPRINT,
    validate_report,
)


def _report() -> dict:
    return json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))


def test_e5_small_window_census_report_passes() -> None:
    assert (
        validate_report(
            _report(), expected_result_fingerprint=EXPECTED_RESULT_FINGERPRINT
        )
        == []
    )


def test_e5_small_window_census_report_rejects_hidden_tail_loss() -> None:
    changed = copy.deepcopy(_report())
    changed["coverage"]["uncovered_content_token_count"] = 1
    assert "E5-small window-census coverage changed" in validate_report(changed)


def test_e5_small_window_census_report_rejects_workload_drift() -> None:
    changed = copy.deepcopy(_report())
    changed["workload"]["window_count"] += 1
    assert "E5-small window-census workload arithmetic changed" in validate_report(changed)


def test_e5_small_window_census_report_rejects_model_inference() -> None:
    changed = copy.deepcopy(_report())
    changed["execution"]["model_inference_runs"] = 1
    assert "E5-small window-census execution boundary changed" in validate_report(changed)
