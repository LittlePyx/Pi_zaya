from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from tools.retrieval_eval.validate_semantic_retrieval_report import validate_report


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_cases_v1.json"
REPORT = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_benchmark_v1.json"


def _report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_semantic_retrieval_report_is_frozen_and_rejected() -> None:
    report = _report()
    assert validate_report(report, FIXTURE) == []
    assert report["acceptance"]["raw_dense_gain"] == {
        "scoped": 0.125,
        "full_library": 0.208333,
        "passes_all_required_modes": False,
    }
    assert report["acceptance"]["safe_rrf_gain"] == {
        "scoped": 0.0,
        "full_library": 0.083333,
        "passes_all_required_modes": False,
    }
    assert report["decision"]["status"] == "rejected_for_product_promotion"


def test_semantic_retrieval_report_rejects_metric_or_activation_drift() -> None:
    changed = deepcopy(_report())
    changed["channels"]["safe_rrf"]["full_library"]["recall@10"] = 0.75
    assert any("gain is stale" in error for error in validate_report(changed, FIXTURE))

    enabled = deepcopy(_report())
    enabled["candidate"]["product_enabled"] = True
    assert any("product-disabled" in error for error in validate_report(enabled, FIXTURE))
