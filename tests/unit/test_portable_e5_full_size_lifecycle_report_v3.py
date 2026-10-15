from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_portable_e5_full_size_lifecycle_report_v3 import (
    DEFAULT_REPORT,
    validate_report,
)


def _report() -> dict:
    return json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))


def test_release_component_lifecycle_pass_is_valid() -> None:
    assert validate_report(_report()) == []


def test_report_rejects_ranking_rewrite() -> None:
    changed = copy.deepcopy(_report())
    changed["queries"][0]["ranked_chunk_ids"].reverse()
    assert "release-component lifecycle query evidence changed" in validate_report(
        changed
    )


def test_report_rejects_slow_query_rewrite() -> None:
    changed = copy.deepcopy(_report())
    changed["timings"]["warm_query_p95_ms"] = 1001.0
    assert "release-component lifecycle gates changed" in validate_report(changed)


def test_report_rejects_path_layout_rewrite() -> None:
    changed = copy.deepcopy(_report())
    changed["execution"]["storage_layout_max_path_chars"] = 239
    assert "release-component lifecycle execution changed" in validate_report(changed)


def test_report_rejects_incomplete_cleanup() -> None:
    changed = copy.deepcopy(_report())
    changed["removal"]["cleanup_completed_after_worker_exit"] = False
    assert "release-component lifecycle removal changed" in validate_report(changed)


def test_report_rejects_product_enablement() -> None:
    changed = copy.deepcopy(_report())
    changed["decision"]["product_enabled"] = True
    assert "release-component lifecycle decision changed" in validate_report(changed)


def test_report_rejects_fingerprint_rewrite() -> None:
    changed = copy.deepcopy(_report())
    changed["result_fingerprint"] = "0" * 64
    assert "release-component lifecycle fingerprint changed" in validate_report(changed)
