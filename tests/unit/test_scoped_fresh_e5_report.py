from __future__ import annotations

import copy
import json

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.run_scoped_fresh_e5 import (
    DEFAULT_PROTOCOL,
    DEFAULT_REPORT,
    validate_report,
)
from tools.retrieval_eval.validate_scoped_fresh_e5_report import EXPECTED_REPORT_SHA256


def _inputs() -> tuple[dict, dict]:
    return (
        json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8")),
        json.loads(DEFAULT_REPORT.read_text(encoding="utf-8")),
    )


def test_scoped_fresh_e5_report_is_pinned_and_valid() -> None:
    protocol, report = _inputs()
    assert sha256_file(DEFAULT_REPORT) == EXPECTED_REPORT_SHA256
    assert validate_report(report, protocol=protocol) == []


def test_scoped_fresh_e5_report_rejects_quality_mutation() -> None:
    protocol, report = _inputs()
    changed = copy.deepcopy(report)
    changed["quality_gate"]["recall_at_10_gain"] = 0.0
    errors = validate_report(changed, protocol=protocol)
    assert "scoped fresh E5 quality gate is inconsistent" in errors


def test_scoped_fresh_e5_report_rejects_runtime_mutation() -> None:
    protocol, report = _inputs()
    changed = copy.deepcopy(report)
    changed["selector_trace"][0]["source_escape_count"] = 1
    errors = validate_report(changed, protocol=protocol)
    assert "scoped fresh E5 runtime gate is inconsistent" in errors


def test_scoped_fresh_e5_report_rejects_decision_mutation() -> None:
    protocol, report = _inputs()
    changed = copy.deepcopy(report)
    changed["decision"]["product_enabled"] = True
    errors = validate_report(changed, protocol=protocol)
    assert "scoped fresh E5 decision is inconsistent" in errors


def test_scoped_fresh_e5_report_rejects_result_fingerprint_mutation() -> None:
    protocol, report = _inputs()
    changed = copy.deepcopy(report)
    changed["result_fingerprint"] = "0" * 64
    errors = validate_report(changed, protocol=protocol)
    assert "scoped fresh E5 result fingerprint is inconsistent" in errors
