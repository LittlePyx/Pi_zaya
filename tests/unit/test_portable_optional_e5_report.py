from __future__ import annotations

import copy
import json

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.run_portable_optional_e5 import (
    DEFAULT_PROTOCOL,
    DEFAULT_REPORT,
    validate_report,
)
from tools.retrieval_eval.validate_portable_optional_e5_report import (
    EXPECTED_REPORT_SHA256,
    EXPECTED_RESULT_FINGERPRINT,
)


def _inputs() -> tuple[dict, dict]:
    return (
        json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8")),
        json.loads(DEFAULT_REPORT.read_text(encoding="utf-8")),
    )


def _validate(protocol: dict, report: dict) -> list[str]:
    return validate_report(
        report,
        protocol=protocol,
        expected_result_fingerprint=EXPECTED_RESULT_FINGERPRINT,
    )


def test_portable_optional_e5_rejection_is_pinned_and_valid() -> None:
    protocol, report = _inputs()
    assert sha256_file(DEFAULT_REPORT) == EXPECTED_REPORT_SHA256
    assert _validate(protocol, report) == []
    assert report["gate"]["checks"]["import_time"] is False
    assert report["decision"]["status"] == protocol["decision"]["failure_status"]


def test_portable_optional_e5_report_rejects_import_timing_mutation() -> None:
    protocol, report = _inputs()
    changed = copy.deepcopy(report)
    changed["runtime"]["import_ms"] = 1.0
    assert "portable optional E5 gate is inconsistent" in _validate(protocol, changed)


def test_portable_optional_e5_report_rejects_product_enablement() -> None:
    protocol, report = _inputs()
    changed = copy.deepcopy(report)
    changed["execution"]["product_enabled"] = True
    assert "portable optional E5 gate is inconsistent" in _validate(protocol, changed)


def test_portable_optional_e5_report_rejects_decision_mutation() -> None:
    protocol, report = _inputs()
    changed = copy.deepcopy(report)
    changed["decision"]["status"] = protocol["decision"]["pass_status"]
    assert "portable optional E5 decision is inconsistent" in _validate(protocol, changed)


def test_portable_optional_e5_report_rejects_fingerprint_mutation() -> None:
    protocol, report = _inputs()
    changed = copy.deepcopy(report)
    changed["result_fingerprint"] = "0" * 64
    assert "portable optional E5 result fingerprint is inconsistent" in _validate(
        protocol,
        changed,
    )
