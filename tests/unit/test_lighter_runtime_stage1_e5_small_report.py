from __future__ import annotations

import copy
import json

from tools.retrieval_eval.run_lighter_runtime_stage1_e5_small import (
    DEFAULT_REPORT,
    EXPECTED_RESULT_FINGERPRINT,
    validate_report,
)
from tools.retrieval_eval.validate_lighter_runtime_stage1_e5_small_protocol import (
    DEFAULT_PROTOCOL,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def _report() -> dict:
    return json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))


def test_e5_small_stage1_report_passes() -> None:
    assert (
        validate_report(
            _report(),
            protocol=_protocol(),
            expected_fingerprint=EXPECTED_RESULT_FINGERPRINT,
        )
        == []
    )


def test_e5_small_stage1_report_rejects_unauthorized_full_build() -> None:
    changed = copy.deepcopy(_report())
    changed["document_projection"]["projected_full_build_ms"] = 600001.0
    errors = validate_report(
        changed, protocol=_protocol(), expected_fingerprint=EXPECTED_RESULT_FINGERPRINT
    )
    assert "E5 Stage-1 gate arithmetic changed" in errors


def test_e5_small_stage1_report_rejects_integrity_drift() -> None:
    changed = copy.deepcopy(_report())
    changed["integrity"]["source_after_sha256"] = "0" * 64
    errors = validate_report(changed, protocol=_protocol(), expected_fingerprint=None)
    assert "E5 Stage-1 gate arithmetic changed" in errors


def test_e5_small_stage1_report_rejects_product_enablement() -> None:
    changed = copy.deepcopy(_report())
    changed["decision"]["product_enabled"] = True
    errors = validate_report(changed, protocol=_protocol(), expected_fingerprint=None)
    assert "E5 Stage-1 decision changed" in errors
