from __future__ import annotations

import copy
import json

from tools.retrieval_eval.run_lighter_runtime_stage2_e5_small import (
    DEFAULT_CONTROL,
    DEFAULT_REPORT,
    EXPECTED_RESULT_FINGERPRINT,
    validate_report,
)
from tools.retrieval_eval.validate_lighter_runtime_stage2_e5_small_protocol import (
    DEFAULT_PROTOCOL,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def _control() -> dict:
    return json.loads(DEFAULT_CONTROL.read_text(encoding="utf-8"))


def _report() -> dict:
    return json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))


def test_e5_stage2_report_passes() -> None:
    assert (
        validate_report(
            _report(),
            protocol=_protocol(),
            control=_control(),
            expected=EXPECTED_RESULT_FINGERPRINT,
        )
        == []
    )


def test_e5_stage2_report_rejects_quality_drift() -> None:
    changed = copy.deepcopy(_report())
    changed["channels"]["e5_small_window_protected_dense_tail"]["modes"]["scoped"][
        "metrics"
    ]["recall@10"] = 0.375
    errors = validate_report(
        changed, protocol=_protocol(), control=_control(), expected=None
    )
    assert "E5 Stage-2 quality arithmetic changed" in errors


def test_e5_stage2_report_rejects_runtime_drift() -> None:
    changed = copy.deepcopy(_report())
    changed["runtime"]["full_build_ms"] = 600001.0
    errors = validate_report(
        changed, protocol=_protocol(), control=_control(), expected=None
    )
    assert "E5 Stage-2 runtime arithmetic changed" in errors


def test_e5_stage2_report_rejects_product_enablement() -> None:
    changed = copy.deepcopy(_report())
    changed["decision"]["product_enabled"] = True
    errors = validate_report(
        changed, protocol=_protocol(), control=_control(), expected=None
    )
    assert "E5 Stage-2 decision changed" in errors
