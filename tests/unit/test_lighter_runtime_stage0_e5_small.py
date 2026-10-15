from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_lighter_runtime_stage0_e5_small import (
    DEFAULT_REPORT,
    validate_report,
)


def _report() -> dict:
    return json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))


def test_e5_small_stage0_report_passes() -> None:
    assert validate_report(_report()) == []


def test_e5_small_stage0_report_rejects_session_run() -> None:
    changed = copy.deepcopy(_report())
    changed["onnx"]["session_run_calls"] = 1
    assert "E5-small Stage-0 ONNX graph evidence changed" in validate_report(changed)


def test_e5_small_stage0_report_rejects_tokenizer_drift() -> None:
    changed = copy.deepcopy(_report())
    changed["tokenizer"]["files"][0]["sha256"] = "0" * 64
    assert "E5-small Stage-0 tokenizer evidence changed" in validate_report(changed)


def test_e5_small_stage0_report_rejects_product_enablement() -> None:
    changed = copy.deepcopy(_report())
    changed["execution"]["product_enabled"] = True
    assert "E5-small Stage-0 execution boundary changed" in validate_report(changed)
