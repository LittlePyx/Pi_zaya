from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_lighter_runtime_stage0 import DEFAULT_REPORT, validate_report


def _report() -> dict:
    return json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))


def test_lighter_runtime_stage0_report_is_valid() -> None:
    assert validate_report(_report()) == []


def test_lighter_runtime_stage0_rejects_artifact_drift() -> None:
    changed = copy.deepcopy(_report())
    changed["candidate"]["artifact_sha256"] = "0" * 64
    assert "lighter runtime stage-0 candidate evidence changed" in validate_report(changed)


def test_lighter_runtime_stage0_rejects_hidden_runtime_dependency() -> None:
    changed = copy.deepcopy(_report())
    changed["environment"]["distributions"].append(
        {"name": "torch", "version": "latest", "installed_bytes": 1}
    )
    assert "lighter runtime stage-0 isolated environment changed" in validate_report(changed)


def test_lighter_runtime_stage0_rejects_inference_or_product_enablement() -> None:
    changed = copy.deepcopy(_report())
    changed["execution"]["model_inference_runs"] = 1
    changed["execution"]["product_enabled"] = True
    assert "lighter runtime stage-0 execution boundary changed" in validate_report(changed)


def test_lighter_runtime_stage0_rejects_non_cpu_session() -> None:
    changed = copy.deepcopy(_report())
    changed["onnx"]["session_providers"] = ["AzureExecutionProvider", "CPUExecutionProvider"]
    assert "lighter runtime stage-0 ONNX graph evidence changed" in validate_report(changed)
