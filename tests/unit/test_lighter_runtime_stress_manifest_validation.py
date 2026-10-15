from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_lighter_runtime_stress_manifest import (
    DEFAULT_MANIFEST,
    validate_manifest,
)


def _manifest() -> dict:
    return json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))


def test_lighter_runtime_stress_manifest_is_valid() -> None:
    assert validate_manifest(_manifest()) == []


def test_lighter_runtime_stress_manifest_rejects_sample_drift() -> None:
    changed = copy.deepcopy(_manifest())
    changed["items"][0]["chunk_id"] = changed["items"][1]["chunk_id"]
    assert "lighter runtime stress item membership changed" in validate_manifest(changed)


def test_lighter_runtime_stress_manifest_rejects_paper_text() -> None:
    changed = copy.deepcopy(_manifest())
    changed["items"][0]["text"] = "paper text"
    assert "lighter runtime stress item schema or privacy boundary changed" in validate_manifest(changed)


def test_lighter_runtime_stress_manifest_rejects_inference() -> None:
    changed = copy.deepcopy(_manifest())
    changed["execution"]["onnx_session_run_calls"] = 1
    assert "lighter runtime stress execution boundary changed" in validate_manifest(changed)
