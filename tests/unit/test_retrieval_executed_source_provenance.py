from __future__ import annotations

import json
from pathlib import Path

import tools.retrieval_eval.validate_lighter_runtime_stage2_e5_small_protocol as stage2
import tools.retrieval_eval.validate_portable_e5_product_startup_protocol as portable_v2
import tools.retrieval_eval.validate_portable_optional_e5_protocol as portable_v1
import tools.retrieval_eval.validate_product_e5_selector_shadow_protocol as selector
import tools.retrieval_eval.validate_product_e5_user_journey_protocol as journey
import tools.retrieval_eval.validate_scoped_fresh_e5_protocol as scoped


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_stage2_historical_validation_uses_recorded_source_identity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    protocol = _load(stage2.DEFAULT_PROTOCOL)
    missing = tmp_path / "refactored.py"
    monkeypatch.setattr(stage2, "DEFAULT_RETRIEVER", missing)
    monkeypatch.setattr(stage2, "DEFAULT_PROTECTED_TAIL", missing)
    assert stage2.validate_protocol(protocol) == []
    errors = stage2.validate_protocol(protocol, inspect_implementation_files=True)
    assert "E5 Stage-2 retriever implementation changed" in errors
    assert "E5 Stage-2 protected-tail implementation changed" in errors


def test_selector_historical_validation_does_not_rehash_live_implementation(
    monkeypatch,
) -> None:
    protocol = _load(selector.DEFAULT_PROTOCOL)
    original = selector._verify_paths

    def fake_verify(payload, pairs, *, label):
        if label == "product E5 selector implementation":
            return ["product E5 selector implementation changed: simulated"]
        return original(payload, pairs, label=label)

    monkeypatch.setattr(selector, "_verify_paths", fake_verify)
    assert selector.validate_protocol(protocol) == []
    assert (
        "product E5 selector implementation changed: simulated"
        in selector.validate_protocol(protocol, inspect_implementation_files=True)
    )


def test_journey_historical_validation_does_not_rehash_live_implementation(
    monkeypatch,
) -> None:
    protocol = _load(journey.DEFAULT_PROTOCOL)
    original = journey._matches_file

    def fake_match(section, path_key, sha_key):
        if path_key.startswith(("executor_", "index_", "runtime_", "selector_", "fusion_")):
            return False
        return original(section, path_key, sha_key)

    monkeypatch.setattr(journey, "_matches_file", fake_match)
    assert journey.validate_protocol(protocol) == []
    errors = journey.validate_protocol(protocol, inspect_implementation_files=True)
    assert "product E5 user-journey source changed: executor" in errors


def test_scoped_report_validation_separates_live_preflight_hashes(monkeypatch) -> None:
    protocol = _load(scoped.DEFAULT_PROTOCOL)
    monkeypatch.setattr(scoped, "_raw_sha256", lambda _path: "simulated-refactor")
    assert scoped.validate_protocol(protocol) == []
    errors = scoped.validate_protocol(protocol, inspect_files=True)
    assert "scoped fresh E5 product source changed: executor_path" in errors


def test_portable_v1_report_validation_separates_executed_runner_hashes(
    monkeypatch,
) -> None:
    protocol = _load(portable_v1.DEFAULT_PROTOCOL)
    monkeypatch.setattr(portable_v1, "_raw_sha256", lambda _path: "simulated-refactor")
    assert portable_v1.validate_protocol(protocol) == []
    errors = portable_v1.validate_protocol(protocol, inspect_runtime_files=True)
    assert "portable optional E5 runner_path changed" in errors


def test_portable_v2_report_validation_separates_executed_runner_hashes(
    monkeypatch,
) -> None:
    protocol = _load(portable_v2.DEFAULT_PROTOCOL)
    original = portable_v2._raw_sha256

    def fake_hash(path: Path) -> str:
        if path.suffix == ".py":
            return "simulated-refactor"
        return original(path)

    monkeypatch.setattr(portable_v2, "_raw_sha256", fake_hash)
    assert portable_v2.validate_protocol(protocol) == []
    errors = portable_v2.validate_protocol(protocol, inspect_runtime_files=True)
    assert "portable E5 product-startup runner_path changed" in errors
