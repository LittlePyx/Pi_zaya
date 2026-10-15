from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_lighter_runtime_stage1_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_lighter_runtime_stage1_protocol_is_valid() -> None:
    assert validate_protocol(_protocol()) == []


def test_lighter_runtime_stage1_rejects_old_parent_pointer() -> None:
    changed = copy.deepcopy(_protocol())
    changed["read_only_parent_index"]["relative_path"] = (
        "test_results/retrieval_cpu_full_index_feasibility_v1/index"
    )
    assert "lighter runtime stage-1 parent index changed" in validate_protocol(changed)


def test_lighter_runtime_stage1_rejects_vector_parity_relaxation() -> None:
    changed = copy.deepcopy(_protocol())
    changed["measurement"]["document_vector_parity"]["minimum_mean_cosine"] = 0.8
    assert "lighter runtime stage-1 measurement changed" in validate_protocol(changed)


def test_lighter_runtime_stage1_rejects_timeout_or_repeat_relaxation() -> None:
    changed = copy.deepcopy(_protocol())
    changed["operational_gates"]["maximum_subprocess_wall_clock_seconds"] = 3600
    changed["execution"]["maximum_runs"] = 2
    errors = validate_protocol(changed)
    assert "lighter runtime stage-1 operational gates changed" in errors
    assert "lighter runtime stage-1 execution boundary changed" in errors


def test_lighter_runtime_stage1_rejects_heavy_runtime_or_product_surface() -> None:
    changed = copy.deepcopy(_protocol())
    changed["runtime"]["torch_allowed"] = True
    changed["execution"]["product_enabled"] = True
    errors = validate_protocol(changed)
    assert "lighter runtime stage-1 runtime contract changed" in errors
    assert "lighter runtime stage-1 execution boundary changed" in errors
