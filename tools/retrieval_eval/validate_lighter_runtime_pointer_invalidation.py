from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file


DEFAULT_INVALIDATION = (
    REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage1_pointer_invalidation_v1.json"
)
EXPECTED_INVALIDATION_SHA256 = "277ffe5f46d499df0c6b8cc895284dcb708814751549ec11a0b5f2005865583a"


def validate_invalidation(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("invalidation_id")
        != "retrieval_lighter_runtime_stage1_pointer_invalidation_v1"
        or payload.get("status") != "retained_before_first_onnx_session_run"
    ):
        errors.append("lighter runtime pointer invalidation identity changed")
    invalidated = payload.get("invalidated_protocol") or {}
    source = REPO_ROOT / str(invalidated.get("path") or "")
    if (
        not source.is_file()
        or invalidated.get("sha256") != sha256_file(source)
        or invalidated.get("field")
        != "candidates[0].stage_1_quality_route.read_only_parent_index"
        or invalidated.get("frozen_value")
        != "test_results/retrieval_cpu_full_index_feasibility_v1/index"
        or invalidated.get("retained_index_value")
        != "test_results/retrieval_cpu_full_index_feasibility_v1/index_v1"
        or invalidated.get("retained_index_tree_sha256")
        != "8c8fa74b7ce72c30687933b0f41d7e6b399615cf4d3a271313c0bfb85e169b50"
    ):
        errors.append("lighter runtime invalidated pointer evidence changed")
    scope = payload.get("scope") or {}
    if (
        scope.get("stage_0_result_affected") is not False
        or scope.get("stage_1_execution_under_invalidated_protocol_allowed") is not False
        or scope.get("historical_protocol_mutation_allowed") is not False
    ):
        errors.append("lighter runtime pointer invalidation scope changed")
    recovery = payload.get("recovery") or {}
    if (
        recovery.get("protocol_path")
        != "docs/retrieval_lighter_runtime_stage1_qwen_protocol_v1.json"
        or recovery.get("must_pin_stage_0_report") is not True
        or recovery.get("must_pin_stress_manifest") is not True
        or recovery.get("must_pin_retained_index_v1") is not True
        or recovery.get("must_exist_before_first_inference") is not True
    ):
        errors.append("lighter runtime pointer recovery changed")
    execution = payload.get("execution") or {}
    if execution != {
        "onnx_session_run_calls_before_retention": 0,
        "queries_encoded_before_retention": 0,
        "documents_embedded_before_retention": 0,
        "vectors_created_before_retention": 0,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }:
        errors.append("lighter runtime pointer invalidation execution boundary changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_INVALIDATION.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] lighter runtime pointer invalidation cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_invalidation(payload)
    if sha256_file(DEFAULT_INVALIDATION) != EXPECTED_INVALIDATION_SHA256:
        errors.append("lighter runtime pointer invalidation SHA-256 changed")
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] invalid Stage-1 parent pointer is retained and cannot authorize inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
