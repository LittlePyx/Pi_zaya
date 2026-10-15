from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_product_shadow_vectorized_protocol_v1.json"
)


def sha256_file_variants(path: Path) -> set[str]:
    data = path.read_bytes()
    lf = data.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    return {
        hashlib.sha256(candidate).hexdigest()
        for candidate in (data, lf, crlf)
    }


def validate_protocol(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id") != "retrieval_product_shadow_vectorized_v1"
    ):
        errors.append("vectorized shadow protocol identity changed")
    parent = payload.get("parent") or {}
    for path_key, hash_key in (
        ("protocol_path", "protocol_sha256"),
        ("rejected_report_path", "rejected_report_sha256"),
    ):
        path = REPO_ROOT / str(parent.get(path_key) or "")
        if not path.is_file() or parent.get(hash_key) not in sha256_file_variants(path):
            errors.append(f"vectorized shadow parent changed: {path_key}")
    if (
        parent.get("rejected_result_fingerprint")
        != "48e54e620d73681f3fbaf8ad7d3e86982b51f5079d7ab0f1d66878113500461e"
    ):
        errors.append("vectorized shadow rejected result changed")
    if parent.get("rejected_reason") != "warm_shadow_p95_ms_261.435_exceeded_250.0":
        errors.append("vectorized shadow eligibility reason changed")

    changes = payload.get("allowed_implementation_changes") or {}
    for path_key, hash_key in (
        ("dense_retriever_path", "dense_retriever_sha256"),
        ("provider_path", "provider_sha256"),
        ("shadow_runtime_path", "shadow_runtime_sha256"),
    ):
        path = REPO_ROOT / str(changes.get(path_key) or "")
        if not path.is_file() or changes.get(hash_key) not in sha256_file_variants(path):
            errors.append(f"vectorized shadow implementation changed: {path_key}")
    expected_change_contract = {
        "scoring_backend": "numpy_float32_exact_shortlist",
        "exact_shortlist_rule": "float32_matrix_vector_boundary_then_original_python_double_precision_score_and_tie_order",
        "error_margin_rule": "max(1e-7,8*dimension*float32_epsilon)",
        "insufficient_shortlist_policy": "full_python_scalar_fallback",
        "numpy_absent_policy": "full_python_scalar_fallback",
    }
    if any(
        changes.get(key) != value for key, value in expected_change_contract.items()
    ):
        errors.append("vectorized shadow scoring contract changed")

    evidence = payload.get("equivalence_evidence") or {}
    if (
        evidence.get("seeded_unit_query_count") != 40
        or evidence.get("real_index_query_vector_count") != 30
        or evidence.get("real_index_shape") != [2251, 1024]
        or evidence.get("real_index_top_k") != 40
        or evidence.get("exact_id_order_equal") is not True
        or evidence.get("exact_scores_equal") is not True
    ):
        errors.append("vectorized shadow rank-equivalence evidence changed")

    inherited = payload.get("inherited_unchanged") or {}
    if inherited != {
        "fixture_path": "docs/retrieval_user_journey_expansion_v1.json",
        "case_count": 18,
        "modes_per_case": ["scoped", "full_library"],
        "submission_count": 36,
        "model_provider_fusion_and_index": "exactly_parent_protocol",
        "activation_privacy_fail_open_and_read_only_boundaries": "exactly_parent_protocol",
        "controlled_operational_gate": "exactly_parent_protocol",
        "maximum_warm_shadow_p95_ms": 250.0,
        "maximum_foreground_submission_p95_ms": 5.0,
    }:
        errors.append("vectorized shadow inherited experiment changed")
    execution = payload.get("execution") or {}
    if execution != {
        "maximum_replacement_runs": 1,
        "report_path": "docs/retrieval_product_shadow_vectorized_operational_v1.json",
        "telemetry_path": "test_results/retrieval_product_shadow_vectorized_v1/observations.jsonl",
        "no_parameter_or_threshold_changes_after_measurement": True,
        "product_enabled_during_or_after_run": False,
        "ordinary_user_surface_exposed": False,
        "pass_decision": "passed_for_explicit_opt_in_design",
        "reject_decision": "rejected_keep_dense_disabled",
    }:
        errors.append("vectorized shadow execution boundary changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] vectorized shadow protocol cannot be read: {type(exc).__name__}")
        return 1
    errors = validate_protocol(payload)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[OK] vectorized shadow replacement is frozen and product-disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
