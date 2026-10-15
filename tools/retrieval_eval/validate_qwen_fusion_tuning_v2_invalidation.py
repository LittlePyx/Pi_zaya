from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.run_qwen_fusion_tuning_v2 import (
    _load_contracts,
    _result_fingerprint,
    validate_tuning_report,
)


DEFAULT_INVALIDATION = (
    REPO_ROOT / "docs" / "retrieval_qwen_fusion_tuning_v2_invalidation.json"
)


def validate_invalidation(payload: dict) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("report_kind") != "qwen_fusion_tuning_v2_invalidation_v1"
        or payload.get("status") != "invalid_protocol_execution"
    ):
        errors.append("fusion invalidation identity changed")
    protocol_path = REPO_ROOT / str(payload.get("protocol_path") or "")
    report_path = REPO_ROOT / str(payload.get("invalid_report_path") or "")
    if not protocol_path.is_file() or payload.get("protocol_sha256") != sha256_file(
        protocol_path
    ):
        errors.append("invalidated fusion protocol identity changed")
    if not report_path.is_file() or payload.get("invalid_report_sha256") != sha256_file(
        report_path
    ):
        errors.append("invalidated fusion report identity changed")
        return errors
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        protocol, predecessor, _batch, qwen_report = _load_contracts()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return errors + [f"invalidation evidence cannot be read: {type(exc).__name__}"]
    fingerprint = _result_fingerprint(report.get("channels") or {})
    if (
        payload.get("invalid_report_result_fingerprint") != fingerprint
        or report.get("result_fingerprint") != fingerprint
    ):
        errors.append("invalidated fusion result fingerprint changed")
    report_errors = validate_tuning_report(
        report,
        protocol=protocol,
        predecessor=predecessor,
        qwen_report=qwen_report,
    )
    if report_errors != [
        "fusion retriever request depth did not match the frozen top_k contract"
    ]:
        errors.append(
            "invalid report no longer has the single pinned depth-contract failure"
        )

    mismatch = payload.get("detected_contract_mismatch") or {}
    if mismatch != {
        "frozen_retriever_top_k": 10,
        "evaluation_fixture_max_k": 20,
        "actual_retriever_request_top_k": 20,
        "frozen_candidate_multiplier": 4,
        "expected_dense_candidate_depth": 40,
        "actual_dense_candidate_depth": 80,
        "quota_1_expected_dense_tail_position": 10,
        "quota_1_actual_first_dense_tail_position": 11,
        "quota_2_expected_dense_tail_positions": [9, 10],
        "quota_2_actual_dense_tail_positions": [11, 12],
        "cause": "The benchmark passed its Recall@20 maximum K directly to fusion instead of adapting the frozen top-10 fusion layout before evaluating deeper diagnostic ranks.",
    }:
        errors.append("depth-contract failure diagnosis changed")
    observed = payload.get("observed_diagnostic_only") or {}
    if observed != {
        "reported_nominee_candidate_id": "ungated_rrf_2_1_k20_v2",
        "ungated_rrf_recall_at_10_gain_scoped": 0.0,
        "ungated_rrf_recall_at_10_gain_full_library": 0.051724,
        "protected_tail_quota_1_strict_recall_gain": False,
        "protected_tail_quota_2_strict_recall_gain": False,
    }:
        errors.append("invalid diagnostic outcome changed")
    decision = payload.get("effective_decision") or {}
    if decision != {
        "nominee_candidate_id": "",
        "confirmation_allowed": False,
        "confirmation_accessed": False,
        "frozen_acceptance_accessed": False,
        "remeasurement_allowed_under_v2": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
        "next_action": "expand_real_user_evaluation_evidence_before_any_fusion_v3",
    }:
        errors.append("effective no-promotion decision changed")
    if payload.get("report_preserved_unmodified") is not True:
        errors.append("invalid measurement report was not preserved")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_INVALIDATION.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] fusion invalidation cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_invalidation(payload)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] invalid Qwen fusion v2 execution is pinned and cannot promote")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
