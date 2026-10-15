from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file


DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "retrieval_qwen_fusion_protocol_v2.json"


def _ordered_ids_sha256(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _path_contract(row: dict[str, Any], path_key: str, sha_key: str) -> bool:
    path = REPO_ROOT / str(row.get(path_key) or "")
    return path.is_file() and row.get(sha_key) == sha256_file(path)


def validate_protocol(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("fusion protocol schema changed")
    if payload.get("curation_status") != "frozen_before_implementation_and_measurement":
        errors.append(
            "fusion protocol was not frozen before implementation and measurement"
        )

    predecessor = payload.get("predecessor_evidence") or {}
    path_pairs = (
        ("candidate_protocol_path", "candidate_protocol_sha256"),
        ("candidate_batch_path", "candidate_batch_sha256"),
        ("qwen_screening_report_path", "qwen_screening_report_sha256"),
        ("zero_nomination_report_path", "zero_nomination_report_sha256"),
    )
    for path_key, sha_key in path_pairs:
        if not _path_contract(predecessor, path_key, sha_key):
            errors.append(f"predecessor identity changed: {path_key}")

    try:
        predecessor_protocol = json.loads(
            (REPO_ROOT / predecessor["candidate_protocol_path"]).read_text(
                encoding="utf-8"
            )
        )
        qwen_report = json.loads(
            (REPO_ROOT / predecessor["qwen_screening_report_path"]).read_text(
                encoding="utf-8"
            )
        )
        nomination = json.loads(
            (REPO_ROOT / predecessor["zero_nomination_report_path"]).read_text(
                encoding="utf-8"
            )
        )
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        return errors + [f"predecessor evidence cannot be read: {type(exc).__name__}"]

    boundary = payload.get("evidence_boundary") or {}
    if not _path_contract(
        boundary, "development_fixture_path", "development_fixture_sha256"
    ):
        errors.append("development fixture identity changed")
    if not _path_contract(
        boundary, "frozen_acceptance_path", "frozen_acceptance_sha256"
    ):
        errors.append("frozen acceptance identity changed")
    tuning_ids = list(
        predecessor_protocol.get("inner_split", {}).get("screening_case_ids") or []
    )
    confirmation_ids = list(
        predecessor_protocol.get("inner_split", {}).get("confirmation_case_ids") or []
    )
    if (
        boundary.get("tuning_case_count") != 29
        or boundary.get("ordered_tuning_case_ids_sha256")
        != _ordered_ids_sha256(tuning_ids)
        or boundary.get("confirmation_case_count") != 12
        or boundary.get("ordered_confirmation_case_ids_sha256")
        != _ordered_ids_sha256(confirmation_ids)
    ):
        errors.append("tuning or confirmation partition changed")
    if (
        boundary.get("confirmation_hidden_until_one_tuning_nominee") is not True
        or boundary.get("frozen_acceptance_allowed_for_tuning") is not False
    ):
        errors.append("hidden evidence boundary weakened")

    representation = payload.get("frozen_representation") or {}
    measured = qwen_report.get("candidate") or {}
    expected_representation = {
        "candidate_id": measured.get("candidate_id"),
        "model_id": measured.get("model_id"),
        "revision": measured.get("revision"),
        "model_safetensors_sha256": measured.get("model_safetensors_sha256"),
        "provider_identity": measured.get("provider_identity"),
        "dimension": measured.get("dimension"),
        "pooling": measured.get("pooling"),
        "padding_side": measured.get("padding_side"),
        "query_prefix": measured.get("query_prefix"),
        "passage_prefix": measured.get("passage_prefix"),
        "max_length": measured.get("max_length"),
        "parent_chunks_unchanged": True,
        "query_translation_and_expansion_unchanged": True,
        "cross_encoder": False,
        "additional_model_variants": 0,
        "additional_index_builds": 0,
    }
    if representation != expected_representation:
        errors.append("frozen Qwen representation changed")
    decision = nomination.get("decision") or {}
    if (
        decision.get("nominee_candidate_id")
        or decision.get("confirmation_allowed") is not False
    ):
        errors.append("predecessor zero-nominee boundary changed")

    shared = payload.get("shared_fusion_contract") or {}
    expected_shared = {
        "top_k": 10,
        "candidate_multiplier": 4,
        "protected_sparse_prefix": 5,
        "protected_prefix_order_unchanged": True,
        "dense_failure_falls_back_to_sparse": True,
        "deduplicate_by_chunk_identity": True,
        "no_case_ids_or_qrels_available_to_fusion": True,
        "no_source_gate_for_new_dense_tail_evidence": True,
        "product_enabled": False,
    }
    if shared != expected_shared:
        errors.append("shared fusion safety contract changed")

    candidates = payload.get("frozen_candidates") or []
    expected_candidates = [
        ("ungated_rrf_2_1_k20_v2", "rrf", None, 0),
        ("protected_tail_dense_quota_1_v2", "protected_dense_tail", 1, None),
        ("protected_tail_dense_quota_2_v2", "protected_dense_tail", 2, None),
    ]
    observed = [
        (
            row.get("candidate_id"),
            row.get("algorithm"),
            row.get("dense_novelty_quota"),
            row.get("dense_source_gate_depth"),
        )
        for row in candidates
    ]
    if observed != expected_candidates:
        errors.append("frozen fusion candidate set changed")
    if candidates:
        rrf = candidates[0]
        if (rrf.get("sparse_weight"), rrf.get("dense_weight"), rrf.get("rrf_k")) != (
            2.0,
            1.0,
            20.0,
        ):
            errors.append("ungated RRF parameters changed")
    for row in candidates[1:]:
        if (
            row.get("sparse_tail_window") != 5
            or row.get("dense_candidate_depth") != 40
            or row.get("dense_novelty_excludes_sparse_top") != 10
            or row.get("sparse_survivor_selection")
            != "best_dense_rank_then_sparse_rank"
            or row.get("sparse_survivor_output_order") != "original_sparse_rank"
            or row.get("dense_novelty_output_order") != "dense_rank"
        ):
            errors.append(f"protected-tail contract changed: {row.get('candidate_id')}")

    tuning = payload.get("tuning_screen") or {}
    required_checks = [
        "recall@5_equal_to_product_in_both_modes",
        "wrong_source_rate@5_equal_to_product_in_both_modes",
        "recall@10_no_regression_vs_product_in_both_modes",
        "mrr_no_regression_vs_product_in_both_modes",
        "ndcg@10_no_regression_vs_product_in_both_modes",
        "strict_recall@10_gain_in_at_least_one_mode",
    ]
    if (
        tuning.get("maximum_runs") != 1
        or tuning.get("all_three_candidates_in_one_run") is not True
        or tuning.get("data") != "tuning_partition_only"
        or tuning.get("required_checks") != required_checks
        or tuning.get("all_required") is not True
        or tuning.get("screening_result_is_tuning_evidence_only") is not True
    ):
        errors.append("tuning screen boundary changed")

    nomination_contract = payload.get("nomination") or {}
    if (
        nomination_contract.get("maximum_nominees") != 1
        or nomination_contract.get("eligible_candidates_only") is not True
    ):
        errors.append("nomination boundary changed")
    confirmation = payload.get("confirmation") or {}
    if (
        confirmation.get("maximum_runs") != 1
        or confirmation.get("nominee_only") is not True
        or confirmation.get("data") != "confirmation_partition_only"
        or confirmation.get("required_checks") != required_checks
        or confirmation.get("all_required") is not True
        or confirmation.get("no_retry_or_candidate_substitution_after_result")
        is not True
    ):
        errors.append("confirmation boundary changed")

    combined = payload.get("combined_development_gate") or {}
    if (
        combined.get("recall@10_min_gain_in_both_modes") != 0.02
        or combined.get("all_required") is not True
    ):
        errors.append("combined development gate changed")
    acceptance = payload.get("frozen_acceptance_gate") or {}
    if (
        acceptance.get("maximum_runs") != 1
        or acceptance.get("run_only_after_all_prior_gates_pass") is not True
        or acceptance.get("recall@10_min_gain_vs_product_in_both_modes") != 0.15
        or acceptance.get("preserve_product_recall@5") is not True
        or acceptance.get("preserve_product_wrong_source_rate@5") is not True
    ):
        errors.append("frozen acceptance gate changed")
    failure = payload.get("failure_policy") or {}
    if failure != {
        "candidate_configuration_changes_after_any_tuning_result": False,
        "rejected_candidate_retry_under_v2": False,
        "confirmation_access_without_nominee": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
        "next_failure_action": "expand_real_user_evaluation_evidence_before_any_fusion_v3",
    }:
        errors.append("failure or product exposure policy changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] fusion protocol cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_protocol(payload)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] Qwen fusion protocol v2 frozen before implementation and measurement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
