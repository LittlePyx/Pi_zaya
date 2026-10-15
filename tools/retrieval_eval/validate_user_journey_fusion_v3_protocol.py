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


DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "retrieval_user_journey_fusion_v3_protocol.json"


def _ids_sha256(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _path_is_pinned(evidence: dict[str, Any], prefix: str) -> bool:
    path = REPO_ROOT / str(evidence.get(f"{prefix}_path") or "")
    return path.is_file() and evidence.get(f"{prefix}_sha256") == sha256_file(path)


def validate_protocol(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("id") != "retrieval_user_journey_fusion_v3_protocol"
        or payload.get("curation_status")
        != "frozen_before_fusion_v3_runner_and_measurement"
    ):
        errors.append("fusion-v3 protocol identity changed")
    evidence = payload.get("evidence") or {}
    for prefix in (
        "fixture",
        "bm25_baseline",
        "qwen_dense_protocol",
        "qwen_dense_report",
        "fusion_v2_invalidation",
        "protected_tail_implementation",
        "semantic_acceptance",
    ):
        if not _path_is_pinned(evidence, prefix):
            errors.append(f"fusion-v3 evidence identity changed: {prefix}")
    try:
        fixture = json.loads(
            (REPO_ROOT / evidence["fixture_path"]).read_text(encoding="utf-8")
        )
        dense_protocol = json.loads(
            (REPO_ROOT / evidence["qwen_dense_protocol_path"]).read_text(
                encoding="utf-8"
            )
        )
        dense_report = json.loads(
            (REPO_ROOT / evidence["qwen_dense_report_path"]).read_text(encoding="utf-8")
        )
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        return errors + [f"fusion-v3 evidence cannot be read: {type(exc).__name__}"]
    design_gate = (dense_report.get("analysis") or {}).get(
        "fusion_v3_design_condition"
    ) or {}
    if (
        design_gate.get("passes_all") is not True
        or design_gate.get("product_eligibility") is not False
    ):
        errors.append("fusion-v3 design lacks passing diagnostic evidence")
    split = fixture.get("split") or {}
    partition = payload.get("partition") or {}
    if (
        partition.get("calibration_source") != "fixture.split.calibration_case_ids"
        or partition.get("calibration_case_count") != 12
        or partition.get("ordered_calibration_case_ids_sha256")
        != _ids_sha256(list(split.get("calibration_case_ids") or []))
        or partition.get("holdout_source") != "fixture.split.holdout_case_ids"
        or partition.get("holdout_case_count") != 6
        or partition.get("ordered_holdout_case_ids_sha256")
        != _ids_sha256(list(split.get("holdout_case_ids") or []))
        or partition.get("holdout_hidden_until_calibration_passes") is not True
        or partition.get("maximum_calibration_runs") != 1
        or partition.get("maximum_holdout_runs") != 1
    ):
        errors.append("fusion-v3 partition boundary changed")
    dense_representation = dense_protocol.get("frozen_representation") or {}
    expected_representation = {
        key: dense_representation.get(key)
        for key in (
            "candidate_id",
            "model_id",
            "revision",
            "provider_identity",
            "dimension",
            "pooling",
            "padding_side",
            "query_prefix",
            "passage_prefix",
            "max_length",
            "allow_download",
            "trust_remote_code",
        )
    }
    expected_representation["existing_index_reused_without_build"] = True
    if payload.get("frozen_representation") != expected_representation:
        errors.append("fusion-v3 Qwen representation changed")
    candidate = payload.get("frozen_candidate") or {}
    expected_candidate = {
        "candidate_id": "protected_tail_dense_quota_2_v3",
        "algorithm": "protected_dense_tail",
        "protected_sparse_prefix": 5,
        "sparse_tail_window": 5,
        "dense_novelty_quota": 2,
        "candidate_multiplier": 4,
        "fusion_output_depth": 10,
        "sparse_and_dense_candidate_depth": 40,
        "dense_novelty_excludes_sparse_ranks": 10,
        "sparse_survivor_selection": "best_dense_rank_then_sparse_rank",
        "sparse_survivor_output_order": "original_sparse_rank",
        "dense_novelty_output_order": "dense_rank",
        "tail_layout": "sparse_1_to_5_then_three_dense_supported_sparse_6_to_10_then_two_dense_novelties_at_9_to_10",
        "dense_failure_fallback": "original_sparse_top_10",
        "candidate_count": 1,
        "rationale": "The calibration diagnostic found a full-library dense-only relevant passage at dense rank 2 while BM25-only evidence remained inside the protected top five.",
    }
    if candidate != expected_candidate:
        errors.append("fusion-v3 candidate changed")
    depth = payload.get("depth_contract") or {}
    if depth != {
        "selection_ranking_depth": 10,
        "selection_reported_metric_depths": [5, 10],
        "candidate_retrieval_depth": 40,
        "diagnostic_depth_20_during_selection": False,
        "post_selection_depth_20_policy": "requires_a_separate_frozen_continuation_contract_after_selection",
        "evaluator_must_not_pass_maximum_unrelated_metric_k_to_fusion": True,
    }:
        errors.append("fusion-v3 ranking/reporting depth contract changed")
    product = payload.get("product_control") or {}
    if (
        product.get("strategy")
        != "current_product_bm25_query_pipeline_at_selection_depth_10"
        or product.get("query_expansion") is not False
        or product.get("translation_behavior_unchanged") is not True
        or product.get("must_match_frozen_baseline_projection_for")
        != [
            "recall@5",
            "recall@10",
            "mrr",
            "ndcg@10",
            "contract_recall@10",
            "wrong_source_rate@5",
        ]
    ):
        errors.append("fusion-v3 product control changed")
    required_checks = [
        "recall@5_equal_to_product",
        "wrong_source_rate@5_equal_to_product",
        "recall@10_no_regression",
        "mrr_no_regression",
        "ndcg@10_no_regression",
        "contract_recall@10_no_regression",
        "wrong_source_rate@10_no_regression",
    ]
    calibration = payload.get("calibration_gate") or {}
    holdout = payload.get("holdout_gate") or {}
    combined = payload.get("combined_gate") or {}
    if (
        calibration.get("required_in_both_modes") != required_checks
        or calibration.get("strict_recall@10_gain_in_at_least_one_mode") is not True
        or calibration.get("all_required") is not True
        or calibration.get("failure_action")
        != "stop_without_holdout_semantic_or_product_access"
    ):
        errors.append("fusion-v3 calibration gate changed")
    if holdout != {
        "run_only_after_calibration_passes": True,
        "candidate_configuration_immutable": True,
        "same_checks_as_calibration": True,
        "strict_recall@10_gain_in_at_least_one_mode": True,
        "no_retry_or_substitution": True,
    }:
        errors.append("fusion-v3 holdout gate changed")
    if (
        combined.get("required_after_calibration_and_holdout") is not True
        or combined.get("recall@10_minimum_gain_in_at_least_one_mode") != 0.02
        or combined.get("wrong_source_rate@5_and_10_no_regression") is not True
    ):
        errors.append("fusion-v3 combined gate changed")
    semantic = payload.get("semantic_acceptance_gate") or {}
    if semantic != {
        "run_only_after_combined_gate_passes": True,
        "maximum_runs": 1,
        "recall@10_minimum_gain_vs_product_in_both_modes": 0.15,
        "preserve_product_recall@5": True,
        "preserve_product_wrong_source_rate@5": True,
    }:
        errors.append("fusion-v3 semantic acceptance gate changed")
    if payload.get("safety_boundary") != {
        "model_prompt_or_index_variants": 0,
        "additional_fusion_candidates": 0,
        "cross_encoder_runs": 0,
        "external_web_or_model_knowledge_for_labels": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }:
        errors.append("fusion-v3 safety boundary changed")
    return errors


def main() -> int:
    try:
        protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] fusion-v3 protocol cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_protocol(protocol)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] one-candidate fusion-v3 protocol frozen before runner and measurement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
