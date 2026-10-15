from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file


DEFAULT_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage2_e5_small_protocol_v1.json"
)
DEFAULT_UMBRELLA = REPO_ROOT / "docs" / "retrieval_lighter_runtime_screen_protocol_v1.json"
DEFAULT_STAGE1_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage1_e5_small_protocol_v1.json"
)
DEFAULT_STAGE1_REPORT = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage1_e5_small_v1.json"
DEFAULT_CONTROL = (
    REPO_ROOT / "docs" / "retrieval_user_journey_fusion_v3_1_semantic_acceptance_v2.json"
)
DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_cases_v1.json"
DEFAULT_RETRIEVER = REPO_ROOT / "kb" / "parent_collapsed_dense_retriever.py"
DEFAULT_PROTECTED_TAIL = REPO_ROOT / "kb" / "protected_dense_tail_retriever.py"
EXPECTED_PROTOCOL_SHA256 = "84647d00ef87a3aae2f4bce1208491a813c9d08f6e1baff527804a9aa05a013b"


def validate_protocol(
    payload: dict[str, Any],
    *,
    inspect_implementation_files: bool = False,
) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id") != "retrieval_lighter_runtime_stage2_e5_small_v1"
        or payload.get("status") != "frozen_before_single_full_build"
    ):
        errors.append("E5 Stage-2 protocol identity changed")
    authorization = payload.get("authorization") or {}
    if (
        authorization.get("umbrella_protocol_sha256") != sha256_file(DEFAULT_UMBRELLA)
        or authorization.get("stage_1_protocol_sha256")
        != sha256_file(DEFAULT_STAGE1_PROTOCOL)
        or authorization.get("stage_1_report_sha256") != sha256_file(DEFAULT_STAGE1_REPORT)
        or authorization.get("stage_1_result_fingerprint")
        != "0750c32c42c09f8f9613734662ff1e6c3c7bee1169c9341b51e5afed778cd645"
        or authorization.get("stage_1_decision")
        != "passed_stage_1_authorize_single_full_build"
        or authorization.get("full_builds_consumed_by_qwen") != 0
        or authorization.get("maximum_full_builds_across_batch") != 1
        or authorization.get("accepted_product_control_sha256")
        != sha256_file(DEFAULT_CONTROL)
    ):
        errors.append("E5 Stage-2 authorization changed")
    representation = payload.get("representation") or {}
    if (
        representation.get("parent_count") != 1673
        or representation.get("window_count") != 2016
        or representation.get("content_window_tokens") != 448
        or representation.get("minimum_content_overlap_tokens") != 64
        or representation.get("nominal_content_stride_tokens") != 384
        or representation.get("parent_score_collapse")
        != "maximum_window_cosine_then_parent_id"
        or representation.get("retriever_source_sha256")
        != "814d22eea98096a46d7499bb1cdcd6c890e2b6b808efaca97ef79bede77d1724"
        or representation.get("returned_result_identity")
        != "complete_immutable_parent_chunk"
        or representation.get("child_window_text_returned") is not False
        or representation.get("silent_tail_loss_allowed") is not False
    ):
        errors.append("E5 Stage-2 representation changed")
    index_format = payload.get("index_format") or {}
    if (
        index_format.get("schema_version") != 1
        or index_format.get("vectors_format") != "float32_little_endian_l2_row_major"
        or index_format.get("window_metadata_fields")
        != ["parent_id", "parent_position", "window_ordinal"]
        or index_format.get("staging_directory_then_atomic_generation_rename") is not True
        or index_format.get("atomic_current_pointer_replace") is not True
        or index_format.get("readback_shape_hash_and_parent_mapping_required") is not True
        or index_format.get("paper_text_stored_in_index") is not False
        or index_format.get("query_text_stored_in_index") is not False
    ):
        errors.append("E5 Stage-2 index format changed")
    execution = payload.get("execution") or {}
    if (
        execution.get("maximum_runs") != 1
        or execution.get("hard_subprocess_timeout_seconds") != 660
        or execution.get("raw_query_vectors_retained_in_tracked_report") is not False
        or execution.get("report_persisted_before_outcome_validation") is not True
        or execution.get("product_enabled") is not False
        or execution.get("ordinary_user_surface_exposed") is not False
    ):
        errors.append("E5 Stage-2 execution changed")
    runtime = payload.get("runtime_gates") or {}
    if (
        runtime.get("maximum_provider_load_ms") != 30000.0
        or runtime.get("maximum_full_build_ms") != 600000.0
        or runtime.get("minimum_parent_chunks_per_second") != 2.75
        or runtime.get("maximum_peak_process_rss_mib") != 6144.0
        or runtime.get("maximum_warm_query_p95_ms") != 1000.0
        or runtime.get("maximum_index_bytes") != 10000000
        or runtime.get("maximum_truncated_query_count") != 0
        or runtime.get("maximum_truncated_parent_count") != 0
        or runtime.get("maximum_uncovered_content_token_count") != 0
        or runtime.get("require_exact_parent_window_counts") is not True
        or runtime.get("require_finite_unit_vectors") is not True
        or runtime.get("require_atomic_readback") is not True
        or runtime.get("require_source_artifact_tokenizer_unchanged") is not True
        or runtime.get("all_required") is not True
    ):
        errors.append("E5 Stage-2 runtime gates changed")
    semantic = payload.get("semantic_evaluation") or {}
    if (
        semantic.get("fixture_sha256") != sha256_file(DEFAULT_FIXTURE)
        or semantic.get("case_count") != 12
        or semantic.get("modes") != ["scoped", "full_library"]
        or semantic.get("raw_dense_is_diagnostic_only") is not True
        or semantic.get("protected_tail_source_sha256")
        != "f120174d277fd190008fb0c1d5749315df83cb75fd1ca143aa864991c28d95b2"
        or semantic.get("protected_tail")
        != {
            "protected_sparse_prefix": 5,
            "sparse_tail_window": 5,
            "dense_novelty_quota": 2,
            "candidate_multiplier": 4,
            "fusion_output_depth": 10,
            "candidate_retrieval_depth": 40,
        }
        or semantic.get("minimum_protected_recall_at_10_gain_vs_product_each_mode")
        != 0.15
        or semantic.get("protected_recall_at_5_equal_to_product_each_mode") is not True
        or semantic.get("protected_wrong_source_at_5_equal_to_product_each_mode")
        is not True
        or semantic.get("protected_mrr_no_regression_vs_product_each_mode") is not True
        or semantic.get("protected_ndcg_at_10_no_regression_vs_product_each_mode")
        is not True
        or semantic.get("all_required") is not True
    ):
        errors.append("E5 Stage-2 semantic gates changed")
    decision = payload.get("decision") or {}
    boundary = payload.get("product_boundary") or {}
    if (
        decision.get("product_enablement_allowed") is not False
        or decision.get("preparation_design_allowed_only_on_pass") is not True
        or boundary.get("library_papers_only") is not True
        or boundary.get("evidence_retrieval_only") is not True
        or boundary.get("writing_or_submission_workflow_allowed") is not False
        or boundary.get("product_enabled") is not False
        or boundary.get("ordinary_user_surface_exposed") is not False
    ):
        errors.append("E5 Stage-2 product boundary changed")
    if inspect_implementation_files:
        if (
            not DEFAULT_RETRIEVER.is_file()
            or representation.get("retriever_source_sha256")
            != sha256_file(DEFAULT_RETRIEVER)
        ):
            errors.append("E5 Stage-2 retriever implementation changed")
        if (
            not DEFAULT_PROTECTED_TAIL.is_file()
            or semantic.get("protected_tail_source_sha256")
            != sha256_file(DEFAULT_PROTECTED_TAIL)
        ):
            errors.append("E5 Stage-2 protected-tail implementation changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] E5 Stage-2 protocol cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_protocol(payload)
    if sha256_file(DEFAULT_PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        errors.append("E5 Stage-2 protocol SHA-256 changed")
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] E5 Stage-2 single full-build protocol is frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
