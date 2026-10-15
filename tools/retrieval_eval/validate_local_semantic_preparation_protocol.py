from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.local_semantic_preparation_contract import LOCAL_SEMANTIC_STATES
from kb.retrieval_eval_fixture import sha256_file


DEFAULT_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_local_semantic_preparation_protocol_v1.json"
)
DEFAULT_STAGE2_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage2_e5_small_protocol_v1.json"
)
DEFAULT_STAGE2_REPORT = (
    REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage2_e5_small_v1.json"
)
EXPECTED_PROTOCOL_SHA256 = "75416104e64b461106be8161630adf471dd1eda8524229135ea9bedb54775a06"
EXPECTED_FILES = {
    "onnx/model.onnx": (
        470268510,
        "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665",
    ),
    "onnx/config.json": (
        653,
        "bbb7c1333fc4b3e27fbc9cd5d2070aabcc1d4dfb99917c3633e772f97545a6b6",
    ),
    "onnx/sentencepiece.bpe.model": (
        5069051,
        "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    ),
    "onnx/special_tokens_map.json": (
        167,
        "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7",
    ),
    "onnx/tokenizer.json": (
        17082730,
        "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    ),
    "onnx/tokenizer_config.json": (
        443,
        "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b",
    ),
}
EXPECTED_ACTIONS = {
    "unavailable_in_build": "update_app",
    "not_prepared": "prepare",
    "downloading": "cancel",
    "verifying": "cancel",
    "ready_to_index": "resume",
    "indexing": "cancel",
    "cancelling": "wait",
    "interrupted": "resume",
    "stale": "rebuild",
    "repairable_error": "retry",
    "ready": "none",
}


def validate_protocol(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id") != "retrieval_local_semantic_preparation_v1"
        or payload.get("status") != "frozen_before_product_preparation_implementation"
    ):
        errors.append("local semantic preparation protocol identity changed")
    authorization = payload.get("authorization") or {}
    if (
        authorization.get("stage_2_protocol_sha256")
        != sha256_file(DEFAULT_STAGE2_PROTOCOL)
        or authorization.get("stage_2_report_sha256")
        != sha256_file(DEFAULT_STAGE2_REPORT)
        or authorization.get("stage_2_result_fingerprint")
        != "18fdba85d35503d2f5f60e38191df412482ec7fc1783341b4354fa0f37848a05"
        or authorization.get("stage_2_decision")
        != "runtime_candidate_eligible_for_optional_preparation_design"
    ):
        errors.append("local semantic preparation authorization changed")
    candidate = payload.get("candidate") or {}
    if (
        candidate.get("candidate_id")
        != "multilingual_e5_small_official_onnx_v1"
        or candidate.get("model_revision")
        != "03415a4be176a1620747c692ed433219fabc3def"
        or candidate.get("execution_provider") != "CPUExecutionProvider"
        or candidate.get("embedding_dimension") != 384
        or candidate.get("content_window_tokens") != 448
        or candidate.get("minimum_content_overlap_tokens") != 64
        or candidate.get("nominal_content_stride_tokens") != 384
        or candidate.get("parent_score_collapse")
        != "maximum_window_cosine_then_parent_id"
        or candidate.get("returned_result_identity")
        != "complete_immutable_parent_chunk"
    ):
        errors.append("local semantic candidate contract changed")
    artifacts = payload.get("artifacts") or {}
    observed_files = {
        str(item.get("repository_path") or ""): (
            item.get("bytes"),
            item.get("sha256"),
        )
        for item in artifacts.get("files") or []
        if isinstance(item, dict)
    }
    if (
        observed_files != EXPECTED_FILES
        or artifacts.get("maximum_total_bytes") != sum(item[0] for item in EXPECTED_FILES.values())
        or artifacts.get("download_requires_explicit_user_action") is not True
        or artifacts.get("automatic_query_time_download_allowed") is not False
        or artifacts.get("allowed_scheme") != "https"
        or artifacts.get("allowed_host") != "huggingface.co"
        or artifacts.get("immutable_revision_required_in_url") is not True
        or artifacts.get("resume_uses_validated_range_response_only") is not True
        or artifacts.get("sha256_checked_before_atomic_publish") is not True
        or artifacts.get("unverified_file_never_becomes_current") is not True
    ):
        errors.append("local semantic artifact integrity contract changed")
    dependencies = payload.get("runtime_dependencies") or {}
    if (
        dependencies.get("onnxruntime_version") != "1.23.2"
        or dependencies.get("tokenizers_version") != "0.21.4"
        or dependencies.get("network_install_at_preparation_time_allowed") is not False
        or dependencies.get("missing_dependencies_state") != "unavailable_in_build"
        or dependencies.get("mode_control_visible_when_missing") is not False
    ):
        errors.append("local semantic runtime dependency contract changed")
    index = payload.get("library_index") or {}
    if (
        index.get("root") != "<db_dir>/local_semantic_index/e5_small_v1"
        or index.get("shared_with_legacy_qwen_dense_index") is not False
        or index.get("window_metadata_fields")
        != ["parent_id", "parent_position", "window_ordinal"]
        or index.get("paper_text_stored_in_index") is not False
        or index.get("query_text_stored_in_index") is not False
        or index.get("staging_directory_then_atomic_generation_rename") is not True
        or index.get("atomic_current_pointer_replace") is not True
        or index.get("previous_ready_generation_preserved_until_commit") is not True
        or index.get("corpus_change_during_build_publishes_new_index") is not False
    ):
        errors.append("local semantic index atomicity contract changed")
    lifecycle = payload.get("lifecycle") or {}
    if (
        tuple(lifecycle.get("states") or []) != LOCAL_SEMANTIC_STATES
        or lifecycle.get("restart_reconciliation")
        != {
            "downloading": "interrupted",
            "verifying": "interrupted",
            "indexing": "interrupted",
            "cancelling": "interrupted",
        }
        or lifecycle.get("cancel_retains_verified_files") is not True
        or lifecycle.get("cancel_preserves_previous_ready_index") is not True
        or lifecycle.get("single_active_task_per_library") is not True
        or lifecycle.get("duplicate_prepare_returns_existing_task") is not True
    ):
        errors.append("local semantic lifecycle contract changed")
    public = payload.get("public_contract") or {}
    if (
        public.get("path_or_raw_exception_exposed") is not False
        or public.get("remediation_by_state") != EXPECTED_ACTIONS
        or public.get("mode_control_visible_only_when_ready") is not True
        or public.get("bm25_remains_default") is not True
        or public.get("raw_dense_mode_exposed") is not False
        or public.get("external_web_or_model_consent_is_separate") is not True
    ):
        errors.append("local semantic public remediation contract changed")
    gates = payload.get("promotion_gates") or {}
    boundary = payload.get("product_boundary") or {}
    if (
        gates
        != {
            "contract_validator_and_mutation_tests_pass": True,
            "artifact_download_resume_cancel_hash_tests_pass": True,
            "atomic_index_cancel_stale_repair_tests_pass": True,
            "production_shadow_replay_passes": True,
            "expanded_judged_failure_slices_pass": True,
            "portable_windows_runtime_dependency_gate_passes": True,
            "ordinary_user_surface_allowed_before_all_gates": False,
        }
        or boundary.get("library_papers_only") is not True
        or boundary.get("evidence_retrieval_only") is not True
        or boundary.get("writing_or_submission_workflow_allowed") is not False
        or boundary.get("product_enabled") is not False
        or boundary.get("ordinary_user_surface_exposed") is not False
    ):
        errors.append("local semantic promotion or product boundary changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] local semantic preparation protocol cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_protocol(payload)
    if sha256_file(DEFAULT_PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        errors.append("local semantic preparation protocol SHA-256 changed")
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] local semantic preparation lifecycle protocol is frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
