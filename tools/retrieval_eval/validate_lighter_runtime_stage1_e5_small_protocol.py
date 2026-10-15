from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.analyze_e5_small_window_census import (
    DEFAULT_REPORT as DEFAULT_CENSUS_REPORT,
)
from tools.retrieval_eval.validate_e5_small_window_census_protocol import (
    DEFAULT_PROTOCOL as DEFAULT_CENSUS_PROTOCOL,
)
from tools.retrieval_eval.validate_lighter_runtime_stage0_e5_small import (
    DEFAULT_REPORT as DEFAULT_STAGE0_REPORT,
)


DEFAULT_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage1_e5_small_protocol_v1.json"
)
DEFAULT_UMBRELLA = REPO_ROOT / "docs" / "retrieval_lighter_runtime_screen_protocol_v1.json"
DEFAULT_STRESS_MANIFEST = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stress_manifest_v1.json"
DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_cases_v1.json"
EXPECTED_PROTOCOL_SHA256 = "01309083feb769ef99cfe61a0dd94b44b193c772e540bc5740c6f7ac31f5ad39"


def validate_protocol(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id")
        != "retrieval_lighter_runtime_stage1_e5_small_v1"
        or payload.get("status") != "frozen_before_stage_1_inference"
    ):
        errors.append("E5-small Stage-1 protocol identity changed")

    authorization = payload.get("authorization") or {}
    if (
        authorization.get("umbrella_protocol_sha256") != sha256_file(DEFAULT_UMBRELLA)
        or authorization.get("stage_0_report_sha256") != sha256_file(DEFAULT_STAGE0_REPORT)
        or authorization.get("window_census_protocol_sha256")
        != sha256_file(DEFAULT_CENSUS_PROTOCOL)
        or authorization.get("window_census_report_sha256")
        != sha256_file(DEFAULT_CENSUS_REPORT)
        or authorization.get("window_census_result_fingerprint")
        != "9ff2c8f4dc9f81ade0cbae5286ade071a1d0e8fcfeaecc9cae6ba6f33b33c3cf"
        or authorization.get("window_census_decision")
        != "window_census_passed_freeze_stage_1"
        or authorization.get("stress_manifest_sha256")
        != sha256_file(DEFAULT_STRESS_MANIFEST)
    ):
        errors.append("E5-small Stage-1 authorization changed")

    candidate = payload.get("candidate") or {}
    if (
        candidate.get("candidate_id") != "multilingual_e5_small_official_onnx_v1"
        or candidate.get("model_revision") != "03415a4be176a1620747c692ed433219fabc3def"
        or candidate.get("artifact_bytes") != 470268510
        or candidate.get("artifact_sha256")
        != "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665"
        or candidate.get("onnx_inputs")
        != ["input_ids", "attention_mask", "token_type_ids"]
        or candidate.get("onnx_output") != "last_hidden_state"
        or candidate.get("embedding_dimension") != 384
    ):
        errors.append("E5-small Stage-1 candidate changed")

    tokenizer = payload.get("tokenizer") or {}
    if (
        tokenizer.get("tokenizer_json_sha256")
        != "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"
        or tokenizer.get("padding_side") != "right"
        or tokenizer.get("bos_token_id") != 0
        or tokenizer.get("pad_token_id") != 1
        or tokenizer.get("eos_token_id") != 2
        or tokenizer.get("maximum_model_tokens") != 512
        or tokenizer.get("local_files_only") is not True
    ):
        errors.append("E5-small Stage-1 tokenizer changed")

    encoding = payload.get("encoding") or {}
    if (
        encoding.get("content_window_tokens") != 448
        or encoding.get("minimum_content_overlap_tokens") != 64
        or encoding.get("nominal_content_stride_tokens") != 384
        or encoding.get("pooling")
        != "attention_masked_mean_over_last_hidden_state"
        or encoding.get("normalization") != "l2_float32"
        or encoding.get("full_parent_content_coverage_required") is not True
        or encoding.get("silent_tail_loss_allowed") is not False
        or encoding.get("parent_score_collapse")
        != "maximum_window_cosine_then_parent_id"
        or encoding.get("returned_result_identity")
        != "complete_immutable_parent_chunk"
    ):
        errors.append("E5-small Stage-1 encoding changed")

    runtime = payload.get("runtime") or {}
    if (
        runtime.get("python_version") != "3.10.11"
        or runtime.get("onnxruntime_version") != "1.23.2"
        or runtime.get("tokenizers_version") != "0.21.4"
        or runtime.get("execution_provider") != "CPUExecutionProvider"
        or runtime.get("intra_op_threads") != 8
        or runtime.get("inter_op_threads") != 1
        or runtime.get("torch_allowed") is not False
        or runtime.get("transformers_allowed") is not False
        or runtime.get("remote_code_allowed") is not False
        or runtime.get("user_site_allowed") is not False
    ):
        errors.append("E5-small Stage-1 runtime changed")

    measurement = payload.get("measurement") or {}
    if (
        measurement.get("semantic_fixture_sha256") != sha256_file(DEFAULT_FIXTURE)
        or measurement.get("public_warmup_count") != 1
        or measurement.get("timed_query_count") != 12
        or measurement.get("stress_parent_count") != 64
        or measurement.get("stress_window_count") != 74
        or measurement.get("full_parent_count") != 1673
        or measurement.get("full_window_count") != 2016
        or measurement.get("maximum_batch_size") != 8
        or measurement.get("maximum_padded_tokens_per_batch") != 4096
        or measurement.get("embed_only_stress_windows") is not True
        or measurement.get("tokenize_full_corpus_for_exact_projection") is not True
        or measurement.get("quality_evaluation_allowed") is not False
        or measurement.get("full_index_build_allowed") is not False
    ):
        errors.append("E5-small Stage-1 measurement changed")

    gates = payload.get("operational_gates") or {}
    if (
        gates.get("maximum_subprocess_wall_clock_seconds") != 600
        or gates.get("maximum_provider_load_ms") != 30000.0
        or gates.get("maximum_warm_query_p95_ms") != 1000.0
        or gates.get("maximum_peak_process_rss_mib") != 4096.0
        or gates.get("maximum_projected_full_build_ms") != 600000.0
        or gates.get("minimum_projected_parent_chunks_per_second") != 2.75
        or gates.get("maximum_input_tokens") != 512
        or gates.get("maximum_truncated_query_count") != 0
        or gates.get("maximum_truncated_parent_count") != 0
        or gates.get("maximum_uncovered_content_token_count") != 0
        or gates.get("require_exact_window_counts") is not True
        or gates.get("require_finite_unit_vectors") is not True
        or gates.get("require_source_artifact_tokenizer_unchanged") is not True
        or gates.get("all_required") is not True
    ):
        errors.append("E5-small Stage-1 gates changed")

    execution = payload.get("execution") or {}
    if (
        execution.get("maximum_runs") != 1
        or execution.get("hard_timeout_required") is not True
        or execution.get("raw_vectors_retained_in_tracked_report") is not False
        or execution.get("report_persisted_before_outcome_validation") is not True
        or execution.get("product_enabled") is not False
        or execution.get("ordinary_user_surface_exposed") is not False
    ):
        errors.append("E5-small Stage-1 execution changed")

    stage2 = payload.get("stage_2_if_authorized") or {}
    if (
        stage2.get("maximum_full_builds_across_candidate_batch") != 1
        or stage2.get("complete_window_count") != 2016
        or stage2.get("maximum_full_build_ms") != 600000.0
        or stage2.get("minimum_parent_chunks_per_second") != 2.75
        or stage2.get("maximum_peak_process_rss_mib") != 6144.0
        or stage2.get("maximum_warm_query_p95_ms") != 1000.0
        or stage2.get("maximum_index_bytes") != 10000000
        or stage2.get("evaluate_raw_dense") is not True
        or stage2.get("evaluate_protected_dense_tail") is not True
        or stage2.get("protected_dense_tail_implementation_unchanged") is not True
        or stage2.get("minimum_protected_recall_at_10_gain_vs_product_each_mode")
        != 0.15
        or stage2.get("protected_recall_at_5_equal_to_product_each_mode") is not True
        or stage2.get("protected_wrong_source_at_5_equal_to_product_each_mode")
        is not True
        or stage2.get("protected_mrr_no_regression_vs_product_each_mode") is not True
        or stage2.get("protected_ndcg_at_10_no_regression_vs_product_each_mode")
        is not True
        or stage2.get("source_corpus_unchanged_required") is not True
    ):
        errors.append("E5-small Stage-2 authorization boundary changed")

    boundary = payload.get("product_boundary") or {}
    if (
        boundary.get("library_papers_only") is not True
        or boundary.get("evidence_retrieval_only") is not True
        or boundary.get("writing_or_submission_workflow_allowed") is not False
        or boundary.get("product_enabled") is not False
        or boundary.get("ordinary_user_surface_exposed") is not False
    ):
        errors.append("E5-small Stage-1 product boundary changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] E5-small Stage-1 protocol cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_protocol(payload)
    if sha256_file(DEFAULT_PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        errors.append("E5-small Stage-1 protocol SHA-256 changed")
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] E5-small Stage-1 protocol is frozen before model inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
