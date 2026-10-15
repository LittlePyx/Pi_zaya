from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file


DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage1_qwen_protocol_v1.json"


def _ref_matches(payload: dict[str, Any], path_key: str, sha_key: str) -> bool:
    path = REPO_ROOT / str(payload.get(path_key) or "")
    return path.is_file() and payload.get(sha_key) == sha256_file(path)


def validate_protocol(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id") != "retrieval_lighter_runtime_stage1_qwen_v1"
        or payload.get("status") != "frozen_before_tokenizer_load_or_first_onnx_session_run"
    ):
        errors.append("lighter runtime stage-1 protocol identity changed")
    authorization = payload.get("authorization") or {}
    for path_key, sha_key in (
        ("umbrella_protocol_path", "umbrella_protocol_sha256"),
        ("pointer_invalidation_path", "pointer_invalidation_sha256"),
        ("stage_0_report_path", "stage_0_report_sha256"),
        ("stress_manifest_path", "stress_manifest_sha256"),
    ):
        if not _ref_matches(authorization, path_key, sha_key):
            errors.append(f"lighter runtime stage-1 authorization changed: {path_key}")
    if authorization.get("stage_0_decision") != "passed_stage_0":
        errors.append("lighter runtime stage-1 stage-0 decision changed")

    candidate = payload.get("candidate") or {}
    if (
        candidate.get("candidate_id")
        != "qwen3_embedding_0_6b_official_quint8_avx2_onnx_v1"
        or candidate.get("artifact_bytes") != 600127788
        or candidate.get("artifact_sha256")
        != "a7eda29cc374b01ae75eb1af1ab65c3c61a64c43f39f16e08c6e37c075dcbf4d"
        or candidate.get("onnx_output") != "sentence_embedding"
        or candidate.get("dimension") != 1024
    ):
        errors.append("lighter runtime stage-1 candidate contract changed")

    tokenizer = payload.get("tokenizer") or {}
    if (
        tokenizer.get("revision") != "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
        or tokenizer.get("tokenizer_json_sha256")
        != "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"
        or tokenizer.get("local_files_only") is not True
        or tokenizer.get("padding_side") != "left"
        or tokenizer.get("padding_token_id") != 151643
        or tokenizer.get("maximum_length") != 8192
    ):
        errors.append("lighter runtime stage-1 tokenizer contract changed")

    runtime = payload.get("runtime") or {}
    if (
        runtime.get("python_version") != "3.10.11"
        or runtime.get("requirements_sha256")
        != sha256_file(REPO_ROOT / str(runtime.get("requirements_path") or ""))
        or runtime.get("onnxruntime_version") != "1.23.2"
        or runtime.get("execution_provider") != "CPUExecutionProvider"
        or runtime.get("intra_op_threads") != 8
        or runtime.get("torch_allowed") is not False
        or runtime.get("transformers_allowed") is not False
        or runtime.get("user_site_allowed") is not False
    ):
        errors.append("lighter runtime stage-1 runtime contract changed")

    source = payload.get("source_corpus") or {}
    parent = payload.get("read_only_parent_index") or {}
    if (
        source.get("runtime_db_argument_required") is not True
        or source.get("tree_sha256")
        != "deb60a97bdfc50b0273de07ebe1c775352ffb2c5e4adeb3e925d48f7f25cafaf"
        or source.get("chunk_count") != 1673
        or source.get("total_effective_qwen_tokens_from_parent_diagnostic") != 489070
        or source.get("writes_allowed") is not False
    ):
        errors.append("lighter runtime stage-1 source corpus changed")
    if (
        parent.get("relative_path")
        != "test_results/retrieval_cpu_full_index_feasibility_v1/index_v1"
        or parent.get("tree_sha256")
        != "8c8fa74b7ce72c30687933b0f41d7e6b399615cf4d3a271313c0bfb85e169b50"
        or parent.get("build_id") != "7138dcaa15c242bd88bc14042166e879"
        or parent.get("vectors_sha256")
        != "bd037e035ab993126887a951f1d0aa63780e554dbfcdc85efb2f8f457874bc2e"
        or parent.get("writes_allowed") is not False
    ):
        errors.append("lighter runtime stage-1 parent index changed")

    measurement = payload.get("measurement") or {}
    if (
        measurement.get("unique_semantic_query_count") != 12
        or measurement.get("stress_document_count") != 64
        or measurement.get("embed_only_stress_manifest_documents") is not True
        or measurement.get("maximum_batch_size") != 8
        or measurement.get("maximum_padded_tokens_per_batch") != 4096
        or measurement.get("projection_method")
        != "stress_embedding_ms_times_full_corpus_attention_work_proxy_over_stress_attention_work_proxy"
        or measurement.get("document_vector_parity")
        != {"minimum_mean_cosine": 0.98, "minimum_worst_cosine": 0.95}
    ):
        errors.append("lighter runtime stage-1 measurement changed")
    quality = payload.get("quality") or {}
    if (
        quality.get("fixture_sha256")
        != sha256_file(REPO_ROOT / str(quality.get("fixture_path") or ""))
        or quality.get("accepted_parent_sha256")
        != sha256_file(REPO_ROOT / str(quality.get("accepted_parent_path") or ""))
        or quality.get("minimum_recall_at_10_gain_each_mode") != 0.15
        or quality.get("all_required") is not True
    ):
        errors.append("lighter runtime stage-1 quality gates changed")
    gates = payload.get("operational_gates") or {}
    if (
        gates.get("maximum_subprocess_wall_clock_seconds") != 600
        or gates.get("maximum_provider_load_ms") != 30000.0
        or gates.get("maximum_warm_query_p95_ms") != 1000.0
        or gates.get("maximum_peak_process_rss_mib") != 4096.0
        or gates.get("maximum_projected_full_build_ms") != 600000.0
        or gates.get("minimum_projected_throughput_chunks_per_second") != 2.75
        or gates.get("maximum_truncated_stress_document_count") != 0
        or gates.get("all_required") is not True
    ):
        errors.append("lighter runtime stage-1 operational gates changed")
    execution = payload.get("execution") or {}
    if (
        execution.get("maximum_runs") != 1
        or execution.get("raw_report_persist_before_validation") is not True
        or execution.get("product_enabled") is not False
        or execution.get("ordinary_user_surface_exposed") is not False
        or execution.get("pass_decision") != "passed_stage_1_authorize_single_full_build"
        or execution.get("reject_decision") != "rejected_stage_1"
    ):
        errors.append("lighter runtime stage-1 execution boundary changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] lighter runtime stage-1 protocol cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_protocol(payload)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] corrected Qwen ONNX stage-1 protocol is frozen before inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
