from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file


DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "retrieval_lighter_runtime_screen_protocol_v1.json"


def validate_protocol(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1 or payload.get("protocol_id") != "retrieval_lighter_runtime_screen_v1":
        errors.append("lighter runtime protocol identity changed")
    if payload.get("status") != "frozen_before_artifact_download":
        errors.append("lighter runtime protocol is not frozen before download")

    authorization = payload.get("authorization") or {}
    parent_path = REPO_ROOT / str(authorization.get("parent_report_path") or "")
    if authorization != {
        "parent_report_path": "docs/retrieval_cpu_padding_diagnostic_v1.json",
        "parent_report_sha256": "af1a6fc67929af03ba3c1afaf3fe3a3c37330796343f8d1d251df8c57596973a",
        "parent_result_fingerprint": "58ecfe98fc10f3e628320e870879d6b091f741a1e4c2069153be264467b78658",
        "parent_decision": "skip_native_qwen_and_screen_lighter_runtime",
    } or not parent_path.is_file() or authorization.get("parent_report_sha256") != sha256_file(parent_path):
        errors.append("lighter runtime authorization changed")

    boundary = payload.get("product_boundary") or {}
    if boundary != {
        "library_papers_only": True,
        "evidence_retrieval_only": True,
        "writing_or_submission_workflow_allowed": False,
        "core_release_requirements_modified": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }:
        errors.append("lighter runtime product boundary changed")

    host = payload.get("host_contract") or {}
    if host.get("platform") != "windows_x64" or host.get("required_instruction_set") != "AVX2" or host.get("observed_instruction_set") != "AVX2" or host.get("release_python_version") != "3.10.11":
        errors.append("lighter runtime host contract changed")

    runtime = payload.get("runtime_contract") or {}
    if runtime != {
        "python_version": "3.10.11",
        "onnxruntime_version": "1.23.2",
        "execution_provider": "CPUExecutionProvider",
        "intra_op_threads": 8,
        "inter_op_threads": 1,
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "torch_allowed": False,
        "transformers_allowed": False,
        "optimum_allowed": False,
        "remote_code_allowed": False,
        "isolated_user_site": True,
        "product_requirements_modified": False,
    }:
        errors.append("lighter runtime dependency contract changed")

    budget = payload.get("candidate_budget") or {}
    if budget != {
        "maximum_candidates": 2,
        "evaluation_order_is_binding": True,
        "stop_after_first_stage_1_pass": True,
        "maximum_stage_1_runs_per_candidate": 1,
        "maximum_stage_2_full_builds_across_batch": 1,
        "maximum_total_primary_artifact_bytes_if_both_needed": 1100396298,
        "download_only_current_candidate": True,
    }:
        errors.append("lighter runtime candidate budget changed")

    candidates = list(payload.get("candidates") or [])
    expected = {
        "qwen3_embedding_0_6b_official_quint8_avx2_onnx_v1": {
            "order": 1,
            "model_id": "Qwen/Qwen3-Embedding-0.6B",
            "revision": "bd58e9fd4b0467770da53678d8da25481c0f959b",
            "path": "onnx/model_quint8_avx2.onnx",
            "bytes": 600127788,
            "sha256": "a7eda29cc374b01ae75eb1af1ab65c3c61a64c43f39f16e08c6e37c075dcbf4d",
            "dimension": 1024,
            "maximum_length": 8192,
        },
        "multilingual_e5_small_official_onnx_v1": {
            "order": 2,
            "model_id": "intfloat/multilingual-e5-small",
            "revision": "03415a4be176a1620747c692ed433219fabc3def",
            "path": "onnx/model.onnx",
            "bytes": 470268510,
            "sha256": "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665",
            "dimension": 384,
            "maximum_length": 512,
        },
    }
    if len(candidates) != 2:
        errors.append("lighter runtime candidate cardinality changed")
    seen: set[str] = set()
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        expected_candidate = expected.get(candidate_id)
        if not expected_candidate or candidate_id in seen:
            errors.append("lighter runtime candidate identity or uniqueness changed")
            continue
        seen.add(candidate_id)
        artifact = candidate.get("primary_artifact") or {}
        tokenizer = candidate.get("tokenizer") or {}
        encoding = candidate.get("encoding") or {}
        if (
            candidate.get("evaluation_order") != expected_candidate["order"]
            or candidate.get("model_id") != expected_candidate["model_id"]
            or candidate.get("model_revision") != expected_candidate["revision"]
            or artifact.get("path") != expected_candidate["path"]
            or artifact.get("bytes") != expected_candidate["bytes"]
            or artifact.get("sha256") != expected_candidate["sha256"]
            or encoding.get("dimension") != expected_candidate["dimension"]
            or tokenizer.get("maximum_length") != expected_candidate["maximum_length"]
            or not str(artifact.get("url") or "").startswith("https://huggingface.co/")
        ):
            errors.append(f"{candidate_id} frozen identity changed")
    if seen != set(expected):
        errors.append("lighter runtime candidate set changed")

    stage_0 = payload.get("stage_0_preflight") or {}
    for key in (
        "artifact_sha256_and_size_required",
        "onnx_parse_and_checker_required",
        "graph_inputs_outputs_recorded",
        "cpu_provider_only_required",
        "dependency_tree_and_installed_bytes_recorded",
        "fresh_python_3_10_11_environment_required",
        "system_and_user_site_packages_forbidden",
    ):
        if stage_0.get(key) is not True:
            errors.append(f"lighter runtime stage 0 gate changed: {key}")
    if stage_0.get("model_inference_allowed") is not False or stage_0.get("paper_text_access_allowed") is not False:
        errors.append("lighter runtime stage 0 execution boundary changed")

    stage_1 = payload.get("stage_1_bounded_screen") or {}
    if (
        stage_1.get("fixture_sha256") != sha256_file(REPO_ROOT / stage_1.get("fixture_path", ""))
        or stage_1.get("query_count") != 12
        or stage_1.get("stress_document_count") != 64
        or stage_1.get("maximum_wall_clock_seconds_per_candidate") != 600
        or stage_1.get("maximum_projected_full_build_ms") != 600000.0
        or stage_1.get("maximum_peak_process_rss_mib") != 4096.0
        or stage_1.get("all_applicable_gates_required") is not True
    ):
        errors.append("lighter runtime stage 1 gates changed")

    stage_2 = payload.get("stage_2_single_winner") or {}
    if (
        stage_2.get("requires_stage_1_pass") is not True
        or stage_2.get("maximum_full_builds_across_batch") != 1
        or stage_2.get("maximum_full_build_ms") != 600000.0
        or stage_2.get("minimum_throughput_chunks_per_second") != 2.75
        or stage_2.get("maximum_peak_process_rss_mib") != 6144.0
        or stage_2.get("same_semantic_quality_gates_as_accepted_parent") is not True
        or stage_2.get("atomic_publish_and_readback_required") is not True
    ):
        errors.append("lighter runtime stage 2 gates changed")

    excluded = {str(row.get("route") or "") for row in payload.get("excluded_routes") or []}
    if excluded != {
        "native_pytorch_qwen_cpu",
        "length_aware_native_qwen_cpu",
        "e5_base_parent_collapsed_multivector",
        "cross_encoder_productization",
    }:
        errors.append("lighter runtime excluded routes changed")

    execution = payload.get("execution") or {}
    if execution != {
        "artifact_downloads": 0,
        "model_inference_runs": 0,
        "full_index_builds": 0,
        "report_path": "docs/retrieval_lighter_runtime_screen_v1.json",
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }:
        errors.append("lighter runtime execution boundary changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] lighter runtime protocol cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_protocol(payload)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] two-candidate ONNX screen is frozen before artifact download")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
