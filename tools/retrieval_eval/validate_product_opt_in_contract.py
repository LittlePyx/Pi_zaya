from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = REPO_ROOT / "docs" / "retrieval_product_opt_in_contract_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_contract(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1 or payload.get("contract_id") != "retrieval_product_opt_in_v1":
        errors.append("product opt-in contract identity changed")

    boundary = payload.get("product_boundary") or {}
    if boundary != {
        "purpose": "retrieve_traceable_evidence_from_papers_already_in_the_user_library",
        "external_web_content_allowed": False,
        "uncited_model_knowledge_allowed": False,
        "writing_or_submission_workflow_allowed": False,
        "embedding_model_contributes_content": False,
    }:
        errors.append("product boundary changed")

    evidence = payload.get("accepted_evidence") or {}
    for path_key, hash_key in (
        ("semantic_report_path", "semantic_report_sha256"),
        ("operational_report_path", "operational_report_sha256"),
    ):
        path = REPO_ROOT / str(evidence.get(path_key) or "")
        if not path.is_file() or evidence.get(hash_key) != _sha256(path):
            errors.append(f"accepted product evidence changed: {path_key}")
    if evidence.get("semantic_result_fingerprint") != "a3c81c2805cd887011219b68a33e6bbf3ed8688f5209b8483263e1a41ce2847a":
        errors.append("semantic result fingerprint changed")
    if evidence.get("operational_result_fingerprint") != "b63ec97dc7b5562c1c85e4c532a442725d4b55842caa9a172343dd94992600d0":
        errors.append("operational result fingerprint changed")

    request = payload.get("request_contract") or {}
    if request != {
        "field": "retrieval_mode",
        "accepted_values": ["bm25", "hybrid_local"],
        "default": "bm25",
        "unknown_value_policy": "normalize_to_bm25",
        "scope": "per_generation_request",
        "external_supplement_mode_independent": True,
    }:
        errors.append("request contract changed")

    candidate = payload.get("hybrid_candidate") or {}
    if candidate != {
        "provider_identity": "55229348d5bbe4b1e1935aeb8ffe57cc03c527685b1d08d44f11a56a54525cb8",
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "fusion": "protected_dense_tail_5_3_2",
        "output_depth": 10,
        "candidate_depth": 40,
        "product_index_location": "db_dir/dense_index",
        "test_artifact_dependency_allowed": False,
    }:
        errors.append("accepted hybrid candidate changed")

    readiness = payload.get("readiness_contract") or {}
    if readiness != {
        "public_states": ["ready", "unavailable"],
        "required_for_ready": [
            "optional_runtime_dependencies_importable",
            "pinned_model_revision_cached_locally",
            "committed_dense_index_present",
            "index_matches_current_corpus_and_provider",
            "vectorized_exact_shortlist_available",
        ],
        "readiness_check_downloads_model": False,
        "readiness_check_builds_index": False,
        "public_payload_exposes_paths": False,
        "public_payload_exposes_exception_text": False,
    }:
        errors.append("readiness contract changed")

    runtime = payload.get("runtime_contract") or {}
    if runtime != {
        "generation_request_downloads_model": False,
        "generation_request_builds_or_refreshes_index": False,
        "hybrid_failure_policy": "return_original_bm25_ranking",
        "fallback_blocks_answer": False,
        "fallback_requires_user_repair": False,
        "source_scope_must_be_preserved": True,
        "sparse_prefix_must_be_preserved": True,
        "top_k_above_output_depth_policy": "append_original_bm25_remainder_without_duplicates",
    }:
        errors.append("runtime fail-open contract changed")

    experience = payload.get("user_experience_contract") or {}
    if (
        experience.get("show_hybrid_choice_only_when_ready") is not True
        or experience.get("show_unavailable_hybrid_dead_option") is not False
        or experience.get("fallback_disclosure") != "本轮已自动使用 BM25 完成，无需用户修复。"
        or experience.get("persist_choice_per_conversation") is not True
        or experience.get("reset_hidden_or_unavailable_choice_to_bm25") is not True
    ):
        errors.append("user experience contract changed")

    rollout = payload.get("rollout_boundary") or {}
    if rollout != {
        "broad_availability_requires_system_managed_preparation": True,
        "implicit_dependency_install_allowed": False,
        "implicit_model_download_allowed": False,
        "implicit_index_build_allowed": False,
        "ordinary_user_control_may_ship_before_preparation_workflow": False,
    }:
        errors.append("rollout boundary changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] product opt-in contract cannot be read: {type(exc).__name__}")
        return 1
    errors = validate_contract(payload)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[OK] local hybrid product opt-in boundary is frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
