from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file


DEFAULT_REGISTRY = REPO_ROOT / "docs" / "retrieval_parent_collapsed_multivector_candidate_v1.json"


def validate_candidate(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("schema version changed")
    if payload.get("id") != "parent_collapsed_multivector_e5_v1":
        errors.append("candidate ID changed")
    if payload.get("curation_status") != "frozen_before_development_measurement":
        errors.append("candidate was not frozen before development measurement")
    for key, expected_id, expected_count, tuning in (
        ("development_fixture", "research_qa_library_development_v1_chunk_qrels", 41, True),
        ("frozen_acceptance_fixture", "research_qa_semantic_paraphrase_v1", 12, False),
    ):
        fixture = payload.get(key) if isinstance(payload.get(key), dict) else {}
        path = REPO_ROOT / str(fixture.get("path") or "")
        if not path.is_file() or fixture.get("sha256") != sha256_file(path):
            errors.append(f"{key} identity or SHA-256 changed")
        if fixture.get("id") != expected_id or fixture.get("case_count") != expected_count:
            errors.append(f"{key} case contract changed")
        if fixture.get("allowed_for_tuning") is not tuning:
            errors.append(f"{key} tuning policy changed")
    frozen = payload.get("frozen_acceptance_fixture") or {}
    if frozen.get("maximum_candidate_replays") != 1:
        errors.append("frozen acceptance replay count must remain one")

    candidate = payload.get("embedding_candidate") if isinstance(payload.get("embedding_candidate"), dict) else {}
    expected_candidate = {
        "model_id": "intfloat/multilingual-e5-base",
        "revision": "d128750597153bb5987e10b1c3493a34e5a4502a",
        "provider_identity": "b6f461d0da1959a877bc465919b54dd2121149e360a8c2c973ddfcda1e0ae99b",
        "encoder_max_length": 512,
        "dimension": 768,
        "dtype": "float16",
        "device": "cuda",
        "allow_download": False,
        "trust_remote_code": False,
        "product_enabled": False,
    }
    for key, value in expected_candidate.items():
        if candidate.get(key) != value:
            errors.append(f"embedding candidate {key} changed")

    representation = payload.get("representation_contract") if isinstance(payload.get("representation_contract"), dict) else {}
    expected_representation = {
        "child_window_unit": "tokenizer_content_token",
        "fast_tokenizer_offsets_required": True,
        "child_window_tokens": 256,
        "child_stride_tokens": 192,
        "child_overlap_tokens": 64,
        "final_child_may_be_shorter": True,
        "minimum_tail_child_tokens": 65,
        "minimum_parent_tokens_for_children": 321,
        "heading_injection": False,
        "maximum_children_per_parent": None,
        "parent_score": "maximum cosine score across full parent and all child representations",
        "tie_break": "original parent corpus order",
        "returned_id": "original parent chunk id",
        "returned_text": "original parent chunk text",
        "returned_metadata": "original parent chunk metadata",
        "storage_chunks_changed": False,
        "page_locators_changed": False,
    }
    for key, value in expected_representation.items():
        if representation.get(key) != value:
            errors.append(f"representation contract {key} changed")
    if representation.get("child_window_tokens", 0) - representation.get("child_stride_tokens", 0) != representation.get("child_overlap_tokens"):
        errors.append("child overlap arithmetic changed")

    fusion = payload.get("fusion_contract") if isinstance(payload.get("fusion_contract"), dict) else {}
    expected_fusion = {
        "algorithm": "rrf",
        "rrf_k": 20.0,
        "sparse_weight": 2.0,
        "dense_weight": 1.0,
        "candidate_multiplier": 4,
        "protected_sparse_prefix": 5,
        "dense_source_gate_depth": 5,
        "cross_encoder": False,
    }
    for key, value in expected_fusion.items():
        if fusion.get(key) != value:
            errors.append(f"fusion contract {key} changed")

    development = payload.get("development_gates") if isinstance(payload.get("development_gates"), dict) else {}
    if development.get("raw_multivector_recall_at_10_min_gain_in_any_mode") != 0.02:
        errors.append("development raw Dense gain gate changed")
    if development.get("safe_rrf_recall_at_10_min_gain_in_any_mode") != 0.02:
        errors.append("development safe RRF gain gate changed")
    if development.get("representation_count_multiplier_max") != 2.0:
        errors.append("representation multiplier gate changed")
    if development.get("all_required") is not True:
        errors.append("development gates must all be required")
    replay = payload.get("frozen_replay_gates") if isinstance(payload.get("frozen_replay_gates"), dict) else {}
    if replay.get("run_only_if_all_development_gates_pass") is not True:
        errors.append("frozen replay is not development-gated")
    if replay.get("safe_rrf_recall_at_10_min_gain_vs_product") != 0.15:
        errors.append("frozen Recall@10 gate changed")
    if replay.get("required_modes") != ["scoped", "full_library"]:
        errors.append("frozen required modes changed")
    if replay.get("all_required") is not True:
        errors.append("frozen gates must all be required")
    execution = payload.get("execution_policy") if isinstance(payload.get("execution_policy"), dict) else {}
    if execution.get("development_measurement_runs") != 1:
        errors.append("development run count changed")
    for key in ("normal_runtime_downloads", "normal_runtime_model_loading", "ordinary_user_surface_exposed"):
        if execution.get(key) is not False:
            errors.append(f"execution policy {key} must remain false")
    return errors


def main() -> int:
    path = DEFAULT_REGISTRY
    if not path.is_file():
        print(f"[FAIL] missing candidate registry: {path}")
        return 2
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] invalid candidate registry: {type(exc).__name__}")
        return 2
    errors = validate_candidate(payload)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] parent-collapsed multi-vector candidate frozen before measurement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
