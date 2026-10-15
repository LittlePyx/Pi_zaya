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


DEFAULT_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_user_journey_qwen_dense_protocol_v1.json"
)


def _ids_sha256(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _path_is_pinned(evidence: dict[str, Any], prefix: str) -> bool:
    path = REPO_ROOT / str(evidence.get(f"{prefix}_path") or "")
    return path.is_file() and evidence.get(f"{prefix}_sha256") == sha256_file(path)


def validate_protocol(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("id")
        != "retrieval_user_journey_qwen_dense_diagnostic_protocol_v1"
        or payload.get("curation_status")
        != "frozen_before_diagnostic_implementation_and_measurement"
    ):
        errors.append("Qwen dense diagnostic protocol identity changed")
    evidence = payload.get("evidence") or {}
    for prefix in (
        "fixture",
        "bm25_baseline",
        "bm25_outcome",
        "qwen_predecessor_report",
        "candidate_batch",
        "fusion_v2_invalidation",
    ):
        if not _path_is_pinned(evidence, prefix):
            errors.append(f"Qwen diagnostic evidence identity changed: {prefix}")
    try:
        fixture = json.loads(
            (REPO_ROOT / evidence["fixture_path"]).read_text(encoding="utf-8")
        )
        qwen = json.loads(
            (REPO_ROOT / evidence["qwen_predecessor_report_path"]).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        return errors + [
            f"Qwen diagnostic evidence cannot be read: {type(exc).__name__}"
        ]
    split = fixture.get("split") or {}
    partition = payload.get("partition") or {}
    calibration_ids = list(split.get("calibration_case_ids") or [])
    holdout_ids = list(split.get("holdout_case_ids") or [])
    if (
        partition.get("source") != "fixture.split.calibration_case_ids"
        or partition.get("case_count") != 12
        or partition.get("ordered_case_ids_sha256") != _ids_sha256(calibration_ids)
        or partition.get("holdout_case_count") != 6
        or partition.get("ordered_holdout_case_ids_sha256") != _ids_sha256(holdout_ids)
        or partition.get("holdout_access_allowed") is not False
        or partition.get("semantic_acceptance_access_allowed") is not False
    ):
        errors.append("Qwen diagnostic calibration/holdout boundary changed")
    measured = qwen.get("candidate") or {}
    representation = payload.get("frozen_representation") or {}
    expected_representation = {
        key: measured.get(key)
        for key in (
            "candidate_id",
            "model_id",
            "revision",
            "model_safetensors_sha256",
            "provider_identity",
            "dimension",
            "pooling",
            "padding_side",
            "query_prefix",
            "passage_prefix",
            "max_length",
            "batch_size",
            "allow_download",
            "trust_remote_code",
        )
    }
    expected_representation.update(
        {
            "parent_chunks_unchanged": True,
            "query_translation_or_expansion": False,
        }
    )
    if representation != expected_representation:
        errors.append("Qwen diagnostic representation changed")
    predecessor_index = qwen.get("index") or {}
    status = predecessor_index.get("status") or {}
    expected_index = {
        "relative_path": "test_results/retrieval_candidate_screening_v1/qwen3_embedding_0_6b_v1/index_v1",
        "build_id": status.get("build_id"),
        "corpus_fingerprint": status.get("corpus_fingerprint"),
        "document_count": status.get("document_count"),
        "chunk_count": status.get("chunk_count"),
        "provider_identity": status.get("provider_identity"),
        "size_bytes": predecessor_index.get("size_bytes"),
        "rebuild_allowed": False,
    }
    if payload.get("existing_index") != expected_index:
        errors.append("Qwen diagnostic existing-index contract changed")
    contract = payload.get("diagnostic_contract") or {}
    if contract != {
        "channels": [
            "bm25_product_queries_frozen_control",
            "qwen_dense_single_vector",
        ],
        "modes": ["scoped", "full_library"],
        "retrieval_ranking_depth": 20,
        "reported_metric_depths": [5, 10, 20],
        "query_embedding_cache": True,
        "expected_unique_query_embeddings": 12,
        "maximum_runs": 1,
        "no_fusion_or_reranking": True,
        "no_case_ids_or_qrels_available_to_retriever": True,
        "report_is_diagnostic_only": True,
    }:
        errors.append("Qwen dense diagnostic depth or execution contract changed")
    analysis = payload.get("predeclared_analysis") or {}
    if (
        analysis.get("comparison_depth") != 10
        or analysis.get("classify_each_calibration_case_as")
        != ["both_hit", "bm25_only", "dense_only", "neither"]
        or (analysis.get("fusion_v3_may_be_designed_only_if") or {}).get(
            "dense_only_case_count_minimum_in_either_mode"
        )
        != 2
        or (analysis.get("fusion_v3_may_be_designed_only_if") or {}).get(
            "dense_recall_at_10_strict_gain_in_at_least_one_mode"
        )
        is not True
        or analysis.get("fusion_v3_eligibility_is_not_product_eligibility") is not True
    ):
        errors.append("Qwen diagnostic predeclared analysis changed")
    safety = payload.get("safety_boundary") or {}
    if safety != {
        "model_or_prompt_variants": 0,
        "index_builds": 0,
        "holdout_runs": 0,
        "fusion_candidates": 0,
        "cross_encoder_runs": 0,
        "external_web_or_model_knowledge_for_labels": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }:
        errors.append("Qwen diagnostic safety boundary changed")
    return errors


def main() -> int:
    try:
        protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] Qwen diagnostic protocol cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_protocol(protocol)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] calibration-only Qwen dense diagnostic frozen before measurement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
