from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.transformers_embedding_provider import (
    TransformersMeanPoolEmbeddingProvider,
)


DEFAULT_REGISTRY = REPO_ROOT / "docs" / "retrieval_multilingual_e5_large_instruct_candidate_v1.json"
MODEL_ID = "intfloat/multilingual-e5-large-instruct"
REVISION = "274baa43b0e13e37fafa6428dbc7938e62e5c439"
QUERY_PREFIX = (
    "Instruct: Given an academic literature question, retrieve relevant passages "
    "from the user paper library that provide evidence for answering the question\nQuery: "
)
PROVIDER_IDENTITY = "3dd860965cb586f731d13ac87efd0d473343197ca50632fae0434accd95d1c65"
BASE_MODEL_ID = "intfloat/multilingual-e5-base"
BASE_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"


def validate_candidate(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1 or payload.get("curation_status") != "frozen_before_development_measurement":
        errors.append("candidate is not frozen before development measurement")
    selected = payload.get("selected_candidate") if isinstance(payload.get("selected_candidate"), dict) else {}
    expected = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "declared_license": "mit",
        "dimension": 1024,
        "max_length": 512,
        "pooling": "attention_masked_mean",
        "normalization": "l2",
        "query_prefix": QUERY_PREFIX,
        "passage_prefix": "",
        "provider_identity": PROVIDER_IDENTITY,
        "model_safetensors_bytes": 1119825680,
        "model_safetensors_sha256": "dd6b6e4f52db0a7aff83a13d10e6c5342ef9f6ab799bad3221f4b35ef390fa85",
        "dtype": "float16",
        "device": "cuda",
        "trust_remote_code": False,
        "product_enabled": False,
    }
    for key, value in expected.items():
        if selected.get(key) != value:
            errors.append(f"selected candidate {key} changed")
    descriptor = TransformersMeanPoolEmbeddingProvider(
        model_id=MODEL_ID,
        revision=REVISION,
        dimension=1024,
        query_prefix=QUERY_PREFIX,
        passage_prefix="",
        max_length=512,
        batch_size=4,
        device="cuda",
        dtype="float16",
        allow_download=False,
    ).descriptor
    if descriptor.identity != PROVIDER_IDENTITY or selected.get("model_fingerprint") != descriptor.model_fingerprint:
        errors.append("provider encoding identity changed")
    baseline = (
        payload.get("comparison_baseline")
        if isinstance(payload.get("comparison_baseline"), dict)
        else {}
    )
    expected_baseline = {
        "model_id": BASE_MODEL_ID,
        "revision": BASE_REVISION,
        "dimension": 768,
        "max_length": 512,
        "pooling": "attention_masked_mean",
        "normalization": "l2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "model_safetensors_bytes": 1112201288,
        "model_safetensors_sha256": "a18a44fad1d0b46ded15928144138cff1135d5cc8233bdd90be5f18822de09a7",
        "dtype": "float16",
        "device": "cuda",
        "trust_remote_code": False,
    }
    for key, value in expected_baseline.items():
        if baseline.get(key) != value:
            errors.append(f"comparison baseline {key} changed")
    alternative = payload.get("rejected_alternative") if isinstance(payload.get("rejected_alternative"), dict) else {}
    if alternative.get("model_id") != "BAAI/bge-m3" or alternative.get("revision") != "5617a9f61b028005a4858fdac845db406aefb181":
        errors.append("BGE-M3 comparison identity changed")
    if alternative.get("equivalent_safetensors_primary_weight_available") is not False:
        errors.append("BGE-M3 safe-weight decision changed")
    if alternative.get("decision") != "rejected_for_this_candidate_round":
        errors.append("BGE-M3 candidate decision changed")
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
    if (payload.get("frozen_acceptance_fixture") or {}).get("maximum_candidate_replays") != 1:
        errors.append("frozen replay count changed")
    fusion = payload.get("fusion_contract") if isinstance(payload.get("fusion_contract"), dict) else {}
    if fusion != {
        "algorithm": "rrf",
        "rrf_k": 20.0,
        "sparse_weight": 2.0,
        "dense_weight": 1.0,
        "candidate_multiplier": 4,
        "protected_sparse_prefix": 5,
        "dense_source_gate_depth": 5,
        "cross_encoder": False,
    }:
        errors.append("source-safe fusion contract changed")
    gates = payload.get("development_gates") if isinstance(payload.get("development_gates"), dict) else {}
    if gates.get("raw_large_recall_at_10_min_gain_in_any_mode") != 0.05:
        errors.append("raw large development gain gate changed")
    if gates.get("safe_rrf_recall_at_10_min_gain_in_both_modes") != 0.02:
        errors.append("safe RRF development gain gate changed")
    if gates.get("index_size_ratio_vs_base_max") != 1.5 or gates.get("all_required") is not True:
        errors.append("resource or all-pass development gate changed")
    replay = payload.get("frozen_replay_gates") if isinstance(payload.get("frozen_replay_gates"), dict) else {}
    if replay.get("run_only_if_all_development_gates_pass") is not True:
        errors.append("frozen replay is not development-gated")
    if replay.get("safe_rrf_recall_at_10_min_gain_vs_product") != 0.15:
        errors.append("frozen semantic gain gate changed")
    privacy = payload.get("privacy_and_execution") if isinstance(payload.get("privacy_and_execution"), dict) else {}
    for key in (
        "normal_runtime_downloads",
        "normal_runtime_model_loading",
        "paper_text_uploaded",
        "ordinary_user_surface_exposed",
    ):
        if privacy.get(key) is not False:
            errors.append(f"privacy/execution {key} must remain false")
    if privacy.get("benchmark_download_requires_explicit_flag") is not True:
        errors.append("benchmark download consent changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] candidate registry cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_candidate(payload)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print(f"[OK] E5-large-instruct candidate frozen: {MODEL_ID}@{REVISION[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
