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
from tools.retrieval_eval.validate_candidate_comparison_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)
from tools.retrieval_eval.transformers_embedding_provider import (
    TransformersPoolingEmbeddingProvider,
)


DEFAULT_BATCH = REPO_ROOT / "docs" / "retrieval_candidate_batch_v1.json"


def _contract_sha(candidate: dict[str, Any]) -> str:
    contract = {
        "hf_revision": candidate.get("revision"),
        "pooling": candidate.get("pooling"),
        "query_prefix": candidate.get("query_prefix"),
        "passage_prefix": candidate.get("passage_prefix"),
        "max_length": candidate.get("benchmark_max_length"),
        "dtype": candidate.get("dtype"),
        "padding_side": candidate.get("padding_side"),
    }
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_batch(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1 or payload.get("curation_status") != "frozen_before_screening":
        errors.append("candidate batch is not frozen before screening")
    try:
        protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"comparison protocol cannot be read: {type(exc).__name__}"]
    errors.extend(validate_protocol(protocol))
    protocol_ref = payload.get("protocol") or {}
    if protocol_ref.get("sha256") != sha256_file(DEFAULT_PROTOCOL):
        errors.append("comparison protocol identity changed")
    candidates = list(payload.get("candidates") or [])
    if payload.get("candidate_count") != 2 or len(candidates) != 2:
        errors.append("candidate batch cardinality changed")
    expected = {
        "qwen3_embedding_0_6b_v1": {
            "model_id": "Qwen/Qwen3-Embedding-0.6B",
            "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            "declared_license": "apache-2.0",
            "dimension": 1024,
            "benchmark_max_length": 8192,
            "pooling": "last_token",
            "padding_side": "left",
            "query_prefix": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
            "passage_prefix": "",
            "model_safetensors_bytes": 1191586416,
            "model_safetensors_sha256": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
            "minimum_transformers_version": "4.51.0",
            "batch_size": 1,
        },
        "granite_embedding_278m_multilingual_v1": {
            "model_id": "ibm-granite/granite-embedding-278m-multilingual",
            "revision": "a9cb5338491faf32b73dd17b714a31821c021bbf",
            "declared_license": "apache-2.0",
            "dimension": 768,
            "benchmark_max_length": 512,
            "pooling": "cls",
            "padding_side": "right",
            "query_prefix": "",
            "passage_prefix": "",
            "model_safetensors_bytes": 556109872,
            "model_safetensors_sha256": "b522423ae15d1a6fd48c55ef4f599c0c2c0737603da94e4b4815760c2c06ebac",
            "minimum_transformers_version": "4.15.0",
            "batch_size": 4,
        },
    }
    budget = protocol["candidate_budget"]
    seen: set[str] = set()
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id in seen or candidate_id not in expected:
            errors.append("candidate identity or uniqueness changed")
            continue
        seen.add(candidate_id)
        for key, value in expected[candidate_id].items():
            if candidate.get(key) != value:
                errors.append(f"{candidate_id} {key} changed")
        if candidate.get("encoding_contract_sha256") != _contract_sha(candidate):
            errors.append(f"{candidate_id} encoding contract hash changed")
        descriptor = TransformersPoolingEmbeddingProvider(
            model_id=candidate["model_id"],
            revision=candidate["revision"],
            dimension=candidate["dimension"],
            query_prefix=candidate["query_prefix"],
            passage_prefix=candidate["passage_prefix"],
            max_length=candidate["benchmark_max_length"],
            batch_size=candidate["batch_size"],
            device=candidate["device"],
            dtype=candidate["dtype"],
            pooling=candidate["pooling"],
            padding_side=candidate["padding_side"],
        ).descriptor
        if descriptor.model_fingerprint != (
            "hf-contract-sha256:" + str(candidate.get("encoding_contract_sha256") or "")
        ):
            errors.append(f"{candidate_id} provider fingerprint changed")
        if (
            candidate.get("model_safetensors_bytes", 0)
            > budget["maximum_safetensors_bytes_each"]
            or candidate.get("dimension", 0) > budget["maximum_embedding_dimension"]
        ):
            errors.append(f"{candidate_id} exceeds resource budget")
        if (
            candidate.get("standard_auto_model") is not True
            or candidate.get("trust_remote_code") is not False
            or candidate.get("screening_runs") != 0
            or candidate.get("product_enabled") is not False
        ):
            errors.append(f"{candidate_id} execution boundary changed")
    excluded = {str(row.get("model_id") or ""): str(row.get("reason_code") or "") for row in payload.get("excluded_alternatives") or []}
    for model_id in (
        "Snowflake/snowflake-arctic-embed-m-v2.0",
        "Alibaba-NLP/gte-multilingual-base",
        "jinaai/jina-embeddings-v3",
        "nomic-ai/nomic-embed-text-v2-moe",
        "BAAI/bge-m3",
        "google/embeddinggemma-300m",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "intfloat/multilingual-e5-large",
    ):
        if not excluded.get(model_id):
            errors.append(f"excluded alternative missing: {model_id}")
    runtime = payload.get("benchmark_runtime") or {}
    requirements_path = REPO_ROOT / str(runtime.get("requirements_path") or "")
    expected_runtime = {
        "requirements_path": "tools/retrieval_eval/candidate_runtime_requirements_v1.txt",
        "requirements_sha256": sha256_file(requirements_path),
        "python_version": "3.9.7",
        "torch_version": "2.3.1+cu121",
        "transformers_version": "4.51.3",
        "tokenizers_version": "0.21.4",
        "huggingface_hub_version": "0.30.2",
        "safetensors_version": "0.5.3",
        "virtual_environment": True,
        "inherits_system_torch": False,
        "isolated_torch": True,
        "gpu_contract": "NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, compute capability 8.6",
        "product_requirements_modified": False,
    }
    if runtime != expected_runtime:
        errors.append("benchmark runtime contract changed")
    boundary = payload.get("execution_boundary") or {}
    for key in (
        "screening_fixture_accessed_before_freeze",
        "confirmation_fixture_accessed_by_candidates",
        "frozen_acceptance_fixture_accessed",
        "measurement_downloads",
        "paper_text_uploaded",
        "ordinary_user_surface_exposed",
        "product_enabled",
    ):
        if boundary.get(key) is not False:
            errors.append(f"execution boundary {key} must remain false")
    if boundary.get("model_download_requires_explicit_prepare") is not True:
        errors.append("model preparation consent boundary changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_BATCH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] candidate batch cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_batch(payload)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] two-candidate dense screening batch frozen before measurement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
