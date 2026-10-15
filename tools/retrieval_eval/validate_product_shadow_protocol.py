from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "retrieval_product_shadow_protocol_v1.json"


def sha256_file_variants(path: Path) -> set[str]:
    data = path.read_bytes()
    lf = data.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    return {
        hashlib.sha256(candidate).hexdigest()
        for candidate in (data, lf, crlf)
    }


def validate_protocol(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id") != "retrieval_product_shadow_v1"
    ):
        errors.append("product shadow protocol identity changed")
    parent = payload.get("parent_evidence") or {}
    for path_key, sha_key in (
        ("semantic_acceptance_path", "semantic_acceptance_sha256"),
        ("fusion_protocol_path", "fusion_protocol_sha256"),
        ("retriever_path", "retriever_sha256"),
    ):
        path = REPO_ROOT / str(parent.get(path_key) or "")
        if not path.is_file() or parent.get(sha_key) not in sha256_file_variants(path):
            errors.append(f"product shadow parent evidence changed: {path_key}")
    if (
        parent.get("semantic_result_fingerprint")
        != ("a3c81c2805cd887011219b68a33e6bbf3ed8688f5209b8483263e1a41ce2847a")
        or parent.get("semantic_decision") != "passed_for_product_shadow"
    ):
        errors.append("product shadow semantic eligibility changed")

    boundary = payload.get("product_boundary") or {}
    expected_boundary = {
        "enabled_by_default": False,
        "activation_environment_variable": "KB_DENSE_SHADOW_ENABLED",
        "required_activation_value": "1",
        "requires_explicit_index_path": True,
        "index_path_environment_variable": "KB_DENSE_SHADOW_INDEX_DIR",
        "requires_local_observability_path": True,
        "changes_product_ranking": False,
        "changes_answer_evidence": False,
        "exposes_user_surface": False,
        "allows_external_network": False,
        "allows_index_build": False,
        "allows_raw_query_or_evidence_telemetry": False,
    }
    if boundary != expected_boundary:
        errors.append("product shadow safety boundary changed")

    candidate = payload.get("candidate") or {}
    expected_candidate = {
        "candidate_id": "qwen3_embedding_0_6b_v1",
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "provider_identity": "55229348d5bbe4b1e1935aeb8ffe57cc03c527685b1d08d44f11a56a54525cb8",
        "dimension": 1024,
        "pooling": "last_token",
        "padding_side": "left",
        "query_prefix": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        "passage_prefix": "",
        "max_length": 8192,
        "dtype": "float16",
        "device": "cuda",
        "batch_size": 1,
        "allow_download": False,
        "trust_remote_code": False,
        "fusion": {
            "strategy": "protected_dense_tail",
            "output_depth": 10,
            "candidate_depth": 40,
            "candidate_multiplier": 4,
            "protected_sparse_prefix": 5,
            "sparse_tail_window": 5,
            "dense_novelty_quota": 2,
        },
    }
    if candidate != expected_candidate:
        errors.append("product shadow accepted candidate changed")

    execution = payload.get("execution") or {}
    if (
        execution.get("mode") != "asynchronous_fail_open_shadow"
        or execution.get("max_in_flight") != 1
    ):
        errors.append("product shadow concurrency boundary changed")
    if (
        execution.get("failure_policy")
        != "record_redacted_category_and_leave_product_output_unchanged"
    ):
        errors.append("product shadow fail-open policy changed")
    if execution.get("index_policy") != "load_existing_matching_snapshot_only":
        errors.append("product shadow read-only index policy changed")

    telemetry = payload.get("telemetry") or {}
    fields = set(telemetry.get("fields") or [])
    if (
        telemetry.get("event_name") != "pi_zaya.retrieval_shadow"
        or telemetry.get("content_free") is not True
    ):
        errors.append("product shadow telemetry identity changed")
    if {"query", "prompt", "evidence", "source_path", "source_title", "model"} & fields:
        errors.append("product shadow telemetry admits raw content")

    gate = payload.get("controlled_operational_gate") or {}
    expected_gate = {
        "minimum_completed_queries": 18,
        "required_completion_rate": 1.0,
        "required_product_output_unchanged_rate": 1.0,
        "required_protected_prefix_preserved_rate": 1.0,
        "maximum_source_escape_count": 0,
        "maximum_warm_shadow_p95_ms": 250.0,
        "maximum_foreground_submission_p95_ms": 5.0,
        "fail_open_tests_required": True,
        "privacy_replay_required": True,
        "product_enablement_on_pass": False,
    }
    if gate != expected_gate:
        errors.append("product shadow operational gate changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] product shadow protocol cannot be read: {type(exc).__name__}")
        return 1
    errors = validate_protocol(payload)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[OK] product shadow protocol is frozen and product-disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
