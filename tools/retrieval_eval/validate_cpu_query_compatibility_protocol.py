from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "retrieval_cpu_query_compatibility_protocol_v3.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_protocol(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1 or payload.get("protocol_id") != "retrieval_cpu_query_compatibility_v3":
        errors.append("CPU query compatibility protocol identity changed")
    predecessor = payload.get("predecessor") or {}
    predecessor_path = REPO_ROOT / str(predecessor.get("protocol_path") or "")
    invalidation_path = REPO_ROOT / str(predecessor.get("invalidation_path") or "")
    if (
        predecessor
        != {
            "protocol_path": "docs/retrieval_cpu_query_compatibility_protocol_v2.json",
            "protocol_sha256": "e524378d1be22de93c3de4a77949c8dd80dc8e35c99786598a27367aed1eaf37",
            "invalidation_path": "docs/retrieval_cpu_query_compatibility_execution_invalidation_v2.json",
            "invalidation_sha256": "7b574e7f2233e926d261256e12e4dedae0eb773172adbf1a546f19d71ee5c3eb",
            "attempted_measurement_runs": 1,
            "usable_outcomes": 0,
            "quality_metrics_observed_by_operator": False,
            "candidate_or_gate_changes": 0,
        }
        or not predecessor_path.is_file()
        or predecessor.get("protocol_sha256") != _sha256(predecessor_path)
        or not invalidation_path.is_file()
        or predecessor.get("invalidation_sha256") != _sha256(invalidation_path)
    ):
        errors.append("CPU query predecessor or zero-query invalidation changed")
    parent = payload.get("accepted_parent") or {}
    parent_path = REPO_ROOT / str(parent.get("report_path") or "")
    if not parent_path.is_file() or parent.get("report_sha256") != _sha256(parent_path):
        errors.append("accepted parent report changed")
    if parent.get("result_fingerprint") != "a3c81c2805cd887011219b68a33e6bbf3ed8688f5209b8483263e1a41ce2847a":
        errors.append("accepted parent result changed")
    fixture = payload.get("fixture") or {}
    fixture_path = REPO_ROOT / str(fixture.get("path") or "")
    if (
        not fixture_path.is_file()
        or fixture.get("sha256") != _sha256(fixture_path)
        or fixture.get("case_count") != 12
        or fixture.get("modes") != ["scoped", "full_library"]
    ):
        errors.append("CPU query fixture changed")
    index = payload.get("read_only_index") or {}
    if index != {
        "relative_path": "test_results/retrieval_user_journey_fusion_v3_1_semantic/qwen_index_v1",
        "tree_sha256": "fef412fcf632b98c8a5dbbe7223e9986996045b105b7f2a371d30b28c25da2da",
        "tree_file_count": 5,
        "tree_size_bytes": 7277202,
        "build_id": "21e9718aa59f483988f3e88571a3c01c",
        "corpus_fingerprint": "19dd55b81535593219b0fdb8744b6c74cdfe51d68c3567dfd822780be927ec7a",
        "current_pointer_sha256": "440a80aa69338582b263463a69d8eb4a5161e80e0b3a1d03831c76b686a4a332",
        "manifest_sha256": "9ad735efacdbbe9d4d6c8c33ad5605a7939e1558a622300a6e0c1eeab02544d2",
        "vectors_sha256": "04bd9d7904695538004d6cd52f77c974fd16e3cada22e8a4fa0bd994541b32d8",
        "document_embedding_runtime": "cuda_float16_parent",
        "writes_allowed": False,
        "build_allowed": False,
    }:
        errors.append("CPU query read-only index boundary changed")
    candidate = payload.get("cpu_query_candidate") or {}
    if candidate != {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "provider_identity": "a5bb526562ff581e073496f8fc14077e119373c5051a70a78aec7306770434e9",
        "device": "cpu",
        "dtype": "float32",
        "pooling": "last_token",
        "padding_side": "left",
        "max_length": 8192,
        "allow_download": False,
        "trust_remote_code": False,
    }:
        errors.append("CPU query candidate changed")
    if payload.get("fusion") != {
        "strategy": "protected_dense_tail",
        "protected_sparse_prefix": 5,
        "sparse_tail_window": 5,
        "dense_novelty_quota": 2,
        "candidate_multiplier": 4,
        "ranking_depth": 10,
    }:
        errors.append("CPU query fusion changed")
    gate = payload.get("predeclared_gate") or {}
    if gate != {
        "minimum_semantic_recall_at_10_gain_each_mode": 0.15,
        "require_recall_at_5_equal_to_product": True,
        "require_wrong_source_at_5_equal_to_product": True,
        "require_no_regression_vs_parent_fusion_metrics": [
            "recall@5",
            "recall@10",
            "mrr",
            "ndcg@10",
        ],
        "require_wrong_source_at_5_no_worse_than_parent": True,
        "maximum_provider_load_ms": 30000.0,
        "maximum_warm_query_embedding_p95_ms": 1000.0,
        "required_query_completion_rate": 1.0,
        "require_index_tree_unchanged": True,
    }:
        errors.append("CPU query gate changed")
    execution = payload.get("execution") or {}
    if (
        execution.get("prior_attempted_measurement_runs") != 1
        or execution.get("prior_usable_outcomes") != 0
        or execution.get("maximum_recovery_runs") != 1
        or execution.get("raw_report_persist_before_validation") is not True
        or execution.get("unique_query_count") != 12
        or execution.get("no_document_embeddings_created") is not True
        or execution.get("no_product_index_changes") is not True
        or execution.get("product_enabled") is not False
        or execution.get("ordinary_user_surface_exposed") is not False
        or execution.get("pass_decision") != "passed_for_cpu_full_index_feasibility"
        or execution.get("reject_decision") != "rejected_cpu_qwen_query_runtime"
    ):
        errors.append("CPU query execution boundary changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] CPU query protocol cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_protocol(payload)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[OK] CPU query compatibility v3 recovery is frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
