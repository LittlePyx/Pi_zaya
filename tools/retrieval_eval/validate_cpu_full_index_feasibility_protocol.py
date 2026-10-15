from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_cpu_full_index_feasibility_protocol_v1.json"
)


def _sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def validate_protocol(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id") != "retrieval_cpu_full_index_feasibility_v1"
    ):
        errors.append("CPU full-index protocol identity changed")
    authorization = payload.get("authorization") or {}
    query_report = REPO_ROOT / str(authorization.get("cpu_query_report_path") or "")
    query_protocol = REPO_ROOT / str(
        authorization.get("cpu_query_protocol_path") or ""
    )
    if (
        authorization
        != {
            "cpu_query_report_path": "docs/retrieval_cpu_query_compatibility_v1.json",
            "cpu_query_report_sha256": "52ed6a51d22b0e6f0b9fc2533e15faf5c66398e1a46b0e0c358568c34015e0ad",
            "cpu_query_result_fingerprint": "6063af5f27991d74db0a044218b78385f5ff4318e9deb229e27d21b564730ab4",
            "cpu_query_decision": "passed_for_cpu_full_index_feasibility",
            "cpu_query_protocol_path": "docs/retrieval_cpu_query_compatibility_protocol_v3.json",
            "cpu_query_protocol_sha256": "45b994bafaff8e0068847594af5af894365c43dc19e80b8dff1dddc580f94023",
        }
        or not query_report.is_file()
        or authorization.get("cpu_query_report_sha256") != _sha256(query_report)
        or not query_protocol.is_file()
        or authorization.get("cpu_query_protocol_sha256") != _sha256(query_protocol)
    ):
        errors.append("CPU full-index authorization changed")
    else:
        outcome = json.loads(query_report.read_text(encoding="utf-8"))
        if (
            outcome.get("result_fingerprint")
            != authorization["cpu_query_result_fingerprint"]
            or (outcome.get("decision") or {}).get("status")
            != authorization["cpu_query_decision"]
            or (outcome.get("decision") or {}).get("cpu_full_index_study_allowed")
            is not True
        ):
            errors.append("CPU full-index authorization outcome changed")
    fixture = payload.get("fixture") or {}
    fixture_path = REPO_ROOT / str(fixture.get("path") or "")
    if (
        not fixture_path.is_file()
        or fixture.get("sha256") != _sha256(fixture_path)
        or fixture.get("case_count") != 12
        or fixture.get("modes") != ["scoped", "full_library"]
    ):
        errors.append("CPU full-index fixture changed")
    if payload.get("isolated_corpus") != {
        "runtime_db_argument_required": True,
        "tree_sha256": "deb60a97bdfc50b0273de07ebe1c775352ffb2c5e4adeb3e925d48f7f25cafaf",
        "committed_docs_sha256": "3455d33a36b6055d533d3d75c5db1a5dc1108009391e14e623d39eba909327df",
        "corpus_fingerprint": "19dd55b81535593219b0fdb8744b6c74cdfe51d68c3567dfd822780be927ec7a",
        "document_count": 10,
        "chunk_count": 1673,
        "character_count": 1488995,
        "writes_allowed": False,
    }:
        errors.append("CPU full-index corpus boundary changed")
    if payload.get("dedicated_index") != {
        "relative_path": "test_results/retrieval_cpu_full_index_feasibility_v1/index_v1",
        "must_not_exist_before_run": True,
        "maximum_builds": 1,
        "product_index_reuse_allowed": False,
        "product_index_writes_allowed": False,
        "atomic_publish_required": True,
        "readback_required": True,
    }:
        errors.append("CPU full-index destination boundary changed")
    if payload.get("cpu_provider") != {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "provider_identity": "a5bb526562ff581e073496f8fc14077e119373c5051a70a78aec7306770434e9",
        "device": "cpu",
        "dtype": "float32",
        "pooling": "last_token",
        "padding_side": "left",
        "max_length": 8192,
        "document_batch_size": 8,
        "allow_download": False,
        "trust_remote_code": False,
    }:
        errors.append("CPU full-index provider changed")
    if payload.get("fusion") != {
        "strategy": "protected_dense_tail",
        "protected_sparse_prefix": 5,
        "sparse_tail_window": 5,
        "dense_novelty_quota": 2,
        "candidate_multiplier": 4,
        "ranking_depth": 10,
    }:
        errors.append("CPU full-index fusion changed")
    if payload.get("quality_gate") != {
        "minimum_semantic_recall_at_10_gain_each_mode": 0.15,
        "require_recall_at_5_equal_to_product": True,
        "require_wrong_source_at_5_equal_to_product": True,
        "require_no_regression_vs_cpu_query_compatibility_metrics": [
            "recall@5",
            "recall@10",
            "mrr",
            "ndcg@10",
        ],
        "require_wrong_source_at_5_no_worse_than_cpu_query_compatibility": True,
    }:
        errors.append("CPU full-index quality gate changed")
    if payload.get("operational_gate") != {
        "maximum_provider_load_ms": 30000.0,
        "maximum_full_build_ms": 600000.0,
        "minimum_build_throughput_chunks_per_second": 2.75,
        "maximum_peak_process_rss_mib": 6144.0,
        "maximum_index_size_bytes": 10485760,
        "maximum_warm_query_embedding_p95_ms": 1000.0,
        "required_query_completion_rate": 1.0,
        "maximum_truncated_document_count": 0,
        "require_all_chunks_embedded": True,
        "require_no_reuse": True,
        "require_source_corpus_unchanged": True,
    }:
        errors.append("CPU full-index operational gate changed")
    execution = payload.get("execution") or {}
    if (
        execution.get("maximum_runs") != 1
        or execution.get("raw_report_persist_before_validation") is not True
        or execution.get("product_enabled") is not False
        or execution.get("ordinary_user_surface_exposed") is not False
        or execution.get("pass_decision")
        != "passed_for_cpu_optional_runtime_footprint_study"
        or execution.get("reject_decision") != "rejected_cpu_full_index_runtime"
    ):
        errors.append("CPU full-index execution boundary changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] CPU full-index protocol cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_protocol(payload)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[OK] CPU full-index feasibility protocol is frozen before embedding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
