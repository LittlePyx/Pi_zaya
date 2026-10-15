from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.lighter_runtime_onnx_worker import _raw_sha256, _tree_fingerprint
from tools.retrieval_eval.run_product_e5_user_journey import retrieval_snapshot_sha256
from tools.retrieval_eval.validate_scoped_fresh_cases import (
    DEFAULT_FIXTURE,
    EXPECTED_FIXTURE_SHA256,
)


DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "retrieval_scoped_fresh_e5_protocol_v1.json"
AUTHORING_PROTOCOL = REPO_ROOT / "docs" / "retrieval_scoped_fresh_authoring_protocol_v1.json"
PRODUCT_BOUNDARY = {
    "library_papers_only": True,
    "evidence_retrieval_only": True,
    "writing_or_submission_workflow_allowed": False,
    "external_web_or_model_knowledge_used": False,
    "product_enabled": False,
    "ordinary_user_surface_exposed": False,
}


def validate_protocol(payload: dict[str, Any], *, inspect_files: bool = False) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id") != "retrieval_scoped_fresh_e5_v1"
        or payload.get("status") != "frozen_before_candidate_replay"
    ):
        errors.append("scoped fresh E5 protocol identity changed")

    authorization = payload.get("authorization") or {}
    if authorization != {
        "authoring_protocol_path": "docs/retrieval_scoped_fresh_authoring_protocol_v1.json",
        "authoring_protocol_sha256": sha256_file(AUTHORING_PROTOCOL),
        "fixture_path": "docs/retrieval_scoped_fresh_cases_v1.json",
        "fixture_sha256": EXPECTED_FIXTURE_SHA256,
        "fixture_case_count": 12,
        "fixture_label_count": 21,
        "fixture_document_count": 12,
        "fixture_chunk_count": 702,
        "fixture_corpus_fingerprint": "5d65f3a033d62e15c3b90b7c32314452cb49334652d1db0f0e74a55a2b125317",
        "retriever_executed_before_protocol_freeze": False,
        "candidate_configuration_changes_allowed": 0,
    }:
        errors.append("scoped fresh E5 authorization changed")

    existing = payload.get("existing_corpus_and_index") or {}
    if existing != {
        "working_db_path": "test_results/retrieval_product_e5_user_journey_v1/db",
        "working_db_retrieval_snapshot_sha256": "09d20706edb877e5534f115f92b3252c0917caf23f3e91af55d37d7209557535",
        "index_path": "test_results/retrieval_product_e5_user_journey_v1/index",
        "index_tree_sha256": "4b6858d3c1afc5239e4e6eaa98f6eabe7b87b24f2df7b58315764128046b23e6",
        "build_id": "a66067c83cd0440abc8381b31a7b154f",
        "document_count": 31,
        "parent_count": 2251,
        "window_count": 2663,
        "index_bytes": 4870347,
        "index_rebuild_allowed": False,
        "index_write_allowed": False,
        "database_write_allowed": False,
    }:
        errors.append("scoped fresh existing corpus/index contract changed")

    candidate = payload.get("candidate") or {}
    selector = candidate.get("selector_configuration") or {}
    sources = candidate.get("product_sources") or {}
    if (
        candidate.get("candidate_id") != "multilingual_e5_small_official_onnx_v1"
        or candidate.get("model_id") != "intfloat/multilingual-e5-small"
        or candidate.get("model_revision")
        != "03415a4be176a1620747c692ed433219fabc3def"
        or candidate.get("model_bytes") != 470268510
        or candidate.get("model_sha256")
        != "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665"
        or candidate.get("tokenizer_bytes") != 17082730
        or candidate.get("tokenizer_sha256")
        != "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"
        or candidate.get("artifact_tree_sha256")
        != "5671f8ed0e55e080c2bfb535acd2059d29cd371b510640048bef56eae66f3e01"
        or candidate.get("embedding_dimension") != 384
        or candidate.get("representation_id")
        != "e5_small_full_coverage_parent_collapsed_windows_v1"
        or selector
        != {
            "retrieval_mode": "hybrid_local",
            "source_scope": "scoped",
            "protected_sparse_prefix": 5,
            "evaluated_k": 10,
        }
        or sources
        != {
            "executor_path": "kb/e5_small_embedding_runtime.py",
            "executor_sha256": "25713a37f3c00f075bb076a0c31a8eab491a46c43d32d46fe677ccd000e60e3d",
            "index_path": "kb/local_semantic_window_index.py",
            "index_sha256": "38aabbe8ee861e8e2b59fac153cce08d9ad154057878e2617621f1faab48a36e",
            "runtime_path": "kb/local_semantic_runtime.py",
            "runtime_sha256": "3b9b20ef237eba07c2c0375abe635e6df79e24b7cbfa12d82f43bde412ddadc3",
            "selector_path": "kb/local_retrieval_selector.py",
            "selector_sha256": "a6787eb8fb650d51ece6c728dc8290be8c33ee065f15f9df8be9114024725417",
            "fusion_path": "kb/protected_dense_tail_retriever.py",
            "fusion_sha256": "f120174d277fd190008fb0c1d5749315df83cb75fd1ca143aa864991c28d95b2",
        }
    ):
        errors.append("scoped fresh E5 candidate changed")

    runtime = payload.get("runtime_environment") or {}
    if runtime != {
        "python_version": "3.10.11",
        "model_runtime_python": "test_results/retrieval_lighter_runtime_v1/.venv/Scripts/python.exe",
        "product_dependency_site_packages": ".ci_venv_2/Lib/site-packages",
        "python_no_user_site": True,
        "package_install_allowed": False,
        "numpy_version": "2.2.6",
        "onnxruntime_version": "1.23.2",
        "tokenizers_version": "0.21.4",
        "rank_bm25_version": "0.2.2",
        "rank_bm25_source_sha256": "2f28cc795415C01E9F3DB5A8ED019774F9CBA747B272C9C304271589B8081AC6",
    }:
        errors.append("scoped fresh E5 runtime environment changed")

    execution = payload.get("execution") or {}
    if execution != {
        "modes": ["scoped"],
        "selection_call_count": 12,
        "full_library_calls_allowed": 0,
        "embedding_build_calls_allowed": 0,
        "result_root": "test_results/retrieval_scoped_fresh_e5_v1",
        "worker_result_path": "test_results/retrieval_scoped_fresh_e5_v1/worker_v1.json",
        "report_path": "docs/retrieval_scoped_fresh_e5_v1.json",
        "hard_subprocess_timeout_seconds": 240,
        "allow_external_network": False,
        "allow_model_download": False,
        "candidate_enabled_only_inside_offline_replay": True,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
        "report_persisted_before_outcome_validation": True,
    }:
        errors.append("scoped fresh E5 execution boundary changed")

    runtime_gates = payload.get("runtime_gates") or {}
    if runtime_gates != {
        "maximum_subprocess_wall_ms": 180000.0,
        "maximum_peak_process_rss_mib": 6144.0,
        "maximum_warm_selector_p95_ms": 1000.0,
        "required_index_build_calls": 0,
        "required_selection_calls": 12,
        "required_cold_load_count": 1,
        "minimum_warm_cache_hit_count": 11,
        "maximum_fallback_count": 0,
        "maximum_source_escape_count": 0,
        "required_protected_prefix_rate": 1.0,
        "required_product_sparse_input_unchanged_rate": 1.0,
        "require_working_db_unchanged": True,
        "require_existing_index_unchanged": True,
        "require_artifact_model_tokenizer_unchanged": True,
        "all_required": True,
    }:
        errors.append("scoped fresh E5 runtime gates changed")
    quality = payload.get("quality_gates") or {}
    if quality != {
        "required_no_regression": [
            "recall@5",
            "recall@10",
            "mrr",
            "ndcg@10",
            "contract_recall@10",
        ],
        "minimum_recall_at_10_gain": 0.02,
        "maximum_case_regressions": {
            "recall@10": 0,
            "contract_recall@10": 0,
            "mrr": 0,
        },
        "all_required": True,
    }:
        errors.append("scoped fresh E5 quality gates changed")
    if payload.get("decision") != {
        "pass_status": "fresh_scoped_suite_passed_authorize_packaging_experiment",
        "runtime_failure_status": "fresh_scoped_runtime_rejected_keep_bm25",
        "quality_failure_status": "fresh_scoped_quality_rejected_keep_bm25",
        "product_enablement_allowed": False,
        "user_mode_control_allowed": False,
        "candidate_tuning_on_failure_allowed": False,
    }:
        errors.append("scoped fresh E5 decision changed")
    if payload.get("product_boundary") != PRODUCT_BOUNDARY:
        errors.append("scoped fresh E5 product boundary changed")

    if not DEFAULT_FIXTURE.is_file() or sha256_file(DEFAULT_FIXTURE) != EXPECTED_FIXTURE_SHA256:
        errors.append("scoped fresh E5 sealed fixture changed")
    if inspect_files:
        for path_key, sha_key in (
            ("executor_path", "executor_sha256"),
            ("index_path", "index_sha256"),
            ("runtime_path", "runtime_sha256"),
            ("selector_path", "selector_sha256"),
            ("fusion_path", "fusion_sha256"),
        ):
            path = REPO_ROOT / str(sources.get(path_key) or "")
            if not path.is_file() or _raw_sha256(path) != str(
                sources.get(sha_key) or ""
            ):
                errors.append(f"scoped fresh E5 product source changed: {path_key}")
        try:
            snapshot_sha256 = retrieval_snapshot_sha256(
                REPO_ROOT / str(existing.get("working_db_path") or "")
            )
            index_sha256 = _tree_fingerprint(
                REPO_ROOT / str(existing.get("index_path") or "")
            )
        except (OSError, RuntimeError):
            snapshot_sha256 = ""
            index_sha256 = ""
        if (
            existing.get("working_db_retrieval_snapshot_sha256") != snapshot_sha256
            or existing.get("index_tree_sha256") != index_sha256
        ):
            errors.append("scoped fresh E5 existing corpus/index files changed")
        for path_key, sha_key in (
            ("model_path", "model_sha256"),
            ("tokenizer_path", "tokenizer_sha256"),
        ):
            path = REPO_ROOT / str(candidate.get(path_key) or "")
            if not path.is_file() or _raw_sha256(path) != str(candidate.get(sha_key) or ""):
                errors.append(f"scoped fresh E5 {path_key} changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] scoped fresh E5 protocol unreadable: {type(exc).__name__}")
        return 1
    errors = validate_protocol(payload)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[OK] scoped fresh E5 protocol is frozen before candidate replay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
