from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_product_e5_selector_shadow_protocol_v1.json"
)


def _sha256(path: Path) -> str:
    value = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(value).hexdigest()


def _verify_paths(
    payload: dict[str, Any],
    pairs: tuple[tuple[str, str], ...],
    *,
    label: str,
) -> list[str]:
    errors: list[str] = []
    for path_key, sha_key in pairs:
        path = REPO_ROOT / str(payload.get(path_key) or "")
        if not path.is_file() or payload.get(sha_key) != _sha256(path):
            errors.append(f"{label} changed: {path_key}")
    return errors


def validate_protocol(
    payload: dict[str, Any],
    *,
    inspect_implementation_files: bool = False,
) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id")
        != "retrieval_product_e5_selector_shadow_v1"
        or payload.get("status")
        != "refrozen_after_preexecution_dependency_correction"
    ):
        errors.append("product E5 selector shadow protocol identity changed")

    authorization = payload.get("authorization") or {}
    errors.extend(
        _verify_paths(
            authorization,
            (
                ("preparation_protocol_path", "preparation_protocol_sha256"),
                ("runtime_parity_report_path", "runtime_parity_report_sha256"),
                ("stage_2_report_path", "stage_2_report_sha256"),
                (
                    "preexecution_invalidation_path",
                    "preexecution_invalidation_sha256",
                ),
            ),
            label="product E5 selector authorization",
        )
    )
    if (
        authorization.get("stage_2_result_fingerprint")
        != "18fdba85d35503d2f5f60e38191df412482ec7fc1783341b4354fa0f37848a05"
        or authorization.get("stage_2_decision")
        != "runtime_candidate_eligible_for_optional_preparation_design"
        or authorization.get("completed_full_builds_before_retry") != 0
    ):
        errors.append("product E5 selector Stage-2 authorization changed")

    source = payload.get("source_corpus") or {}
    errors.extend(
        _verify_paths(
            source,
            (
                ("qrels_path", "qrels_sha256"),
                ("semantic_fixture_path", "semantic_fixture_sha256"),
            ),
            label="product E5 selector source fixture",
        )
    )
    expected_source = {
        "runtime_db_argument_required": True,
        "source_tree_sha256": "deb60a97bdfc50b0273de07ebe1c775352ffb2c5e4adeb3e925d48f7f25cafaf",
        "qrels_path": "docs/retrieval_eval_unseen_v2.json",
        "qrels_sha256": "4062ce7f0a1ce4afc86f44fbe74fdc375bbeeeb6b2bdc934344e4e0d53d749d7",
        "semantic_fixture_path": "docs/retrieval_semantic_paraphrase_cases_v1.json",
        "semantic_fixture_sha256": "98417031dc54af544a717d74d70938906b87c35c29232fe35fd903fffd03bc3d",
        "document_count": 10,
        "parent_chunk_count": 1673,
        "semantic_case_count": 12,
        "corpus_fingerprint": "d1602fb0d3a7320806dc1cf1a9c3615d7fc1864471e93270f2135f8f44f27516",
        "source_writes_allowed": False,
        "working_copy_required": True,
    }
    if source != expected_source:
        errors.append("product E5 selector source corpus changed")

    candidate = payload.get("candidate") or {}
    sources = candidate.get("product_sources") or {}
    if inspect_implementation_files:
        errors.extend(
            _verify_paths(
                sources,
                (
                    ("executor_path", "executor_sha256"),
                    ("index_path", "index_sha256"),
                    ("runtime_path", "runtime_sha256"),
                    ("selector_path", "selector_sha256"),
                    ("fusion_path", "fusion_sha256"),
                ),
                label="product E5 selector implementation",
            )
        )
    expected_candidate = {
        "candidate_id": "multilingual_e5_small_official_onnx_v1",
        "model_id": "intfloat/multilingual-e5-small",
        "model_revision": "03415a4be176a1620747c692ed433219fabc3def",
        "model_path": "test_results/retrieval_lighter_runtime_v1/artifacts/multilingual_e5_small_official_onnx_v1/model.onnx",
        "model_bytes": 470268510,
        "model_sha256": "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665",
        "tokenizer_path": "test_results/retrieval_lighter_runtime_v1/artifacts/multilingual_e5_small_official_onnx_v1/tokenizer/tokenizer.json",
        "tokenizer_bytes": 17082730,
        "tokenizer_sha256": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
        "embedding_dimension": 384,
        "expected_parent_count": 1673,
        "expected_window_count": 2016,
        "representation_id": "e5_small_full_coverage_parent_collapsed_windows_v1",
        "product_sources": sources,
    }
    if candidate != expected_candidate:
        errors.append("product E5 selector candidate changed")

    execution = payload.get("execution") or {}
    expected_execution = {
        "maximum_completed_full_builds": 1,
        "preexecution_invalidation_count": 1,
        "hard_subprocess_timeout_seconds": 660,
        "working_root": "test_results/retrieval_product_e5_shadow_v1",
        "working_db_path": "test_results/retrieval_product_e5_shadow_v1/db",
        "product_index_path": "test_results/retrieval_product_e5_shadow_v1/index",
        "attempt_receipt_path": "test_results/retrieval_product_e5_shadow_v1/attempt_v1.json",
        "worker_result_path": "test_results/retrieval_product_e5_shadow_v1/worker_v1.json",
        "report_path": "docs/retrieval_product_e5_selector_shadow_v1.json",
        "copy_source_before_model_load": True,
        "validate_source_before_and_after": True,
        "allow_external_network": False,
        "allow_model_download": False,
        "allow_user_library_write": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
        "candidate_enabled_only_inside_offline_replay": True,
        "report_persisted_before_outcome_validation": True,
    }
    if execution != expected_execution:
        errors.append("product E5 selector execution boundary changed")

    runtime_environment = payload.get("runtime_environment") or {}
    if runtime_environment != {
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
        errors.append("product E5 selector runtime environment changed")

    selector = payload.get("selector_contract") or {}
    expected_selector = {
        "requested_mode": "hybrid_local",
        "selection_call_count": 24,
        "modes": ["scoped", "full_library"],
        "protected_sparse_prefix": 5,
        "sparse_tail_window": 5,
        "dense_novelty_quota": 2,
        "candidate_multiplier": 4,
        "output_depth": 10,
        "product_sparse_evaluation_depth": 20,
        "scoped_dense_sources": "sources_present_in_product_sparse_candidate_set",
        "full_library_dense_sources": "all_frozen_corpus_sources",
        "fallback_allowed": False,
        "product_sparse_input_mutation_allowed": False,
    }
    if selector != expected_selector:
        errors.append("product E5 selector contract changed")

    runtime = payload.get("runtime_gates") or {}
    expected_runtime = {
        "maximum_full_build_ms": 600000.0,
        "maximum_peak_process_rss_mib": 6144.0,
        "maximum_warm_selector_p95_ms": 1000.0,
        "maximum_index_bytes": 10000000,
        "required_parent_count": 1673,
        "required_window_count": 2016,
        "required_completed_selection_calls": 24,
        "required_cold_load_count": 1,
        "minimum_warm_cache_hit_count": 23,
        "maximum_fallback_count": 0,
        "maximum_source_escape_count": 0,
        "required_protected_prefix_rate": 1.0,
        "required_product_sparse_input_unchanged_rate": 1.0,
        "require_atomic_readback": True,
        "require_source_artifact_tokenizer_unchanged": True,
        "all_required": True,
    }
    if runtime != expected_runtime:
        errors.append("product E5 selector runtime gates changed")

    quality = payload.get("quality_gates") or {}
    expected_quality = {
        "channels": [
            "bm25_product_queries_selection_control",
            "e5_small_product_selector_candidate",
        ],
        "minimum_recall_at_10_gain_vs_product_each_mode": 0.15,
        "recall_at_5_equal_to_product_each_mode": True,
        "wrong_source_at_5_equal_to_product_each_mode": True,
        "mrr_no_regression_vs_product_each_mode": True,
        "ndcg_at_10_no_regression_vs_product_each_mode": True,
        "all_required": True,
    }
    if quality != expected_quality:
        errors.append("product E5 selector quality gates changed")

    decision = payload.get("decision") or {}
    if decision != {
        "pass_status": "product_selector_shadow_passed_expand_judged_cases",
        "runtime_failure_status": "product_selector_runtime_rejected_keep_bm25",
        "quality_failure_status": "product_selector_quality_rejected_keep_bm25",
        "product_enablement_allowed": False,
        "user_mode_control_allowed": False,
    }:
        errors.append("product E5 selector decision boundary changed")
    boundary = payload.get("product_boundary") or {}
    if boundary != {
        "library_papers_only": True,
        "evidence_retrieval_only": True,
        "writing_or_submission_workflow_allowed": False,
        "external_web_or_model_knowledge_used": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }:
        errors.append("product E5 selector product boundary changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] product E5 selector protocol unreadable: {type(exc).__name__}")
        return 1
    errors = validate_protocol(payload)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[OK] product E5 selector shadow protocol is frozen and product-disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
