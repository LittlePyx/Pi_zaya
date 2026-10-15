from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "retrieval_product_e5_user_journey_protocol_v1.json"


def _sha256(path: Path) -> str:
    value = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(value).hexdigest()


def _matches_file(section: dict[str, Any], path_key: str, sha_key: str) -> bool:
    path = REPO_ROOT / str(section.get(path_key) or "")
    return path.is_file() and section.get(sha_key) == _sha256(path)


def validate_protocol(
    payload: dict[str, Any],
    *,
    inspect_implementation_files: bool = False,
) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id") != "retrieval_product_e5_user_journey_v1"
        or payload.get("status")
        != "refrozen_after_retrieval_snapshot_scope_correction"
    ):
        errors.append("product E5 user-journey protocol identity changed")

    authorization = payload.get("authorization") or {}
    if (
        not _matches_file(
            authorization,
            "product_selector_report_path",
            "product_selector_report_sha256",
        )
        or authorization.get("product_selector_result_fingerprint")
        != "2baee02e77afd8e58a23313b01d89e119a1c7e2a42c7390eea48ef1fa58463b7"
        or authorization.get("product_selector_decision")
        != "product_selector_shadow_passed_expand_judged_cases"
    ):
        errors.append("product E5 user-journey authorization changed")

    source = payload.get("source_corpus") or {}
    if (
        not _matches_file(source, "fixture_path", "fixture_sha256")
        or source
        != {
            "runtime_db_argument_required": True,
            "retrieval_snapshot_sha256": "09d20706edb877e5534f115f92b3252c0917caf23f3e91af55d37d7209557535",
            "retrieval_snapshot_algorithm": "sha256(relative_path_nul_raw_bytes_nul_for_docs_json_and_sorted_chunk_jsonl_only)",
            "committed_docs_sha256": "12677731b639c272dff36c61fb8dbe3e64b69802c9bb19452ec995d049d5ec98",
            "index_corpus_fingerprint": "7e3261179b44da0e2baa430ea14243ef6885a49ee73d3e921b48bfc57f35bdf5",
            "fixture_path": "docs/retrieval_user_journey_expansion_v1.json",
            "fixture_sha256": "6463cf63ba1853fc8704eead89de0ee36061481613621b1fa72ea4b7fe5429ca",
            "document_count": 31,
            "parent_chunk_count": 2251,
            "expected_window_count": 2663,
            "case_count": 18,
            "selection_call_count": 36,
            "corpus_fingerprint": "20ad35d505d2936dbd80e90df6199e4b3d51fdf4cf8872a36c9211c5d6c99d9f",
            "source_writes_allowed": False,
            "working_copy_required": True,
            "copy_only_docs_and_chunks": True,
        }
    ):
        errors.append("product E5 user-journey source corpus changed")

    evaluation = payload.get("evaluation_status") or {}
    if evaluation != {
        "labels_pre_registered_before_e5_execution": True,
        "external_web_or_model_knowledge_used_for_labels": False,
        "historical_holdout_already_accessed_by_qwen_candidate": True,
        "new_untouched_holdout_claim_allowed": False,
        "candidate_tuning_from_this_run_allowed": False,
        "strata": {
            "zh": 12,
            "en": 6,
            "single_paper_mechanism": 4,
            "single_paper_boundary_or_limitation": 4,
            "experimental_configuration_or_metric": 4,
            "cross_paper_evidence_comparison": 4,
            "equation_figure_or_table_location": 2,
            "multi_paper_scope": 4,
        },
    }:
        errors.append("product E5 user-journey evaluation status changed")

    candidate = payload.get("candidate") or {}
    product_sources = candidate.get("product_sources") or {}
    if inspect_implementation_files:
        for prefix in ("executor", "index", "runtime", "selector", "fusion"):
            if not _matches_file(
                product_sources,
                f"{prefix}_path",
                f"{prefix}_sha256",
            ):
                errors.append(f"product E5 user-journey source changed: {prefix}")
    if candidate != {
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
        "representation_id": "e5_small_full_coverage_parent_collapsed_windows_v1",
        "configuration_changes_after_selector_shadow": 0,
        "product_sources": product_sources,
    }:
        errors.append("product E5 user-journey candidate changed")

    if payload.get("runtime_environment") != {
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
        errors.append("product E5 user-journey runtime environment changed")

    if payload.get("execution") != {
        "maximum_completed_full_builds": 1,
        "hard_subprocess_timeout_seconds": 660,
        "working_root": "test_results/retrieval_product_e5_user_journey_v1",
        "working_db_path": "test_results/retrieval_product_e5_user_journey_v1/db",
        "product_index_path": "test_results/retrieval_product_e5_user_journey_v1/index",
        "attempt_receipt_path": "test_results/retrieval_product_e5_user_journey_v1/attempt_v1.json",
        "worker_result_path": "test_results/retrieval_product_e5_user_journey_v1/worker_v1.json",
        "report_path": "docs/retrieval_product_e5_user_journey_v1.json",
        "allow_external_network": False,
        "allow_model_download": False,
        "allow_user_library_write": False,
        "candidate_enabled_only_inside_offline_replay": True,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
        "report_persisted_before_outcome_validation": True,
    }:
        errors.append("product E5 user-journey execution boundary changed")

    if payload.get("runtime_gates") != {
        "maximum_full_build_ms": 600000.0,
        "maximum_peak_process_rss_mib": 6144.0,
        "maximum_warm_selector_p95_ms": 1000.0,
        "maximum_index_bytes": 10000000,
        "required_parent_count": 2251,
        "required_window_count": 2663,
        "required_completed_selection_calls": 36,
        "required_cold_load_count": 1,
        "minimum_warm_cache_hit_count": 35,
        "maximum_fallback_count": 0,
        "maximum_source_escape_count": 0,
        "required_protected_prefix_rate": 1.0,
        "required_product_sparse_input_unchanged_rate": 1.0,
        "require_atomic_readback": True,
        "require_source_artifact_tokenizer_unchanged": True,
        "all_required": True,
    }:
        errors.append("product E5 user-journey runtime gates changed")
    if payload.get("quality_gates") != {
        "required_no_regression_each_mode": [
            "recall@5",
            "recall@10",
            "mrr",
            "ndcg@10",
            "contract_recall@10",
        ],
        "wrong_source_rate_no_regression_each_mode": [
            "wrong_source_rate@5",
            "wrong_source_rate@10",
        ],
        "minimum_recall_at_10_gain_in_at_least_one_mode": 0.02,
        "maximum_case_regressions_each_mode": {
            "recall@10": 0,
            "contract_recall@10": 0,
            "mrr": 0,
            "wrong_source_rate@10_increase": 0,
        },
        "all_required": True,
    }:
        errors.append("product E5 user-journey quality gates changed")
    if payload.get("decision") != {
        "pass_status": "current_domain_suite_passed_authorize_packaging_experiment",
        "runtime_failure_status": "current_domain_runtime_rejected_keep_bm25",
        "quality_failure_status": "current_domain_quality_rejected_keep_bm25",
        "product_enablement_allowed": False,
        "user_mode_control_allowed": False,
        "candidate_tuning_on_failure_allowed": False,
    }:
        errors.append("product E5 user-journey decision boundary changed")
    if payload.get("product_boundary") != {
        "library_papers_only": True,
        "evidence_retrieval_only": True,
        "writing_or_submission_workflow_allowed": False,
        "external_web_or_model_knowledge_used": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }:
        errors.append("product E5 user-journey product boundary changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] product E5 user-journey protocol unreadable: {type(exc).__name__}")
        return 1
    errors = validate_protocol(payload)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[OK] product E5 user-journey protocol is frozen and product-disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
