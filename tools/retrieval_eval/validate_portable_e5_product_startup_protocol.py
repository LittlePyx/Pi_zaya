from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.lighter_runtime_onnx_worker import _raw_sha256


DEFAULT_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_portable_e5_product_startup_protocol_v2.json"
)


def validate_protocol(
    payload: dict[str, Any],
    *,
    inspect_runtime_files: bool = False,
) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id") != "retrieval_portable_e5_product_startup_v2"
        or payload.get("status")
        != "refrozen_after_pre_query_component_path_wiring_failure"
    ):
        errors.append("portable E5 product-startup protocol identity changed")

    source_v1 = payload.get("source_v1") or {}
    if source_v1 != {
        "report_path": "docs/retrieval_portable_optional_e5_v1.json",
        "report_sha256": (
            "a938b1130c600aa8d9b1ccb11630848bba54ecef204d593348fde10764f42a89"
        ),
        "result_fingerprint": (
            "2120f98ec523d98c0e890a66be008d71a65a67f04e468b3edf61ada48625df7d"
        ),
        "decision_status": "optional_component_rejected_keep_bm25_investigate_smaller_runtime",
        "failed_gate": "import_time",
        "verdict_may_be_changed": False,
    }:
        errors.append("portable E5 product-startup v1 rejection lineage changed")

    base = payload.get("base_portable") or {}
    if base != {
        "version": "0.1.0-beta.13",
        "commit": "9c78a7635896",
        "archive_path": (
            ".runtime/release-verify-beta13-20260825-1436/"
            "Pi_zaya-v0.1.0-beta.13-windows-x64.zip"
        ),
        "archive_root": "Pi_zaya-v0.1.0-beta.13-windows-x64",
        "archive_bytes": 94876265,
        "archive_sha256": (
            "ef9247b8af3856deee6b9c8ea5b2605cd0af58f77c41aa7acb571f98a75f9968"
        ),
        "python_version": "3.10.11",
        "modified_by_experiment": False,
    }:
        errors.append("portable E5 product-startup base archive changed")

    component = payload.get("component") or {}
    if component != {
        "candidate_id": "multilingual_e5_small_official_onnx_v1",
        "archive_path": (
            "test_results/e5p1/Pi_zaya-local-semantic-e5-small-v1-windows-x64.zip"
        ),
        "archive_root": "c",
        "archive_bytes": 296365646,
        "archive_sha256": (
            "54d0f4fbd1c07913f5915dafbab224cf5510072fd767fcffd62da0e0b8ac3ad3"
        ),
        "raw_bytes": 541333859,
        "component_tree_sha256": (
            "17b7c352cd6836ac6d2f4958db15ef05ddba650e4299494e7fe270d7e9ab81b9"
        ),
        "combined_download_bytes": 391241911,
        "base_archive_growth_bytes": 0,
        "contains_user_library_data": False,
        "product_enabled_by_installation": False,
    }:
        errors.append("portable E5 product-startup component changed")

    execution = payload.get("execution") or {}
    expected_execution = {
        "runner_path": "tools/retrieval_eval/run_portable_optional_e5_product_startup.py",
        "runner_sha256": (
            "677cc9c23f53e054455c7dccf49c5ad17257e38dad4cab741765e22cafb21199"
        ),
        "worker_path": (
            "tools/retrieval_eval/portable_optional_e5_product_startup_worker.py"
        ),
        "worker_sha256": (
            "63cf4d6086509efb13d6fa80ae0fa30ba8d20622871118a87317e96645933c31"
        ),
        "fixture_path": "docs/retrieval_scoped_fresh_cases_v1.json",
        "fixture_sha256": (
            "3472826090e032739903db91dd8c359adbc68bff6fcbe40ec9f136bb945b1d27"
        ),
        "working_db_path": "test_results/retrieval_product_e5_user_journey_v1/db",
        "existing_index_path": "test_results/retrieval_product_e5_user_journey_v1/index",
        "result_root": "test_results/e5p3",
        "report_path": "docs/retrieval_portable_e5_product_startup_v2.json",
        "hard_subprocess_timeout_seconds": 90,
        "allow_external_network": False,
        "allow_model_download": False,
        "allow_archive_write": False,
        "allow_user_library_write": False,
        "index_build_calls_allowed": 0,
        "evaluation_helper_imported_before_startup_timing": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }
    execution_without_queries = {
        key: value for key, value in execution.items() if key != "expected_queries"
    }
    if execution_without_queries != expected_execution:
        errors.append("portable E5 product-startup execution boundary changed")

    report_path = REPO_ROOT / str(source_v1.get("report_path") or "")
    expected_queries: list[dict[str, Any]] = []
    if report_path.is_file() and sha256_file(report_path) == source_v1.get("report_sha256"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_queries = [
            {
                "case_id": row.get("case_id"),
                "first_relevant_rank": row.get("first_relevant_rank"),
                "ranked_chunk_ids": row.get("ranked_chunk_ids"),
            }
            for row in report.get("queries") or []
        ]
    else:
        errors.append("portable E5 product-startup v1 report changed")
    if execution.get("expected_queries") != expected_queries:
        errors.append("portable E5 product-startup expected ranking changed")

    if payload.get("pre_execution_history") != [
        {
            "attempt": 1,
            "outcome": "aborted_before_runtime_load_and_queries",
            "failure_id": (
                "worker_executor_used_db_relative_paths_instead_of_"
                "verified_component_paths"
            ),
            "report_created": False,
            "candidate_queries_executed": 0,
            "allowed_refreeze_scope": [
                "execution.worker_sha256",
                "execution.result_root",
                "worker_executor_path_wiring",
            ],
            "gates_or_rankings_changed": False,
        }
    ]:
        errors.append("portable E5 product-startup pre-execution history changed")

    if payload.get("gates") != {
        "maximum_component_raw_bytes": 550000000,
        "maximum_component_archive_bytes": 550000000,
        "maximum_combined_download_bytes": 650000000,
        "maximum_component_extraction_ms": 120000.0,
        "maximum_embedded_worker_wall_ms": 60000.0,
        "maximum_product_import_ms": 5000.0,
        "maximum_cold_load_ms": 30000.0,
        "maximum_warm_query_p95_ms": 1000.0,
        "maximum_peak_process_rss_mib": 2048.0,
        "require_v1_rejection_preserved": True,
        "require_fresh_short_extraction": True,
        "require_exact_v1_ranking": True,
        "require_zero_index_builds": True,
        "require_zero_source_escapes_or_fallbacks": True,
        "all_required": True,
    }:
        errors.append("portable E5 product-startup gates changed")
    if payload.get("decision") != {
        "pass_status": "product_startup_boundary_passed_authorize_component_lifecycle_design",
        "failure_status": "product_startup_boundary_rejected_keep_bm25_screen_smaller_delivery",
        "bundling_into_every_install_allowed": False,
        "product_enablement_allowed": False,
        "user_mode_control_allowed": False,
    }:
        errors.append("portable E5 product-startup decision changed")
    if payload.get("product_boundary") != {
        "library_papers_only": True,
        "selected_paper_scope_only": True,
        "evidence_retrieval_only": True,
        "writing_or_submission_workflow_allowed": False,
        "external_web_or_model_knowledge_used": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }:
        errors.append("portable E5 product-startup product boundary changed")

    fixture_path = REPO_ROOT / str(execution.get("fixture_path") or "")
    if not fixture_path.is_file() or _raw_sha256(fixture_path) != str(
        execution.get("fixture_sha256") or ""
    ):
        errors.append("portable E5 product-startup fixture_path changed")

    if inspect_runtime_files:
        for path_key, sha_key in (
            ("runner_path", "runner_sha256"),
            ("worker_path", "worker_sha256"),
        ):
            path = REPO_ROOT / str(execution.get(path_key) or "")
            if not path.is_file() or _raw_sha256(path) != str(
                execution.get(sha_key) or ""
            ):
                errors.append(f"portable E5 product-startup {path_key} changed")
        for label, contract in (("base", base), ("component", component)):
            archive = REPO_ROOT / str(contract.get("archive_path") or "")
            if (
                not archive.is_file()
                or archive.stat().st_size != int(contract.get("archive_bytes") or -1)
                or _raw_sha256(archive) != str(contract.get("archive_sha256") or "")
            ):
                errors.append(f"portable E5 product-startup {label} archive file changed")
        for key in ("working_db_path", "existing_index_path"):
            if not (REPO_ROOT / str(execution.get(key) or "")).is_dir():
                errors.append(f"portable E5 product-startup required {key} is missing")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] portable E5 product-startup protocol unreadable: {type(exc).__name__}")
        return 1
    errors = validate_protocol(payload)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[OK] portable E5 product-startup v2 protocol is frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
