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
from tools.retrieval_eval.run_portable_e5_architecture_startup import (
    _component_archive_payload,
)


DEFAULT_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_portable_e5_architecture_startup_protocol_v3.json"
)

_SOURCE_V2 = {
    "report_path": "docs/retrieval_portable_e5_product_startup_v2.json",
    "report_sha256": "5f8216155254b9b5fe61d46cc8375ed3b023ea68cbc81e27bdf338c9cf21c5ca",
    "result_fingerprint": "7789569cae9397c475472d0d0a63d312b8789f06de7531b357e4521cd5d48d85",
    "decision_status": "product_startup_boundary_rejected_keep_bm25_screen_smaller_delivery",
    "failed_gates": ["component_tree_reproduced", "product_import_time"],
    "verdict_may_be_changed": False,
}

_ARCHITECTURE_CANDIDATE = {
    "candidate_id": "multilingual_e5_small_extracted_query_policy_v3",
    "change_kind": "product_query_import_boundary_extracted",
    "query_policy_path": "kb/product_query_policy.py",
    "query_policy_sha256": "2cca7e7ec349b5d22409f9105bd952d632ecdfb1a12974aa738189c4f56ed01e",
    "boundary_test_path": "tests/unit/test_product_query_policy_boundary.py",
    "boundary_test_sha256": "ec01011ed13b24248798b50e2930b783fafd40df1d41d7397144a13742f76b30",
    "previous_product_import_ms": 9046.063,
    "relevance_rerun_allowed": False,
    "delivery_remeasurement_allowed": True,
}

_BASE_PORTABLE = {
    "version": "0.1.0-beta.13",
    "commit": "9c78a7635896",
    "archive_path": (
        ".runtime/release-verify-beta13-20260825-1436/"
        "Pi_zaya-v0.1.0-beta.13-windows-x64.zip"
    ),
    "archive_root": "Pi_zaya-v0.1.0-beta.13-windows-x64",
    "archive_bytes": 94876265,
    "archive_sha256": "ef9247b8af3856deee6b9c8ea5b2605cd0af58f77c41aa7acb571f98a75f9968",
    "python_version": "3.10.11",
    "modified_by_experiment": False,
}

_COMPONENT = {
    "candidate_id": "multilingual_e5_small_official_onnx_v1",
    "archive_path": "test_results/e5p1/Pi_zaya-local-semantic-e5-small-v1-windows-x64.zip",
    "archive_root": "c",
    "archive_bytes": 296365646,
    "archive_sha256": "54d0f4fbd1c07913f5915dafbab224cf5510072fd767fcffd62da0e0b8ac3ad3",
    "raw_bytes": 541333859,
    "payload_file_count": 680,
    "payload_sha256": "568ecdc099cacb4c99943f5d333a8661f4194975ebebafa0caedae0379fc54ef",
    "combined_download_bytes": 391241911,
    "base_archive_growth_bytes": 0,
    "contains_user_library_data": False,
    "product_enabled_by_installation": False,
}

_EXECUTION = {
    "runner_path": "tools/retrieval_eval/run_portable_e5_architecture_startup.py",
    "runner_sha256": "3ccc53f39b5217f6d78b63c7df14d82ef37fa16ef860121076e7f314817d6acb",
    "worker_path": "tools/retrieval_eval/portable_e5_architecture_startup_worker.py",
    "worker_sha256": "6551d985741b23ca6a6bc352b5d1964f24f9dcd4c5e871a22fe838fa25a83157",
    "fixture_path": "docs/retrieval_scoped_fresh_cases_v1.json",
    "fixture_sha256": "3472826090e032739903db91dd8c359adbc68bff6fcbe40ec9f136bb945b1d27",
    "working_db_path": "test_results/retrieval_product_e5_user_journey_v1/db",
    "existing_index_path": "test_results/retrieval_product_e5_user_journey_v1/index",
    "result_root": "test_results/e5p4",
    "report_path": "docs/retrieval_portable_e5_architecture_startup_v3.json",
    "hard_subprocess_timeout_seconds": 90,
    "allow_external_network": False,
    "allow_model_download": False,
    "allow_archive_write": False,
    "allow_user_library_write": False,
    "index_build_calls_allowed": 0,
    "python_dont_write_bytecode": True,
    "evaluation_helper_imported_before_startup_timing": False,
    "forbid_retrieval_engine_import": True,
    "forbid_llm_import": True,
    "product_enabled": False,
    "ordinary_user_surface_exposed": False,
}

_GATES = {
    "maximum_component_raw_bytes": 550000000,
    "maximum_component_archive_bytes": 550000000,
    "maximum_combined_download_bytes": 650000000,
    "maximum_component_extraction_ms": 120000.0,
    "maximum_embedded_worker_wall_ms": 60000.0,
    "maximum_product_import_ms": 5000.0,
    "maximum_cold_load_ms": 30000.0,
    "maximum_warm_query_p95_ms": 1000.0,
    "maximum_peak_process_rss_mib": 2048.0,
    "require_v2_rejection_preserved": True,
    "require_archive_payload_reproduced_before_worker": True,
    "require_fresh_short_extraction": True,
    "require_exact_v2_ranking": True,
    "require_zero_index_builds": True,
    "require_zero_source_escapes_or_fallbacks": True,
    "require_lightweight_product_import_boundary": True,
    "all_required": True,
}

_DECISION = {
    "pass_status": "architecture_startup_passed_authorize_component_lifecycle_contract",
    "failure_status": "architecture_startup_rejected_keep_bm25_profile_remaining_runtime",
    "bundling_into_every_install_allowed": False,
    "product_enablement_allowed": False,
    "user_mode_control_allowed": False,
}

_PRODUCT_BOUNDARY = {
    "library_papers_only": True,
    "selected_paper_scope_only": True,
    "evidence_retrieval_only": True,
    "writing_or_submission_workflow_allowed": False,
    "external_web_or_model_knowledge_used": False,
    "product_enabled": False,
    "ordinary_user_surface_exposed": False,
}


def _expected_queries(source: dict[str, Any], errors: list[str]) -> list[dict[str, Any]]:
    report_path = REPO_ROOT / str(source.get("report_path") or "")
    if not report_path.is_file() or sha256_file(report_path) != source.get("report_sha256"):
        errors.append("portable E5 architecture-startup v2 report changed")
        return []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return [
        {
            "case_id": row.get("case_id"),
            "first_relevant_rank": row.get("first_relevant_rank"),
            "ranked_chunk_ids": row.get("ranked_chunk_ids"),
        }
        for row in report.get("queries") or []
    ]


def validate_protocol(
    payload: dict[str, Any],
    *,
    inspect_runtime_files: bool = False,
) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id") != "retrieval_portable_e5_architecture_startup_v3"
        or payload.get("status") != "frozen_before_architecture_qualified_execution"
    ):
        errors.append("portable E5 architecture-startup protocol identity changed")
    if payload.get("source_v2") != _SOURCE_V2:
        errors.append("portable E5 architecture-startup v2 rejection lineage changed")
    if payload.get("architecture_candidate") != _ARCHITECTURE_CANDIDATE:
        errors.append("portable E5 architecture-startup candidate changed")
    if payload.get("base_portable") != _BASE_PORTABLE:
        errors.append("portable E5 architecture-startup base archive changed")
    if payload.get("component") != _COMPONENT:
        errors.append("portable E5 architecture-startup component changed")

    execution = payload.get("execution") or {}
    execution_without_queries = {
        key: value for key, value in execution.items() if key != "expected_queries"
    }
    if execution_without_queries != _EXECUTION:
        errors.append("portable E5 architecture-startup execution boundary changed")
    if execution.get("expected_queries") != _expected_queries(_SOURCE_V2, errors):
        errors.append("portable E5 architecture-startup expected ranking changed")
    if payload.get("gates") != _GATES:
        errors.append("portable E5 architecture-startup gates changed")
    if payload.get("decision") != _DECISION:
        errors.append("portable E5 architecture-startup decision changed")
    if payload.get("product_boundary") != _PRODUCT_BOUNDARY:
        errors.append("portable E5 architecture-startup product boundary changed")

    fixture_path = REPO_ROOT / str(execution.get("fixture_path") or "")
    if not fixture_path.is_file() or _raw_sha256(fixture_path) != str(
        execution.get("fixture_sha256") or ""
    ):
        errors.append("portable E5 architecture-startup fixture changed")

    if inspect_runtime_files:
        for contract in (_EXECUTION, _ARCHITECTURE_CANDIDATE):
            for path_key, sha_key in (
                ("runner_path", "runner_sha256"),
                ("worker_path", "worker_sha256"),
                ("query_policy_path", "query_policy_sha256"),
                ("boundary_test_path", "boundary_test_sha256"),
            ):
                if path_key not in contract:
                    continue
                path = REPO_ROOT / str(contract[path_key])
                if not path.is_file() or _raw_sha256(path) != str(contract[sha_key]):
                    errors.append(f"portable E5 architecture-startup {path_key} changed")
        for label, contract in (("base", _BASE_PORTABLE), ("component", _COMPONENT)):
            archive = REPO_ROOT / str(contract["archive_path"])
            if (
                not archive.is_file()
                or archive.stat().st_size != int(contract["archive_bytes"])
                or _raw_sha256(archive) != str(contract["archive_sha256"])
            ):
                errors.append(f"portable E5 architecture-startup {label} archive changed")
        component_archive = REPO_ROOT / _COMPONENT["archive_path"]
        if component_archive.is_file():
            observed = _component_archive_payload(
                component_archive,
                archive_root=str(_COMPONENT["archive_root"]),
            )
            if observed != {
                "sha256": _COMPONENT["payload_sha256"],
                "file_count": _COMPONENT["payload_file_count"],
                "bytes": _COMPONENT["raw_bytes"],
            }:
                errors.append("portable E5 architecture-startup archive payload changed")
        for key in ("working_db_path", "existing_index_path"):
            if not (REPO_ROOT / str(execution.get(key) or "")).is_dir():
                errors.append(f"portable E5 architecture-startup required {key} is missing")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] portable E5 architecture-startup protocol unreadable: {type(exc).__name__}")
        return 1
    errors = validate_protocol(payload)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[OK] portable E5 architecture-startup v3 protocol is frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
