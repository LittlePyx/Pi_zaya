from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.validate_portable_e5_full_size_lifecycle_protocol import (
    _BASE,
    _BOUNDARY,
    _COMPONENT,
    _expected_queries,
    _raw_sha256,
    _tree_identity,
)
from tools.retrieval_eval.validate_portable_e5_full_size_lifecycle_report import (
    validate_report as validate_v1_rejection,
)


DEFAULT_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_portable_e5_full_size_lifecycle_protocol_v2.json"
)
EXPECTED_PROTOCOL_SHA256 = (
    "c66c4456ea669f57e684d24016bf2f616aa3b8bc63aa18b1f24423314b95a8df"
)
_AUTHORIZATION = {
    "lifecycle_contract_path": "docs/retrieval_local_semantic_component_lifecycle_contract_v2.json",
    "lifecycle_contract_sha256": "b10047eb5921dcbcabf8e88c026cfa60d1ed25dfddf905d8337874083c0dacfc",
    "architecture_report_path": "docs/retrieval_portable_e5_architecture_startup_v3.json",
    "architecture_report_sha256": "1843fe12db7290cbcb07e892187f0882cee7f24c791790354b6a79eee1c6bc4e",
    "architecture_result_fingerprint": "07b64dd0541c3c6bf3205f95aeee96c8bd70dd68f55760f13f4a4cabc3f91384",
    "architecture_decision": "architecture_startup_passed_authorize_component_lifecycle_contract",
    "rejected_v1_protocol_path": "docs/retrieval_portable_e5_full_size_lifecycle_protocol_v1.json",
    "rejected_v1_protocol_sha256": "dc5205129d50fda61320cfb49023db216e719113f55260d86959171d61eee752",
    "rejected_v1_report_path": "docs/retrieval_portable_e5_full_size_lifecycle_v1.json",
    "rejected_v1_report_sha256": "8d8f3fdc5414d523dd408ff5d45dc0e3255e50c89a758cf04e244dad6ef60abb",
    "rejected_v1_result_fingerprint": "362c52a8dc745fd62f4ca7739614f95314989cccb771f1fae2fd4b01ef136225",
    "rejected_v1_decision": "full_size_lifecycle_rejected_keep_bm25_repair_backend",
    "rejected_v1_root_cause": "atomic_generation_layout_exceeds_traditional_windows_path_budget",
    "rejected_v1_longest_target_absolute_path_characters": 275,
    "rejected_v1_verdict_may_be_changed": False,
}
_PATH_REPAIR = {
    "component_store_relative_path": "c",
    "generation_id_characters": 16,
    "staging_prefix": ".s-",
    "maximum_component_relative_path_characters": 146,
    "maximum_windows_target_path_characters": 259,
    "expected_full_size_layout_max_path_characters": 239,
    "preflight_before_staging_creation": True,
    "legacy_32_character_generation_ids_readable": True,
    "overlong_layout_manifest_quarantined": False,
}
_IMPLEMENTATION_FILES = {
    "kb/local_semantic_component.py": "0282021c099fe6d50d0de421e802e8332b17e50a1f71794711d3f57f5ee35e4c",
    "kb/local_semantic_component_download.py": "60a3be837f468556dec441a3cb9872ee3497adfe45a21cedf32779bff7174101",
    "kb/local_semantic_component_lifecycle.py": "9555f3c1c5cde3f5e43475b2cbdcb4ecc1b2e12c13d14e3e6218522a3e02ba6b",
    "kb/local_semantic_component_lifecycle_store.py": "e7185b400efa79efbc285c2f6025b7a780915d478f9b53acf1e8f2342fdef310",
    "kb/local_semantic_component_lifecycle_contract.py": "5987c1fa69050619e6d5404cc36d50c8b8b96e6350c47d041682315302984a46",
    "kb/portable_local_semantic_runtime.py": "99bf5debfe626351d2d70b4ead5a3b0396cbc7f27e82834b894d7608f3381a83",
}
_EXECUTION_WITHOUT_QUERIES = {
    "runner_path": "tools/retrieval_eval/run_portable_e5_full_size_lifecycle.py",
    "runner_sha256": "707e86f5553cbf3e40747162fe6cf6e1eded15642a0e91419acddd1431650dfe",
    "worker_path": "tools/retrieval_eval/portable_e5_lifecycle_worker.py",
    "worker_sha256": "e8c0b62cfac451fd2e83229677f88300246d26a65dd973d0b779d86668390a43",
    "implementation_files": _IMPLEMENTATION_FILES,
    "fixture_path": "docs/retrieval_scoped_fresh_cases_v1.json",
    "fixture_sha256": "3472826090e032739903db91dd8c359adbc68bff6fcbe40ec9f136bb945b1d27",
    "source_db_path": "test_results/retrieval_product_e5_user_journey_v1/db",
    "source_db_file_count": 33,
    "source_db_bytes": 3577032,
    "source_db_tree_sha256": "b5087ac5984900448320fd6aea87d45b674f2bad92810ebc0f787515fcd1249d",
    "source_index_path": "test_results/retrieval_product_e5_user_journey_v1/index",
    "source_index_file_count": 5,
    "source_index_bytes": 4870347,
    "source_index_tree_sha256": "bd71f8bfc4d5df827220e4270b2088a28808d88eb8946e6f6bcf42a7aa31f0b6",
    "result_root": "test_results/e5p6",
    "report_path": "docs/retrieval_portable_e5_full_size_lifecycle_v2.json",
    "component_store_relative_path": "c",
    "hard_subprocess_timeout_seconds": 180,
    "allow_external_network": False,
    "allow_model_download": False,
    "allow_user_library_write": False,
    "allow_product_enablement": False,
    "case_ids": [
        "fresh-fdm-integration-time-boundary",
        "fresh-scinerf-overlap-rate",
    ],
}
_GATES = {
    "maximum_base_extraction_ms": 120000.0,
    "maximum_install_ready_ms": 120000.0,
    "maximum_cold_load_ms": 30000.0,
    "maximum_warm_query_p95_ms": 1000.0,
    "maximum_stale_repair_ms": 60000.0,
    "maximum_removal_ms": 30000.0,
    "maximum_peak_process_rss_mib": 2048.0,
    "maximum_component_path_chars": 259,
    "expected_component_layout_max_path_chars": 239,
    "require_embedded_python_3_10_11": True,
    "require_exact_v3_rankings": True,
    "require_hybrid_without_fallback": True,
    "require_zero_source_escapes": True,
    "require_one_incremental_stale_repair": True,
    "require_zero_network": True,
    "require_bm25_continuity": True,
    "require_component_and_index_disabled_before_delete": True,
    "require_cleanup_after_worker_exit": True,
    "require_product_and_ui_disabled": True,
    "all_required": True,
}
_DECISION = {
    "pass_status": "full_size_short_path_lifecycle_passed_authorize_management_api_e2e",
    "failure_status": "full_size_short_path_lifecycle_rejected_keep_bm25_repair_backend",
    "product_enablement_allowed": False,
    "ordinary_user_surface_allowed": False,
}


def _validate_v1_lineage(errors: list[str]) -> None:
    protocol_path = REPO_ROOT / _AUTHORIZATION["rejected_v1_protocol_path"]
    report_path = REPO_ROOT / _AUTHORIZATION["rejected_v1_report_path"]
    if (
        not protocol_path.is_file()
        or sha256_file(protocol_path)
        != _AUTHORIZATION["rejected_v1_protocol_sha256"]
    ):
        errors.append("short-path lifecycle rejected v1 protocol changed")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        errors.append("short-path lifecycle rejected v1 report unreadable")
        return
    if (
        sha256_file(report_path) != _AUTHORIZATION["rejected_v1_report_sha256"]
        or validate_v1_rejection(report)
        or report.get("result_fingerprint")
        != _AUTHORIZATION["rejected_v1_result_fingerprint"]
        or (report.get("decision") or {}).get("status")
        != _AUTHORIZATION["rejected_v1_decision"]
        or (report.get("decision") or {}).get("verdict_may_be_changed")
        is not _AUTHORIZATION["rejected_v1_verdict_may_be_changed"]
        or (report.get("attribution") or {}).get("root_cause")
        != _AUTHORIZATION["rejected_v1_root_cause"]
        or (report.get("attribution") or {}).get(
            "longest_target_absolute_path_characters"
        )
        != _AUTHORIZATION["rejected_v1_longest_target_absolute_path_characters"]
    ):
        errors.append("short-path lifecycle rejected v1 evidence changed")


def validate_protocol(
    payload: dict[str, Any],
    *,
    inspect_runtime_files: bool = False,
) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id")
        != "retrieval_portable_e5_full_size_lifecycle_v2"
        or payload.get("status")
        != "frozen_before_single_short_path_lifecycle_execution"
    ):
        errors.append("short-path lifecycle protocol identity changed")
    if payload.get("authorization") != _AUTHORIZATION:
        errors.append("short-path lifecycle authorization changed")
    if payload.get("base_portable") != _BASE:
        errors.append("short-path lifecycle base archive changed")
    if payload.get("component") != _COMPONENT:
        errors.append("short-path lifecycle component changed")
    if payload.get("path_repair") != _PATH_REPAIR:
        errors.append("short-path lifecycle repair boundary changed")
    execution = payload.get("execution") or {}
    without_queries = {
        key: value for key, value in execution.items() if key != "expected_queries"
    }
    if without_queries != _EXECUTION_WITHOUT_QUERIES:
        errors.append("short-path lifecycle execution boundary changed")
    if execution.get("expected_queries") != _expected_queries(errors):
        errors.append("short-path lifecycle expected rankings changed")
    if payload.get("gates") != _GATES:
        errors.append("short-path lifecycle gates changed")
    if payload.get("decision") != _DECISION:
        errors.append("short-path lifecycle decision changed")
    if payload.get("product_boundary") != _BOUNDARY:
        errors.append("short-path lifecycle product boundary changed")

    for path_key, sha_key in (
        ("lifecycle_contract_path", "lifecycle_contract_sha256"),
        ("architecture_report_path", "architecture_report_sha256"),
    ):
        path = REPO_ROOT / str(_AUTHORIZATION[path_key])
        if not path.is_file() or sha256_file(path) != _AUTHORIZATION[sha_key]:
            errors.append("short-path lifecycle authorization file changed")
    _validate_v1_lineage(errors)

    if inspect_runtime_files:
        for path_key, sha_key in (
            ("runner_path", "runner_sha256"),
            ("worker_path", "worker_sha256"),
            ("fixture_path", "fixture_sha256"),
        ):
            path = REPO_ROOT / str(execution.get(path_key) or "")
            if not path.is_file() or _raw_sha256(path) != str(
                execution.get(sha_key) or ""
            ):
                errors.append(f"short-path lifecycle {path_key} changed")
        for relative, expected_sha in _IMPLEMENTATION_FILES.items():
            path = REPO_ROOT / relative
            if not path.is_file() or _raw_sha256(path) != expected_sha:
                errors.append("short-path lifecycle implementation changed")
                break
        for label, contract in (("base", _BASE), ("component", _COMPONENT)):
            archive = REPO_ROOT / str(contract["archive_path"])
            if (
                not archive.is_file()
                or archive.stat().st_size != int(contract["archive_bytes"])
                or _raw_sha256(archive) != str(contract["archive_sha256"])
            ):
                errors.append(f"short-path lifecycle {label} archive changed")
        for label, path_key, count_key, bytes_key, sha_key in (
            (
                "source db",
                "source_db_path",
                "source_db_file_count",
                "source_db_bytes",
                "source_db_tree_sha256",
            ),
            (
                "source index",
                "source_index_path",
                "source_index_file_count",
                "source_index_bytes",
                "source_index_tree_sha256",
            ),
        ):
            root = REPO_ROOT / str(execution.get(path_key) or "")
            observed = _tree_identity(root) if root.is_dir() else {}
            expected = {
                "file_count": execution.get(count_key),
                "bytes": execution.get(bytes_key),
                "sha256": execution.get(sha_key),
            }
            if observed != expected:
                errors.append(f"short-path lifecycle {label} changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] short-path lifecycle protocol unreadable: {type(exc).__name__}")
        return 1
    errors = validate_protocol(payload)
    if sha256_file(DEFAULT_PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        errors.append("short-path lifecycle protocol SHA-256 changed")
    if errors:
        for error in dict.fromkeys(errors):
            print(f"[FAIL] {error}")
        return 1
    print("[OK] portable E5 full-size lifecycle v2 protocol is frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
