from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from tools.release.validate_local_semantic_component_release import (
    validate_component_release,
)
from tools.retrieval_eval.validate_portable_e5_full_size_lifecycle_protocol import (
    _BASE,
    _BOUNDARY,
    _expected_queries,
    _raw_sha256,
    _tree_identity,
)
from tools.retrieval_eval.validate_portable_e5_full_size_lifecycle_report_v2 import (
    validate_report as validate_v2_report,
)


DEFAULT_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_portable_e5_full_size_lifecycle_protocol_v3.json"
)
EXPECTED_PROTOCOL_SHA256 = (
    "ff3d3ad62761e4143cebf75480db00d1338fba05456075db57556003f5a11b99"
)
_AUTHORIZATION = {
    "lifecycle_contract_path": "docs/retrieval_local_semantic_component_lifecycle_contract_v2.json",
    "lifecycle_contract_sha256": "b10047eb5921dcbcabf8e88c026cfa60d1ed25dfddf905d8337874083c0dacfc",
    "full_size_v2_protocol_path": "docs/retrieval_portable_e5_full_size_lifecycle_protocol_v2.json",
    "full_size_v2_protocol_sha256": "c66c4456ea669f57e684d24016bf2f616aa3b8bc63aa18b1f24423314b95a8df",
    "full_size_v2_report_path": "docs/retrieval_portable_e5_full_size_lifecycle_v2.json",
    "full_size_v2_report_sha256": "1fed1e34dde206b1f51d4d5f6aaae6538fd5ca25f68652f08a22e701459c069b",
    "full_size_v2_result_fingerprint": "7440961c0be3e681e4c1ac1bb6a9de8dc7a4d78d5c9e7719438893924907709d",
    "full_size_v2_decision": "full_size_short_path_lifecycle_passed_authorize_management_api_e2e",
    "release_builder_path": "tools/release/build_local_semantic_component.py",
    "release_builder_sha256": "7ae8815cc22b671bfeadad0f38c4ea2449534fb2f6f4e9933db78744653c2e4e",
    "release_validator_path": "tools/release/validate_local_semantic_component_release.py",
    "release_validator_sha256": "e1d540591b6a8ac29bf65f063d1553de0d4051a387af83e2a225d5213b99a711",
}
_COMPONENT = {
    "component_id": "pizaya_local_semantic_e5_small_windows_x64_v1",
    "candidate_id": "multilingual_e5_small_official_onnx_v1",
    "model_revision": "03415a4be176a1620747c692ed433219fabc3def",
    "release_id": "v0.1.0-beta.13-component.1",
    "immutable_manifest_url": (
        "https://github.com/LittlePyx/Pi_zaya/releases/download/"
        "v0.1.0-beta.13-component.1/"
        "Pi_zaya-local-semantic-e5-small-v1-windows-x64.zip"
    ),
    "manifest_url_is_validation_only_and_not_fetched": True,
    "archive_path": (
        "test_results/e5p7/release/"
        "Pi_zaya-local-semantic-e5-small-v1-windows-x64.zip"
    ),
    "archive_name": "Pi_zaya-local-semantic-e5-small-v1-windows-x64.zip",
    "archive_root": "c",
    "archive_bytes": 299558739,
    "archive_sha256": "ad8bfb006b91bd9714ad4758c2da00f18557451a293279c11e3b3b24c8123ea8",
    "payload_bytes": 538888249,
    "payload_file_count": 360,
    "payload_sha256": "1561576403759fbe072c9fe4a5d14fe9bf3e03e3b0da62d0841a6328aa6c317c",
    "contains_user_library_data": False,
}
_RELEASE_HANDOFF = {
    "manifest_path": (
        "test_results/e5p7/release/"
        "Pi_zaya-local-semantic-e5-small-v1-windows-x64.manifest.json"
    ),
    "manifest_sha256": "bcfb6ffa14d32a3da54522652364da2f0e9a84ddac926bc49653a2db6a50b367",
    "deterministic_zip_metadata": True,
    "cached_bytecode_entries_removed": 320,
    "previous_payload_file_count": 680,
    "release_payload_file_count": 360,
    "maximum_component_relative_path_characters": 146,
    "expected_full_size_layout_max_path_characters": 214,
    "external_network_used": False,
    "release_published": False,
    "product_enabled": False,
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
    "runner_sha256": "492ed145b4ef9af5dcb43fd690356cf391000a7b57b65945ce304d3c2a71fae8",
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
    "result_root": "test_results/e5p8",
    "report_path": "docs/retrieval_portable_e5_full_size_lifecycle_v3.json",
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
    "expected_component_layout_max_path_chars": 214,
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
    "pass_status": "release_component_lifecycle_passed_authorize_opt_in_ui_e2e",
    "failure_status": "release_component_lifecycle_rejected_keep_bm25_repair_release_handoff",
    "product_enablement_allowed": False,
    "ordinary_user_surface_allowed": False,
}


def _validate_authorization_files(errors: list[str]) -> None:
    for path_key, sha_key in (
        ("lifecycle_contract_path", "lifecycle_contract_sha256"),
        ("full_size_v2_protocol_path", "full_size_v2_protocol_sha256"),
        ("full_size_v2_report_path", "full_size_v2_report_sha256"),
        ("release_builder_path", "release_builder_sha256"),
        ("release_validator_path", "release_validator_sha256"),
    ):
        path = REPO_ROOT / _AUTHORIZATION[path_key]
        if not path.is_file() or sha256_file(path) != _AUTHORIZATION[sha_key]:
            errors.append("release-component lifecycle authorization file changed")
    try:
        v2_report = json.loads(
            (REPO_ROOT / _AUTHORIZATION["full_size_v2_report_path"]).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        errors.append("release-component lifecycle v2 report unreadable")
        return
    if (
        validate_v2_report(v2_report)
        or v2_report.get("result_fingerprint")
        != _AUTHORIZATION["full_size_v2_result_fingerprint"]
        or (v2_report.get("decision") or {}).get("status")
        != _AUTHORIZATION["full_size_v2_decision"]
    ):
        errors.append("release-component lifecycle v2 evidence changed")


def validate_protocol(
    payload: dict[str, Any],
    *,
    inspect_runtime_files: bool = False,
) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id")
        != "retrieval_portable_e5_full_size_lifecycle_v3"
        or payload.get("status")
        != "frozen_before_single_release_component_lifecycle_execution"
    ):
        errors.append("release-component lifecycle protocol identity changed")
    if payload.get("authorization") != _AUTHORIZATION:
        errors.append("release-component lifecycle authorization changed")
    if payload.get("base_portable") != _BASE:
        errors.append("release-component lifecycle base archive changed")
    if payload.get("component") != _COMPONENT:
        errors.append("release-component lifecycle component changed")
    if payload.get("release_handoff") != _RELEASE_HANDOFF:
        errors.append("release-component lifecycle handoff changed")
    execution = payload.get("execution") or {}
    without_queries = {
        key: value for key, value in execution.items() if key != "expected_queries"
    }
    if without_queries != _EXECUTION_WITHOUT_QUERIES:
        errors.append("release-component lifecycle execution boundary changed")
    if execution.get("expected_queries") != _expected_queries(errors):
        errors.append("release-component lifecycle expected rankings changed")
    if payload.get("gates") != _GATES:
        errors.append("release-component lifecycle gates changed")
    if payload.get("decision") != _DECISION:
        errors.append("release-component lifecycle decision changed")
    if payload.get("product_boundary") != _BOUNDARY:
        errors.append("release-component lifecycle product boundary changed")
    _validate_authorization_files(errors)

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
                errors.append(f"release-component lifecycle {path_key} changed")
        for relative, expected_sha in _IMPLEMENTATION_FILES.items():
            path = REPO_ROOT / relative
            if not path.is_file() or _raw_sha256(path) != expected_sha:
                errors.append("release-component lifecycle implementation changed")
                break
        for label, contract in (("base", _BASE), ("component", _COMPONENT)):
            archive = REPO_ROOT / str(contract["archive_path"])
            if (
                not archive.is_file()
                or archive.stat().st_size != int(contract["archive_bytes"])
                or _raw_sha256(archive) != str(contract["archive_sha256"])
            ):
                errors.append(f"release-component lifecycle {label} archive changed")
        manifest_path = REPO_ROOT / _RELEASE_HANDOFF["manifest_path"]
        archive_path = REPO_ROOT / _COMPONENT["archive_path"]
        if (
            not manifest_path.is_file()
            or _raw_sha256(manifest_path) != _RELEASE_HANDOFF["manifest_sha256"]
            or validate_component_release(archive_path, manifest_path)[1]
        ):
            errors.append("release-component lifecycle release handoff changed")
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
                errors.append(f"release-component lifecycle {label} changed")
    return list(dict.fromkeys(errors))


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] release-component protocol unreadable: {type(exc).__name__}")
        return 1
    errors = validate_protocol(payload)
    if sha256_file(DEFAULT_PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        errors.append("release-component lifecycle protocol SHA-256 changed")
    if errors:
        for error in dict.fromkeys(errors):
            print(f"[FAIL] {error}")
        return 1
    print("[OK] portable E5 release-component lifecycle v3 protocol is frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
