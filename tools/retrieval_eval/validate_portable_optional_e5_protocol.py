from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.lighter_runtime_onnx_worker import _raw_sha256, _tree_fingerprint


DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "retrieval_portable_optional_e5_protocol_v1.json"
FRESH_SCOPED_REPORT = REPO_ROOT / "docs" / "retrieval_scoped_fresh_e5_v1.json"
FRESH_SCOPED_FIXTURE = REPO_ROOT / "docs" / "retrieval_scoped_fresh_cases_v1.json"


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _combined_tree_sha256(paths: list[Path]) -> str:
    return hashlib.sha256(
        "".join(_tree_fingerprint(path) for path in paths).encode("ascii")
    ).hexdigest()


def validate_protocol(
    payload: dict[str, Any],
    *,
    inspect_runtime_files: bool = False,
) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id") != "retrieval_portable_optional_e5_v1"
        or payload.get("status")
        != "refrozen_after_pre_execution_path_length_failure_before_worker_execution"
    ):
        errors.append("portable optional E5 protocol identity changed")

    authorization = payload.get("authorization") or {}
    if authorization != {
        "fresh_scoped_report_path": "docs/retrieval_scoped_fresh_e5_v1.json",
        "fresh_scoped_report_sha256": (
            "90db9c97eef1196ab1e93f5531f662735daeef53c28adbb8a2a54cf5d0bb7fd8"
        ),
        "fresh_scoped_result_fingerprint": (
            "23a3a7262c8cde3a95ed2cbfbe6ea251318f05327cbf149438e9331fcbea7c76"
        ),
        "fresh_scoped_decision": (
            "fresh_scoped_suite_passed_authorize_packaging_experiment"
        ),
        "product_or_ui_enablement_authorized": False,
    }:
        errors.append("portable optional E5 authorization changed")

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
        "python_runtime": "embedded",
        "python_version": "3.10.11",
        "source_dirty": False,
        "modified_by_experiment": False,
    }:
        errors.append("portable optional E5 base archive contract changed")

    candidate = payload.get("candidate") or {}
    if candidate != {
        "candidate_id": "multilingual_e5_small_official_onnx_v1",
        "model_id": "intfloat/multilingual-e5-small",
        "model_revision": "03415a4be176a1620747c692ed433219fabc3def",
        "artifact_source_path": (
            "test_results/retrieval_lighter_runtime_v1/artifacts/"
            "multilingual_e5_small_official_onnx_v1"
        ),
        "artifact_bytes": 492421554,
        "artifact_tree_sha256": (
            "5671f8ed0e55e080c2bfb535acd2059d29cd371b510640048bef56eae66f3e01"
        ),
        "onnxruntime_version": "1.23.2",
        "tokenizers_version": "0.21.4",
    }:
        errors.append("portable optional E5 candidate changed")

    component = payload.get("optional_component") or {}
    overlay = component.get("runtime_overlay") or {}
    entries = overlay.get("entries") or []
    expected_entries = [
        {
            "name": "onnxruntime",
            "bytes": 41484471,
            "tree_sha256": (
                "498fc06cfc1c45f78761149bab242aacf4b8578cf2fdf3d0f3e6a9d7fc9bedfc"
            ),
        },
        {
            "name": "onnxruntime-1.23.2.dist-info",
            "bytes": 65958,
            "tree_sha256": (
                "c6f5a912c725468e18f1ea7c4459a29e0cab326afdddf702d33a69bc42cc4bc5"
            ),
        },
        {
            "name": "tokenizers",
            "bytes": 7349974,
            "tree_sha256": (
                "dbf64bcf578bd3de138ed1b58dc7f3ca62e1bb94663c7d59633bf78b5cd43f86"
            ),
        },
        {
            "name": "tokenizers-0.21.4.dist-info",
            "bytes": 10777,
            "tree_sha256": (
                "4d0a322ae0f4b66c2945b73fcfe6387fb5a0d71cf377029284cec36be5a5b07c"
            ),
        },
    ]
    if component != {
        "component_id": "pizaya_local_semantic_e5_small_windows_x64_v1",
        "delivery": "separate_optional_archive",
        "base_archive_growth_bytes": 0,
        "artifact_source_path": (
            "test_results/retrieval_lighter_runtime_v1/artifacts/"
            "multilingual_e5_small_official_onnx_v1"
        ),
        "runtime_overlay": {
            "strategy": "minimal_binary_overlay_reusing_base_numpy_coloredlogs_packaging",
            "source_site_packages": (
                "test_results/retrieval_lighter_runtime_v1/.venv/Lib/site-packages"
            ),
            "combined_bytes": 48911180,
            "combined_tree_sha256": (
                "4dc3b33398c560f3218d0c1b746cd446eddedb714633765dcecb742becb08b32"
            ),
            "entries": expected_entries,
            "release_status": "experiment_only_not_a_distributable_pip_environment",
            "formal_release_requires_dependency_license_and_installer_lifecycle_review": True,
        },
        "contains_user_library_data": False,
        "installation_alone_enables_product": False,
    }:
        errors.append("portable optional E5 component contract changed")

    execution = payload.get("execution") or {}
    if execution != {
        "runner_path": "tools/retrieval_eval/run_portable_optional_e5.py",
        "runner_sha256": (
            "bd8e8ad733722b0f88a69fd55d2a79970ec1b12af88e3549a843cd2fed5c1ce4"
        ),
        "worker_path": "tools/retrieval_eval/portable_optional_e5_worker.py",
        "worker_sha256": (
            "d3e4b415a7e70a0fe7678ddba09dab4bdd44c863daa956971dbe868acad6cb9c"
        ),
        "fixture_path": "docs/retrieval_scoped_fresh_cases_v1.json",
        "fixture_sha256": (
            "3472826090e032739903db91dd8c359adbc68bff6fcbe40ec9f136bb945b1d27"
        ),
        "working_db_path": "test_results/retrieval_product_e5_user_journey_v1/db",
        "existing_index_path": "test_results/retrieval_product_e5_user_journey_v1/index",
        "result_root": "test_results/e5p1",
        "component_directory_name": "c",
        "component_archive_name": (
            "Pi_zaya-local-semantic-e5-small-v1-windows-x64.zip"
        ),
        "report_path": "docs/retrieval_portable_optional_e5_v1.json",
        "hard_subprocess_timeout_seconds": 90,
        "allow_external_network": False,
        "allow_model_download": False,
        "allow_base_archive_write": False,
        "allow_artifact_source_write": False,
        "allow_user_library_write": False,
        "index_build_calls_allowed": 0,
        "selected_scope_query_ids": [
            "fresh-fdm-integration-time-boundary",
            "fresh-scinerf-overlap-rate",
        ],
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }:
        errors.append("portable optional E5 execution boundary changed")

    if payload.get("pre_execution_history") != [
        {
            "attempt": 1,
            "outcome": "aborted_during_runtime_overlay_copy",
            "failure_id": "windows_destination_path_too_long",
            "component_archive_created": False,
            "report_created": False,
            "embedded_worker_started": False,
            "candidate_queries_executed": 0,
            "frozen_inputs_reverified_unchanged": True,
            "allowed_refreeze_scope": [
                "execution.result_root",
                "execution.component_directory_name",
            ],
        }
    ]:
        errors.append("portable optional E5 pre-execution history changed")

    if payload.get("gates") != {
        "maximum_component_raw_bytes": 550000000,
        "maximum_component_archive_bytes": 550000000,
        "maximum_combined_download_bytes": 650000000,
        "maximum_component_assembly_ms": 120000.0,
        "maximum_component_compression_ms": 300000.0,
        "maximum_embedded_worker_wall_ms": 60000.0,
        "maximum_import_ms": 5000.0,
        "maximum_cold_load_ms": 30000.0,
        "maximum_warm_query_p95_ms": 1000.0,
        "maximum_peak_process_rss_mib": 2048.0,
        "required_query_count": 2,
        "require_base_archive_unchanged": True,
        "require_source_artifacts_unchanged": True,
        "require_zero_base_archive_growth": True,
        "require_no_user_data": True,
        "require_zero_index_builds": True,
        "require_zero_source_escapes_or_fallbacks": True,
        "all_required": True,
    }:
        errors.append("portable optional E5 gates changed")
    if payload.get("decision") != {
        "pass_status": "optional_component_viable_keep_base_unchanged_design_opt_in_lifecycle",
        "failure_status": "optional_component_rejected_keep_bm25_investigate_smaller_runtime",
        "bundling_into_every_install_allowed": False,
        "product_enablement_allowed": False,
        "user_mode_control_allowed": False,
    }:
        errors.append("portable optional E5 decision changed")
    if payload.get("product_boundary") != {
        "library_papers_only": True,
        "selected_paper_scope_only": True,
        "evidence_retrieval_only": True,
        "writing_or_submission_workflow_allowed": False,
        "external_web_or_model_knowledge_used": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }:
        errors.append("portable optional E5 product boundary changed")

    if (
        not FRESH_SCOPED_REPORT.is_file()
        or sha256_file(FRESH_SCOPED_REPORT)
        != str(authorization.get("fresh_scoped_report_sha256") or "")
    ):
        errors.append("portable optional E5 authorization report changed")
    if (
        not FRESH_SCOPED_FIXTURE.is_file()
        or sha256_file(FRESH_SCOPED_FIXTURE)
        != str(execution.get("fixture_sha256") or "")
    ):
        errors.append("portable optional E5 fixture changed")

    if inspect_runtime_files:
        for path_key, sha_key in (
            ("runner_path", "runner_sha256"),
            ("worker_path", "worker_sha256"),
        ):
            path = REPO_ROOT / str(execution.get(path_key) or "")
            if not path.is_file() or _raw_sha256(path) != str(
                execution.get(sha_key) or ""
            ):
                errors.append(f"portable optional E5 {path_key} changed")
        base_path = REPO_ROOT / str(base.get("archive_path") or "")
        if (
            not base_path.is_file()
            or base_path.stat().st_size != int(base.get("archive_bytes") or -1)
            or _raw_sha256(base_path) != str(base.get("archive_sha256") or "")
        ):
            errors.append("portable optional E5 base archive file changed")

        artifact_path = REPO_ROOT / str(candidate.get("artifact_source_path") or "")
        if (
            not artifact_path.is_dir()
            or _directory_bytes(artifact_path) != int(candidate.get("artifact_bytes") or -1)
            or _tree_fingerprint(artifact_path)
            != str(candidate.get("artifact_tree_sha256") or "")
        ):
            errors.append("portable optional E5 artifact source changed")

        source_site = REPO_ROOT / str(overlay.get("source_site_packages") or "")
        runtime_paths = [source_site / str(item.get("name") or "") for item in entries]
        entry_mismatch = len(entries) != len(expected_entries)
        for item, path in zip(entries, runtime_paths):
            if (
                not path.exists()
                or _directory_bytes(path) != int(item.get("bytes") or -1)
                or _tree_fingerprint(path) != str(item.get("tree_sha256") or "")
            ):
                entry_mismatch = True
        if (
            entry_mismatch
            or sum(_directory_bytes(path) for path in runtime_paths if path.exists())
            != int(overlay.get("combined_bytes") or -1)
            or any(not path.exists() for path in runtime_paths)
            or _combined_tree_sha256(runtime_paths)
            != str(overlay.get("combined_tree_sha256") or "")
        ):
            errors.append("portable optional E5 runtime overlay files changed")

        for key in ("working_db_path", "existing_index_path"):
            if not (REPO_ROOT / str(execution.get(key) or "")).is_dir():
                errors.append(f"portable optional E5 required {key} is missing")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] portable optional E5 protocol unreadable: {type(exc).__name__}")
        return 1
    errors = validate_protocol(payload)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[OK] portable optional E5 protocol is frozen before component assembly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
