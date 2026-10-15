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


DEFAULT_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_portable_e5_full_size_lifecycle_protocol_v1.json"
)
EXPECTED_PROTOCOL_SHA256 = (
    "dc5205129d50fda61320cfb49023db216e719113f55260d86959171d61eee752"
)
_AUTHORIZATION = {
    "lifecycle_contract_path": "docs/retrieval_local_semantic_component_lifecycle_contract_v2.json",
    "lifecycle_contract_sha256": "b10047eb5921dcbcabf8e88c026cfa60d1ed25dfddf905d8337874083c0dacfc",
    "architecture_report_path": "docs/retrieval_portable_e5_architecture_startup_v3.json",
    "architecture_report_sha256": "1843fe12db7290cbcb07e892187f0882cee7f24c791790354b6a79eee1c6bc4e",
    "architecture_result_fingerprint": "07b64dd0541c3c6bf3205f95aeee96c8bd70dd68f55760f13f4a4cabc3f91384",
    "architecture_decision": "architecture_startup_passed_authorize_component_lifecycle_contract",
}
_BASE = {
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
}
_COMPONENT = {
    "component_id": "pizaya_local_semantic_e5_small_windows_x64_v1",
    "candidate_id": "multilingual_e5_small_official_onnx_v1",
    "model_revision": "03415a4be176a1620747c692ed433219fabc3def",
    "release_id": "lifecycle-fixture-v1",
    "immutable_manifest_url": (
        "https://github.com/LittlePyx/Pi_zaya/releases/download/lifecycle-fixture-v1/"
        "Pi_zaya-local-semantic-e5-small-v1-windows-x64.zip"
    ),
    "manifest_url_is_validation_only_and_not_fetched": True,
    "archive_path": (
        "test_results/e5p1/Pi_zaya-local-semantic-e5-small-v1-windows-x64.zip"
    ),
    "archive_name": "Pi_zaya-local-semantic-e5-small-v1-windows-x64.zip",
    "archive_root": "c",
    "archive_bytes": 296365646,
    "archive_sha256": "54d0f4fbd1c07913f5915dafbab224cf5510072fd767fcffd62da0e0b8ac3ad3",
    "payload_bytes": 541333859,
    "payload_file_count": 680,
    "payload_sha256": "568ecdc099cacb4c99943f5d333a8661f4194975ebebafa0caedae0379fc54ef",
    "contains_user_library_data": False,
}
_IMPLEMENTATION_FILES = {
    "kb/local_semantic_component.py": "272e4325726cff505ab4e49f8eb761f9bb7116c4fce76979956ceeadaab95b45",
    "kb/local_semantic_component_download.py": "60a3be837f468556dec441a3cb9872ee3497adfe45a21cedf32779bff7174101",
    "kb/local_semantic_component_lifecycle.py": "ccb8ea1bb3b364692729646a6eb369712183b42f4ce48403c39d05b6718d3b9e",
    "kb/local_semantic_component_lifecycle_store.py": "e7185b400efa79efbc285c2f6025b7a780915d478f9b53acf1e8f2342fdef310",
    "kb/local_semantic_component_lifecycle_contract.py": "5987c1fa69050619e6d5404cc36d50c8b8b96e6350c47d041682315302984a46",
    "kb/portable_local_semantic_runtime.py": "99bf5debfe626351d2d70b4ead5a3b0396cbc7f27e82834b894d7608f3381a83",
}
_EXECUTION_WITHOUT_QUERIES = {
    "runner_path": "tools/retrieval_eval/run_portable_e5_full_size_lifecycle.py",
    "runner_sha256": "c9d9482266b46ebd3d0cc90232aa2caafdb7b331666992cb0efe2000bc23c914",
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
    "result_root": "test_results/e5p5",
    "report_path": "docs/retrieval_portable_e5_full_size_lifecycle_v1.json",
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
    "pass_status": "full_size_lifecycle_passed_authorize_management_api_e2e",
    "failure_status": "full_size_lifecycle_rejected_keep_bm25_repair_backend",
    "product_enablement_allowed": False,
    "ordinary_user_surface_allowed": False,
}
_BOUNDARY = {
    "library_papers_only": True,
    "selected_paper_scope_only": True,
    "evidence_retrieval_only": True,
    "writing_or_submission_workflow_allowed": False,
    "external_web_or_model_knowledge_used": False,
    "product_enabled": False,
    "ordinary_user_surface_exposed": False,
}


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_identity(root: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_sha = _raw_sha256(path)
        digest.update(f"{relative}\0{size}\0{file_sha}\n".encode("utf-8"))
        count += 1
        total += size
    return {"file_count": count, "bytes": total, "sha256": digest.hexdigest()}


def _expected_queries(errors: list[str]) -> list[dict[str, Any]]:
    report_path = REPO_ROOT / _AUTHORIZATION["architecture_report_path"]
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        errors.append("full-size lifecycle architecture report unreadable")
        return []
    return [
        {
            "case_id": item.get("case_id"),
            "first_relevant_rank": item.get("first_relevant_rank"),
            "ranked_chunk_ids": item.get("ranked_chunk_ids"),
        }
        for item in report.get("queries") or []
    ]


def validate_protocol(
    payload: dict[str, Any],
    *,
    inspect_runtime_files: bool = False,
) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id")
        != "retrieval_portable_e5_full_size_lifecycle_v1"
        or payload.get("status")
        != "frozen_before_single_full_size_lifecycle_execution"
    ):
        errors.append("full-size lifecycle protocol identity changed")
    if payload.get("authorization") != _AUTHORIZATION:
        errors.append("full-size lifecycle authorization changed")
    if payload.get("base_portable") != _BASE:
        errors.append("full-size lifecycle base archive changed")
    if payload.get("component") != _COMPONENT:
        errors.append("full-size lifecycle component changed")
    execution = payload.get("execution") or {}
    without_queries = {
        key: value for key, value in execution.items() if key != "expected_queries"
    }
    if without_queries != _EXECUTION_WITHOUT_QUERIES:
        errors.append("full-size lifecycle execution boundary changed")
    if execution.get("expected_queries") != _expected_queries(errors):
        errors.append("full-size lifecycle expected rankings changed")
    if payload.get("gates") != _GATES:
        errors.append("full-size lifecycle gates changed")
    if payload.get("decision") != _DECISION:
        errors.append("full-size lifecycle decision changed")
    if payload.get("product_boundary") != _BOUNDARY:
        errors.append("full-size lifecycle product boundary changed")

    for path_key, sha_key in (
        ("lifecycle_contract_path", "lifecycle_contract_sha256"),
        ("architecture_report_path", "architecture_report_sha256"),
    ):
        path = REPO_ROOT / str(_AUTHORIZATION[path_key])
        if not path.is_file() or sha256_file(path) != _AUTHORIZATION[sha_key]:
            errors.append("full-size lifecycle authorization file changed")
    if inspect_runtime_files:
        for path_key, sha_key in (
            ("runner_path", "runner_sha256"),
            ("worker_path", "worker_sha256"),
            ("fixture_path", "fixture_sha256"),
        ):
            path = REPO_ROOT / str(execution.get(path_key) or "")
            if not path.is_file() or _raw_sha256(path) != str(execution.get(sha_key) or ""):
                errors.append(f"full-size lifecycle {path_key} changed")
        for relative, expected_sha in _IMPLEMENTATION_FILES.items():
            path = REPO_ROOT / relative
            if not path.is_file() or _raw_sha256(path) != expected_sha:
                errors.append("full-size lifecycle implementation changed")
                break
        for label, contract in (("base", _BASE), ("component", _COMPONENT)):
            archive = REPO_ROOT / str(contract["archive_path"])
            if (
                not archive.is_file()
                or archive.stat().st_size != int(contract["archive_bytes"])
                or _raw_sha256(archive) != str(contract["archive_sha256"])
            ):
                errors.append(f"full-size lifecycle {label} archive changed")
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
                errors.append(f"full-size lifecycle {label} changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] full-size lifecycle protocol unreadable: {type(exc).__name__}")
        return 1
    errors = validate_protocol(payload)
    if sha256_file(DEFAULT_PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        errors.append("full-size lifecycle protocol SHA-256 changed")
    if errors:
        for error in dict.fromkeys(errors):
            print(f"[FAIL] {error}")
        return 1
    print("[OK] portable E5 full-size lifecycle v1 protocol is frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
