from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file


DEFAULT_CONTRACT = (
    REPO_ROOT
    / "docs"
    / "retrieval_local_semantic_component_lifecycle_contract_v2.json"
)
EXPECTED_CONTRACT_SHA256 = (
    "b10047eb5921dcbcabf8e88c026cfa60d1ed25dfddf905d8337874083c0dacfc"
)
EXPECTED_ARCHITECTURE_FINGERPRINT = (
    "07b64dd0541c3c6bf3205f95aeee96c8bd70dd68f55760f13f4a4cabc3f91384"
)
EXPECTED_INTERNAL_STATES = (
    "not_installed",
    "checking_capacity",
    "downloading",
    "verifying_archive",
    "installing",
    "health_checking",
    "ready_to_index",
    "indexing",
    "ready",
    "cancelling",
    "paused_by_user",
    "recovering",
    "needs_space",
    "degraded_bm25",
    "removing",
    "unavailable_in_build",
)
EXPECTED_PUBLIC_STATES = (
    "not_installed",
    "preparing",
    "ready",
    "paused",
    "needs_space",
    "bm25_available",
    "removing",
    "unavailable",
)
EXPECTED_RECOVERY = {
    "network_timeout_or_5xx": (
        "resume_with_exponential_backoff_then_degraded_bm25_and_retry_on_next_app_start"
    ),
    "invalid_range_response": "discard_partial_and_restart_current_archive_from_zero",
    "archive_size_or_hash_mismatch": (
        "discard_staged_archive_and_retry_once_from_immutable_manifest_then_quarantine_manifest_version"
    ),
    "unsafe_archive_entry": (
        "discard_staging_quarantine_manifest_version_and_keep_bm25"
    ),
    "payload_hash_mismatch": (
        "discard_staging_and_retry_clean_install_once_then_quarantine_manifest_version"
    ),
    "install_or_health_check_failure": (
        "rollback_to_previous_healthy_generation_or_bm25_then_retry_clean_install_once"
    ),
    "process_or_power_interruption": (
        "reconcile_ledger_remove_unpublished_staging_and_resume_automatically"
    ),
    "corpus_changed_during_index_build": (
        "discard_unpublished_index_generation_and_rebuild_latest_snapshot"
    ),
    "stale_or_corrupt_index": "keep_bm25_available_and_rebuild_in_background",
    "locked_file_during_removal": (
        "disable_component_immediately_and_schedule_owned_path_cleanup_on_restart"
    ),
    "unexpected_internal_exception": (
        "record_stable_reason_id_keep_bm25_and_schedule_one_clean_recovery"
    ),
}


def _lineage_errors(authorization: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_files = {
        "legacy_preparation_protocol_path": (
            "legacy_preparation_protocol_sha256",
            "75416104e64b461106be8161630adf471dd1eda8524229135ea9bedb54775a06",
        ),
        "architecture_protocol_path": (
            "architecture_protocol_sha256",
            "31f5bb3dc53f2ef20f662932ee4c3b38893ac5e895109a535cb41a2cdd81e944",
        ),
        "architecture_report_path": (
            "architecture_report_sha256",
            "1843fe12db7290cbcb07e892187f0882cee7f24c791790354b6a79eee1c6bc4e",
        ),
    }
    for path_key, (sha_key, expected_sha) in expected_files.items():
        path = REPO_ROOT / str(authorization.get(path_key) or "")
        if (
            authorization.get(sha_key) != expected_sha
            or not path.is_file()
            or sha256_file(path) != expected_sha
        ):
            errors.append("local semantic component lifecycle lineage changed")
            break
    report_path = REPO_ROOT / str(
        authorization.get("architecture_report_path") or ""
    )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        report = {}
    decision = report.get("decision") or {}
    if (
        authorization.get("legacy_protocol_verdict_may_be_changed") is not False
        or authorization.get("architecture_result_fingerprint")
        != EXPECTED_ARCHITECTURE_FINGERPRINT
        or report.get("result_fingerprint") != EXPECTED_ARCHITECTURE_FINGERPRINT
        or authorization.get("architecture_decision")
        != "architecture_startup_passed_authorize_component_lifecycle_contract"
        or decision.get("status")
        != "architecture_startup_passed_authorize_component_lifecycle_contract"
        or decision.get("product_enabled") is not False
        or decision.get("user_mode_control_visible") is not False
    ):
        errors.append("local semantic component lifecycle authorization changed")
    return errors


def validate_contract(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 2
        or payload.get("contract_id")
        != "retrieval_local_semantic_component_lifecycle_v2"
        or payload.get("status")
        != "frozen_before_portable_component_lifecycle_implementation"
    ):
        errors.append("local semantic component lifecycle identity changed")

    errors.extend(_lineage_errors(payload.get("authorization") or {}))

    candidate = payload.get("candidate") or {}
    if (
        candidate.get("candidate_id")
        != "multilingual_e5_small_official_onnx_v1"
        or candidate.get("model_revision")
        != "03415a4be176a1620747c692ed433219fabc3def"
        or candidate.get("retrieval_scope")
        != "explicit_selected_library_papers_only"
        or candidate.get("fusion") != "bm25_plus_dense_rrf"
        or candidate.get("bm25_default") is not True
    ):
        errors.append("local semantic component candidate boundary changed")

    consent = payload.get("consent") or {}
    if (
        consent.get("required_before_network_or_disk_mutation") is not True
        or consent.get("disclose_archive_bytes") != 296365646
        or consent.get("disclose_installed_bytes") != 541333859
        or consent.get("disclose_no_library_content_uploaded") is not True
        or consent.get("disclose_no_external_content_or_model_knowledge_added")
        is not True
        or consent.get("separate_from_external_web_or_model_knowledge_consent")
        is not True
        or consent.get("query_time_implicit_consent_allowed") is not False
    ):
        errors.append("local semantic component consent boundary changed")

    package = payload.get("component_package") or {}
    if (
        package.get("archive_bytes") != 296365646
        or package.get("archive_sha256")
        != "54d0f4fbd1c07913f5915dafbab224cf5510072fd767fcffd62da0e0b8ac3ad3"
        or package.get("payload_bytes") != 541333859
        or package.get("payload_file_count") != 680
        or package.get("payload_sha256")
        != "568ecdc099cacb4c99943f5d333a8661f4194975ebebafa0caedae0379fc54ef"
        or package.get("delivery_locator_source")
        != "versioned_release_component_manifest"
        or package.get("legacy_direct_model_file_download_used") is not False
        or package.get("archive_hash_checked_before_extraction") is not True
        or package.get("reject_path_escape_symlink_and_reparse_entries") is not True
        or package.get("every_extracted_file_hashed_before_publish") is not True
        or package.get("previous_healthy_generation_preserved_until_health_pass")
        is not True
        or package.get("installation_alone_enables_retrieval") is not False
    ):
        errors.append("local semantic component package integrity changed")

    capacity = payload.get("capacity") or {}
    if (
        capacity.get("first_install_required_free_bytes") != 971917233
        or capacity.get("check_before_download") is not True
        or capacity.get("check_before_extraction") is not True
        or capacity.get("check_before_index_build") is not True
        or capacity.get("reclaim_only_owned_staging_archives_and_superseded_generations")
        is not True
        or capacity.get("never_delete_library_pdfs_markdown_or_user_notes") is not True
        or capacity.get("bm25_remains_available") is not True
    ):
        errors.append("local semantic component capacity boundary changed")

    lifecycle = payload.get("lifecycle") or {}
    if (
        tuple(lifecycle.get("internal_states") or ()) != EXPECTED_INTERNAL_STATES
        or tuple(lifecycle.get("public_states") or ()) != EXPECTED_PUBLIC_STATES
        or lifecycle.get("active_query_never_waits_for_preparation") is not True
        or lifecycle.get("active_query_uses_bm25_until_ready") is not True
        or lifecycle.get("stale_index_is_never_queried") is not True
        or lifecycle.get("user_cancel_stays_paused_across_restart") is not True
        or lifecycle.get("non_user_interruption_recovers_automatically_after_restart")
        is not True
        or lifecycle.get("product_mode_control_visible_before_backend_recovery_e2e")
        is not False
    ):
        errors.append("local semantic component lifecycle transitions changed")

    recovery = payload.get("automatic_recovery") or {}
    observed_recovery = {key: recovery.get(key) for key in EXPECTED_RECOVERY}
    if (
        observed_recovery != EXPECTED_RECOVERY
        or recovery.get("automatic_attempt_limit_per_manifest_version") != 2
        or recovery.get("backoff_seconds") != [2, 10]
        or recovery.get("raw_exception_or_path_exposed") is not False
        or recovery.get("integrity_failure_requires_user_retry") is not False
        or recovery.get("index_failure_requires_user_retry") is not False
    ):
        errors.append("local semantic component automatic recovery changed")

    removal = payload.get("removal") or {}
    if (
        removal.get("disable_new_semantic_queries_before_delete") is not True
        or removal.get("remove_component_generations") is not True
        or removal.get("remove_semantic_index_generations") is not True
        or removal.get("preserve_library_pdfs_markdown_notes_chat_and_bm25_index")
        is not True
        or removal.get("bm25_available_during_and_after_removal") is not True
        or removal.get("idempotent") is not True
    ):
        errors.append("local semantic component removal boundary changed")

    public = payload.get("public_contract") or {}
    if (
        public.get("technical_pipeline_state_exposed") is not False
        or public.get("raw_exception_path_or_checksum_exposed") is not False
        or public.get("retry_button_for_integrity_or_index_failures_exposed")
        is not False
        or public.get("no_failure_state_without_bm25_or_one_concrete_user_action")
        is not True
    ):
        errors.append("local semantic component public failure contract changed")

    gates = payload.get("promotion_gates") or {}
    boundary = payload.get("product_boundary") or {}
    if (
        gates.get("contract_validator_and_mutation_tests_pass") is not True
        or any(
            gates.get(key) is not False
            for key in (
                "component_manifest_download_resume_capacity_hash_extract_tests_pass",
                "automatic_retry_quarantine_restart_tests_pass",
                "atomic_install_health_rollback_tests_pass",
                "index_stale_cancel_rebuild_tests_pass",
                "removal_and_locked_file_cleanup_tests_pass",
                "backend_api_contract_tests_pass",
                "ordinary_user_end_to_end_recovery_tests_pass",
                "ordinary_user_surface_allowed_before_all_implementation_gates",
            )
        )
        or boundary
        != {
            "library_papers_only": True,
            "selected_paper_scope_only": True,
            "evidence_retrieval_only": True,
            "writing_or_submission_workflow_allowed": False,
            "external_web_content_used": False,
            "uncited_model_knowledge_used": False,
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
        }
    ):
        errors.append("local semantic component promotion or product boundary changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            "[FAIL] local semantic component lifecycle contract unreadable: "
            f"{type(exc).__name__}"
        )
        return 1
    errors = validate_contract(payload)
    if sha256_file(DEFAULT_CONTRACT) != EXPECTED_CONTRACT_SHA256:
        errors.append("local semantic component lifecycle contract SHA-256 changed")
    if errors:
        for error in dict.fromkeys(errors):
            print(f"[FAIL] {error}")
        return 1
    print("[OK] local semantic component lifecycle v2 contract is frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
