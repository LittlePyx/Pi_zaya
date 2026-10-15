from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.run_portable_e5_full_size_lifecycle import (
    _fingerprint,
    evaluate_gate,
)
from tools.retrieval_eval.validate_portable_e5_full_size_lifecycle_protocol_v3 import (
    EXPECTED_PROTOCOL_SHA256,
    validate_protocol,
)


DEFAULT_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_portable_e5_full_size_lifecycle_protocol_v3.json"
)
DEFAULT_REPORT = (
    REPO_ROOT / "docs" / "retrieval_portable_e5_full_size_lifecycle_v3.json"
)
EXPECTED_REPORT_SHA256 = (
    "a23a0135c88d5f676eaacc2eb2166fa301ee10fc6d5f08745dc5aec564b739a0"
)
EXPECTED_RESULT_FINGERPRINT = (
    "b1bde7acb30be4c2b67b23bbd71df10c61e49ed2e66a39d92c3da87dd09db80d"
)


def _load_protocol(errors: list[str]) -> dict[str, Any]:
    try:
        protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        errors.append("release-component lifecycle protocol unreadable")
        return {}
    if (
        sha256_file(DEFAULT_PROTOCOL) != EXPECTED_PROTOCOL_SHA256
        or validate_protocol(protocol)
    ):
        errors.append("release-component lifecycle protocol changed")
    return protocol


def validate_report(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    protocol = _load_protocol(errors)
    if (
        payload.get("schema_version") != 1
        or payload.get("report_kind")
        != "retrieval_portable_e5_full_size_lifecycle_v3"
        or payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
    ):
        errors.append("release-component lifecycle report identity changed")
    timings = payload.get("timings") or {}
    if any(
        float(timings.get(key) or 0) <= 0
        for key in (
            "base_extraction_ms",
            "worker_wall_ms",
            "install_ready_ms",
            "cold_load_ms",
            "warm_query_p95_ms",
            "stale_repair_ms",
            "removal_ms",
        )
    ):
        errors.append("release-component lifecycle timings changed")
    runtime = payload.get("runtime") or {}
    if (
        runtime.get("python_version") != "3.10.11"
        or runtime.get("cold_start") is not True
        or runtime.get("index_cache_hit") is not False
        or float(runtime.get("peak_process_rss_mib") or 0) <= 0
    ):
        errors.append("release-component lifecycle runtime changed")

    expected_queries = {
        str(item["case_id"]): item
        for item in (protocol.get("execution") or {}).get("expected_queries") or []
    }
    observed_queries = {
        str(item.get("case_id") or ""): item for item in payload.get("queries") or []
    }
    if set(observed_queries) != set(expected_queries) or any(
        observed_queries[case_id].get("ranked_chunk_ids")
        != expected.get("ranked_chunk_ids")
        or observed_queries[case_id].get("first_relevant_rank")
        != expected.get("first_relevant_rank")
        or observed_queries[case_id].get("used_mode") != "hybrid_local"
        or observed_queries[case_id].get("fallback_reason_id") != ""
        or observed_queries[case_id].get("source_escape_count") != 0
        for case_id, expected in expected_queries.items()
    ):
        errors.append("release-component lifecycle query evidence changed")
    repair = payload.get("stale_repair") or {}
    if (
        repair.get("state_before") != "stale"
        or repair.get("state_after") != "ready"
        or repair.get("lifecycle_state") != "ready"
        or repair.get("index_build_call_count") != 1
    ):
        errors.append("release-component lifecycle stale repair changed")
    removal = payload.get("removal") or {}
    public = removal.get("public") or {}
    if (
        removal.get("created") is not True
        or removal.get("lifecycle_state") != "not_installed"
        or removal.get("component_pointer_present") is not False
        or removal.get("index_pointer_present") is not False
        or removal.get("cleanup_pending_after_worker") is not True
        or removal.get("cleanup_completed_after_worker_exit") is not True
        or removal.get("remaining_component_generation_count") != 0
        or public.get("state") != "not_installed"
        or public.get("action_id") != "enable"
        or public.get("bm25_available") is not True
    ):
        errors.append("release-component lifecycle removal changed")
    execution = payload.get("execution") or {}
    if (
        execution.get("explicit_consent") is not True
        or execution.get("download_network_used") is not False
        or execution.get("local_archive_adapter_calls") != 1
        or execution.get("transport_factory_calls") != 1
        or execution.get("bm25_available_in_all_public_states") is not True
        or execution.get("product_enabled") is not False
        or execution.get("ordinary_user_surface_exposed") is not False
        or execution.get("storage_layout_max_path_chars") != 214
    ):
        errors.append("release-component lifecycle execution changed")
    expected_gate = evaluate_gate(payload, protocol) if protocol else {}
    observed_gate = payload.get("gate") or {}
    if not expected_gate or observed_gate != expected_gate or not all(observed_gate.values()):
        errors.append("release-component lifecycle gates changed")
    decision = payload.get("decision") or {}
    if (
        decision.get("status")
        != "release_component_lifecycle_passed_authorize_opt_in_ui_e2e"
        or decision.get("all_gates_passed") is not True
        or decision.get("product_enabled") is not False
        or decision.get("ordinary_user_surface_exposed") is not False
        or decision.get("next_action") != "add_management_api_and_failure_path_e2e"
    ):
        errors.append("release-component lifecycle decision changed")
    if (
        payload.get("result_fingerprint") != EXPECTED_RESULT_FINGERPRINT
        or _fingerprint(payload) != EXPECTED_RESULT_FINGERPRINT
    ):
        errors.append("release-component lifecycle fingerprint changed")
    return list(dict.fromkeys(errors))


def main() -> int:
    try:
        payload = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] release-component report unreadable: {type(exc).__name__}")
        return 1
    errors = validate_report(payload)
    if sha256_file(DEFAULT_REPORT) != EXPECTED_REPORT_SHA256:
        errors.append("release-component lifecycle report SHA-256 changed")
    if errors:
        for error in dict.fromkeys(errors):
            print(f"[FAIL] {error}")
        return 1
    print("[OK] portable E5 release-component lifecycle v3 pass is pinned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
