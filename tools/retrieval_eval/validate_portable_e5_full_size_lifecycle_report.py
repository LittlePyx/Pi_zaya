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
DEFAULT_REPORT = (
    REPO_ROOT / "docs" / "retrieval_portable_e5_full_size_lifecycle_v1.json"
)
EXPECTED_PROTOCOL_SHA256 = (
    "dc5205129d50fda61320cfb49023db216e719113f55260d86959171d61eee752"
)
EXPECTED_REPORT_SHA256 = (
    "8d8f3fdc5414d523dd408ff5d45dc0e3255e50c89a758cf04e244dad6ef60abb"
)
EXPECTED_RESULT_FINGERPRINT = (
    "362c52a8dc745fd62f4ca7739614f95314989cccb771f1fae2fd4b01ef136225"
)


def _fingerprint(payload: dict[str, Any]) -> str:
    source = {key: value for key, value in payload.items() if key != "result_fingerprint"}
    raw = json.dumps(
        source,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_report(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("report_kind")
        != "retrieval_portable_e5_full_size_lifecycle_v1"
        or payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
        or sha256_file(DEFAULT_PROTOCOL) != EXPECTED_PROTOCOL_SHA256
    ):
        errors.append("full-size lifecycle rejection identity changed")
    execution = payload.get("execution") or {}
    if (
        execution.get("single_clean_execution_started") is not True
        or execution.get("worker_completed") is not False
        or execution.get("automatic_attempt_count") != 2
        or execution.get("external_network_used") is not False
        or execution.get("user_library_written") is not False
        or execution.get("product_enabled") is not False
        or execution.get("ordinary_user_surface_exposed") is not False
    ):
        errors.append("full-size lifecycle rejection execution changed")
    failure = payload.get("failure") or {}
    if (
        failure.get("stage") != "atomic_component_extraction"
        or failure.get("internal_reason_id") != "component_archive_invalid"
        or failure.get("lifecycle_terminal_state") != "degraded_bm25"
        or failure.get("manifest_quarantined") is not True
        or failure.get("public_state") != "bm25_available"
        or failure.get("public_action_id") != "none"
        or failure.get("bm25_available") is not True
        or failure.get("user_repair_required") is not False
    ):
        errors.append("full-size lifecycle rejection failure contract changed")
    attribution = payload.get("attribution") or {}
    if (
        attribution.get("component_archive_bytes") != 296365646
        or attribution.get("component_archive_sha256")
        != "54d0f4fbd1c07913f5915dafbab224cf5510072fd767fcffd62da0e0b8ac3ad3"
        or attribution.get("zip_crc_test_passed") is not True
        or attribution.get("payload_previously_reproduced") is not True
        or attribution.get("longest_archive_member_characters") != 148
        or attribution.get("atomic_staging_root_characters") != 128
        or attribution.get("longest_target_absolute_path_characters") != 275
        or attribution.get("traditional_windows_max_path_characters") != 260
        or attribution.get("root_cause")
        != "atomic_generation_layout_exceeds_traditional_windows_path_budget"
        or attribution.get("inference") is not True
    ):
        errors.append("full-size lifecycle rejection attribution changed")
    gate = payload.get("gate") or {}
    if (
        gate.get("component_archive_integrity") is not True
        or gate.get("zero_network") is not True
        or gate.get("bm25_continuity") is not True
        or gate.get("product_remains_disabled") is not True
        or any(
            gate.get(key) is not False
            for key in (
                "full_size_install_completed",
                "runtime_health_completed",
                "ranking_exact",
                "stale_repair_completed",
                "safe_removal_completed",
                "all_required",
            )
        )
    ):
        errors.append("full-size lifecycle rejection gates changed")
    decision = payload.get("decision") or {}
    if (
        decision.get("status")
        != "full_size_lifecycle_rejected_keep_bm25_repair_backend"
        or decision.get("all_gates_passed") is not False
        or decision.get("verdict_may_be_changed") is not False
        or decision.get("product_enabled") is not False
        or decision.get("ordinary_user_surface_exposed") is not False
        or decision.get("next_action")
        != "shorten_and_preflight_windows_component_storage_layout_then_freeze_new_candidate"
    ):
        errors.append("full-size lifecycle rejection decision changed")
    if (
        payload.get("result_fingerprint") != EXPECTED_RESULT_FINGERPRINT
        or _fingerprint(payload) != EXPECTED_RESULT_FINGERPRINT
    ):
        errors.append("full-size lifecycle rejection fingerprint changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] full-size lifecycle rejection unreadable: {type(exc).__name__}")
        return 1
    errors = validate_report(payload)
    if sha256_file(DEFAULT_REPORT) != EXPECTED_REPORT_SHA256:
        errors.append("full-size lifecycle rejection report SHA-256 changed")
    if errors:
        for error in dict.fromkeys(errors):
            print(f"[FAIL] {error}")
        return 1
    print("[OK] portable E5 full-size lifecycle v1 rejection is pinned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
