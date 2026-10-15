from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file


DEFAULT_INVALIDATION = (
    REPO_ROOT / "docs" / "retrieval_user_journey_fusion_v3_execution_invalidation.json"
)
V3_REPORT = REPO_ROOT / "docs" / "retrieval_user_journey_fusion_v3_selection.json"


def validate_invalidation(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("report_kind")
        != "retrieval_user_journey_fusion_v3_execution_invalidation"
        or payload.get("status") != "invalid_execution_closed_without_candidate_result"
    ):
        errors.append("fusion-v3 invalidation identity changed")
    protocol = REPO_ROOT / str(payload.get("protocol_path") or "")
    if (
        not protocol.is_file()
        or payload.get("protocol_sha256") != sha256_file(protocol)
        or payload.get("attempted_runner_path")
        != "tools/retrieval_eval/run_user_journey_fusion_v3_selection.py"
        or payload.get("attempted_runner_sha256")
        != "14fd282fa605f9d603ce4b2e4197ede901ab3cd5a3c98142fbcf22c757eedf6b"
    ):
        errors.append("fusion-v3 invalidation source identity changed")
    boundary = payload.get("execution_boundary") or {}
    if boundary != {
        "calibration_partition_executed": True,
        "calibration_query_count": 12,
        "calibration_candidate_result_observed": False,
        "canonical_report_written": False,
        "holdout_accessed": False,
        "semantic_acceptance_accessed": False,
        "product_enabled": False,
    }:
        errors.append("fusion-v3 invalidation execution boundary changed")
    failure = payload.get("failure") or {}
    if (
        failure.get("stage") != "pre_gate_product_control_projection_validation"
        or failure.get("exception") != "ValueError"
        or "rank depth 20" not in str(failure.get("root_cause") or "")
        or "rank depth 10" not in str(failure.get("root_cause") or "")
        or failure.get("candidate_or_gate_quality_conclusion_allowed") is not False
    ):
        errors.append("fusion-v3 invalidation failure account changed")
    decision = payload.get("effective_decision") or {}
    if (
        decision.get("fusion_v3_protocol_closed") is not True
        or decision.get("selection_result") != "none"
        or decision.get("holdout_allowed_under_v3") is not False
        or decision.get("semantic_acceptance_allowed_under_v3") is not False
        or decision.get("product_exposure_allowed") is not False
        or decision.get("candidate_configuration_change_allowed") is not False
        or "sole change is depth-aware BM25 MRR projection"
        not in str(decision.get("replacement_protocol_allowed") or "")
    ):
        errors.append("fusion-v3 invalidation no-promotion decision changed")
    if V3_REPORT.exists():
        errors.append("closed fusion-v3 execution unexpectedly has a canonical report")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_INVALIDATION.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] fusion-v3 invalidation cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_invalidation(payload)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print(
        "[OK] invalid fusion-v3 execution is closed before holdout and product access"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
