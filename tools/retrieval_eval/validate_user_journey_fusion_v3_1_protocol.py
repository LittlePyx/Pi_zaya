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
    REPO_ROOT / "docs" / "retrieval_user_journey_fusion_v3_1_protocol.json"
)
DEFAULT_V3_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_user_journey_fusion_v3_protocol.json"
)
DEFAULT_INVALIDATION = (
    REPO_ROOT / "docs" / "retrieval_user_journey_fusion_v3_execution_invalidation.json"
)
UNCHANGED_SECTIONS = (
    "partition",
    "frozen_representation",
    "frozen_candidate",
    "depth_contract",
    "product_control",
    "calibration_gate",
    "holdout_gate",
    "combined_gate",
    "semantic_acceptance_gate",
    "safety_boundary",
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_protocol(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("id")
        != "retrieval_user_journey_fusion_v3_1_replacement_protocol"
        or payload.get("curation_status")
        != "frozen_after_v3_infrastructure_invalidation_before_replacement_measurement"
    ):
        errors.append("fusion-v3.1 protocol identity changed")
    predecessor = payload.get("predecessor") or {}
    if (
        predecessor.get("protocol_path")
        != "docs/retrieval_user_journey_fusion_v3_protocol.json"
        or predecessor.get("protocol_sha256") != sha256_file(DEFAULT_V3_PROTOCOL)
        or predecessor.get("invalidation_path")
        != "docs/retrieval_user_journey_fusion_v3_execution_invalidation.json"
        or predecessor.get("invalidation_sha256") != sha256_file(DEFAULT_INVALIDATION)
    ):
        errors.append("fusion-v3.1 predecessor identity changed")
    try:
        v3 = json.loads(DEFAULT_V3_PROTOCOL.read_text(encoding="utf-8"))
        invalidation = json.loads(DEFAULT_INVALIDATION.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return errors + [
            f"fusion-v3.1 predecessor cannot be read: {type(exc).__name__}"
        ]
    expected_hashes = {key: _canonical_sha256(v3[key]) for key in UNCHANGED_SECTIONS}
    if payload.get("unchanged_contract_section_sha256") != expected_hashes:
        errors.append(
            "fusion-v3.1 changed an inherited candidate, depth, gate, or safety contract"
        )
    if (invalidation.get("effective_decision") or {}).get(
        "candidate_configuration_change_allowed"
    ) is not False:
        errors.append("fusion-v3.1 lacks a closed unchanged-candidate invalidation")
    correction = payload.get("sole_correction") or {}
    if correction != {
        "component": "frozen_product_control_projection",
        "metric": "mrr",
        "source_ranking_depth": 20,
        "selection_ranking_depth": 10,
        "rule": "If the frozen baseline first_relevant_rank is null or greater than 10, expected selection-depth MRR is 0. Otherwise expected MRR remains the frozen baseline value.",
        "affected_case_mode_pairs": [
            "scoped:journey-dualscale-resolution-results",
            "full_library:journey-piln-vs-pidl-supervision",
        ],
        "candidate_result_or_gate_observed_before_correction": False,
    }:
        errors.append("fusion-v3.1 sole depth-projection correction changed")
    if payload.get("replacement_execution") != {
        "maximum_runs": 1,
        "same_command_flow": "calibration_then_automatic_conditional_holdout",
        "candidate_changes": 0,
        "representation_changes": 0,
        "index_builds": 0,
        "ranking_or_reporting_depth_changes": 0,
        "gate_changes": 0,
        "holdout_policy_changes": 0,
        "product_enabled": False,
    }:
        errors.append("fusion-v3.1 replacement execution boundary changed")
    return errors


def effective_protocol(payload: dict[str, Any]) -> dict[str, Any]:
    errors = validate_protocol(payload)
    if errors:
        raise ValueError("invalid fusion-v3.1 protocol: " + "; ".join(errors))
    v3 = json.loads(DEFAULT_V3_PROTOCOL.read_text(encoding="utf-8"))
    return {
        **v3,
        "id": payload["id"],
        "curation_status": payload["curation_status"],
        "replacement_protocol": payload,
    }


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] fusion-v3.1 protocol cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_protocol(payload)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] unchanged-candidate fusion-v3.1 replacement frozen before measurement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
