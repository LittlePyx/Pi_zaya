from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file


DEFAULT_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_user_journey_fusion_v3_1_semantic_protocol.json"
)


def _path_is_pinned(evidence: dict[str, Any], prefix: str) -> bool:
    path = REPO_ROOT / str(evidence.get(f"{prefix}_path") or "")
    return path.is_file() and evidence.get(f"{prefix}_sha256") == sha256_file(path)


def validate_protocol(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("id")
        != "retrieval_user_journey_fusion_v3_1_semantic_acceptance_protocol"
        or payload.get("curation_status")
        != "frozen_after_selection_before_semantic_acceptance_runner_and_measurement"
    ):
        errors.append("fusion-v3.1 semantic protocol identity changed")
    evidence = payload.get("evidence") or {}
    for prefix in (
        "replacement_protocol",
        "selection_report",
        "semantic_fixture",
        "semantic_predecessor_report",
        "protected_tail_implementation",
    ):
        if not _path_is_pinned(evidence, prefix):
            errors.append(f"fusion-v3.1 semantic evidence identity changed: {prefix}")
    try:
        selection = json.loads(
            (REPO_ROOT / evidence["selection_report_path"]).read_text(encoding="utf-8")
        )
        predecessor = json.loads(
            (REPO_ROOT / evidence["semantic_predecessor_report_path"]).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        return errors + [
            f"semantic acceptance evidence cannot be read: {type(exc).__name__}"
        ]
    eligibility = payload.get("eligibility") or {}
    if (
        (selection.get("decision") or {}).get("outcome")
        != "eligible_for_semantic_acceptance"
        or selection.get("product_enabled") is not False
        or eligibility
        != {
            "required_selection_outcome": "eligible_for_semantic_acceptance",
            "selection_product_enabled": False,
            "case_count": 12,
            "maximum_runs": 1,
        }
    ):
        errors.append("fusion-v3.1 semantic eligibility changed")
    selected = selection.get("candidate") or {}
    dense_protocol = json.loads(
        (
            REPO_ROOT / "docs" / "retrieval_user_journey_qwen_dense_protocol_v1.json"
        ).read_text(encoding="utf-8")
    )
    expected_candidate = {
        "candidate_id": selected.get("candidate_id"),
        "protected_sparse_prefix": selected.get("protected_sparse_prefix"),
        "sparse_tail_window": selected.get("sparse_tail_window"),
        "dense_novelty_quota": selected.get("dense_novelty_quota"),
        "candidate_multiplier": selected.get("candidate_multiplier"),
        "fusion_output_depth": selected.get("fusion_output_depth"),
        "candidate_retrieval_depth": selected.get("sparse_and_dense_candidate_depth"),
        "qwen_provider_identity": dense_protocol["frozen_representation"][
            "provider_identity"
        ],
        "existing_index_reused_without_build": True,
    }
    if payload.get("unchanged_candidate") != expected_candidate:
        errors.append("fusion-v3.1 semantic candidate changed")
    if payload.get("execution_contract") != {
        "ranking_depth": 10,
        "reported_metric_depths": [5, 10],
        "scoped_and_full_library": True,
        "query_embedding_cache": True,
        "expected_unique_query_embeddings": 12,
        "product_control_must_match_predecessor_for": [
            "recall@5",
            "recall@10",
            "wrong_source_rate@5",
        ],
        "fusion_or_model_variants": 0,
        "index_builds": 0,
    }:
        errors.append("fusion-v3.1 semantic execution contract changed")
    acceptance = payload.get("acceptance_gate") or {}
    if acceptance != {
        "metric": "recall@10",
        "minimum_gain_vs_product_in_each_mode": 0.15,
        "required_modes": ["scoped", "full_library"],
        "recall@5_equal_to_product_in_each_mode": True,
        "wrong_source_rate@5_equal_to_product_in_each_mode": True,
        "all_required": True,
    }:
        errors.append("fusion-v3.1 semantic acceptance gate changed")
    predecessor_product = predecessor.get("channels", {}).get("product_bm25", {})
    if any(
        predecessor_product.get(mode, {}).get("recall@10") is None
        for mode in ("scoped", "full_library")
    ):
        errors.append("semantic predecessor product control is incomplete")
    if payload.get("failure_policy") != {
        "retry_or_candidate_change": False,
        "product_shadow_allowed_on_failure": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }:
        errors.append("fusion-v3.1 semantic failure policy changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"[FAIL] semantic acceptance protocol cannot be read: {type(exc).__name__}"
        )
        return 2
    errors = validate_protocol(payload)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] fusion-v3.1 semantic acceptance frozen before runner and measurement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
