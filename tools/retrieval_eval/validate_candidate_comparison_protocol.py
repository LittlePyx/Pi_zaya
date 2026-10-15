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


DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "retrieval_candidate_comparison_protocol_v1.json"
SEED = "pizaya-dense-inner-confirmation-v1"


def _expected_split(cases: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    groups: dict[str, list[str]] = {}
    for case in cases:
        groups.setdefault(str(case.get("locale") or ""), []).append(str(case.get("id") or ""))
    quotas = {"en": 1, "zh": 11}
    confirmation: list[str] = []
    for locale, ids in groups.items():
        ranked = sorted(
            ids,
            key=lambda case_id: hashlib.sha256(
                f"{SEED}:{locale}:{case_id}".encode()
            ).hexdigest(),
        )
        confirmation.extend(ranked[: quotas.get(locale, 0)])
    confirmation = sorted(confirmation)
    all_ids = {str(case.get("id") or "") for case in cases}
    return sorted(all_ids - set(confirmation)), confirmation


def validate_protocol(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("protocol schema version changed")
    if payload.get("curation_status") != "frozen_before_candidate_shortlist":
        errors.append("protocol must be frozen before candidate shortlisting")
    development = payload.get("development_fixture") or {}
    development_path = REPO_ROOT / str(development.get("path") or "")
    try:
        fixture = json.loads(development_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"development fixture cannot be read: {type(exc).__name__}"]
    cases = list(fixture.get("cases") or [])
    if (
        development.get("sha256") != sha256_file(development_path)
        or development.get("case_count") != 41
        or development.get("allowed_for_tuning") is not True
    ):
        errors.append("development fixture contract changed")
    split = payload.get("inner_split") or {}
    expected_screen, expected_confirmation = _expected_split(cases)
    if split.get("seed") != SEED or split.get("confirmation_locale_quota") != {"en": 1, "zh": 11}:
        errors.append("inner split seed or locale quota changed")
    if split.get("screening_case_ids") != expected_screen:
        errors.append("screening membership changed")
    if split.get("confirmation_case_ids") != expected_confirmation:
        errors.append("confirmation membership changed")
    if split.get("screening_case_count") != 29 or split.get("confirmation_case_count") != 12:
        errors.append("inner split counts changed")
    if split.get("confirmation_hidden_from_candidate_measurement") is not True:
        errors.append("confirmation partition is not protected")
    budget = payload.get("candidate_budget") or {}
    expected_budget = {
        "maximum_new_candidates": 3,
        "maximum_variants_per_candidate": 1,
        "maximum_safetensors_bytes_each": 1500000000,
        "maximum_embedding_dimension": 1024,
        "required_declared_license": True,
        "safetensors_required": True,
        "standard_transformers_auto_model_required": True,
        "trust_remote_code": False,
        "immutable_revision_required": True,
        "weight_sha256_required": True,
        "official_query_contract_required": True,
        "paper_text_uploaded": False,
    }
    if budget != expected_budget:
        errors.append("candidate safety or resource budget changed")
    retrieval = payload.get("frozen_retrieval_contract") or {}
    expected_fusion = {
        "algorithm": "rrf",
        "rrf_k": 20.0,
        "sparse_weight": 2.0,
        "dense_weight": 1.0,
        "candidate_multiplier": 4,
        "protected_sparse_prefix": 5,
        "dense_source_gate_depth": 5,
    }
    if retrieval.get("fusion") != expected_fusion:
        errors.append("source-safe fusion contract changed")
    if retrieval.get("parent_chunks_unchanged") is not True or retrieval.get("cross_encoder") is not False:
        errors.append("representation or reranker scope changed")
    if retrieval.get("no_parameter_or_prompt_changes_after_any_result") is not True:
        errors.append("post-result tuning prohibition changed")
    screening = payload.get("screening_stage") or {}
    if screening.get("all_candidates_measured_once") is not True or screening.get("all_required") is not True:
        errors.append("screening run or all-pass rule changed")
    if screening.get("data") != "screening_case_ids_only":
        errors.append("screening data boundary changed")
    resources = screening.get("resource_checks") or {}
    if resources.get("index_size_ratio_vs_dense_control_max") != 1.5:
        errors.append("screening index budget changed")
    nomination = payload.get("nomination") or {}
    if nomination.get("maximum_nominees") != 1 or nomination.get("eligible_candidates_only") is not True:
        errors.append("nomination cardinality or eligibility changed")
    confirmation = payload.get("confirmation_stage") or {}
    if (
        confirmation.get("maximum_runs") != 1
        or confirmation.get("nominee_only") is not True
        or confirmation.get("data") != "confirmation_case_ids_only"
        or confirmation.get("all_required") is not True
    ):
        errors.append("confirmation boundary changed")
    combined = payload.get("combined_development_gate") or {}
    if (
        combined.get("raw_recall_at_10_min_gain_in_any_mode") != 0.05
        or combined.get("safe_rrf_recall_at_10_min_gain_in_both_modes") != 0.02
        or combined.get("all_required") is not True
    ):
        errors.append("combined development promotion gate changed")
    acceptance = payload.get("frozen_acceptance_gate") or {}
    acceptance_path = REPO_ROOT / str(acceptance.get("path") or "")
    if (
        acceptance.get("sha256") != sha256_file(acceptance_path)
        or acceptance.get("case_count") != 12
        or acceptance.get("allowed_for_tuning") is not False
        or acceptance.get("maximum_candidate_replays") != 1
        or acceptance.get("run_only_after_all_prior_gates_pass") is not True
        or acceptance.get("safe_rrf_recall_at_10_min_gain_vs_product_in_both_modes") != 0.15
    ):
        errors.append("frozen acceptance boundary changed")
    failure = payload.get("failure_policy") or {}
    if (
        failure.get("rejected_candidate_retest_under_v1") is not False
        or failure.get("product_enabled") is not False
        or failure.get("ordinary_user_surface_exposed") is not False
        or failure.get("next_iteration_requires_new_versioned_protocol_or_new_evidence") is not True
    ):
        errors.append("failure or product-exposure policy changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] comparison protocol cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_protocol(payload)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] dense candidate comparison protocol frozen before shortlisting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
