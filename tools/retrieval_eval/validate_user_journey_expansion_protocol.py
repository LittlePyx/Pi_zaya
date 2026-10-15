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
    REPO_ROOT / "docs" / "retrieval_user_journey_expansion_protocol_v1.json"
)


def _valid_path_hash(row: dict[str, Any], path_key: str, sha_key: str) -> bool:
    path = REPO_ROOT / str(row.get(path_key) or "")
    return path.is_file() and row.get(sha_key) == sha256_file(path)


def validate_protocol(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("user-journey protocol schema changed")
    if (
        payload.get("curation_status")
        != "frozen_before_question_authoring_or_qrel_inspection"
    ):
        errors.append("user-journey protocol was not frozen before authoring")
    predecessor = payload.get("predecessor_boundary") or {}
    for path_key, sha_key in (
        ("development_fixture_path", "development_fixture_sha256"),
        ("semantic_fixture_path", "semantic_fixture_sha256"),
        ("fusion_v2_invalidation_path", "fusion_v2_invalidation_sha256"),
    ):
        if not _valid_path_hash(predecessor, path_key, sha_key):
            errors.append(f"predecessor evidence changed: {path_key}")
    if any(
        predecessor.get(key) is not True
        for key in (
            "previous_case_ids_forbidden",
            "previous_query_text_reuse_forbidden",
            "previous_qrels_not_extended_in_place",
        )
    ):
        errors.append("previous-evidence independence boundary changed")

    product = payload.get("product_boundary") or {}
    if (
        product.get("library_sources_only") is not True
        or product.get("external_web_or_model_knowledge_for_labels") is not False
        or product.get("answer_generation_evaluated") is not False
        or len(product.get("allowed_user_goals") or []) != 5
        or len(product.get("forbidden_user_goals") or []) != 5
    ):
        errors.append("reading-only product boundary changed")

    target = payload.get("target_fixture") or {}
    if (
        target.get("path") != "docs/retrieval_user_journey_expansion_v1.json"
        or target.get("case_count") != 18
        or target.get("locale_counts") != {"zh": 12, "en": 6}
        or target.get("journey_type_counts")
        != {
            "single_paper_mechanism": 4,
            "single_paper_boundary_or_limitation": 4,
            "experimental_configuration_or_metric": 4,
            "cross_paper_evidence_comparison": 4,
            "equation_figure_or_table_location": 2,
        }
        or target.get("minimum_distinct_primary_documents") != 12
        or target.get("maximum_primary_cases_per_document") != 2
        or target.get("cross_paper_case_count") != 4
        or target.get("cross_paper_minimum_relevant_documents_each") != 2
    ):
        errors.append("target fixture coverage contract changed")

    sequence = payload.get("authoring_sequence") or {}
    if sequence.get("retriever_execution_before_step_5") is not False:
        errors.append("retriever-free authoring boundary changed")
    label = payload.get("label_policy") or {}
    if (
        label.get("unit") != "indexed_chunk"
        or label.get("minimum_relevant_chunks_per_case") != 1
        or label.get("maximum_relevant_chunks_per_case") != 6
        or label.get("maximum_chunks_per_evidence_contract") != 3
        or label.get("grades")
        != {
            "3": "direct evidence that resolves the requested reading question",
            "2": "direct partial or contextual evidence needed with another labeled chunk",
        }
        or label.get("scope_contains_every_relevant_document") is not True
        or label.get("cross_paper_requires_relevance_in_each_compared_document")
        is not True
        or label.get("labels_selected_without_retriever_rank") is not True
        or label.get("unresolved_or_ambiguous_questions_are_dropped_not_forced")
        is not True
    ):
        errors.append("qrel label policy changed")
    required_fields = [
        "chunk_id",
        "doc_id",
        "grade",
        "page_start",
        "page_end",
        "text_sha256",
        "evidence_role",
    ]
    if label.get("required_label_fields") != required_fields:
        errors.append("qrel evidence audit fields changed")

    independence = payload.get("query_independence_checks") or {}
    if (
        independence.get("exact_case_id_overlap_with_previous") != 0
        or independence.get("normalized_exact_query_overlap_with_previous") != 0
        or independence.get("english_max_contiguous_evidence_token_copy") != 5
        or independence.get(
            "chinese_queries_against_english_evidence_are_translation_paraphrases"
        )
        is not True
        or independence.get("retrieval_result_fields_forbidden_in_fixture")
        != ["bm25_rank", "dense_rank", "fusion_rank", "retrieved_score"]
    ):
        errors.append("query independence or leakage checks changed")

    split = payload.get("deterministic_split") or {}
    if split != {
        "algorithm": "ascending_sha256(seed:locale:case_id)_within_locale",
        "seed": "pizaya-user-journey-holdout-v1",
        "holdout_locale_quota": {"zh": 4, "en": 2},
        "calibration_case_count": 12,
        "holdout_case_count": 6,
        "holdout_hidden_from_future_candidate_design": True,
        "holdout_maximum_candidate_runs": 1,
        "split_materialized_only_after_fixture_audit": True,
    }:
        errors.append("deterministic holdout boundary changed")
    future = payload.get("future_use") or {}
    if future != {
        "first_required_report": "product BM25 baseline on all 18 cases before any fusion-v3 protocol",
        "calibration_allowed_for_future_protocol_design": True,
        "holdout_allowed_for_tuning": False,
        "fusion_v2_candidate_results_cannot_be_reclassified_with_new_fixture": True,
        "product_enabled": False,
    }:
        errors.append("future-use or product boundary changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"[FAIL] user-journey expansion protocol cannot be read: {type(exc).__name__}"
        )
        return 2
    errors = validate_protocol(payload)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] retrieval user-journey expansion protocol frozen before authoring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
