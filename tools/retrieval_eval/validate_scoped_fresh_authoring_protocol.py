from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "retrieval_scoped_fresh_authoring_protocol_v1.json"


def _sha256(path: Path) -> str:
    value = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(value).hexdigest()


def _ids_sha256(values: list[str]) -> str:
    raw = json.dumps(sorted(values), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_protocol(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id") != "retrieval_scoped_fresh_authoring_v1"
        or payload.get("status") != "frozen_before_source_chunk_inspection"
    ):
        errors.append("scoped fresh authoring protocol identity changed")
    source = payload.get("source") or {}
    base_path = REPO_ROOT / str(source.get("base_fixture_path") or "")
    if (
        not base_path.is_file()
        or source.get("base_fixture_sha256") != _sha256(base_path)
        or source.get("retrieval_snapshot_sha256")
        != "09d20706edb877e5534f115f92b3252c0917caf23f3e91af55d37d7209557535"
        or source.get("corpus_fingerprint")
        != "20ad35d505d2936dbd80e90df6199e4b3d51fdf4cf8872a36c9211c5d6c99d9f"
        or source.get("previous_evidence_chunk_count") != 35
        or source.get("previous_evidence_chunk_ids_sha256")
        != "cd815404a0c9b5a6ce09134c5c4c0acd44739b03d037ef41b26b3a02b035d482"
    ):
        errors.append("scoped fresh authoring source changed")
    documents = payload.get("document_selection") or {}
    document_ids = list(documents.get("document_ids") or [])
    if (
        documents.get("rule")
        != "all_and_only_logical_documents_never_used_in_previous_scope_doc_ids"
        or documents.get("document_count") != 12
        or document_ids
        != [
            "3d-sp-video",
            "dl-spi-review",
            "fdm-metamaterials",
            "foveated-spi",
            "qclfm",
            "rt-spi-supp",
            "scigs",
            "scinerf",
            "seq-cs",
            "spd-review",
            "sph-biological",
            "spi-prospects",
        ]
        or documents.get("ordered_document_ids_sha256") != _ids_sha256(document_ids)
        or documents.get("one_case_per_document") is not True
        or documents.get("single_document_scope_only") is not True
    ):
        errors.append("scoped fresh authoring document selection changed")
    boundary = payload.get("authoring_boundary") or {}
    if (
        boundary.get("bm25_or_dense_retrieval_during_authoring_allowed") is not False
        or boundary.get("existing_candidate_rankings_or_scores_allowed") is not False
        or boundary.get("external_web_allowed") is not False
        or boundary.get("external_model_knowledge_allowed") is not False
        or boundary.get("query_must_be_answerable_from_registered_evidence") is not True
        or boundary.get("query_must_not_copy_a_complete_evidence_sentence") is not True
        or boundary.get("evidence_chunk_ids_must_be_disjoint_from_previous_suite") is not True
        or boundary.get("evidence_text_sha256_required") is not True
        or boundary.get("page_span_required") is not True
    ):
        errors.append("scoped fresh authoring boundary changed")
    case = payload.get("case_contract") or {}
    if (
        case.get("case_count") != 12
        or case.get("locale_quota") != {"zh": 6, "en": 6}
        or case.get("minimum_labels_per_case") != 1
        or case.get("maximum_labels_per_case") != 2
        or case.get("allowed_grades") != [2, 3]
        or case.get("output_path") != "docs/retrieval_scoped_fresh_cases_v1.json"
    ):
        errors.append("scoped fresh authoring case contract changed")
    future = payload.get("future_evaluation") or {}
    if future != {
        "candidate_configuration_changes_allowed": 0,
        "modes": ["scoped"],
        "minimum_recall_at_10_gain_vs_product": 0.02,
        "recall_at_5_no_regression": True,
        "mrr_no_regression": True,
        "ndcg_at_10_no_regression": True,
        "contract_recall_at_10_no_regression": True,
        "maximum_case_recall_at_10_regressions": 0,
        "maximum_case_mrr_regressions": 0,
        "maximum_case_contract_recall_at_10_regressions": 0,
        "product_enablement_on_pass": False,
    }:
        errors.append("scoped fresh authoring future gate changed")
    if payload.get("product_boundary") != {
        "library_papers_only": True,
        "evidence_retrieval_only": True,
        "writing_or_submission_workflow_allowed": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }:
        errors.append("scoped fresh authoring product boundary changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] scoped fresh authoring protocol unreadable: {type(exc).__name__}")
        return 1
    errors = validate_protocol(payload)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[OK] scoped fresh authoring protocol is frozen before source inspection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
