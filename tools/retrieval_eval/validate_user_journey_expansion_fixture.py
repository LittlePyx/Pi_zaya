from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import validate_qrels_fixture
from kb.store import load_all_chunks
from kb.tokenize import tokenize
from tools.retrieval_eval.build_user_journey_expansion_fixture import (
    DEFAULT_BASE_FIXTURE,
    DEFAULT_DB_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    DEFAULT_PROTOCOL,
    DEFAULT_REGISTRY,
    build_fixture,
    validate_registry,
)


def _normalized_query(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _longest_contiguous_overlap(left: str, right: str) -> int:
    a = tokenize(left)
    b = tokenize(right)
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    longest = 0
    for left_token in a:
        current = [0] * (len(b) + 1)
        for index, right_token in enumerate(b, start=1):
            if left_token == right_token:
                current[index] = previous[index - 1] + 1
                longest = max(longest, current[index])
        previous = current
    return longest


def validate_fixture(payload: dict[str, Any]) -> list[str]:
    errors = validate_qrels_fixture(payload, repo_root=REPO_ROOT)
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    registry = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
    errors.extend(validate_registry(registry, protocol=protocol, manifest=manifest))
    try:
        expected = build_fixture()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return errors + [
            f"user-journey fixture cannot be regenerated: {type(exc).__name__}"
        ]
    if payload != expected:
        errors.append(
            "user-journey fixture differs from rank-independent local-source projection"
        )
        return errors
    if (
        payload.get("curation_status")
        != "frozen_before_any_retrieval_baseline_or_fusion_v3_design"
        or payload.get("retriever_executed_during_authoring_or_labeling") is not False
        or payload.get("answer_generation_evaluated") is not False
    ):
        errors.append("fixture freeze or reading-only boundary changed")
    base = json.loads(DEFAULT_BASE_FIXTURE.read_text(encoding="utf-8"))
    if payload.get("corpus") != base.get("corpus"):
        errors.append("user-journey corpus identity changed")

    cases = payload.get("cases") or []
    locale_counts = Counter(str(case.get("locale") or "") for case in cases)
    type_counts = Counter(str(case.get("evaluation_focus") or "") for case in cases)
    primary_counts = Counter(str(case.get("primary_doc_id") or "") for case in cases)
    target = protocol["target_fixture"]
    if dict(locale_counts) != target["locale_counts"]:
        errors.append("fixture locale distribution changed")
    if dict(type_counts) != target["journey_type_counts"]:
        errors.append("fixture journey distribution changed")
    if (
        len(primary_counts) < target["minimum_distinct_primary_documents"]
        or max(primary_counts.values(), default=0)
        > target["maximum_primary_cases_per_document"]
    ):
        errors.append("fixture primary-document diversity changed")

    previous_sources = [
        json.loads(DEFAULT_BASE_FIXTURE.read_text(encoding="utf-8")),
        json.loads(
            (
                REPO_ROOT / protocol["predecessor_boundary"]["semantic_fixture_path"]
            ).read_text(encoding="utf-8")
        ),
    ]
    previous_ids = {
        str(case.get("id") or "")
        for source in previous_sources
        for case in source.get("cases") or []
    }
    previous_queries = {
        _normalized_query(case.get("query") or case.get("question"))
        for source in previous_sources
        for case in source.get("cases") or []
    }
    chunk_text = {
        str(chunk.get("id") or ""): str(chunk.get("text") or "")
        for chunk in load_all_chunks(DEFAULT_DB_ROOT)
    }
    for case in cases:
        case_id = str(case.get("id") or "")
        query = str(case.get("query") or "")
        if case_id in previous_ids or _normalized_query(query) in previous_queries:
            errors.append(f"fixture reuses previous query evidence: {case_id}")
        labels = case.get("labels") or []
        relevant_docs = {str(label.get("doc_id") or "") for label in labels}
        if case.get(
            "evaluation_focus"
        ) == "cross_paper_evidence_comparison" and relevant_docs != set(
            case.get("scope_doc_ids") or []
        ):
            errors.append(f"cross-paper qrels are incomplete: {case_id}")
        if case.get("locale") == "en":
            longest = max(
                (
                    _longest_contiguous_overlap(
                        query, chunk_text[str(label["chunk_id"])]
                    )
                    for label in labels
                ),
                default=0,
            )
            if (
                longest
                > protocol["query_independence_checks"][
                    "english_max_contiguous_evidence_token_copy"
                ]
            ):
                errors.append(
                    f"English question copies too much evidence text: {case_id}"
                )
    forbidden_fields = protocol["query_independence_checks"][
        "retrieval_result_fields_forbidden_in_fixture"
    ]
    serialized = json.dumps(payload, ensure_ascii=False)
    if any(f'"{field}"' in serialized for field in forbidden_fields):
        errors.append("fixture contains forbidden retrieval-result fields")

    split = payload.get("split") or {}
    all_ids = {str(case["id"]) for case in cases}
    holdout = set(split.get("holdout_case_ids") or [])
    calibration = set(split.get("calibration_case_ids") or [])
    if (
        holdout & calibration
        or holdout | calibration != all_ids
        or len(holdout) != 6
        or len(calibration) != 12
        or split.get("holdout_allowed_for_tuning") is not False
    ):
        errors.append("fixture calibration/holdout split changed")
    holdout_locales = Counter(
        str(case["locale"]) for case in cases if str(case["id"]) in holdout
    )
    if dict(holdout_locales) != protocol["deterministic_split"]["holdout_locale_quota"]:
        errors.append("fixture holdout locale quota changed")
    expected_holdout: list[str] = []
    for locale, quota in protocol["deterministic_split"][
        "holdout_locale_quota"
    ].items():
        locale_ids = [str(case["id"]) for case in cases if case["locale"] == locale]
        ranked = sorted(
            locale_ids,
            key=lambda case_id: hashlib.sha256(
                f"{protocol['deterministic_split']['seed']}:{locale}:{case_id}".encode()
            ).hexdigest(),
        )
        expected_holdout.extend(ranked[: int(quota)])
    if split.get("holdout_case_ids") != sorted(expected_holdout):
        errors.append("fixture deterministic holdout membership changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] user-journey fixture cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_fixture(payload)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] 18 user-journey qrels frozen before retrieval with 12/6 split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
