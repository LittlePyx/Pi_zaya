from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.validate_user_journey_expansion_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


DEFAULT_MANIFEST = (
    REPO_ROOT / "docs" / "retrieval_user_journey_authoring_manifest_v1.json"
)


def validate_manifest(payload: dict) -> list[str]:
    errors: list[str] = []
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    protocol_errors = validate_protocol(protocol)
    if protocol_errors:
        return ["source protocol invalid: " + "; ".join(protocol_errors)]
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "questions_frozen_before_qrel_chunk_inspection"
    ):
        errors.append("authoring manifest identity changed")
    if payload.get(
        "protocol_path"
    ) != "docs/retrieval_user_journey_expansion_protocol_v1.json" or payload.get(
        "protocol_sha256"
    ) != sha256_file(DEFAULT_PROTOCOL):
        errors.append("authoring manifest protocol identity changed")
    if (
        payload.get("authoring_basis")
        != "paper titles, library-level reading intents, and protocol slots only"
        or payload.get("retriever_executed") is not False
        or payload.get("qrel_chunk_inspection_started") is not False
    ):
        errors.append("rank-independent authoring boundary changed")
    cases = payload.get("cases") or []
    if len(cases) != 18:
        errors.append("authoring case count changed")
        return errors
    ids = [str(case.get("id") or "") for case in cases]
    if len(set(ids)) != len(ids) or any(
        not case_id.startswith("journey-") for case_id in ids
    ):
        errors.append("authoring case IDs are missing or duplicated")
    locale_counts = Counter(str(case.get("locale") or "") for case in cases)
    type_counts = Counter(str(case.get("journey_type") or "") for case in cases)
    primary_counts = Counter(str(case.get("primary_doc_id") or "") for case in cases)
    target = protocol["target_fixture"]
    if dict(locale_counts) != target["locale_counts"]:
        errors.append("authoring locale distribution changed")
    if dict(type_counts) != target["journey_type_counts"]:
        errors.append("authoring journey-type distribution changed")
    if len(primary_counts) < target["minimum_distinct_primary_documents"]:
        errors.append("authoring document diversity is too low")
    if (
        max(primary_counts.values(), default=0)
        > target["maximum_primary_cases_per_document"]
    ):
        errors.append("a primary document exceeds the case cap")
    previous = json.loads(
        (
            REPO_ROOT / protocol["predecessor_boundary"]["development_fixture_path"]
        ).read_text(encoding="utf-8")
    )
    semantic = json.loads(
        (
            REPO_ROOT / protocol["predecessor_boundary"]["semantic_fixture_path"]
        ).read_text(encoding="utf-8")
    )
    previous_ids = {
        str(case.get("id") or "")
        for source in (previous, semantic)
        for case in source.get("cases") or []
    }
    previous_queries = {
        " ".join(str(case.get("query") or "").casefold().split())
        for source in (previous, semantic)
        for case in source.get("cases") or []
    }
    if set(ids) & previous_ids:
        errors.append("authoring case ID overlaps previous evidence")
    for case in cases:
        question = str(case.get("question") or "").strip()
        scope = [str(value or "") for value in case.get("scope_doc_ids") or []]
        primary = str(case.get("primary_doc_id") or "")
        if not question or " ".join(question.casefold().split()) in previous_queries:
            errors.append(
                f"question is missing or reuses previous text: {case.get('id')}"
            )
        if (
            not primary
            or primary not in scope
            or not str(case.get("reading_intent") or "").strip()
        ):
            errors.append(
                f"question scope or reading intent is incomplete: {case.get('id')}"
            )
        is_cross = case.get("journey_type") == "cross_paper_evidence_comparison"
        if (is_cross and len(set(scope)) < 2) or (
            not is_cross and len(set(scope)) != 1
        ):
            errors.append(
                f"question scope does not match journey type: {case.get('id')}"
            )
    forbidden_fragments = (
        "draft",
        "rewrite",
        "polish",
        "manuscript",
        "投稿",
        "润色",
        "撰写",
    )
    if any(
        fragment in str(case.get("question") or "").casefold()
        for case in cases
        for fragment in forbidden_fragments
    ):
        errors.append("authoring manifest contains a writing-assistance request")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] authoring manifest cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_manifest(payload)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] 18 user-journey questions frozen before qrel chunk inspection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
