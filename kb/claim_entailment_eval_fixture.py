from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from .retrieval_eval_fixture import sha256_text
from .tokenize import tokenize


CLAIM_ENTAILMENT_SCHEMA_VERSION = 1
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LABELS = {"entailment", "contradiction", "insufficient"}


def _claim_token_coverage(evidence: str, claim: str) -> float:
    evidence_tokens = set(tokenize(evidence))
    claim_tokens = set(tokenize(claim))
    return len(evidence_tokens & claim_tokens) / len(claim_tokens) if claim_tokens else 0.0


def validate_claim_entailment_fixture(
    payload: Mapping[str, Any],
    qrels: Mapping[str, Any],
    *,
    qrels_sha256: str,
) -> list[str]:
    """Validate a model-independent claim/evidence semantic gold set."""
    errors: list[str] = []
    if int(payload.get("schema_version") or 0) != CLAIM_ENTAILMENT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {CLAIM_ENTAILMENT_SCHEMA_VERSION}")
    source = payload.get("source_qrels") if isinstance(payload.get("source_qrels"), Mapping) else {}
    if str(source.get("id") or "") != str(qrels.get("id") or ""):
        errors.append("source qrels ID does not match")
    if str(source.get("sha256") or "") != qrels_sha256:
        errors.append("source qrels SHA-256 does not match")

    qrel_cases = {
        str(case.get("id") or ""): case
        for case in list(qrels.get("cases") or [])
        if isinstance(case, Mapping)
    }
    evidence_rows = [
        row for row in list(payload.get("evidence") or []) if isinstance(row, Mapping)
    ]
    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    evidence_text_by_id: dict[str, str] = {}
    for row in evidence_rows:
        evidence_id = str(row.get("id") or "").strip()
        source_case_id = str(row.get("source_case_id") or "").strip()
        if not evidence_id or evidence_id in evidence_by_id:
            errors.append("evidence IDs must be non-empty and unique")
            continue
        source_case = qrel_cases.get(source_case_id)
        if source_case is None:
            errors.append(f"{evidence_id}: source qrels case does not exist")
            continue
        registered_labels = {
            str(label.get("chunk_id") or ""): str(label.get("text_sha256") or "")
            for label in list(source_case.get("labels") or [])
            if isinstance(label, Mapping)
        }
        segments = [
            segment
            for segment in list(row.get("segments") or [])
            if isinstance(segment, Mapping)
        ]
        if not segments:
            errors.append(f"{evidence_id}: evidence segments are required")
            continue
        text_parts: list[str] = []
        for segment in segments:
            chunk_id = str(segment.get("chunk_id") or "").strip()
            chunk_sha256 = str(segment.get("chunk_text_sha256") or "").strip()
            quote = str(segment.get("quote") or "").strip()
            if registered_labels.get(chunk_id) != chunk_sha256:
                errors.append(f"{evidence_id}: segment is not bound to its frozen qrels label")
            if len(quote) < 40:
                errors.append(f"{evidence_id}: evidence quote is too short")
            text_parts.append(quote)
        evidence_by_id[evidence_id] = row
        evidence_text_by_id[evidence_id] = " ".join(text_parts)

    raw_cases = [
        row for row in list(payload.get("cases") or []) if isinstance(row, Mapping)
    ]
    expected_count = int(payload.get("expected_case_count") or 0)
    if expected_count <= 0 or len(raw_cases) != expected_count:
        errors.append("case count does not match the frozen contract")
    case_ids: set[str] = set()
    label_counts: Counter[str] = Counter()
    locale_counts: Counter[str] = Counter()
    cases_per_evidence: dict[str, set[str]] = {}
    lexical_adversaries = 0
    cross_lingual_cases = 0
    minimum_lexical_coverage = float(payload.get("minimum_lexical_adversary_claim_coverage") or 0.45)
    for row in raw_cases:
        case_id = str(row.get("id") or "").strip()
        evidence_id = str(row.get("evidence_id") or "").strip()
        claim = str(row.get("claim") or "").strip()
        label = str(row.get("label") or "").strip()
        locale = str(row.get("locale") or "").strip()
        adversary_type = str(row.get("adversary_type") or "").strip()
        if not case_id or case_id in case_ids:
            errors.append("case IDs must be non-empty and unique")
            continue
        case_ids.add(case_id)
        if evidence_id not in evidence_by_id:
            errors.append(f"{case_id}: evidence ID does not exist")
            continue
        if label not in _LABELS:
            errors.append(f"{case_id}: unsupported semantic label")
        if locale not in {"en", "zh"} or bool(_CJK_RE.search(claim)) != (locale == "zh"):
            errors.append(f"{case_id}: claim locale does not match")
        if len(claim) < 20 or not str(row.get("rationale") or "").strip():
            errors.append(f"{case_id}: claim and rationale are required")
        evidence_text = evidence_text_by_id.get(evidence_id, "")
        coverage = _claim_token_coverage(evidence_text, claim)
        declared_coverage = row.get("claim_token_coverage")
        if not isinstance(declared_coverage, (int, float)) or abs(float(declared_coverage) - coverage) > 0.000001:
            errors.append(f"{case_id}: claim token coverage is stale")
        if label == "contradiction" and adversary_type.startswith("lexical_"):
            lexical_adversaries += 1
            if coverage < minimum_lexical_coverage:
                errors.append(f"{case_id}: lexical adversary coverage is below the frozen floor")
        if locale == "zh":
            cross_lingual_cases += 1
        label_counts[label] += 1
        locale_counts[locale] += 1
        cases_per_evidence.setdefault(evidence_id, set()).add(label)

    expected_labels = {
        str(key): int(value)
        for key, value in dict(payload.get("expected_label_counts") or {}).items()
    }
    expected_locales = {
        str(key): int(value)
        for key, value in dict(payload.get("expected_locale_counts") or {}).items()
    }
    if dict(label_counts) != expected_labels:
        errors.append("semantic label composition changed")
    if dict(locale_counts) != expected_locales:
        errors.append("claim locale composition changed")
    if any(labels != _LABELS for labels in cases_per_evidence.values()):
        errors.append("each evidence item must have entailment, contradiction, and insufficient cases")
    if len(cases_per_evidence) != len(evidence_by_id):
        errors.append("every evidence item must be exercised")
    if lexical_adversaries < int(payload.get("minimum_lexical_adversaries") or 0):
        errors.append("lexical contradiction adversary minimum is not met")
    if cross_lingual_cases < int(payload.get("minimum_cross_lingual_cases") or 0):
        errors.append("cross-lingual semantic case minimum is not met")
    return errors


def claim_entailment_fixture_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    cases = [row for row in list(payload.get("cases") or []) if isinstance(row, Mapping)]
    return {
        "fixture_id": str(payload.get("id") or ""),
        "fixture_sha256": sha256_text(payload),
        "case_count": len(cases),
        "label_counts": dict(Counter(str(row.get("label") or "") for row in cases)),
        "locale_counts": dict(Counter(str(row.get("locale") or "") for row in cases)),
        "lexical_adversary_count": sum(
            1
            for row in cases
            if str(row.get("label") or "") == "contradiction"
            and str(row.get("adversary_type") or "").startswith("lexical_")
        ),
    }
