from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .retrieval_eval_fixture import (
    load_qrels_fixture,
    sha256_file,
    sha256_text,
    validate_qrels_fixture,
)
from .tokenize import tokenize


SEMANTIC_PARAPHRASE_SCHEMA_VERSION = 1
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _token_jaccard(left: object, right: object) -> float:
    left_tokens = set(tokenize(str(left or "")))
    right_tokens = set(tokenize(str(right or "")))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def build_semantic_paraphrase_qrels(
    registry: Mapping[str, Any],
    base_fixture: Mapping[str, Any],
    *,
    registry_path: str,
    registry_sha256: str,
    base_fixture_path: str,
    base_fixture_sha256: str,
) -> dict[str, Any]:
    """Project pre-registered semantic queries onto already-frozen qrels.

    Relevance labels, contract bindings, corpus scope, and chunk hashes are
    inherited byte-for-byte from the base fixture. Only the query surface and
    declared semantic-slice metadata can change.
    """
    if int(registry.get("schema_version") or 0) != SEMANTIC_PARAPHRASE_SCHEMA_VERSION:
        raise ValueError(
            f"semantic registry schema_version must be {SEMANTIC_PARAPHRASE_SCHEMA_VERSION}"
        )
    expected_base = registry.get("base_fixture")
    expected_base = expected_base if isinstance(expected_base, Mapping) else {}
    if str(expected_base.get("id") or "") != str(base_fixture.get("id") or ""):
        raise ValueError("semantic registry base fixture ID does not match")
    if str(expected_base.get("sha256") or "") != base_fixture_sha256:
        raise ValueError("semantic registry base fixture SHA-256 does not match")

    base_cases = {
        str(case.get("id") or ""): case
        for case in list(base_fixture.get("cases") or [])
        if isinstance(case, Mapping)
    }
    max_overlap = float(registry.get("max_source_query_token_jaccard") or 0.45)
    raw_cases = [
        case
        for case in list(registry.get("cases") or [])
        if isinstance(case, Mapping)
    ]
    expected_count = int(registry.get("expected_case_count") or 0)
    if expected_count <= 0 or len(raw_cases) != expected_count:
        raise ValueError("semantic registry case count does not match its frozen contract")

    projected_cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_base_ids: set[str] = set()
    locale_counts: Counter[str] = Counter()
    transformation_counts: Counter[str] = Counter()
    for raw_case in raw_cases:
        case_id = str(raw_case.get("id") or "").strip()
        base_case_id = str(raw_case.get("base_case_id") or "").strip()
        query = str(raw_case.get("query") or "").strip()
        locale = str(raw_case.get("locale") or "").strip()
        transformations = [
            str(value or "").strip()
            for value in list(raw_case.get("transformations") or [])
            if str(value or "").strip()
        ]
        withheld_terms = [
            str(value or "").strip()
            for value in list(raw_case.get("withheld_terms") or [])
            if str(value or "").strip()
        ]
        base_case = base_cases.get(base_case_id)
        if not case_id or case_id in seen_ids:
            raise ValueError("semantic case IDs must be non-empty and unique")
        if not base_case_id or base_case_id in seen_base_ids or base_case is None:
            raise ValueError(f"{case_id}: base case must exist and be used once")
        if locale not in {"en", "zh"} or bool(_CJK_RE.search(query)) != (locale == "zh"):
            raise ValueError(f"{case_id}: query locale does not match its declared locale")
        if "conceptual_paraphrase" not in transformations:
            raise ValueError(f"{case_id}: conceptual_paraphrase must be declared")
        if len(withheld_terms) < 2:
            raise ValueError(f"{case_id}: at least two evidence surface terms must be withheld")
        query_folded = query.casefold()
        leaked_terms = [term for term in withheld_terms if term.casefold() in query_folded]
        if leaked_terms:
            raise ValueError(f"{case_id}: query leaks withheld evidence terms: {leaked_terms}")
        source_query = str(base_case.get("query") or "")
        overlap = _token_jaccard(source_query, query)
        if not query or query == source_query or overlap > max_overlap:
            raise ValueError(
                f"{case_id}: source-query token Jaccard {overlap:.4f} exceeds {max_overlap:.4f}"
            )

        projected = deepcopy(dict(base_case))
        projected.update(
            {
                "id": case_id,
                "query": query,
                "locale": locale,
                "coverage_tags": sorted(
                    set(projected.get("coverage_tags") or [])
                    | {"semantic_paraphrase_v1", *transformations}
                ),
                "semantic_slice": {
                    "base_case_id": base_case_id,
                    "source_query_sha256": sha256_text(source_query),
                    "transformations": transformations,
                    "withheld_terms": withheld_terms,
                    "source_query_token_jaccard": round(overlap, 6),
                    "rationale": str(raw_case.get("rationale") or "").strip(),
                },
            }
        )
        projected_cases.append(projected)
        seen_ids.add(case_id)
        seen_base_ids.add(base_case_id)
        locale_counts[locale] += 1
        transformation_counts.update(transformations)

    expected_locales = {
        str(key): int(value)
        for key, value in dict(registry.get("expected_locale_counts") or {}).items()
    }
    if dict(locale_counts) != expected_locales:
        raise ValueError("semantic registry locale composition changed")
    minimum_cross_lingual = int(registry.get("minimum_cross_lingual_cases") or 0)
    if transformation_counts["cross_lingual"] < minimum_cross_lingual:
        raise ValueError("semantic registry no longer meets its cross-lingual minimum")

    label_policy = deepcopy(dict(base_fixture.get("label_policy") or {}))
    label_policy["query_curation"] = {
        "version": SEMANTIC_PARAPHRASE_SCHEMA_VERSION,
        "frozen_before_model_measurement": True,
        "maximum_source_query_token_jaccard": max_overlap,
        "withheld_surface_terms_required": 2,
        "relevance_labels_inherited_without_reselection": True,
    }
    payload = {
        "schema_version": int(base_fixture.get("schema_version") or 0),
        "id": str(registry.get("id") or ""),
        "curation_status": "frozen_before_semantic_model_measurement",
        "description": str(registry.get("description") or "").strip(),
        "source_fixture": {
            "path": registry_path.replace("\\", "/"),
            "sha256": registry_sha256,
            "sha256_newlines": "normalized_lf",
        },
        "source_qrels": {
            "path": base_fixture_path.replace("\\", "/"),
            "sha256": base_fixture_sha256,
            "id": str(base_fixture.get("id") or ""),
            "labels_inherited": True,
        },
        "label_policy": label_policy,
        "ks": deepcopy(list(base_fixture.get("ks") or [1, 3, 5, 10, 20])),
        "corpus": deepcopy(dict(base_fixture.get("corpus") or {})),
        "slice_contract": {
            "expected_case_count": expected_count,
            "locale_counts": dict(locale_counts),
            "transformation_counts": dict(transformation_counts),
            "acceptance_metric": "Recall@10",
            "minimum_dense_gain_over_product_bm25": 0.15,
            "secondary_safety_metrics": ["Recall@5", "wrong_source_rate@5"],
        },
        "cases": projected_cases,
    }
    errors = validate_qrels_fixture(payload)
    if errors:
        raise ValueError("invalid semantic qrels projection: " + "; ".join(errors))
    return payload


def load_semantic_paraphrase_qrels(
    registry_path: Path,
    base_fixture_path: Path,
) -> dict[str, Any]:
    import json

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    base_fixture = json.loads(base_fixture_path.read_text(encoding="utf-8"))
    return build_semantic_paraphrase_qrels(
        registry,
        base_fixture,
        registry_path=registry_path.as_posix(),
        registry_sha256=sha256_file(registry_path),
        base_fixture_path=base_fixture_path.as_posix(),
        base_fixture_sha256=sha256_file(base_fixture_path),
    )


def load_retrieval_benchmark_fixture(
    fixture_path: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Load either ordinary qrels or a pre-registered semantic projection."""
    import json

    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    base = raw.get("base_fixture") if isinstance(raw.get("base_fixture"), Mapping) else {}
    if int(raw.get("schema_version") or 0) != SEMANTIC_PARAPHRASE_SCHEMA_VERSION or not base:
        return load_qrels_fixture(fixture_path)
    raw_base_path = str(base.get("path") or "").strip()
    if not raw_base_path:
        raise ValueError("semantic registry base fixture path is required")
    base_path = (repo_root / raw_base_path).resolve()
    try:
        registry_label = fixture_path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        registry_label = fixture_path.name
    try:
        base_label = base_path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        base_label = base_path.name
    return build_semantic_paraphrase_qrels(
        raw,
        json.loads(base_path.read_text(encoding="utf-8")),
        registry_path=registry_label,
        registry_sha256=sha256_file(fixture_path),
        base_fixture_path=base_label,
        base_fixture_sha256=sha256_file(base_path),
    )
