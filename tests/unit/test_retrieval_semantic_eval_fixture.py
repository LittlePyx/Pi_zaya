from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from kb.retrieval_eval_fixture import sha256_file, validate_qrels_fixture
from kb.retrieval_semantic_eval_fixture import (
    build_semantic_paraphrase_qrels,
    load_retrieval_benchmark_fixture,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_cases_v1.json"
BASE_PATH = REPO_ROOT / "docs" / "retrieval_eval_unseen_v2.json"


def _build(registry: dict | None = None) -> dict:
    source = registry or json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    base = json.loads(BASE_PATH.read_text(encoding="utf-8"))
    return build_semantic_paraphrase_qrels(
        source,
        base,
        registry_path="docs/retrieval_semantic_paraphrase_cases_v1.json",
        registry_sha256=sha256_file(REGISTRY_PATH),
        base_fixture_path="docs/retrieval_eval_unseen_v2.json",
        base_fixture_sha256=sha256_file(BASE_PATH),
    )


def test_semantic_fixture_inherits_qrels_and_freezes_composition() -> None:
    payload = _build()
    assert validate_qrels_fixture(payload, repo_root=REPO_ROOT) == []
    assert len(payload["cases"]) == 12
    assert payload["slice_contract"]["locale_counts"] == {"en": 6, "zh": 6}
    assert payload["slice_contract"]["transformation_counts"]["cross_lingual"] == 8
    assert payload["slice_contract"]["minimum_dense_gain_over_product_bm25"] == 0.15

    base = json.loads(BASE_PATH.read_text(encoding="utf-8"))
    base_cases = {case["id"]: case for case in base["cases"]}
    for case in payload["cases"]:
        source = base_cases[case["semantic_slice"]["base_case_id"]]
        assert case["labels"] == source["labels"]
        assert case["contracts"] == source["contracts"]
        assert case["relevance"] == source["relevance"]
        assert case["scope_doc_ids"] == source["scope_doc_ids"]
        assert case["query"] != source["query"]
        assert case["semantic_slice"]["source_query_token_jaccard"] <= 0.45


def test_semantic_fixture_rejects_withheld_term_leakage() -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    broken = deepcopy(registry)
    broken["cases"][0]["query"] += " selection mechanism"
    with pytest.raises(ValueError, match="leaks withheld evidence terms"):
        _build(broken)


def test_semantic_fixture_rejects_post_hoc_case_or_base_changes() -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    missing_case = deepcopy(registry)
    missing_case["cases"].pop()
    with pytest.raises(ValueError, match="case count"):
        _build(missing_case)

    wrong_base = deepcopy(registry)
    wrong_base["base_fixture"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="base fixture SHA-256"):
        _build(wrong_base)


def test_benchmark_loader_accepts_regular_and_semantic_fixtures() -> None:
    regular = load_retrieval_benchmark_fixture(BASE_PATH, repo_root=REPO_ROOT)
    semantic = load_retrieval_benchmark_fixture(REGISTRY_PATH, repo_root=REPO_ROOT)
    assert regular["id"] == "research_qa_unseen_corpus_v2_chunk_qrels"
    assert semantic["id"] == "research_qa_semantic_paraphrase_v1"
    assert len(semantic["cases"]) == 12
