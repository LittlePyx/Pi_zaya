from __future__ import annotations

from copy import deepcopy

import pytest

from kb.retrieval_benchmark import (
    evaluate_bm25_baseline,
    raw_bm25_search,
    validate_indexed_corpus,
)
from kb.retrieval_eval_fixture import (
    corpus_fingerprint,
    map_chunks_to_logical_docs,
    project_research_qa_qrels,
    validate_qrels_fixture,
)


def _source_payload() -> dict:
    return {
        "benchmark": {
            "id": "fixture-v1",
            "corpus": [
                {"id": "alpha", "sha256": "a" * 64},
                {"id": "beta", "sha256": "b" * 64},
            ],
        },
        "docs": [
            {"id": "alpha", "dir": "alpha", "file": "alpha"},
            {"id": "beta", "dir": "beta", "file": "beta"},
        ],
        "suites": {
            "acceptance": {
                "coverage": {"method_detail": ["alpha-method"], "en_locale": ["alpha-method"]}
            }
        },
        "cases": [
            {
                "id": "alpha-method",
                "question": "Which catalyst improves photon conversion?",
                "evaluationFocus": "method_detail",
                "queryScope": "current_paper",
                "docIds": ["alpha"],
                "expected": {
                    "claimEvidenceContracts": [
                        {
                            "id": "alpha-catalyst",
                            "docId": "alpha",
                            "sourcePage": 2,
                            "evidenceTerms": ["silver catalyst", "photon conversion"],
                        }
                    ]
                },
            }
        ],
    }


def _chunks(root: str = "C:/private") -> list[dict]:
    return [
        {
            "id": "doc-alpha:0",
            "text": "Background material without the reported mechanism.",
            "meta": {"source_path": f"{root}/alpha/alpha.en.md", "page_start": 1},
        },
        {
            "id": "doc-alpha:1",
            "text": "A silver catalyst improves photon conversion in the measured device.",
            "meta": {"source_path": f"{root}/alpha/alpha.en.md", "page_start": 2},
        },
        {
            "id": "doc-alpha:2",
            "text": "The appendix lists unrelated acquisition settings.",
            "meta": {"source_path": f"{root}/alpha/alpha.en.md", "page_start": 3},
        },
        {
            "id": "doc-beta:0",
            "text": "Photon conversion was evaluated without a silver catalyst.",
            "meta": {"source_path": f"{root}/beta/beta.en.md", "page_start": 4},
        },
    ]


def _projected() -> dict:
    return project_research_qa_qrels(
        _source_payload(),
        _chunks(),
        source_fixture_path="docs/source.json",
        source_fixture_sha256="c" * 64,
    )


def test_projection_freezes_page_bound_evidence_without_paths() -> None:
    payload = _projected()

    assert validate_qrels_fixture(payload) == []
    assert payload["corpus"]["document_count"] == 2
    assert payload["corpus"]["chunk_count"] == 4
    assert payload["cases"][0]["relevance"] == {"doc-alpha:1": 3}
    assert payload["cases"][0]["coverage_tags"] == ["en_locale", "method_detail"]
    assert payload["cases"][0]["contracts"][0]["relevant_chunk_ids"] == ["doc-alpha:1"]
    assert "C:/private" not in str(payload)


def test_corpus_fingerprint_is_path_independent_but_text_sensitive() -> None:
    chunks_a = _chunks("C:/one")
    chunks_b = _chunks("D:/another/root")
    map_a, _ = map_chunks_to_logical_docs(chunks_a, _source_payload()["docs"])
    map_b, _ = map_chunks_to_logical_docs(chunks_b, _source_payload()["docs"])

    assert corpus_fingerprint(chunks_a, map_a) == corpus_fingerprint(chunks_b, map_b)
    chunks_b[1]["text"] += " changed"
    assert corpus_fingerprint(chunks_a, map_a) != corpus_fingerprint(chunks_b, map_b)


def test_projection_rejects_contract_without_page_evidence() -> None:
    source = _source_payload()
    source["cases"][0]["expected"]["claimEvidenceContracts"][0]["sourcePage"] = 9

    with pytest.raises(ValueError, match="no evidence-term match"):
        project_research_qa_qrels(
            source,
            _chunks(),
            source_fixture_path="docs/source.json",
            source_fixture_sha256="c" * 64,
        )


def test_bm25_baseline_reports_scoped_and_full_library_metrics() -> None:
    payload = _projected()
    report = evaluate_bm25_baseline(payload, _chunks(), product_search=raw_bm25_search)

    assert set(report["channels"]) == {"bm25_raw", "bm25_product_queries"}
    scoped = report["channels"]["bm25_raw"]["modes"]["scoped"]
    full = report["channels"]["bm25_raw"]["modes"]["full_library"]
    assert scoped["metrics"]["recall@1"] == 1.0
    assert scoped["metrics"]["mrr"] == 1.0
    assert scoped["groups"]["coverage"]["method_detail"]["case_count"] == 1
    assert full["metrics"]["wrong_source_rate@20"] > 0.0


def test_index_validation_detects_labelled_text_drift() -> None:
    payload = _projected()
    changed = deepcopy(_chunks())
    changed[1]["text"] = "The registered evidence was removed."

    _chunk_map, errors = validate_indexed_corpus(payload, changed)

    assert any("corpus fingerprint" in error for error in errors)
    assert any("labelled chunk text changed" in error for error in errors)
