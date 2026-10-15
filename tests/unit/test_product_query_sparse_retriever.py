from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import tools.retrieval_eval.product_query_sparse_retriever as adapter
from tools.retrieval_eval.product_query_sparse_retriever import (
    ProductQuerySparseRetriever,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_product_query_adapter_pins_product_baseline_depth(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_product_search(
        query,
        _retriever,
        top_k,
        _settings,
        *,
        allow_translate,
        allow_expand,
        whole_library,
    ):
        observed.update(
            query=query,
            top_k=top_k,
            allow_translate=allow_translate,
            allow_expand=allow_expand,
            whole_library=whole_library,
        )
        hits = [{"id": f"doc:{index}", "text": "evidence"} for index in range(top_k)]
        return hits, [1.0] * top_k, query, False, [query]

    monkeypatch.setattr(adapter, "search_hits_with_fallback", fake_product_search)
    retriever = ProductQuerySparseRetriever(
        [{"id": "doc:0", "text": "evidence", "meta": {}}],
        whole_library=True,
    )

    assert len(retriever.search("query", top_k=80)) == 20
    assert observed == {
        "query": "query",
        "top_k": 20,
        "allow_translate": True,
        "allow_expand": False,
        "whole_library": True,
    }


def test_product_query_adapter_preserves_empty_corpus_behavior() -> None:
    retriever = ProductQuerySparseRetriever([], whole_library=False)
    assert retriever.is_empty is True
    assert retriever.search("evidence", top_k=10) == []


def test_product_query_adapter_does_not_import_dense_benchmark() -> None:
    script = (
        "import sys; "
        "import tools.retrieval_eval.product_query_sparse_retriever; "
        "raise SystemExit('tools.retrieval_eval.run_dense_research_qa_benchmark' "
        "in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
