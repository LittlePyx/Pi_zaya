from __future__ import annotations

from types import SimpleNamespace

from kb.product_query_policy import search_hits_with_fallback
from kb.retriever import BM25Retriever


class ProductQuerySparseRetriever:
    """Expose the deterministic product-query BM25 path to evaluation channels."""

    def __init__(self, chunks: list[dict], *, whole_library: bool) -> None:
        self._retriever = BM25Retriever(chunks)
        self._whole_library = bool(whole_library)
        self._settings = SimpleNamespace(api_key=None, query_expansion_enabled=False)
        # Frozen qrels are judged through rank 20. Keeping this product-query
        # window stable prevents a hybrid caller's wider pool from changing the
        # protected sparse prefix used by the BM25 control.
        self._evaluation_depth = 20

    @property
    def is_empty(self) -> bool:
        return self._retriever.is_empty

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        hits, _scores, _used_query, _translated, _variants = (
            search_hits_with_fallback(
                query,
                self._retriever,
                self._evaluation_depth,
                self._settings,
                allow_translate=True,
                allow_expand=False,
                whole_library=self._whole_library,
            )
        )
        return list(hits[:top_k])
