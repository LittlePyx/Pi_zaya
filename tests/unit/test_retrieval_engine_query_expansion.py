from __future__ import annotations

import json
import pytest

from kb.retrieval_engine import (
    _expand_query_via_llm,
    _merge_expanded_results,
    _search_hits_with_fallback,
)


# ---------------------------------------------------------------------------
# _expand_query_via_llm
# ---------------------------------------------------------------------------

def test_expand_query_returns_original_on_empty(monkeypatch):
    """Empty/very short input returns just the original query (or empty)."""
    assert _expand_query_via_llm(None, "") == []
    assert _expand_query_via_llm(None, "ab") == ["ab"]


def test_expand_query_uses_cache(monkeypatch):
    """Cache hit returns cached variants without calling LLM."""
    call_count = 0

    def fake_cache_get(bucket, key):
        nonlocal call_count
        if bucket == "query_expand":
            return ["cached_variant"]
        return None

    def fake_cache_set(bucket, key, val, **kw):
        nonlocal call_count
        call_count += 1

    import kb.retrieval_engine as re
    monkeypatch.setattr(re, "_cache_get", fake_cache_get)
    monkeypatch.setattr(re, "_cache_set", fake_cache_set)

    result = _expand_query_via_llm(None, "test query")
    assert "test query" in result
    assert "cached_variant" in result


def test_expand_query_llm_failure_returns_original(monkeypatch):
    """LLM failure/timeout returns just [original]."""
    call_count = 0

    def fake_cache_get(bucket, key):
        return None  # No cache hit

    def fake_cache_set(bucket, key, val, **kw):
        nonlocal call_count
        call_count += 1

    class FakeSettings:
        api_key = "test-key"
        timeout_s = 60.0
        max_retries = 0

    class FakeDeepSeek:
        def chat(self, **kw):
            raise RuntimeError("LLM timeout")

    import kb.retrieval_engine as re
    monkeypatch.setattr(re, "_cache_get", fake_cache_get)
    monkeypatch.setattr(re, "_cache_set", fake_cache_set)
    monkeypatch.setattr(re, "DeepSeekChat", lambda s: FakeDeepSeek())

    result = _expand_query_via_llm(FakeSettings(), "test query")
    assert result == ["test query"]


def test_expand_query_parses_llm_output(monkeypatch):
    """LLM returns well-formed variants."""
    def fake_cache_get(bucket, key):
        return None

    called_with = {}

    def fake_cache_set(bucket, key, val, **kw):
        called_with["val"] = val

    class FakeSettings:
        api_key = "test-key"
        timeout_s = 60.0
        max_retries = 0

    class FakeDeepSeek:
        def chat(self, messages, temperature, max_tokens):
            return "query variant one\nquery variant two"

    import kb.retrieval_engine as re
    monkeypatch.setattr(re, "_cache_get", fake_cache_get)
    monkeypatch.setattr(re, "_cache_set", fake_cache_set)
    monkeypatch.setattr(re, "DeepSeekChat", lambda s: FakeDeepSeek())

    result = _expand_query_via_llm(FakeSettings(), "test query")
    # Original query always first
    assert result[0] == "test query"
    assert "query variant one" in result
    assert "query variant two" in result


def test_expand_query_none_output(monkeypatch):
    """LLM returns 'NONE' — no variants generated."""
    def fake_cache_get(bucket, key):
        return None

    class FakeSettings:
        api_key = "test-key"
        timeout_s = 60.0
        max_retries = 0

    class FakeDeepSeek:
        def chat(self, messages, temperature, max_tokens):
            return "NONE"

    import kb.retrieval_engine as re
    monkeypatch.setattr(re, "_cache_get", fake_cache_get)
    monkeypatch.setattr(re, "_cache_set", lambda *a, **kw: None)
    monkeypatch.setattr(re, "DeepSeekChat", lambda s: FakeDeepSeek())

    result = _expand_query_via_llm(FakeSettings(), "test query")
    assert result == ["test query"]


# ---------------------------------------------------------------------------
# _merge_expanded_results  (RRF)
# ---------------------------------------------------------------------------

def _make_meta_hit(chunk_id: str, text: str, score: float = 1.0) -> dict:
    return {"chunk_id": chunk_id, "text": text, "score": score, "meta": {"chunk_id": chunk_id}}


def test_merge_expanded_results_disjoint():
    """Two completely disjoint result sets are merged."""
    r1 = ([_make_meta_hit("a", "hit a")], [1.0], "q1")
    r2 = ([_make_meta_hit("b", "hit b")], [1.0], "q2")
    merged_hits, merged_scores = _merge_expanded_results([r1, r2], top_k=10)
    assert len(merged_hits) == 2
    assert {h["chunk_id"] for h in merged_hits} == {"a", "b"}


def test_merge_expanded_results_overlap():
    """Overlapping hits get higher RRF scores."""
    r1 = ([_make_meta_hit("a", "hit a"), _make_meta_hit("b", "hit b")], [1.0, 0.5], "q1")
    r2 = ([_make_meta_hit("a", "hit a")], [1.0], "q2")
    merged_hits, merged_scores = _merge_expanded_results([r1, r2], top_k=10)
    ids = [h["chunk_id"] for h in merged_hits]
    # "a" is in both sets so it should rank first
    assert ids[0] == "a"
    assert "b" in ids


def test_merge_expanded_results_empty():
    """Empty inputs produce empty output."""
    merged_hits, merged_scores = _merge_expanded_results([], top_k=10)
    assert merged_hits == []
    assert merged_scores == []


def test_merge_expanded_results_top_k():
    """top_k limits output."""
    r1 = ([_make_meta_hit(f"chunk_{i}", f"hit {i}") for i in range(10)], [1.0] * 10, "q1")
    merged_hits, merged_scores = _merge_expanded_results([r1], top_k=3)
    assert len(merged_hits) == 3


# ---------------------------------------------------------------------------
# _search_hits_with_fallback — expansion integration
# ---------------------------------------------------------------------------

class _FakeRetriever:
    """Simple BM25-like retriever mock."""
    def __init__(self, results: dict[str, list[dict]]):
        self._results = results

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        return self._results.get(query, [])[:top_k]


class _FakeSettings:
    api_key = None  # No API key = no translation/expansion
    timeout_s = 60.0
    max_retries = 0
    query_expansion_enabled = False


def test_search_hits_fallback_basic():
    """Basic retrieval without translation/expansion returns original query results."""
    retriever = _FakeRetriever({
        "test query": [{"text": "result 1", "score": 5.0, "meta": {"source_path": "doc.md"}}],
    })
    hits, scores, used_query, used_trans, variants = _search_hits_with_fallback(
        "test query", retriever, top_k=10, settings=_FakeSettings(),
    )
    assert len(hits) == 1
    assert used_query == "test query"
    assert used_trans is False
    assert "test query" in variants


def test_search_hits_fallback_with_expansion(monkeypatch):
    """Expansion path is triggered when allow_expand=True and settings enable it."""

    class FakeSettings:
        api_key = "test-key"
        timeout_s = 60.0
        max_retries = 0
        query_expansion_enabled = True

    retriever = _FakeRetriever({
        "test query": [{"text": "result 1", "score": 5.0, "meta": {"source_path": "doc.md", "chunk_id": "c1"}}],
        "expanded variant": [{"text": "result 2", "score": 5.0, "meta": {"source_path": "doc.md", "chunk_id": "c2"}}],
    })

    def fake_expand(settings, prompt):
        return ["test query", "expanded variant"]

    def fake_translate(settings, prompt):
        return None  # No translation

    import kb.retrieval_engine as re
    monkeypatch.setattr(re, "_expand_query_via_llm", fake_expand)
    monkeypatch.setattr(re, "_translate_query_for_search", fake_translate)

    hits, scores, used_query, used_trans, variants = _search_hits_with_fallback(
        "test query", retriever, top_k=10, settings=FakeSettings(),
        allow_expand=True,
    )
    # Should have merged results from both queries
    assert len(hits) >= 1
    assert "expanded variant" in variants
