from __future__ import annotations

import json
from pathlib import Path

import pytest

from kb.retrieval_engine import (
    _expand_query_via_llm,
    _group_hits_by_doc_for_refs,
    _merge_expanded_results,
    _reading_roadmap_source_role_bonus,
    _search_hits_with_fallback,
    _translate_query_for_search,
)


def test_reading_roadmap_source_role_bonus_prefers_reviews_and_comparisons():
    prompt = "我刚开始看单像素成像，想建立主线，应该先读哪几篇？"

    assert _reading_roadmap_source_role_bonus(
        prompt,
        "NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md",
    ) > _reading_roadmap_source_role_bonus(
        prompt,
        "Robust application method.en.md",
    )
    assert _reading_roadmap_source_role_bonus(
        prompt,
        "Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md",
    ) > 0
    assert _reading_roadmap_source_role_bonus(
        "请解释这个公式",
        "Principles and prospects for single-pixel imaging.en.md",
    ) == 0


def test_reading_roadmap_grouping_keeps_complementary_foundation_and_comparison_docs(
    tmp_path,
    monkeypatch,
):
    import kb.retrieval_engine as retrieval_engine

    monkeypatch.setattr(retrieval_engine, "_is_temp_source_path", lambda _path: False)
    sources = [
        tmp_path / "NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md",
        tmp_path / "LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.en.md",
        tmp_path / "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md",
        tmp_path / "Application of single-pixel imaging for one narrow task.en.md",
    ]
    for source in sources:
        source.write_text(
            "# Introduction\n\nGrounded single-pixel imaging evidence.\n",
            encoding="utf-8",
        )

    hits = [
        {
            "score": score,
            "text": "Grounded single-pixel imaging evidence.",
            "meta": {"source_path": str(source), "heading_path": "Introduction"},
        }
        for source, score in zip(sources, [1.0, 1.1, 1.2, 12.0])
    ]

    docs = _group_hits_by_doc_for_refs(
        hits,
        "我刚开始看深度学习单像素成像，想建立主线，应该先读哪几篇？",
        3,
        deep_read=False,
        llm_rerank=False,
    )

    names = [
        Path(str((doc.get("meta") or {}).get("source_path") or "")).name
        for doc in docs
    ]
    assert names == [sources[1].name, sources[0].name, sources[2].name]


from kb.retrieval_heuristics import _doc_term_bonus, _query_term_profile


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


def test_merge_expanded_results_uses_hit_id_before_text_fallback():
    shared_prefix = "same text prefix " * 12
    r1 = (
        [
            {"id": "hit-a", "text": f"{shared_prefix}a", "score": 1.0, "meta": {}},
            {"id": "hit-b", "text": f"{shared_prefix}b", "score": 0.9, "meta": {}},
        ],
        [1.0, 0.9],
        "q1",
    )

    merged_hits, _merged_scores = _merge_expanded_results([r1], top_k=10)

    assert [hit["id"] for hit in merged_hits] == ["hit-a", "hit-b"]


def test_merge_expanded_results_counts_identity_once_per_result_set():
    duplicate = {"id": "hit-a", "text": "same hit", "score": 1.0, "meta": {}}
    r1 = ([duplicate, dict(duplicate)], [1.0, 0.9], "q1")

    merged_hits, merged_scores = _merge_expanded_results([r1], top_k=10)

    assert [hit["id"] for hit in merged_hits] == ["hit-a"]
    assert merged_scores == pytest.approx([1.0 / 20.0])


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


def test_translate_query_uses_mixed_metric_anchors_without_llm(monkeypatch):
    class Settings:
        api_key = "configured"

    import kb.retrieval_engine as retrieval_engine

    monkeypatch.setattr(
        retrieval_engine,
        "DeepSeekChat",
        lambda _settings: (_ for _ in ()).throw(AssertionError("translation LLM should not run")),
    )

    translated = _translate_query_for_search(
        Settings(),
        "ECCV-2022 Simple Baselines 论文的 SIDD 基准测试里，PSNR 最高的模型是谁？如果并列请全部列出。",
    )

    assert translated is not None
    assert "SIDD" in translated
    assert "PSNR" in translated
    assert "benchmark" in translated
    assert "highest" in translated
    assert "model" in translated
    assert "tie" in translated


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


def test_search_hits_fallback_does_not_index_internal_query_scope_instructions():
    retriever = _FakeRetriever({
        "highest SIDD PSNR model": [
            {"text": "Table 6 comparison", "score": 8.0, "meta": {"source_path": "paper.md"}},
        ],
    })
    prompt = (
        "highest SIDD PSNR model\n\n"
        "QUERY SCOPE: Full library.\n"
        "- Search and synthesize across the whole indexed literature library.\n"
        "- When multiple papers are relevant, organize the answer by paper."
    )

    hits, _scores, used_query, _used_trans, variants = _search_hits_with_fallback(
        prompt,
        retriever,
        top_k=10,
        settings=_FakeSettings(),
    )

    assert hits[0]["text"] == "Table 6 comparison"
    assert used_query == "highest SIDD PSNR model"
    assert variants == ["highest SIDD PSNR model"]


def test_search_hits_fallback_keeps_translated_hits_when_original_has_weak_match(monkeypatch):
    query = "中文伪命中"
    translated = "english translated query"
    retriever = _FakeRetriever(
        {
            query: [
                {
                    "id": "weak-cn",
                    "text": "weak lexical match",
                    "score": 0.2,
                    "meta": {"source_path": "weak.md"},
                }
            ],
            translated: [
                {
                    "id": "relevant-en",
                    "text": "relevant English evidence",
                    "score": 8.0,
                    "meta": {"source_path": "relevant.md"},
                }
            ],
        }
    )

    import kb.retrieval_engine as re

    monkeypatch.setattr(re, "_translate_query_for_search", lambda _settings, _prompt: translated)
    monkeypatch.setattr(re, "_deterministic_query_variants", lambda _prompt: [])

    hits, scores, _used_query, used_trans, variants = _search_hits_with_fallback(
        query,
        retriever,
        top_k=10,
        settings=_FakeSettings(),
    )

    assert {hit["id"] for hit in hits} == {"weak-cn", "relevant-en"}
    assert hits[0]["id"] == "relevant-en"
    assert max(scores) < 0.2  # RRF scale; it must not be compared with BM25.
    assert used_trans is True
    assert variants == [query, translated]


def test_search_hits_fallback_uses_wider_candidate_window_for_whole_library(monkeypatch):
    requested_limits: list[int] = []

    class RecordingRetriever(_FakeRetriever):
        def search(self, query: str, top_k: int = 10) -> list[dict]:
            requested_limits.append(top_k)
            return super().search(query, top_k=top_k)

    retriever = RecordingRetriever(
        {
            "library question": [
                {"id": f"hit-{idx}", "text": f"result {idx}", "score": 10.0 - idx, "meta": {"source_path": f"doc-{idx}.md"}}
                for idx in range(100)
            ]
        }
    )
    import kb.retrieval_engine as re

    monkeypatch.setattr(re, "_translate_query_for_search", lambda _settings, _prompt: None)
    monkeypatch.setattr(re, "_deterministic_query_variants", lambda _prompt: [])

    hits, _scores, _used_query, _used_trans, _variants = _search_hits_with_fallback(
        "library question",
        retriever,
        top_k=6,
        settings=_FakeSettings(),
        whole_library=True,
    )

    assert requested_limits == [96]
    assert len(hits) == 96


def test_deep_learning_topic_qualifier_penalizes_generic_single_pixel_paper():
    profile = _query_term_profile(
        "深度学习给单像素成像带来哪些优势？",
        "deep learning single-pixel imaging advantages",
    )

    relevant = _doc_term_bonus(
        profile,
        "Deep learning for real-time single-pixel video.md",
        ["A neural network reconstructs single-pixel measurements in real time."],
    )
    generic = _doc_term_bonus(
        profile,
        "Hadamard single-pixel imaging versus Fourier single-pixel imaging.md",
        ["We compare two conventional single-pixel sampling bases."],
    )
    unrelated = _doc_term_bonus(
        profile,
        "Deep learning for classical literature.md",
        ["A neural network classifies historical Japanese characters."],
    )

    assert relevant > generic
    assert relevant - generic >= 4.0
    assert relevant > unrelated
    assert relevant - unrelated >= 4.0


def test_doc_grouping_uses_topic_qualifier_before_bounded_candidate_cutoff(tmp_path, monkeypatch):
    import kb.retrieval_engine as re

    monkeypatch.setattr(re, "_is_temp_source_path", lambda _source_path: False)
    generic_sources = []
    hits = []
    for idx in range(12):
        source = tmp_path / f"generic-{idx}-single-pixel-imaging.md"
        source.write_text("# Generic SPI\n\nConventional single-pixel sampling basis.", encoding="utf-8")
        generic_sources.append(source)
        hits.append(
            {
                "id": f"generic-{idx}",
                "text": "Conventional single-pixel sampling basis.",
                "score": 20.0 - idx,
                "meta": {"source_path": str(source), "heading_path": "Introduction"},
            }
        )
    relevant_source = tmp_path / "deep-learning-single-pixel-imaging.md"
    relevant_source.write_text(
        "# Deep learning SPI\n\nA neural network reconstructs single-pixel measurements in real time.",
        encoding="utf-8",
    )
    hits.append(
        {
            "id": "relevant",
            "text": "A neural network reconstructs single-pixel measurements in real time.",
            "score": 8.5,
            "meta": {"source_path": str(relevant_source), "heading_path": "Introduction"},
        }
    )

    docs = _group_hits_by_doc_for_refs(
        hits,
        prompt_text="深度学习单像素成像有哪些优势？",
        top_k_docs=1,
        deep_query="deep learning single-pixel imaging advantages",
    )

    assert docs
    assert (docs[0].get("meta") or {}).get("source_path") == str(relevant_source)


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
