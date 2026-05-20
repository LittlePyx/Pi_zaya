from __future__ import annotations

from pathlib import Path

from kb.config import Settings
import kb.retrieval_engine as retrieval_engine


def test_llm_refs_pack_docwise_items_respects_settings_timeout_on_retry(monkeypatch):
    observed_timeouts: list[float] = []
    call_count = {"n": 0}

    class FakeDeepSeekChat:
        def __init__(self, settings):
            observed_timeouts.append(float(getattr(settings, "timeout_s", 0.0) or 0.0))

        def chat(self, *, messages, temperature=0.0, max_tokens=0):
            del messages, temperature, max_tokens
            call_count["n"] += 1
            if call_count["n"] % 2 == 1:
                return "{}"
            return '{"score": 82, "what": "Paper summary.", "why": "Directly relevant.", "section": "Method"}'

    monkeypatch.setattr(retrieval_engine, "DeepSeekChat", FakeDeepSeekChat)

    settings = Settings(
        text_api_key="sk-test",
        text_base_url="https://example.invalid/v1",
        text_model="deepseek-chat",
        vision_api_key="sk-test",
        vision_base_url="https://example.invalid/v1",
        vision_model="qwen3-vl-plus",
        db_dir=Path("."),
        chat_db_path=Path("chat.sqlite3"),
        library_db_path=Path("library.sqlite3"),
        timeout_s=12.0,
        max_retries=0,
    )
    items = [
        {
            "i": 1,
            "headings": ["Method"],
            "locs": [{"heading_path": "Method", "snippet": "Direct evidence snippet."}],
            "overview_snippets": ["This paper proposes a method."],
            "snippets": ["Direct evidence snippet."],
            "anchors": ["dynamic supersampling"],
        }
    ]

    arr = retrieval_engine._llm_refs_pack_docwise_items(
        settings,
        question="Which paper most directly discusses dynamic supersampling?",
        items=items,
    )

    assert len(arr) == 1
    assert observed_timeouts == [12.0, 16.0]


def test_llm_refs_pack_skips_batch_for_multi_paper_list_prompt(monkeypatch):
    monkeypatch.setattr(retrieval_engine, "_cache_get", lambda *args, **kwargs: None)
    monkeypatch.setattr(retrieval_engine, "_cache_set", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        retrieval_engine,
        "_llm_refs_pack_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not call batch path")),
    )
    observed: dict[str, object] = {}

    def fake_docwise(settings, *, question, items, on_item=None):
        del on_item
        observed["question"] = question
        observed["timeout_s"] = float(getattr(settings, "timeout_s", 0.0) or 0.0)
        observed["item_count"] = len(items or [])
        return [
            {"i": 1, "score": 84, "what": "Paper A summary.", "why": "Mentions SCI directly.", "section": "Abstract"},
            {"i": 2, "score": 73, "what": "Paper B summary.", "why": "Discusses SCI-related imaging.", "section": "Introduction"},
        ]

    monkeypatch.setattr(retrieval_engine, "_llm_refs_pack_docwise_items", fake_docwise)

    settings = Settings(
        text_api_key="sk-test",
        text_base_url="https://example.invalid/v1",
        text_model="deepseek-chat",
        vision_api_key="sk-test",
        vision_base_url="https://example.invalid/v1",
        vision_model="qwen3-vl-plus",
        db_dir=Path("."),
        chat_db_path=Path("chat.sqlite3"),
        library_db_path=Path("library.sqlite3"),
        timeout_s=12.0,
        max_retries=0,
    )
    docs = [
        {
            "meta": {
                "source_path": r"db\paper-a\paper-a.en.md",
                "ref_headings": ["Abstract"],
                "ref_show_snippets": ["Snapshot Compressive Imaging (SCI) is introduced in the abstract."],
                "ref_overview_snippets": ["This paper studies SCI reconstruction."],
            }
        },
        {
            "meta": {
                "source_path": r"db\paper-b\paper-b.en.md",
                "ref_headings": ["1. Introduction"],
                "ref_show_snippets": ["The introduction discusses SCI and dynamic scene recovery."],
                "ref_overview_snippets": ["This paper discusses SCI-related imaging."],
            }
        },
    ]

    out = retrieval_engine._llm_refs_pack(
        settings,
        question="Which papers in my library mention Snapshot Compressive Imaging (SCI)?",
        docs=docs,
    )

    assert observed["item_count"] == 2
    assert observed["timeout_s"] == 10.0
    assert "Snapshot Compressive Imaging" in str(observed["question"] or "")
    assert sorted(out.keys()) == [1, 2]
