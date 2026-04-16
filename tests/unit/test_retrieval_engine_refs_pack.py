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
        api_key="sk-test",
        base_url="https://example.invalid/v1",
        model="qwen3-vl-plus",
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
