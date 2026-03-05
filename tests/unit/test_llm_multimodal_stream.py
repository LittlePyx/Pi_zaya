from __future__ import annotations

from kb.llm import DeepSeekChat


class _FakeSettings:
    api_key = "test-key"
    base_url = "https://example.com/v1"
    model = "qwen3-vl-plus"
    timeout_s = 5.0
    max_retries = 0


def test_chat_stream_falls_back_to_chat_for_multimodal():
    ds = DeepSeekChat.__new__(DeepSeekChat)
    ds._settings = _FakeSettings()
    ds._client = None

    called = {"chat": 0}

    def _fake_chat(messages, temperature=0.2, max_tokens=1200):
      called["chat"] += 1
      return "ok"

    ds.chat = _fake_chat  # type: ignore[method-assign]

    out = list(ds.chat_stream(messages=[{
      "role": "user",
      "content": [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
      ],
    }]))

    assert out == ["ok"]
    assert called["chat"] == 1
