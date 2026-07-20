from __future__ import annotations

import time

from kb.llm import DeepSeekChat


class _FakeSettings:
    text_api_key = "test-key"
    text_base_url = "https://example.com/v1"
    text_model = "deepseek-chat"
    vision_api_key = "test-key"
    vision_base_url = "https://example.com/v1"
    vision_model = "qwen3-vl-plus"
    timeout_s = 5.0
    max_retries = 0
    auto_route = False


class _FakeDelta:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeEvent:
    def __init__(self, content: str) -> None:
        self.choices = [type("Choice", (), {"delta": _FakeDelta(content)})()]


class _FakeCompletions:
    def __init__(self, *, error: Exception | None = None, pieces: list[str] | None = None) -> None:
        self.error = error
        self.pieces = list(pieces or [])
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return [_FakeEvent(piece) for piece in self.pieces]


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = type("Chat", (), {"completions": completions})()


def test_chat_stream_falls_back_to_chat_for_multimodal():
    ds = DeepSeekChat.__new__(DeepSeekChat)
    ds._settings = _FakeSettings()
    ds._text_client = None
    ds._vision_client = None

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


def test_chat_stream_uses_secondary_provider_when_primary_fails_before_output(monkeypatch):
    settings = _FakeSettings()
    settings.auto_route = True
    settings.timeout_s = 60.0
    primary_completions = _FakeCompletions(error=ConnectionError("primary unavailable"))
    secondary_completions = _FakeCompletions(pieces=["fallback", " answer"])
    ds = DeepSeekChat.__new__(DeepSeekChat)
    ds._settings = settings
    ds._text_client = _FakeClient(primary_completions)
    ds._vision_client = _FakeClient(secondary_completions)
    monkeypatch.setenv("KB_LLM_PRIMARY_STREAM_TIMEOUT_S", "18")

    out = list(ds.chat_stream(messages=[{"role": "user", "content": "hello"}]))

    assert out == ["fallback", " answer"]
    assert primary_completions.calls[0]["model"] == "deepseek-chat"
    assert primary_completions.calls[0]["timeout"] == 18.0
    assert secondary_completions.calls[0]["model"] == "qwen3-vl-plus"
    assert secondary_completions.calls[0]["timeout"] == 60.0


def test_chat_stream_does_not_switch_provider_after_partial_output():
    class _PartialFailureCompletions(_FakeCompletions):
        def create(self, **kwargs):
            self.calls.append(dict(kwargs))

            def events():
                yield _FakeEvent("partial")
                raise ConnectionError("stream interrupted")

            return events()

    settings = _FakeSettings()
    settings.auto_route = True
    primary_completions = _PartialFailureCompletions()
    secondary_completions = _FakeCompletions(pieces=["duplicate"])
    ds = DeepSeekChat.__new__(DeepSeekChat)
    ds._settings = settings
    ds._text_client = _FakeClient(primary_completions)
    ds._vision_client = _FakeClient(secondary_completions)

    iterator = ds.chat_stream(messages=[{"role": "user", "content": "hello"}])
    assert next(iterator) == "partial"
    try:
        next(iterator)
    except ConnectionError:
        pass
    else:
        raise AssertionError("partial stream failure should propagate")
    assert secondary_completions.calls == []


def test_chat_stream_falls_back_when_primary_has_no_visible_token(monkeypatch):
    class _SlowCompletions(_FakeCompletions):
        def create(self, **kwargs):
            self.calls.append(dict(kwargs))
            time.sleep(0.12)
            return [_FakeEvent("too late")]

    settings = _FakeSettings()
    settings.auto_route = True
    primary_completions = _SlowCompletions()
    secondary_completions = _FakeCompletions(pieces=["timely", " fallback"])
    ds = DeepSeekChat.__new__(DeepSeekChat)
    ds._settings = settings
    ds._text_client = _FakeClient(primary_completions)
    ds._vision_client = _FakeClient(secondary_completions)
    monkeypatch.setattr(ds, "_first_visible_token_timeout_s", lambda: 0.03)

    out = list(ds.chat_stream(messages=[{"role": "user", "content": "hello"}]))

    assert out == ["timely", " fallback"]
    assert len(primary_completions.calls) == 1
    assert len(secondary_completions.calls) == 1


def test_chat_supports_bounded_single_attempt_retry() -> None:
    class _ResponseCompletions:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def create(self, **kwargs):
            self.calls.append(dict(kwargs))
            message = type("Message", (), {"content": "bounded answer"})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    completions = _ResponseCompletions()
    ds = DeepSeekChat.__new__(DeepSeekChat)
    ds._settings = _FakeSettings()
    ds._text_client = _FakeClient(completions)  # type: ignore[arg-type]
    ds._vision_client = ds._text_client

    answer = ds.chat(
        messages=[{"role": "user", "content": "hello"}],
        timeout_s=18.0,
        max_retries=0,
    )

    assert answer == "bounded answer"
    assert len(completions.calls) == 1
    assert completions.calls[0]["timeout"] == 18.0
