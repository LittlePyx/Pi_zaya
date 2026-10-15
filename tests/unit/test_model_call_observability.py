from __future__ import annotations

import json
import time

from kb.llm import DeepSeekChat
from kb.model_call_observability import (
    extract_provider_token_usage,
    record_model_call_observation,
    stable_model_error_category,
)


class _Settings:
    text_api_key = "test-key"
    text_base_url = "https://private-provider.invalid/v1"
    text_model = "private-model-name"
    vision_api_key = "test-key"
    vision_base_url = "https://private-vision.invalid/v1"
    vision_model = "private-vision-name"
    timeout_s = 5.0
    max_retries = 0
    auto_route = False


class _Completions:
    def __init__(self, response) -> None:
        self.response = response

    def create(self, **_kwargs):
        return self.response


class _Client:
    def __init__(self, response) -> None:
        self.chat = type("Chat", (), {"completions": _Completions(response)})()


def _response(*, usage=None):
    message = type("Message", (), {"content": "private response text"})()
    choice = type("Choice", (), {"message": message})()
    return type("Response", (), {"choices": [choice], "usage": usage})()


def test_extract_provider_token_usage_accepts_known_exact_aliases_only() -> None:
    usage = type(
        "Usage",
        (),
        {"prompt_tokens": 17, "completion_tokens": 5, "total_tokens": 22},
    )()
    assert extract_provider_token_usage(type("Response", (), {"usage": usage})()) == {
        "input_token_count": 17,
        "output_token_count": 5,
        "total_token_count": 22,
    }
    assert extract_provider_token_usage({"usage": {"prompt_tokens": -1}}) == {}
    assert extract_provider_token_usage({"usage": {"unrelated": 99}}) == {}


def test_record_model_call_marks_missing_usage_without_estimating(monkeypatch, tmp_path) -> None:
    path = tmp_path / "model.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(path))

    assert record_model_call_observation(
        route_kind="text",
        started_at=time.monotonic(),
        outcome="completed",
        attempt_count=1,
        streaming=False,
        usage_source={},
    )

    attributes = json.loads(path.read_text(encoding="utf-8"))["attributes"]
    assert attributes["usage_reported"] is False
    assert "input_token_count" not in attributes
    assert "output_token_count" not in attributes
    assert "total_token_count" not in attributes


def test_chat_records_provider_usage_without_prompt_response_or_model(monkeypatch, tmp_path) -> None:
    path = tmp_path / "chat.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(path))
    usage = type(
        "Usage",
        (),
        {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    )()
    ds = DeepSeekChat.__new__(DeepSeekChat)
    ds._settings = _Settings()
    ds._text_client = _Client(_response(usage=usage))
    ds._vision_client = ds._text_client

    assert ds.chat([{"role": "user", "content": "private prompt"}]) == "private response text"

    stored_text = path.read_text(encoding="utf-8")
    attributes = json.loads(stored_text)["attributes"]
    assert attributes["model_route_id"] == "text_generation"
    assert attributes["usage_reported"] is True
    assert attributes["input_token_count"] == 11
    assert attributes["output_token_count"] == 7
    assert "private prompt" not in stored_text
    assert "private response text" not in stored_text
    assert "private-model-name" not in stored_text
    assert "private-provider" not in stored_text


def test_chat_stream_records_usage_only_when_provider_includes_it(monkeypatch, tmp_path) -> None:
    path = tmp_path / "stream.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(path))
    delta = type("Delta", (), {"content": "visible answer"})()
    choice = type("Choice", (), {"delta": delta})()
    event = type(
        "Event",
        (),
        {
            "choices": [choice],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
        },
    )()
    ds = DeepSeekChat.__new__(DeepSeekChat)
    ds._settings = _Settings()
    ds._text_client = _Client([event])
    ds._vision_client = ds._text_client

    assert list(ds.chat_stream([{"role": "user", "content": "private prompt"}])) == [
        "visible answer"
    ]

    stored_text = path.read_text(encoding="utf-8")
    attributes = json.loads(stored_text)["attributes"]
    assert attributes["streaming"] is True
    assert attributes["usage_reported"] is True
    assert attributes["total_token_count"] == 12
    assert "private prompt" not in stored_text
    assert "visible answer" not in stored_text


def test_stable_model_error_category_never_returns_exception_text() -> None:
    assert stable_model_error_category(TimeoutError("private prompt timed out")) == "timeout"
    assert stable_model_error_category(ConnectionError("private host")) == "connection"
    assert stable_model_error_category(RuntimeError("private evidence")) == "provider_error"
