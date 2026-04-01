from __future__ import annotations

from kb.generation_state_runtime import (
    _gen_store_answer_provenance_async,
    _is_live_assistant_text,
    _live_assistant_task_id,
    _live_assistant_text,
    _should_run_provenance_async_refine,
)


def test_live_assistant_helpers_roundtrip():
    text = _live_assistant_text("task-123", live_assistant_prefix="__LIVE__: ")

    assert _is_live_assistant_text(text, live_assistant_prefix="__LIVE__: ") is True
    assert _live_assistant_task_id(text, live_assistant_prefix="__LIVE__: ") == "task-123"
    assert _is_live_assistant_text("plain text", live_assistant_prefix="__LIVE__: ") is False


def test_should_run_provenance_async_refine_requires_flags_and_api_key():
    class _Settings:
        api_key = "test-key"

    task = {
        "paper_guide_mode": True,
        "paper_guide_bound_source_path": "/tmp/demo.pdf",
        "llm_rerank": True,
        "settings_obj": _Settings(),
    }

    assert _should_run_provenance_async_refine(task, environ={"KB_PROVENANCE_ASYNC_LLM": "1"}) is True
    assert _should_run_provenance_async_refine(task, environ={"KB_PROVENANCE_ASYNC_LLM": "0"}) is False
    assert _should_run_provenance_async_refine({**task, "llm_rerank": False}, environ={"KB_PROVENANCE_ASYNC_LLM": "1"}) is False
    assert _should_run_provenance_async_refine({**task, "settings_obj": object()}, environ={"KB_PROVENANCE_ASYNC_LLM": "1"}) is False


def test_gen_store_answer_provenance_async_enables_llm_rerank():
    captured: dict[str, object] = {}

    def _fake_store(
        task: dict,
        *,
        answer: str,
        answer_hits: list[dict],
        support_resolution: list[dict] | None = None,
    ) -> None:
        captured["task"] = dict(task)
        captured["answer"] = answer
        captured["answer_hits"] = list(answer_hits)
        captured["support_resolution"] = list(support_resolution or [])

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, name=None):
            self._target = target

        def start(self):
            if callable(self._target):
                self._target()

    class _ThreadingModule:
        Thread = _ImmediateThread

    _gen_store_answer_provenance_async(
        {"llm_rerank": False, "paper_guide_mode": True, "paper_guide_bound_source_path": "/tmp/demo.pdf"},
        answer="demo answer",
        answer_hits=[{"text": "x"}],
        store_answer_provenance=_fake_store,
        threading_module=_ThreadingModule,
    )

    assert captured["answer"] == "demo answer"
    assert captured["answer_hits"] == [{"text": "x"}]
    assert captured["support_resolution"] == []
    assert isinstance(captured["task"], dict)
    assert captured["task"].get("llm_rerank") is True
