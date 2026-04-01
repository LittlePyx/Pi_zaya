from __future__ import annotations

from pathlib import Path
import threading
import time

from kb.converter.config import ConvertConfig, LlmConfig
from kb.converter.llm_worker import LLMWorker


class _FakeClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url


class _FakeResp:
    def __init__(self, content: str):
        msg = type("Msg", (), {"content": content})()
        choice = type("Choice", (), {"message": msg})()
        self.choices = [choice]


def _make_cfg(tmp_path) -> ConvertConfig:
    return ConvertConfig(
        pdf_path=tmp_path / "dummy.pdf",
        out_dir=tmp_path,
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=LlmConfig(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="qwen3-vl-plus",
        ),
        llm_workers=1,
    )


def test_call_llm_page_to_markdown_reuses_cached_exact_request(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMWorker, "_ensure_openai_class", lambda self: _FakeClient)
    LLMWorker._shared_page_ocr_cache.clear()
    LLMWorker._reset_shared_llm_gate_for_tests()
    worker = LLMWorker(_make_cfg(tmp_path))

    calls = {"n": 0}

    def _fake_llm_create(**kwargs):
        calls["n"] += 1
        return _FakeResp("cached markdown")

    monkeypatch.setattr(worker, "_llm_create", _fake_llm_create)

    out1 = worker.call_llm_page_to_markdown(
        b"same-image",
        page_number=0,
        total_pages=2,
        hint="same hint",
        speed_mode="normal",
        is_references_page=False,
    )
    out2 = worker.call_llm_page_to_markdown(
        b"same-image",
        page_number=0,
        total_pages=2,
        hint="same hint",
        speed_mode="normal",
        is_references_page=False,
    )

    assert out1 == "cached markdown"
    assert out2 == "cached markdown"
    assert calls["n"] == 1


def test_call_llm_page_to_markdown_cache_key_changes_with_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMWorker, "_ensure_openai_class", lambda self: _FakeClient)
    LLMWorker._shared_page_ocr_cache.clear()
    LLMWorker._reset_shared_llm_gate_for_tests()
    worker = LLMWorker(_make_cfg(tmp_path))

    calls = {"n": 0}

    def _fake_llm_create(**kwargs):
        calls["n"] += 1
        return _FakeResp(f"markdown-{calls['n']}")

    monkeypatch.setattr(worker, "_llm_create", _fake_llm_create)

    out1 = worker.call_llm_page_to_markdown(
        b"same-image",
        page_number=0,
        total_pages=2,
        hint="hint-a",
        speed_mode="normal",
        is_references_page=False,
    )
    out2 = worker.call_llm_page_to_markdown(
        b"same-image",
        page_number=0,
        total_pages=2,
        hint="hint-b",
        speed_mode="normal",
        is_references_page=False,
    )

    assert out1 == "markdown-1"
    assert out2 == "markdown-2"
    assert calls["n"] == 2


def test_call_llm_page_to_markdown_skips_cache_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMWorker, "_ensure_openai_class", lambda self: _FakeClient)
    monkeypatch.setenv("KB_PDF_VISION_PAGE_CACHE", "0")
    LLMWorker._shared_page_ocr_cache.clear()
    LLMWorker._reset_shared_llm_gate_for_tests()
    worker = LLMWorker(_make_cfg(tmp_path))

    calls = {"n": 0}

    def _fake_llm_create(**kwargs):
        calls["n"] += 1
        return _FakeResp("uncached markdown")

    monkeypatch.setattr(worker, "_llm_create", _fake_llm_create)

    out1 = worker.call_llm_page_to_markdown(
        b"same-image",
        page_number=0,
        total_pages=2,
        hint="same hint",
        speed_mode="normal",
        is_references_page=False,
    )
    out2 = worker.call_llm_page_to_markdown(
        b"same-image",
        page_number=0,
        total_pages=2,
        hint="same hint",
        speed_mode="normal",
        is_references_page=False,
    )

    assert out1 == "uncached markdown"
    assert out2 == "uncached markdown"
    assert calls["n"] == 2


def test_call_llm_page_to_markdown_applies_lower_max_tokens_override(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMWorker, "_ensure_openai_class", lambda self: _FakeClient)
    LLMWorker._shared_page_ocr_cache.clear()
    LLMWorker._reset_shared_llm_gate_for_tests()
    worker = LLMWorker(_make_cfg(tmp_path))

    captured = {}

    def _fake_llm_create(**kwargs):
        captured["max_tokens"] = kwargs.get("max_tokens")
        return _FakeResp("token-capped markdown")

    monkeypatch.setattr(worker, "_llm_create", _fake_llm_create)

    out = worker.call_llm_page_to_markdown(
        b"same-image",
        page_number=1,
        total_pages=4,
        hint="",
        speed_mode="normal",
        is_references_page=False,
        max_tokens_override=3072,
    )

    assert out == "token-capped markdown"
    assert captured["max_tokens"] == 3072


def test_multiple_workers_share_process_level_inflight_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMWorker, "_ensure_openai_class", lambda self: _FakeClient)
    monkeypatch.setenv("KB_LLM_MAX_INFLIGHT", "1")
    LLMWorker._reset_shared_llm_gate_for_tests(limit=1)

    worker_a = LLMWorker(_make_cfg(tmp_path))
    worker_b = LLMWorker(_make_cfg(tmp_path))

    state = {"active": 0, "max_active": 0, "calls": 0}
    lock = threading.Lock()
    barrier = threading.Barrier(3)

    def _fake_guard_timeout(self, *, timeout_s: float, has_image_payload: bool, **kwargs):
        del timeout_s, has_image_payload, kwargs
        with lock:
            state["active"] += 1
            state["calls"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.12)
        with lock:
            state["active"] -= 1
        return _FakeResp("ok")

    monkeypatch.setattr(LLMWorker, "_client_create_with_guard_timeout", _fake_guard_timeout)

    errs = []

    def _run(worker):
        try:
            barrier.wait(timeout=2.0)
            out = worker._llm_create(messages=[{"role": "user", "content": "hi"}], max_tokens=32)
            assert str(out.choices[0].message.content) == "ok"
        except Exception as exc:
            errs.append(exc)

    t1 = threading.Thread(target=_run, args=(worker_a,))
    t2 = threading.Thread(target=_run, args=(worker_b,))
    t1.start()
    t2.start()
    barrier.wait(timeout=2.0)
    t1.join(timeout=3.0)
    t2.join(timeout=3.0)

    assert errs == []
    assert state["calls"] == 2
    assert state["max_active"] == 1
