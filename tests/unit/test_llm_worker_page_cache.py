from __future__ import annotations

from pathlib import Path
import threading
import time

import pytest

from kb.converter.config import ConvertConfig, LlmConfig
from kb.converter.llm_worker import LLMWorker


class _FakeClient:
    def __init__(self, api_key: str, base_url: str, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = kwargs.get("max_retries")


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


def test_converter_client_disables_hidden_sdk_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMWorker, "_ensure_openai_class", lambda self: _FakeClient)
    worker = LLMWorker(_make_cfg(tmp_path))

    assert worker._client.max_retries == 0


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
        captured.update(kwargs)
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
    assert "_request_timeout_s" not in captured
    assert "_hard_timeout_s" not in captured


def test_call_llm_page_to_markdown_ultra_fast_keeps_2048_token_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMWorker, "_ensure_openai_class", lambda self: _FakeClient)
    LLMWorker._shared_page_ocr_cache.clear()
    LLMWorker._reset_shared_llm_gate_for_tests()
    worker = LLMWorker(_make_cfg(tmp_path))
    captured = {}

    def _fake_llm_create(**kwargs):
        captured.update(kwargs)
        return _FakeResp("fast markdown")

    monkeypatch.setattr(worker, "_llm_create", _fake_llm_create)
    monkeypatch.setenv("KB_PDF_ULTRA_FAST_VISION_TIMEOUT_S", "33")

    out = worker.call_llm_page_to_markdown(
        b"fast-image",
        page_number=1,
        total_pages=4,
        speed_mode="ultra_fast",
        is_references_page=False,
    )

    assert out == "fast markdown"
    assert captured["max_tokens"] == 2048
    assert captured["_request_timeout_s"] == 33.0
    assert captured["_hard_timeout_s"] == 43.0
    assert captured["_semaphore_timeout_s"] == 30.0
    assert captured["_max_retries"] == 0


def test_llm_create_applies_explicit_fast_request_and_hard_timeouts(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMWorker, "_ensure_openai_class", lambda self: _FakeClient)
    LLMWorker._reset_shared_llm_gate_for_tests()
    worker = LLMWorker(_make_cfg(tmp_path))
    worker._llm_gate = None
    captured = {}

    def _fake_guard(**kwargs):
        captured.update(kwargs)
        return _FakeResp("ok")

    monkeypatch.setattr(worker, "_client_create_with_guard_timeout", _fake_guard)

    out = worker._llm_create(
        messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}]}],
        max_tokens=32,
        _request_timeout_s=24,
        _hard_timeout_s=34,
        _semaphore_timeout_s=12,
        _max_retries=0,
    )

    assert out.choices[0].message.content == "ok"
    assert captured["timeout_s"] == 24.0
    assert captured["hard_timeout_s_override"] == 34
    assert captured["has_image_payload"] is True
    assert "_request_timeout_s" not in captured


def test_normal_vision_call_is_clamped_to_shared_page_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMWorker, "_ensure_openai_class", lambda self: _FakeClient)
    monkeypatch.setenv("KB_PDF_VISION_PAGE_BUDGET_S", "40")
    monkeypatch.delenv("KB_PDF_VISION_TIMEOUT_S", raising=False)
    LLMWorker._reset_shared_llm_gate_for_tests()
    worker = LLMWorker(_make_cfg(tmp_path))
    worker._llm_gate = None
    captured = {}

    def _fake_guard(**kwargs):
        captured.update(kwargs)
        return _FakeResp("ok")

    monkeypatch.setattr(worker, "_client_create_with_guard_timeout", _fake_guard)
    with worker.vision_page_budget("normal") as deadline:
        out = worker._llm_create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}
                    ],
                }
            ],
            max_tokens=32,
        )
        assert worker.current_vision_page_deadline() == deadline

    assert out.choices[0].message.content == "ok"
    assert 1.0 <= captured["timeout_s"] <= 40.0
    assert 1.0 <= captured["hard_timeout_s_override"] <= 40.0
    assert worker.current_vision_page_deadline() is None


def test_vision_call_uses_configured_timeout_without_implicit_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMWorker, "_ensure_openai_class", lambda self: _FakeClient)
    monkeypatch.delenv("KB_PDF_VISION_TIMEOUT_S", raising=False)
    LLMWorker._reset_shared_llm_gate_for_tests()
    worker = LLMWorker(_make_cfg(tmp_path))
    worker._llm_gate = None
    captured = {}

    def _fake_guard(**kwargs):
        captured.update(kwargs)
        return _FakeResp("ok")

    monkeypatch.setattr(worker, "_client_create_with_guard_timeout", _fake_guard)
    out = worker._llm_create(
        messages=[
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}],
            }
        ],
        max_tokens=32,
    )

    assert out.choices[0].message.content == "ok"
    assert captured["timeout_s"] == 45.0


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


def test_shared_inflight_gate_releases_slot_after_provider_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMWorker, "_ensure_openai_class", lambda self: _FakeClient)
    monkeypatch.setenv("KB_LLM_MAX_INFLIGHT", "1")
    LLMWorker._reset_shared_llm_gate_for_tests(limit=1)
    worker_a = LLMWorker(_make_cfg(tmp_path))
    worker_b = LLMWorker(_make_cfg(tmp_path))
    calls = {"count": 0}

    def _fail_then_succeed(self, *, timeout_s: float, has_image_payload: bool, **kwargs):
        del self, timeout_s, has_image_payload, kwargs
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("simulated provider failure")
        return _FakeResp("recovered")

    monkeypatch.setattr(LLMWorker, "_client_create_with_guard_timeout", _fail_then_succeed)

    with pytest.raises(RuntimeError, match="simulated provider failure"):
        worker_a._llm_create(
            messages=[{"role": "user", "content": "first"}],
            max_tokens=32,
            max_retries=0,
        )
    recovered = worker_b._llm_create(
        messages=[{"role": "user", "content": "second"}],
        max_tokens=32,
        max_retries=0,
    )

    assert recovered.choices[0].message.content == "recovered"
    assert calls["count"] == 2


def test_workers_share_cross_process_global_inflight_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMWorker, "_ensure_openai_class", lambda self: _FakeClient)
    monkeypatch.setenv("KB_LLM_MAX_INFLIGHT", "2")
    monkeypatch.setenv("KB_LLM_GLOBAL_COORDINATOR", str(tmp_path / "global-inflight"))
    monkeypatch.setenv("KB_LLM_GLOBAL_MAX_INFLIGHT", "1")
    monkeypatch.setenv("KB_LLM_GLOBAL_MIN_INFLIGHT", "1")
    monkeypatch.setenv("KB_LLM_GLOBAL_REQUIRED", "1")
    monkeypatch.setenv("KB_LLM_GLOBAL_OWNER", "test-process")
    LLMWorker._reset_shared_llm_gate_for_tests(limit=2)

    worker_a = LLMWorker(_make_cfg(tmp_path))
    worker_b = LLMWorker(_make_cfg(tmp_path))
    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()
    barrier = threading.Barrier(3)

    def _fake_guard_timeout(self, *, timeout_s: float, has_image_payload: bool, **kwargs):
        del self, timeout_s, has_image_payload, kwargs
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.1)
        with lock:
            state["active"] -= 1
        return _FakeResp("ok")

    monkeypatch.setattr(LLMWorker, "_client_create_with_guard_timeout", _fake_guard_timeout)
    errors = []

    def _run(worker):
        try:
            barrier.wait(timeout=2.0)
            worker._llm_create(messages=[{"role": "user", "content": "hi"}], max_tokens=32)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_run, args=(worker,)) for worker in (worker_a, worker_b)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=2.0)
    for thread in threads:
        thread.join(timeout=3.0)

    assert errors == []
    assert state["max_active"] == 1


def test_provider_timeout_reduces_dynamic_global_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMWorker, "_ensure_openai_class", lambda self: _FakeClient)
    monkeypatch.setenv("KB_LLM_MAX_INFLIGHT", "4")
    monkeypatch.setenv("KB_LLM_GLOBAL_COORDINATOR", str(tmp_path / "global-inflight"))
    monkeypatch.setenv("KB_LLM_GLOBAL_MAX_INFLIGHT", "4")
    monkeypatch.setenv("KB_LLM_GLOBAL_MIN_INFLIGHT", "2")
    monkeypatch.setenv("KB_LLM_GLOBAL_REQUIRED", "1")
    monkeypatch.setenv("KB_LLM_GLOBAL_OWNER", "timeout-test")
    LLMWorker._reset_shared_llm_gate_for_tests(limit=4)
    worker = LLMWorker(_make_cfg(tmp_path))

    def _timeout(self, *, timeout_s: float, has_image_payload: bool, **kwargs):
        del self, timeout_s, has_image_payload, kwargs
        raise TimeoutError("simulated provider timeout")

    monkeypatch.setattr(LLMWorker, "_client_create_with_guard_timeout", _timeout)
    with pytest.raises(TimeoutError, match="simulated provider timeout"):
        worker._llm_create(
            messages=[{"role": "user", "content": "first"}],
            max_tokens=32,
            _max_retries=0,
        )

    snapshot = worker.get_global_llm_inflight_snapshot()
    assert snapshot["configured_limit"] == 4
    assert snapshot["effective_limit"] == 3
    assert snapshot["timeout_events"] == 1
    assert snapshot["limit_reductions"] == 1


def test_ultra_fast_vision_circuit_skips_later_requests_after_timeouts(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMWorker, "_ensure_openai_class", lambda self: _FakeClient)
    monkeypatch.setenv("KB_PDF_VISION_PAGE_CACHE", "0")
    monkeypatch.setenv("KB_PDF_VISION_CIRCUIT_BREAKER", "1")
    monkeypatch.setenv("KB_PDF_VISION_CIRCUIT_ULTRA_FAST_THRESHOLD", "2")
    monkeypatch.setenv("KB_PDF_VISION_CIRCUIT_ULTRA_FAST_COOLDOWN_S", "60")
    LLMWorker._reset_shared_vision_circuit_for_tests()
    worker = LLMWorker(_make_cfg(tmp_path))
    calls = 0

    def _timeout(**kwargs):
        nonlocal calls
        calls += 1
        raise TimeoutError("simulated provider timeout")

    monkeypatch.setattr(worker, "_llm_create", _timeout)

    outputs = [
        worker.call_llm_page_to_markdown(
            f"page-{page}".encode(),
            page_number=page,
            total_pages=6,
            speed_mode="ultra_fast",
            is_references_page=False,
        )
        for page in range(6)
    ]

    assert outputs == [None] * 6
    assert calls == 2
    assert worker.get_last_vl_error_code() == "circuit_open"


def test_ordinary_vision_business_error_does_not_open_circuit(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMWorker, "_ensure_openai_class", lambda self: _FakeClient)
    monkeypatch.setenv("KB_PDF_VISION_PAGE_CACHE", "0")
    monkeypatch.setenv("KB_PDF_VISION_CIRCUIT_BREAKER", "1")
    monkeypatch.setenv("KB_PDF_VISION_CIRCUIT_ULTRA_FAST_THRESHOLD", "1")
    LLMWorker._reset_shared_vision_circuit_for_tests()
    worker = LLMWorker(_make_cfg(tmp_path))
    calls = 0

    def _business_then_success(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("ordinary document validation error")
        return _FakeResp("recovered markdown")

    monkeypatch.setattr(worker, "_llm_create", _business_then_success)

    first = worker.call_llm_page_to_markdown(
        b"page-a",
        page_number=0,
        total_pages=2,
        speed_mode="ultra_fast",
        is_references_page=False,
    )
    second = worker.call_llm_page_to_markdown(
        b"page-b",
        page_number=1,
        total_pages=2,
        speed_mode="ultra_fast",
        is_references_page=False,
    )

    assert first is None
    assert second == "recovered markdown"
    assert calls == 2
