from __future__ import annotations

import threading
from types import SimpleNamespace

import kb.converter.reference_page_vl as ref_module


class _DummyPix:
    def tobytes(self, fmt: str) -> bytes:
        return f"{fmt}-bytes".encode("utf-8")


class _DummyPage:
    def __init__(self):
        self.rect = SimpleNamespace(width=612.0, height=792.0)
        self.calls = []

    def get_pixmap(self, clip=None, dpi=0, alpha=False):
        self.calls.append({"clip": clip, "dpi": dpi, "alpha": alpha})
        return _DummyPix()


class _DummyLlmWorker:
    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()
        self._idx = 0

    def call_llm_page_to_markdown(self, png_bytes, **kwargs):
        with self._lock:
            self._idx += 1
            idx = self._idx
            self.calls.append({"png_bytes": png_bytes, **kwargs})
        return f"[{idx}] Reference line {idx}"


class _DummyConverter:
    def __init__(self):
        self.llm_worker = _DummyLlmWorker()
        self._active_speed_config = {"dpi": 220}
        self.dpi = 200


def test_choose_reference_crop_max_tokens_override_defaults(monkeypatch):
    monkeypatch.delenv("KB_PDF_VISION_REFS_CROP_MAX_TOKENS", raising=False)

    assert ref_module.choose_reference_crop_max_tokens_override(speed_mode="normal") == 1536
    assert ref_module.choose_reference_crop_max_tokens_override(speed_mode="ultra_fast") == 1024
    assert ref_module.choose_reference_crop_max_tokens_override(speed_mode="no_llm") is None


def test_choose_reference_crop_max_tokens_override_honors_env(monkeypatch):
    monkeypatch.setenv("KB_PDF_VISION_REFS_CROP_MAX_TOKENS", "2048")

    assert ref_module.choose_reference_crop_max_tokens_override(speed_mode="normal") == 2048


def test_convert_references_page_with_column_vl_passes_crop_token_override(monkeypatch):
    page = _DummyPage()
    converter = _DummyConverter()

    monkeypatch.delenv("KB_PDF_VISION_DPI", raising=False)
    monkeypatch.delenv("KB_PDF_VISION_REFS_CROP_MAX_TOKENS", raising=False)
    monkeypatch.setattr(ref_module, "build_reference_column_crop_rects", lambda **kwargs: ["left", "right"])

    md = ref_module.convert_references_page_with_column_vl(
        converter,
        page=page,
        page_index=0,
        total_pages=3,
        page_hint="refs",
        speed_mode="normal",
    )

    assert "[1] Reference line 1" in md
    assert "[2] Reference line 2" in md
    assert len(converter.llm_worker.calls) == 2
    assert all(call["is_references_page"] is True for call in converter.llm_worker.calls)
    assert all(call["max_tokens_override"] == 1536 for call in converter.llm_worker.calls)
    assert all(int(call["dpi"]) == 240 for call in page.calls)
