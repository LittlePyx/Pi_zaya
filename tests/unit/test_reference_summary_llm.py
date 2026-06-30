from __future__ import annotations

from types import SimpleNamespace

from api import reference_summary_llm as summary_llm


class FakeChat:
    calls: list[dict] = []
    response = "本文提出一种自适应采样方法，并通过实验显示其能提升重建质量。"

    def __init__(self, settings: object) -> None:
        self.settings = settings

    def chat(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self.response


def _settings() -> SimpleNamespace:
    return SimpleNamespace(api_key="key", timeout_s=60.0, max_retries=3)


def test_translate_summary_to_zh_returns_source_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("KB_CITE_SUMMARY_TRANSLATE_ZH", "0")

    out = summary_llm._translate_summary_to_zh(
        "We propose an adaptive sampling method and experiments show improved reconstruction quality.",
        load_settings_func=_settings,
        chat_cls=FakeChat,
    )

    assert out.startswith("We propose an adaptive sampling method")


def test_translate_summary_to_zh_uses_chat_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("KB_CITE_SUMMARY_TRANSLATE_ZH", "1")
    FakeChat.calls = []

    out = summary_llm._translate_summary_to_zh(
        "We propose an adaptive sampling method and experiments show improved reconstruction quality.",
        load_settings_func=_settings,
        chat_cls=FakeChat,
    )

    assert out == "本文提出一种自适应采样方法，并通过实验显示其能提升重建质量。"
    assert FakeChat.calls
    assert FakeChat.calls[0]["temperature"] == 0.0


def test_llm_summarize_abstract_zh_respects_disabled_flag(monkeypatch) -> None:
    monkeypatch.setenv("KB_CITE_SUMMARY_USE_LLM", "0")

    out = summary_llm._llm_summarize_abstract_zh(
        "Adaptive sampling for imaging",
        "We propose an adaptive sampling method and experiments show improved reconstruction quality.",
        load_settings_func=_settings,
        chat_cls=FakeChat,
        is_summary_quality_ok=lambda text: True,
    )

    assert out == ""


def test_llm_summarize_abstract_zh_accepts_quality_checked_chinese(monkeypatch) -> None:
    monkeypatch.setenv("KB_CITE_SUMMARY_USE_LLM", "1")
    FakeChat.response = "本文提出一种自适应采样方法，并通过实验显示其能提升重建质量。"

    out = summary_llm._llm_summarize_abstract_zh(
        "Adaptive sampling for imaging",
        "We propose an adaptive sampling method and experiments show improved reconstruction quality.",
        load_settings_func=_settings,
        chat_cls=FakeChat,
        is_summary_quality_ok=lambda text: True,
    )

    assert out == "本文提出一种自适应采样方法，并通过实验显示其能提升重建质量。"


def test_llm_summarize_abstract_zh_rejects_non_chinese_output(monkeypatch) -> None:
    monkeypatch.setenv("KB_CITE_SUMMARY_USE_LLM", "1")
    FakeChat.response = "We propose a method and experiments show improved reconstruction quality."

    out = summary_llm._llm_summarize_abstract_zh(
        "Adaptive sampling for imaging",
        "We propose an adaptive sampling method and experiments show improved reconstruction quality.",
        load_settings_func=_settings,
        chat_cls=FakeChat,
        is_summary_quality_ok=lambda text: True,
    )

    assert out == ""


def test_finalize_abstract_summary_line_prefers_llm_then_translation() -> None:
    out, generation = summary_llm._finalize_abstract_summary_line(
        title="Adaptive sampling",
        abstract_text="We propose an adaptive sampling method and experiments show improved reconstruction quality.",
        llm_summarize_abstract_zh=lambda **kwargs: "LLM summary",
        translate_summary_to_zh=lambda text: "Translated summary",
    )

    assert out == "LLM summary"
    assert generation == "llm_abstract"

    out2, generation2 = summary_llm._finalize_abstract_summary_line(
        title="Adaptive sampling",
        abstract_text="We propose an adaptive sampling method and experiments show improved reconstruction quality.",
        llm_summarize_abstract_zh=lambda **kwargs: "",
        translate_summary_to_zh=lambda text: "Translated summary",
    )

    assert out2 == "Translated summary"
    assert generation2 == "translated_abstract"
