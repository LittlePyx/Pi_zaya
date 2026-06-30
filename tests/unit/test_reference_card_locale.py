from api import reference_card_locale
from api.reference_card_locale import (
    _prefer_zh_ref_card_locale,
    _prompt_strongly_prefers_english,
    _ref_card_user_locale,
    _refs_card_locale_pref,
)


def test_refs_card_locale_pref_uses_valid_env(monkeypatch):
    monkeypatch.setenv("KB_REFS_CARD_LOCALE", "zh")

    assert _refs_card_locale_pref() == "zh"


def test_refs_card_locale_pref_falls_back_to_prefs(monkeypatch):
    monkeypatch.delenv("KB_REFS_CARD_LOCALE", raising=False)
    monkeypatch.setattr(reference_card_locale, "load_prefs", lambda: {"refs_card_locale": "en"})

    assert _refs_card_locale_pref() == "en"


def test_ref_card_user_locale_infers_prompt_language(monkeypatch):
    monkeypatch.setenv("KB_REFS_CARD_LOCALE", "auto")
    monkeypatch.setattr(reference_card_locale, "load_prefs", lambda: {})

    assert _ref_card_user_locale("请解释这篇论文") == "zh"
    assert _ref_card_user_locale("Summarize this paper") == "en"
    assert _prefer_zh_ref_card_locale("请解释", "fallback") is True


def test_prompt_strongly_prefers_english_requires_latin_text_only():
    assert _prompt_strongly_prefers_english("Compare these papers") is True
    assert _prompt_strongly_prefers_english("比较 these papers") is False


def test_reference_ui_reuses_shared_locale_policy():
    from api import reference_ui

    assert reference_ui._ref_card_user_locale is reference_card_locale._ref_card_user_locale
    assert reference_ui._prefer_zh_ref_card_locale is reference_card_locale._prefer_zh_ref_card_locale
