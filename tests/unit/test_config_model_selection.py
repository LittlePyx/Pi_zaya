from __future__ import annotations

from kb import config


def _clear_provider_env(monkeypatch) -> None:
    for name in (
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "KB_DEEPSEEK_THINKING_MODE",
        "QWEN_API_KEY",
        "QWEN_TEXT_MODEL",
        "QWEN_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "load_dotenv", lambda *_args, **_kwargs: None)


def test_deepseek_default_model_does_not_inherit_openai_model(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr(config, "_load_runtime_prefs", lambda: {})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")

    settings = config.load_settings()

    assert settings.text_model == "deepseek-v4-flash"
    assert settings.text_base_url == "https://api.deepseek.com/v1"
    assert settings.deepseek_thinking_mode == "disabled"


def test_deepseek_explicit_v4_pro_is_preserved(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr(config, "_load_runtime_prefs", lambda: {})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

    settings = config.load_settings()

    assert settings.text_model == "deepseek-v4-pro"
    assert settings.deepseek_thinking_mode == "enabled"


def test_stored_official_deepseek_alias_migrates_to_explicit_v4(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr(
        config,
        "_load_runtime_prefs",
        lambda: {
            "text_api_key": "stored-key",
            "text_base_url": "https://api.deepseek.com/v1",
            "text_model": "deepseek-chat",
        },
    )

    settings = config.load_settings()

    assert settings.text_model == "deepseek-v4-flash"
    assert settings.deepseek_thinking_mode == "disabled"


def test_deepseek_reasoner_alias_preserves_thinking_intent(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr(config, "_load_runtime_prefs", lambda: {})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")

    settings = config.load_settings()

    assert settings.text_model == "deepseek-v4-flash"
    assert settings.deepseek_thinking_mode == "enabled"


def test_deepseek_thinking_mode_can_be_explicitly_overridden(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr(config, "_load_runtime_prefs", lambda: {})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("KB_DEEPSEEK_THINKING_MODE", "enabled")

    settings = config.load_settings()

    assert settings.text_model == "deepseek-v4-flash"
    assert settings.deepseek_thinking_mode == "enabled"
