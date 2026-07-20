from __future__ import annotations

from kb import config


def _clear_provider_env(monkeypatch) -> None:
    for name in (
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
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


def test_deepseek_explicit_v4_pro_is_preserved(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr(config, "_load_runtime_prefs", lambda: {})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

    settings = config.load_settings()

    assert settings.text_model == "deepseek-v4-pro"


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
