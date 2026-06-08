from __future__ import annotations

import time
from types import SimpleNamespace

from api.routers import settings as settings_router
from kb import config as config_module


def test_update_settings_persists_sidebar_collapsed(monkeypatch):
    stored: dict[str, object] = {}

    def load_prefs() -> dict[str, object]:
        return dict(stored)

    def save_prefs(data: dict[str, object]) -> None:
        stored.clear()
        stored.update(data)

    monkeypatch.setattr(settings_router, "load_prefs", load_prefs)
    monkeypatch.setattr(settings_router, "save_prefs", save_prefs)

    assert settings_router.update_settings(settings_router.PrefsPatch(sidebar_collapsed=True)) == {"ok": True}
    assert stored["sidebar_collapsed"] is True

    assert settings_router.update_settings(settings_router.PrefsPatch(sidebar_collapsed=False)) == {"ok": True}
    assert stored["sidebar_collapsed"] is False


def test_update_settings_persists_auto_backup_preference(monkeypatch):
    stored: dict[str, object] = {}

    def load_prefs() -> dict[str, object]:
        return dict(stored)

    def save_prefs(data: dict[str, object]) -> None:
        stored.clear()
        stored.update(data)

    monkeypatch.setattr(settings_router, "load_prefs", load_prefs)
    monkeypatch.setattr(settings_router, "save_prefs", save_prefs)

    assert settings_router.update_settings(settings_router.PrefsPatch(auto_backup_enabled=True)) == {"ok": True}
    assert stored["auto_backup_enabled"] is True

    assert settings_router.update_settings(settings_router.PrefsPatch(auto_backup_enabled=False)) == {"ok": True}
    assert stored["auto_backup_enabled"] is False


def test_update_settings_persists_api_credentials_and_clears_cache(monkeypatch):
    stored: dict[str, object] = {}
    cleared = {"count": 0}

    def load_prefs() -> dict[str, object]:
        return dict(stored)

    def save_prefs(data: dict[str, object]) -> None:
        stored.clear()
        stored.update(data)

    def get_settings():
        return None

    def cache_clear() -> None:
        cleared["count"] += 1

    get_settings.cache_clear = cache_clear  # type: ignore[attr-defined]
    monkeypatch.setattr(settings_router, "load_prefs", load_prefs)
    monkeypatch.setattr(settings_router, "save_prefs", save_prefs)
    monkeypatch.setattr(settings_router, "get_settings", get_settings)

    assert settings_router.update_settings(settings_router.PrefsPatch(
        text_api_key=' "sk-text" ',
        text_base_url="https://text.example/v1/",
        text_model="text-model",
        vision_api_key="sk-vision",
        vision_base_url="https://vision.example/v1/",
        vision_model="vision-model",
    )) == {"ok": True}

    assert stored["text_api_key"] == "sk-text"
    assert stored["text_base_url"] == "https://text.example/v1"
    assert stored["text_model"] == "text-model"
    assert stored["vision_api_key"] == "sk-vision"
    assert stored["vision_base_url"] == "https://vision.example/v1"
    assert stored["vision_model"] == "vision-model"
    assert cleared["count"] == 1


def test_get_settings_returns_text_and_vision_connection(monkeypatch):
    settings_router._LLM_TEST_RESULTS.clear()
    settings = SimpleNamespace(
        model="text-model",
        base_url="https://text.example/v1",
        api_key="sk-text",
        text_api_key="sk-text",
        text_base_url="https://text.example/v1",
        text_model="text-model",
        vision_api_key="sk-vision",
        vision_base_url="https://vision.example/v1",
        vision_model="vision-model",
        vision_uses_text_fallback=False,
        auto_route=True,
        db_dir="db",
    )

    monkeypatch.setattr(settings_router, "get_settings", lambda: settings)
    monkeypatch.setattr(settings_router, "load_prefs", lambda: {"text_api_key": "sk-text", "theme": "light"})

    payload = settings_router.get_all_settings()

    assert payload["has_api_key"] is True
    assert payload["connection"]["text"]["model"] == "text-model"
    assert payload["connection"]["vision"]["model"] == "vision-model"
    assert payload["connection"]["vision"]["uses_text_fallback"] is False
    assert payload["connection"]["auto_route"] is True
    assert payload["readiness"]["overall"]["status"] == "ok"
    assert payload["readiness"]["providers"]["text"]["status"] == "configured"
    assert payload["prefs"] == {"theme": "light"}


def test_llm_connection_test_accepts_transient_overrides(monkeypatch):
    settings_router._LLM_TEST_RESULTS.clear()
    settings = SimpleNamespace(
        text_api_key="saved-text",
        text_base_url="https://saved-text.example/v1",
        text_model="saved-text-model",
        vision_api_key="saved-vision",
        vision_base_url="https://saved-vision.example/v1",
        vision_model="saved-vision-model",
        timeout_s=9.0,
    )
    observed: dict[str, object] = {}

    def fake_test_chat_completion(**kwargs):
        observed.update(kwargs)
        return {"ok": True, "reply": "OK"}

    monkeypatch.setattr(settings_router, "get_settings", lambda: settings)
    monkeypatch.setattr(settings_router, "_test_chat_completion", fake_test_chat_completion)

    result = settings_router.test_llm(settings_router.ConnectionTestBody(
        target="vision",
        api_key=" transient-key ",
        base_url="https://transient.example/v1/",
        model="transient-model",
    ))

    assert result["ok"] is True
    assert result["reply"] == "OK"
    assert isinstance(result["checked_at"], float)
    assert observed == {
        "api_key": "transient-key",
        "base_url": "https://transient.example/v1",
        "model": "transient-model",
        "timeout_s": 9.0,
    }


def test_llm_readiness_reports_missing_text_key(monkeypatch):
    settings_router._LLM_TEST_RESULTS.clear()
    settings = SimpleNamespace(
        text_api_key=None,
        text_base_url="https://text.example/v1",
        text_model="text-model",
        vision_api_key=None,
        vision_base_url="https://text.example/v1",
        vision_model="text-model",
        vision_uses_text_fallback=True,
        auto_route=False,
    )

    monkeypatch.setattr(settings_router, "get_settings", lambda: settings)

    payload = settings_router.get_llm_readiness()

    assert payload["overall"] == {"status": "error", "reason": "missing_api_key", "target": "text"}
    assert payload["providers"]["text"]["status"] == "missing"
    assert payload["providers"]["text"]["severity"] == "error"


def test_llm_readiness_keeps_last_failed_test(monkeypatch):
    settings_router._LLM_TEST_RESULTS.clear()
    settings = SimpleNamespace(
        text_api_key="saved-text",
        text_base_url="https://saved-text.example/v1",
        text_model="saved-text-model",
        vision_api_key="saved-vision",
        vision_base_url="https://saved-vision.example/v1",
        vision_model="saved-vision-model",
        vision_uses_text_fallback=False,
        auto_route=True,
        timeout_s=9.0,
    )

    monkeypatch.setattr(settings_router, "get_settings", lambda: settings)
    monkeypatch.setattr(
        settings_router,
        "_test_chat_completion",
        lambda **kwargs: {"ok": False, "error": "401 unauthorized"},
    )

    result = settings_router.test_llm(settings_router.ConnectionTestBody(target="text"))
    payload = settings_router.get_llm_readiness()

    assert result["ok"] is False
    assert result["error_type"] == "auth"
    assert payload["overall"] == {"status": "error", "reason": "auth", "target": "text"}
    assert payload["providers"]["text"]["status"] == "failed"
    assert payload["providers"]["text"]["last_test"]["error_type"] == "auth"


def test_app_readiness_warns_after_recent_restore(monkeypatch, tmp_path):
    settings_router._LLM_TEST_RESULTS.clear()
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    settings = SimpleNamespace(
        app_env="development",
        production=False,
        auth_required=False,
        text_api_key="sk-text",
        text_base_url="https://text.example/v1",
        text_model="text-model",
        vision_api_key="sk-vision",
        vision_base_url="https://vision.example/v1",
        vision_model="vision-model",
        vision_uses_text_fallback=False,
        auto_route=True,
        db_dir=db_dir,
        chat_db_path=tmp_path / "chat.sqlite3",
        library_db_path=tmp_path / "library.sqlite3",
    )
    restore_event = {
        "event": "restore",
        "status": "restored",
        "backup": "backup-recent.zip",
        "created_at": time.time(),
        "ok": True,
        "restart_required": True,
        "components": {"chat": True, "library": True, "db": True},
        "restored": [{"target": str(tmp_path / "chat.sqlite3")}],
        "pre_restore_backup": {"path": str(tmp_path / "backup.zip")},
    }
    monkeypatch.setattr(settings_router, "latest_restore_review_state", lambda: {
        "restore": restore_event,
        "acknowledgement": None,
        "acknowledged": False,
    })

    payload = settings_router.production_readiness_payload(settings)
    item = next(item for item in payload["items"] if item["key"] == "recent_restore")

    assert item["severity"] == "warning"
    assert item["action"] == "restart_and_check"
    assert payload["restore"]["latest"]["backup"] == "backup-recent.zip"
    assert payload["restore"]["acknowledged"] is False
    assert "restored" not in payload["restore"]["latest"]
    assert "pre_restore_backup" not in payload["restore"]["latest"]


def test_app_readiness_omits_recent_restore_after_acknowledgement(monkeypatch, tmp_path):
    settings_router._LLM_TEST_RESULTS.clear()
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    settings = SimpleNamespace(
        app_env="development",
        production=False,
        auth_required=False,
        text_api_key="sk-text",
        text_base_url="https://text.example/v1",
        text_model="text-model",
        vision_api_key="sk-vision",
        vision_base_url="https://vision.example/v1",
        vision_model="vision-model",
        vision_uses_text_fallback=False,
        auto_route=True,
        db_dir=db_dir,
        chat_db_path=tmp_path / "chat.sqlite3",
        library_db_path=tmp_path / "library.sqlite3",
    )
    restore_event = {
        "event": "restore",
        "status": "restored",
        "backup": "backup-recent.zip",
        "created_at": time.time(),
        "ok": True,
        "restart_required": True,
    }
    monkeypatch.setattr(settings_router, "latest_restore_review_state", lambda: {
        "restore": restore_event,
        "acknowledgement": {
            "event": "restore_review_acknowledged",
            "status": "acknowledged",
            "backup": "backup-recent.zip",
            "created_at": time.time(),
            "ok": True,
            "restore_created_at": restore_event["created_at"],
        },
        "acknowledged": True,
    })

    payload = settings_router.production_readiness_payload(settings)

    assert all(item["key"] != "recent_restore" for item in payload["items"])
    assert payload["restore"]["acknowledged"] is True


def test_load_settings_uses_local_api_prefs_when_env_keys_missing(monkeypatch):
    for key in (
        "DEEPSEEK_API_KEY",
        "QWEN_API_KEY",
        "OPENAI_API_KEY",
        "DEEPSEEK_BASE_URL",
        "QWEN_BASE_URL",
        "OPENAI_BASE_URL",
        "DEEPSEEK_MODEL",
        "QWEN_MODEL",
        "OPENAI_MODEL",
    ):
        monkeypatch.setenv(key, "")

    monkeypatch.setattr(config_module, "_load_runtime_prefs", lambda: {
        "text_api_key": "sk-text",
        "text_base_url": "https://text.example/v1/",
        "text_model": "text-model",
        "vision_api_key": "sk-vision",
        "vision_base_url": "https://vision.example/v1/",
        "vision_model": "vision-model",
        "auto_backup_enabled": True,
    })

    settings = config_module.load_settings()

    assert settings.text_api_key == "sk-text"
    assert settings.text_base_url == "https://text.example/v1"
    assert settings.text_model == "text-model"
    assert settings.vision_api_key == "sk-vision"
    assert settings.vision_base_url == "https://vision.example/v1"
    assert settings.vision_model == "vision-model"
    assert settings.auto_route is True
    assert settings.vision_uses_text_fallback is False
    assert settings.auto_backup_enabled is True
