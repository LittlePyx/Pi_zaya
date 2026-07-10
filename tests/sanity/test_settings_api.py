from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from api import deps
from api.routers import settings as settings_router
from kb import config as config_module
from kb.user_issue_store import UserIssueStore


def _app_readiness_settings(
    tmp_path,
    *,
    production: bool = False,
    remote_enabled: bool = False,
    remote_url: str = "",
    remote_token: str = "",
):
    db_dir = tmp_path / "db"
    db_dir.mkdir(exist_ok=True)
    return SimpleNamespace(
        app_env="production" if production else "development",
        production=production,
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
        user_issues_db_path=tmp_path / "user_issues.sqlite3",
        user_issues_remote_enabled=remote_enabled,
        user_issues_remote_url=remote_url,
        user_issues_remote_token=remote_token,
    )


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


def test_update_settings_persists_quality_data_sharing_preference(monkeypatch):
    stored: dict[str, object] = {}
    monkeypatch.setattr(settings_router.secrets, "token_urlsafe", lambda _n=18: "anon-client-token")

    def load_prefs() -> dict[str, object]:
        return dict(stored)

    def save_prefs(data: dict[str, object]) -> None:
        stored.clear()
        stored.update(data)

    monkeypatch.setattr(settings_router, "load_prefs", load_prefs)
    monkeypatch.setattr(settings_router, "save_prefs", save_prefs)

    assert settings_router.update_settings(settings_router.PrefsPatch(quality_data_sharing_enabled=True)) == {"ok": True}
    assert stored["quality_data_sharing_enabled"] is True
    assert stored["quality_data_client_id"] == "anon-client-token"

    disabled = settings_router.update_settings(settings_router.PrefsPatch(quality_data_sharing_enabled=False))
    assert disabled["ok"] is True
    assert disabled["quality_data_cleanup"]["ok"] is True
    assert stored["quality_data_sharing_enabled"] is False
    assert "quality_data_client_id" not in stored


def test_settings_preferences_honor_configured_user_prefs_path(monkeypatch, tmp_path):
    prefs_path = tmp_path / "runtime" / "prefs.json"
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))
    monkeypatch.setattr(settings_router.secrets, "token_urlsafe", lambda _n=18: "anon-client-token")

    assert settings_router.update_settings(settings_router.PrefsPatch(quality_data_sharing_enabled=True)) == {"ok": True}

    stored = json.loads(prefs_path.read_text(encoding="utf-8"))
    assert stored["quality_data_sharing_enabled"] is True
    assert stored["quality_data_client_id"] == "anon-client-token"


def test_save_prefs_uses_configured_path_and_replaces_atomically(monkeypatch, tmp_path):
    prefs_path = tmp_path / "runtime" / "prefs.json"
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))

    deps.save_prefs({"theme": "dark", "quality_data_sharing_enabled": False})
    deps.save_prefs({"theme": "light", "quality_data_sharing_enabled": True})

    assert deps.load_prefs() == {"theme": "light", "quality_data_sharing_enabled": True}
    assert not list(prefs_path.parent.glob(f".{prefs_path.name}.*.tmp"))


def test_load_prefs_ignores_non_object_json(monkeypatch, tmp_path):
    prefs_path = tmp_path / "runtime" / "prefs.json"
    prefs_path.parent.mkdir(parents=True)
    prefs_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))

    assert deps.load_prefs() == {}


def test_settings_routes_recover_from_non_object_prefs(monkeypatch, tmp_path):
    prefs_path = tmp_path / "runtime" / "prefs.json"
    prefs_path.parent.mkdir(parents=True)
    prefs_path.write_text(json.dumps(["broken"]), encoding="utf-8")
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))

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
        db_dir=tmp_path / "db",
        chat_db_path=tmp_path / "chat.sqlite3",
        library_db_path=tmp_path / "library.sqlite3",
        user_issues_db_path=tmp_path / "user_issues.sqlite3",
        user_issues_remote_enabled=False,
        user_issues_remote_url="",
        user_issues_remote_token="",
        auth_required=False,
        production=False,
        app_env="development",
    )
    monkeypatch.setattr(settings_router, "get_settings", lambda: settings)

    payload = settings_router.get_all_settings()
    assert payload["prefs"] == {}

    assert settings_router.update_settings(settings_router.PrefsPatch(theme="light")) == {"ok": True}
    assert json.loads(prefs_path.read_text(encoding="utf-8")) == {"theme": "light"}


def test_update_settings_discards_unsent_quality_data_when_disabled(monkeypatch, tmp_path):
    prefs_path = tmp_path / "user_prefs.json"
    prefs_path.write_text(json.dumps({"quality_data_sharing_enabled": True}), encoding="utf-8")
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_ENABLED", "1")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_URL", "https://collector.example/api/user-issues/ingest")
    monkeypatch.setattr(UserIssueStore, "flush_remote_outbox_async", lambda self, limit=20: None)
    db_path = tmp_path / "user_issues.sqlite3"
    store = UserIssueStore(db_path)
    store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Render failed",
        fingerprint="render-failed",
    )
    assert store.remote_outbox_summary()["pending"] == 1

    stored: dict[str, object] = {"quality_data_sharing_enabled": True}

    def load_prefs() -> dict[str, object]:
        return dict(stored)

    def save_prefs(data: dict[str, object]) -> None:
        stored.clear()
        stored.update(data)

    def get_settings():
        return SimpleNamespace(user_issues_db_path=db_path)

    get_settings.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(settings_router, "load_prefs", load_prefs)
    monkeypatch.setattr(settings_router, "save_prefs", save_prefs)
    monkeypatch.setattr(settings_router, "get_settings", get_settings)

    disabled = settings_router.update_settings(settings_router.PrefsPatch(quality_data_sharing_enabled=False))
    assert disabled["ok"] is True
    assert disabled["quality_data_cleanup"]["ok"] is True
    assert disabled["quality_data_cleanup"]["removed"] == 1
    assert stored["quality_data_sharing_enabled"] is False
    assert UserIssueStore(db_path).remote_outbox_summary()["pending"] == 0


def test_update_settings_persists_quality_data_opt_out_when_cleanup_fails(monkeypatch):
    stored: dict[str, object] = {
        "quality_data_sharing_enabled": True,
        "quality_data_client_id": "existing-client",
    }

    def load_prefs() -> dict[str, object]:
        return dict(stored)

    def save_prefs(data: dict[str, object]) -> None:
        stored.clear()
        stored.update(data)

    monkeypatch.setattr(settings_router, "load_prefs", load_prefs)
    monkeypatch.setattr(settings_router, "save_prefs", save_prefs)
    monkeypatch.setattr(
        settings_router,
        "_discard_unsent_quality_data_outbox",
        lambda: {"ok": False, "removed": 0, "error": "database locked"},
    )

    disabled = settings_router.update_settings(settings_router.PrefsPatch(quality_data_sharing_enabled=False))

    assert disabled["ok"] is True
    assert disabled["quality_data_cleanup"] == {
        "ok": False,
        "removed": 0,
        "error": "database locked",
    }
    assert stored["quality_data_sharing_enabled"] is False
    assert "quality_data_client_id" not in stored


def test_update_settings_discards_stale_quality_data_when_enabled_from_off(monkeypatch, tmp_path):
    prefs_path = tmp_path / "user_prefs.json"
    prefs_path.write_text(json.dumps({"quality_data_sharing_enabled": True}), encoding="utf-8")
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_ENABLED", "1")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_URL", "https://collector.example/api/user-issues/ingest")
    monkeypatch.setattr(UserIssueStore, "flush_remote_outbox_async", lambda self, limit=20: None)
    monkeypatch.setattr(settings_router.secrets, "token_urlsafe", lambda _n=18: "new-consent-client")
    db_path = tmp_path / "user_issues.sqlite3"
    store = UserIssueStore(db_path)
    store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Render failed before current consent",
        fingerprint="stale-render-failed",
    )
    assert store.remote_outbox_summary()["pending"] == 1

    stored: dict[str, object] = {"quality_data_sharing_enabled": False}

    def load_prefs() -> dict[str, object]:
        return dict(stored)

    def save_prefs(data: dict[str, object]) -> None:
        stored.clear()
        stored.update(data)

    def get_settings():
        return SimpleNamespace(user_issues_db_path=db_path)

    get_settings.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(settings_router, "load_prefs", load_prefs)
    monkeypatch.setattr(settings_router, "save_prefs", save_prefs)
    monkeypatch.setattr(settings_router, "get_settings", get_settings)

    assert settings_router.update_settings(settings_router.PrefsPatch(quality_data_sharing_enabled=True)) == {"ok": True}
    assert stored["quality_data_sharing_enabled"] is True
    assert stored["quality_data_client_id"] == "new-consent-client"
    assert UserIssueStore(db_path).remote_outbox_summary()["pending"] == 0


def test_update_settings_refuses_quality_data_enable_when_stale_cleanup_fails(monkeypatch):
    stored: dict[str, object] = {"quality_data_sharing_enabled": False}
    saved: list[dict[str, object]] = []

    def load_prefs() -> dict[str, object]:
        return dict(stored)

    def save_prefs(data: dict[str, object]) -> None:
        saved.append(dict(data))
        stored.clear()
        stored.update(data)

    monkeypatch.setattr(settings_router, "load_prefs", load_prefs)
    monkeypatch.setattr(settings_router, "save_prefs", save_prefs)
    monkeypatch.setattr(
        settings_router,
        "_discard_unsent_quality_data_outbox",
        lambda: {"ok": False, "removed": 0, "error": "database locked"},
    )

    with pytest.raises(HTTPException) as exc:
        settings_router.update_settings(settings_router.PrefsPatch(quality_data_sharing_enabled=True))

    assert exc.value.status_code == 500
    assert "database locked" in str(exc.value.detail)
    assert stored == {"quality_data_sharing_enabled": False}
    assert saved == []


def test_update_settings_rejects_invalid_choice_preferences(monkeypatch):
    stored: dict[str, object] = {}

    def load_prefs() -> dict[str, object]:
        return dict(stored)

    def save_prefs(data: dict[str, object]) -> None:
        stored.clear()
        stored.update(data)

    monkeypatch.setattr(settings_router, "load_prefs", load_prefs)
    monkeypatch.setattr(settings_router, "save_prefs", save_prefs)

    with pytest.raises(HTTPException) as exc:
        settings_router.update_settings(settings_router.PrefsPatch(theme="sepia"))

    assert exc.value.status_code == 400
    assert "theme must be one of" in str(exc.value.detail)
    assert stored == {}


def test_update_settings_rejects_out_of_range_numeric_preferences():
    invalid_values = [
        {"top_k": 1},
        {"top_k": 21},
        {"temperature": -0.1},
        {"temperature": 1.1},
        {"max_tokens": 511},
        {"max_tokens": 3073},
    ]

    for patch in invalid_values:
        with pytest.raises(ValidationError):
            settings_router.PrefsPatch(**patch)


def test_update_settings_rejects_invalid_model_base_urls_before_persisting(monkeypatch):
    stored: dict[str, object] = {}
    saved: list[dict[str, object]] = []

    def load_prefs() -> dict[str, object]:
        return dict(stored)

    def save_prefs(data: dict[str, object]) -> None:
        saved.append(dict(data))
        stored.clear()
        stored.update(data)

    monkeypatch.setattr(settings_router, "load_prefs", load_prefs)
    monkeypatch.setattr(settings_router, "save_prefs", save_prefs)

    for base_url in (
        "file:///tmp/model-proxy",
        "https://user:pass@proxy.example/v1",
        "https://proxy.example/v1?token=secret",
        "https://proxy.example/v1#fragment",
    ):
        with pytest.raises(HTTPException) as exc:
            settings_router.update_settings(settings_router.PrefsPatch(text_base_url=base_url))
        assert exc.value.status_code == 400

    assert stored == {}
    assert saved == []


def test_settings_models_reject_overlong_sensitive_and_path_values():
    with pytest.raises(ValidationError):
        settings_router.PrefsPatch(text_api_key="k" * 4097)
    with pytest.raises(ValidationError):
        settings_router.PrefsPatch(pdf_dir="p" * 1201)
    with pytest.raises(ValidationError):
        settings_router.ConnectionTestBody(api_key="k" * 4097)
    with pytest.raises(ValidationError):
        settings_router.PickDirRequest(target="pdf", initial_dir="p" * 1201)


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
    monkeypatch.setattr(
        settings_router,
        "load_prefs",
        lambda: {
            "text_api_key": "sk-text",
            "text_base_url": "https://stored-text.example/v1",
            "theme": "light",
            "max_tokens": 2048,
            "quality_data_sharing_enabled": True,
            "quality_data_client_id": "anon-client-token",
            "remote_token": "remote-token-should-not-leak",
            "client_secret": "client-secret-should-not-leak",
            "future_display_pref": "not-yet-public",
            "nested": {"token": "nested-secret-should-not-leak"},
        },
    )

    payload = settings_router.get_all_settings()

    assert payload["has_api_key"] is True
    assert payload["connection"]["text"]["model"] == "text-model"
    assert payload["connection"]["vision"]["model"] == "vision-model"
    assert payload["connection"]["vision"]["uses_text_fallback"] is False
    assert payload["connection"]["auto_route"] is True
    assert payload["readiness"]["overall"] == {
        "status": "warning",
        "reason": "configured_not_tested",
        "target": "text",
    }
    assert payload["readiness"]["providers"]["text"]["status"] == "configured"
    assert payload["prefs"] == {
        "theme": "light",
        "max_tokens": 2048,
        "quality_data_sharing_enabled": True,
    }
    assert "anon-client-token" not in str(payload)
    assert "remote-token-should-not-leak" not in str(payload)
    assert "client-secret-should-not-leak" not in str(payload)
    assert "nested-secret-should-not-leak" not in str(payload)


def test_public_prefs_allow_only_ui_preferences_and_drop_sensitive_unknowns() -> None:
    payload = settings_router._public_prefs(
        {
            "theme": "dark",
            "top_k": 8,
            "text_model": "saved-model",
            "quality_data_client_id": "anon-client-token",
            "github_token": "ghp_secretsecretsecret",
            "client_secret": "client-secret",
            "future_display_pref": "not-yet-public",
            "nested": {"theme": "light"},
        }
    )

    assert payload == {"theme": "dark", "top_k": 8}


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


def test_llm_connection_test_redacts_sensitive_error_text(monkeypatch):
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

    def fake_test_chat_completion(**kwargs):
        return {
            "ok": False,
            "error": (
                "401 unauthorized for sk-secretsecretsecret at "
                "https://proxy.example/v1?token=private and Bearer abcdefghijklmnop"
            ),
        }

    monkeypatch.setattr(settings_router, "get_settings", lambda: settings)
    monkeypatch.setattr(settings_router, "_test_chat_completion", fake_test_chat_completion)

    result = settings_router.test_llm(settings_router.ConnectionTestBody(target="text"))
    payload = settings_router.get_llm_readiness()

    assert result["ok"] is False
    assert result["error_type"] == "auth"
    assert "sk-secret" not in result["error"]
    assert "private" not in result["error"]
    assert "abcdefghijklmnop" not in result["error"]
    assert "[token]" in result["error"]
    assert "Bearer [redacted]" in result["error"]
    assert "https://proxy.example/v1" in result["error"]
    assert payload["providers"]["text"]["last_test"]["error"] == result["error"]


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


def test_llm_readiness_warns_when_configured_providers_have_not_been_tested(monkeypatch):
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
    )

    monkeypatch.setattr(settings_router, "get_settings", lambda: settings)

    payload = settings_router.get_llm_readiness()

    assert payload["overall"] == {
        "status": "warning",
        "reason": "configured_not_tested",
        "target": "text",
    }
    assert payload["providers"]["text"]["severity"] == "warning"
    assert payload["providers"]["vision"]["severity"] == "warning"


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


@pytest.mark.parametrize(
    ("remote_url", "detail_fragment"),
    [
        ("http://collector.example/api/user-issues/ingest", "must use HTTPS"),
        ("https://user:pass@collector.example/api/user-issues/ingest", "must not include embedded"),
        ("https://127.0.0.1/api/user-issues/ingest", "local/private host"),
        ("https://collector.example:bad/api/user-issues/ingest", "valid http(s) URL"),
    ],
)
def test_app_readiness_blocks_unsafe_remote_quality_telemetry_in_production(
    monkeypatch,
    tmp_path,
    remote_url,
    detail_fragment,
):
    monkeypatch.setenv("KB_USER_ISSUES_ALLOW_LOCAL_REMOTE", "1")
    monkeypatch.setattr(settings_router, "latest_restore_review_state", lambda: {})
    settings = _app_readiness_settings(
        tmp_path,
        production=True,
        remote_enabled=True,
        remote_url=remote_url,
        remote_token="collect-secret",
    )

    payload = settings_router.production_readiness_payload(settings)
    item = next(item for item in payload["items"] if item["key"] == "user_issues_remote")

    assert item["severity"] == "error"
    assert item["action"] == "fix_user_issues_remote_url"
    assert detail_fragment in item["detail"]


def test_app_readiness_allows_explicit_local_quality_telemetry_for_nonproduction_tests(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_USER_ISSUES_ALLOW_LOCAL_REMOTE", "1")
    monkeypatch.setattr(settings_router, "latest_restore_review_state", lambda: {})
    settings = _app_readiness_settings(
        tmp_path,
        production=False,
        remote_enabled=True,
        remote_url="http://127.0.0.1:9009/api/user-issues/ingest",
        remote_token="collect-secret",
    )

    payload = settings_router.production_readiness_payload(settings)
    item = next(item for item in payload["items"] if item["key"] == "user_issues_remote")

    assert item["severity"] == "ok"
    assert item["detail"] == "Enabled"


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


def test_load_settings_keeps_public_auth_off_by_default_in_production(monkeypatch):
    monkeypatch.setattr(config_module, "_load_runtime_prefs", lambda: {})
    monkeypatch.setenv("KB_ENV", "production")
    monkeypatch.setenv("KB_APP_ENV", "")
    monkeypatch.setenv("KB_PRIVATE_INSTANCE_AUTH", "")
    monkeypatch.setenv("KB_ENABLE_AUTH_GATE", "")
    monkeypatch.setenv("KB_REQUIRE_AUTH", "")
    for key in (
        "KB_ACCESS_TOKEN",
        "KB_API_TOKEN",
        "KB_AUTH_TOKEN",
        "KB_ACCESS_TOKEN_SHA256",
        "KB_API_TOKEN_SHA256",
        "KB_AUTH_TOKEN_SHA256",
    ):
        monkeypatch.setenv(key, "")

    settings = config_module.load_settings()

    assert settings.production is True
    assert settings.auth_required is False


def test_load_settings_does_not_make_token_an_implicit_login_gate(monkeypatch):
    monkeypatch.setattr(config_module, "_load_runtime_prefs", lambda: {})
    monkeypatch.setenv("KB_ENV", "production")
    monkeypatch.setenv("KB_APP_ENV", "")
    monkeypatch.setenv("KB_PRIVATE_INSTANCE_AUTH", "")
    monkeypatch.setenv("KB_ENABLE_AUTH_GATE", "")
    monkeypatch.setenv("KB_REQUIRE_AUTH", "")
    monkeypatch.setenv("KB_ACCESS_TOKEN", "configured-but-not-required")
    monkeypatch.setenv("KB_API_TOKEN", "")
    monkeypatch.setenv("KB_AUTH_TOKEN", "")
    monkeypatch.setenv("KB_ACCESS_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_API_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_AUTH_TOKEN_SHA256", "")

    settings = config_module.load_settings()

    assert settings.access_token == "configured-but-not-required"
    assert settings.auth_required is False


def test_load_settings_keeps_auth_off_when_auth_gate_is_not_enabled(monkeypatch):
    monkeypatch.setattr(config_module, "_load_runtime_prefs", lambda: {})
    monkeypatch.setenv("KB_ENV", "production")
    monkeypatch.setenv("KB_APP_ENV", "")
    monkeypatch.setenv("KB_PRIVATE_INSTANCE_AUTH", "1")
    monkeypatch.setenv("KB_ENABLE_AUTH_GATE", "")
    monkeypatch.setenv("KB_REQUIRE_AUTH", "1")
    monkeypatch.setenv("KB_ACCESS_TOKEN", "private-token")
    monkeypatch.setenv("KB_API_TOKEN", "")
    monkeypatch.setenv("KB_AUTH_TOKEN", "")
    monkeypatch.setenv("KB_ACCESS_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_API_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_AUTH_TOKEN_SHA256", "")

    settings = config_module.load_settings()

    assert settings.auth_required is False
    assert settings.access_token == "private-token"


def test_load_settings_keeps_auth_off_without_private_instance_marker(monkeypatch):
    monkeypatch.setattr(config_module, "_load_runtime_prefs", lambda: {})
    monkeypatch.setenv("KB_ENV", "production")
    monkeypatch.setenv("KB_APP_ENV", "")
    monkeypatch.setenv("KB_PRIVATE_INSTANCE_AUTH", "")
    monkeypatch.setenv("KB_ENABLE_AUTH_GATE", "1")
    monkeypatch.setenv("KB_REQUIRE_AUTH", "1")
    monkeypatch.setenv("KB_ACCESS_TOKEN", "private-token")
    monkeypatch.setenv("KB_API_TOKEN", "")
    monkeypatch.setenv("KB_AUTH_TOKEN", "")
    monkeypatch.setenv("KB_ACCESS_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_API_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_AUTH_TOKEN_SHA256", "")

    settings = config_module.load_settings()

    assert settings.auth_required is False
    assert settings.access_token == "private-token"


def test_load_settings_enables_auth_only_when_explicitly_required(monkeypatch):
    monkeypatch.setattr(config_module, "_load_runtime_prefs", lambda: {})
    monkeypatch.setenv("KB_ENV", "production")
    monkeypatch.setenv("KB_APP_ENV", "")
    monkeypatch.setenv("KB_PRIVATE_INSTANCE_AUTH", "1")
    monkeypatch.setenv("KB_ENABLE_AUTH_GATE", "1")
    monkeypatch.setenv("KB_REQUIRE_AUTH", "1")
    monkeypatch.setenv("KB_ACCESS_TOKEN", "private-token")
    monkeypatch.setenv("KB_API_TOKEN", "")
    monkeypatch.setenv("KB_AUTH_TOKEN", "")
    monkeypatch.setenv("KB_ACCESS_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_API_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_AUTH_TOKEN_SHA256", "")

    settings = config_module.load_settings()

    assert settings.auth_required is True
    assert settings.access_token == "private-token"


def test_load_settings_ignores_auth_gate_flags_in_local_user_mode(monkeypatch):
    monkeypatch.setattr(config_module, "_load_runtime_prefs", lambda: {})
    monkeypatch.setenv("KB_ENV", "development")
    monkeypatch.setenv("KB_APP_ENV", "")
    monkeypatch.setenv("KB_PRIVATE_INSTANCE_AUTH", "1")
    monkeypatch.setenv("KB_ENABLE_AUTH_GATE", "1")
    monkeypatch.setenv("KB_REQUIRE_AUTH", "1")
    monkeypatch.setenv("KB_ALLOW_LOCAL_AUTH_GATE", "")
    monkeypatch.setenv("KB_ACCESS_TOKEN", "local-token")
    monkeypatch.setenv("KB_API_TOKEN", "")
    monkeypatch.setenv("KB_AUTH_TOKEN", "")
    monkeypatch.setenv("KB_ACCESS_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_API_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_AUTH_TOKEN_SHA256", "")

    settings = config_module.load_settings()

    assert settings.production is False
    assert settings.auth_required is False
    assert settings.access_token == "local-token"


def test_load_settings_allows_explicit_local_auth_gate_testing(monkeypatch):
    monkeypatch.setattr(config_module, "_load_runtime_prefs", lambda: {})
    monkeypatch.setenv("KB_ENV", "development")
    monkeypatch.setenv("KB_APP_ENV", "")
    monkeypatch.setenv("KB_PRIVATE_INSTANCE_AUTH", "1")
    monkeypatch.setenv("KB_ENABLE_AUTH_GATE", "1")
    monkeypatch.setenv("KB_REQUIRE_AUTH", "1")
    monkeypatch.setenv("KB_ALLOW_LOCAL_AUTH_GATE", "1")
    monkeypatch.setenv("KB_ACCESS_TOKEN", "local-private-token")
    monkeypatch.setenv("KB_API_TOKEN", "")
    monkeypatch.setenv("KB_AUTH_TOKEN", "")
    monkeypatch.setenv("KB_ACCESS_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_API_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_AUTH_TOKEN_SHA256", "")

    settings = config_module.load_settings()

    assert settings.production is False
    assert settings.auth_required is True
    assert settings.access_token == "local-private-token"


def test_load_settings_reads_configured_user_prefs_path(monkeypatch, tmp_path):
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
    prefs_path = tmp_path / "runtime" / "prefs.json"
    prefs_path.parent.mkdir(parents=True)
    prefs_path.write_text(
        json.dumps(
            {
                "text_api_key": "sk-text",
                "text_base_url": "https://text.example/v1/",
                "text_model": "text-model",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))

    settings = config_module.load_settings()

    assert settings.text_api_key == "sk-text"
    assert settings.text_base_url == "https://text.example/v1"
    assert settings.text_model == "text-model"
