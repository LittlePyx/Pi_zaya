from __future__ import annotations

from api.routers import settings as settings_router


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
