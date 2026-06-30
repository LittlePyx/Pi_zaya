from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api import deps
from api.main import app
from api.routers import maintenance as maintenance_router


def _clear_settings_cache() -> None:
    try:
        deps.get_settings.cache_clear()
    except Exception:
        pass


def _sqlite_with_row(path: Path, table: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(f"create table {table}(id integer primary key)")
        conn.execute(f"insert into {table} default values")
        conn.commit()
    finally:
        conn.close()


def _settings(tmp_path: Path) -> SimpleNamespace:
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    (db_dir / "docs.json").write_text("{}", encoding="utf-8")
    chat_db = tmp_path / "chat.sqlite3"
    library_db = tmp_path / "library.sqlite3"
    _sqlite_with_row(chat_db, "messages")
    _sqlite_with_row(library_db, "papers")
    return SimpleNamespace(
        app_env="development",
        production=False,
        auth_required=False,
        access_token=None,
        access_token_sha256=None,
        text_api_key="sk-test",
        text_base_url="https://text.example/v1",
        text_model="text-model",
        vision_api_key="sk-vision",
        vision_base_url="https://vision.example/v1",
        vision_model="vision-model",
        vision_uses_text_fallback=False,
        db_dir=db_dir,
        chat_db_path=chat_db,
        library_db_path=library_db,
        auto_backup_enabled=None,
    )


def test_maintenance_backup_and_diagnostics_routes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KB_REQUIRE_AUTH", "0")
    monkeypatch.setenv("KB_ENABLE_INTERNAL_API", "1")
    monkeypatch.setenv("KB_BACKUP_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("KB_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))
    monkeypatch.setenv("KB_RESTORE_AUDIT_PATH", str(tmp_path / "restore_audit.jsonl"))
    _clear_settings_cache()
    settings = _settings(tmp_path)
    monkeypatch.setattr(maintenance_router, "get_settings", lambda: settings)
    client = TestClient(app)

    initial_status = client.get("/api/maintenance/status")
    assert initial_status.status_code == 200
    initial_status_payload = initial_status.json()
    assert initial_status_payload["data_protection"]["enabled"] is False
    assert initial_status_payload["data_protection"]["can_toggle"] is True
    assert initial_status_payload["data_protection"]["backup_count"] == 0
    assert initial_status_payload["auto_backup"]["source"] == "development_default"
    assert initial_status_payload["auto_backup"]["locked"] is False

    created = client.post("/api/maintenance/backups", json={"label": "route smoke"})
    assert created.status_code == 200
    created_payload = created.json()
    assert created_payload["name"].startswith("backup-")

    status = client.get("/api/maintenance/status")
    assert status.status_code == 200
    status_payload = status.json()
    assert status_payload["data_protection"]["backup_count"] == 1
    assert status_payload["data_protection"]["latest_backup"]["name"] == created_payload["name"]

    listing = client.get("/api/maintenance/backups")
    assert listing.status_code == 200
    assert listing.json()["items"][0]["name"] == created_payload["name"]

    backup = client.get(f"/api/maintenance/backups/{created_payload['name']}")
    assert backup.status_code == 200
    assert backup.headers["content-type"].startswith("application/zip")
    backup_path = tmp_path / "backup.zip"
    backup_path.write_bytes(backup.content)
    with zipfile.ZipFile(backup_path, "r") as zf:
        assert "manifest.json" in zf.namelist()
        assert "chat.sqlite3" in zf.namelist()

    verified = client.get(f"/api/maintenance/backups/{created_payload['name']}/verify")
    assert verified.status_code == 200
    verify_payload = verified.json()
    assert verify_payload["ok"] is True
    assert verify_payload["checks"]["sqlite"]["chat.sqlite3"]["ok"] is True

    dry_run = client.get(f"/api/maintenance/backups/{created_payload['name']}/restore-dry-run")
    assert dry_run.status_code == 200
    dry_run_payload = dry_run.json()
    assert dry_run_payload["ok"] is True
    assert dry_run_payload["can_restore"] is True
    assert dry_run_payload["sqlite"]["chat.sqlite3"]["tables"]["messages"] == 1
    assert any(item["archive"] == "db/" for item in dry_run_payload["destinations"])

    bad_restore = client.post(
        f"/api/maintenance/backups/{created_payload['name']}/restore",
        json={"confirm": "restore"},
    )
    assert bad_restore.status_code == 400
    restore = client.post(
        f"/api/maintenance/backups/{created_payload['name']}/restore",
        json={
            "confirm": f"RESTORE {created_payload['name']}",
            "create_pre_restore_backup": False,
        },
    )
    assert restore.status_code == 200
    restore_payload = restore.json()
    assert restore_payload["ok"] is True
    assert restore_payload["restart_required"] is True

    restore_review = client.post(
        "/api/maintenance/restore-review/acknowledge",
        json={"checks": {"api_restarted": True}},
    )
    assert restore_review.status_code == 200
    restore_review_payload = restore_review.json()
    assert restore_review_payload["ok"] is True
    assert restore_review_payload["backup"] == created_payload["name"]
    audit = client.get("/api/maintenance/restore-audit")
    assert audit.status_code == 200
    audit_payload = audit.json()
    assert audit_payload["items"][0]["event"] == "restore_review_acknowledged"
    assert audit_payload["items"][1]["event"] == "restore"
    assert audit_payload["items"][1]["backup"] == created_payload["name"]
    assert "restored" not in audit_payload["items"][1]

    diagnostics = client.get("/api/maintenance/diagnostics/export")
    assert diagnostics.status_code == 200
    assert diagnostics.headers["content-type"].startswith("application/zip")
    diagnostics_path = tmp_path / "diagnostics.zip"
    diagnostics_path.write_bytes(diagnostics.content)
    with zipfile.ZipFile(diagnostics_path, "r") as zf:
        assert "diagnostics.json" in zf.namelist()
        assert "chat.sqlite3" not in zf.namelist()

    for label in ("cleanup-1", "cleanup-2"):
        response = client.post("/api/maintenance/backups", json={"label": label})
        assert response.status_code == 200
    cleanup = client.post("/api/maintenance/backups/cleanup", json={"keep": 1})
    assert cleanup.status_code == 200
    cleanup_payload = cleanup.json()
    assert cleanup_payload["ok"] is True
    assert cleanup_payload["keep"] == 1
    assert cleanup_payload["deleted"] >= 1
    kept = client.get("/api/maintenance/backups")
    assert kept.status_code == 200
    assert len(kept.json()["items"]) == 1
