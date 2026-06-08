from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path
from types import SimpleNamespace

from kb import maintenance


def _sqlite_with_row(path: Path, table: str = "items") -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(f"create table {table}(id integer primary key, value text)")
        conn.execute(f"insert into {table}(value) values (?)", ("hello",))
        conn.commit()
    finally:
        conn.close()


def _settings(tmp_path: Path) -> SimpleNamespace:
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    (db_dir / "docs.json").write_text('{"docs":[]}', encoding="utf-8")
    chat_db = tmp_path / "chat.sqlite3"
    library_db = tmp_path / "library.sqlite3"
    _sqlite_with_row(chat_db, "messages")
    _sqlite_with_row(library_db, "papers")
    return SimpleNamespace(
        app_env="production",
        production=True,
        auth_required=True,
        access_token="secret-access-token",
        access_token_sha256=None,
        text_api_key="sk-secret-text",
        text_base_url="https://text.example/v1",
        text_model="text-model",
        vision_api_key="sk-secret-vision",
        vision_base_url="https://vision.example/v1",
        vision_model="vision-model",
        vision_uses_text_fallback=False,
        db_dir=db_dir,
        chat_db_path=chat_db,
        library_db_path=library_db,
        auto_backup_enabled=None,
    )


def test_diagnostics_archive_contains_only_redacted_summaries(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(maintenance, "ROOT", tmp_path)
    monkeypatch.setenv("KB_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))
    settings = _settings(tmp_path)
    (tmp_path / "server.log").write_text("Authorization: Bearer sk-secret-text\nnormal line", encoding="utf-8")

    archive = maintenance.create_diagnostics_archive(settings, readiness_payload={"status": "error"})

    raw = archive.read_bytes()
    assert b"sk-secret-text" not in raw
    assert b"secret-access-token" not in raw
    with zipfile.ZipFile(archive, "r") as zf:
        names = set(zf.namelist())
        assert "diagnostics.json" in names
        assert "logs/server.log.tail.txt" in names
        payload = json.loads(zf.read("diagnostics.json").decode("utf-8"))
        assert payload["readiness"]["status"] == "error"
        assert payload["sqlite"]["chat_db"]["tables"]["messages"] == 1
        assert payload["config"]["model"]["has_text_key"] is True
        assert "<redacted>" in zf.read("logs/server.log.tail.txt").decode("utf-8")
        assert "chat.sqlite3" not in names


def test_backup_archive_includes_recoverable_data_but_redacts_prefs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(maintenance, "ROOT", tmp_path)
    monkeypatch.setenv("KB_BACKUP_DIR", str(tmp_path / "backups"))
    settings = _settings(tmp_path)
    (tmp_path / "user_prefs.json").write_text(
        json.dumps({"theme": "light", "text_api_key": "sk-secret-text"}),
        encoding="utf-8",
    )

    info = maintenance.create_backup_archive(settings, label="manual test")

    archive = Path(info["path"])
    assert info["name"].startswith("backup-")
    with zipfile.ZipFile(archive, "r") as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "chat.sqlite3" in names
        assert "library.sqlite3" in names
        assert "db/docs.json" in names
        prefs = json.loads(zf.read("user_prefs.redacted.json").decode("utf-8"))
        assert prefs == {"theme": "light", "text_api_key": "<redacted>"}
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest["label"] == "manual test"

    backup_db = tmp_path / "restored.sqlite3"
    with zipfile.ZipFile(archive, "r") as zf:
        backup_db.write_bytes(zf.read("chat.sqlite3"))
    conn = sqlite3.connect(backup_db)
    try:
        assert conn.execute("select count(*) from messages").fetchone()[0] == 1
    finally:
        conn.close()


def test_backup_listing_and_resolution(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(maintenance, "ROOT", tmp_path)
    monkeypatch.setenv("KB_BACKUP_DIR", str(tmp_path / "backups"))
    settings = _settings(tmp_path)

    info = maintenance.create_backup_archive(settings)
    items = maintenance.list_backup_archives()

    assert [item["name"] for item in items] == [info["name"]]
    assert maintenance.resolve_backup_archive(info["name"]).name == info["name"]


def test_auto_snapshot_defaults_to_production_and_rate_limits(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(maintenance, "ROOT", tmp_path)
    monkeypatch.setenv("KB_BACKUP_DIR", str(tmp_path / "backups"))
    monkeypatch.delenv("KB_AUTO_BACKUP", raising=False)
    monkeypatch.setenv("KB_AUTO_BACKUP_MIN_INTERVAL_S", "999")
    maintenance._AUTO_BACKUP_LAST.clear()
    settings = _settings(tmp_path)

    dev_settings = SimpleNamespace(**settings.__dict__)
    dev_settings.production = False
    disabled = maintenance.create_auto_snapshot(dev_settings, action="library delete")
    assert disabled["enabled"] is False
    assert disabled["reason"] == "disabled"

    first = maintenance.create_auto_snapshot(settings, action="library delete", label="paper")
    second = maintenance.create_auto_snapshot(settings, action="library delete", label="paper")

    assert first["enabled"] is True
    assert first["created"] is True
    assert first["backup"]["name"].startswith("backup-")
    assert second["created"] is False
    assert second["reason"] == "rate_limited"


def test_auto_snapshot_uses_user_preference_when_env_unset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(maintenance, "ROOT", tmp_path)
    monkeypatch.setenv("KB_BACKUP_DIR", str(tmp_path / "backups"))
    monkeypatch.delenv("KB_AUTO_BACKUP", raising=False)
    monkeypatch.setenv("KB_AUTO_BACKUP_MIN_INTERVAL_S", "0")
    maintenance._AUTO_BACKUP_LAST.clear()
    settings = _settings(tmp_path)
    settings.production = False
    settings.auto_backup_enabled = True

    snapshot = maintenance.create_auto_snapshot(settings, action="library delete")
    status = maintenance.maintenance_status(settings)

    assert snapshot["enabled"] is True
    assert snapshot["created"] is True
    assert snapshot["source"] == "user"
    assert status["data_protection"]["enabled"] is True
    assert status["data_protection"]["can_toggle"] is True
    assert status["auto_backup"]["source"] == "user"
    assert status["auto_backup"]["locked"] is False


def test_maintenance_status_summarizes_data_protection(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(maintenance, "ROOT", tmp_path)
    monkeypatch.setenv("KB_BACKUP_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("KB_AUTO_BACKUP", "1")
    monkeypatch.setenv("KB_AUTO_BACKUP_STRICT", "1")
    monkeypatch.setenv("KB_AUTO_BACKUP_MIN_INTERVAL_S", "12.5")
    settings = _settings(tmp_path)
    info = maintenance.create_backup_archive(settings, label="status")

    status = maintenance.maintenance_status(settings)

    assert status["data_protection"]["enabled"] is True
    assert status["data_protection"]["can_toggle"] is False
    assert status["data_protection"]["backup_count"] == 1
    assert status["data_protection"]["latest_backup"]["name"] == info["name"]
    assert status["auto_backup"] == {
        "enabled": True,
        "strict": True,
        "min_interval_s": 12.5,
        "source": "env",
        "locked": True,
    }
    assert status["backups"]["count"] == 1
    assert status["backups"]["keep"] == 30


def test_auto_snapshot_strict_failure_blocks_operation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(maintenance, "ROOT", tmp_path)
    monkeypatch.setenv("KB_AUTO_BACKUP", "1")
    monkeypatch.setenv("KB_AUTO_BACKUP_STRICT", "1")
    monkeypatch.setenv("KB_AUTO_BACKUP_MIN_INTERVAL_S", "0")
    maintenance._AUTO_BACKUP_LAST.clear()
    settings = _settings(tmp_path)

    def fail_backup(*args, **kwargs):
        raise RuntimeError("backup disk full")

    monkeypatch.setattr(maintenance, "create_backup_archive", fail_backup)
    snapshot = maintenance.create_auto_snapshot(settings, action="library delete")

    assert snapshot["created"] is False
    assert snapshot["reason"] == "failed"
    assert snapshot["block_operation"] is True
    assert "backup disk full" in snapshot["error"]


def test_backup_verification_checks_zip_manifest_and_sqlite(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(maintenance, "ROOT", tmp_path)
    monkeypatch.setenv("KB_BACKUP_DIR", str(tmp_path / "backups"))
    settings = _settings(tmp_path)

    info = maintenance.create_backup_archive(settings, label="verify")
    result = maintenance.verify_backup_archive(Path(info["path"]))

    assert result["ok"] is True
    assert result["checks"]["zip"]["ok"] is True
    assert result["checks"]["required_files"]["chat.sqlite3"]["present"] is True
    assert result["checks"]["sqlite"]["chat.sqlite3"]["ok"] is True
    assert result["checks"]["sqlite"]["library.sqlite3"]["ok"] is True


def test_backup_cleanup_keeps_newest_archives(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(maintenance, "ROOT", tmp_path)
    monkeypatch.setenv("KB_BACKUP_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("KB_BACKUP_KEEP_N", "2")
    settings = _settings(tmp_path)

    first = maintenance.create_backup_archive(settings, label="first")
    second = maintenance.create_backup_archive(settings, label="second")
    third = maintenance.create_backup_archive(settings, label="third")

    result = maintenance.cleanup_backup_archives()
    names = [item["name"] for item in maintenance.list_backup_archives()]

    assert result["ok"] is True
    assert result["keep"] == 2
    assert result["deleted"] == 1
    assert third["name"] in names
    assert second["name"] in names
    assert first["name"] not in names


def test_restore_dry_run_reports_targets_without_overwriting(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(maintenance, "ROOT", tmp_path)
    monkeypatch.setenv("KB_BACKUP_DIR", str(tmp_path / "backups"))
    settings = _settings(tmp_path)
    info = maintenance.create_backup_archive(settings, label="restore dry run")
    original_chat_size = Path(settings.chat_db_path).stat().st_size

    report = maintenance.restore_dry_run_backup_archive(settings, Path(info["path"]))

    assert report["ok"] is True
    assert report["can_restore"] is True
    assert report["extracted_file_count"] >= 3
    assert Path(settings.chat_db_path).stat().st_size == original_chat_size
    destinations = {item["archive"]: item for item in report["destinations"]}
    assert destinations["chat.sqlite3"]["target"] == str(settings.chat_db_path)
    assert destinations["library.sqlite3"]["source_exists"] is True
    assert destinations["db/"]["source_file_count"] >= 1
    assert report["sqlite"]["chat.sqlite3"]["tables"]["messages"] == 1


def test_restore_backup_archive_restores_files_with_pre_restore_snapshot(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(maintenance, "ROOT", tmp_path)
    monkeypatch.setenv("KB_BACKUP_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("KB_RESTORE_AUDIT_PATH", str(tmp_path / "restore_audit.jsonl"))
    settings = _settings(tmp_path)
    info = maintenance.create_backup_archive(settings, label="restore-source")

    conn = sqlite3.connect(settings.chat_db_path)
    try:
        conn.execute("insert into messages(value) values (?)", ("new row",))
        conn.commit()
    finally:
        conn.close()
    (Path(settings.db_dir) / "docs.json").write_text('{"docs":["mutated"]}', encoding="utf-8")

    result = maintenance.restore_backup_archive(
        settings,
        Path(info["path"]),
        confirm=f"RESTORE {info['name']}",
    )

    assert result["ok"] is True
    assert result["restart_required"] is True
    assert result["pre_restore_backup"]["name"].startswith("backup-")
    assert len(result["restored"]) == 3
    conn = sqlite3.connect(settings.chat_db_path)
    try:
        assert conn.execute("select count(*) from messages").fetchone()[0] == 1
    finally:
        conn.close()
    assert (Path(settings.db_dir) / "docs.json").read_text(encoding="utf-8") == '{"docs":[]}'
    audit = (tmp_path / "restore_audit.jsonl").read_text(encoding="utf-8")
    assert '"status": "restored"' in audit
    audit_event = json.loads(audit.strip().splitlines()[-1])
    assert audit_event["restart_required"] is True


def test_restore_audit_events_return_latest_valid_event(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KB_RESTORE_AUDIT_PATH", str(tmp_path / "restore_audit.jsonl"))
    (tmp_path / "restore_audit.jsonl").write_text(
        'not-json\n'
        '{"event":"restore","status":"restored","backup":"backup-old.zip","created_at":1}\n'
        '{"event":"restore","status":"failed","backup":"backup-new.zip","created_at":2}\n',
        encoding="utf-8",
    )

    events = maintenance.list_restore_audit_events(limit=1)

    assert len(events) == 1
    assert events[0]["backup"] == "backup-new.zip"
    assert maintenance.latest_restore_audit_event()["status"] == "failed"


def test_restore_review_acknowledgement_matches_latest_restore(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(maintenance, "ROOT", tmp_path)
    monkeypatch.setenv("KB_BACKUP_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("KB_RESTORE_AUDIT_PATH", str(tmp_path / "restore_audit.jsonl"))
    settings = _settings(tmp_path)
    info = maintenance.create_backup_archive(settings, label="restore-source")
    restore = maintenance.restore_backup_archive(
        settings,
        Path(info["path"]),
        confirm=f"RESTORE {info['name']}",
        create_pre_restore_backup=False,
    )
    assert restore["ok"] is True

    acknowledged = maintenance.acknowledge_latest_restore_review(checks={"api_restarted": True})
    state = maintenance.latest_restore_review_state()

    assert acknowledged["ok"] is True
    assert acknowledged["status"] == "acknowledged"
    assert state["acknowledged"] is True
    assert state["restore"]["backup"] == info["name"]
    assert state["acknowledgement"]["backup"] == info["name"]
    assert maintenance.latest_restore_audit_event()["event"] == "restore_review_acknowledged"
    assert maintenance.latest_restore_operation_event()["status"] == "restored"
    public_events = maintenance.public_restore_audit_events(limit=5)
    assert public_events[0]["event"] == "restore_review_acknowledged"
    assert public_events[1]["event"] == "restore"
    assert public_events[1]["restored_count"] == 3
    assert "restored" not in public_events[1]
    assert "pre_restore_backup" in public_events[1]

    again = maintenance.acknowledge_latest_restore_review()
    assert again["ok"] is True
    assert again["status"] == "already_acknowledged"


def test_restore_backup_archive_requires_exact_confirmation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(maintenance, "ROOT", tmp_path)
    monkeypatch.setenv("KB_BACKUP_DIR", str(tmp_path / "backups"))
    settings = _settings(tmp_path)
    info = maintenance.create_backup_archive(settings)

    result = maintenance.restore_backup_archive(settings, Path(info["path"]), confirm="restore")

    assert result["ok"] is False
    assert result["status"] == "confirmation_failed"
    assert "confirmation text mismatch" in result["errors"]
