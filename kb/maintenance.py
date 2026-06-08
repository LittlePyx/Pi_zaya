from __future__ import annotations

import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from kb import runtime_state
from kb.bg_queue_state import snapshot as bg_snapshot
from kb.config import Settings


ROOT = Path(__file__).resolve().parent.parent
SECRET_FIELD_NAMES = {
    "api_key",
    "access_token",
    "auth_token",
    "deepseek_api_key",
    "openai_api_key",
    "qwen_api_key",
    "text_api_key",
    "vision_api_key",
    "kb_access_token",
    "kb_access_token_sha256",
    "kb_api_token",
    "kb_auth_token",
}
SECRET_ENV_PATTERNS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
_AUTO_BACKUP_LOCK = threading.Lock()
_AUTO_BACKUP_LAST: dict[str, float] = {}
_RESTORE_AUDIT_LOCK = threading.Lock()


def _now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _archive_id(prefix: str, label: str = "") -> str:
    clean_label = re.sub(r"[^A-Za-z0-9._-]+", "-", str(label or "").strip()).strip("-._")
    suffix = f"-{clean_label[:44]}" if clean_label else ""
    return f"{prefix}-{_now_stamp()}-{uuid.uuid4().hex[:8]}{suffix}.zip"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _auto_backup_config(settings: Settings) -> dict[str, Any]:
    production_default = bool(getattr(settings, "production", False))
    raw_auto_backup = os.environ.get("KB_AUTO_BACKUP")
    if raw_auto_backup is not None:
        return {
            "enabled": _env_bool("KB_AUTO_BACKUP", production_default),
            "source": "env",
            "locked": True,
        }
    user_pref = getattr(settings, "auto_backup_enabled", None)
    if user_pref is not None:
        return {
            "enabled": bool(user_pref),
            "source": "user",
            "locked": False,
        }
    return {
        "enabled": production_default,
        "source": "production_default" if production_default else "development_default",
        "locked": False,
    }


def _clean_action(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-._") or "operation"


def backup_dir() -> Path:
    return Path(os.environ.get("KB_BACKUP_DIR") or (ROOT / ".runtime" / "backups")).expanduser().resolve()


def diagnostics_dir() -> Path:
    return Path(os.environ.get("KB_DIAGNOSTICS_DIR") or (ROOT / ".runtime" / "diagnostics")).expanduser().resolve()


def restore_audit_path() -> Path:
    return Path(os.environ.get("KB_RESTORE_AUDIT_PATH") or (ROOT / ".runtime" / "restore_audit.jsonl")).expanduser().resolve()


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default).encode("utf-8")


def _write_json(zf: zipfile.ZipFile, name: str, payload: object) -> None:
    zf.writestr(name, _json_bytes(payload))


def _redact_value(key: str, value: object) -> object:
    lower = str(key or "").lower()
    if lower in SECRET_FIELD_NAMES or any(part.lower() in lower for part in ("api_key", "access_token", "auth_token")):
        return "<redacted>" if str(value or "").strip() else ""
    return value


def redact_mapping(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in dict(data or {}).items():
        if isinstance(value, dict):
            out[key] = redact_mapping(value)
        else:
            out[key] = _redact_value(key, value)
    return out


_LOG_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|authorization|bearer)\b\s*[:= ]+\s*([A-Za-z0-9._:\-/+=]{8,})"
)
_TOKEN_RE = re.compile(r"\b(sk-[A-Za-z0-9._-]{8,}|Bearer\s+[A-Za-z0-9._\-=]{8,})\b")


def redact_text(text: str) -> str:
    redacted = _LOG_SECRET_RE.sub(lambda m: f"{m.group(1)}=<redacted>", str(text or ""))
    return _TOKEN_RE.sub("<redacted>", redacted)


def _path_stats(path: Path) -> dict[str, Any]:
    p = Path(path)
    exists = p.exists()
    if not exists:
        return {"path": str(p), "exists": False, "is_dir": False, "size_bytes": 0, "file_count": 0}
    if p.is_file():
        return {"path": str(p), "exists": True, "is_dir": False, "size_bytes": p.stat().st_size, "file_count": 1}
    file_count = 0
    size = 0
    suffix_counts: dict[str, int] = {}
    for item in p.rglob("*"):
        if not item.is_file():
            continue
        file_count += 1
        try:
            size += item.stat().st_size
        except OSError:
            pass
        suffix = item.suffix.lower() or "<none>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    return {
        "path": str(p),
        "exists": True,
        "is_dir": True,
        "size_bytes": size,
        "file_count": file_count,
        "suffix_counts": dict(sorted(suffix_counts.items())),
    }


def _sqlite_table_counts(path: Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False, "tables": {}}
    tables: dict[str, int] = {}
    try:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "select name from sqlite_master where type='table' and name not like 'sqlite_%' order by name"
            ).fetchall()
            for (name,) in rows:
                try:
                    tables[str(name)] = int(conn.execute(f'select count(*) from "{name}"').fetchone()[0])
                except sqlite3.Error:
                    tables[str(name)] = -1
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return {"path": str(p), "exists": True, "error": str(exc), "tables": {}}
    return {"path": str(p), "exists": True, "size_bytes": p.stat().st_size, "tables": tables}


def _git_summary() -> dict[str, str]:
    def run_git(args: list[str]) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=str(ROOT),
                text=True,
                capture_output=True,
                timeout=3,
                check=False,
            )
        except Exception:
            return ""
        return (result.stdout or "").strip()

    return {
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": run_git(["rev-parse", "--short", "HEAD"]),
        "dirty": "yes" if run_git(["status", "--short"]) else "no",
    }


def _env_summary(settings: Settings) -> dict[str, Any]:
    present = {
        key: bool(str(os.environ.get(key) or "").strip())
        for key in sorted(set(k for k in os.environ if k.startswith("KB_") or any(part in k for part in SECRET_ENV_PATTERNS)))
    }
    return {
        "app_env": getattr(settings, "app_env", "development"),
        "production": bool(getattr(settings, "production", False)),
        "auth_required": bool(getattr(settings, "auth_required", False)),
        "auth_token_configured": bool(getattr(settings, "access_token", None) or getattr(settings, "access_token_sha256", None)),
        "model": {
            "text_base_url": getattr(settings, "text_base_url", ""),
            "text_model": getattr(settings, "text_model", ""),
            "has_text_key": bool(getattr(settings, "text_api_key", None)),
            "vision_base_url": getattr(settings, "vision_base_url", ""),
            "vision_model": getattr(settings, "vision_model", ""),
            "has_vision_key": bool(getattr(settings, "vision_api_key", None)),
            "vision_uses_text_fallback": bool(getattr(settings, "vision_uses_text_fallback", False)),
        },
        "env_present": present,
    }


def _task_state_summary() -> dict[str, Any]:
    bg = bg_snapshot(runtime_state.BG_STATE, runtime_state.BG_LOCK)
    with runtime_state.GEN_LOCK:
        gen_count = len(runtime_state.GEN_TASKS)
        gen_status_counts: dict[str, int] = {}
        for task in runtime_state.GEN_TASKS.values():
            status = str((task or {}).get("status") or "unknown")
            gen_status_counts[status] = gen_status_counts.get(status, 0) + 1
    with runtime_state.CITATION_LOCK:
        citation_count = len(runtime_state.CITATION_TASKS)
    return {
        "conversion": {
            "running": bool(bg.get("running")),
            "active_count": int(bg.get("active_count") or 0),
            "queued_count": len(bg.get("queue") or []),
            "done": int(bg.get("done") or 0),
            "total": int(bg.get("total") or 0),
            "last": redact_text(str(bg.get("last") or ""))[:220],
        },
        "generation": {
            "count": gen_count,
            "status_counts": gen_status_counts,
        },
        "citation_tasks": {"count": citation_count},
    }


def _candidate_log_files(limit: int = 12) -> list[Path]:
    names = [
        "server.log",
        "api_server.log",
        "backend_stdout.log",
        "backend_stderr.log",
        "vite_server.log",
        "api_server.log",
    ]
    candidates: list[Path] = [ROOT / name for name in names]
    for folder in (ROOT / ".logs", ROOT / ".runtime", ROOT / "logs"):
        if folder.exists():
            candidates.extend(sorted(folder.glob("*.log"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True))
    seen: set[str] = set()
    out: list[Path] = []
    for path in candidates:
        key = str(path.resolve())
        if key in seen or not path.exists() or not path.is_file():
            continue
        seen.add(key)
        out.append(path)
        if len(out) >= limit:
            break
    return out


def _tail_text(path: Path, max_bytes: int = 64_000) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(max(0, size - max_bytes))
            raw = fh.read(max_bytes)
    except OSError:
        return ""
    return redact_text(raw.decode("utf-8", "replace"))


def _safe_user_prefs() -> dict[str, Any]:
    path = ROOT / "user_prefs.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return redact_mapping(data if isinstance(data, dict) else {})


def build_diagnostics_payload(settings: Settings, *, readiness_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "created_at": time.time(),
        "app": {
            "name": "Pi-zaya",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cwd": str(ROOT),
            "git": _git_summary(),
        },
        "config": _env_summary(settings),
        "readiness": readiness_payload or {},
        "paths": {
            "db_dir": _path_stats(Path(getattr(settings, "db_dir", ""))),
            "chat_db": _path_stats(Path(getattr(settings, "chat_db_path", ""))),
            "library_db": _path_stats(Path(getattr(settings, "library_db_path", ""))),
            "web_dist": _path_stats(ROOT / "web" / "dist"),
        },
        "sqlite": {
            "chat_db": _sqlite_table_counts(Path(getattr(settings, "chat_db_path", ""))),
            "library_db": _sqlite_table_counts(Path(getattr(settings, "library_db_path", ""))),
        },
        "maintenance": maintenance_status(settings),
        "runtime": _task_state_summary(),
        "prefs": _safe_user_prefs(),
        "logs": [
            {"name": path.name, "path": str(path), "size_bytes": path.stat().st_size}
            for path in _candidate_log_files()
        ],
    }


def create_diagnostics_archive(settings: Settings, *, readiness_payload: dict[str, Any] | None = None) -> Path:
    out_dir = diagnostics_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / _archive_id("diagnostics")
    payload = build_diagnostics_payload(settings, readiness_payload=readiness_payload)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _write_json(zf, "diagnostics.json", payload)
        _write_json(zf, "README.json", {
            "kind": "diagnostics",
            "privacy": "Contains configuration summaries, counts, readiness and redacted log tails. It does not include chat rows, library rows, chunks, PDFs, or API keys.",
        })
        for path in _candidate_log_files():
            zf.writestr(f"logs/{path.name}.tail.txt", _tail_text(path))
    return archive


def _backup_status_summary(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    return {
        "name": str(item.get("name") or ""),
        "created_at": float(item.get("created_at") or 0.0),
        "label": str(item.get("label") or ""),
        "size_bytes": int(item.get("size_bytes") or 0),
    }


def maintenance_status(settings: Settings) -> dict[str, Any]:
    auto_backup = _auto_backup_config(settings)
    auto_enabled = bool(auto_backup["enabled"])
    strict = _env_bool("KB_AUTO_BACKUP_STRICT", False)
    min_interval_s = max(0.0, _env_float("KB_AUTO_BACKUP_MIN_INTERVAL_S", 30.0))
    keep_count = max(1, _env_int("KB_BACKUP_KEEP_N", 30))
    backups = list_backup_archives()
    latest = backups[0] if backups else None
    latest_summary = _backup_status_summary(latest)
    return {
        "data_protection": {
            "enabled": bool(auto_enabled),
            "status": "enabled" if auto_enabled else "disabled",
            "can_toggle": not bool(auto_backup["locked"]),
            "manual_backup_available": True,
            "backup_count": len(backups),
            "latest_backup": latest_summary,
        },
        "auto_backup": {
            "enabled": bool(auto_enabled),
            "strict": bool(strict),
            "min_interval_s": min_interval_s,
            "source": str(auto_backup["source"]),
            "locked": bool(auto_backup["locked"]),
        },
        "backups": {
            "count": len(backups),
            "latest": latest_summary,
            "keep": keep_count,
            "directory": str(backup_dir()),
        },
    }


def _copy_sqlite_database(source: Path, dest: Path) -> bool:
    if not source.exists() or not source.is_file():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        src = sqlite3.connect(str(source))
        dst = sqlite3.connect(str(dest))
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
    except sqlite3.Error:
        shutil.copy2(source, dest)
    return True


def _add_file(zf: zipfile.ZipFile, path: Path, arcname: str) -> None:
    if path.exists() and path.is_file():
        zf.write(path, arcname)


def _iter_dir_files(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return [path for path in root.rglob("*") if path.is_file()]


def create_backup_archive(settings: Settings, *, label: str = "") -> dict[str, Any]:
    out_dir = backup_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / _archive_id("backup", label)
    db_dir = Path(getattr(settings, "db_dir", "")).expanduser()
    chat_db = Path(getattr(settings, "chat_db_path", "")).expanduser()
    library_db = Path(getattr(settings, "library_db_path", "")).expanduser()
    manifest: dict[str, Any] = {
        "kind": "backup",
        "created_at": time.time(),
        "label": str(label or ""),
        "files": [],
        "notes": "API keys and access tokens are not included. Re-enter them after restore if needed.",
    }
    with tempfile.TemporaryDirectory(prefix="kb_backup_") as tmp:
        tmp_root = Path(tmp)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for source, arcname in ((chat_db, "chat.sqlite3"), (library_db, "library.sqlite3")):
                temp_db = tmp_root / arcname
                if _copy_sqlite_database(source, temp_db):
                    _add_file(zf, temp_db, arcname)
                    manifest["files"].append({"source": str(source), "archive": arcname, "size_bytes": temp_db.stat().st_size})
            for file_path in _iter_dir_files(db_dir):
                rel = file_path.relative_to(db_dir).as_posix()
                arcname = f"db/{rel}"
                _add_file(zf, file_path, arcname)
                manifest["files"].append({"source": str(file_path), "archive": arcname, "size_bytes": file_path.stat().st_size})
            safe_prefs = _safe_user_prefs()
            if safe_prefs:
                _write_json(zf, "user_prefs.redacted.json", safe_prefs)
                manifest["files"].append({"source": str(ROOT / "user_prefs.json"), "archive": "user_prefs.redacted.json", "redacted": True})
            _write_json(zf, "manifest.json", manifest)
    return backup_info(archive)


def create_auto_snapshot(
    settings: Settings,
    *,
    action: str,
    label: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action_key = _clean_action(action)
    auto_backup = _auto_backup_config(settings)
    enabled = bool(auto_backup["enabled"])
    strict = _env_bool("KB_AUTO_BACKUP_STRICT", False)
    min_interval_s = max(0.0, _env_float("KB_AUTO_BACKUP_MIN_INTERVAL_S", 30.0))
    payload: dict[str, Any] = {
        "enabled": bool(enabled),
        "created": False,
        "reason": "",
        "action": action_key,
        "label": str(label or ""),
        "metadata": dict(metadata or {}),
        "strict": bool(strict),
        "source": str(auto_backup["source"]),
        "locked": bool(auto_backup["locked"]),
        "block_operation": False,
    }
    if not enabled:
        payload["reason"] = "disabled"
        return payload

    backup_label = f"auto-{action_key}"
    extra_label = _clean_action(label) if str(label or "").strip() else ""
    if extra_label:
        backup_label = f"{backup_label}-{extra_label}"

    with _AUTO_BACKUP_LOCK:
        now = time.monotonic()
        has_last = action_key in _AUTO_BACKUP_LAST
        last = float(_AUTO_BACKUP_LAST.get(action_key) or 0.0)
        remaining = min_interval_s - max(0.0, now - last)
        if has_last and min_interval_s > 0 and remaining > 0:
            payload.update({
                "reason": "rate_limited",
                "retry_after_s": round(remaining, 3),
            })
            return payload
        try:
            info = create_backup_archive(settings, label=backup_label)
        except Exception as exc:
            payload.update({
                "reason": "failed",
                "error": str(exc)[:240] or "backup failed",
                "block_operation": bool(strict),
            })
            return payload
        _AUTO_BACKUP_LAST[action_key] = time.monotonic()

    payload.update({
        "created": True,
        "reason": "created",
        "backup": info,
    })
    return payload


def backup_info(path: Path) -> dict[str, Any]:
    p = Path(path)
    manifest: dict[str, Any] = {}
    try:
        with zipfile.ZipFile(p, "r") as zf:
            if "manifest.json" in zf.namelist():
                manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
    except Exception:
        manifest = {}
    stat = p.stat()
    return {
        "name": p.name,
        "created_at": float(manifest.get("created_at") or stat.st_mtime),
        "label": str(manifest.get("label") or ""),
        "size_bytes": stat.st_size,
        "path": str(p),
    }


def _verify_sqlite_bytes(zf: zipfile.ZipFile, name: str, tmp_root: Path) -> dict[str, Any]:
    target = tmp_root / Path(name).name
    target.write_bytes(zf.read(name))
    result: dict[str, Any] = {"name": name, "ok": False, "tables": {}, "integrity": ""}
    try:
        conn = sqlite3.connect(str(target))
        try:
            integrity = str(conn.execute("pragma integrity_check").fetchone()[0] or "")
            rows = conn.execute(
                "select name from sqlite_master where type='table' and name not like 'sqlite_%' order by name"
            ).fetchall()
            tables: dict[str, int] = {}
            for (table_name,) in rows:
                try:
                    tables[str(table_name)] = int(conn.execute(f'select count(*) from "{table_name}"').fetchone()[0])
                except sqlite3.Error:
                    tables[str(table_name)] = -1
            result.update({"ok": integrity.lower() == "ok", "integrity": integrity, "tables": tables})
        finally:
            conn.close()
    except sqlite3.Error as exc:
        result.update({"ok": False, "integrity": str(exc)})
    return result


def _safe_archive_rel(name: str) -> Path:
    raw = str(name or "").replace("\\", "/")
    parts = [part for part in raw.split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"unsafe archive path: {name}")
    return Path(*parts)


def _safe_extract_zip(zf: zipfile.ZipFile, root: Path) -> list[Path]:
    root_resolved = Path(root).resolve()
    extracted: list[Path] = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        rel = _safe_archive_rel(info.filename)
        target = (root_resolved / rel).resolve()
        if target != root_resolved and root_resolved not in target.parents:
            raise ValueError(f"unsafe archive path: {info.filename}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(zf.read(info))
        extracted.append(target)
    return extracted


def _restore_runtime_blockers() -> list[str]:
    blockers: list[str] = []
    try:
        bg = bg_snapshot(runtime_state.BG_STATE, runtime_state.BG_LOCK)
        if bool(bg.get("running")) or int(bg.get("active_count") or 0) > 0 or list(bg.get("active_tasks") or []):
            blockers.append("background conversion is running")
        if list(bg.get("queue") or []):
            blockers.append("background conversion queue is not empty")
    except Exception:
        pass
    try:
        with runtime_state.GEN_LOCK:
            running = [
                task
                for task in runtime_state.GEN_TASKS.values()
                if str((task or {}).get("status") or "").strip().lower() in {"queued", "pending", "running"}
            ]
        if running:
            blockers.append("chat generation task is running")
    except Exception:
        pass
    return blockers


def _append_restore_audit(event: dict[str, Any]) -> None:
    payload = {
        "created_at": time.time(),
        **dict(event or {}),
    }
    path = restore_audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _RESTORE_AUDIT_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def list_restore_audit_events(limit: int = 20) -> list[dict[str, Any]]:
    count = max(1, min(int(limit or 20), 200))
    path = restore_audit_path()
    if not path.exists() or not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    events.append(payload)
                    if len(events) > 1000:
                        events = events[-500:]
    except OSError:
        return []
    events = events[-count:]
    return list(reversed(events))


def latest_restore_audit_event() -> dict[str, Any] | None:
    events = list_restore_audit_events(limit=1)
    return events[0] if events else None


def _audit_event_time(event: dict[str, Any] | None, key: str = "created_at") -> float:
    if not isinstance(event, dict):
        return 0.0
    try:
        return float(event.get(key) or 0.0)
    except Exception:
        return 0.0


def latest_restore_operation_event() -> dict[str, Any] | None:
    for event in list_restore_audit_events(limit=200):
        if str(event.get("event") or "") == "restore":
            return event
    return None


def latest_restore_review_state() -> dict[str, Any]:
    restore = latest_restore_operation_event()
    acknowledgement = None
    if restore:
        restore_created_at = _audit_event_time(restore)
        restore_backup = str(restore.get("backup") or "")
        for event in list_restore_audit_events(limit=200):
            if str(event.get("event") or "") != "restore_review_acknowledged":
                continue
            if str(event.get("status") or "") != "acknowledged":
                continue
            if str(event.get("backup") or "") != restore_backup:
                continue
            acknowledged_restore_at = _audit_event_time(event, "restore_created_at")
            if acknowledged_restore_at and restore_created_at and abs(acknowledged_restore_at - restore_created_at) <= 0.001:
                acknowledgement = event
                break
    return {
        "restore": restore,
        "acknowledgement": acknowledgement,
        "acknowledged": bool(restore and acknowledgement),
    }


def acknowledge_latest_restore_review(checks: dict[str, bool] | None = None) -> dict[str, Any]:
    state = latest_restore_review_state()
    restore = state.get("restore")
    if not isinstance(restore, dict):
        return {
            "ok": False,
            "status": "no_restore",
            "errors": ["no restore event has been recorded"],
            "audit_path": str(restore_audit_path()),
        }
    restore_status = str(restore.get("status") or "")
    backup = str(restore.get("backup") or "")
    restore_created_at = _audit_event_time(restore)
    if restore_status != "restored":
        return {
            "ok": False,
            "status": "restore_not_successful",
            "backup": backup,
            "restore_status": restore_status,
            "restore_created_at": restore_created_at,
            "errors": ["latest restore did not complete successfully"],
            "audit_path": str(restore_audit_path()),
        }
    if bool(state.get("acknowledged")):
        acknowledgement = dict(state.get("acknowledgement") or {})
        return {
            "ok": True,
            "status": "already_acknowledged",
            "backup": backup,
            "restore_created_at": restore_created_at,
            "acknowledged_at": _audit_event_time(acknowledgement),
            "audit_path": str(restore_audit_path()),
        }

    payload = {
        "event": "restore_review_acknowledged",
        "status": "acknowledged",
        "ok": True,
        "backup": backup,
        "restore_status": restore_status,
        "restore_created_at": restore_created_at,
        "checks": {str(key): bool(value) for key, value in dict(checks or {}).items()},
    }
    _append_restore_audit(payload)
    return {
        "ok": True,
        "status": "acknowledged",
        "backup": backup,
        "restore_created_at": restore_created_at,
        "acknowledged_at": time.time(),
        "audit_path": str(restore_audit_path()),
    }


def _public_restore_audit_event(event: dict[str, Any]) -> dict[str, Any]:
    raw_components = event.get("components")
    raw_checks = event.get("checks")
    restored = event.get("restored")
    pre_restore_backup = event.get("pre_restore_backup")
    pre_restore_name = ""
    if isinstance(pre_restore_backup, dict):
        pre_restore_name = str(pre_restore_backup.get("name") or "")
    return {
        "event": str(event.get("event") or ""),
        "status": str(event.get("status") or ""),
        "ok": bool(event.get("ok")),
        "backup": str(event.get("backup") or ""),
        "created_at": _audit_event_time(event),
        "restore_created_at": _audit_event_time(event, "restore_created_at"),
        "restart_required": bool(event.get("restart_required")),
        "components": dict(raw_components) if isinstance(raw_components, dict) else {},
        "checks": dict(raw_checks) if isinstance(raw_checks, dict) else {},
        "errors": [str(item) for item in list(event.get("errors") or [])[:4] if str(item or "").strip()],
        "warnings": [str(item) for item in list(event.get("warnings") or [])[:4] if str(item or "").strip()],
        "restored_count": len(restored) if isinstance(restored, list) else 0,
        "pre_restore_backup": pre_restore_name,
    }


def public_restore_audit_events(limit: int = 20) -> list[dict[str, Any]]:
    return [_public_restore_audit_event(event) for event in list_restore_audit_events(limit=limit)]


def _sqlite_sidecar_paths(path: Path) -> list[Path]:
    p = Path(path)
    return [p.with_name(p.name + suffix) for suffix in ("-wal", "-shm")]


def _remove_sqlite_sidecars(path: Path) -> list[str]:
    warnings: list[str] = []
    for sidecar in _sqlite_sidecar_paths(path):
        try:
            if sidecar.exists() and sidecar.is_file():
                sidecar.unlink()
        except OSError as exc:
            warnings.append(f"failed to remove {sidecar.name}: {exc}")
    return warnings


def _copy_file_atomic(source: Path, target: Path, *, sqlite_sidecars: bool = False) -> dict[str, Any]:
    src = Path(source)
    dest = Path(target).expanduser()
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(str(src))
    dest.parent.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    if sqlite_sidecars:
        warnings.extend(_remove_sqlite_sidecars(dest))
    tmp = dest.with_name(f".{dest.name}.restore-{uuid.uuid4().hex[:10]}.tmp")
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dest)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    if sqlite_sidecars:
        warnings.extend(_remove_sqlite_sidecars(dest))
    return {
        "kind": "file",
        "target": str(dest),
        "source": str(src),
        "size_bytes": dest.stat().st_size if dest.exists() else 0,
        "warnings": warnings,
    }


def _validate_restore_directory_target(target: Path) -> Path:
    dest = Path(target).expanduser().resolve()
    if not dest.name or str(dest) == str(dest.anchor):
        raise ValueError(f"refusing to restore into unsafe directory target: {dest}")
    return dest


def _replace_directory_atomic(source: Path, target: Path) -> dict[str, Any]:
    src = Path(source)
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(str(src))
    dest = _validate_restore_directory_target(target)
    dest.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:10]
    tmp = dest.parent / f".{dest.name}.restore-{token}.tmp"
    old = dest.parent / f".{dest.name}.restore-old-{token}"
    warnings: list[str] = []
    shutil.copytree(src, tmp)
    moved_old = False
    try:
        if dest.exists():
            dest.rename(old)
            moved_old = True
        tmp.rename(dest)
    except Exception:
        if not dest.exists() and moved_old and old.exists():
            try:
                old.rename(dest)
            except Exception:
                pass
        raise
    finally:
        try:
            if tmp.exists():
                shutil.rmtree(tmp)
        except OSError:
            pass
    if old.exists():
        try:
            if old.is_dir():
                shutil.rmtree(old)
            else:
                old.unlink()
        except OSError as exc:
            warnings.append(f"old target retained at {old}: {exc}")
    files = [path for path in dest.rglob("*") if path.is_file()] if dest.exists() and dest.is_dir() else []
    return {
        "kind": "directory",
        "target": str(dest),
        "source": str(src),
        "file_count": len(files),
        "warnings": warnings,
    }


def verify_backup_archive(path: Path) -> dict[str, Any]:
    p = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {
        "zip": {"ok": False},
        "sqlite": {},
        "required_files": {},
    }
    manifest: dict[str, Any] = {}

    if not p.exists() or not p.is_file():
        return {
            "ok": False,
            "name": p.name,
            "path": str(p),
            "errors": ["backup archive not found"],
            "warnings": [],
            "checks": checks,
        }

    try:
        with zipfile.ZipFile(p, "r") as zf:
            entries = zf.infolist()
            names = set(zf.namelist())
            bad_entry = zf.testzip()
            if bad_entry:
                errors.append(f"zip entry failed CRC check: {bad_entry}")
            checks["zip"] = {
                "ok": bad_entry is None,
                "file_count": len(entries),
                "uncompressed_size_bytes": sum(int(info.file_size or 0) for info in entries),
            }

            if "manifest.json" in names:
                try:
                    manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
                except Exception as exc:
                    errors.append(f"manifest.json is unreadable: {exc}")
            else:
                errors.append("manifest.json is missing")

            required = {
                "chat.sqlite3": "chat database",
                "library.sqlite3": "library database",
                "db/docs.json": "knowledge-base document index",
            }
            for entry, label in required.items():
                present = entry in names
                checks["required_files"][entry] = {"present": present, "label": label}
                if not present:
                    if entry in {"chat.sqlite3", "library.sqlite3"}:
                        errors.append(f"{entry} is missing")
                    else:
                        warnings.append(f"{entry} is missing")

            with tempfile.TemporaryDirectory(prefix="kb_backup_verify_") as tmp:
                tmp_root = Path(tmp)
                for db_name in ("chat.sqlite3", "library.sqlite3"):
                    if db_name not in names:
                        continue
                    sqlite_check = _verify_sqlite_bytes(zf, db_name, tmp_root)
                    checks["sqlite"][db_name] = sqlite_check
                    if not bool(sqlite_check.get("ok")):
                        errors.append(f"{db_name} failed SQLite integrity check")
    except zipfile.BadZipFile:
        errors.append("archive is not a valid zip file")
    except Exception as exc:
        errors.append(str(exc)[:240] or "backup verification failed")

    stat = p.stat()
    return {
        "ok": not errors,
        "name": p.name,
        "created_at": float(manifest.get("created_at") or stat.st_mtime),
        "label": str(manifest.get("label") or ""),
        "size_bytes": stat.st_size,
        "path": str(p),
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "verified_at": time.time(),
    }


def _restore_file_destination(extracted_root: Path, archive_name: str, target: Path, label: str) -> dict[str, Any]:
    source = extracted_root / _safe_archive_rel(archive_name)
    source_exists = source.exists() and source.is_file()
    target_path = Path(target).expanduser()
    return {
        "kind": "file",
        "label": label,
        "archive": archive_name,
        "target": str(target_path),
        "source_exists": bool(source_exists),
        "target_exists": target_path.exists(),
        "source_size_bytes": source.stat().st_size if source_exists else 0,
        "target_size_bytes": target_path.stat().st_size if target_path.exists() and target_path.is_file() else 0,
        "action": "replace" if target_path.exists() else "create",
    }


def _restore_db_dir_destination(extracted_root: Path, target: Path) -> dict[str, Any]:
    source = extracted_root / "db"
    target_path = Path(target).expanduser()
    files = [path for path in source.rglob("*") if path.is_file()] if source.exists() and source.is_dir() else []
    size = 0
    suffix_counts: dict[str, int] = {}
    for path in files:
        try:
            size += path.stat().st_size
        except OSError:
            pass
        suffix = path.suffix.lower() or "<none>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    key_files = {
        "docs.json": (source / "docs.json").exists(),
        "references_index.json": (source / "references_index.json").exists(),
        "crossref_cache.json": (source / "crossref_cache.json").exists(),
    }
    chunk_count = len(list((source / "chunks").glob("*.jsonl"))) if (source / "chunks").exists() else 0
    return {
        "kind": "directory",
        "label": "knowledge base directory",
        "archive": "db/",
        "target": str(target_path),
        "source_exists": source.exists() and source.is_dir(),
        "target_exists": target_path.exists(),
        "source_file_count": len(files),
        "source_size_bytes": size,
        "source_suffix_counts": dict(sorted(suffix_counts.items())),
        "chunk_file_count": chunk_count,
        "key_files": key_files,
        "action": "replace_directory_contents" if target_path.exists() else "create_directory",
    }


def restore_dry_run_backup_archive(settings: Settings, path: Path) -> dict[str, Any]:
    p = Path(path)
    verification = verify_backup_archive(p)
    errors: list[str] = list(verification.get("errors") or [])
    warnings: list[str] = list(verification.get("warnings") or [])
    destinations: list[dict[str, Any]] = []
    sqlite_checks: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    extracted_count = 0

    if not p.exists() or not p.is_file():
        errors.append("backup archive not found")
    else:
        try:
            with tempfile.TemporaryDirectory(prefix="kb_restore_dry_run_") as tmp:
                extracted_root = Path(tmp)
                with zipfile.ZipFile(p, "r") as zf:
                    if "manifest.json" in zf.namelist():
                        try:
                            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
                        except Exception:
                            manifest = {}
                    extracted = _safe_extract_zip(zf, extracted_root)
                    extracted_count = len(extracted)

                chat_dest = _restore_file_destination(
                    extracted_root,
                    "chat.sqlite3",
                    Path(str(getattr(settings, "chat_db_path", "") or "")),
                    "chat database",
                )
                library_dest = _restore_file_destination(
                    extracted_root,
                    "library.sqlite3",
                    Path(str(getattr(settings, "library_db_path", "") or "")),
                    "library database",
                )
                db_dest = _restore_db_dir_destination(extracted_root, Path(str(getattr(settings, "db_dir", "") or "")))
                destinations = [chat_dest, library_dest, db_dest]

                for item in (chat_dest, library_dest):
                    if not bool(item.get("source_exists")):
                        errors.append(f"{item.get('archive')} is missing")
                        continue
                    sqlite_path = extracted_root / _safe_archive_rel(str(item.get("archive") or ""))
                    sqlite_checks[str(item.get("archive") or "")] = _sqlite_table_counts(sqlite_path)

                if not bool(db_dest.get("source_exists")):
                    warnings.append("db/ directory is missing")
                elif not bool((db_dest.get("key_files") or {}).get("docs.json")):
                    warnings.append("db/docs.json is missing")
        except zipfile.BadZipFile:
            errors.append("archive is not a valid zip file")
        except Exception as exc:
            errors.append(str(exc)[:240] or "restore dry-run failed")

    user_prefs_note = (
        "user_prefs.redacted.json can help inspect preferences, but API keys are redacted and must be re-entered after restore."
    )
    steps = [
        "Stop the FastAPI server and background workers.",
        "Create a fresh manual backup of the current state.",
        "Copy chat.sqlite3, library.sqlite3, and db/ from the selected backup into their configured target paths.",
        "Re-enter model/API credentials if they were stored only in user preferences.",
        "Restart the server and run production readiness checks.",
    ]
    stat = p.stat() if p.exists() else None
    return {
        "ok": not errors,
        "can_restore": not errors,
        "name": p.name,
        "created_at": float(manifest.get("created_at") or (stat.st_mtime if stat else 0)),
        "label": str(manifest.get("label") or ""),
        "size_bytes": int(stat.st_size if stat else 0),
        "verified": verification,
        "extracted_file_count": extracted_count,
        "destinations": destinations,
        "sqlite": sqlite_checks,
        "errors": errors,
        "warnings": warnings,
        "notes": [user_prefs_note],
        "restore_steps": steps,
        "checked_at": time.time(),
    }


def restore_backup_archive(
    settings: Settings,
    path: Path,
    *,
    confirm: str,
    components: dict[str, bool] | None = None,
    create_pre_restore_backup: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    p = Path(path)
    expected_confirm = f"RESTORE {p.name}"
    selected = {
        "chat": True,
        "library": True,
        "db": True,
        **{str(k): bool(v) for k, v in dict(components or {}).items()},
    }
    result: dict[str, Any] = {
        "ok": False,
        "name": p.name,
        "expected_confirmation": expected_confirm,
        "components": selected,
        "pre_restore_backup": None,
        "restored": [],
        "errors": [],
        "warnings": [],
        "restart_required": False,
        "audit_path": str(restore_audit_path()),
    }

    def finish(status: str) -> dict[str, Any]:
        result["status"] = status
        try:
            _append_restore_audit({
                "event": "restore",
                "status": status,
                "backup": p.name,
                "components": selected,
                "ok": bool(result.get("ok")),
                "errors": list(result.get("errors") or []),
                "warnings": list(result.get("warnings") or []),
                "restart_required": bool(result.get("restart_required")),
                "pre_restore_backup": result.get("pre_restore_backup"),
                "restored": list(result.get("restored") or []),
            })
        except Exception:
            pass
        return result

    if str(confirm or "") != expected_confirm:
        result["errors"].append("confirmation text mismatch")
        return finish("confirmation_failed")

    plan = restore_dry_run_backup_archive(settings, p)
    result["dry_run"] = plan
    if not bool(plan.get("can_restore")):
        result["errors"].extend([str(item) for item in list(plan.get("errors") or [])])
        if not result["errors"]:
            result["errors"].append("restore dry-run did not pass")
        return finish("dry_run_failed")

    blockers = [] if force else _restore_runtime_blockers()
    if blockers:
        result["errors"].extend(blockers)
        return finish("blocked")

    if bool(selected.get("db")):
        db_target = Path(str(getattr(settings, "db_dir", "") or "")).expanduser().resolve()
        try:
            if db_target in p.resolve().parents:
                result["errors"].append("backup archive is inside the target db directory; move it outside before restore")
                return finish("blocked")
        except Exception:
            pass

    try:
        if create_pre_restore_backup:
            result["pre_restore_backup"] = create_backup_archive(settings, label=f"pre-restore-{p.stem[:32]}")

        with tempfile.TemporaryDirectory(prefix="kb_restore_apply_") as tmp:
            extracted_root = Path(tmp)
            with zipfile.ZipFile(p, "r") as zf:
                _safe_extract_zip(zf, extracted_root)

            restored: list[dict[str, Any]] = []
            warnings: list[str] = []

            if bool(selected.get("db")):
                restored.append(_replace_directory_atomic(extracted_root / "db", Path(str(getattr(settings, "db_dir", "") or ""))))
            if bool(selected.get("chat")):
                item = _copy_file_atomic(
                    extracted_root / "chat.sqlite3",
                    Path(str(getattr(settings, "chat_db_path", "") or "")),
                    sqlite_sidecars=True,
                )
                restored.append(item)
            if bool(selected.get("library")):
                item = _copy_file_atomic(
                    extracted_root / "library.sqlite3",
                    Path(str(getattr(settings, "library_db_path", "") or "")),
                    sqlite_sidecars=True,
                )
                restored.append(item)

            for item in restored:
                warnings.extend([str(value) for value in list(item.get("warnings") or []) if str(value or "").strip()])
            result["restored"] = restored
            result["warnings"].extend(warnings)
            result["ok"] = True
            result["restart_required"] = True
            return finish("restored")
    except Exception as exc:
        result["errors"].append(str(exc)[:240] or "restore failed")
        return finish("failed")


def list_backup_archives() -> list[dict[str, Any]]:
    root = backup_dir()
    if not root.exists():
        return []
    items = [backup_info(path) for path in root.glob("backup-*.zip") if path.is_file()]
    return sorted(items, key=lambda item: float(item.get("created_at") or 0), reverse=True)


def resolve_backup_archive(name: str) -> Path:
    clean = Path(str(name or "")).name
    if not clean.startswith("backup-") or not clean.endswith(".zip"):
        raise FileNotFoundError(clean)
    path = (backup_dir() / clean).resolve()
    root = backup_dir().resolve()
    if root not in path.parents or not path.exists() or not path.is_file():
        raise FileNotFoundError(clean)
    return path


def cleanup_backup_archives(*, keep: int | None = None, dry_run: bool = False) -> dict[str, Any]:
    keep_count = max(1, int(keep if keep is not None else _env_int("KB_BACKUP_KEEP_N", 30)))
    items = list_backup_archives()
    victims = items[keep_count:]
    deleted: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for item in victims:
        name = str(item.get("name") or "")
        if not name:
            continue
        entry = {
            "name": name,
            "size_bytes": int(item.get("size_bytes") or 0),
            "created_at": float(item.get("created_at") or 0),
            "dry_run": bool(dry_run),
        }
        if dry_run:
            deleted.append(entry)
            continue
        try:
            resolve_backup_archive(name).unlink()
            deleted.append(entry)
        except Exception as exc:
            failed.append({**entry, "error": str(exc)[:240]})
    return {
        "ok": not failed,
        "keep": keep_count,
        "before": len(items),
        "deleted": len(deleted),
        "failed": len(failed),
        "dry_run": bool(dry_run),
        "items": deleted,
        "errors": failed,
    }
