from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import secrets
import time
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.deps import get_settings, load_prefs, save_prefs
from api.internal_access import management_api_allowed, require_management_api
from api.security import auth_token_configured, management_auth_required, management_token_configured, request_is_authenticated
from kb.file_ops import _pick_directory_dialog
from kb.maintenance import latest_restore_review_state
from kb.user_issue_store import UserIssueStore
from kb.version import read_app_version

router = APIRouter(prefix="/api", tags=["settings"])
_PATH_PREF_KEYS = {"pdf_dir", "md_dir"}
_API_KEY_PREF_KEYS = {"text_api_key", "vision_api_key"}
_LLM_PREF_KEYS = {"text_base_url", "text_model", "vision_base_url", "vision_model"}
_BOOL_PREF_KEYS = {"auto_backup_enabled", "quality_data_sharing_enabled"}
_PRIVATE_PREF_KEYS = _API_KEY_PREF_KEYS | {"quality_data_client_id"}
_PUBLIC_PREF_KEYS = {
    "top_k",
    "temperature",
    "max_tokens",
    "deep_read",
    "show_context",
    "theme",
    "pdf_dir",
    "md_dir",
    "answer_contract_v1",
    "answer_depth_auto",
    "answer_mode_hint",
    "answer_output_mode",
    "refs_card_locale",
    "ui_locale",
    "sidebar_collapsed",
    "auto_backup_enabled",
    "quality_data_sharing_enabled",
}
_ANSWER_MODE_HINTS = {"", "reading", "compare", "idea", "experiment", "troubleshoot", "writing"}
_ANSWER_OUTPUT_MODES = {"", "reading_guide", "fact_answer", "critical_review"}
_CHOICE_PREF_VALUES = {
    "theme": {"light", "dark"},
    "ui_locale": {"zh", "en"},
    "refs_card_locale": {"auto", "zh", "en"},
    "answer_mode_hint": _ANSWER_MODE_HINTS,
    "answer_output_mode": _ANSWER_OUTPUT_MODES,
}
_LLM_TEST_RESULTS: dict[str, dict] = {}
_RESTORE_NOTICE_WINDOW_S = 7 * 24 * 60 * 60
_RESTORE_FAILURE_NOTICE_WINDOW_S = 24 * 60 * 60
_SENSITIVE_PREF_KEY_RE = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?token|auth[_-]?token|token|secret|password|"
    r"passphrase|authorization|cookie|credential|private[_-]?key|client[_-]?secret)(?:$|[_-])",
    flags=re.IGNORECASE,
)
_TOKEN_TEXT_RE = re.compile(r"\b(?:sk|pk|ghp|github_pat|xoxb|xoxp|ya29|AIza)[A-Za-z0-9_\-]{12,}\b")
_AUTH_VALUE_RE = re.compile(r"\b(Bearer|Token|Api[-_ ]?Key)\s+([A-Za-z0-9._\-]{8,})", flags=re.IGNORECASE)
_URL_QUERY_RE = re.compile(r"(https?://[^\s?#]+)(?:\?[^ \t\r\n\"'<>]*)?")
_MAX_PATH_PREF_CHARS = 1200
_MAX_API_KEY_CHARS = 4096
_MAX_BASE_URL_CHARS = 500
_MAX_MODEL_CHARS = 200
_MAX_HINT_CHARS = 40


def _validate_model_base_url(raw: str, *, key: str) -> str:
    value = str(raw or "").replace("\x00", "").strip().rstrip("/")
    if not value:
        return ""
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(400, f"{key} must be an http(s) URL")
    if parsed.username or parsed.password:
        raise HTTPException(400, f"{key} must not include credentials")
    if parsed.query or parsed.fragment:
        raise HTTPException(400, f"{key} must not include query or fragment")
    return value


def _classify_connection_error(error: object) -> str:
    text = str(error or "").lower()
    if "api key" in text or "authentication" in text or "unauthorized" in text or "401" in text:
        return "auth"
    if "forbidden" in text or "403" in text:
        return "permission"
    if "model" in text and ("not found" in text or "does not exist" in text or "invalid" in text):
        return "model"
    if "base_url" in text or "invalid url" in text or "unsupported protocol" in text:
        return "base_url"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "connection" in text or "network" in text or "name resolution" in text or "connect" in text:
        return "network"
    return "unknown"


def _public_error_text(error: object, *, limit: int = 800) -> str:
    text = str(error or "").replace("\x00", " ")
    text = _AUTH_VALUE_RE.sub(r"\1 [redacted]", text)
    text = _TOKEN_TEXT_RE.sub("[token]", text)
    text = _URL_QUERY_RE.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[: int(max(0, limit))]


def _normalize_pref_value(key: str, value):
    if key in _API_KEY_PREF_KEYS:
        raw = str(value or "").replace("\x00", "").strip()
        if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
            raw = raw[1:-1].strip()
        return raw
    if key in _LLM_PREF_KEYS:
        raw = str(value or "").replace("\x00", "").strip()
        if key.endswith("_base_url"):
            raw = _validate_model_base_url(raw, key=key)
        return raw
    if key in _BOOL_PREF_KEYS:
        if isinstance(value, bool):
            return value
        raw = str(value or "").strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
        raise HTTPException(400, f"{key} must be a boolean")
    if key in _CHOICE_PREF_VALUES:
        raw = str(value or "").replace("\x00", "").strip().lower()
        if raw not in _CHOICE_PREF_VALUES[key]:
            allowed = ", ".join(sorted(_CHOICE_PREF_VALUES[key]))
            raise HTTPException(400, f"{key} must be one of: {allowed}")
        return raw
    if key not in _PATH_PREF_KEYS:
        return value
    raw = str(value or "").strip()
    if not raw:
        raise HTTPException(400, f"{key} cannot be empty")
    if len(raw) > _MAX_PATH_PREF_CHARS:
        raise HTTPException(400, f"{key} is too long")
    try:
        path = Path(raw).expanduser().resolve(strict=False)
    except Exception as exc:
        raise HTTPException(400, f"invalid {key}: {exc}") from exc
    if path.exists() and not path.is_dir():
        raise HTTPException(400, f"{key} must be a directory")
    return str(path)


def _discard_unsent_quality_data_outbox() -> dict:
    try:
        settings = get_settings()
        db_path = getattr(settings, "user_issues_db_path", None)
        if not db_path:
            db_dir = Path(getattr(settings, "db_dir", Path.cwd() / "db")).expanduser().resolve()
            db_path = db_dir.parent / "user_issues.sqlite3"
        return UserIssueStore(Path(db_path)).discard_unsent_remote_outbox()
    except Exception as exc:
        return {"ok": False, "removed": 0, "error": str(exc)[:240]}


def _discard_stale_quality_data_before_enable() -> None:
    result = _discard_unsent_quality_data_outbox()
    if not bool(result.get("ok")):
        detail = str(result.get("error") or "failed to clear pending quality data before enabling sharing")
        raise HTTPException(500, detail[:240])


def _pref_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    raw = str(value or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _connection_status(s) -> dict:
    return {
        "text": {
            "has_api_key": bool(s.text_api_key),
            "base_url": s.text_base_url,
            "model": s.text_model,
        },
        "vision": {
            "has_api_key": bool(s.vision_api_key),
            "base_url": s.vision_base_url,
            "model": s.vision_model,
            "uses_text_fallback": bool(getattr(s, "vision_uses_text_fallback", False)),
        },
        "auto_route": bool(s.auto_route),
    }


def _pref_key_is_sensitive(key: object) -> bool:
    clean = str(key or "").strip()
    return bool(clean in _PRIVATE_PREF_KEYS or _SENSITIVE_PREF_KEY_RE.search(clean))


def _public_prefs(prefs: dict, *, include_paths: bool = True) -> dict:
    out: dict[str, object] = {}
    for key, value in dict(prefs or {}).items():
        clean = str(key or "")
        if clean not in _PUBLIC_PREF_KEYS or _pref_key_is_sensitive(clean):
            continue
        if not include_paths and clean in _PATH_PREF_KEYS:
            continue
        if isinstance(value, (dict, list, tuple)):
            continue
        out[clean] = value
    return out


def _provider_fingerprint(*, api_key: str | None, base_url: str, model: str) -> str:
    raw = "|".join([str(api_key or ""), str(base_url or ""), str(model or "")])
    return hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest()[:16]


def _provider_readiness(s, target: Literal["text", "vision"]) -> dict:
    if target == "vision":
        has_key = bool(s.vision_api_key)
        base_url = s.vision_base_url
        model = s.vision_model
        uses_text_fallback = bool(getattr(s, "vision_uses_text_fallback", False))
        fingerprint = _provider_fingerprint(api_key=s.vision_api_key, base_url=base_url, model=model)
    else:
        has_key = bool(s.text_api_key)
        base_url = s.text_base_url
        model = s.text_model
        uses_text_fallback = False
        fingerprint = _provider_fingerprint(api_key=s.text_api_key, base_url=base_url, model=model)

    last_test = _LLM_TEST_RESULTS.get(target)
    if last_test and last_test.get("fingerprint") != fingerprint:
        last_test = None

    if not has_key:
        status = "missing"
        severity = "error"
        reason = "missing_api_key"
    elif uses_text_fallback:
        status = "fallback"
        severity = "warning"
        reason = "vision_uses_text_fallback"
    elif last_test and bool(last_test.get("ok")):
        status = "ok"
        severity = "ok"
        reason = "last_test_ok"
    elif last_test:
        status = "failed"
        severity = "error"
        reason = str(last_test.get("error_type") or "unknown")
    else:
        status = "configured"
        severity = "warning"
        reason = "configured_not_tested"

    public_last_test = None
    if last_test:
        public_last_test = {
            "ok": bool(last_test.get("ok")),
            "checked_at": float(last_test.get("checked_at") or 0.0),
            "error": str(last_test.get("error") or ""),
            "error_type": str(last_test.get("error_type") or ""),
            "reply": str(last_test.get("reply") or ""),
        }

    return {
        "target": target,
        "has_api_key": has_key,
        "base_url": base_url,
        "model": model,
        "uses_text_fallback": uses_text_fallback,
        "status": status,
        "severity": severity,
        "reason": reason,
        "last_test": public_last_test,
    }


def _readiness_payload(s) -> dict:
    text = _provider_readiness(s, "text")
    vision = _provider_readiness(s, "vision")
    if text["severity"] == "error":
        overall = {"status": "error", "reason": text["reason"], "target": "text"}
    elif vision["severity"] == "error":
        overall = {"status": "error", "reason": vision["reason"], "target": "vision"}
    elif text["severity"] == "warning":
        overall = {"status": "warning", "reason": text["reason"], "target": "text"}
    elif vision["severity"] == "warning":
        overall = {"status": "warning", "reason": vision["reason"], "target": "vision"}
    else:
        overall = {"status": "ok", "reason": "ready", "target": ""}
    return {"providers": {"text": text, "vision": vision}, "overall": overall}


def _readiness_status(items: list[dict]) -> str:
    severities = {str(item.get("severity") or "") for item in items}
    if "error" in severities:
        return "error"
    if "warning" in severities:
        return "warning"
    return "ok"


def _readiness_item(
    key: str,
    *,
    severity: Literal["ok", "warning", "error"],
    label: str,
    detail: str = "",
    action: str = "",
) -> dict:
    return {
        "key": key,
        "status": severity,
        "severity": severity,
        "label": label,
        "detail": detail,
        "action": action,
    }


def _check_directory(key: str, label: str, path_value: object) -> dict:
    path = Path(str(path_value)).expanduser()
    if path.exists() and path.is_dir():
        return _readiness_item(key, severity="ok", label=label, detail=str(path))
    parent = path.parent
    if parent.exists() and os.access(parent, os.W_OK):
        return _readiness_item(
            key,
            severity="warning",
            label=label,
            detail=f"Directory does not exist yet: {path}",
            action="create_directory",
        )
    return _readiness_item(
        key,
        severity="error",
        label=label,
        detail=f"Directory parent is not writable or missing: {parent}",
        action="fix_path",
    )


def _check_file_parent(key: str, label: str, path_value: object) -> dict:
    path = Path(str(path_value)).expanduser()
    parent = path.parent
    if parent.exists() and os.access(parent, os.W_OK):
        return _readiness_item(key, severity="ok", label=label, detail=str(path))
    return _readiness_item(
        key,
        severity="error",
        label=label,
        detail=f"Parent directory is not writable or missing: {parent}",
        action="fix_path",
    )


def _restore_event_age_s(event: dict | None) -> float | None:
    if not isinstance(event, dict):
        return None
    try:
        created_at = float(event.get("created_at") or 0.0)
    except Exception:
        return None
    if created_at <= 0:
        return None
    return max(0.0, time.time() - created_at)


def _public_restore_event(event: dict | None) -> dict | None:
    if not isinstance(event, dict):
        return None
    errors = [str(item) for item in list(event.get("errors") or [])[:3] if str(item or "").strip()]
    warnings = [str(item) for item in list(event.get("warnings") or [])[:3] if str(item or "").strip()]
    try:
        created_at = float(event.get("created_at") or 0.0)
    except Exception:
        created_at = 0.0
    raw_components = event.get("components")
    components = dict(raw_components) if isinstance(raw_components, dict) else {}
    return {
        "event": str(event.get("event") or ""),
        "status": str(event.get("status") or ""),
        "backup": str(event.get("backup") or ""),
        "created_at": created_at,
        "ok": bool(event.get("ok")),
        "restart_required": bool(event.get("restart_required")),
        "components": components,
        "errors": errors,
        "warnings": warnings,
    }


def _restore_readiness_item(event: dict | None, *, acknowledged: bool = False) -> dict | None:
    if not isinstance(event, dict):
        return None
    status = str(event.get("status") or "").strip().lower()
    age_s = _restore_event_age_s(event)
    backup = str(event.get("backup") or "selected backup")
    if status == "restored":
        if acknowledged:
            return None
        if age_s is not None and age_s > _RESTORE_NOTICE_WINDOW_S:
            return None
        return _readiness_item(
            "recent_restore",
            severity="warning",
            label="Recent restore",
            detail=(
                f"Backup {backup} was restored recently. Restart the API, then verify API keys, "
                "the knowledge-base index, chat history, and library data."
            ),
            action="restart_and_check",
        )
    if status in {"failed", "blocked", "dry_run_failed"}:
        if age_s is not None and age_s > _RESTORE_FAILURE_NOTICE_WINDOW_S:
            return None
        return _readiness_item(
            "recent_restore",
            severity="warning",
            label="Recent restore",
            detail=f"The latest restore attempt for {backup} ended as {status}. Inspect the restore audit before launch.",
            action="inspect_restore_audit",
        )
    return None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _quality_remote_host_is_local(hostname: str) -> bool:
    host = str(hostname or "").strip("[]").lower().split("%", 1)[0]
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host in {"localhost"} or host.endswith(".localhost") or host.endswith(".local")
    return bool(
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_unspecified
        or ip.is_reserved
        or ip.is_multicast
    )


def _quality_remote_url_block_reason(url: str, *, production: bool = False) -> str:
    raw = str(url or "").strip()
    if not raw:
        return "missing_remote_url"
    try:
        parsed = urlsplit(raw)
        try:
            parsed.port
        except ValueError:
            return "invalid_remote_url"
    except Exception:
        return "invalid_remote_url"

    scheme = str(parsed.scheme or "").strip().lower()
    hostname = str(parsed.hostname or "").strip()
    if scheme not in {"http", "https"} or not hostname:
        return "invalid_remote_url"
    if parsed.username or parsed.password:
        return "remote_url_credentials"

    is_local = _quality_remote_host_is_local(hostname)
    local_allowed = _env_bool("KB_USER_ISSUES_ALLOW_LOCAL_REMOTE", False)
    if is_local and (production or not local_allowed):
        return "local_remote_url"
    if scheme != "https" and not ((not production) and is_local and local_allowed and scheme == "http"):
        return "insecure_remote_url"
    return ""


def _quality_remote_detail(block_reason: str) -> str:
    details = {
        "missing_remote_url": "Enabled but KB_USER_ISSUES_REMOTE_URL is not configured.",
        "invalid_remote_url": "KB_USER_ISSUES_REMOTE_URL must be a valid http(s) URL with a host.",
        "remote_url_credentials": "KB_USER_ISSUES_REMOTE_URL must not include embedded user:pass credentials.",
        "local_remote_url": "KB_USER_ISSUES_REMOTE_URL points to a local/private host; enable KB_USER_ISSUES_ALLOW_LOCAL_REMOTE only for private tests.",
        "insecure_remote_url": "KB_USER_ISSUES_REMOTE_URL must use HTTPS for real user deployments.",
    }
    return details.get(block_reason, "Remote quality telemetry is not safely configured.")


def _production_readiness_payload(s) -> dict:
    items: list[dict] = []
    llm = _readiness_payload(s)
    text_ready = llm["providers"]["text"]
    vision_ready = llm["providers"]["vision"]
    items.append(_readiness_item(
        "text_llm",
        severity=text_ready["severity"],
        label="Text model",
        detail=text_ready["reason"],
        action="configure_text_api_key" if text_ready["severity"] == "error" else "",
    ))
    items.append(_readiness_item(
        "vision_llm",
        severity=vision_ready["severity"],
        label="Vision model",
        detail=vision_ready["reason"],
        action="configure_vision_api_key" if vision_ready["severity"] == "error" else "",
    ))
    items.append(_check_directory("db_dir", "Knowledge base directory", getattr(s, "db_dir", "")))
    items.append(_check_file_parent("chat_db", "Chat database", getattr(s, "chat_db_path", "")))
    items.append(_check_file_parent("library_db", "Library database", getattr(s, "library_db_path", "")))
    items.append(_check_file_parent("user_issues_db", "User issues database", getattr(s, "user_issues_db_path", "")))
    remote_issue_enabled = bool(getattr(s, "user_issues_remote_enabled", False))
    remote_issue_url = str(getattr(s, "user_issues_remote_url", "") or "").strip()
    remote_issue_token = str(getattr(s, "user_issues_remote_token", "") or "").strip()
    production = bool(getattr(s, "production", False))
    remote_issue_url_block = (
        _quality_remote_url_block_reason(remote_issue_url, production=production) if remote_issue_enabled else ""
    )
    if remote_issue_enabled and remote_issue_url_block:
        items.append(_readiness_item(
            "user_issues_remote",
            severity="error" if production else "warning",
            label="Remote quality telemetry",
            detail=_quality_remote_detail(remote_issue_url_block),
            action="fix_user_issues_remote_url",
        ))
    elif remote_issue_enabled and not remote_issue_token:
        items.append(_readiness_item(
            "user_issues_remote",
            severity="warning",
            label="Remote quality telemetry",
            detail="Enabled without KB_USER_ISSUES_REMOTE_TOKEN; use a token for public collectors.",
            action="set_user_issues_remote_token",
        ))
    elif remote_issue_enabled:
        items.append(_readiness_item(
            "user_issues_remote",
            severity="ok",
            label="Remote quality telemetry",
            detail="Enabled",
        ))

    auth_required = bool(getattr(s, "auth_required", False))
    if auth_required and not auth_token_configured(s):
        items.append(_readiness_item(
            "api_auth",
            severity="error",
            label="API access protection",
            detail="Auth is required but no KB_ACCESS_TOKEN or KB_ACCESS_TOKEN_SHA256 is configured.",
            action="set_access_token",
        ))
    elif auth_required:
        items.append(_readiness_item("api_auth", severity="ok", label="API access protection", detail="Enabled"))
    else:
        items.append(_readiness_item(
            "api_auth",
            severity="ok",
            label="API access protection",
            detail="Disabled; public access",
        ))

    management_required = management_auth_required(s)
    if management_required and not management_token_configured(s):
        items.append(_readiness_item(
            "management_auth",
            severity="error",
            label="Management API protection",
            detail="Management writes are protected but no management access token is configured.",
            action="set_management_access_token",
        ))
    elif management_required:
        items.append(_readiness_item(
            "management_auth",
            severity="ok",
            label="Management API protection",
            detail="Enabled for server settings and library changes.",
        ))
    else:
        items.append(_readiness_item(
            "management_auth",
            severity="warning" if production else "ok",
            label="Management API protection",
            detail="Disabled; management writes are public.",
            action="enable_management_auth" if production else "",
        ))

    raw_origins = (os.environ.get("KB_API_ALLOW_ORIGINS") or os.environ.get("KB_CORS_ALLOW_ORIGINS") or "").strip()
    if production and raw_origins == "*":
        items.append(_readiness_item(
            "cors",
            severity="error",
            label="CORS origins",
            detail="Wildcard CORS is unsafe in production.",
            action="set_allowed_origins",
        ))
    else:
        items.append(_readiness_item("cors", severity="ok", label="CORS origins", detail=raw_origins or "local defaults"))

    root = Path(__file__).resolve().parents[2]
    dist_index = root / "web" / "dist" / "index.html"
    if dist_index.exists():
        items.append(_readiness_item("frontend_build", severity="ok", label="Frontend build", detail=str(dist_index)))
    else:
        items.append(_readiness_item(
            "frontend_build",
            severity="error" if production else "warning",
            label="Frontend build",
            detail="web/dist/index.html is missing.",
            action="run_npm_build",
        ))

    try:
        restore_state = latest_restore_review_state()
    except Exception:
        restore_state = {}
    latest_restore = restore_state.get("restore") if isinstance(restore_state, dict) else None
    acknowledgement = restore_state.get("acknowledgement") if isinstance(restore_state, dict) else None
    restore_acknowledged = bool(restore_state.get("acknowledged")) if isinstance(restore_state, dict) else False
    if not isinstance(latest_restore, dict):
        latest_restore = None
    if not isinstance(acknowledgement, dict):
        acknowledgement = None
    restore_item = _restore_readiness_item(latest_restore, acknowledged=restore_acknowledged)
    if restore_item:
        items.append(restore_item)

    status = _readiness_status(items)
    return {
        "status": status,
        "env": str(getattr(s, "app_env", "development") or "development"),
        "production": production,
        "auth_required": auth_required,
        "management_auth_required": management_required,
        "items": items,
        "llm": llm,
        "restore": {
            "latest": _public_restore_event(latest_restore),
            "acknowledgement": _public_restore_event(acknowledgement),
            "acknowledged": restore_acknowledged,
        },
    }


def _public_readiness_payload(payload: dict, *, reveal_paths: bool) -> dict:
    if reveal_paths:
        return payload
    result = dict(payload or {})
    items: list[dict] = []
    for raw_item in list(result.get("items") or []):
        item = dict(raw_item) if isinstance(raw_item, dict) else {}
        if str(item.get("key") or "") in {"db_dir", "chat_db", "library_db", "user_issues_db", "frontend_build"}:
            item["detail"] = "Hidden outside management access."
        items.append(item)
    result["items"] = items
    return result


def production_readiness_payload(s) -> dict:
    return _production_readiness_payload(s)


@router.get("/settings")
def get_all_settings(request: Request = None):
    s = get_settings()
    prefs = load_prefs()
    has_management_access = bool(request is not None and management_api_allowed(request))
    return {
        "model": s.model,
        "base_url": s.base_url,
        "has_api_key": bool(s.api_key),
        "connection": _connection_status(s),
        "readiness": _readiness_payload(s),
        "app_readiness": _public_readiness_payload(
            _production_readiness_payload(s),
            reveal_paths=has_management_access,
        ),
        "db_dir": str(s.db_dir) if has_management_access else "",
        "prefs": _public_prefs(prefs, include_paths=has_management_access),
    }


@router.get("/settings/readiness")
def get_llm_readiness():
    return _readiness_payload(get_settings())


@router.get("/readiness")
def get_readiness(request: Request):
    return _public_readiness_payload(
        production_readiness_payload(get_settings()),
        reveal_paths=management_api_allowed(request),
    )


class PrefsPatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    top_k: int | None = Field(None, ge=2, le=20)
    temperature: float | None = Field(None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(None, ge=512, le=3072)
    deep_read: bool | None = None
    show_context: bool | None = None
    theme: str | None = Field(None, max_length=_MAX_HINT_CHARS)
    pdf_dir: str | None = Field(None, max_length=_MAX_PATH_PREF_CHARS)
    md_dir: str | None = Field(None, max_length=_MAX_PATH_PREF_CHARS)
    answer_contract_v1: bool | None = None
    answer_depth_auto: bool | None = None
    answer_mode_hint: str | None = Field(None, max_length=_MAX_HINT_CHARS)
    answer_output_mode: str | None = Field(None, max_length=_MAX_HINT_CHARS)
    refs_card_locale: str | None = Field(None, max_length=_MAX_HINT_CHARS)
    ui_locale: str | None = Field(None, max_length=_MAX_HINT_CHARS)
    sidebar_collapsed: bool | None = None
    text_api_key: str | None = Field(None, max_length=_MAX_API_KEY_CHARS)
    text_base_url: str | None = Field(None, max_length=_MAX_BASE_URL_CHARS)
    text_model: str | None = Field(None, max_length=_MAX_MODEL_CHARS)
    vision_api_key: str | None = Field(None, max_length=_MAX_API_KEY_CHARS)
    vision_base_url: str | None = Field(None, max_length=_MAX_BASE_URL_CHARS)
    vision_model: str | None = Field(None, max_length=_MAX_MODEL_CHARS)
    auto_backup_enabled: bool | None = None
    quality_data_sharing_enabled: bool | None = None

    @field_validator("theme", "answer_mode_hint", "answer_output_mode", "refs_card_locale", "ui_locale")
    @classmethod
    def _clean_small_choice(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return str(value).replace("\x00", "").strip().lower()

    @field_validator("text_api_key", "vision_api_key", "text_base_url", "vision_base_url", "text_model", "vision_model", "pdf_dir", "md_dir")
    @classmethod
    def _clean_text_value(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return str(value).replace("\x00", "").strip()


@router.patch("/settings", dependencies=[Depends(require_management_api)])
def update_settings(body: PrefsPatch):
    prefs = load_prefs()
    patch = body.model_dump(exclude_none=True)
    previous_quality_sharing_enabled = _pref_bool(prefs.get("quality_data_sharing_enabled"))
    quality_sharing_disabled = False
    quality_sharing_enabled_from_off = False
    quality_cleanup_result: dict | None = None
    for k, v in patch.items():
        normalized = _normalize_pref_value(k, v)
        if k in (_API_KEY_PREF_KEYS | _LLM_PREF_KEYS) and not normalized:
            prefs.pop(k, None)
        else:
            prefs[k] = normalized
        if k == "quality_data_sharing_enabled" and normalized is False:
            quality_sharing_disabled = True
            prefs.pop("quality_data_client_id", None)
        elif k == "quality_data_sharing_enabled" and normalized is True and not str(prefs.get("quality_data_client_id") or "").strip():
            quality_sharing_enabled_from_off = not previous_quality_sharing_enabled
            prefs["quality_data_client_id"] = secrets.token_urlsafe(18)
        elif k == "quality_data_sharing_enabled" and normalized is True:
            quality_sharing_enabled_from_off = not previous_quality_sharing_enabled
    if quality_sharing_enabled_from_off:
        _discard_stale_quality_data_before_enable()
    save_prefs(prefs)
    if quality_sharing_disabled:
        quality_cleanup_result = _discard_unsent_quality_data_outbox()
    try:
        get_settings.cache_clear()
    except Exception:
        pass
    response: dict[str, object] = {"ok": True}
    if quality_cleanup_result is not None:
        response["quality_data_cleanup"] = {
            "ok": bool(quality_cleanup_result.get("ok")),
            "removed": int(quality_cleanup_result.get("removed") or 0),
            "error": str(quality_cleanup_result.get("error") or "")[:240],
        }
    return response


class PickDirRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    target: Literal["pdf", "md"]
    initial_dir: str | None = Field(None, max_length=_MAX_PATH_PREF_CHARS)


def _require_server_file_picker_allowed(request: Request) -> None:
    settings = get_settings()
    if not bool(getattr(settings, "production", False)):
        return
    if management_api_allowed(request):
        return
    if bool(getattr(settings, "auth_required", False)) and request_is_authenticated(request, settings=settings):
        return
    raise HTTPException(status_code=404, detail="Not found")


@router.post("/settings/pick-dir", dependencies=[Depends(require_management_api)])
def pick_dir(body: PickDirRequest, request: Request):
    _require_server_file_picker_allowed(request)
    prefs = load_prefs()
    key = "pdf_dir" if body.target == "pdf" else "md_dir"
    initial = (body.initial_dir or "").strip() or str(prefs.get(key) or "").strip()
    picked = _pick_directory_dialog(initial)
    if not picked:
        return {"ok": False, "path": None}
    return {"ok": True, "path": picked}


class ConnectionTestBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    target: Literal["text", "vision"] = "text"
    api_key: str | None = Field(None, max_length=_MAX_API_KEY_CHARS)
    base_url: str | None = Field(None, max_length=_MAX_BASE_URL_CHARS)
    model: str | None = Field(None, max_length=_MAX_MODEL_CHARS)

    @field_validator("api_key", "base_url", "model")
    @classmethod
    def _clean_override(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return str(value).replace("\x00", "").strip()


def _test_chat_completion(*, api_key: str | None, base_url: str, model: str, timeout_s: float) -> dict:
    if not api_key:
        return {"ok": False, "error": "API key is missing", "error_type": "auth"}
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Hi, reply OK in one word."}],
        temperature=0.0,
        max_tokens=16,
        timeout=timeout_s,
    )
    reply = (resp.choices[0].message.content or "").strip()
    return {"ok": True, "reply": reply}


@router.post("/settings/test-llm", dependencies=[Depends(require_management_api)])
def test_llm(body: ConnectionTestBody | None = None):
    checked_at = time.time()
    target = body.target if body else "text"
    api_key: str | None = None
    base_url = ""
    model = ""
    try:
        s = get_settings()
        override_api_key = _normalize_pref_value("text_api_key", body.api_key) if body and body.api_key is not None else ""
        override_base_url = _normalize_pref_value("text_base_url", body.base_url) if body and body.base_url is not None else ""
        override_model = _normalize_pref_value("text_model", body.model) if body and body.model is not None else ""
        if target == "vision":
            api_key = override_api_key or s.vision_api_key
            base_url = override_base_url or s.vision_base_url
            model = override_model or s.vision_model
            result = _test_chat_completion(
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout_s=s.timeout_s,
            )
        else:
            api_key = override_api_key or s.text_api_key
            base_url = override_base_url or s.text_base_url
            model = override_model or s.text_model
            result = _test_chat_completion(
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout_s=s.timeout_s,
            )
        error_type = str(result.get("error_type") or "")
        if not result.get("ok") and not error_type:
            error_type = _classify_connection_error(result.get("error"))
            result["error_type"] = error_type
        if not result.get("ok"):
            result["error"] = _public_error_text(result.get("error"))
        result["checked_at"] = checked_at
        _LLM_TEST_RESULTS[target] = {
            "ok": bool(result.get("ok")),
            "reply": str(result.get("reply") or ""),
            "error": str(result.get("error") or ""),
            "error_type": error_type,
            "checked_at": checked_at,
            "fingerprint": _provider_fingerprint(api_key=api_key, base_url=base_url, model=model),
        }
        return result
    except Exception as e:
        error_type = _classify_connection_error(e)
        error_text = _public_error_text(e)
        _LLM_TEST_RESULTS[target] = {
            "ok": False,
            "reply": "",
            "error": error_text,
            "error_type": error_type,
            "checked_at": checked_at,
            "fingerprint": _provider_fingerprint(api_key=api_key, base_url=base_url, model=model),
        }
        return {"ok": False, "error": error_text, "error_type": error_type, "checked_at": checked_at}


@router.get("/health")
def health():
    s = get_settings()
    auth_required = bool(getattr(s, "auth_required", False))
    return {
        "status": "ok",
        "version": read_app_version(),
        "env": str(getattr(s, "app_env", "development") or "development"),
        "production": bool(getattr(s, "production", False)),
        "auth": {
            "required": auth_required,
            "configured": auth_token_configured(s) if auth_required else False,
        },
    }
