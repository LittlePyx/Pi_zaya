from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import re
import secrets
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import requests


_WINDOWS_PATH_RE = re.compile(r"(^|[\s(\"'=])([A-Za-z]:[\\/][^\s\"'<>|]+)")
_FILE_URL_RE = re.compile(r"file:\/\/\/[^\s\"'<>|]+", flags=re.IGNORECASE)
_UNC_PATH_RE = re.compile(r"\\\\[^\s\"'<>|]+")
_UNIX_PATH_RE = re.compile(r"(^|[\s(\"'=])(/(?:Users|home|mnt|var|tmp|private)/[^\s\"'<>]+)", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_AUTH_SECRET_RE = re.compile(
    r"\b((?:authorization|x[-_]?api[-_]?key|api[-_]?key|access[-_]?token|refresh[-_]?token|cookie|set-cookie)\s*[:=]\s*)"
    r"(?:bearer\s+)?[A-Za-z0-9._~+/=\-]{8,}",
    flags=re.IGNORECASE,
)
_BEARER_RE = re.compile(r"\bbearer\s+[A-Za-z0-9._~+/=\-]{8,}", flags=re.IGNORECASE)
_TOKEN_RE = re.compile(r"\b(?:sk|pk|ghp|github_pat|xoxb|xoxp|ya29|AIza)[A-Za-z0-9_\-]{12,}\b")
_LONG_HASH_RE = re.compile(r"\b[A-Fa-f0-9]{32,}\b")
_HTTP_URL_RE = re.compile(r"https?://[^\s\"'<>]+", flags=re.IGNORECASE)
_URL_QUERY_RE = re.compile(r"(https?://[^\s?#]+)(?:[?#][^ \t\r\n\"'<>]*)?")
_SENSITIVE_KEY_RE = re.compile(
    r"(?:api[_-]?key|token|secret|password|authorization|cookie|"
    r"(?:^|[_-])user[_-]?agent(?:$|[_-])|^ua$|browser[_-]?agent|"
    r"pdf[_-]?path|md[_-]?path|"
    r"source[_-]?path|absolute[_-]?path|local[_-]?path|file[_-]?path|path|"
    r"pdf[_-]?name|md[_-]?name|source[_-]?name|document[_-]?name|file[_-]?name|filename|"
    r"(?:^|[_-])(?:title|main|raw|prompt|query|question|answer|message|content|body|excerpt|quote|abstract)"
    r"(?:$|[_-]?(?:text|markdown|content|body|raw)$)|"
    r"(?:pdf|md|markdown|raw|full|source|document|page)[_-]?text)$",
    flags=re.IGNORECASE,
)
_FREEFORM_SAMPLE_KEY_RE = re.compile(
    r"(?:^|[_-])(?:sample|samples|example|examples|evidence|snippet|snippets)"
    r"(?:$|[_-]?(?:text|texts|markdown|content|body|raw|items?|list|names?|values?)$)",
    flags=re.IGNORECASE,
)
_DOCUMENT_COLLECTION_KEY_RE = re.compile(
    r"(?:^|[_-])(?:paper|papers|document|documents|file|files)"
    r"(?:$|[_-]?(?:list|names?|titles?|items?)$)|"
    r"(?:^|[_-])(?:source|sources)[_-](?:list|names?|titles?|items?)$",
    flags=re.IGNORECASE,
)
_REMOTE_PAYLOAD_DICT_LIMIT = 100
_REMOTE_PAYLOAD_LIST_LIMIT = 20
_REMOTE_PAYLOAD_STRING_LIMIT = 500
_PREFS_WRITE_LOCK = threading.Lock()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _remote_token_configured() -> bool:
    return bool(str(os.environ.get("KB_USER_ISSUES_REMOTE_TOKEN") or "").strip())


def _allow_unauthenticated_remote() -> bool:
    return _env_bool("KB_USER_ISSUES_ALLOW_UNAUTHENTICATED_REMOTE", False)


def _prefs_path() -> Path:
    configured = str(os.environ.get("KB_USER_PREFS_PATH") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parent.parent / "user_prefs.json"


def _read_prefs() -> dict[str, Any]:
    try:
        data = json.loads(_prefs_path().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def _write_prefs(data: Mapping[str, Any]) -> None:
    path = _prefs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(data), ensure_ascii=False, indent=2)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(str(tmp_path), str(path))
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _bool_value(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    raw = str(value or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _prefs_bool(name: str, default: bool = False) -> bool:
    return _bool_value(_read_prefs().get(name), default)


def _prefs_text(name: str, default: str = "") -> str:
    data = _read_prefs()
    return str(data.get(name) or default or "").strip()


def _quality_data_client_id() -> str:
    env_client_id = str(os.environ.get("KB_USER_ISSUES_CLIENT_ID") or "").strip()
    if env_client_id:
        return env_client_id

    with _PREFS_WRITE_LOCK:
        prefs = _read_prefs()
        if not _bool_value(prefs.get("quality_data_sharing_enabled"), False):
            return str(prefs.get("quality_data_client_id") or "").strip()
        existing = str(prefs.get("quality_data_client_id") or "").strip()
        if existing:
            return existing
        generated = secrets.token_urlsafe(18)
        prefs["quality_data_client_id"] = generated
        try:
            _write_prefs(prefs)
        except Exception:
            pass
        return generated


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except Exception:
        return float(default)


def _clean_text(value: Any, *, limit: int = 2000) -> str:
    text = str(value if value is not None else "").replace("\x00", " ")
    text = _URL_QUERY_RE.sub(r"\1", text)
    text = _FILE_URL_RE.sub("[local-path]", text)
    text = _UNC_PATH_RE.sub("[local-path]", text)
    text = _WINDOWS_PATH_RE.sub(r"\1[local-path]", text)
    text = _UNIX_PATH_RE.sub(r"\1[local-path]", text)
    text = _EMAIL_RE.sub("[email]", text)
    text = _AUTH_SECRET_RE.sub(r"\1[token]", text)
    text = _BEARER_RE.sub("Bearer [token]", text)
    text = _TOKEN_RE.sub("[token]", text)
    text = _LONG_HASH_RE.sub("[hash]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[: int(max(0, limit))]


def _safe_scalar(value: Any, *, limit: int = _REMOTE_PAYLOAD_STRING_LIMIT) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return _clean_text(value, limit=limit)


def _clean_identifier(value: Any, *, limit: int = 128) -> str:
    text = str(value if value is not None else "").replace("\x00", " ").strip()
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text[: int(max(0, limit))]


def _clean_route(value: Any, *, limit: int = 500) -> str:
    text = _clean_text(value, limit=limit)
    if not text:
        return ""
    for sep in ("?", "#"):
        idx = text.find(sep)
        if idx >= 0:
            text = text[:idx]
    return text[: int(max(0, limit))].strip()


def _payload_key_requires_redaction(key: str, value: Any) -> bool:
    if _SENSITIVE_KEY_RE.search(key) or _FREEFORM_SAMPLE_KEY_RE.search(key):
        return True
    if _DOCUMENT_COLLECTION_KEY_RE.search(key):
        return not (value is None or isinstance(value, (bool, int, float)))
    return False


def _safe_payload(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "[depth-limit]"
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, raw_val in list(value.items())[:_REMOTE_PAYLOAD_DICT_LIMIT]:
            clean_key = _clean_text(key, limit=120)
            if not clean_key:
                continue
            if _payload_key_requires_redaction(clean_key, raw_val):
                out[clean_key] = "[redacted]"
                continue
            out[clean_key] = _safe_payload(raw_val, depth=depth + 1)
        return out
    if isinstance(value, list):
        return [_safe_payload(item, depth=depth + 1) for item in value[:_REMOTE_PAYLOAD_LIST_LIMIT]]
    return _safe_scalar(value)


def _stable_client_id(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:32]


def _fingerprint_contains_sensitive_text(value: Any) -> bool:
    text = str(value if value is not None else "")
    if not text:
        return False
    if (
        _WINDOWS_PATH_RE.search(text)
        or _FILE_URL_RE.search(text)
        or _UNC_PATH_RE.search(text)
        or _UNIX_PATH_RE.search(text)
        or _EMAIL_RE.search(text)
        or _AUTH_SECRET_RE.search(text)
        or _BEARER_RE.search(text)
        or _TOKEN_RE.search(text)
        or _HTTP_URL_RE.search(text)
    ):
        return True
    return _URL_QUERY_RE.sub(r"\1", text) != text


def _safe_fingerprint(value: Any) -> str:
    if value is None:
        return ""
    if _fingerprint_contains_sensitive_text(value):
        redacted = _clean_text(value, limit=1000).lower()
        return "fp-" + hashlib.sha256(redacted.encode("utf-8", errors="ignore")).hexdigest()[:32]
    return _clean_identifier(value, limit=128)


def _remote_issue_summary(source: str, summary: Any) -> str:
    clean_source = _clean_identifier(source, limit=120).lower()
    if clean_source == "research_qa_failure_case":
        return "Research QA failure"
    if clean_source == "frontend":
        return "Frontend issue"
    return _clean_text(summary, limit=500)


def _remote_issue_detail(source: str, detail: Any) -> str:
    clean_source = _clean_identifier(source, limit=120).lower()
    if clean_source == "frontend":
        return ""
    return _clean_text(detail, limit=1200)


def _is_local_or_private_host(host_key: str) -> bool:
    host = str(host_key or "").strip("[]").lower().split("%", 1)[0]
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return (
            host in {"localhost"}
            or host.endswith(".localhost")
            or host.endswith(".local")
        )
    return bool(
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_unspecified
        or ip.is_reserved
        or ip.is_multicast
    )


def _remote_url_state() -> dict[str, Any]:
    url = str(os.environ.get("KB_USER_ISSUES_REMOTE_URL") or "").strip()
    host = ""
    host_key = ""
    scheme = ""
    has_credentials = False
    has_valid_port = True
    if url:
        try:
            parsed = urlparse(url)
            scheme = str(parsed.scheme or "").strip().lower()
            has_credentials = bool(parsed.username or parsed.password)
            try:
                port = parsed.port
            except ValueError:
                port = None
                has_valid_port = False
            host_key = (parsed.hostname or "").strip("[]").lower()
            if host_key:
                host = f"{host_key}:{port}" if port is not None else host_key
            else:
                host = (parsed.netloc or parsed.path.split("/", 1)[0]).rsplit("@", 1)[-1]
                host_key = host.split(":", 1)[0].strip("[]").lower()
        except Exception:
            host = ""
            host_key = ""
            scheme = ""
            has_credentials = False
            has_valid_port = False
    is_local = _is_local_or_private_host(host_key)
    has_valid_scheme = scheme in {"http", "https"}
    local_allowed = _env_bool("KB_USER_ISSUES_ALLOW_LOCAL_REMOTE", False)
    secure_transport = scheme == "https" or (is_local and local_allowed and scheme == "http")
    valid_target = bool(host_key and has_valid_scheme and has_valid_port and not has_credentials)
    return {
        "url": url,
        "host": host,
        "host_key": host_key,
        "scheme": scheme,
        "has_valid_scheme": has_valid_scheme,
        "has_valid_port": has_valid_port,
        "has_credentials": has_credentials,
        "is_local": is_local,
        "local_allowed": local_allowed,
        "secure_transport": secure_transport,
        "allowed": bool(url and valid_target and ((not is_local) or local_allowed) and secure_transport),
    }


def _remote_url_resolution_error(url_state: Mapping[str, Any]) -> str:
    host_key = str(url_state.get("host_key") or "").strip("[]").lower()
    if not host_key or bool(url_state.get("local_allowed")):
        return ""
    if _is_local_or_private_host(host_key):
        return "remote host is local/private"
    try:
        infos = socket.getaddrinfo(host_key, None, type=socket.SOCK_STREAM)
    except Exception as exc:
        return f"remote host DNS lookup failed: {_clean_text(exc, limit=240)}"
    addresses: set[str] = set()
    for info in infos:
        try:
            sockaddr = info[4]
            address = str(sockaddr[0] if sockaddr else "").strip()
        except Exception:
            address = ""
        if not address or address in addresses:
            continue
        addresses.add(address)
        if _is_local_or_private_host(address):
            return "remote host resolves to local/private address"
    return ""


def user_issue_remote_enabled() -> bool:
    url_state = _remote_url_state()
    return bool(
        _env_bool("KB_USER_ISSUES_REMOTE_ENABLED", False)
        and url_state["allowed"]
        and (_remote_token_configured() or _allow_unauthenticated_remote())
        and _prefs_bool("quality_data_sharing_enabled", False)
    )


def user_issue_quality_data_sharing_enabled() -> bool:
    return _prefs_bool("quality_data_sharing_enabled", False)


def user_issue_remote_status() -> dict[str, Any]:
    url_state = _remote_url_state()
    quality_data_sharing = user_issue_quality_data_sharing_enabled()
    remote_enabled = _env_bool("KB_USER_ISSUES_REMOTE_ENABLED", False)
    token_configured = _remote_token_configured()
    allow_unauthenticated = _allow_unauthenticated_remote()
    if not remote_enabled:
        block_reason = "env_disabled"
    elif not url_state["url"]:
        block_reason = "missing_remote_url"
    elif not url_state["has_valid_scheme"]:
        block_reason = "invalid_remote_url"
    elif not url_state["has_valid_port"] or not url_state["host_key"]:
        block_reason = "invalid_remote_url"
    elif url_state["has_credentials"]:
        block_reason = "remote_url_credentials"
    elif url_state["is_local"] and not url_state["local_allowed"]:
        block_reason = "local_remote_url"
    elif not url_state["secure_transport"]:
        block_reason = "insecure_remote_url"
    elif not quality_data_sharing:
        block_reason = "user_opt_out"
    elif not token_configured and not allow_unauthenticated:
        block_reason = "missing_remote_token"
    else:
        block_reason = ""
    return {
        "enabled": user_issue_remote_enabled(),
        "remote_enabled": remote_enabled,
        "remote_url_configured": bool(url_state["url"]),
        "remote_url_host": _clean_text(url_state["host"], limit=160),
        "remote_url_scheme": _clean_text(url_state["scheme"], limit=20),
        "remote_url_has_valid_scheme": bool(url_state["has_valid_scheme"]),
        "remote_url_has_valid_port": bool(url_state["has_valid_port"]),
        "remote_url_has_credentials": bool(url_state["has_credentials"]),
        "remote_url_is_local": bool(url_state["is_local"]),
        "remote_url_local_allowed": bool(url_state["local_allowed"]),
        "remote_url_secure": bool(url_state["secure_transport"]),
        "remote_url_allowed": bool(url_state["allowed"]),
        "remote_block_reason": block_reason,
        "remote_token_configured": token_configured,
        "remote_token_required": not allow_unauthenticated,
        "remote_unauthenticated_allowed": allow_unauthenticated,
        "quality_data_sharing_enabled": quality_data_sharing,
    }


def build_remote_issue_payload(issue: Mapping[str, Any]) -> dict[str, Any]:
    client_id = _stable_client_id(_quality_data_client_id())
    project_channel = _clean_text(os.environ.get("KB_USER_ISSUES_CLIENT_CHANNEL") or "", limit=120)
    issue_source = _clean_text(issue.get("source"), limit=120)
    payload = {
        "schema": "pi-zaya.user_issue.v1",
        "client": {
            "installation_id": client_id,
            "channel": project_channel,
            "quality_data_sharing": _prefs_bool("quality_data_sharing_enabled", False),
        },
        "issue": {
            "fingerprint": _safe_fingerprint(issue.get("fingerprint")),
            "source": issue_source,
            "domain": _clean_text(issue.get("domain"), limit=120),
            "severity": _clean_text(issue.get("severity"), limit=40),
            "status": _clean_text(issue.get("status"), limit=40),
            "summary": _remote_issue_summary(issue_source, issue.get("summary")),
            "detail": _remote_issue_detail(issue_source, issue.get("detail")),
            "route": _clean_route(issue.get("route"), limit=500),
            "first_seen_at": _safe_scalar(issue.get("first_seen_at")),
            "last_seen_at": _safe_scalar(issue.get("last_seen_at")),
            "occurrence_count": _safe_scalar(issue.get("occurrence_count")),
            "context": _safe_payload(issue.get("context") if isinstance(issue.get("context"), Mapping) else {}),
            "payload": _safe_payload(issue.get("payload") if isinstance(issue.get("payload"), Mapping) else {}),
        },
    }
    return payload


def build_remote_smoke_test_payload() -> dict[str, Any]:
    return build_remote_issue_payload(
        {
            "fingerprint": f"collector-smoke-test-{int(time.time())}",
            "source": "collector_smoke_test",
            "domain": "quality_data",
            "severity": "info",
            "status": "open",
            "summary": "Quality collector smoke test",
            "detail": "Manual collector connectivity check from settings.",
            "route": "/settings/privacy",
            "context": {
                "kind": "collector_smoke_test",
                "created_at": time.time(),
            },
            "payload": {
                "test": True,
            },
        }
    )


def post_remote_issue_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    url = str(os.environ.get("KB_USER_ISSUES_REMOTE_URL") or "").strip()
    if not url:
        return {"ok": False, "enabled": False, "status_code": 0, "error": "remote URL is not configured"}
    remote_status = user_issue_remote_status()
    if not bool(remote_status.get("enabled")):
        return {
            "ok": False,
            "enabled": False,
            "status_code": 0,
            "error": str(remote_status.get("remote_block_reason") or "remote reporting is disabled"),
        }
    resolution_error = _remote_url_resolution_error(_remote_url_state())
    if resolution_error:
        return {"ok": False, "enabled": True, "status_code": 0, "error": resolution_error}
    token = str(os.environ.get("KB_USER_ISSUES_REMOTE_TOKEN") or "").strip()
    timeout_s = max(0.5, min(15.0, _env_float("KB_USER_ISSUES_REMOTE_TIMEOUT_S", 2.5)))
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Pi-zaya-KB/1.0 user-issue-reporter",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.post(url, json=dict(payload), headers=headers, timeout=timeout_s, allow_redirects=False)
    except Exception as exc:
        return {"ok": False, "enabled": True, "status_code": 0, "error": _clean_text(exc, limit=500)}
    status_code = int(getattr(resp, "status_code", 0) or 0)
    if 200 <= status_code < 300:
        return {"ok": True, "enabled": True, "status_code": status_code, "error": ""}
    if 300 <= status_code < 400:
        return {
            "ok": False,
            "enabled": True,
            "status_code": status_code,
            "error": "remote redirects are not allowed",
        }
    try:
        text = str(getattr(resp, "text", "") or "")
    except Exception:
        text = ""
    return {
        "ok": False,
        "enabled": True,
        "status_code": status_code,
        "error": _clean_text(text or f"HTTP {status_code}", limit=500),
    }
