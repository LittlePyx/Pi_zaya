from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from kb.user_issue_remote import (
    build_remote_issue_payload,
    post_remote_issue_payload,
    user_issue_quality_data_sharing_enabled,
    user_issue_remote_enabled,
)

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
_SENSITIVE_PAYLOAD_KEY_RE = re.compile(
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
_ISSUE_JSON_MAX_CHARS = 20_000
_ISSUE_PAYLOAD_DICT_LIMIT = 100
_ISSUE_PAYLOAD_LIST_LIMIT = 20
_ISSUE_PAYLOAD_STRING_LIMIT = 500
_DEFAULT_MAX_ISSUES = 5_000
_DEFAULT_MAX_EVENTS = 20_000
_DEFAULT_MAX_SENT_OUTBOX = 5_000
_DEFAULT_SENT_OUTBOX_RETENTION_DAYS = 30.0


def _clean_text(value: Any, *, limit: int = 2000) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ").strip()
    if len(text) > limit:
        return text[:limit]
    return text


def _redact_text(value: Any, *, limit: int = 2000) -> str:
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


def _clean_identifier(value: Any, *, limit: int = 128) -> str:
    text = str(value if value is not None else "").replace("\x00", " ").strip()
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text[: int(max(0, limit))]


def _clean_route(value: Any, *, limit: int = 500) -> str:
    text = _redact_text(value, limit=limit)
    if not text:
        return ""
    for sep in ("?", "#"):
        idx = text.find(sep)
        if idx >= 0:
            text = text[:idx]
    return text[: int(max(0, limit))].strip()


def _safe_issue_scalar(value: Any, *, limit: int = _ISSUE_PAYLOAD_STRING_LIMIT) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return _redact_text(value, limit=limit)


def _payload_key_requires_redaction(key: str, value: Any) -> bool:
    if _SENSITIVE_PAYLOAD_KEY_RE.search(key) or _FREEFORM_SAMPLE_KEY_RE.search(key):
        return True
    if _DOCUMENT_COLLECTION_KEY_RE.search(key):
        return not (value is None or isinstance(value, (bool, int, float)))
    return False


def _safe_issue_payload(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "[depth-limit]"
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, raw_val in list(value.items())[:_ISSUE_PAYLOAD_DICT_LIMIT]:
            clean_key = _redact_text(key, limit=120)
            if not clean_key:
                continue
            if _payload_key_requires_redaction(clean_key, raw_val):
                out[clean_key] = "[redacted]"
                continue
            out[clean_key] = _safe_issue_payload(raw_val, depth=depth + 1)
        return out
    if isinstance(value, list):
        return [_safe_issue_payload(item, depth=depth + 1) for item in value[:_ISSUE_PAYLOAD_LIST_LIMIT]]
    return _safe_issue_scalar(value)


def _safe_json(value: Any, *, max_chars: int = _ISSUE_JSON_MAX_CHARS) -> str:
    try:
        text = json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = json.dumps({"value": _redact_text(value, limit=4000)}, ensure_ascii=False, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return json.dumps(
        {
            "_truncated": True,
            "preview": text[: max(0, int(max_chars))],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _loads_json(value: str) -> Any:
    try:
        return json.loads(value or "{}")
    except Exception:
        return {}


def _severity(value: Any) -> str:
    text = _clean_text(value, limit=40).lower()
    if text in {"error", "warning", "info"}:
        return text
    if text in {"failed", "blocked", "critical"}:
        return "error"
    if text in {"warn", "attention", "review"}:
        return "warning"
    return "info"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except Exception:
        return default


def _bounded_env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 100_000) -> int:
    try:
        raw = int(os.environ.get(name, "") or default)
    except Exception:
        raw = int(default)
    return max(int(min_value), min(int(max_value), int(raw)))


def _bounded_env_float(name: str, default: float, *, min_value: float = 0.0, max_value: float = 3650.0) -> float:
    try:
        raw = float(os.environ.get(name, "") or default)
    except Exception:
        raw = float(default)
    return max(float(min_value), min(float(max_value), float(raw)))


def _fingerprint(parts: list[Any]) -> str:
    joined = "\n".join(_clean_text(part, limit=1000).lower() for part in parts)
    return hashlib.sha256(joined.encode("utf-8", errors="ignore")).hexdigest()


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


def _row_to_issue(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "fingerprint": str(row["fingerprint"]),
        "source": str(row["source"]),
        "domain": str(row["domain"]),
        "severity": str(row["severity"]),
        "status": str(row["status"]),
        "summary": str(row["summary"]),
        "detail": str(row["detail"] or ""),
        "first_seen_at": float(row["first_seen_at"]),
        "last_seen_at": float(row["last_seen_at"]),
        "occurrence_count": int(row["occurrence_count"]),
        "context": _loads_json(str(row["context_json"] or "{}")),
        "payload": _loads_json(str(row["payload_json"] or "{}")),
    }


def _remote_retry_delay_s(attempts: int) -> float:
    clean = max(1, int(attempts or 1))
    return float(min(3600, 30 * (2 ** min(clean - 1, 7))))


def _remote_outbox_flush_interval_s() -> float:
    raw = str(os.environ.get("KB_USER_ISSUES_OUTBOX_FLUSH_INTERVAL_S") or "").strip()
    if not raw:
        return 60.0
    try:
        return max(0.0, min(3600.0, float(raw)))
    except Exception:
        return 60.0


def _remote_outbox_claim_lease_s() -> float:
    raw = str(os.environ.get("KB_USER_ISSUES_OUTBOX_CLAIM_LEASE_S") or "").strip()
    if not raw:
        return 120.0
    try:
        return max(5.0, min(3600.0, float(raw)))
    except Exception:
        return 120.0


def _issue_event_coalesce_window_s() -> float:
    raw = str(os.environ.get("KB_USER_ISSUES_EVENT_COALESCE_S") or "").strip()
    if not raw:
        return 60.0
    try:
        return max(0.0, min(3600.0, float(raw)))
    except Exception:
        return 60.0


def _max_local_issues() -> int:
    return _bounded_env_int("KB_USER_ISSUES_MAX_ISSUES", _DEFAULT_MAX_ISSUES)


def _max_local_events() -> int:
    return _bounded_env_int("KB_USER_ISSUES_MAX_EVENTS", _DEFAULT_MAX_EVENTS)


def _max_sent_remote_outbox() -> int:
    return _bounded_env_int("KB_USER_ISSUES_MAX_SENT_OUTBOX", _DEFAULT_MAX_SENT_OUTBOX)


def _sent_remote_outbox_retention_s() -> float:
    days = _bounded_env_float(
        "KB_USER_ISSUES_SENT_OUTBOX_RETENTION_DAYS",
        _DEFAULT_SENT_OUTBOX_RETENTION_DAYS,
    )
    return days * 24 * 60 * 60


_OUTBOX_WORKER_LOCK = threading.Lock()
_OUTBOX_WORKER_THREAD: threading.Thread | None = None


class UserIssueStore:
    """Durable local issue log for user-facing problems and hidden diagnostics."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_issues (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  fingerprint TEXT NOT NULL UNIQUE,
                  source TEXT NOT NULL,
                  domain TEXT NOT NULL,
                  severity TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'open',
                  summary TEXT NOT NULL,
                  detail TEXT NOT NULL DEFAULT '',
                  first_seen_at REAL NOT NULL,
                  last_seen_at REAL NOT NULL,
                  occurrence_count INTEGER NOT NULL DEFAULT 1,
                  context_json TEXT NOT NULL DEFAULT '{}',
                  payload_json TEXT NOT NULL DEFAULT '{}'
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_issue_events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  issue_id INTEGER NOT NULL,
                  created_at REAL NOT NULL,
                  route TEXT NOT NULL DEFAULT '',
                  user_agent TEXT NOT NULL DEFAULT '',
                  context_json TEXT NOT NULL DEFAULT '{}',
                  payload_json TEXT NOT NULL DEFAULT '{}',
                  FOREIGN KEY(issue_id) REFERENCES user_issues(id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_issue_remote_outbox (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  issue_id INTEGER NOT NULL,
                  event_id INTEGER NOT NULL UNIQUE,
                  status TEXT NOT NULL DEFAULT 'pending',
                  payload_json TEXT NOT NULL DEFAULT '{}',
                  attempts INTEGER NOT NULL DEFAULT 0,
                  next_attempt_at REAL NOT NULL DEFAULT 0,
                  last_error TEXT NOT NULL DEFAULT '',
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL,
                  sent_at REAL NOT NULL DEFAULT 0,
                  FOREIGN KEY(issue_id) REFERENCES user_issues(id) ON DELETE CASCADE,
                  FOREIGN KEY(event_id) REFERENCES user_issue_events(id) ON DELETE CASCADE
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_issues_last_seen ON user_issues(last_seen_at DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_issues_status ON user_issues(status, severity, last_seen_at DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_issue_events_issue ON user_issue_events(issue_id, created_at DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_issue_remote_outbox_due ON user_issue_remote_outbox(status, next_attempt_at, created_at);")

    def _prune_history(self, conn: sqlite3.Connection, *, now: float) -> None:
        sent_retention_s = _sent_remote_outbox_retention_s()
        if sent_retention_s > 0:
            conn.execute(
                """
                DELETE FROM user_issue_remote_outbox
                WHERE status='sent'
                  AND sent_at > 0
                  AND sent_at < ?;
                """,
                (now - sent_retention_s,),
            )
        max_sent_outbox = _max_sent_remote_outbox()
        if max_sent_outbox > 0:
            conn.execute(
                """
                DELETE FROM user_issue_remote_outbox
                WHERE id IN (
                  SELECT id
                  FROM user_issue_remote_outbox
                  WHERE status='sent'
                  ORDER BY sent_at DESC, id DESC
                  LIMIT -1 OFFSET ?
                );
                """,
                (max_sent_outbox,),
            )
        max_events = _max_local_events()
        if max_events > 0:
            conn.execute(
                """
                DELETE FROM user_issue_events
                WHERE id IN (
                  SELECT e.id
                  FROM user_issue_events e
                  WHERE NOT EXISTS (
                    SELECT 1
                    FROM user_issue_remote_outbox o
                    WHERE o.event_id=e.id
                      AND o.status!='sent'
                  )
                  ORDER BY e.created_at DESC, e.id DESC
                  LIMIT -1 OFFSET ?
                );
                """,
                (max_events,),
            )
        max_issues = _max_local_issues()
        if max_issues > 0:
            conn.execute(
                """
                DELETE FROM user_issues
                WHERE id IN (
                  SELECT ui.id
                  FROM user_issues ui
                  WHERE NOT EXISTS (
                    SELECT 1
                    FROM user_issue_remote_outbox o
                    WHERE o.issue_id=ui.id
                      AND o.status!='sent'
                  )
                  ORDER BY ui.last_seen_at DESC, ui.id DESC
                  LIMIT -1 OFFSET ?
                );
                """,
                (max_issues,),
            )

    def _enqueue_remote_outbox(
        self,
        conn: sqlite3.Connection,
        *,
        issue_id: int,
        event_id: int,
        issue: Mapping[str, Any],
        now: float,
    ) -> None:
        payload = build_remote_issue_payload(issue)
        conn.execute(
            """
            INSERT INTO user_issue_remote_outbox (
              issue_id, event_id, status, payload_json, attempts,
              next_attempt_at, last_error, created_at, updated_at, sent_at
            )
            VALUES (?, ?, 'pending', ?, 0, ?, '', ?, ?, 0)
            ON CONFLICT(event_id) DO UPDATE SET
              status='pending',
              payload_json=excluded.payload_json,
              next_attempt_at=excluded.next_attempt_at,
              updated_at=excluded.updated_at,
              last_error='';
            """,
            (issue_id, event_id, _safe_json(payload), now, now, now),
        )

    def flush_remote_outbox_async(self, *, limit: int = 20) -> None:
        db_path = self._db_path

        def _worker() -> None:
            try:
                UserIssueStore(db_path).flush_remote_outbox(limit=limit)
            except Exception:
                return

        threading.Thread(target=_worker, daemon=True).start()

    def _release_remote_outbox_claims(self, outbox_ids: list[int], *, error: str) -> int:
        if not outbox_ids:
            return 0
        now = time.time()
        clean_error = _clean_text(error, limit=500)
        released = 0
        with self._connect() as conn:
            for outbox_id in outbox_ids:
                cursor = conn.execute(
                    """
                    UPDATE user_issue_remote_outbox
                    SET status='pending',
                        next_attempt_at=?,
                        updated_at=?,
                        last_error=?
                    WHERE id=?
                      AND status='sending';
                    """,
                    (now, now, clean_error, int(outbox_id)),
                )
                released += int(cursor.rowcount or 0)
        return released

    def record_issue(
        self,
        *,
        source: str,
        domain: str,
        severity: str = "info",
        summary: str,
        detail: str = "",
        route: str = "",
        user_agent: str = "",
        context: Mapping[str, Any] | None = None,
        payload: Mapping[str, Any] | None = None,
        fingerprint: str = "",
        forward_remote: bool = True,
    ) -> dict[str, Any]:
        now = time.time()
        clean_source = _redact_text(source or "frontend", limit=120) or "frontend"
        clean_domain = _redact_text(domain or "general", limit=120) or "general"
        clean_severity = _severity(severity)
        clean_summary = _redact_text(summary or "User issue", limit=500) or "User issue"
        clean_detail = _redact_text(detail, limit=4000)
        clean_route = _clean_route(route, limit=500)
        clean_user_agent = _redact_text(user_agent, limit=500)
        fallback_fp = _fingerprint([clean_source, clean_domain, clean_severity, clean_summary, clean_detail[:1000]])
        supplied_fp = _clean_identifier(fingerprint, limit=128)
        issue_fp = fallback_fp if _fingerprint_contains_sensitive_text(fingerprint) else (supplied_fp or fallback_fp)
        clean_context = _safe_issue_payload(dict(context or {}))
        clean_payload = _safe_issue_payload(dict(payload or {}))
        context_json = _safe_json(clean_context)
        payload_json = _safe_json(clean_payload)
        should_flush_remote = False

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_issues (
                  fingerprint, source, domain, severity, status, summary, detail,
                  first_seen_at, last_seen_at, occurrence_count, context_json, payload_json
                )
                VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(fingerprint) DO UPDATE SET
                  source=excluded.source,
                  domain=excluded.domain,
                  severity=excluded.severity,
                  summary=excluded.summary,
                  detail=excluded.detail,
                  last_seen_at=excluded.last_seen_at,
                  occurrence_count=user_issues.occurrence_count + 1,
                  context_json=excluded.context_json,
                  payload_json=excluded.payload_json;
                """,
                (
                    issue_fp,
                    clean_source,
                    clean_domain,
                    clean_severity,
                    clean_summary,
                    clean_detail,
                    now,
                    now,
                    context_json,
                    payload_json,
                ),
            )
            row = conn.execute("SELECT * FROM user_issues WHERE fingerprint=?", (issue_fp,)).fetchone()
            issue_id = int(row["id"])
            event_id = 0
            remote_queue_now = bool(forward_remote) and user_issue_quality_data_sharing_enabled()
            remote_send_ready_now = remote_queue_now and user_issue_remote_enabled()
            latest_event = conn.execute(
                """
                SELECT id, created_at
                FROM user_issue_events
                WHERE issue_id=?
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                (issue_id,),
            ).fetchone()
            latest_event_has_outbox = False
            if latest_event and remote_queue_now:
                latest_event_has_outbox = bool(
                    conn.execute(
                        "SELECT 1 FROM user_issue_remote_outbox WHERE event_id=? LIMIT 1;",
                        (int(latest_event["id"]),),
                    ).fetchone()
                )
            coalesce_window_s = _issue_event_coalesce_window_s()
            coalesced = bool(
                latest_event
                and coalesce_window_s > 0
                and now - float(latest_event["created_at"] or 0.0) <= coalesce_window_s
                and (not remote_queue_now or latest_event_has_outbox)
            )
            if not coalesced:
                event_cursor = conn.execute(
                    """
                    INSERT INTO user_issue_events (
                      issue_id, created_at, route, user_agent, context_json, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (issue_id, now, clean_route, clean_user_agent, context_json, payload_json),
                )
                event_id = int(event_cursor.lastrowid or 0)
            issue = _row_to_issue(row)
            if event_id and remote_queue_now:
                remote_issue = {
                    **issue,
                    "route": clean_route,
                }
                self._enqueue_remote_outbox(
                    conn,
                    issue_id=issue_id,
                    event_id=event_id,
                    issue=remote_issue,
                    now=now,
                )
                should_flush_remote = remote_send_ready_now
            self._prune_history(conn, now=now)
        if should_flush_remote:
            try:
                self.flush_remote_outbox_async(limit=10)
            except Exception:
                pass
        return issue

    def flush_remote_outbox(self, *, limit: int = 20) -> dict[str, Any]:
        max_limit = max(1, min(200, int(limit or 20)))
        if not user_issue_remote_enabled():
            summary = self.remote_outbox_summary()
            return {"ok": False, "enabled": False, "sent": 0, "failed": 0, "summary": summary}
        now = time.time()
        lease_until = now + _remote_outbox_claim_lease_s()
        claimed_rows: list[sqlite3.Row] = []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM user_issue_remote_outbox
                WHERE status != 'sent'
                  AND next_attempt_at <= ?
                ORDER BY created_at ASC
                LIMIT ?;
                """,
                (now, max_limit),
            ).fetchall()
            for row in rows:
                cursor = conn.execute(
                    """
                    UPDATE user_issue_remote_outbox
                    SET status='sending',
                        next_attempt_at=?,
                        updated_at=?
                    WHERE id=?
                      AND status!='sent'
                      AND next_attempt_at <= ?;
                    """,
                    (lease_until, now, int(row["id"]), now),
                )
                if int(cursor.rowcount or 0) == 1:
                    claimed_rows.append(row)
        sent = 0
        failed = 0
        claimed_ids = [int(row["id"]) for row in claimed_rows]
        for row_index, row in enumerate(claimed_rows):
            outbox_id = int(row["id"])
            if not user_issue_remote_enabled():
                remaining_ids = claimed_ids[row_index:]
                released = self._release_remote_outbox_claims(
                    remaining_ids,
                    error="remote reporting is disabled",
                )
                return {
                    "ok": False,
                    "enabled": False,
                    "sent": sent,
                    "failed": failed,
                    "released": released,
                    "summary": self.remote_outbox_summary(),
                }
            attempts = int(row["attempts"] or 0)
            payload = _loads_json(str(row["payload_json"] or "{}"))
            if not isinstance(payload, Mapping):
                payload = {}
            try:
                result = post_remote_issue_payload(payload)
            except Exception as exc:
                result = {
                    "ok": False,
                    "enabled": True,
                    "status_code": 0,
                    "error": str(exc),
                }
            attempted_at = time.time()
            next_attempts = attempts + 1
            if bool(result.get("ok")):
                with self._connect() as conn:
                    conn.execute(
                        """
                        UPDATE user_issue_remote_outbox
                        SET status='sent',
                            attempts=?,
                            updated_at=?,
                            sent_at=?,
                            last_error=''
                        WHERE id=?;
                        """,
                        (next_attempts, attempted_at, attempted_at, outbox_id),
                    )
                sent += 1
            else:
                error = _redact_text(result.get("error") or f"HTTP {result.get('status_code') or 0}", limit=500)
                next_attempt_at = attempted_at + _remote_retry_delay_s(next_attempts)
                with self._connect() as conn:
                    conn.execute(
                        """
                        UPDATE user_issue_remote_outbox
                        SET status='pending',
                            attempts=?,
                            next_attempt_at=?,
                            updated_at=?,
                            last_error=?
                        WHERE id=?;
                        """,
                            (next_attempts, next_attempt_at, attempted_at, error, outbox_id),
                    )
                failed += 1
        with self._connect() as conn:
            self._prune_history(conn, now=time.time())
        return {
            "ok": failed == 0,
            "enabled": True,
            "sent": sent,
            "failed": failed,
            "summary": self.remote_outbox_summary(),
        }

    def remote_outbox_summary(self) -> dict[str, Any]:
        with self._connect() as conn:
            total = int(conn.execute("SELECT COUNT(*) FROM user_issue_remote_outbox;").fetchone()[0] or 0)
            pending = int(conn.execute("SELECT COUNT(*) FROM user_issue_remote_outbox WHERE status!='sent';").fetchone()[0] or 0)
            sent = int(conn.execute("SELECT COUNT(*) FROM user_issue_remote_outbox WHERE status='sent';").fetchone()[0] or 0)
            retryable = int(
                conn.execute(
                    "SELECT COUNT(*) FROM user_issue_remote_outbox WHERE status!='sent' AND next_attempt_at <= ?;",
                    (time.time(),),
                ).fetchone()[0]
                or 0
            )
            row = conn.execute(
                """
                SELECT last_error, attempts, next_attempt_at
                FROM user_issue_remote_outbox
                WHERE status!='sent' AND last_error != ''
                ORDER BY updated_at DESC
                LIMIT 1;
                """
            ).fetchone()
        latest_error = _redact_text(row["last_error"], limit=500) if row else ""
        return {
            "total": total,
            "pending": pending,
            "retryable": retryable,
            "sent": sent,
            "latest_error": latest_error,
            "latest_attempts": int(row["attempts"] or 0) if row else 0,
            "next_attempt_at": float(row["next_attempt_at"] or 0.0) if row else 0.0,
        }

    def discard_unsent_remote_outbox(self) -> dict[str, Any]:
        """Drop queued remote sends when the user opts out of quality data sharing."""

        with self._connect() as conn:
            total = int(conn.execute("SELECT COUNT(*) FROM user_issue_remote_outbox;").fetchone()[0] or 0)
            pending = int(conn.execute("SELECT COUNT(*) FROM user_issue_remote_outbox WHERE status!='sent';").fetchone()[0] or 0)
            sent = int(conn.execute("SELECT COUNT(*) FROM user_issue_remote_outbox WHERE status='sent';").fetchone()[0] or 0)
            conn.execute("DELETE FROM user_issue_remote_outbox WHERE status!='sent';")
        return {
            "ok": True,
            "removed": pending,
            "retained_sent": sent,
            "total_before": total,
        }

    def list_issues(self, *, limit: int = 100, status: str = "open") -> list[dict[str, Any]]:
        clean_status = _clean_text(status, limit=40).lower()
        max_limit = max(1, min(1000, int(limit or 100)))
        with self._connect() as conn:
            if clean_status and clean_status != "all":
                rows = conn.execute(
                    """
                    SELECT * FROM user_issues
                    WHERE status=?
                    ORDER BY last_seen_at DESC
                    LIMIT ?;
                    """,
                    (clean_status, max_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM user_issues
                    ORDER BY last_seen_at DESC
                    LIMIT ?;
                    """,
                    (max_limit,),
                ).fetchall()
        return [_row_to_issue(row) for row in rows]

    def summary(self) -> dict[str, Any]:
        with self._connect() as conn:
            total = int(conn.execute("SELECT COUNT(*) FROM user_issues;").fetchone()[0] or 0)
            open_total = int(conn.execute("SELECT COUNT(*) FROM user_issues WHERE status='open';").fetchone()[0] or 0)
            severity_rows = conn.execute(
                """
                SELECT severity, COUNT(*) AS count
                FROM user_issues
                WHERE status='open'
                GROUP BY severity;
                """
            ).fetchall()
            source_rows = conn.execute(
                """
                SELECT source, COUNT(*) AS count
                FROM user_issues
                WHERE status='open'
                GROUP BY source
                ORDER BY count DESC
                LIMIT 12;
                """
            ).fetchall()
        return {
            "total": total,
            "open": open_total,
            "severity": {str(row["severity"]): int(row["count"]) for row in severity_rows},
            "sources": [{"source": str(row["source"]), "count": int(row["count"])} for row in source_rows],
            "remote_outbox": self.remote_outbox_summary(),
        }


def start_remote_outbox_worker(db_path: str | Path, *, limit: int = 20) -> bool:
    """Start a small daemon that retries due remote quality-data sends."""

    global _OUTBOX_WORKER_THREAD
    interval_s = _remote_outbox_flush_interval_s()
    if interval_s <= 0:
        return False
    clean_limit = max(1, min(200, int(limit or 20)))
    clean_db_path = Path(db_path).expanduser()
    with _OUTBOX_WORKER_LOCK:
        if _OUTBOX_WORKER_THREAD and _OUTBOX_WORKER_THREAD.is_alive():
            return False

        def _worker() -> None:
            while True:
                try:
                    if user_issue_remote_enabled():
                        UserIssueStore(clean_db_path).flush_remote_outbox(limit=clean_limit)
                except Exception:
                    pass
                time.sleep(interval_s)

        _OUTBOX_WORKER_THREAD = threading.Thread(
            target=_worker,
            name="kb_user_issue_outbox",
            daemon=True,
        )
        _OUTBOX_WORKER_THREAD.start()
    return True


def record_library_quality_issues(db_path: str | Path, overview: Mapping[str, Any]) -> dict[str, Any]:
    store = UserIssueStore(db_path)
    context = {
        "scope": _clean_text(overview.get("scope"), limit=120),
        "status": _clean_text(overview.get("status"), limit=80),
        "summary": overview.get("summary") if isinstance(overview.get("summary"), Mapping) else {},
    }
    recorded = 0

    for raw_issue in list(overview.get("top_issues") or [])[:20]:
        if not isinstance(raw_issue, Mapping):
            continue
        code = _clean_text(raw_issue.get("code") or raw_issue.get("label"), limit=160)
        label = _clean_text(raw_issue.get("label") or code, limit=240)
        if not label:
            continue
        papers = _safe_int(raw_issue.get("papers"))
        count = _safe_int(raw_issue.get("count"))
        strategy = _clean_text(raw_issue.get("repair_strategy"), limit=400)
        detail = f"{papers} papers, {count} occurrences"
        if strategy:
            detail = f"{detail}; {strategy}"
        store.record_issue(
            source="library_quality_overview",
            domain="conversion",
            severity=_clean_text(raw_issue.get("severity"), limit=40) or "warning",
            summary=label,
            detail=detail,
            context=context,
            payload={"issue": dict(raw_issue)},
            fingerprint=_fingerprint(["library_quality_overview", "conversion", code or label]),
        )
        recorded += 1

    domains = overview.get("domains") if isinstance(overview.get("domains"), Mapping) else {}
    for domain_name, raw_domain in list(domains.items())[:12]:
        if not isinstance(raw_domain, Mapping):
            continue
        if raw_domain.get("available") is False:
            continue
        status = _clean_text(raw_domain.get("status"), limit=40).lower()
        if status in {"", "good", "unknown"}:
            continue
        failures = [item for item in list(raw_domain.get("top_failures") or []) if isinstance(item, Mapping)]
        failure_text = ", ".join(
            f"{_clean_text(item.get('name'), limit=120)} x{_safe_int(item.get('count'))}"
            for item in failures[:6]
            if _clean_text(item.get("name"), limit=120)
        )
        domain_summary = raw_domain.get("summary") if isinstance(raw_domain.get("summary"), Mapping) else {}
        store.record_issue(
            source="library_quality_overview",
            domain=_clean_text(domain_name, limit=120) or "quality",
            severity="error" if status == "error" else "warning",
            summary=f"{_clean_text(domain_name, limit=120) or 'quality'} status: {status}",
            detail=failure_text,
            context=context,
            payload={"domain": domain_summary, "top_failures": failures[:6]},
            fingerprint=_fingerprint(["library_quality_overview", domain_name, status, failure_text]),
        )
        recorded += 1

    for raw_case in list(overview.get("failure_cases") or [])[:30]:
        if not isinstance(raw_case, Mapping):
            continue
        case_id = _clean_text(raw_case.get("id"), limit=160)
        failures = [item for item in list(raw_case.get("failures") or []) if isinstance(item, Mapping)]
        names = [
            _clean_text(item.get("name"), limit=120)
            for item in failures[:8]
            if _clean_text(item.get("name"), limit=120)
        ]
        question = _clean_text(raw_case.get("question"), limit=500)
        fingerprint_material = case_id or question or ", ".join(names) or "unknown"
        store.record_issue(
            source="research_qa_failure_case",
            domain="research_qa",
            severity="error",
            summary="Research QA failure",
            detail=", ".join(names),
            context=context,
            payload={"case": dict(raw_case)},
            fingerprint=_fingerprint(["research_qa_failure_case", fingerprint_material]),
        )
        recorded += 1

    return {"ok": True, "recorded": recorded}
