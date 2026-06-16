from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping

from kb.user_issue_remote import report_user_issue_remote


def _clean_text(value: Any, *, limit: int = 2000) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ").strip()
    if len(text) > limit:
        return text[:limit]
    return text


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)
    except Exception:
        return json.dumps({"value": _clean_text(value, limit=4000)}, ensure_ascii=False, sort_keys=True)


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


def _fingerprint(parts: list[Any]) -> str:
    joined = "\n".join(_clean_text(part, limit=1000).lower() for part in parts)
    return hashlib.sha256(joined.encode("utf-8", errors="ignore")).hexdigest()


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


class UserIssueStore:
    """Durable local issue log for user-facing problems and hidden diagnostics."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_issues_last_seen ON user_issues(last_seen_at DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_issues_status ON user_issues(status, severity, last_seen_at DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_issue_events_issue ON user_issue_events(issue_id, created_at DESC);")

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
        clean_source = _clean_text(source or "frontend", limit=120) or "frontend"
        clean_domain = _clean_text(domain or "general", limit=120) or "general"
        clean_severity = _severity(severity)
        clean_summary = _clean_text(summary or "User issue", limit=500) or "User issue"
        clean_detail = _clean_text(detail, limit=4000)
        clean_route = _clean_text(route, limit=500)
        clean_user_agent = _clean_text(user_agent, limit=500)
        issue_fp = _clean_text(fingerprint, limit=128) or _fingerprint(
            [clean_source, clean_domain, clean_severity, clean_summary, clean_detail[:1000]]
        )
        context_json = _safe_json(dict(context or {}))
        payload_json = _safe_json(dict(payload or {}))

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
            conn.execute(
                """
                INSERT INTO user_issue_events (
                  issue_id, created_at, route, user_agent, context_json, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (issue_id, now, clean_route, clean_user_agent, context_json, payload_json),
            )
            issue = _row_to_issue(row)
        if bool(forward_remote):
            try:
                report_user_issue_remote(issue)
            except Exception:
                pass
        return issue

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
        }


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
        summary = question or case_id or "Research QA failure"
        store.record_issue(
            source="research_qa_failure_case",
            domain="research_qa",
            severity="error",
            summary=summary,
            detail=", ".join(names),
            context=context,
            payload={"case": dict(raw_case)},
            fingerprint=_fingerprint(["research_qa_failure_case", case_id or summary]),
        )
        recorded += 1

    return {"ok": True, "recorded": recorded}
