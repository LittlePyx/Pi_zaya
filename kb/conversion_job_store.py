from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any


_ACTIVE_STATES = {"queued", "running", "cancelling"}
_RECOVERABLE_STATES = {"interrupted"}
_TERMINAL_STATES = {
    "interrupted",
    "succeeded",
    "cancelled",
    "conversion_failed",
    "quality_blocked",
    "index_failed",
}
_PUBLIC_OUTCOME_BY_STATE = {
    "interrupted": "interrupted",
    "succeeded": "success",
    "cancelled": "cancelled",
    "conversion_failed": "conversion_failed",
    "quality_blocked": "quality_blocked",
    "index_failed": "index_failed",
}


def _normalized_path_key(value: str | Path) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return os.path.normcase(str(Path(raw).expanduser().resolve(strict=False))).casefold()
    except Exception:
        return raw.replace("\\", "/").casefold()


def _compact_repair_context(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    issue_codes = [
        str(item or "").strip()
        for item in list(value.get("issue_codes") or [])
        if str(item or "").strip()
    ]
    retry_pages = sorted(
        {
            int(item)
            for item in list(value.get("retry_pages") or [])
            if str(item or "").isdigit() and int(item) > 0
        }
    )[:500]
    out: dict[str, Any] = {
        "action": str(value.get("action") or "")[:80],
        "scope": str(value.get("scope") or "")[:120],
        "reason": str(value.get("reason") or "")[:500],
        "source": str(value.get("source") or "")[:120],
        "repair_run_id": str(value.get("repair_run_id") or "")[:160],
        "issue_codes": issue_codes[:30],
        "retry_pages": retry_pages,
    }
    return {
        key: item
        for key, item in out.items()
        if (item if not isinstance(item, list) else bool(item))
    }


def _decode_json_object(raw: Any) -> dict[str, Any]:
    try:
        value = json.loads(str(raw or "{}"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


class ConversionJobStore:
    """Durable conversion-job ledger stored beside the PDF library metadata."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversion_jobs (
                  task_id TEXT PRIMARY KEY,
                  pdf_path TEXT NOT NULL,
                  pdf_key TEXT NOT NULL,
                  name TEXT NOT NULL,
                  out_root TEXT NOT NULL,
                  db_dir TEXT NOT NULL DEFAULT '',
                  no_llm INTEGER NOT NULL DEFAULT 0,
                  eq_image_fallback INTEGER NOT NULL DEFAULT 0,
                  replace_existing INTEGER NOT NULL DEFAULT 0,
                  speed_mode TEXT NOT NULL DEFAULT 'balanced',
                  repair_context_json TEXT NOT NULL DEFAULT '{}',
                  state TEXT NOT NULL,
                  stage TEXT NOT NULL DEFAULT 'queued',
                  operation TEXT NOT NULL DEFAULT 'conversion',
                  outcome TEXT NOT NULL DEFAULT '',
                  retry_action TEXT NOT NULL DEFAULT '',
                  message TEXT NOT NULL DEFAULT '',
                  detail TEXT NOT NULL DEFAULT '',
                  blocked_reason TEXT NOT NULL DEFAULT '',
                  page_done INTEGER NOT NULL DEFAULT 0,
                  page_total INTEGER NOT NULL DEFAULT 0,
                  reused_page_count INTEGER NOT NULL DEFAULT 0,
                  attempt INTEGER NOT NULL DEFAULT 1,
                  owner_session TEXT NOT NULL DEFAULT '',
                  created_at REAL NOT NULL,
                  queued_at REAL NOT NULL,
                  started_at REAL NOT NULL DEFAULT 0,
                  updated_at REAL NOT NULL,
                  interrupted_at REAL NOT NULL DEFAULT 0,
                  finished_at REAL NOT NULL DEFAULT 0
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversion_jobs_state_updated
                ON conversion_jobs(state, updated_at DESC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversion_jobs_pdf_updated
                ON conversion_jobs(pdf_key, updated_at DESC);
                """
            )

    @staticmethod
    def _task_values(task: dict[str, Any]) -> dict[str, Any]:
        task_id = str(task.get("_tid") or task.get("task_id") or "").strip()
        pdf_path = str(task.get("pdf") or task.get("pdf_path") or "").strip()
        if not task_id:
            raise ValueError("conversion task_id is required")
        if not pdf_path:
            raise ValueError("conversion pdf path is required")
        return {
            "task_id": task_id,
            "pdf_path": pdf_path,
            "pdf_key": _normalized_path_key(pdf_path),
            "name": str(task.get("name") or Path(pdf_path).name),
            "out_root": str(task.get("out_root") or ""),
            "db_dir": str(task.get("db_dir") or ""),
            "no_llm": int(bool(task.get("no_llm", False))),
            "eq_image_fallback": int(bool(task.get("eq_image_fallback", False))),
            "replace_existing": int(bool(task.get("replace", False))),
            "speed_mode": str(task.get("speed_mode") or "balanced")[:40],
            "repair_context_json": json.dumps(
                _compact_repair_context(task.get("repair_context")),
                ensure_ascii=False,
                sort_keys=True,
            ),
        }

    def create_queued(self, task: dict[str, Any], *, owner_session: str) -> bool:
        values = self._task_values(task)
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            active = conn.execute(
                """
                SELECT task_id FROM conversion_jobs
                WHERE pdf_key = ? AND state IN ('queued', 'running', 'cancelling')
                LIMIT 1
                """,
                (values["pdf_key"],),
            ).fetchone()
            if active:
                return False
            conn.execute(
                """
                UPDATE conversion_jobs
                SET state = 'cancelled', stage = '', outcome = 'cancelled',
                    retry_action = '', message = 'Recovery was replaced by a new conversion.',
                    detail = '', blocked_reason = '', updated_at = ?, finished_at = ?
                WHERE pdf_key = ? AND state = 'interrupted'
                """,
                (now, now, values["pdf_key"]),
            )
            conn.execute(
                """
                INSERT INTO conversion_jobs (
                  task_id, pdf_path, pdf_key, name, out_root, db_dir,
                  no_llm, eq_image_fallback, replace_existing, speed_mode,
                  repair_context_json, state, stage, operation, outcome,
                  retry_action, message, detail, blocked_reason, page_done,
                  page_total, reused_page_count, attempt, owner_session,
                  created_at, queued_at, started_at, updated_at,
                  interrupted_at, finished_at
                ) VALUES (
                  :task_id, :pdf_path, :pdf_key, :name, :out_root, :db_dir,
                  :no_llm, :eq_image_fallback, :replace_existing, :speed_mode,
                  :repair_context_json, 'queued', 'queued', 'conversion', '',
                  '', '', '', '', 0, 0, 0, 1, :owner_session,
                  :created_at, :queued_at, 0, :updated_at, 0, 0
                )
                """,
                {
                    **values,
                    "owner_session": str(owner_session or ""),
                    "created_at": now,
                    "queued_at": now,
                    "updated_at": now,
                },
            )
        return True

    def delete_queued(self, task_id: str, *, owner_session: str = "") -> bool:
        params: list[Any] = [str(task_id or "")]
        owner_clause = ""
        if owner_session:
            owner_clause = " AND owner_session = ?"
            params.append(str(owner_session))
        with self._connect() as conn:
            cur = conn.execute(
                f"DELETE FROM conversion_jobs WHERE task_id = ? AND state = 'queued'{owner_clause}",
                tuple(params),
            )
            return int(cur.rowcount or 0) > 0

    def mark_running(self, task_id: str, *, owner_session: str) -> bool:
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE conversion_jobs
                SET state = 'running', stage = 'converting', owner_session = ?,
                    started_at = CASE WHEN started_at > 0 THEN started_at ELSE ? END,
                    updated_at = ?, blocked_reason = '', message = ''
                WHERE task_id = ? AND state = 'queued'
                """,
                (str(owner_session or ""), now, now, str(task_id or "")),
            )
            return int(cur.rowcount or 0) > 0

    def update_progress(
        self,
        task_id: str,
        *,
        page_done: int,
        page_total: int,
        stage: str = "",
        reused_page_count: int | None = None,
    ) -> bool:
        clean_stage = str(stage or "").strip().lower()
        now = time.time()
        fields = [
            "page_done = MAX(page_done, ?)",
            "page_total = MAX(page_total, ?)",
            "updated_at = ?",
        ]
        params: list[Any] = [max(0, int(page_done or 0)), max(0, int(page_total or 0)), now]
        if clean_stage:
            fields.append("stage = ?")
            params.append(clean_stage[:40])
        if reused_page_count is not None:
            fields.append("reused_page_count = MAX(reused_page_count, ?)")
            params.append(max(0, int(reused_page_count)))
        params.append(str(task_id or ""))
        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE conversion_jobs SET {', '.join(fields)} "
                "WHERE task_id = ? AND state IN ('queued', 'running', 'cancelling')",
                tuple(params),
            )
            return int(cur.rowcount or 0) > 0

    def update_stage(self, task_id: str, stage: str) -> bool:
        clean_stage = str(stage or "").strip().lower()[:40]
        if not clean_stage:
            return False
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE conversion_jobs SET stage = ?, updated_at = ?
                WHERE task_id = ? AND state IN ('queued', 'running', 'cancelling')
                """,
                (clean_stage, time.time(), str(task_id or "")),
            )
            return int(cur.rowcount or 0) > 0

    def mark_cancelling(self, task_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE conversion_jobs
                SET state = 'cancelling', stage = 'cancelling', updated_at = ?
                WHERE task_id = ? AND state = 'running'
                """,
                (time.time(), str(task_id or "")),
            )
            return int(cur.rowcount or 0) > 0

    def mark_cancelled(self, task_id: str, *, message: str = "Conversion was cancelled.") -> bool:
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE conversion_jobs
                SET state = 'cancelled', stage = '', outcome = 'cancelled',
                    retry_action = 'reconvert', message = ?, detail = '',
                    blocked_reason = '', updated_at = ?, finished_at = ?
                WHERE task_id = ? AND state IN ('queued', 'interrupted')
                """,
                (str(message or "")[:500], now, now, str(task_id or "")),
            )
            return int(cur.rowcount or 0) > 0

    def finish(self, task_id: str, result: dict[str, Any]) -> bool:
        outcome = str(result.get("outcome") or "conversion_failed").strip().lower()
        state = "succeeded" if outcome == "success" else outcome
        if state not in _TERMINAL_STATES - {"interrupted"}:
            state = "conversion_failed"
            outcome = "conversion_failed"
        now = float(result.get("finished_at") or time.time())
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE conversion_jobs
                SET state = ?, stage = '', outcome = ?, retry_action = ?,
                    message = ?, detail = ?, page_done = MAX(page_done, ?),
                    page_total = MAX(page_total, ?), updated_at = ?, finished_at = ?,
                    blocked_reason = ''
                WHERE task_id = ? AND state IN ('queued', 'running', 'cancelling')
                """,
                (
                    state,
                    outcome,
                    str(result.get("retry_action") or "")[:40],
                    str(result.get("message") or "")[:500],
                    str(result.get("detail") or "")[:500],
                    max(0, int(result.get("page_done") or 0)),
                    max(0, int(result.get("page_total") or 0)),
                    now,
                    now,
                    str(task_id or ""),
                ),
            )
            return int(cur.rowcount or 0) > 0

    def reconcile_after_restart(self, *, owner_session: str) -> list[dict[str, Any]]:
        now = time.time()
        owner = str(owner_session or "")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT * FROM conversion_jobs
                WHERE state IN ('queued', 'running', 'cancelling')
                  AND owner_session <> ?
                ORDER BY updated_at DESC
                """,
                (owner,),
            ).fetchall()
            if rows:
                task_ids = [str(row["task_id"] or "") for row in rows]
                placeholders = ",".join("?" for _ in task_ids)
                conn.execute(
                    f"""
                    UPDATE conversion_jobs
                    SET state = 'interrupted', stage = 'interrupted',
                        outcome = 'interrupted', retry_action = 'resume',
                        message = 'Conversion was interrupted when Pi_zaya stopped.',
                        detail = '', blocked_reason = '', interrupted_at = ?,
                        updated_at = ?, finished_at = ?
                    WHERE task_id IN ({placeholders})
                    """,
                    (now, now, now, *task_ids),
                )
        return [self._public_recoverable(dict(row), interrupted_at=now) for row in rows]

    def mark_blocked(self, task_id: str, *, reason: str, message: str) -> bool:
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE conversion_jobs
                SET blocked_reason = ?, message = ?, updated_at = ?
                WHERE task_id = ? AND state = 'interrupted'
                """,
                (
                    str(reason or "")[:80],
                    str(message or "")[:500],
                    now,
                    str(task_id or ""),
                ),
            )
            return int(cur.rowcount or 0) > 0

    def queue_for_resume(self, task_id: str, *, owner_session: str) -> dict[str, Any] | None:
        tid = str(task_id or "").strip()
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM conversion_jobs WHERE task_id = ? AND state = 'interrupted'",
                (tid,),
            ).fetchone()
            if not row:
                return None
            active = conn.execute(
                """
                SELECT task_id FROM conversion_jobs
                WHERE pdf_key = ? AND state IN ('queued', 'running', 'cancelling')
                LIMIT 1
                """,
                (str(row["pdf_key"] or ""),),
            ).fetchone()
            if active:
                return None
            cur = conn.execute(
                """
                UPDATE conversion_jobs
                SET state = 'queued', stage = 'queued', outcome = '',
                    retry_action = '', message = '', detail = '', blocked_reason = '',
                    attempt = attempt + 1, owner_session = ?, queued_at = ?,
                    started_at = 0, updated_at = ?, interrupted_at = 0, finished_at = 0
                WHERE task_id = ? AND state = 'interrupted'
                """,
                (str(owner_session or ""), now, now, tid),
            )
            if int(cur.rowcount or 0) <= 0:
                return None
            updated = conn.execute("SELECT * FROM conversion_jobs WHERE task_id = ?", (tid,)).fetchone()
        return self._task_from_row(updated) if updated else None

    def revert_resume(self, task_id: str, *, message: str) -> bool:
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE conversion_jobs
                SET state = 'interrupted', stage = 'interrupted', outcome = 'interrupted',
                    retry_action = 'resume', message = ?, updated_at = ?,
                    interrupted_at = ?, finished_at = ?
                WHERE task_id = ? AND state = 'queued'
                """,
                (str(message or "")[:500], now, now, now, str(task_id or "")),
            )
            return int(cur.rowcount or 0) > 0

    def get_recoverable(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversion_jobs WHERE task_id = ? AND state = 'interrupted'",
                (str(task_id or ""),),
            ).fetchone()
        return self._public_recoverable(dict(row)) if row else None

    def get_recoverable_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversion_jobs WHERE task_id = ? AND state = 'interrupted'",
                (str(task_id or ""),),
            ).fetchone()
        return self._task_from_row(row) if row else None

    def job_state(self, task_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state FROM conversion_jobs WHERE task_id = ?",
                (str(task_id or ""),),
            ).fetchone()
        return str(row["state"] or "") if row else ""

    def update_recoverable_pdf_path(self, old_path: Path | str, new_path: Path | str) -> int:
        old_key = _normalized_path_key(old_path)
        new_pdf = str(Path(new_path))
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE conversion_jobs
                SET pdf_path = ?, pdf_key = ?, name = ?, updated_at = ?
                WHERE pdf_key = ? AND state = 'interrupted'
                """,
                (
                    new_pdf,
                    _normalized_path_key(new_pdf),
                    Path(new_pdf).name,
                    time.time(),
                    old_key,
                ),
            )
            return max(0, int(cur.rowcount or 0))

    def list_recoverable(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM conversion_jobs
                WHERE state = 'interrupted'
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(1, min(500, int(limit))),),
            ).fetchall()
        items: list[dict[str, Any]] = []
        seen_pdf_keys: set[str] = set()
        for row in rows:
            data = dict(row)
            pdf_key = str(data.get("pdf_key") or "")
            if pdf_key and pdf_key in seen_pdf_keys:
                continue
            if pdf_key:
                seen_pdf_keys.add(pdf_key)
            items.append(self._public_recoverable(data))
        return items

    def list_recent_results(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM conversion_jobs
                WHERE state IN (
                  'interrupted', 'succeeded', 'cancelled', 'conversion_failed',
                  'quality_blocked', 'index_failed'
                )
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(1, min(200, int(limit))),),
            ).fetchall()
        return [self._public_result(dict(row)) for row in rows]

    @staticmethod
    def _task_from_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        data = dict(row)
        task = {
            "_tid": str(data.get("task_id") or ""),
            "pdf": str(data.get("pdf_path") or ""),
            "name": str(data.get("name") or ""),
            "out_root": str(data.get("out_root") or ""),
            "db_dir": str(data.get("db_dir") or ""),
            "no_llm": bool(data.get("no_llm")),
            "eq_image_fallback": bool(data.get("eq_image_fallback")),
            "replace": bool(data.get("replace_existing")),
            "speed_mode": str(data.get("speed_mode") or "balanced"),
            "resumed": True,
            "resume_attempt": max(1, int(data.get("attempt") or 1)),
        }
        repair_context = _decode_json_object(data.get("repair_context_json"))
        if repair_context:
            task["repair_context"] = repair_context
        return task

    @staticmethod
    def _cached_page_count(data: dict[str, Any]) -> int:
        try:
            cache_pages = (
                Path(str(data.get("out_root") or ""))
                / Path(str(data.get("pdf_path") or "")).stem
                / ".conversion_cache"
                / "pages"
            )
            if not cache_pages.is_dir():
                return 0
            return sum(
                1
                for entry in cache_pages.glob("*/entry.json")
                if entry.is_file() and (entry.parent / "page.txt").is_file()
            )
        except Exception:
            return 0

    def _public_recoverable(
        self,
        data: dict[str, Any],
        *,
        interrupted_at: float | None = None,
    ) -> dict[str, Any]:
        return {
            "task_id": str(data.get("task_id") or ""),
            "pdf": str(data.get("pdf_path") or ""),
            "name": str(data.get("name") or Path(str(data.get("pdf_path") or "")).name),
            "state": "interrupted",
            "stage": "interrupted",
            "message": str(data.get("message") or "Conversion was interrupted when Pi_zaya stopped.")[:500],
            "blocked_reason": str(data.get("blocked_reason") or "")[:80],
            "replace": bool(data.get("replace_existing")),
            "speed_mode": str(data.get("speed_mode") or "balanced")[:40],
            "no_llm": bool(data.get("no_llm")),
            "page_done": max(0, int(data.get("page_done") or 0)),
            "page_total": max(0, int(data.get("page_total") or 0)),
            "reused_page_count": max(0, int(data.get("reused_page_count") or 0)),
            "cached_page_count": self._cached_page_count(data),
            "attempt": max(1, int(data.get("attempt") or 1)),
            "created_at": float(data.get("created_at") or 0.0),
            "updated_at": float(data.get("updated_at") or time.time()),
            "interrupted_at": float(interrupted_at if interrupted_at is not None else data.get("interrupted_at") or 0.0),
        }

    @staticmethod
    def _public_result(data: dict[str, Any]) -> dict[str, Any]:
        state = str(data.get("state") or "conversion_failed")
        outcome = _PUBLIC_OUTCOME_BY_STATE.get(state, "conversion_failed")
        started_at = float(data.get("started_at") or 0.0)
        finished_at = float(data.get("finished_at") or data.get("updated_at") or 0.0)
        return {
            "task_id": str(data.get("task_id") or ""),
            "pdf": str(data.get("pdf_path") or ""),
            "name": str(data.get("name") or Path(str(data.get("pdf_path") or "")).name),
            "outcome": outcome,
            "operation": str(data.get("operation") or "conversion"),
            "message": str(data.get("message") or "")[:500],
            "detail": str(data.get("detail") or "")[:500],
            "retry_action": str(data.get("retry_action") or ("resume" if outcome == "interrupted" else "")),
            "replace": bool(data.get("replace_existing")),
            "speed_mode": str(data.get("speed_mode") or "")[:40],
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_s": max(0.0, finished_at - started_at) if started_at > 0 else 0.0,
            "page_done": max(0, int(data.get("page_done") or 0)),
            "page_total": max(0, int(data.get("page_total") or 0)),
            "reused_page_count": max(0, int(data.get("reused_page_count") or 0)),
            "blocked_reason": str(data.get("blocked_reason") or "")[:80],
        }
