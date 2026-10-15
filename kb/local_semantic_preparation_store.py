from __future__ import annotations

import re
import sqlite3
import time
import uuid
from pathlib import Path

from .local_semantic_preparation_contract import (
    LocalSemanticPreparationStatus,
    normalize_preparation_state,
    preparation_is_active,
    reconcile_preparation_state_after_restart,
)


_REASON_RE = re.compile(r"^[a-z0-9_]{0,80}$")
_RUNNING_STATES = {"downloading", "verifying", "indexing"}


def _reason_id(value: object, default: str = "") -> str:
    reason = str(value or "").strip().lower()
    return reason if _REASON_RE.fullmatch(reason) else default


class LocalSemanticPreparationStore:
    """Durable single-library preparation ledger with path-free public records."""

    def __init__(self, db_dir: Path | str) -> None:
        self.db_dir = Path(db_dir).expanduser().resolve(strict=False)
        self.path = self.db_dir / "local_semantic_preparation.sqlite3"
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        connection.execute("PRAGMA busy_timeout=30000;")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS local_semantic_preparation_jobs (
                  task_id TEXT PRIMARY KEY,
                  operation TEXT NOT NULL,
                  state TEXT NOT NULL,
                  reason_id TEXT NOT NULL DEFAULT '',
                  phase TEXT NOT NULL DEFAULT '',
                  completed_units INTEGER NOT NULL DEFAULT 0,
                  total_units INTEGER NOT NULL DEFAULT 0,
                  completed_bytes INTEGER NOT NULL DEFAULT 0,
                  total_bytes INTEGER NOT NULL DEFAULT 0,
                  eta_seconds INTEGER,
                  document_count INTEGER NOT NULL DEFAULT 0,
                  chunk_count INTEGER NOT NULL DEFAULT 0,
                  owner_session TEXT NOT NULL DEFAULT '',
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL,
                  finished_at REAL NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_local_semantic_preparation_updated
                ON local_semantic_preparation_jobs(updated_at DESC)
                """
            )

    @staticmethod
    def _status(row: sqlite3.Row | None) -> LocalSemanticPreparationStatus:
        if row is None:
            return LocalSemanticPreparationStatus(
                state="not_prepared",
                reason_id="artifacts_not_prepared",
            )
        raw_state = str(row["state"] or "")
        state = normalize_preparation_state(raw_state)
        return LocalSemanticPreparationStatus(
            state=state,
            reason_id=(
                _reason_id(row["reason_id"], "preparation_state_invalid")
                if state == raw_state
                else "preparation_state_invalid"
            ),
            task_id=str(row["task_id"] or ""),
            phase=str(row["phase"] or "")[:40],
            completed_units=max(0, int(row["completed_units"] or 0)),
            total_units=max(0, int(row["total_units"] or 0)),
            completed_bytes=max(0, int(row["completed_bytes"] or 0)),
            total_bytes=max(0, int(row["total_bytes"] or 0)),
            eta_seconds=(
                max(0, int(row["eta_seconds"]))
                if row["eta_seconds"] is not None
                else None
            ),
            document_count=max(0, int(row["document_count"] or 0)),
            chunk_count=max(0, int(row["chunk_count"] or 0)),
        )

    def current(self) -> LocalSemanticPreparationStatus:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM local_semantic_preparation_jobs
                ORDER BY updated_at DESC, created_at DESC LIMIT 1
                """
            ).fetchone()
        return self._status(row)

    def start_or_get(
        self,
        *,
        operation: str,
        owner_session: str,
        initial_state: str = "downloading",
        total_units: int = 0,
        total_bytes: int = 0,
    ) -> tuple[LocalSemanticPreparationStatus, bool]:
        if initial_state not in _RUNNING_STATES:
            raise ValueError("a preparation job must start in a running state")
        clean_operation = str(operation or "prepare").strip().lower()
        if clean_operation not in {"prepare", "resume", "rebuild", "retry"}:
            raise ValueError("unsupported local semantic preparation operation")
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """
                SELECT * FROM local_semantic_preparation_jobs
                WHERE state IN ('downloading', 'verifying', 'indexing', 'cancelling')
                ORDER BY updated_at DESC LIMIT 1
                """
            ).fetchone()
            if active is not None:
                return self._status(active), False
            task_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO local_semantic_preparation_jobs (
                  task_id, operation, state, reason_id, phase,
                  completed_units, total_units, completed_bytes, total_bytes,
                  eta_seconds, document_count, chunk_count, owner_session,
                  created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 0, ?, 0, ?, NULL, 0, 0, ?, ?, ?)
                """,
                (
                    task_id,
                    clean_operation,
                    initial_state,
                    f"artifact_{initial_state}" if initial_state != "indexing" else "index_building",
                    "artifacts" if initial_state != "indexing" else "index",
                    max(0, int(total_units)),
                    max(0, int(total_bytes)),
                    str(owner_session or "")[:80],
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM local_semantic_preparation_jobs WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return self._status(row), True

    def update_progress(
        self,
        task_id: str,
        *,
        state: str,
        reason_id: str,
        phase: str,
        completed_units: int,
        total_units: int,
        completed_bytes: int,
        total_bytes: int,
        eta_seconds: int | None,
        document_count: int = 0,
        chunk_count: int = 0,
    ) -> bool:
        if state not in _RUNNING_STATES | {"verifying", "ready_to_index"}:
            raise ValueError("invalid progress state")
        status = LocalSemanticPreparationStatus(
            state=state,
            reason_id=_reason_id(reason_id, "preparation_in_progress"),
            task_id=str(task_id or ""),
            phase=str(phase or "")[:40],
            completed_units=completed_units,
            total_units=total_units,
            completed_bytes=completed_bytes,
            total_bytes=total_bytes,
            eta_seconds=eta_seconds,
            document_count=document_count,
            chunk_count=chunk_count,
        )
        with self._connect() as connection:
            current = connection.execute(
                "SELECT state FROM local_semantic_preparation_jobs WHERE task_id = ?",
                (status.task_id,),
            ).fetchone()
            if current is None or str(current["state"] or "") == "cancelling":
                return False
            cursor = connection.execute(
                """
                UPDATE local_semantic_preparation_jobs SET
                  state = ?, reason_id = ?, phase = ?, completed_units = ?,
                  total_units = ?, completed_bytes = ?, total_bytes = ?,
                  eta_seconds = ?, document_count = ?, chunk_count = ?, updated_at = ?
                WHERE task_id = ? AND state IN ('downloading', 'verifying', 'indexing')
                """,
                (
                    status.state,
                    status.reason_id,
                    status.phase,
                    status.completed_units,
                    status.total_units,
                    status.completed_bytes,
                    status.total_bytes,
                    status.eta_seconds,
                    status.document_count,
                    status.chunk_count,
                    time.time(),
                    status.task_id,
                ),
            )
        return int(cursor.rowcount or 0) > 0

    def request_cancel(self, task_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE local_semantic_preparation_jobs
                SET state = 'cancelling', reason_id = 'cancel_requested',
                    phase = 'cancelling', eta_seconds = NULL, updated_at = ?
                WHERE task_id = ? AND state IN ('downloading', 'verifying', 'indexing')
                """,
                (time.time(), str(task_id or "")),
            )
        return int(cursor.rowcount or 0) > 0

    def should_cancel(self, task_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state FROM local_semantic_preparation_jobs WHERE task_id = ?",
                (str(task_id or ""),),
            ).fetchone()
        return bool(row is not None and str(row["state"] or "") == "cancelling")

    def finish(
        self,
        task_id: str,
        *,
        state: str,
        reason_id: str,
        document_count: int = 0,
        chunk_count: int = 0,
    ) -> bool:
        if state not in {"interrupted", "repairable_error", "ready", "stale"}:
            raise ValueError("invalid terminal preparation state")
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE local_semantic_preparation_jobs
                SET state = ?, reason_id = ?, phase = '', eta_seconds = NULL,
                    document_count = ?, chunk_count = ?, updated_at = ?, finished_at = ?
                WHERE task_id = ?
                  AND state IN ('downloading', 'verifying', 'ready_to_index',
                                'indexing', 'cancelling')
                """,
                (
                    state,
                    _reason_id(reason_id, "preparation_failed"),
                    max(0, int(document_count)),
                    max(0, int(chunk_count)),
                    now,
                    now,
                    str(task_id or ""),
                ),
            )
        return int(cursor.rowcount or 0) > 0

    def reconcile_after_restart(self, *, owner_session: str) -> LocalSemanticPreparationStatus:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM local_semantic_preparation_jobs
                ORDER BY updated_at DESC, created_at DESC LIMIT 1
                """
            ).fetchone()
            if row is None or not preparation_is_active(row["state"]):
                return self._status(row)
            if str(row["owner_session"] or "") == str(owner_session or ""):
                return self._status(row)
            reconciled = reconcile_preparation_state_after_restart(row["state"])
            connection.execute(
                """
                UPDATE local_semantic_preparation_jobs
                SET state = ?, reason_id = 'app_restarted', phase = '',
                    eta_seconds = NULL, updated_at = ?, finished_at = ?
                WHERE task_id = ?
                """,
                (reconciled, now, now, str(row["task_id"] or "")),
            )
            updated = connection.execute(
                "SELECT * FROM local_semantic_preparation_jobs WHERE task_id = ?",
                (str(row["task_id"] or ""),),
            ).fetchone()
        return self._status(updated)
