from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

from .local_semantic_component_lifecycle_contract import (
    COMPONENT_ACTIVE_STATES,
    COMPONENT_LIFECYCLE_STATES,
    LocalSemanticComponentLifecycleRecord,
    component_lifecycle_is_active,
    normalize_component_lifecycle_state,
    normalize_internal_reason,
)


_OPERATIONS = {"enable", "resume", "recover", "remove"}
_UNSET = object()


class LocalSemanticComponentLifecycleStore:
    """Durable v2 lifecycle ledger; raw failure ids remain internal."""

    def __init__(self, db_dir: Path | str) -> None:
        self.db_dir = Path(db_dir).expanduser().resolve(strict=False)
        self.path = self.db_dir / "local_semantic_component_lifecycle.sqlite3"
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
                CREATE TABLE IF NOT EXISTS local_semantic_component_jobs (
                  task_id TEXT PRIMARY KEY,
                  operation TEXT NOT NULL,
                  state TEXT NOT NULL,
                  reason_id TEXT NOT NULL DEFAULT '',
                  manifest_identity TEXT NOT NULL DEFAULT '',
                  owner_session TEXT NOT NULL DEFAULT '',
                  attempt INTEGER NOT NULL DEFAULT 0,
                  consented INTEGER NOT NULL DEFAULT 0,
                  completed_bytes INTEGER NOT NULL DEFAULT 0,
                  total_bytes INTEGER NOT NULL DEFAULT 0,
                  eta_seconds INTEGER,
                  required_additional_bytes INTEGER NOT NULL DEFAULT 0,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL,
                  finished_at REAL NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_local_semantic_component_updated
                ON local_semantic_component_jobs(updated_at DESC)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS local_semantic_component_quarantine (
                  manifest_identity TEXT PRIMARY KEY,
                  reason_id TEXT NOT NULL,
                  created_at REAL NOT NULL
                )
                """
            )

    @staticmethod
    def _record(row: sqlite3.Row | None) -> LocalSemanticComponentLifecycleRecord:
        if row is None:
            return LocalSemanticComponentLifecycleRecord(
                state="not_installed",
                reason_id="component_not_installed",
            )
        raw_state = str(row["state"] or "")
        state = normalize_component_lifecycle_state(raw_state)
        return LocalSemanticComponentLifecycleRecord(
            state=state,
            reason_id=(
                normalize_internal_reason(row["reason_id"])
                if state == raw_state
                else "component_state_invalid"
            ),
            task_id=str(row["task_id"] or ""),
            operation=str(row["operation"] or ""),
            manifest_identity=str(row["manifest_identity"] or ""),
            owner_session=str(row["owner_session"] or ""),
            attempt=max(0, int(row["attempt"] or 0)),
            consented=bool(row["consented"]),
            completed_bytes=max(0, int(row["completed_bytes"] or 0)),
            total_bytes=max(0, int(row["total_bytes"] or 0)),
            eta_seconds=(
                max(0, int(row["eta_seconds"]))
                if row["eta_seconds"] is not None
                else None
            ),
            required_additional_bytes=max(
                0, int(row["required_additional_bytes"] or 0)
            ),
        )

    def current(self) -> LocalSemanticComponentLifecycleRecord:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM local_semantic_component_jobs
                ORDER BY updated_at DESC, created_at DESC LIMIT 1
                """
            ).fetchone()
        return self._record(row)

    def start_or_get(
        self,
        *,
        operation: str,
        initial_state: str,
        manifest_identity: str,
        owner_session: str,
        consented: bool,
        total_bytes: int = 0,
    ) -> tuple[LocalSemanticComponentLifecycleRecord, bool]:
        operation = str(operation or "").strip().lower()
        if operation not in _OPERATIONS:
            raise ValueError("unsupported component lifecycle operation")
        if initial_state not in COMPONENT_ACTIVE_STATES:
            raise ValueError("component lifecycle job must start active")
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            placeholders = ",".join("?" for _state in COMPONENT_ACTIVE_STATES)
            active = connection.execute(
                f"""
                SELECT * FROM local_semantic_component_jobs
                WHERE state IN ({placeholders})
                ORDER BY updated_at DESC LIMIT 1
                """,
                tuple(COMPONENT_ACTIVE_STATES),
            ).fetchone()
            if active is not None:
                return self._record(active), False
            task_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO local_semantic_component_jobs (
                  task_id, operation, state, reason_id, manifest_identity,
                  owner_session, attempt, consented, completed_bytes,
                  total_bytes, eta_seconds, required_additional_bytes,
                  created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, 0, ?, NULL, 0, ?, ?)
                """,
                (
                    task_id,
                    operation,
                    initial_state,
                    "component_operation_started",
                    str(manifest_identity or "")[:160],
                    str(owner_session or "")[:80],
                    1 if consented else 0,
                    max(0, int(total_bytes)),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM local_semantic_component_jobs WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return self._record(row), True

    def transition(
        self,
        task_id: str,
        *,
        state: str,
        reason_id: str,
        attempt: int | None = None,
        completed_bytes: int | None = None,
        total_bytes: int | None = None,
        eta_seconds: int | None | object = _UNSET,
        required_additional_bytes: int | None = None,
        consented: bool | None = None,
        finished: bool = False,
    ) -> bool:
        if state not in COMPONENT_LIFECYCLE_STATES:
            raise ValueError("invalid component lifecycle state")
        updates = ["state = ?", "reason_id = ?", "updated_at = ?"]
        values: list[object] = [state, normalize_internal_reason(reason_id), time.time()]
        for column, value in (
            ("attempt", attempt),
            ("completed_bytes", completed_bytes),
            ("total_bytes", total_bytes),
            ("required_additional_bytes", required_additional_bytes),
        ):
            if value is not None:
                updates.append(f"{column} = ?")
                values.append(max(0, int(value)))
        if eta_seconds is not _UNSET:
            updates.append("eta_seconds = ?")
            values.append(
                None if eta_seconds is None else max(0, int(eta_seconds))
            )
        if consented is not None:
            updates.append("consented = ?")
            values.append(1 if consented else 0)
        if finished:
            updates.append("finished_at = ?")
            values.append(time.time())
        values.append(str(task_id or ""))
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE local_semantic_component_jobs SET {', '.join(updates)} "
                "WHERE task_id = ? AND (state != 'cancelling' "
                "OR ? IN ('paused_by_user', 'removing', 'not_installed'))",
                (*values, state),
            )
        return int(cursor.rowcount or 0) > 0

    def request_cancel(self, task_id: str) -> bool:
        cancellable = COMPONENT_ACTIVE_STATES - {"cancelling", "removing"}
        placeholders = ",".join("?" for _state in cancellable)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE local_semantic_component_jobs
                SET state = 'cancelling', reason_id = 'cancel_requested',
                    eta_seconds = NULL, updated_at = ?
                WHERE task_id = ? AND state IN ({placeholders})
                """,
                (time.time(), str(task_id or ""), *tuple(cancellable)),
            )
        return int(cursor.rowcount or 0) > 0

    def request_remove(self, task_id: str) -> bool:
        cancellable = COMPONENT_ACTIVE_STATES - {"removing"}
        placeholders = ",".join("?" for _state in cancellable)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE local_semantic_component_jobs
                SET operation = 'remove', state = 'cancelling',
                    reason_id = 'remove_requested', eta_seconds = NULL,
                    updated_at = ?
                WHERE task_id = ? AND state IN ({placeholders})
                """,
                (time.time(), str(task_id or ""), *tuple(cancellable)),
            )
        return int(cursor.rowcount or 0) > 0

    def should_cancel(self, task_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state FROM local_semantic_component_jobs WHERE task_id = ?",
                (str(task_id or ""),),
            ).fetchone()
        return bool(row is not None and str(row["state"] or "") == "cancelling")

    def reconcile_after_restart(
        self,
        *,
        owner_session: str,
    ) -> tuple[LocalSemanticComponentLifecycleRecord, bool]:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM local_semantic_component_jobs
                ORDER BY updated_at DESC, created_at DESC LIMIT 1
                """
            ).fetchone()
            record = self._record(row)
            if (
                row is None
                or not component_lifecycle_is_active(record.state)
                or record.owner_session == str(owner_session or "")
            ):
                return record, False
            if record.state == "cancelling" and record.operation != "remove":
                state = "paused_by_user"
                reason_id = "cancelled_by_user"
                finished_at = now
                should_resume = False
            else:
                state = "recovering"
                reason_id = "app_restarted"
                finished_at = 0
                should_resume = record.consented or record.operation == "remove"
            connection.execute(
                """
                UPDATE local_semantic_component_jobs
                SET state = ?, reason_id = ?, owner_session = ?,
                    eta_seconds = NULL, updated_at = ?, finished_at = ?
                WHERE task_id = ?
                """,
                (
                    state,
                    reason_id,
                    str(owner_session or "")[:80],
                    now,
                    finished_at,
                    record.task_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM local_semantic_component_jobs WHERE task_id = ?",
                (record.task_id,),
            ).fetchone()
        return self._record(updated), should_resume

    def quarantine(self, manifest_identity: str, *, reason_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO local_semantic_component_quarantine (
                  manifest_identity, reason_id, created_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(manifest_identity) DO UPDATE SET
                  reason_id = excluded.reason_id,
                  created_at = excluded.created_at
                """,
                (
                    str(manifest_identity or "")[:160],
                    normalize_internal_reason(reason_id),
                    time.time(),
                ),
            )

    def quarantine_reason(self, manifest_identity: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT reason_id FROM local_semantic_component_quarantine
                WHERE manifest_identity = ?
                """,
                (str(manifest_identity or "")[:160],),
            ).fetchone()
        return normalize_internal_reason(row["reason_id"]) if row is not None else ""
