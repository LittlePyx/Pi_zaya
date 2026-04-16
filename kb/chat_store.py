from __future__ import annotations

import sqlite3
import time
import uuid
import json
from pathlib import Path

DEFAULT_ACTIVE_CONVERSATION_LIMIT = 400


def _normalize_conversation_mode(mode: str) -> str:
    m = str(mode or "").strip().lower()
    if m in {"paper_guide", "normal"}:
        return m
    return "normal"


class ChatStore:
    """
    A tiny local chat persistence layer.
    - One sqlite file
    - Multiple conversations
    - Append-only messages
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self, *, timeout_s: float | None = None) -> sqlite3.Connection:
        # WAL helps concurrent reads while Streamlit reruns.
        try:
            timeout_final = float(timeout_s if timeout_s is not None else 30.0)
        except Exception:
            timeout_final = 30.0
        timeout_final = max(0.05, timeout_final)
        conn = sqlite3.connect(str(self._db_path), timeout=timeout_final, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            conn.execute(f"PRAGMA busy_timeout={int(timeout_final * 1000)};")
        except Exception:
            pass
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                  id TEXT PRIMARY KEY,
                  title TEXT NOT NULL,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  conv_id TEXT NOT NULL,
                  role TEXT NOT NULL,
                  content TEXT NOT NULL,
                  attachments_json TEXT NOT NULL DEFAULT '[]',
                  meta_json TEXT NOT NULL DEFAULT '{}',
                  created_at REAL NOT NULL,
                  FOREIGN KEY(conv_id) REFERENCES conversations(id)
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv_id ON messages(conv_id);")
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN attachments_json TEXT NOT NULL DEFAULT '[]'")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN meta_json TEXT NOT NULL DEFAULT '{}'")
            except sqlite3.OperationalError:
                pass
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS message_refs (
                  user_msg_id INTEGER PRIMARY KEY,
                  conv_id TEXT NOT NULL,
                  prompt TEXT NOT NULL,
                  prompt_sig TEXT NOT NULL,
                  hits_json TEXT NOT NULL,
                  scores_json TEXT NOT NULL,
                  rendered_payload_json TEXT NOT NULL DEFAULT '',
                  rendered_payload_sig TEXT NOT NULL DEFAULT '',
                  render_status TEXT NOT NULL DEFAULT '',
                  render_error TEXT NOT NULL DEFAULT '',
                  render_error_detail TEXT NOT NULL DEFAULT '',
                  render_built_at REAL NOT NULL DEFAULT 0,
                  render_attempts INTEGER NOT NULL DEFAULT 0,
                  render_evidence_sig TEXT NOT NULL DEFAULT '',
                  render_locale TEXT NOT NULL DEFAULT '',
                  used_query TEXT NOT NULL,
                  used_translation INTEGER NOT NULL DEFAULT 0,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL
                );
                """
            )
            try:
                conn.execute("ALTER TABLE message_refs ADD COLUMN rendered_payload_json TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE message_refs ADD COLUMN rendered_payload_sig TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE message_refs ADD COLUMN render_status TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE message_refs ADD COLUMN render_error TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE message_refs ADD COLUMN render_error_detail TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE message_refs ADD COLUMN render_built_at REAL NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE message_refs ADD COLUMN render_attempts INTEGER NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE message_refs ADD COLUMN render_evidence_sig TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE message_refs ADD COLUMN render_locale TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_message_refs_conv_id ON message_refs(conv_id);")
            # Projects (ChatGPT-style): optional grouping for conversations
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                  id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL
                );
                """
            )
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN project_id TEXT;")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN archived INTEGER NOT NULL DEFAULT 0;")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN archived_at REAL;")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN mode TEXT NOT NULL DEFAULT 'normal';")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN bound_source_path TEXT NOT NULL DEFAULT '';")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN bound_source_name TEXT NOT NULL DEFAULT '';")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN bound_source_ready INTEGER NOT NULL DEFAULT 0;")
            except sqlite3.OperationalError:
                pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_project_id ON conversations(project_id);")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_scope_archived_updated "
                "ON conversations(project_id, archived, updated_at DESC);"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_sources (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  conv_id TEXT NOT NULL,
                  source_path TEXT NOT NULL,
                  source_name TEXT NOT NULL,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL,
                  UNIQUE(conv_id, source_path)
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversation_sources_conv_id ON conversation_sources(conv_id);")

    def _archive_excess_conversations(
        self,
        conn: sqlite3.Connection,
        *,
        project_id: str | None,
        active_limit: int = DEFAULT_ACTIVE_CONVERSATION_LIMIT,
    ) -> int:
        keep_n = max(1, int(active_limit))
        if project_id is None:
            rows = conn.execute(
                """
                SELECT id
                FROM conversations
                WHERE project_id IS NULL AND COALESCE(archived, 0) = 0
                ORDER BY updated_at DESC, created_at DESC, id DESC
                LIMIT -1 OFFSET ?
                """,
                (keep_n,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id
                FROM conversations
                WHERE project_id = ? AND COALESCE(archived, 0) = 0
                ORDER BY updated_at DESC, created_at DESC, id DESC
                LIMIT -1 OFFSET ?
                """,
                (project_id, keep_n),
            ).fetchall()
        ids = [str(r["id"] or "").strip() for r in rows if str(r["id"] or "").strip()]
        if not ids:
            return 0
        now = time.time()
        conn.executemany(
            "UPDATE conversations SET archived = 1, archived_at = ? WHERE id = ?",
            [(now, cid) for cid in ids],
        )
        return len(ids)

    def _touch_conversation_active(self, conn: sqlite3.Connection, conv_id: str, now: float) -> str | None:
        row = conn.execute("SELECT project_id FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        if not row:
            return None
        project_id = row["project_id"]
        conn.execute(
            "UPDATE conversations SET updated_at = ?, archived = 0, archived_at = NULL WHERE id = ?",
            (now, conv_id),
        )
        return str(project_id).strip() if isinstance(project_id, str) and project_id.strip() else None

    def create_project(self, name: str) -> str:
        pid = uuid.uuid4().hex
        now = time.time()
        name = (name or "").strip() or "未命名项目"
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO projects (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (pid, name, now, now),
            )
        return pid

    def list_projects(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, created_at, updated_at FROM projects ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_project(self, project_id: str) -> dict | None:
        if not (project_id or "").strip():
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, name, created_at, updated_at FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
        return dict(row) if row else None

    def rename_project(self, project_id: str, name: str) -> bool:
        if not (project_id or "").strip():
            return False
        name = (name or "").strip()
        if not name:
            return False
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE projects SET name = ?, updated_at = ? WHERE id = ?",
                (name, now, project_id),
            )
        return cur.rowcount > 0

    def delete_project(self, project_id: str) -> None:
        if not (project_id or "").strip():
            return
        with self._connect() as conn:
            conn.execute("UPDATE conversations SET project_id = NULL WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    def create_conversation(
        self,
        title: str = "新对话",
        project_id: str | None = None,
        *,
        mode: str = "normal",
        bound_source_path: str = "",
        bound_source_name: str = "",
        bound_source_ready: bool = False,
    ) -> str:
        conv_id = uuid.uuid4().hex
        now = time.time()
        mode_norm = _normalize_conversation_mode(mode)
        source_path = str(bound_source_path or "").strip()
        source_name = str(bound_source_name or "").strip()
        source_ready = 1 if bool(bound_source_ready and source_path) else 0
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations ("
                "id, title, created_at, updated_at, project_id, archived, archived_at, "
                "mode, bound_source_path, bound_source_name, bound_source_ready"
                ") VALUES (?, ?, ?, ?, ?, 0, NULL, ?, ?, ?, ?)",
                (
                    conv_id,
                    title.strip() or "新对话",
                    now,
                    now,
                    project_id,
                    mode_norm,
                    source_path,
                    source_name,
                    source_ready,
                ),
            )
            if source_path:
                conn.execute(
                    """
                    INSERT INTO conversation_sources (conv_id, source_path, source_name, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(conv_id, source_path)
                    DO UPDATE SET source_name = excluded.source_name, updated_at = excluded.updated_at
                    """,
                    (conv_id, source_path, source_name or Path(source_path).name, now, now),
                )
            self._archive_excess_conversations(conn, project_id=project_id)
        return conv_id

    def list_conversations(
        self,
        project_id: str | None = None,
        limit: int = 50,
        *,
        include_archived: bool = False,
    ) -> list[dict]:
        with self._connect() as conn:
            if not bool(include_archived):
                self._archive_excess_conversations(conn, project_id=project_id)
            if project_id is None:
                rows = conn.execute(
                    "SELECT id, title, created_at, updated_at, project_id, "
                    "COALESCE(archived, 0) AS archived, archived_at, "
                    "COALESCE(mode, 'normal') AS mode, "
                    "COALESCE(bound_source_path, '') AS bound_source_path, "
                    "COALESCE(bound_source_name, '') AS bound_source_name, "
                    "COALESCE(bound_source_ready, 0) AS bound_source_ready "
                    "FROM conversations "
                    "WHERE project_id IS NULL "
                    + ("" if include_archived else "AND COALESCE(archived, 0) = 0 ")
                    + "ORDER BY updated_at DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, title, created_at, updated_at, project_id, "
                    "COALESCE(archived, 0) AS archived, archived_at, "
                    "COALESCE(mode, 'normal') AS mode, "
                    "COALESCE(bound_source_path, '') AS bound_source_path, "
                    "COALESCE(bound_source_name, '') AS bound_source_name, "
                    "COALESCE(bound_source_ready, 0) AS bound_source_ready "
                    "FROM conversations "
                    "WHERE project_id = ? "
                    + ("" if include_archived else "AND COALESCE(archived, 0) = 0 ")
                    + "ORDER BY updated_at DESC LIMIT ?",
                    (project_id, int(limit)),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, conv_id: str, *, timeout_s: float | None = None) -> dict | None:
        conv_id = (conv_id or "").strip()
        if not conv_id:
            return None
        with self._connect(timeout_s=timeout_s) as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at, project_id, "
                "COALESCE(archived, 0) AS archived, archived_at, "
                "COALESCE(mode, 'normal') AS mode, "
                "COALESCE(bound_source_path, '') AS bound_source_path, "
                "COALESCE(bound_source_name, '') AS bound_source_name, "
                "COALESCE(bound_source_ready, 0) AS bound_source_ready "
                "FROM conversations WHERE id = ?",
                (conv_id,),
            ).fetchone()
        return dict(row) if row else None

    def set_conversation_guide(
        self,
        conv_id: str,
        *,
        mode: str | None = None,
        bound_source_path: str | None = None,
        bound_source_name: str | None = None,
        bound_source_ready: bool | None = None,
    ) -> bool:
        cid = str(conv_id or "").strip()
        if not cid:
            return False
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT mode, bound_source_path, bound_source_name, bound_source_ready, project_id "
                "FROM conversations WHERE id = ?",
                (cid,),
            ).fetchone()
            if not row:
                return False
            mode_cur = _normalize_conversation_mode(str(row["mode"] or "normal"))
            path_cur = str(row["bound_source_path"] or "").strip()
            name_cur = str(row["bound_source_name"] or "").strip()
            ready_cur = bool(int(row["bound_source_ready"] or 0))
            mode_next = mode_cur if mode is None else _normalize_conversation_mode(mode)
            path_next = path_cur if bound_source_path is None else str(bound_source_path or "").strip()
            name_next = name_cur if bound_source_name is None else str(bound_source_name or "").strip()
            if bound_source_ready is None:
                ready_next = ready_cur
            else:
                ready_next = bool(bound_source_ready and path_next)
            if not path_next:
                ready_next = False
            conn.execute(
                "UPDATE conversations "
                "SET mode = ?, bound_source_path = ?, bound_source_name = ?, bound_source_ready = ?, "
                "updated_at = ?, archived = 0, archived_at = NULL "
                "WHERE id = ?",
                (
                    mode_next,
                    path_next,
                    name_next,
                    1 if ready_next else 0,
                    now,
                    cid,
                ),
            )
            if path_next:
                conn.execute(
                    """
                    INSERT INTO conversation_sources (conv_id, source_path, source_name, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(conv_id, source_path)
                    DO UPDATE SET source_name = excluded.source_name, updated_at = excluded.updated_at
                    """,
                    (cid, path_next, name_next or Path(path_next).name, now, now),
                )
            project_id = str(row["project_id"] or "").strip() or None
            self._archive_excess_conversations(conn, project_id=project_id)
        return True

    def set_conversation_project(self, conv_id: str, project_id: str | None) -> bool:
        conv_id = (conv_id or "").strip()
        if not conv_id:
            return False
        now = time.time()
        with self._connect() as conn:
            old = conn.execute("SELECT project_id FROM conversations WHERE id = ?", (conv_id,)).fetchone()
            if not old:
                return False
            old_project_id = old["project_id"]
            cur = conn.execute(
                "UPDATE conversations SET project_id = ?, updated_at = ?, archived = 0, archived_at = NULL WHERE id = ?",
                (project_id, now, conv_id),
            )
            old_pid = str(old_project_id).strip() if isinstance(old_project_id, str) and old_project_id.strip() else None
            self._archive_excess_conversations(conn, project_id=project_id)
            if old_pid != project_id:
                self._archive_excess_conversations(conn, project_id=old_pid)
        return cur.rowcount > 0

    def delete_conversation(self, conv_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM conversation_sources WHERE conv_id = ?", (conv_id,))
            conn.execute("DELETE FROM message_refs WHERE conv_id = ?", (conv_id,))
            conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))

    def bind_conversation_source(self, conv_id: str, source_path: str, source_name: str = "") -> bool:
        cid = str(conv_id or "").strip()
        src = str(source_path or "").strip()
        if (not cid) or (not src):
            return False
        name = str(source_name or "").strip() or Path(src).name
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_sources (conv_id, source_path, source_name, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(conv_id, source_path)
                DO UPDATE SET source_name = excluded.source_name, updated_at = excluded.updated_at
                """,
                (cid, src, name, now, now),
            )
            project_id = self._touch_conversation_active(conn, cid, now)
            self._archive_excess_conversations(conn, project_id=project_id)
        return True

    def list_conversation_sources(self, conv_id: str, limit: int = 8) -> list[dict]:
        cid = str(conv_id or "").strip()
        if not cid:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT source_path, source_name, created_at, updated_at
                FROM conversation_sources
                WHERE conv_id = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (cid, max(1, int(limit))),
            ).fetchall()
        return [
            {
                "source_path": str(r["source_path"] or ""),
                "source_name": str(r["source_name"] or ""),
                "created_at": float(r["created_at"] or 0.0),
                "updated_at": float(r["updated_at"] or 0.0),
            }
            for r in rows
        ]

    def get_messages(self, conv_id: str, limit: int | None = None) -> list[dict]:
        sql = "SELECT id, role, content, attachments_json, meta_json, created_at FROM messages WHERE conv_id = ? ORDER BY id ASC"
        params: tuple = (conv_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (conv_id, int(limit))

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return self._hydrate_message_rows(rows)

    def get_messages_upto_id(self, conv_id: str, max_id: int, limit: int | None = None) -> list[dict]:
        mid = int(max_id or 0)
        if mid <= 0:
            return self.get_messages(conv_id, limit=limit)
        sql = "SELECT id, role, content, attachments_json, meta_json, created_at FROM messages WHERE conv_id = ? AND id <= ? ORDER BY id ASC"
        params: tuple = (conv_id, mid)
        if limit is not None:
            sql += " LIMIT ?"
            params = (conv_id, mid, int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return self._hydrate_message_rows(rows)

    def get_messages_page(
        self,
        conv_id: str,
        limit: int = 24,
        before_id: int | None = None,
    ) -> tuple[list[dict], bool, int | None, int | None]:
        page_size = max(1, min(200, int(limit or 24)))
        before = int(before_id or 0)
        params: tuple
        sql = (
            "SELECT id, role, content, attachments_json, meta_json, created_at "
            "FROM messages WHERE conv_id = ?"
        )
        params = (conv_id,)
        if before > 0:
            sql += " AND id < ?"
            params = (conv_id, before)
        sql += " ORDER BY id DESC LIMIT ?"
        params = (*params, page_size + 1)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        has_more_before = len(rows) > page_size
        page_rows = list(rows[:page_size])
        page_rows.reverse()
        out = self._hydrate_message_rows(page_rows)
        oldest_loaded_id = int(out[0]["id"]) if out else None
        newest_loaded_id = int(out[-1]["id"]) if out else None
        return out, has_more_before, oldest_loaded_id, newest_loaded_id

    def _hydrate_message_rows(self, rows: list[sqlite3.Row]) -> list[dict]:
        out: list[dict] = []
        for row in rows:
            rec = dict(row)
            try:
                attachments = json.loads(rec.get("attachments_json") or "[]")
            except Exception:
                attachments = []
            if not isinstance(attachments, list):
                attachments = []
            try:
                meta = json.loads(rec.get("meta_json") or "{}")
            except Exception:
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            rec["attachments"] = attachments
            rec["meta"] = meta
            if isinstance(meta.get("provenance"), dict):
                rec["provenance"] = dict(meta.get("provenance") or {})
            rec.pop("attachments_json", None)
            rec.pop("meta_json", None)
            out.append(rec)
        return out

    def append_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        attachments: list[dict] | None = None,
        meta: dict | None = None,
    ) -> int:
        role = (role or "").strip()
        if role not in ("user", "assistant", "system"):
            role = "user"
        content = (content or "").strip()
        try:
            attachments_json = json.dumps(list(attachments or []), ensure_ascii=False, default=str)
        except Exception:
            attachments_json = "[]"
        try:
            meta_json = json.dumps(dict(meta or {}), ensure_ascii=False, default=str)
        except Exception:
            meta_json = "{}"
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO messages (conv_id, role, content, attachments_json, meta_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (conv_id, role, content, attachments_json, meta_json, now),
            )
            project_id = self._touch_conversation_active(conn, conv_id, now)
            self._archive_excess_conversations(conn, project_id=project_id)
            try:
                return int(cur.lastrowid or 0)
            except Exception:
                return 0

    def update_message_content(self, message_id: int, content: str) -> bool:
        mid = int(message_id or 0)
        if mid <= 0:
            return False
        text = (content or "").strip()
        now = time.time()
        with self._connect() as conn:
            row = conn.execute("SELECT conv_id, meta_json FROM messages WHERE id = ?", (mid,)).fetchone()
            if not row:
                return False
            try:
                meta = json.loads(row["meta_json"] or "{}")
            except Exception:
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            meta.pop("render_cache", None)
            try:
                meta_json = json.dumps(meta, ensure_ascii=False, default=str)
            except Exception:
                meta_json = "{}"
            conn.execute("UPDATE messages SET content = ?, meta_json = ? WHERE id = ?", (text, meta_json, mid))
            project_id = self._touch_conversation_active(conn, str(row["conv_id"] or ""), now)
            self._archive_excess_conversations(conn, project_id=project_id)
        return True

    def update_message_meta(self, message_id: int, meta: dict) -> bool:
        mid = int(message_id or 0)
        if mid <= 0:
            return False
        try:
            meta_json = json.dumps(dict(meta or {}), ensure_ascii=False, default=str)
        except Exception:
            meta_json = "{}"
        now = time.time()
        with self._connect() as conn:
            row = conn.execute("SELECT conv_id FROM messages WHERE id = ?", (mid,)).fetchone()
            if not row:
                return False
            conn.execute("UPDATE messages SET meta_json = ? WHERE id = ?", (meta_json, mid))
            project_id = self._touch_conversation_active(conn, str(row["conv_id"] or ""), now)
            self._archive_excess_conversations(conn, project_id=project_id)
        return True

    def merge_message_meta(self, message_id: int, patch: dict) -> bool:
        mid = int(message_id or 0)
        if mid <= 0:
            return False
        patch_dict = dict(patch or {})
        now = time.time()
        with self._connect() as conn:
            row = conn.execute("SELECT conv_id, meta_json FROM messages WHERE id = ?", (mid,)).fetchone()
            if not row:
                return False
            try:
                current = json.loads(row["meta_json"] or "{}")
            except Exception:
                current = {}
            if not isinstance(current, dict):
                current = {}
            current.update(patch_dict)
            try:
                meta_json = json.dumps(current, ensure_ascii=False, default=str)
            except Exception:
                meta_json = "{}"
            conn.execute("UPDATE messages SET meta_json = ? WHERE id = ?", (meta_json, mid))
            project_id = self._touch_conversation_active(conn, str(row["conv_id"] or ""), now)
            self._archive_excess_conversations(conn, project_id=project_id)
        return True

    def set_message_render_cache(self, message_id: int, cache_payload: dict | None) -> bool:
        mid = int(message_id or 0)
        if mid <= 0:
            return False
        with self._connect() as conn:
            row = conn.execute("SELECT meta_json FROM messages WHERE id = ?", (mid,)).fetchone()
            if not row:
                return False
            try:
                current = json.loads(row["meta_json"] or "{}")
            except Exception:
                current = {}
            if not isinstance(current, dict):
                current = {}
            next_meta = dict(current)
            if isinstance(cache_payload, dict) and cache_payload:
                next_meta["render_cache"] = dict(cache_payload)
            else:
                next_meta.pop("render_cache", None)
            if next_meta == current:
                return True
            try:
                meta_json = json.dumps(next_meta, ensure_ascii=False, default=str)
            except Exception:
                meta_json = "{}"
            conn.execute("UPDATE messages SET meta_json = ? WHERE id = ?", (meta_json, mid))
        return True

    def delete_message(self, message_id: int) -> bool:
        mid = int(message_id or 0)
        if mid <= 0:
            return False
        now = time.time()
        with self._connect() as conn:
            row = conn.execute("SELECT conv_id FROM messages WHERE id = ?", (mid,)).fetchone()
            if not row:
                return False
            conn.execute("DELETE FROM message_refs WHERE user_msg_id = ?", (mid,))
            conn.execute("DELETE FROM messages WHERE id = ?", (mid,))
            project_id = self._touch_conversation_active(conn, str(row["conv_id"] or ""), now)
            self._archive_excess_conversations(conn, project_id=project_id)
        return True

    def upsert_message_refs(
        self,
        *,
        user_msg_id: int,
        conv_id: str,
        prompt: str,
        prompt_sig: str,
        hits: list[dict],
        scores: list[float],
        used_query: str,
        used_translation: bool,
        rendered_payload: dict | None = None,
        rendered_payload_sig: str = "",
        render_status: str | None = None,
        render_error: str | None = None,
        render_error_detail: str | None = None,
        render_built_at: float | None = None,
        render_attempts: int | None = None,
        render_evidence_sig: str | None = None,
        render_locale: str | None = None,
    ) -> bool:
        mid = int(user_msg_id or 0)
        if mid <= 0:
            return False
        now = time.time()
        conv_id = (conv_id or "").strip()
        prompt = (prompt or "").strip()
        prompt_sig = (prompt_sig or "").strip()
        used_query = (used_query or "").strip()
        try:
            hits_json = json.dumps(list(hits or []), ensure_ascii=False, default=str)
        except Exception:
            hits_json = "[]"
        try:
            scores_json = json.dumps(list(scores or []), ensure_ascii=False, default=str)
        except Exception:
            scores_json = "[]"
        try:
            rendered_payload_json = (
                json.dumps(dict(rendered_payload or {}), ensure_ascii=False, default=str)
                if isinstance(rendered_payload, dict)
                else ""
            )
        except Exception:
            rendered_payload_json = ""
        rendered_sig = str(rendered_payload_sig or "").strip() if rendered_payload_json else ""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT user_msg_id, created_at, render_status, render_error, render_error_detail,
                       render_built_at, render_attempts, render_evidence_sig, render_locale
                FROM message_refs
                WHERE user_msg_id = ?
                """,
                (mid,),
            ).fetchone()
            next_render_status = str(render_status).strip() if render_status is not None else str((row["render_status"] if row else "") or "").strip()
            next_render_error = str(render_error).strip() if render_error is not None else str((row["render_error"] if row else "") or "").strip()
            next_render_error_detail = (
                str(render_error_detail).strip()
                if render_error_detail is not None
                else str((row["render_error_detail"] if row else "") or "").strip()
            )
            try:
                next_render_built_at = float(render_built_at) if render_built_at is not None else float((row["render_built_at"] if row else 0.0) or 0.0)
            except Exception:
                next_render_built_at = 0.0
            try:
                next_render_attempts = int(render_attempts) if render_attempts is not None else int((row["render_attempts"] if row else 0) or 0)
            except Exception:
                next_render_attempts = 0
            next_render_attempts = max(0, next_render_attempts)
            next_render_evidence_sig = (
                str(render_evidence_sig).strip()
                if render_evidence_sig is not None
                else str((row["render_evidence_sig"] if row else "") or "").strip()
            )
            next_render_locale = (
                str(render_locale).strip()
                if render_locale is not None
                else str((row["render_locale"] if row else "") or "").strip()
            )
            if row:
                created_at = float(row["created_at"] or now)
                conn.execute(
                    """
                    UPDATE message_refs
                    SET conv_id = ?, prompt = ?, prompt_sig = ?, hits_json = ?, scores_json = ?,
                        rendered_payload_json = ?, rendered_payload_sig = ?,
                        render_status = ?, render_error = ?, render_error_detail = ?,
                        render_built_at = ?, render_attempts = ?, render_evidence_sig = ?, render_locale = ?,
                        used_query = ?, used_translation = ?, updated_at = ?
                    WHERE user_msg_id = ?
                    """,
                    (
                        conv_id,
                        prompt,
                        prompt_sig,
                        hits_json,
                        scores_json,
                        rendered_payload_json,
                        rendered_sig,
                        next_render_status,
                        next_render_error,
                        next_render_error_detail,
                        next_render_built_at,
                        next_render_attempts,
                        next_render_evidence_sig,
                        next_render_locale,
                        used_query,
                        1 if bool(used_translation) else 0,
                        now,
                        mid,
                    ),
                )
            else:
                created_at = now
                conn.execute(
                    """
                    INSERT INTO message_refs
                    (
                        user_msg_id, conv_id, prompt, prompt_sig, hits_json, scores_json,
                        rendered_payload_json, rendered_payload_sig,
                        render_status, render_error, render_error_detail,
                        render_built_at, render_attempts, render_evidence_sig, render_locale,
                        used_query, used_translation, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        mid,
                        conv_id,
                        prompt,
                        prompt_sig,
                        hits_json,
                        scores_json,
                        rendered_payload_json,
                        rendered_sig,
                        next_render_status,
                        next_render_error,
                        next_render_error_detail,
                        next_render_built_at,
                        next_render_attempts,
                        next_render_evidence_sig,
                        next_render_locale,
                        used_query,
                        1 if bool(used_translation) else 0,
                        created_at,
                        now,
                    ),
                )
        return True

    def set_message_refs_rendered_payload(
        self,
        *,
        user_msg_id: int,
        rendered_payload: dict | None,
        rendered_payload_sig: str = "",
        render_status: str | None = None,
        render_error: str | None = None,
        render_error_detail: str | None = None,
        render_built_at: float | None = None,
        render_attempts: int | None = None,
        render_evidence_sig: str | None = None,
        render_locale: str | None = None,
    ) -> bool:
        mid = int(user_msg_id or 0)
        if mid <= 0:
            return False
        try:
            rendered_payload_json = (
                json.dumps(dict(rendered_payload or {}), ensure_ascii=False, default=str)
                if isinstance(rendered_payload, dict)
                else ""
            )
        except Exception:
            rendered_payload_json = ""
        rendered_sig = str(rendered_payload_sig or "").strip() if rendered_payload_json else ""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT user_msg_id, render_status, render_error, render_error_detail,
                       render_built_at, render_attempts, render_evidence_sig, render_locale
                FROM message_refs
                WHERE user_msg_id = ?
                """,
                (mid,),
            ).fetchone()
            if not row:
                return False
            next_render_status = str(render_status).strip() if render_status is not None else str(row["render_status"] or "").strip()
            next_render_error = str(render_error).strip() if render_error is not None else str(row["render_error"] or "").strip()
            next_render_error_detail = (
                str(render_error_detail).strip()
                if render_error_detail is not None
                else str(row["render_error_detail"] or "").strip()
            )
            try:
                next_render_built_at = float(render_built_at) if render_built_at is not None else float(row["render_built_at"] or 0.0)
            except Exception:
                next_render_built_at = 0.0
            try:
                next_render_attempts = int(render_attempts) if render_attempts is not None else int(row["render_attempts"] or 0)
            except Exception:
                next_render_attempts = 0
            next_render_attempts = max(0, next_render_attempts)
            next_render_evidence_sig = (
                str(render_evidence_sig).strip()
                if render_evidence_sig is not None
                else str(row["render_evidence_sig"] or "").strip()
            )
            next_render_locale = (
                str(render_locale).strip()
                if render_locale is not None
                else str(row["render_locale"] or "").strip()
            )
            conn.execute(
                """
                UPDATE message_refs
                SET rendered_payload_json = ?, rendered_payload_sig = ?,
                    render_status = ?, render_error = ?, render_error_detail = ?,
                    render_built_at = ?, render_attempts = ?, render_evidence_sig = ?, render_locale = ?
                WHERE user_msg_id = ?
                """,
                (
                    rendered_payload_json,
                    rendered_sig,
                    next_render_status,
                    next_render_error,
                    next_render_error_detail,
                    next_render_built_at,
                    next_render_attempts,
                    next_render_evidence_sig,
                    next_render_locale,
                    mid,
                ),
            )
        return True

    def set_message_refs_render_state(
        self,
        *,
        user_msg_id: int,
        render_status: str | None = None,
        render_error: str | None = None,
        render_error_detail: str | None = None,
        render_built_at: float | None = None,
        render_attempts: int | None = None,
        render_evidence_sig: str | None = None,
        render_locale: str | None = None,
    ) -> bool:
        mid = int(user_msg_id or 0)
        if mid <= 0:
            return False
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT user_msg_id, render_status, render_error, render_error_detail,
                       render_built_at, render_attempts, render_evidence_sig, render_locale
                FROM message_refs
                WHERE user_msg_id = ?
                """,
                (mid,),
            ).fetchone()
            if not row:
                return False
            next_render_status = str(render_status).strip() if render_status is not None else str(row["render_status"] or "").strip()
            next_render_error = str(render_error).strip() if render_error is not None else str(row["render_error"] or "").strip()
            next_render_error_detail = (
                str(render_error_detail).strip()
                if render_error_detail is not None
                else str(row["render_error_detail"] or "").strip()
            )
            try:
                next_render_built_at = float(render_built_at) if render_built_at is not None else float(row["render_built_at"] or 0.0)
            except Exception:
                next_render_built_at = 0.0
            try:
                next_render_attempts = int(render_attempts) if render_attempts is not None else int(row["render_attempts"] or 0)
            except Exception:
                next_render_attempts = 0
            next_render_attempts = max(0, next_render_attempts)
            next_render_evidence_sig = (
                str(render_evidence_sig).strip()
                if render_evidence_sig is not None
                else str(row["render_evidence_sig"] or "").strip()
            )
            next_render_locale = (
                str(render_locale).strip()
                if render_locale is not None
                else str(row["render_locale"] or "").strip()
            )
            conn.execute(
                """
                UPDATE message_refs
                SET render_status = ?, render_error = ?, render_error_detail = ?,
                    render_built_at = ?, render_attempts = ?, render_evidence_sig = ?, render_locale = ?
                WHERE user_msg_id = ?
                """,
                (
                    next_render_status,
                    next_render_error,
                    next_render_error_detail,
                    next_render_built_at,
                    next_render_attempts,
                    next_render_evidence_sig,
                    next_render_locale,
                    mid,
                ),
            )
        return True

    def list_message_refs(self, conv_id: str, *, timeout_s: float | None = None) -> dict[int, dict]:
        with self._connect(timeout_s=timeout_s) as conn:
            rows = conn.execute(
                """
                SELECT user_msg_id, conv_id, prompt, prompt_sig, hits_json, scores_json,
                       rendered_payload_json, rendered_payload_sig,
                       render_status, render_error, render_error_detail,
                       render_built_at, render_attempts, render_evidence_sig, render_locale,
                       used_query, used_translation, created_at, updated_at
                FROM message_refs
                WHERE conv_id = ?
                ORDER BY user_msg_id ASC
                """,
                (conv_id,),
            ).fetchall()
        out: dict[int, dict] = {}
        for r in rows:
            try:
                mid = int(r["user_msg_id"] or 0)
            except Exception:
                mid = 0
            if mid <= 0:
                continue
            try:
                hits = json.loads(r["hits_json"] or "[]")
            except Exception:
                hits = []
            if not isinstance(hits, list):
                hits = []
            try:
                scores = json.loads(r["scores_json"] or "[]")
            except Exception:
                scores = []
            if not isinstance(scores, list):
                scores = []
            try:
                rendered_payload = json.loads(r["rendered_payload_json"] or "{}")
            except Exception:
                rendered_payload = {}
            if not isinstance(rendered_payload, dict):
                rendered_payload = {}
            out[mid] = {
                "user_msg_id": mid,
                "conv_id": str(r["conv_id"] or ""),
                "prompt": str(r["prompt"] or ""),
                "prompt_sig": str(r["prompt_sig"] or ""),
                "hits": hits,
                "scores": scores,
                "rendered_payload": rendered_payload,
                "rendered_payload_sig": str(r["rendered_payload_sig"] or ""),
                "render_status": str(r["render_status"] or ""),
                "render_error": str(r["render_error"] or ""),
                "render_error_detail": str(r["render_error_detail"] or ""),
                "render_built_at": float(r["render_built_at"] or 0.0),
                "render_attempts": int(r["render_attempts"] or 0),
                "render_evidence_sig": str(r["render_evidence_sig"] or ""),
                "render_locale": str(r["render_locale"] or ""),
                "used_query": str(r["used_query"] or ""),
                "used_translation": bool(int(r["used_translation"] or 0)),
                "created_at": float(r["created_at"] or 0.0),
                "updated_at": float(r["updated_at"] or 0.0),
            }
        return out

    def set_title_if_default(self, conv_id: str, new_title: str) -> None:
        new_title = (new_title or "").strip()
        if not new_title:
            return
        new_title = new_title.replace("\n", " ").strip()
        new_title = new_title[:80]

        with self._connect() as conn:
            row = conn.execute("SELECT title FROM conversations WHERE id = ?", (conv_id,)).fetchone()
            if not row:
                return
            if (row["title"] or "").strip() not in ("新对话", ""):
                return
            now = time.time()
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ?, archived = 0, archived_at = NULL WHERE id = ?",
                (new_title, now, conv_id),
            )
            project_id = self._touch_conversation_active(conn, conv_id, now)
            self._archive_excess_conversations(conn, project_id=project_id)
