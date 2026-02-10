from __future__ import annotations

import sqlite3
import time
import uuid
import json
from pathlib import Path


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

    def _connect(self) -> sqlite3.Connection:
        # WAL helps concurrent reads while Streamlit reruns.
        conn = sqlite3.connect(str(self._db_path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
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
                  created_at REAL NOT NULL,
                  FOREIGN KEY(conv_id) REFERENCES conversations(id)
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv_id ON messages(conv_id);")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS message_refs (
                  user_msg_id INTEGER PRIMARY KEY,
                  conv_id TEXT NOT NULL,
                  prompt TEXT NOT NULL,
                  prompt_sig TEXT NOT NULL,
                  hits_json TEXT NOT NULL,
                  scores_json TEXT NOT NULL,
                  used_query TEXT NOT NULL,
                  used_translation INTEGER NOT NULL DEFAULT 0,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_message_refs_conv_id ON message_refs(conv_id);")

    def create_conversation(self, title: str = "新对话") -> str:
        conv_id = uuid.uuid4().hex
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conv_id, title.strip() or "新对话", now, now),
            )
        return conv_id

    def list_conversations(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_conversation(self, conv_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM message_refs WHERE conv_id = ?", (conv_id,))
            conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))

    def get_messages(self, conv_id: str, limit: int | None = None) -> list[dict]:
        sql = "SELECT id, role, content, created_at FROM messages WHERE conv_id = ? ORDER BY id ASC"
        params: tuple = (conv_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (conv_id, int(limit))

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_messages_upto_id(self, conv_id: str, max_id: int, limit: int | None = None) -> list[dict]:
        mid = int(max_id or 0)
        if mid <= 0:
            return self.get_messages(conv_id, limit=limit)
        sql = "SELECT id, role, content, created_at FROM messages WHERE conv_id = ? AND id <= ? ORDER BY id ASC"
        params: tuple = (conv_id, mid)
        if limit is not None:
            sql += " LIMIT ?"
            params = (conv_id, mid, int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def append_message(self, conv_id: str, role: str, content: str) -> int:
        role = (role or "").strip()
        if role not in ("user", "assistant", "system"):
            role = "user"
        content = (content or "").strip()
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO messages (conv_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conv_id, role, content, now),
            )
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
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
            row = conn.execute("SELECT conv_id FROM messages WHERE id = ?", (mid,)).fetchone()
            if not row:
                return False
            conn.execute("UPDATE messages SET content = ? WHERE id = ?", (text, mid))
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, row["conv_id"]))
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
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, row["conv_id"]))
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
        with self._connect() as conn:
            row = conn.execute("SELECT user_msg_id, created_at FROM message_refs WHERE user_msg_id = ?", (mid,)).fetchone()
            if row:
                created_at = float(row["created_at"] or now)
                conn.execute(
                    """
                    UPDATE message_refs
                    SET conv_id = ?, prompt = ?, prompt_sig = ?, hits_json = ?, scores_json = ?,
                        used_query = ?, used_translation = ?, updated_at = ?
                    WHERE user_msg_id = ?
                    """,
                    (
                        conv_id,
                        prompt,
                        prompt_sig,
                        hits_json,
                        scores_json,
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
                    (user_msg_id, conv_id, prompt, prompt_sig, hits_json, scores_json, used_query, used_translation, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        mid,
                        conv_id,
                        prompt,
                        prompt_sig,
                        hits_json,
                        scores_json,
                        used_query,
                        1 if bool(used_translation) else 0,
                        created_at,
                        now,
                    ),
                )
        return True

    def list_message_refs(self, conv_id: str) -> dict[int, dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT user_msg_id, conv_id, prompt, prompt_sig, hits_json, scores_json, used_query,
                       used_translation, created_at, updated_at
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
            out[mid] = {
                "user_msg_id": mid,
                "conv_id": str(r["conv_id"] or ""),
                "prompt": str(r["prompt"] or ""),
                "prompt_sig": str(r["prompt_sig"] or ""),
                "hits": hits,
                "scores": scores,
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
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (new_title, now, conv_id),
            )
