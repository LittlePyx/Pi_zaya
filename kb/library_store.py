# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


class LibraryStore:
    """
    Minimal PDF library index:
    - keyed by sha1 to detect duplicates quickly
    - stores final pdf path and created_at
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pdf_files (
                  sha1 TEXT PRIMARY KEY,
                  path TEXT NOT NULL,
                  created_at REAL NOT NULL,
                  citation_meta TEXT
                );
                """
            )
            # Add citation_meta column if it doesn't exist (for existing databases)
            try:
                conn.execute("ALTER TABLE pdf_files ADD COLUMN citation_meta TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

    def get_by_sha1(self, sha1: str) -> dict | None:
        sha1 = (sha1 or "").strip().lower()
        if not sha1:
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT sha1, path, created_at, citation_meta FROM pdf_files WHERE sha1 = ?", (sha1,)).fetchone()
        return dict(row) if row else None
    
    def get_by_path(self, path: Path) -> dict | None:
        """Get PDF record by path (for citation lookup)."""
        path_s = str(Path(path))
        with self._connect() as conn:
            row = conn.execute("SELECT sha1, path, created_at, citation_meta FROM pdf_files WHERE path = ?", (path_s,)).fetchone()
        return dict(row) if row else None
    
    def get_citation_meta(self, path: Path) -> dict | None:
        """Get stored citation metadata for a PDF path."""
        record = self.get_by_path(path)
        if not record or not record.get("citation_meta"):
            return None
        try:
            import json
            return json.loads(record["citation_meta"])
        except Exception:
            return None
    
    def set_citation_meta(self, path: Path, citation_meta: dict | None) -> None:
        """Store citation metadata for a PDF path."""
        path_s = str(Path(path))
        import json
        meta_json = json.dumps(citation_meta) if citation_meta else None
        with self._connect() as conn:
            conn.execute(
                "UPDATE pdf_files SET citation_meta = ? WHERE path = ?",
                (meta_json, path_s)
            )

    def upsert(self, sha1: str, path: Path, citation_meta: dict | None = None) -> None:
        sha1 = (sha1 or "").strip().lower()
        path_s = str(Path(path))
        now = time.time()
        import json
        meta_json = json.dumps(citation_meta) if citation_meta else None
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO pdf_files (sha1, path, created_at, citation_meta) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(sha1) DO UPDATE SET path=excluded.path, citation_meta=excluded.citation_meta",
                (sha1, path_s, now, meta_json),
            )

    def update_path(self, old_path: Path, new_path: Path) -> int:
        """
        Best-effort path update when a PDF file is renamed/moved on disk.
        Returns affected rows count.
        """
        old_s = str(Path(old_path))
        new_s = str(Path(new_path))
        with self._connect() as conn:
            cur = conn.execute("UPDATE pdf_files SET path = ? WHERE path = ?", (new_s, old_s))
            return int(getattr(cur, "rowcount", 0) or 0)

    def delete_by_path(self, path: Path) -> int:
        """
        Best-effort removal when a PDF file is deleted on disk.
        Returns affected rows count.
        """
        path_s = str(Path(path))
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM pdf_files WHERE path = ?", (path_s,))
            return int(getattr(cur, "rowcount", 0) or 0)
