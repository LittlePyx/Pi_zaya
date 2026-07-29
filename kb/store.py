from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import threading
import time
from contextlib import AbstractContextManager
from errno import EACCES, EAGAIN
from pathlib import Path
from typing import BinaryIO


_DB_WRITE_LOCK_FILE = ".kb_write.lock"
_DB_WRITE_LOCKS_GUARD = threading.Lock()
_DB_WRITE_LOCKS: dict[str, threading.Lock] = {}


def _db_lock_timeout_s(value: float | None = None) -> float:
    if value is not None:
        return max(0.05, float(value))
    try:
        return max(0.05, float(os.environ.get("KB_DB_WRITE_LOCK_TIMEOUT_S", "600") or 600))
    except Exception:
        return 600.0


def _thread_lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve(strict=False)).casefold()
    with _DB_WRITE_LOCKS_GUARD:
        lock = _DB_WRITE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _DB_WRITE_LOCKS[key] = lock
        return lock


class DatabaseWriteLock(AbstractContextManager["DatabaseWriteLock"]):
    """Cross-process exclusive lock for a database read-modify-write transaction."""

    def __init__(self, db_dir: Path, *, timeout_s: float | None = None, poll_s: float = 0.05):
        self.db_dir = Path(db_dir).expanduser().resolve(strict=False)
        self.path = self.db_dir / _DB_WRITE_LOCK_FILE
        self.timeout_s = _db_lock_timeout_s(timeout_s)
        self.poll_s = max(0.01, float(poll_s or 0.05))
        self._thread_lock = _thread_lock_for(self.path)
        self._thread_acquired = False
        self._os_acquired = False
        self._handle: BinaryIO | None = None

    @staticmethod
    def _try_os_lock(handle: BinaryIO) -> bool:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as exc:
            if exc.errno in {EACCES, EAGAIN} or getattr(exc, "winerror", 0) in {33, 36}:
                return False
            raise

    @staticmethod
    def _unlock_os(handle: BinaryIO) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def acquire(self) -> "DatabaseWriteLock":
        deadline = time.monotonic() + self.timeout_s
        remaining = max(0.0, deadline - time.monotonic())
        if not self._thread_lock.acquire(timeout=remaining):
            raise TimeoutError(f"Timed out waiting for database write lock: {self.path}")
        self._thread_acquired = True
        try:
            self.db_dir.mkdir(parents=True, exist_ok=True)
            handle = self.path.open("a+b")
            self._handle = handle
            handle.seek(0, os.SEEK_END)
            if handle.tell() <= 0:
                handle.write(b"\0")
                handle.flush()
            while not self._try_os_lock(handle):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for database write lock: {self.path}")
                time.sleep(min(self.poll_s, max(0.0, deadline - time.monotonic())))
            self._os_acquired = True
            return self
        except Exception:
            self.release()
            raise

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        try:
            if handle is not None:
                try:
                    if self._os_acquired:
                        self._unlock_os(handle)
                finally:
                    self._os_acquired = False
                    handle.close()
        finally:
            if self._thread_acquired:
                self._thread_acquired = False
                self._thread_lock.release()

    def __enter__(self) -> "DatabaseWriteLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


def db_write_lock(db_dir: Path, *, timeout_s: float | None = None) -> DatabaseWriteLock:
    return DatabaseWriteLock(db_dir, timeout_s=timeout_s)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            try:
                target_mode = stat.S_IMODE(path.stat().st_mode)
            except FileNotFoundError:
                target_mode = 0o644
            os.chmod(tmp_path, target_mode)
        deadline = time.monotonic() + 2.0
        while True:
            try:
                os.replace(tmp_path, path)
                break
            except OSError as exc:
                retryable_windows_error = os.name == "nt" and getattr(exc, "winerror", 0) in {5, 32}
                if not retryable_windows_error or time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        if os.name != "nt":
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def compute_file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_doc_id(path: Path) -> str:
    # Stable ID based on absolute path.
    s = str(path.resolve()).encode("utf-8", errors="ignore")
    return hashlib.sha1(s).hexdigest()[:16]


def _docs_index_path(db_dir: Path) -> Path:
    return db_dir / "docs.json"


def _chunks_dir(db_dir: Path) -> Path:
    return db_dir / "chunks"


def doc_chunks_path(db_dir: Path, doc_id: str) -> Path:
    return _chunks_dir(db_dir) / f"{doc_id}.jsonl"


def load_docs_index(db_dir: Path) -> dict:
    p = _docs_index_path(db_dir)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(Path(path), json.dumps(payload, ensure_ascii=False, indent=2))


def save_docs_index(db_dir: Path, docs: dict) -> None:
    p = _docs_index_path(db_dir)
    atomic_write_json(p, docs)


def write_doc_chunks(db_dir: Path, doc_id: str, chunks: list[dict]) -> None:
    d = _chunks_dir(db_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = doc_chunks_path(db_dir, doc_id)
    lines: list[str] = []
    for i, c in enumerate(chunks):
        row = dict(c)
        row["id"] = f"{doc_id}:{i}"
        lines.append(json.dumps(row, ensure_ascii=False))
    payload = "\n".join(lines)
    if lines:
        payload += "\n"
    _atomic_write_text(p, payload)


def delete_doc_chunks(db_dir: Path, doc_id: str) -> bool:
    p = doc_chunks_path(db_dir, doc_id)
    if not p.exists():
        return False
    p.unlink()
    return True


def _doc_index_is_ready(rec: dict | None) -> bool:
    if not isinstance(rec, dict):
        return False
    status = str(rec.get("index_status") or "").strip().lower()
    if status and status not in {"ready", "quality_degraded"}:
        return False
    if status == "quality_degraded":
        gate = rec.get("quality_gate") if isinstance(rec.get("quality_gate"), dict) else {}
        # Degraded documents are safe only after the quality gate explicitly
        # recorded page-level isolation.  Legacy/malformed records without that
        # affirmative contract may still contain corrupt chunks.
        if gate.get("indexable") is not True:
            return False
    if int(rec.get("num_chunks") or 0) <= 0:
        return False
    return True


def load_all_chunks(db_dir: Path, *, include_non_ready: bool = False) -> list[dict]:
    chunks: list[dict] = []
    d = _chunks_dir(db_dir)
    if not d.exists():
        return chunks
    docs_index = load_docs_index(db_dir)
    # The docs index is the commit point. A missing/empty index means no chunk
    # has been published yet, even if a writer has already staged its JSONL.
    enforce_index = not bool(include_non_ready)
    for p in sorted(d.glob("*.jsonl")):
        if enforce_index:
            doc_id = p.stem
            rec = docs_index.get(doc_id)
            if not _doc_index_is_ready(rec):
                continue
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    chunks.append(json.loads(line))
        except FileNotFoundError:
            # A purge publishes the docs-index removal before deleting JSONL.
            # Readers that enumerated the old filename should treat it as gone.
            continue
    return chunks


def prune_missing_docs(db_dir: Path, docs_index: dict) -> int:
    removed = 0
    to_delete: list[str] = []
    for doc_id, rec in docs_index.items():
        path = Path(rec.get("path", ""))
        if not path.exists():
            to_delete.append(doc_id)

    for doc_id in to_delete:
        docs_index.pop(doc_id, None)
        p = doc_chunks_path(db_dir, doc_id)
        if p.exists():
            p.unlink()
        removed += 1
    return removed
