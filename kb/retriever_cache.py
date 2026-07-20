from __future__ import annotations

import threading
from pathlib import Path

from kb.retriever import BM25Retriever
from kb.store import load_all_chunks


_CACHE_LOCK = threading.Condition(threading.RLock())
_CACHE: dict[str, tuple[tuple[int, int], BM25Retriever, int]] = {}
_LOADING: set[tuple[str, tuple[int, int]]] = set()
_PREWARM_STARTED: set[str] = set()
_MAX_CACHE_ROOTS = 3


def _cache_key(db_dir: Path) -> str:
    return str(Path(db_dir).expanduser().resolve()).casefold()


def _docs_signature(db_dir: Path) -> tuple[int, int]:
    path = Path(db_dir) / "docs.json"
    try:
        stat = path.stat()
        return int(stat.st_mtime_ns), int(stat.st_size)
    except FileNotFoundError:
        return 0, 0


def get_cached_retriever(db_dir: Path) -> tuple[BM25Retriever, int, bool]:
    """Return a retriever, rebuilding only after the committed docs index changes."""

    root = Path(db_dir).expanduser().resolve()
    key = _cache_key(root)
    signature = _docs_signature(root)
    loading_key = (key, signature)
    with _CACHE_LOCK:
        while loading_key in _LOADING:
            _CACHE_LOCK.wait()
        cached = _CACHE.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2], True
        _LOADING.add(loading_key)

    try:
        chunks = load_all_chunks(root)
        retriever = BM25Retriever(chunks)
    except Exception:
        with _CACHE_LOCK:
            _LOADING.discard(loading_key)
            _CACHE_LOCK.notify_all()
        raise

    with _CACHE_LOCK:
        _CACHE[key] = (signature, retriever, len(chunks))
        while len(_CACHE) > _MAX_CACHE_ROOTS:
            oldest_key = next(iter(_CACHE))
            if oldest_key == key and len(_CACHE) > 1:
                oldest_key = next(k for k in _CACHE if k != key)
            _CACHE.pop(oldest_key, None)
        _LOADING.discard(loading_key)
        _CACHE_LOCK.notify_all()
    return retriever, len(chunks), False


def warm_retriever_async(db_dir: Path) -> threading.Thread | None:
    root = Path(db_dir).expanduser().resolve()
    key = _cache_key(root)
    with _CACHE_LOCK:
        if key in _PREWARM_STARTED:
            return None
        _PREWARM_STARTED.add(key)

    def run() -> None:
        try:
            get_cached_retriever(root)
        except Exception:
            # Startup prewarm is opportunistic; foreground generation retains
            # the normal error path and can retry later.
            pass

    worker = threading.Thread(target=run, name="kb-retriever-prewarm", daemon=True)
    worker.start()
    return worker


def clear_retriever_cache() -> None:
    """Clear process-local state (primarily for tests)."""

    with _CACHE_LOCK:
        _CACHE.clear()
        _LOADING.clear()
        _PREWARM_STARTED.clear()
        _CACHE_LOCK.notify_all()
