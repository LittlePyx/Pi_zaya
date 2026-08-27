from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable

logger = logging.getLogger(__name__)

ResearchNoteRecoveryHandler = Callable[..., object]

_LOCK = threading.Condition(threading.RLock())
_PENDING_PATHS: OrderedDict[str, str] = OrderedDict()
_PENDING_REASONS: set[str] = set()
_SCAN_ALL = False
_RUNNING = False
_HANDLER: ResearchNoteRecoveryHandler | None = None
_HANDLER_USERS = 0
_WORKER: threading.Thread | None = None
_PATH_BATCH_LIMIT = 32


def _path_identity(value: object) -> str:
    return str(value or "").replace("\x00", "").strip().replace("\\", "/").casefold()[:1_200]


def register_research_note_recovery_handler(handler: ResearchNoteRecoveryHandler) -> None:
    """Register the process-local note maintenance handler and start its worker."""
    if not callable(handler):
        raise TypeError("research note recovery handler must be callable")
    global _HANDLER, _HANDLER_USERS, _WORKER
    with _LOCK:
        if _HANDLER is handler:
            _HANDLER_USERS += 1
        else:
            _HANDLER = handler
            _HANDLER_USERS = 1
        if _WORKER is None or not _WORKER.is_alive():
            _WORKER = threading.Thread(
                target=_recovery_worker_loop,
                name="research-note-evidence-recovery",
                daemon=True,
            )
            _WORKER.start()
        _LOCK.notify_all()


def unregister_research_note_recovery_handler(
    handler: ResearchNoteRecoveryHandler | None = None,
) -> None:
    """Detach a handler and discard work that cannot safely outlive its app."""
    global _HANDLER, _HANDLER_USERS, _SCAN_ALL
    with _LOCK:
        if handler is not None and _HANDLER is not handler:
            return
        if handler is not None and _HANDLER_USERS > 1:
            _HANDLER_USERS -= 1
            return
        _HANDLER = None
        _HANDLER_USERS = 0
        _PENDING_PATHS.clear()
        _PENDING_REASONS.clear()
        _SCAN_ALL = False
        _LOCK.notify_all()


def enqueue_research_note_source_recovery(
    source_paths: Iterable[object],
    *,
    reason: str,
) -> bool:
    """Queue a deduplicated, best-effort revalidation for recovered sources."""
    global _PENDING_PATHS
    normalized_reason = str(reason or "source_recovered").replace("\x00", "").strip()[:80]
    clean_paths: list[tuple[str, str]] = []
    for raw_path in source_paths:
        path = str(raw_path or "").replace("\x00", "").strip()[:1_200]
        identity = _path_identity(path)
        if identity:
            clean_paths.append((identity, path))
    if not clean_paths:
        return False
    with _LOCK:
        if _HANDLER is None:
            return False
        for identity, path in clean_paths:
            _PENDING_PATHS.setdefault(identity, path)
        _PENDING_REASONS.add(normalized_reason or "source_recovered")
        _LOCK.notify_all()
    return True


def enqueue_research_note_source_recovery_scan(*, reason: str) -> bool:
    """Queue a bounded scan after a successful whole-library recovery event."""
    global _SCAN_ALL
    normalized_reason = str(reason or "library_reindexed").replace("\x00", "").strip()[:80]
    with _LOCK:
        if _HANDLER is None:
            return False
        _SCAN_ALL = True
        _PENDING_PATHS.clear()
        _PENDING_REASONS.add(normalized_reason or "library_reindexed")
        _LOCK.notify_all()
    return True


def wait_for_research_note_recovery_idle(timeout: float = 5.0) -> bool:
    """Wait for queued recovery work; intended for deterministic integration tests."""
    import time

    deadline = time.monotonic() + max(0.0, float(timeout))
    with _LOCK:
        while _RUNNING or _SCAN_ALL or bool(_PENDING_PATHS):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _LOCK.wait(timeout=remaining)
        return True


def _recovery_worker_loop() -> None:
    global _RUNNING, _SCAN_ALL
    while True:
        with _LOCK:
            while _HANDLER is None or (not _SCAN_ALL and not _PENDING_PATHS):
                _LOCK.wait()
            handler = _HANDLER
            scan_all = _SCAN_ALL
            if scan_all:
                source_paths: tuple[str, ...] | None = None
                _SCAN_ALL = False
                _PENDING_PATHS.clear()
            else:
                rows = list(_PENDING_PATHS.items())[:_PATH_BATCH_LIMIT]
                for identity, _path in rows:
                    _PENDING_PATHS.pop(identity, None)
                source_paths = tuple(path for _identity, path in rows)
            reason = ",".join(sorted(_PENDING_REASONS))[:240] or "source_recovered"
            _PENDING_REASONS.clear()
            _RUNNING = True
        try:
            if handler is not None:
                handler(source_paths=source_paths, reason=reason)
        except Exception:
            logger.exception(
                "research_note_source_recovery_failed",
                extra={"reason": reason, "scan_all": scan_all},
            )
        finally:
            with _LOCK:
                _RUNNING = False
                _LOCK.notify_all()
