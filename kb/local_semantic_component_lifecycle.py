from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from .local_semantic_component import (
    InstalledLocalSemanticComponent,
    LocalSemanticComponentCancelled,
    LocalSemanticComponentError,
    LocalSemanticComponentManifest,
    LocalSemanticComponentStore,
    local_semantic_component_root,
)
from .local_semantic_component_download import (
    ComponentDownloadProgress,
    LocalSemanticComponentDownloader,
)
from .local_semantic_component_lifecycle_contract import (
    LocalSemanticComponentLifecycleRecord,
    component_lifecycle_is_active,
)
from .local_semantic_component_lifecycle_store import (
    LocalSemanticComponentLifecycleStore,
)
from .local_semantic_component_observability import (
    record_local_semantic_component_observation,
)


LOCAL_SEMANTIC_COMPONENT_PRODUCT_ENABLED = True
_INTEGRITY_CODES = {
    "component_archive_hash_mismatch",
    "component_archive_invalid",
    "component_archive_entry_unsafe",
    "component_archive_duplicate_path",
    "component_payload_file_count_mismatch",
    "component_payload_size_mismatch",
    "component_payload_hash_mismatch",
}
_IMMEDIATE_QUARANTINE_CODES = {
    "component_archive_entry_unsafe",
    "component_archive_duplicate_path",
}
_NON_RETRYABLE_CODES = {"component_storage_path_too_long"}
_RETRY_DELAYS_SECONDS = (2.0, 10.0)


class _Downloader(Protocol):
    def download(self, manifest, transport, *, should_cancel, on_progress):
        ...


class _ComponentStore(Protocol):
    root: Path

    def preflight_download(self, manifest, *, current_generation_bytes=0):
        ...

    def install_from_archive(
        self,
        archive_path,
        manifest,
        *,
        should_cancel,
        health_check,
    ):
        ...

    def load_current(self):
        ...

    def remove_all(self):
        ...

    def retry_pending_cleanup(self) -> bool:
        ...


class LocalSemanticComponentLifecycleError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(str(code or "semantic_component_lifecycle_failed"))
        self.code = str(code or "semantic_component_lifecycle_failed")


def _background_runner(callback: Callable[[], None]) -> None:
    threading.Thread(
        target=callback,
        name="pi-zaya-local-semantic-component",
        daemon=True,
    ).start()


def _manifest_identity(manifest: LocalSemanticComponentManifest) -> str:
    return f"{manifest.release_id}:{manifest.archive_sha256}"


class LocalSemanticComponentLifecycleService:
    """Run optional-component work unattended while every query retains BM25."""

    def __init__(
        self,
        db_dir: Path | str,
        manifest: LocalSemanticComponentManifest,
        *,
        transport_factory: Callable[[], object],
        component_store: _ComponentStore | None = None,
        downloader: _Downloader | None = None,
        lifecycle_store: LocalSemanticComponentLifecycleStore | None = None,
        health_check: Callable[[Path], bool] = lambda _root: True,
        index_ready: Callable[[Path], bool] = lambda _root: False,
        prepare_index: Callable[..., object] = lambda *_args, **_kwargs: None,
        remove_index: Callable[[], None] = lambda: None,
        release_runtime_handles: Callable[[], None] = lambda: None,
        runner: Callable[[Callable[[], None]], None] = _background_runner,
        retry_wait: Callable[[float], None] = time.sleep,
        owner_session: str | None = None,
        retry_limit: int = 2,
    ) -> None:
        self.db_dir = Path(db_dir).expanduser().resolve(strict=False)
        self.manifest = manifest
        self.manifest_identity = _manifest_identity(manifest)
        component_root = local_semantic_component_root(self.db_dir)
        self.component_store = component_store or LocalSemanticComponentStore(
            component_root
        )
        self.downloader = downloader or LocalSemanticComponentDownloader(
            self.component_store.root / "downloads"
        )
        self.lifecycle_store = lifecycle_store or LocalSemanticComponentLifecycleStore(
            self.db_dir
        )
        self.transport_factory = transport_factory
        self.health_check = health_check
        self.index_ready = index_ready
        self.prepare_index = prepare_index
        self.remove_index = remove_index
        self.release_runtime_handles = release_runtime_handles
        self.runner = runner
        self.retry_wait = retry_wait
        self.owner_session = str(owner_session or uuid.uuid4().hex)
        self.retry_limit = max(1, min(2, int(retry_limit)))
        self.component_store.retry_pending_cleanup()
        reconciled, should_resume = self.lifecycle_store.reconcile_after_restart(
            owner_session=self.owner_session
        )
        if should_resume:
            record_local_semantic_component_observation(
                reconciled,
                phase="started",
            )
            self.runner(lambda: self._run_existing(reconciled.task_id))
        elif (
            reconciled.state == "degraded_bm25"
            and reconciled.consented
            and not self.lifecycle_store.quarantine_reason(self.manifest_identity)
        ):
            self._start_prepare(operation="recover", consented=True)

    def status(self) -> LocalSemanticComponentLifecycleRecord:
        record = self.lifecycle_store.current()
        if record.state == "ready" and not self._all_ready():
            return self._start_prepare(operation="recover", consented=True)[0]
        return record

    def enable(
        self,
        *,
        consent: bool,
    ) -> tuple[LocalSemanticComponentLifecycleRecord, bool]:
        if not consent:
            raise LocalSemanticComponentLifecycleError(
                "semantic_component_consent_required"
            )
        current = self.lifecycle_store.current()
        if current.state == "ready" and self._all_ready():
            return current, False
        if component_lifecycle_is_active(current.state):
            return current, False
        return self._start_prepare(operation="enable", consented=True)

    def resume(self) -> tuple[LocalSemanticComponentLifecycleRecord, bool]:
        current = self.lifecycle_store.current()
        if current.state not in {"paused_by_user", "needs_space", "degraded_bm25"}:
            raise LocalSemanticComponentLifecycleError(
                "semantic_component_resume_not_allowed"
            )
        if self.lifecycle_store.quarantine_reason(self.manifest_identity):
            return self._finish_quarantined(operation="resume")
        return self._start_prepare(operation="resume", consented=current.consented)

    def cancel(self, task_id: str) -> LocalSemanticComponentLifecycleRecord:
        if not self.lifecycle_store.request_cancel(str(task_id or "")):
            raise LocalSemanticComponentLifecycleError(
                "semantic_component_task_not_cancellable"
            )
        return self.lifecycle_store.current()

    def remove(self) -> tuple[LocalSemanticComponentLifecycleRecord, bool]:
        current = self.lifecycle_store.current()
        if component_lifecycle_is_active(current.state):
            if current.operation == "remove":
                return current, False
            if not self.lifecycle_store.request_remove(current.task_id):
                raise LocalSemanticComponentLifecycleError(
                    "semantic_component_remove_not_allowed"
                )
            requested = self.lifecycle_store.current()
            record_local_semantic_component_observation(
                requested,
                phase="started",
            )
            return requested, False
        record, created = self.lifecycle_store.start_or_get(
            operation="remove",
            initial_state="removing",
            manifest_identity=self.manifest_identity,
            owner_session=self.owner_session,
            consented=False,
        )
        if created:
            record_local_semantic_component_observation(record, phase="started")
            self.runner(lambda: self._run_remove(record.task_id))
        return self.lifecycle_store.current(), created

    def _start_prepare(
        self,
        *,
        operation: str,
        consented: bool,
    ) -> tuple[LocalSemanticComponentLifecycleRecord, bool]:
        if self.lifecycle_store.quarantine_reason(self.manifest_identity):
            return self._finish_quarantined(operation=operation)
        record, created = self.lifecycle_store.start_or_get(
            operation=operation,
            initial_state="checking_capacity",
            manifest_identity=self.manifest_identity,
            owner_session=self.owner_session,
            consented=consented,
            total_bytes=self.manifest.archive_bytes,
        )
        if created:
            record_local_semantic_component_observation(record, phase="started")
            self.runner(lambda: self._run_prepare(record.task_id))
        return self.lifecycle_store.current(), created

    def _finish_quarantined(
        self,
        *,
        operation: str,
    ) -> tuple[LocalSemanticComponentLifecycleRecord, bool]:
        record, created = self.lifecycle_store.start_or_get(
            operation=operation,
            initial_state="checking_capacity",
            manifest_identity=self.manifest_identity,
            owner_session=self.owner_session,
            consented=True,
        )
        if created:
            record_local_semantic_component_observation(record, phase="started")
            self.lifecycle_store.transition(
                record.task_id,
                state="degraded_bm25",
                reason_id="component_manifest_quarantined",
                finished=True,
            )
            record_local_semantic_component_observation(
                self.lifecycle_store.current(),
                phase="terminal",
            )
        return self.lifecycle_store.current(), created

    def _run_existing(self, task_id: str) -> None:
        record = self.lifecycle_store.current()
        if record.task_id != task_id:
            return
        if record.operation == "remove":
            self._run_remove(task_id)
        else:
            self._run_prepare(task_id)

    def _run_prepare(self, task_id: str) -> None:
        started_at = time.perf_counter()
        current = self.lifecycle_store.current()
        first_attempt = max(1, current.attempt or 1)
        for attempt in range(first_attempt, self.retry_limit + 1):
            try:
                self._prepare_once(task_id, attempt=attempt)
                self._raise_if_cancelled(task_id)
                self._observe_prepare_terminal(task_id, started_at=started_at)
                return
            except LocalSemanticComponentCancelled:
                self._finish_cancel_or_remove(task_id)
                self._observe_prepare_terminal(task_id, started_at=started_at)
                return
            except LocalSemanticComponentError as exc:
                if self.lifecycle_store.should_cancel(task_id):
                    self._finish_cancel_or_remove(task_id)
                    self._observe_prepare_terminal(task_id, started_at=started_at)
                    return
                if exc.code == "semantic_component_needs_space":
                    self.lifecycle_store.transition(
                        task_id,
                        state="needs_space",
                        reason_id=exc.code,
                        required_additional_bytes=exc.required_additional_bytes,
                        eta_seconds=None,
                        finished=True,
                    )
                    self._observe_prepare_terminal(task_id, started_at=started_at)
                    return
                if exc.code in _INTEGRITY_CODES:
                    self._discard_downloaded_archive()
                if exc.code in _IMMEDIATE_QUARANTINE_CODES:
                    self._quarantine_and_degrade(task_id, reason_id=exc.code)
                    self._observe_prepare_terminal(task_id, started_at=started_at)
                    return
                if exc.code in _NON_RETRYABLE_CODES:
                    self._degrade(task_id, reason_id=exc.code)
                    self._observe_prepare_terminal(task_id, started_at=started_at)
                    return
                if attempt < self.retry_limit:
                    self._prepare_retry(task_id, attempt=attempt, reason_id=exc.code)
                    continue
                if exc.code in _INTEGRITY_CODES:
                    self.lifecycle_store.quarantine(
                        self.manifest_identity,
                        reason_id=exc.code,
                    )
                self._degrade(task_id, reason_id=exc.code)
                self._observe_prepare_terminal(task_id, started_at=started_at)
                return
            except Exception:
                if self.lifecycle_store.should_cancel(task_id):
                    self._finish_cancel_or_remove(task_id)
                    self._observe_prepare_terminal(task_id, started_at=started_at)
                    return
                if attempt < self.retry_limit:
                    self._prepare_retry(
                        task_id,
                        attempt=attempt,
                        reason_id="component_internal_failure",
                    )
                    continue
                self._degrade(task_id, reason_id="component_internal_failure")
                self._observe_prepare_terminal(task_id, started_at=started_at)
                return

    def _prepare_once(self, task_id: str, *, attempt: int) -> None:
        self._raise_if_cancelled(task_id)
        installed = self._matching_healthy_component()
        if installed is None:
            self.lifecycle_store.transition(
                task_id,
                state="checking_capacity",
                reason_id="component_capacity_check",
                attempt=attempt,
                completed_bytes=0,
                total_bytes=self.manifest.archive_bytes,
                eta_seconds=None,
                required_additional_bytes=0,
            )
            self.component_store.preflight_download(
                self.manifest,
                current_generation_bytes=self._current_generation_bytes(),
            )
            self._raise_if_cancelled(task_id)
            self.lifecycle_store.transition(
                task_id,
                state="downloading",
                reason_id="component_downloading",
                attempt=attempt,
            )
            download = self.downloader.download(
                self.manifest,
                self.transport_factory(),
                should_cancel=lambda: self.lifecycle_store.should_cancel(task_id),
                on_progress=lambda item: self._download_progress(task_id, item),
            )
            self._raise_if_cancelled(task_id)
            self.lifecycle_store.transition(
                task_id,
                state="verifying_archive",
                reason_id="component_verifying",
                completed_bytes=self.manifest.archive_bytes,
                total_bytes=self.manifest.archive_bytes,
                eta_seconds=None,
            )
            self.lifecycle_store.transition(
                task_id,
                state="installing",
                reason_id="component_installing",
            )

            def health(root: Path) -> bool:
                self.lifecycle_store.transition(
                    task_id,
                    state="health_checking",
                    reason_id="component_health_checking",
                )
                return self.health_check(root)

            installed = self.component_store.install_from_archive(
                download.archive_path,
                self.manifest,
                should_cancel=lambda: self.lifecycle_store.should_cancel(task_id),
                health_check=health,
            )
        self._raise_if_cancelled(task_id)
        self.lifecycle_store.transition(
            task_id,
            state="ready_to_index",
            reason_id="semantic_index_not_ready",
            completed_bytes=0,
            total_bytes=0,
            eta_seconds=None,
        )
        if not self.index_ready(installed.root):
            self.lifecycle_store.transition(
                task_id,
                state="indexing",
                reason_id="semantic_index_building",
                completed_bytes=0,
                total_bytes=0,
                eta_seconds=None,
            )
            self.prepare_index(
                installed.root,
                should_cancel=lambda: self.lifecycle_store.should_cancel(task_id),
                on_progress=lambda completed, total: self._index_progress(
                    task_id, completed=completed, total=total
                ),
            )
        self._raise_if_cancelled(task_id)
        if not self.index_ready(installed.root):
            raise LocalSemanticComponentError("semantic_index_not_ready")
        self.lifecycle_store.transition(
            task_id,
            state="ready",
            reason_id="component_and_index_ready",
            completed_bytes=self.manifest.archive_bytes,
            total_bytes=self.manifest.archive_bytes,
            eta_seconds=None,
            required_additional_bytes=0,
            consented=True,
            finished=True,
        )
        self._discard_downloaded_archive()

    def _prepare_retry(self, task_id: str, *, attempt: int, reason_id: str) -> None:
        self.lifecycle_store.transition(
            task_id,
            state="recovering",
            reason_id=reason_id,
            attempt=attempt,
            eta_seconds=None,
        )
        self._raise_if_cancelled(task_id)
        self.retry_wait(_RETRY_DELAYS_SECONDS[min(attempt - 1, 1)])
        self._raise_if_cancelled(task_id)

    def _finish_cancel_or_remove(self, task_id: str) -> None:
        current = self.lifecycle_store.current()
        if current.task_id == task_id and current.operation == "remove":
            self._run_remove(task_id)
            return
        self.lifecycle_store.transition(
            task_id,
            state="paused_by_user",
            reason_id="cancelled_by_user",
            eta_seconds=None,
            finished=True,
        )

    def _run_remove(self, task_id: str) -> None:
        started_at = time.perf_counter()
        for attempt in range(1, self.retry_limit + 1):
            try:
                self.lifecycle_store.transition(
                    task_id,
                    state="removing",
                    reason_id="component_removing",
                    attempt=attempt,
                    eta_seconds=None,
                )
                self.release_runtime_handles()
                result = self.component_store.remove_all()
                self.remove_index()
                self.lifecycle_store.transition(
                    task_id,
                    state="not_installed",
                    reason_id=(
                        "owned_cleanup_scheduled"
                        if result.cleanup_pending
                        else "component_removed"
                    ),
                    completed_bytes=0,
                    total_bytes=0,
                    eta_seconds=None,
                    required_additional_bytes=0,
                    consented=False,
                    finished=True,
                )
                record_local_semantic_component_observation(
                    self.lifecycle_store.current(),
                    phase="terminal",
                    started_at=started_at,
                )
                return
            except Exception:
                if attempt < self.retry_limit:
                    self.lifecycle_store.transition(
                        task_id,
                        state="recovering",
                        reason_id="component_removal_retrying",
                        attempt=attempt,
                        eta_seconds=None,
                    )
                    self.retry_wait(_RETRY_DELAYS_SECONDS[min(attempt - 1, 1)])
                    continue
                self._degrade(task_id, reason_id="component_removal_incomplete")
                record_local_semantic_component_observation(
                    self.lifecycle_store.current(),
                    phase="terminal",
                    started_at=started_at,
                )
                return

    def _observe_prepare_terminal(self, task_id: str, *, started_at: float) -> None:
        current = self.lifecycle_store.current()
        if current.task_id != task_id or current.operation == "remove":
            return
        record_local_semantic_component_observation(
            current,
            phase="terminal",
            started_at=started_at,
        )

    def _raise_if_cancelled(self, task_id: str) -> None:
        if self.lifecycle_store.should_cancel(task_id):
            raise LocalSemanticComponentCancelled()

    def _download_progress(
        self,
        task_id: str,
        item: ComponentDownloadProgress,
    ) -> None:
        self.lifecycle_store.transition(
            task_id,
            state="downloading",
            reason_id="component_downloading",
            completed_bytes=item.completed_bytes,
            total_bytes=item.total_bytes,
            eta_seconds=item.eta_seconds,
        )

    def _index_progress(self, task_id: str, *, completed: int, total: int) -> None:
        self.lifecycle_store.transition(
            task_id,
            state="indexing",
            reason_id="semantic_index_building",
            completed_bytes=max(0, int(completed)),
            total_bytes=max(0, int(total)),
            eta_seconds=None,
        )

    def _matching_healthy_component(
        self,
    ) -> InstalledLocalSemanticComponent | None:
        try:
            installed = self.component_store.load_current()
            if (
                installed.archive_sha256 != self.manifest.archive_sha256
                or installed.payload_sha256 != self.manifest.payload_sha256
                or not self.health_check(installed.root)
            ):
                return None
            return installed
        except Exception:
            return None

    def _all_ready(self) -> bool:
        installed = self._matching_healthy_component()
        if installed is None:
            return False
        try:
            return bool(self.index_ready(installed.root))
        except Exception:
            return False

    def _current_generation_bytes(self) -> int:
        try:
            root = self.component_store.load_current().root
        except Exception:
            return 0
        total = 0
        try:
            for path in root.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
        except OSError:
            return 0
        return total

    def _discard_downloaded_archive(self) -> None:
        root = self.component_store.root / "downloads"
        (root / f"{self.manifest.archive_sha256}.zip").unlink(missing_ok=True)

    def _quarantine_and_degrade(self, task_id: str, *, reason_id: str) -> None:
        self.lifecycle_store.quarantine(
            self.manifest_identity,
            reason_id=reason_id,
        )
        self._degrade(task_id, reason_id=reason_id)

    def _degrade(self, task_id: str, *, reason_id: str) -> None:
        self.lifecycle_store.transition(
            task_id,
            state="degraded_bm25",
            reason_id=reason_id,
            eta_seconds=None,
            finished=True,
        )
