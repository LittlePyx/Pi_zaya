from __future__ import annotations

import importlib.metadata
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from .e5_small_embedding_runtime import E5RuntimeError, E5SmallOnnxExecutor
from .local_semantic_artifacts import (
    CANDIDATE_ID,
    TOTAL_ARTIFACT_BYTES,
    ArtifactPreparationCancelled,
    ArtifactPreparationError,
    ArtifactProgress,
    RequestsArtifactTransport,
    VerifiedLocalSemanticArtifacts,
)
from .local_semantic_preparation_contract import (
    LocalSemanticPreparationStatus,
    preparation_is_active,
)
from .local_semantic_preparation_store import LocalSemanticPreparationStore
from .local_semantic_window_index import (
    LocalSemanticIndexBuildProgress,
    LocalSemanticIndexError,
    LocalSemanticWindowIndexStore,
)


LOCAL_SEMANTIC_PRODUCT_ENABLED = False
_DEPENDENCY_VERSIONS = {
    "numpy": "2.2.6",
    "onnxruntime": "1.23.2",
    "tokenizers": "0.21.4",
}


class _Artifacts(Protocol):
    root: Path

    def quick_ready(self) -> bool:
        ...

    def is_ready(self) -> bool:
        ...

    def prepare(self, transport, *, should_cancel, on_progress):
        ...


class _IndexStore(Protocol):
    def status(self):
        ...

    def build(self, executor, *, should_cancel, on_progress):
        ...


class LocalSemanticPreparationServiceError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(str(code or "preparation_service_failed"))
        self.code = str(code or "preparation_service_failed")


def local_semantic_model_root(db_dir: Path | str) -> Path:
    root = Path(db_dir).expanduser().resolve(strict=False)
    return root.parent / "_local_models" / CANDIDATE_ID


def local_semantic_runtime_dependencies_available() -> bool:
    if (sys.version_info.major, sys.version_info.minor, sys.version_info.micro) != (3, 10, 11):
        return False
    try:
        if any(
            importlib.metadata.version(distribution) != expected
            for distribution, expected in _DEPENDENCY_VERSIONS.items()
        ):
            return False
        import onnxruntime as ort

        return "CPUExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _background_runner(callback: Callable[[], None]) -> None:
    threading.Thread(
        target=callback,
        name="pi-zaya-local-semantic-preparation",
        daemon=True,
    ).start()


class LocalSemanticPreparationService:
    """Own the complete local semantic preparation and repair lifecycle."""

    def __init__(
        self,
        db_dir: Path | str,
        *,
        dependency_probe: Callable[[], bool] = local_semantic_runtime_dependencies_available,
        store: LocalSemanticPreparationStore | None = None,
        artifacts: _Artifacts | None = None,
        index_store: _IndexStore | None = None,
        transport_factory: Callable[[], object] = RequestsArtifactTransport,
        executor_factory: Callable[[Path, Path], object] | None = None,
        runner: Callable[[Callable[[], None]], None] = _background_runner,
        owner_session: str | None = None,
    ) -> None:
        self.db_dir = Path(db_dir).expanduser().resolve(strict=False)
        self.dependency_probe = dependency_probe
        self.store = store or LocalSemanticPreparationStore(self.db_dir)
        self.artifacts = artifacts or VerifiedLocalSemanticArtifacts(
            local_semantic_model_root(self.db_dir)
        )
        self.index_store = index_store or LocalSemanticWindowIndexStore(self.db_dir)
        self.transport_factory = transport_factory
        self.executor_factory = executor_factory or self._executor
        self.runner = runner
        self.owner_session = str(owner_session or uuid.uuid4().hex)
        self.store.reconcile_after_restart(owner_session=self.owner_session)

    def _executor(self, model_path: Path, tokenizer_path: Path) -> E5SmallOnnxExecutor:
        return E5SmallOnnxExecutor(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
        )

    @staticmethod
    def _synthesized(
        state: str,
        reason_id: str,
        *,
        previous: LocalSemanticPreparationStatus,
        document_count: int = 0,
        chunk_count: int = 0,
    ) -> LocalSemanticPreparationStatus:
        return LocalSemanticPreparationStatus(
            state=state,
            reason_id=reason_id,
            task_id=previous.task_id,
            document_count=document_count,
            chunk_count=chunk_count,
        )

    def status(self) -> LocalSemanticPreparationStatus:
        current = self.store.current()
        if preparation_is_active(current.state):
            return current
        if not self.dependency_probe():
            return self._synthesized(
                "unavailable_in_build",
                "runtime_dependencies_missing",
                previous=current,
            )
        if current.state in {"interrupted", "repairable_error"}:
            return current
        if not self.artifacts.quick_ready():
            return self._synthesized(
                "not_prepared",
                "artifacts_not_prepared",
                previous=current,
            )
        index = self.index_store.status()
        if index.state == "ready":
            return self._synthesized(
                "ready",
                "ready",
                previous=current,
                document_count=int(index.document_count),
                chunk_count=int(index.parent_count),
            )
        if index.state == "missing":
            return self._synthesized(
                "ready_to_index",
                "index_not_prepared",
                previous=current,
            )
        if index.state == "stale":
            return self._synthesized(
                "stale",
                "library_changed",
                previous=current,
            )
        return self._synthesized(
            "repairable_error",
            "index_repair_required",
            previous=current,
        )

    def start(self, operation: str) -> tuple[LocalSemanticPreparationStatus, bool]:
        current = self.store.current()
        if preparation_is_active(current.state):
            return current, False
        status = self.status()
        clean_operation = str(operation or "").strip().lower()
        if status.action_id != clean_operation:
            raise LocalSemanticPreparationServiceError("action_not_allowed")
        initial_state = "indexing" if self.artifacts.quick_ready() else "downloading"
        started, created = self.store.start_or_get(
            operation=clean_operation,
            owner_session=self.owner_session,
            initial_state=initial_state,
            total_units=0 if initial_state == "indexing" else 6,
            total_bytes=TOTAL_ARTIFACT_BYTES,
        )
        if created:
            self.runner(lambda: self._run(started.task_id))
        return self.status(), created

    def cancel(self, task_id: str) -> LocalSemanticPreparationStatus:
        if not self.store.request_cancel(str(task_id or "")):
            raise LocalSemanticPreparationServiceError("task_not_cancellable")
        return self.store.current()

    def _run(self, task_id: str) -> None:
        try:
            if not self.artifacts.is_ready():
                self.artifacts.prepare(
                    self.transport_factory(),
                    should_cancel=lambda: self.store.should_cancel(task_id),
                    on_progress=lambda item: self._artifact_progress(task_id, item),
                )
            if self.store.should_cancel(task_id):
                raise ArtifactPreparationCancelled()
            index_started = time.monotonic()
            self.store.update_progress(
                task_id,
                state="indexing",
                reason_id="index_building",
                phase="index",
                completed_units=0,
                total_units=0,
                completed_bytes=TOTAL_ARTIFACT_BYTES,
                total_bytes=TOTAL_ARTIFACT_BYTES,
                eta_seconds=None,
            )
            executor = self.executor_factory(
                self.artifacts.root / "model.onnx",
                self.artifacts.root / "tokenizer" / "tokenizer.json",
            )
            result = self.index_store.build(
                executor,
                should_cancel=lambda: self.store.should_cancel(task_id),
                on_progress=lambda item: self._index_progress(
                    task_id,
                    item,
                    started=index_started,
                ),
            )
            if self.store.should_cancel(task_id):
                raise ArtifactPreparationCancelled()
            self.store.finish(
                task_id,
                state="ready",
                reason_id="ready",
                document_count=int(result.document_count),
                chunk_count=int(result.parent_count),
            )
        except (ArtifactPreparationCancelled,):
            self.store.finish(
                task_id,
                state="interrupted",
                reason_id="cancelled_by_user",
            )
        except (ArtifactPreparationError, LocalSemanticIndexError, E5RuntimeError) as exc:
            if self.store.should_cancel(task_id) or getattr(exc, "code", "") == "cancelled_by_user":
                state, reason = "interrupted", "cancelled_by_user"
            else:
                state, reason = "repairable_error", str(getattr(exc, "code", "") or "preparation_failed")
            self.store.finish(task_id, state=state, reason_id=reason)
        except Exception:
            state = "interrupted" if self.store.should_cancel(task_id) else "repairable_error"
            reason = "cancelled_by_user" if state == "interrupted" else "preparation_internal_failure"
            self.store.finish(task_id, state=state, reason_id=reason)

    def _artifact_progress(self, task_id: str, item: ArtifactProgress) -> None:
        self.store.update_progress(
            task_id,
            state="downloading",
            reason_id="artifact_downloading",
            phase=item.phase,
            completed_units=item.completed_units,
            total_units=item.total_units,
            completed_bytes=item.completed_bytes,
            total_bytes=item.total_bytes,
            eta_seconds=item.eta_seconds,
        )

    def _index_progress(
        self,
        task_id: str,
        item: LocalSemanticIndexBuildProgress,
        *,
        started: float,
    ) -> None:
        elapsed = max(0.0, time.monotonic() - started)
        eta = None
        if item.completed_windows > 0 and elapsed > 0:
            eta = max(
                0,
                round(
                    (item.total_windows - item.completed_windows)
                    * elapsed
                    / item.completed_windows
                ),
            )
        self.store.update_progress(
            task_id,
            state="indexing",
            reason_id="index_building",
            phase="index",
            completed_units=item.completed_windows,
            total_units=item.total_windows,
            completed_bytes=TOTAL_ARTIFACT_BYTES,
            total_bytes=TOTAL_ARTIFACT_BYTES,
            eta_seconds=eta,
        )


_SERVICES_LOCK = threading.Lock()
_SERVICES: dict[str, LocalSemanticPreparationService] = {}


def get_local_semantic_preparation_service(
    db_dir: Path | str,
) -> LocalSemanticPreparationService:
    root = Path(db_dir).expanduser().resolve(strict=False)
    key = str(root).casefold()
    with _SERVICES_LOCK:
        service = _SERVICES.get(key)
        if service is None:
            service = LocalSemanticPreparationService(root)
            _SERVICES[key] = service
        return service
