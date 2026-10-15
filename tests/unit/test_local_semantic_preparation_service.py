from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kb.local_semantic_artifacts import (
    ArtifactPreparationError,
    ArtifactProgress,
)
from kb.local_semantic_preparation_service import (
    LocalSemanticPreparationService,
    LocalSemanticPreparationServiceError,
)
from kb.local_semantic_preparation_store import LocalSemanticPreparationStore
from kb.local_semantic_window_index import LocalSemanticIndexBuildProgress


class _Artifacts:
    def __init__(self, root: Path, *, ready: bool = False) -> None:
        self.root = root
        self.ready = ready
        self.error = ""

    def quick_ready(self) -> bool:
        return self.ready

    def is_ready(self) -> bool:
        return self.ready

    def prepare(self, _transport, *, should_cancel, on_progress):
        if should_cancel():
            from kb.local_semantic_artifacts import ArtifactPreparationCancelled

            raise ArtifactPreparationCancelled()
        if self.error:
            raise ArtifactPreparationError(self.error)
        on_progress(ArtifactProgress("artifacts", 1, 1, 10, 10, 0))
        self.ready = True


class _Index:
    def __init__(self, *, state: str = "missing") -> None:
        self.state = state
        self.build_calls = 0

    def status(self):
        return SimpleNamespace(
            state=self.state,
            document_count=3,
            parent_count=7,
        )

    def build(self, _executor, *, should_cancel, on_progress):
        self.build_calls += 1
        if should_cancel():
            from kb.local_semantic_window_index import LocalSemanticIndexError

            raise LocalSemanticIndexError("cancelled_by_user")
        on_progress(LocalSemanticIndexBuildProgress(7, 7, 0, 7))
        self.state = "ready"
        return SimpleNamespace(document_count=3, parent_count=7)


def _service(
    tmp_path: Path,
    *,
    dependencies: bool = True,
    artifacts_ready: bool = False,
    index_state: str = "missing",
    runner=lambda callback: callback(),
):
    artifacts = _Artifacts(tmp_path / "models", ready=artifacts_ready)
    index = _Index(state=index_state)
    service = LocalSemanticPreparationService(
        tmp_path / "db",
        dependency_probe=lambda: dependencies,
        store=LocalSemanticPreparationStore(tmp_path / "db"),
        artifacts=artifacts,
        index_store=index,
        transport_factory=lambda: object(),
        executor_factory=lambda _model, _tokenizer: object(),
        runner=runner,
        owner_session="test-session",
    )
    return service, artifacts, index


def test_missing_runtime_is_hidden_and_owned_by_app_update(tmp_path: Path) -> None:
    service, _artifacts, _index = _service(tmp_path, dependencies=False)
    status = service.status()
    assert status.state == "unavailable_in_build"
    assert status.action_id == "update_app"
    assert status.available is False


def test_prepare_runs_artifact_and_index_to_ready(tmp_path: Path) -> None:
    service, artifacts, index = _service(tmp_path)
    assert service.status().action_id == "prepare"

    status, created = service.start("prepare")

    assert created is True
    assert artifacts.ready is True
    assert index.build_calls == 1
    assert status.state == "ready"
    assert status.available is True
    assert status.document_count == 3
    assert status.chunk_count == 7


def test_artifact_failure_becomes_retryable_without_raw_detail(tmp_path: Path) -> None:
    service, artifacts, index = _service(tmp_path)
    artifacts.error = "artifact_integrity_mismatch"

    status, created = service.start("prepare")

    assert created is True
    assert index.build_calls == 0
    assert status.state == "repairable_error"
    assert status.reason_id == "artifact_integrity_mismatch"
    assert status.action_id == "retry"


def test_deferred_cancel_finishes_as_resumable_interrupted(tmp_path: Path) -> None:
    pending = []
    service, _artifacts, index = _service(tmp_path, runner=pending.append)
    running, created = service.start("prepare")
    assert created is True and running.state == "downloading"

    cancelling = service.cancel(running.task_id)
    pending.pop()()

    assert cancelling.state == "cancelling"
    assert index.build_calls == 0
    assert service.status().state == "interrupted"
    assert service.status().action_id == "resume"


def test_ready_artifacts_with_stale_index_get_rebuild_action(tmp_path: Path) -> None:
    service, _artifacts, _index = _service(
        tmp_path,
        artifacts_ready=True,
        index_state="stale",
    )
    status = service.status()
    assert status.state == "stale"
    assert status.action_id == "rebuild"


def test_wrong_action_is_rejected_before_background_work(tmp_path: Path) -> None:
    service, _artifacts, _index = _service(tmp_path)
    with pytest.raises(LocalSemanticPreparationServiceError) as captured:
        service.start("rebuild")
    assert captured.value.code == "action_not_allowed"
