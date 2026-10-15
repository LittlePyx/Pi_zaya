from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kb.local_semantic_component import (
    CANDIDATE_ID,
    MODEL_REVISION,
    InstalledLocalSemanticComponent,
    LocalSemanticComponentError,
    LocalSemanticComponentManifest,
)
from kb.local_semantic_component_download import ComponentDownloadProgress
from kb.local_semantic_component_lifecycle import (
    LocalSemanticComponentLifecycleError,
    LocalSemanticComponentLifecycleService,
)
from kb.local_semantic_component_lifecycle_store import (
    LocalSemanticComponentLifecycleStore,
)


def _manifest() -> LocalSemanticComponentManifest:
    return LocalSemanticComponentManifest(
        component_id="component",
        release_id="component-v1",
        download_url="https://github.com/release/component-v1/component.zip",
        archive_name="component.zip",
        archive_root="c",
        archive_bytes=8,
        archive_sha256=hashlib.sha256(b"archive").hexdigest(),
        payload_bytes=10,
        payload_file_count=2,
        payload_sha256=hashlib.sha256(b"payload").hexdigest(),
        candidate_id=CANDIDATE_ID,
        model_revision=MODEL_REVISION,
    )


class _ComponentStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.current = None
        self.preflight_error: Exception | None = None
        self.install_errors: list[Exception] = []
        self.preflight_calls = 0
        self.install_calls = 0
        self.remove_calls = 0
        self.cleanup_calls = 0

    def preflight_download(self, _manifest, *, current_generation_bytes=0):
        self.preflight_calls += 1
        if self.preflight_error:
            raise self.preflight_error
        return SimpleNamespace(sufficient=True)

    def install_from_archive(
        self,
        _archive,
        manifest,
        *,
        should_cancel,
        health_check,
    ):
        self.install_calls += 1
        if should_cancel():
            from kb.local_semantic_component import LocalSemanticComponentCancelled

            raise LocalSemanticComponentCancelled()
        if self.install_errors:
            raise self.install_errors.pop(0)
        generation = self.root / f"generation-{self.install_calls}"
        generation.mkdir(parents=True, exist_ok=True)
        if not health_check(generation):
            raise LocalSemanticComponentError("component_install_health_failed")
        self.current = InstalledLocalSemanticComponent(
            generation_id="0" * 32,
            root=generation,
            component_id=manifest.component_id,
            release_id=manifest.release_id,
            archive_sha256=manifest.archive_sha256,
            payload_sha256=manifest.payload_sha256,
        )
        return self.current

    def load_current(self):
        if self.current is None:
            raise LocalSemanticComponentError("component_not_installed")
        return self.current

    def remove_all(self):
        self.remove_calls += 1
        self.current = None
        return SimpleNamespace(cleanup_pending=False)

    def retry_pending_cleanup(self) -> bool:
        self.cleanup_calls += 1
        return True


class _Downloader:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls = 0
        self.errors: list[Exception] = []

    def download(self, manifest, _transport, *, should_cancel, on_progress):
        self.calls += 1
        if should_cancel():
            from kb.local_semantic_component import LocalSemanticComponentCancelled

            raise LocalSemanticComponentCancelled()
        if self.errors:
            raise self.errors.pop(0)
        self.root.mkdir(parents=True, exist_ok=True)
        archive = self.root / f"{manifest.archive_sha256}.zip"
        archive.write_bytes(b"archive")
        on_progress(ComponentDownloadProgress(8, 8, 0, 0))
        return SimpleNamespace(archive_path=archive)


class _Index:
    def __init__(self) -> None:
        self.ready = False
        self.prepare_calls = 0
        self.remove_calls = 0

    def is_ready(self, _component_root: Path) -> bool:
        return self.ready

    def prepare(self, _component_root: Path, *, should_cancel, on_progress):
        self.prepare_calls += 1
        if should_cancel():
            from kb.local_semantic_component import LocalSemanticComponentCancelled

            raise LocalSemanticComponentCancelled()
        on_progress(1, 1)
        self.ready = True

    def remove(self) -> None:
        self.remove_calls += 1
        self.ready = False


def _service(
    tmp_path: Path,
    *,
    runner=lambda callback: callback(),
    retry_wait=lambda _seconds: None,
    owner_session: str = "session-one",
    lifecycle_store=None,
):
    manifest = _manifest()
    component = _ComponentStore(tmp_path / "component")
    downloader = _Downloader(component.root / "downloads")
    index = _Index()
    transports = []
    service = LocalSemanticComponentLifecycleService(
        tmp_path / "db",
        manifest,
        transport_factory=lambda: transports.append(object()) or transports[-1],
        component_store=component,
        downloader=downloader,
        lifecycle_store=lifecycle_store,
        health_check=lambda _root: True,
        index_ready=index.is_ready,
        prepare_index=index.prepare,
        remove_index=index.remove,
        runner=runner,
        retry_wait=retry_wait,
        owner_session=owner_session,
    )
    return service, component, downloader, index, transports


def _observed_attributes(path: Path) -> list[dict]:
    return [
        json.loads(line)["attributes"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_enable_requires_explicit_consent_before_work(tmp_path: Path) -> None:
    service, component, downloader, _index, transports = _service(tmp_path)
    with pytest.raises(LocalSemanticComponentLifecycleError) as captured:
        service.enable(consent=False)
    assert captured.value.code == "semantic_component_consent_required"
    assert component.preflight_calls == 0
    assert downloader.calls == 0
    assert transports == []


def test_happy_path_reaches_ready_only_after_component_and_index(tmp_path: Path) -> None:
    service, component, downloader, index, transports = _service(tmp_path)
    status, created = service.enable(consent=True)
    assert created is True
    assert status.state == "ready"
    assert status.available is True
    assert component.install_calls == 1
    assert downloader.calls == 1
    assert index.prepare_calls == 1 and index.ready is True
    assert len(transports) == 1
    assert status.public_payload()["bm25_available"] is True


def test_network_failure_retries_automatically_without_user_action(tmp_path: Path) -> None:
    waits = []
    service, _component, downloader, index, _transports = _service(
        tmp_path,
        retry_wait=waits.append,
    )
    downloader.errors.append(
        LocalSemanticComponentError("component_download_transport_failed")
    )
    status, _created = service.enable(consent=True)
    assert status.state == "ready"
    assert downloader.calls == 2
    assert waits == [2.0]
    assert index.ready is True


def test_repeated_integrity_failure_is_quarantined_and_publicly_generic(
    tmp_path: Path,
) -> None:
    service, component, _downloader, _index, _transports = _service(tmp_path)
    component.install_errors.extend(
        [
            LocalSemanticComponentError("component_payload_hash_mismatch"),
            LocalSemanticComponentError("component_payload_hash_mismatch"),
        ]
    )
    status, _created = service.enable(consent=True)
    assert status.state == "degraded_bm25"
    assert status.public_payload()["reason_id"] == (
        "semantic_component_temporarily_unavailable"
    )
    assert status.public_payload()["action_id"] == "none"
    assert service.lifecycle_store.quarantine_reason(service.manifest_identity) == (
        "component_payload_hash_mismatch"
    )

    before = component.install_calls
    second, _created = service.enable(consent=True)
    assert second.state == "degraded_bm25"
    assert component.install_calls == before


def test_low_disk_has_exact_concrete_action_and_never_downloads(tmp_path: Path) -> None:
    service, component, downloader, _index, transports = _service(tmp_path)
    component.preflight_error = LocalSemanticComponentError(
        "semantic_component_needs_space",
        required_additional_bytes=1234,
    )
    status, _created = service.enable(consent=True)
    assert status.state == "needs_space"
    assert status.public_payload()["action_id"] == "free_space"
    assert status.public_payload()["required_additional_bytes"] == 1234
    assert status.public_payload()["bm25_available"] is True
    assert downloader.calls == 0 and transports == []


def test_user_cancel_stays_paused_and_keeps_partial_work_resumable(
    tmp_path: Path,
) -> None:
    pending = []
    service, _component, downloader, _index, _transports = _service(
        tmp_path,
        runner=pending.append,
    )
    running, _created = service.enable(consent=True)
    assert running.state == "checking_capacity"
    service.cancel(running.task_id)
    pending.pop()()
    status = service.status()
    assert status.state == "paused_by_user"
    assert status.public_payload()["action_id"] == "resume"
    assert downloader.calls == 0


def test_non_user_restart_resumes_without_new_consent(tmp_path: Path) -> None:
    original_pending = []
    first, component, downloader, index, _transports = _service(
        tmp_path,
        runner=original_pending.append,
        owner_session="old",
    )
    running, _created = first.enable(consent=True)
    assert running.state == "checking_capacity"

    resumed_pending = []
    resumed = LocalSemanticComponentLifecycleService(
        tmp_path / "db",
        _manifest(),
        transport_factory=lambda: object(),
        component_store=component,
        downloader=downloader,
        lifecycle_store=LocalSemanticComponentLifecycleStore(tmp_path / "db"),
        health_check=lambda _root: True,
        index_ready=index.is_ready,
        prepare_index=index.prepare,
        remove_index=index.remove,
        runner=resumed_pending.append,
        retry_wait=lambda _seconds: None,
        owner_session="new",
    )
    assert resumed.status().state == "recovering"
    assert len(resumed_pending) == 1
    resumed_pending.pop()()
    assert resumed.status().state == "ready"


def test_remove_during_preparation_cancels_then_removes_in_same_task(
    tmp_path: Path,
) -> None:
    pending = []
    service, component, downloader, index, _transports = _service(
        tmp_path,
        runner=pending.append,
    )
    running, _created = service.enable(consent=True)
    removing, created = service.remove()
    assert created is False
    assert removing.operation == "remove"
    pending.pop()()
    assert service.status().state == "not_installed"
    assert component.remove_calls == 1
    assert index.remove_calls == 1
    assert downloader.calls == 0


def test_stale_ready_state_triggers_background_recovery_and_index_rebuild(
    tmp_path: Path,
) -> None:
    service, _component, downloader, index, _transports = _service(tmp_path)
    ready, _created = service.enable(consent=True)
    assert ready.state == "ready"
    initial_downloads = downloader.calls
    index.ready = False

    recovered = service.status()

    assert recovered.state == "ready"
    assert index.prepare_calls == 2
    assert downloader.calls == initial_downloads


def test_enable_lifecycle_observation_closes_with_ready_without_identity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "observations.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(path))
    service, _component, _downloader, _index, _transports = _service(tmp_path)

    ready, _created = service.enable(consent=True)

    events = _observed_attributes(path)
    assert ready.state == "ready"
    assert [item["outcome_id"] for item in events] == [
        "local_semantic_component.started",
        "local_semantic_component.ready",
    ]
    assert events[-1]["lifecycle_operation_id"] == "enable"
    assert events[-1]["lifecycle_state_id"] == "ready"
    assert events[-1]["bm25_available"] is True
    stored = path.read_text(encoding="utf-8")
    assert ready.task_id not in stored
    assert service.manifest_identity not in stored


def test_low_disk_lifecycle_observation_closes_with_actionable_category(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "observations.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(path))
    service, component, _downloader, _index, _transports = _service(tmp_path)
    component.preflight_error = LocalSemanticComponentError(
        "semantic_component_needs_space",
        required_additional_bytes=1234,
    )

    status, _created = service.enable(consent=True)

    events = _observed_attributes(path)
    assert status.state == "needs_space"
    assert events[-1]["outcome_id"] == "local_semantic_component.needs_space"
    assert events[-1]["fallback_reason_id"] == "storage_capacity"
    assert events[-1]["bm25_available"] is True
    assert "1234" not in path.read_text(encoding="utf-8")


def test_cancel_then_remove_lifecycle_observation_has_terminal_outcomes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "observations.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(path))
    pending = []
    service, _component, _downloader, _index, _transports = _service(
        tmp_path,
        runner=pending.append,
    )

    running, _created = service.enable(consent=True)
    service.cancel(running.task_id)
    pending.pop()()
    removing, remove_created = service.remove()
    assert removing.operation == "remove" and remove_created is True
    pending.pop()()

    events = _observed_attributes(path)
    assert [item["outcome_id"] for item in events] == [
        "local_semantic_component.started",
        "local_semantic_component.paused",
        "local_semantic_component.started",
        "local_semantic_component.removed",
    ]
    assert events[1]["fallback_reason_id"] == "user_pause"
    assert events[-1]["consented"] is False
    assert all(item["bm25_available"] is True for item in events)


def test_integrity_failure_lifecycle_observation_reports_containment_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "observations.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(path))
    service, component, _downloader, _index, _transports = _service(tmp_path)
    component.install_errors.extend(
        [
            LocalSemanticComponentError("component_payload_hash_mismatch"),
            LocalSemanticComponentError("component_payload_hash_mismatch"),
        ]
    )

    status, _created = service.enable(consent=True)

    events = _observed_attributes(path)
    assert status.state == "degraded_bm25"
    assert events[-1]["outcome_id"] == "local_semantic_component.bm25_contained"
    assert events[-1]["fallback_reason_id"] == "component_integrity"
    assert events[-1]["attempt_count"] == 2
    stored = path.read_text(encoding="utf-8")
    assert "component_payload_hash_mismatch" not in stored
    assert events[-1]["bm25_available"] is True
