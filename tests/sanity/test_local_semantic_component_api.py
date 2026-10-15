from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routers import retrieval as retrieval_router
from kb.local_semantic_component import (
    CANDIDATE_ID,
    MODEL_REVISION,
    InstalledLocalSemanticComponent,
    LocalSemanticComponentCancelled,
    LocalSemanticComponentError,
    LocalSemanticComponentManifest,
    local_semantic_component_root,
)
from kb.local_semantic_component_download import ComponentDownloadProgress
from kb.local_semantic_component_lifecycle import (
    LocalSemanticComponentLifecycleService,
)
from kb.local_semantic_component_lifecycle_contract import (
    LocalSemanticComponentLifecycleRecord,
)
from kb.local_semantic_component_lifecycle_store import (
    LocalSemanticComponentLifecycleStore,
)
from kb.local_semantic_component_service import (
    LOCAL_SEMANTIC_COMPONENT_MANIFEST_ENV,
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

    def preflight_download(self, _manifest, *, current_generation_bytes=0):
        del current_generation_bytes
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
            raise LocalSemanticComponentCancelled()
        if self.install_errors:
            raise self.install_errors.pop(0)
        generation = self.root / f"generation-{self.install_calls}"
        generation.mkdir(parents=True, exist_ok=True)
        if not health_check(generation):
            raise LocalSemanticComponentError("component_install_health_failed")
        self.current = InstalledLocalSemanticComponent(
            generation_id="0" * 16,
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
        return True


class _Downloader:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls = 0
        self.errors: list[Exception] = []

    def download(self, manifest, _transport, *, should_cancel, on_progress):
        self.calls += 1
        if should_cancel():
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
    owner_session: str = "api-session",
    lifecycle_store=None,
    component=None,
    downloader=None,
    index=None,
):
    component = component or _ComponentStore(tmp_path / "component")
    downloader = downloader or _Downloader(component.root / "downloads")
    index = index or _Index()
    transports = []
    service = LocalSemanticComponentLifecycleService(
        tmp_path / "db",
        _manifest(),
        transport_factory=lambda: transports.append(object()) or transports[-1],
        component_store=component,
        downloader=downloader,
        lifecycle_store=lifecycle_store,
        health_check=lambda _root: True,
        index_ready=index.is_ready,
        prepare_index=index.prepare,
        remove_index=index.remove,
        runner=runner,
        retry_wait=lambda _seconds: None,
        owner_session=owner_session,
    )
    return service, component, downloader, index, transports


def _bind(monkeypatch, tmp_path: Path, service) -> TestClient:
    monkeypatch.setattr(
        retrieval_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=tmp_path / "db"),
    )
    monkeypatch.setattr(
        retrieval_router,
        "get_local_semantic_component_service",
        lambda _db_dir: service,
    )
    return TestClient(app)


def test_unconfigured_component_status_is_read_only_and_not_a_dead_install_choice(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(LOCAL_SEMANTIC_COMPONENT_MANIFEST_ENV, raising=False)
    monkeypatch.setattr(
        retrieval_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=tmp_path / "db"),
    )

    response = TestClient(app).get("/api/retrieval/local-component")

    assert response.status_code == 200
    payload = response.json()
    assert payload["install_available"] is False
    assert payload["available_modes"] == ["bm25"]
    assert payload["show_mode_control"] is False
    assert payload["product_enabled"] is True
    assert payload["hybrid_local"]["state"] == "unavailable"
    assert payload["hybrid_local"]["action_id"] == "update_app"
    assert payload["component"]["external_content_or_model_knowledge_added"] is False
    assert not (tmp_path / "db").exists()
    assert str(tmp_path) not in response.text


def test_unconfigured_component_can_still_be_removed_without_touching_library_data(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(LOCAL_SEMANTIC_COMPONENT_MANIFEST_ENV, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-app-data"))
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    docs = db_dir / "docs.json"
    chunks = db_dir / "chunks" / "paper.jsonl"
    docs.write_text('{"paper": {}}', encoding="utf-8")
    chunks.parent.mkdir()
    chunks.write_text('{"text": "library evidence"}\n', encoding="utf-8")
    index = db_dir / "local_semantic_index" / "e5_small_v1"
    (index / "generations" / ("1" * 32)).mkdir(parents=True)
    (index / "current.json").write_text("{}", encoding="utf-8")
    component_root = local_semantic_component_root(db_dir)
    generation = component_root / "generations" / ("2" * 16)
    generation.mkdir(parents=True)
    (generation / "owned.bin").write_bytes(b"owned")
    monkeypatch.setattr(
        retrieval_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=db_dir),
    )

    response = TestClient(app).delete("/api/retrieval/local-component")

    assert response.status_code == 200
    assert response.json()["hybrid_local"]["state"] == "not_installed"
    assert response.json()["hybrid_local"]["bm25_available"] is True
    assert docs.read_text(encoding="utf-8") == '{"paper": {}}'
    assert chunks.read_text(encoding="utf-8") == '{"text": "library evidence"}\n'
    assert not generation.exists()
    assert not index.exists()
    assert str(tmp_path) not in response.text


def test_enable_requires_explicit_component_consent_through_http(
    monkeypatch,
    tmp_path: Path,
) -> None:
    service, component, downloader, _index, transports = _service(tmp_path)
    client = _bind(monkeypatch, tmp_path, service)

    response = client.post(
        "/api/retrieval/local-component/enable",
        json={"consent": False},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "semantic_component_consent_required"}
    assert component.preflight_calls == 0
    assert downloader.calls == 0
    assert transports == []


def test_duplicate_enable_returns_same_active_task_without_duplicate_work(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pending = []
    service, component, downloader, _index, transports = _service(
        tmp_path,
        runner=pending.append,
    )
    client = _bind(monkeypatch, tmp_path, service)

    first = client.post(
        "/api/retrieval/local-component/enable",
        json={"consent": True},
    ).json()
    second = client.post(
        "/api/retrieval/local-component/enable",
        json={"consent": True},
    ).json()

    assert first["created"] is True and second["created"] is False
    assert first["hybrid_local"]["task_id"] == second["hybrid_local"]["task_id"]
    assert len(pending) == 1
    assert component.preflight_calls == 0 and downloader.calls == 0
    assert transports == []


def test_low_disk_returns_only_concrete_space_action_and_keeps_bm25(
    monkeypatch,
    tmp_path: Path,
) -> None:
    service, component, downloader, _index, transports = _service(tmp_path)
    component.preflight_error = LocalSemanticComponentError(
        "semantic_component_needs_space",
        required_additional_bytes=987654,
    )
    client = _bind(monkeypatch, tmp_path, service)

    response = client.post(
        "/api/retrieval/local-component/enable",
        json={"consent": True},
    )

    public = response.json()["hybrid_local"]
    assert public["state"] == "needs_space"
    assert public["action_id"] == "free_space"
    assert public["required_additional_bytes"] == 987654
    assert public["bm25_available"] is True
    assert downloader.calls == 0 and transports == []
    assert str(tmp_path) not in response.text


def test_repeated_integrity_failure_is_quarantined_without_retry_action(
    monkeypatch,
    tmp_path: Path,
) -> None:
    service, component, _downloader, _index, _transports = _service(tmp_path)
    component.install_errors.extend(
        [
            LocalSemanticComponentError("component_payload_hash_mismatch"),
            LocalSemanticComponentError("component_payload_hash_mismatch"),
        ]
    )
    client = _bind(monkeypatch, tmp_path, service)

    response = client.post(
        "/api/retrieval/local-component/enable",
        json={"consent": True},
    )

    public = response.json()["hybrid_local"]
    assert component.install_calls == 2
    assert service.lifecycle_store.quarantine_reason(service.manifest_identity)
    assert public["state"] == "bm25_available"
    assert public["action_id"] == "none"
    assert public["bm25_available"] is True
    assert "hash_mismatch" not in response.text
    assert str(tmp_path) not in response.text


def test_cancel_resume_and_ready_journey_needs_no_manual_repair(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pending = []
    service, _component, downloader, index, _transports = _service(
        tmp_path,
        runner=pending.append,
    )
    client = _bind(monkeypatch, tmp_path, service)

    started = client.post(
        "/api/retrieval/local-component/enable",
        json={"consent": True},
    ).json()
    task_id = started["hybrid_local"]["task_id"]
    cancelling = client.post(
        "/api/retrieval/local-component/cancel",
        json={"task_id": task_id},
    ).json()
    assert cancelling["hybrid_local"]["state"] == "preparing"
    assert cancelling["hybrid_local"]["action_id"] == "none"
    pending.pop(0)()
    paused = client.get("/api/retrieval/local-component").json()
    assert paused["hybrid_local"]["state"] == "paused"
    assert paused["hybrid_local"]["action_id"] == "resume"

    resumed = client.post("/api/retrieval/local-component/resume").json()
    assert resumed["created"] is True
    pending.pop(0)()
    ready = client.get("/api/retrieval/local-component").json()
    assert ready["hybrid_local"]["state"] == "ready"
    assert ready["hybrid_local"]["action_id"] == "remove"
    assert downloader.calls == 1 and index.prepare_calls == 1
    assert ready["available_modes"] == ["bm25", "hybrid_local"]
    assert ready["show_mode_control"] is False


def test_ready_shipped_component_exposes_explicit_mode_control(
    monkeypatch,
    tmp_path: Path,
) -> None:
    service, *_rest = _service(tmp_path)
    service.enable(consent=True)
    client = _bind(monkeypatch, tmp_path, service)
    monkeypatch.setattr(
        retrieval_router,
        "local_semantic_component_management_available",
        lambda: True,
    )

    payload = client.get("/api/retrieval/local-component").json()

    assert payload["product_enabled"] is True
    assert payload["install_available"] is True
    assert payload["show_mode_control"] is True
    assert payload["default_mode"] == "bm25"
    assert payload["available_modes"] == ["bm25", "hybrid_local"]


def test_restart_resumes_existing_consented_task_through_status_api(
    monkeypatch,
    tmp_path: Path,
) -> None:
    original_pending = []
    first, component, downloader, index, _transports = _service(
        tmp_path,
        runner=original_pending.append,
        owner_session="old",
    )
    running, _created = first.enable(consent=True)
    resumed_pending = []
    resumed, *_rest = _service(
        tmp_path,
        runner=resumed_pending.append,
        owner_session="new",
        lifecycle_store=LocalSemanticComponentLifecycleStore(tmp_path / "db"),
        component=component,
        downloader=downloader,
        index=index,
    )
    client = _bind(monkeypatch, tmp_path, resumed)

    recovering = client.get("/api/retrieval/local-component").json()
    assert recovering["hybrid_local"]["state"] == "preparing"
    assert recovering["hybrid_local"]["task_id"] == running.task_id
    assert len(resumed_pending) == 1
    resumed_pending.pop(0)()
    ready = client.get("/api/retrieval/local-component").json()
    assert ready["hybrid_local"]["state"] == "ready"
    assert ready["hybrid_local"]["bm25_available"] is True


def test_stale_ready_index_repairs_automatically_on_status(
    monkeypatch,
    tmp_path: Path,
) -> None:
    service, _component, downloader, index, _transports = _service(tmp_path)
    ready, _created = service.enable(consent=True)
    assert ready.state == "ready"
    initial_downloads = downloader.calls
    index.ready = False
    client = _bind(monkeypatch, tmp_path, service)

    payload = client.get("/api/retrieval/local-component").json()

    assert payload["hybrid_local"]["state"] == "ready"
    assert index.prepare_calls == 2
    assert downloader.calls == initial_downloads


def test_remove_is_idempotent_and_keeps_bm25_available(
    monkeypatch,
    tmp_path: Path,
) -> None:
    service, component, _downloader, index, _transports = _service(tmp_path)
    service.enable(consent=True)
    client = _bind(monkeypatch, tmp_path, service)

    first = client.delete("/api/retrieval/local-component")
    second = client.delete("/api/retrieval/local-component")

    assert first.status_code == 200 and second.status_code == 200
    for response in (first, second):
        public = response.json()["hybrid_local"]
        assert public["state"] == "not_installed"
        assert public["action_id"] == "enable"
        assert public["bm25_available"] is True
    assert component.remove_calls == 2
    assert index.remove_calls == 2


@pytest.mark.parametrize(
    ("internal_state", "public_state", "action_id"),
    [
        ("not_installed", "not_installed", "enable"),
        ("downloading", "preparing", "cancel"),
        ("ready", "ready", "remove"),
        ("paused_by_user", "paused", "resume"),
        ("needs_space", "needs_space", "free_space"),
        ("degraded_bm25", "bm25_available", "none"),
        ("removing", "removing", "none"),
        ("unavailable_in_build", "unavailable", "update_app"),
    ],
)
def test_api_exposes_only_eight_public_states(
    monkeypatch,
    tmp_path: Path,
    internal_state: str,
    public_state: str,
    action_id: str,
) -> None:
    record = LocalSemanticComponentLifecycleRecord(
        state=internal_state,
        reason_id="component_payload_hash_mismatch",
        task_id="task-1",
    )
    service = SimpleNamespace(status=lambda: record)
    response = _bind(monkeypatch, tmp_path, service).get(
        "/api/retrieval/local-component"
    )

    assert response.status_code == 200
    public = response.json()["hybrid_local"]
    assert public["state"] == public_state
    assert public["action_id"] == action_id
    assert public["bm25_available"] is True
    assert "payload_hash_mismatch" not in response.text


def test_component_management_api_requires_management_auth_in_production(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("KB_ENV", "production")
    monkeypatch.setenv("KB_APP_ENV", "")
    monkeypatch.setenv("KB_REQUIRE_MANAGEMENT_AUTH", "1")
    monkeypatch.setenv("KB_MANAGEMENT_ACCESS_TOKEN", "component-admin")
    monkeypatch.setenv("KB_MANAGEMENT_ACCESS_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_PRIVATE_INSTANCE_AUTH", "0")
    monkeypatch.setenv("KB_ENABLE_AUTH_GATE", "0")
    monkeypatch.setenv("KB_REQUIRE_AUTH", "0")
    record = LocalSemanticComponentLifecycleRecord(
        state="not_installed",
        reason_id="component_not_installed",
    )
    service = SimpleNamespace(status=lambda: record)
    client = _bind(monkeypatch, tmp_path, service)

    denied = client.get("/api/retrieval/local-component")
    allowed = client.get(
        "/api/retrieval/local-component",
        headers={"X-KB-Management-Token": "component-admin"},
    )

    assert denied.status_code == 401
    assert denied.json()["detail"] == "Management authentication required"
    assert allowed.status_code == 200
    assert allowed.json()["hybrid_local"]["state"] == "not_installed"
