from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app
from api.routers import retrieval as retrieval_router
from kb.local_semantic_preparation_contract import LocalSemanticPreparationStatus


class _Service:
    def __init__(self, readiness: LocalSemanticPreparationStatus) -> None:
        self._readiness = readiness

    def status(self):
        return self._readiness

    def start(self, action: str):
        self.action = action
        return self._readiness, True

    def cancel(self, task_id: str):
        self.cancelled_task_id = task_id
        return LocalSemanticPreparationStatus(
            state="cancelling",
            reason_id="cancel_requested",
            task_id=task_id,
        )


def test_unavailable_local_retrieval_is_not_exposed_as_dead_choice(monkeypatch, tmp_path) -> None:
    service = _Service(
        LocalSemanticPreparationStatus(
            state="unavailable_in_build",
            reason_id="runtime_dependencies_missing",
        )
    )
    monkeypatch.setattr(
        retrieval_router,
        "get_local_semantic_preparation_service",
        lambda _db_dir: service,
    )
    monkeypatch.setattr(
        retrieval_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=tmp_path),
    )

    response = TestClient(app).get("/api/retrieval/local-readiness")

    assert response.status_code == 200
    assert response.json() == {
        "schema_version": 1,
        "default_mode": "bm25",
        "available_modes": ["bm25"],
        "hybrid_local": {
            "state": "unavailable_in_build",
            "reason_id": "runtime_dependencies_missing",
            "task_id": "",
            "phase": "",
            "completed_units": 0,
            "total_units": 0,
            "completed_bytes": 0,
            "total_bytes": 0,
            "eta_seconds": None,
            "document_count": 0,
            "chunk_count": 0,
            "available": False,
            "mode": "hybrid_local",
            "action_id": "update_app",
        },
        "show_mode_control": False,
    }
    assert str(tmp_path) not in response.text


def test_ready_local_retrieval_stays_hidden_until_promotion(monkeypatch, tmp_path) -> None:
    service = _Service(
        LocalSemanticPreparationStatus(
            state="ready",
            reason_id="ready",
            document_count=31,
            chunk_count=2251,
        )
    )
    monkeypatch.setattr(
        retrieval_router,
        "get_local_semantic_preparation_service",
        lambda _db_dir: service,
    )
    monkeypatch.setattr(
        retrieval_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=tmp_path),
    )

    payload = TestClient(app).get("/api/retrieval/local-readiness").json()

    assert payload["available_modes"] == ["bm25"]
    assert payload["show_mode_control"] is False
    assert payload["hybrid_local"]["document_count"] == 31
    assert payload["hybrid_local"]["chunk_count"] == 2251
    assert payload["hybrid_local"]["action_id"] == "none"


def test_management_prepare_endpoint_dispatches_only_the_requested_action(
    monkeypatch,
    tmp_path,
) -> None:
    service = _Service(
        LocalSemanticPreparationStatus(
            state="downloading",
            reason_id="artifact_downloading",
            task_id="task-1",
        )
    )
    monkeypatch.setattr(
        retrieval_router,
        "get_local_semantic_preparation_service",
        lambda _db_dir: service,
    )
    monkeypatch.setattr(
        retrieval_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=tmp_path),
    )

    response = TestClient(app).post(
        "/api/retrieval/local-preparation",
        json={"action": "prepare"},
    )

    assert response.status_code == 200
    assert response.json()["created"] is True
    assert response.json()["hybrid_local"]["task_id"] == "task-1"
    assert service.action == "prepare"


def test_management_cancel_endpoint_is_task_scoped(monkeypatch, tmp_path) -> None:
    service = _Service(
        LocalSemanticPreparationStatus(
            state="downloading",
            reason_id="artifact_downloading",
            task_id="task-1",
        )
    )
    monkeypatch.setattr(
        retrieval_router,
        "get_local_semantic_preparation_service",
        lambda _db_dir: service,
    )
    monkeypatch.setattr(
        retrieval_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=tmp_path),
    )

    response = TestClient(app).post(
        "/api/retrieval/local-preparation/cancel",
        json={"task_id": "task-1"},
    )

    assert response.status_code == 200
    assert response.json()["hybrid_local"]["state"] == "cancelling"
    assert service.cancelled_task_id == "task-1"
