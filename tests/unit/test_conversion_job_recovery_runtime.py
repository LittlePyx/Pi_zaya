from __future__ import annotations

from pathlib import Path
from threading import Lock
from types import SimpleNamespace

from kb import task_runtime
from kb.conversion_job_store import ConversionJobStore


def _runtime_state() -> dict:
    return {
        "queue": [],
        "active_tasks": [],
        "recent_tasks": [],
        "active_count": 0,
        "running": False,
        "done": 0,
        "total": 0,
        "current": "",
        "cur_page_done": 0,
        "cur_page_total": 0,
        "cur_page_msg": "",
        "conversion_stage": "",
        "cancel": False,
        "last": "",
    }


def _persist_interrupted(
    tmp_path: Path,
    *,
    task_id: str,
    create_pdf: bool,
    no_llm: bool,
) -> ConversionJobStore:
    pdf = tmp_path / "papers" / f"{task_id}.pdf"
    if create_pdf:
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(b"%PDF-1.4\n%%EOF")
    store = ConversionJobStore(tmp_path / "library.sqlite3")
    store.create_queued(
        {
            "_tid": task_id,
            "pdf": str(pdf),
            "name": pdf.name,
            "out_root": str(tmp_path / "markdown"),
            "db_dir": str(tmp_path / "db"),
            "no_llm": no_llm,
            "replace": True,
            "speed_mode": "no_llm" if no_llm else "balanced",
        },
        owner_session="dead-backend",
    )
    store.mark_running(task_id, owner_session="dead-backend")
    return store


def _install_isolated_runtime(monkeypatch, store: ConversionJobStore, *, vision_api_key: str | None):
    monkeypatch.setattr(task_runtime, "_BG_STATE", _runtime_state())
    monkeypatch.setattr(task_runtime, "_BG_LOCK", Lock())
    monkeypatch.setattr(task_runtime, "_BG_SESSION_ID", "new-backend")
    monkeypatch.setattr(task_runtime, "_BG_RECONCILED_DB_PATHS", set())
    monkeypatch.setattr(task_runtime, "_bg_job_store", lambda: store)
    monkeypatch.setattr(task_runtime, "_bg_ensure_started", lambda: None)
    monkeypatch.setattr(
        task_runtime,
        "load_settings",
        lambda: SimpleNamespace(
            library_db_path=store.db_path,
            vision_api_key=vision_api_key,
        ),
    )


def test_resume_missing_source_stays_recoverable_without_queue_loop(tmp_path: Path, monkeypatch):
    store = _persist_interrupted(tmp_path, task_id="missing", create_pdf=False, no_llm=True)
    _install_isolated_runtime(monkeypatch, store, vision_api_key=None)

    result = task_runtime._bg_resume_task("missing")
    snapshot = task_runtime._bg_snapshot()

    assert result["state"] == "blocked"
    assert result["blocked_reason"] == "source_missing"
    assert snapshot["queue"] == []
    assert snapshot["running"] is False
    assert snapshot["recoverable_count"] == 1
    assert snapshot["recoverable_tasks"][0]["blocked_reason"] == "source_missing"


def test_resume_missing_api_key_stays_recoverable(tmp_path: Path, monkeypatch):
    store = _persist_interrupted(tmp_path, task_id="needs-key", create_pdf=True, no_llm=False)
    _install_isolated_runtime(monkeypatch, store, vision_api_key=None)

    result = task_runtime._bg_resume_task("needs-key")

    assert result["state"] == "blocked"
    assert result["blocked_reason"] == "api_key_missing"
    assert task_runtime._BG_STATE["queue"] == []
    assert store.list_recoverable()[0]["blocked_reason"] == "api_key_missing"


def test_resume_is_idempotent_and_keeps_same_task_id(tmp_path: Path, monkeypatch):
    store = _persist_interrupted(tmp_path, task_id="resume-me", create_pdf=True, no_llm=True)
    _install_isolated_runtime(monkeypatch, store, vision_api_key=None)

    first = task_runtime._bg_resume_task("resume-me")
    second = task_runtime._bg_resume_task("resume-me")
    snapshot = task_runtime._bg_snapshot()

    assert first == {
        "matched": True,
        "enqueued": True,
        "task_id": "resume-me",
        "state": "queued",
    }
    assert second["state"] == "already_busy"
    assert len(snapshot["queue"]) == 1
    assert snapshot["queue"][0]["_tid"] == "resume-me"
    assert snapshot["recoverable_count"] == 0


def test_cancel_interrupted_task_dismisses_recovery(tmp_path: Path, monkeypatch):
    store = _persist_interrupted(tmp_path, task_id="dismiss-me", create_pdf=True, no_llm=True)
    _install_isolated_runtime(monkeypatch, store, vision_api_key=None)
    task_runtime._bg_reconcile_persisted_jobs()

    result = task_runtime._bg_cancel_task("dismiss-me")

    assert result["matched"] is True
    assert result["state"] == "recovery_cancelled"
    assert store.list_recoverable() == []
    assert store.list_recent_results()[0]["outcome"] == "cancelled"
