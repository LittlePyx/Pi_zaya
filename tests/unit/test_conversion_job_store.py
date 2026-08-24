from __future__ import annotations

import json
import multiprocessing
import os
import sqlite3
from pathlib import Path

from kb.conversion_job_store import ConversionJobStore


def _write_running_job_then_crash(db_path: str, task: dict) -> None:
    store = ConversionJobStore(db_path)
    store.create_queued(task, owner_session="crashed-backend")
    store.mark_running(str(task.get("_tid") or ""), owner_session="crashed-backend")
    store.update_progress(str(task.get("_tid") or ""), page_done=2, page_total=6)
    os._exit(17)


def _task(tmp_path: Path, *, task_id: str = "task-1", no_llm: bool = False) -> dict:
    pdf = tmp_path / "papers" / "paper.pdf"
    return {
        "_tid": task_id,
        "pdf": str(pdf),
        "name": pdf.name,
        "out_root": str(tmp_path / "markdown"),
        "db_dir": str(tmp_path / "db"),
        "no_llm": no_llm,
        "eq_image_fallback": False,
        "replace": True,
        "speed_mode": "balanced",
        "repair_context": {
            "action": "reconvert",
            "scope": "pages",
            "issue_codes": ["weak_structure"],
            "retry_pages": [5, 3, 5],
        },
        # Unknown fields, especially credentials, must never enter the ledger.
        "api_key": "do-not-persist-this-secret",
    }


def test_conversion_job_store_migrates_existing_library_db_without_credentials(tmp_path: Path):
    db_path = tmp_path / "library.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE pdf_files (sha1 TEXT PRIMARY KEY, path TEXT NOT NULL, created_at REAL NOT NULL)")

    store = ConversionJobStore(db_path)
    assert store.create_queued(_task(tmp_path), owner_session="old-session") is True

    with sqlite3.connect(db_path) as conn:
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        row = conn.execute(
            "SELECT task_id, repair_context_json FROM conversion_jobs WHERE task_id = 'task-1'"
        ).fetchone()

    assert {"pdf_files", "conversion_jobs"}.issubset(tables)
    assert row is not None and row[0] == "task-1"
    assert json.loads(str(row[1]))["issue_codes"] == ["weak_structure"]
    assert json.loads(str(row[1]))["retry_pages"] == [3, 5]
    assert b"do-not-persist-this-secret" not in db_path.read_bytes()


def test_restart_reconciliation_is_recoverable_and_reuses_page_cache(tmp_path: Path):
    store = ConversionJobStore(tmp_path / "library.sqlite3")
    task = _task(tmp_path)
    assert store.create_queued(task, owner_session="old-session") is True
    assert store.mark_running("task-1", owner_session="old-session") is True
    assert store.update_progress("task-1", page_done=3, page_total=8, stage="converting") is True

    cache_page = tmp_path / "markdown" / "paper" / ".conversion_cache" / "pages" / "00001"
    cache_page.mkdir(parents=True)
    (cache_page / "entry.json").write_text("{}", encoding="utf-8")
    (cache_page / "page.txt").write_text("page one", encoding="utf-8")

    interrupted = store.reconcile_after_restart(owner_session="new-session")
    recoverable = store.list_recoverable()

    assert len(interrupted) == 1
    assert len(recoverable) == 1
    assert recoverable[0]["state"] == "interrupted"
    assert recoverable[0]["page_done"] == 3
    assert recoverable[0]["page_total"] == 8
    assert recoverable[0]["cached_page_count"] == 1
    assert store.reconcile_after_restart(owner_session="new-session") == []

    resumed = store.queue_for_resume("task-1", owner_session="new-session")
    assert resumed is not None
    assert resumed["_tid"] == "task-1"
    assert resumed["resumed"] is True
    assert store.queue_for_resume("task-1", owner_session="new-session") is None
    assert store.list_recoverable() == []


def test_terminal_transition_is_exactly_once(tmp_path: Path):
    store = ConversionJobStore(tmp_path / "library.sqlite3")
    assert store.create_queued(_task(tmp_path), owner_session="session") is True
    assert store.mark_running("task-1", owner_session="session") is True
    success = {
        "outcome": "success",
        "retry_action": "",
        "message": "Conversion and index update completed.",
        "detail": "",
        "page_done": 8,
        "page_total": 8,
        "finished_at": 120.0,
    }
    failure = {
        **success,
        "outcome": "index_failed",
        "retry_action": "reindex",
        "message": "Index failed.",
        "finished_at": 121.0,
    }

    assert store.finish("task-1", success) is True
    assert store.finish("task-1", failure) is False

    results = store.list_recent_results()
    assert len(results) == 1
    assert results[0]["outcome"] == "success"
    assert results[0]["page_done"] == 8
    assert store.list_recoverable() == []


def test_new_conversion_supersedes_stale_recovery_for_same_pdf(tmp_path: Path):
    store = ConversionJobStore(tmp_path / "library.sqlite3")
    assert store.create_queued(_task(tmp_path, task_id="old"), owner_session="old-session") is True
    store.reconcile_after_restart(owner_session="new-session")

    assert store.create_queued(_task(tmp_path, task_id="new"), owner_session="new-session") is True
    assert store.list_recoverable() == []
    results = store.list_recent_results()
    old = next(item for item in results if item["task_id"] == "old")
    assert old["outcome"] == "cancelled"


def test_backend_process_crash_leaves_running_job_recoverable(tmp_path: Path):
    db_path = tmp_path / "library.sqlite3"
    task = _task(tmp_path, task_id="crash-task", no_llm=True)
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_write_running_job_then_crash,
        args=(str(db_path), task),
    )
    process.start()
    process.join(timeout=20)

    assert process.exitcode == 17
    store = ConversionJobStore(db_path)
    interrupted = store.reconcile_after_restart(owner_session="restarted-backend")
    assert [item["task_id"] for item in interrupted] == ["crash-task"]
    assert store.list_recoverable()[0]["page_done"] == 2


def test_recoverable_job_follows_source_rename(tmp_path: Path):
    store = ConversionJobStore(tmp_path / "library.sqlite3")
    task = _task(tmp_path, task_id="rename-task", no_llm=True)
    assert store.create_queued(task, owner_session="old") is True
    store.reconcile_after_restart(owner_session="new")
    renamed = tmp_path / "papers" / "renamed.pdf"

    assert store.update_recoverable_pdf_path(task["pdf"], renamed) == 1
    recoverable = store.list_recoverable()[0]
    assert recoverable["pdf"] == str(renamed)
    assert recoverable["name"] == "renamed.pdf"
