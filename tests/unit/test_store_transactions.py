from __future__ import annotations

import json
import multiprocessing
import os
import stat
import threading
from pathlib import Path

import pytest

from kb import store
from kb.store import db_write_lock, load_all_chunks, load_docs_index, save_docs_index, write_doc_chunks


def _merge_doc_worker(
    db_dir_raw: str,
    doc_id: str,
    acquired: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    db_dir = Path(db_dir_raw)
    with db_write_lock(db_dir, timeout_s=10):
        docs = load_docs_index(db_dir)
        acquired.set()
        if not release.wait(10):
            raise TimeoutError("test worker was not released")
        docs[doc_id] = {"doc_id": doc_id, "path": f"/{doc_id}.md", "num_chunks": 1}
        save_docs_index(db_dir, docs)


def test_database_write_lock_serializes_process_read_modify_write(tmp_path: Path):
    ctx = multiprocessing.get_context("spawn")
    db_dir = tmp_path / "db"
    first_acquired = ctx.Event()
    first_release = ctx.Event()
    second_acquired = ctx.Event()
    second_release = ctx.Event()
    second_release.set()

    first = ctx.Process(target=_merge_doc_worker, args=(str(db_dir), "first", first_acquired, first_release))
    second = ctx.Process(target=_merge_doc_worker, args=(str(db_dir), "second", second_acquired, second_release))
    first.start()
    try:
        assert first_acquired.wait(10)
        second.start()
        assert not second_acquired.wait(0.4)

        with pytest.raises(TimeoutError):
            with db_write_lock(db_dir, timeout_s=0.1):
                pass
    finally:
        first_release.set()
        for process in (first, second):
            if process.pid is None:
                continue
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join(5)

    assert first.exitcode == 0
    assert second.exitcode == 0
    assert second_acquired.is_set()
    assert set(load_docs_index(db_dir)) == {"first", "second"}


def test_database_write_lock_is_released_when_owner_process_is_terminated(tmp_path: Path):
    ctx = multiprocessing.get_context("spawn")
    db_dir = tmp_path / "db"
    acquired = ctx.Event()
    never_release = ctx.Event()
    owner = ctx.Process(target=_merge_doc_worker, args=(str(db_dir), "abandoned", acquired, never_release))
    owner.start()
    assert acquired.wait(10)

    owner.terminate()
    owner.join(10)
    assert not owner.is_alive()

    with db_write_lock(db_dir, timeout_s=2):
        save_docs_index(db_dir, {"recovered": {"doc_id": "recovered", "num_chunks": 1}})

    assert set(load_docs_index(db_dir)) == {"recovered"}


def test_docs_index_is_atomically_replaced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    db_dir = tmp_path / "db"
    old_docs = {"old": {"doc_id": "old", "num_chunks": 1}}
    new_docs = {"new": {"doc_id": "new", "num_chunks": 2}}
    save_docs_index(db_dir, old_docs)
    target = db_dir / "docs.json"
    original_replace = store.os.replace
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def blocking_replace(src, dst):
        assert Path(dst) == target
        assert json.loads(Path(src).read_text(encoding="utf-8")) == new_docs
        entered.set()
        assert release.wait(10)
        return original_replace(src, dst)

    def writer() -> None:
        try:
            save_docs_index(db_dir, new_docs)
        except BaseException as exc:  # pragma: no cover - assertion surfaced below
            errors.append(exc)

    monkeypatch.setattr(store.os, "replace", blocking_replace)
    thread = threading.Thread(target=writer)
    thread.start()
    assert entered.wait(10)
    assert load_docs_index(db_dir) == old_docs
    release.set()
    thread.join(10)

    assert not thread.is_alive()
    assert errors == []
    assert load_docs_index(db_dir) == new_docs
    assert list(db_dir.glob(".docs.json.*.tmp")) == []


def test_chunk_file_is_atomically_replaced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    db_dir = tmp_path / "db"
    write_doc_chunks(db_dir, "paper", [{"text": "old", "meta": {}}])
    target = db_dir / "chunks" / "paper.jsonl"
    original_replace = store.os.replace
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def blocking_replace(src, dst):
        assert Path(dst) == target
        rows = [json.loads(line) for line in Path(src).read_text(encoding="utf-8").splitlines()]
        assert [row["text"] for row in rows] == ["new one", "new two"]
        entered.set()
        assert release.wait(10)
        return original_replace(src, dst)

    def writer() -> None:
        try:
            write_doc_chunks(
                db_dir,
                "paper",
                [{"text": "new one", "meta": {}}, {"text": "new two", "meta": {}}],
            )
        except BaseException as exc:  # pragma: no cover - assertion surfaced below
            errors.append(exc)

    monkeypatch.setattr(store.os, "replace", blocking_replace)
    thread = threading.Thread(target=writer)
    thread.start()
    assert entered.wait(10)
    old_rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert [row["text"] for row in old_rows] == ["old"]
    release.set()
    thread.join(10)

    assert not thread.is_alive()
    assert errors == []
    new_rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert [row["text"] for row in new_rows] == ["new one", "new two"]
    assert list(target.parent.glob(".paper.jsonl.*.tmp")) == []


def test_failed_atomic_replace_preserves_old_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    db_dir = tmp_path / "db"
    old_docs = {"old": {"doc_id": "old", "num_chunks": 1}}
    save_docs_index(db_dir, old_docs)

    def fail_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(store.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        save_docs_index(db_dir, {"new": {"doc_id": "new", "num_chunks": 1}})

    assert load_docs_index(db_dir) == old_docs
    assert list(db_dir.glob(".docs.json.*.tmp")) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes are not meaningful on Windows")
def test_atomic_replace_preserves_existing_file_mode(tmp_path: Path):
    db_dir = tmp_path / "db"
    save_docs_index(db_dir, {"old": {"doc_id": "old", "num_chunks": 1}})
    target = db_dir / "docs.json"
    target.chmod(0o640)

    save_docs_index(db_dir, {"new": {"doc_id": "new", "num_chunks": 1}})

    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_empty_docs_index_does_not_load_orphan_chunks(tmp_path: Path):
    db_dir = tmp_path / "db"
    write_doc_chunks(db_dir, "orphan", [{"text": "stale", "meta": {}}])

    assert load_all_chunks(db_dir) == []

    save_docs_index(db_dir, {})

    assert load_all_chunks(db_dir) == []
    assert [row["text"] for row in load_all_chunks(db_dir, include_non_ready=True)] == ["stale"]


def test_load_all_chunks_includes_indexable_degraded_documents(tmp_path: Path):
    db_dir = tmp_path / "db"
    write_doc_chunks(db_dir, "degraded", [{"text": "usable page", "meta": {"evidence_ready": True}}])
    write_doc_chunks(db_dir, "blocked", [{"text": "blocked page", "meta": {}}])
    save_docs_index(
        db_dir,
        {
            "degraded": {
                "doc_id": "degraded",
                "path": "/degraded.md",
                "num_chunks": 1,
                "index_status": "quality_degraded",
                "quality_gate": {"status": "degraded", "indexable": True},
            },
            "blocked": {
                "doc_id": "blocked",
                "path": "/blocked.md",
                "num_chunks": 1,
                "index_status": "quality_blocked",
                "quality_gate": {"status": "blocked", "indexable": False},
            },
        },
    )

    assert [row["text"] for row in load_all_chunks(db_dir)] == ["usable page"]


def test_load_all_chunks_rejects_degraded_document_without_explicit_indexable_gate(
    tmp_path: Path,
):
    db_dir = tmp_path / "db"
    write_doc_chunks(
        db_dir,
        "legacy-degraded",
        [{"text": "possibly corrupt legacy page", "meta": {"evidence_ready": True}}],
    )
    save_docs_index(
        db_dir,
        {
            "legacy-degraded": {
                "doc_id": "legacy-degraded",
                "path": "/legacy-degraded.md",
                "num_chunks": 1,
                "index_status": "quality_degraded",
                "quality_gate": {"status": "degraded"},
            }
        },
    )

    assert load_all_chunks(db_dir) == []
    assert [row["text"] for row in load_all_chunks(db_dir, include_non_ready=True)] == [
        "possibly corrupt legacy page"
    ]


def test_chunk_deleted_after_reader_enumeration_is_ignored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    db_dir = tmp_path / "db"
    target = db_dir / "chunks" / "paper.jsonl"
    write_doc_chunks(db_dir, "paper", [{"text": "published", "meta": {}}])
    save_docs_index(
        db_dir,
        {"paper": {"doc_id": "paper", "path": "/paper.md", "num_chunks": 1, "index_status": "ready"}},
    )
    original_open = Path.open
    entered = threading.Event()
    release = threading.Event()
    results: list[list[dict]] = []
    errors: list[BaseException] = []

    def blocking_open(path: Path, *args, **kwargs):
        if path == target:
            entered.set()
            assert release.wait(10)
        return original_open(path, *args, **kwargs)

    def reader() -> None:
        try:
            results.append(load_all_chunks(db_dir))
        except BaseException as exc:  # pragma: no cover - assertion surfaced below
            errors.append(exc)

    monkeypatch.setattr(Path, "open", blocking_open)
    thread = threading.Thread(target=reader)
    thread.start()
    assert entered.wait(10)
    target.unlink()
    release.set()
    thread.join(10)

    assert not thread.is_alive()
    assert errors == []
    assert results == [[]]


def test_library_purge_waits_for_same_database_writer_lock(tmp_path: Path):
    from api.routers import library as library_router

    db_dir = tmp_path / "db"
    md_path = tmp_path / "paper.md"
    md_path.write_text("# Paper\n", encoding="utf-8")
    doc_id = "paper"
    write_doc_chunks(db_dir, doc_id, [{"text": "paper", "meta": {"source_path": str(md_path)}}])
    save_docs_index(
        db_dir,
        {doc_id: {"doc_id": doc_id, "path": str(md_path), "num_chunks": 1, "index_status": "ready"}},
    )
    finished = threading.Event()
    result: list[dict] = []

    def purge() -> None:
        result.append(library_router._purge_library_index_for_markdown(db_dir, md_path))
        finished.set()

    with db_write_lock(db_dir, timeout_s=2):
        thread = threading.Thread(target=purge)
        thread.start()
        assert not finished.wait(0.2)

    thread.join(10)
    assert finished.is_set()
    assert result[0]["errors"] == []
    assert result[0]["docs_removed"] == 1
    assert result[0]["chunks_removed"] == 1
