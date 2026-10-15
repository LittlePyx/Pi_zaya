from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from kb.chat_store import ChatStore
from kb.citation_shelf_concurrency import CitationShelfConflict


def item(key):
    return {"key": key, "main": key, "title": key}


def operation(key, *, clear=False, remove=()):
    return {"id": key, "upserts": [] if clear or remove else [item(key)],
            "clear": clear, "removedKeys": list(remove), "order": []}


def write(store, revision, *ops):
    return store.save_citation_shelf(items=[], expected_revision=revision, mutation_id=ops[-1]["id"],
                                     mutations=list(ops), allow_empty_overwrite=True)


def keys(record):
    return {entry["key"] for entry in record["items"]}


def test_stale_snapshot_cannot_erase_concurrent_addition(tmp_path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    first = write(store, 0, operation("A"))
    write(store, first["revision"], operation("B"))
    with pytest.raises(CitationShelfConflict):
        write(store, first["revision"], operation("C"))
    fresh = store.get_citation_shelf()
    assert keys(write(store, fresh["revision"], operation("C"))) == {"A", "B", "C"}


def test_lost_response_receipts_survive_restart_and_do_not_resurrect_after_clear(tmp_path):
    path = tmp_path / "chat.sqlite3"
    store = ChatStore(path)
    old = write(store, 0, operation("A"))
    cleared = write(store, old["revision"], operation("clear", clear=True))
    later = write(store, cleared["revision"], operation("B"))
    reopened = ChatStore(path)
    repeated = write(reopened, 0, operation("A"))
    assert keys(repeated) == {"B"}
    assert repeated["revision"] == later["revision"]
    assert repeated["acknowledged_mutation_id"] == "A"
    replay = write(reopened, later["revision"], operation("A"), operation("C"))
    assert keys(replay) == {"B", "C"}
    assert reopened.get_citation_shelf()["concurrency_guarded"]


def test_delete_retry_does_not_remove_a_later_readd(tmp_path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    first = write(store, 0, operation("A"))
    deleted = write(store, first["revision"], operation("delete", remove=["A"]))
    readd = {**operation("A"), "id": "readd"}
    later = write(store, deleted["revision"], readd)
    assert keys(write(store, first["revision"], operation("delete", remove=["A"]))) == {"A"}
    assert store.get_citation_shelf()["revision"] == later["revision"]


def test_guarded_scope_rejects_legacy_replace_delete_but_accepts_atomic_append(tmp_path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    write(store, 0, operation("A"))
    with pytest.raises(CitationShelfConflict):
        store.save_citation_shelf(items=[item("B")])
    with pytest.raises(CitationShelfConflict):
        store.delete_citation_shelf()
    assert keys(store.append_citation_shelf_item(item=item("B"))) == {"A", "B"}


def test_two_database_connections_cannot_both_commit_same_revision(tmp_path):
    path = tmp_path / "chat.sqlite3"
    stores = [ChatStore(path), ChatStore(path)]
    barrier = Barrier(2)
    def attempt(index):
        barrier.wait()
        try:
            write(stores[index], 0, operation(str(index)))
            return "saved"
        except CitationShelfConflict:
            return "conflict"
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(attempt, range(2))) == ["conflict", "saved"]
    assert len(stores[0].get_citation_shelf()["items"]) == 1


def test_receipts_are_scope_specific(tmp_path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    a, b = store.create_project("A"), store.create_project("B")
    for project in (a, b):
        result = store.save_citation_shelf(project_id=project, items=[], expected_revision=0,
            mutation_id="same-id", mutations=[operation("same-id")], allow_empty_overwrite=True)
        assert keys(result) == {"same-id"}


def test_operations_require_revision_and_receipt_id(tmp_path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    with pytest.raises(CitationShelfConflict):
        store.save_citation_shelf(items=[], mutations=[operation("A")])


def test_receipts_are_removed_only_when_their_scope_is_deleted(tmp_path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    project = store.create_project("A")
    store.save_citation_shelf(project_id=project, items=[item("A")], expected_revision=0, mutation_id="A")
    assert store.delete_project(project)
    with store._connect() as conn:
        assert conn.execute("SELECT count(*) FROM citation_shelf_write_receipts WHERE scope_id=?", (project,)).fetchone()[0] == 0
