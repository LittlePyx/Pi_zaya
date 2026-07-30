import json
import threading
import time
from pathlib import Path

from kb.chat_store import ChatStore


def test_citation_shelf_project_scope_persists_across_conversations(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("paper project")
    conv_a = store.create_conversation("guide a", project_id=project_id)
    conv_b = store.create_conversation("guide b", project_id=project_id)

    saved = store.save_citation_shelf(
        conv_id=conv_a,
        items=[
            {
                "key": "ref-1",
                "main": "High-resolution single-photon imaging",
                "title": "High-resolution single-photon imaging",
                "doi": "10.1038/demo",
                "tags": ["method"],
                "note": "Important upstream method.",
            }
        ],
        open=True,
    )

    assert saved is not None
    assert saved["scope"] == "project"
    assert saved["scope_id"] == project_id
    assert saved["open"] is True
    assert saved["revision"] == 1

    loaded = store.get_citation_shelf(conv_id=conv_b)

    assert loaded is not None
    assert loaded["scope_id"] == project_id
    assert loaded["open"] is True
    assert loaded["items"][0]["key"] == "ref-1"
    assert loaded["items"][0]["tags"] == ["method"]
    assert loaded["items"][0]["note"] == "Important upstream method."


def test_citation_shelf_default_scope_and_delete(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("no project")

    saved = store.save_citation_shelf(
        conv_id=conv_id,
        items=[{"key": "ref-2", "main": "Default shelf item"}],
        open=False,
    )

    assert saved is not None
    assert saved["scope_id"] == "__default__"
    assert saved["items"][0]["main"] == "Default shelf item"

    deleted = store.delete_citation_shelf(conv_id=conv_id)
    assert deleted is not None
    assert deleted["items"] == []
    assert deleted["open"] is False

    loaded = store.get_citation_shelf(conv_id=conv_id)
    assert loaded is not None
    assert loaded["items"] == []
    assert loaded["revision"] == 0


def test_delete_conversation_removes_conversation_scoped_shelf_only(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("conversation shelf")

    conversation_shelf = store.save_citation_shelf(
        conv_id=conv_id,
        scope="conversation",
        items=[{"key": "conv-ref", "main": "Conversation scoped reference"}],
        open=True,
    )
    project_shelf = store.save_citation_shelf(
        conv_id=conv_id,
        items=[{"key": "project-ref", "main": "Default scoped reference"}],
        open=True,
    )

    assert conversation_shelf is not None
    assert conversation_shelf["scope"] == "conversation"
    assert project_shelf is not None
    assert project_shelf["scope"] == "project"

    assert store.delete_conversation(conv_id) is True
    assert store.delete_conversation(conv_id) is False

    with store._connect() as conn:
        leaked_conversation_rows = conn.execute(
            "SELECT COUNT(*) FROM citation_shelves WHERE scope = 'conversation' AND scope_id = ?",
            (conv_id,),
        ).fetchone()[0]
        default_project_rows = conn.execute(
            "SELECT COUNT(*) FROM citation_shelves WHERE scope = 'project' AND scope_id = '__default__'"
        ).fetchone()[0]

    assert leaked_conversation_rows == 0
    assert default_project_rows == 1


def test_citation_shelf_rejects_missing_conversation_scope(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")

    assert store.get_citation_shelf(conv_id="missing") is None
    assert store.save_citation_shelf(conv_id="missing", items=[]) is None


def test_citation_shelf_skips_items_without_identity(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("no project")

    saved = store.save_citation_shelf(
        conv_id=conv_id,
        items=[{}, {"key": "   "}, {"key": "ref-3", "main": "Stable item"}],
        open=True,
    )

    assert saved is not None
    assert len(saved["items"]) == 1
    assert saved["items"][0]["key"] == "ref-3"


def test_citation_shelf_append_duplicate_merges_richer_metadata(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("merge duplicate")

    saved = store.save_citation_shelf(
        conv_id=conv_id,
        items=[
            {
                "key": "ref-merge",
                "main": "Untitled",
                "title": "Untitled",
                "sourcePath": "F:/papers/current.en.md",
                "anchor": "ref-12",
                "tags": ["method"],
                "note": "Keep my manual note.",
            }
        ],
        open=True,
    )
    assert saved is not None

    appended = store.append_citation_shelf_item(
        conv_id=conv_id,
        item={
            "key": "ref-merge",
            "main": "Sparse 3-D transform-domain filtering",
            "title": "Sparse 3-D transform-domain filtering",
            "doi": "10.1109/tip.2007.901238",
            "sourcePath": "F:/papers/current.en.md",
            "anchor": "ref-12",
            "headingPath": "References",
            "evidenceQuote": "This reference is a baseline method for image denoising comparisons.",
            "tags": ["baseline"],
            "note": "Incoming note should not overwrite.",
            "metadata_quality": {"status": "ready", "source": "crossref"},
            "metadataRepairStatus": "ready",
        },
        open=True,
    )

    assert appended is not None
    assert len(appended["items"]) == 1
    item = appended["items"][0]
    assert item["main"] == "Sparse 3-D transform-domain filtering"
    assert item["title"] == "Sparse 3-D transform-domain filtering"
    assert item["doi"] == "10.1109/tip.2007.901238"
    assert item["headingPath"] == "References"
    assert item["shelfExcerpt"] == "This reference is a baseline method for image denoising comparisons."
    assert item["note"] == "Keep my manual note."
    assert item["tags"] == ["method", "baseline"]
    assert item["metadata_quality"]["status"] == "ready"
    assert item["metadataRepairStatus"] == "ready"

    downgraded = store.append_citation_shelf_item(
        conv_id=conv_id,
        item={
            "key": "ref-merge",
            "title": "Sparse 3-D transform-domain filtering",
            "doi": "10.1109/tip.2007.901238",
            "metadata_quality": {"status": "pending", "source": "local"},
            "metadataRepairStatus": "pending",
        },
        open=True,
    )

    assert downgraded is not None
    item = downgraded["items"][0]
    assert item["metadata_quality"]["status"] == "ready"
    assert item["metadata_quality"]["source"] == "crossref"
    assert item["metadataRepairStatus"] == "ready"


def test_citation_shelf_richer_duplicate_clears_resolved_evidence_quality_flags(
    tmp_path: Path,
) -> None:
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("refresh duplicate evidence")
    store.save_citation_shelf(
        conv_id=conv_id,
        items=[
            {
                "key": "cassi-old",
                "title": "CASSI architecture",
                "doi": "10.1364/oe.15.012913",
                "card_quality_flags": ["missing_evidence_quote", "missing_precise_location"],
                "cardView": {
                    "quality": {
                        "flags": ["missing_evidence_quote", "missing_precise_location"]
                    }
                },
            }
        ],
        open=True,
    )

    refreshed = store.append_citation_shelf_item(
        conv_id=conv_id,
        item={
            "key": "cassi-new",
            "title": "CASSI architecture",
            "doi": "10.1364/oe.15.012913",
            "evidenceQuote": (
                "Two dispersive elements are arranged in opposition around a "
                "binary-valued aperture."
            ),
            "blockId": "blk_cassi_abstract",
            "headingPath": "Abstract",
            "pageStart": 1,
        },
        open=True,
    )

    assert refreshed is not None
    assert len(refreshed["items"]) == 1
    item = refreshed["items"][0]
    assert item["evidenceQuote"].startswith("Two dispersive elements")
    assert not item.get("card_quality_flags")
    card_view = item.get("cardView") if isinstance(item.get("cardView"), dict) else {}
    quality = card_view.get("quality") if isinstance(card_view.get("quality"), dict) else {}
    assert not quality.get("flags")

def test_citation_shelf_save_sanitizes_large_nested_payload(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("large payload")
    long_text = "x" * 8000

    saved = store.save_citation_shelf(
        conv_id=conv_id,
        items=[
            {
                "key": "ref-large",
                "main": "Large payload reference",
                "doi": "10.1000/large-payload",
                "debugBlob": long_text,
                "nested": {
                    "quote": long_text,
                    "badFloat": float("nan"),
                    "values": [long_text for _ in range(50)],
                    "empty": "",
                },
                "tags": ["a" * 120, "", None],
                "note": "n" * 5000,
            }
        ],
        open=True,
    )

    assert saved is not None
    item = saved["items"][0]
    assert len(item["debugBlob"]) <= 700
    assert len(item["nested"]["quote"]) <= 700
    assert "badFloat" not in item["nested"]
    assert len(item["nested"]["values"]) == 32
    assert len(item["nested"]["values"][0]) <= 700
    assert item["tags"] == ["a" * 60]
    assert len(item["note"]) == 4000


def test_citation_shelf_get_sanitizes_legacy_oversized_payload(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("legacy payload")
    now = time.time()
    long_text = "y" * 8000
    legacy_items = [
        {
            "key": "legacy-ref",
            "main": "Legacy reference",
            "doi": "10.1000/legacy-payload",
            "debugBlob": long_text,
            "nested": {"quote": long_text, "badFloat": float("inf")},
            "tags": ["b" * 120],
            "note": "m" * 5000,
        }
    ]
    with store._connect() as conn:
        conn.execute(
            """
            INSERT INTO citation_shelves (scope, scope_id, items_json, open, revision, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("project", "__default__", json.dumps(legacy_items), 1, 7, now, now),
        )

    loaded = store.get_citation_shelf(conv_id=conv_id)

    assert loaded is not None
    assert loaded["revision"] == 7
    item = loaded["items"][0]
    assert len(item["debugBlob"]) <= 700
    assert len(item["nested"]["quote"]) <= 700
    assert "badFloat" not in item["nested"]
    assert item["tags"] == ["b" * 60]
    assert len(item["note"]) == 4000


def test_citation_shelf_concurrent_appends_keep_all_items(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("concurrent papers")
    conv_id = store.create_conversation("basket", project_id=project_id)
    worker_count = 10
    start = threading.Barrier(worker_count + 1)
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def append_one(index: int) -> None:
        try:
            start.wait(timeout=5)
            record = store.append_citation_shelf_item(
                conv_id=conv_id,
                item={
                    "key": f"ref-concurrent-{index}",
                    "main": f"Concurrent reference {index}",
                    "doi": f"10.1000/concurrent-{index}",
                },
                open=True,
            )
            if record is None:
                raise AssertionError("citation shelf append unexpectedly returned None")
        except BaseException as exc:
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=append_one, args=(idx,), daemon=True) for idx in range(worker_count)]
    for thread in threads:
        thread.start()
    start.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=10)

    assert not [thread for thread in threads if thread.is_alive()]
    assert errors == []

    loaded = store.get_citation_shelf(conv_id=conv_id)
    assert loaded is not None
    assert loaded["revision"] == worker_count
    assert {item["doi"] for item in loaded["items"]} == {
        f"10.1000/concurrent-{idx}" for idx in range(worker_count)
    }
