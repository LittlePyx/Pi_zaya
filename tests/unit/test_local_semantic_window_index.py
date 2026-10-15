from __future__ import annotations

from pathlib import Path

import pytest

from kb.e5_small_embedding_runtime import E5EmbeddingCancelled, E5ParentWindow
from kb.local_semantic_window_index import (
    LocalSemanticIndexError,
    LocalSemanticWindowIndexStore,
)
from kb.store import save_docs_index, write_doc_chunks


def _vector(axis: int) -> tuple[float, ...]:
    values = [0.0] * 384
    values[axis] = 1.0
    return tuple(values)


def _publish(db_dir: Path, documents: dict[str, list[str]]) -> None:
    docs = {}
    for doc_id, texts in documents.items():
        write_doc_chunks(
            db_dir,
            doc_id,
            [
                {
                    "text": text,
                    "meta": {
                        "source_path": f"C:/private/{doc_id}.pdf",
                        "evidence_ready": True,
                    },
                }
                for text in texts
            ],
        )
        docs[doc_id] = {
            "doc_id": doc_id,
            "path": f"C:/private/{doc_id}.md",
            "num_chunks": len(texts),
            "index_status": "ready",
        }
    save_docs_index(db_dir, docs)


class _Executor:
    def __init__(self) -> None:
        self.embedded_parent_ids: list[str] = []
        self.cancel_after_embed = None
        self.change_after_embed = None

    def build_windows(self, chunks, *, should_cancel=lambda: False):
        windows = []
        for position, chunk in enumerate(chunks):
            if should_cancel():
                raise E5EmbeddingCancelled()
            count = 2 if "long" in str(chunk.get("text") or "") else 1
            for ordinal in range(count):
                windows.append(
                    E5ParentWindow(
                        parent_id=str(chunk["id"]),
                        parent_position=position,
                        window_ordinal=ordinal,
                        input_ids=(0, position + 3, ordinal + 20, 2),
                    )
                )
        return windows

    def embed_windows(self, windows, *, should_cancel=lambda: False, on_progress=lambda _x: None):
        vectors = []
        for window in windows:
            if should_cancel():
                raise E5EmbeddingCancelled()
            self.embedded_parent_ids.append(window.parent_id)
            vectors.append(_vector(window.parent_position % 3))
        if self.change_after_embed:
            self.change_after_embed()
        if self.cancel_after_embed:
            self.cancel_after_embed()
        return vectors

    def embed_query(self, text: str):
        return _vector(1 if "beta" in text else 0)


def test_atomic_build_load_search_noop_and_text_free_manifest(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    _publish(db_dir, {"a": ["alpha"], "b": ["beta long"]})
    executor = _Executor()
    store = LocalSemanticWindowIndexStore(db_dir)

    built = store.build(executor)  # type: ignore[arg-type]
    loaded = store.load(executor)  # type: ignore[arg-type]
    hit = loaded.as_retriever(executor).search("beta", top_k=1)  # type: ignore[arg-type]
    repeated = store.build(executor)  # type: ignore[arg-type]

    assert built.parent_count == 2 and built.window_count == 3
    assert built.embedded_parent_count == 2 and built.embedded_window_count == 3
    assert hit[0]["id"] == "b:0"
    assert repeated.no_op is True and repeated.build_id == built.build_id
    assert store.status().state == "ready"
    generation = store.generations_dir / built.build_id
    text = (generation / "manifest.json").read_text(encoding="utf-8")
    assert "C:/private" not in text
    assert "alpha" not in text and "beta long" not in text


def test_incremental_rebuild_reuses_unchanged_parent_windows(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    _publish(db_dir, {"a": ["alpha"], "b": ["beta long"]})
    store = LocalSemanticWindowIndexStore(db_dir)
    first_executor = _Executor()
    first = store.build(first_executor)  # type: ignore[arg-type]
    _publish(db_dir, {"a": ["alpha changed"], "b": ["beta long"]})
    second_executor = _Executor()
    second = store.build(second_executor)  # type: ignore[arg-type]

    assert second.build_id != first.build_id
    assert second.reused_parent_count == 1 and second.reused_window_count == 2
    assert second.embedded_parent_count == 1 and second.embedded_window_count == 1
    assert second_executor.embedded_parent_ids == ["a:0"]


def test_cancel_after_embedding_preserves_previous_pointer(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    _publish(db_dir, {"a": ["alpha"]})
    store = LocalSemanticWindowIndexStore(db_dir)
    first = store.build(_Executor())  # type: ignore[arg-type]
    pointer = store.current_path.read_bytes()
    _publish(db_dir, {"a": ["changed"]})
    cancel = {"value": False}
    executor = _Executor()
    executor.cancel_after_embed = lambda: cancel.update(value=True)

    with pytest.raises(LocalSemanticIndexError) as captured:
        store.build(executor, should_cancel=lambda: cancel["value"])  # type: ignore[arg-type]

    assert captured.value.code == "cancelled_by_user"
    assert store.current_path.read_bytes() == pointer
    _publish(db_dir, {"a": ["alpha"]})
    assert store.load(_Executor()).manifest["build_id"] == first.build_id  # type: ignore[arg-type]


def test_corpus_change_during_embedding_aborts_before_commit(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    _publish(db_dir, {"a": ["alpha"]})
    store = LocalSemanticWindowIndexStore(db_dir)
    store.build(_Executor())  # type: ignore[arg-type]
    pointer = store.current_path.read_bytes()
    _publish(db_dir, {"a": ["beta"]})
    executor = _Executor()
    executor.change_after_embed = lambda: _publish(db_dir, {"a": ["third"]})

    with pytest.raises(LocalSemanticIndexError) as captured:
        store.build(executor)  # type: ignore[arg-type]

    assert captured.value.code == "corpus_changed"
    assert store.current_path.read_bytes() == pointer


def test_pointer_commit_failure_removes_orphan_and_preserves_previous(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    _publish(db_dir, {"a": ["alpha"]})
    store = LocalSemanticWindowIndexStore(db_dir)
    first = store.build(_Executor())  # type: ignore[arg-type]
    pointer = store.current_path.read_bytes()
    _publish(db_dir, {"a": ["beta"]})

    def fail(_generation: Path) -> None:
        raise OSError("private commit detail")

    with pytest.raises(LocalSemanticIndexError) as captured:
        store.build(_Executor(), _before_commit=fail)  # type: ignore[arg-type]

    assert captured.value.code == "commit_failed"
    assert store.current_path.read_bytes() == pointer
    assert [path.name for path in store.generations_dir.iterdir()] == [first.build_id]


def test_cancel_at_commit_checkpoint_removes_orphan_and_preserves_previous(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    _publish(db_dir, {"a": ["alpha"]})
    store = LocalSemanticWindowIndexStore(db_dir)
    first = store.build(_Executor())  # type: ignore[arg-type]
    pointer = store.current_path.read_bytes()
    _publish(db_dir, {"a": ["beta"]})
    cancel = {"value": False}

    with pytest.raises(LocalSemanticIndexError) as captured:
        store.build(
            _Executor(),  # type: ignore[arg-type]
            should_cancel=lambda: cancel["value"],
            _before_commit=lambda _generation: cancel.update(value=True),
        )

    assert captured.value.code == "cancelled_by_user"
    assert store.current_path.read_bytes() == pointer
    assert [path.name for path in store.generations_dir.iterdir()] == [first.build_id]


def test_corrupt_vectors_and_stale_corpus_are_not_ready(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    _publish(db_dir, {"a": ["alpha"]})
    store = LocalSemanticWindowIndexStore(db_dir)
    built = store.build(_Executor())  # type: ignore[arg-type]
    _publish(db_dir, {"a": ["changed"]})
    assert store.status().state == "stale"
    _publish(db_dir, {"a": ["alpha"]})
    vectors = store.generations_dir / built.build_id / "vectors.f32"
    vectors.write_bytes(vectors.read_bytes()[:-1])
    assert store.status().state == "corrupt"
