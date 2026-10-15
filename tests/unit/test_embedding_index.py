from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from kb.embedding_index import (
    EmbeddingIndexBuildError,
    EmbeddingIndexError,
    EmbeddingIndexStore,
)
from kb.embedding_provider import (
    CallableEmbeddingProvider,
    EmbeddingProviderDescriptor,
    LocalEmbeddingProvider,
)
from kb.store import save_docs_index, write_doc_chunks


class RecordingProvider:
    def __init__(self, *, fingerprint: str = "weights-sha256:test-v1") -> None:
        self.descriptor = EmbeddingProviderDescriptor(
            kind="local",
            model_id="test/multilingual-embedding",
            revision="1",
            model_fingerprint=fingerprint,
            dimension=3,
        )
        self.document_calls: list[list[str]] = []
        self.fail_on = ""

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        if "alpha" in lowered:
            return [1.0, 0.0, 0.0]
        if "beta" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        rows = [str(text) for text in texts]
        self.document_calls.append(rows)
        if self.fail_on and any(self.fail_on in text for text in rows):
            raise RuntimeError("simulated provider outage with private details")
        return [self._vector(text) for text in rows]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def _publish_documents(
    db_dir: Path,
    documents: dict[str, list[str]],
    *,
    path_prefix: str = "C:/private/library",
) -> None:
    docs: dict[str, dict] = {}
    for doc_id, texts in documents.items():
        write_doc_chunks(
            db_dir,
            doc_id,
            [
                {
                    "text": text,
                    "meta": {
                        "source_path": f"{path_prefix}/{doc_id}.pdf",
                        "page_start": index + 1,
                        "page_end": index + 1,
                        "evidence_ready": True,
                    },
                }
                for index, text in enumerate(texts)
            ],
        )
        docs[doc_id] = {
            "doc_id": doc_id,
            "path": f"{path_prefix}/{doc_id}.md",
            "num_chunks": len(texts),
            "index_status": "ready",
        }
    save_docs_index(db_dir, docs)


def test_provider_descriptor_and_callable_adapter_validate_shape() -> None:
    descriptor = EmbeddingProviderDescriptor(
        kind="local",
        model_id="local/test",
        revision="r1",
        model_fingerprint="sha256:abc",
        dimension=2,
    )
    provider = CallableEmbeddingProvider(
        descriptor,
        lambda texts: [[3.0, 4.0] for _text in texts],
        lambda _text: [0.0, 2.0],
    )

    assert provider.descriptor.identity == descriptor.identity
    assert provider.embed_documents(["a", "b"]) == [(0.6, 0.8), (0.6, 0.8)]
    assert provider.embed_query("question") == (0.0, 1.0)

    broken = CallableEmbeddingProvider(
        descriptor,
        lambda _texts: [[1.0, 0.0, 0.0]],
        lambda _text: [1.0, 0.0],
    )
    with pytest.raises(ValueError, match="dimension mismatch"):
        broken.embed_documents(["a"])

    local = LocalEmbeddingProvider(
        model_id="local/test",
        revision="r1",
        model_fingerprint="sha256:abc",
        dimension=2,
        embed_documents=lambda texts: [[1.0, 0.0] for _text in texts],
        embed_query=lambda _text: [1.0, 0.0],
    )
    assert local.descriptor.kind == "local"


def test_atomic_index_build_load_search_and_noop(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    _publish_documents(
        db_dir,
        {"paper-a": ["alpha finding"], "paper-b": ["beta evidence", "context"]},
    )
    provider = RecordingProvider()
    store = EmbeddingIndexStore(db_dir)

    built = store.build(provider)
    loaded = store.load(provider)
    hits = loaded.as_retriever(provider).search("beta question", top_k=1)
    repeated = store.build(provider)

    assert built.document_count == 2
    assert built.chunk_count == 3
    assert built.embedded_document_count == 2
    assert built.reused_document_count == 0
    assert hits[0]["id"] == "paper-b:0"
    assert store.status(provider).state == "ready"
    assert repeated.no_op is True
    assert repeated.build_id == built.build_id
    assert len(provider.document_calls) == 2

    generation = store.index_dir / "generations" / built.build_id
    manifest_text = (generation / "manifest.json").read_text(encoding="utf-8")
    assert "C:/private/library" not in manifest_text
    assert "alpha finding" not in manifest_text
    assert json.loads(manifest_text)["chunks"][0]["chunk_id"] == "paper-a:0"


def test_incremental_build_only_embeds_text_changed_document(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    provider = RecordingProvider()
    store = EmbeddingIndexStore(db_dir)
    _publish_documents(
        db_dir,
        {"paper-a": ["alpha original"], "paper-b": ["beta unchanged", "context"]},
    )
    first = store.build(provider)
    provider.document_calls.clear()

    _publish_documents(
        db_dir,
        {"paper-a": ["alpha changed"], "paper-b": ["beta unchanged", "context"]},
    )
    second = store.build(provider)

    assert second.build_id != first.build_id
    assert second.embedded_document_count == 1
    assert second.embedded_chunk_count == 1
    assert second.reused_document_count == 1
    assert second.reused_chunk_count == 2
    assert provider.document_calls == [["alpha changed"]]
    assert store.load(provider).manifest["documents"][1]["status"] == "ready"


def test_path_change_reuses_all_embeddings_but_commits_new_docs_signature(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    provider = RecordingProvider()
    store = EmbeddingIndexStore(db_dir)
    _publish_documents(
        db_dir,
        {"old-paper-a": ["alpha finding"], "old-paper-b": ["beta evidence"]},
        path_prefix="D:/first/location",
    )
    first = store.build(provider)
    provider.document_calls.clear()

    _publish_documents(
        db_dir,
        {"new-paper-a": ["alpha finding"], "new-paper-b": ["beta evidence"]},
        path_prefix="E:/moved/library",
    )
    assert store.status(provider).state == "stale"
    second = store.build(provider)

    assert second.build_id != first.build_id
    assert second.embedded_document_count == 0
    assert second.reused_document_count == 2
    assert provider.document_calls == []
    assert store.status(provider).state == "ready"
    manifest_path = store.index_dir / "generations" / second.build_id / "manifest.json"
    assert "E:/moved/library" not in manifest_path.read_text(encoding="utf-8")


def test_provider_failure_preserves_previous_commit_and_records_safe_error(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    provider = RecordingProvider()
    store = EmbeddingIndexStore(db_dir)
    _publish_documents(db_dir, {"paper-a": ["alpha original"]})
    first = store.build(provider)
    old_pointer = store.current_path.read_bytes()

    _publish_documents(db_dir, {"paper-a": ["private changed content"]})
    provider.fail_on = "private"
    with pytest.raises(EmbeddingIndexBuildError) as captured:
        store.build(provider)

    assert captured.value.code == "provider_failed"
    assert store.current_path.read_bytes() == old_pointer
    status = store.status(provider)
    assert status.state == "stale"
    assert status.last_build_state == "failed"
    assert status.last_error_category == "RuntimeError"
    last_build = store.build_status_path.read_text(encoding="utf-8")
    assert "private changed content" not in last_build
    assert "private details" not in last_build

    _publish_documents(db_dir, {"paper-a": ["alpha original"]})
    assert store.load(provider).manifest["build_id"] == first.build_id


def test_failed_atomic_pointer_commit_never_publishes_partial_generation(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    provider = RecordingProvider()
    store = EmbeddingIndexStore(db_dir)
    _publish_documents(db_dir, {"paper-a": ["alpha original"]})
    first = store.build(provider)
    old_pointer = store.current_path.read_bytes()

    _publish_documents(db_dir, {"paper-a": ["beta replacement"]})

    def fail_before_commit(_generation: Path) -> None:
        raise OSError("simulated current pointer failure")

    with pytest.raises(EmbeddingIndexBuildError) as captured:
        store.build(provider, _before_commit=fail_before_commit)

    assert captured.value.code == "commit_failed"
    assert store.current_path.read_bytes() == old_pointer
    assert store.status(provider).last_error_category == "commit_failed"
    _publish_documents(db_dir, {"paper-a": ["alpha original"]})
    assert store.load(provider).manifest["build_id"] == first.build_id


def test_corpus_change_during_provider_call_aborts_before_commit(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    provider = RecordingProvider()
    store = EmbeddingIndexStore(db_dir)
    _publish_documents(db_dir, {"paper-a": ["alpha original"]})
    first = store.build(provider)
    old_pointer = store.current_path.read_bytes()

    _publish_documents(db_dir, {"paper-a": ["beta replacement"]})
    original_embed = provider.embed_documents

    def embed_then_change_corpus(texts: Sequence[str]) -> list[list[float]]:
        vectors = original_embed(texts)
        _publish_documents(db_dir, {"paper-a": ["concurrent third version"]})
        return vectors

    provider.embed_documents = embed_then_change_corpus  # type: ignore[method-assign]
    with pytest.raises(EmbeddingIndexBuildError) as captured:
        store.build(provider)

    assert captured.value.code == "corpus_changed"
    assert store.current_path.read_bytes() == old_pointer
    status = store.status(provider)
    assert status.state == "stale"
    assert status.last_error_category == "corpus_changed"

    _publish_documents(db_dir, {"paper-a": ["alpha original"]})
    assert store.load(provider).manifest["build_id"] == first.build_id


def test_mismatched_provider_and_corrupt_vector_are_unavailable(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    provider = RecordingProvider()
    store = EmbeddingIndexStore(db_dir)
    _publish_documents(db_dir, {"paper-a": ["alpha finding"]})
    built = store.build(provider)

    other_provider = RecordingProvider(fingerprint="weights-sha256:different")
    assert store.status(other_provider).state == "provider_mismatch"
    with pytest.raises(EmbeddingIndexError) as mismatch:
        store.load(other_provider)
    assert mismatch.value.code == "provider_mismatch"

    vectors_path = store.index_dir / "generations" / built.build_id / "vectors.f32"
    vectors_path.write_bytes(vectors_path.read_bytes()[:-1])
    assert store.status(provider).state == "corrupt"
    with pytest.raises(EmbeddingIndexError) as corrupt:
        store.load(provider)
    assert corrupt.value.code == "corrupt_vectors"
