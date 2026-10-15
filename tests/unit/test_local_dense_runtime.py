from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from kb.local_dense_runtime import (
    LOCAL_RETRIEVAL_MODES,
    QWEN_PROVIDER_IDENTITY,
    LocalDenseRuntimeService,
    build_qwen_product_provider,
    normalize_retrieval_mode,
    product_dense_index_dir,
)
from kb.retrieval_shadow import QWEN_PROVIDER_IDENTITY as SHADOW_PROVIDER_IDENTITY


@dataclass
class _Status:
    state: str = "ready"
    provider_identity: str = QWEN_PROVIDER_IDENTITY
    document_count: int = 3
    chunk_count: int = 7


class _Retriever:
    is_empty = False

    def search(self, _query: str, top_k: int = 6) -> list[dict]:
        return [{"id": "dense"}][:top_k]


class _Loaded:
    def as_retriever(self, _provider):
        return _Retriever()


class _Store:
    def __init__(self, status: _Status | None = None) -> None:
        self._status = status or _Status()
        self.load_calls = 0

    def status(self, _provider) -> _Status:
        return self._status

    def load(self, _provider) -> _Loaded:
        self.load_calls += 1
        return _Loaded()


def _service(store: _Store, **overrides) -> LocalDenseRuntimeService:
    true = lambda: True
    return LocalDenseRuntimeService(
        dependency_probe=overrides.get("dependency_probe", true),
        accelerator_probe=overrides.get("accelerator_probe", true),
        model_cache_probe=overrides.get("model_cache_probe", true),
        vector_backend_probe=overrides.get("vector_backend_probe", true),
        provider_factory=build_qwen_product_provider,
        index_store_factory=lambda _db, _index: store,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "bm25"),
        ("", "bm25"),
        ("unknown", "bm25"),
        ("BM25", "bm25"),
        ("hybrid-local", "hybrid_local"),
    ],
)
def test_retrieval_mode_is_closed_and_defaults_to_bm25(raw, expected) -> None:
    assert normalize_retrieval_mode(raw) == expected
    assert LOCAL_RETRIEVAL_MODES == ("bm25", "hybrid_local")


def test_product_provider_matches_accepted_identity_and_is_cache_only() -> None:
    provider = build_qwen_product_provider()
    assert provider.descriptor.identity == QWEN_PROVIDER_IDENTITY
    assert provider.descriptor.identity == SHADOW_PROVIDER_IDENTITY
    assert provider.allow_download is False
    assert provider.requested_device == "cuda"


def test_product_index_location_never_uses_test_artifacts(tmp_path: Path) -> None:
    assert product_dense_index_dir(tmp_path) == tmp_path.resolve() / "dense_index"


@pytest.mark.parametrize(
    ("probe", "reason"),
    [
        ("dependency_probe", "runtime_dependencies_missing"),
        ("vector_backend_probe", "vector_backend_unavailable"),
        ("accelerator_probe", "local_accelerator_unavailable"),
        ("model_cache_probe", "pinned_model_not_cached"),
    ],
)
def test_readiness_redacts_unavailable_runtime_details(tmp_path: Path, probe: str, reason: str) -> None:
    readiness = _service(_Store(), **{probe: lambda: False}).readiness(tmp_path)
    assert readiness.public_payload() == {
        "state": "unavailable",
        "reason_id": reason,
        "available": False,
        "mode": "hybrid_local",
        "document_count": 0,
        "chunk_count": 0,
    }
    assert str(tmp_path) not in str(readiness.public_payload())


@pytest.mark.parametrize("state", ["missing", "stale", "provider_mismatch", "corrupt"])
def test_readiness_requires_current_matching_index(tmp_path: Path, state: str) -> None:
    readiness = _service(_Store(_Status(state=state))).readiness(tmp_path)
    assert readiness.available is False
    assert readiness.reason_id == f"dense_index_{state}"


def test_ready_payload_contains_counts_but_no_paths(tmp_path: Path) -> None:
    readiness = _service(_Store()).readiness(tmp_path)
    assert readiness.public_payload() == {
        "state": "ready",
        "reason_id": "ready",
        "available": True,
        "mode": "hybrid_local",
        "document_count": 3,
        "chunk_count": 7,
    }


def test_runtime_cache_invalidates_when_index_pointer_changes(tmp_path: Path) -> None:
    index_dir = product_dense_index_dir(tmp_path)
    index_dir.mkdir()
    pointer = index_dir / "current.json"
    pointer.write_text("one", encoding="utf-8")
    store = _Store()
    service = _service(store)

    first = service.load(tmp_path)
    second = service.load(tmp_path)
    pointer.write_text("a different pointer", encoding="utf-8")
    third = service.load(tmp_path)

    assert first.cold_start is True and first.index_cache_hit is False
    assert second.cold_start is False and second.index_cache_hit is True
    assert third.cold_start is True and third.index_cache_hit is False
    assert store.load_calls == 2


def test_load_ready_checks_prerequisites_before_opening_index(tmp_path: Path) -> None:
    store = _Store()
    service = _service(store, model_cache_probe=lambda: False)
    with pytest.raises(Exception) as exc_info:
        service.load_ready(tmp_path)
    assert getattr(exc_info.value, "reason_id", "") == "pinned_model_not_cached"
    assert store.load_calls == 0
