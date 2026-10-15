from __future__ import annotations

import copy
from pathlib import Path

import pytest

import kb.local_retrieval_selector as selector_module
from kb.local_retrieval_selector import select_product_retrieval
from kb.local_semantic_runtime import LocalSemanticRuntime, LocalSemanticUnavailable


def _hit(number: int, *, source: str = "doc-a") -> dict:
    return {
        "id": f"{source}:chunk-{number}",
        "score": float(100 - number),
        "text": f"evidence {number}",
        "meta": {"source_path": f"C:/private/{source}.md"},
    }


class _Dense:
    def __init__(self, hits: list[dict], *, fail: bool = False) -> None:
        self._hits = hits
        self._fail = fail

    @property
    def is_empty(self) -> bool:
        return not self._hits

    def search(self, _query: str, top_k: int = 6) -> list[dict]:
        if self._fail:
            raise RuntimeError("private dense failure")
        return copy.deepcopy(self._hits[:top_k])

    def search_scoped(self, query: str, *, source_paths, top_k: int = 6) -> list[dict]:
        if self._fail:
            raise RuntimeError("private dense failure")
        allowed = {str(path).replace("\\", "/").casefold() for path in source_paths}
        return copy.deepcopy([
            hit for hit in self._hits
            if str(hit["meta"]["source_path"]).replace("\\", "/").casefold() in allowed
        ][:top_k])


class _Service:
    def __init__(self, dense: _Dense | None = None, *, reason: str = "") -> None:
        self._dense = dense or _Dense([])
        self._reason = reason

    def load_ready(self, _db_dir: Path) -> LocalSemanticRuntime:
        if self._reason:
            raise LocalSemanticUnavailable(self._reason)
        return LocalSemanticRuntime(self._dense, index_cache_hit=True, cold_start=False)


def _ids(hits: list[dict]) -> list[str]:
    return [str(hit["id"]) for hit in hits]


@pytest.mark.parametrize("mode", [None, "", "unknown", "BM25"])
def test_default_and_unknown_modes_return_exact_bm25_values(mode, tmp_path: Path) -> None:
    sparse = [_hit(index) for index in range(1, 13)]
    original = copy.deepcopy(sparse)
    result = select_product_retrieval(
        query="private question",
        sparse_hits=sparse,
        db_dir=tmp_path,
        retrieval_mode=mode,
        source_scope="library",
        runtime_service=_Service(_Dense([_hit(99, source="dense")])),
        product_enabled=True,
    )
    assert result.hits == original
    assert result.used_mode == "bm25"
    assert result.fallback_reason_id == ""
    assert sparse == original


def test_hybrid_preserves_prefix_adds_two_dense_hits_and_appends_sparse_remainder(tmp_path: Path) -> None:
    sparse = [_hit(index) for index in range(1, 16)]
    original = copy.deepcopy(sparse)
    dense = _Dense([_hit(7), _hit(50, source="doc-b"), _hit(51, source="doc-c")])

    result = select_product_retrieval(
        query="semantic paraphrase",
        sparse_hits=sparse,
        db_dir=tmp_path,
        retrieval_mode="hybrid_local",
        source_scope="library",
        runtime_service=_Service(dense),
        product_enabled=True,
    )

    assert _ids(result.hits[:5]) == _ids(original[:5])
    assert {"doc-b:chunk-50", "doc-c:chunk-51"}.issubset(_ids(result.hits[:10]))
    assert len(result.hits) == len(original)
    assert len(_ids(result.hits)) == len(set(_ids(result.hits)))
    assert result.used_mode == "hybrid_local"
    assert result.dense_novelty_count == 2
    assert sparse == original


def test_scoped_hybrid_cannot_admit_another_source(tmp_path: Path) -> None:
    sparse = [_hit(index) for index in range(1, 11)]
    dense = _Dense([_hit(90, source="other"), _hit(8), _hit(91, source="other")])

    result = select_product_retrieval(
        query="current paper only",
        sparse_hits=sparse,
        db_dir=tmp_path,
        retrieval_mode="hybrid_local",
        source_scope="current_paper",
        source_paths=["C:/private/doc-a.md"],
        runtime_service=_Service(dense),
        product_enabled=True,
    )

    assert all(str(hit["id"]).startswith("doc-a:") for hit in result.hits)
    assert result.used_mode == "hybrid_local"


@pytest.mark.parametrize(
    "reason",
    [
        "runtime_dependencies_missing",
        "local_semantic_artifacts_not_ready",
        "local_semantic_index_missing",
        "local_semantic_index_stale",
    ],
)
def test_unavailable_runtime_returns_untouched_bm25(reason: str, tmp_path: Path) -> None:
    sparse = [_hit(index) for index in range(1, 12)]
    result = select_product_retrieval(
        query="question with C:/private/path",
        sparse_hits=sparse,
        db_dir=tmp_path,
        retrieval_mode="hybrid_local",
        source_scope="library",
        runtime_service=_Service(reason=reason),
        product_enabled=True,
    )
    assert result.hits == sparse
    assert result.used_mode == "bm25"
    assert result.fallback_reason_id == reason
    assert "private" not in str(result.trace_payload())


def test_dense_query_failure_returns_untouched_bm25(tmp_path: Path) -> None:
    sparse = [_hit(index) for index in range(1, 12)]
    result = select_product_retrieval(
        query="private query",
        sparse_hits=sparse,
        db_dir=tmp_path,
        retrieval_mode="hybrid_local",
        source_scope="library",
        runtime_service=_Service(_Dense([], fail=True)),
        product_enabled=True,
    )
    assert result.hits == sparse
    assert result.used_mode == "bm25"
    assert result.fallback_reason_id == "dense_query_failed"
    assert all("_fusion_slot" not in hit for hit in result.hits)


def test_unpromoted_manual_hybrid_request_is_exact_bm25(tmp_path: Path) -> None:
    sparse = [_hit(index) for index in range(1, 12)]
    result = select_product_retrieval(
        query="manual hidden request",
        sparse_hits=sparse,
        db_dir=tmp_path,
        retrieval_mode="hybrid_local",
        source_scope="library",
        runtime_service=_Service(_Dense([_hit(99, source="dense")])),
        product_enabled=False,
    )
    assert result.hits == sparse
    assert result.used_mode == "bm25"
    assert result.fallback_reason_id == "local_semantic_not_promoted"


def test_promoted_default_loader_uses_ready_component_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sparse = [_hit(index) for index in range(1, 12)]
    dense = _Dense([_hit(90, source="other"), _hit(8)])
    calls: list[Path] = []

    def load_ready(db_dir: Path):
        calls.append(db_dir)
        return LocalSemanticRuntime(dense, index_cache_hit=True, cold_start=False)

    monkeypatch.setattr(
        selector_module,
        "load_local_semantic_component_runtime",
        load_ready,
    )

    result = select_product_retrieval(
        query="selected paper paraphrase",
        sparse_hits=sparse,
        db_dir=tmp_path,
        retrieval_mode="hybrid_local",
        source_scope="current_paper",
        source_paths=["C:/private/doc-a.md"],
        product_enabled=True,
    )

    assert calls == [tmp_path.resolve()]
    assert result.used_mode == "hybrid_local"
    assert all(str(hit["id"]).startswith("doc-a:") for hit in result.hits)


def test_component_runtime_failure_is_redacted_and_fails_open(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sparse = [_hit(index) for index in range(1, 12)]

    def fail(_db_dir: Path):
        raise selector_module.PortableLocalSemanticRuntimeError(
            "component_lifecycle_not_ready"
        )

    monkeypatch.setattr(
        selector_module,
        "load_local_semantic_component_runtime",
        fail,
    )
    result = select_product_retrieval(
        query="selected paper paraphrase",
        sparse_hits=sparse,
        db_dir=tmp_path,
        retrieval_mode="hybrid_local",
        source_scope="current_paper",
        source_paths=["C:/private/doc-a.md"],
        product_enabled=True,
    )

    assert result.hits == sparse
    assert result.used_mode == "bm25"
    assert result.fallback_reason_id == "component_lifecycle_not_ready"


def test_scoped_dense_recovers_evidence_with_no_sparse_hits(tmp_path: Path) -> None:
    result = select_product_retrieval(
        query="semantic paraphrase", sparse_hits=[], db_dir=tmp_path,
        retrieval_mode="hybrid_local", source_scope="current_paper",
        source_paths=["C:/private/doc-a.md"],
        runtime_service=_Service(_Dense([_hit(90)])), product_enabled=True,
    )
    assert _ids(result.hits) == ["doc-a:chunk-90"]
    assert result.used_mode == "hybrid_local"


def test_scoped_dense_filters_before_global_top_k(tmp_path: Path) -> None:
    result = select_product_retrieval(
        query="semantic paraphrase", sparse_hits=[_hit(1)], db_dir=tmp_path,
        retrieval_mode="hybrid_local", source_scope="basket",
        source_paths=["C:/private/doc-a.md", "C:/private/doc-b.md"],
        runtime_service=_Service(_Dense([
            *[_hit(i, source="outside") for i in range(1, 81)],
            _hit(90, source="doc-b"),
        ])), product_enabled=True,
    )
    assert _ids(result.hits) == ["doc-a:chunk-1", "doc-b:chunk-90"]
    assert result.dense_novelty_count == 1


def test_scoped_dense_requires_explicit_paths_before_loading_runtime(tmp_path: Path) -> None:
    class UnexpectedRuntime:
        def load_ready(self, _root):
            raise AssertionError("missing scope must not load runtime")

    result = select_product_retrieval(
        query="question", sparse_hits=[_hit(1)], db_dir=tmp_path,
        retrieval_mode="hybrid_local", source_scope="current_paper",
        runtime_service=UnexpectedRuntime(), product_enabled=True,
    )
    assert result.hits == [_hit(1)]
    assert result.used_mode == "bm25"
    assert result.fallback_reason_id == "local_semantic_scope_sources_missing"
