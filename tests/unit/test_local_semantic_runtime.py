from __future__ import annotations

from pathlib import Path

import pytest

from kb.local_semantic_runtime import (
    LocalSemanticRuntimeService,
    LocalSemanticUnavailable,
)


class _Artifacts:
    def __init__(self, ready: bool = True) -> None:
        self.ready = ready
        self.validation_calls = 0

    def is_ready(self) -> bool:
        self.validation_calls += 1
        return self.ready


class _Retriever:
    is_empty = False

    def search(self, _query: str, top_k: int = 6):
        return [{"id": "e5"}][:top_k]


class _Loaded:
    def as_retriever(self, _executor):
        return _Retriever()


class _Index:
    def __init__(self) -> None:
        self.load_calls = 0

    def load(self, _executor):
        self.load_calls += 1
        return _Loaded()


def _service(artifacts: _Artifacts, index: _Index, *, dependencies: bool = True):
    return LocalSemanticRuntimeService(
        dependency_probe=lambda: dependencies,
        artifacts_factory=lambda _root: artifacts,
        index_factory=lambda _db, _index: index,
        executor_factory=lambda _model, _tokenizer: object(),
    )


def test_runtime_requires_packaged_dependencies_before_artifact_access(tmp_path: Path) -> None:
    artifacts = _Artifacts()
    service = _service(artifacts, _Index(), dependencies=False)
    with pytest.raises(LocalSemanticUnavailable) as captured:
        service.load_ready(tmp_path)
    assert captured.value.reason_id == "runtime_dependencies_missing"
    assert artifacts.validation_calls == 0


def test_runtime_requires_full_artifact_verification(tmp_path: Path) -> None:
    artifacts = _Artifacts(ready=False)
    service = _service(artifacts, _Index())
    with pytest.raises(LocalSemanticUnavailable) as captured:
        service.load_ready(tmp_path)
    assert captured.value.reason_id == "local_semantic_artifacts_not_ready"
    assert artifacts.validation_calls == 1


def test_runtime_caches_only_matching_corpus_index_and_artifact_signatures(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    model_root = tmp_path / "_local_models" / "multilingual_e5_small_official_onnx_v1"
    index_root = db_dir / "local_semantic_index" / "e5_small_v1"
    db_dir.mkdir(parents=True)
    model_root.joinpath("tokenizer").mkdir(parents=True)
    index_root.mkdir(parents=True)
    for path, value in (
        (db_dir / "docs.json", "docs"),
        (model_root / "ready.json", "ready"),
        (model_root / "model.onnx", "model"),
        (model_root / "tokenizer" / "tokenizer.json", "tokenizer"),
        (index_root / "current.json", "index"),
    ):
        path.write_text(value, encoding="utf-8")
    artifacts = _Artifacts()
    index = _Index()
    service = _service(artifacts, index)

    first = service.load_ready(db_dir)
    second = service.load_ready(db_dir)
    index_root.joinpath("current.json").write_text("changed index pointer", encoding="utf-8")
    third = service.load_ready(db_dir)

    assert first.cold_start is True and first.index_cache_hit is False
    assert second.cold_start is False and second.index_cache_hit is True
    assert third.cold_start is True and third.index_cache_hit is False
    assert artifacts.validation_calls == 2
    assert index.load_calls == 2
