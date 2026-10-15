from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kb.portable_local_semantic_runtime import (
    CANDIDATE_ID,
    COMPONENT_ID,
    COMPONENT_PYTHON_VERSION,
    MODEL_REVISION,
    PortableLocalSemanticRuntimeAdapter,
    PortableLocalSemanticRuntimeError,
)


class _Retriever:
    is_empty = False

    def search(self, _query: str, top_k: int = 6):
        return [{"id": "dense"}][:top_k]


class _Loaded:
    def as_retriever(self, _executor):
        return _Retriever()


class _Index:
    def __init__(self, root: Path) -> None:
        self.index_dir = root
        self.generations_dir = root / "generations"
        self.current_path = root / "current.json"
        self.state = "missing"
        self.build_calls = 0
        self.load_calls = 0

    def status(self):
        return SimpleNamespace(state=self.state)

    def build(self, _executor, *, should_cancel, on_progress):
        self.build_calls += 1
        assert should_cancel() is False
        on_progress(SimpleNamespace(completed_windows=3, total_windows=3))
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.current_path.write_text("ready", encoding="utf-8")
        self.state = "ready"
        return SimpleNamespace(build_id="build")

    def load(self, _executor):
        self.load_calls += 1
        return _Loaded()


class _ComponentStore:
    def __init__(self, root: Path) -> None:
        self.installed = SimpleNamespace(root=root)

    def load_current(self):
        return self.installed


def _component(root: Path, *, candidate_id: str = CANDIDATE_ID) -> None:
    root.mkdir(parents=True)
    (root / "install.json").write_text("{}", encoding="utf-8")
    (root / "artifacts" / "tokenizer").mkdir(parents=True)
    (root / "artifacts" / "model.onnx").write_bytes(b"model")
    (root / "artifacts" / "tokenizer" / "tokenizer.json").write_bytes(b"tokens")
    (root / "component-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "component_id": COMPONENT_ID,
                "platform": "windows-x64",
                "python_version": COMPONENT_PYTHON_VERSION,
                "candidate_id": candidate_id,
                "model_revision": MODEL_REVISION,
                "contains_user_library_data": False,
                "product_enabled_by_installation": False,
            }
        ),
        encoding="utf-8",
    )


def _adapter(tmp_path: Path):
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    (db_dir / "docs.json").write_text("{}", encoding="utf-8")
    component_root = tmp_path / "component"
    _component(component_root)
    index = _Index(db_dir / "local_semantic_index" / "e5_small_v1")
    activations = []
    executors = []
    adapter = PortableLocalSemanticRuntimeAdapter(
        db_dir,
        component_store=_ComponentStore(component_root),  # type: ignore[arg-type]
        index_store=index,  # type: ignore[arg-type]
        artifact_probe=lambda root: (root / "artifacts" / "model.onnx").is_file(),
        dependency_activator=lambda root: activations.append(root) or str(root / "site"),
        executor_factory=lambda root: executors.append(root) or object(),
    )
    return adapter, component_root, index, activations, executors


def test_health_requires_exact_inner_component_manifest(tmp_path: Path) -> None:
    adapter, component_root, _index, activations, executors = _adapter(tmp_path)
    _component_manifest = component_root / "component-manifest.json"
    payload = json.loads(_component_manifest.read_text(encoding="utf-8"))
    payload["candidate_id"] = "wrong"
    _component_manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert adapter.health_check(component_root) is False
    assert activations == [] and executors == []


def test_health_is_cached_per_immutable_generation_signature(tmp_path: Path) -> None:
    adapter, component_root, _index, activations, executors = _adapter(tmp_path)
    assert adapter.health_check(component_root) is True
    assert adapter.health_check(component_root) is True
    assert activations == [component_root]
    assert executors == [component_root]


def test_prepare_real_index_boundary_and_load_cached_runtime(tmp_path: Path) -> None:
    adapter, component_root, index, _activations, _executors = _adapter(tmp_path)
    progress = []
    assert adapter.health_check(component_root) is True
    adapter.prepare_index(
        component_root,
        should_cancel=lambda: False,
        on_progress=lambda completed, total: progress.append((completed, total)),
    )
    first = adapter.load_ready()
    second = adapter.load_ready()
    assert progress == [(3, 3)]
    assert index.build_calls == 1 and index.load_calls == 1
    assert first.cold_start is True and first.index_cache_hit is False
    assert second.cold_start is False and second.index_cache_hit is True
    assert first.retriever.search("query", top_k=1) == [{"id": "dense"}]


def test_load_rejects_stale_or_missing_index(tmp_path: Path) -> None:
    adapter, component_root, _index, _activations, _executors = _adapter(tmp_path)
    assert adapter.health_check(component_root) is True
    with pytest.raises(PortableLocalSemanticRuntimeError) as captured:
        adapter.load_ready()
    assert captured.value.code == "component_or_index_not_ready"


def test_load_rejects_a_different_database_scope(tmp_path: Path) -> None:
    adapter, _component_root, _index, _activations, _executors = _adapter(tmp_path)
    with pytest.raises(PortableLocalSemanticRuntimeError) as captured:
        adapter.load_ready(tmp_path / "other-db")
    assert captured.value.code == "runtime_database_mismatch"


def test_release_clears_cached_runtime_and_remove_disables_index_first(
    tmp_path: Path,
) -> None:
    adapter, component_root, index, _activations, executors = _adapter(tmp_path)
    adapter.health_check(component_root)
    adapter.prepare_index(
        component_root,
        should_cancel=lambda: False,
        on_progress=lambda _completed, _total: None,
    )
    adapter.load_ready()
    generation = index.generations_dir / "one"
    generation.mkdir(parents=True)
    (generation / "vectors.f32").write_bytes(b"vectors")

    adapter.release_runtime_handles()
    adapter.remove_index()

    assert not index.current_path.exists()
    assert not index.generations_dir.exists()
    assert len(executors) == 1
