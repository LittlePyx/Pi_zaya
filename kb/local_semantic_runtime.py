from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .candidate_retriever import CandidateRetriever
from .e5_small_embedding_runtime import E5SmallOnnxExecutor
from .local_semantic_artifacts import VerifiedLocalSemanticArtifacts
from .local_semantic_preparation_service import (
    local_semantic_model_root,
    local_semantic_runtime_dependencies_available,
)
from .local_semantic_window_index import (
    LocalSemanticIndexError,
    LocalSemanticWindowIndexStore,
    local_semantic_index_dir,
)


class LocalSemanticUnavailable(RuntimeError):
    def __init__(self, reason_id: str) -> None:
        super().__init__(str(reason_id or "local_semantic_runtime_unavailable"))
        self.reason_id = str(reason_id or "local_semantic_runtime_unavailable")


@dataclass(frozen=True)
class LocalSemanticRuntime:
    retriever: CandidateRetriever
    index_cache_hit: bool
    cold_start: bool


ArtifactsFactory = Callable[[Path], VerifiedLocalSemanticArtifacts]
IndexFactory = Callable[[Path, Path], LocalSemanticWindowIndexStore]
ExecutorFactory = Callable[[Path, Path], E5SmallOnnxExecutor]


def _artifacts(root: Path) -> VerifiedLocalSemanticArtifacts:
    return VerifiedLocalSemanticArtifacts(root)


def _index(db_dir: Path, index_dir: Path) -> LocalSemanticWindowIndexStore:
    return LocalSemanticWindowIndexStore(db_dir, index_dir=index_dir)


def _executor(model_path: Path, tokenizer_path: Path) -> E5SmallOnnxExecutor:
    return E5SmallOnnxExecutor(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )


class LocalSemanticRuntimeService:
    """Read only fully verified E5 artifacts and a matching committed index."""

    def __init__(
        self,
        *,
        dependency_probe: Callable[[], bool] = local_semantic_runtime_dependencies_available,
        artifacts_factory: ArtifactsFactory = _artifacts,
        index_factory: IndexFactory = _index,
        executor_factory: ExecutorFactory = _executor,
    ) -> None:
        self._dependency_probe = dependency_probe
        self._artifacts_factory = artifacts_factory
        self._index_factory = index_factory
        self._executor_factory = executor_factory
        self._lock = threading.Lock()
        self._runtime_key: tuple[object, ...] | None = None
        self._runtime: LocalSemanticRuntime | None = None

    @staticmethod
    def _signature(path: Path) -> tuple[int, int]:
        try:
            stat = path.stat()
            return int(stat.st_mtime_ns), int(stat.st_size)
        except OSError:
            return 0, 0

    def _key(self, db_dir: Path, model_root: Path, index_dir: Path) -> tuple[object, ...]:
        return (
            str(db_dir).casefold(),
            self._signature(db_dir / "docs.json"),
            self._signature(index_dir / "current.json"),
            self._signature(model_root / "ready.json"),
            self._signature(model_root / "model.onnx"),
            self._signature(model_root / "tokenizer" / "tokenizer.json"),
        )

    def load_ready(self, db_dir: Path | str) -> LocalSemanticRuntime:
        root = Path(db_dir).expanduser().resolve(strict=False)
        model_root = local_semantic_model_root(root)
        index_dir = local_semantic_index_dir(root)
        key = self._key(root, model_root, index_dir)
        with self._lock:
            if self._runtime is not None and self._runtime_key == key:
                return LocalSemanticRuntime(
                    retriever=self._runtime.retriever,
                    index_cache_hit=True,
                    cold_start=False,
                )
        if not self._dependency_probe():
            raise LocalSemanticUnavailable("runtime_dependencies_missing")
        artifacts = self._artifacts_factory(model_root)
        if not artifacts.is_ready():
            raise LocalSemanticUnavailable("local_semantic_artifacts_not_ready")
        try:
            executor = self._executor_factory(
                model_root / "model.onnx",
                model_root / "tokenizer" / "tokenizer.json",
            )
            loaded = self._index_factory(root, index_dir).load(executor)
            runtime = LocalSemanticRuntime(
                retriever=loaded.as_retriever(executor),
                index_cache_hit=False,
                cold_start=True,
            )
        except LocalSemanticIndexError as exc:
            raise LocalSemanticUnavailable(f"local_semantic_index_{exc.code}") from exc
        except LocalSemanticUnavailable:
            raise
        except Exception as exc:
            raise LocalSemanticUnavailable("local_semantic_runtime_unavailable") from exc
        with self._lock:
            self._runtime_key = key
            self._runtime = runtime
        return runtime


def local_semantic_runtime_failure_reason(exc: Exception) -> str:
    if isinstance(exc, LocalSemanticUnavailable):
        return exc.reason_id
    if isinstance(exc, LocalSemanticIndexError):
        return f"local_semantic_index_{exc.code}"
    return "local_semantic_runtime_unavailable"


DEFAULT_LOCAL_SEMANTIC_RUNTIME = LocalSemanticRuntimeService()
