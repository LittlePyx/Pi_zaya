from __future__ import annotations

import importlib.util
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .candidate_retriever import CandidateRetriever
from .embedding_index import EmbeddingIndexError, EmbeddingIndexStore
from .transformers_embedding_provider import TransformersPoolingEmbeddingProvider


LOCAL_RETRIEVAL_MODES = ("bm25", "hybrid_local")
QWEN_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
QWEN_MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QWEN_PROVIDER_IDENTITY = (
    "55229348d5bbe4b1e1935aeb8ffe57cc03c527685b1d08d44f11a56a54525cb8"
)


def normalize_retrieval_mode(value: object) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    return raw if raw in LOCAL_RETRIEVAL_MODES else "bm25"


def product_dense_index_dir(db_dir: Path) -> Path:
    return Path(db_dir).expanduser().resolve(strict=False) / "dense_index"


def build_qwen_product_provider() -> TransformersPoolingEmbeddingProvider:
    provider = TransformersPoolingEmbeddingProvider(
        model_id=QWEN_MODEL_ID,
        revision=QWEN_MODEL_REVISION,
        dimension=1024,
        query_prefix=(
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
            "Query:"
        ),
        passage_prefix="",
        max_length=8192,
        batch_size=1,
        device="cuda",
        dtype="float16",
        allow_download=False,
        pooling="last_token",
        padding_side="left",
    )
    if provider.descriptor.identity != QWEN_PROVIDER_IDENTITY:
        raise RuntimeError("local Dense provider identity mismatch")
    return provider


def _dependencies_available() -> bool:
    required = ("huggingface_hub", "numpy", "safetensors", "torch", "transformers")
    return all(importlib.util.find_spec(name) is not None for name in required)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _pinned_model_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=QWEN_MODEL_ID,
                revision=QWEN_MODEL_REVISION,
                local_files_only=True,
            )
        )
        has_weights = any(snapshot.glob("*.safetensors"))
        has_tokenizer = (snapshot / "tokenizer.json").is_file() or (
            snapshot / "tokenizer_config.json"
        ).is_file()
        return bool((snapshot / "config.json").is_file() and has_weights and has_tokenizer)
    except Exception:
        return False


def _vectorized_backend_available() -> bool:
    try:
        import numpy

        return bool(getattr(numpy, "float32", None))
    except Exception:
        return False


@dataclass(frozen=True)
class LocalDenseReadiness:
    state: str
    reason_id: str
    available: bool
    mode: str = "hybrid_local"
    document_count: int = 0
    chunk_count: int = 0

    def public_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LocalDenseRuntime:
    retriever: CandidateRetriever
    index_cache_hit: bool
    cold_start: bool


class LocalDenseUnavailable(RuntimeError):
    def __init__(self, reason_id: str) -> None:
        super().__init__(str(reason_id or "runtime_unavailable"))
        self.reason_id = str(reason_id or "runtime_unavailable")


DependencyProbe = Callable[[], bool]
ProviderFactory = Callable[[], TransformersPoolingEmbeddingProvider]
IndexStoreFactory = Callable[[Path, Path], EmbeddingIndexStore]


def _index_store(db_dir: Path, index_dir: Path) -> EmbeddingIndexStore:
    return EmbeddingIndexStore(db_dir, index_dir=index_dir)


class LocalDenseRuntimeService:
    """Cache-only product runtime for the one accepted local Dense candidate."""

    def __init__(
        self,
        *,
        dependency_probe: DependencyProbe = _dependencies_available,
        accelerator_probe: DependencyProbe = _cuda_available,
        model_cache_probe: DependencyProbe = _pinned_model_cached,
        vector_backend_probe: DependencyProbe = _vectorized_backend_available,
        provider_factory: ProviderFactory = build_qwen_product_provider,
        index_store_factory: IndexStoreFactory = _index_store,
    ) -> None:
        self._dependency_probe = dependency_probe
        self._accelerator_probe = accelerator_probe
        self._model_cache_probe = model_cache_probe
        self._vector_backend_probe = vector_backend_probe
        self._provider_factory = provider_factory
        self._index_store_factory = index_store_factory
        self._lock = threading.Lock()
        self._runtime_key: tuple[str, tuple[int, int], tuple[int, int]] | None = None
        self._runtime: LocalDenseRuntime | None = None

    @staticmethod
    def _signature(path: Path) -> tuple[int, int]:
        try:
            stat = path.stat()
            return int(stat.st_mtime_ns), int(stat.st_size)
        except OSError:
            return 0, 0

    @staticmethod
    def _unavailable(reason_id: str) -> LocalDenseReadiness:
        return LocalDenseReadiness(
            state="unavailable",
            reason_id=str(reason_id or "runtime_unavailable"),
            available=False,
        )

    def _prerequisite_failure(self) -> str:
        try:
            if not self._dependency_probe():
                return "runtime_dependencies_missing"
            if not self._vector_backend_probe():
                return "vector_backend_unavailable"
            if not self._accelerator_probe():
                return "local_accelerator_unavailable"
            if not self._model_cache_probe():
                return "pinned_model_not_cached"
        except Exception:
            return "runtime_unavailable"
        return ""

    def readiness(self, db_dir: Path) -> LocalDenseReadiness:
        root = Path(db_dir).expanduser().resolve(strict=False)
        index_dir = product_dense_index_dir(root)
        prerequisite_failure = self._prerequisite_failure()
        if prerequisite_failure:
            return self._unavailable(prerequisite_failure)
        try:
            provider = self._provider_factory()
            status = self._index_store_factory(root, index_dir).status(provider)
        except Exception:
            return self._unavailable("runtime_unavailable")
        if status.state != "ready":
            reason = status.state if status.state in {
                "missing",
                "stale",
                "provider_mismatch",
                "corrupt",
            } else "unavailable"
            return self._unavailable(f"dense_index_{reason}")
        if status.provider_identity != QWEN_PROVIDER_IDENTITY:
            return self._unavailable("dense_index_provider_mismatch")
        return LocalDenseReadiness(
            state="ready",
            reason_id="ready",
            available=True,
            document_count=status.document_count,
            chunk_count=status.chunk_count,
        )

    def load_ready(self, db_dir: Path) -> LocalDenseRuntime:
        prerequisite_failure = self._prerequisite_failure()
        if prerequisite_failure:
            raise LocalDenseUnavailable(prerequisite_failure)
        return self.load(db_dir)

    def load(self, db_dir: Path) -> LocalDenseRuntime:
        root = Path(db_dir).expanduser().resolve(strict=False)
        index_dir = product_dense_index_dir(root)
        key = (
            str(root).casefold(),
            self._signature(root / "docs.json"),
            self._signature(index_dir / "current.json"),
        )
        with self._lock:
            if self._runtime is not None and self._runtime_key == key:
                return LocalDenseRuntime(
                    retriever=self._runtime.retriever,
                    index_cache_hit=True,
                    cold_start=False,
                )
        provider = self._provider_factory()
        loaded = self._index_store_factory(root, index_dir).load(provider)
        runtime = LocalDenseRuntime(
            retriever=loaded.as_retriever(provider),
            index_cache_hit=False,
            cold_start=True,
        )
        with self._lock:
            self._runtime_key = key
            self._runtime = runtime
        return runtime


def dense_runtime_failure_reason(exc: Exception) -> str:
    if isinstance(exc, LocalDenseUnavailable):
        return exc.reason_id
    if isinstance(exc, EmbeddingIndexError):
        return f"dense_index_{exc.code}"
    return "dense_runtime_unavailable"


DEFAULT_LOCAL_DENSE_RUNTIME = LocalDenseRuntimeService()
