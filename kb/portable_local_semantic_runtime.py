from __future__ import annotations

import importlib
import json
import shutil
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .candidate_retriever import CandidateRetriever
from .e5_small_embedding_runtime import E5RuntimeError, E5SmallOnnxExecutor
from .local_semantic_artifacts import VerifiedLocalSemanticArtifacts
from .local_semantic_component import (
    CANDIDATE_ID,
    MODEL_REVISION,
    LocalSemanticComponentError,
    LocalSemanticComponentStore,
)
from .local_semantic_window_index import (
    LocalSemanticIndexError,
    LocalSemanticWindowIndexStore,
)


COMPONENT_ID = "pizaya_local_semantic_e5_small_windows_x64_v1"
COMPONENT_PYTHON_VERSION = "3.10.11"
ONNXRUNTIME_VERSION = "1.23.2"
TOKENIZERS_VERSION = "0.21.4"
NUMPY_VERSION = "2.2.6"


class PortableLocalSemanticRuntimeError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(str(code or "portable_semantic_runtime_failed"))
        self.code = str(code or "portable_semantic_runtime_failed")


@dataclass(frozen=True)
class PortableLocalSemanticRuntime:
    retriever: CandidateRetriever
    index_cache_hit: bool
    cold_start: bool


def _inside(path: Path, root: Path) -> bool:
    resolved = path.resolve(strict=False)
    base = root.resolve(strict=False)
    return resolved == base or base in resolved.parents


def _component_manifest_valid(component_root: Path) -> bool:
    try:
        payload = json.loads(
            (component_root / "component-manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("schema_version") == 1
        and payload.get("component_id") == COMPONENT_ID
        and payload.get("platform") == "windows-x64"
        and payload.get("python_version") == COMPONENT_PYTHON_VERSION
        and payload.get("candidate_id") == CANDIDATE_ID
        and payload.get("model_revision") == MODEL_REVISION
        and payload.get("contains_user_library_data") is False
        and payload.get("product_enabled_by_installation") is False
    )


def _default_artifact_probe(component_root: Path) -> bool:
    return VerifiedLocalSemanticArtifacts(component_root / "artifacts").is_ready()


def _module_path(module: Any) -> Path:
    return Path(str(getattr(module, "__file__", "") or "")).resolve(strict=False)


def _default_dependency_activator(component_root: Path) -> str:
    site_packages = (component_root / "python" / "site-packages").resolve(strict=False)
    if not site_packages.is_dir():
        raise PortableLocalSemanticRuntimeError("component_site_packages_missing")
    site_text = str(site_packages)
    if site_text not in sys.path:
        sys.path.insert(0, site_text)
    try:
        onnxruntime = importlib.import_module("onnxruntime")
        tokenizers = importlib.import_module("tokenizers")
        numpy = importlib.import_module("numpy")
        if (
            not _inside(_module_path(onnxruntime), site_packages)
            or not _inside(_module_path(tokenizers), site_packages)
            or str(getattr(onnxruntime, "__version__", "")) != ONNXRUNTIME_VERSION
            or str(getattr(tokenizers, "__version__", "")) != TOKENIZERS_VERSION
            or str(getattr(numpy, "__version__", "")) != NUMPY_VERSION
            or "CPUExecutionProvider" not in onnxruntime.get_available_providers()
        ):
            raise PortableLocalSemanticRuntimeError(
                "component_runtime_dependency_mismatch"
            )
    except PortableLocalSemanticRuntimeError:
        raise
    except Exception as exc:
        raise PortableLocalSemanticRuntimeError(
            "component_runtime_dependency_unavailable"
        ) from exc
    return site_text


def _default_executor(component_root: Path) -> E5SmallOnnxExecutor:
    artifacts = component_root / "artifacts"
    return E5SmallOnnxExecutor(
        model_path=artifacts / "model.onnx",
        tokenizer_path=artifacts / "tokenizer" / "tokenizer.json",
    )


class PortableLocalSemanticRuntimeAdapter:
    """Bind one verified optional component generation to the real E5 index."""

    def __init__(
        self,
        db_dir: Path | str,
        *,
        component_store: LocalSemanticComponentStore | None = None,
        index_store: LocalSemanticWindowIndexStore | None = None,
        artifact_probe: Callable[[Path], bool] = _default_artifact_probe,
        dependency_activator: Callable[[Path], str] = _default_dependency_activator,
        executor_factory: Callable[[Path], Any] = _default_executor,
    ) -> None:
        self.db_dir = Path(db_dir).expanduser().resolve(strict=False)
        self.component_store = component_store
        self.index_store = index_store or LocalSemanticWindowIndexStore(self.db_dir)
        self.artifact_probe = artifact_probe
        self.dependency_activator = dependency_activator
        self.executor_factory = executor_factory
        self._lock = threading.Lock()
        self._component_root: Path | None = None
        self._health_key: tuple[object, ...] | None = None
        self._executor: Any | None = None
        self._runtime_key: tuple[object, ...] | None = None
        self._runtime: PortableLocalSemanticRuntime | None = None
        self._activated_site_path = ""

    @staticmethod
    def _signature(path: Path) -> tuple[int, int]:
        try:
            item = path.stat()
            return int(item.st_mtime_ns), int(item.st_size)
        except OSError:
            return 0, 0

    def _health_signature(self, component_root: Path) -> tuple[object, ...]:
        return (
            str(component_root).casefold(),
            self._signature(component_root / "install.json"),
            self._signature(component_root / "component-manifest.json"),
            self._signature(component_root / "artifacts" / "model.onnx"),
            self._signature(
                component_root / "artifacts" / "tokenizer" / "tokenizer.json"
            ),
        )

    def health_check(self, component_root: Path | str) -> bool:
        root = Path(component_root).expanduser().resolve(strict=False)
        key = self._health_signature(root)
        with self._lock:
            if self._executor is not None and self._health_key == key:
                return True
        try:
            if not _component_manifest_valid(root) or not self.artifact_probe(root):
                return False
            site_path = self.dependency_activator(root)
            executor = self.executor_factory(root)
        except Exception:
            return False
        with self._lock:
            self._component_root = root
            self._health_key = key
            self._executor = executor
            self._runtime_key = None
            self._runtime = None
            self._activated_site_path = str(site_path or "")
        return True

    def index_ready(self, component_root: Path | str) -> bool:
        root = Path(component_root).expanduser().resolve(strict=False)
        with self._lock:
            if self._component_root != root or self._executor is None:
                return False
        try:
            return self.index_store.status().state == "ready"
        except Exception:
            return False

    def prepare_index(
        self,
        component_root: Path | str,
        *,
        should_cancel: Callable[[], bool],
        on_progress: Callable[[int, int], None],
    ) -> object:
        root = Path(component_root).expanduser().resolve(strict=False)
        if not self.health_check(root):
            raise PortableLocalSemanticRuntimeError("component_runtime_health_failed")
        with self._lock:
            executor = self._executor
        if executor is None:
            raise PortableLocalSemanticRuntimeError("component_runtime_health_failed")
        try:
            return self.index_store.build(
                executor,
                should_cancel=should_cancel,
                on_progress=lambda item: on_progress(
                    int(item.completed_windows), int(item.total_windows)
                ),
            )
        except LocalSemanticIndexError as exc:
            if exc.code == "cancelled_by_user":
                from .local_semantic_component import LocalSemanticComponentCancelled

                raise LocalSemanticComponentCancelled() from exc
            raise LocalSemanticComponentError(f"semantic_index_{exc.code}") from exc
        except E5RuntimeError as exc:
            raise LocalSemanticComponentError(f"semantic_runtime_{exc.code}") from exc

    def load_ready(
        self,
        db_dir: Path | str | None = None,
    ) -> PortableLocalSemanticRuntime:
        if db_dir is not None:
            requested = Path(db_dir).expanduser().resolve(strict=False)
            if requested != self.db_dir:
                raise PortableLocalSemanticRuntimeError("runtime_database_mismatch")
        if self.component_store is None:
            raise PortableLocalSemanticRuntimeError("component_store_unavailable")
        try:
            installed = self.component_store.load_current()
        except LocalSemanticComponentError as exc:
            raise PortableLocalSemanticRuntimeError("component_not_ready") from exc
        if not self.health_check(installed.root) or not self.index_ready(installed.root):
            raise PortableLocalSemanticRuntimeError("component_or_index_not_ready")
        runtime_key = (
            self._health_key,
            self._signature(self.index_store.current_path),
            self._signature(self.db_dir / "docs.json"),
        )
        with self._lock:
            if self._runtime is not None and self._runtime_key == runtime_key:
                return PortableLocalSemanticRuntime(
                    retriever=self._runtime.retriever,
                    index_cache_hit=True,
                    cold_start=False,
                )
            executor = self._executor
        try:
            loaded = self.index_store.load(executor)
            runtime = PortableLocalSemanticRuntime(
                retriever=loaded.as_retriever(executor),
                index_cache_hit=False,
                cold_start=True,
            )
        except LocalSemanticIndexError as exc:
            raise PortableLocalSemanticRuntimeError(
                f"semantic_index_{exc.code}"
            ) from exc
        with self._lock:
            self._runtime_key = runtime_key
            self._runtime = runtime
        return runtime

    def release_runtime_handles(self) -> None:
        with self._lock:
            site_path = self._activated_site_path
            self._component_root = None
            self._health_key = None
            self._executor = None
            self._runtime_key = None
            self._runtime = None
            self._activated_site_path = ""
        if site_path:
            while site_path in sys.path:
                sys.path.remove(site_path)

    def remove_index(self) -> None:
        """Disable the pointer first, then remove only the regenerable index."""

        index_dir = self.index_store.index_dir.resolve(strict=False)
        db_dir = self.db_dir.resolve(strict=False)
        if db_dir not in index_dir.parents:
            raise PortableLocalSemanticRuntimeError("semantic_index_path_invalid")
        self.index_store.current_path.unlink(missing_ok=True)
        generations = self.index_store.generations_dir
        if generations.is_dir():
            shutil.rmtree(generations)
        try:
            index_dir.rmdir()
        except OSError:
            pass
