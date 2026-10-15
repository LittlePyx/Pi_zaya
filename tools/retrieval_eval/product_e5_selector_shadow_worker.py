from __future__ import annotations

import argparse
import copy
import ctypes
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.e5_small_embedding_runtime import E5SmallOnnxExecutor
from kb.hybrid_retriever import _hit_identity, _source_identity
from kb.local_retrieval_selector import select_product_retrieval
from kb.local_semantic_artifacts import VerifiedLocalSemanticArtifacts
from kb.local_semantic_runtime import LocalSemanticRuntimeService
from kb.local_semantic_window_index import LocalSemanticWindowIndexStore
from kb.retrieval_benchmark import evaluate_candidate_channel, validate_indexed_corpus
from kb.retrieval_semantic_eval_fixture import load_retrieval_benchmark_fixture
from kb.store import atomic_write_json, load_all_chunks
from tools.retrieval_eval.product_query_sparse_retriever import (
    ProductQuerySparseRetriever,
)
from tools.retrieval_eval.run_user_journey_fusion_v3_selection import (
    compact_selection_channel,
)


PRODUCT_CHANNEL = "bm25_product_queries_selection_control"
CANDIDATE_CHANNEL = "e5_small_product_selector_candidate"


def _peak_rss_mib() -> float:
    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("page_fault_count", wintypes.DWORD),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.WinDLL(
            "psapi",
            use_last_error=True,
        ).GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        success = get_process_memory_info(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if not success:
            raise ctypes.WinError(ctypes.get_last_error())
        return round(counters.peak_working_set_size / (1024 * 1024), 3)
    import resource

    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0
    return round(raw / divisor, 3)


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _canonical_sha256(value: object) -> str:
    import hashlib

    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class _ActualSelectorRetriever:
    def __init__(
        self,
        chunks: list[dict],
        *,
        whole_library: bool,
        db_dir: Path,
        runtime_service: LocalSemanticRuntimeService,
        trace: list[dict[str, Any]],
    ) -> None:
        self._sparse = ProductQuerySparseRetriever(
            chunks,
            whole_library=whole_library,
        )
        self._whole_library = bool(whole_library)
        self._source_paths = [str(chunk.get("meta", {}).get("source_path") or "") for chunk in chunks]
        self._allowed_sources = {_source_identity(chunk) for chunk in chunks}
        self._db_dir = db_dir
        self._runtime_service = runtime_service
        self._trace = trace

    @property
    def is_empty(self) -> bool:
        return self._sparse.is_empty

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        sparse_hits = self._sparse.search(query, top_k=max(20, int(top_k)))
        before = copy.deepcopy(sparse_hits)
        before_sha = _canonical_sha256(before)
        started = time.perf_counter()
        selection = select_product_retrieval(
            query=query,
            sparse_hits=sparse_hits,
            db_dir=self._db_dir,
            retrieval_mode="hybrid_local",
            source_scope="library" if self._whole_library else "scoped",
            source_paths=self._source_paths,
            runtime_service=self._runtime_service,
            product_enabled=True,
        )
        duration_ms = (time.perf_counter() - started) * 1000.0
        after_sha = _canonical_sha256(sparse_hits)
        prefix_depth = min(5, len(before))
        protected = [
            _hit_identity(hit) for hit in selection.hits[:prefix_depth]
        ] == [_hit_identity(hit) for hit in before[:prefix_depth]]
        allowed_sources = self._allowed_sources
        source_escapes = (
            0
            if self._whole_library
            else sum(
                _source_identity(hit) not in allowed_sources
                for hit in selection.hits[: max(1, int(top_k))]
            )
        )
        self._trace.append(
            {
                "ordinal": len(self._trace) + 1,
                "mode": "full_library" if self._whole_library else "scoped",
                "duration_ms": round(duration_ms, 3),
                "requested_mode": selection.requested_mode,
                "used_mode": selection.used_mode,
                "fallback_reason_id": selection.fallback_reason_id,
                "protected_prefix_preserved": protected,
                "dense_novelty_count": selection.dense_novelty_count,
                "source_escape_count": source_escapes,
                "product_sparse_input_unchanged": before_sha == after_sha,
                "cold_start": selection.cold_start,
                "index_cache_hit": selection.index_cache_hit,
            }
        )
        return list(selection.hits[: max(1, int(top_k))])


def run_worker(
    *,
    db_dir: Path,
    index_dir: Path,
    artifact_root: Path,
    fixture_path: Path,
) -> dict[str, Any]:
    fixture = {
        **load_retrieval_benchmark_fixture(fixture_path, repo_root=REPO_ROOT),
        "ks": [5, 10],
    }
    chunks = load_all_chunks(db_dir)
    _mapping, errors = validate_indexed_corpus(fixture, chunks)
    if errors:
        raise RuntimeError("working corpus mismatch: " + "; ".join(errors))

    artifacts = VerifiedLocalSemanticArtifacts(artifact_root)
    if not artifacts.is_ready():
        raise RuntimeError("verified E5 artifacts are unavailable")
    model_path = artifact_root / "model.onnx"
    tokenizer_path = artifact_root / "tokenizer" / "tokenizer.json"

    provider_started = time.perf_counter()
    executor = E5SmallOnnxExecutor(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    provider_load_ms = (time.perf_counter() - provider_started) * 1000.0
    store = LocalSemanticWindowIndexStore(db_dir, index_dir=index_dir)
    build = store.build(executor)
    loaded = store.load(executor)
    status = store.status()
    atomic_readback = (
        status.state == "ready"
        and status.build_id == build.build_id
        and len(loaded.chunks) == build.parent_count
        and len(loaded.windows) == build.window_count
        and len(loaded.vectors) == build.window_count
    )

    runtime_service = LocalSemanticRuntimeService(
        artifacts_factory=lambda _root: VerifiedLocalSemanticArtifacts(artifact_root),
        index_factory=lambda root, _index: LocalSemanticWindowIndexStore(
            root,
            index_dir=index_dir,
        ),
        executor_factory=lambda _model, _tokenizer: E5SmallOnnxExecutor(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
        ),
    )

    def product_factory(selected: list[dict], whole_library: bool):
        return ProductQuerySparseRetriever(selected, whole_library=whole_library)

    trace: list[dict[str, Any]] = []

    def candidate_factory(selected: list[dict], whole_library: bool):
        return _ActualSelectorRetriever(
            selected,
            whole_library=whole_library,
            db_dir=db_dir,
            runtime_service=runtime_service,
            trace=trace,
        )

    product = compact_selection_channel(
        evaluate_candidate_channel(
            fixture,
            chunks,
            channel_name=PRODUCT_CHANNEL,
            mode_retriever_factory=product_factory,
        )
    )
    candidate = compact_selection_channel(
        evaluate_candidate_channel(
            fixture,
            chunks,
            channel_name=CANDIDATE_CHANNEL,
            mode_retriever_factory=candidate_factory,
        )
    )
    if len(trace) != len(fixture["cases"]) * 2:
        raise RuntimeError("actual selector call count changed")
    return {
        "schema_version": 1,
        "worker_kind": "retrieval_product_e5_selector_shadow_worker_v1",
        "runtime": {
            "provider_load_ms": round(provider_load_ms, 3),
            "full_build_ms": round(build.duration_ms, 3),
            "peak_process_rss_mib": _peak_rss_mib(),
            "python_version": ".".join(str(item) for item in sys.version_info[:3]),
        },
        "index": {
            "build_id": build.build_id,
            "document_count": build.document_count,
            "parent_count": build.parent_count,
            "window_count": build.window_count,
            "embedded_parent_count": build.embedded_parent_count,
            "embedded_window_count": build.embedded_window_count,
            "reused_parent_count": build.reused_parent_count,
            "reused_window_count": build.reused_window_count,
            "no_op": build.no_op,
            "index_bytes": _directory_bytes(index_dir),
            "atomic_readback_passed": atomic_readback,
        },
        "selector_trace": trace,
        "channels": {
            PRODUCT_CHANNEL: product,
            CANDIDATE_CHANNEL: candidate,
        },
        "execution": {
            "full_build_count": 0 if build.no_op else 1,
            "selection_call_count": len(trace),
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
            "network_used": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = run_worker(
            db_dir=args.db_dir.resolve(),
            index_dir=args.index_dir.resolve(),
            artifact_root=args.artifact_root.resolve(),
            fixture_path=args.fixture.resolve(),
        )
        atomic_write_json(args.output.resolve(), payload)
    except Exception as exc:
        print(f"[FAIL] product E5 selector worker: {type(exc).__name__}: {exc}")
        return 1
    print(
        "[OK] product E5 selector worker completed "
        f"{payload['index']['parent_count']} parents/"
        f"{payload['index']['window_count']} windows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
