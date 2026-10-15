from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def _peak_rss_mib() -> float:
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
    query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    query.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    query.restype = wintypes.BOOL
    if not query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return round(counters.peak_working_set_size / (1024 * 1024), 3)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(float(item) for item in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * max(0.0, min(1.0, float(percentile)))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--db-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    component_root = args.component_root.resolve()
    optional_site = component_root / "python" / "site-packages"
    sys.path.insert(0, str(optional_site))
    sys.path.insert(0, str(repo_root))

    try:
        product_import_started = time.perf_counter()
        import onnxruntime
        import tokenizers

        from kb.e5_small_embedding_runtime import E5SmallOnnxExecutor
        from kb.hybrid_retriever import _source_identity
        from kb.local_retrieval_selector import select_product_retrieval
        from kb.local_semantic_artifacts import VerifiedLocalSemanticArtifacts
        from kb.local_semantic_runtime import LocalSemanticRuntimeService
        from kb.local_semantic_window_index import LocalSemanticWindowIndexStore
        from kb.product_query_policy import search_hits_with_fallback
        from kb.retriever import BM25Retriever
        from kb.store import atomic_write_json, load_all_chunks

        product_import_ms = (time.perf_counter() - product_import_started) * 1000.0
        retrieval_engine_imported = "kb.retrieval_engine" in sys.modules
        llm_imported = "kb.llm" in sys.modules

        fixture = json.loads(args.fixture.resolve().read_text(encoding="utf-8"))
        cases_by_id = {str(case["id"]): case for case in fixture["cases"]}
        case_ids = [
            "fresh-fdm-integration-time-boundary",
            "fresh-scinerf-overlap-rate",
        ]
        chunks = load_all_chunks(args.db_dir.resolve())
        artifact_root = component_root / "artifacts"
        model_path = artifact_root / "model.onnx"
        tokenizer_path = artifact_root / "tokenizer" / "tokenizer.json"
        runtime_service = LocalSemanticRuntimeService(
            dependency_probe=lambda: True,
            artifacts_factory=lambda _root: VerifiedLocalSemanticArtifacts(artifact_root),
            index_factory=lambda root, _index: LocalSemanticWindowIndexStore(
                root,
                index_dir=args.index_dir.resolve(),
            ),
            executor_factory=lambda _model, _tokenizer: E5SmallOnnxExecutor(
                model_path=model_path,
                tokenizer_path=tokenizer_path,
            ),
        )
        cold_started = time.perf_counter()
        cold = runtime_service.load_ready(args.db_dir.resolve())
        cold_load_ms = (time.perf_counter() - cold_started) * 1000.0

        query_rows: list[dict[str, object]] = []
        for case_id in case_ids:
            case = cases_by_id[case_id]
            index_doc_id = next(
                str(item["index_doc_id"])
                for item in fixture["corpus"]["documents"]
                if item["doc_id"] == case["primary_doc_id"]
            )
            selected = [
                dict(chunk)
                for chunk in chunks
                if str(chunk.get("id") or "").startswith(index_doc_id + ":")
            ]
            sparse = BM25Retriever(selected)
            sparse_hits, _scores, _used_query, _translated, _variants = (
                search_hits_with_fallback(
                    str(case["query"]),
                    sparse,
                    20,
                    SimpleNamespace(api_key=None, query_expansion_enabled=False),
                    allow_translate=True,
                    allow_expand=False,
                    whole_library=False,
                )
            )
            sparse_hits = list(sparse_hits[:20])
            allowed_sources = {_source_identity(hit) for hit in sparse_hits}
            started = time.perf_counter()
            selection = select_product_retrieval(
                query=str(case["query"]),
                sparse_hits=sparse_hits,
                db_dir=args.db_dir.resolve(),
                retrieval_mode="hybrid_local",
                source_scope="scoped",
                runtime_service=runtime_service,
                product_enabled=True,
            )
            duration_ms = (time.perf_counter() - started) * 1000.0
            ranked_ids = [str(hit.get("id") or "") for hit in selection.hits[:10]]
            relevant = {str(item) for item in case["relevance"]}
            query_rows.append(
                {
                    "case_id": case_id,
                    "duration_ms": round(duration_ms, 3),
                    "used_mode": selection.used_mode,
                    "fallback_reason_id": selection.fallback_reason_id,
                    "cold_start": selection.cold_start,
                    "index_cache_hit": selection.index_cache_hit,
                    "source_escape_count": sum(
                        _source_identity(hit) not in allowed_sources
                        for hit in selection.hits[:10]
                    ),
                    "first_relevant_rank": next(
                        (
                            rank
                            for rank, chunk_id in enumerate(ranked_ids, start=1)
                            if chunk_id in relevant
                        ),
                        None,
                    ),
                    "ranked_chunk_ids": ranked_ids,
                }
            )

        durations = [float(row["duration_ms"]) for row in query_rows]
        output = {
            "schema_version": 1,
            "worker_kind": "retrieval_portable_e5_architecture_startup_v3",
            "runtime": {
                "python_version": ".".join(str(item) for item in sys.version_info[:3]),
                "onnxruntime_version": importlib.metadata.version("onnxruntime"),
                "tokenizers_version": importlib.metadata.version("tokenizers"),
                "onnxruntime_module_path": str(Path(onnxruntime.__file__).resolve()),
                "tokenizers_module_path": str(Path(tokenizers.__file__).resolve()),
                "product_import_ms": round(product_import_ms, 3),
                "cold_load_ms": round(cold_load_ms, 3),
                "query_p95_ms": round(_percentile(durations, 0.95), 3),
                "peak_process_rss_mib": _peak_rss_mib(),
            },
            "cold_load": {
                "cold_start": cold.cold_start,
                "index_cache_hit": cold.index_cache_hit,
            },
            "queries": query_rows,
            "execution": {
                "query_count": len(query_rows),
                "index_build_call_count": 0,
                "network_used": False,
                "evaluation_helper_imported_before_startup_timing": False,
                "retrieval_engine_imported_during_product_import": retrieval_engine_imported,
                "llm_imported_during_product_import": llm_imported,
                "product_enabled": False,
                "ordinary_user_surface_exposed": False,
            },
        }
        atomic_write_json(args.output.resolve(), output)
    except Exception as exc:
        print(f"[FAIL] portable E5 architecture startup: {type(exc).__name__}: {exc}")
        return 1
    print(
        "[OK] portable E5 architecture startup completed "
        f"imports={output['runtime']['product_import_ms']}ms "
        f"cold={output['runtime']['cold_load_ms']}ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
