from __future__ import annotations

import argparse
import copy
import ctypes
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
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
from kb.retrieval_benchmark import validate_indexed_corpus
from kb.retrieval_metrics import evaluate_ranking, mean_metrics
from kb.retrieval_semantic_eval_fixture import load_retrieval_benchmark_fixture
from kb.store import atomic_write_json, load_all_chunks
from tools.retrieval_eval.product_query_sparse_retriever import (
    ProductQuerySparseRetriever,
)


PRODUCT_CHANNEL = "bm25_product_queries_scoped_control"
CANDIDATE_CHANNEL = "e5_small_product_selector_scoped_candidate"


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
    import resource

    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0
    return round(raw / divisor, 3)


def _canonical_sha256(value: object) -> str:
    import hashlib

    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _percentile(values: Sequence[float], percentile: float) -> float:
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


def _contract_recall(
    ranked_ids: Sequence[str],
    contracts: Sequence[Mapping[str, Any]],
    k: int,
) -> float:
    retrieved = set(str(item) for item in ranked_ids[: max(0, int(k))])
    if not contracts:
        return 0.0
    covered = sum(
        bool(retrieved & set(str(item) for item in contract["relevant_chunk_ids"]))
        for contract in contracts
    )
    return covered / len(contracts)


def _case_row(
    case: Mapping[str, Any],
    ranked_ids: list[str],
    *,
    latency_ms: float,
) -> dict[str, Any]:
    relevance = {
        str(chunk_id): float(grade)
        for chunk_id, grade in dict(case.get("relevance") or {}).items()
    }
    contracts = [
        item for item in list(case.get("contracts") or []) if isinstance(item, Mapping)
    ]
    metrics = evaluate_ranking(ranked_ids, relevance, ks=(1, 3, 5, 10))
    for k in (1, 3, 5, 10):
        metrics[f"contract_recall@{k}"] = _contract_recall(ranked_ids, contracts, k)
    return {
        "id": str(case.get("id") or ""),
        "locale": str(case.get("locale") or ""),
        "evaluation_focus": str(case.get("evaluation_focus") or ""),
        "first_relevant_rank": next(
            (rank for rank, chunk_id in enumerate(ranked_ids, start=1) if chunk_id in relevance),
            None,
        ),
        "ranked_chunk_ids": ranked_ids,
        "latency_ms": round(latency_ms, 3),
        "metrics": {key: round(value, 6) for key, value in metrics.items()},
    }


def _channel(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows]
    metrics = mean_metrics([row["metrics"] for row in rows])
    return {
        "channel": name,
        "mode": "scoped",
        "case_count": len(rows),
        "metrics": {key: round(value, 6) for key, value in metrics.items()},
        "latency_ms": {
            "p50": round(_percentile(latencies, 0.5), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
            "max": round(max(latencies, default=0.0), 3),
        },
        "cases": rows,
    }


def run_worker(
    *,
    db_dir: Path,
    index_dir: Path,
    artifact_root: Path,
    fixture_path: Path,
) -> dict[str, Any]:
    fixture = load_retrieval_benchmark_fixture(fixture_path, repo_root=REPO_ROOT)
    index_to_logical = {
        str(item["index_doc_id"]): str(item["doc_id"])
        for item in fixture["corpus"]["documents"]
    }
    chunks = [
        chunk
        for chunk in load_all_chunks(db_dir)
        if str(chunk.get("id") or "").split(":", 1)[0] in index_to_logical
    ]
    chunk_to_doc, errors = validate_indexed_corpus(fixture, chunks)
    if errors:
        raise RuntimeError("scoped fresh working corpus mismatch: " + "; ".join(errors))

    artifacts = VerifiedLocalSemanticArtifacts(artifact_root)
    if not artifacts.is_ready():
        raise RuntimeError("verified E5 artifacts are unavailable")
    model_path = artifact_root / "model.onnx"
    tokenizer_path = artifact_root / "tokenizer" / "tokenizer.json"
    index_store = LocalSemanticWindowIndexStore(db_dir, index_dir=index_dir)
    status = index_store.status()
    if status.state != "ready":
        raise RuntimeError(f"existing E5 index is not ready: {status.reason_id}")

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

    product_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for case in fixture["cases"]:
        scope_doc_ids = set(str(item) for item in case["scope_doc_ids"])
        selected = [
            dict(chunk)
            for chunk in chunks
            if chunk_to_doc[str(chunk["id"])] in scope_doc_ids
        ]
        sparse = ProductQuerySparseRetriever(selected, whole_library=False)
        sparse_started = time.perf_counter()
        sparse_hits = sparse.search(str(case["query"]), top_k=20)
        sparse_ms = (time.perf_counter() - sparse_started) * 1000.0
        before = copy.deepcopy(sparse_hits)
        before_sha = _canonical_sha256(before)
        selector_started = time.perf_counter()
        selection = select_product_retrieval(
            query=str(case["query"]),
            sparse_hits=sparse_hits,
            db_dir=db_dir,
            retrieval_mode="hybrid_local",
            source_scope="scoped",
            source_paths=[str(chunk.get("meta", {}).get("source_path") or "") for chunk in selected],
            runtime_service=runtime_service,
            product_enabled=True,
        )
        selector_ms = (time.perf_counter() - selector_started) * 1000.0
        after_sha = _canonical_sha256(sparse_hits)
        allowed_sources = {_source_identity(hit) for hit in before}
        candidate_hits = list(selection.hits[:10])
        prefix_depth = min(5, len(before))
        trace.append(
            {
                "ordinal": len(trace) + 1,
                "case_id": str(case["id"]),
                "mode": "scoped",
                "duration_ms": round(selector_ms, 3),
                "requested_mode": selection.requested_mode,
                "used_mode": selection.used_mode,
                "fallback_reason_id": selection.fallback_reason_id,
                "protected_prefix_preserved": [
                    _hit_identity(hit) for hit in candidate_hits[:prefix_depth]
                ]
                == [_hit_identity(hit) for hit in before[:prefix_depth]],
                "dense_novelty_count": selection.dense_novelty_count,
                "source_escape_count": sum(
                    _source_identity(hit) not in allowed_sources for hit in candidate_hits
                ),
                "product_sparse_input_unchanged": before_sha == after_sha,
                "cold_start": selection.cold_start,
                "index_cache_hit": selection.index_cache_hit,
            }
        )
        product_rows.append(
            _case_row(
                case,
                [str(hit.get("id") or "") for hit in before[:10]],
                latency_ms=sparse_ms,
            )
        )
        candidate_rows.append(
            _case_row(
                case,
                [str(hit.get("id") or "") for hit in candidate_hits],
                latency_ms=sparse_ms + selector_ms,
            )
        )

    return {
        "schema_version": 1,
        "worker_kind": "retrieval_scoped_fresh_e5_worker_v1",
        "runtime": {
            "peak_process_rss_mib": _peak_rss_mib(),
            "python_version": ".".join(str(item) for item in sys.version_info[:3]),
        },
        "index": {
            "build_id": status.build_id,
            "document_count": status.document_count,
            "parent_count": status.parent_count,
            "window_count": status.window_count,
        },
        "selector_trace": trace,
        "channels": {
            PRODUCT_CHANNEL: _channel(PRODUCT_CHANNEL, product_rows),
            CANDIDATE_CHANNEL: _channel(CANDIDATE_CHANNEL, candidate_rows),
        },
        "execution": {
            "index_build_call_count": 0,
            "selection_call_count": len(trace),
            "full_library_call_count": 0,
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
        print(f"[FAIL] scoped fresh E5 worker: {type(exc).__name__}: {exc}")
        return 1
    print(f"[OK] scoped fresh E5 worker completed {payload['execution']['selection_call_count']} calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
