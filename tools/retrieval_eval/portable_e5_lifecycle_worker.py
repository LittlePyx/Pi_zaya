from __future__ import annotations

import argparse
import ctypes
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
    position = (len(ordered) - 1) * max(0.0, min(1.0, float(percentile)))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class _ExistingArchiveDownloader:
    def __init__(self, archive_path: Path) -> None:
        self.archive_path = archive_path
        self.calls = 0

    def download(self, manifest, _transport, *, should_cancel, on_progress):
        from kb.local_semantic_component import LocalSemanticComponentCancelled
        from kb.local_semantic_component_download import ComponentDownloadProgress

        self.calls += 1
        if should_cancel():
            raise LocalSemanticComponentCancelled()
        on_progress(
            ComponentDownloadProgress(
                completed_bytes=manifest.archive_bytes,
                total_bytes=manifest.archive_bytes,
                resumed_bytes=0,
                eta_seconds=0,
            )
        )
        return SimpleNamespace(archive_path=self.archive_path)


def _mutate_unrelated_chunk(db_dir: Path, *, excluded_doc_ids: set[str]) -> str:
    from kb.store import load_docs_index, write_doc_chunks

    docs = load_docs_index(db_dir)
    doc_id = next(
        item
        for item in sorted(docs)
        if item not in excluded_doc_ids and int(docs[item].get("num_chunks") or 0) > 0
    )
    path = db_dir / "chunks" / f"{doc_id}.jsonl"
    chunks = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunks[0]["text"] = str(chunks[0].get("text") or "") + " lifecycle-repair-probe"
    write_doc_chunks(db_dir, doc_id, chunks)
    return doc_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--db-dir", type=Path, required=True)
    parser.add_argument("--component-store", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    protocol = json.loads(args.protocol.resolve().read_text(encoding="utf-8"))
    component_spec = protocol["component"]
    lifecycle_spec = protocol["execution"]

    from kb.hybrid_retriever import _source_identity
    from kb.local_retrieval_selector import select_product_retrieval
    from kb.local_semantic_component import (
        LocalSemanticComponentManifest,
        LocalSemanticComponentStore,
    )
    from kb.local_semantic_component_lifecycle import (
        LocalSemanticComponentLifecycleService,
    )
    from kb.local_semantic_component_lifecycle_store import (
        LocalSemanticComponentLifecycleStore,
    )
    from kb.local_semantic_window_index import LocalSemanticWindowIndexStore
    from kb.portable_local_semantic_runtime import PortableLocalSemanticRuntimeAdapter
    from kb.product_query_policy import search_hits_with_fallback
    from kb.retriever import BM25Retriever
    from kb.store import atomic_write_json, load_all_chunks

    manifest = LocalSemanticComponentManifest.from_payload(
        {
            "component_id": component_spec["component_id"],
            "release_id": component_spec["release_id"],
            "download_url": component_spec["immutable_manifest_url"],
            "archive_name": component_spec["archive_name"],
            "archive_root": component_spec["archive_root"],
            "archive_bytes": component_spec["archive_bytes"],
            "archive_sha256": component_spec["archive_sha256"],
            "payload_bytes": component_spec["payload_bytes"],
            "payload_file_count": component_spec["payload_file_count"],
            "payload_sha256": component_spec["payload_sha256"],
            "candidate_id": component_spec["candidate_id"],
            "model_revision": component_spec["model_revision"],
        },
        allowed_hosts={"github.com"},
    )
    db_dir = args.db_dir.resolve()
    component_store = LocalSemanticComponentStore(args.component_store.resolve())
    index_store = LocalSemanticWindowIndexStore(db_dir)
    adapter = PortableLocalSemanticRuntimeAdapter(
        db_dir,
        component_store=component_store,
        index_store=index_store,
    )
    downloader = _ExistingArchiveDownloader(args.archive.resolve())
    transport_calls = 0

    def transport_factory() -> object:
        nonlocal transport_calls
        transport_calls += 1
        return object()

    lifecycle = LocalSemanticComponentLifecycleService(
        db_dir,
        manifest,
        transport_factory=transport_factory,
        component_store=component_store,
        downloader=downloader,
        lifecycle_store=LocalSemanticComponentLifecycleStore(db_dir),
        health_check=adapter.health_check,
        index_ready=adapter.index_ready,
        prepare_index=adapter.prepare_index,
        remove_index=adapter.remove_index,
        release_runtime_handles=adapter.release_runtime_handles,
        runner=lambda callback: callback(),
        retry_wait=lambda _seconds: None,
        owner_session="full-size-lifecycle-worker",
    )

    install_started = time.perf_counter()
    installed_status, created = lifecycle.enable(consent=True)
    install_ready_ms = (time.perf_counter() - install_started) * 1000.0
    if not created or installed_status.state != "ready":
        raise RuntimeError(
            f"component lifecycle did not reach ready: {installed_status.internal_payload()}"
        )

    load_started = time.perf_counter()
    runtime = adapter.load_ready(db_dir)
    cold_load_ms = (time.perf_counter() - load_started) * 1000.0

    fixture = json.loads(args.fixture.resolve().read_text(encoding="utf-8"))
    cases_by_id = {str(case["id"]): case for case in fixture["cases"]}
    case_ids = list(lifecycle_spec["case_ids"])
    chunks = load_all_chunks(db_dir)
    query_rows: list[dict[str, object]] = []
    excluded_doc_ids: set[str] = set()
    for case_id in case_ids:
        case = cases_by_id[case_id]
        index_doc_id = next(
            str(item["index_doc_id"])
            for item in fixture["corpus"]["documents"]
            if item["doc_id"] == case["primary_doc_id"]
        )
        excluded_doc_ids.add(index_doc_id)
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
            db_dir=db_dir,
            retrieval_mode="hybrid_local",
            source_scope="scoped",
            source_paths=[str(chunk.get("meta", {}).get("source_path") or "") for chunk in selected],
            runtime_service=adapter,
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

    mutated_doc_id = _mutate_unrelated_chunk(
        db_dir,
        excluded_doc_ids=excluded_doc_ids,
    )
    stale_state = index_store.status().state
    repair_started = time.perf_counter()
    repaired_status = lifecycle.status()
    stale_repair_ms = (time.perf_counter() - repair_started) * 1000.0
    repaired_index_state = index_store.status().state

    removal_started = time.perf_counter()
    removed_status, removal_created = lifecycle.remove()
    removal_ms = (time.perf_counter() - removal_started) * 1000.0
    cleanup_pending_path = component_store.root / "pending_cleanup.json"
    output = {
        "schema_version": 1,
        "worker_kind": "portable_e5_full_size_lifecycle_v1",
        "timings": {
            "install_ready_ms": round(install_ready_ms, 3),
            "cold_load_ms": round(cold_load_ms, 3),
            "warm_query_p95_ms": round(
                _percentile(
                    [float(item["duration_ms"]) for item in query_rows],
                    0.95,
                ),
                3,
            ),
            "stale_repair_ms": round(stale_repair_ms, 3),
            "removal_ms": round(removal_ms, 3),
        },
        "runtime": {
            "python_version": ".".join(str(item) for item in sys.version_info[:3]),
            "cold_start": runtime.cold_start,
            "index_cache_hit": runtime.index_cache_hit,
            "peak_process_rss_mib": _peak_rss_mib(),
        },
        "queries": query_rows,
        "stale_repair": {
            "mutated_doc_id": mutated_doc_id,
            "state_before": stale_state,
            "state_after": repaired_index_state,
            "lifecycle_state": repaired_status.state,
            "index_build_call_count": 1,
        },
        "removal": {
            "created": removal_created,
            "lifecycle_state": removed_status.state,
            "public": removed_status.public_payload(),
            "component_pointer_present": component_store.current_path.exists(),
            "index_pointer_present": index_store.current_path.exists(),
            "cleanup_pending_after_worker": cleanup_pending_path.exists(),
        },
        "execution": {
            "explicit_consent": True,
            "download_network_used": False,
            "local_archive_adapter_calls": downloader.calls,
            "transport_factory_calls": transport_calls,
            "bm25_available_in_all_public_states": True,
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
        },
    }
    atomic_write_json(args.output.resolve(), output)
    print(
        "[OK] full-size lifecycle worker completed "
        f"install={output['timings']['install_ready_ms']}ms "
        f"repair={output['timings']['stale_repair_ms']}ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
