from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.dense_retriever import DenseRetriever
from kb.embedding_index import EmbeddingIndexStore
from kb.protected_dense_tail_retriever import ProtectedDenseTailRetriever
from kb.retrieval_benchmark import evaluate_candidate_channel, validate_indexed_corpus
from kb.retrieval_eval_fixture import sha256_file
from kb.retrieval_semantic_eval_fixture import load_retrieval_benchmark_fixture
from kb.store import atomic_write_json
from kb.transformers_embedding_provider import TransformersPoolingEmbeddingProvider
from tools.retrieval_eval.run_cpu_query_compatibility import (
    DEFAULT_REPORT as DEFAULT_CPU_QUERY_REPORT,
    TimedCachedQueryEmbedder,
    _percentile,
    _release_provider,
    _stable_channel_projection,
    _tree_fingerprint,
)
from tools.retrieval_eval.product_query_sparse_retriever import (
    ProductQuerySparseRetriever,
)
from tools.retrieval_eval.run_user_journey_fusion_v3_1_semantic_acceptance import (
    evaluate_acceptance,
)
from tools.retrieval_eval.run_user_journey_fusion_v3_selection import (
    PRODUCT_CHANNEL,
    compact_selection_channel,
)
from tools.retrieval_eval.validate_cpu_full_index_feasibility_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_cases_v1.json"
DEFAULT_INDEX = (
    REPO_ROOT
    / "test_results"
    / "retrieval_cpu_full_index_feasibility_v1"
    / "index_v1"
)
DEFAULT_ATTEMPT = (
    REPO_ROOT
    / "test_results"
    / "retrieval_cpu_full_index_feasibility_v1"
    / "attempt_v1.json"
)
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_cpu_full_index_feasibility_v1.json"
CHANNEL = "cpu_full_index_protected_dense_tail"
EXPECTED_RESULT_FINGERPRINT: str | None = (
    "3085c4ed8cf73b2e68fa071b569610ef1812902c726f1af7b6e91e7ddaae00af"
)
EXPECTED_REPORT_SHA256: str | None = (
    "866e2abb58566cb0f569e5eb4c3dca20e6f7764d60793dd68c2ea4e8f85f282a"
)
EXPECTED_INDEX: dict[str, Any] | None = {
    "relative_path": "test_results/retrieval_cpu_full_index_feasibility_v1/index_v1",
    "tree_sha256": "8c8fa74b7ce72c30687933b0f41d7e6b399615cf4d3a271313c0bfb85e169b50",
    "size_bytes": 7277202,
    "state": "ready",
    "reason": "embedding index matches the committed corpus and provider",
    "build_id": "7138dcaa15c242bd88bc14042166e879",
    "corpus_fingerprint": "19dd55b81535593219b0fdb8744b6c74cdfe51d68c3567dfd822780be927ec7a",
    "provider_identity": "a5bb526562ff581e073496f8fc14077e119373c5051a70a78aec7306770434e9",
    "document_count": 10,
    "chunk_count": 1673,
    "readback_chunk_count": 1673,
    "current_pointer_sha256": "ac37a6dfb3e89d4cb30f9d95a27c1e6411d565e009d0ae8b0adb79232baa7d65",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _index_size_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _peak_process_rss_mib() -> float:
    if sys.platform == "win32":
        size_type = ctypes.c_size_t

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", size_type),
                ("WorkingSetSize", size_type),
                ("QuotaPeakPagedPoolUsage", size_type),
                ("QuotaPagedPoolUsage", size_type),
                ("QuotaPeakNonPagedPoolUsage", size_type),
                ("QuotaNonPagedPoolUsage", size_type),
                ("PagefileUsage", size_type),
                ("PeakPagefileUsage", size_type),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            raise OSError("peak process memory is unavailable")
        return round(float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0), 3)
    import resource

    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return round(value / (1024.0 * 1024.0), 3)
    return round(value / 1024.0, 3)


def _model_parameter_dtype(provider: TransformersPoolingEmbeddingProvider) -> str:
    model = getattr(provider, "_model", None)
    if model is None:
        return ""
    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration):
        return ""
    return str(getattr(parameter, "dtype", "")).removeprefix("torch.")


def _provider(contract: dict[str, Any]) -> TransformersPoolingEmbeddingProvider:
    return TransformersPoolingEmbeddingProvider(
        model_id=str(contract["model_id"]),
        revision=str(contract["revision"]),
        dimension=1024,
        query_prefix=(
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
            "Query:"
        ),
        passage_prefix="",
        max_length=int(contract["max_length"]),
        batch_size=int(contract["document_batch_size"]),
        device=str(contract["device"]),
        dtype=str(contract["dtype"]),
        allow_download=bool(contract["allow_download"]),
        pooling=str(contract["pooling"]),
        padding_side=str(contract["padding_side"]),
    )


def _result_fingerprint(report: dict[str, Any]) -> str:
    projection = {
        "channels": {
            name: _stable_channel_projection(channel)
            for name, channel in (report.get("channels") or {}).items()
        },
        "semantic_acceptance": report.get("semantic_acceptance"),
        "gate": report.get("gate"),
        "decision": report.get("decision"),
    }
    canonical = json.dumps(projection, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def evaluate_gate(
    *,
    product: dict[str, Any],
    cpu_query_parent: dict[str, Any],
    candidate: dict[str, Any],
    semantic_acceptance: dict[str, Any],
    provider_load_ms: float,
    build: dict[str, Any],
    index_status: dict[str, Any],
    index_size_bytes: int,
    peak_process_rss_mib: float,
    query_durations_ms: list[float],
    expected_query_count: int,
    runtime_info: dict[str, Any],
    source_corpus_unchanged: bool,
    quality_gate: dict[str, Any],
    operational_gate: dict[str, Any],
) -> dict[str, Any]:
    mode_checks: dict[str, dict[str, bool]] = {}
    metrics = quality_gate[
        "require_no_regression_vs_cpu_query_compatibility_metrics"
    ]
    for mode in ("scoped", "full_library"):
        product_metrics = product["modes"][mode]["metrics"]
        parent_metrics = cpu_query_parent["modes"][mode]["metrics"]
        candidate_metrics = candidate["modes"][mode]["metrics"]
        checks = {
            f"no_regression_{metric}": candidate_metrics[metric]
            >= parent_metrics[metric]
            for metric in metrics
        }
        checks["recall_at_5_equal_to_product"] = (
            candidate_metrics["recall@5"] == product_metrics["recall@5"]
        )
        checks["wrong_source_at_5_equal_to_product"] = (
            candidate_metrics["wrong_source_rate@5"]
            == product_metrics["wrong_source_rate@5"]
        )
        checks["wrong_source_at_5_no_worse_than_cpu_query_parent"] = (
            candidate_metrics["wrong_source_rate@5"]
            <= parent_metrics["wrong_source_rate@5"]
        )
        mode_checks[mode] = checks
    build_ms = float(build.get("duration_ms") or 0.0)
    embedded_chunks = int(build.get("embedded_chunk_count") or 0)
    throughput = (
        round(embedded_chunks / (build_ms / 1000.0), 6) if build_ms > 0.0 else 0.0
    )
    query_count = len(query_durations_ms)
    expected_chunks = int(index_status.get("chunk_count") or 0)
    operational_checks = {
        "provider_load_ms": provider_load_ms
        <= float(operational_gate["maximum_provider_load_ms"]),
        "full_build_ms": build_ms
        <= float(operational_gate["maximum_full_build_ms"]),
        "build_throughput": throughput
        >= float(operational_gate["minimum_build_throughput_chunks_per_second"]),
        "peak_process_rss_mib": peak_process_rss_mib
        <= float(operational_gate["maximum_peak_process_rss_mib"]),
        "index_size_bytes": index_size_bytes
        <= int(operational_gate["maximum_index_size_bytes"]),
        "query_completion_rate": (
            round(query_count / expected_query_count, 6)
            >= float(operational_gate["required_query_completion_rate"])
        ),
        "warm_query_embedding_p95_ms": (
            _percentile(query_durations_ms, 0.95)
            <= float(operational_gate["maximum_warm_query_embedding_p95_ms"])
        ),
        "document_truncation": int(
            runtime_info.get("truncated_document_count") or 0
        )
        <= int(operational_gate["maximum_truncated_document_count"]),
        "all_chunks_embedded": (
            embedded_chunks == expected_chunks and expected_chunks > 0
        ),
        "no_reuse": (
            int(build.get("reused_document_count") or 0) == 0
            and int(build.get("reused_chunk_count") or 0) == 0
            and build.get("no_op") is False
        ),
        "index_ready_and_read_back": (
            index_status.get("state") == "ready"
            and int(index_status.get("document_count") or 0) == 10
            and expected_chunks == 1673
        ),
        "source_corpus_unchanged": bool(source_corpus_unchanged),
    }
    passes = bool(
        semantic_acceptance.get("passes_all")
        and all(all(checks.values()) for checks in mode_checks.values())
        and all(operational_checks.values())
    )
    return {
        "semantic_acceptance_passes": bool(semantic_acceptance.get("passes_all")),
        "mode_checks": mode_checks,
        "operational_checks": operational_checks,
        "passes_all": passes,
    }


def _build_summary(report: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    runtime = report["runtime"]
    build = report["build"]
    timings = list(runtime["query_timings"])
    durations = [float(row["duration_ms"]) for row in timings]
    build_ms = round(float(build["duration_ms"]), 3)
    embedded = int(build["embedded_chunk_count"])
    return {
        "provider_load_ms": round(float(runtime["provider_load_ms"]), 3),
        "full_build_ms": build_ms,
        "build_throughput_chunks_per_second": round(
            embedded / (build_ms / 1000.0), 6
        ),
        "peak_process_rss_mib": round(float(runtime["peak_process_rss_mib"]), 3),
        "index_size_bytes": int(report["index"]["size_bytes"]),
        "unique_query_count": len(durations),
        "query_completion_rate": round(
            len(durations) / int(protocol["fixture"]["case_count"]), 6
        ),
        "query_embedding_p50_ms": _percentile(durations, 0.50),
        "query_embedding_p95_ms": _percentile(durations, 0.95),
        "truncated_document_count": int(
            runtime["provider"].get("truncated_document_count") or 0
        ),
    }


def validate_report(
    report: dict[str, Any],
    *,
    protocol: dict[str, Any],
    fixture: dict[str, Any],
    cpu_query_parent: dict[str, Any],
    expected_result_fingerprint: str | None = EXPECTED_RESULT_FINGERPRINT,
    expected_index: dict[str, Any] | None = EXPECTED_INDEX,
) -> list[str]:
    errors: list[str] = []
    if (
        report.get("schema_version") != 1
        or report.get("report_kind") != "retrieval_cpu_full_index_feasibility_v1"
    ):
        errors.append("CPU full-index report identity changed")
    if (
        report.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL)
        or report.get("fixture_sha256") != sha256_file(DEFAULT_FIXTURE)
        or report.get("cpu_query_parent_sha256")
        != sha256_file(DEFAULT_CPU_QUERY_REPORT)
    ):
        errors.append("CPU full-index report source identity changed")
    if report.get("case_ids") != [str(case["id"]) for case in fixture["cases"]]:
        errors.append("CPU full-index case membership changed")
    if report.get("candidate") != protocol["cpu_provider"]:
        errors.append("CPU full-index candidate changed")
    source = report.get("source_corpus") or {}
    if (
        source.get("before_sha256") != protocol["isolated_corpus"]["tree_sha256"]
        or source.get("before_sha256") != source.get("after_sha256")
        or source.get("unchanged") is not True
    ):
        errors.append("CPU full-index source corpus changed")
    build = report.get("build") or {}
    if (
        int(build.get("document_count") or 0)
        != protocol["isolated_corpus"]["document_count"]
        or int(build.get("chunk_count") or 0)
        != protocol["isolated_corpus"]["chunk_count"]
        or int(build.get("embedded_document_count") or 0) != 10
        or int(build.get("embedded_chunk_count") or 0) != 1673
        or str(build.get("corpus_fingerprint") or "")
        != protocol["isolated_corpus"]["corpus_fingerprint"]
    ):
        errors.append("CPU full-index build membership changed")
    index = report.get("index") or {}
    if (
        index.get("relative_path") != protocol["dedicated_index"]["relative_path"]
        or index.get("state") != "ready"
        or index.get("provider_identity")
        != protocol["cpu_provider"]["provider_identity"]
        or index.get("build_id") != build.get("build_id")
        or index.get("readback_chunk_count") != 1673
        or not str(index.get("tree_sha256") or "")
        or int(index.get("size_bytes") or 0) <= 0
    ):
        errors.append("CPU full-index committed index changed")
    if expected_index is not None and index != expected_index:
        errors.append("pinned CPU full-index identity changed")
    channels = report.get("channels") or {}
    if list(channels) != [PRODUCT_CHANNEL, CHANNEL]:
        errors.append("CPU full-index channels changed")
        return errors
    if _stable_channel_projection(channels[PRODUCT_CHANNEL]) != _stable_channel_projection(
        cpu_query_parent["channels"][PRODUCT_CHANNEL]
    ):
        errors.append("CPU full-index product control changed")
    timings = list((report.get("runtime") or {}).get("query_timings") or [])
    if [row.get("ordinal") for row in timings] != list(range(1, 13)) or any(
        set(row) != {"ordinal", "duration_ms"} for row in timings
    ):
        errors.append("CPU full-index query timing membership changed")
    expected_summary = _build_summary(report, protocol)
    if report.get("summary") != expected_summary:
        errors.append("CPU full-index runtime summary is inconsistent")
    expected_acceptance = evaluate_acceptance(
        channels[PRODUCT_CHANNEL],
        channels[CHANNEL],
        minimum_gain=float(
            protocol["quality_gate"][
                "minimum_semantic_recall_at_10_gain_each_mode"
            ]
        ),
    )
    if report.get("semantic_acceptance") != expected_acceptance:
        errors.append("CPU full-index semantic acceptance is inconsistent")
    durations = [float(row["duration_ms"]) for row in timings]
    expected_gate = evaluate_gate(
        product=channels[PRODUCT_CHANNEL],
        cpu_query_parent=cpu_query_parent["channels"][
            "cpu_float32_protected_dense_tail"
        ],
        candidate=channels[CHANNEL],
        semantic_acceptance=expected_acceptance,
        provider_load_ms=expected_summary["provider_load_ms"],
        build=build,
        index_status=index,
        index_size_bytes=int(index.get("size_bytes") or 0),
        peak_process_rss_mib=expected_summary["peak_process_rss_mib"],
        query_durations_ms=durations,
        expected_query_count=12,
        runtime_info=report["runtime"]["provider"],
        source_corpus_unchanged=bool(source.get("unchanged")),
        quality_gate=protocol["quality_gate"],
        operational_gate=protocol["operational_gate"],
    )
    if report.get("gate") != expected_gate:
        errors.append("CPU full-index gate is inconsistent")
    expected_decision = {
        "status": (
            protocol["execution"]["pass_decision"]
            if expected_gate["passes_all"]
            else protocol["execution"]["reject_decision"]
        ),
        "cpu_optional_runtime_footprint_study_allowed": bool(
            expected_gate["passes_all"]
        ),
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }
    if report.get("decision") != expected_decision:
        errors.append("CPU full-index decision changed")
    fingerprint = _result_fingerprint(report)
    if report.get("result_fingerprint") != fingerprint:
        errors.append("CPU full-index result fingerprint is inconsistent")
    if expected_result_fingerprint and fingerprint != expected_result_fingerprint:
        errors.append("pinned CPU full-index result changed")
    return errors


def _begin_attempt(*, semantic_db_root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    if DEFAULT_ATTEMPT.exists():
        raise ValueError("CPU full-index attempt receipt already exists")
    if DEFAULT_REPORT.exists():
        raise ValueError("CPU full-index report already exists")
    if DEFAULT_INDEX.exists():
        raise ValueError("CPU full-index destination already exists")
    source_sha = _tree_fingerprint(semantic_db_root)
    if source_sha != protocol["isolated_corpus"]["tree_sha256"]:
        raise ValueError("CPU full-index source corpus identity mismatch")
    receipt = {
        "schema_version": 1,
        "attempt_kind": "retrieval_cpu_full_index_feasibility_v1",
        "status": "started",
        "started_at": _utc_now(),
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "source_corpus_sha256": source_sha,
        "process_id": os.getpid(),
        "maximum_runs": 1,
    }
    atomic_write_json(DEFAULT_ATTEMPT, receipt)
    return receipt


def _finish_attempt(receipt: dict[str, Any], *, status: str) -> None:
    atomic_write_json(
        DEFAULT_ATTEMPT,
        {
            **receipt,
            "status": status,
            "finished_at": _utc_now(),
            "report_persisted": DEFAULT_REPORT.exists(),
        },
    )


def measure(
    *,
    semantic_db_root: Path,
    protocol: dict[str, Any],
    fixture: dict[str, Any],
    cpu_query_parent: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    source_before = _tree_fingerprint(semantic_db_root)
    provider = _provider(protocol["cpu_provider"])
    if provider.descriptor.identity != protocol["cpu_provider"]["provider_identity"]:
        raise ValueError("CPU full-index provider identity mismatch")
    try:
        provider.prepare()
        provider_load_ms = float(provider.runtime_info()["load_duration_ms"])
        store = EmbeddingIndexStore(semantic_db_root, index_dir=DEFAULT_INDEX)
        build_result = store.build(provider)
        status = store.status(provider)
        loaded = store.load(provider)
        semantic_fixture = {**fixture, "ks": [5, 10]}
        _mapping, corpus_errors = validate_indexed_corpus(
            semantic_fixture, loaded.chunks
        )
        if corpus_errors:
            raise ValueError("CPU full-index corpus readback mismatch")
        timed_query = TimedCachedQueryEmbedder(provider.embed_query)
        vectors = {
            str(chunk.get("id") or ""): vector
            for chunk, vector in zip(loaded.chunks, loaded.embeddings)
        }

        def product_factory(chunks: list[dict], whole_library: bool):
            return ProductQuerySparseRetriever(chunks, whole_library=whole_library)

        def fusion_factory(chunks: list[dict], whole_library: bool):
            dense = DenseRetriever(
                chunks,
                [vectors[str(chunk.get("id") or "")] for chunk in chunks],
                timed_query,
                min_score=-1.0,
            )
            fusion = protocol["fusion"]
            return ProtectedDenseTailRetriever(
                product_factory(chunks, whole_library),
                dense,
                dense_novelty_quota=int(fusion["dense_novelty_quota"]),
                candidate_multiplier=int(fusion["candidate_multiplier"]),
                protected_sparse_prefix=int(fusion["protected_sparse_prefix"]),
                sparse_tail_window=int(fusion["sparse_tail_window"]),
            )

        product = compact_selection_channel(
            evaluate_candidate_channel(
                semantic_fixture,
                loaded.chunks,
                channel_name=PRODUCT_CHANNEL,
                mode_retriever_factory=product_factory,
            )
        )
        candidate = compact_selection_channel(
            evaluate_candidate_channel(
                semantic_fixture,
                loaded.chunks,
                channel_name=CHANNEL,
                mode_retriever_factory=fusion_factory,
            )
        )
        source_after = _tree_fingerprint(semantic_db_root)
        index_tree = _tree_fingerprint(DEFAULT_INDEX)
        index_size = _index_size_bytes(DEFAULT_INDEX)
        provider_runtime = provider.runtime_info()
        provider_runtime["model_parameter_dtype"] = _model_parameter_dtype(provider)
        durations = [float(row["duration_ms"]) for row in timed_query.timings]
        build = {
            "build_id": build_result.build_id,
            "corpus_fingerprint": build_result.corpus_fingerprint,
            "document_count": build_result.document_count,
            "chunk_count": build_result.chunk_count,
            "embedded_document_count": build_result.embedded_document_count,
            "embedded_chunk_count": build_result.embedded_chunk_count,
            "reused_document_count": build_result.reused_document_count,
            "reused_chunk_count": build_result.reused_chunk_count,
            "no_op": build_result.no_op,
            "duration_ms": round(float(build_result.duration_ms), 3),
        }
        index = {
            "relative_path": protocol["dedicated_index"]["relative_path"],
            "tree_sha256": index_tree,
            "size_bytes": index_size,
            "state": status.state,
            "reason": status.reason,
            "build_id": status.build_id,
            "corpus_fingerprint": status.corpus_fingerprint,
            "provider_identity": status.provider_identity,
            "document_count": status.document_count,
            "chunk_count": status.chunk_count,
            "readback_chunk_count": len(loaded.chunks),
            "current_pointer_sha256": _raw_sha256(DEFAULT_INDEX / "current.json"),
        }
        semantic_acceptance = evaluate_acceptance(
            product,
            candidate,
            minimum_gain=float(
                protocol["quality_gate"][
                    "minimum_semantic_recall_at_10_gain_each_mode"
                ]
            ),
        )
        peak_rss = _peak_process_rss_mib()
        gate = evaluate_gate(
            product=product,
            cpu_query_parent=cpu_query_parent["channels"][
                "cpu_float32_protected_dense_tail"
            ],
            candidate=candidate,
            semantic_acceptance=semantic_acceptance,
            provider_load_ms=round(provider_load_ms, 3),
            build=build,
            index_status=index,
            index_size_bytes=index_size,
            peak_process_rss_mib=peak_rss,
            query_durations_ms=durations,
            expected_query_count=12,
            runtime_info=provider_runtime,
            source_corpus_unchanged=source_before == source_after,
            quality_gate=protocol["quality_gate"],
            operational_gate=protocol["operational_gate"],
        )
        report: dict[str, Any] = {
            "schema_version": 1,
            "report_kind": "retrieval_cpu_full_index_feasibility_v1",
            "measured_at": _utc_now(),
            "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
            "fixture_sha256": sha256_file(DEFAULT_FIXTURE),
            "cpu_query_parent_sha256": sha256_file(DEFAULT_CPU_QUERY_REPORT),
            "case_ids": [str(case["id"]) for case in fixture["cases"]],
            "candidate": dict(protocol["cpu_provider"]),
            "attempt": {
                "started_at": receipt["started_at"],
                "maximum_runs": receipt["maximum_runs"],
                "raw_report_persisted_before_validation": True,
            },
            "source_corpus": {
                "before_sha256": source_before,
                "after_sha256": source_after,
                "unchanged": source_before == source_after,
            },
            "build": build,
            "index": index,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "provider_load_ms": round(provider_load_ms, 3),
                "peak_process_rss_mib": peak_rss,
                "provider": provider_runtime,
                "query_timings": timed_query.timings,
            },
            "channels": {PRODUCT_CHANNEL: product, CHANNEL: candidate},
            "semantic_acceptance": semantic_acceptance,
            "gate": gate,
            "decision": {
                "status": (
                    protocol["execution"]["pass_decision"]
                    if gate["passes_all"]
                    else protocol["execution"]["reject_decision"]
                ),
                "cpu_optional_runtime_footprint_study_allowed": bool(
                    gate["passes_all"]
                ),
                "product_enabled": False,
                "ordinary_user_surface_exposed": False,
            },
        }
        report["summary"] = _build_summary(report, protocol)
        report["result_fingerprint"] = _result_fingerprint(report)
        return report
    finally:
        _release_provider(provider)


def _load_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    errors = validate_protocol(protocol)
    if errors:
        raise ValueError("CPU full-index protocol invalid: " + "; ".join(errors))
    fixture = load_retrieval_benchmark_fixture(DEFAULT_FIXTURE, repo_root=REPO_ROOT)
    parent = json.loads(DEFAULT_CPU_QUERY_REPORT.read_text(encoding="utf-8"))
    return protocol, fixture, parent


def _failure_report(
    *,
    protocol: dict[str, Any],
    receipt: dict[str, Any],
    semantic_db_root: Path,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "report_kind": "retrieval_cpu_full_index_feasibility_v1_execution_failure",
        "measured_at": _utc_now(),
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "attempt": {
            "started_at": receipt["started_at"],
            "maximum_runs": 1,
            "raw_report_persisted_before_validation": True,
        },
        "failure": {
            "category": type(exc).__name__,
            "message": "CPU full-index execution failed before a valid outcome",
        },
        "source_corpus_sha256": (
            _tree_fingerprint(semantic_db_root) if semantic_db_root.exists() else ""
        ),
        "dedicated_index_exists": DEFAULT_INDEX.exists(),
        "peak_process_rss_mib": _peak_process_rss_mib(),
        "decision": {
            "status": "execution_failed_no_retry",
            "cpu_optional_runtime_footprint_study_allowed": False,
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run or validate the one-shot CPU full-index feasibility study."
    )
    parser.add_argument("--semantic-db-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    protocol, fixture, cpu_query_parent = _load_contracts()
    if args.validate_only:
        try:
            report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[FAIL] CPU full-index report cannot be read: {type(exc).__name__}")
            return 2
        errors = validate_report(
            report,
            protocol=protocol,
            fixture=fixture,
            cpu_query_parent=cpu_query_parent,
        )
        if EXPECTED_REPORT_SHA256 and sha256_file(DEFAULT_REPORT) != EXPECTED_REPORT_SHA256:
            errors.append("pinned CPU full-index report file changed")
        if errors:
            print("[FAIL] " + "; ".join(errors))
            return 1
        print(f"[OK] CPU full-index feasibility: {report['decision']['status']}")
        return 0
    if args.semantic_db_root is None:
        parser.error("--semantic-db-root is required unless --validate-only is used")
    semantic_db_root = args.semantic_db_root.expanduser().resolve()
    receipt = _begin_attempt(semantic_db_root=semantic_db_root, protocol=protocol)
    try:
        report = measure(
            semantic_db_root=semantic_db_root,
            protocol=protocol,
            fixture=fixture,
            cpu_query_parent=cpu_query_parent,
            receipt=receipt,
        )
    except Exception as exc:
        atomic_write_json(
            DEFAULT_REPORT,
            _failure_report(
                protocol=protocol,
                receipt=receipt,
                semantic_db_root=semantic_db_root,
                exc=exc,
            ),
        )
        _finish_attempt(receipt, status="execution_failed_no_retry")
        print(f"[FAIL] CPU full-index execution failed: {type(exc).__name__}")
        print(f"[FAIL] retained {DEFAULT_REPORT}")
        return 1
    atomic_write_json(DEFAULT_REPORT, report)
    errors = validate_report(
        report,
        protocol=protocol,
        fixture=fixture,
        cpu_query_parent=cpu_query_parent,
        expected_result_fingerprint=None,
        expected_index=None,
    )
    if errors:
        _finish_attempt(receipt, status="validation_failed_no_retry")
        print("[FAIL] raw CPU full-index report preserved before validation")
        print("[FAIL] " + "; ".join(errors))
        return 1
    _finish_attempt(receipt, status="completed")
    print(json.dumps({"summary": report["summary"], "decision": report["decision"]}, indent=2))
    print(f"[OK] wrote {DEFAULT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
