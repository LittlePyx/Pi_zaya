from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
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
from kb.embedding_provider import CallableEmbeddingProvider, EmbeddingProviderDescriptor
from kb.protected_dense_tail_retriever import ProtectedDenseTailRetriever
from kb.retrieval_benchmark import evaluate_candidate_channel, validate_indexed_corpus
from kb.retrieval_eval_fixture import sha256_file
from kb.retrieval_semantic_eval_fixture import load_retrieval_benchmark_fixture
from kb.store import atomic_write_json
from tools.retrieval_eval.run_cpu_full_index_feasibility import _tree_fingerprint
from tools.retrieval_eval.run_cpu_query_compatibility import (
    _percentile,
    _stable_channel_projection,
)
from tools.retrieval_eval.product_query_sparse_retriever import (
    ProductQuerySparseRetriever,
)
from tools.retrieval_eval.run_user_journey_fusion_v3_1_semantic_acceptance import (
    evaluate_acceptance,
)
from tools.retrieval_eval.run_user_journey_fusion_v3_selection import (
    FUSION_CHANNEL,
    PRODUCT_CHANNEL,
    compact_selection_channel,
)
from tools.retrieval_eval.validate_lighter_runtime_pointer_invalidation import (
    DEFAULT_INVALIDATION,
    validate_invalidation,
)
from tools.retrieval_eval.validate_lighter_runtime_stage0 import (
    DEFAULT_REPORT as DEFAULT_STAGE0_REPORT,
    validate_report as validate_stage0_report,
)
from tools.retrieval_eval.validate_lighter_runtime_stage1_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)
from tools.retrieval_eval.validate_lighter_runtime_stress_manifest import (
    DEFAULT_MANIFEST,
    validate_manifest,
)


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_cases_v1.json"
DEFAULT_PARENT_REPORT = (
    REPO_ROOT / "docs" / "retrieval_user_journey_fusion_v3_1_semantic_acceptance_v2.json"
)
DEFAULT_INDEX = (
    REPO_ROOT / "test_results" / "retrieval_cpu_full_index_feasibility_v1" / "index_v1"
)
DEFAULT_ARTIFACT = (
    REPO_ROOT
    / "test_results"
    / "retrieval_lighter_runtime_v1"
    / "artifacts"
    / "qwen3_embedding_0_6b_official_quint8_avx2_onnx_v1"
    / "model_quint8_avx2.onnx"
)
DEFAULT_PYTHON = (
    REPO_ROOT / "test_results" / "retrieval_lighter_runtime_v1" / ".venv" / "Scripts" / "python.exe"
)
DEFAULT_WORKER = REPO_ROOT / "tools" / "retrieval_eval" / "lighter_runtime_onnx_worker.py"
DEFAULT_WORKER_RESULT = (
    REPO_ROOT / "test_results" / "retrieval_lighter_runtime_v1" / "stage1_qwen_worker_v1.json"
)
DEFAULT_ATTEMPT = (
    REPO_ROOT / "test_results" / "retrieval_lighter_runtime_v1" / "stage1_qwen_attempt_v1.json"
)
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage1_qwen_v1.json"
EXPECTED_REPORT_SHA256: str | None = (
    "d19e9576d0e2b272b27d155d76d7840c95d64374dd5af8f0a4f6df7400e0c9bb"
)
EXPECTED_RESULT_FINGERPRINT: str | None = (
    "9bb2553b02b0b2d59faabb352cd19dcae719f01ab71ea9edd398d70cc1ba8beb"
)
CHANNEL = "qwen_official_quint8_avx2_protected_dense_tail"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokenizer_root() -> Path:
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Qwen--Qwen3-Embedding-0.6B"
        / "snapshots"
        / "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


def _load_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = _load_json(DEFAULT_PROTOCOL)
    protocol_errors = validate_protocol(protocol)
    if protocol_errors:
        raise ValueError("Stage-1 protocol invalid: " + "; ".join(protocol_errors))
    invalidation = _load_json(DEFAULT_INVALIDATION)
    invalidation_errors = validate_invalidation(invalidation)
    if invalidation_errors:
        raise ValueError("pointer invalidation invalid: " + "; ".join(invalidation_errors))
    stage0 = _load_json(DEFAULT_STAGE0_REPORT)
    stage0_errors = validate_stage0_report(stage0)
    if stage0_errors:
        raise ValueError("Stage-0 report invalid: " + "; ".join(stage0_errors))
    manifest = _load_json(DEFAULT_MANIFEST)
    manifest_errors = validate_manifest(manifest)
    if manifest_errors:
        raise ValueError("stress manifest invalid: " + "; ".join(manifest_errors))
    fixture = load_retrieval_benchmark_fixture(DEFAULT_FIXTURE, repo_root=REPO_ROOT)
    parent = _load_json(DEFAULT_PARENT_REPORT)
    return protocol, manifest, fixture, parent


def _preflight(
    *,
    semantic_db_root: Path,
    protocol: dict[str, Any],
    require_absent_execution: bool,
) -> dict[str, Any]:
    if require_absent_execution:
        for path in (DEFAULT_ATTEMPT, DEFAULT_WORKER_RESULT, DEFAULT_REPORT):
            if path.exists():
                raise RuntimeError(f"Stage-1 execution artifact already exists: {path.name}")
    if not DEFAULT_PYTHON.is_file() or not DEFAULT_WORKER.is_file():
        raise RuntimeError("isolated Stage-1 runtime or worker is missing")
    if not DEFAULT_ARTIFACT.is_file():
        raise RuntimeError("verified ONNX artifact is missing")
    candidate = protocol["candidate"]
    if (
        DEFAULT_ARTIFACT.stat().st_size != candidate["artifact_bytes"]
        or _raw_sha256(DEFAULT_ARTIFACT) != candidate["artifact_sha256"]
    ):
        raise RuntimeError("ONNX artifact failed Stage-1 preflight")
    source_sha = _tree_fingerprint(semantic_db_root)
    index_sha = _tree_fingerprint(DEFAULT_INDEX)
    if source_sha != protocol["source_corpus"]["tree_sha256"]:
        raise RuntimeError("source corpus failed Stage-1 preflight")
    if index_sha != protocol["read_only_parent_index"]["tree_sha256"]:
        raise RuntimeError("parent index_v1 failed Stage-1 preflight")
    tokenizer_root = _tokenizer_root()
    tokenizer = protocol["tokenizer"]
    for filename, size_key, sha_key in (
        ("tokenizer.json", "tokenizer_json_bytes", "tokenizer_json_sha256"),
        ("tokenizer_config.json", "tokenizer_config_bytes", "tokenizer_config_sha256"),
    ):
        path = tokenizer_root / filename
        if (
            not path.is_file()
            or path.stat().st_size != tokenizer[size_key]
            or _raw_sha256(path) != tokenizer[sha_key]
        ):
            raise RuntimeError(f"pinned {filename} failed Stage-1 preflight")
    environment = subprocess.run(
        [
            str(DEFAULT_PYTHON),
            "-I",
            str(REPO_ROOT / "tools" / "retrieval_eval" / "prepare_lighter_runtime_candidate.py"),
            "--environment-only",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    if environment.returncode != 0:
        raise RuntimeError("isolated Stage-1 environment preflight failed")
    return {
        "source_sha256": source_sha,
        "index_sha256": index_sha,
        "artifact_sha256": candidate["artifact_sha256"],
        "tokenizer_root": tokenizer_root,
        "environment_check": environment.stdout.strip(),
    }


def _begin_attempt(preflight: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    receipt = {
        "schema_version": 1,
        "attempt_kind": "retrieval_lighter_runtime_stage1_qwen_v1",
        "status": "running",
        "started_at": _utc_now(),
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "stress_manifest_sha256": sha256_file(DEFAULT_MANIFEST),
        "source_sha256": preflight["source_sha256"],
        "index_sha256": preflight["index_sha256"],
        "artifact_sha256": preflight["artifact_sha256"],
        "maximum_runs": int(protocol["execution"]["maximum_runs"]),
        "process_id": os.getpid(),
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
            "worker_result_persisted": DEFAULT_WORKER_RESULT.exists(),
            "report_persisted": DEFAULT_REPORT.exists(),
        },
    )


def _run_worker(
    *, semantic_db_root: Path, tokenizer_root: Path, timeout_seconds: int
) -> tuple[dict[str, Any], float]:
    command = [
        str(DEFAULT_PYTHON),
        "-I",
        str(DEFAULT_WORKER),
        "--semantic-db-root",
        str(semantic_db_root),
        "--tokenizer-root",
        str(tokenizer_root),
        "--output",
        str(DEFAULT_WORKER_RESULT),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("isolated ONNX worker exceeded the frozen wall-clock limit") from exc
    wall_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode != 0:
        tail = (completed.stdout + completed.stderr).strip().splitlines()[-1:]
        category = tail[0][:240] if tail else "worker_returned_nonzero"
        raise RuntimeError(f"isolated ONNX worker failed: {category}")
    if not DEFAULT_WORKER_RESULT.is_file():
        raise RuntimeError("isolated ONNX worker returned without a retained result")
    return _load_json(DEFAULT_WORKER_RESULT), wall_ms


class _FixedQueryVectors:
    def __init__(self, fixture: dict[str, Any], vectors: dict[str, list[float]]) -> None:
        self._vectors = {
            str(case["query"]): tuple(float(value) for value in vectors[str(case["id"])])
            for case in fixture["cases"]
        }

    def __call__(self, query: str) -> tuple[float, ...]:
        try:
            return self._vectors[str(query)]
        except KeyError as exc:
            raise ValueError("retrieval requested a query outside the frozen Stage-1 fixture") from exc


def _load_parent_index(
    semantic_db_root: Path,
) -> tuple[list[dict], list[tuple[float, ...]], dict[str, Any]]:
    pointer = _load_json(DEFAULT_INDEX / "current.json")
    manifest = _load_json(
        DEFAULT_INDEX / "generations" / str(pointer["build_id"]) / "manifest.json"
    )
    descriptor = EmbeddingProviderDescriptor(**manifest["provider"])
    provider = CallableEmbeddingProvider(
        descriptor,
        lambda _texts: [],
        lambda _text: [1.0] + [0.0] * (descriptor.dimension - 1),
    )
    loaded = EmbeddingIndexStore(semantic_db_root, index_dir=DEFAULT_INDEX).load(provider)
    return loaded.chunks, loaded.embeddings, loaded.manifest


def _vector_parity(
    *,
    worker_vectors: dict[str, list[float]],
    manifest: dict[str, Any],
    parent_vectors: list[tuple[float, ...]],
) -> dict[str, Any]:
    cosines: list[float] = []
    by_id: list[dict[str, Any]] = []
    for item in manifest["items"]:
        chunk_id = str(item["chunk_id"])
        candidate = worker_vectors[chunk_id]
        parent = parent_vectors[int(item["parent_vector_index"])]
        cosine = sum(float(left) * float(right) for left, right in zip(candidate, parent))
        cosines.append(cosine)
        by_id.append({"chunk_id": chunk_id, "cosine": round(cosine, 8)})
    return {
        "count": len(cosines),
        "mean_cosine": round(sum(cosines) / len(cosines), 8),
        "minimum_cosine": round(min(cosines), 8),
        "p05_cosine": round(sorted(cosines)[max(0, math.ceil(0.05 * len(cosines)) - 1)], 8),
        "rows": by_id,
    }


def _evaluate_quality(
    *,
    protocol: dict[str, Any],
    fixture: dict[str, Any],
    chunks: list[dict],
    parent: dict[str, Any],
    parent_embeddings: list[tuple[float, ...]],
    query_vectors: dict[str, list[float]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    semantic_fixture = {**fixture, "ks": [5, 10]}
    _mapping, errors = validate_indexed_corpus(semantic_fixture, chunks)
    if errors:
        raise ValueError("Stage-1 semantic corpus mismatch: " + "; ".join(errors))
    fixed_query = _FixedQueryVectors(fixture, query_vectors)

    def product_factory(selected_chunks: list[dict], whole_library: bool):
        return ProductQuerySparseRetriever(selected_chunks, whole_library=whole_library)

    vector_by_id = {
        str(chunk.get("id") or ""): vector
        for chunk, vector in zip(chunks, parent_embeddings)
    }

    def fusion_factory(selected_chunks: list[dict], whole_library: bool):
        dense = DenseRetriever(
            selected_chunks,
            [vector_by_id[str(chunk.get("id") or "")] for chunk in selected_chunks],
            fixed_query,
            min_score=-1.0,
        )
        frozen = parent["candidate"]
        return ProtectedDenseTailRetriever(
            product_factory(selected_chunks, whole_library),
            dense,
            dense_novelty_quota=int(frozen["dense_novelty_quota"]),
            candidate_multiplier=int(frozen["candidate_multiplier"]),
            protected_sparse_prefix=int(frozen["protected_sparse_prefix"]),
            sparse_tail_window=int(frozen["sparse_tail_window"]),
        )

    product = compact_selection_channel(
        evaluate_candidate_channel(
            semantic_fixture,
            chunks,
            channel_name=PRODUCT_CHANNEL,
            mode_retriever_factory=product_factory,
        )
    )
    candidate = compact_selection_channel(
        evaluate_candidate_channel(
            semantic_fixture,
            chunks,
            channel_name=CHANNEL,
            mode_retriever_factory=fusion_factory,
        )
    )
    acceptance = evaluate_acceptance(
        product,
        candidate,
        minimum_gain=float(protocol["quality"]["minimum_recall_at_10_gain_each_mode"]),
    )
    return product, candidate, acceptance


def _quality_checks(
    *, product: dict[str, Any], candidate: dict[str, Any], parent: dict[str, Any]
) -> dict[str, dict[str, bool]]:
    checks: dict[str, dict[str, bool]] = {}
    for mode in ("scoped", "full_library"):
        product_metrics = product["modes"][mode]["metrics"]
        candidate_metrics = candidate["modes"][mode]["metrics"]
        parent_metrics = parent["channels"][FUSION_CHANNEL]["modes"][mode]["metrics"]
        mode_checks = {
            f"no_regression_{metric}": candidate_metrics[metric] >= parent_metrics[metric]
            for metric in ("recall@5", "recall@10", "mrr", "ndcg@10")
        }
        mode_checks["recall_at_5_equal_to_product"] = (
            candidate_metrics["recall@5"] == product_metrics["recall@5"]
        )
        mode_checks["wrong_source_at_5_equal_to_product"] = (
            candidate_metrics["wrong_source_rate@5"]
            == product_metrics["wrong_source_rate@5"]
        )
        checks[mode] = mode_checks
    return checks


def _evaluate_gate(
    *,
    protocol: dict[str, Any],
    parent: dict[str, Any],
    product: dict[str, Any],
    candidate: dict[str, Any],
    acceptance: dict[str, Any],
    worker: dict[str, Any],
    parity: dict[str, Any],
    subprocess_wall_ms: float,
    source_unchanged: bool,
    index_unchanged: bool,
    artifact_unchanged: bool,
) -> dict[str, Any]:
    mode_checks = _quality_checks(product=product, candidate=candidate, parent=parent)
    runtime = worker["runtime"]
    documents = worker["documents"]
    query = worker["query"]
    gates = protocol["operational_gates"]
    parity_gate = protocol["measurement"]["document_vector_parity"]
    durations = [float(row["duration_ms"]) for row in query["timings"]]
    operational = {
        "subprocess_wall_clock": subprocess_wall_ms
        <= float(gates["maximum_subprocess_wall_clock_seconds"]) * 1000.0,
        "provider_load": float(runtime["provider_load_ms"])
        <= float(gates["maximum_provider_load_ms"]),
        "warm_query_p95": _percentile(durations, 0.95)
        <= float(gates["maximum_warm_query_p95_ms"]),
        "peak_process_rss": float(runtime["peak_process_rss_mib"])
        <= float(gates["maximum_peak_process_rss_mib"]),
        "projected_full_build": float(documents["projected_full_build_ms"])
        <= float(gates["maximum_projected_full_build_ms"]),
        "projected_throughput": float(documents["projected_throughput_chunks_per_second"])
        >= float(gates["minimum_projected_throughput_chunks_per_second"]),
        "query_completion": len(query["timings"]) == 12,
        "query_truncation": int(query["truncated_count"])
        <= int(gates["maximum_truncated_query_count"]),
        "stress_document_completion": len(documents["vectors"]) == 64,
        "stress_document_truncation": int(documents["truncated_stress_count"])
        <= int(gates["maximum_truncated_stress_document_count"]),
        "mean_document_vector_parity": float(parity["mean_cosine"])
        >= float(parity_gate["minimum_mean_cosine"]),
        "worst_document_vector_parity": float(parity["minimum_cosine"])
        >= float(parity_gate["minimum_worst_cosine"]),
        "source_unchanged": source_unchanged,
        "index_unchanged": index_unchanged,
        "artifact_unchanged": artifact_unchanged,
    }
    passes = bool(
        acceptance.get("passes_all")
        and all(all(values.values()) for values in mode_checks.values())
        and all(operational.values())
    )
    return {
        "semantic_acceptance_passes": bool(acceptance.get("passes_all")),
        "mode_checks": mode_checks,
        "operational_checks": operational,
        "passes_all": passes,
    }


def _result_fingerprint(report: dict[str, Any]) -> str:
    projection = {
        "channels": {
            name: _stable_channel_projection(channel)
            for name, channel in (report.get("channels") or {}).items()
        },
        "semantic_acceptance": report.get("semantic_acceptance"),
        "document_vector_parity": report.get("document_vector_parity"),
        "gate": report.get("gate"),
        "decision": report.get("decision"),
    }
    canonical = json.dumps(projection, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _report_from_worker(
    *,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    fixture: dict[str, Any],
    parent: dict[str, Any],
    semantic_db_root: Path,
    worker: dict[str, Any],
    wall_ms: float,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    chunks, parent_vectors, _parent_manifest = _load_parent_index(semantic_db_root)
    parity = _vector_parity(
        worker_vectors=worker["documents"]["vectors"],
        manifest=manifest,
        parent_vectors=parent_vectors,
    )
    product, candidate, acceptance = _evaluate_quality(
        protocol=protocol,
        fixture=fixture,
        chunks=chunks,
        parent=parent,
        parent_embeddings=parent_vectors,
        query_vectors=worker["query"]["vectors"],
    )
    source_after = _tree_fingerprint(semantic_db_root)
    index_after = _tree_fingerprint(DEFAULT_INDEX)
    artifact_after = _raw_sha256(DEFAULT_ARTIFACT)
    gate = _evaluate_gate(
        protocol=protocol,
        parent=parent,
        product=product,
        candidate=candidate,
        acceptance=acceptance,
        worker=worker,
        parity=parity,
        subprocess_wall_ms=wall_ms,
        source_unchanged=receipt["source_sha256"] == source_after,
        index_unchanged=receipt["index_sha256"] == index_after,
        artifact_unchanged=receipt["artifact_sha256"] == artifact_after,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_kind": "retrieval_lighter_runtime_stage1_qwen_v1",
        "measured_at": _utc_now(),
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "stress_manifest_sha256": sha256_file(DEFAULT_MANIFEST),
        "stage_0_report_sha256": sha256_file(DEFAULT_STAGE0_REPORT),
        "accepted_parent_sha256": sha256_file(DEFAULT_PARENT_REPORT),
        "case_ids": [str(case["id"]) for case in fixture["cases"]],
        "candidate": dict(protocol["candidate"]),
        "attempt": {
            "started_at": receipt["started_at"],
            "maximum_runs": receipt["maximum_runs"],
            "subprocess_wall_ms": round(wall_ms, 3),
            "raw_report_persisted_before_validation": True,
        },
        "integrity": {
            "source_before_sha256": receipt["source_sha256"],
            "source_after_sha256": source_after,
            "index_before_sha256": receipt["index_sha256"],
            "index_after_sha256": index_after,
            "artifact_before_sha256": receipt["artifact_sha256"],
            "artifact_after_sha256": artifact_after,
        },
        "runtime": {
            **worker["runtime"],
            "query_timings": worker["query"]["timings"],
            "truncated_query_count": worker["query"]["truncated_count"],
        },
        "document_projection": {
            key: value for key, value in worker["documents"].items() if key != "vectors"
        },
        "document_vector_parity": parity,
        "worker_execution": worker["execution"],
        "channels": {PRODUCT_CHANNEL: product, CHANNEL: candidate},
        "semantic_acceptance": acceptance,
        "gate": gate,
        "decision": {
            "status": (
                protocol["execution"]["pass_decision"]
                if gate["passes_all"]
                else protocol["execution"]["reject_decision"]
            ),
            "single_full_build_allowed": bool(gate["passes_all"]),
            "evaluate_second_candidate": not bool(gate["passes_all"]),
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
        },
    }
    durations = [float(row["duration_ms"]) for row in worker["query"]["timings"]]
    report["summary"] = {
        "provider_load_ms": float(worker["runtime"]["provider_load_ms"]),
        "query_p50_ms": _percentile(durations, 0.50),
        "query_p95_ms": _percentile(durations, 0.95),
        "stress_document_encode_ms": float(worker["documents"]["encode_ms"]),
        "projected_full_build_ms": float(worker["documents"]["projected_full_build_ms"]),
        "projected_throughput_chunks_per_second": float(
            worker["documents"]["projected_throughput_chunks_per_second"]
        ),
        "peak_process_rss_mib": float(worker["runtime"]["peak_process_rss_mib"]),
        "mean_document_vector_cosine": parity["mean_cosine"],
        "minimum_document_vector_cosine": parity["minimum_cosine"],
    }
    report["result_fingerprint"] = _result_fingerprint(report)
    return report


def validate_report(
    report: dict[str, Any],
    *,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    fixture: dict[str, Any],
    parent: dict[str, Any],
    expected_result_fingerprint: str | None = EXPECTED_RESULT_FINGERPRINT,
) -> list[str]:
    errors: list[str] = []
    if (
        report.get("schema_version") != 1
        or report.get("report_kind") != "retrieval_lighter_runtime_stage1_qwen_v1"
    ):
        errors.append("lighter runtime Stage-1 report identity changed")
    if (
        report.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL)
        or report.get("stress_manifest_sha256") != sha256_file(DEFAULT_MANIFEST)
        or report.get("stage_0_report_sha256") != sha256_file(DEFAULT_STAGE0_REPORT)
        or report.get("accepted_parent_sha256") != sha256_file(DEFAULT_PARENT_REPORT)
    ):
        errors.append("lighter runtime Stage-1 report source identity changed")
    if report.get("case_ids") != [str(case["id"]) for case in fixture["cases"]]:
        errors.append("lighter runtime Stage-1 case membership changed")
    if report.get("candidate") != protocol["candidate"]:
        errors.append("lighter runtime Stage-1 candidate changed")
    channels = report.get("channels") or {}
    if list(channels) != [PRODUCT_CHANNEL, CHANNEL]:
        errors.append("lighter runtime Stage-1 channels changed")
        return errors
    if _stable_channel_projection(channels[PRODUCT_CHANNEL]) != _stable_channel_projection(
        parent["channels"][PRODUCT_CHANNEL]
    ):
        errors.append("lighter runtime Stage-1 product control changed")
    acceptance = evaluate_acceptance(
        channels[PRODUCT_CHANNEL],
        channels[CHANNEL],
        minimum_gain=float(protocol["quality"]["minimum_recall_at_10_gain_each_mode"]),
    )
    if report.get("semantic_acceptance") != acceptance:
        errors.append("lighter runtime Stage-1 semantic acceptance arithmetic changed")
    parity = report.get("document_vector_parity") or {}
    if (
        parity.get("count") != 64
        or len(parity.get("rows") or []) != 64
        or [str(row.get("chunk_id") or "") for row in parity.get("rows") or []]
        != [str(item["chunk_id"]) for item in manifest["items"]]
    ):
        errors.append("lighter runtime Stage-1 vector parity membership changed")
    integrity = report.get("integrity") or {}
    source_unchanged = integrity.get("source_before_sha256") == integrity.get("source_after_sha256")
    index_unchanged = integrity.get("index_before_sha256") == integrity.get("index_after_sha256")
    artifact_unchanged = integrity.get("artifact_before_sha256") == integrity.get("artifact_after_sha256")
    worker_projection = {
        "runtime": {
            key: value for key, value in (report.get("runtime") or {}).items() if key != "query_timings"
        },
        "query": {
            "timings": (report.get("runtime") or {}).get("query_timings") or [],
            "truncated_count": int(
                (report.get("runtime") or {}).get("truncated_query_count") or 0
            ),
        },
        "documents": {
            **(report.get("document_projection") or {}),
            "vectors": {str(item["chunk_id"]): [] for item in manifest["items"]},
        },
    }
    expected_gate = _evaluate_gate(
        protocol=protocol,
        parent=parent,
        product=channels[PRODUCT_CHANNEL],
        candidate=channels[CHANNEL],
        acceptance=acceptance,
        worker=worker_projection,
        parity=parity,
        subprocess_wall_ms=float((report.get("attempt") or {}).get("subprocess_wall_ms") or 0.0),
        source_unchanged=source_unchanged,
        index_unchanged=index_unchanged,
        artifact_unchanged=artifact_unchanged,
    )
    if report.get("gate") != expected_gate:
        errors.append("lighter runtime Stage-1 gate arithmetic changed")
    expected_decision = {
        "status": (
            protocol["execution"]["pass_decision"]
            if expected_gate["passes_all"]
            else protocol["execution"]["reject_decision"]
        ),
        "single_full_build_allowed": bool(expected_gate["passes_all"]),
        "evaluate_second_candidate": not bool(expected_gate["passes_all"]),
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }
    if report.get("decision") != expected_decision:
        errors.append("lighter runtime Stage-1 decision changed")
    fingerprint = _result_fingerprint(report)
    if report.get("result_fingerprint") != fingerprint:
        errors.append("lighter runtime Stage-1 result fingerprint is inconsistent")
    if expected_result_fingerprint and fingerprint != expected_result_fingerprint:
        errors.append("pinned lighter runtime Stage-1 result changed")
    return errors


def _failure_report(
    *, protocol: dict[str, Any], receipt: dict[str, Any], exc: Exception
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "report_kind": "retrieval_lighter_runtime_stage1_qwen_v1_execution_failure",
        "failed_at": _utc_now(),
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "attempt": receipt,
        "error_category": type(exc).__name__,
        "decision": {
            "status": protocol["execution"]["reject_decision"],
            "single_full_build_allowed": False,
            "evaluate_second_candidate": True,
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or validate the one-shot Qwen ONNX Stage-1 screen")
    parser.add_argument("--semantic-db-root", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        protocol, manifest, fixture, parent = _load_contracts()
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[FAIL] lighter runtime Stage-1 contracts: {type(exc).__name__}: {exc}")
        return 2
    if args.validate_only:
        try:
            report = _load_json(DEFAULT_REPORT)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[FAIL] lighter runtime Stage-1 report cannot be read: {type(exc).__name__}")
            return 2
        errors = validate_report(
            report,
            protocol=protocol,
            manifest=manifest,
            fixture=fixture,
            parent=parent,
        )
        if EXPECTED_REPORT_SHA256 and sha256_file(DEFAULT_REPORT) != EXPECTED_REPORT_SHA256:
            errors.append("pinned lighter runtime Stage-1 report SHA-256 changed")
        if errors:
            print("[FAIL] " + "; ".join(errors))
            return 1
        print(f"[OK] lighter runtime Stage-1: {report['decision']['status']}")
        return 0
    if args.semantic_db_root is None:
        parser.error("--semantic-db-root is required unless --validate-only is used")
    semantic_db_root = args.semantic_db_root.expanduser().resolve()
    try:
        preflight = _preflight(
            semantic_db_root=semantic_db_root,
            protocol=protocol,
            require_absent_execution=True,
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[FAIL] lighter runtime Stage-1 preflight: {type(exc).__name__}: {exc}")
        return 1
    if args.preflight:
        print(
            "[OK] lighter runtime Stage-1 zero-inference preflight passed: "
            f"{preflight['environment_check']}"
        )
        return 0

    receipt = _begin_attempt(preflight, protocol)
    try:
        worker, wall_ms = _run_worker(
            semantic_db_root=semantic_db_root,
            tokenizer_root=preflight["tokenizer_root"],
            timeout_seconds=int(protocol["operational_gates"]["maximum_subprocess_wall_clock_seconds"]),
        )
        report = _report_from_worker(
            protocol=protocol,
            manifest=manifest,
            fixture=fixture,
            parent=parent,
            semantic_db_root=semantic_db_root,
            worker=worker,
            wall_ms=wall_ms,
            receipt=receipt,
        )
        atomic_write_json(DEFAULT_REPORT, report)
        errors = validate_report(
            report,
            protocol=protocol,
            manifest=manifest,
            fixture=fixture,
            parent=parent,
            expected_result_fingerprint=None,
        )
        _finish_attempt(receipt, status="completed")
        if errors:
            print("[FAIL] retained lighter runtime Stage-1 report: " + "; ".join(errors))
            return 1
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        atomic_write_json(DEFAULT_REPORT, _failure_report(protocol=protocol, receipt=receipt, exc=exc))
        _finish_attempt(receipt, status="failed")
        print(f"[FAIL] retained lighter runtime Stage-1 failure: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps({"summary": report["summary"], "decision": report["decision"]}, indent=2))
    print(f"[OK] wrote {DEFAULT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
