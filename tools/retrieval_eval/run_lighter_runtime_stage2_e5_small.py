from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.parent_collapsed_dense_retriever import ParentCollapsedDenseRetriever
from kb.protected_dense_tail_retriever import ProtectedDenseTailRetriever
from kb.retrieval_benchmark import evaluate_candidate_channel, validate_indexed_corpus
from kb.retrieval_semantic_eval_fixture import load_retrieval_benchmark_fixture
from kb.store import load_all_chunks
from tools.retrieval_eval.lighter_runtime_e5_small_full_build_worker import read_window_index
from tools.retrieval_eval.lighter_runtime_e5_small_worker import _verify_tokenizer_files
from tools.retrieval_eval.lighter_runtime_onnx_worker import _raw_sha256, _tree_fingerprint
from tools.retrieval_eval.product_query_sparse_retriever import (
    ProductQuerySparseRetriever,
)
from tools.retrieval_eval.run_lighter_runtime_stage1_e5_small import (
    DEFAULT_PYTHON,
    DEFAULT_PREPARER,
    DEFAULT_STAGE0_REPORT,
    DEFAULT_TOKENIZER_ROOT,
    _percentile,
)
from tools.retrieval_eval.run_user_journey_fusion_v3_1_semantic_acceptance import (
    evaluate_acceptance,
)
from tools.retrieval_eval.run_user_journey_fusion_v3_selection import (
    PRODUCT_CHANNEL,
    compact_selection_channel,
)
from tools.retrieval_eval.run_cpu_query_compatibility import _stable_channel_projection
from tools.retrieval_eval.validate_lighter_runtime_stage0_e5_small import (
    validate_report as validate_stage0_report,
)
from tools.retrieval_eval.validate_lighter_runtime_stage1_e5_small_protocol import (
    DEFAULT_PROTOCOL as DEFAULT_STAGE1_PROTOCOL,
)
from tools.retrieval_eval.run_lighter_runtime_stage1_e5_small import (
    DEFAULT_REPORT as DEFAULT_STAGE1_REPORT,
    EXPECTED_RESULT_FINGERPRINT as EXPECTED_STAGE1_FINGERPRINT,
    validate_report as validate_stage1_report,
)
from tools.retrieval_eval.validate_lighter_runtime_stage2_e5_small_protocol import (
    DEFAULT_CONTROL,
    DEFAULT_FIXTURE,
    DEFAULT_PROTOCOL,
    validate_protocol,
)


RAW_CHANNEL = "e5_small_window_raw_dense"
FUSION_CHANNEL = "e5_small_window_protected_dense_tail"
DEFAULT_ARTIFACT = (
    REPO_ROOT
    / "test_results"
    / "retrieval_lighter_runtime_v1"
    / "artifacts"
    / "multilingual_e5_small_official_onnx_v1"
    / "model.onnx"
)
DEFAULT_WORKER = (
    REPO_ROOT / "tools" / "retrieval_eval" / "lighter_runtime_e5_small_full_build_worker.py"
)
DEFAULT_INDEX = (
    REPO_ROOT / "test_results" / "retrieval_lighter_runtime_v1" / "stage2_e5_small_index_v1"
)
DEFAULT_ATTEMPT = (
    REPO_ROOT / "test_results" / "retrieval_lighter_runtime_v1" / "stage2_e5_small_attempt_v1.json"
)
DEFAULT_WORKER_RESULT = (
    REPO_ROOT / "test_results" / "retrieval_lighter_runtime_v1" / "stage2_e5_small_worker_v1.json"
)
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage2_e5_small_v1.json"
EXPECTED_REPORT_SHA256: str | None = (
    "9cbccb2413da6ff66725120b529cae0b111adf7fe9947a67cb1e876ee357c73e"
)
EXPECTED_RESULT_FINGERPRINT: str | None = (
    "18fdba85d35503d2f5f60e38191df412482ec7fc1783341b4354fa0f37848a05"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized_sha256(path: Path) -> str:
    value = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


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
            raise ValueError("E5 Stage-2 query is outside the frozen fixture") from exc


def _load_contracts(
    *,
    inspect_implementation_files: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = _load_json(DEFAULT_PROTOCOL)
    errors = validate_protocol(
        protocol,
        inspect_implementation_files=inspect_implementation_files,
    )
    if errors:
        raise RuntimeError("E5 Stage-2 protocol invalid: " + "; ".join(errors))
    stage0 = _load_json(DEFAULT_STAGE0_REPORT)
    errors = validate_stage0_report(stage0)
    if errors:
        raise RuntimeError("E5 Stage-0 report invalid: " + "; ".join(errors))
    stage1_protocol = _load_json(DEFAULT_STAGE1_PROTOCOL)
    stage1 = _load_json(DEFAULT_STAGE1_REPORT)
    errors = validate_stage1_report(
        stage1,
        protocol=stage1_protocol,
        expected_fingerprint=EXPECTED_STAGE1_FINGERPRINT,
    )
    if errors or stage1.get("decision", {}).get("single_full_build_allowed") is not True:
        raise RuntimeError("E5 Stage-1 does not authorize the full build")
    control = _load_json(DEFAULT_CONTROL)
    return protocol, stage0, stage1, control


def _preflight(
    *, semantic_db_root: Path, protocol: dict[str, Any], stage0: dict[str, Any]
) -> dict[str, Any]:
    for path in (DEFAULT_ATTEMPT, DEFAULT_WORKER_RESULT, DEFAULT_REPORT, DEFAULT_INDEX):
        if path.exists():
            raise RuntimeError(f"E5 Stage-2 execution artifact already exists: {path.name}")
    if not DEFAULT_PYTHON.is_file() or not DEFAULT_WORKER.is_file():
        raise RuntimeError("E5 Stage-2 Python or worker is missing")
    candidate = protocol["candidate"]
    if (
        not DEFAULT_ARTIFACT.is_file()
        or DEFAULT_ARTIFACT.stat().st_size != int(candidate["artifact_bytes"])
        or _raw_sha256(DEFAULT_ARTIFACT) != candidate["artifact_sha256"]
    ):
        raise RuntimeError("E5 Stage-2 artifact changed")
    source_sha = _tree_fingerprint(semantic_db_root)
    if source_sha != protocol["source_corpus"]["tree_sha256"]:
        raise RuntimeError("E5 Stage-2 source changed")
    tokenizer_sha = _verify_tokenizer_files(DEFAULT_TOKENIZER_ROOT, stage0)
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    check = subprocess.run(
        [
            str(DEFAULT_PYTHON),
            str(DEFAULT_PREPARER),
            "--candidate-id",
            "multilingual_e5_small_official_onnx_v1",
            "--environment-only",
        ],
        cwd=str(REPO_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if check.returncode:
        raise RuntimeError("E5 Stage-2 environment preflight failed")
    return {
        "source_sha256": source_sha,
        "artifact_sha256": _raw_sha256(DEFAULT_ARTIFACT),
        "tokenizer_tree_sha256": tokenizer_sha,
        "environment_check": check.stdout.strip(),
    }


def _run_worker(semantic_db_root: Path, timeout_seconds: int) -> tuple[dict[str, Any], float]:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    command = [
        str(DEFAULT_PYTHON),
        str(DEFAULT_WORKER),
        "--semantic-db-root",
        str(semantic_db_root),
        "--tokenizer-root",
        str(DEFAULT_TOKENIZER_ROOT),
        "--index-root",
        str(DEFAULT_INDEX),
        "--output",
        str(DEFAULT_WORKER_RESULT),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("E5 Stage-2 worker exceeded hard timeout") from exc
    wall_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode:
        tail = (completed.stdout + "\n" + completed.stderr).strip().splitlines()[-1:]
        raise RuntimeError(tail[0][:240] if tail else "E5 Stage-2 worker returned nonzero")
    return _load_json(DEFAULT_WORKER_RESULT), wall_ms


def _group_window_vectors(
    chunks: list[dict], index: dict[str, Any]
) -> tuple[list[tuple[float, ...]], dict[str, list[tuple[int, tuple[float, ...]]]]]:
    grouped: dict[str, list[tuple[int, tuple[float, ...]]]] = {}
    for row, vector in zip(index["windows"], index["vectors"]):
        grouped.setdefault(str(row["parent_id"]), []).append(
            (int(row["window_ordinal"]), tuple(vector))
        )
    parents: list[tuple[float, ...]] = []
    children: dict[str, list[tuple[int, tuple[float, ...]]]] = {}
    for chunk in chunks:
        parent_id = str(chunk["id"])
        rows = sorted(grouped[parent_id])
        parents.append(rows[0][1])
        children[parent_id] = rows[1:]
    return parents, children


def _evaluate_channels(
    *,
    protocol: dict[str, Any],
    fixture: dict[str, Any],
    chunks: list[dict],
    index: dict[str, Any],
    query_vectors: dict[str, list[float]],
) -> dict[str, Any]:
    semantic_fixture = {**fixture, "ks": [5, 10]}
    _mapping, errors = validate_indexed_corpus(semantic_fixture, chunks)
    if errors:
        raise RuntimeError("E5 Stage-2 fixture/corpus mismatch: " + "; ".join(errors))
    parent_vectors, child_vectors = _group_window_vectors(chunks, index)
    vector_by_id = {
        str(chunk["id"]): vector for chunk, vector in zip(chunks, parent_vectors)
    }
    fixed_query = _FixedQueryVectors(fixture, query_vectors)

    def product_factory(selected_chunks: list[dict], whole_library: bool):
        return ProductQuerySparseRetriever(selected_chunks, whole_library=whole_library)

    def dense_factory(selected_chunks: list[dict], _whole_library: bool):
        selected_ids = {str(chunk["id"]) for chunk in selected_chunks}
        return ParentCollapsedDenseRetriever(
            selected_chunks,
            [vector_by_id[str(chunk["id"])] for chunk in selected_chunks],
            {key: value for key, value in child_vectors.items() if key in selected_ids},
            fixed_query,
            min_score=-1.0,
        )

    tail = protocol["semantic_evaluation"]["protected_tail"]

    def fusion_factory(selected_chunks: list[dict], whole_library: bool):
        return ProtectedDenseTailRetriever(
            product_factory(selected_chunks, whole_library),
            dense_factory(selected_chunks, whole_library),
            dense_novelty_quota=int(tail["dense_novelty_quota"]),
            candidate_multiplier=int(tail["candidate_multiplier"]),
            protected_sparse_prefix=int(tail["protected_sparse_prefix"]),
            sparse_tail_window=int(tail["sparse_tail_window"]),
        )

    channels = {}
    for name, factory in (
        (PRODUCT_CHANNEL, product_factory),
        (RAW_CHANNEL, dense_factory),
        (FUSION_CHANNEL, fusion_factory),
    ):
        channels[name] = compact_selection_channel(
            evaluate_candidate_channel(
                semantic_fixture, chunks, channel_name=name, mode_retriever_factory=factory
            )
        )
    return channels


def _quality_gate(protocol: dict[str, Any], channels: dict[str, Any]) -> dict[str, Any]:
    contract = protocol["semantic_evaluation"]
    modes: dict[str, Any] = {}
    for mode in ("scoped", "full_library"):
        product = channels[PRODUCT_CHANNEL]["modes"][mode]["metrics"]
        candidate = channels[FUSION_CHANNEL]["modes"][mode]["metrics"]
        checks = {
            "recall_at_10_gain": candidate["recall@10"] - product["recall@10"]
            >= float(contract["minimum_protected_recall_at_10_gain_vs_product_each_mode"]),
            "recall_at_5_equal": candidate["recall@5"] == product["recall@5"],
            "wrong_source_at_5_equal": candidate["wrong_source_rate@5"]
            == product["wrong_source_rate@5"],
            "mrr_no_regression": candidate["mrr"] >= product["mrr"],
            "ndcg_at_10_no_regression": candidate["ndcg@10"] >= product["ndcg@10"],
        }
        modes[mode] = {
            "recall_at_10_gain": round(candidate["recall@10"] - product["recall@10"], 6),
            "checks": checks,
            "passes_all": all(checks.values()),
        }
    return {"modes": modes, "passes_all": all(row["passes_all"] for row in modes.values())}


def _runtime_gate(
    *,
    protocol: dict[str, Any],
    worker: dict[str, Any],
    wall_ms: float,
    integrity: dict[str, Any],
) -> dict[str, Any]:
    gates = protocol["runtime_gates"]
    runtime = worker["runtime"]
    representation = worker["representation"]
    index = worker["index"]
    query = worker["query"]
    execution = worker["execution"]
    durations = [float(row["duration_ms"]) for row in query["timings"]]
    checks = {
        "hard_wall_clock": wall_ms
        <= float(protocol["execution"]["hard_subprocess_timeout_seconds"]) * 1000.0,
        "provider_load": runtime["provider_load_ms"] <= gates["maximum_provider_load_ms"],
        "full_build": runtime["full_build_ms"] <= gates["maximum_full_build_ms"],
        "parent_throughput": runtime["parent_chunks_per_second"]
        >= gates["minimum_parent_chunks_per_second"],
        "peak_rss": runtime["peak_process_rss_mib"] <= gates["maximum_peak_process_rss_mib"],
        "query_p95": _percentile(durations, 0.95)
        <= gates["maximum_warm_query_p95_ms"],
        "index_bytes": index["index_bytes"] <= gates["maximum_index_bytes"],
        "query_completion": len(query["timings"]) == 12,
        "query_truncation": query["truncated_count"] == 0,
        "parent_count": representation["parent_count"] == 1673,
        "window_count": representation["window_count"] == 2016,
        "parent_truncation": representation["truncated_parent_count"] == 0,
        "content_coverage": representation["uncovered_content_token_count"] == 0,
        "maximum_input": representation["maximum_input_tokens"] <= 512,
        "document_unit_vectors": 0.99999
        <= representation["vector_summary"]["minimum_norm"]
        <= representation["vector_summary"]["maximum_norm"]
        <= 1.00001,
        "query_unit_vectors": 0.99999
        <= query["vector_summary"]["minimum_norm"]
        <= query["vector_summary"]["maximum_norm"]
        <= 1.00001,
        "atomic_readback": index["atomic_readback_passed"] is True,
        "full_build_count": execution["full_build_count"] == 1,
        "worker_quality_boundary": execution["quality_evaluations"] == 0,
        "cpu_provider": runtime["session_providers"] == ["CPUExecutionProvider"],
        "source_unchanged": integrity["source_before_sha256"]
        == integrity["source_after_sha256"],
        "artifact_unchanged": integrity["artifact_before_sha256"]
        == integrity["artifact_after_sha256"],
        "tokenizer_unchanged": integrity["tokenizer_before_sha256"]
        == integrity["tokenizer_after_sha256"],
    }
    return {"checks": checks, "passes_all": all(checks.values())}


def _result_fingerprint(report: dict[str, Any]) -> str:
    projection = {
        "index": report.get("index"),
        "channels": {
            name: _stable_channel_projection(channel)
            for name, channel in (report.get("channels") or {}).items()
        },
        "runtime_gate": report.get("runtime_gate"),
        "quality_gate": report.get("quality_gate"),
        "decision": report.get("decision"),
    }
    return hashlib.sha256(
        json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _build_report(
    *,
    protocol: dict[str, Any],
    stage0: dict[str, Any],
    control: dict[str, Any],
    worker: dict[str, Any],
    wall_ms: float,
    receipt: dict[str, Any],
    semantic_db_root: Path,
) -> dict[str, Any]:
    index = read_window_index(DEFAULT_INDEX)
    chunks = load_all_chunks(semantic_db_root)
    fixture = load_retrieval_benchmark_fixture(DEFAULT_FIXTURE, repo_root=REPO_ROOT)
    channels = _evaluate_channels(
        protocol=protocol,
        fixture=fixture,
        chunks=chunks,
        index=index,
        query_vectors=worker["query"]["vectors"],
    )
    if _stable_channel_projection(channels[PRODUCT_CHANNEL]) != _stable_channel_projection(
        control["channels"][PRODUCT_CHANNEL]
    ):
        raise RuntimeError("E5 Stage-2 product BM25 control changed")
    integrity = {
        "source_before_sha256": receipt["source_sha256"],
        "source_after_sha256": _tree_fingerprint(semantic_db_root),
        "artifact_before_sha256": receipt["artifact_sha256"],
        "artifact_after_sha256": _raw_sha256(DEFAULT_ARTIFACT),
        "tokenizer_before_sha256": receipt["tokenizer_tree_sha256"],
        "tokenizer_after_sha256": _verify_tokenizer_files(DEFAULT_TOKENIZER_ROOT, stage0),
    }
    runtime_gate = _runtime_gate(
        protocol=protocol, worker=worker, wall_ms=wall_ms, integrity=integrity
    )
    quality_gate = _quality_gate(protocol, channels)
    passed = runtime_gate["passes_all"] and quality_gate["passes_all"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_kind": "retrieval_lighter_runtime_stage2_e5_small_v1",
        "measured_at": _utc_now(),
        "protocol_sha256": _normalized_sha256(DEFAULT_PROTOCOL),
        "stage_1_report_sha256": _normalized_sha256(DEFAULT_STAGE1_REPORT),
        "fixture_sha256": _normalized_sha256(DEFAULT_FIXTURE),
        "accepted_control_sha256": _normalized_sha256(DEFAULT_CONTROL),
        "candidate": protocol["candidate"],
        "attempt": {
            "started_at": receipt["started_at"],
            "maximum_runs": 1,
            "subprocess_wall_ms": round(wall_ms, 3),
            "report_persisted_before_outcome_validation": True,
        },
        "integrity": integrity,
        "runtime": worker["runtime"],
        "representation": worker["representation"],
        "index": {key: value for key, value in worker["index"].items()},
        "query": {
            "timings": worker["query"]["timings"],
            "truncated_count": worker["query"]["truncated_count"],
            "vector_summary": worker["query"]["vector_summary"],
        },
        "worker_execution": worker["execution"],
        "channels": channels,
        "runtime_gate": runtime_gate,
        "quality_gate": quality_gate,
        "decision": {
            "status": (
                protocol["decision"]["pass_status"]
                if passed
                else protocol["decision"]["failure_status"]
            ),
            "passes_all": passed,
            "preparation_design_allowed": passed,
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
        },
    }
    report["result_fingerprint"] = _result_fingerprint(report)
    return report


def validate_report(
    report: dict[str, Any], *, protocol: dict[str, Any], control: dict[str, Any], expected: str | None
) -> list[str]:
    errors: list[str] = []
    if (
        report.get("schema_version") != 1
        or report.get("report_kind") != "retrieval_lighter_runtime_stage2_e5_small_v1"
    ):
        errors.append("E5 Stage-2 report identity changed")
    if (
        report.get("protocol_sha256") != _normalized_sha256(DEFAULT_PROTOCOL)
        or report.get("stage_1_report_sha256") != _normalized_sha256(DEFAULT_STAGE1_REPORT)
        or report.get("fixture_sha256") != _normalized_sha256(DEFAULT_FIXTURE)
        or report.get("accepted_control_sha256") != _normalized_sha256(DEFAULT_CONTROL)
        or report.get("candidate") != protocol["candidate"]
    ):
        errors.append("E5 Stage-2 source identity changed")
    channels = report.get("channels") or {}
    if list(channels) != [PRODUCT_CHANNEL, RAW_CHANNEL, FUSION_CHANNEL]:
        errors.append("E5 Stage-2 channels changed")
        return errors
    if _stable_channel_projection(channels[PRODUCT_CHANNEL]) != _stable_channel_projection(
        control["channels"][PRODUCT_CHANNEL]
    ):
        errors.append("E5 Stage-2 product control changed")
    quality = _quality_gate(protocol, channels)
    if report.get("quality_gate") != quality:
        errors.append("E5 Stage-2 quality arithmetic changed")
    worker = {
        "runtime": report.get("runtime") or {},
        "representation": report.get("representation") or {},
        "index": report.get("index") or {},
        "query": report.get("query") or {},
        "execution": report.get("worker_execution") or {},
    }
    runtime = _runtime_gate(
        protocol=protocol,
        worker=worker,
        wall_ms=float((report.get("attempt") or {}).get("subprocess_wall_ms") or 0.0),
        integrity=report.get("integrity") or {},
    )
    if report.get("runtime_gate") != runtime:
        errors.append("E5 Stage-2 runtime arithmetic changed")
    passed = runtime["passes_all"] and quality["passes_all"]
    expected_decision = {
        "status": (
            protocol["decision"]["pass_status"]
            if passed
            else protocol["decision"]["failure_status"]
        ),
        "passes_all": passed,
        "preparation_design_allowed": passed,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }
    if report.get("decision") != expected_decision:
        errors.append("E5 Stage-2 decision changed")
    fingerprint = _result_fingerprint(report)
    if report.get("result_fingerprint") != fingerprint:
        errors.append("E5 Stage-2 fingerprint is inconsistent")
    if expected and fingerprint != expected:
        errors.append("pinned E5 Stage-2 result changed")
    return errors


def _failure_report(protocol: dict[str, Any], receipt: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "report_kind": "retrieval_lighter_runtime_stage2_e5_small_v1_execution_failure",
        "failed_at": _utc_now(),
        "protocol_sha256": _normalized_sha256(DEFAULT_PROTOCOL),
        "attempt": receipt,
        "error_category": type(exc).__name__,
        "decision": {
            "status": protocol["decision"]["failure_status"],
            "passes_all": False,
            "preparation_design_allowed": False,
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or validate the sole E5 Stage-2 full build")
    parser.add_argument("--semantic-db-root", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        protocol, stage0, _stage1, control = _load_contracts(
            inspect_implementation_files=not args.validate_only,
        )
        if args.validate_only:
            report = _load_json(DEFAULT_REPORT)
            errors = validate_report(
                report,
                protocol=protocol,
                control=control,
                expected=EXPECTED_RESULT_FINGERPRINT,
            )
            if EXPECTED_REPORT_SHA256 and _normalized_sha256(DEFAULT_REPORT) != EXPECTED_REPORT_SHA256:
                errors.append("E5 Stage-2 report SHA-256 changed")
            if errors:
                raise RuntimeError("; ".join(errors))
            print(f"[OK] E5 Stage-2: {report['decision']['status']}")
            return 0
        if args.semantic_db_root is None:
            raise ValueError("--semantic-db-root is required")
        semantic_db_root = args.semantic_db_root.resolve()
        preflight = _preflight(
            semantic_db_root=semantic_db_root, protocol=protocol, stage0=stage0
        )
        if args.preflight:
            print("[OK] E5 Stage-2 zero-inference preflight passed")
            return 0
        receipt = {
            **preflight,
            "started_at": _utc_now(),
            "maximum_runs": 1,
            "candidate_id": protocol["candidate"]["candidate_id"],
        }
        _atomic_write(DEFAULT_ATTEMPT, receipt)
        try:
            worker, wall_ms = _run_worker(
                semantic_db_root,
                int(protocol["execution"]["hard_subprocess_timeout_seconds"]),
            )
            report = _build_report(
                protocol=protocol,
                stage0=stage0,
                control=control,
                worker=worker,
                wall_ms=wall_ms,
                receipt=receipt,
                semantic_db_root=semantic_db_root,
            )
            _atomic_write(DEFAULT_REPORT, report)
            errors = validate_report(
                report, protocol=protocol, control=control, expected=None
            )
            if errors:
                raise RuntimeError("; ".join(errors))
        except Exception as exc:
            if not DEFAULT_REPORT.exists():
                _atomic_write(DEFAULT_REPORT, _failure_report(protocol, receipt, exc))
            raise
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[FAIL] E5 Stage-2: {type(exc).__name__}: {exc}")
        return 1
    print(f"[OK] E5 Stage-2: {report['decision']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
