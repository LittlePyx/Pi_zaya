from __future__ import annotations

import argparse
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

from tools.retrieval_eval.analyze_e5_small_window_census import (
    DEFAULT_REPORT as DEFAULT_CENSUS_REPORT,
    EXPECTED_RESULT_FINGERPRINT as EXPECTED_CENSUS_FINGERPRINT,
    _tree_fingerprint,
    validate_report as validate_census_report,
)
from tools.retrieval_eval.lighter_runtime_onnx_worker import _raw_sha256
from tools.retrieval_eval.validate_lighter_runtime_stage0_e5_small import (
    DEFAULT_REPORT as DEFAULT_STAGE0_REPORT,
    validate_report as validate_stage0_report,
)
from tools.retrieval_eval.validate_lighter_runtime_stage1_e5_small_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)
from tools.retrieval_eval.validate_lighter_runtime_stress_manifest import (
    DEFAULT_MANIFEST,
    validate_manifest,
)


DEFAULT_ARTIFACT = (
    REPO_ROOT
    / "test_results"
    / "retrieval_lighter_runtime_v1"
    / "artifacts"
    / "multilingual_e5_small_official_onnx_v1"
    / "model.onnx"
)
DEFAULT_TOKENIZER_ROOT = DEFAULT_ARTIFACT.parent / "tokenizer"
DEFAULT_PYTHON = (
    REPO_ROOT / "test_results" / "retrieval_lighter_runtime_v1" / ".venv" / "Scripts" / "python.exe"
)
DEFAULT_WORKER = REPO_ROOT / "tools" / "retrieval_eval" / "lighter_runtime_e5_small_worker.py"
DEFAULT_PREPARER = REPO_ROOT / "tools" / "retrieval_eval" / "prepare_lighter_runtime_candidate.py"
DEFAULT_ATTEMPT = (
    REPO_ROOT / "test_results" / "retrieval_lighter_runtime_v1" / "stage1_e5_small_attempt_v1.json"
)
DEFAULT_WORKER_RESULT = (
    REPO_ROOT / "test_results" / "retrieval_lighter_runtime_v1" / "stage1_e5_small_worker_v1.json"
)
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage1_e5_small_v1.json"
EXPECTED_REPORT_SHA256: str | None = (
    "19a42c13a779e4eabb37438d45749485fa6864f85d3c924c6f5a6163be1502cc"
)
EXPECTED_RESULT_FINGERPRINT: str | None = (
    "0750c32c42c09f8f9613734662ff1e6c3c7bee1169c9341b51e5afed778cd645"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized_sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


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


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(max(0.0, float(value)) for value in values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) * percentile + 0.999999) - 1)))
    return round(ordered[index], 3)


def _verify_tokenizer(stage0: dict[str, Any]) -> str:
    rows = list((stage0.get("tokenizer") or {}).get("files") or [])
    if len(rows) != 5:
        raise RuntimeError("E5 Stage-0 tokenizer file set changed")
    for row in rows:
        path = DEFAULT_TOKENIZER_ROOT / str(row["local_name"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or _raw_sha256(path) != str(row["sha256"])
        ):
            raise RuntimeError(f"E5 tokenizer file changed: {path.name}")
    return _tree_fingerprint(DEFAULT_TOKENIZER_ROOT)


def _load_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = _load_json(DEFAULT_PROTOCOL)
    errors = validate_protocol(protocol)
    if errors:
        raise RuntimeError("E5 Stage-1 protocol invalid: " + "; ".join(errors))
    stage0 = _load_json(DEFAULT_STAGE0_REPORT)
    errors = validate_stage0_report(stage0)
    if errors:
        raise RuntimeError("E5 Stage-0 report invalid: " + "; ".join(errors))
    census = _load_json(DEFAULT_CENSUS_REPORT)
    errors = validate_census_report(
        census, expected_result_fingerprint=EXPECTED_CENSUS_FINGERPRINT
    )
    if errors:
        raise RuntimeError("E5 census report invalid: " + "; ".join(errors))
    manifest = _load_json(DEFAULT_MANIFEST)
    errors = validate_manifest(manifest)
    if errors:
        raise RuntimeError("E5 stress manifest invalid: " + "; ".join(errors))
    return protocol, stage0, census, manifest


def _preflight(
    *,
    semantic_db_root: Path,
    protocol: dict[str, Any],
    stage0: dict[str, Any],
    require_absent_execution: bool,
) -> dict[str, Any]:
    if require_absent_execution:
        for path in (DEFAULT_ATTEMPT, DEFAULT_WORKER_RESULT, DEFAULT_REPORT):
            if path.exists():
                raise RuntimeError(f"E5 Stage-1 execution artifact already exists: {path.name}")
    if not DEFAULT_PYTHON.is_file() or not DEFAULT_WORKER.is_file():
        raise RuntimeError("E5 isolated Python or worker is missing")
    candidate = protocol["candidate"]
    if (
        not DEFAULT_ARTIFACT.is_file()
        or DEFAULT_ARTIFACT.stat().st_size != int(candidate["artifact_bytes"])
        or _raw_sha256(DEFAULT_ARTIFACT) != str(candidate["artifact_sha256"])
    ):
        raise RuntimeError("E5 artifact identity changed")
    source_sha = _tree_fingerprint(semantic_db_root)
    if source_sha != protocol["source_corpus"]["tree_sha256"]:
        raise RuntimeError("E5 source corpus identity changed")
    tokenizer_sha = _verify_tokenizer(stage0)
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
    if check.returncode != 0:
        raise RuntimeError("E5 isolated environment preflight failed")
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
        raise RuntimeError("E5 isolated worker exceeded the hard timeout") from exc
    wall_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode != 0:
        tail = (completed.stdout + "\n" + completed.stderr).strip().splitlines()[-1:]
        raise RuntimeError(tail[0][:240] if tail else "E5 worker returned nonzero")
    if not DEFAULT_WORKER_RESULT.is_file():
        raise RuntimeError("E5 worker completed without a retained result")
    return _load_json(DEFAULT_WORKER_RESULT), wall_ms


def _unit_vector_summary_passes(summary: dict[str, Any], *, count: int) -> bool:
    return bool(
        summary.get("count") == count
        and summary.get("dimension") == 384
        and 0.99999 <= float(summary.get("minimum_norm") or 0.0) <= 1.00001
        and 0.99999 <= float(summary.get("maximum_norm") or 0.0) <= 1.00001
        and len(str(summary.get("matrix_sha256") or "")) == 64
    )


def _evaluate_gate(
    *,
    protocol: dict[str, Any],
    worker: dict[str, Any],
    subprocess_wall_ms: float,
    source_unchanged: bool,
    artifact_unchanged: bool,
    tokenizer_unchanged: bool,
) -> dict[str, Any]:
    gates = protocol["operational_gates"]
    measurement = protocol["measurement"]
    runtime = worker.get("runtime") or {}
    query = worker.get("query") or {}
    documents = worker.get("documents") or {}
    execution = worker.get("execution") or {}
    durations = [float(row["duration_ms"]) for row in query.get("timings") or []]
    checks = {
        "subprocess_wall_clock": subprocess_wall_ms
        <= float(gates["maximum_subprocess_wall_clock_seconds"]) * 1000.0,
        "provider_load": float(runtime.get("provider_load_ms") or float("inf"))
        <= float(gates["maximum_provider_load_ms"]),
        "warm_query_p95": _percentile(durations, 0.95)
        <= float(gates["maximum_warm_query_p95_ms"]),
        "peak_process_rss": float(runtime.get("peak_process_rss_mib") or float("inf"))
        <= float(gates["maximum_peak_process_rss_mib"]),
        "projected_full_build": float(
            documents.get("projected_full_build_ms") or float("inf")
        )
        <= float(gates["maximum_projected_full_build_ms"]),
        "projected_parent_throughput": float(
            documents.get("projected_parent_chunks_per_second") or 0.0
        )
        >= float(gates["minimum_projected_parent_chunks_per_second"]),
        "query_completion": len(query.get("timings") or [])
        == int(measurement["timed_query_count"]),
        "query_truncation": int(query.get("truncated_count") or 0)
        <= int(gates["maximum_truncated_query_count"]),
        "exact_parent_count": documents.get("parent_count")
        == int(measurement["full_parent_count"]),
        "exact_full_window_count": documents.get("full_window_count")
        == int(measurement["full_window_count"]),
        "exact_stress_parent_count": documents.get("stress_parent_count")
        == int(measurement["stress_parent_count"]),
        "exact_stress_window_count": documents.get("stress_window_count")
        == int(measurement["stress_window_count"]),
        "parent_truncation": int(documents.get("truncated_parent_count") or 0)
        <= int(gates["maximum_truncated_parent_count"]),
        "content_coverage": int(documents.get("uncovered_content_token_count") or 0)
        <= int(gates["maximum_uncovered_content_token_count"]),
        "maximum_input_tokens": int(documents.get("maximum_input_tokens") or 0)
        <= int(gates["maximum_input_tokens"]),
        "query_unit_vectors": _unit_vector_summary_passes(
            query.get("vectors") or {}, count=int(measurement["timed_query_count"])
        ),
        "stress_unit_vectors": _unit_vector_summary_passes(
            documents.get("vectors") or {}, count=int(measurement["stress_window_count"])
        ),
        "worker_execution_boundary": execution.get("full_corpus_windows_embedded") == 0
        and execution.get("quality_evaluations") == 0
        and execution.get("product_enabled") is False,
        "cpu_provider_only": runtime.get("session_providers") == ["CPUExecutionProvider"],
        "source_unchanged": source_unchanged,
        "artifact_unchanged": artifact_unchanged,
        "tokenizer_unchanged": tokenizer_unchanged,
    }
    return {"checks": checks, "passes_all": all(checks.values())}


def _result_fingerprint(report: dict[str, Any]) -> str:
    projection = {
        "query_vectors": (report.get("query") or {}).get("vectors"),
        "document_vectors": (report.get("document_projection") or {}).get("vectors"),
        "stress_work": (report.get("document_projection") or {}).get("stress_work"),
        "full_corpus_work": (report.get("document_projection") or {}).get(
            "full_corpus_work"
        ),
        "gate": report.get("gate"),
        "decision": report.get("decision"),
    }
    canonical = json.dumps(projection, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _build_report(
    *,
    protocol: dict[str, Any],
    worker: dict[str, Any],
    receipt: dict[str, Any],
    wall_ms: float,
    semantic_db_root: Path,
    stage0: dict[str, Any],
) -> dict[str, Any]:
    source_after = _tree_fingerprint(semantic_db_root)
    artifact_after = _raw_sha256(DEFAULT_ARTIFACT)
    tokenizer_after = _verify_tokenizer(stage0)
    worker_integrity = worker.get("integrity") or {}
    source_unchanged = bool(
        receipt["source_sha256"]
        == worker_integrity.get("source_before_sha256")
        == worker_integrity.get("source_after_sha256")
        == source_after
    )
    artifact_unchanged = bool(
        receipt["artifact_sha256"]
        == worker_integrity.get("artifact_before_sha256")
        == worker_integrity.get("artifact_after_sha256")
        == artifact_after
    )
    tokenizer_unchanged = bool(
        receipt["tokenizer_tree_sha256"]
        == worker_integrity.get("tokenizer_before_sha256")
        == worker_integrity.get("tokenizer_after_sha256")
        == tokenizer_after
    )
    gate = _evaluate_gate(
        protocol=protocol,
        worker=worker,
        subprocess_wall_ms=wall_ms,
        source_unchanged=source_unchanged,
        artifact_unchanged=artifact_unchanged,
        tokenizer_unchanged=tokenizer_unchanged,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_kind": "retrieval_lighter_runtime_stage1_e5_small_v1",
        "measured_at": _utc_now(),
        "protocol_sha256": _normalized_sha256(DEFAULT_PROTOCOL),
        "stage_0_report_sha256": _normalized_sha256(DEFAULT_STAGE0_REPORT),
        "window_census_report_sha256": _normalized_sha256(DEFAULT_CENSUS_REPORT),
        "stress_manifest_sha256": _normalized_sha256(DEFAULT_MANIFEST),
        "candidate": dict(protocol["candidate"]),
        "attempt": {
            "started_at": receipt["started_at"],
            "maximum_runs": receipt["maximum_runs"],
            "subprocess_wall_ms": round(wall_ms, 3),
            "report_persisted_before_outcome_validation": True,
        },
        "integrity": {
            "source_before_sha256": receipt["source_sha256"],
            "source_after_sha256": source_after,
            "artifact_before_sha256": receipt["artifact_sha256"],
            "artifact_after_sha256": artifact_after,
            "tokenizer_before_sha256": receipt["tokenizer_tree_sha256"],
            "tokenizer_after_sha256": tokenizer_after,
        },
        "runtime": worker["runtime"],
        "query": worker["query"],
        "document_projection": worker["documents"],
        "worker_execution": worker["execution"],
        "gate": gate,
        "decision": {
            "status": (
                protocol["execution"]["pass_decision"]
                if gate["passes_all"]
                else protocol["execution"]["reject_decision"]
            ),
            "single_full_build_allowed": bool(gate["passes_all"]),
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
        },
    }
    durations = [float(row["duration_ms"]) for row in worker["query"]["timings"]]
    report["summary"] = {
        "provider_load_ms": float(worker["runtime"]["provider_load_ms"]),
        "query_p50_ms": _percentile(durations, 0.50),
        "query_p95_ms": _percentile(durations, 0.95),
        "stress_window_encode_ms": float(worker["documents"]["encode_ms"]),
        "projected_full_build_ms": float(
            worker["documents"]["projected_full_build_ms"]
        ),
        "projected_parent_chunks_per_second": float(
            worker["documents"]["projected_parent_chunks_per_second"]
        ),
        "peak_process_rss_mib": float(worker["runtime"]["peak_process_rss_mib"]),
    }
    report["result_fingerprint"] = _result_fingerprint(report)
    return report


def validate_report(
    report: dict[str, Any], *, protocol: dict[str, Any], expected_fingerprint: str | None
) -> list[str]:
    errors: list[str] = []
    if (
        report.get("schema_version") != 1
        or report.get("report_kind") != "retrieval_lighter_runtime_stage1_e5_small_v1"
    ):
        errors.append("E5 Stage-1 report identity changed")
    if (
        report.get("protocol_sha256") != _normalized_sha256(DEFAULT_PROTOCOL)
        or report.get("stage_0_report_sha256") != _normalized_sha256(DEFAULT_STAGE0_REPORT)
        or report.get("window_census_report_sha256")
        != _normalized_sha256(DEFAULT_CENSUS_REPORT)
        or report.get("stress_manifest_sha256") != _normalized_sha256(DEFAULT_MANIFEST)
        or report.get("candidate") != protocol["candidate"]
    ):
        errors.append("E5 Stage-1 report source identity changed")
    integrity = report.get("integrity") or {}
    worker = {
        "runtime": report.get("runtime") or {},
        "query": report.get("query") or {},
        "documents": report.get("document_projection") or {},
        "execution": report.get("worker_execution") or {},
    }
    expected_gate = _evaluate_gate(
        protocol=protocol,
        worker=worker,
        subprocess_wall_ms=float((report.get("attempt") or {}).get("subprocess_wall_ms") or 0.0),
        source_unchanged=integrity.get("source_before_sha256")
        == integrity.get("source_after_sha256"),
        artifact_unchanged=integrity.get("artifact_before_sha256")
        == integrity.get("artifact_after_sha256"),
        tokenizer_unchanged=integrity.get("tokenizer_before_sha256")
        == integrity.get("tokenizer_after_sha256"),
    )
    if report.get("gate") != expected_gate:
        errors.append("E5 Stage-1 gate arithmetic changed")
    expected_decision = {
        "status": (
            protocol["execution"]["pass_decision"]
            if expected_gate["passes_all"]
            else protocol["execution"]["reject_decision"]
        ),
        "single_full_build_allowed": bool(expected_gate["passes_all"]),
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }
    if report.get("decision") != expected_decision:
        errors.append("E5 Stage-1 decision changed")
    fingerprint = _result_fingerprint(report)
    if report.get("result_fingerprint") != fingerprint:
        errors.append("E5 Stage-1 result fingerprint is inconsistent")
    if expected_fingerprint and fingerprint != expected_fingerprint:
        errors.append("pinned E5 Stage-1 result changed")
    return errors


def _failure_report(protocol: dict[str, Any], receipt: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "report_kind": "retrieval_lighter_runtime_stage1_e5_small_v1_execution_failure",
        "failed_at": _utc_now(),
        "protocol_sha256": _normalized_sha256(DEFAULT_PROTOCOL),
        "attempt": receipt,
        "error_category": type(exc).__name__,
        "decision": {
            "status": protocol["execution"]["reject_decision"],
            "single_full_build_allowed": False,
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or validate one-shot E5-small Stage 1")
    parser.add_argument("--semantic-db-root", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        protocol, stage0, _census, _manifest = _load_contracts()
        if args.validate_only:
            report = _load_json(DEFAULT_REPORT)
            errors = validate_report(
                report,
                protocol=protocol,
                expected_fingerprint=EXPECTED_RESULT_FINGERPRINT,
            )
            if EXPECTED_REPORT_SHA256 and _normalized_sha256(DEFAULT_REPORT) != EXPECTED_REPORT_SHA256:
                errors.append("E5 Stage-1 report SHA-256 changed")
            if errors:
                raise RuntimeError("; ".join(errors))
            print(f"[OK] E5-small Stage-1: {report['decision']['status']}")
            return 0
        if args.semantic_db_root is None:
            raise ValueError("--semantic-db-root is required")
        semantic_db_root = args.semantic_db_root.resolve()
        preflight = _preflight(
            semantic_db_root=semantic_db_root,
            protocol=protocol,
            stage0=stage0,
            require_absent_execution=True,
        )
        if args.preflight:
            print("[OK] E5-small Stage-1 zero-inference preflight passed")
            return 0
        receipt = {
            **preflight,
            "started_at": _utc_now(),
            "maximum_runs": int(protocol["execution"]["maximum_runs"]),
            "candidate_id": protocol["candidate"]["candidate_id"],
        }
        _atomic_write(DEFAULT_ATTEMPT, receipt)
        try:
            worker, wall_ms = _run_worker(
                semantic_db_root,
                int(protocol["operational_gates"]["maximum_subprocess_wall_clock_seconds"]),
            )
            report = _build_report(
                protocol=protocol,
                worker=worker,
                receipt=receipt,
                wall_ms=wall_ms,
                semantic_db_root=semantic_db_root,
                stage0=stage0,
            )
            _atomic_write(DEFAULT_REPORT, report)
            errors = validate_report(report, protocol=protocol, expected_fingerprint=None)
            if errors:
                raise RuntimeError("; ".join(errors))
        except Exception as exc:
            if not DEFAULT_REPORT.exists():
                _atomic_write(DEFAULT_REPORT, _failure_report(protocol, receipt, exc))
            raise
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[FAIL] E5-small Stage-1: {type(exc).__name__}: {exc}")
        return 1
    print(f"[OK] E5-small Stage-1: {report['decision']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
