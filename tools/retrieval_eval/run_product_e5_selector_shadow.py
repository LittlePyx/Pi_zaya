from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_benchmark import validate_indexed_corpus
from kb.retrieval_eval_fixture import sha256_file
from kb.retrieval_semantic_eval_fixture import load_retrieval_benchmark_fixture
from kb.store import atomic_write_json, load_all_chunks
from tools.retrieval_eval.lighter_runtime_onnx_worker import (
    _raw_sha256,
    _tree_fingerprint,
)
from tools.retrieval_eval.run_lighter_runtime_stage1_e5_small import DEFAULT_PYTHON
from tools.retrieval_eval.validate_product_e5_selector_shadow_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


PRODUCT_CHANNEL = "bm25_product_queries_selection_control"
CANDIDATE_CHANNEL = "e5_small_product_selector_candidate"
DEFAULT_WORKER = (
    REPO_ROOT / "tools" / "retrieval_eval" / "product_e5_selector_shadow_worker.py"
)
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_product_e5_selector_shadow_v1.json"
EXPECTED_RESULT_FINGERPRINT: str | None = (
    "2baee02e77afd8e58a23313b01d89e119a1c7e2a42c7390eea48ef1fa58463b7"
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(max(0.0, float(value)) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * max(0.0, min(1.0, percentile))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    weight = position - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 3)


def summarize_selector_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    warm = [row for row in rows if not bool(row.get("cold_start"))]
    count = len(rows)
    return {
        "completed_selection_calls": count,
        "cold_load_count": sum(bool(row.get("cold_start")) for row in rows),
        "warm_cache_hit_count": sum(bool(row.get("index_cache_hit")) for row in warm),
        "fallback_count": sum(bool(row.get("fallback_reason_id")) for row in rows),
        "source_escape_count": sum(
            max(0, int(row.get("source_escape_count") or 0)) for row in rows
        ),
        "protected_prefix_rate": round(
            sum(bool(row.get("protected_prefix_preserved")) for row in rows) / count,
            6,
        )
        if count
        else 0.0,
        "product_sparse_input_unchanged_rate": round(
            sum(bool(row.get("product_sparse_input_unchanged")) for row in rows)
            / count,
            6,
        )
        if count
        else 0.0,
        "warm_selector_p95_ms": _percentile(
            [float(row.get("duration_ms") or 0.0) for row in warm],
            0.95,
        ),
        "dense_novelty_count": sum(
            max(0, int(row.get("dense_novelty_count") or 0)) for row in rows
        ),
        "mode_counts": {
            mode: sum(str(row.get("mode") or "") == mode for row in rows)
            for mode in ("scoped", "full_library")
        },
    }


def evaluate_runtime_gate(
    worker: dict[str, Any],
    protocol: dict[str, Any],
    integrity: dict[str, Any],
) -> dict[str, Any]:
    gates = protocol["runtime_gates"]
    runtime = worker.get("runtime") or {}
    index = worker.get("index") or {}
    execution = worker.get("execution") or {}
    summary = summarize_selector_trace(list(worker.get("selector_trace") or []))
    source_contract = protocol["source_corpus"]
    expected_source_sha = source_contract.get(
        "source_tree_sha256",
        source_contract.get("retrieval_snapshot_sha256", ""),
    )
    source_unchanged = (
        integrity.get("source_before_sha256")
        == integrity.get("source_after_sha256")
        == expected_source_sha
        and integrity.get("working_db_before_sha256")
        == integrity.get("working_db_after_sha256")
    )
    artifact_unchanged = (
        bool(integrity.get("artifact_tree_before_sha256"))
        and integrity.get("artifact_tree_before_sha256")
        == integrity.get("artifact_tree_after_sha256")
        and integrity.get("model_before_sha256")
        == integrity.get("model_after_sha256")
        == protocol["candidate"]["model_sha256"]
        and integrity.get("tokenizer_before_sha256")
        == integrity.get("tokenizer_after_sha256")
        == protocol["candidate"]["tokenizer_sha256"]
    )
    checks = {
        "single_completed_full_build": execution.get("full_build_count") == 1
        and index.get("no_op") is False,
        "full_build_time": float(runtime.get("full_build_ms") or math.inf)
        <= float(gates["maximum_full_build_ms"]),
        "peak_process_rss": float(runtime.get("peak_process_rss_mib") or math.inf)
        <= float(gates["maximum_peak_process_rss_mib"]),
        "warm_selector_p95": summary["warm_selector_p95_ms"]
        <= float(gates["maximum_warm_selector_p95_ms"]),
        "index_bytes": int(index.get("index_bytes") or 0)
        <= int(gates["maximum_index_bytes"]),
        "parent_count": int(index.get("parent_count") or 0)
        == int(gates["required_parent_count"]),
        "window_count": int(index.get("window_count") or 0)
        == int(gates["required_window_count"]),
        "selection_call_count": summary["completed_selection_calls"]
        == int(gates["required_completed_selection_calls"]),
        "cold_load_count": summary["cold_load_count"]
        == int(gates["required_cold_load_count"]),
        "warm_cache_hits": summary["warm_cache_hit_count"]
        >= int(gates["minimum_warm_cache_hit_count"]),
        "fallback_count": summary["fallback_count"]
        <= int(gates["maximum_fallback_count"]),
        "source_escape_count": summary["source_escape_count"]
        <= int(gates["maximum_source_escape_count"]),
        "protected_prefix": summary["protected_prefix_rate"]
        >= float(gates["required_protected_prefix_rate"]),
        "product_sparse_input_unchanged": summary[
            "product_sparse_input_unchanged_rate"
        ]
        >= float(gates["required_product_sparse_input_unchanged_rate"]),
        "atomic_readback": bool(index.get("atomic_readback_passed")),
        "source_unchanged": source_unchanged,
        "artifact_tokenizer_unchanged": artifact_unchanged,
        "product_remained_disabled": execution.get("product_enabled") is False
        and execution.get("ordinary_user_surface_exposed") is False,
        "network_unused": execution.get("network_used") is False,
    }
    return {"summary": summary, "checks": checks, "passes_all": all(checks.values())}


def evaluate_quality_gate(
    channels: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    gates = protocol["quality_gates"]
    product = channels[PRODUCT_CHANNEL]
    candidate = channels[CANDIDATE_CHANNEL]
    modes: dict[str, Any] = {}
    for mode in ("scoped", "full_library"):
        product_metrics = product["modes"][mode]["metrics"]
        candidate_metrics = candidate["modes"][mode]["metrics"]
        recall_gain = round(
            candidate_metrics["recall@10"] - product_metrics["recall@10"],
            6,
        )
        checks = {
            "recall_at_10_gain": recall_gain
            >= float(gates["minimum_recall_at_10_gain_vs_product_each_mode"]),
            "recall_at_5_equal": candidate_metrics["recall@5"]
            == product_metrics["recall@5"],
            "wrong_source_at_5_equal": candidate_metrics["wrong_source_rate@5"]
            == product_metrics["wrong_source_rate@5"],
            "mrr_no_regression": candidate_metrics["mrr"] >= product_metrics["mrr"],
            "ndcg_at_10_no_regression": candidate_metrics["ndcg@10"]
            >= product_metrics["ndcg@10"],
        }
        modes[mode] = {
            "recall_at_10_gain": recall_gain,
            "checks": checks,
            "passes_all": all(checks.values()),
        }
    return {
        "modes": modes,
        "passes_all": all(item["passes_all"] for item in modes.values()),
    }


def _decision(
    protocol: dict[str, Any],
    runtime_gate: dict[str, Any],
    quality_gate: dict[str, Any],
) -> dict[str, Any]:
    contract = protocol["decision"]
    if not runtime_gate["passes_all"]:
        status = contract["runtime_failure_status"]
    elif not quality_gate["passes_all"]:
        status = contract["quality_failure_status"]
    else:
        status = contract["pass_status"]
    return {
        "status": status,
        "product_enabled": False,
        "user_mode_control_visible": False,
        "next_action": (
            "expand_judged_cases"
            if status == contract["pass_status"]
            else "keep_bm25_and_diagnose_failed_gate"
        ),
    }


def _result_fingerprint(report: dict[str, Any]) -> str:
    projection = {
        "index": {
            key: report["index"].get(key)
            for key in (
                "document_count",
                "parent_count",
                "window_count",
                "embedded_parent_count",
                "embedded_window_count",
                "reused_parent_count",
                "reused_window_count",
                "no_op",
                "atomic_readback_passed",
            )
        },
        "selector_trace": [
            {key: row.get(key) for key in row if key != "duration_ms"}
            for row in report["selector_trace"]
        ],
        "channels": report["channels"],
        "runtime_gate": {
            "checks": report["runtime_gate"]["checks"],
            "passes_all": report["runtime_gate"]["passes_all"],
        },
        "quality_gate": report["quality_gate"],
        "decision": report["decision"],
    }
    canonical = json.dumps(projection, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_report(
    report: dict[str, Any],
    *,
    protocol: dict[str, Any],
    expected_result_fingerprint: str | None = EXPECTED_RESULT_FINGERPRINT,
) -> list[str]:
    errors: list[str] = []
    if (
        report.get("schema_version") != 1
        or report.get("report_kind")
        != "retrieval_product_e5_selector_shadow_v1"
    ):
        errors.append("product E5 selector report identity changed")
    if report.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL):
        errors.append("product E5 selector protocol identity changed")
    worker = {
        "runtime": report.get("runtime") or {},
        "index": report.get("index") or {},
        "selector_trace": report.get("selector_trace") or [],
        "channels": report.get("channels") or {},
        "execution": report.get("execution") or {},
    }
    expected_runtime = evaluate_runtime_gate(
        worker,
        protocol,
        report.get("integrity") or {},
    )
    if report.get("runtime_gate") != expected_runtime:
        errors.append("product E5 selector runtime gate is inconsistent")
    try:
        expected_quality = evaluate_quality_gate(worker["channels"], protocol)
    except (KeyError, TypeError):
        expected_quality = {}
        errors.append("product E5 selector channels are malformed")
    if report.get("quality_gate") != expected_quality:
        errors.append("product E5 selector quality gate is inconsistent")
    expected_decision = _decision(protocol, expected_runtime, expected_quality)
    if report.get("decision") != expected_decision:
        errors.append("product E5 selector decision is inconsistent")
    try:
        fingerprint = _result_fingerprint(report)
    except (KeyError, TypeError):
        fingerprint = ""
    if report.get("result_fingerprint") != fingerprint:
        errors.append("product E5 selector result fingerprint is inconsistent")
    if expected_result_fingerprint and fingerprint != expected_result_fingerprint:
        errors.append("pinned product E5 selector result changed")
    return errors


def _preflight(source_db: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    execution = protocol["execution"]
    working_root = REPO_ROOT / execution["working_root"]
    report_path = REPO_ROOT / execution["report_path"]
    if working_root.exists() or report_path.exists():
        raise RuntimeError("product E5 selector execution artifacts already exist")
    runtime_environment = protocol["runtime_environment"]
    model_python = REPO_ROOT / runtime_environment["model_runtime_python"]
    dependency_root = REPO_ROOT / runtime_environment["product_dependency_site_packages"]
    rank_bm25_path = dependency_root / "rank_bm25.py"
    if (
        not source_db.is_dir()
        or not DEFAULT_PYTHON.is_file()
        or model_python.resolve() != DEFAULT_PYTHON.resolve()
        or not dependency_root.is_dir()
        or not rank_bm25_path.is_file()
        or not DEFAULT_WORKER.is_file()
    ):
        raise RuntimeError("product E5 selector source, Python, or worker is missing")
    if _raw_sha256(rank_bm25_path) != str(runtime_environment["rank_bm25_source_sha256"]).lower():
        raise RuntimeError("product E5 selector rank-bm25 source changed")
    source_sha = _tree_fingerprint(source_db)
    if source_sha != protocol["source_corpus"]["source_tree_sha256"]:
        raise RuntimeError("product E5 selector source corpus changed")
    fixture_path = REPO_ROOT / protocol["source_corpus"]["semantic_fixture_path"]
    fixture = load_retrieval_benchmark_fixture(fixture_path, repo_root=REPO_ROOT)
    chunks = load_all_chunks(source_db)
    _mapping, errors = validate_indexed_corpus(fixture, chunks)
    if errors:
        raise RuntimeError("product E5 selector qrels mismatch: " + "; ".join(errors))
    candidate = protocol["candidate"]
    model_path = REPO_ROOT / candidate["model_path"]
    tokenizer_path = REPO_ROOT / candidate["tokenizer_path"]
    for path, size, sha in (
        (model_path, candidate["model_bytes"], candidate["model_sha256"]),
        (tokenizer_path, candidate["tokenizer_bytes"], candidate["tokenizer_sha256"]),
    ):
        if not path.is_file() or path.stat().st_size != int(size) or _raw_sha256(path) != sha:
            raise RuntimeError(f"product E5 selector artifact changed: {path.name}")
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(dependency_root)
    check = subprocess.run(
        [
            str(DEFAULT_PYTHON),
            "-c",
            (
                "import importlib.metadata as m,sys;"
                "assert sys.version_info[:3]==(3,10,11);"
                "assert m.version('numpy')=='2.2.6';"
                "assert m.version('onnxruntime')=='1.23.2';"
                "assert m.version('tokenizers')=='0.21.4';"
                "assert m.version('rank-bm25')=='0.2.2';"
                "import rank_bm25;"
                f"assert __import__('pathlib').Path(rank_bm25.__file__).resolve()=="
                f"__import__('pathlib').Path({str(rank_bm25_path)!r}).resolve()"
            ),
        ],
        cwd=str(REPO_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if check.returncode:
        raise RuntimeError("product E5 selector isolated runtime changed")
    artifact_root = model_path.parent
    return {
        "source_sha256": source_sha,
        "fixture_path": fixture_path,
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "artifact_root": artifact_root,
        "artifact_tree_sha256": _tree_fingerprint(artifact_root),
        "product_dependency_site_packages": dependency_root,
    }


def run(source_db: Path) -> tuple[dict[str, Any], list[str]]:
    protocol = _load_json(DEFAULT_PROTOCOL)
    protocol_errors = validate_protocol(protocol, inspect_implementation_files=True)
    if protocol_errors:
        raise RuntimeError("product E5 selector protocol invalid: " + "; ".join(protocol_errors))
    preflight = _preflight(source_db, protocol)
    execution = protocol["execution"]
    working_root = REPO_ROOT / execution["working_root"]
    working_db = REPO_ROOT / execution["working_db_path"]
    index_dir = REPO_ROOT / execution["product_index_path"]
    attempt_path = REPO_ROOT / execution["attempt_receipt_path"]
    worker_path = REPO_ROOT / execution["worker_result_path"]
    report_path = REPO_ROOT / execution["report_path"]

    working_root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source_db, working_db)
    working_before = _tree_fingerprint(working_db)
    if working_before != preflight["source_sha256"]:
        raise RuntimeError("product E5 selector working copy changed during copy")
    atomic_write_json(
        attempt_path,
        {
            "schema_version": 1,
            "attempt_kind": "retrieval_product_e5_selector_shadow_v1",
            "started_at": _utc_now(),
            "maximum_completed_full_builds": 1,
            "source_sha256": preflight["source_sha256"],
            "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        },
    )

    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(preflight["product_dependency_site_packages"])
    command = [
        str(DEFAULT_PYTHON),
        str(DEFAULT_WORKER),
        "--db-dir",
        str(working_db),
        "--index-dir",
        str(index_dir),
        "--artifact-root",
        str(preflight["artifact_root"]),
        "--fixture",
        str(preflight["fixture_path"]),
        "--output",
        str(worker_path),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=environment,
            capture_output=True,
            text=True,
            timeout=int(execution["hard_subprocess_timeout_seconds"]),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("product E5 selector worker exceeded hard timeout") from exc
    wall_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode:
        lines = (completed.stdout + "\n" + completed.stderr).strip().splitlines()
        failures = [line for line in lines if "[FAIL]" in line]
        detail = failures[-1:] or lines[-1:]
        raise RuntimeError(detail[0][:240] if detail else "product E5 selector worker failed")
    worker = _load_json(worker_path)
    if (
        worker.get("schema_version") != 1
        or worker.get("worker_kind")
        != "retrieval_product_e5_selector_shadow_worker_v1"
    ):
        raise RuntimeError("product E5 selector worker identity changed")

    integrity = {
        "source_before_sha256": preflight["source_sha256"],
        "source_after_sha256": _tree_fingerprint(source_db),
        "working_db_before_sha256": working_before,
        "working_db_after_sha256": _tree_fingerprint(working_db),
        "artifact_tree_before_sha256": preflight["artifact_tree_sha256"],
        "artifact_tree_after_sha256": _tree_fingerprint(preflight["artifact_root"]),
        "model_before_sha256": protocol["candidate"]["model_sha256"],
        "model_after_sha256": _raw_sha256(preflight["model_path"]),
        "tokenizer_before_sha256": protocol["candidate"]["tokenizer_sha256"],
        "tokenizer_after_sha256": _raw_sha256(preflight["tokenizer_path"]),
    }
    runtime_gate = evaluate_runtime_gate(worker, protocol, integrity)
    quality_gate = evaluate_quality_gate(worker["channels"], protocol)
    decision = _decision(protocol, runtime_gate, quality_gate)
    report = {
        "schema_version": 1,
        "report_kind": "retrieval_product_e5_selector_shadow_v1",
        "generated_at": _utc_now(),
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "source": {
            "document_count": protocol["source_corpus"]["document_count"],
            "parent_chunk_count": protocol["source_corpus"]["parent_chunk_count"],
            "semantic_case_count": protocol["source_corpus"]["semantic_case_count"],
            "corpus_fingerprint": protocol["source_corpus"]["corpus_fingerprint"],
        },
        "runtime": {**worker["runtime"], "subprocess_wall_ms": round(wall_ms, 3)},
        "index": worker["index"],
        "selector_trace": worker["selector_trace"],
        "channels": worker["channels"],
        "execution": worker["execution"],
        "integrity": integrity,
        "runtime_gate": runtime_gate,
        "quality_gate": quality_gate,
        "decision": decision,
    }
    report["result_fingerprint"] = _result_fingerprint(report)
    atomic_write_json(report_path, report)
    errors = validate_report(report, protocol=protocol)
    return report, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        report, errors = run(args.source_db_root.resolve())
    except Exception as exc:
        print(f"[FAIL] product E5 selector shadow: {type(exc).__name__}: {exc}")
        return 1
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    status = report["decision"]["status"]
    if status != "product_selector_shadow_passed_expand_judged_cases":
        print(f"[REJECT] product E5 selector shadow: {status}")
        return 2
    print(
        "[OK] product E5 selector shadow passed: "
        f"{report['result_fingerprint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
