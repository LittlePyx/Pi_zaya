from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from kb.store import atomic_write_json
from tools.retrieval_eval.lighter_runtime_onnx_worker import _raw_sha256, _tree_fingerprint
from tools.retrieval_eval.run_lighter_runtime_stage1_e5_small import DEFAULT_PYTHON
from tools.retrieval_eval.run_product_e5_selector_shadow import (
    _load_json,
    _utc_now,
    summarize_selector_trace,
)
from tools.retrieval_eval.run_product_e5_user_journey import retrieval_snapshot_sha256
from tools.retrieval_eval.scoped_fresh_e5_worker import (
    CANDIDATE_CHANNEL,
    PRODUCT_CHANNEL,
)
from tools.retrieval_eval.validate_scoped_fresh_cases import (
    DEFAULT_FIXTURE,
    validate_fixture,
)
from tools.retrieval_eval.validate_scoped_fresh_e5_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


DEFAULT_WORKER = REPO_ROOT / "tools" / "retrieval_eval" / "scoped_fresh_e5_worker.py"
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_scoped_fresh_e5_v1.json"
EXPECTED_RESULT_FINGERPRINT: str | None = (
    "23a3a7262c8cde3a95ed2cbfbe6ea251318f05327cbf149438e9331fcbea7c76"
)


def _case_regressions(
    product: dict[str, Any],
    candidate: dict[str, Any],
    *,
    metric: str,
) -> list[str]:
    product_rows = {str(row["id"]): row for row in product.get("cases") or []}
    candidate_rows = {str(row["id"]): row for row in candidate.get("cases") or []}
    if set(product_rows) != set(candidate_rows):
        raise ValueError("scoped product/candidate case membership changed")
    return [
        case_id
        for case_id in sorted(product_rows)
        if float(candidate_rows[case_id]["metrics"][metric])
        < float(product_rows[case_id]["metrics"][metric])
    ]


def evaluate_quality_gate(
    channels: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    gates = protocol["quality_gates"]
    product = channels[PRODUCT_CHANNEL]
    candidate = channels[CANDIDATE_CHANNEL]
    product_metrics = product["metrics"]
    candidate_metrics = candidate["metrics"]
    aggregate_checks = {
        f"{metric}_no_regression": float(candidate_metrics[metric])
        >= float(product_metrics[metric])
        for metric in gates["required_no_regression"]
    }
    gain = round(
        float(candidate_metrics["recall@10"]) - float(product_metrics["recall@10"]),
        6,
    )
    aggregate_checks["minimum_recall_at_10_gain"] = gain >= float(
        gates["minimum_recall_at_10_gain"]
    )
    regressions = {
        metric: _case_regressions(product, candidate, metric=metric)
        for metric in gates["maximum_case_regressions"]
    }
    case_checks = {
        f"maximum_{metric}_case_regressions": len(regressions[metric]) <= int(maximum)
        for metric, maximum in gates["maximum_case_regressions"].items()
    }
    checks = {**aggregate_checks, **case_checks}
    return {
        "recall_at_10_gain": gain,
        "case_regressions": regressions,
        "checks": checks,
        "passes_all": all(checks.values()),
    }


def evaluate_runtime_gate(
    worker: dict[str, Any],
    protocol: dict[str, Any],
    integrity: dict[str, Any],
) -> dict[str, Any]:
    gates = protocol["runtime_gates"]
    expected_index = protocol["existing_corpus_and_index"]
    runtime = worker.get("runtime") or {}
    index = worker.get("index") or {}
    execution = worker.get("execution") or {}
    summary = summarize_selector_trace(list(worker.get("selector_trace") or []))
    checks = {
        "subprocess_wall_time": float(runtime.get("subprocess_wall_ms") or math.inf)
        <= float(gates["maximum_subprocess_wall_ms"]),
        "peak_process_rss": float(runtime.get("peak_process_rss_mib") or math.inf)
        <= float(gates["maximum_peak_process_rss_mib"]),
        "warm_selector_p95": summary["warm_selector_p95_ms"]
        <= float(gates["maximum_warm_selector_p95_ms"]),
        "existing_index_identity": index
        == {
            "build_id": expected_index["build_id"],
            "document_count": expected_index["document_count"],
            "parent_count": expected_index["parent_count"],
            "window_count": expected_index["window_count"],
        },
        "zero_index_build_calls": execution.get("index_build_call_count")
        == int(gates["required_index_build_calls"]),
        "selection_call_count": summary["completed_selection_calls"]
        == int(gates["required_selection_calls"]),
        "scoped_only": summary["mode_counts"] == {"scoped": 12, "full_library": 0}
        and execution.get("full_library_call_count") == 0,
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
        "product_sparse_input_unchanged": summary["product_sparse_input_unchanged_rate"]
        >= float(gates["required_product_sparse_input_unchanged_rate"]),
        "working_db_unchanged": integrity.get("working_db_before_sha256")
        == integrity.get("working_db_after_sha256")
        == expected_index["working_db_retrieval_snapshot_sha256"],
        "existing_index_unchanged": integrity.get("index_before_sha256")
        == integrity.get("index_after_sha256")
        == expected_index["index_tree_sha256"],
        "artifacts_unchanged": integrity.get("artifact_before_sha256")
        == integrity.get("artifact_after_sha256")
        == protocol["candidate"]["artifact_tree_sha256"]
        and integrity.get("model_after_sha256") == protocol["candidate"]["model_sha256"]
        and integrity.get("tokenizer_after_sha256")
        == protocol["candidate"]["tokenizer_sha256"],
        "exact_python": runtime.get("python_version")
        == protocol["runtime_environment"]["python_version"],
        "product_remained_disabled": execution.get("product_enabled") is False
        and execution.get("ordinary_user_surface_exposed") is False,
        "network_unused": execution.get("network_used") is False,
    }
    return {"summary": summary, "checks": checks, "passes_all": all(checks.values())}


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
        "candidate_tuning_allowed": False,
        "next_action": (
            "run_selected_scope_portable_packaging_experiment"
            if status == contract["pass_status"]
            else "keep_bm25_and_record_non_tunable_failure"
        ),
    }


def _channel_fingerprint(channel: dict[str, Any]) -> dict[str, Any]:
    return {
        "channel": channel.get("channel"),
        "mode": channel.get("mode"),
        "case_count": channel.get("case_count"),
        "metrics": channel.get("metrics"),
        "cases": [
            {
                "id": row.get("id"),
                "first_relevant_rank": row.get("first_relevant_rank"),
                "ranked_chunk_ids": row.get("ranked_chunk_ids"),
                "metrics": row.get("metrics"),
            }
            for row in channel.get("cases") or []
        ],
    }


def result_fingerprint(report: dict[str, Any]) -> str:
    projection = {
        "index": report.get("index"),
        "selector_trace": [
            {key: value for key, value in row.items() if key != "duration_ms"}
            for row in report.get("selector_trace") or []
        ],
        "channels": {
            name: _channel_fingerprint(channel)
            for name, channel in (report.get("channels") or {}).items()
        },
        "runtime_checks": (report.get("runtime_gate") or {}).get("checks"),
        "quality_gate": report.get("quality_gate"),
        "decision": report.get("decision"),
    }
    raw = json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_report(
    report: dict[str, Any],
    *,
    protocol: dict[str, Any],
    expected_result_fingerprint: str | None = EXPECTED_RESULT_FINGERPRINT,
) -> list[str]:
    errors: list[str] = []
    if (
        report.get("schema_version") != 1
        or report.get("report_kind") != "retrieval_scoped_fresh_e5_v1"
        or report.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL)
        or report.get("fixture_sha256") != sha256_file(DEFAULT_FIXTURE)
    ):
        errors.append("scoped fresh E5 report identity changed")
    worker = {
        "runtime": report.get("runtime") or {},
        "index": report.get("index") or {},
        "selector_trace": report.get("selector_trace") or [],
        "channels": report.get("channels") or {},
        "execution": report.get("execution") or {},
    }
    expected_runtime = evaluate_runtime_gate(worker, protocol, report.get("integrity") or {})
    if report.get("runtime_gate") != expected_runtime:
        errors.append("scoped fresh E5 runtime gate is inconsistent")
    try:
        expected_quality = evaluate_quality_gate(worker["channels"], protocol)
    except (KeyError, TypeError, ValueError):
        expected_quality = {}
        errors.append("scoped fresh E5 channels are malformed")
    if report.get("quality_gate") != expected_quality:
        errors.append("scoped fresh E5 quality gate is inconsistent")
    expected_decision = _decision(protocol, expected_runtime, expected_quality)
    if report.get("decision") != expected_decision:
        errors.append("scoped fresh E5 decision is inconsistent")
    fingerprint = result_fingerprint(report)
    if report.get("result_fingerprint") != fingerprint:
        errors.append("scoped fresh E5 result fingerprint is inconsistent")
    if expected_result_fingerprint and fingerprint != expected_result_fingerprint:
        errors.append("pinned scoped fresh E5 result changed")
    return errors


def _preflight(protocol: dict[str, Any], *, require_fresh_output: bool) -> dict[str, Path]:
    errors = validate_protocol(protocol, inspect_files=True)
    fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    errors.extend(validate_fixture(fixture))
    if errors:
        raise RuntimeError("scoped fresh E5 preflight invalid: " + "; ".join(errors))
    execution = protocol["execution"]
    result_root = REPO_ROOT / execution["result_root"]
    report_path = REPO_ROOT / execution["report_path"]
    if require_fresh_output and (result_root.exists() or report_path.exists()):
        raise RuntimeError("scoped fresh E5 execution artifacts already exist")

    runtime = protocol["runtime_environment"]
    model_python = REPO_ROOT / runtime["model_runtime_python"]
    dependency_root = REPO_ROOT / runtime["product_dependency_site_packages"]
    rank_path = dependency_root / "rank_bm25.py"
    if (
        not DEFAULT_WORKER.is_file()
        or not model_python.is_file()
        or model_python.resolve() != DEFAULT_PYTHON.resolve()
        or not rank_path.is_file()
        or _raw_sha256(rank_path) != runtime["rank_bm25_source_sha256"].lower()
    ):
        raise RuntimeError("scoped fresh E5 isolated runtime dependency changed")
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(dependency_root)
    check = subprocess.run(
        [
            str(model_python),
            "-c",
            (
                "import importlib.metadata as m,sys;"
                "assert sys.version_info[:3]==(3,10,11);"
                "assert m.version('numpy')=='2.2.6';"
                "assert m.version('onnxruntime')=='1.23.2';"
                "assert m.version('tokenizers')=='0.21.4';"
                "assert m.version('rank-bm25')=='0.2.2';"
                "import rank_bm25"
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
        raise RuntimeError("scoped fresh E5 isolated package versions changed")
    return {
        "result_root": result_root,
        "report_path": report_path,
        "worker_path": REPO_ROOT / execution["worker_result_path"],
        "db_dir": REPO_ROOT / protocol["existing_corpus_and_index"]["working_db_path"],
        "index_dir": REPO_ROOT / protocol["existing_corpus_and_index"]["index_path"],
        "artifact_root": (REPO_ROOT / protocol["candidate"]["model_path"]).parent,
        "model_python": model_python,
        "dependency_root": dependency_root,
    }


def run() -> tuple[dict[str, Any], list[str]]:
    protocol = _load_json(DEFAULT_PROTOCOL)
    paths = _preflight(protocol, require_fresh_output=True)
    paths["result_root"].mkdir(parents=True, exist_ok=False)
    before = {
        "working_db": retrieval_snapshot_sha256(paths["db_dir"]),
        "index": _tree_fingerprint(paths["index_dir"]),
        "artifact": _tree_fingerprint(paths["artifact_root"]),
    }
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(paths["dependency_root"])
    command = [
        str(paths["model_python"]),
        str(DEFAULT_WORKER),
        "--db-dir",
        str(paths["db_dir"]),
        "--index-dir",
        str(paths["index_dir"]),
        "--artifact-root",
        str(paths["artifact_root"]),
        "--fixture",
        str(DEFAULT_FIXTURE),
        "--output",
        str(paths["worker_path"]),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=environment,
            capture_output=True,
            text=True,
            timeout=int(protocol["execution"]["hard_subprocess_timeout_seconds"]),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("scoped fresh E5 worker exceeded hard timeout") from exc
    wall_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode:
        lines = (completed.stdout + "\n" + completed.stderr).strip().splitlines()
        failures = [line for line in lines if "[FAIL]" in line]
        detail = failures[-1:] or lines[-1:]
        raise RuntimeError(detail[0][:300] if detail else "scoped fresh E5 worker failed")
    worker = _load_json(paths["worker_path"])
    if worker.get("worker_kind") != "retrieval_scoped_fresh_e5_worker_v1":
        raise RuntimeError("scoped fresh E5 worker identity changed")

    candidate = protocol["candidate"]
    integrity = {
        "working_db_before_sha256": before["working_db"],
        "working_db_after_sha256": retrieval_snapshot_sha256(paths["db_dir"]),
        "index_before_sha256": before["index"],
        "index_after_sha256": _tree_fingerprint(paths["index_dir"]),
        "artifact_before_sha256": before["artifact"],
        "artifact_after_sha256": _tree_fingerprint(paths["artifact_root"]),
        "model_after_sha256": _raw_sha256(REPO_ROOT / candidate["model_path"]),
        "tokenizer_after_sha256": _raw_sha256(REPO_ROOT / candidate["tokenizer_path"]),
    }
    worker["runtime"]["subprocess_wall_ms"] = round(wall_ms, 3)
    runtime_gate = evaluate_runtime_gate(worker, protocol, integrity)
    quality_gate = evaluate_quality_gate(worker["channels"], protocol)
    decision = _decision(protocol, runtime_gate, quality_gate)
    report = {
        "schema_version": 1,
        "report_kind": "retrieval_scoped_fresh_e5_v1",
        "generated_at": _utc_now(),
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "fixture_sha256": sha256_file(DEFAULT_FIXTURE),
        "source": {
            "case_count": 12,
            "label_count": 21,
            "document_count": 12,
            "chunk_count": 702,
            "scope": "selected_single_paper_only",
            "candidate_tuned_after_sealing": False,
        },
        "runtime": worker["runtime"],
        "index": worker["index"],
        "selector_trace": worker["selector_trace"],
        "channels": worker["channels"],
        "execution": worker["execution"],
        "integrity": integrity,
        "runtime_gate": runtime_gate,
        "quality_gate": quality_gate,
        "decision": decision,
    }
    report["result_fingerprint"] = result_fingerprint(report)
    atomic_write_json(paths["report_path"], report)
    return report, validate_report(report, protocol=protocol)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        protocol = _load_json(DEFAULT_PROTOCOL)
        if args.validate_only:
            _preflight(protocol, require_fresh_output=True)
            print("[OK] scoped fresh E5 replay preflight passed; candidate not executed")
            return 0
        report, errors = run()
    except Exception as exc:
        print(f"[FAIL] scoped fresh E5 replay: {type(exc).__name__}: {exc}")
        return 1
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    status = report["decision"]["status"]
    if status != "fresh_scoped_suite_passed_authorize_packaging_experiment":
        print(f"[REJECT] scoped fresh E5 replay: {status}")
        return 2
    print(f"[OK] scoped fresh E5 replay passed: {report['result_fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
