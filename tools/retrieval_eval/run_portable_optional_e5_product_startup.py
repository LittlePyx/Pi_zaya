from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from kb.store import atomic_write_json
from tools.retrieval_eval.lighter_runtime_onnx_worker import _raw_sha256, _tree_fingerprint


DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "retrieval_portable_e5_product_startup_protocol_v2.json"
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_portable_e5_product_startup_v2.json"
DEFAULT_WORKER = (
    REPO_ROOT / "tools" / "retrieval_eval" / "portable_optional_e5_product_startup_worker.py"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _preflight(protocol: dict[str, Any], *, require_fresh_output: bool) -> dict[str, Path]:
    from tools.retrieval_eval.validate_portable_e5_product_startup_protocol import (
        validate_protocol,
    )

    errors = validate_protocol(protocol, inspect_runtime_files=True)
    if errors:
        raise RuntimeError("portable E5 product-startup protocol invalid: " + "; ".join(errors))
    execution = protocol["execution"]
    result_root = REPO_ROOT / execution["result_root"]
    report_path = REPO_ROOT / execution["report_path"]
    if require_fresh_output and (result_root.exists() or report_path.exists()):
        raise RuntimeError("portable E5 product-startup outputs already exist")
    return {
        "result_root": result_root,
        "report_path": report_path,
        "base_archive": REPO_ROOT / protocol["base_portable"]["archive_path"],
        "component_archive": REPO_ROOT / protocol["component"]["archive_path"],
        "base_extract_root": result_root / "base",
        "component_extract_root": result_root / "component",
        "component_root": result_root
        / "component"
        / protocol["component"]["archive_root"],
        "worker_result": result_root / "worker_v2.json",
        "db_dir": REPO_ROOT / execution["working_db_path"],
        "index_dir": REPO_ROOT / execution["existing_index_path"],
        "fixture": REPO_ROOT / execution["fixture_path"],
    }


def evaluate_gate(
    report: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    gates = protocol["gates"]
    inherited = report.get("inherited_v1") or {}
    runtime = report.get("runtime") or {}
    timings = report.get("timings") or {}
    integrity = report.get("integrity") or {}
    execution = report.get("execution") or {}
    queries = report.get("queries") or []
    component_root = str(report.get("component_root") or "").casefold()
    expected_queries = protocol["execution"]["expected_queries"]
    checks = {
        "v1_rejection_preserved": inherited
        == {
            "report_sha256": protocol["source_v1"]["report_sha256"],
            "result_fingerprint": protocol["source_v1"]["result_fingerprint"],
            "decision_status": protocol["source_v1"]["decision_status"],
            "only_failed_gate": "import_time",
        },
        "base_archive_unchanged": integrity.get("base_archive_before_sha256")
        == integrity.get("base_archive_after_sha256")
        == protocol["base_portable"]["archive_sha256"],
        "component_archive_unchanged": integrity.get("component_archive_before_sha256")
        == integrity.get("component_archive_after_sha256")
        == protocol["component"]["archive_sha256"],
        "component_tree_reproduced": integrity.get("extracted_component_tree_sha256")
        == protocol["component"]["component_tree_sha256"],
        "inherited_size_contract": protocol["component"]["raw_bytes"]
        <= gates["maximum_component_raw_bytes"]
        and protocol["component"]["archive_bytes"]
        <= gates["maximum_component_archive_bytes"]
        and protocol["component"]["combined_download_bytes"]
        <= gates["maximum_combined_download_bytes"],
        "fresh_short_extraction": report.get("component_root")
        and component_root.endswith("\\component\\c"),
        "extraction_time": float(timings.get("component_extraction_ms") or float("inf"))
        <= gates["maximum_component_extraction_ms"],
        "worker_wall": float(timings.get("embedded_worker_wall_ms") or float("inf"))
        <= gates["maximum_embedded_worker_wall_ms"],
        "exact_runtime_versions": runtime.get("python_version") == "3.10.11"
        and runtime.get("onnxruntime_version") == "1.23.2"
        and runtime.get("tokenizers_version") == "0.21.4",
        "overlay_modules_used": bool(component_root)
        and component_root in str(runtime.get("onnxruntime_module_path") or "").casefold()
        and component_root in str(runtime.get("tokenizers_module_path") or "").casefold(),
        "product_import_time": float(runtime.get("product_import_ms") or float("inf"))
        <= gates["maximum_product_import_ms"],
        "cold_load_time": float(runtime.get("cold_load_ms") or float("inf"))
        <= gates["maximum_cold_load_ms"],
        "warm_query_p95": float(runtime.get("query_p95_ms") or float("inf"))
        <= gates["maximum_warm_query_p95_ms"],
        "peak_process_rss": float(runtime.get("peak_process_rss_mib") or float("inf"))
        <= gates["maximum_peak_process_rss_mib"],
        "cold_index_load": report.get("cold_load")
        == {"cold_start": True, "index_cache_hit": False},
        "ranking_unchanged": len(queries) == len(expected_queries)
        and all(
            row.get("case_id") == expected.get("case_id")
            and row.get("ranked_chunk_ids") == expected.get("ranked_chunk_ids")
            and row.get("first_relevant_rank") == expected.get("first_relevant_rank")
            for row, expected in zip(queries, expected_queries)
        ),
        "selected_scope_clean": all(row.get("used_mode") == "hybrid_local" for row in queries)
        and all(not row.get("fallback_reason_id") for row in queries)
        and all(int(row.get("source_escape_count") or 0) == 0 for row in queries),
        "product_timing_boundary": execution.get(
            "evaluation_helper_imported_before_startup_timing"
        )
        is False,
        "zero_build_network_or_enablement": execution.get("index_build_call_count") == 0
        and execution.get("network_used") is False
        and execution.get("product_enabled") is False
        and execution.get("ordinary_user_surface_exposed") is False,
    }
    return {"checks": checks, "passes_all": all(checks.values())}


def _decision(protocol: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    decision = protocol["decision"]
    return {
        "status": (
            decision["pass_status"] if gate["passes_all"] else decision["failure_status"]
        ),
        "v1_verdict_changed": False,
        "bundle_with_base_application": False,
        "product_enabled": False,
        "user_mode_control_visible": False,
        "next_action": (
            "design_selected_scope_component_lifecycle"
            if gate["passes_all"]
            else "keep_bm25_and_screen_smaller_delivery_candidate"
        ),
    }


def result_fingerprint(report: dict[str, Any]) -> str:
    projection = {
        "inherited_v1": report.get("inherited_v1"),
        "runtime_identity": {
            key: (report.get("runtime") or {}).get(key)
            for key in ("python_version", "onnxruntime_version", "tokenizers_version")
        },
        "cold_load": report.get("cold_load"),
        "queries": [
            {key: value for key, value in row.items() if key != "duration_ms"}
            for row in report.get("queries") or []
        ],
        "execution": report.get("execution"),
        "integrity": report.get("integrity"),
        "gate_checks": (report.get("gate") or {}).get("checks"),
        "decision": report.get("decision"),
    }
    raw = json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_report(
    report: dict[str, Any],
    *,
    protocol: dict[str, Any],
    expected_result_fingerprint: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if (
        report.get("schema_version") != 1
        or report.get("report_kind") != "retrieval_portable_e5_product_startup_v2"
        or report.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL)
    ):
        errors.append("portable E5 product-startup report identity changed")
    gate = evaluate_gate(report, protocol)
    if report.get("gate") != gate:
        errors.append("portable E5 product-startup gate is inconsistent")
    decision = _decision(protocol, gate)
    if report.get("decision") != decision:
        errors.append("portable E5 product-startup decision is inconsistent")
    fingerprint = result_fingerprint(report)
    if report.get("result_fingerprint") != fingerprint:
        errors.append("portable E5 product-startup fingerprint is inconsistent")
    if expected_result_fingerprint and fingerprint != expected_result_fingerprint:
        errors.append("pinned portable E5 product-startup result changed")
    return errors


def run() -> tuple[dict[str, Any], list[str]]:
    protocol = _load_json(DEFAULT_PROTOCOL)
    paths = _preflight(protocol, require_fresh_output=True)
    paths["result_root"].mkdir(parents=True, exist_ok=False)
    base_before = _raw_sha256(paths["base_archive"])
    component_before = _raw_sha256(paths["component_archive"])

    base_started = time.perf_counter()
    with zipfile.ZipFile(paths["base_archive"]) as archive:
        archive.extractall(paths["base_extract_root"])
    base_extraction_ms = (time.perf_counter() - base_started) * 1000.0
    component_started = time.perf_counter()
    with zipfile.ZipFile(paths["component_archive"]) as archive:
        archive.extractall(paths["component_extract_root"])
    component_extraction_ms = (time.perf_counter() - component_started) * 1000.0

    embedded_python = (
        paths["base_extract_root"]
        / protocol["base_portable"]["archive_root"]
        / "runtime"
        / "python"
        / "python.exe"
    )
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    command = [
        str(embedded_python),
        str(DEFAULT_WORKER),
        "--repo-root",
        str(REPO_ROOT),
        "--component-root",
        str(paths["component_root"]),
        "--db-dir",
        str(paths["db_dir"]),
        "--index-dir",
        str(paths["index_dir"]),
        "--fixture",
        str(paths["fixture"]),
        "--output",
        str(paths["worker_result"]),
    ]
    worker_started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(embedded_python.parent.parent.parent),
            env=environment,
            capture_output=True,
            text=True,
            timeout=protocol["execution"]["hard_subprocess_timeout_seconds"],
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("portable E5 product-startup worker exceeded timeout") from exc
    worker_ms = (time.perf_counter() - worker_started) * 1000.0
    if completed.returncode:
        lines = (completed.stdout + "\n" + completed.stderr).strip().splitlines()
        failures = [line for line in lines if "[FAIL]" in line]
        detail = failures[-1:] or lines[-1:]
        raise RuntimeError(detail[0][:300] if detail else "product-startup worker failed")
    worker = _load_json(paths["worker_result"])
    if worker.get("worker_kind") != "retrieval_portable_optional_e5_product_startup_v2":
        raise RuntimeError("portable E5 product-startup worker identity changed")

    source_v1 = _load_json(REPO_ROOT / protocol["source_v1"]["report_path"])
    failed_checks = [
        key for key, value in source_v1["gate"]["checks"].items() if value is False
    ]
    inherited_v1 = {
        "report_sha256": sha256_file(REPO_ROOT / protocol["source_v1"]["report_path"]),
        "result_fingerprint": source_v1["result_fingerprint"],
        "decision_status": source_v1["decision"]["status"],
        "only_failed_gate": failed_checks[0] if failed_checks == ["import_time"] else None,
    }
    report = {
        "schema_version": 1,
        "report_kind": "retrieval_portable_e5_product_startup_v2",
        "generated_at": _utc_now(),
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "inherited_v1": inherited_v1,
        "component_root": str(paths["component_root"]),
        "timings": {
            "base_extraction_ms": round(base_extraction_ms, 3),
            "component_extraction_ms": round(component_extraction_ms, 3),
            "embedded_worker_wall_ms": round(worker_ms, 3),
        },
        "runtime": worker["runtime"],
        "cold_load": worker["cold_load"],
        "queries": worker["queries"],
        "execution": worker["execution"],
        "integrity": {
            "base_archive_before_sha256": base_before,
            "base_archive_after_sha256": _raw_sha256(paths["base_archive"]),
            "component_archive_before_sha256": component_before,
            "component_archive_after_sha256": _raw_sha256(paths["component_archive"]),
            "extracted_component_tree_sha256": _tree_fingerprint(paths["component_root"]),
        },
    }
    report["gate"] = evaluate_gate(report, protocol)
    report["decision"] = _decision(protocol, report["gate"])
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
            print("[OK] portable E5 product-startup preflight passed; nothing extracted")
            return 0
        report, errors = run()
    except Exception as exc:
        print(f"[FAIL] portable E5 product-startup experiment: {type(exc).__name__}: {exc}")
        return 1
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    if report["decision"]["status"] != protocol["decision"]["pass_status"]:
        print(f"[REJECT] portable E5 product startup: {report['decision']['status']}")
        return 2
    print(f"[OK] portable E5 product startup passed: {report['result_fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
