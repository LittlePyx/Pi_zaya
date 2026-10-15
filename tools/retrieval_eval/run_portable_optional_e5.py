from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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
from tools.retrieval_eval.run_product_e5_selector_shadow import _load_json, _utc_now
from tools.retrieval_eval.validate_scoped_fresh_cases import DEFAULT_FIXTURE


DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "retrieval_portable_optional_e5_protocol_v1.json"
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_portable_optional_e5_v1.json"
DEFAULT_WORKER = REPO_ROOT / "tools" / "retrieval_eval" / "portable_optional_e5_worker.py"
EXPECTED_RESULT_FINGERPRINT: str | None = None


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _copy_component(protocol: dict[str, Any], component_root: Path) -> dict[str, Any]:
    runtime = protocol["optional_component"]["runtime_overlay"]
    source_site = REPO_ROOT / runtime["source_site_packages"]
    target_site = component_root / "python" / "site-packages"
    target_site.mkdir(parents=True, exist_ok=False)
    package_rows: list[dict[str, Any]] = []
    for item in runtime["entries"]:
        source = source_site / item["name"]
        target = target_site / item["name"]
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        package_rows.append(
            {
                "name": item["name"],
                "bytes": _directory_bytes(target) if target.is_dir() else target.stat().st_size,
                "tree_sha256": _tree_fingerprint(target) if target.is_dir() else _raw_sha256(target),
            }
        )
    artifact_source = REPO_ROOT / protocol["optional_component"]["artifact_source_path"]
    artifact_target = component_root / "artifacts"
    shutil.copytree(artifact_source, artifact_target)
    manifest = {
        "schema_version": 1,
        "component_id": protocol["optional_component"]["component_id"],
        "platform": "windows-x64",
        "python_version": "3.10.11",
        "candidate_id": protocol["candidate"]["candidate_id"],
        "model_revision": protocol["candidate"]["model_revision"],
        "runtime_entries": package_rows,
        "artifact_tree_sha256": _tree_fingerprint(artifact_target),
        "contains_user_library_data": False,
        "product_enabled_by_installation": False,
    }
    atomic_write_json(component_root / "component-manifest.json", manifest)
    return manifest


def _zip_component(component_root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive_path,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as output:
        for path in sorted(component_root.rglob("*")):
            if path.is_file():
                arcname = Path(component_root.name) / path.relative_to(component_root)
                output.write(path, arcname.as_posix())


def _preflight(protocol: dict[str, Any], *, require_fresh_output: bool) -> dict[str, Path]:
    from tools.retrieval_eval.validate_portable_optional_e5_protocol import validate_protocol

    errors = validate_protocol(protocol, inspect_runtime_files=True)
    if errors:
        raise RuntimeError("portable optional E5 protocol invalid: " + "; ".join(errors))
    execution = protocol["execution"]
    result_root = REPO_ROOT / execution["result_root"]
    report_path = REPO_ROOT / execution["report_path"]
    if require_fresh_output and (result_root.exists() or report_path.exists()):
        raise RuntimeError("portable optional E5 execution artifacts already exist")
    base_archive = REPO_ROOT / protocol["base_portable"]["archive_path"]
    expected_python = f"{protocol['base_portable']['archive_root']}/runtime/python/python.exe"
    with zipfile.ZipFile(base_archive) as archive:
        if expected_python not in set(archive.namelist()):
            raise RuntimeError("base portable archive lacks its embedded Python runtime")
    return {
        "result_root": result_root,
        "report_path": report_path,
        "base_archive": base_archive,
        "component_root": result_root / "stage" / execution["component_directory_name"],
        "component_archive": result_root / execution["component_archive_name"],
        "base_extract_root": result_root / "base",
        "worker_result": result_root / "worker_v1.json",
        "db_dir": REPO_ROOT / execution["working_db_path"],
        "index_dir": REPO_ROOT / execution["existing_index_path"],
    }


def evaluate_gate(
    report: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    gates = protocol["gates"]
    package = report.get("package") or {}
    timings = report.get("timings") or {}
    runtime = report.get("runtime") or {}
    queries = report.get("queries") or []
    integrity = report.get("integrity") or {}
    component_root = str(package.get("component_root") or "").casefold()
    checks = {
        "base_archive_unchanged": integrity.get("base_archive_before_sha256")
        == integrity.get("base_archive_after_sha256")
        == protocol["base_portable"]["archive_sha256"],
        "artifact_source_unchanged": integrity.get("artifact_before_sha256")
        == integrity.get("artifact_after_sha256")
        == protocol["candidate"]["artifact_tree_sha256"],
        "runtime_source_unchanged": integrity.get("runtime_overlay_before_sha256")
        == integrity.get("runtime_overlay_after_sha256")
        == protocol["optional_component"]["runtime_overlay"]["combined_tree_sha256"],
        "separate_optional_archive": package.get("base_archive_growth_bytes") == 0
        and package.get("separate_optional_archive") is True,
        "component_raw_size": int(package.get("component_raw_bytes") or 0)
        <= int(gates["maximum_component_raw_bytes"]),
        "component_zip_size": int(package.get("component_archive_bytes") or 0)
        <= int(gates["maximum_component_archive_bytes"]),
        "combined_download_size": int(package.get("combined_download_bytes") or 0)
        <= int(gates["maximum_combined_download_bytes"]),
        "no_user_data": package.get("contains_user_library_data") is False,
        "assembly_time": float(timings.get("component_assembly_ms") or float("inf"))
        <= float(gates["maximum_component_assembly_ms"]),
        "compression_time": float(timings.get("component_compression_ms") or float("inf"))
        <= float(gates["maximum_component_compression_ms"]),
        "embedded_worker_wall": float(timings.get("embedded_worker_wall_ms") or float("inf"))
        <= float(gates["maximum_embedded_worker_wall_ms"]),
        "exact_runtime_versions": runtime.get("python_version") == "3.10.11"
        and runtime.get("onnxruntime_version") == "1.23.2"
        and runtime.get("tokenizers_version") == "0.21.4",
        "overlay_modules_used": component_root
        and component_root in str(runtime.get("onnxruntime_module_path") or "").casefold()
        and component_root in str(runtime.get("tokenizers_module_path") or "").casefold(),
        "import_time": float(runtime.get("import_ms") or float("inf"))
        <= float(gates["maximum_import_ms"]),
        "cold_load_time": float(runtime.get("cold_load_ms") or float("inf"))
        <= float(gates["maximum_cold_load_ms"]),
        "warm_query_p95": float(runtime.get("query_p95_ms") or float("inf"))
        <= float(gates["maximum_warm_query_p95_ms"]),
        "peak_process_rss": float(runtime.get("peak_process_rss_mib") or float("inf"))
        <= float(gates["maximum_peak_process_rss_mib"]),
        "cold_index_load": (report.get("cold_load") or {})
        == {"cold_start": True, "index_cache_hit": False},
        "selected_scope_queries": len(queries) == int(gates["required_query_count"])
        and all(row.get("used_mode") == "hybrid_local" for row in queries)
        and all(not row.get("fallback_reason_id") for row in queries)
        and all(int(row.get("source_escape_count") or 0) == 0 for row in queries)
        and all(row.get("first_relevant_rank") is not None for row in queries),
        "zero_build_and_network": (report.get("execution") or {}).get(
            "index_build_call_count"
        )
        == 0
        and (report.get("execution") or {}).get("network_used") is False,
        "product_remained_disabled": (report.get("execution") or {}).get(
            "product_enabled"
        )
        is False
        and (report.get("execution") or {}).get("ordinary_user_surface_exposed")
        is False,
    }
    return {"checks": checks, "passes_all": all(checks.values())}


def _decision(protocol: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    contract = protocol["decision"]
    status = contract["pass_status"] if gate["passes_all"] else contract["failure_status"]
    return {
        "status": status,
        "bundle_with_base_application": False,
        "product_enabled": False,
        "user_mode_control_visible": False,
        "next_action": (
            "design_explicit_selected_scope_opt_in_and_component_lifecycle"
            if gate["passes_all"]
            else "keep_bm25_and_investigate_smaller_runtime_candidate"
        ),
    }


def result_fingerprint(report: dict[str, Any]) -> str:
    projection = {
        "package": {
            key: value
            for key, value in (report.get("package") or {}).items()
            if key != "component_root"
        },
        "runtime_identity": {
            key: (report.get("runtime") or {}).get(key)
            for key in (
                "python_version",
                "onnxruntime_version",
                "tokenizers_version",
            )
        },
        "cold_load": report.get("cold_load"),
        "queries": [
            {key: value for key, value in row.items() if key != "duration_ms"}
            for row in report.get("queries") or []
        ],
        "execution": report.get("execution"),
        "gate_checks": (report.get("gate") or {}).get("checks"),
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
        or report.get("report_kind") != "retrieval_portable_optional_e5_v1"
        or report.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL)
    ):
        errors.append("portable optional E5 report identity changed")
    gate = evaluate_gate(report, protocol)
    if report.get("gate") != gate:
        errors.append("portable optional E5 gate is inconsistent")
    decision = _decision(protocol, gate)
    if report.get("decision") != decision:
        errors.append("portable optional E5 decision is inconsistent")
    fingerprint = result_fingerprint(report)
    if report.get("result_fingerprint") != fingerprint:
        errors.append("portable optional E5 result fingerprint is inconsistent")
    if expected_result_fingerprint and fingerprint != expected_result_fingerprint:
        errors.append("pinned portable optional E5 result changed")
    return errors


def run() -> tuple[dict[str, Any], list[str]]:
    protocol = _load_json(DEFAULT_PROTOCOL)
    paths = _preflight(protocol, require_fresh_output=True)
    paths["result_root"].mkdir(parents=True, exist_ok=False)
    base_before = _raw_sha256(paths["base_archive"])
    artifact_source = REPO_ROOT / protocol["optional_component"]["artifact_source_path"]
    source_site = REPO_ROOT / protocol["optional_component"]["runtime_overlay"][
        "source_site_packages"
    ]
    runtime_entries = protocol["optional_component"]["runtime_overlay"]["entries"]
    runtime_before = hashlib.sha256(
        "".join(item["tree_sha256"] for item in runtime_entries).encode("ascii")
    ).hexdigest()
    artifact_before = _tree_fingerprint(artifact_source)

    extract_started = time.perf_counter()
    with zipfile.ZipFile(paths["base_archive"]) as archive:
        archive.extractall(paths["base_extract_root"])
    extract_ms = (time.perf_counter() - extract_started) * 1000.0

    assembly_started = time.perf_counter()
    component_manifest = _copy_component(protocol, paths["component_root"])
    assembly_ms = (time.perf_counter() - assembly_started) * 1000.0
    component_raw_bytes = _directory_bytes(paths["component_root"])

    compression_started = time.perf_counter()
    _zip_component(paths["component_root"], paths["component_archive"])
    compression_ms = (time.perf_counter() - compression_started) * 1000.0

    bundle = paths["base_extract_root"] / protocol["base_portable"]["archive_root"]
    embedded_python = bundle / "runtime" / "python" / "python.exe"
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
        str(DEFAULT_FIXTURE),
        "--output",
        str(paths["worker_result"]),
    ]
    worker_started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(bundle),
            env=environment,
            capture_output=True,
            text=True,
            timeout=int(protocol["execution"]["hard_subprocess_timeout_seconds"]),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("portable optional E5 worker exceeded hard timeout") from exc
    worker_ms = (time.perf_counter() - worker_started) * 1000.0
    if completed.returncode:
        lines = (completed.stdout + "\n" + completed.stderr).strip().splitlines()
        failures = [line for line in lines if "[FAIL]" in line]
        detail = failures[-1:] or lines[-1:]
        raise RuntimeError(detail[0][:300] if detail else "portable optional E5 worker failed")
    worker = _load_json(paths["worker_result"])
    if worker.get("worker_kind") != "retrieval_portable_optional_e5_worker_v1":
        raise RuntimeError("portable optional E5 worker identity changed")

    runtime_after = hashlib.sha256(
        "".join(
            _tree_fingerprint(source_site / item["name"])
            for item in runtime_entries
        ).encode("ascii")
    ).hexdigest()
    integrity = {
        "base_archive_before_sha256": base_before,
        "base_archive_after_sha256": _raw_sha256(paths["base_archive"]),
        "artifact_before_sha256": artifact_before,
        "artifact_after_sha256": _tree_fingerprint(artifact_source),
        "runtime_overlay_before_sha256": runtime_before,
        "runtime_overlay_after_sha256": runtime_after,
        "component_tree_sha256": _tree_fingerprint(paths["component_root"]),
        "component_archive_sha256": _raw_sha256(paths["component_archive"]),
    }
    base_bytes = paths["base_archive"].stat().st_size
    component_archive_bytes = paths["component_archive"].stat().st_size
    package = {
        "base_archive_bytes": base_bytes,
        "base_archive_growth_bytes": 0,
        "component_raw_bytes": component_raw_bytes,
        "component_archive_bytes": component_archive_bytes,
        "combined_download_bytes": base_bytes + component_archive_bytes,
        "component_to_base_ratio": round(component_archive_bytes / base_bytes, 6),
        "separate_optional_archive": True,
        "contains_user_library_data": component_manifest["contains_user_library_data"],
        "runtime_overlay_bytes": sum(int(item["bytes"]) for item in component_manifest["runtime_entries"]),
        "model_artifact_bytes": protocol["candidate"]["artifact_bytes"],
        "component_root": str(paths["component_root"]),
    }
    report = {
        "schema_version": 1,
        "report_kind": "retrieval_portable_optional_e5_v1",
        "generated_at": _utc_now(),
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "package": package,
        "timings": {
            "base_extraction_ms": round(extract_ms, 3),
            "component_assembly_ms": round(assembly_ms, 3),
            "component_compression_ms": round(compression_ms, 3),
            "embedded_worker_wall_ms": round(worker_ms, 3),
        },
        "runtime": worker["runtime"],
        "cold_load": worker["cold_load"],
        "queries": worker["queries"],
        "execution": worker["execution"],
        "integrity": integrity,
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
            print("[OK] portable optional E5 preflight passed; package not assembled")
            return 0
        report, errors = run()
    except Exception as exc:
        print(f"[FAIL] portable optional E5 experiment: {type(exc).__name__}: {exc}")
        return 1
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    if report["decision"]["status"] != protocol["decision"]["pass_status"]:
        print(f"[REJECT] portable optional E5: {report['decision']['status']}")
        return 2
    print(f"[OK] portable optional E5 passed: {report['result_fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
