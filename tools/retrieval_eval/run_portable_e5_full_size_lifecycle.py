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

from kb.local_semantic_component import (
    LocalSemanticComponentStore,
    _require_windows_path_budget,
)
from kb.retrieval_eval_fixture import sha256_file
from kb.store import atomic_write_json


DEFAULT_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_portable_e5_full_size_lifecycle_protocol_v1.json"
)
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_portable_e5_full_size_lifecycle_v1.json"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fingerprint(report: dict[str, Any]) -> str:
    payload = {
        "report_kind": report.get("report_kind"),
        "protocol_sha256": report.get("protocol_sha256"),
        "runtime": report.get("runtime"),
        "queries": report.get("queries"),
        "stale_repair": report.get("stale_repair"),
        "removal": report.get("removal"),
        "execution": report.get("execution"),
        "gate": report.get("gate"),
        "decision": report.get("decision"),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _paths(protocol: dict[str, Any]) -> dict[str, Path]:
    execution = protocol["execution"]
    result_root = REPO_ROOT / execution["result_root"]
    base_extract = result_root / "base"
    base_root = base_extract / protocol["base_portable"]["archive_root"]
    return {
        "result_root": result_root,
        "report": REPO_ROOT / execution["report_path"],
        "worker": REPO_ROOT / execution["worker_path"],
        "worker_output": result_root / "worker.json",
        "base_archive": REPO_ROOT / protocol["base_portable"]["archive_path"],
        "base_extract": base_extract,
        "embedded_python": base_root / "runtime" / "python" / "python.exe",
        "component_archive": REPO_ROOT / protocol["component"]["archive_path"],
        "source_db": REPO_ROOT / execution["source_db_path"],
        "source_index": REPO_ROOT / execution["source_index_path"],
        "working_db": result_root / "db",
        "working_index": result_root
        / "db"
        / "local_semantic_index"
        / "e5_small_v1",
        "component_store": result_root
        / str(execution.get("component_store_relative_path") or "component_store"),
        "fixture": REPO_ROOT / execution["fixture_path"],
    }


def _preflight(
    protocol: dict[str, Any],
    *,
    require_fresh_output: bool,
) -> dict[str, Path]:
    protocol_id = str(protocol.get("protocol_id") or "")
    if protocol_id == "retrieval_portable_e5_full_size_lifecycle_v3":
        from tools.retrieval_eval.validate_portable_e5_full_size_lifecycle_protocol_v3 import (
            validate_protocol,
        )
    elif protocol_id == "retrieval_portable_e5_full_size_lifecycle_v2":
        from tools.retrieval_eval.validate_portable_e5_full_size_lifecycle_protocol_v2 import (
            validate_protocol,
        )
    else:
        from tools.retrieval_eval.validate_portable_e5_full_size_lifecycle_protocol import (
            validate_protocol,
        )

    errors = validate_protocol(protocol, inspect_runtime_files=True)
    if errors:
        raise RuntimeError("full-size lifecycle protocol invalid: " + "; ".join(errors))
    paths = _paths(protocol)
    if require_fresh_output and (
        paths["result_root"].exists() or paths["report"].exists()
    ):
        raise RuntimeError("full-size lifecycle outputs already exist")
    return paths


def evaluate_gate(
    report: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, bool]:
    gates = protocol["gates"]
    timings = report.get("timings") or {}
    runtime = report.get("runtime") or {}
    execution = report.get("execution") or {}
    repair = report.get("stale_repair") or {}
    removal = report.get("removal") or {}
    expected_queries = {
        str(item["case_id"]): item for item in protocol["execution"]["expected_queries"]
    }
    observed_queries = {str(item.get("case_id") or ""): item for item in report.get("queries") or []}
    ranking_exact = set(observed_queries) == set(expected_queries) and all(
        observed_queries[case_id].get("ranked_chunk_ids")
        == expected.get("ranked_chunk_ids")
        and observed_queries[case_id].get("first_relevant_rank")
        == expected.get("first_relevant_rank")
        for case_id, expected in expected_queries.items()
    )
    return {
        "base_extraction_time": float(timings.get("base_extraction_ms") or 0)
        <= float(gates["maximum_base_extraction_ms"]),
        "install_ready_time": float(timings.get("install_ready_ms") or 0)
        <= float(gates["maximum_install_ready_ms"]),
        "cold_load_time": float(timings.get("cold_load_ms") or 0)
        <= float(gates["maximum_cold_load_ms"]),
        "warm_query_p95": float(timings.get("warm_query_p95_ms") or 0)
        <= float(gates["maximum_warm_query_p95_ms"]),
        "stale_repair_time": float(timings.get("stale_repair_ms") or 0)
        <= float(gates["maximum_stale_repair_ms"]),
        "removal_time": float(timings.get("removal_ms") or 0)
        <= float(gates["maximum_removal_ms"]),
        "peak_memory": float(runtime.get("peak_process_rss_mib") or 0)
        <= float(gates["maximum_peak_process_rss_mib"]),
        "embedded_python_exact": runtime.get("python_version") == "3.10.11",
        "ranking_exact": ranking_exact,
        "hybrid_without_fallback": bool(observed_queries)
        and all(
            item.get("used_mode") == "hybrid_local"
            and not item.get("fallback_reason_id")
            for item in observed_queries.values()
        ),
        "zero_source_escapes": all(
            int(item.get("source_escape_count") or 0) == 0
            for item in observed_queries.values()
        ),
        "stale_repair_completed": repair.get("state_before") == "stale"
        and repair.get("state_after") == "ready"
        and repair.get("lifecycle_state") == "ready"
        and repair.get("index_build_call_count") == 1,
        "no_network": execution.get("download_network_used") is False,
        "one_local_archive_handoff": execution.get("local_archive_adapter_calls") == 1
        and execution.get("transport_factory_calls") == 1,
        "bm25_continuity": execution.get("bm25_available_in_all_public_states") is True,
        "component_disabled_after_removal": removal.get("lifecycle_state")
        == "not_installed"
        and removal.get("component_pointer_present") is False,
        "index_disabled_after_removal": removal.get("index_pointer_present") is False,
        "cleanup_completed_after_worker_exit": removal.get(
            "cleanup_completed_after_worker_exit"
        )
        is True
        and int(removal.get("remaining_component_generation_count") or 0) == 0,
        "product_remains_disabled": execution.get("product_enabled") is False
        and execution.get("ordinary_user_surface_exposed") is False,
        **(
            {
                "windows_path_budget": int(
                    execution.get("storage_layout_max_path_chars") or 0
                )
                <= int(gates["maximum_component_path_chars"])
                and int(execution.get("storage_layout_max_path_chars") or 0) > 0,
                **(
                    {
                        "windows_path_layout_exact": int(
                            execution.get("storage_layout_max_path_chars") or 0
                        )
                        == int(gates["expected_component_layout_max_path_chars"])
                    }
                    if "expected_component_layout_max_path_chars" in gates
                    else {}
                ),
            }
            if "maximum_component_path_chars" in gates
            else {}
        ),
    }


def run(protocol: dict[str, Any], *, protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    paths = _preflight(protocol, require_fresh_output=True)
    paths["result_root"].mkdir(parents=True)
    shutil.copytree(paths["source_db"], paths["working_db"])
    shutil.copytree(paths["source_index"], paths["working_index"])
    extraction_started = time.perf_counter()
    with zipfile.ZipFile(paths["base_archive"]) as archive:
        archive.extractall(paths["base_extract"])
    base_extraction_ms = (time.perf_counter() - extraction_started) * 1000.0
    if not paths["embedded_python"].is_file():
        raise RuntimeError("embedded Python is missing after base extraction")
    generation_root = paths["component_store"] / "generations"
    storage_layout_max_path_chars = _require_windows_path_budget(
        paths["component_archive"],
        archive_root=str(protocol["component"]["archive_root"]),
        staging=generation_root / ".s-0000000000000000",
        generation=generation_root / "0000000000000000",
    )

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        str(paths["embedded_python"]),
        str(paths["worker"]),
        "--repo-root",
        str(REPO_ROOT),
        "--protocol",
        str(protocol_path),
        "--archive",
        str(paths["component_archive"]),
        "--db-dir",
        str(paths["working_db"]),
        "--component-store",
        str(paths["component_store"]),
        "--fixture",
        str(paths["fixture"]),
        "--output",
        str(paths["worker_output"]),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=float(protocol["execution"]["hard_subprocess_timeout_seconds"]),
        check=False,
    )
    worker_wall_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode != 0 or not paths["worker_output"].is_file():
        raise RuntimeError(
            "full-size lifecycle worker failed: "
            + (completed.stdout + "\n" + completed.stderr)[-4000:]
        )
    worker = json.loads(paths["worker_output"].read_text(encoding="utf-8"))
    component_store = LocalSemanticComponentStore(paths["component_store"])
    cleanup_completed = component_store.retry_pending_cleanup()
    generations_dir = paths["component_store"] / "generations"
    remaining_generations = (
        sum(1 for path in generations_dir.iterdir() if path.is_dir())
        if generations_dir.is_dir()
        else 0
    )

    report = {
        "schema_version": 1,
        "report_kind": str(protocol["protocol_id"]),
        "protocol_sha256": sha256_file(protocol_path),
        "timings": {
            "base_extraction_ms": round(base_extraction_ms, 3),
            "worker_wall_ms": round(worker_wall_ms, 3),
            **dict(worker["timings"]),
        },
        "runtime": worker["runtime"],
        "queries": worker["queries"],
        "stale_repair": worker["stale_repair"],
        "removal": {
            **dict(worker["removal"]),
            "cleanup_completed_after_worker_exit": cleanup_completed,
            "remaining_component_generation_count": remaining_generations,
        },
        "execution": {
            **dict(worker["execution"]),
            "storage_layout_max_path_chars": storage_layout_max_path_chars,
        },
    }
    gate = evaluate_gate(report, protocol)
    passed = bool(gate) and all(gate.values())
    report["gate"] = gate
    report["decision"] = {
        "status": (
            protocol["decision"]["pass_status"]
            if passed
            else protocol["decision"]["failure_status"]
        ),
        "all_gates_passed": passed,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
        "next_action": (
            "add_management_api_and_failure_path_e2e"
            if passed
            else "keep_bm25_and_repair_component_lifecycle"
        ),
    }
    report["result_fingerprint"] = _fingerprint(report)
    atomic_write_json(paths["report"], report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    try:
        protocol_path = args.protocol.resolve()
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        _preflight(protocol, require_fresh_output=not args.validate_only)
        if args.validate_only:
            print("[OK] full-size E5 lifecycle preflight passed; no work executed")
            return 0
        report = run(protocol, protocol_path=protocol_path)
    except Exception as exc:
        print(f"[FAIL] full-size E5 lifecycle: {type(exc).__name__}: {exc}")
        return 1
    print(
        f"[OK] full-size E5 lifecycle: {report['decision']['status']} "
        f"fingerprint={report['result_fingerprint']}"
    )
    return 0 if report["decision"]["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
