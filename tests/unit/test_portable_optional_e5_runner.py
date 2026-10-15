from __future__ import annotations

import copy
import json

from tools.retrieval_eval.run_portable_optional_e5 import (
    _decision,
    evaluate_gate,
    result_fingerprint,
)
from tools.retrieval_eval.validate_portable_optional_e5_protocol import DEFAULT_PROTOCOL


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def _passing_report() -> dict:
    protocol = _protocol()
    base = protocol["base_portable"]
    candidate = protocol["candidate"]
    overlay = protocol["optional_component"]["runtime_overlay"]
    component_root = "C:/experiment/Pi_zaya-local-semantic-e5-small-v1-windows-x64"
    return {
        "package": {
            "base_archive_bytes": base["archive_bytes"],
            "base_archive_growth_bytes": 0,
            "component_raw_bytes": 541_500_000,
            "component_archive_bytes": 500_000_000,
            "combined_download_bytes": 594_876_265,
            "component_to_base_ratio": 5.27,
            "separate_optional_archive": True,
            "contains_user_library_data": False,
            "runtime_overlay_bytes": overlay["combined_bytes"],
            "model_artifact_bytes": candidate["artifact_bytes"],
            "component_root": component_root,
        },
        "timings": {
            "base_extraction_ms": 1000.0,
            "component_assembly_ms": 10_000.0,
            "component_compression_ms": 20_000.0,
            "embedded_worker_wall_ms": 15_000.0,
        },
        "runtime": {
            "python_version": "3.10.11",
            "onnxruntime_version": "1.23.2",
            "tokenizers_version": "0.21.4",
            "onnxruntime_module_path": f"{component_root}/python/site-packages/onnxruntime/__init__.py",
            "tokenizers_module_path": f"{component_root}/python/site-packages/tokenizers/__init__.py",
            "import_ms": 500.0,
            "cold_load_ms": 5000.0,
            "query_p95_ms": 100.0,
            "peak_process_rss_mib": 1200.0,
        },
        "cold_load": {"cold_start": True, "index_cache_hit": False},
        "queries": [
            {
                "case_id": case_id,
                "duration_ms": 100.0,
                "used_mode": "hybrid_local",
                "fallback_reason_id": "",
                "source_escape_count": 0,
                "first_relevant_rank": rank,
            }
            for rank, case_id in enumerate(
                protocol["execution"]["selected_scope_query_ids"],
                start=1,
            )
        ],
        "execution": {
            "query_count": 2,
            "index_build_call_count": 0,
            "network_used": False,
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
        },
        "integrity": {
            "base_archive_before_sha256": base["archive_sha256"],
            "base_archive_after_sha256": base["archive_sha256"],
            "artifact_before_sha256": candidate["artifact_tree_sha256"],
            "artifact_after_sha256": candidate["artifact_tree_sha256"],
            "runtime_overlay_before_sha256": overlay["combined_tree_sha256"],
            "runtime_overlay_after_sha256": overlay["combined_tree_sha256"],
        },
    }


def test_portable_optional_e5_gate_passes_separate_component() -> None:
    assert evaluate_gate(_passing_report(), _protocol())["passes_all"] is True


def test_portable_optional_e5_gate_rejects_base_growth() -> None:
    report = _passing_report()
    report["package"]["base_archive_growth_bytes"] = 1
    assert evaluate_gate(report, _protocol())["passes_all"] is False


def test_portable_optional_e5_gate_rejects_source_escape() -> None:
    report = _passing_report()
    report["queries"][0]["source_escape_count"] = 1
    assert evaluate_gate(report, _protocol())["passes_all"] is False


def test_portable_optional_e5_decision_never_enables_product() -> None:
    decision = _decision(_protocol(), {"passes_all": True})
    assert decision["status"].startswith("optional_component_viable")
    assert decision["bundle_with_base_application"] is False
    assert decision["product_enabled"] is False
    assert decision["user_mode_control_visible"] is False


def test_portable_optional_e5_result_fingerprint_ignores_timing_and_paths() -> None:
    report = _passing_report()
    report["gate"] = evaluate_gate(report, _protocol())
    report["decision"] = _decision(_protocol(), report["gate"])
    changed = copy.deepcopy(report)
    changed["timings"]["component_assembly_ms"] = 99_000.0
    changed["runtime"]["import_ms"] = 4_000.0
    changed["queries"][0]["duration_ms"] = 900.0
    changed["package"]["component_root"] = "D:/another-path/component"
    assert result_fingerprint(report) == result_fingerprint(changed)
