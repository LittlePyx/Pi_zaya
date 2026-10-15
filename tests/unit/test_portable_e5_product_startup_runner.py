from __future__ import annotations

import copy
import json

from tools.retrieval_eval.run_portable_optional_e5_product_startup import (
    _decision,
    evaluate_gate,
    result_fingerprint,
)
from tools.retrieval_eval.validate_portable_e5_product_startup_protocol import (
    DEFAULT_PROTOCOL,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def _passing_report() -> dict:
    protocol = _protocol()
    component = protocol["component"]
    component_root = "F:\\test_results\\e5p2\\component\\c"
    return {
        "inherited_v1": {
            "report_sha256": protocol["source_v1"]["report_sha256"],
            "result_fingerprint": protocol["source_v1"]["result_fingerprint"],
            "decision_status": protocol["source_v1"]["decision_status"],
            "only_failed_gate": "import_time",
        },
        "component_root": component_root,
        "timings": {
            "base_extraction_ms": 1000.0,
            "component_extraction_ms": 20_000.0,
            "embedded_worker_wall_ms": 10_000.0,
        },
        "runtime": {
            "python_version": "3.10.11",
            "onnxruntime_version": "1.23.2",
            "tokenizers_version": "0.21.4",
            "onnxruntime_module_path": f"{component_root}\\python\\site-packages\\onnxruntime\\__init__.py",
            "tokenizers_module_path": f"{component_root}\\python\\site-packages\\tokenizers\\__init__.py",
            "product_import_ms": 1000.0,
            "cold_load_ms": 2500.0,
            "query_p95_ms": 90.0,
            "peak_process_rss_mib": 1200.0,
        },
        "cold_load": {"cold_start": True, "index_cache_hit": False},
        "queries": [
            {
                **expected,
                "duration_ms": 80.0,
                "used_mode": "hybrid_local",
                "fallback_reason_id": "",
                "source_escape_count": 0,
            }
            for expected in protocol["execution"]["expected_queries"]
        ],
        "execution": {
            "query_count": 2,
            "index_build_call_count": 0,
            "network_used": False,
            "evaluation_helper_imported_before_startup_timing": False,
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
        },
        "integrity": {
            "base_archive_before_sha256": protocol["base_portable"]["archive_sha256"],
            "base_archive_after_sha256": protocol["base_portable"]["archive_sha256"],
            "component_archive_before_sha256": component["archive_sha256"],
            "component_archive_after_sha256": component["archive_sha256"],
            "extracted_component_tree_sha256": component["component_tree_sha256"],
        },
    }


def test_product_startup_gate_passes_product_only_boundary() -> None:
    assert evaluate_gate(_passing_report(), _protocol())["passes_all"] is True


def test_product_startup_gate_rejects_v1_verdict_rewrite() -> None:
    report = _passing_report()
    report["inherited_v1"]["decision_status"] = "passed"
    assert evaluate_gate(report, _protocol())["passes_all"] is False


def test_product_startup_gate_rejects_ranking_change() -> None:
    report = _passing_report()
    report["queries"][0]["ranked_chunk_ids"].reverse()
    assert evaluate_gate(report, _protocol())["passes_all"] is False


def test_product_startup_gate_rejects_import_over_budget() -> None:
    report = _passing_report()
    report["runtime"]["product_import_ms"] = 5000.001
    assert evaluate_gate(report, _protocol())["passes_all"] is False


def test_product_startup_decision_never_enables_product() -> None:
    decision = _decision(_protocol(), {"passes_all": True})
    assert decision["status"].startswith("product_startup_boundary_passed")
    assert decision["v1_verdict_changed"] is False
    assert decision["product_enabled"] is False
    assert decision["user_mode_control_visible"] is False


def test_product_startup_fingerprint_ignores_timings() -> None:
    report = _passing_report()
    report["gate"] = evaluate_gate(report, _protocol())
    report["decision"] = _decision(_protocol(), report["gate"])
    changed = copy.deepcopy(report)
    changed["timings"]["component_extraction_ms"] = 90_000.0
    changed["runtime"]["product_import_ms"] = 4000.0
    changed["queries"][0]["duration_ms"] = 900.0
    assert result_fingerprint(report) == result_fingerprint(changed)
