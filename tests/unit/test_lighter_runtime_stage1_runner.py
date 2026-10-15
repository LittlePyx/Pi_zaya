from __future__ import annotations

import copy
import json
from pathlib import Path

from tools.retrieval_eval.run_lighter_runtime_stage1 import (
    DEFAULT_PARENT_REPORT,
    DEFAULT_PROTOCOL,
    DEFAULT_REPORT,
    DEFAULT_MANIFEST,
    FUSION_CHANNEL,
    PRODUCT_CHANNEL,
    _evaluate_gate,
    _result_fingerprint,
    validate_report,
)
from kb.retrieval_semantic_eval_fixture import load_retrieval_benchmark_fixture


def _contracts() -> tuple[dict, dict, dict, dict]:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    fixture = load_retrieval_benchmark_fixture(
        Path(protocol["quality"]["fixture_path"]), repo_root=DEFAULT_PROTOCOL.parents[1]
    )
    parent = json.loads(DEFAULT_PARENT_REPORT.read_text(encoding="utf-8"))
    return protocol, manifest, fixture, parent


def test_stage1_gate_rejects_runtime_or_vector_parity_regression() -> None:
    protocol, _manifest, _fixture, parent = _contracts()
    product = parent["channels"][PRODUCT_CHANNEL]
    candidate = copy.deepcopy(parent["channels"][FUSION_CHANNEL])
    candidate["channel"] = "qwen_official_quint8_avx2_protected_dense_tail"
    acceptance = {"passes_all": True}
    worker = {
        "runtime": {"provider_load_ms": 1000, "peak_process_rss_mib": 1000},
        "query": {"timings": [{"duration_ms": 10}] * 12, "truncated_count": 0},
        "documents": {
            "projected_full_build_ms": 1000,
            "projected_throughput_chunks_per_second": 100,
            "truncated_stress_count": 0,
            "vectors": {str(i): [] for i in range(64)},
        },
    }
    parity = {"mean_cosine": 0.99, "minimum_cosine": 0.96}
    passed = _evaluate_gate(
        protocol=protocol,
        parent=parent,
        product=product,
        candidate=candidate,
        acceptance=acceptance,
        worker=worker,
        parity=parity,
        subprocess_wall_ms=1000,
        source_unchanged=True,
        index_unchanged=True,
        artifact_unchanged=True,
    )
    assert passed["passes_all"] is True
    worker["runtime"]["peak_process_rss_mib"] = 5000
    parity["minimum_cosine"] = 0.8
    rejected = _evaluate_gate(
        protocol=protocol,
        parent=parent,
        product=product,
        candidate=candidate,
        acceptance=acceptance,
        worker=worker,
        parity=parity,
        subprocess_wall_ms=1000,
        source_unchanged=True,
        index_unchanged=True,
        artifact_unchanged=True,
    )
    assert rejected["operational_checks"]["peak_process_rss"] is False
    assert rejected["operational_checks"]["worst_document_vector_parity"] is False
    assert rejected["passes_all"] is False


def test_stage1_result_fingerprint_ignores_timing_but_not_gate() -> None:
    payload = {
        "channels": {},
        "semantic_acceptance": {"passes_all": True},
        "document_vector_parity": {"mean_cosine": 0.99},
        "gate": {"passes_all": True},
        "decision": {"status": "passed"},
        "runtime": {"query_p95_ms": 1},
    }
    changed = copy.deepcopy(payload)
    changed["runtime"]["query_p95_ms"] = 999
    assert _result_fingerprint(payload) == _result_fingerprint(changed)
    changed["gate"]["passes_all"] = False
    assert _result_fingerprint(payload) != _result_fingerprint(changed)


def test_stage1_pinned_report_validation_when_available() -> None:
    if not DEFAULT_REPORT.is_file():
        return
    protocol, manifest, fixture, parent = _contracts()
    report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    if report.get("report_kind") == "retrieval_lighter_runtime_stage1_qwen_v1":
        assert validate_report(
            report,
            protocol=protocol,
            manifest=manifest,
            fixture=fixture,
            parent=parent,
            expected_result_fingerprint=None,
        ) == []
