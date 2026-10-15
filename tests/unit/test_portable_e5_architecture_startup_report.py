from __future__ import annotations

import copy
import json

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.run_portable_e5_architecture_startup import (
    DEFAULT_PROTOCOL,
    DEFAULT_REPORT,
    validate_report,
)
from tools.retrieval_eval.validate_portable_e5_architecture_startup_report import (
    EXPECTED_REPORT_SHA256,
    EXPECTED_RESULT_FINGERPRINT,
)


def _inputs() -> tuple[dict, dict]:
    return (
        json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8")),
        json.loads(DEFAULT_REPORT.read_text(encoding="utf-8")),
    )


def _validate(protocol: dict, report: dict) -> list[str]:
    return validate_report(
        report,
        protocol=protocol,
        expected_result_fingerprint=EXPECTED_RESULT_FINGERPRINT,
    )


def test_architecture_startup_pass_is_pinned() -> None:
    protocol, report = _inputs()
    assert sha256_file(DEFAULT_REPORT) == EXPECTED_REPORT_SHA256
    assert _validate(protocol, report) == []
    assert report["gate"]["passes_all"] is True
    assert report["decision"]["status"] == protocol["decision"]["pass_status"]


def test_architecture_startup_report_rejects_import_mutation() -> None:
    protocol, report = _inputs()
    changed = copy.deepcopy(report)
    changed["runtime"]["product_import_ms"] = 5000.001
    assert "portable E5 architecture-startup gate is inconsistent" in _validate(
        protocol,
        changed,
    )


def test_architecture_startup_report_rejects_ranking_mutation() -> None:
    protocol, report = _inputs()
    changed = copy.deepcopy(report)
    changed["queries"][0]["ranked_chunk_ids"].reverse()
    assert "portable E5 architecture-startup gate is inconsistent" in _validate(
        protocol,
        changed,
    )


def test_architecture_startup_report_rejects_engine_import_mutation() -> None:
    protocol, report = _inputs()
    changed = copy.deepcopy(report)
    changed["execution"]["retrieval_engine_imported_during_product_import"] = True
    assert "portable E5 architecture-startup gate is inconsistent" in _validate(
        protocol,
        changed,
    )


def test_architecture_startup_report_rejects_payload_mutation() -> None:
    protocol, report = _inputs()
    changed = copy.deepcopy(report)
    changed["integrity"]["component_payload_before_worker"]["sha256"] = "0" * 64
    assert "portable E5 architecture-startup gate is inconsistent" in _validate(
        protocol,
        changed,
    )


def test_architecture_startup_report_rejects_fingerprint_mutation() -> None:
    protocol, report = _inputs()
    changed = copy.deepcopy(report)
    changed["result_fingerprint"] = "0" * 64
    assert "portable E5 architecture-startup fingerprint is inconsistent" in _validate(
        protocol,
        changed,
    )
