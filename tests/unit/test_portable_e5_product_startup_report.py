from __future__ import annotations

import copy
import json

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.run_portable_optional_e5_product_startup import (
    DEFAULT_PROTOCOL,
    DEFAULT_REPORT,
    validate_report,
)
from tools.retrieval_eval.validate_portable_e5_product_startup_report import (
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


def test_portable_e5_product_startup_rejection_is_pinned() -> None:
    protocol, report = _inputs()
    assert sha256_file(DEFAULT_REPORT) == EXPECTED_REPORT_SHA256
    assert _validate(protocol, report) == []
    assert report["gate"]["checks"]["product_import_time"] is False
    assert report["decision"]["status"] == protocol["decision"]["failure_status"]


def test_portable_e5_product_startup_report_rejects_import_mutation() -> None:
    protocol, report = _inputs()
    changed = copy.deepcopy(report)
    changed["runtime"]["product_import_ms"] = 1.0
    assert "portable E5 product-startup gate is inconsistent" in _validate(
        protocol,
        changed,
    )


def test_portable_e5_product_startup_report_rejects_ranking_mutation() -> None:
    protocol, report = _inputs()
    changed = copy.deepcopy(report)
    changed["queries"][0]["ranked_chunk_ids"].reverse()
    assert "portable E5 product-startup gate is inconsistent" in _validate(
        protocol,
        changed,
    )


def test_portable_e5_product_startup_report_rejects_v1_rewrite() -> None:
    protocol, report = _inputs()
    changed = copy.deepcopy(report)
    changed["inherited_v1"]["decision_status"] = "passed"
    assert "portable E5 product-startup gate is inconsistent" in _validate(
        protocol,
        changed,
    )


def test_portable_e5_product_startup_report_rejects_fingerprint_mutation() -> None:
    protocol, report = _inputs()
    changed = copy.deepcopy(report)
    changed["result_fingerprint"] = "0" * 64
    assert "portable E5 product-startup fingerprint is inconsistent" in _validate(
        protocol,
        changed,
    )
