from __future__ import annotations

import copy
import json

from tools.retrieval_eval.run_product_e5_runtime_parity import (
    DEFAULT_REPORT,
    validate_report,
)


def _report() -> dict:
    return json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))


def test_product_e5_runtime_parity_report_passes() -> None:
    assert validate_report(_report()) == []


def test_parity_rejects_query_float_drift() -> None:
    changed = copy.deepcopy(_report())
    changed["query_parity"]["maximum_absolute_difference"] = 0.000001
    assert "product E5 query-vector parity changed" in validate_report(changed)


def test_parity_rejects_window_mapping_drift() -> None:
    changed = copy.deepcopy(_report())
    changed["representation_parity"]["benchmark_windows_sha256"] = "0" * 64
    assert "product E5 representation parity changed" in validate_report(changed)


def test_parity_rejects_document_build_or_product_enablement() -> None:
    changed = copy.deepcopy(_report())
    changed["execution"]["document_vectors_built"] = 1
    changed["execution"]["product_enabled"] = True
    assert "product E5 parity execution boundary changed" in validate_report(changed)
