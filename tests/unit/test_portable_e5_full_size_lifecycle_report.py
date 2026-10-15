from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_portable_e5_full_size_lifecycle_report import (
    DEFAULT_REPORT,
    validate_report,
)


def _report() -> dict:
    return json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))


def test_full_size_lifecycle_rejection_passes() -> None:
    assert validate_report(_report()) == []


def test_report_rejects_relabelled_pass() -> None:
    changed = copy.deepcopy(_report())
    changed["decision"]["all_gates_passed"] = True
    assert "full-size lifecycle rejection decision changed" in validate_report(changed)


def test_report_rejects_hidden_manual_retry() -> None:
    changed = copy.deepcopy(_report())
    changed["failure"]["user_repair_required"] = True
    assert "full-size lifecycle rejection failure contract changed" in validate_report(
        changed
    )


def test_report_rejects_archive_integrity_rewrite() -> None:
    changed = copy.deepcopy(_report())
    changed["attribution"]["zip_crc_test_passed"] = False
    assert "full-size lifecycle rejection attribution changed" in validate_report(changed)


def test_report_rejects_path_length_rewrite() -> None:
    changed = copy.deepcopy(_report())
    changed["attribution"]["longest_target_absolute_path_characters"] = 259
    assert "full-size lifecycle rejection attribution changed" in validate_report(changed)


def test_report_rejects_fingerprint_rewrite() -> None:
    changed = copy.deepcopy(_report())
    changed["result_fingerprint"] = "0" * 64
    assert "full-size lifecycle rejection fingerprint changed" in validate_report(changed)
