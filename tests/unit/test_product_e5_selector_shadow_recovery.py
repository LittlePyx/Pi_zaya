from __future__ import annotations

import copy
import json

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.finalize_product_e5_selector_shadow_recovery import (
    DEFAULT_RECOVERY,
    validate_recovery,
)
from tools.retrieval_eval.validate_product_e5_selector_shadow_protocol import (
    DEFAULT_PROTOCOL,
)


def _recovery() -> dict:
    return json.loads(DEFAULT_RECOVERY.read_text(encoding="utf-8"))


def test_product_e5_selector_measurement_recovery_passes() -> None:
    assert validate_recovery(
        _recovery(),
        protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
    ) == []


def test_product_e5_selector_measurement_recovery_rejects_second_build() -> None:
    changed = copy.deepcopy(_recovery())
    changed["measurement_fix"]["second_full_build_allowed"] = True

    assert "product E5 selector measurement fix boundary changed" in validate_recovery(
        changed,
        protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
    )


def test_product_e5_selector_measurement_recovery_rejects_partial_trace() -> None:
    changed = copy.deepcopy(_recovery())
    changed["recovery_replay"]["selection_call_count"] = 23

    assert "product E5 selector recovery replay changed" in validate_recovery(
        changed,
        protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
    )
