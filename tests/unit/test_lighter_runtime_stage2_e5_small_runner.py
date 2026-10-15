from __future__ import annotations

import copy
import json

from tools.retrieval_eval.run_lighter_runtime_stage2_e5_small import (
    FUSION_CHANNEL,
    PRODUCT_CHANNEL,
    _quality_gate,
)
from tools.retrieval_eval.validate_lighter_runtime_stage2_e5_small_protocol import (
    DEFAULT_PROTOCOL,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def _channel(recall5: float, recall10: float, wrong5: float, mrr: float, ndcg: float) -> dict:
    return {
        "modes": {
            mode: {
                "metrics": {
                    "recall@5": recall5,
                    "recall@10": recall10,
                    "wrong_source_rate@5": wrong5,
                    "mrr": mrr,
                    "ndcg@10": ndcg,
                }
            }
            for mode in ("scoped", "full_library")
        }
    }


def test_e5_stage2_quality_gate_passes_protected_gain() -> None:
    channels = {
        PRODUCT_CHANNEL: _channel(0.25, 0.30, 0.0, 0.1, 0.1),
        FUSION_CHANNEL: _channel(0.25, 0.50, 0.0, 0.2, 0.2),
    }
    assert _quality_gate(_protocol(), channels)["passes_all"] is True


def test_e5_stage2_quality_gate_rejects_mrr_regression() -> None:
    channels = {
        PRODUCT_CHANNEL: _channel(0.25, 0.30, 0.0, 0.2, 0.1),
        FUSION_CHANNEL: _channel(0.25, 0.50, 0.0, 0.1, 0.2),
    }
    result = _quality_gate(_protocol(), channels)
    assert result["passes_all"] is False
    assert result["modes"]["scoped"]["checks"]["mrr_no_regression"] is False


def test_e5_stage2_quality_gate_rejects_wrong_source_regression() -> None:
    channels = {
        PRODUCT_CHANNEL: _channel(0.25, 0.30, 0.0, 0.1, 0.1),
        FUSION_CHANNEL: _channel(0.25, 0.50, 0.1, 0.2, 0.2),
    }
    changed = copy.deepcopy(channels)
    assert _quality_gate(_protocol(), changed)["passes_all"] is False
