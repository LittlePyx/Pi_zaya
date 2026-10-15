from __future__ import annotations

import math

import pytest

from tools.retrieval_eval.lighter_runtime_onnx_worker import (
    _normalize_rows,
    _pack_by_padded_token_budget,
    _work_summary,
)


def test_lighter_runtime_packing_respects_batch_and_token_budget() -> None:
    lengths = [100, 500, 600, 3000, 5000]
    batches = _pack_by_padded_token_budget(
        lengths, maximum_batch_size=2, maximum_padded_tokens=1000
    )
    assert batches == [[0, 1], [2], [3], [4]]
    assert all(len(batch) <= 2 for batch in batches)


def test_lighter_runtime_packing_keeps_oversize_as_singleton() -> None:
    batches = _pack_by_padded_token_budget(
        [5000, 100, 200], maximum_batch_size=8, maximum_padded_tokens=4096
    )
    assert batches == [[1, 2], [0]]
    summary = _work_summary([5000, 100, 200], batches)
    assert summary["oversize_singleton_count"] == 1
    assert summary["peak_padded_tokens"] == 5000


def test_lighter_runtime_work_proxy_uses_padded_quadratic_cost() -> None:
    summary = _work_summary([2, 3], [[0, 1]])
    assert summary["real_tokens"] == 5
    assert summary["padded_tokens"] == 6
    assert summary["attention_work_proxy"] == 18
    assert summary["padding_amplification"] == 1.2


def test_lighter_runtime_normalizes_finite_vectors() -> None:
    rows = _normalize_rows([[3.0, 4.0], [0.0, 2.0]], expected_count=2, expected_dimension=2)
    assert rows[0] == pytest.approx([0.6, 0.8])
    assert math.sqrt(sum(value * value for value in rows[1])) == pytest.approx(1.0)


def test_lighter_runtime_rejects_output_shape_or_zero_norm() -> None:
    with pytest.raises(RuntimeError, match="output shape changed"):
        _normalize_rows([[1.0, 2.0]], expected_count=2, expected_dimension=2)
    with pytest.raises(RuntimeError, match="invalid vectors"):
        _normalize_rows([[0.0, 0.0]], expected_count=1, expected_dimension=2)
