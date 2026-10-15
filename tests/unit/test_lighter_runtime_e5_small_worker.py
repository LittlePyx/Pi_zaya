from __future__ import annotations

import math

import pytest

from tools.retrieval_eval.lighter_runtime_e5_small_worker import (
    _masked_mean_normalize,
    build_document_window_ids,
)


def test_e5_document_window_ids_cover_tail_without_decode_roundtrip() -> None:
    windows = build_document_window_ids(
        list(range(1000)),
        prefix_ids=[10, 11, 12],
        bos_token_id=0,
        eos_token_id=2,
        window_tokens=448,
        stride_tokens=384,
    )
    assert len(windows) == 3
    assert all(len(row) == 453 for row in windows)
    assert windows[0][:4] == [0, 10, 11, 12]
    assert windows[-1][-2:] == [999, 2]


def test_e5_masked_mean_excludes_right_padding_and_normalizes() -> None:
    vectors = _masked_mean_normalize(
        [
            [[2.0, 0.0], [0.0, 2.0], [100.0, 100.0]],
            [[3.0, 4.0], [9.0, 9.0], [9.0, 9.0]],
        ],
        [[1, 1, 0], [1, 0, 0]],
        expected_dimension=2,
    )
    assert vectors[0] == pytest.approx([math.sqrt(0.5), math.sqrt(0.5)])
    assert vectors[1] == pytest.approx([0.6, 0.8])


def test_e5_masked_mean_rejects_shape_or_zero_vector() -> None:
    with pytest.raises(RuntimeError, match="shape changed"):
        _masked_mean_normalize([[[1.0, 2.0]]], [[1, 1]], expected_dimension=2)
    with pytest.raises(RuntimeError, match="invalid vectors"):
        _masked_mean_normalize([[[0.0, 0.0]]], [[1]], expected_dimension=2)
