from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from kb.e5_small_embedding_runtime import (
    CONTENT_STRIDE_TOKENS,
    CONTENT_WINDOW_TOKENS,
    E5EmbeddingCancelled,
    E5RuntimeError,
    E5SmallOnnxExecutor,
    build_document_window_ids,
    build_parent_windows,
    masked_mean_normalize,
    pack_by_padded_token_budget,
    uncovered_token_count,
    window_spans,
)


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = True):
        if text == "passage: ":
            return SimpleNamespace(ids=[9, 10, 11])
        ids = [20 + index for index, _character in enumerate(text)]
        return SimpleNamespace(ids=([0, *ids, 2] if add_special_tokens else ids))


class _Session:
    def __init__(self) -> None:
        self.calls = []

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, outputs, inputs):
        self.calls.append((outputs, inputs))
        ids = inputs["input_ids"]
        hidden = np.zeros((ids.shape[0], ids.shape[1], 384), dtype=np.float32)
        hidden[:, :, 0] = ids
        hidden[:, :, 1] = 1.0
        return [hidden]


def test_tail_pulled_windows_cover_every_token_with_minimum_overlap() -> None:
    spans = window_spans(1000)
    assert spans == [(0, 448), (384, 832), (552, 1000)]
    assert uncovered_token_count(1000, spans) == 0
    assert min(left[1] - right[0] for left, right in zip(spans, spans[1:])) >= 64
    assert CONTENT_WINDOW_TOKENS - CONTENT_STRIDE_TOKENS == 64


def test_document_rows_keep_prefix_and_special_tokens_outside_content_budget() -> None:
    rows = build_document_window_ids(range(1000), prefix_ids=[7, 8, 9])
    assert len(rows) == 3
    assert all(row[:4] == (0, 7, 8, 9) and row[-1] == 2 for row in rows)
    assert max(len(row) for row in rows) == 453


def test_parent_window_metadata_is_deterministic_and_parent_text_is_not_returned() -> None:
    windows = build_parent_windows(
        _Tokenizer(),
        [{"id": "a", "text": "x" * 500}, {"id": "b", "text": "y"}],
    )
    assert [item.public_metadata() for item in windows] == [
        {"parent_id": "a", "parent_position": 0, "window_ordinal": 0},
        {"parent_id": "a", "parent_position": 0, "window_ordinal": 1},
        {"parent_id": "b", "parent_position": 1, "window_ordinal": 0},
    ]
    assert all("text" not in item.public_metadata() for item in windows)


def test_masked_mean_excludes_padding_and_l2_normalizes() -> None:
    hidden = np.array([[[3.0, 4.0], [0.0, 2.0], [100.0, 100.0]]], dtype=np.float32)
    vectors = masked_mean_normalize(
        hidden,
        np.array([[1, 1, 0]], dtype=np.int64),
        expected_dimension=2,
    )
    expected = np.array([1.5, 3.0], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    assert np.allclose(vectors[0], expected)


def test_batches_are_length_sorted_and_respect_padded_budget() -> None:
    lengths = [453, 10, 400, 20, 300, 30, 200, 40, 100]
    batches = pack_by_padded_token_budget(lengths)
    assert sorted(index for batch in batches for index in batch) == list(range(len(lengths)))
    assert all(len(batch) <= 8 for batch in batches)
    assert all(max(lengths[index] for index in batch) * len(batch) <= 4096 for batch in batches)


def test_executor_uses_exact_prefix_pooling_and_restores_corpus_order() -> None:
    session = _Session()
    executor = E5SmallOnnxExecutor(
        model_path="unused",
        tokenizer_path="unused",
        session=session,
        tokenizer=_Tokenizer(),
    )
    query = executor.embed_query("ab")
    windows = executor.build_windows([{"id": "a", "text": "x" * 500}])
    progress = []
    vectors = executor.embed_windows(windows, on_progress=progress.append)
    assert len(query) == 384
    assert len(vectors) == 2
    assert all(np.isclose(np.linalg.norm(vector), 1.0) for vector in vectors)
    assert progress[-1].completed_windows == 2
    assert progress[-1].total_windows == 2


def test_cancel_is_checked_during_tokenization_and_before_each_batch() -> None:
    with pytest.raises(E5EmbeddingCancelled):
        build_parent_windows(
            _Tokenizer(),
            [{"id": "a", "text": "x"}],
            should_cancel=lambda: True,
        )
    executor = E5SmallOnnxExecutor(
        model_path="unused",
        tokenizer_path="unused",
        session=_Session(),
        tokenizer=_Tokenizer(),
    )
    windows = executor.build_windows([{"id": "a", "text": "x"}])
    with pytest.raises(E5EmbeddingCancelled):
        executor.embed_windows(windows, should_cancel=lambda: True)


def test_executor_rejects_non_cpu_provider_and_long_query() -> None:
    session = _Session()
    session.get_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    with pytest.raises(E5RuntimeError) as provider_error:
        E5SmallOnnxExecutor(
            model_path="unused",
            tokenizer_path="unused",
            session=session,
            tokenizer=_Tokenizer(),
        )
    assert provider_error.value.code == "execution_provider_invalid"

    executor = E5SmallOnnxExecutor(
        model_path="unused",
        tokenizer_path="unused",
        session=_Session(),
        tokenizer=_Tokenizer(),
    )
    with pytest.raises(E5RuntimeError) as query_error:
        executor.embed_query("x" * 600)
    assert query_error.value.code == "query_too_long"
