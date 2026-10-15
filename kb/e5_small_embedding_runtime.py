from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


EMBEDDING_DIMENSION = 384
MAXIMUM_MODEL_TOKENS = 512
CONTENT_WINDOW_TOKENS = 448
CONTENT_STRIDE_TOKENS = 384
MINIMUM_CONTENT_OVERLAP_TOKENS = 64
MAXIMUM_BATCH_SIZE = 8
MAXIMUM_PADDED_TOKENS = 4096
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
BOS_TOKEN_ID = 0
PAD_TOKEN_ID = 1
EOS_TOKEN_ID = 2


class _Encoding(Protocol):
    ids: Sequence[int]


class E5Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = True) -> _Encoding:
        ...


@dataclass(frozen=True)
class E5ParentWindow:
    parent_id: str
    parent_position: int
    window_ordinal: int
    input_ids: tuple[int, ...]

    def public_metadata(self) -> dict[str, object]:
        return {
            "parent_id": self.parent_id,
            "parent_position": self.parent_position,
            "window_ordinal": self.window_ordinal,
        }


@dataclass(frozen=True)
class E5EmbeddingProgress:
    completed_batches: int
    total_batches: int
    completed_windows: int
    total_windows: int


class E5RuntimeError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(str(code or "e5_runtime_failed"))
        self.code = str(code or "e5_runtime_failed")


class E5EmbeddingCancelled(E5RuntimeError):
    def __init__(self) -> None:
        super().__init__("cancelled_by_user")


def window_spans(
    content_token_count: int,
    *,
    window_tokens: int = CONTENT_WINDOW_TOKENS,
    stride_tokens: int = CONTENT_STRIDE_TOKENS,
) -> list[tuple[int, int]]:
    count = max(0, int(content_token_count))
    window = int(window_tokens)
    stride = int(stride_tokens)
    if window <= 0 or stride <= 0 or stride > window:
        raise ValueError("window and stride must be positive with stride <= window")
    if count <= window:
        return [(0, count)]
    starts = [0]
    while starts[-1] + window < count:
        next_start = min(starts[-1] + stride, count - window)
        if next_start <= starts[-1]:
            raise E5RuntimeError("window_construction_failed")
        starts.append(next_start)
    return [(start, min(count, start + window)) for start in starts]


def uncovered_token_count(
    content_token_count: int,
    spans: Sequence[tuple[int, int]],
) -> int:
    count = max(0, int(content_token_count))
    if count == 0:
        return 0
    cursor = 0
    uncovered = 0
    for start, end in sorted((int(start), int(end)) for start, end in spans):
        if start > cursor:
            uncovered += start - cursor
        cursor = max(cursor, end)
    return uncovered + max(0, count - cursor)


def build_document_window_ids(
    content_ids: Sequence[int],
    *,
    prefix_ids: Sequence[int],
    bos_token_id: int = BOS_TOKEN_ID,
    eos_token_id: int = EOS_TOKEN_ID,
    window_tokens: int = CONTENT_WINDOW_TOKENS,
    stride_tokens: int = CONTENT_STRIDE_TOKENS,
) -> list[tuple[int, ...]]:
    spans = window_spans(
        len(content_ids),
        window_tokens=window_tokens,
        stride_tokens=stride_tokens,
    )
    if uncovered_token_count(len(content_ids), spans):
        raise E5RuntimeError("window_content_uncovered")
    rows = [
        (
            int(bos_token_id),
            *(int(value) for value in prefix_ids),
            *(int(value) for value in content_ids[start:end]),
            int(eos_token_id),
        )
        for start, end in spans
    ]
    if any(len(row) > MAXIMUM_MODEL_TOKENS for row in rows):
        raise E5RuntimeError("document_window_too_long")
    return rows


def pack_by_padded_token_budget(
    token_lengths: Sequence[int],
    *,
    maximum_batch_size: int = MAXIMUM_BATCH_SIZE,
    maximum_padded_tokens: int = MAXIMUM_PADDED_TOKENS,
) -> list[list[int]]:
    ordered = sorted(
        range(len(token_lengths)),
        key=lambda index: (int(token_lengths[index]), index),
    )
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for index in ordered:
        length = int(token_lengths[index])
        if length <= 0 or length > MAXIMUM_MODEL_TOKENS:
            raise E5RuntimeError("input_token_length_invalid")
        proposed_max = max(current_max, length)
        proposed_size = len(current) + 1
        if current and (
            proposed_size > maximum_batch_size
            or proposed_max * proposed_size > maximum_padded_tokens
        ):
            batches.append(current)
            current = []
            current_max = 0
        current.append(index)
        current_max = max(current_max, length)
    if current:
        batches.append(current)
    return batches


def masked_mean_normalize(
    hidden_state: Any,
    attention_mask: Any,
    *,
    expected_dimension: int = EMBEDDING_DIMENSION,
) -> list[tuple[float, ...]]:
    import numpy as np

    hidden = np.asarray(hidden_state, dtype=np.float32)
    mask = np.asarray(attention_mask, dtype=np.float32)
    if hidden.ndim != 3 or hidden.shape[:2] != mask.shape:
        raise E5RuntimeError("hidden_state_shape_invalid")
    if hidden.shape[2] != expected_dimension:
        raise E5RuntimeError("embedding_dimension_invalid")
    denominator = np.clip(mask.sum(axis=1, keepdims=True), 1.0, None)
    pooled = (hidden * mask[:, :, None]).sum(axis=1) / denominator
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    if (
        not np.isfinite(hidden).all()
        or not np.isfinite(pooled).all()
        or not np.isfinite(norms).all()
        or np.any(norms <= 0)
    ):
        raise E5RuntimeError("embedding_vector_invalid")
    normalized = (pooled / norms).astype(np.float32, copy=False)
    return [tuple(float(value) for value in row) for row in normalized]


def encode_id_rows(
    *,
    session: Any,
    rows: Sequence[Sequence[int]],
    pad_token_id: int = PAD_TOKEN_ID,
    output_name: str = "last_hidden_state",
    dimension: int = EMBEDDING_DIMENSION,
) -> list[tuple[float, ...]]:
    import numpy as np

    if not rows:
        return []
    maximum = max(len(row) for row in rows)
    if maximum > MAXIMUM_MODEL_TOKENS:
        raise E5RuntimeError("input_token_length_invalid")
    input_ids = np.full((len(rows), maximum), pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((len(rows), maximum), dtype=np.int64)
    token_type_ids = np.zeros((len(rows), maximum), dtype=np.int64)
    for index, row in enumerate(rows):
        input_ids[index, : len(row)] = row
        attention_mask[index, : len(row)] = 1
    hidden = session.run(
        [output_name],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )[0]
    return masked_mean_normalize(
        hidden,
        attention_mask,
        expected_dimension=dimension,
    )


def build_parent_windows(
    tokenizer: E5Tokenizer,
    chunks: Sequence[dict],
    *,
    should_cancel: Callable[[], bool] = lambda: False,
) -> list[E5ParentWindow]:
    prefix_ids = tokenizer.encode(PASSAGE_PREFIX, add_special_tokens=False).ids
    windows: list[E5ParentWindow] = []
    for parent_position, chunk in enumerate(chunks):
        if should_cancel():
            raise E5EmbeddingCancelled()
        parent_id = str(chunk.get("id") or "").strip()
        if not parent_id:
            raise E5RuntimeError("parent_id_missing")
        content_ids = tokenizer.encode(
            str(chunk.get("text") or ""),
            add_special_tokens=False,
        ).ids
        for ordinal, row in enumerate(
            build_document_window_ids(content_ids, prefix_ids=prefix_ids)
        ):
            windows.append(
                E5ParentWindow(
                    parent_id=parent_id,
                    parent_position=parent_position,
                    window_ordinal=ordinal,
                    input_ids=row,
                )
            )
    return windows


class E5SmallOnnxExecutor:
    """CPU-only executor for the exact accepted E5-small representation."""

    def __init__(
        self,
        *,
        model_path: Path | str,
        tokenizer_path: Path | str,
        session: Any | None = None,
        tokenizer: E5Tokenizer | None = None,
    ) -> None:
        if tokenizer is None:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(Path(tokenizer_path)))
            tokenizer.no_padding()
            tokenizer.no_truncation()
        if session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 8
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(
                str(Path(model_path)),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        if list(session.get_providers()) != ["CPUExecutionProvider"]:
            raise E5RuntimeError("execution_provider_invalid")
        self.tokenizer = tokenizer
        self.session = session

    def embed_query(self, text: str) -> tuple[float, ...]:
        clean = str(text or "").strip()
        if not clean:
            raise E5RuntimeError("query_empty")
        ids = tuple(
            int(value)
            for value in self.tokenizer.encode(
                QUERY_PREFIX + clean,
                add_special_tokens=True,
            ).ids
        )
        if len(ids) > MAXIMUM_MODEL_TOKENS:
            raise E5RuntimeError("query_too_long")
        return encode_id_rows(session=self.session, rows=[ids])[0]

    def build_windows(
        self,
        chunks: Sequence[dict],
        *,
        should_cancel: Callable[[], bool] = lambda: False,
    ) -> list[E5ParentWindow]:
        return build_parent_windows(
            self.tokenizer,
            chunks,
            should_cancel=should_cancel,
        )

    def embed_windows(
        self,
        windows: Sequence[E5ParentWindow],
        *,
        should_cancel: Callable[[], bool] = lambda: False,
        on_progress: Callable[[E5EmbeddingProgress], None] = lambda _progress: None,
    ) -> list[tuple[float, ...]]:
        lengths = [len(window.input_ids) for window in windows]
        batches = pack_by_padded_token_budget(lengths)
        vectors: list[tuple[float, ...] | None] = [None] * len(windows)
        completed_windows = 0
        for batch_number, batch in enumerate(batches, start=1):
            if should_cancel():
                raise E5EmbeddingCancelled()
            encoded = encode_id_rows(
                session=self.session,
                rows=[windows[index].input_ids for index in batch],
            )
            for index, vector in zip(batch, encoded):
                vectors[index] = vector
            completed_windows += len(batch)
            on_progress(
                E5EmbeddingProgress(
                    completed_batches=batch_number,
                    total_batches=len(batches),
                    completed_windows=completed_windows,
                    total_windows=len(windows),
                )
            )
        if any(vector is None for vector in vectors):
            raise E5RuntimeError("embedding_matrix_incomplete")
        return [vector for vector in vectors if vector is not None]
