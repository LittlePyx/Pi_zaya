from __future__ import annotations

import math
from collections.abc import Sequence


Embedding = Sequence[float]


def normalize_embedding(
    values: Embedding,
    *,
    expected_dim: int | None = None,
) -> tuple[float, ...]:
    """Validate and L2-normalize one embedding vector."""

    vector = tuple(float(value) for value in values)
    if not vector:
        raise ValueError("embedding vector cannot be empty")
    if expected_dim is not None and len(vector) != expected_dim:
        raise ValueError(
            f"embedding dimension mismatch: expected {expected_dim}, got {len(vector)}"
        )
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding vector must contain only finite values")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        raise ValueError("embedding vector norm must be positive")
    return tuple(value / norm for value in vector)


def normalize_embedding_batch(
    values: Sequence[Embedding],
    *,
    expected_count: int,
    expected_dim: int,
) -> list[tuple[float, ...]]:
    """Validate the exact shape of a provider response and normalize it."""

    if len(values) != expected_count:
        raise ValueError(
            "embedding count mismatch: "
            f"expected {expected_count}, got {len(values)}"
        )
    return [
        normalize_embedding(vector, expected_dim=expected_dim)
        for vector in values
    ]
