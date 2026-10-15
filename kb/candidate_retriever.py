from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class CandidateRetriever(Protocol):
    """Minimal contract for retrieval channels that return local evidence chunks."""

    @property
    def is_empty(self) -> bool: ...

    def search(self, query: str, top_k: int = 6) -> list[dict]: ...


class SourceScopedCandidateRetriever(CandidateRetriever, Protocol):
    """Rank evidence within explicit source paths before taking top K."""

    def search_scoped(
        self, query: str, *, source_paths: Sequence[str], top_k: int = 6,
    ) -> list[dict]: ...
