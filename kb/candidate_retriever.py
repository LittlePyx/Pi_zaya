from __future__ import annotations

from typing import Protocol


class CandidateRetriever(Protocol):
    """Minimal contract for retrieval channels that return local evidence chunks."""

    @property
    def is_empty(self) -> bool: ...

    def search(self, query: str, top_k: int = 6) -> list[dict]: ...
