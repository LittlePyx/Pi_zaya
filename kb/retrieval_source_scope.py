from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def source_path_key(value: object) -> str:
    return str(value or "").strip().replace("\\", "/").casefold()


@dataclass(frozen=True)
class RetrievalSourceScope:
    """Explicit local source paths; an empty selection never means all sources."""

    paths: frozenset[str]

    @classmethod
    def from_paths(cls, paths: Sequence[str] | None) -> RetrievalSourceScope:
        return cls(frozenset(key for path in paths or () if (key := source_path_key(path))))

    def contains(self, hit: dict) -> bool:
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        return source_path_key(meta.get("source_path")) in self.paths
