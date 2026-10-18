from __future__ import annotations

from difflib import SequenceMatcher
from functools import lru_cache


@lru_cache(maxsize=2048)
def _cached_ratio(left: str, right: str) -> float:
    return float(SequenceMatcher(None, left, right).ratio())


def provenance_similarity(left: str, right: str) -> float:
    """Reuse exact ordered text comparisons, never source-binding decisions."""
    # Provenance probes are already bounded to 420, 520 or 2000 characters.
    # Keep both entry count and retained text size bounded for other callers.
    if len(left) > 2000 or len(right) > 2000:
        return float(SequenceMatcher(None, left, right).ratio())
    return _cached_ratio(left, right)
