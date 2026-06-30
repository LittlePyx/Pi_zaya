from __future__ import annotations

from typing import Callable


def enrich_bibliometrics_and_summary(
    meta: dict,
    *,
    enrich_bibliometrics: Callable[[dict], dict],
    ensure_summary_line: Callable[..., dict],
    allow_crossref_abstract: bool = True,
) -> dict:
    out = dict(meta or {})
    enriched = enrich_bibliometrics(out)
    if isinstance(enriched, dict):
        out = enriched
    return ensure_summary_line(out, allow_crossref_abstract=allow_crossref_abstract)
