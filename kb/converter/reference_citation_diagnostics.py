from __future__ import annotations

import re

from kb.inpaper_citation_grounding import iter_inpaper_numeric_citations, parse_ref_num_set


_ARRAY_CONTEXT = re.compile(
    r"(?:channels?|heads?|blocks?|dimensions?|sizes?|resolutions?|widths?|depths?|"
    r"batch\s+sizes?|sequence\s+lengths?|kernel\s+sizes?|strides?|values?)"
    r"\s+(?:are|is|=)\s*$", re.IGNORECASE,
)
_TRAINING_INTERVAL = re.compile(
    r"(?:noise[-\s]+levels?\s+range\s+[A-Za-z]|"
    r"pre[-\s]+trained\s+model\s+on\s+[A-Za-z])\s*=\s*$", re.IGNORECASE,
)


def plausible_reference_citation_indices(body: str) -> set[int]:
    """Bibliography diagnostics exclude declared arrays and training intervals."""
    cited: set[int] = set()
    for spec, start, _end, _style in iter_inpaper_numeric_citations(body):
        prefix = str(body or "")[max(0, int(start) - 96):int(start)]
        if _ARRAY_CONTEXT.search(prefix) or re.search(r"(?:\\in|∈|range\s+of)\s*$", prefix, re.I):
            continue
        if re.fullmatch(r"\s*\d+\s*,\s*\d+\s*", spec) and _TRAINING_INTERVAL.search(prefix):
            continue
        cited.update(parse_ref_num_set(spec, max_items=256))
    return cited
