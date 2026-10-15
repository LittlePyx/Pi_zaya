from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


_CITATION_RE = re.compile(r"(?:\[[0-9][0-9,\-\s]*\]|\[\[CITE:[^\]]+\]\])")
_NUMERIC_CITATION_RE = re.compile(r"(?<!\[)\[(\d+(?:\s*(?:,|-)\s*\d+)*)\](?!\])")


@dataclass(frozen=True)
class CitationStructure:
    citation_present: bool
    citation_numbers: list[int]
    bound_citation_numbers: list[int]
    unresolved_citation_numbers: list[int]
    binding: str
    structurally_bound: bool
    structural_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def has_citation_marker(claim: str) -> bool:
    return bool(_CITATION_RE.search(str(claim or "")))


def citation_numbers(claim: str, *, limit: int = 24) -> list[int]:
    numbers: list[int] = []
    seen: set[int] = set()
    for match in _NUMERIC_CITATION_RE.finditer(str(claim or "")):
        for part in re.split(r"\s*,\s*", match.group(1)):
            raw = part.strip()
            if "-" in raw:
                start_raw, end_raw = [item.strip() for item in raw.split("-", 1)]
                try:
                    start = int(start_raw)
                    end = int(end_raw)
                except ValueError:
                    continue
                if start <= 0 or end < start or end - start >= limit:
                    continue
                candidates = range(start, end + 1)
            else:
                try:
                    candidates = (int(raw),)
                except ValueError:
                    continue
            for number in candidates:
                if number <= 0 or number in seen:
                    continue
                seen.add(number)
                numbers.append(number)
                if len(numbers) >= limit:
                    return numbers
    return numbers


def inspect_citation_structure(claim: str, *, evidence_hit_count: int) -> CitationStructure:
    present = has_citation_marker(claim)
    numbers = citation_numbers(claim)
    hit_count = max(0, int(evidence_hit_count or 0))
    bound = [number for number in numbers if number <= hit_count]
    unresolved = [number for number in numbers if number > hit_count]
    structurally_bound = bool(present and numbers and not unresolved and bound)
    if not present:
        binding = "none"
        reason = "missing_citation"
    elif not numbers:
        binding = "unbound"
        reason = "unbound_citation"
    elif unresolved and bound:
        binding = "partial"
        reason = "citation_index_out_of_range"
    elif unresolved:
        binding = "unresolved"
        reason = "citation_index_out_of_range"
    elif not bound:
        binding = "unresolved"
        reason = "no_evidence_hits" if hit_count <= 0 else "citation_index_out_of_range"
    else:
        binding = "bound"
        reason = ""
    return CitationStructure(
        citation_present=present,
        citation_numbers=numbers,
        bound_citation_numbers=bound,
        unresolved_citation_numbers=unresolved,
        binding=binding,
        structurally_bound=structurally_bound,
        structural_reason=reason,
    )
