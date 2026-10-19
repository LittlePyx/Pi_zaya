from __future__ import annotations

import re
from collections import defaultdict
from decimal import Decimal


_ROW = re.compile(
    r'\b(?P<metric>PSNR|PNSR|SSIM|LPIPS)\s*(?:\([^;=\n]*\))?\s*/\s*'
    r'(?P<method>[^/;|\n=]{2,60})\s*/\s*(?P<condition>[^;|\n=]{1,100})\s*=\s*'
    r'(?P<value>-?\d+(?:\.\d+)?)', re.I,
)


def table_order_covers(claim: str, evidence: str, comparison: re.Match, *, positive: bool) -> bool:
    """Require the named table methods and every comparable metric to agree."""
    rows = [(r['metric'], r['method'], r['condition'], r['value']) for r in _ROW.finditer(evidence)]
    for header in re.finditer(r'\b(PSNR|PNSR|SSIM|LPIPS)\s+(?:results?|values?)\s*:\s*([^\n]+)', evidence, re.I):
        rows.extend((header[1], r[1], '', r[2]) for r in re.finditer(r'([^,;:|\n=]{1,60}?)\s*=\s*(-?\d+(?:\.\d+)?)', header[2]))
    methods = {method.strip().casefold() for _, method, _, _ in rows}

    def named(surface: str) -> set[str]:
        names = {method for method in methods if re.search(rf'(?<![A-Za-z0-9]){re.escape(method)}(?![A-Za-z0-9])', surface, re.I)}
        if 'ours' in methods and re.search(r'该方法|本方法|\bour\s+method\b', surface, re.I):
            names.add('ours')
        return names

    left = named(claim[:comparison.start()]); right = named(claim[comparison.end():])
    if len(left) != 1 or not right or left & right:
        return False
    first = next(iter(left))
    groups: dict[tuple[str, str], dict[str, list[Decimal]]] = defaultdict(lambda: defaultdict(list))
    for metric, method, condition, value in rows:
        key = metric.upper().replace('PNSR', 'PSNR'), condition.strip().casefold()
        groups[key][method.strip().casefold()].append(Decimal(value))
    comparable = []
    for (metric, _condition), values in groups.items():
        for second in right:
            if first not in values and second not in values:
                continue
            if first not in values or second not in values or len(set(values[first])) != 1 or len(set(values[second])) != 1:
                return False
            a, b = values[first][0], values[second][0]
            better = a < b if metric == 'LPIPS' else a > b
            comparable.append(better == positive)
    return bool(comparable) and all(comparable)
