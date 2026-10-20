from __future__ import annotations

import re

from kb.citation_marker_context import numeric_marker_is_content
from kb.evidence_term_mapping import specific_method_identities, _METHOD_IDENTITY_EQUIVALENTS


def explicit_owned_comparison_clause(claim: str, source_name: str) -> str:
    """Scope an explicit while/whereas comparison to its named source only."""
    quantities = re.sub(r'\[\d{1,5}\](?:\([^\n)]*\))?', '', claim)
    quantities = re.sub(r'\b(?:equations?|eq\.?|propositions?|theorems?|lemmas?|sections?|figures?|tables?)\s*[（(\[]?\d+[)）\]]?|(?:命题|定理|引理|公式|方程|第)\s*[（(]?\d+[)）]?(?:节|章|式)?', '', quantities, flags=re.I)
    if re.search(r'(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])', quantities):
        return claim
    owner = {name for name in specific_method_identities(claim) if re.search(rf'(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])', source_name, re.I)}
    if not owner:
        return claim
    boundaries = [0]
    for match in re.finditer(r',\s*(?:while|whereas)\s+|，\s*而\s*', claim, re.I):
        if not numeric_marker_is_content(claim, match.start()):
            boundaries.extend((match.start(), match.end()))
    boundaries.append(len(claim))
    parts = [claim[boundaries[i]:boundaries[i + 1]].strip() for i in range(0, len(boundaries) - 1, 2)]
    if len(parts) < 2:
        return claim
    identities = [specific_method_identities(part) for part in parts]
    if not all(identities) or any(identities[i] & identities[j] for i in range(len(parts)) for j in range(i)):
        return claim
    matches = [part for part, names in zip(parts, identities) if owner & names]
    return matches[0] if len(matches) == 1 else claim


def owned_source_body_claim(claim: str, source_name: str) -> str:
    """Exclude only this verified source's name/known alias from body scoring."""
    claim = explicit_owned_comparison_clause(claim, source_name)
    owner = str(source_name).replace('\\', '/').rsplit('/', 1)[-1].split(':', 1)[0].strip()
    owner = re.sub(r'(?:\.en|\.md|\.pdf)+$', '', owner, flags=re.I).lower()
    if not re.fullmatch(r'[a-z][a-z0-9]+', owner):
        return claim
    names = {owner}
    for equivalents in _METHOD_IDENTITY_EQUIVALENTS:
        if owner in equivalents:
            names.update(equivalents)
    return re.sub(r'(?<![A-Za-z0-9])(?:' + '|'.join(re.escape(n) for n in names) + r')(?![A-Za-z0-9])', '', claim, flags=re.I)
