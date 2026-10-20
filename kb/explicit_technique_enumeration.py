from __future__ import annotations

import re

from kb.citation_sentences import citation_sentence_spans


def technique_pair_quantities(claim: str, evidence: str) -> set[tuple[str, str, str]]:
    """Count a source's explicit pair of named techniques, not nearby numbers.

    This only covers the anaphoric count 'these two techniques'; measurements,
    architecture/data counts and all remaining quantities keep their checks.
    """
    if not re.search(r'\bthese\s+(?:two|2)\s+techniques\b', claim, re.I):
        return set()
    technique = r'(?:(?!and\b|or\b)[a-z-]+\s+){0,3}(?:learning|normalization|regularization|augmentation|optimization|dropout|masking|sampling|pooling|encoding|decoding)'
    pair = re.compile(rf'(?<![A-Za-z-])(?P<left>{technique})\s+and\s+(?P<right>{technique})\s+can\s+benefit\s+from\s+each\s+other', re.I)
    for start, end in citation_sentence_spans(evidence):
        sentence = evidence[start:end]
        if re.search(r'\b(?:not|never|neither|nor|false|incorrect)\b', sentence, re.I):
            continue
        matches = list(pair.finditer(sentence))
        identity = lambda name: re.sub(r'^.*\bthat\s+', '', name.lower()).strip()
        if len(matches) == 1 and identity(matches[0]['left']) != identity(matches[0]['right']):
            # Do not count an embedded pair as an entire longer enumeration.
            remainder = sentence[:matches[0].start()] + sentence[matches[0].end():]
            remainder = re.sub(r',?\s+and\s+their\s+(?:integration|combination)\b.*', '', remainder, flags=re.I)
            if not re.search(r'\band\b|\bor\b|\d|\b(?:three|four|several|many|other\s+techniques)\b', remainder, re.I):
                return {('2', '', '')}
    return set()
