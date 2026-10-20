from __future__ import annotations

import re

from kb.citation_marker_context import numeric_marker_is_content
from kb.evidence_term_mapping import specific_method_identities

_LINK = r'\(\#[^\s)]+(?:\s+"[^"]*")?\)'
_MARKERS = r'(?:\s*\[\d{1,5}\](?:' + _LINK + r')?|\s*\[\[(?:SUPPORT|CITE):[^\]\n]+\]\])*'


def explicit_clause_boundaries(text: str) -> list[int]:
    """Separate independently asserted, explicitly contrasted source facts.

    Return offsets only: punctuation, qualifiers, variables and citations stay
    byte-for-byte in their original clause. Conditional/denial scopes, quotes,
    parentheses and literal code/math cannot acquire independent assertions.
    """
    result = []
    # Link titles contain sections, parentheses and quotation marks. They are
    # routing decoration, not sentence scope; keep offsets but mask their text.
    surface = re.sub(r'\[\[(?:SUPPORT|CITE):[^\]\n]+\]\]|\[\d{1,5}\](?:' + _LINK + r')?', lambda m: ' ' * len(m[0]), text, flags=re.I)
    for match in re.finditer(r'[,，]' + _MARKERS + r'(?=\s*(?:while\b|whereas\b|而\s*|并指出|且\s*|and\s+(?:adds?|injects?|augments?)\b))', text, re.I):
        at = match.start()
        prefix, suffix = text[:at], text[match.end():]
        if numeric_marker_is_content(text, at):
            continue
        current = re.split(r'[。;；!?\n]|\.(?=\s+[A-Z])', surface[:at])[-1]
        following = surface[match.end():]
        if any(current.count(a) != current.count(b) for a,b in (('(',')'),('（','）'),('“','”'),('‘','’'))) or current.count('"') % 2:
            continue
        if re.search(r'\b(?:if|unless|suppose|assuming|false\s+that|incorrect\s+that|not\s+(?:the\s+case|true|claim|state))\b|假设|如果|若|并非|并未|不能断言', current, re.I):
            continue
        # An explicit inline marker already ends one claim. A following
        # coordinated assertion must be checked against its own passage.
        cited = re.search(r'(?:\[\d{1,5}\](?:' + _LINK + r')?|\[\[(?:SUPPORT|CITE):[^\]\n]+\]\])\s*$', prefix, re.I) or '[' in match[0]
        left_ids = specific_method_identities(current)
        right_ids = specific_method_identities(re.split(r'[。.;；\n]', following)[0])
        methods = left_ids and right_ids and left_ids.isdisjoint(right_ids)
        loss = re.search(r'\bL\s*[_{}$ ]*[12]\b', current, re.I) and re.match(r'\s*(?:while|whereas)\s+(?:the\s+)?\$?L\s*[_{}$ ]*[12]\b', following, re.I)
        sentence = current + ',' + re.split(r'[。;；!?\n]|\.(?=\s+[A-Z])', following, maxsplit=1)[0]
        quantities = re.sub(r'\[\[(?:SUPPORT|CITE):[^\]\n]+\]\]|\[\d{1,5}\](?:' + _LINK + r')?', '', sentence, flags=re.I)
        quantities = re.sub(r'\b(?:propositions?|theorems?|lemmas?|sections?|figures?|tables?)\s*\d+|(?:命题|定理|引理)\s*\d+', '', quantities, flags=re.I)
        if not loss and re.search(r'(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])', quantities):
            continue
        augmentation = re.match(r'\s*and\s+(?:adds?|injects?|augments?)\b[^.;\n]{0,65}(?:synthetic|additional)\s+(?:noise|sample)', following, re.I)
        if cited or methods or loss or augmentation:
            result.append(match.end())
    return result
