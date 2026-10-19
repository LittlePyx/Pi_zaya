from __future__ import annotations

import re
import unicodedata


def remove_new_unpaired_bold(original: str, retained: str) -> str:
    """Do not leave half of an originally paired emphasis after claim pruning."""
    token = re.compile(r'(?P<protected>\$\$[\s\S]*?\$\$|\$(?:\\.|[^$\n])+\$|`+[^`\n]*`+)|(?P<bold>(?<![\\*])\*\*(?!\*))')
    def markers(text: str) -> list[re.Match[str]]:
        return [match for match in token.finditer(text) if match.lastgroup == 'bold']
    if len(markers(original)) % 2:
        return retained
    stack: list[re.Match[str]] = []
    unmatched = []
    for match in markers(retained):
        left = retained[match.start()-1] if match.start() else ' '
        right = retained[match.end()] if match.end() < len(retained) else ' '
        left_punct = unicodedata.category(left).startswith('P')
        right_punct = unicodedata.category(right).startswith('P')
        opens = not right.isspace() and (not right_punct or left.isspace() or left_punct)
        closes = not left.isspace() and (not left_punct or right.isspace() or right_punct)
        if closes and stack:
            stack.pop()
        elif opens:
            stack.append(match)
        else:
            unmatched.append(match)
    for match in sorted([*stack, *unmatched], key=lambda m: m.start(), reverse=True):
        retained = retained[:match.start()] + retained[match.end():]
    return retained


def repair_pruned_line_emphasis(original: str, retained: str) -> str:
    """Keep pruning on one line from consuming another paragraph's emphasis."""
    originals = original.splitlines()
    result = []
    fenced = False
    for line in retained.splitlines():
        if line.strip().startswith(('```', '~~~', '$$')):
            fenced = not fenced
            result.append(line)
            continue
        prefix = re.sub(r'\*\*', '', line).lstrip()[:16]
        matches = [raw for raw in originals if prefix and re.sub(r'\*\*', '', raw).lstrip().startswith(prefix)]
        result.append(remove_new_unpaired_bold(matches[0], line) if not fenced and len(matches) == 1 else line)
    return '\n'.join(result)
