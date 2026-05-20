from __future__ import annotations

import re

_RE_WORD = re.compile(
    r"[A-Za-z0-9_]+|[\u4e00-\u9fff]",  # English/number tokens OR single CJK char
    flags=re.UNICODE,
)


def tokenize(text: str, *, bigram_cjk: bool = False) -> list[str]:
    # Lowercase only for latin tokens.
    out: list[str] = []
    for t in _RE_WORD.findall(text):
        if len(t) == 1 and "\u4e00" <= t <= "\u9fff":
            out.append(t)
        else:
            out.append(t.lower())
    # Optionally append CJK bigrams for better phrase-level matching.
    # E.g. "\u91cd\u5efa\u8d28\u91cf" -> unigrams ["\u91cd", "\u5efa", "\u8d28", "\u91cf"] + bigrams ["\u91cd\u5efa", "\u5efa\u8d28", "\u8d28\u91cf"]
    if bigram_cjk:
        cjk_region: list[str] = []
        for t in out:
            if len(t) == 1 and "\u4e00" <= t <= "\u9fff":
                cjk_region.append(t)
            else:
                if len(cjk_region) >= 2:
                    out.extend(cjk_region[i] + cjk_region[i + 1] for i in range(len(cjk_region) - 1))
                cjk_region.clear()
        if len(cjk_region) >= 2:
            out.extend(cjk_region[i] + cjk_region[i + 1] for i in range(len(cjk_region) - 1))
    return out

