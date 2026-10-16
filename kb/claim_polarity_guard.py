from __future__ import annotations

import re
from dataclasses import dataclass


_WORD = re.compile(r"[a-z]+(?:-[a-z]+)?", re.I)
_BREAK = re.compile(r"[,;，。！？\n]|(?<!\d)[.!?](?!\d)|\b(?:while|whereas|but)\b", re.I)
_STOP = {"a", "an", "the", "all", "any", "only", "out", "of", "on", "in", "by", "for", "to", "is", "are", "was", "were", "and", "or", "that", "it", "its", "our"}
_ACTIONS = {
    "memory": (r"stor(?:e|es|ed|ing)|retain(?:s|ed|ing)?|remember(?:s|ed|ing)?|keep(?:s|ing)?", r"discard(?:s|ed|ing)?|delet(?:e|es|ed|ing)|forget(?:s|ting)?|drop(?:s|ped|ping)?"),
    "training": (r"train(?:s|ed|ing)?|fine-tun(?:e|es|ed|ing)", r"freez(?:e|es|ing)|frozen|fix(?:es|ed|ing)?"),
    "filter": (r"filter(?:s|ed|ing)?", r""),
    "direction": (r"increas(?:e|es|ed|ing)|improv(?:e|es|ed|ing)|rais(?:e|es|ed|ing)|higher", r"decreas(?:e|es|ed|ing)|reduc(?:e|es|ed|ing)|lower"),
    "required_estimation": (r"estimat(?:e|es|ed|ing)", r""),
}
_ZH_ALIASES = (
    ("视觉编码器", "vision encoder"), ("语言模型", "language model"),
    ("无关信息", "irrelevant information"), ("相关信息", "relevant information"),
    ("重建精度", "reconstruction accuracy"), ("重建质量", "reconstruction quality"),
    ("先前", "previous"), ("之前", "previous"), ("历史", "previous"),
    ("提示", "prompts"), ("预测", "predictions"),
    ("不能", "cannot"), ("无法", "cannot"), ("没有", "not"), ("未", "not"), ("不", "not"),
    ("存储", "stores"), ("保留", "retains"), ("记住", "remembers"), ("丢弃", "discards"),
    ("训练", "trains"), ("冻结", "freezes"), ("过滤", "filters"),
    ("提高", "increases"), ("降低", "decreases"), ("与", "and"),
)


def _action_surface(text: str) -> str:
    value = text.lower()
    for source, target in _ZH_ALIASES:
        value = value.replace(source, f" {target} ")
    return re.sub(
        r"(vision encoder|language model)\s*(?:被|保持)\s*(freezes|trains)",
        lambda match: f"{match[1]} is {'frozen' if match[2] == 'freezes' else 'trained'}",
        value,
    )


@dataclass(frozen=True)
class _ActionFact:
    kind: str
    positive: bool
    object_terms: frozenset[str]


def _terms(words: list[str]) -> frozenset[str]:
    return frozenset(word[:-1] if word.endswith("s") and not word.endswith("ss") else word for word in words if word not in _STOP and len(word) >= 3)


def _facts(text: str) -> list[_ActionFact]:
    facts: list[_ActionFact] = []
    for clause in _BREAK.split(_action_surface(text)):
        words = _WORD.findall(clause)
        for index, word in enumerate(words):
            for kind, (positive, negative) in _ACTIONS.items():
                polarity = True if re.fullmatch(positive, word) else False if negative and re.fullmatch(negative, word) else None
                if polarity is None:
                    continue
                prefix = words[max(0, index - 3):index]
                if kind == "required_estimation" and not set(prefix) & {"without", "require", "requires", "required", "need", "needs", "must"}:
                    continue  # Optional estimation is not contradicted by denoising without it.
                # "not only" is additive, not a negation of this action.
                negated = any(token in {"not", "cannot", "never", "without"} for token in prefix) and prefix[-2:] != ["not", "only"]
                if negated:
                    polarity = not polarity
                copulas = {"is", "are", "was", "were", "be", "been"}
                offset = 1 if prefix and prefix[-1] in copulas else 2 if len(prefix) >= 2 and prefix[-1] == "not" and prefix[-2] in copulas else 0
                object_words = words[max(0, index - offset - 4):index - offset] if offset else words[index + 1:index + 6]
                terms = _terms(object_words)
                if len(terms) >= 2:
                    facts.append(_ActionFact(kind, polarity, terms))
    return facts


def claim_polarity_conflicts(claim: str, evidence: str) -> bool:
    """Veto aligned explicit opposites; never use absence as proof of negation.

    This is a conservative lexical constraint, not an entailment classifier.
    Numbers, units and source identity retain their existing dedicated checks.
    """
    clean_claim = re.sub(r"\[\d+\](?:\([^)]*\))?", " ", str(claim or ""))
    claim_facts, evidence_facts = _facts(clean_claim), _facts(str(evidence or ""))
    for fact in claim_facts:
        aligned = [other for other in evidence_facts if other.kind == fact.kind and len(other.object_terms & fact.object_terms) >= 2]
        if aligned and all(other.positive != fact.positive for other in aligned):
            return True
    return False


def has_checkable_action_fact(claim: str) -> bool:
    return bool(_facts(str(claim or "")))
