from __future__ import annotations

import re


_OUTPERFORM = re.compile(r"outperform(?:s|ed|ing)?|胜过|优于", re.I)
_INFORMATION = re.compile(
    r"(?P<en>additional|more|less|same)\s+(?:training\s+)?information"
    r"(?:\s+available)?(?:\s+during\s+training)?|"
    r"(?P<zh>更多|额外|更少|相同)(?:的)?训练信息", re.I,
)


def _relations(text: str) -> list[tuple[str, str, str]]:
    relations = []
    for clause in re.split(r"[;；。!?！？]|(?<!\d)\.(?!\d)", text):
        action = _OUTPERFORM.search(clause)
        if action is None:
            continue
        information = _INFORMATION.search(clause, action.end())
        direction = "object"
        if information is None:
            information = _INFORMATION.search(clause[:action.start()])
            direction = "subject"
        if information is None or min(abs(information.start() - action.end()), abs(action.start() - information.end())) > 100:
            continue
        # The training qualifier must belong to the comparator, rather than
        # appear elsewhere in a paragraph about training.
        if information["en"] and not re.search(r"\btraining\b", information[0], re.I):
            continue
        amount = information["en"] or information["zh"]
        amount = {"additional": "more", "更多": "more", "额外": "more", "更少": "less", "相同": "same"}.get(amount.lower(), amount.lower())
        prefix = clause[:action.start()]
        if re.search(r"never|impossible|绝对不能|永远不会|不可能", prefix, re.I):
            modality = "impossible"
        elif re.search(r"cannot\s+be\s+expected|not\s+(?:be\s+)?expected|不能(?:期待|指望)|不应期待|不支持.{0,40}应当", prefix, re.I):
            modality = "not_expected"
        elif re.search(r"can\s+be\s+expected|expected\s+to|应当|可(?:以)?期待", prefix, re.I):
            modality = "expected"
        elif re.search(r"cannot|does\s+not|不(?:能|会)", prefix, re.I):
            modality = "negative"
        else:
            modality = "asserted"
        relations.append((amount, modality, direction))
    return relations


def training_information_relation_covered(claim: str, evidence: str) -> bool:
    """Keep a training-information comparison's amount and modal boundary.

    This is a veto for explicit comparisons, not a source of positive support.
    In particular, 'cannot be expected' does not mean 'can never outperform'.
    """
    requested = _relations(claim)
    return not requested or all(relation in _relations(evidence) for relation in requested)
