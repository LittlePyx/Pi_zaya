from __future__ import annotations

import re


_CATEGORIES = {
    'clean': re.compile(r'(?<![非不])干净(?:训练)?(?:目标|标签|参照|图像|数据)|\bclean\s+(?:(?:training|target|reference|ground[- ]truth)\s+)?(?:images?|examples?|targets?|labels?|data)', re.I),
    'noisy_pairs': re.compile(r'(?<![非未不])(?:配对|成对)(?:的)?(?:含噪|带噪|噪声)(?:图像|数据)|(?:含噪|带噪|噪声)(?:图像)?对|\bpaired\s+noisy\s+(?:training\s+)?(?:images?|examples?|data)|\bnoisy\s+image\s+pairs|\bpairs\s+of\s+noisy\s+images', re.I),
}
_WAIVER = re.compile(r"(?:does?\s+not|don't|doesn't|not)\s+(?:require\w*|need\w*)\b|(?:require\w*|need\w*)\s+(?:neither|no)\b|不(?:再)?(?:需要|要求)|无需|无须|不必|不用|免除|省去|\b(?:dispose\s+of|dispense\s+with)\s+(?:the\s+)?|\bwithout\s+(?:(?:observing|using|needing)\s+)?", re.I)
_NEW_REQUIREMENT = re.compile(r'\b(?:but|whereas|however|while|require\w*|need\w*)\b|但|而|不过|(?<!不)(?:需要|要求)', re.I)


def _waived_categories(text: str) -> set[str]:
    result: set[str] = set()
    for sentence in re.split(r'[.;。!?！？；\n]', text):
        for waiver in _WAIVER.finditer(sentence):
            if waiver[0].lower().startswith('without') and re.search(r"\b(?:cannot|can't|not)\b[^,;]{0,30}$", sentence[:waiver.start()], re.I):
                continue
            scope = _NEW_REQUIREMENT.split(sentence[waiver.end():], maxsplit=1)[0]
            result.update(name for name, pattern in _CATEGORIES.items() if pattern.search(scope))
        neither = re.search(r'\bneither\b(?P<objects>.+?\bnor\b.+?)\s+(?:is|are)\s+(?:needed|required|necessary)\b', sentence, re.I)
        if neither:
            result.update(name for name, pattern in _CATEGORIES.items() if pattern.search(neither['objects']))
        for passive in re.finditer(r'\bno\s+(?P<objects>.+?)\s+(?:(?:is|are)\s+)?(?:needed|required|necessary)\b|(?P<subjects>.+?)\s+(?:is|are)\s+not\s+(?:needed|required|necessary)\b', sentence, re.I):
            objects = passive['objects'] or passive['subjects']
            # The subject of a passive waiver cannot include the subjects of
            # an earlier positive requirement, even across "and" or "but".
            objects = re.split(r',|\b(?:but|whereas|however|while)\b|但|而|不过', objects, flags=re.I)[-1]
            objects = re.split(r'\b(?:is|are)\s+(?:needed|required|necessary)\b', objects, flags=re.I)[-1]
            result.update(name for name, pattern in _CATEGORIES.items() if pattern.search(objects))
    return result


def waived_data_requirements_covered(claim: str, evidence: str) -> bool:
    """A noise-model premise cannot establish waived data requirements."""
    requested = _waived_categories(claim)
    covered = _waived_categories(evidence)
    # Choosing a denoiser by the self-supervised objective is a different
    # operation from collecting clean training targets. Require the complete
    # source loss relation and restricted minimization result, not a title.
    selection_reference = (re.search(r'干净参照.{0,20}(?:选择|选出)(?:最优)?去噪器', claim)
                           and re.search(r'最小化自监督损失|方差项|噪声方差', claim)
                           and re.search(r'与.{0,25}无关|不影响最优解|不会改变最优解', claim)
                           and not re.search(r'训练数据|训练目标|训练图像|训练标签', claim))
    surface = re.sub(r'\s+', ' ', evidence)
    if selection_reference and re.search(r'(?:^|[.!?])\s*(?:That is,\s*)?the self-supervised loss is the sum of the ordinary supervised loss and the variance of the noise', surface, re.I) and re.search(r'(?:^|[.!?])\s*By minimizing the self-supervised loss over a class of J\s*-invariant functions, one may find the optimal denoiser', surface, re.I):
        covered.add('clean')
    return requested <= covered
