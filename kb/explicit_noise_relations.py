from __future__ import annotations

import re
from kb.citation_sentences import citation_sentence_spans


IID_PATTERN = r"\bi\.\s*i\.\s*d\.|\biid\b|\bindependent(?:ly)?\s+and\s+identically\s+distributed\b|独立同分布"
_PAIR = re.compile(r"\b([A-Z])\s*(?:[,，、]\s*|and\s+|和|与)\s*([A-Z])\b")
_PIXEL_SCOPE = re.compile(r"(?:同一幅|单幅|图内|像素).{0,32}分量|pixel[- ]wise|noise\s+components", re.I)
_COMPONENT_RELATION = re.compile(r'(?:噪声)?分量[^，,。.;!?]{0,50}独立|独立[^，,。.;!?]{0,25}(?:噪声)?分量|(?:noise\s+)?components?[^,.;!?]{0,50}independen\w*|independen\w*[^,.;!?]{0,25}(?:noise\s+)?components?', re.I)
_COMPONENT_PREMISE = re.compile(r'noise\s+components?[^,.;!?]{0,50}independen\w*|independen\w*[^,.;!?]{0,25}noise\s+components?|noise\s+(?:(?:to\s+be|is)\s+)?pixel[- ]wise\s+independen\w*|noise\s+is\s+independent\s+conditional|噪声分量[^，,。.;!?]{0,30}独立', re.I)
_COMPATIBILITY = re.compile(r'同时成立|不矛盾|相容|\bcompatible\b|\bcan\s+(?:both\s+)?hold\s+simultaneously\b|\bcan\s+both\s+hold\b|\bnot\s+mutually\s+exclusive\b', re.I)
_RAW_NOISE = re.compile(
    r"(?:raw\s+(?:network\s+)?output|uncorrected\s+output)"
    r"(?:\s+of\s+(?:the|our)\s+network)?\s+(?:is|remains?)\s+"
    r"(?P<state>not\s+(?:still\s+)?noisy|(?:still\s+)?noisy|not\s+(?:a\s+)?clean(?:\s+image)?)", re.I,
)


def _negated_relation_context(text: str, match: re.Match[str]) -> bool:
    start = next((a for a, b in citation_sentence_spans(text) if a <= match.start() < b), 0)
    prefix = text[max(start, match.start() - 80):match.start()]
    return bool(re.search(r"\b(?:not|never|cannot|false)\b|并非|不(?:是|要求|需要)", prefix, re.I))


def _relation_pair(text: str) -> re.Match[str] | None:
    normalized = re.sub(r"\\(?:boldsymbol|mathbf|mathcal|mathrm)\s*\{([A-Z])\}", r"\1", text)
    return _PAIR.search(normalized.replace('$', ''))


def _positive_scoped(text: str, match: re.Match[str]) -> bool:
    prefix = re.split(r'[.;。！？\n]|\b(?:but|while|whereas|however|and)\b|但|而', text[:match.start()], flags=re.I)[-1][-80:]
    own = re.sub(r'\bnot\s+(?:only|mutually\s+exclusive)\b', '', prefix + match[0], flags=re.I)
    return not re.search(r'\b(?:not|never|cannot|false)\b|并非|不(?:是|要求|需要|独立|相容)|无需|未', own, re.I)


def _sample_independence_statement(evidence: str) -> bool:
    for start, end in citation_sentence_spans(evidence):
        statement = re.sub(r'\\(?:boldsymbol|mathbf|mathcal|mathrm)\s*\{([A-Z])\}', r'\1', evidence[start:end]).replace('$', '')
        for relation in re.finditer(IID_PATTERN + r'|\bindependen\w*\b|独立', statement, re.I):
            if not _positive_scoped(statement, relation):
                continue
            if re.search(rf'(?:two|both)\s+(?:(?:{IID_PATTERN})\s+)?(?:noise\s+)?(?:samples|draws)|两个?噪声样本', statement, re.I):
                return True
            pair = _PAIR.search(statement)
            if pair and 0 <= relation.start() - pair.end() <= 64 and not re.search(r'\b(?:but|while|whereas|components?|pixels?)\b|[;；]', statement[pair.end():relation.start()], re.I):
                return True
            if re.search(r'original|原始', statement, re.I) and re.search(r'synthetic|合成', statement, re.I) and not re.search(r'components?|分量', statement, re.I):
                return True
    return False


def iid_relation_covered(claim: str, evidence: str) -> bool:
    if re.search(r'前者.*后者|former.*latter', claim, re.I) and re.search(r'不要求|不需要|不相关|\bnot\s+(?:require|need)', claim, re.I):
        if not re.search(r'前者.*后者|former.*latter', evidence, re.I):
            return False
    comparison = bool(re.search(r'两篇|前者.*后者|two\s+papers|former.*latter', claim, re.I) and re.search(r'独立|independen', claim, re.I))
    component = any(_positive_scoped(claim, m) for m in _COMPONENT_RELATION.finditer(claim)) or bool(comparison and re.search(r'分量|components?', claim, re.I))
    if component and not any(_positive_scoped(evidence, m) for m in _COMPONENT_PREMISE.finditer(evidence)):
        return False
    original = re.search(r'(?:原始噪声|原噪声)(?!图像|图片|图|观测)|original\s+noise\b', claim, re.I)
    generated = re.search(r'(?:额外|合成|生成).{0,5}噪声|(?:synthetic|additional)\s+(?:(?:additional|synthetic|generated|spatially|correlated)\s+){0,3}(?:noise|samples?)\b', claim, re.I)
    if original and generated and not (
        (re.search(r'original\s+(?:noise|noisy\s+image)|原始(?:噪声|含噪图像)', evidence, re.I) and re.search(r'synthetic\s+(?:noise\s+)?sample|合成噪声', evidence, re.I))
        or _sample_independence_statement(evidence)
    ):
        return False
    compatibility = bool(component and re.search(r'噪声样本|noise\s+samples?', claim, re.I) and _COMPATIBILITY.search(claim))
    samples = bool(re.search(r'独立|independen', claim, re.I) and re.search(r'额外|合成|synthetic', claim, re.I) and re.search(r'噪声|noise|samples?', claim, re.I)) or bool(re.search(r'(?:two|both)\s+(?:i\.\s*i\.\s*d\.\s+)?(?:noise\s+)?samples|两个?[^。；;]{0,24}噪声样本', claim, re.I))
    if samples and original and generated:
        surface = re.sub(r'\\(?:boldsymbol|mathbf|mathcal|mathrm)\s*\{([A-Z])\}', r'\1', claim)
        names = set(re.findall(r'(?<![A-Za-z0-9_])([A-Z])(?![A-Za-z0-9_])', re.sub(IID_PATTERN, '', surface, flags=re.I)))
        if len(names) >= 2 and any(not re.search(rf'(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])', evidence) for name in names):
            return False
    if samples and re.search(r'和的一半|half\s+of\s+(?:their|the)\s+sum|half[- ]sum', claim, re.I) and not re.search(r'\bhalf\b|一半|/\s*2\b|\\frac\s*\{1\}\s*\{2\}', evidence, re.I):
        return False
    if (samples or compatibility) and not _sample_independence_statement(evidence):
        return False
    if compatibility and not any(_positive_scoped(evidence, m) for m in _COMPATIBILITY.finditer(evidence)):
        return False
    requested = re.search(IID_PATTERN, claim, re.I)
    if not requested:
        return True
    # Negated/qualified assertions cannot gain positive support through an
    # abbreviation alias. Ordinary polarity checks still run independently.
    if _negated_relation_context(claim, requested):
        return False
    claim_pair = _relation_pair(claim)
    sample_scope = re.search(r"额外|合成|生成|\b(?:additional|synthetic|generated)\b", claim, re.I)
    spans = citation_sentence_spans(evidence)
    for actual in re.finditer(IID_PATTERN, evidence, re.I):
        if _negated_relation_context(evidence, actual):
            continue
        start, end = next(((a, b) for a, b in spans if a <= actual.start() < b), (0, len(evidence)))
        statement = evidence[start:end]
        if _PIXEL_SCOPE.search(claim) and not _PIXEL_SCOPE.search(statement):
            continue
        source_pair = _relation_pair(statement)
        if claim_pair and (not source_pair or set(claim_pair.groups()) != set(source_pair.groups())):
            continue
        if sample_scope and not (source_pair or re.search(
            rf"(?:two|both)\s+(?:(?:{IID_PATTERN})\s+)?(?:noise\s+)?(?:samples|draws)", statement, re.I,
        )):
            continue
        return True
    return False


def signal_dependency_covered(claim: str, evidence: str) -> bool:
    """Keep signal dependence separate from the accompanying noise premise."""
    requested = re.search(r'信号(?:(?:中|具有|有|存在|的|统计|一定|某种|某些|表现出|一定程度(?:的)?)\s*){0,4}(?:相关|依赖)|\bsignal[^.;\n]{0,65}(?:statistical\s+dependenc|statistically\s+predictable|not\s+pixel[- ]wise\s+independent|\bcorrelat)', claim, re.I)
    if not requested:
        return True
    pattern = (r'\bsignal\s*,?\s*(?:however\s*,?\s*)?(?:is\s+assumed\s+to\s+contain\s+(?:statistical\s+)?dependencies|'
               r'(?:exhibits?|contains?|has)\s+(?:some\s+|complex\s+|statistical\s+)*(?:dependencies|correlation)|'
               r'(?:is\s+)?(?:spatially\s+)?correlated|is\s+not\s+pixel[- ]wise\s+independent)|'
               r'信号(?:(?:中|具有|有|存在|的|统计|一定)\s*){0,4}(?:相关|依赖)')
    return any(not _negated_relation_context(evidence, m) for m in re.finditer(pattern, evidence, re.I))


def white_noise_restriction_covered(claim: str, evidence: str) -> bool:
    """A permitted noise type does not establish an exclusive restriction."""
    pattern = (r'\b(?:(?:limited|restricted)\s+to|only\s+(?:supports?|handles?|works\s+with|applies\s+to|applicable\s+to))\s+'
               r'[^,.;。；\n]{0,55}\b(?:white\s+noise|independent\s+noise)|'
               r'(?:仅|只能|只)(?:适用|支持|处理)[^，,。.;\n]{0,35}(?:白噪声|独立噪声)')
    requested = any(not _negated_relation_context(claim, m) for m in re.finditer(pattern, claim, re.I))
    return not requested or any(not _negated_relation_context(evidence, m) for m in re.finditer(pattern, evidence, re.I))


def raw_output_noise_covered(claim: str, evidence: str) -> bool:
    wanted = _RAW_NOISE.search(claim)
    if not wanted:
        return True
    supplied = _RAW_NOISE.search(evidence)
    if not supplied or any(m['state'].lower().startswith(('not noisy','not still noisy')) for m in (wanted,supplied)):
        return False
    return 'clean' in wanted['state'].lower() or 'clean' not in supplied['state'].lower()


def sampling_ability_covered(claim: str, evidence: str) -> bool:
    """Not needing a sampling prior does not establish inability to sample."""
    sampling = r"sampl\w*\s+from\s+(?:(?:the|a|known)\s+){0,2}noise\s+distribution|从噪声分布中?采样"
    sampling += r"|可采样|能够采样|能采样|\bsampling\s+ability\b"
    if re.search(r'noise|噪声', claim, re.I):
        sampling += r'|从(?:该|此)分布中?采样|从中采样|可采样的噪声分布|noise\s+distribution\s+to\s+sampl\w*\s+from'
    sampling += r'|draw\w*[^.;\n]{0,60}(?:synthetic\s+)?noise\s+sample|(?:synthetic\s+)?noise\s+sample[^.;\n]{0,30}(?:can|may)\s+be\s+drawn'
    inability = r"不具备.{0,16}(?:能力|这一点)|(?:所)?不具备(?:的)?(?=[。；,，\s\[\]]|$)|无法.{0,25}采样|\b(?:cannot|unable\s+to)\b.{0,45}sampl|\b(?:lack\w*|not\s+(?:have|possess))\b.{0,25}ability"
    if not re.search(sampling, claim, re.I):
        return True
    if not re.search(inability, claim, re.I):
        return bool(re.search(sampling + r'|draw\w*[^.;\n]{0,60}(?:synthetic\s+)?noise\s+sample', evidence, re.I))
    return bool(re.search(
        r"\b(?:cannot|unable\s+to)\b[^.;\n]{0,45}sampl|"
        r"\b(?:lack\w*|not\s+(?:have|possess))\s+(?:the\s+)?ability\s+to\s+sampl|"
        r"(?:不具备|没有|缺乏)[^。；\n]{0,30}采样能力|无法从噪声分布中?采样",
        evidence, re.I,
    ))


def known_noise_model_covered(claim: str, evidence: str) -> bool:
    """Sampling ability alone does not explicitly state a known noise model."""
    scope = bool(re.search(r'noise\s+model|噪声模型', claim, re.I))
    known = re.compile(r'\bknown(?:,?\s+samplable)?\s+noise\s+(?:model|distribution)|已知(?:的)?[、，]?(?:(?:且)?(?:(?:可(?:以)?|能(?:够)?)从中采样的|可采样的))?噪声(?:模型|分布)' + (r'|\bknown\s+model\b' if scope else ''), re.I)

    known = re.compile(known.pattern + r'|noise\s+(?:operator|distribution|model)[^.;\n]{0,35}\bis\s+known\b|噪声(?:分布|模型)[^。；\n]{0,15}(?:是|为)已知', re.I)

    def positive(text: str, match: re.Match) -> bool:
        prefix = re.split(r'\bbut\b|\bhowever\b|但|而', text[:match.start()], flags=re.I)[-1][-32:]
        return not (re.search(r'(?:does\s+not|not|without|不需要|无需)\s*(?:require\w*\s+)?(?:a\s+)?$', prefix, re.I)
                    or re.match(r'\s+(?:is\s+)?not\s+(?:required|known)', text[match.end():], re.I))

    return not any(positive(claim, m) for m in known.finditer(claim)) or any(positive(evidence, m) for m in known.finditer(evidence))


def named_synthetic_sample_covered(claim: str, evidence: str) -> bool:
    """A distribution statement about N cannot label N as the synthetic sample."""
    labels = re.findall(r'(?i:synthetic\s+(?:noise\s+)?sample|synthetic\s+noise|合成(?:噪声)?样本|合成噪声|合成样本)\s*\$?([A-Z])(?![A-Za-z])', claim)
    labels += re.findall(r'(?<![A-Za-z])\$?([A-Z])\$?\s*(?:视为|是|为)[^。；\n]{0,40}(?:合成(?:噪声)?样本)|(?<![A-Za-z])([A-Z])\s+(?i:is\s+(?:the|a|an)\s+synthetic\s+(?:noise\s+)?sample)', claim)
    labels = [next((v for v in name if v), '') if isinstance(name, tuple) else name for name in labels]
    if any(not re.search(rf'(?<![A-Za-z0-9_]){name}(?![A-Za-z0-9_])', evidence) for name in labels):
        return False
    named = re.findall(r'(?i:synthetic\s+(?:noise\s+)?sample|合成噪声样本)\s*[（(]?\s*\$?([A-Za-z])\s*(?:\\sim|∼|~)', claim)
    for variable in named:
        name = re.escape(variable)
        direct = rf'\b{name}\s+(?i:is\s+(?:a\s+)?synthetic\s+(?:noise\s+)?sample)\b|(?i:synthetic\s+(?:noise\s+)?sample)\s*\$?{name}(?![A-Za-z])'
        if not re.search(direct, evidence):
            return False
    declared = re.findall(r'(?<![A-Za-z])([A-Z])\s*(?i:is\s+(?:the|a|an)\s+synthetic\s+(?:noise\s+)?sample)|把\s*([A-Z])\s*视为[^。；\n]{0,40}合成(?:噪声)?样本', claim)
    for pair in declared:
        variable = next(value for value in pair if value)
        direct = rf'(?i:synthetic\s+(?:noise\s+)?sample)\s*\$?{variable}(?![A-Za-z])|(?<![A-Za-z]){variable}\s+(?i:is\s+(?:the|a|an)\s+synthetic\s+(?:noise\s+)?sample)'
        if not re.search(direct, evidence):
            return False
    return True
