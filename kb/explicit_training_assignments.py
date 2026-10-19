from __future__ import annotations

import re


def _named_training_roles_covered(claim: str, evidence: str) -> bool:
    description = r'(?:(?i:the|a|an|observed|noisy|singly[- ]noisy|doubly[- ]noisy|unseen|clean|image|observation)\s+){0,7}'
    variable = r'\$?([A-Z])(?![A-Za-z0-9_])\$?'
    def roles(text: str, role: str) -> set[str]:
        prefix = rf'(?i:(?:(?:training|network|regression)\s+)?{role}\s+(?:is|are|=)\s+)'
        return set(re.findall(prefix + description + variable, text))
    wanted_targets = roles(claim,'target')
    source_targets = roles(evidence,'target') | set(re.findall(r'(?i:predict)\s+'+variable+r'\s+(?i:given|from)\b',evidence))
    if wanted_targets and not wanted_targets <= source_targets:
        return False
    wanted_inputs = roles(claim,'input')
    source_inputs = roles(evidence,'input') | set(re.findall(r'(?i:given)\s+'+variable+r'\s+(?i:as\s+input)\b',evidence))
    if wanted_inputs and not wanted_inputs <= source_inputs:
        return False
    descriptive_input = re.search(r'(?i:network\s+input\s+is\s+(?:the\s+)?(?:doubly[- ]noisy|noisier)\s+image)',claim)
    return not descriptive_input or bool(re.search(r'\binput\b',evidence,re.I) and re.search(r'doubly[- ]noisy|noisier\s+version',evidence,re.I))


def named_equation_covered(claim: str, evidence: str) -> bool:
    """Do not borrow a numbered equation from a different source paragraph."""
    # A figure/table/equation number used merely to locate a numeric result
    # remains a locator. This veto covers the stated training/definition itself.
    if not re.search(r'\b(?:train\w*|loss|input|target|split\w*|decompos\w*)\b|训练|损失|输入|目标|划分|分解', claim, re.I):
        return True
    numbers = re.findall(r'\b(?:equations?|eqs?\.?)\s*[（(]?\s*(\d+)\s*[)）]?|(?:公式|等式|式)\s*[（(]\s*(\d+)\s*[)）]', claim, re.I)
    return all(re.search(rf'\\tag\s*\{{\s*{number}\s*\}}|\b(?:equations?|eqs?\.?|loss\s+function)\s*[（(]?\s*{number}(?!\d)\s*[)）]?|(?:公式|等式|式)\s*[（(]\s*{number}\s*[)）]', evidence, re.I) for pair in numbers for number in pair if number)


def explicit_training_assignments_covered(claim: str, evidence: str) -> bool:
    """Veto an explicit input/target formula absent from the cited passage.

    Shared mask symbols in a loss do not define the training pair. This only
    checks already written definitions; it supplies no positive entailment.
    """
    if not _named_training_roles_covered(claim,evidence):
        return False
    feeding = r'feed\w*[^.;。\n]{0,50}Bernoulli[- ]sampled|fed[^.;。\n]{0,50}Bernoulli[- ]sampled|Bernoulli[- ]sampled[^.;。\n]{0,80}(?:fed|feeding)|伯努利采样[^。；\n]{0,30}(?:送入|输入网络)'
    if re.search(feeding, claim, re.I) and not re.search(feeding, evidence, re.I):
        return False
    if re.search(r'\bloss\b|损失', claim, re.I):
        compact = lambda text: re.sub(r'\s+|_|\\(?:left|right)\b', '', text)
        factors = [compact(m[1]) for m in re.finditer(r'\$([^$\n]+)\$', claim) if re.search(r'1\s*-', m[1]) and not re.search(r':=|=', m[1])]
        if factors and not all(factor in compact(evidence) for factor in factors):
            return False
    split = re.search(r'\b(?:equation|eq\.?)\s*\(?([0-9]+)\)?', claim, re.I)
    if split and re.search(r'splits?|decompos\w*|划分|分解', claim, re.I) and re.search(r'visible|可见', claim, re.I) and re.search(r'held[- ]out|留出', claim, re.I):
        def symbols(text: str) -> str:
            text = re.sub(r'\\(?:boldsymbol|mathbf|mathcal)\{([A-Za-z0-9]+)\}', r'\1', text)
            return re.sub(r'\s+|_|\\(?:left|right)\b', '', text)
        raw = symbols(evidence)
        names = {symbols(m[1]) for m in re.finditer(r'\$([^$\n]+)\$', claim)}
        assignments = list(re.finditer(r':=([A-Za-z][A-Za-z0-9]*)\\odot([A-Za-z][A-Za-z0-9]*)', raw))
        if not re.search(rf'\\tag\{{{split[1]}\}}', raw) or not any(
            a[1] in names and a[2] in names and f':=(1-{a[1]})\\odot{a[2]}' in raw for a in assignments
        ):
            return False
    training_roles = bool(re.search(r'\b(?:input|target)\b|输入|目标', claim, re.I))

    def normalized(text: str) -> str:
        # The shared claim surface removes Markdown underscores. Keep the
        # subscript variable itself on both sides, including its actual index.
        text = re.sub(r'\\triangleq\b|≜|:=', '=', text).replace('−', '-')
        return re.sub(r'\s+', '', re.sub(r'\\(?:left|right)\b', '', text)).replace('_', '')

    definitions = []
    for match in re.finditer(r'(?<!\$)\$([^$\n]+)\$(?!\$)', claim):
        expression = match[1]
        operator = re.search(r':=|\\triangleq\b|≜|(?<!=)=(?!=)', expression)
        if operator is None:
            continue
        subject = re.sub(r'\\(?:boldsymbol|mathbf|mathcal)\{([^{}]+)\}', r'\1', expression[:operator.start()])
        # A declared observation/variable is still a definition without an
        # "input" or "target" label. Do not treat arbitrary equalities as
        # definitions or invent algebraic equivalents.
        if training_roles or operator[0] != '=' or re.fullmatch(r'\s*[A-Za-z](?:_[A-Za-z0-9{}]+)?\s*', subject):
            definitions.append(normalized(expression))
    definitions.extend(normalized(m[0]) for m in re.finditer(r'\b[A-Z]\s*(?:≜|:=|=)\s*[A-Z](?:\s*[+-]\s*[A-Z])+', claim))
    if not definitions:
        return True
    source = normalized(evidence)
    return all(definition in source for definition in definitions)


def prediction_average_covered(claim: str, evidence: str) -> bool:
    """Training with dropout does not itself state how predictions combine."""
    average = r'averag(?:e|es|ed|ing)\b(?!\s+pooling)'
    relation = rf'(?:{average}|mean\s+of)[^.;。\n]{{0,100}}(?:predictions?|outputs?)|(?:predictions?|outputs?)[^.;。\n]{{0,100}}{average}|(?:预测|输出)[^。；\n]{{0,60}}(?:取平均|平均)|(?:取平均|平均)[^。；\n]{{0,60}}(?:预测|输出)'
    return not re.search(relation, claim, re.I) or bool(re.search(relation, evidence, re.I))


def noise_augmentation_training_covered(claim: str, evidence: str) -> bool:
    """An experiment's noise level is not an additional-noise training recipe."""
    if not re.search(r'\btrain\w*\b|\bnetwork\b|训练|(?:网络|模型).{0,8}预测', claim, re.I):
        return True
    addition = r'(?:add|augment|inject)\w*[^.;。\n]{0,65}noise|(?:再|额外|进一步)(?:加入|添加|叠加)(?:合成)?噪声|(?:加入|添加|叠加)合成噪声'
    if not re.search(addition, claim, re.I):
        return True
    if not re.search(r'additional\s+(?:synthetic\s+)?noise|synthetic\s+noise|noisier\s+version|augment\w*[^.;\n]{0,60}noise|(?:额外|合成)噪声', evidence, re.I):
        return False
    predicts_original = r'predict\w*[^.;\n]{0,30}original\s+noisy\s+image|预测原(?:始)?(?:噪声|带噪|含噪)图像'
    return not re.search(predicts_original, claim, re.I) or bool(re.search(r'predict\s+Y\s+given\s+Z|predict\w*[^.;\n]{0,30}original\s+noisy\s+image|预测原(?:始)?(?:噪声|带噪|含噪)图像', evidence, re.I))
