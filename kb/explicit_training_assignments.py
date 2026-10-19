from __future__ import annotations

import re


def explicit_training_assignments_covered(claim: str, evidence: str) -> bool:
    """Veto an explicit input/target formula absent from the cited passage.

    Shared mask symbols in a loss do not define the training pair. This only
    checks already written definitions; it supplies no positive entailment.
    """
    if not re.search(r'\b(?:input|target)\b|输入|目标', claim, re.I):
        return True

    def normalized(text: str) -> str:
        # The shared claim surface removes Markdown underscores. Keep the
        # subscript variable itself on both sides, including its actual index.
        return re.sub(r'\s+', '', re.sub(r'\\(?:left|right)\b', '', text)).replace('_', '')

    definitions = [normalized(m[1]) for m in re.finditer(r'(?<!\$)\$([^$\n]+)\$(?!\$)', claim) if ':=' in m[1]]
    if not definitions:
        return True
    source = normalized(evidence)
    return all(definition in source for definition in definitions)


def noise_augmentation_training_covered(claim: str, evidence: str) -> bool:
    """An experiment's noise level is not an additional-noise training recipe."""
    addition = r'(?:add|augment|inject)\w*[^.;。\n]{0,65}noise|(?:再|额外|进一步)(?:加入|添加|叠加)(?:合成)?噪声'
    if not re.search(addition, claim, re.I):
        return True
    if not re.search(r'additional\s+(?:synthetic\s+)?noise|synthetic\s+noise|noisier\s+version|augment\w*[^.;\n]{0,60}noise|(?:额外|合成)噪声', evidence, re.I):
        return False
    predicts_original = r'predict\w*[^.;\n]{0,30}original\s+noisy\s+image|预测原(?:始)?(?:噪声|带噪|含噪)图像'
    return not re.search(predicts_original, claim, re.I) or bool(re.search(r'predict\s+Y\s+given\s+Z|predict\w*[^.;\n]{0,30}original\s+noisy\s+image|预测原(?:始)?(?:噪声|带噪|含噪)图像', evidence, re.I))
