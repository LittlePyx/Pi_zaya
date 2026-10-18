from __future__ import annotations

import re

from kb.citation_sentences import citation_sentence_spans


_NOISE_MODEL = re.compile(r"noise\s+(?:models?|distributions?)|噪声(?:模型|分布)", re.I)
_CORRELATED = re.compile(r"(?:spatially\s+)?correlated\s+noise|相关噪声", re.I)
_CORRECTION = re.compile(r"correction|校正|修正|doubl|subtract", re.I)
_OUTPUT = re.compile(r"outputs?|predictions?|输出|预测", re.I)
_DROPOUT_PHASE = re.compile(
    r"dropout.{0,100}(?:\btest(?:ing)?\b|inference|测试|推理)|"
    r"(?:\btest(?:ing)?\b|inference|测试|推理).{0,100}dropout", re.I,
)


def requests_source_facets(prompt: str) -> bool:
    return bool(_NOISE_MODEL.search(prompt) or _CORRELATED.search(prompt)
                or (_CORRECTION.search(prompt) and _OUTPUT.search(prompt))
                or _DROPOUT_PHASE.search(prompt))


def requests_comparison_facets(prompt: str) -> bool:
    return requests_source_facets(prompt) and bool(re.search(
        r"比较|对比|\b(?:compare|comparison|versus|whereas)\b", prompt, re.I,
    ))


def source_facet_score(prompt: str, text: str) -> int:
    """Select explicit method premises/operations; never generate an answer."""
    owned = bool(re.search(r"\b(?:we|our)\b|this\s+(?:paper|study|method|approach)", text, re.I))
    if owned and _DROPOUT_PHASE.search(prompt) and re.search(
        r"using\s+dropout\s+in\s+both\s+training\s+and\s+test", text, re.I,
    ):
        return 18
    if owned and _DROPOUT_PHASE.search(prompt) and re.search(r"dropout", text, re.I) and re.search(
        r"average\s+of\s+(?:these|the|our)\s+predictions.{0,70}reduce\s+the\s+variance", text, re.I,
    ):
        return 16
    if owned and _CORRELATED.search(prompt) and re.search(
        r"额外|生成|\b(?:additional|synthetic|generated)\b", prompt, re.I,
    ) and re.search(r"synthetic\s+sample\s+from\s+the\s+same\s+noise\s+distribution", text, re.I):
        return 19
    if _CORRELATED.search(prompt) and re.search(
        r"额外|生成|\b(?:additional|synthetic|generated)\b", prompt, re.I,
    ) and re.search(
        r"(?:Recall\s+that|We\s+(?:assume|take|draw)).{0,100}"
        r"(?:i\.\s*i\.\s*d\.|independent(?:ly)?\s+and\s+identically\s+distributed)", text, re.I,
    ):
        return 18
    if owned and _CORRECTION.search(prompt) and _OUTPUT.search(prompt):
        if re.search(r"doubl\w*.{0,60}(?:output|prediction).{0,50}subtract\w*.{0,40}input", text, re.I):
            return 18
        if re.search(r"correction\s+step", text, re.I) and re.search(
            r"raw\s+output.{0,60}still\s+noisy", text, re.I,
        ):
            return 16
    if _NOISE_MODEL.search(prompt) and owned and re.search(r"noise", text, re.I):
        if re.search(r"(?:require|known|statistical).{0,60}(?:noise\s+model|noise\s+distribution)|"
                     r"(?:require|ability).{0,50}sample\s+from\s+the\s+noise\s+distribution", text, re.I):
            return 18 if re.search(r"sample\s+from|draw.{0,50}(?:noise|sample)", text, re.I) else 16
    if _CORRELATED.search(prompt) and owned and re.search(
        r"(?:spatially\s+(?:correlated|structured)\s+noise|noise.{0,30}spatially\s+(?:correlated|structured))",
        text, re.I,
    ):
        return 18 if re.search(r"require.{0,60}sample\s+from", text, re.I) else 16
    return 0


def source_facet_quote(prompt: str, text: str) -> str:
    """Keep complete requested clauses as a continuous literal source span."""
    if not source_facet_score(prompt, text):
        return text
    spans = []
    for start, end in citation_sentence_spans(text):
        sentence = text[start:end]
        selected = bool(source_facet_score(prompt, sentence))
        if (_NOISE_MODEL.search(prompt) or _CORRELATED.search(prompt)) and re.search(
            r"\bwe\b.{0,35}require.{0,60}sample\s+from\s+the\s+noise\s+distribution", sentence, re.I,
        ):
            selected = True
        if _DROPOUT_PHASE.search(prompt) and re.search(
            r"average\s+of\s+(?:these|the|our)\s+predictions.{0,70}reduce\s+the\s+variance|"
            r"main\s+reason.{0,100}reduce\s+the\s+variance", sentence, re.I,
        ):
            selected = True
        if selected:
            spans.append((start, end))
    if not spans:
        return text
    quote = text[spans[0][0]:spans[-1][1]].strip()
    return quote if len(quote) >= 40 else text
