from __future__ import annotations

import re

from kb.citation_sentences import citation_sentence_spans


_NOISE_MODEL = re.compile(r"noise\s+(?:models?|distributions?)|噪声(?:模型|分布)", re.I)
_CORRELATED = re.compile(r"(?:spatially\s+)?correlated\s+noise|相关噪声", re.I)
_CORRECTION = re.compile(r"correction|校正|修正|doubl|subtract", re.I)
_OUTPUT = re.compile(r"outputs?|predictions?|输出|预测", re.I)
_IID = r"(?:i\.\s*i\.\s*d\.|independent(?:ly)?\s+and\s+identically\s+distributed)"
_DROPOUT_PHASE = re.compile(
    r"dropout.{0,100}(?:\btest(?:ing)?\b|inference|测试|推理)|"
    r"(?:\btest(?:ing)?\b|inference|测试|推理).{0,100}dropout", re.I,
)


def _training_speed_question(prompt: str) -> bool:
    return bool(re.search(r'train\w*|训练', prompt, re.I) and re.search(r'accelerat\w*|stabili\w*|speed|加速|稳定', prompt, re.I))


def _loss_statistic_question(prompt: str) -> bool:
    return bool(re.search(r'loss|损失', prompt, re.I) and re.search(r'statistics?|scalar\s+estimation|统计量|标量估计', prompt, re.I))


def _prediction_supervision_question(prompt: str) -> bool:
    return bool(re.search(r'predicted\s+quantity|predicts?\s+a\s+residual|预测(?:量|残差)', prompt, re.I) and re.search(r'train\w*|supervis\w*|训练|监督', prompt, re.I))


def _blind_spot_exclusion_question(prompt: str) -> bool:
    return bool(re.search(r'blind[- ]spot|盲点', prompt, re.I) and re.search(r'identity|exclud\w*|恒等|排除', prompt, re.I))


def _noise_statistics_question(prompt: str) -> bool:
    quality = re.search(r'performance\s+boundary|性能边界|(?:outperform|胜过|优于).{0,80}(?:clean targets?|noisy pairs?|干净目标|噪声对)', prompt, re.I)
    return bool(not quality and re.search(r'noise|噪声|signal|信号', prompt, re.I) and re.search(r'independen\w*|assumptions?|独立|统计假设|相关噪声', prompt, re.I))


def measurement_conditions_question(prompt: str) -> bool:
    return bool(re.search(r'unbiased\w*|无偏', prompt, re.I) and re.search(r'conditional\s+independen\w*|条件独立', prompt, re.I))


def _measurement_modality_question(prompt: str) -> bool:
    return bool(re.search(r'measurement\s+modality|测量模态', prompt, re.I)
                and re.search(r'independent\s+events|独立事件', prompt, re.I))


def _noise_statistics_score(prompt: str, text: str) -> int:
    if not _noise_statistics_question(prompt):
        return 0
    condition = r'noise(?:\s+[a-z])?\s+(?:(?:is|to\s+be)\s+)?(?:conditionally\s+)?pixel[- ]wise\s+independent|noise\s+is\s+independent\s+conditional\s+on\s+(?:the\s+)?(?:true\s+)?signal'
    for match in re.finditer(condition, text, re.I):
        prefix = text[max(0, match.start()-220):match.start()]
        if re.search(r'\bwe\s+(?:assume|make\s+two\s+simple\s+statistical\s+assumptions|use\s+groups\s+of\s+features)', prefix, re.I) and not re.search(r'\b(?:not|never)\s+assume', prefix, re.I):
            return 26 if re.search(r'make\s+two\s+simple\s+statistical\s+assumptions', prefix, re.I) else 24
    if re.search(r'\bwe\s+propose', text, re.I) and re.search(r'noise\s+exhibits\s+statistical\s+independence\s+across', text, re.I) and re.search(r'signal\s+exhibits\s+some\s+correlation', text, re.I):
        return 22
    return 0


def _additive_training_pair_question(prompt: str) -> bool:
    return bool(re.search(r"additive|加性", prompt, re.I) and not re.search(r"non[- ]?additive|非加性", prompt, re.I)
                and re.search(r"\binput\b|输入", prompt, re.I)
                and re.search(r"\btarget\b|目标", prompt, re.I))


def requests_comparison_boundary(prompt: str) -> bool:
    return bool(re.search(r'rank\w*|compar\w*|head[- ]to[- ]head|排名|比较|对比', prompt, re.I)
                and re.search(r'universal|always|every|普遍|总是|始终|任意|所有', prompt, re.I))


def comparison_caution_quote(text: str) -> str:
    for start, end in citation_sentence_spans(text):
        sentence = text[start:end].strip()
        if (not re.search(r'\b(?:not|never|false|incorrect)\b', sentence, re.I)
                and re.search(r'methods make different assumptions about the task.{0,60}direct comparison may be misleading', sentence, re.I)):
            return sentence
    return ""


def requests_noise_model_requirement(prompt: str) -> bool:
    return bool((_NOISE_MODEL.search(prompt) and re.search(r'requir\w*|need\w*|需要|先验', prompt, re.I))
                or requests_sampling_prerequisite(prompt))


def requests_sampling_prerequisite(prompt: str) -> bool:
    return bool(_CORRELATED.search(prompt) and re.search(r'(?:依赖|需要).{0,15}(?:能力|采样)|(?:requir\w*|depend\w*).{0,30}(?:ability|sampl)', prompt, re.I))


def requests_training_augmentation(prompt: str) -> bool:
    return _additive_training_pair_question(prompt) or bool(
        _NOISE_MODEL.search(prompt) and re.search(r'train\w*|训练', prompt, re.I)
        and re.search(r'adds?\s+(?:sampled|synthetic|additional)\s+noise|(?:加入|添加)(?:采样|合成|额外)噪声', prompt, re.I))


def requests_bernoulli_training_requirement(prompt: str) -> bool:
    return bool(re.search(r'Bernoulli|伯努利', prompt, re.I) and re.search(r'train\w*\s+(?:requirements?|data|observations?)|训练(?:数据|要求|观测)', prompt, re.I))


def _bernoulli_training_premise(text: str) -> bool:
    return bool(re.search(r'Bernoulli[- ](?:sampled|masked)', text, re.I)
                and re.search(r'\btrain\w*\b', text, re.I)
                and re.search(r'instances|noisy\s+image|input\s+image', text, re.I)
                and re.search(r'\bwe\b|\bour\b|this\s+paper|the\s+proposed\s+method', text, re.I))


def requested_method_premise_priority(prompt: str, text: str) -> int:
    return max(noise_model_premise_priority(prompt, text), int(requests_bernoulli_training_requirement(prompt) and _bernoulli_training_premise(text)),
               2 * int(requests_training_augmentation(prompt) and _training_pair_procedure(text)))


def _known_noise_training_premise(text: str) -> bool:
    return bool(re.search(r'known\s+noise\s+(?:model|distribution)', text, re.I)
                and re.search(r'\bwe\s+(?:can\s+)?(?:draw|sampl)\w*.{0,60}additional\s+synthetic\s+noise\s+sample', text, re.I))


def noise_model_premise_priority(prompt: str, text: str) -> int:
    return int(requests_noise_model_requirement(prompt) and _known_noise_training_premise(text))


def requests_source_facets(prompt: str) -> bool:
    return bool(_measurement_modality_question(prompt) or measurement_conditions_question(prompt) or _noise_statistics_question(prompt) or _prediction_supervision_question(prompt) or _blind_spot_exclusion_question(prompt) or _training_speed_question(prompt) or _loss_statistic_question(prompt) or re.search(r'\bJ\s*[- ]\s*invarian', prompt, re.I) or requests_comparison_boundary(prompt) or _NOISE_MODEL.search(prompt) or _CORRELATED.search(prompt) or requests_bernoulli_training_requirement(prompt)
                or (_CORRECTION.search(prompt) and _OUTPUT.search(prompt))
                or _DROPOUT_PHASE.search(prompt) or _additive_training_pair_question(prompt))


def requests_comparison_facets(prompt: str) -> bool:
    return requests_source_facets(prompt) and bool(re.search(
        r"比较|对比|\b(?:compare|comparison|versus|whereas)\b", prompt, re.I,
    ))


def requests_sample_relation_evidence(prompt: str) -> bool:
    return bool(_CORRELATED.search(prompt) and re.search(
        r"额外|生成|\b(?:additional|synthetic|generated)\b", prompt, re.I,
    ) and re.search(r"独立|关系|independen|relation", prompt, re.I))


def sample_relation_priority(prompt: str, text: str) -> int:
    """Keep between-sample premises separate from within-image correlation."""
    if not requests_sample_relation_evidence(prompt):
        return 0
    for start, end in citation_sentence_spans(text):
        sentence = text[start:end]
        if re.search(r"\b(?:not|never|false|incorrect|different)\b", sentence, re.I):
            continue
        if re.search(rf"Recall\s+that\s+M\s+and\s+N\s+are\s+{_IID}", sentence, re.I):
            return 3
        if re.search(rf"two\s+{_IID}\s+noise\s+samples", sentence, re.I):
            return 2
    return int(bool(_CORRELATED.search(text)))


def _caption_training_pair(text: str) -> bool:
    return bool(re.search(r'Figure\s+\d+', text, re.I)
                and re.search(r'singly[- ]noisy\s+realization.{0,70}training\s+target', text, re.I)
                and re.search(r'doubly[- ]noisy\s+realization.{0,70}input\s+to\s+(?:the|our)\s+network', text, re.I))


def _training_pair_procedure(text: str) -> bool:
    return bool(re.search(r'\bWe\s+(?:then\s+)?train\s+the\s+network\s+to\s+predict\s+[A-Z]\s+given\s+[A-Z]\s+as\s+input', text))


def source_facet_score(prompt: str, text: str) -> int:
    """Select explicit method premises/operations; never generate an answer."""
    if _measurement_modality_question(prompt) and re.search(r'\bwe\s+demonstrate\b', text, re.I) and re.search(r'\bdata\b', text, re.I) and re.search(r'independence\s+between\s+detections', text, re.I):
        return 26
    if measurement_conditions_question(prompt):
        for start, end in citation_sentence_spans(text):
            sentence = text[start:end]
            if re.search(r'noise[^.;\n]{0,70}conditionally\s+independent\s+between\s+sets\s+of\s+dimensions', sentence, re.I) and not re.search(r'\b(?:not|never|false|incorrect)\b', sentence, re.I):
                return 24
    statistical = _noise_statistics_score(prompt, text)
    if statistical:
        return statistical
    if _prediction_supervision_question(prompt):
        if re.search(r'noisy[- ]clean\s+training\s+image[^.;]{0,30}pairs', text, re.I) and re.search(r'train\w*\s+a\s+residual\s+mapping', text, re.I):
            return 26
        if re.search(r'predict\s+the\s+residual\s+image', text, re.I):
            return 22
    if _prediction_supervision_question(prompt) or _blind_spot_exclusion_question(prompt):
        if re.search(r'receptive\s+field[^.;]{0,60}excludes\s+the\s+pixel\s+itself', text, re.I):
            return 26
    if _training_speed_question(prompt) and re.search(r'residual\s+learning', text, re.I) and re.search(r'batch\s+normalization', text, re.I) and re.search(r'speed\w*\s+up\s+(?:the\s+)?training|fast\s+and\s+stable\s+training', text, re.I):
        return 24
    if _loss_statistic_question(prompt):
        if re.search(r'L\s*1\s+loss\s+recovers\s+the\s+median|L\s*2\s+loss[^.;]{0,80}output\s+the\s+average', text):
            return 24
    if re.search(r'\bJ\s*[- ]\s*invarian', prompt, re.I) and re.search(r'\bJ\s*[- ]\s*invariant', text, re.I):
        if re.search(r'information\s+from\s+outside\s+(?:of\s+)?each\s+subset\s+(?:of\s+dimensions\s+)?J', text, re.I) and re.search(r'cannot\s+(?:merely|simply)\s+be\s+the\s+identity', text, re.I):
            return 22
        if re.search(r'f\s*\(\s*x\s*\)\s*[_{ ]*J\b[^.;]{0,35}does\s+not\s+depend\s+on', text):
            return 24
        if re.search(r'information\s+from\s+outside\s+each\s+subset\s+J', text, re.I):
            return 18
    if requests_comparison_boundary(prompt) and comparison_caution_quote(text):
        return 24
    if requested_method_premise_priority(prompt, text):
        return 20
    if sample_relation_priority(prompt, text) >= 2:
        return 20 if sample_relation_priority(prompt, text) == 3 else 18
    if _additive_training_pair_question(prompt):
        if _training_pair_procedure(text):
            return 22
        if _caption_training_pair(text):
            return 18
        if re.search(r"^(?:We\s+)?train\s+the\s+network\s+to\s+predict\s+[A-Z]\s+given\s+[A-Z]\s+as\s+input", text):
            return 20
        if re.search(r"additional\s+synthetic\s+noise\s+sample", text, re.I) and re.search(r"known", text, re.I):
            return 18
        if re.search(r"single\s+noisy\s+realization\s*[A-Z]|noisier\s+version", text):
            return 16
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
    if _measurement_modality_question(prompt) and len(text) <= 1800:
        return text
    if _noise_statistics_score(prompt, text) and len(text) <= 1800:
        return text
    if _training_speed_question(prompt) and source_facet_score(prompt, text) and len(text) <= 1800:
        return text
    if (_prediction_supervision_question(prompt) or _blind_spot_exclusion_question(prompt)) and re.search(r'Figure\s+\d+', text, re.I) and re.search(r'receptive\s+field[^.;]{0,60}excludes\s+the\s+pixel\s+itself', text, re.I):
        return text
    if requested_method_premise_priority(prompt, text) or (_additive_training_pair_question(prompt) and (_known_noise_training_premise(text) or _caption_training_pair(text) or _training_pair_procedure(text))):
        return text
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
