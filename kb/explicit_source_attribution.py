from __future__ import annotations

import re

from kb.evidence_term_mapping import _METHOD_IDENTITY_EQUIVALENTS, specific_method_identities
from kb.citation_sentences import citation_sentence_spans
from kb.explicit_training_assignments import explicit_training_assignments_covered
from kb.explicit_noise_relations import named_synthetic_sample_covered
from kb.explicit_source_details import explicit_source_details_covered


_MODEL_ESTIMATION = re.compile(r'estimat\w*\s+(?:(?:the|a|an|unknown)\s+){0,3}noise\s+(?:model|distribution)|noise\s+(?:model|distribution)\s+(?:is|are|can\s+be|may\s+be)\s+(?:estimated|learned)|噪声(?:模型|分布)[^。；;\n]{0,20}(?:估计|估测)|估计(?:的)?未知量', re.I)


def _affirmative_model_estimation(text: str) -> bool:
    for match in _MODEL_ESTIMATION.finditer(text):
        clause=re.split(r'[.;。；:：]|\b(?:but|whereas|however)\b|但|而',text[:match.end()],flags=re.I)[-1]
        if not re.search(r'\b(?:not|never|cannot|without)\b|不|无法|不能|不可',clause,re.I):
            return True
    return False


def _affirmative_external_data(text: str) -> bool:
    for match in re.finditer(r'\bexternal\s+(?:(?:noisy|clean|training)\s+)?(?:data(?:sets?)?|images?)\b|外部(?:训练)?(?:数据集|图像)', text, re.I):
        prefix = re.split(r'[.;。；]|\b(?:but|whereas|however)\b|但|而', text[:match.start()], flags=re.I)[-1][-75:]
        if not re.search(r'\b(?:no|not|without|never)\b|不(?:需要|要求|使用)|无需|无须|不用', prefix, re.I):
            return True
    return False


def _owns_fact(text: str, source_name: str) -> bool:
    return bool(specific_method_identities(text) & specific_method_identities(source_name)
                or any(re.search(rf'(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])', source_name, re.I) for name in specific_method_identities(text))
                or re.search(r'\b(?:our|this)\s+(?:paper|method|model|approach|network|framework)|\bwe\s+(?:(?:can|then|also|do\s+not)\s+)?(?:use|train|require|estimate|learn|draw|sample|propose|assume|consider)|本文|本方法|本研究|我们的(?:方法|模型)',text,re.I))


def _owned_predicate(evidence: str, source_name: str, predicate) -> bool:
    return any(_owns_fact(part,source_name) and predicate(part)
               for start,end in citation_sentence_spans(evidence)
               for part in re.split(r'\b(?:but|while|whereas|however)\b|[。；;]|\.(?=\s+[A-Z])|但|而',evidence[start:end],flags=re.I))


def source_attribution_covered(claim: str, evidence: str, source_name: str) -> bool:
    """Keep named displays, baseline lists and training provenance in the quote."""
    if not explicit_training_assignments_covered(claim, evidence) or not named_synthetic_sample_covered(claim, evidence) or not explicit_source_details_covered(claim, evidence):
        return False
    named_results = ((r'proposition|命题', 'proposition'), (r'theorem|定理', 'theorem'), (r'lemma|引理', 'lemma'), (r'corollary|推论', 'corollary'))
    for aliases, _kind in named_results:
        for number in re.findall(rf'(?:\b(?:{aliases})|(?:{aliases}))\s*([0-9]+)(?![0-9])', claim, re.I):
            if not re.search(rf'(?:{aliases})\s*{number}(?![0-9])', evidence, re.I):
                return False
    no_model = r'(?:needs?\s+no|(?:does?|do)\s+not\s+(?:need|require)|requir\w*\s+no|without\s+(?:requiring|needing))[^.;。；\n]{0,55}noise[- ]+(?:model|distribution)|(?:不需要|无需|无须)[^。；\n]{0,20}噪声(?:模型|分布)'
    if re.search(no_model, claim, re.I):
        predicate = lambda part: bool(re.search(no_model, part, re.I))
        if not predicate(evidence) or (_owned_predicate(claim, source_name, predicate) and not _owned_predicate(evidence, source_name, predicate)):
            return False
    correction = r'correction\s+step|校正步骤|修正步骤'
    if re.search(correction, claim, re.I) and not re.search(correction + r'|doubl\w*[^.;\n]{0,70}subtract\w*', evidence, re.I):
        return False
    exclusion = r'receptive\s+field[^.;。\n]{0,140}(?:exclud\w*|except\s+for)|(?:target|center)\s+(?:pixel|value)[^.;\n]{0,60}(?:invisible|not\s+visible|never\s+visible)|blind[- ]?\s*spot[^.;\n]{0,40}removes?[^.;\n]{0,30}input\s+pixel|感受野[^。；\n]{0,50}(?:排除|不包含)'
    if re.search(exclusion, claim, re.I):
        direct = re.search(r'receptive\s+field[^.;\n]{0,80}exclud\w*[^.;\n]{0,35}(?:pixel\s+itself|center\s+pixel)|(?:affected\s+by|depends\s+on)[^.;\n]{0,110}except\s+(?:for\s+)?(?:the\s+)?pixel\s+itself|center\s+(?:pixel|value)[^.;\n]{0,40}(?:blocked|removed|excluded)|感受野[^。；\n]{0,50}(?:排除|不包含)[^。；\n]{0,15}(?:中心|目标)像素', evidence, re.I)
        negated = re.search(r'\b(?:not|never)\s+(?:exclud\w*|blocked|removed)|without\s+(?:a\s+)?blind[- ]spot|不(?:会)?排除', evidence, re.I)
        if not direct or negated:
            return False
    clean_independence = r'(?:差值|difference|variance\s+term|噪声方差项)[^.;。；\n]{0,45}(?:does\s+not\s+depend\s+on|is\s+independent\s+of|不依赖于?)[^.;。；\n]{0,25}(?:clean\s+(?:reference|target)|ground[- ]truth|干净(?:参照|目标))'
    if re.search(clean_independence, claim, re.I) and not re.search(clean_independence, evidence, re.I):
        return False
    for number, panel in re.findall(r'(?:\b(?:Figure|Fig\.?)|图)\s*(\d+)\s*\(([a-z])\)\s*(?:states?|shows?|illustrates?|depicts?|说明|显示|表明)', claim, re.I):
        label = rf'(?:\b(?:Figure|Fig\.?)|图)\s*{number}(?!\d)'
        if not (re.search(label + rf'\s*(?:\({panel}\)|{panel}\b)', evidence, re.I) or (re.search(label, evidence, re.I) and re.search(rf'\({panel}\)', evidence, re.I))):
            return False
    for concept in (r'partial\s+convolutions?|部分卷积', r'blind[- ]?spot|盲点', r'model\s+uncertainty|模型不确定性', r'statistical\s+independence|统计独立性', r'\bvariance\b|方差'):
        if re.search(concept, claim, re.I) and not re.search(concept, evidence, re.I):
            return False
    speed = r'(?:speed\w*\s+(?:up\s+)?(?:the\s+)?train\w*|(?:fast\w*|accelerat\w*)\s+(?:and\s+\w+\s+)?train\w*|训练(?:速度|加速)|加速训练)'
    if re.search(speed, claim, re.I) and not re.search(speed, evidence, re.I):
        return False
    # A passage about the same method does not substantiate an attributed
    # verbatim quotation. Preserve case and all logical words inside it.
    if re.search(r'paper|authors?|abstract|caption|figure|related[- ]work|explicitly|原文|作者|摘要|图注', claim, re.I) and not re.search(r'\b(?:not|never)\s+(?:state|say|claim)|并未|没有(?:说|声称|指出)', claim, re.I):
        # Token boundaries preserve words, variable case and logical operators,
        # while mathematical punctuation can have typesetting spaces.
        tokens = lambda value: re.findall(r'\w+|[^\w\s]', value.replace('’', "'").replace('‘', "'"))
        source_tokens = tokens(evidence)
        for match in re.finditer(r'"([^"\n]{40,})"|“([^”\n]{40,})”', claim):
            wanted = tokens((match[1] or match[2]).rstrip(',.;:'))
            if not any(source_tokens[i:i + len(wanted)] == wanted for i in range(len(source_tokens) - len(wanted) + 1)):
                return False
    if re.search(r'\binpaint\w*\b|图像修补|图像补全', claim, re.I) and not re.search(r'\binpaint\w*\b|图像修补|图像补全', evidence, re.I):
        return False
    pixels = r'\b(?:un[- ]?corrupted|non[- ]?corrupted)\s+pixels?\b|未(?:被|受)?(?:损坏|污染)的?像素'
    if re.search(pixels, claim, re.I) and not re.search(pixels, evidence, re.I):
        return False
    if re.search(r'noise\s+(?:model|distribution)|噪声(?:模型|分布)',claim,re.I) and _affirmative_model_estimation(claim):
        if not _affirmative_model_estimation(evidence) or (_owns_fact(claim,source_name) and not _owned_predicate(evidence,source_name,_affirmative_model_estimation)):
            return False
    if re.search(r'\bnotation\b|记号|符号.{0,12}(?:写为|记为)', claim, re.I):
        def indexed(text: str) -> set[tuple[str,str,str]]:
            return {(m[1] or '', m[2] or m[3], re.sub(r'\s+','',m[4] or m[5]))
                    for math in re.finditer(r'\${1,2}([^$]+)\${1,2}', text)
                    for m in re.finditer(r'(?:\\(boldsymbol|mathbf|mathcal)\{([A-Za-z])\}|([A-Za-z]))(?:_\s*|\s+)(?:\{([^{}]+)\}|([A-Za-z0-9]+))', math[1])}
        if not indexed(claim) <= indexed(evidence):
            return False
    for metric in (r'Pearson|皮尔逊', r'linear(?:ly)?\s+correlat|线性相关', r'covariance|协方差'):
        if re.search(metric, claim, re.I) and not re.search(metric, evidence, re.I):
            return False
    displays = re.findall(r'\b(?:Figure|Fig\.?)\s*(\d+)\s*(?:(?:likewise|also)\s+)?(?:states?|shows?|illustrates?|depicts?|labels?|confirms?|makes?\s+[^.;]{0,25}explicit)\b|图\s*(\d+)\s*(?:明确)?(?:说明|显示|表明|展示|标注|证实)', claim, re.I)
    for english, chinese in displays:
        number = english or chinese
        if not re.search(rf'\b(?:Figure|Fig\.?)\s*{re.escape(number)}(?!\d)|图\s*{re.escape(number)}(?!\d)', evidence, re.I):
            return False
    performance_nonimplication = re.search(r'不(?:意味(?:着)?|代表)[^。；;]{0,45}(?:质量|性能)[^。；;]{0,20}更(?:高|好)|does\s+not\s+(?:mean|imply)[^.;]{0,50}(?:better|higher)[^.;]{0,20}(?:quality|performance)', claim, re.I)
    if performance_nonimplication and not re.search(r'(?:cannot|not)\s+be\s+expected[^.;]{0,120}outperform|不预期[^。；;]{0,50}(?:胜过|超过)|不保证[^。；;]{0,40}(?:质量|性能)', evidence, re.I):
        return False
    if _affirmative_external_data(claim):
        if not _affirmative_external_data(evidence) or (_owns_fact(claim,source_name) and not _owned_predicate(evidence,source_name,_affirmative_external_data)):
            return False
    if re.search(r'bias\s*[-–—/]\s*variance|偏差\s*[-–—/]\s*方差', claim, re.I):
        if not (re.search(r'\bbias\b|偏差', evidence, re.I) and re.search(r'\bvariance\b|方差', evidence, re.I)):
            return False
    if re.search(r'partition|划分', claim, re.I) and re.search(r'(?<![A-Za-z])J(?![A-Za-z])', claim):
        if not (re.search(r'partition|划分', evidence, re.I) and re.search(r'(?<![A-Za-z])J(?![A-Za-z])', evidence)):
            return False
    if re.search(r'\bbaselines?\b|\be\.g\.|\breported\s+comparisons?\b|hyperparameters?|比较基线|对比基线|超参数', claim, re.I):
        wanted = specific_method_identities(claim)
        present = specific_method_identities(evidence)
        present.update(name for name in wanted if re.search(rf'(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])', source_name, re.I))
        for equivalents in _METHOD_IDENTITY_EQUIVALENTS:
            if present & equivalents:
                present.update(equivalents)
        if wanted - present:
            return False
    return True
