from __future__ import annotations

import re


def explicit_source_details_covered(claim: str, evidence: str) -> bool:
    """Veto extra operations/statistics that a nearby topic cannot establish."""
    stable_training = r'\bstabl\w*[^.;\n]{0,35}\btrain\w*|\bstabili[sz]\w*[^.;\n]{0,35}\btrain\w*|训练[^。；\n]{0,20}稳定|稳定[^。；\n]{0,20}训练'
    if re.search(stable_training, claim, re.I) and not re.search(stable_training, evidence, re.I):
        return False
    for number, statistic in ((1, r'median|中位数'), (2, r'mean|average|均值|平均')):
        loss = rf'(?<![A-Za-z])L\s*_?\s*\{{?{number}\}}?(?!\d)'
        if re.search(loss, claim, re.I) and re.search(statistic, claim, re.I):
            if not (re.search(loss, evidence, re.I) and re.search(statistic, evidence, re.I)):
                return False
    growth = re.search(r'每(?:增加|增添)一层[^。；\n]{0,50}每边(?:扩大|增加)\s*(\d+)\s*(?:个)?像素', claim)
    if growth:
        slope = re.search(r'\(\s*(\d+)\s*d\s*\+\s*\d+\s*\)', evidence, re.I)
        if slope and int(growth[1]) != int(slope[1]):
            return False
        if not slope and not re.search(re.escape(growth[0]), evidence):
            return False
    twice = r'observe\s+each\s+source\s+image\s+twice|对每个源图像观测两次|每个源图像能被观测两次'
    if re.search(twice, claim, re.I) and not re.search(r'observe\s+each\s+source\s+image\s+twice|观测两次', evidence, re.I):
        return False
    expected_clean = r'(?:expected\s+value\s+of\s+(?:the\s+)?(?:noisy|corrupted)\s+input|(?:the\s+)?(?:noisy|corrupted)\s+input\s*[’\x27]s\s+expected\s+value)[^.;\n]{0,25}(?:equals?|is|equaling)\s+(?:the\s+)?clean\s+(?:signal|target)|(?:带噪|含噪|噪声)输入的期望(?:等于|正是)[^。；\n]{0,15}干净(?:目标|信号)'
    if re.search(expected_clean, claim, re.I) and not re.search(expected_clean + r'|expectation\s+of\s+the\s+corrupted\s+input\s+data\s+is\s+the\s+clean\s+target|unbiased\s+estimator', evidence, re.I):
        return False
    mask_loss = r'loss[^.;\n]{0,65}(?:only|exclusively)[^.;\n]{0,45}masked|损失只在[^。；\n]{0,35}(?:掩蔽|遮蔽|盲点)'
    if re.search(mask_loss, claim, re.I) and not re.search(mask_loss + r'|loss[^.;\n]{0,70}measured\s+only[^.;\n]{0,60}masked', evidence, re.I):
        return False
    replacement = r'replac\w*[^.;\n]{0,60}(?:pixel|intensit)|(?:pixel|intensity|value\s+at\s+the\s+center)[^.;\n]{0,55}(?:is\s+)?replaced\s+with|(?:强度|像素值)[^。；\n]{0,12}(?:复制|替换)|把[^。；\n]{0,15}(?:强度|像素)[^。；\n]{0,10}(?:复制|替换)'
    positive_replacement = False
    for match in re.finditer(replacement, claim, re.I):
        prefix = re.split(r'[.;。；\n]', claim[:match.start()])[-1][-40:]
        prefix += re.split(r'复制|替换|replac\w*', match[0], maxsplit=1, flags=re.I)[0]
        if not re.search(r'\b(?:not|never|cannot)(?:\s+(?:be|directly|simply)){0,2}\s*$|(?:不能|不会|无法|不可|无需)(?:被|直接|简单地){0,2}\s*$', prefix, re.I):
            positive_replacement = True
    if positive_replacement:
        neighbor = r'neighbor\w*|surrounding|another\s+pixel|other\s+pixel|邻域|邻近|周围|其他像素'
        if not (re.search(neighbor, claim, re.I) and re.search(neighbor, evidence, re.I) and re.search(r'cop\w*|replac\w*|复制|替换', evidence, re.I)):
            return False
    concepts = (r'gene\s+expression|基因表达', r'individual\s+molecules|RNA\s+molecules|单个分子', r'hyperparameters?|超参数')
    if any(re.search(p, claim, re.I) and not re.search(p, evidence, re.I) for p in concepts):
        return False
    # A dependence claim about the loss gap needs that objective, not a
    # nearby figure that merely defines the input/output coordinate subsets.
    gap = r'(?:second\s+term|variance\s+term|噪声方差项|第二项)[^.;。；\n]{0,90}(?:depend|independent|不随|无关|不依赖)'
    if re.search(gap, claim, re.I) and not re.search(r'loss|objective|损失|目标函数|E\s*[∥|]|\\(?:sum|min)', evidence, re.I):
        return False
    correspondence = r'(?:sampling|采样)[^.;。；\n]{0,50}(?:dropout|丢弃)[^.;。；\n]{0,30}(?:对应|correspond)|(?:dropout|丢弃)[^.;。；\n]{0,50}(?:sampling|采样)[^.;。；\n]{0,30}(?:对应|correspond)'
    if re.search(correspondence, claim, re.I) and not (re.search(r'Bernoulli|伯努利|sampling|采样', evidence, re.I) and re.search(r'dropout|丢弃', evidence, re.I)):
        return False
    intuition = r'halfway|中间位置|一半位置'
    if re.search(intuition, claim, re.I) and re.search(r'paper|intuit|原文|论文|文中', claim, re.I) and not re.search(intuition, evidence, re.I):
        return False
    if re.search(r'J\s*[- ]?\s*(?:invariant|不变)', claim.replace('$',''), re.I):
        denied = re.search(r'(?:does\s+not\s+depend|不依赖)[^。；;\n]{0,65}(?:complement|补集)[^。；;\n]{0,15}J\s*\^?\s*\{?c', claim, re.I)
        if denied and not re.search(re.escape(denied[0]), evidence):
            return False
    return True
