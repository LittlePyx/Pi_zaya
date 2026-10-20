from __future__ import annotations

import re


def explicit_source_details_covered(claim: str, evidence: str) -> bool:
    """Veto extra operations/statistics that a nearby topic cannot establish."""
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
    expected_clean = r'expected\s+value\s+of\s+(?:the\s+)?(?:noisy|corrupted)\s+input[^.;\n]{0,25}(?:equals?|is)\s+(?:the\s+)?clean\s+(?:signal|target)|噪声输入的期望等于[^。；\n]{0,15}干净目标'
    if re.search(expected_clean, claim, re.I) and not re.search(expected_clean + r'|expectation\s+of\s+the\s+corrupted\s+input\s+data\s+is\s+the\s+clean\s+target|unbiased\s+estimator', evidence, re.I):
        return False
    mask_loss = r'loss[^.;\n]{0,65}(?:only|exclusively)[^.;\n]{0,45}masked|损失只在[^。；\n]{0,35}(?:掩蔽|遮蔽|盲点)'
    if re.search(mask_loss, claim, re.I) and not re.search(mask_loss + r'|loss[^.;\n]{0,70}measured\s+only[^.;\n]{0,60}masked', evidence, re.I):
        return False
    replacement = r'replac\w*[^.;\n]{0,60}(?:pixel|intensit)|(?:强度|像素值)[^。；\n]{0,12}(?:复制|替换)|把[^。；\n]{0,15}(?:强度|像素)[^。；\n]{0,10}(?:复制|替换)'
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
    return True
