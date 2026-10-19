from __future__ import annotations

import re

from kb.evidence_term_mapping import _METHOD_IDENTITY_EQUIVALENTS, specific_method_identities


def _affirmative_external_data(text: str) -> bool:
    for match in re.finditer(r'\bexternal\s+(?:(?:noisy|clean|training)\s+)?(?:data(?:sets?)?|images?)\b|外部(?:训练)?(?:数据集|图像)', text, re.I):
        prefix = re.split(r'[.;。；]|\b(?:but|whereas|however)\b|但|而', text[:match.start()], flags=re.I)[-1][-75:]
        if not re.search(r'\b(?:no|not|without|never)\b|不(?:需要|要求|使用)|无需|无须|不用', prefix, re.I):
            return True
    return False


def source_attribution_covered(claim: str, evidence: str, source_name: str) -> bool:
    """Keep named displays, baseline lists and training provenance in the quote."""
    for metric in (r'Pearson|皮尔逊', r'linear(?:ly)?\s+correlat|线性相关', r'covariance|协方差'):
        if re.search(metric, claim, re.I) and not re.search(metric, evidence, re.I):
            return False
    displays = re.findall(r'\b(?:Figure|Fig\.?)\s*(\d+)\s*(?:states?|shows?|illustrates?|depicts?|makes?\s+[^.;]{0,25}explicit)\b|图\s*(\d+)\s*(?:明确)?(?:说明|显示|表明|展示)', claim, re.I)
    for english, chinese in displays:
        number = english or chinese
        if not re.search(rf'\b(?:Figure|Fig\.?)\s*{re.escape(number)}(?!\d)|图\s*{re.escape(number)}(?!\d)', evidence, re.I):
            return False
    performance_nonimplication = re.search(r'不(?:意味(?:着)?|代表)[^。；;]{0,45}(?:质量|性能)[^。；;]{0,20}更(?:高|好)|does\s+not\s+(?:mean|imply)[^.;]{0,50}(?:better|higher)[^.;]{0,20}(?:quality|performance)', claim, re.I)
    if performance_nonimplication and not re.search(r'(?:cannot|not)\s+be\s+expected[^.;]{0,120}outperform|不预期[^。；;]{0,50}(?:胜过|超过)|不保证[^。；;]{0,40}(?:质量|性能)', evidence, re.I):
        return False
    if _affirmative_external_data(claim) and not _affirmative_external_data(evidence):
        return False
    if re.search(r'bias\s*[-–—/]\s*variance|偏差\s*[-–—/]\s*方差', claim, re.I):
        if not (re.search(r'\bbias\b|偏差', evidence, re.I) and re.search(r'\bvariance\b|方差', evidence, re.I)):
            return False
    if re.search(r'partition|划分', claim, re.I) and re.search(r'(?<![A-Za-z])J(?![A-Za-z])', claim):
        if not (re.search(r'partition|划分', evidence, re.I) and re.search(r'(?<![A-Za-z])J(?![A-Za-z])', evidence)):
            return False
    if re.search(r'\bbaselines?\b|\be\.g\.|\breported\s+comparisons?\b|比较基线|对比基线', claim, re.I):
        wanted = specific_method_identities(claim)
        present = specific_method_identities(evidence)
        present.update(name for name in wanted if re.search(rf'(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])', source_name, re.I))
        for equivalents in _METHOD_IDENTITY_EQUIVALENTS:
            if present & equivalents:
                present.update(equivalents)
        if wanted - present:
            return False
    return True
