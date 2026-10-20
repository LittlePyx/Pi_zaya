from __future__ import annotations

from pathlib import Path
import re

from kb.citation_marker_context import numeric_marker_is_content
from kb.n2n_target_condition import _verified_block
from kb.source_question_facets import _known_noise_training_premise, requests_noise_model_requirement


def normalize_verified_noise_sampling_terms(answer: str, *, prompt: str, citation_plan: dict | None) -> str:
    """Use the technical term 采样 for a verified noise-distribution draw.

    This only normalizes an already written verb. It cannot add a premise,
    variable, capability or citation, and leaves literal/code/math text alone.
    """
    if not requests_noise_model_requirement(str(prompt or '')):
        return answer
    slots = [s for s in (citation_plan or {}).get('slots') or [] if isinstance(s,dict) and s.get('source_path')]
    if len({str(Path(s['source_path']).resolve()) for s in slots}) != 1:
        return answer
    verified = [_verified_block(str(s['source_path']),{**s,'text':s.get('evidence_quote')}) for s in slots if all(s.get(k) for k in ('block_id','anchor_id','heading_path','page_start','page_end'))]
    if not any(b and _known_noise_training_premise(str(b.get('raw_text') or '')) for b in verified):
        return answer
    literals = [m.span() for m in re.finditer(r'"[^"\n]*"|“[^”\n]*”|\x27[^\x27\n]*\x27|`[^`\n]*`', answer)]
    def canonical(match: re.Match) -> str:
        line = answer[answer.rfind('\n',0,match.start())+1:match.start()]
        if line.lstrip().startswith('>') or numeric_marker_is_content(answer,match.start()) or any(a<=match.start()<b for a,b in literals):
            return match[0]
        return '采样得到'
    return re.sub(r'抽取(?=(?:额外)?(?:一[个份]|单个)?(?:的)?(?:合成)?噪声样本)',canonical,answer)
