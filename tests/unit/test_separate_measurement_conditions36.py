from __future__ import annotations

from kb.claim_evidence_runtime import _is_high_risk_claim
from kb.direct_question_evidence import direct_question_evidence_hits
from kb.source_question_facets import source_facet_score


def test_a_clipped_theorem_does_not_hide_a_separately_stated_premise(tmp_path):
    source = tmp_path / 'theory.md'
    source.write_text('<!-- kb_page: 2 -->\n# Theory\n\n'
        'Figure 1. This enables self-supervision when the noise is conditionally independent between sets of dimensions.\n\n'
        'Proposition 4. Suppose x is an unbiased estimator of y, i.e.\n\nEquation\n\n'
        'dent from the noise in its complement K, conditional on y.\n', encoding='utf8')
    prompt = 'Proposition 4 uses two conditions: what are unbiasedness and conditional independence?'
    hits = direct_question_evidence_hits(source, prompt)
    assert any('unbiased estimator' in h['text'] for h in hits)
    condition = next(h for h in hits if 'conditionally independent between' in h['text'])
    theorem = next(h for h in hits if 'Proposition 4' in h['text'])
    assert condition['meta']['block_id'] != theorem['meta']['block_id']
    assert condition['meta']['page_start'] == 2
    assert not any('Proposition 4' in h['text'] and 'conditionally independent between' in h['text'] for h in hits)
    assert not direct_question_evidence_hits(source, prompt.replace('4', '9'))
    assert source_facet_score(prompt, 'The noise is not conditionally independent between sets of dimensions.') == 0


def test_statistical_premises_need_the_existing_claim_evidence_audit():
    assert _is_high_risk_claim('The noise samples M and N are independent and identically distributed.')
    assert _is_high_risk_claim('原始噪声 N 与额外噪声样本 M 独立同分布。')
    assert not _is_high_risk_claim('Read the section on independent noise for further details.')
