from __future__ import annotations

import pytest

from api.claim_passage_plan import _supports
from kb.claim_evidence_runtime import audit_and_repair_claim_evidence
from kb.claim_source_eligibility import complete_source_prefix, incomplete_source_prose, retrieval_limit_claim
from kb.evidence_binding import assess_system_a_hit_binding
from kb.source_question_facets import source_facet_score


@pytest.mark.parametrize('claim',[
    'Not stated in the retrieved evidence: the exact probability p used.',
    'The current retrieved evidence cannot establish a universal ranking.',
    'The retrieved excerpts do not specify the probability p.',
    'The provided snippets do not report the probability p.',
    "The selected passages don't give the probability p.",
    '当前检索证据无法确认这项试验。',
])
def test_evidence_limit_is_not_a_source_fact_even_if_terms_overlap(claim):
    quote='The probability p is used for Bernoulli sampling during training of the denoising network.'
    assert retrieval_limit_claim(claim)
    assert not _supports(claim,quote,'Training','method.pdf')
    assert assess_system_a_hit_binding(answer_claim=claim,hit={'text':quote},meta={'citation_plan_evidence_authoritative':True,'citation_plan_full_evidence_quote':quote},heading='Training',evidence_quote=quote,source_name='method.pdf')['suppress_link']


def test_nonnumeric_short_limit_keeps_text_but_cannot_keep_a_false_source_marker():
    claim='Not stated in the retrieved evidence: the exact probability p used.'
    quote='The probability p is used for Bernoulli sampling during training of the denoising network.'
    answer,_=audit_and_repair_claim_evidence(claim+' [1]',[{'text':quote,'meta':{'source_name':'method.pdf'}}],allowed_citation_numbers={1},drop_unsupported_high_risk_claims=True,enforce_user_visible_binding=True)
    assert claim in answer and '[1]' not in answer


@pytest.mark.parametrize('tail',['and therefore','and thus','and therefore and thus'])
def test_complete_atomic_iid_fact_survives_a_missing_following_continuation(tail):
    quote='Recall that M and N are i.i.d. '+tail
    assert incomplete_source_prose(quote)
    assert complete_source_prefix(quote)=='Recall that M and N are i.i.d.'
    assert _supports('M and N are i.i.d. noise samples.',quote,'Method','method.pdf')
    assert not _supports('M and N have variance 900.',quote,'Method','method.pdf')
    assert source_facet_score('空间相关噪声的额外样本独立关系',quote)>0


def test_complete_iid_source_and_quoted_paper_limitation_still_qualify():
    source='M and N are i.i.d. noise samples, and therefore they have exchangeable roles.'
    assert not incomplete_source_prose(source)
    assert _supports('M and N are i.i.d. noise samples.',source,'Method','method.pdf')
    assert not retrieval_limit_claim('The authors state that their method cannot recover arbitrary correlated noise.')
