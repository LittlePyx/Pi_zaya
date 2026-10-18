from __future__ import annotations

import pytest

from api.claim_passage_plan import _supports
from kb.claim_evidence_runtime import _support_score
from kb.explicit_noise_relations import iid_relation_covered
from kb.evidence_binding import assess_system_a_hit_binding
from kb.source_passage_evidence import claim_passage_concepts_covered


def test_iid_abbreviation_covers_the_complete_relation_at_the_unchanged_support_threshold():
    claim='原始噪声与额外生成噪声之间（U,V i.i.d.）'
    source='Recall that U and V are i.i.d. and therefore have exchangeable roles in our noise sampling construction.'
    assert _support_score(claim,source)>=5
    assert _supports(claim,source,'','unseen.pdf')
    binding=assess_system_a_hit_binding(answer_claim=claim,hit={'text':source},meta={},heading='',evidence_quote=source,source_name='unseen.pdf')
    assert not binding['suppress_link']


def test_bare_iid_source_states_the_complete_relation_without_extra_keyword_padding():
    claim='原始噪声与额外生成噪声之间（$M,N$ i.i.d.）'
    source='Recall that M and N are i.i.d. and therefore and thus'
    assert _supports(claim,source,'','unseen.pdf')


@pytest.mark.parametrize('source',[
    'We allow spatially correlated noise models and require the ability to sample noise.',
    'Our method also handles pixel-wise i.i.d. white noise.',
    'Recall that A and B are i.i.d. samples of a noise model.',
    'Recall that U and V are independent but have different distributions.',
    'Recall that U and V have the same marginal distribution but can be dependent.',
    'Recall that U and V are not necessarily i.i.d.',
])
def test_sample_relation_cannot_borrow_a_partial_different_or_pixel_only_statement(source):
    claim='U 和 V 为独立同分布的两份噪声（i.i.d.）'
    assert not iid_relation_covered(claim,source)
    assert not _supports(claim,source,'','unseen.pdf')
    binding=assess_system_a_hit_binding(answer_claim=claim,hit={'text':source},meta={},heading='',evidence_quote=source,source_name='unseen.pdf')
    assert binding['suppress_link']


def test_raw_output_claim_needs_its_explicit_noisiness_not_a_related_correction_operation():
    claim='Note that without our correction step, the raw output of the network is still noisy.'
    source='We recover the clean estimate by doubling our network output and subtracting its input.'
    assert not claim_passage_concepts_covered(claim,source)
    assert not _supports(claim,source,'','unseen.pdf')
    assert _supports(claim,claim,'','unseen.pdf')


def test_clean_output_or_negated_noisiness_cannot_support_a_still_noisy_claim():
    claim='The raw output is still noisy.'
    for source in ('The raw output is not noisy.','The raw output is a clean image.','The raw output is not a clean image.'):
        assert not claim_passage_concepts_covered(claim,source)
    assert not claim_passage_concepts_covered('The raw output is not still noisy.',claim)
