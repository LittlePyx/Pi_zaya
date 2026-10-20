from __future__ import annotations

import pytest

from api.claim_passage_plan import _slot_supports, prepare_claim_passage_plan
from api.citation_occurrence_support import verified_occurrence_candidates
from kb.source_blocks import load_source_blocks


def fixture(tmp_path, raw, heading='Self2Self: Single Image Learning'):
    md = tmp_path / ('noisier2noise.en.md' if heading.startswith('Noisier') else 'self2self.en.md')
    md.write_text('<!-- kb_page: 3 -->\n# ' + heading + '\n## Method\n\n' + raw, encoding='utf8')
    b = next(b for b in load_source_blocks(md) if b['kind'] == 'paragraph')
    return {**{k: b[k] for k in ('block_id','anchor_id','heading_path','page_start','page_end')},
            'source_path':str(md),'source_name':md.name.split('.')[0],
            'evidence_quote':raw,'evidence_source':'verified_source_excerpt',
            'evidence_selection_reason':'prompt_contract_block',
            'selection_reason':'prompt_contract_block','citation_plan_slot':True}


def test_existing_verified_slot_provenance_survives_weak_literal_overlap(tmp_path):
    raw = ('Taking one step further, this paper proposes a self-supervised learning method which only uses the input noisy image itself for training. '
           'In the proposed method, the network is trained with dropout on the pairs of Bernoulli-sampled instances of the input image, '
           'and the result is estimated by averaging the predictions generated from multiple instances of the trained model with dropout.')
    s = fixture(tmp_path, raw)
    claim = 'Self2Self is the method that constructs Bernoulli-masked instances of the very image being denoised.'
    assert _slot_supports(claim, s)
    assert verified_occurrence_candidates(s, [s], claim)
    # An existing selection reason is preserved only after physical recheck;
    # no new canonical/grounded flag or lower body score is granted.
    for key, value in [('block_id','wrong'),('anchor_id','wrong'),('heading_path','Method'),('page_start',4)]:
        assert not _slot_supports(claim, {**s,key:value})
    assert not _slot_supports('The method uses 300 clinical subjects.',s)
    assert 'canonical_answer_evidence' not in s


def test_physical_title_provides_ownership_without_granting_body_facts(tmp_path):
    raw = ('Our method of augmenting training images with additional noise is similar to the method of Noise2Noise. '
           'We also allow spatially correlated noise models, which are problematic for Noise2Void and Noise2Self. '
           'On the other hand, we require the ability to sample from the noise distribution, which those methods do not.')
    s = fixture(tmp_path, raw, 'Noisier2Noise: Learning from Noisy Data')
    claim = 'Noisier2Noise 允许空间相关噪声模型，而这类噪声对 Noise2Void 和 Noise2Self 是有问题的。'
    assert verified_occurrence_candidates(s,[s],claim)
    assert not verified_occurrence_candidates(s,[{**s,'heading_path':'Related Work'}],claim)
    for invalid in ('Noisier2Noise uses 300 clinical subjects.',
                    'Noisier2Noise does not need a known noise model and cannot sample noise.',
                    'Noisier2Noise is restricted to independent white noise.'):
        assert not verified_occurrence_candidates(s,[s],invalid)


@pytest.mark.parametrize('raw',[
    'This experiment evaluates reconstruction quality on 300 images with standard image processing tools.',
    'Noise2Noise requires paired noisy training images. The input image itself is not a sufficient training dataset.',
])
def test_title_and_selection_reason_cannot_supply_missing_body_premises(tmp_path,raw):
    s=fixture(tmp_path,raw,'Noisier2Noise: Spatially Correlated Noise')
    assert not verified_occurrence_candidates(s,[s],'Noisier2Noise supports spatially correlated noise.')


def test_duplicate_intent_slots_do_not_make_one_verified_proposition_ambiguous(tmp_path):
    raw='Proposition 1. Assume the noise components are independent and of zero mean.'
    slot=fixture(tmp_path,raw)
    slot.update(candidate_hits=[1],preferred_system='system_a')
    duplicate={**slot,'claim_type':'compare_result','source_name':''}
    answer='命题1对噪声的要求是各噪声分量相互独立且均值为零 [1]。'
    rec={'meta':{'canonical_hit_paths':[slot['source_path']],'answer_quality':{'citation_plan':{'slots':[slot,duplicate]}}}}
    result=prepare_claim_passage_plan(rec,{},answer)
    slots=result['meta']['answer_quality']['citation_plan']['slots']
    assert len(slots)==1
    assert slots[0]['passage_bound_claims']
    assert 'canonical_answer_evidence' not in slots[0]
    assert len(rec['meta']['answer_quality']['citation_plan']['slots'])==2


def test_conflicting_or_distinct_excerpts_are_not_collapsed(tmp_path):
    from api.claim_passage_plan import _deduplicate_verified_slots

    raw='Assume the noise components are independent and of zero mean. The network predicts the masked pixels.'
    slot=fixture(tmp_path,raw)
    for duplicate in ({**slot,'anchor_id':'wrong'},{**slot,'evidence_quote':raw.split(' The')[0]}):
        assert len(_deduplicate_verified_slots([slot,duplicate]))==2
