from __future__ import annotations

import copy

import pytest

from api.chat_render import _refine_system_a_cite_evidence_from_citation_plan
from api.source_card_contract import refresh_source_citation_cards
from kb.source_blocks import load_source_blocks


SHORT = 'We can therefore recover an estimate of the clean image by doubling our network’s output and subtracting its input. Intuitively, our network learns to output an image halfway between its doubly-noisy input and its best estimate of the clean image. By taking a second step in the direction from its input to this output, we can generate an estimate of the clean image.'
FULL = SHORT + ' We note that, in practice, there are many plausible clean images which could generate a particular noisy realization. Thus, E [ X | Z ] will not be an exact reconstruction of the true clean image, but will instead be the mean of this set of plausible clean images. This is the same uncertainty faced in a setting with clean training targets.'
CLAIM = 'So the colleague\'s premise ("the network directly outputs the clean image") is false for this method, and the correction cannot be removed "without changing the derivation."'


def fixture(tmp_path):
    path=tmp_path/'noisier2noise.en.md'
    path.write_text('<!-- kb_page: 3 -->\n# Noisier2Noise\n## Method\n\n'+FULL,encoding='utf8')
    block=next(b for b in load_source_blocks(path) if b['kind']=='paragraph')
    row={**{k:block[k] for k in ('block_id','anchor_id','heading_path','page_start','page_end')},
         'source_path':str(path),'source_name':'noisier2noise.pdf','evidence_quote':SHORT,'card_evidence':SHORT,
         'summary_line':SHORT,'raw':SHORT,'reader_evidence_quote':FULL,'reader_evidence_source':'verified_source_block',
         'evidence_source':'verified_source_excerpt','answer_claim':CLAIM,'citation_route':'system_a',
         'anchor':'third-occurrence','num':1,'linked_nums':[1],'answer_hit_num':1,'original_num':1,
         'citation_budget_key':'independent-excerpt','selection_reason':'citation_plan_evidence_overlay',
         'citation_plan_slot':False,'citation_plan_evidence_selection_reason':'prompt_contract_block',
         'answer_claims':[
             r'The correction is not cosmetic — it follows from the algebra relating $E[Y|Z]$ to $E[X|Z]$. Dropping it would mean treating $f(Z;\\theta)$ as an estimate of $X$, which contradicts the target the network was trained against ($Y$). '+CLAIM,
             CLAIM+' [1]',
         ]}
    slot={**row,'evidence_quote':FULL,'preferred_system':'system_a','candidate_hits':[1],
          'evidence_selection_reason':'prompt_contract_block','source_passage_bundle':True,'strict_locate':True,'retrieval_score':1000.0}
    return row,{'slots':[slot]}


def test_a_verified_partial_quote_survives_repeated_plan_and_cache_refreshes(tmp_path):
    row,plan=fixture(tmp_path);before=copy.deepcopy((row,plan));current=row
    for _ in range(4):
        current=_refine_system_a_cite_evidence_from_citation_plan([current],plan,render_locale='en')[0]
        current=refresh_source_citation_cards([current],render_locale='en')[0]
        assert current['evidence_quote']==SHORT
        assert current['card_evidence']==SHORT
        assert next(s['text'] for s in current['card_view']['sections'] if s['id']=='evidence')==SHORT
        assert current['block_id']==row['block_id'] and current['anchor']==row['anchor']
    assert (row,plan)==before


@pytest.mark.parametrize('field,value',[
    ('anchor_id','wrong'),('page_start',4),('heading_path','Other'),
    ('evidence_quote',SHORT.replace('doubling','DOUBLING')),
    ('answer_claim','The method is a randomized controlled clinical trial.'),
])
def test_physical_or_semantic_conflicts_do_not_freeze_a_stale_quote(tmp_path,field,value):
    from api.claim_passage_plan import keep_verified_occurrence_excerpt
    row,plan=fixture(tmp_path)
    assert not keep_verified_occurrence_excerpt({**row,field:value},plan['slots'])
    assert not keep_verified_occurrence_excerpt(row,[])
    assert not keep_verified_occurrence_excerpt(row,[{**plan['slots'][0],'page_start':4}])
