from __future__ import annotations

from kb.direct_question_evidence import direct_question_evidence_hits, prioritize_direct_question_hits


def test_bold_statement_label_is_located_and_stops_at_next_bold_statement(tmp_path):
    source=tmp_path/'unseen.md'
    source.write_text('<!-- kb_page: 3 -->\n**Proposition 4.** Assume mutually independent measurements with zero mean noise.\n\nThe loss differs by a fixed variance term.\n\n**Proposition 5.** A different biased estimator.\n\nUnrelated later result.\n',encoding='utf8')
    hits=direct_question_evidence_hits(source,'命题4对噪声有什么要求？')
    assert hits and hits[0]['text'].startswith('**Proposition 4.**')
    assert all('biased' not in h['text'] and 'later result' not in h['text'] for h in hits)
    assert all(h['meta']['page_start']==3 and h['meta']['source_path']==str(source) for h in hits)
    assert not direct_question_evidence_hits(source,'Explain Proposition 7.')


def test_equation_query_carries_only_its_same_page_explanation(tmp_path):
    source=tmp_path/'unseen.md'
    source.write_text('<!-- kb_page: 3 -->\n## Method\n\nWe construct complementary sampled observations.\n\n$$a = m x \\tag{6}$$\n\n$$R = ||F(a)-b||^2 \\tag{7}$$\n\nThe loss is measured only on the masked pixels. This prevents copying the observed input.\n\n## Experiments\n\nUnrelated results.\n<!-- kb_page: 8 -->\nThe same words on a different page must not become the equation explanation.\n',encoding='utf8')
    hits=direct_question_evidence_hits(source,'In Equations (6) and (7), which pixels contribute to the loss?')
    assert any('tag{6}' in h['text'] for h in hits)
    assert any('tag{7}' in h['text'] for h in hits)
    assert any('masked pixels' in h['text'] for h in hits)
    assert all(h['meta']['page_start']==3 and 'Unrelated' not in h['text'] for h in hits)
    assert not direct_question_evidence_hits(source,'Explain Equation (9).')


def test_direct_enrichment_keeps_selected_papers_represented():
    direct=[{'text':f'direct passage {i}','meta':{'source_path':'a.md','block_id':f'a{i}','direct_question_evidence':True}} for i in range(6)]
    selected=[{'text':'the other paper evidence','meta':{'source_path':'b.md','block_id':'b1'}}]
    result=prioritize_direct_question_hits(selected,direct,limit=6,prompt='Compare these two selected papers.')
    assert {h['meta']['source_path'] for h in result}=={'a.md','b.md'}
    assert len(result)==6 and result[0]['text']=='direct passage 0'


def test_code_and_bibliography_cannot_supply_numbered_theorems(tmp_path):
    source=tmp_path/'unseen.md'
    source.write_text('<!-- kb_page: 1 -->\n```text\n**Proposition 4.** A code example, not a theorem.\n```\n\n## References\n\n**Proposition 4.** A bibliography note.\n',encoding='utf8')
    assert not direct_question_evidence_hits(source,'Explain Proposition 4.')


def test_method_premises_are_selected_from_own_source_not_previous_work(tmp_path):
    source=tmp_path/'unseen.md'
    source.write_text('<!-- kb_page: 2 -->\nPrevious work requires a known noise model and noise distribution.\n\nOur method requires a statistical model of the noise distribution and one observation for each training example.\n',encoding='utf8')
    hits=direct_question_evidence_hits(source,'Does this method need a noise model?')
    assert len(hits)==1 and hits[0]['text'].startswith('Our method')
    assert 'statistical model' in hits[0]['text']


def test_negative_noise_model_premise_is_not_rewritten_as_positive(tmp_path):
    source=tmp_path/'unseen.md'
    source.write_text('<!-- kb_page: 2 -->\nOur method does not require an explicit noise model.\n',encoding='utf8')
    hits=direct_question_evidence_hits(source,'Does this method require a noise model?')
    assert hits and hits[0]['text']=='Our method does not require an explicit noise model.'


def test_correction_operation_keeps_the_actual_input_and_output_direction(tmp_path):
    source=tmp_path/'unseen.md'
    source.write_text('<!-- kb_page: 3 -->\nWe recover the estimate by doubling our network output and subtracting its input.\n\n## Other\n\nPrevious work doubles its input instead.\n',encoding='utf8')
    hits=direct_question_evidence_hits(source,'Which output is doubled and which input is subtracted in the correction?')
    assert len(hits)==1 and 'doubling our network output' in hits[0]['text']


def test_dropout_test_phase_is_not_inferred_from_a_general_training_sentence(tmp_path):
    source=tmp_path/'unseen.md'
    source.write_text('<!-- kb_page: 2 -->\nDropout is useful during network training.\n\nThis paper shows that using dropout in both training and test is effective on a single noisy image.\n',encoding='utf8')
    hits=direct_question_evidence_hits(source,'测试阶段也使用 dropout 吗？')
    assert len(hits)==1 and 'both training and test' in hits[0]['text']


def test_dropout_test_question_keeps_prediction_aggregation_as_a_separate_facet(tmp_path):
    source=tmp_path/'ensemble.md'
    source.write_text('<!-- kb_page: 2 -->\nOur solution uses dropout models. The average of these predictions will reduce the variance of the result.\n\nThis paper shows that using dropout in both training and test is effective on a single noisy image.\n',encoding='utf8')
    hits=direct_question_evidence_hits(source,'测试时使用 dropout 吗？多次预测如何组合以降低方差？')
    assert len(hits)==2
    assert 'both training and test' in hits[0]['text']
    assert any('average of these predictions' in h['text'] for h in hits)


def test_comparison_can_locate_another_sources_facets_without_inventing_a_missing_statement(tmp_path):
    source=tmp_path/'other.md'
    source.write_text('<!-- kb_page: 2 -->\nWe also allow spatially correlated noise models, but require the ability to sample from the noise distribution.\n<!-- kb_page: 3 -->\nRecall that U and V are i.i.d. and therefore have exchangeable roles in our noise sampling construction.\n',encoding='utf8')
    prompt='Compare Proposition 4 of the first paper with the other method allowing spatially correlated noise. Distinguish additional noise samples from components within an image.'
    hits=direct_question_evidence_hits(source,prompt)
    assert any('spatially correlated' in h['text'] for h in hits)
    assert any('U and V are i.i.d.' in h['text'] for h in hits)
    assert not direct_question_evidence_hits(source,'Explain Proposition 4 and its noise model.')


def test_supplementary_method_passages_keep_their_own_verified_page(tmp_path):
    from kb.direct_question_evidence import attach_direct_source_evidence
    from kb.paper_guide_answer_selection import _merge_same_source_answer_hits
    source=tmp_path/'method.md'
    source.write_text('<!-- kb_page: 2 -->\nWe recover the estimate by doubling our network output and subtracting its input.\n<!-- kb_page: 4 -->\nOur correction step is necessary because the raw output of the network is still noisy.\n',encoding='utf8')
    hits=sorted(direct_question_evidence_hits(source,'Why is the raw output still noisy without the correction step?'),key=lambda h:h['meta']['page_start'])
    lead=hits[0]
    slot={**lead['meta'],'preferred_system':'system_a','evidence_quote':lead['text'],'candidate_hits':[1]}
    plan=attach_direct_source_evidence({'slots':[slot]},_merge_same_source_answer_hits(hits))
    passages=plan['slots'][0]['direct_source_evidence']
    assert {p['page_start'] for p in passages}=={2,4}
    assert plan['slots'][0]['page_start']==2
    assert plan['slots'][0]['evidence_quote']==lead['text']
    bad={**hits[1],'meta':{**hits[1]['meta'],'page_start':2}}
    wrong=attach_direct_source_evidence({'slots':[slot]},[lead,bad])
    assert all('still noisy' not in p['text'] for p in wrong['slots'][0]['direct_source_evidence'])
    unclassified=[{**h,'meta':{k:v for k,v in h['meta'].items() if k!='direct_question_method_facet_score'}} for h in hits]
    narrow=attach_direct_source_evidence({'slots':[slot]},unclassified)
    assert {p['page_start'] for p in narrow['slots'][0]['direct_source_evidence']}=={2}
    other={**hits[1],'meta':{**hits[1]['meta'],'source_path':str(tmp_path/'other.md')}}
    wrong=attach_direct_source_evidence({'slots':[slot]},[lead,other])
    assert all('still noisy' not in p['text'] for p in wrong['slots'][0]['direct_source_evidence'])
