from __future__ import annotations

from kb.explicit_source_attribution import source_attribution_covered
from kb.source_blocks import load_source_blocks_cached
from kb.source_fact_scope import scope_explicit_source_facts


def test_routing_markers_are_not_paper_method_names_in_a_correct_source_fact(tmp_path):
    p=tmp_path/'estimation.md'
    quote='With the L2 loss, the network learns to output the average of all plausible explanations (e.g., edges shifted by different amounts), which results in spatial blurriness.'
    p.write_text('<!-- kb_page: 2 -->\n# Estimation\n\n'+quote,encoding='utf8')
    block=next(b for b in load_source_blocks_cached(p) if b['kind']=='paragraph')
    slot={'preferred_system':'system_a','source_path':str(p),'source_name':'estimation.pdf','evidence_quote':quote,
        **{k:block[k] for k in ('block_id','anchor_id','heading_path','page_start','page_end')}}
    hit={'text':quote,'meta':{'source_path':str(p)}}
    answer=quote.rstrip('.')+' [[SUPPORT:DOC-1]].'
    def scope(text):return scope_explicit_source_facts(text,citation_plan={'slots':[slot]},answer_hits=[hit],slot_hit_numbers=lambda *_:[1])
    assert scope(answer) == answer
    assert 'BM3D' not in scope('The paper compares baselines, e.g. BM3D, using the L2 loss [[SUPPORT:DOC-1]].')


def test_clean_target_definitions_and_related_work_quotes_need_the_actual_evidence():
    claim="DnCNN's residual target is defined against clean ground truth."
    abstract='Residual learning and batch normalization speed up training and improve denoising.'
    recipe='For DnCNN we train a residual mapping. Here the samples are noisy-clean training image pairs.'
    assert not source_attribution_covered(claim, abstract, 'dncnn.pdf')
    assert source_attribution_covered(claim, recipe, 'dncnn.pdf')
    quoted='As described in the related-work section, the CNN "attempts to predict not the clean signal, but instead the noise at every pixel".'
    assert not source_attribution_covered(quoted, 'Traditionally, training networks requires pairs of noisy and clean images.', 'noise2void.pdf')
    assert source_attribution_covered(quoted, 'The CNN attempts to predict not the clean signal, but instead the noise at every pixel.', 'noise2void.pdf')
