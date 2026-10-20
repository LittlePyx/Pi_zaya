from __future__ import annotations

from kb.direct_question_evidence import direct_question_evidence_hits
from kb.hybrid_retriever import _source_identity
from kb.source_blocks import load_source_blocks_cached
from kb.source_fact_scope import scope_explicit_source_facts


def test_physical_block_ids_keep_a_document_namespace(tmp_path):
    text = '<!-- kb_page: 1 -->\n# Method\n\nResidual learning and batch normalization speed up training.\n\n<!-- kb_page: 2 -->\n\nResidual learning and batch normalization speed up training.'
    papers = [tmp_path / name for name in ('a.md', 'b.md')]
    for p in papers:p.write_text(text, encoding='utf8')
    groups = [direct_question_evidence_hits(p, 'Which techniques accelerate training?') for p in papers]
    assert all(len(hits) == 2 for hits in groups)
    assert len({_source_identity(h) for h in groups[0]}) == 1
    assert {_source_identity(h) for h in groups[0]}.isdisjoint({_source_identity(h) for h in groups[1]})
    assert len({h['id'] for hits in groups for h in hits}) == 4


def test_omitting_a_citation_cannot_bypass_a_single_sources_explicit_fact_veto(tmp_path):
    p=tmp_path/'method.md'
    quote='We know the noise distribution, so we can draw an additional synthetic noise sample M.'
    p.write_text('<!-- kb_page: 2 -->\n# Method\n\n'+quote,encoding='utf8')
    block=next(b for b in load_source_blocks_cached(p) if b['kind']=='paragraph')
    slot={'preferred_system':'system_a','source_path':str(p),'source_name':'method.pdf','evidence_quote':quote,
        **{k:block[k] for k in ('block_id','anchor_id','heading_path','page_start','page_end')}}
    hit={'text':quote,'meta':{'source_path':str(p)}}
    wrong='The synthetic noise sample is used to construct the training target.'
    advice='Read the method section to check the training roles.'
    def scope(answer,hits):return scope_explicit_source_facts(answer,citation_plan={'slots':[slot]},answer_hits=hits,slot_hit_numbers=lambda *_:[1])
    assert wrong not in scope(wrong+'\n\n'+advice,[hit])
    assert scope(advice,[hit]) == advice
    assert scope(wrong,[hit,{'text':'another paper','meta':{'source_path':str(tmp_path/'other.md')}}]) == wrong
    assert scope('> '+wrong,[hit]) == '> '+wrong
    assert scope('"'+wrong+'"',[hit]) == '"'+wrong+'"'
    assert scope(quote,[hit]) == quote
