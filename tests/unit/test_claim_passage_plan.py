from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

from api import chat_render
from api.claim_passage_plan import message_citation_plan, prepare_claim_passage_plan
from kb.source_blocks import load_source_blocks_cached


def _declared_only_fixture(tmp_path):
    rec, pack = _fixture(tmp_path)
    meta = pack['hits'][0]['meta']
    passage = meta['source_passages'][0]
    slot = message_citation_plan(rec)['slots'][0]
    slot['source_passage_bundle'] = True
    slot['direct_source_evidence'] = [{
        **{k: passage[k] for k in ('text', 'block_id', 'anchor_id', 'page_start')},
        'source_path': slot['source_path'],
    }]
    meta.pop('source_passages')
    return rec, pack


def test_declared_retrieved_passage_restores_only_omitted_locator_fields(tmp_path):
    rec, pack = _declared_only_fixture(tmp_path)
    before = copy.deepcopy((rec, pack))
    result = prepare_claim_passage_plan(rec, pack, DIP_CLAIM)
    slots = message_citation_plan(result)['slots']
    assert len(slots) == 2
    assert slots[1]['page_start'] == slots[1]['page_end'] == 4
    assert slots[1]['heading_path'] == 'Noise2Noise / Experiments'
    assert slots[1]['passage_bound_claims'] == [DIP_CLAIM]
    assert slots[1]['candidate_hits'] == [1]
    assert message_citation_plan(result)['budget'] == message_citation_plan(rec)['budget']
    assert (rec, pack) == before
    assert prepare_claim_passage_plan(result, pack, DIP_CLAIM) == result


@pytest.mark.parametrize('failure', [
    'block', 'anchor', 'page', 'heading', 'end', 'source', 'text',
    'missing_anchor', 'missing_page', 'invalid_parent', 'not_bundle',
])
def test_declared_passage_cannot_repair_stale_or_conflicting_identity(tmp_path, failure):
    rec, pack = _declared_only_fixture(tmp_path)
    slot = message_citation_plan(rec)['slots'][0]
    raw = slot['direct_source_evidence'][0]
    if failure == 'block': raw['block_id'] = 'stale'
    if failure == 'anchor': raw['anchor_id'] = 'stale'
    if failure == 'page': raw['page_start'] = 3
    if failure == 'heading': raw['heading_path'] = 'Wrong section'
    if failure == 'end': raw['page_end'] = 99
    if failure == 'source': raw['source_path'] = str(tmp_path / 'namesake.en.md')
    if failure == 'text': raw['text'] = raw['text'].replace('neither clean examples', 'requires clean examples')
    if failure == 'missing_anchor': raw.pop('anchor_id')
    if failure == 'missing_page': raw.pop('page_start')
    if failure == 'invalid_parent': slot['block_id'] = 'stale-parent'
    if failure == 'not_bundle': slot['source_passage_bundle'] = False
    before = copy.deepcopy((rec, pack))
    result = prepare_claim_passage_plan(rec, pack, DIP_CLAIM)
    assert len(message_citation_plan(result)['slots']) == 1
    assert (rec, pack) == before


def test_declared_quote_with_wrong_fact_does_not_gain_a_passage(tmp_path):
    rec, pack = _declared_only_fixture(tmp_path)
    wrong = 'DIP requires clean examples and an explicit model of the corruption process [1].'
    assert len(message_citation_plan(prepare_claim_passage_plan(rec, pack, wrong))['slots']) == 1


def test_declared_passage_remains_bound_when_message_is_rendered_again(tmp_path, monkeypatch):
    rec, pack = _declared_only_fixture(tmp_path)
    rec['content'] = DIP_CLAIM
    monkeypatch.setattr('ui.refs_renderer._is_temp_source_path', lambda _path: False)
    monkeypatch.setattr('ui.refs_renderer._load_reference_index_cached', lambda: {})
    monkeypatch.setattr(chat_render, '_effective_reference_render_pack', lambda value: value)
    monkeypatch.setattr(chat_render, '_answer_aligned_reference_render_pack', lambda value, *args: value)
    monkeypatch.setattr(chat_render, '_effective_citation_render_locale', lambda _value: 'en')
    user = {'id': 1, 'role': 'user', 'content': 'What does Noise2Noise say about DIP?'}
    before = copy.deepcopy((rec, pack))
    first = chat_render.enrich_messages_with_reference_render([user, rec], {1: pack}, conv_id='declared-passages')[-1]
    second = chat_render.enrich_messages_with_reference_render([user, rec], {1: pack}, conv_id='declared-passages')[-1]
    for result in (first, second):
        assert result['content'] == DIP_CLAIM
        details = result['meta']['paper_guide_contracts']['render_packet']['cite_details']
        assert any(d['page_start'] == 4 and 'DIP' in d['evidence_quote'] for d in details)
    assert (rec, pack) == before


def test_declared_passages_do_not_choose_between_two_physical_matches(tmp_path):
    rec, pack = _declared_only_fixture(tmp_path)
    slot = message_citation_plan(rec)['slots'][0]
    path = Path(slot['source_path'])
    path.write_text(path.read_text(encoding='utf8') + '\n\n' + DIP, encoding='utf8')
    blocks = [b for b in load_source_blocks_cached(path) if b['kind'] == 'paragraph' and 'DIP' in b['text']]
    slot['direct_source_evidence'] = [{
        **{k:b[k] for k in ('text', 'block_id', 'anchor_id', 'page_start')},
        'source_path':str(path),
    } for b in blocks]
    assert len(blocks) == 2
    result = prepare_claim_passage_plan(rec, pack, DIP_CLAIM)
    assert len(message_citation_plan(result)['slots']) == 1


def test_declared_passage_is_rejected_after_its_text_changes(tmp_path):
    rec, pack = _declared_only_fixture(tmp_path)
    path = Path(message_citation_plan(rec)['slots'][0]['source_path'])
    path.write_text(path.read_text(encoding='utf8').replace('neither clean examples', 'requires clean examples'), encoding='utf8')
    result = prepare_claim_passage_plan(rec, pack, DIP_CLAIM)
    assert len(message_citation_plan(result)['slots']) == 1


DIP = (
    "DIP was almost 2 dB worse. DIP is not a learning-based solution, and as such very different from our approach, "
    "but it shares the property that neither clean examples nor an explicit model of the corruption is needed."
)
SUMMARY = (
    "In stark contrast to our approach, they need a perfect, explicit computational model of the corruption; "
    "our requirement of the knowledge of the appropriate statistical summary (mean, median) is significantly less limiting."
)
DIP_CLAIM = 'It also notes that DIP shares the property that "neither clean examples nor an explicit model of the corruption is needed" [1].'
ANSWER = (
    'No — the method does not require an explicit model of the corruption process [1]. '
    'Noise2Noise requires the appropriate statistical summary (mean or median), which is "significantly less limiting". '
    f'{DIP_CLAIM}\n\nThe appropriate statistical summary of the target distribution remains necessary [1].'
)


def _fixture(tmp_path, dip_text=DIP):
    path = tmp_path / "noise2noise.en.md"
    path.write_text(f"# Noise2Noise\n\n<!-- kb_page: 4 -->\n\n## Experiments\n\n{dip_text}\n\n"
                    f"<!-- kb_page: 8 -->\n\n## Discussion\n\n{SUMMARY}\n", encoding="utf8")
    blocks = [b for b in load_source_blocks_cached(path) if b["kind"] == "paragraph"]
    passages = [{k:b[k] for k in ("text", "block_id", "anchor_id", "heading_path", "page_start", "page_end")} for b in blocks]
    slot = {"preferred_system":"system_a", "source_path":str(path), "source_name":"Noise2Noise",
            "candidate_hits":[1], "evidence_quote":SUMMARY,
            **{k:passages[1][k] for k in ("block_id", "anchor_id", "heading_path", "page_start", "page_end")}}
    plan = {"intent":"method_explain", "budget":{"system_a":2,"system_b":0}, "slots":[slot]}
    rec = {"id":2, "role":"assistant", "content":ANSWER, "meta":{
        "canonical_hit_paths":[str(path)], "answer_quality":{"citation_plan":plan},
        "paper_guide_contracts":{"citation_plan":plan},
    }}
    pack = {"render_locale":"en", "hits":[{"text":dip_text, "meta":{
        "source_path":str(path), "source_name":"Noise2Noise", "ref_answer_citation_num":1,
        "source_passages":passages,
    }}]}
    return rec, pack


def test_claim_plan_adds_verified_page_and_keeps_number_and_input_immutable(tmp_path):
    rec, pack = _fixture(tmp_path)
    before = copy.deepcopy((rec,pack))
    result = prepare_claim_passage_plan(rec, pack, ANSWER)
    assert (rec,pack) == before
    slots = message_citation_plan(result)["slots"]
    assert len(slots) == 2
    assert [s["page_start"] for s in slots] == [8,4]
    assert slots[1]["candidate_hits"] == [1]
    assert "DIP" in slots[1]["evidence_quote"]
    assert prepare_claim_passage_plan(result,pack,ANSWER) == result


def test_two_claims_share_verified_passage_without_new_number_or_input_mutation(tmp_path):
    rec, pack = _fixture(tmp_path)
    second = "DIP is not a learning-based solution [1]."
    answer = f"{DIP_CLAIM}\n\n{second}"
    before = copy.deepcopy((rec, pack))
    result = prepare_claim_passage_plan(rec, pack, answer)
    slots = message_citation_plan(result)["slots"]
    assert len(slots) == 2
    assert slots[1]["candidate_hits"] == [1]
    assert len(slots[1]["passage_bound_claims"]) == 2
    assert second in slots[1]["passage_bound_claims"]
    assert (rec, pack) == before
    assert prepare_claim_passage_plan(result, pack, answer) == result


@pytest.mark.parametrize("second", [
    "DIP requires clean examples and an explicit model of the corruption process [1].",
    "DIP achieves 99 dB without clean examples or an explicit model [1].",
    "DIP 需要干净样本和显式污染模型 [1]。",
])
def test_second_claim_does_not_inherit_passage_binding_without_support(tmp_path, second):
    rec, pack = _fixture(tmp_path)
    result = prepare_claim_passage_plan(rec, pack, f"{DIP_CLAIM}\n\n{second}")
    slots = message_citation_plan(result)["slots"]
    assert len(slots) == 2
    assert len(slots[1]["passage_bound_claims"]) == 1
    assert second not in slots[1]["passage_bound_claims"]


@pytest.mark.parametrize("corruption", ["page", "anchor", "block", "heading", "text", "namesake", "passage_source", "negative", "number", "no_citation", "code", "math"])
def test_claim_plan_does_not_invent_or_cross_bind_evidence(tmp_path,corruption):
    rec,pack = _fixture(tmp_path)
    passage = pack["hits"][0]["meta"]["source_passages"][0]
    answer = ANSWER
    if corruption == "page":
        passage["page_start"] = passage["page_end"] = 8
    elif corruption == "anchor":
        passage["anchor_id"] = "p_99999"
    elif corruption == "block":
        passage["block_id"] = "blk_wrong"
    elif corruption == "heading":
        passage["heading_path"] = "Noise2Noise / Discussion"
    elif corruption == "text":
        passage["text"] = DIP + " It achieves 99 dB."
    elif corruption == "passage_source":
        passage["source_path"] = str(tmp_path / "other" / "noise2noise.en.md")
    elif corruption == "namesake":
        other = tmp_path / "other" / "noise2noise.en.md"
        other.parent.mkdir()
        other.write_text((tmp_path / "noise2noise.en.md").read_text(encoding="utf8"),encoding="utf8")
        pack["hits"][0]["meta"]["source_path"] = str(other)
    elif corruption == "negative":
        answer = "DIP requires clean examples and an explicit model of the corruption process [1]."
    elif corruption == "number":
        answer = "DIP achieves 99 dB without clean examples or an explicit model [1]."
    elif corruption == "no_citation":
        answer = DIP_CLAIM.replace("[1]", "")
    elif corruption == "code":
        answer = f"```text\n{DIP_CLAIM}\n```"
    elif corruption == "math":
        answer = "$x=[1]$"
    result = prepare_claim_passage_plan(rec,pack,answer)
    assert len(message_citation_plan(result)["slots"]) == 1


def test_negative_requirement_cannot_borrow_affirmative_source(tmp_path):
    rec, pack = _fixture(tmp_path, "DIP requires clean examples and an explicit model of the corruption process for reconstruction.")
    result = prepare_claim_passage_plan(rec,pack,DIP_CLAIM)
    assert len(message_citation_plan(result)["slots"]) == 1


def test_ambiguous_physical_matches_do_not_choose_by_retrieval_order(tmp_path):
    rec, pack = _fixture(tmp_path)
    path = tmp_path / "noise2noise.en.md"
    path.write_text(path.read_text(encoding="utf8")+f"\n<!-- kb_page: 6 -->\n\n## Duplicate\n\n{DIP}\n",encoding="utf8")
    duplicate = [b for b in load_source_blocks_cached(path) if b.get("page_start")==6 and b["kind"]=="paragraph"][0]
    pack["hits"][0]["meta"]["source_passages"].append(duplicate)
    result = prepare_claim_passage_plan(rec,pack,DIP_CLAIM)
    assert len(message_citation_plan(result)["slots"]) == 1


def test_stored_claim_slot_is_revalidated_after_source_change(tmp_path):
    rec,pack = _fixture(tmp_path)
    bound = prepare_claim_passage_plan(rec,pack,ANSWER)
    assert len(message_citation_plan(bound)["slots"]) == 2
    path = tmp_path / "noise2noise.en.md"
    path.write_text(path.read_text(encoding="utf8").replace(DIP,"This replacement passage contains no evidence about the requested method."),encoding="utf8")
    result = prepare_claim_passage_plan(bound,pack,ANSWER)
    assert len(message_citation_plan(result)["slots"]) == 1


@pytest.mark.parametrize("second_claim", [False, True])
def test_full_render_keeps_each_sentence_on_its_own_passage_and_cache(tmp_path, monkeypatch, second_claim):
    rec,pack = _fixture(tmp_path)
    pack["hits"][0]["meta"]["ref_rank"] = {"why":"STALE page 8 Discussion statistical summary explanation"}
    if second_claim:
        rec["content"] = ANSWER.replace("\n\n", "\n\nDIP is not a learning-based solution [1].\n\n")
    monkeypatch.setattr("ui.refs_renderer._is_temp_source_path",lambda path:False)
    monkeypatch.setattr(chat_render,"_effective_reference_render_pack",lambda raw:raw)
    monkeypatch.setattr(chat_render,"_answer_aligned_reference_render_pack",lambda raw,*args:raw)
    monkeypatch.setattr(chat_render,"_effective_citation_render_locale",lambda raw:"en")
    messages = [{"id":1,"role":"user","content":"What statistical condition remains necessary?"},rec]
    rendered = chat_render.enrich_messages_with_reference_render(messages,{1:pack},conv_id="passage-test")[-1]
    for result in (rendered,chat_render.enrich_messages_with_reference_render([messages[0],rendered],{1:pack},conv_id="passage-test")[-1]):
        cards = result["cite_details"]
        assert {c["page_start"] for c in cards} == {4,8}
        anchors = {c["anchor"]:c for c in cards}
        body = result["rendered_body"]
        markers = list(re.finditer(r'\[1\]\(#([^\s)]+)(?:\s+"[^"]*")?\)',body))
        assert len(markers) == (4 if second_claim else 3)
        assert [anchors[m[1]]["page_start"] for m in markers] == ([8,4,4,8] if second_claim else [8,4,8])
        for card in cards:
            if card["page_start"] == 4:
                assert "DIP" in card["evidence_quote"]
                assert card["passage_bound_claims"]
                assert card["block_id"] and card["anchor_id"]
                assert "STALE" not in card["card_support_explanation"]
                assert "statistical summary" not in card["card_support_explanation"]
                stale_primary = {**card,"page_start":8,"page_end":8,"strict_locate":True}
                assert chat_render._refine_system_a_cite_locators_from_final_primary([card],stale_primary)[0]["page_start"] == 4
        assert "DIP" in result["copy_markdown"] and "[1]" in result["copy_markdown"]


def _paragraph_end_fixture(tmp_path, channel):
    introduction = 'Proposition 1. Suppose x is an unbiased estimator of y , i.e.'
    independence = 'dent from the noise in its complement J c , conditional on y . Let f be J -invariant. Then'
    loss = ('That is, the self-supervised loss is the sum of the ordinary supervised loss and the variance of the noise. '
            'By minimizing the self-supervised loss over J-invariant functions, one recovers the optimal denoiser for a given dataset.')
    path = tmp_path / 'noise2self.en.md'
    path.write_text('# Noise2Self\n\n<!-- kb_page: 2 -->\n\n## J minimizing ( 1 ) satisfies\n\n'
                    + '\n\n'.join((introduction, independence, loss)) + '\n', encoding='utf8')
    passages = [{k:b[k] for k in ('text', 'block_id', 'anchor_id', 'heading_path', 'page_start', 'page_end')}
                for b in load_source_blocks_cached(path) if b['kind'] == 'paragraph']
    slot = {'preferred_system':'system_a', 'source_path':str(path), 'source_name':'Noise2Self',
            'candidate_hits':[1], 'evidence_quote':introduction, 'source_passage_bundle':True,
            **{k:passages[0][k] for k in ('block_id', 'anchor_id', 'heading_path', 'page_start', 'page_end')}}
    if channel in {'declared', 'both'}:
        slot['direct_source_evidence'] = [{k:p[k] for k in ('text', 'block_id', 'anchor_id', 'page_start')} for p in passages]
    plan = {'budget':{'system_a':1, 'system_b':0}, 'slots':[slot]}
    answer = ('Conditional independence: the retrieved text reads that $x$ is "…independent from the noise in its '
              'complement $J^c$, conditional on $y$." In other words, once $y$ is fixed, the value of $x$ on the '
              'index set $J$ carries no information about the noise affecting the remaining coordinates $J^c$ [1].\n\n'
              'That is, the self-supervised loss equals the ordinary supervised loss plus the (constant) noise variance, '
              'so minimizing the self-supervised loss over $J$-invariant functions recovers the optimal denoiser [1].')
    rec = {'id':2, 'role':'assistant', 'content':answer, 'meta':{
        'canonical_hit_paths':[str(path)], 'answer_quality':{'citation_plan':plan},
        'paper_guide_contracts':{'citation_plan':plan},
    }}
    meta = {'source_path':str(path), 'source_name':'Noise2Self', 'ref_answer_citation_num':1}
    if channel in {'indexed', 'both'}:
        meta['source_passages'] = passages
    return rec, {'render_locale':'en', 'hits':[{'text':introduction, 'meta':meta}]}, passages


@pytest.mark.parametrize('channel', ['declared', 'both'])
def test_paragraph_end_claims_bind_independent_bodies_on_first_render_and_reload(tmp_path, monkeypatch, channel):
    rec, pack, passages = _paragraph_end_fixture(tmp_path, channel)
    before = copy.deepcopy((rec, pack))
    monkeypatch.setattr('ui.refs_renderer._is_temp_source_path', lambda _path: False)
    monkeypatch.setattr(chat_render, '_effective_reference_render_pack', lambda value: value)
    monkeypatch.setattr(chat_render, '_answer_aligned_reference_render_pack', lambda value, *args: value)
    monkeypatch.setattr(chat_render, '_effective_citation_render_locale', lambda _value: 'en')
    user = {'id':1, 'role':'user', 'content':'What are the assumptions and the loss identity?'}
    first = chat_render.enrich_messages_with_reference_render([user, rec], {1:pack}, conv_id='paragraph-end')[-1]
    second = chat_render.enrich_messages_with_reference_render([user, first], {1:pack}, conv_id='paragraph-end')[-1]
    for result in (first, second):
        assert result['content'] == rec['content']
        cards = result['cite_details']
        assert {c['block_id'] for c in cards} == {p['block_id'] for p in passages[1:]}
        assert all(c['page_start'] == 2 and c['anchor_id'] for c in cards)
        conditional = next(c for c in cards if c['block_id'] == passages[1]['block_id'])
        assert 'Conditional independence' in conditional['answer_claim']
        assert 'In other words' not in conditional['answer_claim']
        assert 'self-supervised loss' not in conditional['answer_claim']
        assert message_citation_plan(result)['budget'] == message_citation_plan(rec)['budget']
    assert (rec, pack) == before


def test_raw_retrieval_alone_does_not_declare_a_new_paragraph_citation(tmp_path):
    from api.claim_passage_plan import prepare_claim_passage_render
    rec, pack, passages = _paragraph_end_fixture(tmp_path, 'indexed')
    result, rendered = prepare_claim_passage_render(rec, pack, rec['content'])
    assert rendered == rec['content']
    assert not any(s.get('block_id') == passages[1]['block_id'] for s in message_citation_plan(result)['slots'])


@pytest.mark.parametrize('failure', ['unverified_anchor', 'ambiguous', 'mixed_sources', 'no_marker', 'code'])
def test_paragraph_distribution_retains_physical_and_citation_boundaries(tmp_path, failure):
    from api.claim_passage_plan import prepare_claim_passage_render
    rec, pack, passages = _paragraph_end_fixture(tmp_path, 'declared')
    slot = message_citation_plan(rec)['slots'][0]
    answer = rec['content']
    if failure == 'unverified_anchor':
        slot['direct_source_evidence'][1]['anchor_id'] = 'stale'
    if failure == 'ambiguous':
        path = Path(slot['source_path'])
        path.write_text(path.read_text(encoding='utf8')+'\n'+passages[1]['text']+'\n', encoding='utf8')
        duplicate = [b for b in load_source_blocks_cached(path) if b['text'] == passages[1]['text']][-1]
        slot['direct_source_evidence'].append(duplicate)
    if failure == 'mixed_sources': answer = answer.replace('[1]', '[1] [2]')
    if failure == 'no_marker': answer = answer.replace('[1]', '')
    if failure == 'code': answer = '```text\n'+answer+'\n```'
    before = copy.deepcopy((rec, pack))
    result, rendered = prepare_claim_passage_render(rec, pack, answer)
    assert rendered == answer
    assert not any(s.get('block_id') == passages[1]['block_id'] for s in message_citation_plan(result)['slots'])
    assert (rec, pack) == before


@pytest.mark.parametrize('channel', ['declared', 'indexed'])
def test_source_block_context_cannot_substitute_for_the_displayed_passage(tmp_path, channel):
    rec, pack = _fixture(tmp_path, DIP+' '+SUMMARY)
    passage = pack['hits'][0]['meta']['source_passages'][0]
    passage['text'] = SUMMARY
    if channel == 'declared':
        slot = message_citation_plan(rec)['slots'][0]
        slot['source_passage_bundle'] = True
        slot['direct_source_evidence'] = [passage]
        pack['hits'][0]['meta'].pop('source_passages')
    result = prepare_claim_passage_plan(rec, pack, DIP_CLAIM)
    assert len(message_citation_plan(result)['slots']) == 1
