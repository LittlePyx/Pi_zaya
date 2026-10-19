from __future__ import annotations

from kb.citation_sentences import citation_sentence_spans, split_definition_bridges
from kb.claim_evidence_runtime import _split_claim_segments
from kb.direct_question_evidence import direct_question_evidence_hits, requests_direct_question_evidence
from kb.source_question_facets import comparison_caution_quote, requests_comparison_boundary
from kb.source_blocks import load_source_blocks_cached
from kb.waived_data_requirements import waived_data_requirements_covered


def test_clean_selection_reference_needs_the_actual_loss_and_restricted_minimization_result():
    claim = '之所以能不用干净参照来选择去噪器，是因为这个方差项与 $f$ 无关，在 $J$-不变函数类上最小化自监督损失时，多出来的常数项不影响最优解的位置。'
    loss = 'That is, the self-supervised loss is the sum of the ordinary supervised loss and the variance of the noise.'
    choice = 'By minimizing the self-supervised loss over a class of J -invariant functions, one may find the optimal denoiser for a given dataset.'
    assert waived_data_requirements_covered(claim, loss + ' ' + choice)
    for wrong in (loss, choice, 'The noise distribution is known.', loss + ' ' + choice.replace('one may find', 'one may not find')):
        assert not waived_data_requirements_covered(claim, wrong)
    training = '训练不需要干净图像或配对噪声图像。'
    assert not waived_data_requirements_covered(training, loss + ' ' + choice)


def test_a_definition_bridge_keeps_the_sampling_premise_and_definition_separate():
    text = r'Because the noise distribution is known, an additional synthetic noise sample can be drawn, giving a noisier version $Z \triangleq Y+M=X+N+M$.'
    spans = [text[a:b].strip() for a, b in citation_sentence_spans(text)]
    assert len(spans) == 2
    assert 'synthetic noise sample' in spans[0] and 'giving' not in spans[0]
    assert spans[1].startswith('giving') and r'Z \triangleq Y+M=X+N+M' in spans[1]
    assert _split_claim_segments(text) == spans
    for atomic in (r'$A, giving Z \triangleq X+N$', r'"text, giving Z := X+N"', '[1](#link "title, giving Z := X+N")', 'sampling gives an improvement, giving good results.'):
        assert split_definition_bridges(atomic) == [atomic]


def test_universal_comparison_retrieves_a_real_warning_with_its_own_locator(tmp_path):
    caution = 'It is important to note that these methods make different assumptions about the task, and thus direct comparison may be misleading.'
    path = tmp_path / 'paper.md'
    path.write_text('<!-- kb_page: 4 -->\n## Comparisons\n\n' + caution, encoding='utf8')
    prompt = 'Do these papers establish a universal ranking for every image?'
    assert requests_comparison_boundary(prompt) and requests_direct_question_evidence(prompt)
    hits = direct_question_evidence_hits(path, prompt)
    assert len(hits) == 1 and hits[0]['text'] == caution
    block = next(b for b in load_source_blocks_cached(path) if b['kind'] == 'paragraph')
    assert all(hits[0]['meta'][k] == block[k] for k in ('block_id', 'anchor_id', 'page_start', 'page_end', 'heading_path'))
    assert not requests_comparison_boundary('Compare the implementation details.')
    assert not comparison_caution_quote('The comparison reports better performance on every image.')
    assert not comparison_caution_quote('It is not true that these methods make different assumptions about the task, and thus direct comparison may be misleading.')



def test_additive_mapping_retains_the_known_distribution_before_the_draw(tmp_path):
    from kb.source_question_facets import source_facet_quote
    from kb.direct_question_evidence import attach_direct_source_evidence
    prefix = r'N ∼$\mathcal{A}$ and $\mathcal{A}$ is a known noise distribution.'
    draw = r'Because $\mathcal{A}$ is known, we can draw an additional synthetic noise sample'
    raw = prefix + ' ' + draw
    prompt = 'For the standard additive procedure, distinguish the network input Z from its training target Y.'
    assert source_facet_quote(prompt, raw) == raw
    path = tmp_path / 'sampling.md'
    path.write_text('<!-- kb_page: 3 -->\n## Method\n\n' + raw, encoding='utf8')
    block = next(b for b in load_source_blocks_cached(path) if b['kind'] == 'paragraph')
    meta = {**block, 'source_path': str(path), 'direct_question_evidence': True}
    hit = {'text': draw, 'meta': meta}
    slot = {**meta, 'preferred_system': 'system_a', 'evidence_quote': draw, 'candidate_hits': [1]}
    plan = attach_direct_source_evidence({'slots': [slot]}, [hit])
    bound = plan['slots'][0]
    assert raw in bound['source_evidence_quotes']
    assert any(p['text'] == raw and p['block_id'] == block['block_id'] for p in bound['direct_source_evidence'])
    assert all(bound[k] == slot[k] for k in ('source_path', 'block_id', 'anchor_id', 'heading_path', 'page_start', 'page_end'))
    # Conflicting physical provenance must not gain a repaired passage.
    wrong = {**meta, 'page_start': 99}
    rejected = attach_direct_source_evidence({'slots': [{**slot, 'page_start': 99}]}, [{'text': draw, 'meta': wrong}])
    assert not rejected['slots'][0].get('direct_source_evidence')



def test_definition_bridge_stays_split_after_citation_attachment():
    from api.citation_display_registry import _citation_claim_context
    import re
    for marker in ('[1]', '[1](#source "title (with parentheses)")'):
        text = r'Because the noise distribution is known, a synthetic sample can be drawn, ' + marker + r' giving $Z \triangleq Y+M=X+N+M$ ' + marker + '.'
        claims = _split_claim_segments(text)
        assert len(claims) == 2 and claims[0].endswith(marker)
        assert claims[1].startswith('giving')
        matches = list(re.finditer(re.escape(marker), text))
        first = _citation_claim_context(text, matches[0].start(), matches[0].end())
        second = _citation_claim_context(text, matches[1].start(), matches[1].end())
        assert 'synthetic sample' in first and 'triangleq' not in first
        assert 'triangleq' in second and 'distribution is known' not in second


def test_sampling_context_recovery_does_not_reintroduce_a_lost_continuation(tmp_path):
    from kb.direct_question_evidence import attach_direct_source_evidence
    draw = 'Because the distribution is known, we can draw an additional synthetic noise sample'
    raw = 'The distribution is a known noise distribution. ' + draw + ' and therefore and thus'
    path = tmp_path / 'incomplete.md'
    path.write_text('<!-- kb_page: 3 -->\n## Method\n\n' + raw, encoding='utf8')
    block = next(b for b in load_source_blocks_cached(path) if b['kind'] == 'paragraph')
    meta = {**block, 'source_path': str(path), 'direct_question_evidence': True}
    slot = {**meta, 'preferred_system': 'system_a', 'evidence_quote': draw, 'candidate_hits': [1]}
    result = attach_direct_source_evidence({'slots': [slot]}, [{'text': draw, 'meta': meta}])
    assert raw not in result['slots'][0].get('source_evidence_quotes', [])
    assert all(p['text'] != raw for p in result['slots'][0].get('direct_source_evidence', []))
