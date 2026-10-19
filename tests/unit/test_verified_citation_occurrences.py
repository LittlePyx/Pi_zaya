from __future__ import annotations

from api.citation_display_registry import remap_system_a_citations_for_display
from kb.source_blocks import load_source_blocks_cached
from api.citation_display_registry import _citation_claim_context
from kb.source_passage_evidence import claim_passage_concepts_covered


def _cards(tmp_path):
    path = tmp_path / 'paper.md'
    path.write_text(
        '# Paper\n\n<!-- kb_page: 1 -->\n\n'
        'Our method is applicable to both pixel-wise i.i.d. white noise and spatially correlated noise.\n\n'
        'Given the sum of two i.i.d. noise samples, the best estimate of one of them is half of their sum.\n',
        encoding='utf8',
    )
    blocks = [b for b in load_source_blocks_cached(path) if b.get('kind') == 'paragraph']
    assert len(blocks) == 2
    return [
        {**b, 'source_path': str(path), 'source_name': 'paper.md', 'num': 1,
         'anchor': f'passage-{i}', 'citation_route': 'system_a',
         'evidence_source': 'verified_source_excerpt', 'evidence_quote': b['raw_text']}
        for i, b in enumerate(blocks)
    ]


def test_later_occurrence_uses_its_own_supported_passage(tmp_path):
    cards = _cards(tmp_path)
    answer = (
        f'{cards[0]["evidence_quote"]} [1](#passage-0).\n\n'
        'Given the sum of two i.i.d. noise samples, the best estimate of one of them is half of their sum [1](#passage-0).'
    )
    rendered, rows, _ = remap_system_a_citations_for_display(answer, cards)
    assert '#passage-0' in rendered.split('\n\n')[0]
    assert '#passage-1' in rendered.split('\n\n')[1]
    assert 'two i.i.d. noise samples' in rows[1]['answer_claim']


def test_single_verified_card_does_not_authorize_an_unrelated_later_claim(tmp_path):
    cards = _cards(tmp_path)[:1]
    answer = (
        f'{cards[0]["evidence_quote"]} [1](#passage-0).\n\n'
        '额外合成噪声样本与原始噪声之间相互独立 [1](#passage-0)。'
    )
    rendered, rows, _ = remap_system_a_citations_for_display(answer, cards)
    assert '#passage-0' in rendered.split('\n\n')[0]
    assert '#passage-0' not in rendered.split('\n\n')[1]
    assert len(rows) == 1
    cards[0].update(citation_plan_slot=True, selection_reason='prompt_contract_block')
    rendered, rows, _ = remap_system_a_citations_for_display(answer, cards)
    assert '#passage-0' in rendered.split('\n\n')[0]
    assert '#passage-0' not in rendered.split('\n\n')[1]
    assert len(rows) == 1


def test_stale_physical_identity_cannot_keep_a_verified_occurrence(tmp_path):
    cards = _cards(tmp_path)[:1]
    cards[0]['anchor_id'] = 'wrong-anchor'
    rendered, rows, _ = remap_system_a_citations_for_display(
        f'{cards[0]["evidence_quote"]} [1](#passage-0).', cards,
    )
    assert '#passage-0' not in rendered
    assert rows == []


def test_trailing_variable_definition_has_its_own_claim_scope():
    text = r'The expectation equals Equation (8), where $\sigma(i)$ denotes the standard deviation of $n(i)$ [1].'
    start = text.index('[1]')
    claim = _citation_claim_context(text, start, start + 3)
    assert claim.startswith('where ')
    assert 'Equation (8)' not in claim
    assert not claim_passage_concepts_covered(claim, 'The expectation equals Equation (8).')
    assert claim_passage_concepts_covered(claim, r'where $\sigma(i)$ denotes the standard deviation of $n(i)$.')


def test_a_distribution_identity_does_not_establish_sampling_ability():
    claim = 'M 与 N 是 i.i.d.，并要求可采样。'
    assert not claim_passage_concepts_covered(claim, 'Recall that M and N are i.i.d.')


def test_parentheses_inside_citation_titles_do_not_expand_the_claim_scope():
    text = 'First fact [1](#first "source: Paper / J minimizing (1) satisfies").\n\nSecond fact [1](#second "source: Paper / Section (2)").'
    start = text.index('[1]')
    end = text.index(').') + 1
    claim = _citation_claim_context(text, start, end)
    assert claim == 'First fact .'
    assert 'Second' not in claim
