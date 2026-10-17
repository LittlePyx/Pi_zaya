from __future__ import annotations

import pytest

from ui import refs_renderer


EVIDENCE = (
    r"In the training stage, we use the noisy images from a wide range of noise levels "
    r"(e.g., $\sigma \in [0,55]$) to train a single DnCNN model. Given a test image "
    "whose noise level belongs to the noise level range, the learned single DnCNN model "
    "can be utilized to denoise it without estimating its noise level."
)


@pytest.mark.parametrize('label', ['range', 'range is', 'interval'])
def test_repeated_interval_without_of_keeps_literal_values_and_existing_source(render, label):
    answer = (
        r'DnCNN uses the noise interval $\sigma\in[0,55]$. '
        f'The Gaussian denoising training {label} [0,55]. '
        'DnCNN can denoise a test image without estimating its noise level [1].'
    )
    body, cards = render(answer)
    assert body.count('[0,55]') == 2
    assert '[1](#kb-cite-' in body
    assert cards
    assert all(card['source_path'] == 'papers/dncnn.en.md' for card in cards)


@pytest.mark.parametrize('claim', [
    'DnCNN is trained with noise levels from the range [0,99] [1].',
    'Noise2Self is trained with noise levels from the range [0,55] [1].',
    'DnCNN requires estimating the test image noise level for the range [0,55] [1].',
])
def test_bare_interval_cannot_grant_support_to_wrong_facts(render, claim):
    test_interval_protection_does_not_grant_support_to_wrong_facts(render, claim)


def test_zero_start_data_pair_cannot_become_a_document_one_citation(render):
    body, cards = render('DnCNN achieves 99 dB accuracy outside [0,1].')
    assert not cards
    assert '[1](#' not in body
    assert '[0,1]' in body


@pytest.fixture
def render(monkeypatch):
    monkeypatch.setattr(refs_renderer, '_is_temp_source_path', lambda path: False)
    monkeypatch.setattr(refs_renderer, '_load_reference_index_cached', lambda: {})
    path = 'papers/dncnn.en.md'
    hits = [{'text': EVIDENCE, 'meta': {'source_path': path, 'source_name': 'DnCNN'}}]
    def annotate(answer):
        return refs_renderer._annotate_inpaper_citations_with_hover_meta(
            answer, hits, canonical_paths=[path], render_locale='en',
        )
    return annotate


@pytest.mark.parametrize('interval', ['[0,55]', '[7,47]'])
def test_plain_interval_preserves_existing_card_and_copy_of_range(render, interval):
    answer = (
        'DnCNN can denoise a test image without estimating its noise level [1]. '
        f'The Gaussian noise level is from the range of {interval}.'
    )
    body, cards = render(answer)
    assert interval in body
    assert '[1](#kb-cite-' in body
    assert len(cards) == 1
    assert 'without estimating' in cards[0]['answer_claim']
    assert cards[0]['source_path'] == 'papers/dncnn.en.md'


def test_real_citation_group_is_not_protected_as_an_interval(render, monkeypatch):
    parsed = []
    parse = refs_renderer._parse_int_set
    def track(spec):
        parsed.append(spec)
        return parse(spec)
    monkeypatch.setattr(refs_renderer, '_parse_int_set', track)
    body, cards = render('DnCNN can denoise a test image without estimating its noise level [1,99].')
    assert '1,99' in parsed
    assert '[1,99]' not in body
    assert '[99](#' not in body


def test_unsupported_claim_beside_interval_does_not_gain_a_card(render):
    body, cards = render('DnCNN achieves 99 dB accuracy [1]. The noise range is [0,55].')
    assert not cards
    assert '[1](#' not in body


@pytest.mark.parametrize('claim', [
    'DnCNN is trained with noise levels from the range of [0,99] [1].',
    'DnCNN requires estimating the test image noise level before denoising [1].',
    'Noise2Self is trained with noise levels from the range of [0,55] [1].',
])
def test_interval_protection_does_not_grant_support_to_wrong_facts(render, claim):
    body, cards = render(claim)
    assert not cards
    assert '[1](#' not in body


@pytest.mark.parametrize('prefix', ['', '`code[42]` ', '$x$ ', '[[CITE:unknown:42]] '])
def test_interval_after_protected_or_replaced_fragment_keeps_its_own_offsets(render, prefix):
    body, _ = render(prefix + 'The noise level is from the range of [7,47].')
    assert '[7,47]' in body
    assert '[47](#' not in body


def test_original_answer_keeps_verified_page_card_through_render_and_cache(tmp_path, monkeypatch):
    import copy
    from api import chat_render
    from kb.source_blocks import load_source_blocks_cached

    source = tmp_path / 'dncnn.en.md'
    source.write_text('# DnCNN\n\n<!-- kb_page: 6 -->\n\n## Blind Gaussian denoising\n\n' + EVIDENCE, encoding='utf8')
    block = next(b for b in load_source_blocks_cached(source) if b['kind'] == 'paragraph')
    passage = {k: block[k] for k in ('text', 'block_id', 'anchor_id', 'heading_path', 'page_start', 'page_end')}
    plan = {'budget': {'system_a': 2, 'system_b': 0}, 'slots': [{
        'preferred_system': 'system_a', 'source_path': str(source), 'source_name': 'DnCNN',
        'candidate_hits': [1], 'evidence_quote': EVIDENCE, 'evidence_selection_reason': 'prompt_contract_block',
        **{k:v for k,v in passage.items() if k != 'text'},
    }]}
    answer = (
        r'Training noise-level interval: DnCNN is trained for blind Gaussian denoising using noisy images drawn '
        r'from a wide range of noise levels, e.g., $\sigma \in [0,55]$, to train a single model [1]. '
        'The same range is used in the general-denoising experimental setting, where Gaussian noise is added '
        'with a noise level from the range of [0,55].\n\n'
        'Does a test image in that range need an explicit noise-level estimate? No. Given a test image whose '
        'noise level belongs to that range, the learned single DnCNN model can denoise it without estimating '
        'its noise level [1].\n\nNot stated in the retrieved evidence: the exact performance numbers for blind '
        'Gaussian denoising, and any explicit noise-level estimation module inside DnCNN [1].'
    )
    rec = {'id': 2, 'role': 'assistant', 'content': answer, 'meta': {
        'canonical_hit_paths': [str(source)], 'answer_quality': {'citation_plan': plan},
        'paper_guide_contracts': {'citation_plan': plan},
    }}
    pack = {'render_locale': 'en', 'hits': [{'text': EVIDENCE, 'meta': {
        'source_path': str(source), 'source_name': 'DnCNN', 'source_passages': [passage],
        'ref_answer_citation_num': 1,
    }}]}
    monkeypatch.setattr(refs_renderer, '_is_temp_source_path', lambda path: False)
    monkeypatch.setattr(chat_render, '_effective_reference_render_pack', lambda raw: raw)
    monkeypatch.setattr(chat_render, '_answer_aligned_reference_render_pack', lambda raw, *args: raw)
    monkeypatch.setattr(chat_render, '_effective_citation_render_locale', lambda raw: 'en')
    user = {'id': 1, 'role': 'user', 'content': 'What training noise interval and test estimation are described?'}
    before = copy.deepcopy((rec, pack))
    result = chat_render.enrich_messages_with_reference_render([user, rec], {1: pack}, conv_id='noise-interval')[-1]
    assert (rec, pack) == before
    cached = chat_render.enrich_messages_with_reference_render([user, result], {1: pack}, conv_id='noise-interval')[-1]
    for message in (result, cached):
        assert message['rendered_body'].count('[0,55]') == 2
        assert '[1](#kb-cite-' in message['rendered_body']
        assert len(message['cite_details']) == 1
        card = message['cite_details'][0]
        assert card['page_start'] == 6 and card['block_id'] == block['block_id']
        assert card['source_path'] == str(source)
        assert 'without estimating' in card['evidence_quote']
        assert message['copy_markdown'].count('[0,55]') == 2
        assert message['copy_text'].count('[0,55]') == 2
