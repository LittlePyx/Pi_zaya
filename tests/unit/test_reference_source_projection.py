from __future__ import annotations

import pytest

from api.reference_source_projection import answer_citation_evidence_quote, verified_reference_detail
from api.routers.references import _overlay_refs_payload_with_answer_citations
from kb.source_blocks import load_source_blocks_cached

SELF = "Assume the noise components are independent and of zero mean. The expectation of the loss function (7) is equal to the supervised loss plus a noise variance term."
RANGE = "Our method is applicable to both pixel-wise i.i.d. white noise and spatially correlated noise."


def detail(tmp_path, text, name):
    path = tmp_path / (name + '.md')
    path.write_text('<!-- kb_page: 3 -->\n## Method\n\n' + text, encoding='utf8')
    block = next(b for b in load_source_blocks_cached(path) if b['kind'] == 'paragraph')
    return {**block, 'source_path': str(path), 'source_name': name + '.pdf', 'evidence_quote': text,
            'evidence_source': 'verified_source_excerpt', 'binding_status': 'grounded',
            'answer_claim': '噪声分量独立且均值为零。', 'answer_claims': ['临床试验招募了300名患者。']}


def test_verified_proof_is_not_shortened_to_its_conclusion(tmp_path):
    row = detail(tmp_path, SELF, 'self')
    row['reader_evidence_quote'] = SELF.split('. ', 1)[1]
    assert answer_citation_evidence_quote(row, prompt='loss function expectation') == SELF
    projected = verified_reference_detail(row)
    assert projected['evidence_quote'] == projected['reader_evidence_quote'] == SELF
    assert '300' not in str(projected['answer_claims'])
    assert all(projected[k] == row[k] for k in ('source_path', 'block_id', 'anchor_id', 'heading_path', 'page_start', 'page_end'))


@pytest.mark.parametrize('key', ['block_id', 'anchor_id', 'heading_path', 'page_start', 'page_end', 'source_path'])
def test_conflicting_physical_identity_cannot_be_projected(tmp_path, key):
    row = detail(tmp_path, SELF, 'self')
    row[key] = 99 if key.startswith('page_') else 'wrong-identity'
    assert verified_reference_detail(row) is None


def test_overlay_copy_and_all_reader_projections_share_the_same_source(tmp_path, monkeypatch):
    monkeypatch.setattr('api.routers.references._ref_card_user_locale', lambda _: 'zh')
    first = detail(tmp_path, SELF, 'self')
    second = detail(tmp_path, RANGE, 'range')
    second['answer_claim'] = '允许空间相关噪声。'
    second['answer_claims'] = ['两个随机量独立同分布。']
    out = _overlay_refs_payload_with_answer_citations(store=None, conv_id='test', payload={'1': {'prompt': '比较噪声前提', 'hits': []}}, answer_citation_state=({1: [first, second]}, set()))
    for hit, row in zip(out['1']['hits'], (first, second)):
        ui = hit['ui_meta']; primary = ui['primary_evidence']; reader = ui['reader_open']
        assert hit['text'] == primary['snippet'] == reader['snippet'] == row['evidence_quote']
        assert all(primary[k] == row[k] for k in ('source_path', 'block_id', 'anchor_id', 'heading_path', 'page_start', 'page_end'))
        assert reader['blockId'] == primary['block_id'] and reader['anchorId'] == primary['anchor_id']
        assert len(ui['summary_line']) >= 12 and len(ui['why_line']) >= 12
        assert '300' not in ui['summary_line'] + ui['why_line']
    self_ui, range_ui = [h['ui_meta'] for h in out['1']['hits']]
    assert '零均值' in self_ui['why_line']
    assert '适用范围' in range_ui['summary_line']
    assert '没有给出' in range_ui['why_line'] and '独立性条件' in range_ui['why_line']


def test_same_paper_occurrences_do_not_mix_the_first_quote_with_another_locator(tmp_path, monkeypatch):
    monkeypatch.setattr('api.routers.references._ref_card_user_locale', lambda _: 'zh')
    path = tmp_path / 'same.md'
    path.write_text('<!-- kb_page: 2 -->\n## Assumptions\n\n' + SELF + '\n\n<!-- kb_page: 5 -->\n## Applicability\n\n' + RANGE, encoding='utf8')
    rows = [{**b, 'source_path': str(path), 'source_name': 'same.pdf', 'evidence_quote': b['raw_text'],
             'evidence_source': 'verified_source_excerpt', 'binding_status': 'grounded'}
            for b in load_source_blocks_cached(path) if b['kind'] == 'paragraph']
    assert len(rows) == 2
    out = _overlay_refs_payload_with_answer_citations(store=None, conv_id='test', payload={'1': {'prompt': 'spatially correlated noise applicability', 'hits': []}}, answer_citation_state=({1: rows}, set()))
    hit = out['1']['hits'][0]; ui = hit['ui_meta']; primary = ui['primary_evidence']
    assert hit['text'] == ui['reader_open']['snippet'] == primary['snippet'] == RANGE
    assert all(primary[k] == rows[1][k] for k in ('block_id', 'anchor_id', 'page_start', 'page_end', 'heading_path'))
    assert ui['heading_path'] == ui['reader_open']['headingPath'] == rows[1]['heading_path']
    assert '适用范围' in ui['summary_line'] and '零均值' not in ui['summary_line']


def test_source_verified_label_cannot_override_a_nonliteral_quote_or_wrong_kind(tmp_path, monkeypatch):
    monkeypatch.setattr('api.routers.references._ref_card_user_locale', lambda _: 'zh')
    row = detail(tmp_path, SELF, 'self')
    row['anchor_kind'] = 'equation'
    assert verified_reference_detail(row)['anchor_kind'] == 'paragraph'
    row['evidence_quote'] = SELF.replace('zero', 'biased')
    assert verified_reference_detail(row) is None
    out = _overlay_refs_payload_with_answer_citations(store=None, conv_id='test', payload={'1': {'prompt': '条件', 'hits': []}}, answer_citation_state=({1: [row]}, set()))
    assert not out['1']['hits']
