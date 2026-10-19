from __future__ import annotations

import copy

import pytest

from kb.refusal_context_excerpt import append_refusal_context_excerpt, bind_refusal_context_excerpt
from kb.source_blocks import load_source_blocks_cached

BOUNDARY = '当前检索到的证据无法确定患者人数，不能据此断言论文全文没有该数据。'
QUOTE = 'It is important to note that these methods make different assumptions about the task, and thus direct comparison may be misleading.'


def _fixture(tmp_path):
    path = tmp_path / 'paper.md'
    raw = 'We report PSNR 32.5 on the image dataset. ' + QUOTE
    path.write_text('<!-- kb_page: 4 -->\n## Experiments\n\n' + raw, encoding='utf8')
    plan = {'budget': {'system_a': 1}, 'slots': [{'preferred_system': 'system_a', 'source_path': str(path), 'heading_path': 'Experiments', 'page_start': 4, 'page_end': 4, 'candidate_hits': [1], 'evidence_quote': raw}]}
    return path, plan


def test_bounded_answer_has_a_literal_context_without_citing_the_limit_or_results(tmp_path):
    path, plan = _fixture(tmp_path); before = copy.deepcopy(plan)
    answer = append_refusal_context_excerpt(BOUNDARY, plan, [str(path)])
    assert answer.startswith(BOUNDARY + '\n\n') and BOUNDARY + ' [1]' not in answer
    assert '> ' + QUOTE + ' [1]' in answer
    assert 'PSNR' not in answer and '32.5' not in answer
    assert append_refusal_context_excerpt(answer, plan, [str(path)]) == answer
    bound = bind_refusal_context_excerpt(answer, plan, [str(path)])
    assert plan == before and bound['budget'] == before['budget']
    slot = bound['slots'][0]
    block = next(b for b in load_source_blocks_cached(path) if b['block_id'] == slot['block_id'])
    assert slot['evidence_quote'] == QUOTE and QUOTE in block['raw_text']
    assert slot['anchor_id'] == block['anchor_id'] and slot['page_start'] == 4
    assert bind_refusal_context_excerpt(answer, bound, [str(path)]) == bound


@pytest.mark.parametrize('failure', ['wrong_source', 'wrong_page', 'wrong_heading', 'stale_block', 'ambiguous', 'numeric_only', 'not_a_limit', 'already_cited'])
def test_context_never_repairs_conflicting_locators_or_supplies_numeric_substitutes(tmp_path, failure):
    path, plan = _fixture(tmp_path); answer = BOUNDARY; paths = [str(path)]
    if failure == 'wrong_source': paths = [str(tmp_path / 'other.md')]
    elif failure == 'wrong_page': plan['slots'][0].update(page_start=3, page_end=3)
    elif failure == 'wrong_heading': plan['slots'][0]['heading_path'] = 'Other'
    elif failure == 'stale_block': plan['slots'][0]['block_id'] = 'stale'
    elif failure == 'ambiguous': path.write_text(path.read_text(encoding='utf8') + '\n\n' + QUOTE, encoding='utf8')
    elif failure == 'numeric_only': plan['slots'][0]['evidence_quote'] = 'We report PSNR 32.5 on the image dataset.'
    elif failure == 'not_a_limit': answer = 'This paper defines a denoising method.'
    elif failure == 'already_cited': answer += '\n\n> ' + QUOTE + ' [1]'
    assert append_refusal_context_excerpt(answer, plan, paths) == answer


def test_unreliable_pages_and_fabricated_reading_quotes_stay_rejected(tmp_path, monkeypatch):
    path, plan = _fixture(tmp_path)
    answer = append_refusal_context_excerpt(BOUNDARY, plan, [str(path)])
    assert bind_refusal_context_excerpt(answer.replace('different assumptions', 'identical assumptions'), plan, [str(path)]) == plan
    monkeypatch.setattr('kb.converter.quality_repair.load_conversion_quality_result', lambda _p: {'source_quality': {'evidence_unreliable_pages': [4]}})
    assert append_refusal_context_excerpt(BOUNDARY, plan, [str(path)]) == BOUNDARY


def test_unrelated_readable_prose_is_not_added_to_fill_a_missing_card(tmp_path):
    path, plan = _fixture(tmp_path)
    unrelated = 'We present a practical algorithm for denoising images using a neural network and stochastic training on the available images.'
    path.write_text('<!-- kb_page: 4 -->\n## Experiments\n\n' + unrelated, encoding='utf8')
    plan['slots'][0]['evidence_quote'] = unrelated
    assert append_refusal_context_excerpt(BOUNDARY, plan, [str(path)]) == BOUNDARY


def test_bounded_refusal_drops_a_cited_fact_without_a_literal_source_witness(tmp_path):
    path, plan = _fixture(tmp_path)
    unsupported = '\n\nThe papers report a randomized clinical trial with 900 patients [1].'
    result = append_refusal_context_excerpt(BOUNDARY + unsupported, plan, [str(path)])
    assert result.startswith(BOUNDARY) and '900' not in result
    assert '> ' + QUOTE + ' [1]' in result


def test_window_limit_does_not_preserve_an_uncited_absolute_paper_classification(tmp_path):
    path, plan = _fixture(tmp_path)
    answer = BOUNDARY + '\n\n这两篇论文不是临床试验研究。\n\n这些都不是临床试验数据。\n\n但这不能证明全文一定没有相关内容。'
    result = append_refusal_context_excerpt(answer, plan, [str(path)])
    assert '不是临床试验' not in result and '不能证明全文' in result
