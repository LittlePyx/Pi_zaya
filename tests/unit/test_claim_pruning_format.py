from __future__ import annotations

from kb.claim_evidence_runtime import _drop_unsupported_uncited_claims
from kb.claim_pruning_format import remove_new_unpaired_bold, repair_pruned_line_emphasis


def test_pruning_a_claim_does_not_leave_half_of_the_conclusion_emphasis():
    original = '**结论：不矛盾。** 实验在500名患者中验证了疗效。'
    retained, dropped = _drop_unsupported_uncited_claims(original)
    assert dropped
    assert '500' not in retained
    assert '不矛盾' in retained
    assert retained.count('**') % 2 == 0


def test_unpruned_bold_and_literal_math_remain_unchanged():
    original = '**A supported claim.** Another supported claim.'
    assert remove_new_unpaired_bold(original, original) == original
    math = r'$x^{**}$ and $y^{**}$'
    assert remove_new_unpaired_bold(math, math) == math
    assert remove_new_unpaired_bold('**Keep.** $x^{**}$', '**Keep. $x^{**}$') == 'Keep. $x^{**}$'


def test_pruning_one_paragraph_cannot_consume_another_paragraphs_bold():
    original = '**结论：不矛盾。** Unsupported inference.\n\nThe relationship is **between two noise samples**, under the stated premise.'
    retained = '**结论：不矛盾。\n\nThe relationship is **between two noise samples**, under the stated premise.'
    expected = retained.replace('**结论', '结论')
    assert repair_pruned_line_emphasis(original, retained) == expected
