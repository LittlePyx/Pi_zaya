from __future__ import annotations

import pytest

from api.claim_passage_plan import _supports
from kb.claim_evidence_runtime import _scope_absolute_negative_claims
from kb.claim_source_eligibility import retrieval_limit_claim


@pytest.mark.parametrize('claim', [
    'The selected excerpts contain no head-to-head comparison of the two methods.',
    'No retrieved snippet reports a direct comparison on any noise type or image.',
    'The cited passages include no randomized clinical trial.',
    'That difference alone means an always better claim is not something these snippets can demonstrate.',
    'Absence of such a comparison in these snippets does not prove the full papers contain none.',
])
def test_explicit_evidence_window_absence_cannot_receive_a_paper_fact_citation(claim):
    assert retrieval_limit_claim(claim)
    assert not _supports(claim, 'A head-to-head comparison of image denoising on several noise types is reported.', 'Experiments', 'paper.pdf')
    answer, _ = _scope_absolute_negative_claims(claim + ' [1]')
    assert claim in answer and '[1]' not in answer


def test_explicit_paper_limitation_is_still_a_paper_claim():
    assert not retrieval_limit_claim('The paper contains no clinical trial.')
    assert not retrieval_limit_claim('The authors report no improvement on correlated noise.')
