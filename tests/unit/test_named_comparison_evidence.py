from __future__ import annotations

from kb.claim_evidence_runtime import _enforce_named_source_section_citations


def _hits():
    return [
        {'text':'Noise components are independent conditional on the true signal.',
         'meta':{'source_name':'AlphaDenoise','title':'AlphaDenoise','source_path':'alpha.md'}},
        {'text':'The network predicts the residual image rather than a clean image.',
         'meta':{'source_name':'BetaDenoise','title':'BetaDenoise','source_path':'beta.md'}},
    ]


def test_each_paper_request_binds_bold_named_sections_to_their_own_evidence():
    answer='**AlphaDenoise 的假设**\nNoise components are independent conditional on the true signal.\n\n**BetaDenoise 的输出**\nThe network predicts the residual image rather than a clean image.'
    repaired,repairs,dropped=_enforce_named_source_section_citations(
        answer,_hits(),prompt='这两篇分别指出论文中的条件和输出依据。',min_support_score=5)
    assert 'true signal [1]' in repaired
    assert 'clean image [2]' in repaired
    assert len(repairs)==2 and not dropped


def test_named_section_cannot_borrow_the_other_sources_fact():
    answer='**AlphaDenoise 的输出**\nThe network predicts the residual image rather than a clean image.'
    repaired,repairs,dropped=_enforce_named_source_section_citations(
        answer,_hits(),prompt='两篇分别给出输出的原文。',min_support_score=5)
    assert '[1]' not in repaired and '[2]' not in repaired
    assert not repairs and dropped


def test_code_and_ambiguous_bold_heading_cannot_authorize_a_source_citation():
    answer='```text\n**AlphaDenoise**\nNoise components are independent conditional on the true signal.\n```\n\n**AlphaDenoise / BetaDenoise**\nNoise components are independent conditional on the true signal.'
    repaired,repairs,dropped=_enforce_named_source_section_citations(
        answer,_hits(),prompt='两篇分别指出条件。',min_support_score=5)
    assert repaired==answer and not repairs and not dropped


def test_existing_heading_marker_cannot_leave_the_previous_papers_ownership_active():
    answer='**AlphaDenoise 的假设**\nNoise components are independent conditional on the true signal.\n\n**BetaDenoise 的输出** [1]\nThe network predicts the residual image rather than a clean image [1].'
    repaired,repairs,dropped=_enforce_named_source_section_citations(
        answer,_hits(),prompt='两篇分别给出输出和假设的原文。',min_support_score=5)
    assert 'clean image [2]' in repaired and 'clean image [1]' not in repaired
    assert any(r['citation']==2 for r in repairs) and not dropped
    ambiguous=answer.replace('**BetaDenoise 的输出** [1]','**AlphaDenoise / BetaDenoise** [2]')
    unchanged,repairs,dropped=_enforce_named_source_section_citations(
        ambiguous,_hits(),prompt='两篇分别给出输出和假设的原文。',min_support_score=5)
    assert not any(r['citation']==2 for r in repairs)
