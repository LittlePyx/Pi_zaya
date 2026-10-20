from __future__ import annotations

from kb.claim_evidence_runtime import _enforce_named_source_section_citations


HITS = [
    {"text": "Self2Self reports improvements over BM3D, TWSC and CDnCNN in its experiments.", "meta": {"source_name": "self2self.pdf"}},
    {"text": "Noisier2Noise requires the ability to sample from a known noise distribution.", "meta": {"source_name": "noisier2noise.pdf"}},
]
PROMPT = "Do these papers establish a universal ranking?"


def test_ranking_question_keeps_both_named_sources_for_mixed_assumptions():
    answer = "Noisier2Noise requires a known noise distribution to sample from, while Self2Self reports improvements over BM3D and TWSC in its experiments [1]."
    repaired, _, _ = _enforce_named_source_section_citations(answer, HITS, prompt=PROMPT, min_support_score=5)
    assert "[2]" in repaired and "[1]" in repaired


def test_excluded_comparator_does_not_receive_a_factual_source_citation():
    answer = "Self2Self reports improvements over BM3D, TWSC and CDnCNN in its experiments, not Noisier2Noise [2][1]."
    repaired, _, _ = _enforce_named_source_section_citations(answer, HITS, prompt=PROMPT, min_support_score=5)
    assert "reports improvements" in repaired and "[1]" in repaired
    assert "[2]" not in repaired


def test_excluded_ranking_subject_and_anaphoric_continuation_keep_own_source():
    answer = "Noisier2Noise requires sampling a known noise distribution, not a ranking against Self2Self [1][2]. It also requires the ability to sample from that distribution [1][2]."
    repaired, _, _ = _enforce_named_source_section_citations(answer, HITS, prompt=PROMPT, min_support_score=5)
    assert "[1]" not in repaired
    assert repaired.count('[2]') == 2


def test_late_source_check_respects_named_section_and_literal_source_detail():
    import copy
    hits = copy.deepcopy(HITS)
    for hit in hits:
        hit['meta']['citation_plan_full_evidence_quote'] = hit['text']
    answer = "**Self2Self**\nSelf2Self reports improvements over BM3D, TWSC and CDnCNN in its experiments [1][2].\n\nSource-grounded detail:Self2Self reports improvements over BM3D, TWSC and CDnCNN in its experiments [1][2]."
    repaired, _, _ = _enforce_named_source_section_citations(answer, hits, prompt=PROMPT, min_support_score=5, inline_only=True)
    assert repaired.count('[1]') == 2
    assert '[2]' not in repaired


def test_selected_pair_evidence_question_binds_each_named_paper():
    answer = 'Self2Self reports improvements over BM3D, TWSC and CDnCNN in its experiments [2].'
    repaired, _, _ = _enforce_named_source_section_citations(answer, HITS, prompt='仅依据这两篇论文，给出所问临床证据。', min_support_score=5)
    assert '[1]' in repaired and '[2]' not in repaired


def test_unresolved_comparison_list_owner_is_not_the_excluded_method():
    answer = 'Noisier2Noise is not named in that contribution summary [2].'
    repaired, _, dropped = _enforce_named_source_section_citations(answer, HITS, prompt=PROMPT, min_support_score=5)
    assert not repaired
    assert dropped[0]['reason'] == 'unresolved_comparison_owner'


def test_scoped_evidence_limit_does_not_receive_unrequested_comparison_backfill():
    from kb.generation_answer_finalize_runtime import _complete_grounded_requested_source_facts, _complete_grounded_requested_source_facets
    boundary = 'The current retrieved evidence cannot establish a universal ranking.'
    quote = 'Also, our method is compared to the single-image extension of N2V and N2S, denoted by N2V(1) and N2S(1), using their codes.'
    hits = [{'text': quote, 'meta': {'source_path': 'self2self.md'}}]
    plan = {'slots': [{'preferred_system': 'system_a', 'source_path': 'self2self.md', 'candidate_hits': [1], 'evidence_quote': quote}]}
    for complete in (_complete_grounded_requested_source_facts, _complete_grounded_requested_source_facets):
        assert complete(boundary, prompt='Do these two papers establish a universal ranking?', citation_plan=plan, answer_hits=hits) == boundary


def test_possessive_and_consequence_continue_one_source_with_support():
    answer = 'Self2Self reports improvements over BM3D, TWSC and CDnCNN [1]. Its experiments report improvements over BM3D [1][2]. So its experiments report improvements over TWSC [1][2].'
    repaired, _, _ = _enforce_named_source_section_citations(answer, HITS, prompt=PROMPT, min_support_score=5)
    assert repaired.count('[1]') == 3
    assert '[2]' not in repaired


def test_paragraph_continuation_checks_support_and_explicit_source_switch():
    answer = 'Self2Self reports improvements over BM3D and TWSC [1].\n\nThis paper reports improvements over CDnCNN [1][2].\n\nIts experiments enrolled 900 patients [1][2].\n\nNoisier2Noise requires the ability to sample from a known noise distribution [1][2].'
    repaired, _, dropped = _enforce_named_source_section_citations(answer, HITS, prompt=PROMPT, min_support_score=5)
    assert '900' not in repaired and dropped
    assert repaired.count('[1]') == 2 and repaired.count('[2]') == 1


def test_mixed_paragraph_does_not_grant_anonymous_followup_a_single_owner():
    answer = 'Self2Self reports improvements over BM3D [1]; Noisier2Noise requires the ability to sample from a known noise distribution [2].\n\nIts paper reports improvements over BM3D [1][2].'
    repaired, _, dropped = _enforce_named_source_section_citations(answer, HITS, prompt=PROMPT, min_support_score=5)
    assert 'Its paper' not in repaired
    assert dropped[-1]['reason'] == 'unresolved_source_continuation'


def test_generic_words_do_not_add_a_second_source_to_anonymous_supported_fact():
    answer = 'This paper reports improvements over BM3D and TWSC [1].'
    repaired, _, _ = _enforce_named_source_section_citations(answer, HITS, prompt=PROMPT, min_support_score=5, inline_only=True)
    assert '[1]' in repaired and '[2]' not in repaired


def test_not_about_a_method_is_not_a_factual_source_mention():
    answer = 'Self2Self reports improvements over BM3D and TWSC [1]. This is a claim about its own benchmark comparisons, not about Noisier2Noise [2].'
    repaired, _, _ = _enforce_named_source_section_citations(answer, HITS, prompt=PROMPT, min_support_score=5)
    assert 'Self2Self reports improvements' in repaired and '[1]' in repaired
    assert '[2]' not in repaired
