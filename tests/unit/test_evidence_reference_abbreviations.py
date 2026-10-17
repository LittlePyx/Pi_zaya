from __future__ import annotations

import pytest

from kb.evidence_text import pick_readable_evidence_text, split_evidence_sentences


@pytest.mark.parametrize('reference', ['Eq. 2', 'Eq. (2)', 'eq. 2', 'Fig. 2b', 'Figs. 2 and 3'])
def test_reference_abbreviation_keeps_its_label_in_the_source_sentence(reference):
    first = f'The signal is assumed to contain statistical dependencies (see {reference} ).'
    second = 'As a result, the network can estimate the signal of a pixel by looking at its surroundings.'
    source = first+' '+second
    assert split_evidence_sentences(source) == [first, second]
    excerpt = pick_readable_evidence_text(source, claim='The signal has statistical dependencies and is estimated from its surroundings.')
    assert excerpt == source


def test_actual_sentence_boundaries_are_not_joined_as_references():
    source = 'The signal is correlated. The noise is independent. This assumption is necessary.'
    assert split_evidence_sentences(source) == [
        'The signal is correlated.', 'The noise is independent.', 'This assumption is necessary.',
    ]
