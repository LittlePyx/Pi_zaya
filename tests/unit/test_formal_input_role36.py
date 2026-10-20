from __future__ import annotations

import pytest

from kb.invariant_input_role import clarify_supported_input_role

ANSWER = ('Formally, $f(x)_J$ does not depend on the value of $x$ restricted to $J$, $x_J$ [1].')
SOURCE = 'The value f(x) restricted to J, f(x)J, does not depend on the value of x restricted to J, xJ.'


def test_formal_definition_can_name_the_existing_arguments_role():
    result, count = clarify_supported_input_role(ANSWER, [{'text': SOURCE}])
    assert count == 1
    assert result.replace('input value', 'value') == ANSWER
    assert clarify_supported_input_role(result, [{'text': SOURCE}]) == (result, 0)


@pytest.mark.parametrize('source', [
    SOURCE.replace('does not depend', 'depends'),
    SOURCE.replace('f(x)J', 'g(x)J'),
    SOURCE.replace('xJ.', 'xK.'),
    SOURCE.replace('x restricted to J', 'y restricted to J'),
    'The input is a noisy image with independent noise.',
])
def test_no_role_from_conflicting_or_merely_topical_source(source):
    assert clarify_supported_input_role(ANSWER, [{'text': source}]) == (ANSWER, 0)


@pytest.mark.parametrize('answer', [
    ANSWER.replace('[1]', ''), ANSWER.replace('x_J$', 'x_K$'),
    '"' + ANSWER + '"', '> ' + ANSWER, '`' + ANSWER + '`',
])
def test_uncited_literal_and_mismatched_answer_are_unchanged(answer):
    assert clarify_supported_input_role(answer, [{'text': SOURCE}]) == (answer, 0)
