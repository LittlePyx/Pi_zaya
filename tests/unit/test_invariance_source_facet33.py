from __future__ import annotations

from kb.source_question_facets import requests_source_facets, source_facet_score


def test_invariance_question_prioritizes_the_actual_definition_not_only_its_consequence():
    prompt = 'What does J-invariance require of the output on a coordinate subset J, and why does that rule prevent an identity predictor?'
    definition = 'It is J -invariant if f ( x ) J does not depend on the value of x J .'
    consequence = 'A J-invariant f has to use information from outside each subset J to predict inside J, so it cannot merely be the identity.'
    assert requests_source_facets(prompt)
    assert source_facet_score(prompt, definition) > source_facet_score(prompt, consequence) > 0
    assert source_facet_score('What is the experimental noise level?', definition) == 0
    assert source_facet_score(prompt, definition.replace('does not depend', 'depends')) == 0


def test_loss_statistics_use_theoretical_statements_and_training_speed_uses_the_combination():
    prompt = 'Which statistics minimize the L2 and L1 losses in the scalar estimation example?'
    caption = 'Figure 3. In case of text removal the mean (L2 loss) is not the correct answer but median (L1) is.'
    for quote in ('The L 1 loss recovers the median of the targets.', 'Using the L 2 loss, the network learns to output the average of all plausible explanations.'):
        assert source_facet_score(prompt, quote) > source_facet_score(prompt, caption)
    speed = 'Which techniques stabilize and accelerate training?'
    assert source_facet_score(speed, 'Residual learning and batch normalization speed up the training and improve denoising performance.') > 0
    assert source_facet_score(speed, 'The architecture incorporates batch normalization to improve performance.') == 0
