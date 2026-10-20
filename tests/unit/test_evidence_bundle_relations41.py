from __future__ import annotations

from copy import deepcopy

import pytest

from kb.paper_guide_answer_selection import _merge_same_source_answer_hits
from kb.explicit_noise_relations import (
    known_noise_model_covered,
    signal_dependency_covered,
    white_noise_restriction_covered,
)
from kb.explicit_source_details import explicit_source_details_covered
from kb.explicit_training_assignments import (
    noise_augmentation_training_covered,
    explicit_training_assignments_covered,
)
from kb.theory_evidence import measurement_conditions_covered
from kb.citation_sentences import citation_sentence_spans, split_definition_bridges


DRAW = 'We train under a fixed, known noise model. Because the distribution is known, we can draw an additional synthetic noise sample M.'
RECIPE = 'Given Y, we create a noisier version, Z = Y + M = X + N + M. We then train the network to predict Y given Z as input, using an L2 loss.'
CORRECTION = 'We can recover an estimate of the clean image by doubling our network output and subtracting its input.'


def _hit(number: int, text: str = '') -> dict:
    return {'text': text or f'Literal source paragraph {number}.', 'score': float(10-number),
            'meta': {'source_path': 'paper.md', 'block_id': f'block{number}',
                     'anchor_id': f'p{number}', 'heading_path': 'Method',
                     'page_start': number, 'page_end': number,
                     'direct_question_evidence': True}}


def test_remerging_preserves_literal_passages_and_their_own_coordinates():
    original = [_hit(1, RECIPE), _hit(2, DRAW)]
    bundle = _merge_same_source_answer_hits(original)[0]
    inputs = [bundle, original[1], _hit(3)]
    snapshot = deepcopy(inputs)
    merged = _merge_same_source_answer_hits(inputs)[0]
    passages = merged['meta']['source_passages']
    assert [p['text'] for p in passages] == [RECIPE, DRAW, _hit(3)['text']]
    assert [(p['block_id'], p['anchor_id'], p['page_start']) for p in passages] == [
        ('block1', 'p1', 1), ('block2', 'p2', 2), ('block3', 'p3', 3)]
    assert merged['text'].count('[Source passage') == 3
    assert _merge_same_source_answer_hits([merged]) == [merged]
    assert inputs == snapshot


def test_remerging_keeps_five_passage_budget_and_never_inherits_missing_locators():
    bundle = _merge_same_source_answer_hits([_hit(1), _hit(2)])[0]
    bundle['meta']['source_passages'][1].pop('block_id')
    bundle['meta']['source_passages'][1].pop('anchor_id')
    merged = _merge_same_source_answer_hits([bundle, *[_hit(i) for i in range(3, 9)]])[0]
    assert len(merged['meta']['source_passages']) == 5
    second = merged['meta']['source_passages'][1]
    assert not second.get('block_id') and not second.get('anchor_id')
    assert second['page_start'] == 2


def test_remerging_does_not_assign_a_foreign_child_to_the_parent_paper():
    bundle = _merge_same_source_answer_hits([_hit(1), _hit(2)])[0]
    bundle['meta']['source_passages'][1]['source_path'] = 'other-paper.md'
    merged = _merge_same_source_answer_hits([bundle, _hit(3)])[0]
    assert _hit(2)['text'] not in merged['text']
    assert [p['block_id'] for p in merged['meta']['source_passages']] == ['block1', 'block3']


@pytest.mark.parametrize('claim', [
    'Noisier2Noise adds a freshly sampled noise realization to the observed image.',
    'It samples new noise from a model and adds it to the observed noisy image.',
])
def test_sampled_noise_addition_needs_the_operation_not_just_a_draw(claim):
    assert not noise_augmentation_training_covered(claim, DRAW)
    assert noise_augmentation_training_covered(claim, RECIPE)
    assert noise_augmentation_training_covered('The measurement is formed by adding noise to the clean signal.', DRAW)


def test_known_statistical_model_is_a_requirement_not_an_augmentation_recipe():
    claim = 'The method needs a known statistical noise model.'
    assert not known_noise_model_covered(claim, RECIPE)
    assert known_noise_model_covered(claim, DRAW)


@pytest.mark.parametrize('claim', ['Noisier2Noise is not restricted to white noise.', 'Noisier2Noise 并不被限制为像素间独立的白噪声。', 'Noisier2Noise 并未被限制为像素间独立的白噪声。'])
def test_denying_white_noise_restriction_needs_an_actual_applicability_statement(claim):
    assert not white_noise_restriction_covered(claim, DRAW)
    assert white_noise_restriction_covered(claim, 'Our method works with spatially correlated noise.')
    assert white_noise_restriction_covered(claim, claim)
    assert not white_noise_restriction_covered(claim, 'Our method cannot handle spatially correlated noise.')


def test_copular_adverb_does_not_hide_the_named_training_target():
    claim = 'The training target is therefore the noisy image Y, not the clean image X.'
    assert not explicit_training_assignments_covered(claim, CORRECTION)
    assert explicit_training_assignments_covered(claim, RECIPE)
    assert not explicit_training_assignments_covered(claim.replace('image Y', 'image Z'), RECIPE)


@pytest.mark.parametrize('expression', ['E[Y|Z]', '$E[Y|Z]$', r'$\mathbb{E}[Y\mid Z]$'])
def test_conditional_output_estimate_needs_its_target_and_conditioning_variable(expression):
    claim = f'We double the raw network output (the estimate of {expression}).'
    assert not explicit_training_assignments_covered(claim, CORRECTION)
    assert explicit_training_assignments_covered(claim, RECIPE)
    assert explicit_training_assignments_covered(claim, 'The network output is E[Y|Z].')
    assert not explicit_training_assignments_covered(claim.replace('Y', 'X'), RECIPE)
    assert not explicit_training_assignments_covered(claim.replace('Z', 'X'), RECIPE)


def test_reading_advice_cannot_presuppose_a_wrong_named_training_target():
    assert not explicit_training_assignments_covered('如何用 M 构造训练目标。', DRAW)
    assert explicit_training_assignments_covered('如何用 Y 构造训练目标。', RECIPE)


def test_statistically_dependent_signal_is_not_in_a_blind_spot_definition():
    claim = 'The signal is assumed to be statistically dependent across neighboring pixels.'
    assert not signal_dependency_covered(claim, 'The receptive field excludes the target pixel, preventing identity mapping.')
    assert signal_dependency_covered(claim, 'The signal is assumed to contain statistical dependencies between neighboring pixels.')


def test_noise_conditioning_object_cannot_be_borrowed_from_a_figure_caption():
    claim = 'The noise in J is independent of noise in its complement Jc conditional on y.'
    caption = 'Noise is conditionally independent between sets of dimensions.'
    premise = 'The noise in the complement Jc, conditional on y.'
    assert not measurement_conditions_covered(claim, caption)
    assert measurement_conditions_covered(claim, premise)
    assert not measurement_conditions_covered(claim.replace('on y', 'on x'), premise)


@pytest.mark.parametrize(('claim', 'proof', 'wrong'), [
    ('The paper mentions potential numerical issues.', 'Bar potential numerical issues, the solution remains the same.', 'Monte Carlo convergence reduces the quality gap.'),
    ('该文列出了准静态场景的限制。', 'Image pairs require (quasi) static scenes.', 'We observe each source image twice.'),
    ('该文明确列出只有在（准）静态场景中才能获得内容（准）恒定的图像对。', 'Image pairs require (quasi) static scenes.', 'We observe each source image twice.'),
])
def test_specific_caveats_need_their_actual_source_concept(claim, proof, wrong):
    assert not explicit_source_details_covered(claim, wrong)
    assert explicit_source_details_covered(claim, proof)


def test_loss_conjunction_preserves_separately_checkable_statistics():
    claim = 'The L2 loss is minimized by the mean of target values, and the L1 loss is minimized by their median.'
    for sentence in (claim, claim.replace(', and', ' and')):
        parts = split_definition_bridges(sentence)
        assert len(parts) == 2 and 'L2' in parts[0] and 'L1' in parts[1]
        assert ''.join(sentence[a:b] for a, b in citation_sentence_spans(sentence)) == sentence
        assert len(split_definition_bridges('It is false that ' + sentence)) == 1
        assert len(split_definition_bridges('If ' + sentence)) == 1


def test_real_source_recipe_survives_selection_merging_and_physical_verification(tmp_path):
    from kb.direct_question_evidence import attach_direct_source_evidence, direct_question_evidence_hits
    from kb.n2n_target_condition import _verified_block

    path = tmp_path / 'paper.md'
    path.write_text('<!-- kb_page: 2 -->\n## Method\n\n' + DRAW + '\n\nOur training procedure is as follows. ' + RECIPE, encoding='utf8')
    hits = direct_question_evidence_hits(path, 'What are the training requirements of the method that adds sampled noise from a statistical noise model?')
    assert len(hits) >= 2
    first = _merge_same_source_answer_hits(hits)
    final = _merge_same_source_answer_hits(first + hits)
    slot = {**hits[0]['meta'], 'preferred_system': 'system_a', 'candidate_hits': [1], 'evidence_quote': hits[0]['text']}
    plan = attach_direct_source_evidence({'slots': [slot]}, final)
    passages = final[0]['meta']['source_passages']
    assert len(passages) <= 5
    assert all(_verified_block(str(path), p) for p in passages)
    verified = plan['slots'][0]['direct_source_evidence']
    assert any('predict Y given Z' in p['text'] for p in verified)
    assert any('known noise model' in p['text'] for p in verified)
