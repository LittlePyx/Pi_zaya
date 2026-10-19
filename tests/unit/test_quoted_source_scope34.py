from __future__ import annotations

from kb.explicit_source_attribution import source_attribution_covered


def test_mathematical_typesetting_space_is_not_a_different_quoted_statement():
    claim = 'The paper states "if input-conditioned target distributions p(y|x) are replaced with arbitrary distributions that have the same conditional expected values."'
    quote = 'The optimal parameters remain unchanged, if input-conditioned target distributions p ( y | x ) are replaced with arbitrary distributions that have the same conditional expected values.'
    assert source_attribution_covered(claim, quote, 'noise2noise.pdf')
    for wrong in (quote.replace('p ( y | x )', 'p ( x | y )'), quote.replace('same conditional', 'same unconditional'), quote.replace('are replaced', 'are not replaced')):
        assert not source_attribution_covered(claim, wrong, 'noise2noise.pdf')


def test_quoted_terminal_comma_does_not_discard_the_source_sentence():
    claim = 'The paper states "if we replace the targets with random numbers whose expectations match the targets," in expectation.'
    quote = 'The estimate remains unchanged if we replace the targets with random numbers whose expectations match the targets.'
    assert source_attribution_covered(claim, quote, 'noise2noise.pdf')
    assert not source_attribution_covered(claim, quote.replace('targets.', 'inputs.'), 'noise2noise.pdf')


def test_own_requirement_is_not_the_other_methods_negative_requirement():
    raw = 'While we require only single noisy images vs. paired images for training, we require a known noise model vs. blind denoising. Another method requires a noise estimate but no training data. A third method does not require a noise model but requires independent noise.'
    assert source_attribution_covered(raw, raw, 'scope.md')
    assert not source_attribution_covered('Our method does not require a noise model.', raw, 'scope.md')
    assert not source_attribution_covered('Self2Self needs no noise-model knowledge.', 'Other methods do not require a noise model. Self2Self uses a known noise model.', 'self2self.pdf')
