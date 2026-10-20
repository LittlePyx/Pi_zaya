from __future__ import annotations

import pytest

from kb.evidence_term_mapping import evidence_alignment_tokens


def test_request_specific_stopwords_and_mutable_results_stay_independent():
    text = 'noise dropout median'
    first = evidence_alignment_tokens(text, extra_stopwords=['noise'])
    assert first == {'dropout', 'median'}
    first.clear()
    first.add('invented-evidence')
    assert evidence_alignment_tokens(text, extra_stopwords=['noise']) == {'dropout', 'median'}
    assert evidence_alignment_tokens(text, extra_stopwords=['dropout']) == {'noise', 'median'}
    assert evidence_alignment_tokens(text) == {'noise', 'dropout', 'median'}


@pytest.mark.parametrize(('text', 'expected'), [
    ('噪声 中位数 非局部均值', {'local', 'means', 'median', 'noise', 'noisy', 'non'}),
    ('独立同分布与空间相关噪声', {'correlated', 'distributed', 'identically', 'independence', 'independent', 'noise', 'noisy', 'spatially'}),
    ('PNSR PSNR denoising', {'denoising', 'pnsr', 'psnr'}),
    ('', set()),
    ('L2 loss and L1 median', {'loss'}),
])
def test_repeated_tokenization_preserves_aliases_and_stopword_iterables(text, expected):
    # These outputs were recorded from the immutable source42 implementation.
    extra = [' MEDIAN ', 'noise', 'noise']
    for _ in range(3):
        assert evidence_alignment_tokens(text, extra_stopwords=iter(extra)) == expected
