from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

import pytest

from kb.provenance_similarity import _cached_ratio, provenance_similarity


@pytest.mark.parametrize("left,right", [
    ("", ""),
    ("tide", "diet"),
    ("diet", "tide"),
    ("噪声独立，均值为零", "噪声不独立，均值非零"),
    (r"E[X|Z] = 2E[Y|Z] - Z", r"E[X|Z] = Z - 2E[Y|Z]"),
    ("a" * 210 + "bc", "a" * 210 + "cb"),
    ("a" * 2101 + "bc", "a" * 2101 + "cb"),
])
def test_provenance_ratio_preserves_difflib_orientation_and_autojunk(left, right):
    expected = float(SequenceMatcher(None, left, right).ratio())
    assert provenance_similarity(left, right) == expected
    assert provenance_similarity(left, right) == expected


def test_repeated_comparisons_are_reused_without_merging_reversed_pairs():
    _cached_ratio.cache_clear()
    assert provenance_similarity("tide", "diet") == 0.25
    assert provenance_similarity("diet", "tide") == 0.5
    for _ in range(20):
        assert provenance_similarity("tide", "diet") == 0.25
    assert _cached_ratio.cache_info().misses == 2
    assert _cached_ratio.cache_info().hits == 20


def test_cache_is_bounded_and_large_inputs_are_not_retained():
    _cached_ratio.cache_clear()
    for i in range(2050):
        provenance_similarity(str(i), "different")
    assert _cached_ratio.cache_info().currsize == 2048
    before = _cached_ratio.cache_info()
    provenance_similarity("x" * 2001, "x")
    assert _cached_ratio.cache_info() == before


def test_concurrent_comparisons_preserve_exact_results():
    pairs = [("tide", "diet"), ("diet", "tide"), ("零均值", "非零均值")] * 30
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda pair: provenance_similarity(*pair), pairs))
    assert results == [float(SequenceMatcher(None, *pair).ratio()) for pair in pairs]
