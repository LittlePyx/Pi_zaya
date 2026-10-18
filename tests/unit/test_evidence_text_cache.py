from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from kb.evidence_text import _clean_display_text_cached, clean_display_text


@pytest.mark.parametrize("raw, expected", [
    ("**independent** and identically distributed", "independent and identically distributed"),
    ("# Assumptions\n- zero-mean noise", "Assumptions zero-mean noise"),
    ("<!-- kb_page: 5 --> **Figure 3.** Mean is not median.", "Figure 3. Mean is not median."),
    ("普通区间 [0,55]，不是引用。", "普通区间 [0,55]，不是引用。"),
    ("conditional $$E[Y|X]=X$$ remains", "conditional $$E[Y|X]=X$$ remains"),
])
def test_cached_display_preserves_conditions_negation_and_math(raw, expected):
    _clean_display_text_cached.cache_clear()
    assert clean_display_text(raw) == expected
    assert clean_display_text(raw) == expected
    assert _clean_display_text_cached.cache_info().hits == 1


def test_display_limit_is_part_of_cache_identity_and_inputs_can_be_unhashable():
    _clean_display_text_cached.cache_clear()
    assert clean_display_text("abcdefgh", max_len=5) == "abcd..."
    assert clean_display_text("abcdefgh", max_len=8) == "abcdefgh"
    assert clean_display_text(["abcdefgh"]) == "['abcdefgh']"


def test_large_display_inputs_are_not_retained_and_capacity_is_bounded():
    _clean_display_text_cached.cache_clear()
    assert clean_display_text("x" * 2001) == "x" * 519 + "..."
    assert _clean_display_text_cached.cache_info().currsize == 0
    for index in range(2050):
        assert clean_display_text(f"**condition {index}**") == f"condition {index}"
    assert _clean_display_text_cached.cache_info().currsize == 2048


def test_concurrent_display_reads_keep_exact_outputs():
    _clean_display_text_cached.cache_clear()
    inputs = [f"**condition {index % 20}** is not optional" for index in range(200)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(clean_display_text, inputs))
    assert values == [f"condition {index % 20} is not optional" for index in range(200)]
