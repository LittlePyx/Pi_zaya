from __future__ import annotations

import re

import pytest

from kb.citation_marker_context import numeric_marker_is_content, strip_citation_decoration, strip_citation_offset


_MARKER = re.compile(r"\[(\d+(?:,\d+)*)\]")


@pytest.mark.parametrize("label", ["range", "range is", "interval", "interval of", "范围为", "区间", "outside", "within"])
def test_plain_interval_survives_citation_cleanup_and_offset(label: str) -> None:
    answer = f"The training {label} [0,55] [1]."
    assert numeric_marker_is_content(answer, answer.index("[0,55]"))
    assert not numeric_marker_is_content(answer, answer.index("[1]"))
    assert strip_citation_decoration(answer, _MARKER) == f"The training {label} [0,55]  ."
    assert strip_citation_offset(answer, 1, _MARKER) == f"The training {label} [0,55] [0]."


@pytest.mark.parametrize("answer", ["The training range [1].", "The studies [1,2].", "The reference interval [1,2,3].", "The studies [0,1].", "The denoiser works without estimating its noise level [1,99]."])
def test_citation_groups_remain_citations(answer: str) -> None:
    assert not numeric_marker_is_content(answer, answer.index("["))


def test_repeated_training_interval_keeps_every_literal_occurrence() -> None:
    answer = r"The range is $\sigma\in[0,55]$. The experiment uses the range [0,55] [1]."
    cleaned = strip_citation_decoration(answer, _MARKER)
    assert cleaned.count("[0,55]") == 2
    assert "[1]" not in cleaned
