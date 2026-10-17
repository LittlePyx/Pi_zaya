from __future__ import annotations

import pytest

from kb.claim_evidence_runtime import _drop_hard_mismatched_claims, _support_score
from kb.evidence_binding import _system_a_fact_quantities


@pytest.mark.parametrize("word,value", [("single", "1"), ("two", "2"), ("four", "4")])
@pytest.mark.parametrize("tail", ["model is trained on noisy images", "network processes video frames", "algorithm uses detectors"])
def test_number_word_does_not_borrow_a_later_nouns_count_unit(word: str, value: str, tail: str) -> None:
    quantities = _system_a_fact_quantities(f"A {word} {tail}.")
    assert (value, "", "") in quantities
    assert not any(unit for number, unit, _ in quantities if number == value)


@pytest.mark.parametrize("surface,expected", [
    ("two independent noisy training images", ("2", "image", "")),
    ("four independent single-pixel detectors", ("4", "detector", "")),
    ("four spatially-separated, single-pixel detectors", ("4", "detector", "")),
    ("one compressed image and one video frame", ("1", "frame", "")),
])
def test_number_word_keeps_its_own_count_unit(surface: str, expected: tuple[str, str, str]) -> None:
    assert expected in _system_a_fact_quantities(surface)


_EVIDENCE = (
    "In the training stage, we use the noisy images from a wide range of noise levels "
    r"(e.g., $\sigma \in [0,55]$) to train a single DnCNN model. "
    "Given a test image whose noise level belongs to the noise level range, the learned "
    "single DnCNN model can be utilized to denoise it without estimating its noise level."
)
_CLAIM = (
    "Training noise-level interval: For blind Gaussian denoising, a single DnCNN model is "
    r"trained on noisy images spanning a wide range of noise levels, e.g., $\sigma \in [0,55]$. "
    r"The experimental setting confirms the Gaussian noise is drawn from the range $[0,55]$ [1]."
)


def test_cited_training_range_survives_repeated_hard_mismatch_audit() -> None:
    assert _support_score(_CLAIM, _EVIDENCE) >= 5
    answer, dropped = _drop_hard_mismatched_claims(_CLAIM, [{"text": _EVIDENCE}])
    assert answer == _CLAIM
    assert dropped == []
    assert _drop_hard_mismatched_claims(answer, [{"text": _EVIDENCE}]) == (answer, [])


@pytest.mark.parametrize("claim", [
    _CLAIM.replace("55", "75"),
    _CLAIM.replace("a single DnCNN model", "a single training image"),
    "DnCNN requires two noisy training images [1].",
    _CLAIM.replace("DnCNN", "Noise2Void"),
])
def test_wrong_range_cardinality_and_method_remain_unsupported(claim: str) -> None:
    assert _support_score(claim, _EVIDENCE) == 0
    answer, dropped = _drop_hard_mismatched_claims(claim, [{"text": _EVIDENCE}])
    assert not answer
    assert dropped
