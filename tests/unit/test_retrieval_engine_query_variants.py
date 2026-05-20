from __future__ import annotations

from kb.retrieval_engine import _deterministic_query_variants


def test_deterministic_query_variants_expand_refocus_mechanism_terms() -> None:
    variants = _deterministic_query_variants(
        "这个 quantum correlation light-field microscope 是怎么把离焦样品重新对焦的？"
    )
    joined = " ".join(variants).lower()

    assert "digital refocusing" in joined
    assert "ray tracing" in joined
    assert "wave propagation" in joined


def test_deterministic_query_variants_expand_thick_sample_tradeoff_terms() -> None:
    variants = _deterministic_query_variants("s2ISM 这篇说的 trade-off 是什么？为什么厚样本会麻烦？")
    joined = " ".join(variants).lower()

    assert "snr" in joined
    assert "optical sectioning" in joined
    assert "out-of-focus background" in joined
