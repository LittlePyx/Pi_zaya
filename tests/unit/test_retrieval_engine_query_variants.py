from __future__ import annotations

from types import SimpleNamespace

from kb.retrieval_engine import (
    _deterministic_query_variants,
    _search_hits_with_fallback,
    _source_prompt_match_score,
)
from kb.retriever import BM25Retriever


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


def test_deterministic_query_variants_expand_piln_taxonomy_terms() -> None:
    variants = _deterministic_query_variants(
        "PILN 和综述里的深度学习单像素成像主线是什么关系？"
    )
    joined = " ".join(variants).lower()

    assert "data-driven" in joined
    assert "model-driven" in joined
    assert "hybrid-driven" in joined
    assert "taxonomy" in joined


def test_deterministic_query_variants_keep_generic_perovskite_laser_query_neutral() -> None:
    variants = _deterministic_query_variants(
        "钙钛矿激光器这篇和我的单像素成像主线关系大吗？"
        "请直接说明它解决的是什么器件问题。"
    )
    joined = " ".join(variants).lower()

    assert "perovskite laser" in joined
    assert "lasing" in joined
    assert "electrically driven" not in joined
    assert "dual-cavity" not in joined
    assert "peled" not in joined


def test_deterministic_query_variants_add_device_terms_only_for_explicit_clues() -> None:
    variants = _deterministic_query_variants(
        "这篇电驱动双腔钙钛矿激光器如何实现电注入？"
    )
    joined = " ".join(variants).lower()

    assert "electrically driven lasing" in joined
    assert "dual-cavity perovskite device" in joined
    assert "peled" in joined


def test_perovskite_laser_alias_matches_lasing_title_not_detector_material() -> None:
    question = (
        "钙钛矿激光器这篇和我的单像素成像主线关系大吗？"
        "请直接说明它解决的是什么器件问题。"
    )

    assert _source_prompt_match_score(
        question,
        "library/Nature-2025-Electrically driven lasing from a dual-cavity perovskite device.en.md",
    ) >= 7.5
    assert _source_prompt_match_score(
        question,
        "library/Perovskite photodetector material for single-pixel imaging.en.md",
    ) == 0.0


def test_perovskite_laser_query_ranks_dual_cavity_device_over_detector_material() -> None:
    laser_source = "library/Nature-2025-Electrically driven lasing from a dual-cavity perovskite device.en.md"
    detector_source = "library/Perovskite photodetector material for single-pixel imaging.en.md"
    retriever = BM25Retriever(
        [
            {
                "id": "laser",
                "text": (
                    "Electrically driven lasing from a dual-cavity perovskite device. "
                    "A high-power microcavity PeLED drives a low-threshold perovskite microcavity."
                ),
                "meta": {"source_path": laser_source},
            },
            {
                "id": "detector",
                "text": (
                    "A perovskite photodetector material improves single-pixel imaging "
                    "sensitivity and reconstruction."
                ),
                "meta": {"source_path": detector_source},
            },
            {
                "id": "spi-review",
                "text": "A review of computational single-pixel imaging methods.",
                "meta": {"source_path": "library/Single-pixel imaging review.en.md"},
            },
            {
                "id": "microscopy",
                "text": "Structured illumination microscopy and optical sectioning.",
                "meta": {"source_path": "library/Structured illumination microscopy.en.md"},
            },
        ]
    )
    question = (
        "钙钛矿激光器这篇和我的单像素成像主线关系大吗？"
        "请直接说明它解决的是什么器件问题。"
    )

    hits, _scores, _used_query, _used_translation, variants = _search_hits_with_fallback(
        question,
        retriever,
        2,
        SimpleNamespace(api_key=None, query_expansion_enabled=False),
    )

    joined = " ".join(variants).lower()
    assert "perovskite laser" in joined
    assert "dual-cavity" not in joined
    assert hits[0]["meta"]["source_path"] == laser_source


def test_optically_pumped_perovskite_laser_query_does_not_promote_dual_cavity_paper() -> None:
    optical_source = "library/Optically pumped perovskite nanowire lasers.en.md"
    dual_cavity_source = (
        "library/Nature-2025-Electrically driven lasing from a dual-cavity "
        "perovskite device.en.md"
    )
    retriever = BM25Retriever(
        [
            {
                "id": "optical",
                "text": (
                    "Optically pumped perovskite nanowire lasers exhibit low-threshold "
                    "lasing and strong optical gain in a Fabry-Perot cavity."
                ),
                "meta": {"source_path": optical_source},
            },
            {
                "id": "dual-cavity",
                "text": (
                    "Electrically driven lasing from a dual-cavity perovskite device. "
                    "A microcavity PeLED provides electrical injection."
                ),
                "meta": {"source_path": dual_cavity_source},
            },
            {
                "id": "spi-review",
                "text": "A review of computational single-pixel imaging methods.",
                "meta": {"source_path": "library/Single-pixel imaging review.en.md"},
            },
            {
                "id": "microscopy",
                "text": "Structured illumination microscopy and optical sectioning.",
                "meta": {"source_path": "library/Structured illumination microscopy.en.md"},
            },
        ]
    )

    hits, _scores, _used_query, _used_translation, variants = _search_hits_with_fallback(
        "帮我找那篇光泵浦钙钛矿激光器论文",
        retriever,
        2,
        SimpleNamespace(api_key=None, query_expansion_enabled=False),
    )

    joined = " ".join(variants).lower()
    assert "optically pumped" in joined
    assert "dual-cavity" not in joined
    assert "electrically driven" not in joined
    assert hits[0]["meta"]["source_path"] == optical_source
