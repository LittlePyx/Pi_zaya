from __future__ import annotations

from pathlib import Path

from api.reference_semantic_badges import (
    _anchor_kind_label,
    _anchor_kind_prefix,
    _build_semantic_badges,
)


def test_anchor_kind_labels_are_localized():
    assert _anchor_kind_prefix("figure") == "图示语义命中"
    assert _anchor_kind_label("figure", 3) == "图3"
    assert _anchor_kind_label("equation", 5) == "公式5"
    assert _anchor_kind_label("custom", 2) == "custom 2"
    assert _anchor_kind_label("figure", 0) == ""


def test_semantic_badges_prefer_anchor_target():
    assert _build_semantic_badges(
        anchor_target_kind="equation",
        anchor_target_number=7,
        anchor_match_score=11.5,
        explicit_doc_match_score=9.0,
    ) == [{"text": "公式语义命中 公式7", "score": 11.5}]


def test_semantic_badges_preserve_decimal_section_label():
    assert _build_semantic_badges(
        anchor_target_kind="section",
        anchor_target_number=5,
        anchor_target_label="5.2",
        anchor_match_score=41.5,
        explicit_doc_match_score=9.0,
    ) == [{"text": "锚点语义命中 section 5.2", "score": 41.5}]


def test_semantic_badges_fall_back_to_doc_direct_link():
    assert _build_semantic_badges(
        anchor_target_kind="",
        anchor_target_number=0,
        anchor_match_score=0.0,
        explicit_doc_match_score=6.1,
    ) == [{"text": "文档语义直连", "score": 6.1}]
    assert _build_semantic_badges(
        anchor_target_kind="",
        anchor_target_number=0,
        anchor_match_score=0.0,
        explicit_doc_match_score=5.9,
    ) == []


def test_reference_ui_no_longer_defines_semantic_badge_helpers():
    source = (Path(__file__).resolve().parents[2] / "api" / "reference_ui.py").read_text(encoding="utf-8")

    assert "def _anchor_kind_prefix" not in source
    assert "def _anchor_kind_label" not in source
    assert "def _build_semantic_badges" not in source
