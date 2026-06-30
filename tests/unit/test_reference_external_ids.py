from __future__ import annotations

from pathlib import Path

from api import reference_external_ids
from api.reference_external_ids import (
    _arxiv_backfill_meta_from_texts,
    _arxiv_doi_from_id,
    _extract_arxiv_id_like,
    _is_weak_meta_value,
    _normalize_doi_like,
    build_doi_url,
)


def test_build_doi_url_preserves_existing_url_and_quotes_raw_doi():
    assert build_doi_url("https://doi.org/10.1234/demo") == "https://doi.org/10.1234/demo"
    assert build_doi_url("10.48550/arXiv.2008.03824") == "https://doi.org/10.48550/arXiv.2008.03824"
    assert build_doi_url("") == ""


def test_arxiv_id_extraction_and_doi_normalization():
    assert _extract_arxiv_id_like("arXiv:2008.03824v2") == "2008.03824"
    assert _extract_arxiv_id_like("https://arxiv.org/abs/2310.02687") == "2310.02687"
    assert _extract_arxiv_id_like("10.48550/arXiv.2008.03824") == "2008.03824"
    assert _arxiv_doi_from_id("2008.03824") == "10.48550/arXiv.2008.03824"
    assert _normalize_doi_like("https://doi.org/10.1364/OE.15.014013") == "10.1364/oe.15.014013"
    assert _normalize_doi_like("arXiv:2008.03824") == "10.48550/arxiv.2008.03824"


def test_arxiv_backfill_meta_from_texts():
    out = _arxiv_backfill_meta_from_texts("reference without id", "arXiv:2008.03824")

    assert out["doi"] == "10.48550/arXiv.2008.03824"
    assert out["doi_url"] == "https://doi.org/10.48550/arXiv.2008.03824"
    assert out["arxiv_id"] == "2008.03824"
    assert out["match_method"] == "arxiv_doi_backfill"
    assert _arxiv_backfill_meta_from_texts("no identifier") == {}


def test_weak_meta_value_detection():
    assert _is_weak_meta_value("title", "A") is True
    assert _is_weak_meta_value("title", "Smith2024") is True
    assert _is_weak_meta_value("title", "A robust neural rendering method") is False
    assert _is_weak_meta_value("authors", "Li") is True
    assert _is_weak_meta_value("venue", "O") is True


def test_reference_ui_reuses_external_id_helpers():
    import api.reference_ui as reference_ui

    assert reference_ui.build_doi_url is reference_external_ids.build_doi_url
    assert reference_ui._normalize_doi_like is reference_external_ids._normalize_doi_like


def test_reference_ui_no_longer_defines_external_id_helpers():
    source = (Path(__file__).resolve().parents[2] / "api" / "reference_ui.py").read_text(encoding="utf-8")

    assert "def _normalize_doi_like" not in source
    assert "def _arxiv_backfill_meta_from_texts" not in source
