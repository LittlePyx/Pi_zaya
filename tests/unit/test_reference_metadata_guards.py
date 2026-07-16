import sys
import types
from pathlib import Path

import api.reference_ui as reference_ui
import pytest
from api.reference_ui import _ensure_summary_line, _merge_meta_prefer_richer, enrich_citation_detail_meta
from ui.chat_widgets import _normalize_math_markdown
from ui import refs_renderer


@pytest.fixture(autouse=True)
def _disable_unmocked_external_abstract_requests(monkeypatch):
    """Keep unit tests deterministic when they exercise a later provider."""
    monkeypatch.setattr(reference_ui, "_summary_from_datacite_description", lambda meta: "")
    monkeypatch.setattr(reference_ui, "_summary_from_europe_pmc_abstract", lambda meta: "")


def test_normalize_math_markdown_protects_table_inline_math_pipes():
    src = (
        "| method | formula |\n"
        "|---|---|\n"
        "| ours | $H(X|Y)$ |\n"
    )
    out = _normalize_math_markdown(src)
    assert "$H(X\\vert{}Y)$" in out
    # Table structure should remain intact.
    assert out.count("\n") == src.count("\n")
    assert "| ours |" in out


def test_resolve_pdf_for_source_accepts_library_pdf_and_rejects_outside(tmp_path: Path):
    pdf_root = tmp_path / "pdfs"
    pdf_root.mkdir(parents=True)
    paper = pdf_root / "Paper.pdf"
    paper.write_bytes(b"%PDF-1.4\n")
    outside = tmp_path / "outside" / "Paper.pdf"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"%PDF-1.4\n")

    assert refs_renderer._resolve_pdf_for_source(pdf_root, str(paper)) == paper.resolve(strict=False)
    assert refs_renderer._resolve_pdf_for_source(pdf_root, "Paper.pdf") == paper.resolve(strict=False)
    assert refs_renderer._resolve_pdf_for_source(pdf_root, str(outside)) is None
    assert refs_renderer._resolve_pdf_for_source(pdf_root, "../outside/Paper.pdf") is None


def test_merge_meta_prefer_richer_keeps_existing_data_on_doi_conflict():
    base = {
        "title": "Correct Title",
        "authors": "A. Author, B. Writer",
        "venue": "Journal A",
        "year": "2022",
        "doi": "10.1000/correct",
        "journal_if": "8.1",
    }
    incoming = {
        "title": "Wrong Title",
        "authors": "X. Wrong",
        "venue": "Journal B",
        "year": "2017",
        "doi": "10.2000/wrong",
        "journal_if": "99.9",
        "citation_count": 9999,
    }
    out = _merge_meta_prefer_richer(base, incoming)
    assert str(out.get("doi") or "") == "10.1000/correct"
    assert str(out.get("title") or "") == "Correct Title"
    assert str(out.get("venue") or "") == "Journal A"
    assert str(out.get("journal_if") or "") == "8.1"
    assert int(out.get("citation_count") or 0) == 0


def test_merge_meta_prefer_richer_keeps_raw_identity_for_low_similarity_external_hit():
    base = {
        "title": "The missing cone problem and low-pass distortion in optical serial sectioning microscopy",
        "authors": "Macias-Garza F, Bovik A",
        "venue": "IEEE Trans. Acoust., Speech, Signal Process.",
        "year": "1988",
    }
    incoming = {
        "title": "Missing Cone Of Frequencies And Low-Pass Distortion In Three-Dimensional Microscopic Images",
        "authors": "F Macias-Garza",
        "venue": "Proceedings of SPIE",
        "year": "2008",
        "doi": "10.1117/12.7976703",
        "doi_url": "https://doi.org/10.1117/12.7976703",
        "citation_count": 22,
        "citation_source": "OpenAlex",
        "journal_if": "1.2",
        "journal_quartile": "Q4",
        "match_method": "bibliographic",
        "title_similarity": 0.4629,
        "match_score": 0.8129,
    }

    out = _merge_meta_prefer_richer(base, incoming)

    assert str(out.get("title") or "") == base["title"]
    assert str(out.get("authors") or "") == base["authors"]
    assert str(out.get("venue") or "") == base["venue"]
    assert str(out.get("year") or "") == base["year"]
    assert str(out.get("doi") or "") == "10.1117/12.7976703"
    assert int(out.get("citation_count") or 0) == 22
    assert str(out.get("journal_if") or "") == "1.2"
    assert str(out.get("external_metadata_status") or "") == "candidate"
    assert str(out.get("external_title") or "") == incoming["title"]
    assert float(out.get("external_title_similarity") or 0) == 0.4629


def test_ensure_summary_line_uses_citation_context_before_metadata_fallback(monkeypatch):
    monkeypatch.setattr(reference_ui, "fetch_crossref_work_by_doi", lambda doi: {})
    monkeypatch.setattr(reference_ui, "_openalex_work_by_doi", lambda doi: {})
    monkeypatch.setattr(reference_ui, "_semantic_scholar_paper_by_doi", lambda doi: {})
    monkeypatch.setattr(reference_ui, "_doi_landing_page_abstract", lambda doi: "")

    out = _ensure_summary_line(
        {
            "title": "Upstream work without abstract",
            "venue": "Example Journal",
            "year": "2024",
            "citation_context": "The current paper cites this method when explaining how adaptive sampling reduces measurements in single-pixel imaging.",
            "answer_claim": "Adaptive sampling can reduce the measurement budget.",
            "location_label": "Introduction / prior work",
        },
        allow_crossref_abstract=True,
    )

    assert str(out.get("summary_source") or "") == "citation_context"
    assert str(out.get("summary_generation") or "") == "citation_context_fallback"
    assert "adaptive sampling reduces measurements" in str(out.get("summary_line") or "")


def test_reference_index_loader_falls_back_to_default_db_outside_streamlit(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    idx_path = tmp_path / "references_index.json"
    idx_path.write_text(json.dumps({"docs": {"demo": {"refs": {}}}}), encoding="utf-8")

    refs_renderer._load_reference_index_file_cached.cache_clear()
    monkeypatch.setattr(refs_renderer.st, "_is_running_with_streamlit", False, raising=False)
    monkeypatch.setattr(refs_renderer, "load_settings", lambda: SimpleNamespace(db_dir=tmp_path))

    data = refs_renderer._load_reference_index_cached()
    assert isinstance(data, dict)
    assert "docs" in data


def test_lookup_journal_if_closes_impact_factor_resources(monkeypatch):
    closed: list[str] = []
    disposed: list[str] = []
    logger_levels: list[int] = []

    class FakeSession:
        def close(self):
            closed.append("session")

    class FakeLogger:
        def setLevel(self, level):
            logger_levels.append(level)

    class FakeEngine:
        logger = FakeLogger()

        def dispose(self):
            disposed.append("engine")

    class FakeFactor:
        def __init__(self):
            self.manager = types.SimpleNamespace(session=FakeSession(), engine=FakeEngine())

        def search(self, value, key=None):
            if key == "issn" and value == "1094-4087":
                return [
                    {
                        "journal": "Optics Express",
                        "issn": "1094-4087",
                        "eissn": "",
                        "factor": "3.8",
                        "jcr": "Q2",
                    }
                ]
            return []

    fake_pkg = types.ModuleType("impact_factor")
    fake_pkg.__path__ = []
    fake_core = types.ModuleType("impact_factor.core")
    fake_core.Factor = FakeFactor
    monkeypatch.setitem(sys.modules, "impact_factor", fake_pkg)
    monkeypatch.setitem(sys.modules, "impact_factor.core", fake_core)

    out = refs_renderer._lookup_journal_if({"venue": "Optics Express", "issn": "1094-4087"})

    assert out == {
        "journal_if": 3.8,
        "journal_quartile": "Q2",
        "journal_if_source": "JCR dataset (impact_factor)",
        "journal_if_matched_journal": "Optics Express",
    }
    assert closed == ["session"]
    assert disposed == ["engine"]
    assert logger_levels


def test_normalize_reference_for_popup_recovers_sparse_conference_reference():
    ref = {
        "raw": "[22] Sankaranarayanan, A. C., Studer, C. & Baraniuk, R. G. CS-MUVI: video compressive sensing for spatial-multiplexing cameras. In Computational Photography 1-10 (IEEE, 2012).",
        "doi": "10.1109/ICCPhot.2012.6215212",
        "doi_url": "https://doi.org/10.1109/ICCPhot.2012.6215212",
        "title": "",
        "authors": "",
        "venue": "",
        "year": "",
    }

    out = refs_renderer._normalize_reference_for_popup(ref)

    assert str(out.get("title") or "") == "CS-MUVI: video compressive sensing for spatial-multiplexing cameras"
    assert str(out.get("authors") or "") == "Sankaranarayanan, A. C., Studer, C. & Baraniuk, R. G"
    assert str(out.get("venue") or "") == "In Computational Photography"
    assert str(out.get("year") or "") == "2012"
    assert str(out.get("pages") or "") == "1-10"


def test_enrich_citation_detail_meta_prefers_canonical_metadata_for_same_doi(monkeypatch):
    monkeypatch.setattr(
        reference_ui,
        "fetch_best_crossref_meta",
        lambda **kwargs: {
            "title": "CS-MUVI: Video compressive sensing for spatial-multiplexing cameras",
            "authors": "Sankaranarayanan A, Studer C, Baraniuk R",
            "venue": "2012 IEEE International Conference on Computational Photography (ICCP)",
            "year": "2012",
            "pages": "1-10",
            "doi": "10.1109/iccphot.2012.6215212",
        },
    )
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_for_reference", lambda **kwargs: None)
    monkeypatch.setattr(
        reference_ui,
        "_enrich_bibliometrics",
        lambda meta: {**dict(meta or {}), "citation_count": 152, "citation_source": "OpenAlex"},
    )

    ref = {
        "num": 22,
        "source_name": "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf",
        "source_path": r"F:\research-papers\2026\Jan\else\kb_chat\db\NatPhoton-2019-Principles and prospects for single-pixel imaging\NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md",
        "raw": "[22] Sankaranarayanan, A. C., Studer, C. & Baraniuk, R. G. CS-MUVI: video compressive sensing for spatial-multiplexing cameras. In Computational Photography 1-10 (IEEE, 2012).",
        "doi": "10.1109/ICCPhot.2012.6215212",
        "doi_url": "https://doi.org/10.1109/ICCPhot.2012.6215212",
        "title": "",
        "authors": "",
        "venue": "",
        "year": "",
    }

    out = enrich_citation_detail_meta(ref)

    assert str(out.get("title") or "") == "CS-MUVI: Video compressive sensing for spatial-multiplexing cameras"
    assert str(out.get("authors") or "") == "Sankaranarayanan A, Studer C, Baraniuk R"
    assert str(out.get("venue") or "") == "2012 IEEE International Conference on Computational Photography (ICCP)"
    assert str(out.get("year") or "") == "2012"
    assert str(out.get("pages") or "") == "1-10"
    assert int(out.get("citation_count") or 0) == 152
    assert str(out.get("citation_source") or "") == "OpenAlex"


def test_enrich_citation_detail_meta_title_only_fallback_recovers_doi(monkeypatch):
    monkeypatch.setattr(reference_ui, "fetch_crossref_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_for_reference", lambda **kwargs: None)

    def _fake_best_crossref_meta(**kwargs):
        if kwargs.get("allow_title_only") and kwargs.get("query_title") == "Real-time methane leak imaging":
            return {
                "title": "Real-time methane leak imaging",
                "authors": "Gibson G, Sun B, Edgar M",
                "venue": "Optics Express",
                "year": "2017",
                "doi": "10.1364/oe.25.002998",
                "doi_url": "https://doi.org/10.1364/oe.25.002998",
            }
        return None

    monkeypatch.setattr(reference_ui, "fetch_best_crossref_meta", _fake_best_crossref_meta)
    monkeypatch.setattr(reference_ui, "_enrich_bibliometrics", lambda meta: dict(meta or {}))

    out = enrich_citation_detail_meta(
        {
            "title": "Real-time methane leak imaging",
            "venue": "Some noisy venue text",
            "year": "2018",
            "doi": "",
            "raw": "",
        }
    )

    assert str(out.get("doi") or "") == "10.1364/oe.25.002998"
    assert str(out.get("doi_url") or "").startswith("https://doi.org/10.1364/oe.25.002998")


def test_enrich_citation_detail_meta_backfills_arxiv_doi_from_raw(monkeypatch):
    monkeypatch.setattr(reference_ui, "fetch_crossref_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_for_reference", lambda **kwargs: None)
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_meta", lambda **kwargs: None)
    monkeypatch.setattr(reference_ui, "fetch_crossref_work_by_doi", lambda doi: {})
    monkeypatch.setattr(reference_ui, "_openalex_work_by_doi", lambda doi: {})
    monkeypatch.setattr(reference_ui, "_semantic_scholar_paper_by_doi", lambda doi: {})
    monkeypatch.setattr(reference_ui, "_doi_landing_page_abstract", lambda doi: "")
    monkeypatch.setattr(reference_ui, "_openalex_arxiv_meta_by_title", lambda title: {})
    monkeypatch.setattr(reference_ui, "_enrich_bibliometrics", lambda meta: dict(meta or {}))

    out = enrich_citation_detail_meta(
        {
            "title": "Neural reflectance fields for appearance acquisition",
            "venue": "arXiv preprint",
            "raw": "[2] Miloš Hašan et al. Neural reflectance fields for appearance acquisition. arXiv preprint arXiv:2008.03824, 2020.",
            "doi": "",
            "doi_url": "",
        }
    )

    assert str(out.get("doi") or "") == "10.48550/arXiv.2008.03824"
    assert str(out.get("doi_url") or "") == "https://doi.org/10.48550/arXiv.2008.03824"


def test_enrich_citation_detail_meta_uses_card_reference_entry_as_raw_seed(monkeypatch):
    monkeypatch.setattr(reference_ui, "fetch_crossref_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_for_reference", lambda **kwargs: None)
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_meta", lambda **kwargs: None)
    monkeypatch.setattr(reference_ui, "fetch_crossref_work_by_doi", lambda doi: {})
    monkeypatch.setattr(reference_ui, "_openalex_work_by_doi", lambda doi: {})
    monkeypatch.setattr(reference_ui, "_semantic_scholar_paper_by_doi", lambda doi: {})
    monkeypatch.setattr(reference_ui, "_doi_landing_page_abstract", lambda doi: "")
    monkeypatch.setattr(reference_ui, "_openalex_arxiv_meta_by_title", lambda title: {})
    monkeypatch.setattr(reference_ui, "_enrich_bibliometrics", lambda meta: dict(meta or {}))

    out = enrich_citation_detail_meta(
        {
            "title": "Single-shot compressive spectral imaging with a dual-disperser architecture",
            "authors": "M. E. Gehm and D. J. Brady",
            "venue": "Optics Express",
            "year": "2007",
            "card_reference_entry": (
                "[24] M. E. Gehm and D. J. Brady. Single-shot compressive spectral imaging "
                "with a dual-disperser architecture. Optics Express, 2007. doi:10.1364/OE.15.014013"
            ),
        }
    )

    assert "M. E. Gehm" in str(out.get("raw") or "")
    assert str(out.get("doi") or "") == "10.1364/OE.15.014013"
    assert str(out.get("doi_url") or "") == "https://doi.org/10.1364/OE.15.014013"


def test_enrich_citation_detail_meta_openalex_arxiv_title_fallback(monkeypatch):
    monkeypatch.setattr(reference_ui, "fetch_crossref_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_for_reference", lambda **kwargs: None)
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_meta", lambda **kwargs: None)
    monkeypatch.setattr(
        reference_ui,
        "_openalex_arxiv_meta_by_title",
        lambda title: {
            "doi": "10.48550/arXiv.2008.03824",
            "doi_url": "https://doi.org/10.48550/arXiv.2008.03824",
            "year": "2020",
            "venue": "arXiv",
        },
    )
    monkeypatch.setattr(reference_ui, "_enrich_bibliometrics", lambda meta: dict(meta or {}))

    out = enrich_citation_detail_meta(
        {
            "title": "Neural reflectance fields for appearance acquisition",
            "venue": "arXiv preprint",
            "raw": "[2] Miloš Hašan et al. Neural reflectance fields for appearance acquisition. arXiv preprint, 2020.",
            "doi": "",
            "doi_url": "",
        }
    )

    assert str(out.get("doi") or "") == "10.48550/arXiv.2008.03824"
    assert str(out.get("doi_url") or "").startswith("https://doi.org/10.48550/arXiv.2008.03824")


def test_enrich_citation_detail_meta_uses_crossref_abstract_summary(monkeypatch):
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_meta", lambda **kwargs: None)
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_for_reference", lambda **kwargs: None)
    monkeypatch.setattr(reference_ui, "_enrich_bibliometrics", lambda meta: dict(meta or {}))
    monkeypatch.setattr(reference_ui, "_llm_summarize_abstract_zh", lambda title, abstract_text: "")
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)
    monkeypatch.setattr(
        reference_ui,
        "fetch_crossref_work_by_doi",
        lambda doi: {
            "abstract": "<jats:p>We propose a compressive imaging method that improves reconstruction quality under low photon counts.</jats:p>",
        },
    )

    out = enrich_citation_detail_meta(
        {
            "doi": "10.1000/demo",
            "title": "Compressive Imaging Under Low Photon Counts",
            "venue": "Optics Letters",
            "year": "2024",
        }
    )

    assert str(out.get("summary_source") or "") == "abstract"
    assert (out.get("summary_quality") or {}).get("status") == "grounded"
    assert (out.get("summary_quality") or {}).get("export_ready") is True
    assert "compressive imaging method" in str(out.get("summary_line") or "").lower()


def test_enrich_citation_detail_meta_falls_back_to_metadata_summary(monkeypatch):
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_meta", lambda **kwargs: None)
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_for_reference", lambda **kwargs: None)
    monkeypatch.setattr(reference_ui, "_enrich_bibliometrics", lambda meta: dict(meta or {}))
    monkeypatch.setattr(reference_ui, "fetch_crossref_work_by_doi", lambda doi: {})

    out = enrich_citation_detail_meta(
        {
            "title": "Snapshot Compressive Imaging with Physics Priors",
            "venue": "Nature Photonics",
            "year": "2025",
        }
    )

    assert str(out.get("summary_source") or "") == "metadata"
    assert (out.get("summary_quality") or {}).get("status") == "fallback"
    assert (out.get("summary_quality") or {}).get("export_ready") is False
    assert "metadata_only" in {
        str(issue.get("code") or "")
        for issue in (out.get("summary_quality") or {}).get("issues", [])
        if isinstance(issue, dict)
    }
    summary_line = str(out.get("summary_line") or "")
    assert "当前仅检索到文献元数据" in summary_line
    assert "建议通过 DOI 查看原文摘要与正文" in summary_line
    assert "Snapshot Compressive Imaging with Physics Priors" not in summary_line


def test_enrich_citation_detail_meta_uses_openalex_abstract_when_crossref_has_none(monkeypatch):
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_meta", lambda **kwargs: None)
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_for_reference", lambda **kwargs: None)
    monkeypatch.setattr(reference_ui, "_enrich_bibliometrics", lambda meta: dict(meta or {}))
    monkeypatch.setattr(reference_ui, "_llm_summarize_abstract_zh", lambda title, abstract_text: "")
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)
    monkeypatch.setattr(reference_ui, "fetch_crossref_work_by_doi", lambda doi: {"abstract": ""})
    monkeypatch.setattr(
        reference_ui,
        "_openalex_work_by_doi",
        lambda doi: {
            "abstract_inverted_index": {
                "We": [0],
                "demonstrate": [1],
                "real-time": [2],
                "methane": [3],
                "leak": [4],
                "imaging": [5],
                "with": [6],
                "a": [7],
                "single-pixel": [8],
                "camera.": [9],
            }
        },
    )

    out = enrich_citation_detail_meta(
        {
            "doi": "10.1364/oe.25.002998",
            "title": "Real-time imaging of methane gas leaks using a single-pixel camera",
            "venue": "Optics Express",
            "year": "2017",
        }
    )

    assert str(out.get("summary_source") or "") == "abstract"
    assert "methane leak imaging" in str(out.get("summary_line") or "").lower()


def test_enrich_citation_detail_meta_uses_semantic_scholar_abstract_when_primary_sources_have_none(monkeypatch):
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_meta", lambda **kwargs: None)
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_for_reference", lambda **kwargs: None)
    monkeypatch.setattr(reference_ui, "_enrich_bibliometrics", lambda meta: dict(meta or {}))
    monkeypatch.setattr(reference_ui, "_llm_summarize_abstract_zh", lambda title, abstract_text: "")
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)
    monkeypatch.setattr(reference_ui, "fetch_crossref_work_by_doi", lambda doi: {"abstract": ""})
    monkeypatch.setattr(reference_ui, "_openalex_work_by_doi", lambda doi: {})
    monkeypatch.setattr(
        reference_ui,
        "_semantic_scholar_paper_by_doi",
        lambda doi: {
            "title": "Adaptive sampling for single-pixel imaging",
            "externalIds": {"DOI": "10.1000/semantic-demo"},
            "abstract": (
                "We develop an adaptive sampling method for single-pixel imaging that selects "
                "informative illumination patterns during acquisition. Experiments show that "
                "the strategy improves reconstruction quality under limited measurements."
            ),
        },
    )

    out = enrich_citation_detail_meta(
        {
            "doi": "10.1000/semantic-demo",
            "title": "Adaptive sampling for single-pixel imaging",
            "venue": "Optics Express",
            "year": "2024",
        }
    )

    assert str(out.get("summary_source") or "") == "abstract"
    assert str(out.get("summary_provider") or "") == "semantic_scholar"
    assert "adaptive sampling method" in str(out.get("summary_line") or "").lower()


def test_enrich_citation_detail_meta_uses_doi_landing_page_abstract_as_last_external_fallback(monkeypatch):
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_meta", lambda **kwargs: None)
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_for_reference", lambda **kwargs: None)
    monkeypatch.setattr(reference_ui, "_enrich_bibliometrics", lambda meta: dict(meta or {}))
    monkeypatch.setattr(reference_ui, "_llm_summarize_abstract_zh", lambda title, abstract_text: "")
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)
    monkeypatch.setattr(reference_ui, "fetch_crossref_work_by_doi", lambda doi: {"abstract": ""})
    monkeypatch.setattr(reference_ui, "_openalex_work_by_doi", lambda doi: {})
    monkeypatch.setattr(reference_ui, "_semantic_scholar_paper_by_doi", lambda doi: {})
    monkeypatch.setattr(
        reference_ui,
        "_doi_landing_page_abstract",
        lambda doi: (
            "We analyze low-pass distortion in three-dimensional microscopic imaging and compare "
            "how missing spatial frequencies affect reconstructed structures. The results show "
            "that the missing-cone geometry imposes characteristic resolution limits."
        ),
    )

    out = enrich_citation_detail_meta(
        {
            "doi": "10.1117/12.7976703",
            "title": "Missing Cone Of Frequencies And Low-Pass Distortion In Three-Dimensional Microscopic Images",
            "venue": "Optical Engineering",
            "year": "1988",
        }
    )

    assert str(out.get("summary_source") or "") == "abstract"
    assert str(out.get("summary_provider") or "") == "doi_landing_page"
    assert "low-pass distortion" in str(out.get("summary_line") or "").lower()


def test_enrich_citation_detail_meta_translates_abstract_summary_to_chinese(monkeypatch):
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_meta", lambda **kwargs: None)
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_for_reference", lambda **kwargs: None)
    monkeypatch.setattr(reference_ui, "_enrich_bibliometrics", lambda meta: dict(meta or {}))
    monkeypatch.setattr(reference_ui, "_llm_summarize_abstract_zh", lambda title, abstract_text: "")
    monkeypatch.setattr(
        reference_ui,
        "fetch_crossref_work_by_doi",
        lambda doi: {"abstract": "<jats:p>We propose a robust spectral reconstruction pipeline for low-light imaging.</jats:p>"},
    )
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: "我们提出了一个面向低照度成像的稳健光谱重建流程。")

    out = enrich_citation_detail_meta(
        {
            "doi": "10.1000/demo-translate",
            "title": "Robust spectral reconstruction for low-light imaging",
            "venue": "Optics Express",
            "year": "2024",
        }
    )

    assert str(out.get("summary_source") or "") == "abstract"
    assert str(out.get("summary_line") or "") == "我们提出了一个面向低照度成像的稳健光谱重建流程。"


def test_enrich_citation_detail_meta_prefers_llm_academic_summary(monkeypatch):
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_meta", lambda **kwargs: None)
    monkeypatch.setattr(reference_ui, "fetch_best_crossref_for_reference", lambda **kwargs: None)
    monkeypatch.setattr(reference_ui, "_enrich_bibliometrics", lambda meta: dict(meta or {}))
    monkeypatch.setattr(
        reference_ui,
        "fetch_crossref_work_by_doi",
        lambda doi: {"abstract": "<jats:p>We propose a robust spectral reconstruction pipeline for low-light imaging.</jats:p>"},
    )
    monkeypatch.setattr(
        reference_ui,
        "_llm_summarize_abstract_zh",
        lambda title, abstract_text: "本文面向低照度光谱成像中的重建稳定性问题，提出了鲁棒重建框架。该方法通过联合先验约束与重建优化提升了弱光条件下的可恢复性。实验表明其在重建质量与稳定性上优于对比方法。",
    )
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: "这条不应被采用")

    out = enrich_citation_detail_meta(
        {
            "doi": "10.1000/demo-llm",
            "title": "Robust spectral reconstruction for low-light imaging",
            "venue": "Optics Express",
            "year": "2024",
        }
    )

    assert str(out.get("summary_source") or "") == "abstract"
    assert "本文面向低照度光谱成像中的重建稳定性问题" in str(out.get("summary_line") or "")
    assert "这条不应被采用" not in str(out.get("summary_line") or "")


def test_inpaper_numeric_citation_stays_plain_when_identity_conflicts(monkeypatch):
    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 50:
            return None
        return {
            "source_path": "doc.en.md",
            "source_name": "doc.pdf",
            "ref_num": 50,
            "ref": {
                "authors": "Phillips D, Sun M",
                "year": "2017",
                "title": "Adaptive foveated single-pixel imaging with dynamic super-sampling",
                "raw": "[50] Phillips D, Sun M, Taylor J, et al. Adaptive foveated single-pixel imaging with dynamic super-sampling. Sci Adv, 2017.",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    md = "Zhang et al. (2018) [50] proposed a CNN-guided strategy."
    hits = [{"meta": {"source_path": "doc.en.md", "source_sha1": "abc"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[50](#" not in out
    assert "[50]" in out
    assert details == []


def test_inpaper_numeric_citation_stays_plain_without_identity_signal(monkeypatch):
    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 116:
            return None
        return {
            "source_path": "doc.en.md",
            "source_name": "doc.pdf",
            "ref_num": 116,
            "ref": {
                "authors": "Wang X, Li Y",
                "year": "2020",
                "title": "A paper",
                "raw": "[116] Wang X, Li Y. A paper. IEEE TCI, 2020.",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    md = "Wang uses DenseNet for reconstruction [116]."
    hits = [{"meta": {"source_path": "doc.en.md", "source_sha1": "abc"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[116](#" not in out
    assert "[116]" in out
    assert details == []


def test_inpaper_citation_link_kept_when_year_signal_matches(monkeypatch):
    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 116:
            return None
        return {
            "source_path": "doc.en.md",
            "source_name": "doc.pdf",
            "ref_num": 116,
            "ref": {
                "authors": "Wang X, Li Y",
                "year": "2020",
                "title": "A paper",
                "raw": "[116] Wang X, Li Y. A paper. IEEE TCI, 2020.",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    md = "Wang et al. (2020) proposed this method [116]."
    hits = [{"meta": {"source_path": "doc.en.md", "source_sha1": "abc"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[116](#" in out
    assert len(details) == 1


def test_structured_citation_token_still_produces_clickable_link(monkeypatch):
    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 24:
            return None
        return {
            "source_path": "doc.en.md",
            "source_name": "doc.pdf",
            "ref_num": 24,
            "ref": {
                "authors": "Gehm M, Brady D",
                "year": "2007",
                "title": "Single-shot compressive spectral imaging with a dual-disperser architecture",
                "raw": "[24] Gehm M, Brady D. Single-shot compressive spectral imaging with a dual-disperser architecture. Opt Express, 2007.",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    sid = refs_renderer._source_cite_id("doc.en.md")
    md = f"Gehm et al. (2007) proposed this [[CITE:{sid}:24]]."
    hits = [{"meta": {"source_path": "doc.en.md", "source_sha1": "abc"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[24](#" in out
    assert len(details) == 1


def test_structured_citation_is_hidden_when_author_year_conflicts(monkeypatch):
    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 24:
            return None
        return {
            "source_path": "doc.en.md",
            "source_name": "doc.pdf",
            "ref_num": 24,
            "ref": {
                "authors": "Townsend P, Foster J",
                "year": "2003",
                "title": "Application of imaging spectroscopy ...",
                "raw": "[24] Townsend P, Foster J. Application of imaging spectroscopy ... IEEE TGRS, 2003.",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    sid = refs_renderer._source_cite_id("doc.en.md")
    md = f"Gehm et al. (2007) proposed this [[CITE:{sid}:24]]."
    hits = [{"meta": {"source_path": "doc.en.md", "source_sha1": "abc"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[24](#" not in out
    assert "[[CITE:" not in out
    assert len(details) == 0


def test_structured_citation_is_clickable_without_local_verification(monkeypatch):
    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 24:
            return None
        return {
            "source_path": "doc.en.md",
            "source_name": "doc.pdf",
            "ref_num": 24,
            "ref": {
                "authors": "Gehm M, Brady D",
                "year": "2007",
                "title": "Single-shot compressive spectral imaging with a dual-disperser architecture",
                "raw": "[24] Gehm M, Brady D. Single-shot compressive spectral imaging with a dual-disperser architecture. Opt Express, 2007.",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    sid = refs_renderer._source_cite_id("doc.en.md")
    md = f"This follows prior work [[CITE:{sid}:24]]."
    hits = [{"meta": {"source_path": "doc.en.md", "source_sha1": "abc"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[24](#" in out
    assert len(details) == 1


def test_numeric_citation_is_hidden_when_source_is_ambiguous(monkeypatch):
    def fake_resolve(_index_data, source_path, ref_num, *, source_sha1=""):
        del _index_data, source_sha1
        if int(ref_num) != 7:
            return None
        if str(source_path).endswith("a.en.md"):
            return {
                "source_path": "a.en.md",
                "source_name": "a.pdf",
                "ref_num": 7,
                "ref": {"raw": "[7] Ref A"},
            }
        if str(source_path).endswith("b.en.md"):
            return {
                "source_path": "b.en.md",
                "source_name": "b.pdf",
                "ref_num": 7,
                "ref": {"raw": "[7] Ref B"},
            }
        return None

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda sp: "a.pdf" if sp.endswith("a.en.md") else "b.pdf")

    md = "Ambiguous citation [7] should not remain as plain non-clickable marker."
    hits = [
        {"meta": {"source_path": "a.en.md", "source_sha1": "aaa"}},
        {"meta": {"source_path": "b.en.md", "source_sha1": "bbb"}},
    ]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[7](#" not in out
    assert "[7]" not in out
    assert details == []


def test_numeric_citation_stays_plain_when_unique_but_ungrounded(monkeypatch):
    def fake_resolve(_index_data, source_path, ref_num, *, source_sha1=""):
        del _index_data, source_sha1
        if int(ref_num) != 19:
            return None
        if str(source_path).endswith("a.en.md"):
            return {
                "source_path": "a.en.md",
                "source_name": "a.pdf",
                "ref_num": 19,
                "ref": {"raw": "[19] Unique Ref In A"},
            }
        if str(source_path).endswith("b.en.md"):
            return None
        return None

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda sp: "a.pdf" if sp.endswith("a.en.md") else "b.pdf")

    md = "Cross-source answer keeps only resolvable marker [19]."
    hits = [
        {"meta": {"source_path": "a.en.md", "source_sha1": "aaa"}},
        {"meta": {"source_path": "b.en.md", "source_sha1": "bbb"}},
    ]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[19](#" not in out
    assert "[19]" in out
    assert details == []


def test_authoritative_answer_numbering_hides_ungrounded_numeric_fallbacks(monkeypatch):
    def fake_resolve(_index_data, source_path, ref_num, *, source_sha1=""):
        del _index_data, source_sha1
        if str(source_path).endswith("a.en.md") and int(ref_num) in {3, 9}:
            return {
                "source_path": "a.en.md",
                "source_name": "a.pdf",
                "ref_num": int(ref_num),
                "ref": {"raw": f"[{int(ref_num)}] Unique but ungrounded bibliography entry"},
            }
        return None

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda sp: "a.pdf" if sp.endswith("a.en.md") else "b.pdf")

    hits = [
        {"meta": {"source_path": "a.en.md", "source_sha1": "aaa", "ref_answer_citation_num": 1}},
        {"meta": {"source_path": "b.en.md", "source_sha1": "bbb", "ref_answer_citation_num": 2}},
    ]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        "An excluded source marker [3] and a synthetic marker [9] must not leak.",
        hits,
        anchor_ns="t-authoritative",
        canonical_paths=["a.en.md", "b.en.md", "excluded.en.md"],
    )

    assert "[3]" not in out
    assert "[9]" not in out
    assert details == []


def test_numeric_citation_range_stays_compact_when_ungrounded(monkeypatch):
    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) not in {11, 12, 13}:
            return None
        return {
            "source_path": "doc.en.md",
            "source_name": "doc.pdf",
            "ref_num": int(ref_num),
            "ref": {
                "authors": "Range Author",
                "year": "2021",
                "title": f"Reference {int(ref_num)}",
                "raw": f"[{int(ref_num)}] Range Author. Reference {int(ref_num)}. Journal, 2021.",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    md = "Prior work [11-13] supports this claim."
    hits = [{"meta": {"source_path": "doc.en.md", "source_sha1": "abc"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[11](#" not in out
    assert "[12](#" not in out
    assert "[13](#" not in out
    assert "[11-13]" in out
    assert details == []


def test_structured_citation_is_hidden_when_sid_cannot_map(monkeypatch):
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})

    md = "Unmapped token [[CITE:sdeadbeef:24]] should be hidden."
    hits = [{"meta": {"source_path": "doc.en.md", "source_sha1": "abc"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[24]" not in out
    assert "CITE" not in out.upper()
    assert details == []
