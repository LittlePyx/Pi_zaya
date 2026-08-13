import json
import time

from kb import reference_index as ref_index
from kb import citation_meta


def test_assess_source_reference_alignment_accepts_matching_rows():
    ref_map = {
        1: "[1] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104-4113, 2016.",
        2: "[2] Mu Qiao, Ziyi Meng, Jiawei Ma, and Xin Yuan. Deep learning for video compressive sensing. APL Photonics, 5(3), 2020.",
        3: "[3] Lishun Wang, Miao Cao, Yong Zhong, and Xin Yuan. Spatial-temporal transformer for video snapshot compressive imaging. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(7):9072-9089, 2022.",
        4: "[4] Patrick Llull, Xuejun Liao, Xin Yuan, Jianbo Yang, David Kittle, Lawrence Carin, Guillermo Sapiro, and David J Brady. Coded aperture compressive temporal imaging. Optics Express, 21(9):10526-10545, 2013.",
    }
    rows = [
        {"text": "Schonberger J Frahm J Structure-from-Motion Revisited 2016 IEEE Conference on Computer Vision and Pattern Recognition CVPR 2016 4104-4113", "author": "Schonberger J, Frahm J", "year": "2016", "doi": "10.1109/cvpr.2016.445"},
        {"text": "Qiao M Meng Z Ma J Yuan X Deep learning for video compressive sensing APL Photonics 2020 5 3 10.1063/1.5140721", "author": "Qiao M, Meng Z, Ma J, Yuan X", "year": "2020", "doi": "10.1063/1.5140721"},
        {"text": "Wang L Cao M Zhong Y Yuan X Spatial-Temporal Transformer for Video Snapshot Compressive Imaging IEEE Transactions on Pattern Analysis and Machine Intelligence 2022 10.1109/tpami.2022.3225382", "author": "Wang L, Cao M, Zhong Y, Yuan X", "year": "2022", "doi": "10.1109/tpami.2022.3225382"},
        {"text": "Llull P Liao X Yuan X Yang J Kittle D Carin L Sapiro G Brady D Coded aperture compressive temporal imaging Optics Express 2013 21 9 10526-10545", "author": "Llull P, Liao X, Yuan X, Yang J, Kittle D, Carin L, Sapiro G, Brady D", "year": "2013", "doi": "10.1364/OE.21.010526"},
    ]

    assert ref_index._assess_source_reference_alignment(ref_map, rows) is True


def test_assess_source_reference_alignment_rejects_unrelated_rows():
    ref_map = {
        1: "[1] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104-4113, 2016.",
        2: "[2] Mu Qiao, Ziyi Meng, Jiawei Ma, and Xin Yuan. Deep learning for video compressive sensing. APL Photonics, 5(3), 2020.",
        3: "[3] Lishun Wang, Miao Cao, Yong Zhong, and Xin Yuan. Spatial-temporal transformer for video snapshot compressive imaging. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(7):9072-9089, 2022.",
        4: "[4] Patrick Llull, Xuejun Liao, Xin Yuan, Jianbo Yang, David Kittle, Lawrence Carin, Guillermo Sapiro, and David J Brady. Coded aperture compressive temporal imaging. Optics Express, 21(9):10526-10545, 2013.",
    }
    rows = [
        {"text": "Brida G Genovese M Experimental realization of sub-shot-noise quantum imaging Nature Photonics 2010 4 227-230", "author": "Brida G, Genovese M", "year": "2010", "doi": "10.1038/nphoton.2010.29"},
        {"text": "Shapiro J Boyd The physics of ghost imaging Quantum Information Processing 2012", "author": "Shapiro J, Boyd R", "year": "2012", "doi": "10.1007/s11128-012-0416-4"},
        {"text": "Levoy M Ng R Adams A Light field microscopy ACM Transactions on Graphics 2006 25 3 924-934", "author": "Levoy M, Ng R, Adams A", "year": "2006", "doi": "10.1145/1141911.1141976"},
        {"text": "Orth A Crozier K Microscopy with microlens arrays high throughput high resolution imaging Optics Express 2012 20 12 13522-13531", "author": "Orth A, Crozier K", "year": "2012", "doi": "10.1364/OE.20.013522"},
    ]

    assert ref_index._assess_source_reference_alignment(ref_map, rows) is False


def test_assess_source_reference_alignment_allows_doi_only_rows_when_local_entries_lack_doi():
    ref_map = {
        1: "[1] G. Brida, M. Genovese, and I. Ruo Berchera. Experimental realization of sub-shot-noise quantum imaging. Light: Science & Applications, 4:227, 2010.",
        2: "[2] Nigam Samantaray, Ivano Ruo-Berchera, Alice Meda, and Marco Genovese. Realization of the quantum field microscope. Light: Science & Applications, 6:e17005, 2017.",
        3: "[3] T. B. Pittman, Y. H. Shih, D. V. Strekalov, and A. V. Sergienko. Optical imaging by means of two-photon entanglement. Phys. Rev. A, 52:R3429-R3432, 1995.",
        4: "[4] Jeffrey H. Shapiro and Robert W. Boyd. The physics of ghost imaging. Quantum Information Processing, 2012.",
    }
    rows = [
        {"doi": "10.1038/s41467-022-35585-8", "text": "10.1038/s41467-022-35585-8"},
        {"doi": "10.1038/s41566-018-0300-7", "text": "10.1038/s41566-018-0300-7"},
        {"doi": "10.1038/s41467-021-24850-x", "text": "10.1038/s41467-021-24850-x"},
        {"doi": "10.1103/PhysRevA.52.R3429", "text": "10.1103/PhysRevA.52.R3429"},
    ]

    assert ref_index._assess_source_reference_alignment(ref_map, rows) is True


def test_assess_source_reference_alignment_rejects_sparse_structured_tail_offset():
    ref_map = {
        1: "[1] R. H. Webb, Confocal optical microscopy. Rep. Prog. Phys. 59, 427-471 (1996).",
        2: "[2] D. Huang et al. Optical coherence tomography. Science 254, 1178-1181 (1991).",
        3: "[3] W. K. Pratt, J. Kane, and H. C. Andrews, Hadamard transform image coding. Proc. IEEE 57, 58-68 (1969).",
        4: "[4] J. A. Decker, Jr., Hadamard-Transform Image Scanning. App",
    }
    rows = [
        {"doi": "10.1088/0034-4885/59/3/003", "author": "Webb", "venue": "Rep. Prog. Phys.", "year": "1996", "volume": "59", "pages": "427", "text": "Webb Rep. Prog. Phys. 1996 59 427 10.1088/0034-4885/59/3/003"},
        {"doi": "10.1126/science.1957169", "author": "Huang", "venue": "Science", "year": "1991", "volume": "254", "pages": "1178", "text": "Huang Science 1991 254 1178 10.1126/science.1957169"},
        {"doi": "10.1364/AO.9.001392", "author": "Decker", "venue": "Appl. Opt.", "year": "1970", "volume": "9", "pages": "1392", "text": "Decker Appl. Opt. 1970 9 1392 10.1364/AO.9.001392"},
        {"doi": "10.1364/OL.18.001745", "author": "Gourlay", "venue": "Opt. Lett.", "year": "1993", "volume": "18", "pages": "1745", "text": "Gourlay Opt. Lett. 1993 18 1745 10.1364/OL.18.001745"},
    ]

    assert ref_index._assess_source_reference_alignment(ref_map, rows) is False


def test_match_source_reference_uses_structured_fields_when_numbering_is_offset():
    rows = [
        {"doi": "10.1364/AO.45.002965", "author": "Gehm", "venue": "Appl. Opt.", "year": "2006", "volume": "45", "pages": "2965", "text": "Gehm Appl. Opt. 2006 45 2965 10.1364/AO.45.002965"},
        {"doi": "10.1364/AO.46.000365", "author": "Cull", "venue": "Appl. Opt.", "year": "2007", "volume": "46", "pages": "365", "text": "Cull Appl. Opt. 2007 46 365 10.1364/AO.46.000365"},
        {"doi": "10.1364/OE.15.005742", "author": "Fernandez", "venue": "Opt. Express", "year": "2007", "volume": "15", "pages": "5742", "text": "Fernandez Opt. Express 2007 15 5742 10.1364/OE.15.005742"},
    ]
    raw = "[12] E. Cull, M. Gehm, D. Brady, C. Hsieh, O. Momtahan, and A. filtering for miniature spectrometers, Appl. Opt. 46, 365-374"

    matched = ref_index._match_source_reference(raw, rows)

    assert matched is not None
    assert matched["doi"] == "10.1364/AO.46.000365"


def test_reference_metadata_classification_splits_actionable_reasons():
    complete = ref_index.classify_reference_metadata(
        {
            "raw": "[1] Gehm M, Brady D. Single-shot compressive spectral imaging. Optics Express, 2007. doi:10.1364/OE.15.014013",
            "title": "Single-shot compressive spectral imaging",
            "authors": "Gehm M, Brady D",
            "venue": "Optics Express",
            "year": "2007",
            "doi": "10.1364/OE.15.014013",
        }
    )
    assert complete["metadata_status"] == "complete"
    assert complete["missing_reason"] == ""

    crossref = ref_index.classify_reference_metadata(
        {
            "raw": "[1] Gehm M, Brady D. Single-shot compressive spectral imaging. Optics Express, 2007.",
            "title": "Single-shot compressive spectral imaging",
            "authors": "Gehm M, Brady D",
            "venue": "Optics Express",
            "year": "2007",
            "doi": "10.1364/OE.15.014013",
            "crossref_ok": True,
            "match_method": "title+doi_backfill",
        }
    )
    assert crossref["metadata_status"] == "crossref_enriched"

    source_mapped = ref_index.classify_reference_metadata(
        {
            "raw": "[1] Gehm M, Brady D. Single-shot compressive spectral imaging. Optics Express, 2007.",
            "title": "Single-shot compressive spectral imaging",
            "authors": "Gehm M, Brady D",
            "year": "2007",
            "match_method": "source_work_reference_order_exact",
        }
    )
    assert source_mapped["metadata_status"] == "crossref_enriched"
    assert source_mapped["metadata_ready"] is True
    assert source_mapped["metadata_action"] == "none"

    source_structured_without_title = ref_index.classify_reference_metadata(
        {
            "raw": "[30] Schechner Y Y, Nayar S K and Belhumeur P N 2001 Multiplexed illumination Proc. CVPR IEEE 2",
            "authors": "Schechner Y Y",
            "venue": "Proc. CVPR IEEE",
            "year": "2003",
            "match_method": "source_work_reference_order_exact",
        }
    )
    assert source_structured_without_title["metadata_status"] == "crossref_enriched"
    assert source_structured_without_title["metadata_ready"] is True
    assert source_structured_without_title["metadata_action"] == "none"

    sparse = ref_index.classify_reference_metadata(
        {
            "raw": "[2] Boyd et al. Alternating direction method of multipliers. 2011. doi:10.1561/2200000016",
            "title": "Alternating direction method of multipliers",
            "doi": "10.1561/2200000016",
        }
    )
    assert sparse["metadata_status"] == "doi_sparse_refreshable"
    assert sparse["metadata_action"] == "auto_backfill"

    bibliographic_ready = ref_index.classify_reference_metadata(
        {
            "raw": "[3] Smith J, Doe A. Snapshot compressive imaging with learned priors. Optics Express, 2020.",
            "title": "Snapshot compressive imaging with learned priors",
            "authors": "Smith J, Doe A",
            "venue": "Optics Express",
            "year": "2020",
        }
    )
    assert bibliographic_ready["metadata_status"] == "bibliographic_ready"
    assert bibliographic_ready["metadata_ready"] is True
    assert bibliographic_ready["metadata_action"] == "none"

    bibliographic_without_authors = ref_index.classify_reference_metadata(
        {
            "raw": "[8] E. Candès. Stable Signal Recovery from Incomplete and Inaccurate Measurements. Commun. Pure Appl. Math. 59, 1207-1223 (2006).",
            "title": "Stable Signal Recovery from Incomplete and Inaccurate Measurements",
            "venue": "Commun. Pure Appl. Math.",
            "year": "2006",
        }
    )
    assert bibliographic_without_authors["metadata_status"] == "bibliographic_ready"

    retryable = ref_index.classify_reference_metadata(
        {
            "raw": "[3] Smith J, Doe A. Snapshot compressive imaging with learned priors. 2020.",
            "title": "Snapshot compressive imaging with learned priors",
            "authors": "Smith J, Doe A",
            "year": "2020",
        }
    )
    assert retryable["metadata_status"] == "title_lookup_retryable"
    assert retryable["metadata_action"] == "retry"

    no_doi = ref_index.classify_reference_metadata(
        {
            "raw": "[4] OpenAI. GPT-4 technical report. Technical report, 2023. https://openai.com/research/gpt-4",
            "title": "GPT-4 technical report",
            "authors": "OpenAI",
            "venue": "OpenAI",
            "year": "2023",
        }
    )
    assert no_doi["metadata_status"] == "no_doi_expected"
    assert no_doi["missing_reason"] == "no_doi_expected"
    assert no_doi["metadata_action"] == "non_article_ok"

    url_only = ref_index.classify_reference_metadata(
        {"raw": "[7] https://raytrix.de/.", "parse_confidence": 0.20}
    )
    assert url_only["metadata_status"] == "no_doi_expected"
    assert url_only["metadata_action"] == "non_article_ok"
    assert url_only["metadata_ready"] is True

    book = ref_index.classify_reference_metadata(
        {
            "raw": "[9] S. Haykin, Communication Systems (Wiley, 2001).",
            "title": "Communication Systems",
            "authors": "S. Haykin",
            "year": "2001",
        }
    )
    assert book["metadata_status"] == "non_article_source_ok"
    assert book["missing_reason"] == "no_doi_expected"
    assert book["metadata_action"] == "non_article_ok"

    publisher_book_without_year = ref_index.classify_reference_metadata(
        {
            "raw": "[31] Novotny, L. & Hecht, B. Principles of Nano-Optics. 2nd edn. (Cambridge: Cambridge University Press).",
            "title": "Principles of Nano-Optics",
            "authors": "Novotny, L. & Hecht, B",
            "venue": "Cambridge University Press",
        }
    )
    assert publisher_book_without_year["metadata_status"] == "non_article_source_ok"
    assert publisher_book_without_year["metadata_ready"] is True

    report_without_venue = ref_index.classify_reference_metadata(
        {
            "raw": "[17] R Ng, M Levoy, M Bredif, G Duval, M Horowitz, and P Hanrahan. Light field photography with a hand-held plenoptic camera, 2005.",
            "title": "Light field photography with a hand-held plenoptic camera",
            "authors": "R Ng, M Levoy, M Bredif, G Duval, M Horowitz, and P Hanrahan",
            "year": "2005",
        }
    )
    assert report_without_venue["metadata_status"] == "non_article_source_ok"
    assert report_without_venue["metadata_ready"] is True

    truncated = ref_index.classify_reference_metadata(
        {"raw": "[5] Smith J. Incomplete reference ...", "parse_confidence": 0.40}
    )
    assert truncated["metadata_status"] == "truncated_reference"

    source_truncated = ref_index.classify_reference_metadata(
        {
            "raw": "[15] D. J. Brady and M. E. Gehm, “Compressive imaging spectrometry,” in Z. ur Rahman, S. E. Reichenbach, and M. A. Neifeld, eds., vol.",
            "title": "Compressive imaging spectrometry",
            "authors": "D. J. Brady and M. E. Gehm",
            "venue": "Proc. SPIE",
        }
    )
    assert source_truncated["metadata_status"] == "truncated_reference"
    assert source_truncated["metadata_action"] == "source_repair"

    low_confidence = ref_index.classify_reference_metadata({"raw": "[6] Bad OCR source 2020.", "parse_confidence": 0.45})
    assert low_confidence["metadata_status"] == "low_confidence_match"


def test_fallback_meta_extracts_multi_author_publisher_book():
    meta = ref_index._fallback_meta_from_raw_reference(
        "[52] C. S. Burrus, R. A. Gopinath, H. Guo, Introduction to Wavelets and Wavelet Transforms: A Primer (Prentice-Hall, 1997)."
    )

    assert meta["authors"] == "C. S. Burrus, R. A. Gopinath, H. Guo"
    assert meta["title"] == "Introduction to Wavelets and Wavelet Transforms: A Primer"
    assert meta["venue"] == "Prentice-Hall, 1997"
    assert meta["year"] == "1997"


def test_title_lookup_allows_quoted_title_without_venue():
    raw = '[39] J. A. Decker, Jr., "Hadamard-Transform Image Scanning," App'
    title = ref_index._extract_query_title(raw)

    assert title == "Hadamard-Transform Image Scanning"
    assert ref_index._should_try_title_lookup(raw, title) is True


def test_lookup_reference_meta_falls_back_to_openalex(monkeypatch):
    raw = (
        "[3] T. B. Pittman, Y. H. Shih, D. V. Strekalov, and A. V. Sergienko. "
        "Optical imaging by means of two-photon quantum entanglement. Phys. Rev. A, "
        "52:R3429-R3432, 1995."
    )
    expected_title = "Optical imaging by means of two-photon quantum entanglement"
    openalex_calls: list[dict] = []

    monkeypatch.setattr(ref_index, "fetch_best_crossref_for_reference", lambda **kwargs: None)
    monkeypatch.setattr(ref_index, "fetch_best_crossref_meta", lambda **kwargs: None)

    def fake_fetch_best_openalex_meta(**kwargs):
        openalex_calls.append(dict(kwargs))
        return {
            "title": expected_title,
            "authors": "T. B. Pittman, Y. H. Shih, D. V. Strekalov, et al",
            "venue": "Physical Review A",
            "year": "1995",
            "volume": "52",
            "pages": "R3429-R3432",
            "doi": "10.1103/PhysRevA.52.R3429",
            "match_method": "openalex_title",
            "match_score": 0.98,
        }

    monkeypatch.setattr(ref_index, "fetch_best_openalex_meta", fake_fetch_best_openalex_meta)
    cache: dict = {}
    stats: dict[str, int] = {}

    meta, doi_hint = ref_index._lookup_crossref_meta_for_entry(
        raw,
        cache,
        crossref_enabled=True,
        enable_title_lookup=True,
        stats=stats,
    )

    assert doi_hint == ""
    assert isinstance(meta, dict)
    assert meta["doi"] == "10.1103/PhysRevA.52.R3429"
    assert meta["match_method"] == "openalex_title"
    assert openalex_calls and openalex_calls[0]["query_title"] == expected_title
    assert int(stats.get("openalex_network_attempts") or 0) == 1
    assert ref_index._is_crossref_meta_cache_hit(next(iter((cache.get("openalex_title") or {}).values())))


def test_raw_reference_fallback_splits_nature_inline_author_title():
    raw = (
        "[2] Stantchev, R. I., Yu, X., Blu, T. & Pickwell-MacPherson, E. "
        "Real-time terahertz imaging with a single-pixel detector. "
        "*Nat. Commun.* **11**, 2535 (2020)."
    )

    out = ref_index._fallback_meta_from_raw_reference(raw)

    assert out["authors"] == "Stantchev, R. I., Yu, X., Blu, T. & Pickwell-MacPherson, E"
    assert out["title"] == "Real-time terahertz imaging with a single-pixel detector"
    assert out["venue"] == "Nat. Commun"
    assert out["year"] == "2020"


def test_raw_reference_fallback_splits_accented_et_al_author_title():
    raw = (
        "[25] Küppers, M. et al. Confocal interferometric scattering microscopy reveals "
        "3D nanoscopic structure and dynamics in live cells. Nat. Commun. 14, 1962 (2023)."
    )

    out = ref_index._fallback_meta_from_raw_reference(raw)

    assert out["authors"] == "Küppers, M. et al"
    assert out["title"] == "Confocal interferometric scattering microscopy reveals 3D nanoscopic structure and dynamics in live cells"
    assert out["venue"] == "Nat. Commun"
    assert out["year"] == "2023"


def test_raw_reference_meta_replaces_venue_fragment_title_and_noisy_authors():
    raw = (
        "[2] Stantchev, R. I., Yu, X., Blu, T. & Pickwell-MacPherson, E. "
        "Real-time terahertz imaging with a single-pixel detector. "
        "*Nat. Commun.* **11**, 2535 (2020)."
    )
    stale = {
        "title": "Nat. Commun. 11, 2535 (2020)",
        "authors": "Stantchev, R. I., Yu, X., Blu, T. & Pickwell-MacPherson, E. Real-time terahertz imaging with a single-pixel detector",
        "year": "2020",
    }

    merged = ref_index._merge_raw_reference_meta(stale, ref_index._fallback_meta_from_raw_reference(raw))

    assert merged["title"] == "Real-time terahertz imaging with a single-pixel detector"
    assert merged["authors"] == "Stantchev, R. I., Yu, X., Blu, T. & Pickwell-MacPherson, E"
    assert merged["venue"] == "Nat. Commun"
    assert ref_index._reference_has_usable_title("Nat. Commun. 11, 2535 (2020)") is False
    assert ref_index._extract_query_title(raw) == "Real-time terahertz imaging with a single-pixel detector"


def test_raw_reference_fallback_keeps_signal_processing_magazine_together():
    raw = (
        "[9] Duarte, M. F. et al. Single-pixel imaging via compressive sampling. "
        "*IEEE Signal Process. Mag.* **25**, 83-91 (2008)."
    )

    out = ref_index._fallback_meta_from_raw_reference(raw)

    assert out["title"] == "Single-pixel imaging via compressive sampling"
    assert out["venue"] == "IEEE Signal Process. Mag"
    assert out["year"] == "2008"


def test_raw_reference_fallback_handles_initial_only_single_author():
    raw = '[1] M. J. E. Golay, "Multi-slit spectrometry," J. Opt. Soc. Am. 39, 437-444 (1949).'

    out = ref_index._fallback_meta_from_raw_reference(raw)

    assert out["authors"] == "M. J. E. Golay"
    assert out["title"] == "Multi-slit spectrometry"
    assert out["venue"] == "J. Opt. Soc. Am"
    assert out["year"] == "1949"


def test_raw_reference_fallback_recovers_ocr_year_digits():
    raw = '[4] BRUNSWIK, E., & KAMIYA, J. Ecological cue-validity of "proximity" and of other gestalt factors. Amer. J. Psychol., 19S3, 66, 20-32.'

    out = ref_index._fallback_meta_from_raw_reference(raw)

    assert out["title"] == "Ecological cue-validity of \"proximity\" and of other gestalt factors"
    assert out["year"] == "1953"


def test_raw_reference_fallback_does_not_make_fake_title_for_compact_titleless_refs():
    raw = "[3] D. L. Donoho, *IEEE Trans. Inf. Theory* **2006**, *52*, 1289."

    out = ref_index._fallback_meta_from_raw_reference(raw)

    assert out["year"] == "2006"
    assert "title" not in out


def test_raw_reference_fallback_splits_comma_style_title_and_venue():
    raw = (
        "[20] A. Gallivanoni, I. Rech, and M. Ghioni, Progress in quenching circuits "
        "for single photon avalanche diodes, *IEEE Trans. Nucl. Sci.* 57, 3815 (2010)"
    )

    out = ref_index._fallback_meta_from_raw_reference(raw)

    assert out["authors"] == "A. Gallivanoni, I. Rech, and M. Ghioni"
    assert out["title"] == "Progress in quenching circuits for single photon avalanche diodes"
    assert out["venue"] == "IEEE Trans. Nucl. Sci"
    assert out["year"] == "2010"


def test_raw_reference_fallback_extracts_conference_venue_from_comma_style_in_clause():
    raw = (
        "[78] Z. Deng, L. Ling, Y. Deng, C. Han, L. Yu, G. Cao, and Y. Wang, "
        "A novel visible light communication system prototype based on SiPM receiver, "
        "in: *Proceedings of the 4th International Conference on Telecommunications and Communication Engineering*, 2019"
    )

    out = ref_index._fallback_meta_from_raw_reference(raw)

    assert out["title"] == "A novel visible light communication system prototype based on SiPM receiver"
    assert out["venue"] == "Proceedings of the 4th International Conference on Telecommunications and Communication Engineering"
    assert out["year"] == "2019"


def test_raw_reference_fallback_extracts_single_author_and_in_venue():
    raw = (
        "[60] D. Fukuda, Single-photon measurement techniques with a superconducting transition edge sensor, "
        "in: *IEICE Transactions on Electronics* 2019, E102. C, pp 230-234"
    )

    out = ref_index._fallback_meta_from_raw_reference(raw)

    assert out["authors"] == "D. Fukuda"
    assert out["title"] == "Single-photon measurement techniques with a superconducting transition edge sensor"
    assert out["venue"] == "IEICE Transactions on Electronics"
    assert out["year"] == "2019"


def test_raw_reference_fallback_keeps_period_style_before_comma_fallback():
    raw = (
        "[10] Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A. Efros, and Xiaolong Wang. "
        "Colmap-free 3d gaussian splatting, 2024."
    )

    out = ref_index._fallback_meta_from_raw_reference(raw)

    assert out["authors"] == "Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A. Efros, and Xiaolong Wang"
    assert out["title"] == "Colmap-free 3d gaussian splatting"
    assert out["year"] == "2024"


def test_reference_title_quality_allows_short_technical_titles():
    assert ref_index._reference_has_usable_title("Compressed sensing") is True
    assert ref_index._reference_has_usable_title(
        "Snapshot Compressive Imaging: Theory, Algorithms, and Applications"
    ) is True
    assert ref_index._is_plausible_reference_title(
        "Plug-and-Play Algorithms for Large-Scale Snapshot Compressive Imaging"
    ) is True
    assert ref_index._reference_has_usable_title("Johannes L Schonberger and Jan-Michael Frahm") is False
    assert ref_index._is_plausible_reference_title("IEEE Transactions on Information Theory") is False


def test_extract_references_map_cleans_noise_on_early_heading_return():
    md_text = (
        "# Demo\n\n"
        "## References\n"
        "[1] A. Author. Good reference one. Journal, 2020.\n"
        "[2] B. Author. Good reference two. Conference, 2021.\n"
        "[2018] Supplemental section marker\n"
        "[2543] OCR noise token\n"
        "## Supplementary Material\n"
        "random text\n"
    )

    out = ref_index.extract_references_map_from_md(md_text)
    assert 1 in out
    assert 2 in out
    assert 2018 not in out
    # Large outlier cleanup is covered by a dedicated test that provides enough evidence.
    assert 2543 in out


def test_extract_references_map_stops_at_plain_supplementary_material_heading():
    md_text = (
        "# Demo\n\n"
        "## References\n"
        "[40] Xin Yuan. Generalized alternating projection based total variation minimization for compressive sensing. "
        "In 2016 IEEE International conference on image processing (ICIP), pages 2539-2543. IEEE, 2016.\n"
        "[41] Xin Yuan, Yang Liu, Jinli Suo, and Qionghai Dai. Plug-and-play algorithms for large-scale snapshot "
        "compressive imaging. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, "
        "pages 1447-1457, 2020.\n"
        "[42] Xin Yuan, David J Brady, and Aggelos K Katsaggelos. Snapshot compressive imaging: Theory, algorithms, "
        "and applications. IEEE Signal Processing Magazine, 38(2):65-88, 2021.\n"
        "[43] Xin Yuan, Yang Liu, Jinli Suo, Fredo Durand, and Qionghai Dai. Plug-and-play algorithms for video "
        "snapshot compressive imaging. IEEE Transactions on Pattern Analysis and Machine Intelligence, "
        "44(10):7093-7111, 2021.\n"
        "[44] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable "
        "effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer "
        "vision and pattern recognition, pages 586-595, 2018.\n"
        "\n"
        "Supplementary Material\n"
        "<!-- kb_page: 11 -->\n"
        "In this supplementary material, additional experiments compare GAP-TV [40], PnP-FFDNet [41], "
        "PnP-FastDVDNet [43] and EfficientSCI [31].\n"
    )

    out = ref_index.extract_references_map_from_md(md_text)

    assert sorted(out.keys()) == [40, 41, 42, 43, 44]
    assert "Plug-and-play algorithms for video snapshot" in out[43]
    assert "EfficientSCI" not in out[43]
    assert "Supplementary Material" not in out[44]


def test_extract_references_map_skips_appendix_running_header_before_sequential_tail():
    md_text = (
        "# Demo\n\n"
        "## References\n"
        "[35] A. Author. Reference thirty five. Journal, 2020.\n"
        "[36] B. Author. Reference thirty six. Journal, 2021.\n"
        "Appendix\n"
        "<!-- kb_page: 21 -->\n"
        "[37] C. Author. Reference thirty seven. Journal, 2022.\n"
        "[38] D. Author. Reference thirty eight. Journal, 2023.\n"
        "[39] E. Author. Restormer. Conference, 2024.\n"
        "[40] F. Author. Reference forty. Conference, 2025.\n"
        "Appendix\n"
        "37. Duplicate publisher footer text.\n"
    )

    out = ref_index.extract_references_map_from_md(md_text)

    assert sorted(out) == [35, 36, 37, 38, 39, 40]
    assert "Restormer" in out[39]
    assert "Duplicate publisher footer" not in out[40]


def test_extract_references_map_stops_before_unheaded_author_biography_page():
    md_text = (
        "# Demo\n\n"
        "<!-- kb_page: 20 -->\n"
        "## References\n"
        "[235] A. Author. Complete reference 235. Journal of Tests, 2024.\n"
        "[236] B. Author. Complete reference 236. Journal of Tests, 2025.\n"
        "<!-- kb_page: 21 -->\n"
        "**Kai Song** received his B.S. degree in 2019. His research interests include single-pixel imaging.\n"
        "**Yaoxing Bian** received his Ph.D. degree in 2022.\n"
    )

    out = ref_index.extract_references_map_from_md(md_text)

    assert sorted(out) == [235, 236]
    assert "Kai Song" not in out[236]
    assert "research interests" not in out[236]


def test_extract_references_map_does_not_use_body_fig_or_section_numbers_as_refs():
    md_text = (
        "# Demo\n\n"
        "## References\n"
        "[1] First real reference. Journal, 2001.\n"
        "[2] Second real reference. Conference, 2002.\n"
        "## 2. System design\n\n"
        "Fig. 1. Schematic of the spectral imager.\n"
        "The model is shown in Eq. 1. and extended in Sec. 2.\n"
    )
    out = ref_index.extract_references_map_from_md(md_text)
    assert sorted(out.keys()) == [1, 2]
    assert "First real reference" in str(out.get(1) or "")
    assert "Second real reference" in str(out.get(2) or "")


def test_extract_references_map_folds_bracketed_page_range_continuation():
    md_text = (
        "# Demo\n\n"
        "## References\n"
        "[10] Dauphin, Y. N., Fan, A., Auli, M., and Grangier, D. Language modeling with gated "
        "convolutional networks. In International Conference on Machine Learning. pp. 933-\n"
        "[941] PMLR (2017)\n"
        "[11] De, S. and Smith, S. Batch normalization biases residual blocks. NeurIPS (2020)\n"
    )

    out = ref_index.extract_references_map_from_md(md_text)

    assert sorted(out.keys()) == [10, 11]
    assert "pp. 933-941 PMLR (2017)" in out[10]
    assert out[11].startswith("[11] De, S.")


def test_extract_references_map_keeps_sequential_reference_after_open_page_range():
    md_text = (
        "# Demo\n\n"
        "## References\n"
        "[10] A. Author. Entry with a damaged page range. Demo Journal, pp. 9-\n"
        "[11] B. Author. Next real entry. Demo Conference, 2021.\n"
    )

    out = ref_index.extract_references_map_from_md(md_text)

    assert sorted(out.keys()) == [10, 11]
    assert out[10].endswith("pp. 9-")
    assert out[11].startswith("[11] B. Author.")


def test_extract_references_map_recovers_unheaded_references_before_methods():
    body = "\n".join(f"Main result paragraph {idx} with Fig. {idx}." for idx in range(70))
    refs = "\n".join(
        f"{idx}. Author, A. et al. Reference title {idx}. Nat. Photon. {10 + idx}, {100 + idx}-{110 + idx} (20{idx:02d})."
        for idx in range(1, 13)
    )
    md_text = f"# Demo\n\n{body}\n\n{refs}\n\n## Methods\n\nExperimental details."

    out = ref_index.extract_references_map_from_md(md_text)

    assert sorted(out.keys()) == list(range(1, 13))
    assert "Reference title 1" in str(out.get(1) or "")
    assert "Reference title 12" in str(out.get(12) or "")


def test_extract_references_map_collects_a_later_data_reference_section():
    md_text = """# Demo

## References
[1] A. Author. First source. Journal of Tests, 2020.
[2] B. Author. Second source. Journal of Tests, 2021.

## Methods
Experimental details and body citations [1].

## Data availability
The dataset is archived online (ref. 3).

## References
[3] C. Author. Data from the demo study. Zenodo https://doi.org/10.5281/zenodo.12345 (2025).

## Acknowledgements
Thanks to the research team.
"""

    out = ref_index.extract_references_map_from_md(md_text)

    assert sorted(out) == [1, 2, 3]
    assert "zenodo.12345" in out[3]
    assert "Acknowledgements" not in out[3]


def test_build_reference_catalog_from_md_marks_gapped_tail_and_confidence():
    md_text = (
        "# Demo\n\n"
        "## References\n"
        "[1] A. Author. First real reference entry. Journal of Testing, 2020.\n"
        "[3] C. Author. Third real reference entry. Conference on Validation, 2022.\n"
    )

    catalog = ref_index.build_reference_catalog_from_md(md_text, source_name="demo.en.md")

    assert str(catalog.get("tail_continuity_status") or "") == "gapped"
    assert list(catalog.get("missing_numbers") or []) == [2]
    rows = list(catalog.get("refs") or [])
    assert len(rows) == 2
    assert int(rows[0].get("reference_number") or 0) == 1
    assert float(rows[0].get("parse_confidence") or 0.0) >= 0.70


def test_author_year_references_get_internal_ids_without_rewriting_source_style():
    md_text = (
        "# Demo\n\n<!-- kb_page: 10 -->\n\n## References\n\n"
        "Tri Dao. 2023. Flashattention-2: Faster attention with better parallelism "
        "and work partitioning. In ICLR.\n"
        "<!-- kb_page: 11 -->\n"
        "Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Re. 2022. "
        "Flashattention: Fast and memory-efficient exact attention with io-awareness. NeurIPS.\n"
    )

    ref_map = ref_index.extract_references_map_from_md(md_text)
    catalog = ref_index.build_reference_catalog_from_md(md_text, source_name="demo.en.md")

    assert sorted(ref_map) == [1, 2]
    assert not ref_map[1].startswith("[1]")
    rows = list(catalog.get("refs") or [])
    assert rows[0]["reference_style"] == "author_year"
    assert rows[0]["synthetic_reference_number"] is True
    assert rows[0]["source_page"] == 10
    assert rows[0]["title"].startswith("Flashattention-2")
    assert rows[1]["source_page"] == 11


def test_reference_index_keeps_author_year_identity_and_reference_page(tmp_path, monkeypatch):
    src_root = tmp_path / "md"
    db_dir = tmp_path / "db"
    paper_dir = src_root / "modernbert"
    paper_dir.mkdir(parents=True)
    md_path = paper_dir / "modernbert.en.md"
    md_path.write_text(
        "# ModernBERT\n\n<!-- kb_page: 3 -->\n\n"
        "ModernBERT uses Flash Attention 2 (Dao, 2023) for local attention.\n\n"
        "<!-- kb_page: 10 -->\n\n## References\n\n"
        "Tri Dao. 2023. Flashattention-2: Faster attention with better parallelism "
        "and work partitioning. In ICLR.\n"
        "Jane Smith and John Doe. 2024. A second complete reference. In Testing.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ref_index, "_crossref_preflight_ok", lambda **_kwargs: False)
    monkeypatch.setattr(ref_index, "_iter_md_files", lambda *args, **kwargs: [md_path])

    ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=False,
        enable_title_lookup=False,
        crossref_time_budget_s=0,
    )
    payload = json.loads((db_dir / "references_index.json").read_text(encoding="utf-8"))
    doc = next(iter(payload["docs"].values()))
    target = doc["refs"]["1"]

    assert target["title"].startswith("Flashattention-2")
    assert target["authors"] == "Tri Dao"
    assert target["year"] == "2023"
    assert target["reference_style"] == "author_year"
    assert target["source_page"] == 10
    assert target["match_method"] == "author_year_catalog"


def test_build_reference_index_persists_reference_catalog_and_quality_fields(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    db_dir = tmp_path / "db"
    src_root.mkdir()
    db_dir.mkdir()
    md_path = src_root / "demo.en.md"
    md_path.write_text(
        (
            "# Demo\n\n"
            "## References\n"
            "[1] A. Author. First real reference entry. Journal of Testing, 2020.\n"
            "[3] C. Author. Third real reference entry. Conference on Validation, 2022.\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(ref_index, "_crossref_preflight_ok", lambda **kwargs: False)
    monkeypatch.setattr(ref_index, "_iter_md_files", lambda *args, **kwargs: [md_path])

    out = ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=False,
        enable_title_lookup=False,
    )

    assert int(out.get("docs_updated") or 0) == 1
    catalog_path = md_path.parent / ref_index.REFERENCE_CATALOG_FILE_NAME
    assert catalog_path.exists()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert str(catalog.get("tail_continuity_status") or "") == "gapped"
    assert list(catalog.get("missing_numbers") or []) == [2]

    data = ref_index.load_reference_index(db_dir)
    doc = next(iter((data.get("docs") or {}).values()))
    assert str(doc.get("reference_catalog_status") or "") == "gapped"
    assert list(doc.get("reference_catalog_missing_numbers") or []) == [2]
    ref = (doc.get("refs") or {}).get("1") or {}
    assert float(ref.get("parse_confidence") or 0.0) > 0.0
    assert str(ref.get("tail_continuity_status") or "") == "gapped"


def test_build_reference_index_keeps_non_article_entries_from_reference_section(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    db_dir = tmp_path / "db"
    src_root.mkdir()
    db_dir.mkdir()
    md_path = src_root / "demo.en.md"
    md_path.write_text(
        (
            "# Demo\n\n"
            "## References\n"
            "[1] A. Author. First real reference entry. Journal of Testing, 2020.\n"
            "[2] Wu, D. et al., Source data for \"Imaging biological tissue with high-throughput single-pixel "
            "compressive holography\", Zenodo.\n"
            "[3] Wu, D. et al. Step-by-step protocol for data acquisition and reconstruction of holographic images "
            "in high-throughput single-pixel holography, Protocol Exchange.\n"
            "[4] Raytrix GmbH. Raytrix light field cameras. https://raytrix.de/.\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(ref_index, "_crossref_preflight_ok", lambda **kwargs: False)
    monkeypatch.setattr(ref_index, "_iter_md_files", lambda *args, **kwargs: [md_path])

    stats = ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=False,
        enable_title_lookup=False,
    )

    data = ref_index.load_reference_index(db_dir)
    doc = next(iter((data.get("docs") or {}).values()))
    refs = doc.get("refs") or {}
    assert sorted(int(k) for k in refs.keys()) == [1, 2, 3, 4]
    assert "Zenodo" in str((refs.get("2") or {}).get("raw") or "")
    assert "Protocol Exchange" in str((refs.get("3") or {}).get("raw") or "")
    assert (refs.get("4") or {}).get("metadata_status") == "no_doi_expected"
    assert int(stats.get("refs_web_source_ok") or 0) == 1


def test_build_reference_index_rebuilds_incremental_doc_when_refs_lag_catalog(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    db_dir = tmp_path / "db"
    src_root.mkdir()
    db_dir.mkdir()
    md_path = src_root / "demo.en.md"
    md_path.write_text(
        (
            "# Demo\n\n"
            "## References\n"
            "[1] A. Author. First real reference entry. Journal of Testing, 2020.\n"
            "[2] Wu, D. et al., Source data for \"Imaging biological tissue with high-throughput single-pixel "
            "compressive holography\", Zenodo.\n"
            "[3] C. Author. Third real reference entry. Conference on Validation, 2022.\n"
        ),
        encoding="utf-8",
    )
    src_key = ref_index._norm_source_key(str(md_path.resolve()))
    sha1 = ref_index.compute_file_sha1(md_path)
    (db_dir / ref_index.INDEX_FILE_NAME).write_text(
        json.dumps(
            {
                "version": 1,
                "updated_at": 1,
                "doc_count": 1,
                "next_cursor": "",
                "docs": {
                    src_key: {
                        "path": str(md_path.resolve()),
                        "name": md_path.name,
                        "stem": md_path.stem,
                        "sha1": sha1,
                        "reference_catalog_status": "continuous",
                        "reference_catalog_ref_count": 3,
                        "reference_catalog_missing_numbers": [],
                        "refs": {
                            "1": {"num": 1, "raw": "[1] A. Author. First real reference entry. Journal of Testing, 2020."},
                            "3": {"num": 3, "raw": "[3] C. Author. Third real reference entry. Conference on Validation, 2022."},
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(ref_index, "_crossref_preflight_ok", lambda **kwargs: False)
    monkeypatch.setattr(ref_index, "_iter_md_files", lambda *args, **kwargs: [md_path])

    out = ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=True,
        enable_title_lookup=False,
    )

    assert int(out.get("docs_updated") or 0) == 1
    data = ref_index.load_reference_index(db_dir)
    doc = next(iter((data.get("docs") or {}).values()))
    refs = doc.get("refs") or {}
    assert sorted(int(k) for k in refs.keys()) == [1, 2, 3]
    assert "Zenodo" in str((refs.get("2") or {}).get("raw") or "")


def test_build_reference_index_rebuilds_unchanged_doc_after_parser_upgrade(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    db_dir = tmp_path / "db"
    src_root.mkdir()
    db_dir.mkdir()
    md_path = src_root / "demo.en.md"
    md_path.write_text(
        (
            "# Demo\n\n"
            "## References\n"
            "[10] A. Author. A real reference with pages 933-941. PMLR, 2017.\n"
            "[11] B. Author. The next real reference. Journal of Testing, 2024.\n"
        ),
        encoding="utf-8",
    )
    src_key = ref_index._norm_source_key(str(md_path.resolve()))
    sha1 = ref_index.compute_file_sha1(md_path)
    (db_dir / ref_index.INDEX_FILE_NAME).write_text(
        json.dumps(
            {
                "version": 1,
                "docs": {
                    src_key: {
                        "path": str(md_path.resolve()),
                        "name": md_path.name,
                        "stem": md_path.stem,
                        "sha1": sha1,
                        "reference_parser_version": ref_index.REFERENCE_PARSER_VERSION - 1,
                        "refs": {
                            "10": {"num": 10, "raw": "[10] A. Author. A real reference, pp. 933-"},
                            "941": {"num": 941, "raw": "[941] PMLR, 2017."},
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(ref_index, "_crossref_preflight_ok", lambda **kwargs: False)
    monkeypatch.setattr(ref_index, "_iter_md_files", lambda *args, **kwargs: [md_path])

    out = ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=True,
        enable_title_lookup=False,
    )

    assert int(out.get("docs_updated") or 0) == 1
    doc = next(iter((ref_index.load_reference_index(db_dir).get("docs") or {}).values()))
    assert int(doc.get("reference_parser_version") or 0) == ref_index.REFERENCE_PARSER_VERSION
    assert sorted(int(key) for key in (doc.get("refs") or {}).keys()) == [10, 11]


def test_cleanup_reference_number_noise_removes_large_gap_outlier():
    ref_map = {
        1: "[1] A",
        2: "[2] B",
        3: "[3] C",
        4: "[4] D",
        5: "[5] E",
        6: "[6] F",
        7: "[7] G",
        8: "[8] H",
        9: "[9] I",
        10: "[10] J",
        11: "[11] K",
        12: "[12] L",
        13: "[13] M",
        14: "[14] N",
        15: "[15] O",
        16: "[16] P",
        17: "[17] Q",
        18: "[18] R",
        19: "[19] S",
        20: "[20] T",
        21: "[21] U",
        22: "[22] V",
        23: "[23] W",
        24: "[24] X",
        25: "[25] Y",
        26: "[26] Z",
        27: "[27] AA",
        28: "[28] BB",
        29: "[29] CC",
        30: "[30] DD",
        31: "[31] EE",
        32: "[32] FF",
        33: "[33] GG",
        34: "[34] HH",
        35: "[35] II",
        36: "[36] JJ",
        37: "[37] KK",
        38: "[38] LL",
        39: "[39] MM",
        40: "[40] NN",
        41: "[41] OO",
        42: "[42] PP",
        43: "[43] QQ",
        44: "[44] RR",
        275: "[275] OCR noise",
        294: "[294] OCR noise",
        948: "[948] OCR noise",
        2543: "[2543] OCR noise",
    }
    cleaned = ref_index._cleanup_reference_number_noise(ref_map)
    assert 44 in cleaned
    assert 275 not in cleaned
    assert 294 not in cleaned
    assert 948 not in cleaned
    assert 2543 not in cleaned


def test_extract_query_title_prefers_quoted_segment():
    raw = (
        "[17] E. Candès, J. Romberg, and T. Tao, "
        "“Robust Uncertainty Principles: Exact Signal Reconstruction from Highly Incomplete Frequency Information,” "
        "IEEE Trans. Inf. Theory"
    )
    title = ref_index._extract_query_title(raw)
    assert title == "Robust Uncertainty Principles: Exact Signal Reconstruction from Highly Incomplete Frequency Information"


def test_fallback_title_from_raw_reference_prefers_year_title_pattern():
    raw = "[14] Gonzalez RC, Woods RE (2006) Digital image processing, 3rd edn. Prentice-Hall, Inc, Upper Saddle River"
    title = ref_index._fallback_title_from_raw_reference(raw)
    assert title == "Digital image processing, 3rd edn"


def test_fallback_reference_meta_protects_author_initial_periods():
    raw = (
        "[4] Jeffrey H. Shapiro and Robert W. Boyd. The physics of ghost imaging. "
        "Quantum Information Processing, Aug 2012."
    )

    meta = ref_index._fallback_meta_from_raw_reference(raw)

    assert meta["title"] == "The physics of ghost imaging"
    assert meta["authors"] == "Jeffrey H. Shapiro and Robert W. Boyd"
    assert meta["venue"] == "Quantum Information Processing"
    assert meta["year"] == "2012"


def test_fallback_reference_meta_extracts_initials_and_abbreviated_venue():
    raw = (
        "[11] Jiuxuan Zhao, Ashley Lyons, Jeff S. Lundeen, and Ryan W. Boyd. "
        "Ghost imaging with entangled photons. Opt. Express, 30(3):3675-3683, Jan 2022."
    )

    meta = ref_index._fallback_meta_from_raw_reference(raw)

    assert meta["title"] == "Ghost imaging with entangled photons"
    assert "Jiuxuan Zhao" in meta["authors"]
    assert meta["venue"] == "Opt. Express"
    assert meta["year"] == "2022"


def test_crossref_meta_uses_publisher_when_container_title_missing():
    meta = citation_meta._meta_from_item(
        {
            "title": ["Introduction to Optical Microscopy"],
            "author": [{"family": "Mertz", "given": "Jerome"}],
            "publisher": "Cambridge University Press",
            "issued": {"date-parts": [[2019]]},
            "DOI": "10.1017/9781108552660",
        }
    )

    assert meta["venue"] == "Cambridge University Press"
    assert meta["authors"] == "Mertz J"


def test_lookup_crossref_meta_for_entry_uses_title_lookup_without_year(monkeypatch):
    raw = (
        "[27] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. "
        "In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104-4113,"
    )

    monkeypatch.setattr(ref_index, "fetch_best_crossref_for_reference", lambda **kwargs: None)
    monkeypatch.setattr(
        ref_index,
        "fetch_best_crossref_meta",
        lambda **kwargs: {
            "title": "Structure-from-Motion Revisited",
            "authors": "Schonberger J, Frahm J",
            "venue": "2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
            "year": "2016",
            "pages": "4104-4113",
            "doi": "10.1109/cvpr.2016.445",
            "match_method": "title",
        },
    )

    meta, doi_hint = ref_index._lookup_crossref_meta_for_entry(
        raw,
        {},
        crossref_enabled=True,
        enable_title_lookup=True,
    )

    assert doi_hint == ""
    assert isinstance(meta, dict)
    assert str(meta.get("doi") or "") == "10.1109/cvpr.2016.445"
    assert str(meta.get("match_method") or "") == "title"


def test_lookup_crossref_meta_for_entry_relaxes_title_threshold_for_quoted_title(monkeypatch):
    raw = (
        "[17] E. Candès, J. Romberg, and T. Tao, "
        "“Robust Uncertainty Principles: Exact Signal Reconstruction from Highly Incomplete Frequency Information,” "
        "IEEE Trans. Inf. Theory"
    )
    seen: dict[str, object] = {}

    monkeypatch.setattr(ref_index, "fetch_best_crossref_for_reference", lambda **kwargs: None)

    def fake_fetch_best_crossref_meta(**kwargs):
        seen.update(kwargs)
        return {
            "title": "Robust Uncertainty Principles: Exact Signal Reconstruction from Highly Incomplete Frequency Information",
            "authors": "Candes E, Romberg J, Tao T",
            "venue": "IEEE Transactions on Information Theory",
            "year": "2006",
            "doi": "10.1109/TIT.2006.871582",
            "match_method": "title",
        }

    monkeypatch.setattr(ref_index, "fetch_best_crossref_meta", fake_fetch_best_crossref_meta)
    meta, _ = ref_index._lookup_crossref_meta_for_entry(
        raw,
        {"doi": {}, "bib": {}, "title": {}},
        crossref_enabled=True,
        enable_title_lookup=True,
    )

    assert isinstance(meta, dict)
    assert str(meta.get("doi") or "") == "10.1109/TIT.2006.871582"
    assert float(seen.get("min_score") or 0.0) <= 0.90
    assert str(seen.get("expected_year") or "") == ""


def test_lookup_crossref_meta_for_entry_retries_stale_none_doi_cache(monkeypatch):
    raw = "[5] Demo entry. doi:10.1000/demo-retry"
    calls = {"n": 0}

    def fake_fetch_best_crossref_meta(**kwargs):
        calls["n"] += 1
        return {
            "title": "Recovered by DOI",
            "authors": "A Demo",
            "venue": "Demo Journal",
            "year": "2022",
            "doi": "10.1000/demo-retry",
            "match_method": "doi",
        }

    monkeypatch.setattr(ref_index, "fetch_best_crossref_meta", fake_fetch_best_crossref_meta)
    monkeypatch.setattr(ref_index, "fetch_best_crossref_for_reference", lambda **kwargs: None)

    cache = {"doi": {"10.1000/demo-retry": None}, "bib": {}, "title": {}}
    meta, doi_hint = ref_index._lookup_crossref_meta_for_entry(
        raw,
        cache,
        crossref_enabled=True,
        enable_title_lookup=True,
    )

    assert doi_hint == "10.1000/demo-retry"
    assert isinstance(meta, dict)
    assert calls["n"] >= 1
    assert isinstance((cache.get("doi") or {}).get("10.1000/demo-retry"), dict)


def test_lookup_crossref_meta_for_entry_retries_stale_none_bib_cache(monkeypatch):
    raw = "[8] A. Demo, B. Demo. Robust demo imaging. IEEE Transactions on Demo, 2021."
    calls = {"n": 0}

    monkeypatch.setattr(ref_index, "fetch_best_crossref_meta", lambda **kwargs: None)

    def fake_fetch_best_crossref_for_reference(**kwargs):
        calls["n"] += 1
        return {
            "title": "Robust demo imaging",
            "authors": "A Demo, B Demo",
            "venue": "IEEE Transactions on Demo",
            "year": "2021",
            "doi": "10.1000/demo-bib",
            "match_method": "bibliographic",
        }

    monkeypatch.setattr(ref_index, "fetch_best_crossref_for_reference", fake_fetch_best_crossref_for_reference)
    key = ref_index.normalize_title_for_match(raw)[:260]
    cache = {"doi": {}, "bib": {key: None}, "title": {}}
    meta, _ = ref_index._lookup_crossref_meta_for_entry(
        raw,
        cache,
        crossref_enabled=True,
        enable_title_lookup=True,
    )

    assert isinstance(meta, dict)
    assert str(meta.get("doi") or "") == "10.1000/demo-bib"
    assert calls["n"] >= 1
    assert isinstance((cache.get("bib") or {}).get(key), dict)


def test_lookup_crossref_meta_for_entry_respects_fresh_negative_cache(monkeypatch):
    raw = '[8] A. Demo, B. Demo. "Robust demo imaging." IEEE Transactions on Demo, 2021.'
    ref_key = ref_index.normalize_title_for_match(raw)[:260]
    title_hint = ref_index._extract_query_title(raw)
    title_key = ref_index.normalize_title_for_match(title_hint)[:260]
    cache = {
        "doi": {},
        "bib": {ref_key: ref_index._crossref_cache_miss("bib_not_found")},
        "title": {title_key: ref_index._crossref_cache_miss("title_not_found")},
    }

    def fail_fetch(**kwargs):
        raise AssertionError("fresh negative cache should skip Crossref retry")

    monkeypatch.setattr(ref_index, "fetch_best_crossref_meta", fail_fetch)
    monkeypatch.setattr(ref_index, "fetch_best_crossref_for_reference", fail_fetch)

    meta, _ = ref_index._lookup_crossref_meta_for_entry(
        raw,
        cache,
        crossref_enabled=True,
        enable_title_lookup=True,
    )

    assert meta is None


def test_lookup_crossref_meta_for_entry_writes_negative_cache_on_miss(monkeypatch):
    raw = "[5] Demo entry. doi:10.1000/demo-miss"
    monkeypatch.setattr(ref_index, "fetch_best_crossref_meta", lambda **kwargs: None)
    monkeypatch.setattr(ref_index, "fetch_best_crossref_for_reference", lambda **kwargs: None)
    cache = {"doi": {}, "bib": {}, "title": {}}

    meta, doi_hint = ref_index._lookup_crossref_meta_for_entry(
        raw,
        cache,
        crossref_enabled=True,
        enable_title_lookup=False,
    )

    assert meta is None
    assert doi_hint == "10.1000/demo-miss"
    assert ref_index._is_fresh_crossref_cache_miss((cache.get("doi") or {}).get("10.1000/demo-miss"))
    assert any(ref_index._is_fresh_crossref_cache_miss(v) for v in (cache.get("bib") or {}).values())


def test_infer_source_doi_from_doc_hints_retries_stale_empty_cache(monkeypatch, tmp_path):
    md_path = tmp_path / "DemoVenue-2024-Demo Paper.en.md"
    md_path.write_text("# Demo Paper\n", encoding="utf-8")
    k = f"{ref_index.normalize_title_for_match('Demo Paper')[:220]}|2024|{ref_index.normalize_title_for_match('DemoVenue')[:120]}"
    cache = {"source_work": {k: ""}}

    monkeypatch.setattr(
        ref_index,
        "fetch_best_crossref_meta",
        lambda **kwargs: {"doi": "10.1000/demo-source"} if str(kwargs.get("query_title") or "").strip() else None,
    )

    doi = ref_index._infer_source_doi_from_doc_hints(
        md_path,
        "# Demo Paper\n",
        cache,
        crossref_enabled=True,
    )

    assert doi == "10.1000/demo-source"
    assert str((cache.get("source_work") or {}).get(k) or "") == "10.1000/demo-source"


def test_infer_source_doi_from_doc_hints_respects_fresh_negative_cache(monkeypatch, tmp_path):
    md_path = tmp_path / "DemoVenue-2024-Demo Paper.en.md"
    md_path.write_text("# Demo Paper\n", encoding="utf-8")
    k = f"{ref_index.normalize_title_for_match('Demo Paper')[:220]}|2024|{ref_index.normalize_title_for_match('DemoVenue')[:120]}"
    cache = {"source_work": {k: ref_index._crossref_cache_miss("source_work_not_found")}}

    monkeypatch.setattr(
        ref_index,
        "fetch_best_crossref_meta",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("fresh negative cache should skip source DOI lookup")),
    )

    doi = ref_index._infer_source_doi_from_doc_hints(
        md_path,
        "# Demo Paper\n",
        cache,
        crossref_enabled=True,
    )

    assert doi == ""


def test_infer_source_doi_from_doc_hints_retries_stale_negative_cache(monkeypatch, tmp_path):
    md_path = tmp_path / "DemoVenue-2024-Demo Paper.en.md"
    md_path.write_text("# Demo Paper\n", encoding="utf-8")
    k = f"{ref_index.normalize_title_for_match('Demo Paper')[:220]}|2024|{ref_index.normalize_title_for_match('DemoVenue')[:120]}"
    stale = ref_index._crossref_cache_miss("source_work_not_found")
    stale["lookup_version"] = int(ref_index.REFERENCE_LOOKUP_VERSION) - 1
    cache = {"source_work": {k: stale}}

    monkeypatch.setattr(
        ref_index,
        "fetch_best_crossref_meta",
        lambda **kwargs: {"doi": "10.1000/source", "title": "Demo Paper"},
    )

    doi = ref_index._infer_source_doi_from_doc_hints(
        md_path,
        "# Demo Paper\n",
        cache,
        crossref_enabled=True,
    )

    assert doi == "10.1000/source"


def test_load_source_reference_rows_retries_stale_empty_cache(monkeypatch):
    cache = {"source_refs": {"doi:10.1000/demo": []}}
    monkeypatch.setattr(
        ref_index,
        "fetch_crossref_references_by_doi",
        lambda doi: [
            {
                "DOI": "10.1000/ref",
                "article-title": "Ref A",
                "author": "A Author",
                "year": "2020",
            }
        ],
    )

    rows = ref_index._load_source_reference_rows(
        "10.1000/demo",
        cache,
        crossref_enabled=True,
    )

    assert isinstance(rows, list)
    assert len(rows) == 1
    cached = (cache.get("source_refs") or {}).get("doi:10.1000/demo")
    assert isinstance(cached, list)
    assert len(cached) == 1


def test_load_source_reference_rows_respects_fresh_negative_cache(monkeypatch):
    cache = {"source_refs": {"doi:10.1000/demo": ref_index._crossref_cache_miss("source_refs_empty")}}
    monkeypatch.setattr(
        ref_index,
        "fetch_crossref_references_by_doi",
        lambda doi: (_ for _ in ()).throw(AssertionError("fresh negative cache should skip source refs lookup")),
    )

    rows = ref_index._load_source_reference_rows(
        "10.1000/demo",
        cache,
        crossref_enabled=True,
    )

    assert rows == []


def test_prepare_doc_context_prefetch_does_not_fetch_source_references(tmp_path, monkeypatch):
    md_path = tmp_path / "demo.en.md"
    md_path.write_text(
        "# Demo\n\nDOI: 10.1000/demo\n\n## References\n"
        "[1] A. Author. Demo title. Demo Journal, 2020.\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        ref_index,
        "_load_source_reference_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prefetch must not hit Crossref source refs")),
    )

    out = ref_index._prepare_doc_context_prefetch(
        md_path,
        pdf_root_obj=None,
        lib_citation_meta_map={},
        crossref_enabled=True,
    )

    assert out["source_doi"] == "10.1000/demo"
    assert out["source_ref_rows"] == []


def test_source_doi_extraction_never_takes_a_cited_work_from_references() -> None:
    markdown = (
        "# Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting\n\n"
        "## Abstract\n\nNo source DOI is printed in the article front matter.\n\n"
        "## References\n\n"
        "Luong et al. Effective Approaches to Attention-based Neural Machine Translation. "
        "doi:10.18653/v1/d15-1166.\n"
    )

    assert ref_index._extract_source_doi_from_md_head(markdown) == ""


def test_source_doi_extraction_keeps_a_front_matter_doi_before_references() -> None:
    markdown = (
        "# FEDformer: Frequency Enhanced Decomposed Transformer\n\n"
        "https://doi.org/10.1201/9781003612742-2\n\n"
        "## References\n\nDOI: 10.18653/v1/d15-1166\n"
    )

    assert ref_index._extract_source_doi_from_md_head(markdown) == "10.1201/9781003612742-2"


def test_incremental_index_revalidates_source_doi_after_lookup_contract_change(
    tmp_path,
    monkeypatch,
) -> None:
    src_root = tmp_path / "src"
    db_dir = tmp_path / "db"
    src_root.mkdir()
    db_dir.mkdir()
    md_path = src_root / "informer.en.md"
    md_path.write_text(
        "# Informer\n\n## References\n\n"
        "[1] Luong et al. Effective Approaches to Attention-based Neural Machine Translation. "
        "doi:10.18653/v1/d15-1166.\n",
        encoding="utf-8",
    )
    source_key = ref_index._norm_source_key(md_path.resolve())
    previous = {
        "version": 1,
        "docs": {
            source_key: {
                "path": str(md_path.resolve()),
                "name": md_path.name,
                "stem": md_path.stem,
                "sha1": ref_index.compute_file_sha1(md_path),
                "source_doi": "10.18653/v1/d15-1166",
                "reference_lookup_version": ref_index.REFERENCE_LOOKUP_VERSION - 1,
                "reference_parser_version": ref_index.REFERENCE_PARSER_VERSION,
                "refs": {},
            }
        },
    }
    (db_dir / "references_index.json").write_text(
        json.dumps(previous),
        encoding="utf-8",
    )
    monkeypatch.setattr(ref_index, "_crossref_preflight_ok", lambda **_kwargs: False)
    monkeypatch.setattr(ref_index, "_iter_md_files", lambda *_args, **_kwargs: [md_path])

    stats = ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=True,
        enable_title_lookup=True,
    )

    document = next(iter(ref_index.load_reference_index(db_dir)["docs"].values()))
    assert document["source_doi"] == ""
    assert document["reference_lookup_version"] == ref_index.REFERENCE_LOOKUP_VERSION
    assert stats["docs_updated"] == 1


def test_infer_source_doi_from_doc_hints_prefers_heading_title_when_filename_is_truncated(monkeypatch, tmp_path):
    doc_dir = tmp_path / "NatCommun-2021-Imaging biological tissue with...pixel compressive holography"
    doc_dir.mkdir()
    md_path = doc_dir / "NatCommun-2021-Imaging biological tissue with...pixel compressive holography.en.md"
    md_path.write_text(
        "# ARTICLE\n\n## Imaging biological tissue with high-throughput single-pixel compressive holography\n",
        encoding="utf-8",
    )

    captured: list[str] = []

    def fake_fetch_best_crossref_meta(**kwargs):
        captured.append(str(kwargs.get("query_title") or ""))
        return {"doi": "10.1038/s41467-021-24990-0"}

    monkeypatch.setattr(ref_index, "fetch_best_crossref_meta", fake_fetch_best_crossref_meta)

    doi = ref_index._infer_source_doi_from_doc_hints(
        md_path,
        md_path.read_text(encoding="utf-8"),
        {},
        crossref_enabled=True,
    )

    assert doi == "10.1038/s41467-021-24990-0"
    assert captured
    assert captured[0] == "Imaging biological tissue with high-throughput single-pixel compressive holography"


def test_infer_source_doi_from_doc_hints_reads_cache_when_crossref_disabled(tmp_path, monkeypatch):
    md_path = tmp_path / "NatCommun-2021-Imaging biological tissue with...pixel compressive holography.en.md"
    md_path.write_text(
        "# Imaging biological tissue with high-throughput single-pixel compressive holography\n",
        encoding="utf-8",
    )
    key = (
        "imaging biological tissue with high throughput single pixel compressive holography"
        "|2021|natcommun"
    )
    cache = {"source_work": {key: "10.1038/s41467-021-24990-0"}}

    monkeypatch.setattr(
        ref_index,
        "fetch_best_crossref_meta",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("offline cache path must not fetch")),
    )

    doi = ref_index._infer_source_doi_from_doc_hints(
        md_path,
        md_path.read_text(encoding="utf-8"),
        cache,
        crossref_enabled=False,
    )

    assert doi == "10.1038/s41467-021-24990-0"


def test_venue_similarity_handles_compact_filename_aliases():
    assert citation_meta._venue_similarity("NatCommun", "Nature Communications") >= 0.94
    assert citation_meta._venue_similarity("SciAdv", "Science Advances") >= 0.94


def test_build_reference_index_supplements_sparse_source_reference_mapping(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    db_dir = tmp_path / "db"
    src_root.mkdir()
    md_path = src_root / "demo.en.md"
    md_path.write_text(
        "# Demo\n\n## References\n"
        "[1] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. "
        "In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104-4113.\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(ref_index, "_crossref_preflight_ok", lambda **kwargs: True)
    monkeypatch.setattr(ref_index, "_iter_md_files", lambda *args, **kwargs: [md_path])
    monkeypatch.setattr(ref_index, "_lookup_pdf_for_md_doc", lambda *args, **kwargs: None)
    monkeypatch.setattr(ref_index, "_extract_source_doi_from_md_head", lambda *args, **kwargs: "")
    monkeypatch.setattr(ref_index, "_infer_source_doi_from_doc_hints", lambda *args, **kwargs: "10.demo/source")
    monkeypatch.setattr(
        ref_index,
        "_load_source_reference_rows",
        lambda *args, **kwargs: [
            {
                "doi": "",
                "title": "",
                "venue": "2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
                "year": "2016",
                "volume": "",
                "pages": "4104-4113",
                "author": "Johannes L Schonberger; Jan-Michael Frahm",
                "unstructured": "",
                "text": "Structure-from-Motion Revisited CVPR 2016 4104-4113",
            }
        ],
    )
    monkeypatch.setattr(
        ref_index,
        "_lookup_crossref_meta_for_entry",
        lambda *args, **kwargs: (
            {
                "title": "Structure-from-Motion Revisited",
                "authors": "Schonberger J, Frahm J",
                "venue": "2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
                "year": "2016",
                "pages": "4104-4113",
                "doi": "10.1109/cvpr.2016.445",
                "match_method": "title",
            },
            "",
        ),
    )

    out = ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=False,
        enable_title_lookup=True,
    )

    assert int(out.get("refs_with_doi") or 0) == 1
    data = ref_index.load_reference_index(db_dir)
    docs = data.get("docs") or {}
    assert len(docs) == 1
    doc = next(iter(docs.values()))
    ref = (doc.get("refs") or {}).get("1") or {}
    assert str(ref.get("doi") or "") == "10.1109/cvpr.2016.445"
    assert str(ref.get("title") or "") == "Structure-from-Motion Revisited"
    assert str(ref.get("authors") or "") == "Johannes L Schonberger; Jan-Michael Frahm"
    assert "source_work_reference_order_exact" in str(ref.get("match_method") or "")
    assert "title" in str(ref.get("match_method") or "")
    assert bool(doc.get("crossref_enriched")) is True


def test_build_reference_index_counts_source_reference_with_raw_metadata_as_ready(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    db_dir = tmp_path / "db"
    src_root.mkdir()
    md_path = src_root / "demo.en.md"
    md_path.write_text(
        "# Demo\n\n## References\n"
        "[1] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. "
        "In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104-4113, 2016.\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(ref_index, "_crossref_preflight_ok", lambda **kwargs: True)
    monkeypatch.setattr(ref_index, "_iter_md_files", lambda *args, **kwargs: [md_path])
    monkeypatch.setattr(ref_index, "_lookup_pdf_for_md_doc", lambda *args, **kwargs: None)
    monkeypatch.setattr(ref_index, "_extract_source_doi_from_md_head", lambda *args, **kwargs: "")
    monkeypatch.setattr(ref_index, "_infer_source_doi_from_doc_hints", lambda *args, **kwargs: "10.demo/source")
    monkeypatch.setattr(
        ref_index,
        "_load_source_reference_rows",
        lambda *args, **kwargs: [
            {
                "doi": "",
                "title": "",
                "venue": "2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
                "year": "2016",
                "volume": "",
                "pages": "4104-4113",
                "author": "",
                "unstructured": "",
                "text": "Structure-from-Motion Revisited CVPR 2016 4104-4113",
            }
        ],
    )
    monkeypatch.setattr(ref_index, "_lookup_crossref_meta_for_entry", lambda *args, **kwargs: (None, ""))

    stats = ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=False,
        enable_title_lookup=True,
    )

    data = ref_index.load_reference_index(db_dir)
    doc = next(iter((data.get("docs") or {}).values()))
    ref = (doc.get("refs") or {}).get("1") or {}
    assert bool(doc.get("crossref_enriched")) is True
    assert ref.get("metadata_status") == "crossref_enriched"
    assert ref.get("metadata_ready") is True
    assert ref.get("metadata_action") == "none"
    assert stats["refs_metadata_ready"] == 1
    assert stats["refs_action_retry_or_source_repair"] == 0


def test_build_reference_index_incremental_rebuilds_stale_crossref_enriched_doc(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    db_dir = tmp_path / "db"
    src_root.mkdir()
    db_dir.mkdir()
    md_path = src_root / "demo.en.md"
    raw_ref = (
        "[1] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. "
        "In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104-4113, 2016."
    )
    md_path.write_text(
        "# Demo\n\n## References\n" + raw_ref + "\n",
        encoding="utf-8",
    )

    src_key = ref_index._norm_source_key(md_path.resolve())
    prev = {
        "version": 1,
        "updated_at": 0,
        "doc_count": 1,
        "next_cursor": 0,
        "docs": {
            src_key: {
                "path": str(md_path.resolve()),
                "name": md_path.name,
                "stem": md_path.stem.lower(),
                "sha1": ref_index.compute_file_sha1(md_path),
                "source_doi": "",
                "crossref_enriched": True,
                "refs": {
                    "1": {
                        "num": 1,
                        "raw": raw_ref,
                        "doi": "",
                        "doi_url": "",
                        "title": "",
                        "authors": "",
                        "venue": "",
                        "year": "",
                        "volume": "",
                        "issue": "",
                        "pages": "",
                        "crossref_ok": False,
                        "match_method": "",
                    }
                },
            }
        },
    }
    (db_dir / "references_index.json").write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")

    monkeypatch.setattr(ref_index, "_crossref_preflight_ok", lambda **kwargs: True)
    monkeypatch.setattr(ref_index, "_iter_md_files", lambda *args, **kwargs: [md_path])
    monkeypatch.setattr(ref_index, "_lookup_pdf_for_md_doc", lambda *args, **kwargs: None)
    monkeypatch.setattr(ref_index, "_extract_source_doi_from_md_head", lambda *args, **kwargs: "")
    monkeypatch.setattr(ref_index, "_infer_source_doi_from_doc_hints", lambda *args, **kwargs: "")
    monkeypatch.setattr(ref_index, "_load_source_reference_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        ref_index,
        "_lookup_crossref_meta_for_entry",
        lambda *args, **kwargs: (
            {
                "title": "Structure-from-Motion Revisited",
                "authors": "Schonberger J, Frahm J",
                "venue": "2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
                "year": "2016",
                "pages": "4104-4113",
                "doi": "10.1109/cvpr.2016.445",
                "match_method": "title",
                "crossref_ok": True,
            },
            "",
        ),
    )

    out = ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=True,
        enable_title_lookup=True,
    )

    assert int(out.get("docs_reused") or 0) == 0
    assert int(out.get("docs_updated") or 0) == 1
    data = ref_index.load_reference_index(db_dir)
    doc = next(iter((data.get("docs") or {}).values()))
    ref = (doc.get("refs") or {}).get("1") or {}
    assert str(ref.get("doi") or "") == "10.1109/cvpr.2016.445"
    assert str(ref.get("title") or "") == "Structure-from-Motion Revisited"
    assert str(ref.get("match_method") or "") == "title"


def test_build_reference_index_incremental_reuses_recent_unresolved_crossref_attempt(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    db_dir = tmp_path / "db"
    src_root.mkdir()
    db_dir.mkdir()
    md_path = src_root / "demo.en.md"
    raw_ref = (
        "[1] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. "
        "In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104-4113, 2016."
    )
    md_path.write_text("# Demo\n\n## References\n" + raw_ref + "\n", encoding="utf-8")

    src_key = ref_index._norm_source_key(md_path.resolve())
    prev = {
        "version": 1,
        "updated_at": 0,
        "doc_count": 1,
        "next_cursor": 0,
        "docs": {
            src_key: {
                "path": str(md_path.resolve()),
                "name": md_path.name,
                "stem": md_path.stem.lower(),
                "sha1": ref_index.compute_file_sha1(md_path),
                "source_doi": "",
                "crossref_enriched": False,
                "crossref_last_attempt_at": time.time(),
                "crossref_unresolved_promising": 1,
                "crossref_sparse_promising": 0,
                "crossref_retry_ttl_s": 24 * 60 * 60,
                "reference_lookup_version": ref_index.REFERENCE_LOOKUP_VERSION,
                "reference_parser_version": ref_index.REFERENCE_PARSER_VERSION,
                "index_status": "ready",
                "quality_gate": {"status": "ready", "indexable": True, "action": "none"},
                "refs": {
                    "1": {
                        "num": 1,
                        "raw": raw_ref,
                        "doi": "",
                        "doi_url": "",
                        "title": "",
                        "authors": "",
                        "venue": "",
                        "year": "",
                        "volume": "",
                        "issue": "",
                        "pages": "",
                        "crossref_ok": False,
                        "match_method": "",
                    }
                },
            }
        },
    }
    (db_dir / "references_index.json").write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")

    def fail_lookup(*args, **kwargs):
        raise AssertionError("recent unresolved references should be reused until retry cooldown expires")

    monkeypatch.setattr(ref_index, "_crossref_preflight_ok", lambda **kwargs: True)
    monkeypatch.setattr(ref_index, "_iter_md_files", lambda *args, **kwargs: [md_path])
    monkeypatch.setattr(ref_index, "_lookup_crossref_meta_for_entry", fail_lookup)

    out = ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=True,
        enable_title_lookup=True,
        quality_gate=True,
    )

    assert int(out.get("docs_reused") or 0) == 1
    assert int(out.get("docs_updated") or 0) == 0
    assert int(out.get("docs_retry_suppressed") or 0) == 1
    assert int(out.get("crossref_network_attempts") or 0) == 0
    assert int(out.get("refs_missing_doi") or 0) == 1


def test_build_reference_index_retries_recent_unresolved_when_lookup_version_changes(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    db_dir = tmp_path / "db"
    src_root.mkdir()
    db_dir.mkdir()
    md_path = src_root / "demo.en.md"
    raw_ref = (
        "[1] T. B. Pittman, Y. H. Shih, D. V. Strekalov, and A. V. Sergienko. "
        "Optical imaging by means of two-photon quantum entanglement. Phys. Rev. A, "
        "52:R3429-R3432, 1995."
    )
    md_path.write_text("# Demo\n\n## References\n" + raw_ref + "\n", encoding="utf-8")

    src_key = ref_index._norm_source_key(md_path.resolve())
    prev = {
        "version": 1,
        "updated_at": 0,
        "doc_count": 1,
        "next_cursor": 0,
        "docs": {
            src_key: {
                "path": str(md_path.resolve()),
                "name": md_path.name,
                "stem": md_path.stem.lower(),
                "sha1": ref_index.compute_file_sha1(md_path),
                "source_doi": "",
                "crossref_enriched": False,
                "crossref_last_attempt_at": time.time(),
                "crossref_unresolved_promising": 1,
                "crossref_sparse_promising": 0,
                "crossref_retry_ttl_s": 24 * 60 * 60,
                "index_status": "ready",
                "quality_gate": {"status": "ready", "indexable": True, "action": "none"},
                "refs": {
                    "1": {
                        "num": 1,
                        "raw": raw_ref,
                        "doi": "",
                        "doi_url": "",
                        "title": "",
                        "authors": "",
                        "venue": "",
                        "year": "",
                        "volume": "",
                        "issue": "",
                        "pages": "",
                        "crossref_ok": False,
                        "match_method": "",
                    }
                },
            }
        },
    }
    (db_dir / "references_index.json").write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")
    calls: list[str] = []

    def fake_lookup(entry, *_args, **_kwargs):
        calls.append(str(entry))
        return (
            {
                "title": "Optical imaging by means of two-photon quantum entanglement",
                "authors": "T. B. Pittman, Y. H. Shih, D. V. Strekalov, et al",
                "venue": "Physical Review A",
                "year": "1995",
                "doi": "10.1103/PhysRevA.52.R3429",
                "match_method": "openalex_title",
                "match_score": 0.98,
            },
            "",
        )

    monkeypatch.setattr(ref_index, "_crossref_preflight_ok", lambda **kwargs: True)
    monkeypatch.setattr(ref_index, "_iter_md_files", lambda *args, **kwargs: [md_path])
    monkeypatch.setattr(ref_index, "_lookup_crossref_meta_for_entry", fake_lookup)

    out = ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=True,
        enable_title_lookup=True,
        quality_gate=True,
    )

    assert calls
    assert int(out.get("docs_updated") or 0) == 1
    assert int(out.get("refs_metadata_status_crossref_enriched") or 0) == 1
    data = ref_index.load_reference_index(db_dir)
    doc = next(iter((data.get("docs") or {}).values()))
    assert int(doc.get("reference_lookup_version") or 0) == ref_index.REFERENCE_LOOKUP_VERSION
    ref = (doc.get("refs") or {}).get("1") or {}
    assert ref.get("crossref_ok") is True
    assert "openalex_title" in str(ref.get("match_method") or "")


def test_build_reference_index_incremental_hydrates_recent_unresolved_from_crossref_cache(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    db_dir = tmp_path / "db"
    src_root.mkdir()
    db_dir.mkdir()
    md_path = src_root / "demo.en.md"
    raw_ref = (
        "[1] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. "
        "In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104-4113, 2016."
    )
    md_path.write_text("# Demo\n\n## References\n" + raw_ref + "\n", encoding="utf-8")

    src_key = ref_index._norm_source_key(md_path.resolve())
    prev = {
        "version": 1,
        "updated_at": 0,
        "doc_count": 1,
        "next_cursor": 0,
        "docs": {
            src_key: {
                "path": str(md_path.resolve()),
                "name": md_path.name,
                "stem": md_path.stem.lower(),
                "sha1": ref_index.compute_file_sha1(md_path),
                "source_doi": "",
                "crossref_enriched": False,
                "crossref_last_attempt_at": time.time(),
                "crossref_unresolved_promising": 1,
                "crossref_sparse_promising": 0,
                "crossref_retry_ttl_s": 24 * 60 * 60,
                "reference_lookup_version": ref_index.REFERENCE_LOOKUP_VERSION,
                "reference_parser_version": ref_index.REFERENCE_PARSER_VERSION,
                "index_status": "ready",
                "quality_gate": {"status": "ready", "indexable": True, "action": "none"},
                "refs": {
                    "1": {
                        "num": 1,
                        "raw": raw_ref,
                        "doi": "",
                        "doi_url": "",
                        "title": "",
                        "authors": "",
                        "venue": "",
                        "year": "",
                        "volume": "",
                        "issue": "",
                        "pages": "",
                        "crossref_ok": False,
                        "match_method": "",
                    }
                },
            }
        },
    }
    (db_dir / "references_index.json").write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")
    ref_key = citation_meta.normalize_title_for_match(raw_ref)[:260]
    (db_dir / "crossref_cache.json").write_text(
        json.dumps(
            {
                "version": 1,
                "doi": {},
                "bib": {
                    ref_key: {
                        "title": "Structure-from-Motion Revisited",
                        "authors": "Schonberger J, Frahm J",
                        "venue": "IEEE Conference on Computer Vision and Pattern Recognition",
                        "year": "2016",
                        "pages": "4104-4113",
                        "doi": "10.1109/CVPR.2016.445",
                    }
                },
                "title": {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    def fail_lookup(*args, **kwargs):
        raise AssertionError("cache hydration must not perform network lookup")

    monkeypatch.setattr(ref_index, "_crossref_preflight_ok", lambda **kwargs: True)
    monkeypatch.setattr(ref_index, "_iter_md_files", lambda *args, **kwargs: [md_path])
    monkeypatch.setattr(ref_index, "_lookup_crossref_meta_for_entry", fail_lookup)

    out = ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=True,
        enable_title_lookup=True,
        quality_gate=True,
    )

    assert int(out.get("docs_updated") or 0) == 1
    assert int(out.get("docs_cache_hydrated") or 0) == 1
    assert int(out.get("refs_cache_hydrated") or 0) == 1
    assert int(out.get("crossref_network_attempts") or 0) == 0
    assert int(out.get("refs_metadata_ready") or 0) == 1
    assert int(out.get("refs_metadata_status_crossref_enriched") or 0) == 1
    assert int(out.get("refs_action_retry_or_source_repair") or 0) == 0
    data = ref_index.load_reference_index(db_dir)
    doc = next(iter((data.get("docs") or {}).values()))
    ref = (doc.get("refs") or {}).get("1") or {}
    assert str(ref.get("doi") or "").lower() == "10.1109/cvpr.2016.445"
    assert str(ref.get("title") or "") == "Structure-from-Motion Revisited"
    assert ref.get("metadata_ready") is True


def test_build_reference_index_incremental_reuses_sparse_but_resolved_doc(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    db_dir = tmp_path / "db"
    src_root.mkdir()
    db_dir.mkdir()
    md_path = src_root / "demo.en.md"
    raw_ref = (
        "[1] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. "
        "In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104-4113, 2016."
    )
    md_path.write_text("# Demo\n\n## References\n" + raw_ref + "\n", encoding="utf-8")

    src_key = ref_index._norm_source_key(md_path.resolve())
    prev = {
        "version": 1,
        "updated_at": 0,
        "doc_count": 1,
        "next_cursor": 0,
        "docs": {
            src_key: {
                "path": str(md_path.resolve()),
                "name": md_path.name,
                "stem": md_path.stem.lower(),
                "sha1": ref_index.compute_file_sha1(md_path),
                "source_doi": "",
                "crossref_enriched": False,
                "reference_parser_version": ref_index.REFERENCE_PARSER_VERSION,
                "refs": {
                    "1": {
                        "num": 1,
                        "raw": raw_ref,
                        "doi": "10.1109/cvpr.2016.445",
                        "doi_url": "https://doi.org/10.1109/cvpr.2016.445",
                        "title": "",
                        "authors": "",
                        "venue": "CVPR",
                        "year": "2016",
                        "volume": "",
                        "issue": "",
                        "pages": "4104-4113",
                        "crossref_ok": True,
                        "match_method": "title",
                    }
                },
            }
        },
    }
    (db_dir / "references_index.json").write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")

    monkeypatch.setattr(ref_index, "_crossref_preflight_ok", lambda **kwargs: True)
    monkeypatch.setattr(ref_index, "_iter_md_files", lambda *args, **kwargs: [md_path])
    monkeypatch.setattr(ref_index, "_lookup_pdf_for_md_doc", lambda *args, **kwargs: None)
    monkeypatch.setattr(ref_index, "_extract_source_doi_from_md_head", lambda *args, **kwargs: "")
    monkeypatch.setattr(ref_index, "_infer_source_doi_from_doc_hints", lambda *args, **kwargs: "")
    monkeypatch.setattr(ref_index, "_load_source_reference_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(ref_index, "_lookup_crossref_meta_for_entry", lambda *args, **kwargs: (None, ""))
    progress_events: list[dict] = []

    out = ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=True,
        enable_title_lookup=True,
        progress_cb=progress_events.append,
    )

    assert int(out.get("docs_reused") or 0) == 1
    assert int(out.get("docs_updated") or 0) == 0
    assert int(out.get("refs_metadata_ready") or 0) == 0
    assert int(out.get("refs_metadata_user_ready") or 0) == 1
    assert int(out.get("refs_action_auto_backfill") or 0) == 1
    final_progress_stats = progress_events[-1]["stats"]
    assert int(final_progress_stats.get("refs_total") or 0) == 1
    assert int(final_progress_stats.get("refs_metadata_ready") or 0) == 0
    assert int(final_progress_stats.get("refs_metadata_user_ready") or 0) == 1


def test_prefer_previous_doc_refs_when_new_is_worse():
    prev_refs = {
        "1": {
            "doi": "10.1000/demo",
            "crossref_ok": True,
            "title": "Good title",
            "authors": "A Demo",
        }
    }
    new_refs = {
        "1": {
            "doi": "",
            "crossref_ok": False,
            "title": "",
            "authors": "",
        }
    }
    assert ref_index._prefer_previous_doc_refs(prev_refs, new_refs) is True


def test_build_reference_index_incremental_keeps_previous_doc_when_rebuild_is_worse(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    db_dir = tmp_path / "db"
    src_root.mkdir()
    db_dir.mkdir()
    md_path = src_root / "demo.en.md"
    raw_ref = (
        "[1] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. "
        "In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104-4113, 2016."
    )
    md_path.write_text("# Demo\n\n## References\n" + raw_ref + "\n", encoding="utf-8")

    src_key = ref_index._norm_source_key(md_path.resolve())
    prev = {
        "version": 1,
        "updated_at": 0,
        "doc_count": 1,
        "next_cursor": 0,
        "docs": {
            src_key: {
                "path": str(md_path.resolve()),
                "name": md_path.name,
                "stem": md_path.stem.lower(),
                "sha1": ref_index.compute_file_sha1(md_path),
                "source_doi": "",
                "crossref_enriched": False,
                "reference_parser_version": ref_index.REFERENCE_PARSER_VERSION,
                "refs": {
                    "1": {
                        "num": 1,
                        "raw": raw_ref,
                        "doi": "",
                        "doi_url": "",
                        "title": "Structure-from-Motion Revisited",
                        "authors": "Schonberger J, Frahm J",
                        "venue": "CVPR",
                        "year": "2016",
                        "volume": "",
                        "issue": "",
                        "pages": "4104-4113",
                        "crossref_ok": False,
                        "match_method": "manual",
                    }
                },
            }
        },
    }
    (db_dir / "references_index.json").write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")

    monkeypatch.setattr(ref_index, "_crossref_preflight_ok", lambda **kwargs: True)
    monkeypatch.setattr(ref_index, "_iter_md_files", lambda *args, **kwargs: [md_path])
    monkeypatch.setattr(ref_index, "_lookup_pdf_for_md_doc", lambda *args, **kwargs: None)
    monkeypatch.setattr(ref_index, "_extract_source_doi_from_md_head", lambda *args, **kwargs: "")
    monkeypatch.setattr(ref_index, "_infer_source_doi_from_doc_hints", lambda *args, **kwargs: "")
    monkeypatch.setattr(ref_index, "_load_source_reference_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(ref_index, "_lookup_crossref_meta_for_entry", lambda *args, **kwargs: (None, ""))

    out = ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=True,
        enable_title_lookup=True,
    )

    assert int(out.get("docs_reused") or 0) == 1
    assert int(out.get("docs_updated") or 0) == 0
    data = ref_index.load_reference_index(db_dir)
    doc = next(iter((data.get("docs") or {}).values()))
    ref = (doc.get("refs") or {}).get("1") or {}
    assert str(ref.get("title") or "") == "Structure-from-Motion Revisited"


def test_build_reference_index_falls_back_to_raw_title_when_meta_has_no_title(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    db_dir = tmp_path / "db"
    src_root.mkdir()
    md_path = src_root / "demo.en.md"
    md_path.write_text(
        "# Demo\n\n## References\n"
        "[1] Gonzalez RC, Woods RE (2006) Digital image processing, 3rd edn. Prentice-Hall, Inc, Upper Saddle River\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(ref_index, "_crossref_preflight_ok", lambda **kwargs: True)
    monkeypatch.setattr(ref_index, "_iter_md_files", lambda *args, **kwargs: [md_path])
    monkeypatch.setattr(ref_index, "_lookup_pdf_for_md_doc", lambda *args, **kwargs: None)
    monkeypatch.setattr(ref_index, "_extract_source_doi_from_md_head", lambda *args, **kwargs: "")
    monkeypatch.setattr(ref_index, "_infer_source_doi_from_doc_hints", lambda *args, **kwargs: "")
    monkeypatch.setattr(ref_index, "_load_source_reference_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        ref_index,
        "_lookup_crossref_meta_for_entry",
        lambda *args, **kwargs: (
            {
                "title": "",
                "authors": "RC Gonzalez; RE Woods",
                "venue": "",
                "year": "2006",
                "doi": "",
                "match_method": "bibliographic",
            },
            "",
        ),
    )

    ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=False,
        enable_title_lookup=True,
    )
    data = ref_index.load_reference_index(db_dir)
    doc = next(iter((data.get("docs") or {}).values()))
    ref = (doc.get("refs") or {}).get("1") or {}
    assert str(ref.get("title") or "") == "Digital image processing, 3rd edn"
    assert "raw_title" in str(ref.get("match_method") or "")


def test_build_reference_index_fills_sparse_meta_from_raw_reference(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    db_dir = tmp_path / "db"
    src_root.mkdir()
    md_path = src_root / "demo.en.md"
    md_path.write_text(
        "# Demo\n\n## References\n"
        "[1] Jeffrey H. Shapiro and Robert W. Boyd. The physics of ghost imaging. "
        "Quantum Information Processing, Aug 2012.\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(ref_index, "_crossref_preflight_ok", lambda **kwargs: False)
    monkeypatch.setattr(ref_index, "_iter_md_files", lambda *args, **kwargs: [md_path])
    monkeypatch.setattr(ref_index, "_lookup_pdf_for_md_doc", lambda *args, **kwargs: None)
    monkeypatch.setattr(ref_index, "_extract_source_doi_from_md_head", lambda *args, **kwargs: "")
    monkeypatch.setattr(ref_index, "_infer_source_doi_from_doc_hints", lambda *args, **kwargs: "")
    monkeypatch.setattr(ref_index, "_load_source_reference_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(ref_index, "_lookup_crossref_meta_for_entry", lambda *args, **kwargs: (None, ""))

    ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=False,
        enable_title_lookup=True,
    )
    data = ref_index.load_reference_index(db_dir)
    doc = next(iter((data.get("docs") or {}).values()))
    ref = (doc.get("refs") or {}).get("1") or {}
    assert str(ref.get("title") or "") == "The physics of ghost imaging"
    assert str(ref.get("authors") or "") == "Jeffrey H. Shapiro and Robert W. Boyd"
    assert str(ref.get("venue") or "") == "Quantum Information Processing"
    assert "raw_meta" in str(ref.get("match_method") or "")


def test_build_reference_index_uses_cached_doi_backfill_when_crossref_offline(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    db_dir = tmp_path / "db"
    src_root.mkdir()
    db_dir.mkdir()
    md_path = src_root / "demo.en.md"
    md_path.write_text(
        "# Demo\n\n## References\n"
        "[1] A. Author. Demo reference with DOI. Demo Journal.\n",
        encoding="utf-8",
    )

    cache_data = {
        "version": 1,
        "updated_at": 0,
        "doi": {
            "10.1234/demo": {
                "title": "Recovered Title From Cached DOI",
                "authors": "A Author",
                "venue": "Demo Journal",
                "year": "2020",
                "volume": "10",
                "issue": "2",
                "pages": "1-10",
                "doi": "10.1234/demo",
                "match_method": "doi",
            }
        },
        "bib": {},
        "source_refs": {},
        "source_work": {},
        "title": {},
    }
    (db_dir / "crossref_cache.json").write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")

    monkeypatch.setattr(ref_index, "_crossref_preflight_ok", lambda **kwargs: False)
    monkeypatch.setattr(ref_index, "_iter_md_files", lambda *args, **kwargs: [md_path])
    monkeypatch.setattr(ref_index, "_lookup_pdf_for_md_doc", lambda *args, **kwargs: None)
    monkeypatch.setattr(ref_index, "_extract_source_doi_from_md_head", lambda *args, **kwargs: "")
    monkeypatch.setattr(ref_index, "_infer_source_doi_from_doc_hints", lambda *args, **kwargs: "")
    monkeypatch.setattr(ref_index, "_load_source_reference_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        ref_index,
        "_lookup_crossref_meta_for_entry",
        lambda *args, **kwargs: (
            {
                "title": "",
                "authors": "",
                "venue": "",
                "year": "",
                "volume": "",
                "issue": "",
                "pages": "",
                "doi": "10.1234/demo",
                "match_method": "bibliographic",
            },
            "10.1234/demo",
        ),
    )

    ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=False,
        enable_title_lookup=True,
    )
    data = ref_index.load_reference_index(db_dir)
    doc = next(iter((data.get("docs") or {}).values()))
    ref = (doc.get("refs") or {}).get("1") or {}
    assert str(ref.get("doi") or "") == "10.1234/demo"
    assert str(ref.get("title") or "") == "Recovered Title From Cached DOI"
    assert "doi_backfill" in str(ref.get("match_method") or "")


def test_build_reference_index_refreshes_sparse_cached_doi_meta(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    db_dir = tmp_path / "db"
    src_root.mkdir()
    db_dir.mkdir()
    md_path = src_root / "demo.en.md"
    md_path.write_text(
        "# Demo\n\n## References\n"
        "[1] Mertz, J. Introduction to Optical Microscopy 2nd edn (Cambridge Univ. Press, 2019). "
        "doi:10.1017/9781108552660\n",
        encoding="utf-8",
    )

    cache_data = {
        "version": 1,
        "updated_at": 0,
        "doi": {
            "10.1017/9781108552660": {
                "title": "Introduction to Optical Microscopy",
                "authors": "Mertz J",
                "venue": "",
                "year": "2019",
                "doi": "10.1017/9781108552660",
                "match_method": "doi",
            }
        },
        "bib": {},
        "source_refs": {},
        "source_work": {},
        "title": {},
    }
    (db_dir / "crossref_cache.json").write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")

    monkeypatch.setattr(ref_index, "_crossref_preflight_ok", lambda **kwargs: True)
    monkeypatch.setattr(ref_index, "_iter_md_files", lambda *args, **kwargs: [md_path])
    monkeypatch.setattr(ref_index, "_lookup_pdf_for_md_doc", lambda *args, **kwargs: None)
    monkeypatch.setattr(ref_index, "_extract_source_doi_from_md_head", lambda *args, **kwargs: "")
    monkeypatch.setattr(ref_index, "_infer_source_doi_from_doc_hints", lambda *args, **kwargs: "")
    monkeypatch.setattr(ref_index, "_load_source_reference_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        ref_index,
        "_lookup_crossref_meta_for_entry",
        lambda *args, **kwargs: (
            {
                "title": "Introduction to Optical Microscopy",
                "authors": "Mertz J",
                "venue": "",
                "year": "2019",
                "doi": "10.1017/9781108552660",
                "match_method": "bibliographic",
            },
            "10.1017/9781108552660",
        ),
    )
    monkeypatch.setattr(
        ref_index,
        "fetch_best_crossref_meta",
        lambda **kwargs: {
            "title": "Introduction to Optical Microscopy",
            "authors": "Mertz J",
            "venue": "Cambridge University Press",
            "year": "2019",
            "doi": "10.1017/9781108552660",
            "match_method": "doi",
        },
    )

    ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=False,
        enable_title_lookup=True,
    )
    data = ref_index.load_reference_index(db_dir)
    doc = next(iter((data.get("docs") or {}).values()))
    ref = (doc.get("refs") or {}).get("1") or {}
    assert str(ref.get("venue") or "") == "Cambridge University Press"
    cache = json.loads((db_dir / "crossref_cache.json").read_text(encoding="utf-8"))
    assert cache["doi"]["10.1017/9781108552660"]["venue"] == "Cambridge University Press"


def test_prefetch_doi_meta_parallel_populates_cache_with_dedup(monkeypatch):
    cache = {"doi": {}, "bib": {}, "title": {}, "source_refs": {}, "source_work": {}}
    ref_map = {
        1: "[1] A. Demo. X. doi:10.1000/demo1",
        2: "[2] B. Demo. Y. doi:10.1000/demo2",
        3: "[3] C. Demo. Z. doi:10.1000/demo1",
    }
    calls: list[str] = []

    def fake_fetch_best_crossref_meta(**kwargs):
        d = str(kwargs.get("doi_hint") or "").strip()
        calls.append(d)
        return {
            "title": f"title-{d}",
            "authors": "A Demo",
            "venue": "Demo Journal",
            "year": "2020",
            "doi": d,
            "match_method": "doi",
        }

    monkeypatch.setattr(ref_index, "fetch_best_crossref_meta", fake_fetch_best_crossref_meta)
    done = ref_index._prefetch_doi_meta_parallel(
        ref_map,
        cache,
        crossref_enabled=True,
        max_workers=4,
        max_prefetch=10,
    )

    doi_cache = cache.get("doi") or {}
    assert int(done) == 2
    assert "10.1000/demo1" in doi_cache
    assert "10.1000/demo2" in doi_cache
    assert len(calls) == 2


def test_prefetch_doi_meta_parallel_skips_when_single_worker(monkeypatch):
    cache = {"doi": {}, "bib": {}, "title": {}, "source_refs": {}, "source_work": {}}
    ref_map = {1: "[1] doi:10.1000/demo1"}
    monkeypatch.setattr(ref_index, "fetch_best_crossref_meta", lambda **kwargs: {"doi": "10.1000/demo1"})

    done = ref_index._prefetch_doi_meta_parallel(
        ref_map,
        cache,
        crossref_enabled=True,
        max_workers=1,
        max_prefetch=10,
    )
    assert int(done) == 0
    assert not (cache.get("doi") or {})


def test_prefetch_reference_meta_parallel_populates_bib_and_title_cache(monkeypatch):
    cache = {"doi": {}, "bib": {}, "title": {}, "source_refs": {}, "source_work": {}}
    ref_map = {
        1: '[1] A. Author, B. Author. "Super Resolution by Coded Imaging". IEEE Transactions on Image Processing, 2020.',
        2: '[2] C. Author, D. Author. "Neural Phase Retrieval with Priors". Proceedings of CVPR, 2021.',
        3: "[3] E. Author, F. Author. Fast compressive calibration. Journal of Optics, 2019.",
    }
    bib_calls: list[str] = []
    title_calls: list[str] = []

    def fake_fetch_best_crossref_for_reference(**kwargs):
        raw = str(kwargs.get("reference_text") or "")
        bib_calls.append(raw)
        if "Fast compressive calibration" in raw:
            return {
                "title": "Fast compressive calibration",
                "authors": "E Author, F Author",
                "venue": "Journal of Optics",
                "year": "2019",
                "doi": "10.1000/jopt.2019.1",
                "match_method": "bibliographic",
            }
        return None

    def fake_fetch_best_crossref_meta(**kwargs):
        title_calls.append(str(kwargs.get("query_title") or ""))
        title = str(kwargs.get("query_title") or "").strip()
        return {
            "title": title or "Recovered Title",
            "authors": "Recovered Author",
            "venue": "Recovered Venue",
            "year": "2021",
            "doi": "10.1000/recovered",
            "match_method": "title",
        }

    monkeypatch.setattr(ref_index, "fetch_best_crossref_for_reference", fake_fetch_best_crossref_for_reference)
    monkeypatch.setattr(ref_index, "fetch_best_crossref_meta", fake_fetch_best_crossref_meta)

    done = ref_index._prefetch_reference_meta_parallel(
        ref_map,
        cache,
        crossref_enabled=True,
        enable_title_lookup=True,
        max_workers=4,
        max_prefetch=10,
    )

    assert int(done) >= 3
    assert len(bib_calls) == 3
    assert len(title_calls) >= 2
    bib_values = list((cache.get("bib") or {}).values())
    assert sum(1 for v in bib_values if ref_index._is_crossref_meta_cache_hit(v)) == 1
    assert sum(1 for v in bib_values if ref_index._is_fresh_crossref_cache_miss(v)) == 2
    assert any(ref_index._is_crossref_meta_cache_hit(v) for v in (cache.get("title") or {}).values())


def test_prefetch_reference_meta_parallel_caches_negative_results(monkeypatch):
    cache = {"doi": {}, "bib": {}, "title": {}, "source_refs": {}, "source_work": {}}
    ref_map = {
        1: '[1] A. Author, B. Author. "Title A". IEEE Trans. Demo, 2020.',
        2: '[2] C. Author, D. Author. "Title B". IEEE Trans. Demo, 2021.',
    }
    monkeypatch.setattr(ref_index, "fetch_best_crossref_for_reference", lambda **kwargs: None)
    monkeypatch.setattr(ref_index, "fetch_best_crossref_meta", lambda **kwargs: None)

    done = ref_index._prefetch_reference_meta_parallel(
        ref_map,
        cache,
        crossref_enabled=True,
        enable_title_lookup=True,
        max_workers=4,
        max_prefetch=10,
    )

    assert int(done) == 0
    assert all(ref_index._is_fresh_crossref_cache_miss(v) for v in (cache.get("bib") or {}).values())
    assert all(ref_index._is_fresh_crossref_cache_miss(v) for v in (cache.get("title") or {}).values())

    monkeypatch.setattr(
        ref_index,
        "fetch_best_crossref_for_reference",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("fresh bib miss should skip retry")),
    )
    monkeypatch.setattr(
        ref_index,
        "fetch_best_crossref_meta",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("fresh title miss should skip retry")),
    )
    done_again = ref_index._prefetch_reference_meta_parallel(
        ref_map,
        cache,
        crossref_enabled=True,
        enable_title_lookup=True,
        max_workers=4,
        max_prefetch=10,
    )
    assert int(done_again) == 0


def test_build_reference_index_skips_order_mapping_when_source_rows_conflict(tmp_path, monkeypatch):
    src_root = tmp_path / "src"
    db_dir = tmp_path / "db"
    src_root.mkdir()
    md_path = src_root / "demo.en.md"
    md_path.write_text(
        "# Demo\n\n## References\n"
        "[1] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104-4113, 2016.\n"
        "[2] Mu Qiao, Ziyi Meng, Jiawei Ma, and Xin Yuan. Deep learning for video compressive sensing. APL Photonics, 5(3), 2020.\n"
        "[3] Lishun Wang, Miao Cao, Yong Zhong, and Xin Yuan. Spatial-temporal transformer for video snapshot compressive imaging. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(7):9072-9089, 2022.\n"
        "[4] Patrick Llull, Xuejun Liao, Xin Yuan, Jianbo Yang, David Kittle, Lawrence Carin, Guillermo Sapiro, and David J Brady. Coded aperture compressive temporal imaging. Optics Express, 21(9):10526-10545, 2013.\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(ref_index, "_crossref_preflight_ok", lambda **kwargs: True)
    monkeypatch.setattr(ref_index, "_iter_md_files", lambda *args, **kwargs: [md_path])
    monkeypatch.setattr(ref_index, "_lookup_pdf_for_md_doc", lambda *args, **kwargs: None)
    monkeypatch.setattr(ref_index, "_extract_source_doi_from_md_head", lambda *args, **kwargs: "")
    monkeypatch.setattr(ref_index, "_infer_source_doi_from_doc_hints", lambda *args, **kwargs: "10.demo/source")
    monkeypatch.setattr(
        ref_index,
        "_load_source_reference_rows",
        lambda *args, **kwargs: [
            {"text": "Brida G Genovese M Experimental realization of sub-shot-noise quantum imaging Nature Photonics 2010 4 227-230", "author": "Brida G, Genovese M", "year": "2010", "doi": "10.1038/nphoton.2010.29"},
            {"text": "Shapiro J Boyd The physics of ghost imaging Quantum Information Processing 2012", "author": "Shapiro J, Boyd R", "year": "2012", "doi": "10.1007/s11128-012-0416-4"},
            {"text": "Levoy M Ng R Adams A Light field microscopy ACM Transactions on Graphics 2006 25 3 924-934", "author": "Levoy M, Ng R, Adams A", "year": "2006", "doi": "10.1145/1141911.1141976"},
            {"text": "Orth A Crozier K Microscopy with microlens arrays high throughput high resolution imaging Optics Express 2012 20 12 13522-13531", "author": "Orth A, Crozier K", "year": "2012", "doi": "10.1364/OE.20.013522"},
        ],
    )
    monkeypatch.setattr(
        ref_index,
        "_lookup_crossref_meta_for_entry",
        lambda raw, *args, **kwargs: (
            {
                "title": ref_index._extract_query_title(raw),
                "authors": "Recovered Authors",
                "venue": "Recovered Venue",
                "year": "2024",
                "doi": "10.9999/recovered",
                "match_method": "title",
            },
            "",
        ),
    )

    out = ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=False,
        enable_title_lookup=True,
    )

    assert int(out.get("refs_source_map_ok") or 0) == 0
    data = ref_index.load_reference_index(db_dir)
    doc = next(iter((data.get("docs") or {}).values()))
    refs = doc.get("refs") or {}
    assert all("source_work_reference" not in str((refs.get(str(i)) or {}).get("match_method") or "") for i in range(1, 5))
