import json

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


def test_build_reference_index_marks_sparse_source_reference_docs_for_retry(tmp_path, monkeypatch):
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

    ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=False,
        enable_title_lookup=True,
    )

    data = ref_index.load_reference_index(db_dir)
    doc = next(iter((data.get("docs") or {}).values()))
    assert bool(doc.get("crossref_enriched")) is False


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

    out = ref_index.build_reference_index(
        src_root=src_root,
        db_dir=db_dir,
        incremental=True,
        enable_title_lookup=True,
    )

    assert int(out.get("docs_reused") or 0) == 1
    assert int(out.get("docs_updated") or 0) == 0


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
    assert len(cache.get("bib") or {}) == 1
    assert any(isinstance(v, dict) for v in (cache.get("title") or {}).values())


def test_prefetch_reference_meta_parallel_does_not_cache_none_results(monkeypatch):
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
    assert (cache.get("bib") or {}) == {}
    assert (cache.get("title") or {}) == {}


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
