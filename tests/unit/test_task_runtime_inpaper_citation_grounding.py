from __future__ import annotations


def test_validate_structured_citations_in_paper_guide_rewrites_using_doi_hint(monkeypatch, tmp_path):
    from kb import task_runtime

    source_path = r"db\doc\paper.en.md"
    locked_sid = task_runtime._cite_source_id(source_path)

    monkeypatch.setattr(task_runtime, "load_reference_index", lambda _db_dir: {"docs": {"demo": {}}})

    refs = {
        1: {
            "raw": "[1] Wrong ref. 2020. doi:10.1000/wrong",
            "authors": "Smith et al.",
            "year": "2020",
            "doi": "10.1000/wrong",
            "title": "Wrong Ref",
        },
        24: {
            "raw": "[24] Gehm M, Brady D. Opt Express, 2007. doi:10.1364/OE.15.014013",
            "authors": "Gehm M, Brady D",
            "year": "2007",
            "doi": "10.1364/OE.15.014013",
            "title": "Correct Ref",
        },
    }

    def fake_resolve(_index, src, ref_num, *, source_sha1=""):
        del _index, source_sha1
        if str(src) != source_path:
            return None
        ref = refs.get(int(ref_num))
        return {"ref": dict(ref)} if isinstance(ref, dict) else None

    monkeypatch.setattr(task_runtime, "resolve_reference_entry", fake_resolve)

    answer, stats = task_runtime._validate_structured_citations(
        "This follows DOI 10.1364/OE.15.014013 [[CITE:sdeadbeef:1]].",
        answer_hits=[
            {
                "text": "Evidence mentions prior work [24].",
                "meta": {
                    "source_path": source_path,
                    "source_sha1": "abc",
                },
            }
        ],
        db_dir=tmp_path,
        locked_source={
            "sid": locked_sid,
            "source_path": source_path,
            "source_sha1": "abc",
        },
        paper_guide_mode=True,
    )

    assert answer == "This follows DOI 10.1364/OE.15.014013 [[CITE:{}:24]].".format(locked_sid)
    assert stats["rewritten"] == 1
    assert stats["dropped"] == 0


def test_build_paper_guide_citation_grounding_block_formats_doc_scoped_candidates():
    from kb import task_runtime

    source_path = r"db\doc\paper.en.md"
    sid = task_runtime._cite_source_id(source_path)

    block = task_runtime._build_paper_guide_citation_grounding_block(
        [
            {
                "text": "This hit mentions prior work [24] and follow-up calibration [25].",
                "meta": {
                    "source_path": source_path,
                    "heading_path": "Results / Resolution analysis",
                    "ref_show_snippets": ["Localized support [24-25] is discussed here."],
                },
            },
            {
                "text": "No in-paper numeric citation in this hit.",
                "meta": {
                    "source_path": source_path,
                    "heading_path": "Discussion",
                },
            },
        ]
    )

    assert "Paper-guide citation grounding hints:" in block
    assert f"DOC-1 | sid={sid} | refs=24, 25" in block
    assert "heading=Results / Resolution analysis" in block
    assert "cue=" in block
    assert "DOC-2" not in block


def test_validate_structured_citations_in_paper_guide_drops_guess_when_no_candidates_or_identity_hints(monkeypatch, tmp_path):
    from kb import task_runtime

    source_path = r"db\doc\paper.en.md"
    locked_sid = task_runtime._cite_source_id(source_path)

    monkeypatch.setattr(task_runtime, "load_reference_index", lambda _db_dir: {"docs": {"demo": {}}})

    def fake_resolve(_index, src, ref_num, *, source_sha1=""):
        del _index, source_sha1
        if str(src) != source_path:
            return None
        if int(ref_num) == 8:
            return {
                "ref": {
                    "raw": "[8] Demo reference 8",
                    "authors": "Demo Author",
                    "year": "2006",
                    "title": "Demo Ref 8",
                }
            }
        return None

    monkeypatch.setattr(task_runtime, "resolve_reference_entry", fake_resolve)

    answer, stats = task_runtime._validate_structured_citations(
        "The paper improves imaging quality [[CITE:sdeadbeef:8]].",
        answer_hits=[
            {
                "text": "Abstract text with no in-paper numeric citations.",
                "meta": {
                    "source_path": source_path,
                    "source_sha1": "abc",
                },
            }
        ],
        db_dir=tmp_path,
        locked_source={
            "sid": locked_sid,
            "source_path": source_path,
            "source_sha1": "abc",
        },
        paper_guide_mode=True,
    )

    assert "[[CITE:" not in answer
    assert stats["dropped"] == 1
