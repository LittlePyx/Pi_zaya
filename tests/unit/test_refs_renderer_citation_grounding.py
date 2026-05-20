from __future__ import annotations

from ui import refs_renderer


def test_numeric_citation_is_hidden_when_local_doi_conflicts(monkeypatch):
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
                "doi": "10.1000/wrong",
                "title": "Wrong ref",
                "raw": "[24] Townsend P, Foster J. Wrong ref. 2003. doi:10.1000/wrong",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    md = "This follows DOI 10.1364/OE.15.014013 [24]."
    hits = [{"meta": {"source_path": "doc.en.md", "source_sha1": "abc"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[24](#" not in out
    assert "[24]" not in out
    assert details == []


def test_numeric_citation_uses_doi_to_resolve_multi_source_ambiguity(monkeypatch):
    def fake_resolve(_index_data, source_path, ref_num, *, source_sha1=""):
        del _index_data, source_sha1
        if int(ref_num) != 24:
            return None
        if str(source_path).endswith("a.en.md"):
            return {
                "source_path": "a.en.md",
                "source_name": "a.pdf",
                "ref_num": 24,
                "ref": {
                    "authors": "Wrong A",
                    "year": "2010",
                    "doi": "10.1000/wrong-a",
                    "raw": "[24] Wrong A. 2010. doi:10.1000/wrong-a",
                },
            }
        if str(source_path).endswith("b.en.md"):
            return {
                "source_path": "b.en.md",
                "source_name": "b.pdf",
                "ref_num": 24,
                "ref": {
                    "authors": "Gehm M, Brady D",
                    "year": "2007",
                    "doi": "10.1364/OE.15.014013",
                    "raw": "[24] Gehm M, Brady D. 2007. doi:10.1364/OE.15.014013",
                },
            }
        return None

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda sp: "a.pdf" if sp.endswith("a.en.md") else "b.pdf")

    md = "This follows DOI 10.1364/OE.15.014013 [24]."
    hits = [
        {"meta": {"source_path": "a.en.md", "source_sha1": "aaa"}},
        {"meta": {"source_path": "b.en.md", "source_sha1": "bbb"}},
    ]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[24](#" in out
    assert len(details) == 1
    assert details[0]["source_path"] == "b.en.md"


def test_numeric_citation_without_identity_signal_stays_clickable(monkeypatch):
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

    assert "[116](#" in out
    assert len(details) == 1


def test_structured_citation_is_hidden_when_context_doi_conflicts(monkeypatch):
    source_path = "doc.en.md"
    sid = refs_renderer._source_cite_id(source_path)

    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 1:
            return None
        return {
            "source_path": source_path,
            "source_name": "doc.pdf",
            "ref_num": 1,
            "ref": {
                "authors": "Wrong A",
                "year": "2020",
                "doi": "10.1000/wrong",
                "title": "Wrong Reference",
                "raw": "[1] Wrong A. Wrong Reference. 2020. doi:10.1000/wrong",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    md = f"This follows DOI 10.1364/OE.15.014013 [[CITE:{sid}:1]]."
    hits = [{"meta": {"source_path": source_path, "source_sha1": "abc"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "#kb-cite-" not in out
    assert "[1]" not in out
    assert "Wrong Reference" not in out
    assert details == []


def test_unresolved_structured_citation_does_not_fall_back_to_hit_number(monkeypatch):
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", lambda *args, **kwargs: None)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    md = "Bad structured cite [[CITE:badbad:1]] and normal evidence [1]."
    hits = [{"text": "supporting snippet", "meta": {"source_path": "doc.en.md"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[[CITE:" not in out
    assert "Bad structured cite  and normal evidence [1](#" in out
    assert len(details) == 1
    assert details[0]["is_inpaper"] is False


def test_structured_citation_detail_points_to_context_matched_reference(monkeypatch):
    source_path = "doc.en.md"
    sid = refs_renderer._source_cite_id(source_path)

    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 24:
            return None
        return {
            "source_path": source_path,
            "source_name": "doc.pdf",
            "ref_num": 24,
            "ref": {
                "authors": "Gehm M, Brady D",
                "year": "2007",
                "doi": "10.1364/OE.15.014013",
                "title": "Single-shot compressive spectral imaging with a dual-disperser architecture",
                "raw": (
                    "[24] Gehm M, Brady D. Single-shot compressive spectral imaging with "
                    "a dual-disperser architecture. Optics Express, 2007. doi:10.1364/OE.15.014013"
                ),
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    md = f"This follows DOI 10.1364/OE.15.014013 [[CITE:{sid}:24]]."
    hits = [{"meta": {"source_path": source_path, "source_sha1": "abc"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[24](#kb-cite-" in out
    assert len(details) == 1
    detail = details[0]
    assert detail["num"] == 24
    assert detail["source_path"] == source_path
    assert detail["doi"] == "10.1364/OE.15.014013"
    assert "dual-disperser architecture" in detail["title"]
    assert "Wrong Reference" not in str(detail)


def test_structured_system_b_detail_carries_answer_context_and_role(monkeypatch):
    source_path = "doc.en.md"
    sid = refs_renderer._source_cite_id(source_path)

    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 4:
            return None
        return {
            "source_path": source_path,
            "source_name": "doc.pdf",
            "ref_num": 4,
            "ref": {
                "authors": "Boyd S",
                "year": "2011",
                "title": "Distributed Optimization and Statistical Learning via ADMM",
                "raw": "[4] Boyd S. Distributed Optimization and Statistical Learning via ADMM. 2011.",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    md = f"ADMM is prior optimization machinery; open ADMM [[CITE:{sid}:4]] to follow the paper's citation trail."
    hits = [{"meta": {"source_path": source_path, "source_sha1": "abc"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[4](#kb-cite-" in out
    assert len(details) == 1
    detail = details[0]
    assert detail["is_inpaper"] is True
    assert "prior optimization machinery" in detail["answer_claim"]
    assert "prior optimization machinery" in detail["citation_context"]
    assert detail["citation_context_source"] == "answer_context"
    assert "prior optimization machinery" in detail["evidence_quote"]
    assert "prior work" in detail["upstream_work_role"].lower()
    assert "origin" in detail["user_question_relation"].lower()
    assert detail["support_relation"] == detail["user_question_relation"]
