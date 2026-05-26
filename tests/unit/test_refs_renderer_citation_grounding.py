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
    def fake_enrich(detail, *, source_path, ref_num, answer_context="", **_kwargs):
        del source_path, ref_num
        detail["answer_claim"] = answer_context
        detail["citation_context"] = "The current paper cites Gehm and Brady when tracing single-shot compressive spectral imaging."
        detail["citation_context_source"] = "source_markdown"
        detail["evidence_quote"] = detail["citation_context"]
        detail["location_label"] = "Related Work / Snapshot compressive imaging"
        return detail

    monkeypatch.setattr(refs_renderer, "enrich_inpaper_detail_context", fake_enrich)

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
    def fake_enrich(detail, *, source_path, ref_num, answer_context="", **_kwargs):
        del source_path, ref_num
        detail["answer_claim"] = answer_context
        detail["citation_context"] = "The current paper cites Wang and Li when discussing DenseNet reconstruction."
        detail["citation_context_source"] = "source_markdown"
        detail["evidence_quote"] = detail["citation_context"]
        detail["location_label"] = "Related Work / Reconstruction"
        return detail

    monkeypatch.setattr(refs_renderer, "enrich_inpaper_detail_context", fake_enrich)

    md = "Wang uses DenseNet for reconstruction [116]."
    hits = [{"meta": {"source_path": "doc.en.md", "source_sha1": "abc"}}]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[116](#" in out
    assert len(details) == 1
    assert details[0]["system_b_trace_complete"] is True


def test_numeric_reference_index_fallback_stays_plain_when_citation_context_is_missing(monkeypatch):
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


def test_structured_cite_routes_to_system_b_for_upstream_origin_context(monkeypatch):
    source_path = "scinerf.en.md"

    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 1:
            return None
        return {
            "source_path": source_path,
            "source_name": "SCINeRF.pdf",
            "ref_num": 1,
            "ref": {
                "authors": "Boyd S",
                "year": "2011",
                "title": "Distributed Optimization and Statistical Learning via ADMM",
                "raw": "[1] Boyd S. Distributed Optimization and Statistical Learning via ADMM. 2011.",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "SCINeRF.pdf")

    sid = refs_renderer._source_cite_id(source_path)
    md = f"ADMM was not invented by this paper; it comes from prior optimization work [[CITE:{sid}:1]]."
    hits = [
        {
            "text": "SCINeRF uses ADMM for optimization in its reconstruction pipeline.",
            "meta": {"source_path": source_path, "source_sha1": "abc"},
        }
    ]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[1](#kb-cite-" in out
    assert len(details) == 1
    detail = details[0]
    assert detail["is_inpaper"] is True
    assert detail["citation_route"] == "system_b"
    assert "ADMM" in " ".join([str(detail.get("title") or ""), str(detail.get("raw") or "")])
    assert detail["routing_reason"] == "structured_cite"


def test_numeric_router_keeps_good_system_a_for_generic_reference_word(monkeypatch):
    source_path = "single_pixel_review.en.md"

    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 1:
            return None
        return {
            "source_path": source_path,
            "source_name": "Review.pdf",
            "ref_num": 1,
            "ref": {
                "authors": "Unrelated A",
                "year": "2001",
                "title": "An unrelated bibliography item",
                "raw": "[1] Unrelated A. An unrelated bibliography item. 2001.",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "Review.pdf")

    md = "Use this single-pixel imaging review as a reference [1]; it explains reconstruction quality and sampling rate."
    hits = [
        {
            "text": "This review explains single-pixel imaging reconstruction quality, sampling rate, and deep learning methods.",
            "meta": {
                "source_path": source_path,
                "source_sha1": "abc",
                "heading_path": "Abstract",
                "evidence_quote": "This review explains single-pixel imaging reconstruction quality, sampling rate, and deep learning methods.",
            },
        }
    ]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[1](#kb-cite-" in out
    assert len(details) == 1
    assert details[0]["citation_route"] == "system_a"
    assert details[0]["card_kind"] == "answer_evidence"


def test_citation_plan_does_not_steal_bare_numeric_system_a_citation(monkeypatch):
    source_path = "scinerf.en.md"

    def fake_resolve(_index_data, _source_path, ref_num, *, source_sha1=""):
        del _index_data, _source_path, source_sha1
        if int(ref_num) != 1:
            return None
        return {
            "source_path": source_path,
            "source_name": "SCINeRF.pdf",
            "ref_num": 1,
            "ref": {
                "authors": "Boyd S",
                "year": "2011",
                "title": "Distributed Optimization and Statistical Learning via ADMM",
                "raw": "[1] Boyd S. Distributed Optimization and Statistical Learning via ADMM. 2011.",
            },
        }

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "SCINeRF.pdf")

    md = "ADMM is the optimization machinery discussed here [1]."
    hits = [
        {
            "text": "SCINeRF uses ADMM for optimization in its reconstruction pipeline.",
            "meta": {"source_path": source_path, "source_sha1": "abc"},
        }
    ]
    plan = {
        "intent": "beginner_overview",
        "budget": {"system_a": 2, "system_b": 1},
        "system_b_enabled": True,
        "slots": [{"preferred_system": "system_b", "candidate_refs": [1]}],
    }
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md,
        hits,
        anchor_ns="t",
        citation_plan=plan,
    )

    assert "[1](#kb-cite-" in out
    assert len(details) == 1
    assert details[0]["citation_route"] == "system_a"
    assert details[0]["routing_reason"] == "retrieval_hit"


def test_citation_router_applies_per_paragraph_budget_for_system_a(monkeypatch):
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", lambda *args, **kwargs: None)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    md = (
        "Structured detection improves optical sectioning [1], dynamic supersampling "
        "changes sampling [2], and light field microscopy captures 3D context [3]."
    )
    hits = [
        {
            "text": "Structured detection improves optical sectioning.",
            "meta": {
                "source_path": "doc.en.md",
                "heading_path": "A",
                "evidence_quote": "Structured detection improves optical sectioning.",
                "primary_block_id": "a",
            },
        },
        {
            "text": "Dynamic supersampling changes the sampling pattern.",
            "meta": {
                "source_path": "doc.en.md",
                "heading_path": "B",
                "evidence_quote": "Dynamic supersampling changes the sampling pattern.",
                "primary_block_id": "b",
            },
        },
        {
            "text": "Light field microscopy captures 3D context.",
            "meta": {
                "source_path": "doc.en.md",
                "heading_path": "C",
                "evidence_quote": "Light field microscopy captures 3D context.",
                "primary_block_id": "c",
            },
        },
    ]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert out.count("#kb-cite-") == 2
    assert "[3](#kb-cite-" not in out
    assert len(details) == 2
    assert all(item["citation_route"] == "system_a" for item in details)


def test_citation_router_reads_system_a_budget_from_citation_plan(monkeypatch):
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", lambda *args, **kwargs: None)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "doc.pdf")

    md = "Structured detection improves sectioning [1], and dynamic supersampling changes sampling [2]."
    hits = [
        {
            "text": "Structured detection improves optical sectioning.",
            "meta": {
                "source_path": "doc.en.md",
                "heading_path": "A",
                "evidence_quote": "Structured detection improves optical sectioning.",
                "primary_block_id": "a",
            },
        },
        {
            "text": "Dynamic supersampling changes the sampling pattern.",
            "meta": {
                "source_path": "doc.en.md",
                "heading_path": "B",
                "evidence_quote": "Dynamic supersampling changes the sampling pattern.",
                "primary_block_id": "b",
            },
        },
    ]
    plan = {"intent": "origin_lookup", "budget": {"system_a": 1, "system_b": 0}, "slots": []}
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(
        md,
        hits,
        anchor_ns="t",
        citation_plan=plan,
    )

    assert out.count("#kb-cite-") == 1
    assert "[2](#kb-cite-" not in out
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

    md = "Bad structured cite [[CITE:badbad:1]] and normal compressive imaging evidence [1]."
    hits = [
        {
            "text": "Compressive imaging evidence supports the reconstruction claim.",
            "meta": {
                "source_path": "doc.en.md",
                "heading_path": "Methods",
                "evidence_quote": "Compressive imaging evidence supports the reconstruction claim.",
            },
        }
    ]
    out, details = refs_renderer._annotate_inpaper_citations_with_hover_meta(md, hits, anchor_ns="t")

    assert "[[CITE:" not in out
    assert "Bad structured cite  and normal compressive imaging evidence [1](#" in out
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
    assert detail["card_kind"] == "upstream_reference"
    assert detail["card_title"] == "Distributed Optimization and Statistical Learning via ADMM"
    assert detail["card_evidence_label"] == "回答里的线索"
    assert "ADMM 优化框架背景" in detail["card_takeaway"]
    assert "answer_context_only" in detail["card_quality_flags"]
    assert "完整引用语境" in detail["card_warning"]
    assert detail["system_b_trace_complete"] is False
    assert "answer_context_only" in detail["system_b_trace_flags"]
    assert detail["system_b_trace_steps"] == ["答案句", "引用语境待核对", "上游文献"]
