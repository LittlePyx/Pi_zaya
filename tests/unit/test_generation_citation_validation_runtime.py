from __future__ import annotations

import json
import re

from kb.generation_citation_validation_runtime import (
    _source_refs_from_index,
    _validate_freeform_numeric_citations,
    _validate_structured_citations,
)


def test_source_refs_from_index_matches_doc_by_sha1():
    refs = _source_refs_from_index(
        {
            "docs": {
                "unrelated": {
                    "sha1": "abc123",
                    "path": r"db\doc\paper.en.md",
                    "refs": {"24": {"raw": "[24] Demo ref"}},
                }
            }
        },
        r"db\other\name.md",
        source_sha1="abc123",
        norm_source_key_local=lambda value: str(value or "").strip().lower(),
    )

    assert refs == {24: {"raw": "[24] Demo ref"}}


def test_validate_structured_citations_rewrites_using_injected_dependencies(tmp_path):
    source_path = r"db\doc\paper.en.md"
    locked_sid = "s1234abcd"
    cite_re = re.compile(r"\[\[CITE:([a-z0-9]+):(\d+)\]\]", re.IGNORECASE)

    def fake_resolve(_index, src, ref_num, *, source_sha1=""):
        del _index, source_sha1
        if str(src) != source_path:
            return None
        if int(ref_num) == 1:
            return {
                "ref": {
                    "raw": "[1] Wrong ref",
                    "authors": "Wrong et al.",
                    "year": "2020",
                    "title": "Wrong Ref",
                }
            }
        if int(ref_num) == 24:
            return {
                "ref": {
                    "raw": "[24] Gehm et al. Demo. 2007.",
                    "authors": "Gehm et al.",
                    "year": "2007",
                    "title": "Correct Ref",
                }
            }
        return None

    answer, stats = _validate_structured_citations(
        "Gehm et al. (2007) support this claim [[CITE:sdeadbeef:1]].",
        answer_hits=[
            {
                "text": "This follows prior work [24].",
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
        sanitize_structured_cite_tokens=lambda text: text,
        cite_canon_re=cite_re,
        cite_source_id=lambda _src: locked_sid,
        hit_source_path=lambda hit: str((hit.get("meta") or {}).get("source_path") or ""),
        load_reference_index=lambda _db_dir: {"docs": {"demo": {}}},
        resolve_reference_entry=fake_resolve,
        source_refs_from_index=lambda _index, _src, *, source_sha1="": {
            1: {"raw": "[1] Wrong ref", "authors": "Wrong et al.", "year": "2020", "title": "Wrong Ref"},
            24: {"raw": "[24] Gehm et al. Demo. 2007.", "authors": "Gehm et al.", "year": "2007", "title": "Correct Ref"},
        },
        extract_candidate_ref_nums_from_hits=lambda _hits, *, source_path="", max_candidates=48: [24],
        extract_citation_context_hints=lambda _text, *, token_start=0, token_end=0: {
            "author": "Gehm",
            "year": "2007",
            "doi": "",
        },
        has_explicit_reference_conflict=lambda ref, hints: str(ref.get("year") or "") != str(hints.get("year") or ""),
        select_support_slot_for_context=lambda slots, *, context_text="": None,
        reference_alignment_score=lambda ref, hints: 10.0 if str(ref.get("year") or "") == str(hints.get("year") or "") else 0.0,
    )

    assert answer == f"Gehm et al. (2007) support this claim [[CITE:{locked_sid}:24]]."
    assert stats["rewritten"] == 1
    assert stats["dropped"] == 0


def test_validate_structured_citations_prefers_support_resolution_ref_spans(tmp_path):
    source_path = r"db\doc\paper.en.md"
    locked_sid = "s1234abcd"
    cite_re = re.compile(r"\[\[CITE:([a-z0-9]+):(\d+)\]\]", re.IGNORECASE)

    def fake_resolve(_index, src, ref_num, *, source_sha1=""):
        del _index, source_sha1
        if str(src) != source_path:
            return None
        if int(ref_num) not in {4, 59}:
            return None
        return {"ref": {"raw": f"[{int(ref_num)}] Demo ref {int(ref_num)}", "title": f"Ref {int(ref_num)}"}}

    answer, stats = _validate_structured_citations(
        "This was reported by Duarte et al. [[CITE:sdeadbeef:59]].",
        answer_hits=[
            {
                "text": "Duarte et al. [4].",
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
        paper_guide_support_resolution=[
            {
                "line_index": 0,
                "cite_policy": "prefer_ref",
                "candidate_refs": [],
                "ref_spans": [{"text": "Duarte et al. [4]", "nums": [4], "scope": "same_clause"}],
                "resolved_ref_num": 0,
            }
        ],
        sanitize_structured_cite_tokens=lambda text: text,
        cite_canon_re=cite_re,
        cite_source_id=lambda _src: locked_sid,
        hit_source_path=lambda hit: str((hit.get("meta") or {}).get("source_path") or ""),
        load_reference_index=lambda _db_dir: {"docs": {"demo": {}}},
        resolve_reference_entry=fake_resolve,
        source_refs_from_index=lambda _index, _src, *, source_sha1="": {
            4: {"raw": "[4] Duarte et al.", "title": "Duarte"},
            59: {"raw": "[59] Compressive sensing", "title": "CS"},
        },
        extract_candidate_ref_nums_from_hits=lambda _hits, *, source_path="", max_candidates=48: [],
        extract_citation_context_hints=lambda _text, *, token_start=0, token_end=0: {
            "author": "",
            "year": "",
            "doi": "",
        },
        has_explicit_reference_conflict=lambda ref, hints: False,
        select_support_slot_for_context=lambda slots, *, context_text="": None,
        reference_alignment_score=lambda ref, hints: 0.0,
    )

    assert answer == f"This was reported by Duarte et al. [[CITE:{locked_sid}:4]]."
    assert stats["rewritten"] == 1


def test_validate_structured_citations_drops_broad_hit_only_candidate_without_local_grounding(tmp_path):
    source_path = r"db\doc\paper.en.md"
    locked_sid = "s1234abcd"
    cite_re = re.compile(r"\[\[CITE:([a-z0-9]+):(\d+)\]\]", re.IGNORECASE)

    def fake_resolve(_index, src, ref_num, *, source_sha1=""):
        del _index, source_sha1
        if str(src) != source_path:
            return None
        if int(ref_num) not in {2, 26}:
            return None
        return {"ref": {"raw": f"[{int(ref_num)}] Demo ref {int(ref_num)}", "title": f"Ref {int(ref_num)}"}}

    answer, stats = _validate_structured_citations(
        "High-fidelity novel-view synthesis improves over traditional methods [[CITE:sdeadbeef:2]].",
        answer_hits=[
            {
                "text": "Broad summary hit mentioning NeRF [26] and appearance decomposition [2].",
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
        paper_guide_candidate_refs_by_source={},
        paper_guide_support_slots=[],
        paper_guide_support_resolution=[],
        sanitize_structured_cite_tokens=lambda text: text,
        cite_canon_re=cite_re,
        cite_source_id=lambda _src: locked_sid,
        hit_source_path=lambda hit: str((hit.get("meta") or {}).get("source_path") or ""),
        load_reference_index=lambda _db_dir: {"docs": {"demo": {}}},
        resolve_reference_entry=fake_resolve,
        source_refs_from_index=lambda _index, _src, *, source_sha1="": {
            2: {"raw": "[2] Demo ref 2", "title": "Ref 2"},
            26: {"raw": "[26] Demo ref 26", "title": "Ref 26"},
        },
        extract_candidate_ref_nums_from_hits=lambda _hits, *, source_path="", max_candidates=48: [2, 26],
        extract_citation_context_hints=lambda _text, *, token_start=0, token_end=0: {
            "author": "",
            "year": "",
            "doi": "",
        },
        has_explicit_reference_conflict=lambda ref, hints: False,
        select_support_slot_for_context=lambda slots, *, context_text="": None,
        reference_alignment_score=lambda ref, hints: 0.0,
    )

    assert "[[CITE:" not in answer
    assert stats["dropped"] == 1


def test_validate_structured_citations_keeps_focused_ref_over_broad_local_slot(tmp_path):
    source_path = r"db\doc\scinerf.en.md"
    locked_sid = "s1234abcd"
    cite_re = re.compile(r"\[\[CITE:([a-z0-9]+):(\d+)\]\]", re.IGNORECASE)

    def fake_resolve(_index, src, ref_num, *, source_sha1=""):
        del _index, source_sha1
        if str(src) != source_path or int(ref_num) not in {4, 18, 20}:
            return None
        return {"ref": {"raw": f"[{int(ref_num)}] Demo", "title": f"Ref {int(ref_num)}"}}

    answer, stats = _validate_structured_citations(
        "ADMM is established prior work [[CITE:s1234abcd:4]].",
        answer_hits=[{"text": "Existing methods use ADMM [4].", "meta": {"source_path": source_path}}],
        db_dir=tmp_path,
        locked_source={"sid": locked_sid, "source_path": source_path},
        paper_guide_mode=True,
        paper_guide_candidate_refs_by_source={source_path: [4]},
        paper_guide_support_slots=[
            {
                "source_path": source_path,
                "candidate_refs": [18, 20],
                "snippet": "Earlier regularized methods [18,20].",
            }
        ],
        paper_guide_support_resolution=[],
        sanitize_structured_cite_tokens=lambda text: text,
        cite_canon_re=cite_re,
        cite_source_id=lambda _src: locked_sid,
        hit_source_path=lambda hit: str((hit.get("meta") or {}).get("source_path") or ""),
        load_reference_index=lambda _db_dir: {"docs": {"demo": {}}},
        resolve_reference_entry=fake_resolve,
        source_refs_from_index=lambda _index, _src, *, source_sha1="": {
            4: {"raw": "[4] ADMM", "title": "ADMM"},
            18: {"raw": "[18] Other", "title": "Other"},
            20: {"raw": "[20] Other", "title": "Other"},
        },
        extract_candidate_ref_nums_from_hits=lambda _hits, *, source_path="", max_candidates=48: [18, 20, 4],
        extract_citation_context_hints=lambda _text, *, token_start=0, token_end=0: {
            "author": "",
            "year": "",
            "doi": "",
        },
        has_explicit_reference_conflict=lambda ref, hints: False,
        select_support_slot_for_context=lambda slots, *, context_text="": slots[0] if slots else None,
        reference_alignment_score=lambda ref, hints: 0.0,
    )

    assert answer == "ADMM is established prior work [[CITE:s1234abcd:4]]."
    assert stats["kept"] == 1
    assert stats["dropped"] == 0


def test_validate_structured_citations_uses_doc_reference_index_when_global_index_missing(tmp_path):
    source_path = tmp_path / "demo.en.md"
    source_path.write_text("# Demo\n", encoding="utf-8")
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "reference_index.json").write_text(
        json.dumps(
            {
                "references": [
                    {
                        "ref_num": 4,
                        "text": "[4] Duarte et al. Robust imaging. 2007.",
                        "doi": "",
                        "year": "2007",
                        "parse_confidence": 0.9,
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    locked_sid = "s1234abcd"
    cite_re = re.compile(r"\[\[CITE:([a-z0-9]+):(\d+)\]\]", re.IGNORECASE)

    answer, stats = _validate_structured_citations(
        "This was reported by Duarte et al. [[CITE:sdeadbeef:59]].",
        answer_hits=[
            {
                "text": "Local evidence text with no reusable numeric cite.",
                "meta": {
                    "source_path": str(source_path),
                    "source_sha1": "abc",
                },
            }
        ],
        db_dir=tmp_path,
        locked_source={
            "sid": locked_sid,
            "source_path": str(source_path),
            "source_sha1": "abc",
        },
        paper_guide_mode=True,
        paper_guide_support_resolution=[
            {
                "line_index": 0,
                "cite_policy": "prefer_ref",
                "candidate_refs": [],
                "ref_spans": [{"text": "Duarte et al. [4]", "nums": [4], "scope": "same_clause"}],
                "resolved_ref_num": 0,
            }
        ],
        sanitize_structured_cite_tokens=lambda text: text,
        cite_canon_re=cite_re,
        cite_source_id=lambda _src: locked_sid,
        hit_source_path=lambda hit: str((hit.get("meta") or {}).get("source_path") or ""),
        load_reference_index=lambda _db_dir: {"docs": {"demo": {}}},
        resolve_reference_entry=lambda _index, _src, _ref_num, *, source_sha1="": None,
        source_refs_from_index=lambda _index, _src, *, source_sha1="": {},
        extract_candidate_ref_nums_from_hits=lambda _hits, *, source_path="", max_candidates=48: [],
        extract_citation_context_hints=lambda _text, *, token_start=0, token_end=0: {
            "author": "",
            "year": "",
            "doi": "",
        },
        has_explicit_reference_conflict=lambda ref, hints: False,
        select_support_slot_for_context=lambda slots, *, context_text="": None,
        reference_alignment_score=lambda ref, hints: 0.0,
    )

    assert answer == f"This was reported by Duarte et al. [[CITE:{locked_sid}:4]]."
    assert stats["rewritten"] == 1


def test_validate_structured_citations_extracts_local_ref_from_anchor_index_when_block_only(tmp_path):
    source_path = tmp_path / "demo.en.md"
    source_path.write_text("# Demo\n", encoding="utf-8")
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "anchor_index.json").write_text(
        json.dumps(
            {
                "anchors": [
                    {
                        "anchor_id": "anc_method",
                        "block_id": "blk_method",
                        "kind": "paragraph",
                        "heading_path": "Methods / APR",
                        "text": "APR was performed using image registration based on phase correlation [35].",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    locked_sid = "s1234abcd"
    cite_re = re.compile(r"\[\[CITE:([a-z0-9]+):(\d+)\]\]", re.IGNORECASE)

    def fake_resolve(_index, src, ref_num, *, source_sha1=""):
        del _index, source_sha1
        if str(src) != str(source_path):
            return None
        if int(ref_num) != 35:
            return None
        return {"ref": {"raw": "[35] APR library ref", "title": "APR Tool"}}

    answer, stats = _validate_structured_citations(
        f"Implementation detail: APR uses phase correlation [[CITE:{locked_sid}:35]].",
        answer_hits=[
            {
                "text": "Broad summary with no in-paper numeric citations.",
                "meta": {
                    "source_path": str(source_path),
                    "source_sha1": "abc",
                },
            }
        ],
        db_dir=tmp_path,
        locked_source={
            "sid": locked_sid,
            "source_path": str(source_path),
            "source_sha1": "abc",
        },
        paper_guide_mode=True,
        paper_guide_support_resolution=[
            {
                "line_index": 0,
                "block_id": "blk_method",
                "anchor_id": "anc_method",
                "cite_policy": "prefer_ref",
                "candidate_refs": [],
                "ref_spans": [],
                "resolved_ref_num": 0,
            }
        ],
        sanitize_structured_cite_tokens=lambda text: text,
        cite_canon_re=cite_re,
        cite_source_id=lambda _src: locked_sid,
        hit_source_path=lambda hit: str((hit.get("meta") or {}).get("source_path") or ""),
        load_reference_index=lambda _db_dir: {"docs": {"demo": {}}},
        resolve_reference_entry=fake_resolve,
        source_refs_from_index=lambda _index, _src, *, source_sha1="": {},
        extract_candidate_ref_nums_from_hits=lambda _hits, *, source_path="", max_candidates=48: [],
        extract_citation_context_hints=lambda _text, *, token_start=0, token_end=0: {
            "author": "",
            "year": "",
            "doi": "",
        },
        has_explicit_reference_conflict=lambda ref, hints: False,
        select_support_slot_for_context=lambda slots, *, context_text="": None,
        reference_alignment_score=lambda ref, hints: 0.0,
    )

    assert answer == f"Implementation detail: APR uses phase correlation [[CITE:{locked_sid}:35]]."
    assert stats["kept"] == 1


# ── Freeform numeric citation validation tests ──────────────────────────


def test_validate_freeform_all_valid():
    """All [n] markers within hit range are kept."""
    hits = [{"text": f"doc {i}"} for i in range(3)]
    answer, stats = _validate_freeform_numeric_citations(
        "Claim from [1] and also [2].",
        answer_hits=hits,
    )
    assert answer == "Claim from [1] and also [2]."
    assert stats["raw_count"] == 2
    assert stats["valid_count"] == 2
    assert stats["out_of_range"] == 0
    assert stats["dropped"] == 0
    assert stats["kept"] == 2


def test_validate_freeform_out_of_range():
    """[n] where n > len(answer_hits) is dropped."""
    hits = [{"text": "doc 1"}, {"text": "doc 2"}, {"text": "doc 3"}]
    answer, stats = _validate_freeform_numeric_citations(
        "Claim from [1] and [5].",
        answer_hits=hits,
    )
    assert " [5]" not in answer
    assert "[1]" in answer
    assert answer.endswith("and.")
    assert stats["raw_count"] == 2
    assert stats["out_of_range"] == 1
    assert stats["dropped"] == 1
    assert stats["kept"] == 1
    assert stats["valid_count"] == 1


def test_validate_freeform_zero_or_negative():
    """[0] is dropped (ref num 0 is invalid). [-1] is not matched by the regex."""
    hits = [{"text": "doc 1"}]
    answer, stats = _validate_freeform_numeric_citations(
        "See [0] and [-1].",
        answer_hits=hits,
    )
    # [0] is dropped because 0 is not a valid positive ref num
    assert "[0]" not in answer
    assert stats["dropped"] == 1
    assert stats["raw_count"] == 1  # only [0] is matched, [-1] is not a valid match
    # 0 fails the n > 0 check in _all_nums_in_spec, not the out_of_range check
    assert stats["out_of_range"] == 0


def test_validate_freeform_no_hits():
    """When answer_hits is empty, all [n] are out of range."""
    answer, stats = _validate_freeform_numeric_citations(
        "Claim from [1] and [2].",
        answer_hits=[],
    )
    assert "[1]" not in answer
    assert "[2]" not in answer
    assert stats["raw_count"] == 2
    assert stats["out_of_range"] == 2
    assert stats["dropped"] == 2
    assert stats["hits_available"] == 0


def test_validate_freeform_mixed():
    """Mixed valid and out-of-range markers: valid kept, invalid dropped."""
    hits = [{"text": "doc 1"}, {"text": "doc 2"}]
    answer, stats = _validate_freeform_numeric_citations(
        "Results from [1] and [99] but also [2].",
        answer_hits=hits,
    )
    assert "[1]" in answer
    assert "[99]" not in answer
    assert "[2]" in answer
    assert stats["raw_count"] == 3
    assert stats["valid_count"] == 2
    assert stats["out_of_range"] == 1
    assert stats["dropped"] == 1
    assert stats["kept"] == 2


def test_validate_freeform_partially_rewrites_range_with_out_of_range_member():
    """A partly valid range keeps the usable hit citations instead of dropping all evidence."""
    hits = [{"text": "doc 1"}, {"text": "doc 2"}]
    dash = chr(0x2013)
    answer, stats = _validate_freeform_numeric_citations(
        f"Range evidence [{1}{dash}{3}] should retain the in-range hits.",
        answer_hits=hits,
    )

    assert f"[1{dash}3]" not in answer
    assert "[1,2]" in answer
    assert stats["raw_count"] == 1
    assert stats["valid_count"] == 1
    assert stats["out_of_range"] == 1
    assert stats["dropped"] == 0
    assert stats["kept"] == 1
    assert stats["rewritten"] == 1


def test_validate_freeform_partially_rewrites_comma_list_with_out_of_range_member():
    hits = [{"text": "doc 1"}, {"text": "doc 2"}]
    answer, stats = _validate_freeform_numeric_citations(
        "Mixed evidence [1, 99, 2] should not lose valid citations.",
        answer_hits=hits,
    )

    assert "[1, 99, 2]" not in answer
    assert "[1,2]" in answer
    assert stats["out_of_range"] == 1
    assert stats["dropped"] == 0
    assert stats["kept"] == 1
    assert stats["rewritten"] == 1


def test_validate_freeform_partially_rewrites_zero_mixed_with_valid_member():
    hits = [{"text": "doc 1"}]
    answer, stats = _validate_freeform_numeric_citations(
        "Mixed zero citation [0,1] should keep only the valid hit.",
        answer_hits=hits,
    )

    assert "[0,1]" not in answer
    assert "[1]" in answer
    assert stats["out_of_range"] == 0
    assert stats["dropped"] == 0
    assert stats["kept"] == 1
    assert stats["rewritten"] == 1


def test_validate_freeform_no_markers():
    """Text without any [n] markers is returned unchanged."""
    hits = [{"text": "doc 1"}]
    answer, stats = _validate_freeform_numeric_citations(
        "Plain text without citations.",
        answer_hits=hits,
    )
    assert answer == "Plain text without citations."
    assert stats["raw_count"] == 0
    assert stats["dropped"] == 0


def test_validate_freeform_empty_hits_list():
    """Empty answer_hits list behaves the same as no hits."""
    answer, stats = _validate_freeform_numeric_citations(
        "Some claim [1] here.",
        answer_hits=[],
    )
    assert "[1]" not in answer
    assert stats["raw_count"] == 1
    assert stats["out_of_range"] == 1
    assert stats["dropped"] == 1


def test_validate_freeform_empty_answer():
    """Empty answer string returns empty."""
    answer, stats = _validate_freeform_numeric_citations(
        "",
        answer_hits=[{"text": "doc 1"}],
    )
    assert answer == ""
    assert stats["raw_count"] == 0


def test_validate_freeform_single_hit_valid():
    """With 1 hit, [1] is valid, [2] is not."""
    hits = [{"text": "doc 1"}]
    answer, stats = _validate_freeform_numeric_citations(
        "See [1] for details.",
        answer_hits=hits,
    )
    assert "[1]" in answer
    assert stats["valid_count"] == 1
    assert stats["kept"] == 1


def test_validate_structured_citations_accepts_confident_author_without_year(tmp_path):
    source_path = r"db\doc\rag.en.md"
    locked_sid = "s1234abcd"
    cite_re = re.compile(r"\[\[CITE:([a-z0-9]+):(\d+)\]\]", re.IGNORECASE)
    ref = {
        "raw": "[26] Karpukhin et al. Dense Passage Retrieval for Open-Domain Question Answering.",
        "authors": "Karpukhin V et al",
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
    }

    answer, stats = _validate_structured_citations(
        "Dense Passage Retrieval was introduced by Karpukhin et al. [[CITE:s1234abcd:26]].",
        answer_hits=[{"text": "The retriever is based on DPR [26].", "meta": {"source_path": source_path}}],
        db_dir=tmp_path,
        locked_source={"sid": locked_sid, "source_path": source_path},
        paper_guide_mode=True,
        paper_guide_candidate_refs_by_source={},
        paper_guide_support_slots=[],
        paper_guide_support_resolution=[],
        sanitize_structured_cite_tokens=lambda text: text,
        cite_canon_re=cite_re,
        cite_source_id=lambda _src: locked_sid,
        hit_source_path=lambda hit: str((hit.get("meta") or {}).get("source_path") or ""),
        load_reference_index=lambda _db_dir: {"docs": {"demo": {}}},
        resolve_reference_entry=lambda _index, src, num, *, source_sha1="": (
            {"ref": ref} if src == source_path and int(num) == 26 else None
        ),
        source_refs_from_index=lambda _index, src, *, source_sha1="": (
            {26: ref} if src == source_path else {}
        ),
        extract_candidate_ref_nums_from_hits=lambda _hits, *, source_path="", max_candidates=48: [],
        extract_citation_context_hints=lambda _text, *, token_start=0, token_end=0: {
            "author": "karpukhin",
            "author_confident": True,
            "year": "",
            "doi": "",
        },
        has_explicit_reference_conflict=lambda _ref, _hints: False,
        select_support_slot_for_context=lambda _slots, *, context_text="": None,
        reference_alignment_score=lambda _ref, _hints: 2.5,
    )

    assert "[[CITE:s1234abcd:26]]" in answer
    assert stats["kept"] == 1
    assert stats["dropped"] == 0
