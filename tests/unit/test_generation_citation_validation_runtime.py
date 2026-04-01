from __future__ import annotations

import re

from kb.generation_citation_validation_runtime import (
    _source_refs_from_index,
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
