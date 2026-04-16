from __future__ import annotations

import json
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
