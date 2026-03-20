from __future__ import annotations

from kb.inpaper_citation_grounding import (
    extract_candidate_ref_nums_from_hits,
    extract_citation_context_hints,
    has_explicit_reference_conflict,
    parse_ref_num_set,
    reference_alignment_score,
)


def test_parse_ref_num_set_supports_ascii_and_unicode_dashes():
    assert parse_ref_num_set("11-13") == [11, 12, 13]
    assert parse_ref_num_set("11\u201313") == [11, 12, 13]
    assert parse_ref_num_set("11\u201413") == [11, 12, 13]
    assert parse_ref_num_set("11\u221213") == [11, 12, 13]


def test_parse_ref_num_set_dedupes_and_skips_large_ranges():
    assert parse_ref_num_set("7, 7, 8-10, 9") == [7, 8, 9, 10]
    assert parse_ref_num_set("1-20") == []


def test_extract_candidate_ref_nums_from_hits_reads_text_and_snippets():
    hits = [
        {
            "text": "Main discussion cites [24].",
            "meta": {
                "source_path": "doc.en.md",
                "ref_show_snippets": [
                    "Supporting refs [30\u201331].",
                    "A second snippet mentions [45, 46].",
                ],
            },
        },
        {
            "text": "Other document [88].",
            "meta": {
                "source_path": "other.en.md",
                "ref_show_snippets": ["Noise [99]."],
            },
        },
    ]

    assert extract_candidate_ref_nums_from_hits(hits, source_path="doc.en.md") == [24, 30, 31, 45, 46]


def test_extract_citation_context_hints_captures_doi_author_and_year():
    answer = "Gehm et al. (2007) and DOI 10.1364/OE.15.014013 support this claim [[CITE:sid:1]]."
    token_start = answer.index("[[CITE:")
    token_end = token_start + len("[[CITE:sid:1]]")

    hints = extract_citation_context_hints(answer, token_start=token_start, token_end=token_end)

    assert hints["doi"] == "10.1364/oe.15.014013"
    assert hints["year"] == "2007"
    assert hints["author"] == "gehm"
    assert hints["author_confident"] is True


def test_reference_alignment_score_prefers_exact_match_and_conflict_detection():
    hints = {
        "doi": "10.1364/oe.15.014013",
        "year": "2007",
        "author": "gehm",
        "author_confident": True,
    }
    good_ref = {
        "authors": "Gehm M, Brady D",
        "year": "2007",
        "doi": "10.1364/OE.15.014013",
        "raw": "[24] Gehm M, Brady D. Opt Express, 2007. doi:10.1364/OE.15.014013",
    }
    bad_ref = {
        "authors": "Smith J",
        "year": "2020",
        "doi": "10.1000/wrong",
        "raw": "[1] Smith J. Wrong paper. 2020. doi:10.1000/wrong",
    }

    assert reference_alignment_score(good_ref, hints) > reference_alignment_score(bad_ref, hints)
    assert has_explicit_reference_conflict(good_ref, hints) is False
    assert has_explicit_reference_conflict(bad_ref, hints) is True
