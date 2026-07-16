from __future__ import annotations

from kb.inpaper_citation_grounding import (
    extract_candidate_ref_nums_from_hits,
    extract_candidate_ref_cue_texts,
    extract_citation_context_hints,
    has_explicit_reference_conflict,
    iter_inpaper_numeric_citations,
    parse_ref_num_set,
    reference_alignment_score,
)


def test_parse_ref_num_set_supports_ascii_and_unicode_dashes():
    assert parse_ref_num_set("11-13") == [11, 12, 13]
    assert parse_ref_num_set("11\u201313") == [11, 12, 13]
    assert parse_ref_num_set("11\u201413") == [11, 12, 13]
    assert parse_ref_num_set("11\u221213") == [11, 12, 13]
    assert parse_ref_num_set("³⁰⁻³³,⁴³") == [30, 31, 32, 33, 43]


def test_iter_inpaper_numeric_citations_accepts_nature_superscripts_but_not_exponents():
    text = (
        "Optical cavity designs remain challenging³⁰⁻³³. "
        "The accepted lasing protocol is used⁴³. "
        "The area is 0.02 mm², current density is 280 A cm⁻², Pb²⁺ is present, and R² is 0.99."
    )

    markers = iter_inpaper_numeric_citations(text)

    assert [parse_ref_num_set(spec) for spec, *_rest in markers] == [[30, 31, 32, 33], [43]]


def test_iter_inpaper_numeric_citations_ignores_numeric_brackets_inside_math():
    text = (
        "The architecture follows prior work [21]. "
        "Its convolution weights have dimensions [9,9,1,64], the dataset has [20,000] images, "
        "and the sum is $\\sum_{i=1}^{[20,000]} y^i$."
    )

    markers = iter_inpaper_numeric_citations(text)

    assert [parse_ref_num_set(spec) for spec, *_rest in markers] == [[21]]


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


def test_extract_candidate_ref_nums_from_hits_reads_unicode_superscripts():
    hits = [
        {
            "text": "Prior cavity designs are difficult³⁰⁻³³, while lasing follows accepted protocols⁴³.",
            "meta": {"source_path": "nature.en.md"},
        }
    ]

    assert extract_candidate_ref_nums_from_hits(hits, source_path="nature.en.md") == [30, 31, 32, 33, 43]


def test_extract_candidate_ref_cue_texts_keeps_numeric_citation_windows():
    hit = {
        "text": (
            "A long introduction sentence that keeps going before the citation appears and still needs trimming "
            "because the actual evidence mentions prior work [24] and should stay visible in the cue."
        ),
        "meta": {
            "ref_show_snippets": [
                "No citation here.",
                "Supporting refs [30-31] appear in the localized snippet.",
            ]
        },
    }

    cues = extract_candidate_ref_cue_texts(hit, max_cues=2, max_chars=72)

    assert len(cues) == 2
    assert "[24]" in cues[0]
    assert "[30-31]" in cues[1]
    assert "No citation here." not in " ".join(cues)


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
