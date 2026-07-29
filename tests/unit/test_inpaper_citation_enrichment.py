from __future__ import annotations

from kb.inpaper_citation_enrichment import (
    enrich_inpaper_detail_context,
    extract_structured_cite_answer_context_line,
    strip_structured_cite_tokens,
)


def test_extract_structured_cite_answer_context_line_strips_internal_tokens() -> None:
    sid = "abc12345"
    text = f"First line.\nThis answer opens prior work [[CITE:{sid}:4]] for context.\nNext line."

    out = extract_structured_cite_answer_context_line(
        text,
        text.index("[[CITE:"),
        text.index("[[CITE:") + len(f"[[CITE:{sid}:4]]"),
    )

    assert out == "This answer opens prior work for context."
    assert "CITE" not in out


def test_extract_structured_cite_answer_context_stops_before_next_sentence() -> None:
    sid = "abc12345"
    token = f"[[CITE:{sid}:4]]"
    text = (
        "1. The detector review explains waveguide cut-off frequency "
        f"{token}. The next sentence discusses deep learning instead."
    )

    out = extract_structured_cite_answer_context_line(
        text,
        text.index(token),
        text.index(token) + len(token),
    )

    assert "waveguide cut-off frequency" in out
    assert "deep learning" not in out


def test_extract_structured_cite_answer_context_marker_after_period_stops_before_next_sentence() -> None:
    sid = "abc12345"
    token = f"[[CITE:{sid}:4]]"
    text = f"Earlier sentence. The detector evidence ends here. {token} Next sentence is unrelated."

    out = extract_structured_cite_answer_context_line(
        text,
        text.index(token),
        text.index(token) + len(token),
    )

    assert out == "The detector evidence ends here."
    assert "unrelated" not in out


def test_extract_structured_cite_answer_context_stays_with_semicolon_clause() -> None:
    sid = "abc12345"
    token = f"[[CITE:{sid}:4]]"
    text = (
        "Hadamard requires 2N^2 measurements; "
        f"the experiment compares PSNR and SSIM {token}."
    )

    out = extract_structured_cite_answer_context_line(
        text,
        text.index(token),
        text.index(token) + len(token),
    )

    assert out == "the experiment compares PSNR and SSIM."
    assert "2N^2" not in out


def test_extract_structured_cite_answer_context_keeps_shared_method_claim_before_navigation() -> None:
    sid = "abc12345"
    token = f"[[CITE:{sid}:4]]"
    text = (
        "ADMM is prior optimization machinery; "
        f"open ADMM {token} to follow the paper's citation trail."
    )

    out = extract_structured_cite_answer_context_line(
        text,
        text.index(token),
        text.index(token) + len(token),
    )

    assert out == (
        "ADMM is prior optimization machinery; "
        "open ADMM to follow the paper's citation trail."
    )


def test_strip_structured_cite_tokens_removes_garbage_forms() -> None:
    out = strip_structured_cite_tokens("Use [CITE:abc12345:2] and [[CITE:abc12345]] plus [[CITE:broken]]")

    assert "Use  and  plus" in out
    assert "CITE" not in out


def test_enrich_inpaper_detail_context_preserves_structured_source_metadata() -> None:
    detail: dict = {"num": 7}

    def fake_extract(_source_path: str, _ref_num: int, *, answer_context: str = "") -> dict:
        assert answer_context == "The answer uses detector-array design."
        return {
            "citation_context": "The method follows a calibrated detector-array design [7].",
            "citation_context_source": "structured_reference_index",
            "heading_path": "Paper / Methods",
            "location_label": "Paper / Methods / p. 3",
            "page_start": 3,
            "line_start": 42,
            "block_id": "blk_00042",
        }

    enrich_inpaper_detail_context(
        detail,
        source_path="paper.en.md",
        ref_num=7,
        answer_context="The answer uses detector-array design.",
        extract_context_fn=fake_extract,
    )

    assert detail["answer_claim"] == "The answer uses detector-array design."
    assert detail["citation_context_source"] == "structured_reference_index"
    assert detail["evidence_source"] == "structured_reference_index"
    assert "detector-array design [7]" in detail["citation_context"]
    assert detail["heading_path"].endswith("Methods")
    assert detail["page_start"] == 3
    assert detail["block_id"] == "blk_00042"


def test_enrich_inpaper_detail_context_falls_back_to_answer_context_when_source_is_missing() -> None:
    detail: dict = {"num": 6}

    enrich_inpaper_detail_context(
        detail,
        source_path="paper.en.md",
        ref_num=6,
        answer_context="This answer mentions an upstream method.",
        extract_context_fn=lambda *_args, **_kwargs: {},
    )

    assert detail["answer_claim"] == "This answer mentions an upstream method."
    assert detail["citation_context_source"] == "answer_context"
    assert detail["evidence_source"] == "answer_context"
    assert detail["summary_source"] == "answer_context"
