from __future__ import annotations

import pytest

from kb.generation_answer_finalize_runtime import (
    _maybe_append_paper_guide_supplement_block,
    _strip_citation_offset,
    _strip_final_answer_citation_markers,
)
from kb.paper_guide_answer_selection import _merge_same_source_answer_hits
from kb.source_passage_evidence import bind_source_passage_primary


@pytest.mark.parametrize("mode", ["disabled", "web"])
def test_local_or_web_only_permission_does_not_generate_model_supplement(
    monkeypatch, mode
):
    monkeypatch.setenv("KB_PAPER_GUIDE_SUPPLEMENT_BLOCK", "1")
    answer = "The retrieved snippets do not specify that detail [1]."
    result = _maybe_append_paper_guide_supplement_block(
        answer,
        paper_guide_mode=True,
        has_hits=True,
        prompt_text="Explain this method.",
        prompt_family="method",
        retrieval_confidence_hint={"low_confidence": True},
        external_supplement_mode=mode,
        build_paper_guide_supplement_lines=lambda **_: pytest.fail(
            "unauthorized model supplement"
        ),
    )
    assert result == answer


def test_numeric_math_is_not_rewritten_as_offset_citation():
    answer = r"$x\in[10000,20000]$; supported [10001]."
    assert _strip_citation_offset(answer) == r"$x\in[10000,20000]$; supported [1]."


def test_paper_guide_final_marker_cleanup_preserves_noise_interval():
    answer = r"Noise range $\sigma\in[0,55]$ [1]."
    assert (
        _strip_final_answer_citation_markers(answer, preserve_numeric_markers=False)
        == r"Noise range $\sigma\in[0,55]$."
    )


def test_explicit_model_permission_preserves_labeled_supplement(monkeypatch):
    monkeypatch.setenv("KB_PAPER_GUIDE_SUPPLEMENT_BLOCK", "1")
    result = _maybe_append_paper_guide_supplement_block(
        "The retrieved snippet gives only a short method description [1].",
        paper_guide_mode=True,
        has_hits=True,
        prompt_text="Explain this method.",
        prompt_family="method",
        retrieval_confidence_hint={"low_confidence": True},
        external_supplement_mode="model",
        build_paper_guide_supplement_lines=lambda **_: [
            "A generic context note, not a claim from the paper."
        ],
    )
    assert "Supplementary note" in result and "not a claim from the paper" in result


def test_bundle_keeps_actual_first_passage_as_primary_locator():
    hits = [
        {
            "text": "First paragraph is source evidence with enough actual context.",
            "meta": {"source_path": "paper.en.md", "page_start": 3, "block_id": "b3"},
        },
        {
            "text": "Second paragraph is other source evidence with another context.",
            "meta": {"source_path": "paper.en.md", "page_start": 5, "block_id": "b5"},
        },
    ]
    bundle = _merge_same_source_answer_hits(hits)[0]
    assert "Source passage 2" in bundle["text"]
    primary = bind_source_passage_primary(
        {"source_path": "paper.en.md", "snippet": bundle["text"]}, {}, [bundle]
    )
    assert primary["snippet"] == hits[0]["text"]
    assert primary["page_start"] == 3 and primary["block_id"] == "b3"
    assert primary["source_path"] == "paper.en.md"


@pytest.mark.parametrize("wrong_field", ["source_path", "page_start"])
def test_compound_excerpt_cannot_borrow_another_source_or_page(wrong_field):
    text = "A real source paragraph contains a sufficiently long physical excerpt."
    passage = {"text": text, "page_start": 3, "block_id": "b3"}
    hit = {"meta": {"source_path": "paper.en.md", "source_passages": [passage]}}
    card = {"source_path": "paper.en.md", "snippet": f"[Source passage 1: Method p. 3] {text}"}
    if wrong_field == "source_path":
        hit["meta"][wrong_field] = "other.en.md"
    else:
        passage[wrong_field] = 5
    assert bind_source_passage_primary(card, {}, [hit]) == {}
