from __future__ import annotations

from kb.citation_card_polish import (
    citation_card_polish_cache_key,
    polish_citation_card_detail,
)


def test_polish_citation_card_accepts_grounded_short_fields() -> None:
    detail = {
        "source_name": "Single pixel imaging review.pdf",
        "heading_path": "Methods",
        "answer_claim": "DMD modulation is the sampling hardware behind this method.",
        "evidence_quote": "A DMD can spatially filter light and redirect the incident beam during measurement.",
        "location_label": "Methods",
    }

    def fake_llm(**_kwargs: object) -> str:
        return (
            '{"card_takeaway":"DMD is the hardware mechanism that makes the sampling strategy concrete.",'
            '"card_claim":"The answer is tying the method to DMD-based optical modulation.",'
            '"card_support_explanation":"The quoted sentence names the DMD action rather than only describing an outcome."}'
        )

    out = polish_citation_card_detail(detail, llm_fn=fake_llm)

    assert out["citation_card_polish_status"] == "full"
    assert out["citation_card_polish_source"] == "llm"
    assert out["card_takeaway"].startswith("DMD is the hardware")
    assert "card_evidence" not in out


def test_polish_citation_card_rejects_markdown_and_generic_output() -> None:
    detail = {
        "source_name": "Fixture.pdf",
        "heading_path": "Abstract",
        "answer_claim": "The work improves single-pixel imaging.",
        "evidence_quote": "Deep learning reduces the sampling ratio while preserving reconstruction quality.",
    }

    def fake_llm(**_kwargs: object) -> str:
        return (
            '{"card_takeaway":"| field | value |",'
            '"card_claim":"This evidence supports the answer.",'
            '"card_support_explanation":"```bad```"}'
        )

    out = polish_citation_card_detail(detail, llm_fn=fake_llm)

    assert out == {
        "citation_card_polish_status": "empty",
        "citation_card_polish_source": "llm_empty",
        "citation_card_polish_checked": True,
    }


def test_citation_card_polish_cache_key_normalizes_frontend_aliases() -> None:
    snake = {
        "source_name": "Fixture.pdf",
        "heading_path": "Abstract",
        "answer_claim": "The paper uses structured illumination.",
        "evidence_quote": "Structured illumination patterns are projected onto the scene.",
        "location_label": "Abstract",
    }
    camel = {
        "sourceName": "Fixture.pdf",
        "headingPath": "Abstract",
        "answerClaim": "The paper uses structured illumination.",
        "evidenceQuote": "Structured illumination patterns are projected onto the scene.",
        "locationLabel": "Abstract",
    }

    assert citation_card_polish_cache_key(snake) == citation_card_polish_cache_key(camel)
