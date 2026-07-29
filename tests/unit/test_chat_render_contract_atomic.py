from __future__ import annotations

from api import chat_render


def test_body_fidelity_does_not_hide_code_array_range_or_year_changes() -> None:
    for answer, rendered in (
        ("Use `[1]` in code.", "Use `` in code."),
        ("Values are [1, 2].", "Values are."),
        ("Range [1-3] is inclusive.", "Range is inclusive."),
        ("Published in [2024].", "Published in."),
    ):
        assert not chat_render._rendered_body_preserves_answer_body(
            answer_body=answer,
            rendered_body=rendered,
        )

    assert chat_render._rendered_body_preserves_answer_body(
        answer_body="Supported claim [4].",
        rendered_body="Supported claim [1](#kb-cite-paper-1).",
        cite_details=[
            {
                "num": 1,
                "linked_nums": [4],
                "anchor": "kb-cite-paper-1",
                "source_path": "paper.md",
            }
        ],
    )

    assert chat_render._rendered_body_preserves_answer_body(
        answer_body="First source claim [2]. Second source claim [3].",
        rendered_body=(
            "First source claim [1](#kb-cite-paper-a). "
            "Second source claim [2](#kb-cite-paper-b)."
        ),
        cite_details=[
            {
                "num": 1,
                "anchor": "kb-cite-paper-a",
                "source_path": "paper-a.md",
            },
            {
                "num": 2,
                "anchor": "kb-cite-paper-b",
                "source_path": "paper-b.md",
            },
        ],
    )


def test_strip_numeric_markers_only_removes_confirmed_citations_outside_code() -> None:
    raw = "Cite [[4]]; keep [1, 2], [1-3], [2024], [], and `[4]`. Plain [4]."

    without_protocol = chat_render._strip_freeform_numeric_citation_markers(raw)
    assert "[[4]]" not in without_protocol
    assert "[1, 2]" in without_protocol
    assert "[1-3]" in without_protocol
    assert "[2024]" in without_protocol
    assert "[]" in without_protocol
    assert "`[4]`" in without_protocol
    assert "Plain [4]" in without_protocol

    confirmed = chat_render._strip_freeform_numeric_citation_markers(
        raw,
        confirmed_numbers={4},
    )
    assert "`[4]`" in confirmed
    assert "Plain [4]" not in confirmed


def test_render_packet_rebuild_keeps_notice_in_display_and_copy() -> None:
    rec = {
        "id": 2,
        "content": "Answer body.",
        "notice": "Evidence is limited.",
        "rendered_body": "Answer body.",
        "rendered_content": "stale display",
        "copy_markdown": "stale copy",
        "copy_text": "stale copy",
        "cite_details": [],
        "meta": {"paper_guide_contracts": {}},
    }

    chat_render._merge_render_packet_contract_meta(
        rec=rec,
        msg_id=2,
        enriched_provenance={},
        ref_pack={"hits": []},
        render_locale="en",
        answer_sig=chat_render._answer_render_signature("Answer body."),
        input_ref_sig="refs-empty",
        citation_plan_sig="plan-empty",
    )

    packet = rec["meta"]["paper_guide_contracts"]["render_packet"]
    assert packet["rendered_body"] == "Answer body."
    assert packet["rendered_content"].startswith("Evidence is limited.")
    assert packet["copy_markdown"].startswith("Evidence is limited.")
    assert packet["copy_text"].startswith("Evidence is limited.")


def test_preserve_existing_rejects_signature_mismatch_even_with_current_refs() -> None:
    answer = "Claim [1]."
    existing_packet = {
        "schema": chat_render._RENDER_CACHE_SCHEMA_VERSION,
        "answer_markdown": answer,
        "answer_sig": "stale-answer",
        "input_ref_sig": "refs-current",
        "citation_plan_sig": "plan-current",
        "locale": "en",
        "notice": "",
        "rendered_body": "Claim [1](#kb-cite-paper-1).",
        "rendered_content": "Claim [1](#kb-cite-paper-1).",
        "copy_markdown": "Claim [1].",
        "copy_text": "Claim [1].",
        "cite_details": [
            {
                "num": 1,
                "anchor": "kb-cite-paper-1",
                "source_path": "paper.md",
                "citation_route": "system_a",
            }
        ],
    }
    rec = {
        "id": 2,
        "content": answer,
        "rendered_body": answer,
        "rendered_content": answer,
        "copy_markdown": answer,
        "copy_text": answer,
        "cite_details": [],
        "meta": {
            "answer_quality": {"output_mode": "citation"},
            "paper_guide_contracts": {"render_packet": existing_packet},
        },
    }

    chat_render._merge_render_packet_contract_meta(
        rec=rec,
        msg_id=2,
        enriched_provenance={},
        ref_pack={
            "hits": [
                {
                    "text": "Claim evidence.",
                    "meta": {"source_path": "paper.md"},
                }
            ]
        },
        render_locale="en",
        answer_sig=chat_render._answer_render_signature(answer),
        input_ref_sig="refs-current",
        citation_plan_sig="plan-current",
    )

    packet = rec["meta"]["paper_guide_contracts"]["render_packet"]
    assert packet["cite_details"] == []
    assert "#kb-cite-paper-1" not in packet["rendered_body"]


def test_prompt_aligned_system_a_evidence_is_not_replaced_by_broader_ref_card_text() -> None:
    exact = (
        "Interferometric detection reaches about 120 nm lateral resolution at "
        "tenfold lower incident illumination power."
    )
    broader = "iISM is broadly applicable for minimally invasive imaging."
    details = [
        {
            "num": 1,
            "anchor": "kb-cite-iism-1",
            "citation_route": "system_a",
            "source_path": "db/iism/iism.en.md",
            "source_name": "iISM.pdf",
            "heading_path": "Abstract",
            "page_start": 1,
            "evidence_quote": exact,
            "raw": exact,
            "selection_reason": "prompt_aligned_source_sentence",
        }
    ]
    ref_pack = {
        "hits": [
            {
                "text": broader,
                "meta": {"source_path": "db/iism/iism.en.md"},
                "ui_meta": {
                    "source_path": "db/iism/iism.en.md",
                    "primary_evidence": {
                        "snippet": broader,
                        "heading_path": "Introduction",
                        "page_start": 2,
                    },
                },
            }
        ]
    }

    out = chat_render._backfill_system_a_cite_details_from_ref_pack(
        details,
        ref_pack,
        render_locale="en",
    )

    assert out[0]["evidence_quote"] == exact
    assert out[0]["heading_path"] == "Abstract"
    assert out[0]["page_start"] == 1
