from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def test_reference_index_cache_refreshes_after_file_change(tmp_path: Path, monkeypatch) -> None:
    index_path = tmp_path / "references_index.json"
    index_path.write_text("old", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(
        chat_render,
        "load_settings",
        lambda: SimpleNamespace(db_dir=tmp_path),
    )

    def fake_load_reference_index(db_dir: Path) -> dict:
        value = (Path(db_dir) / "references_index.json").read_text(encoding="utf-8")
        calls.append(value)
        return {"value": value}

    monkeypatch.setattr(chat_render, "load_reference_index", fake_load_reference_index)
    chat_render._load_reference_index_for_signature.cache_clear()

    assert chat_render._load_reference_index_cached() == {"value": "old"}
    assert chat_render._load_reference_index_cached() == {"value": "old"}
    index_path.write_text("new-and-larger", encoding="utf-8")
    assert chat_render._load_reference_index_cached() == {"value": "new-and-larger"}
    assert calls == ["old", "new-and-larger"]


def test_grounded_plan_repair_cannot_rewrite_answer_prose_in_renderer(monkeypatch) -> None:
    source_path = "db/paper/paper.en.md"
    second_source_path = "db/second/second.en.md"
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Abstract",
                "evidence_quote": "Grounded evidence for the missing step.",
            },
            {
                "preferred_system": "system_a",
                "source_path": second_source_path,
                "heading_path": "Introduction",
                "evidence_quote": "Second source confirms the sequence.",
            },
        ],
    }

    monkeypatch.setattr(
        chat_render,
        "_should_link_inpaper_citations_for_message",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        chat_render,
        "_reading_guide_repair_missing_system_a_citations",
        lambda md, *_args, **_kwargs: (
            f"{md}\nGrounded missing step [1]. Second source confirms it [2]."
        ),
    )

    def fake_annotate(md, _hits, **_kwargs):
        if "[1]" not in md or "[2]" not in md:
            return md, []
        return md.replace("[1]", "[1](#kb-cite-a-1)").replace(
            "[2]", "[2](#kb-cite-a-2)"
        ), [
            {
                "num": 1,
                "anchor": "kb-cite-a-1",
                "citation_route": "system_a",
                "source_path": source_path,
                "heading_path": "Abstract",
                "evidence_quote": "Grounded evidence for the missing step.",
                "answer_claim": "Grounded missing step.",
            },
            {
                "num": 2,
                "anchor": "kb-cite-a-2",
                "citation_route": "system_a",
                "source_path": second_source_path,
                "heading_path": "Introduction",
                "evidence_quote": "Second source confirms the sequence.",
                "answer_claim": "Second source confirms it.",
            },
        ]

    monkeypatch.setattr(
        chat_render,
        "_annotate_inpaper_citations_with_hover_meta",
        fake_annotate,
    )
    rendered = chat_render.enrich_messages_with_reference_render(
        [
            {"id": 1, "role": "user", "content": "Explain the missing step."},
            {
                "id": 2,
                "role": "assistant",
                "content": "Original answer.",
                "meta": {
                    "answer_quality": {
                        "output_mode": "reading_guide",
                        "citation_plan": plan,
                    }
                },
            },
        ],
        {
            1: {
                "hits": [
                    {
                        "text": "Grounded evidence for the missing step.",
                        "meta": {
                            "source_path": source_path,
                            "heading_path": "Abstract",
                        },
                    }
                ]
            }
        },
        conv_id="grounded-baseline",
    )[-1]

    assert rendered["rendered_body"] == "Original answer."
    assert rendered["cite_details"] == []


def test_structured_retry_keeps_system_a_and_system_b(monkeypatch) -> None:
    source_path = "db/paper/paper.en.md"
    marker = "[[CITE:s1234abcd:4]]"
    plan = {
        "budget": {"system_a": 1, "system_b": 1},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "evidence_quote": "Local evidence.",
            },
            {
                "preferred_system": "system_b",
                "source_path": source_path,
                "candidate_refs": [4],
                "candidate_cite_examples": [marker],
            },
        ],
    }
    monkeypatch.setattr(
        chat_render,
        "_should_link_inpaper_citations_for_message",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        chat_render,
        "_reading_guide_repair_missing_system_a_citations",
        lambda md, *_args, **_kwargs: md,
    )

    def fake_primary(md, _hits, **_kwargs):
        body = md.replace(marker, "")
        if "[1]" not in body:
            return body, []
        return body.replace("[1]", "[1](#kb-cite-a-1)"), [
            {
                "num": 1,
                "anchor": "kb-cite-a-1",
                "citation_route": "system_a",
                "source_path": source_path,
            }
        ]

    def fake_fallback(md, _hits, **_kwargs):
        return md.replace(marker, "[4](#kb-cite-b-4)"), [
            {
                "num": 4,
                "anchor": "kb-cite-b-4",
                "is_inpaper": True,
                "source_path": source_path,
                "title": "Upstream method",
            }
        ]

    monkeypatch.setattr(
        chat_render,
        "_annotate_inpaper_citations_with_hover_meta",
        fake_primary,
    )
    monkeypatch.setattr(
        chat_render,
        "_fallback_render_structured_citations",
        fake_fallback,
    )
    rendered = chat_render.enrich_messages_with_reference_render(
        [
            {"id": 1, "role": "user", "content": "Where did the method originate?"},
            {
                "id": 2,
                "role": "assistant",
                "content": f"Existing method {marker}; local evidence [1].",
                "meta": {
                    "answer_quality": {
                        "output_mode": "reading_guide",
                        "citation_plan": plan,
                    }
                },
            },
        ],
        {
            1: {
                "hits": [
                    {
                        "text": "Local evidence [4].",
                        "meta": {"source_path": source_path},
                    }
                ]
            }
        },
        conv_id="mixed-routes",
    )[-1]

    assert "[1](#kb-cite-a-1)" in rendered["rendered_body"]
    assert "[4](#kb-cite-b-4)" in rendered["rendered_body"]
    assert {detail["citation_route"] for detail in rendered["cite_details"]} == {
        "system_a",
        "system_b",
    }


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


def test_prose_preservation_ignores_citation_gap_before_closing_parenthesis() -> None:
    answer = "特别是表 1 中探测器的性能参数 [2]），可用于比较器件。"
    linked = (
        "特别是表 1 中探测器的性能参数 "
        '[2](#kb-cite-demo-2 "source: detector-review.pdf | ref 2")），可用于比较器件。'
    )

    assert chat_render._rendered_body_preserves_answer_body(
        answer_body=answer,
        rendered_body=linked,
        cite_details=[{"num": 2, "citation_route": "system_a"}],
    )
