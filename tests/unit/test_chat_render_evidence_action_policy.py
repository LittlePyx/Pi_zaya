from __future__ import annotations

from api.chat_render import enrich_messages_with_reference_render


def _local_ref_pack() -> dict:
    return {
        "hits": [
            {
                "text": "The local paper reports a five decibel improvement.",
                "score": 9.0,
                "meta": {"source_path": "library/paper.md", "heading_path": "Results"},
                "ui_meta": {
                    "reader_open": {
                        "sourcePath": "library/paper.md",
                        "headingPath": "Results",
                        "highlightSnippet": "five decibel improvement",
                    }
                },
            }
        ]
    }


def test_external_answer_cannot_inherit_local_citations_or_reader_actions() -> None:
    messages = [
        {"id": 1, "role": "user", "content": "Explain an unrelated model concept."},
        {
            "id": 2,
            "role": "assistant",
            "content": "This is model background, not evidence from the open paper [1].",
            "meta": {
                "agent_source_summary": {
                    "kind": "external_not_kb",
                    "source_policy_payload": {
                        "kind": "external_not_kb",
                        "uses_local_knowledge_base": False,
                    },
                },
                "provenance": {
                    "source_path": "library/paper.md",
                    "segments": [{"evidence_mode": "direct", "locate_policy": "required"}],
                },
                "paper_guide_contracts": {
                    "primary_evidence": {"sourcePath": "library/paper.md"},
                    "render_packet": {
                        "answer_markdown": "This is model background, not evidence from the open paper [1].",
                        "cite_details": [{"num": 1, "sourcePath": "library/paper.md"}],
                        "reader_open": {"sourcePath": "library/paper.md"},
                        "locate_target": {"sourcePath": "library/paper.md"},
                    },
                },
            },
            "provenance": {
                "source_path": "library/paper.md",
                "segments": [{"evidence_mode": "direct", "locate_policy": "required"}],
            },
        },
    ]

    rendered = enrich_messages_with_reference_render(
        messages,
        {1: _local_ref_pack()},
        conv_id="external-policy",
    )
    answer = rendered[1]

    assert answer.get("cite_details") == []
    assert "provenance" not in answer
    meta = answer["meta"]
    assert "provenance" not in meta
    packet = meta["paper_guide_contracts"]["render_packet"]
    assert packet.get("cite_details") == []
    assert packet.get("reader_open") == {}
    assert packet.get("locate_target") == {}
    assert "primary_evidence" not in meta["paper_guide_contracts"]


def test_failed_quality_answer_cannot_expose_evidence_actions() -> None:
    messages = [
        {"id": 3, "role": "user", "content": "Was this validated clinically?"},
        {
            "id": 4,
            "role": "assistant",
            "content": "Validation scope:\n\nPlease inspect the paper yourself. [1]",
            "meta": {
                "answer_quality": {"minimum_ok": False},
                "agent_source_summary": {"kind": "local_kb"},
            },
        },
    ]

    rendered = enrich_messages_with_reference_render(
        messages,
        {3: _local_ref_pack()},
        conv_id="failed-quality-policy",
    )

    assert rendered[1].get("cite_details") == []
    assert "provenance" not in rendered[1]
