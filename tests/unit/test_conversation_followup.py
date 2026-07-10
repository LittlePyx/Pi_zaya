from __future__ import annotations

from kb.conversation_followup import (
    build_answer_audit_scope_block,
    filter_hits_to_source_hints,
    order_hits_by_source_hints,
    previous_assistant_reference_hits,
    previous_assistant_source_hints,
)


def test_previous_assistant_source_hints_uses_latest_answer_contract() -> None:
    messages = [
        {"id": 1, "role": "assistant", "meta": {"canonical_hit_paths": ["db/old.md"]}},
        {
            "id": 3,
            "role": "assistant",
            "meta": {
                "canonical_hit_paths": ["db/paper-a.md"],
                "paper_guide_contracts": {
                    "doc_list": [{"source_path": "db/paper-b.md"}],
                },
            },
        },
        {"id": 4, "role": "user", "content": "Audit the previous answer."},
    ]

    assert previous_assistant_source_hints(messages, before_message_id=4) == ["db/paper-b.md"]


def test_previous_assistant_source_hints_recovers_historical_noncontiguous_answer_sources() -> None:
    messages = [
        {
            "id": 20,
            "role": "assistant",
            "content": "Paper A [2], Paper B [6], Paper C [3], and Paper D [5].",
            "meta": {
                "canonical_hit_paths": [
                    "db/candidate-1.md",
                    "db/paper-a.md",
                    "db/paper-c.md",
                    "db/candidate-4.md",
                    "db/paper-d.md",
                    "db/paper-b.md",
                ],
                "paper_guide_contracts": {
                    "doc_list": [
                        {"source_path": "db/candidate-1.md"},
                        {"source_path": "db/paper-a.md"},
                        {"source_path": "db/paper-c.md"},
                        {"source_path": "db/candidate-4.md"},
                    ]
                },
            },
        }
    ]

    hints = previous_assistant_source_hints(messages, before_message_id=21)

    assert hints == [
        "db/paper-a.md",
        "db/paper-b.md",
        "db/paper-c.md",
        "db/paper-d.md",
    ]


def test_previous_assistant_source_hints_falls_back_to_canonical_paths_without_final_contract() -> None:
    messages = [
        {"id": 1, "role": "assistant", "meta": {"canonical_hit_paths": ["db/paper-a.md"]}},
        {"id": 2, "role": "user", "content": "Audit the previous answer."},
    ]

    assert previous_assistant_source_hints(messages, before_message_id=2) == ["db/paper-a.md"]


def test_previous_assistant_reference_hits_uses_refs_owned_by_previous_user_turn() -> None:
    messages = [
        {"id": 10, "role": "user", "content": "Build a route."},
        {"id": 11, "role": "assistant", "content": "Route answer."},
        {"id": 12, "role": "user", "content": "Audit it."},
    ]
    refs = {
        10: {
            "hits": [
                {"text": "A evidence", "meta": {"source_path": "db/paper-a.md"}},
                {"text": "B evidence", "meta": {"source_path": "db/paper-b.md"}},
            ]
        }
    }

    hits = previous_assistant_reference_hits(
        messages,
        refs,
        before_message_id=12,
        source_hints=["db/paper-b.md"],
    )

    assert [hit["text"] for hit in hits] == ["B evidence"]


def test_filter_hits_to_source_hints_excludes_unrelated_papers() -> None:
    hits = [
        {"text": "A", "meta": {"source_path": "db/paper-a.md"}},
        {"text": "unrelated", "meta": {"source_path": "db/visual-perception.md"}},
    ]

    filtered_hits, filtered_scores = filter_hits_to_source_hints(
        hits,
        [3.0, 2.9],
        ["db/paper-a.md"],
    )

    assert [item["text"] for item in filtered_hits] == ["A"]
    assert filtered_scores == [3.0]


def test_filter_hits_to_source_hints_can_fail_closed_for_answer_audits() -> None:
    filtered_hits, filtered_scores = filter_hits_to_source_hints(
        [{"text": "unrelated", "meta": {"source_path": "db/unrelated.md"}}],
        [3.0],
        ["db/paper-a.md"],
        fallback_to_original=False,
    )

    assert filtered_hits == []
    assert filtered_scores == []


def test_order_hits_by_source_hints_preserves_every_authoritative_audit_source() -> None:
    hits = [
        {"text": "B", "meta": {"source_path": "db/paper-b.md"}},
        {"text": "unrelated", "meta": {"source_path": "db/unrelated.md"}},
        {"text": "A", "meta": {"source_path": "db/paper-a.md"}},
        {"text": "C", "meta": {"source_path": "db/paper-c.md"}},
    ]

    ordered = order_hits_by_source_hints(
        hits,
        ["db/paper-a.md", "db/paper-b.md", "db/paper-c.md"],
    )

    assert [hit["text"] for hit in ordered] == ["A", "B", "C"]


def test_answer_audit_scope_forbids_replacing_previous_answer() -> None:
    block = build_answer_audit_scope_block(["db/paper-a.md"])

    assert "Audit the previous assistant answer" in block
    assert "do not replace it with a new reading route" in block
    assert "Do not audit internal citation offsets" in block
    assert "never expose DOC-k labels" in block
    assert "paper-a.md" in block
    assert "Authoritative previous-answer source count: 1" in block
