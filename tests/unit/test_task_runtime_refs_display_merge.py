from __future__ import annotations

from kb.task_runtime import _merge_refs_display_docs_with_answer_hits


def _hit(source_path: str, text: str) -> dict:
    return {"text": text, "meta": {"source_path": source_path, "ref_pack_state": "pending"}}


def test_refs_display_docs_prefer_answer_sources_and_deduplicate():
    merged = _merge_refs_display_docs_with_answer_hits(
        refs_seed_docs=[_hit("db/other.en.md", "other"), _hit("db/answer.en.md", "stale")],
        answer_hits=[_hit("db/answer.en.md", "answer"), _hit("db/second.en.md", "second")],
        limit=3,
    )

    assert [item["meta"]["source_path"] for item in merged] == [
        "db/answer.en.md",
        "db/second.en.md",
        "db/other.en.md",
    ]
    assert all(item["meta"]["ref_pack_state"] == "ready" for item in merged)
    assert [item["meta"].get("ref_display_reason", "") for item in merged] == [
        "answer_hit_top",
        "answer_hit_top",
        "",
    ]


def test_refs_display_docs_only_force_cited_answer_sources_when_answer_has_citations():
    merged = _merge_refs_display_docs_with_answer_hits(
        refs_seed_docs=[_hit("db/seed.en.md", "seed")],
        answer_hits=[
            _hit("db/first.en.md", "first"),
            _hit("db/second.en.md", "second"),
            _hit("db/third.en.md", "third"),
        ],
        limit=4,
        answer="Only [1] and [3] are cited.",
    )

    assert [item["meta"]["source_path"] for item in merged] == [
        "db/first.en.md",
        "db/third.en.md",
        "db/seed.en.md",
    ]


def test_refs_display_docs_preserve_all_non_contiguous_sources_selected_by_answer():
    answer_hits = [_hit(f"db/paper-{idx}.en.md", f"paper {idx}") for idx in range(1, 7)]

    merged = _merge_refs_display_docs_with_answer_hits(
        refs_seed_docs=answer_hits[:4],
        answer_hits=answer_hits,
        limit=4,
        answer="Route uses [2], [6], [3], and [5].",
    )

    assert [item["meta"]["source_path"] for item in merged] == [
        "db/paper-2.en.md",
        "db/paper-3.en.md",
        "db/paper-5.en.md",
        "db/paper-6.en.md",
    ]
