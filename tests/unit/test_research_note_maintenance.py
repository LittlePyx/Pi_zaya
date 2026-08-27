from __future__ import annotations

from kb.research_note_maintenance import (
    enqueue_research_note_source_recovery,
    enqueue_research_note_source_recovery_scan,
    register_research_note_recovery_handler,
    unregister_research_note_recovery_handler,
    wait_for_research_note_recovery_idle,
)


def test_source_recovery_queue_deduplicates_paths_and_drains_in_background() -> None:
    calls: list[dict] = []

    def handler(*, source_paths, reason):
        calls.append({"source_paths": source_paths, "reason": reason})

    unregister_research_note_recovery_handler()
    register_research_note_recovery_handler(handler)
    try:
        assert enqueue_research_note_source_recovery(
            [r"C:\library\Paper.pdf", "c:/library/paper.pdf", r"C:\library\Paper.en.md"],
            reason="conversion_ready",
        ) is True
        assert wait_for_research_note_recovery_idle(timeout=2.0) is True
    finally:
        unregister_research_note_recovery_handler(handler)

    assert calls == [{
        "source_paths": (r"C:\library\Paper.pdf", r"C:\library\Paper.en.md"),
        "reason": "conversion_ready",
    }]


def test_source_recovery_scan_is_ignored_without_handler_and_uses_scan_marker() -> None:
    calls: list[dict] = []

    def handler(*, source_paths, reason):
        calls.append({"source_paths": source_paths, "reason": reason})

    unregister_research_note_recovery_handler()
    assert enqueue_research_note_source_recovery_scan(reason="library_reindex_ready") is False
    register_research_note_recovery_handler(handler)
    try:
        assert enqueue_research_note_source_recovery_scan(reason="library_reindex_ready") is True
        assert wait_for_research_note_recovery_idle(timeout=2.0) is True
    finally:
        unregister_research_note_recovery_handler(handler)

    assert calls == [{"source_paths": None, "reason": "library_reindex_ready"}]
