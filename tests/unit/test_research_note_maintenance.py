from __future__ import annotations

from kb.research_note_maintenance import (
    dispatch_research_note_source_migration,
    enqueue_research_note_source_recovery,
    enqueue_research_note_source_recovery_scan,
    register_research_note_recovery_handler,
    register_research_note_source_migration_handler,
    unregister_research_note_recovery_handler,
    unregister_research_note_source_migration_handler,
    wait_for_research_note_recovery_idle,
)


def test_source_migration_dispatch_is_synchronous_and_reports_handler_failures() -> None:
    calls: list[dict] = []

    def handler(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "notes_updated": 2, "links_updated": 3}

    unregister_research_note_source_migration_handler()
    inactive = dispatch_research_note_source_migration(
        exact_path_migrations={"old.pdf": "new.pdf"},
        prefix_path_migrations=[],
        reason="library_source_renamed",
    )
    assert inactive["handled"] is False

    register_research_note_source_migration_handler(handler)
    try:
        handled = dispatch_research_note_source_migration(
            exact_path_migrations={"old.pdf": "new.pdf"},
            prefix_path_migrations=[("old", "new")],
            reason="library_source_renamed",
        )
    finally:
        unregister_research_note_source_migration_handler(handler)

    assert handled == {
        "ok": True,
        "notes_updated": 2,
        "links_updated": 3,
        "handled": True,
    }
    assert calls[0]["exact_path_migrations"] == {"old.pdf": "new.pdf"}


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
