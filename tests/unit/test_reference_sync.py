from __future__ import annotations

from pathlib import Path

from kb import reference_sync


def _reset_refsync_state() -> None:
    with reference_sync._LOCK:
        reference_sync._THREAD = None
        reference_sync._STATE.update(
            {
                "running": False,
                "status": "idle",
                "stage": "",
                "message": "",
                "error": "",
                "run_id": 0,
                "current": "",
                "docs_done": 0,
                "docs_total": 0,
                "refs_total": 0,
                "refs_with_doi": 0,
                "refs_crossref_ok": 0,
                "refs_source_map_ok": 0,
                "stats": {},
                "started_at": 0.0,
                "finished_at": 0.0,
                "updated_at": 0.0,
            }
        )


def test_progress_stats_from_payload_keeps_top_level_ready_counters() -> None:
    payload = {
        "docs_total": 21,
        "docs_indexed": 6,
        "refs_total": 1304,
        "refs_metadata_ready": 387,
        "refs_metadata_user_ready": 416,
        "refs_crossref_ok": 140,
        "stats": {
            "crossref_network_attempts": 12,
            "docs_indexed": 5,
            "refs_metadata_user_ready": 415,
        },
    }

    stats = reference_sync._progress_stats_from_payload(payload)

    assert stats["docs_total"] == 21
    assert stats["docs_indexed"] == 6
    assert stats["refs_total"] == 1304
    assert stats["refs_metadata_ready"] == 387
    assert stats["refs_metadata_user_ready"] == 416
    assert stats["refs_crossref_ok"] == 140
    assert stats["crossref_network_attempts"] == 12


def test_fmt_done_message_uses_user_ready_and_chinese_summary() -> None:
    msg = reference_sync._fmt_done_message(
        {
            "docs_indexed": 21,
            "refs_total": 1304,
            "refs_metadata_ready": 387,
            "refs_metadata_user_ready": 416,
            "refs_crossref_ok": 140,
            "refs_action_non_article_ok": 26,
            "refs_action_source_repair": 12,
            "crossref_network_attempts": 88,
        }
    )

    assert "参考文献索引已更新" in msg
    assert "元数据就绪 416/1304" in msg
    assert "联网补齐 140" in msg
    assert "待人工处理 12" in msg


def test_start_reference_sync_clears_running_state_when_thread_start_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _reset_refsync_state()

    class FailingThread:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("thread pool unavailable")

    monkeypatch.setattr(reference_sync.threading, "Thread", FailingThread)

    try:
        result = reference_sync.start_reference_sync(
            src_root=tmp_path / "md",
            db_dir=tmp_path / "db",
            crossref_time_budget_s=5.0,
            doi_prefetch_workers=1,
        )
        snap = reference_sync.snapshot()

        assert result["started"] is False
        assert result["reason"] == "thread_start_failed"
        assert result["run_id"] == 1
        assert snap["running"] is False
        assert snap["status"] == "error"
        assert snap["stage"] == "error"
        assert "参考文献后台同步无法启动" in snap["message"]
        assert "thread_start_failed" in snap["error"]
    finally:
        _reset_refsync_state()


def test_start_reference_sync_can_retry_after_thread_start_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _reset_refsync_state()
    thread_names: list[str] = []

    class FailingThread:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("thread pool unavailable")

    class NoopThread:
        def __init__(self, *args, **kwargs) -> None:
            thread_names.append(str(kwargs.get("name") or ""))

        def start(self) -> None:
            return None

    try:
        monkeypatch.setattr(reference_sync.threading, "Thread", FailingThread)
        failed = reference_sync.start_reference_sync(
            src_root=tmp_path / "md",
            db_dir=tmp_path / "db",
            crossref_time_budget_s=5.0,
            doi_prefetch_workers=1,
        )

        monkeypatch.setattr(reference_sync.threading, "Thread", NoopThread)
        retried = reference_sync.start_reference_sync(
            src_root=tmp_path / "md",
            db_dir=tmp_path / "db",
            crossref_time_budget_s=5.0,
            doi_prefetch_workers=1,
        )
        snap = reference_sync.snapshot()

        assert failed["started"] is False
        assert retried["started"] is True
        assert retried["run_id"] == 2
        assert snap["running"] is True
        assert snap["status"] == "running"
        assert thread_names == ["kb-ref-sync-2"]
    finally:
        _reset_refsync_state()
