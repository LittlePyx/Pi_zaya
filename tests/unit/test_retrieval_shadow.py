from __future__ import annotations

import json
import threading
from pathlib import Path

from kb.retrieval_shadow import (
    DenseShadowRuntime,
    RetrievalShadowRunner,
    compare_shadow_rankings,
    schedule_product_retrieval_shadow,
)


def _hit(number: int, *, source: str = "doc-a") -> dict:
    return {
        "id": f"{source}:chunk-{number}",
        "score": float(100 - number),
        "text": f"private evidence {number}",
        "meta": {"source_path": f"C:/private/{source}.md"},
    }


class _FakeDenseRetriever:
    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits

    @property
    def is_empty(self) -> bool:
        return not self._hits

    def search(self, _query: str, top_k: int = 6) -> list[dict]:
        return list(self._hits[:top_k])


def test_compare_shadow_rankings_measures_prefix_overlap_and_source_escape() -> None:
    product = [_hit(index) for index in range(1, 11)]
    shadow = [*_hit_range(1, 8), _hit(11, source="doc-b"), _hit(12, source="doc-b")]

    result = compare_shadow_rankings(
        product_hits=product,
        shadow_hits=shadow,
        scoped=True,
    )

    assert result["protected_prefix_preserved"] is True
    assert result["changed_count_at_5"] == 0
    assert result["overlap_count_at_10"] == 8
    assert result["source_escape_count"] == 2


def _hit_range(start: int, stop: int) -> list[dict]:
    return [_hit(index) for index in range(start, stop + 1)]


def test_async_shadow_is_content_free_and_cannot_mutate_product(
    monkeypatch, tmp_path
) -> None:
    observation_path = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(observation_path))
    product = _hit_range(1, 10)
    original = json.loads(json.dumps(product))
    dense = [
        _hit(6),
        _hit(7),
        _hit(8),
        _hit(11, source="doc-b"),
        _hit(12, source="doc-c"),
    ]
    runner = RetrievalShadowRunner(
        lambda _db, _index: DenseShadowRuntime(
            retriever=_FakeDenseRetriever(dense),
            index_cache_hit=False,
            cold_start=True,
        )
    )

    submission = runner.schedule(
        query="private paper question",
        product_hits=product,
        db_dir=tmp_path,
        index_dir=tmp_path,
        channel="product_chat_shadow",
        source_scope="library",
        scoped=False,
    )
    result = submission.future.result(timeout=3) if submission.future else {}

    assert submission.status == "submitted"
    assert result["status"] == "done"
    assert result["protected_prefix_preserved"] is True
    assert result["dense_novelty_count"] == 2
    assert result["product_output_unchanged"] is True
    assert product == original
    stored = observation_path.read_text(encoding="utf-8")
    assert "private paper question" not in stored
    assert "private evidence" not in stored
    assert "C:/private" not in stored
    attributes = json.loads(stored)["attributes"]
    assert attributes["outcome_id"] == "retrieval_shadow.compared"
    assert attributes["changed_count_at_5"] == 0


def test_scoped_shadow_filters_dense_candidates_to_product_sources(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(tmp_path / "scoped.jsonl"))
    product = _hit_range(1, 10)
    dense = [_hit(11, source="doc-b"), _hit(12, source="doc-c"), _hit(7)]
    runner = RetrievalShadowRunner(
        lambda _db, _index: DenseShadowRuntime(
            retriever=_FakeDenseRetriever(dense),
            index_cache_hit=True,
            cold_start=False,
        )
    )

    submission = runner.schedule(
        query="scoped private question",
        product_hits=product,
        db_dir=tmp_path,
        index_dir=tmp_path,
        channel="controlled_replay_shadow",
        source_scope="current_paper",
        scoped=True,
    )
    result = submission.future.result(timeout=3) if submission.future else {}

    assert result["status"] == "done"
    assert result["source_escape_count"] == 0
    assert result["protected_prefix_preserved"] is True


def test_shadow_runtime_failure_is_redacted_and_fail_open(
    monkeypatch, tmp_path
) -> None:
    observation_path = tmp_path / "failure.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(observation_path))
    product = _hit_range(1, 10)

    def fail(_db: Path, _index: Path) -> DenseShadowRuntime:
        raise RuntimeError("private query C:/private/paper.md")

    submission = RetrievalShadowRunner(fail).schedule(
        query="private query",
        product_hits=product,
        db_dir=tmp_path,
        index_dir=tmp_path,
        channel="product_chat_shadow",
        source_scope="library",
        scoped=False,
    )
    result = submission.future.result(timeout=3) if submission.future else {}

    assert result == {
        "status": "error",
        "fallback_reason_id": "dense_runtime_unavailable",
        "duration_ms": result["duration_ms"],
        "product_output_unchanged": True,
    }
    stored = observation_path.read_text(encoding="utf-8")
    assert "private query" not in stored
    assert "paper.md" not in stored
    assert (
        json.loads(stored)["attributes"]["fallback_reason_id"]
        == "dense_runtime_unavailable"
    )


def test_shadow_skips_when_single_worker_is_busy(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(tmp_path / "busy.jsonl"))
    entered = threading.Event()
    release = threading.Event()

    def wait_loader(_db: Path, _index: Path) -> DenseShadowRuntime:
        entered.set()
        assert release.wait(timeout=3)
        return DenseShadowRuntime(_FakeDenseRetriever([]), False, True)

    runner = RetrievalShadowRunner(wait_loader)
    first = runner.schedule(
        query="one",
        product_hits=_hit_range(1, 10),
        db_dir=tmp_path,
        index_dir=tmp_path,
        channel="controlled_replay_shadow",
        source_scope="library",
        scoped=False,
    )
    assert entered.wait(timeout=3)
    second = runner.schedule(
        query="two",
        product_hits=_hit_range(1, 10),
        db_dir=tmp_path,
        index_dir=tmp_path,
        channel="controlled_replay_shadow",
        source_scope="library",
        scoped=False,
    )
    release.set()
    assert first.future is not None
    first.future.result(timeout=3)

    assert second.status == "busy"
    events = [
        json.loads(line) for line in (tmp_path / "busy.jsonl").read_text().splitlines()
    ]
    assert {event["attributes"]["outcome_id"] for event in events} == {
        "retrieval_shadow.compared",
        "retrieval_shadow.skipped",
    }


def test_product_shadow_activation_is_explicit_and_missing_index_fails_open(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("KB_DENSE_SHADOW_ENABLED", raising=False)
    assert (
        schedule_product_retrieval_shadow(
            query="private",
            product_hits=[],
            db_dir=tmp_path,
            source_scope="library",
        ).status
        == "disabled"
    )

    observation_path = tmp_path / "activation.jsonl"
    monkeypatch.setenv("KB_DENSE_SHADOW_ENABLED", "1")
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(observation_path))
    monkeypatch.setenv("KB_DENSE_SHADOW_INDEX_DIR", str(tmp_path / "missing"))
    result = schedule_product_retrieval_shadow(
        query="private",
        product_hits=[],
        db_dir=tmp_path,
        source_scope="library",
    )
    assert result.status == "index_missing"
    stored = observation_path.read_text(encoding="utf-8")
    assert "private" not in stored
    assert (
        json.loads(stored)["attributes"]["fallback_reason_id"] == "dense_index_missing"
    )
