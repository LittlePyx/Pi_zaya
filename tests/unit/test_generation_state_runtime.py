from __future__ import annotations

from kb.generation_state_runtime import (
    _gen_store_answer_provenance,
    _gen_store_answer_quality_meta,
    _gen_store_paper_guide_contract_meta,
    _gen_store_answer_provenance_async,
    _is_live_assistant_text,
    _live_assistant_task_id,
    _live_assistant_text,
    _should_run_provenance_async_refine,
)


def test_live_assistant_helpers_roundtrip():
    text = _live_assistant_text("task-123", live_assistant_prefix="__LIVE__: ")

    assert _is_live_assistant_text(text, live_assistant_prefix="__LIVE__: ") is True
    assert _live_assistant_task_id(text, live_assistant_prefix="__LIVE__: ") == "task-123"
    assert _is_live_assistant_text("plain text", live_assistant_prefix="__LIVE__: ") is False


def test_should_run_provenance_async_refine_requires_flags_and_api_key():
    class _Settings:
        api_key = "test-key"

    task = {
        "paper_guide_mode": True,
        "paper_guide_bound_source_path": "/tmp/demo.pdf",
        "llm_rerank": True,
        "settings_obj": _Settings(),
    }

    assert _should_run_provenance_async_refine(task, environ={"KB_PROVENANCE_ASYNC_LLM": "1"}) is True
    assert _should_run_provenance_async_refine(task, environ={"KB_PROVENANCE_ASYNC_LLM": "0"}) is False
    assert _should_run_provenance_async_refine({**task, "llm_rerank": False}, environ={"KB_PROVENANCE_ASYNC_LLM": "1"}) is False
    assert _should_run_provenance_async_refine({**task, "settings_obj": object()}, environ={"KB_PROVENANCE_ASYNC_LLM": "1"}) is False


def test_gen_store_answer_provenance_async_enables_llm_rerank():
    captured: dict[str, object] = {}

    def _fake_store(
        task: dict,
        *,
        answer: str,
        answer_hits: list[dict],
        support_resolution: list[dict] | None = None,
    ) -> None:
        captured["task"] = dict(task)
        captured["answer"] = answer
        captured["answer_hits"] = list(answer_hits)
        captured["support_resolution"] = list(support_resolution or [])

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, name=None):
            self._target = target

        def start(self):
            if callable(self._target):
                self._target()

    class _ThreadingModule:
        Thread = _ImmediateThread

    _gen_store_answer_provenance_async(
        {"llm_rerank": False, "paper_guide_mode": True, "paper_guide_bound_source_path": "/tmp/demo.pdf"},
        answer="demo answer",
        answer_hits=[{"text": "x"}],
        store_answer_provenance=_fake_store,
        threading_module=_ThreadingModule,
    )

    assert captured["answer"] == "demo answer"
    assert captured["answer_hits"] == [{"text": "x"}]
    assert captured["support_resolution"] == []
    assert isinstance(captured["task"], dict)
    assert captured["task"].get("llm_rerank") is True


def test_gen_store_answer_provenance_forwards_primary_evidence():
    captured: dict[str, object] = {}

    class _FakeChatStore:
        def __init__(self, _db_path):
            pass

        def merge_message_meta(self, message_id: int, patch: dict) -> bool:
            captured["message_id"] = int(message_id)
            captured["patch"] = dict(patch)
            return True

    def _fake_build_answer_provenance(**kwargs):
        captured["build_kwargs"] = dict(kwargs)
        return {
            "status": "ready",
            "primary_evidence": dict(kwargs.get("primary_evidence") or {}),
        }

    _gen_store_answer_provenance(
        {
            "paper_guide_mode": True,
            "paper_guide_bound_source_path": "/tmp/demo.pdf",
            "paper_guide_bound_source_name": "demo.pdf",
            "chat_db": "/tmp/chat.db",
            "assistant_msg_id": 11,
            "db_dir": "/tmp/db",
        },
        answer="grounded answer",
        answer_hits=[{"text": "APR uses phase correlation"}],
        primary_evidence={
            "source_name": "demo.pdf",
            "block_id": "b-7",
            "heading_path": "Methods / APR",
            "snippet": "APR uses phase correlation for registration.",
        },
        chat_store_cls=_FakeChatStore,
        build_answer_provenance=_fake_build_answer_provenance,
    )

    build_kwargs = dict(captured["build_kwargs"])
    assert build_kwargs["primary_evidence"]["block_id"] == "b-7"
    assert build_kwargs["primary_evidence"]["heading_path"] == "Methods / APR"
    patch = dict(captured["patch"])
    assert patch["provenance"]["primary_evidence"]["snippet"] == "APR uses phase correlation for registration."
    assert captured["message_id"] == 11


def test_gen_store_answer_quality_meta_merges_payload():
    captured: dict[str, object] = {}

    class _FakeChatStore:
        def __init__(self, db_path):
            captured["db_path"] = str(db_path)

        def merge_message_meta(self, message_id: int, patch: dict) -> bool:
            captured["message_id"] = int(message_id)
            captured["patch"] = dict(patch)
            return True

    _gen_store_answer_quality_meta(
        {
            "chat_db": "/tmp/chat.db",
            "assistant_msg_id": 9,
        },
        answer_quality={
            "minimum_ok": True,
            "retrieval_confidence": {
                "low_confidence": True,
                "low_confidence_reason": "strict_family_sparse_hits",
                "candidate_refs_for_notice": [4, 9],
            },
        },
        chat_store_cls=_FakeChatStore,
    )

    assert captured["message_id"] == 9
    assert isinstance(captured["patch"], dict)
    patch = captured["patch"]
    assert "answer_quality" in patch
    assert isinstance(patch["answer_quality"], dict)
    assert patch["answer_quality"]["minimum_ok"] is True
    assert patch["answer_quality"]["retrieval_confidence"]["candidate_refs_for_notice"] == [4, 9]


def test_gen_store_answer_quality_meta_skips_empty_payload():
    called = {"merge": 0}

    class _FakeChatStore:
        def __init__(self, _db_path):
            pass

        def merge_message_meta(self, message_id: int, patch: dict) -> bool:
            called["merge"] += 1
            return True

    _gen_store_answer_quality_meta(
        {
            "chat_db": "/tmp/chat.db",
            "assistant_msg_id": 9,
        },
        answer_quality={},
        chat_store_cls=_FakeChatStore,
    )

    assert called["merge"] == 0


def test_gen_store_paper_guide_contract_meta_merges_payload():
    captured: dict[str, object] = {}

    class _FakeChatStore:
        def __init__(self, db_path):
            captured["db_path"] = str(db_path)

        def merge_message_meta(self, message_id: int, patch: dict) -> bool:
            captured["message_id"] = int(message_id)
            captured["patch"] = dict(patch)
            return True

    _gen_store_paper_guide_contract_meta(
        {
            "chat_db": "/tmp/chat.db",
            "assistant_msg_id": 12,
        },
        paper_guide_contracts={
            "version": 1,
            "intent": {"family": "method"},
            "support_pack": {"family": "method", "support_records": [{"support_id": "slot-1"}]},
        },
        chat_store_cls=_FakeChatStore,
    )

    assert captured["message_id"] == 12
    patch = captured["patch"]
    assert "paper_guide_contracts" in patch
    assert patch["paper_guide_contracts"]["version"] == 1
    assert patch["paper_guide_contracts"]["intent"]["family"] == "method"
    assert patch["paper_guide_contracts"]["support_pack"]["support_records"][0]["support_id"] == "slot-1"


def test_gen_store_paper_guide_contract_meta_skips_empty_payload():
    called = {"merge": 0}

    class _FakeChatStore:
        def __init__(self, _db_path):
            pass

        def merge_message_meta(self, message_id: int, patch: dict) -> bool:
            called["merge"] += 1
            return True

    _gen_store_paper_guide_contract_meta(
        {
            "chat_db": "/tmp/chat.db",
            "assistant_msg_id": 12,
        },
        paper_guide_contracts={},
        chat_store_cls=_FakeChatStore,
    )

    assert called["merge"] == 0
