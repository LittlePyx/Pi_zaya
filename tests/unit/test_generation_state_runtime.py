from __future__ import annotations

from kb.generation_state_runtime import (
    _gen_store_answer,
    _gen_store_answer_contract_meta,
    _gen_store_answer_provenance,
    _gen_store_answer_quality_meta,
    _gen_store_answer_runtime_check_meta,
    _gen_store_paper_guide_contract_meta,
    _gen_store_partial,
    _gen_store_answer_provenance_async,
    _gen_has_active_task_id,
    _gen_has_running_for_conversation,
    _is_live_assistant_text,
    _live_assistant_task_id,
    _live_assistant_text,
    _should_run_provenance_async_refine,
    _strip_internal_generation_markers,
)


def test_live_assistant_helpers_roundtrip():
    text = _live_assistant_text("task-123", live_assistant_prefix="__LIVE__: ")

    assert _is_live_assistant_text(text, live_assistant_prefix="__LIVE__: ") is True
    assert _live_assistant_task_id(text, live_assistant_prefix="__LIVE__: ") == "task-123"
    assert _is_live_assistant_text("plain text", live_assistant_prefix="__LIVE__: ") is False


def test_strip_internal_generation_markers_preserves_cites():
    out = _strip_internal_generation_markers(
        "draft [[SUPPORT:DOC-1-S2]] still cites [[CITE:ref-1:12]]"
    )

    assert "SUPPORT:" not in out
    assert "[[CITE:ref-1:12]]" in out
    assert out == "draft still cites [[CITE:ref-1:12]]"


def test_gen_has_running_for_conversation_ignores_cancel_requested_task():
    from kb import runtime_state as RUNTIME

    session_id = "session-cancel-requested"
    with RUNTIME.GEN_LOCK:
        RUNTIME.GEN_TASKS[session_id] = {
            "id": "task-cancel-requested",
            "session_id": session_id,
            "conv_id": "conv-cancel-requested",
            "chat_db": "/tmp/chat.sqlite3",
            "status": "running",
            "answer_ready": False,
            "cancel": True,
            "created_at": 1.0,
            "updated_at": 1.0,
        }

    try:
        assert _gen_has_running_for_conversation(
            "conv-cancel-requested",
            chat_db_path="/tmp/chat.sqlite3",
        ) is False
    finally:
        with RUNTIME.GEN_LOCK:
            RUNTIME.GEN_TASKS.pop(session_id, None)


def test_gen_has_running_for_conversation_keeps_active_task_blocking():
    from kb import runtime_state as RUNTIME

    session_id = "session-active-running"
    with RUNTIME.GEN_LOCK:
        RUNTIME.GEN_TASKS[session_id] = {
            "id": "task-active-running",
            "session_id": session_id,
            "conv_id": "conv-active-running",
            "chat_db": "/tmp/chat.sqlite3",
            "status": "running",
            "answer_ready": False,
            "cancel": False,
            "created_at": 1.0,
            "updated_at": 1.0,
        }

    try:
        assert _gen_has_running_for_conversation(
            "conv-active-running",
            chat_db_path="/tmp/chat.sqlite3",
        ) is True
    finally:
        with RUNTIME.GEN_LOCK:
            RUNTIME.GEN_TASKS.pop(session_id, None)


def test_gen_has_active_task_id_detects_uncancelled_running_task():
    from kb import runtime_state as RUNTIME

    session_id = "session-active-by-task-id"
    with RUNTIME.GEN_LOCK:
        RUNTIME.GEN_TASKS[session_id] = {
            "id": "task-active-by-task-id",
            "session_id": session_id,
            "conv_id": "conv-active-by-task-id",
            "chat_db": "/tmp/chat.sqlite3",
            "status": "running",
            "answer_ready": False,
            "cancel": False,
            "created_at": 1.0,
            "updated_at": 1.0,
        }

    try:
        assert _gen_has_active_task_id("task-active-by-task-id") is True
        assert _gen_has_active_task_id("missing-task") is False
    finally:
        with RUNTIME.GEN_LOCK:
            RUNTIME.GEN_TASKS.pop(session_id, None)


def test_gen_store_partial_sanitizes_internal_support_markers():
    captured: dict[str, object] = {}

    class _FakeChatStore:
        def __init__(self, db_path):
            captured["db_path"] = str(db_path)

        def update_message_content(self, message_id: int, content: str) -> bool:
            captured["message_id"] = int(message_id)
            captured["content"] = content
            return True

    _gen_store_partial(
        {"chat_db": "/tmp/chat.db", "assistant_msg_id": 7},
        "partial [[SUPPORT:DOC-1]] with [[CITE:ref-1:5]]",
        chat_store_cls=_FakeChatStore,
    )

    assert captured["message_id"] == 7
    assert captured["content"] == "partial with [[CITE:ref-1:5]]"


def test_gen_store_answer_sanitizes_internal_support_markers_for_update_and_append():
    captured: dict[str, object] = {"append_calls": 0}

    class _FakeChatStore:
        def __init__(self, _db_path):
            pass

        def update_message_content(self, message_id: int, content: str) -> bool:
            captured["updated_message_id"] = int(message_id)
            captured["updated_content"] = content
            return False

        def append_message(self, conv_id: str, role: str, content: str) -> int:
            captured["append_calls"] = int(captured["append_calls"]) + 1
            captured["append_conv_id"] = conv_id
            captured["append_role"] = role
            captured["append_content"] = content
            return 99

    _gen_store_answer(
        {"chat_db": "/tmp/chat.db", "conv_id": "conv-1", "assistant_msg_id": 8},
        "final [[SUPPORT:DOC-2]] answer [[CITE:ref-2:9]]",
        chat_store_cls=_FakeChatStore,
    )

    assert captured["updated_message_id"] == 8
    assert captured["updated_content"] == "final answer [[CITE:ref-2:9]]"
    assert captured["append_calls"] == 1
    assert captured["append_conv_id"] == "conv-1"
    assert captured["append_role"] == "assistant"
    assert captured["append_content"] == "final answer [[CITE:ref-2:9]]"


def test_gen_store_answer_sanitizes_agent_trace_suffix_only_in_agent_mode():
    captured: dict[str, object] = {}

    class _FakeChatStore:
        def __init__(self, _db_path):
            pass

        def update_message_content(self, message_id: int, content: str) -> bool:
            captured["message_id"] = int(message_id)
            captured["content"] = content
            return True

    _gen_store_answer(
        {"chat_db": "/tmp/chat.db", "conv_id": "conv-1", "assistant_msg_id": 8, "agent_mode": True},
        "Final answer.\n\nResearch Agent Trace\nPlan\n- retrieve_evidence debug",
        chat_store_cls=_FakeChatStore,
    )

    assert captured["message_id"] == 8
    assert captured["content"] == "Final answer."

    _gen_store_answer(
        {"chat_db": "/tmp/chat.db", "conv_id": "conv-1", "assistant_msg_id": 9, "agent_mode": False},
        "Final answer.\n\nResearch Agent Trace\nPlan\n- user-requested example",
        chat_store_cls=_FakeChatStore,
    )

    assert captured["message_id"] == 9
    assert captured["content"] == "Final answer.\n\nResearch Agent Trace\nPlan\n- user-requested example"


def test_gen_store_partial_sanitizes_agent_trace_json_in_agent_mode():
    captured: dict[str, object] = {}

    class _FakeChatStore:
        def __init__(self, _db_path):
            pass

        def update_message_content(self, message_id: int, content: str) -> bool:
            captured["message_id"] = int(message_id)
            captured["content"] = content
            return True

    _gen_store_partial(
        {"chat_db": "/tmp/chat.db", "assistant_msg_id": 10, "agent_mode": True},
        'Streaming answer.\n\n```json\n{"agent_trace": {"mode": "research_agent"}}\n```',
        chat_store_cls=_FakeChatStore,
    )

    assert captured["message_id"] == 10
    assert captured["content"] == "Streaming answer."


def test_gen_store_paper_guide_contract_meta_sanitizes_agent_render_packet():
    captured: dict[str, object] = {}

    class _FakeChatStore:
        def __init__(self, _db_path):
            pass

        def merge_message_meta(self, message_id: int, patch: dict) -> bool:
            captured["message_id"] = int(message_id)
            captured["patch"] = patch
            return True

    polluted = "Clean answer.\n\nagent_trace: {debug: true}"
    _gen_store_paper_guide_contract_meta(
        {"chat_db": "/tmp/chat.db", "assistant_msg_id": 11, "agent_mode": True},
        paper_guide_contracts={
            "render_packet": {
                "answer_markdown": polluted,
                "rendered_body": polluted,
                "rendered_content": polluted,
                "copy_markdown": polluted,
                "copy_text": polluted,
                "notice": "notice stays",
            }
        },
        chat_store_cls=_FakeChatStore,
    )

    packet = captured["patch"]["paper_guide_contracts"]["render_packet"]
    assert captured["message_id"] == 11
    assert packet["answer_markdown"] == "Clean answer."
    assert packet["rendered_body"] == "Clean answer."
    assert packet["rendered_content"] == "Clean answer."
    assert packet["copy_markdown"] == "Clean answer."
    assert packet["copy_text"] == "Clean answer."
    assert packet["notice"] == "notice stays"


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


def test_gen_store_answer_runtime_check_meta_merges_payload():
    captured: dict[str, object] = {}

    class _FakeChatStore:
        def __init__(self, db_path):
            captured["db_path"] = str(db_path)

        def merge_message_meta(self, message_id: int, patch: dict) -> bool:
            captured["message_id"] = int(message_id)
            captured["patch"] = dict(patch)
            return True

    _gen_store_answer_runtime_check_meta(
        {
            "chat_db": "/tmp/chat.db",
            "assistant_msg_id": 13,
        },
        answer_runtime_check={
            "schema_version": 1,
            "status": "passed",
            "summary": {"failed": [], "needs_review_count": 0},
        },
        chat_store_cls=_FakeChatStore,
    )

    assert captured["message_id"] == 13
    patch = captured["patch"]
    assert isinstance(patch, dict)
    assert patch["answer_runtime_check"]["status"] == "passed"


def test_gen_store_answer_runtime_check_meta_skips_empty_payload():
    called = {"merge": 0}

    class _FakeChatStore:
        def __init__(self, _db_path):
            pass

        def merge_message_meta(self, message_id: int, patch: dict) -> bool:
            called["merge"] += 1
            return True

    _gen_store_answer_runtime_check_meta(
        {
            "chat_db": "/tmp/chat.db",
            "assistant_msg_id": 13,
        },
        answer_runtime_check={},
        chat_store_cls=_FakeChatStore,
    )

    assert called["merge"] == 0


def test_gen_store_answer_contract_meta_merges_payload():
    captured: dict[str, object] = {}

    class _FakeChatStore:
        def __init__(self, db_path):
            captured["db_path"] = str(db_path)

        def merge_message_meta(self, message_id: int, patch: dict) -> bool:
            captured["message_id"] = int(message_id)
            captured["patch"] = dict(patch)
            return True

    _gen_store_answer_contract_meta(
        {
            "chat_db": "/tmp/chat.db",
            "assistant_msg_id": 17,
        },
        answer_contract={
            "schema_version": 1,
            "source_summary": {"kind": "general_api"},
        },
        chat_store_cls=_FakeChatStore,
    )

    assert captured["message_id"] == 17
    patch = captured["patch"]
    assert isinstance(patch, dict)
    assert patch["answer_contract"]["source_summary"]["kind"] == "general_api"


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
