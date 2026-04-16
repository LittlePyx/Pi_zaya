from __future__ import annotations

import sqlite3

import api.routers.references as references_router


class _FakeStore:
    def __init__(self, conversation: dict, refs: dict) -> None:
        self._conversation = dict(conversation)
        self._refs = refs

    def get_conversation(self, conv_id: str):
        del conv_id
        return dict(self._conversation)

    def list_message_refs(self, conv_id: str):
        del conv_id
        return self._refs


def test_get_conversation_refs_reuses_cached_payload_when_signature_is_unchanged(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()
    refs = {
        1: {
            "prompt": "Which paper discusses dynamic supersampling?",
            "hits": [
                {"text": "hit", "meta": {"source_path": r"db\SciAdv-2017\SciAdv-2017.en.md", "ref_pack_state": "ready"}}
            ],
        }
    }
    store = _FakeStore({"mode": "chat"}, refs)
    calls = {"n": 0}

    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(references_router, "_pdf_dir", lambda: None)
    monkeypatch.setattr(references_router, "_md_dir", lambda: None)
    monkeypatch.setattr(references_router, "_lib_store", lambda: None)
    monkeypatch.setattr(references_router, "_warm_conversation_refs_payload_async", lambda **kwargs: None)

    def fake_enrich_refs_payload(*args, **kwargs):
        del args, kwargs
        calls["n"] += 1
        return {1: {"hits": [{"ui_meta": {"summary_line": "cached"}}]}}

    monkeypatch.setattr(references_router, "enrich_refs_payload", fake_enrich_refs_payload)

    out1 = references_router.get_conversation_refs("conv-1")
    out2 = references_router.get_conversation_refs("conv-1")

    assert out1 == out2
    assert calls["n"] == 1


def test_get_conversation_refs_invalidates_cache_when_refs_change(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()
    refs = {
        1: {
            "prompt": "Which paper discusses dynamic supersampling?",
            "hits": [
                {"text": "hit-a", "meta": {"source_path": r"db\SciAdv-2017\SciAdv-2017.en.md", "ref_pack_state": "ready"}}
            ],
        }
    }
    store = _FakeStore({"mode": "chat"}, refs)
    calls = {"n": 0}

    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(references_router, "_pdf_dir", lambda: None)
    monkeypatch.setattr(references_router, "_md_dir", lambda: None)
    monkeypatch.setattr(references_router, "_lib_store", lambda: None)
    monkeypatch.setattr(references_router, "_warm_conversation_refs_payload_async", lambda **kwargs: None)

    def fake_enrich_refs_payload(*args, **kwargs):
        del args, kwargs
        calls["n"] += 1
        return {calls["n"]: {"hits": [{"ui_meta": {"summary_line": f"run-{calls['n']}"}}]}}

    monkeypatch.setattr(references_router, "enrich_refs_payload", fake_enrich_refs_payload)

    out1 = references_router.get_conversation_refs("conv-2")
    refs[1]["hits"][0]["text"] = "hit-b"
    refs[1]["updated_at"] = 2.0
    out2 = references_router.get_conversation_refs("conv-2")

    assert out1 != out2
    assert calls["n"] == 2


def test_get_conversation_refs_returns_fast_pending_payload_without_enrich(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()
    refs = {
        7: {
            "prompt": "Which paper discusses ADMM?",
            "hits": [
                {"text": "pending-a", "meta": {"source_path": r"db\A\A.en.md", "ref_pack_state": "pending"}},
                {"text": "pending-b", "meta": {"source_path": r"db\B\B.en.md", "ref_pack_state": "pending"}},
            ],
        }
    }
    store = _FakeStore({"mode": "chat"}, refs)

    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(references_router, "_warm_conversation_refs_payload_async", lambda **kwargs: None)

    def fail_enrich(*args, **kwargs):
        raise AssertionError("pending refs should bypass expensive enrich")

    monkeypatch.setattr(references_router, "enrich_refs_payload", fail_enrich)

    out = references_router.get_conversation_refs("conv-pending")

    assert bool((out.get(7) or {}).get("pending")) is True
    assert int((out.get(7) or {}).get("pending_hit_count") or 0) == 2
    assert list((out.get(7) or {}).get("hits") or []) == []


def test_get_conversation_refs_pending_payload_includes_pack_primary_evidence(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()
    refs = {
        8: {
            "prompt": "Which paper directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
            "hits": [
                {
                    "text": "Section 2.2 explicitly compares Hadamard single-pixel imaging and Fourier single-pixel imaging.",
                    "meta": {
                        "source_path": r"db\OE-2017\OE-2017.en.md",
                        "ref_pack_state": "pending",
                        "ref_best_heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                    },
                }
            ],
        }
    }
    store = _FakeStore({"mode": "chat"}, refs)

    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(references_router, "_warm_conversation_refs_payload_async", lambda **kwargs: None)
    monkeypatch.setattr(references_router, "enrich_refs_payload", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("pending refs should bypass expensive enrich")))

    out = references_router.get_conversation_refs("conv-pending-primary")

    pack = out[8]
    assert pack["primary_evidence"]["heading_path"] == "2. Comparison of theory / 2.2 Basis patterns generation"
    assert pack["primary_evidence"]["selection_reason"] == "pending_section_seed"
    assert pack["primary_evidence_heading_path"] == "2. Comparison of theory / 2.2 Basis patterns generation"
    assert pack["hits"][0]["ui_meta"]["reader_open"]["primaryEvidence"]["heading_path"] == "2. Comparison of theory / 2.2 Basis patterns generation"


def test_get_conversation_refs_falls_back_to_cached_payload_when_refs_db_is_busy(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()

    class _BusyStore:
        def get_conversation(self, conv_id: str):
            del conv_id
            return {"mode": "chat"}

        def list_message_refs(self, conv_id: str, *, timeout_s=None):
            del conv_id, timeout_s
            raise sqlite3.OperationalError("database is locked")

    cached_payload = {9: {"hits": [{"ui_meta": {"summary_line": "cached"}}]}}
    references_router._store_cached_conversation_refs_payload(
        conv_id="conv-busy",
        signature="sig",
        payload=cached_payload,
    )

    monkeypatch.setattr(references_router, "get_chat_store", lambda: _BusyStore())

    out = references_router.get_conversation_refs("conv-busy")

    assert out == cached_payload


def test_get_conversation_refs_uses_persisted_full_payload_without_reenrich(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()
    refs = {
        11: {
            "prompt": "Which paper defines dynamic supersampling?",
            "prompt_sig": "sig-11",
            "used_query": "dynamic supersampling",
            "used_translation": False,
            "hits": [
                {"text": "hit", "meta": {"source_path": r"db\SciAdv-2017\SciAdv-2017.en.md", "ref_pack_state": "ready"}}
            ],
            "scores": [9.0],
        }
    }
    rendered_payload = {11: {"hits": [{"ui_meta": {"summary_line": "full-persisted"}}]}}
    refs[11]["rendered_payload"] = dict(rendered_payload[11])
    refs[11]["render_status"] = "full"
    refs[11]["render_attempts"] = 1
    refs[11]["rendered_payload_sig"] = references_router._refs_pack_render_signature(
        user_msg_id=11,
        pack=refs[11],
        guide_mode=False,
        guide_source_path="",
        guide_source_name="",
    )
    store = _FakeStore({"mode": "chat"}, refs)

    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(references_router, "_warm_conversation_refs_payload_async", lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not warm when persisted full payload is fresh")))

    def fail_enrich(*args, **kwargs):
        raise AssertionError("persisted full payload should bypass enrich")

    monkeypatch.setattr(references_router, "enrich_refs_payload", fail_enrich)

    out = references_router.get_conversation_refs("conv-rendered")

    assert out == {
        11: {
            "hits": [{"ui_meta": {"summary_line": "full-persisted"}}],
            "render_status": "full",
            "render_attempts": 1,
        }
    }


def test_get_conversation_refs_returns_fast_ready_payload_and_kicks_background_warm(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()
    refs = {
        3: {
            "prompt": "Which paper defines dynamic supersampling?",
            "render_status": "failed",
            "render_error": "render_payload_empty",
            "hits": [
                {"text": "hit", "meta": {"source_path": r"db\SciAdv-2017\SciAdv-2017.en.md", "ref_pack_state": "ready"}}
            ],
        }
    }
    store = _FakeStore({"mode": "chat"}, refs)
    warm_calls: list[dict] = []

    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(references_router, "_pdf_dir", lambda: None)
    monkeypatch.setattr(references_router, "_md_dir", lambda: None)
    monkeypatch.setattr(references_router, "_lib_store", lambda: None)
    monkeypatch.setattr(references_router, "_warm_conversation_refs_payload_async", lambda **kwargs: warm_calls.append(dict(kwargs)))

    def fake_enrich_refs_payload(*args, **kwargs):
        del args
        if bool(kwargs.get("allow_exact_locate")) is False:
            return {3: {"mode": "fast", "hits": [{"ui_meta": {"summary_line": "fast"}}]}}
        return {3: {"mode": "full", "hits": [{"ui_meta": {"summary_line": "full"}}]}}

    monkeypatch.setattr(references_router, "enrich_refs_payload", fake_enrich_refs_payload)

    out = references_router.get_conversation_refs("conv-fast")

    assert out == {
        3: {
            "mode": "fast",
            "payload_mode": "fast",
            "render_status": "failed",
            "render_error": "render_payload_empty",
            "hits": [{"ui_meta": {"summary_line": "fast"}}],
        }
    }
    assert warm_calls == []


def test_get_conversation_refs_surfaces_pack_primary_evidence_from_fast_payload(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()
    refs = {
        4: {
            "prompt": "Which paper defines dynamic supersampling?",
            "hits": [
                {"text": "hit", "meta": {"source_path": r"db\SciAdv-2017\SciAdv-2017.en.md", "ref_pack_state": "ready"}}
            ],
        }
    }
    store = _FakeStore({"mode": "chat"}, refs)

    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(references_router, "_pdf_dir", lambda: None)
    monkeypatch.setattr(references_router, "_md_dir", lambda: None)
    monkeypatch.setattr(references_router, "_lib_store", lambda: None)
    monkeypatch.setattr(references_router, "_warm_conversation_refs_payload_async", lambda **kwargs: None)

    def fake_enrich_refs_payload(*args, **kwargs):
        del args, kwargs
        return {
            4: {
                "hits": [
                    {
                        "ui_meta": {
                            "summary_line": "fast",
                            "primary_evidence": {
                                "source_name": "SciAdv-2017.pdf",
                                "heading_path": "INTRODUCTION / Spatially variant digital supersampling",
                                "snippet": "This technique is known as digital superresolution or supersampling.",
                            },
                        }
                    }
                ]
            }
        }

    monkeypatch.setattr(references_router, "enrich_refs_payload", fake_enrich_refs_payload)

    out = references_router.get_conversation_refs("conv-fast-primary")

    pack = out[4]
    assert pack["primary_evidence"]["source_name"] == "SciAdv-2017.pdf"
    assert pack["primary_evidence"]["heading_path"] == "INTRODUCTION / Spatially variant digital supersampling"
    assert pack["primary_evidence_heading_path"] == "INTRODUCTION / Spatially variant digital supersampling"


def test_warm_conversation_refs_payload_async_uses_bounded_full_variant(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()
    calls: dict[str, object] = {}

    class _ImmediateThread:
        def __init__(self, *, target=None, daemon=None, name=None):
            del daemon, name
            self._target = target

        def start(self):
            if self._target is not None:
                self._target()

    monkeypatch.setattr(references_router.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(references_router, "_pdf_dir", lambda: None)
    monkeypatch.setattr(references_router, "_md_dir", lambda: None)
    monkeypatch.setattr(references_router, "_lib_store", lambda: None)

    def fake_enrich_refs_payload(*args, **kwargs):
        calls["kwargs"] = dict(kwargs)
        return {13: {"hits": [{"ui_meta": {"summary_line": "bounded-full"}}]}}

    monkeypatch.setattr(references_router, "enrich_refs_payload", fake_enrich_refs_payload)
    monkeypatch.setattr(references_router, "_persist_rendered_refs_payloads", lambda **kwargs: calls.setdefault("persisted_payload", kwargs.get("payload")))
    monkeypatch.setattr(references_router, "_store_cached_conversation_refs_payload", lambda **kwargs: calls.setdefault("cache_mode", kwargs.get("mode")))

    references_router._warm_conversation_refs_payload_async(
        conv_id="conv-warm",
        signature="sig-warm",
        refs={
            13: {
                "prompt": "Which paper compares Hadamard and Fourier SPI?",
                "hits": [
                    {
                        "text": "Figure 1 compares Hadamard and Fourier basis patterns.",
                        "meta": {"source_path": r"db\OE-2017\OE-2017.en.md", "ref_pack_state": "ready"},
                    }
                ],
            }
        },
        guide_mode=False,
        guide_source_path="",
        guide_source_name="",
    )

    kwargs = dict(calls.get("kwargs") or {})
    assert kwargs.get("render_variant") == "bounded_full"
    assert kwargs.get("allow_expensive_llm_for_ready") is False
    assert kwargs.get("allow_exact_locate") is True
    assert calls.get("persisted_payload") == {13: {"hits": [{"ui_meta": {"summary_line": "bounded-full"}}]}}
    assert calls.get("cache_mode") == "full"


def test_get_conversation_refs_falls_back_to_cached_payload_when_conversation_read_is_busy(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()

    class _BusyConversationStore:
        def get_conversation(self, conv_id: str, *, timeout_s=None):
            del conv_id, timeout_s
            raise sqlite3.OperationalError("database is locked")

        def list_message_refs(self, conv_id: str, *, timeout_s=None):
            raise AssertionError("should not list refs when conversation read already failed")

    cached_payload = {5: {"hits": [{"ui_meta": {"summary_line": "cached-conversation"}}]}}
    references_router._store_cached_conversation_refs_payload(
        conv_id="conv-conversation-busy",
        signature="sig",
        payload=cached_payload,
    )

    monkeypatch.setattr(references_router, "get_chat_store", lambda: _BusyConversationStore())

    out = references_router.get_conversation_refs("conv-conversation-busy")

    assert out == cached_payload
