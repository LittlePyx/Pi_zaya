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


def test_get_conversation_refs_invalidates_fast_cache_when_full_render_payload_arrives(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()
    refs = {
        12: {
            "prompt": "Which papers in my library mention SCI?",
            "prompt_sig": "sig-12",
            "used_query": "SCI",
            "used_translation": False,
            "hits": [
                {"text": "hit-a", "meta": {"source_path": r"db\A\A.en.md", "ref_pack_state": "ready"}}
            ],
            "scores": [7.1],
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
            12: {
                "payload_mode": "fast",
                "hits": [{"ui_meta": {"summary_line": "fast-only"}}],
            }
        }

    monkeypatch.setattr(references_router, "enrich_refs_payload", fake_enrich_refs_payload)

    out1 = references_router.get_conversation_refs("conv-render-upgrade")

    refs[12]["rendered_payload"] = {"hits": [{"ui_meta": {"summary_line": "full-persisted"}}]}
    refs[12]["render_status"] = "full"
    refs[12]["rendered_payload_sig"] = references_router._refs_pack_render_signature(
        user_msg_id=12,
        pack=refs[12],
        guide_mode=False,
        guide_source_path="",
        guide_source_name="",
    )

    out2 = references_router.get_conversation_refs("conv-render-upgrade")

    assert out1[12]["display_state"] == "pending"
    assert out1[12]["enrichment_pending"] is True
    assert out1[12]["payload_mode"] == "fast"
    assert out1[12]["hits"] == [{"ui_meta": {"summary_line": "fast-only"}}]

    assert out2[12]["display_state"] == "ready"
    assert out2[12]["hits"] == [{"ui_meta": {"summary_line": "full-persisted"}}]
    assert out2[12]["render_status"] == "full"


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
    assert str((out.get(7) or {}).get("display_state") or "") == "pending"
    assert str((out.get(7) or {}).get("suppression_reason") or "") == "pending_enrichment"


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


def test_get_conversation_refs_pending_payload_prefers_authoritative_doc_list_when_available(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()

    class _DocListStore(_FakeStore):
        def __init__(self, conversation: dict, refs: dict, messages: list[dict]) -> None:
            super().__init__(conversation, refs)
            self._messages = list(messages)

        def get_messages(self, conv_id: str):
            del conv_id
            return list(self._messages)

    prompt = "有哪几篇文章提到了SCI（单次曝光压缩成像）"
    refs = {
        15: {
            "prompt": prompt,
            "hits": [
                {
                    "text": "pending broad single-pixel review",
                    "meta": {
                        "source_path": r"db\NatPhoton-2019\NatPhoton-2019.en.md",
                        "ref_pack_state": "pending",
                        "ref_best_heading_path": "Abstract",
                    },
                },
                {
                    "text": "pending single-pixel holography paper",
                    "meta": {
                        "source_path": r"db\NatCommun-2021\NatCommun-2021.en.md",
                        "ref_pack_state": "pending",
                        "ref_best_heading_path": "ARTICLE",
                    },
                },
            ],
        }
    }
    messages = [
        {"id": 15, "role": "user", "content": prompt},
        {
            "id": 16,
            "role": "assistant",
            "content": "根据命中的库内文献，以下 3 篇文章直接涉及 SCI。",
            "meta": {
                "paper_guide_contracts": {
                    "doc_list": [
                        {
                            "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
                            "source_name": "ICIP-2025-SCIGS.pdf",
                            "heading_path": "1. Introduction",
                            "summary_line": "The paper explicitly introduces Snapshot Compressive Imaging (SCI).",
                            "primary_evidence": {
                                "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
                                "source_name": "ICIP-2025-SCIGS.pdf",
                                "heading_path": "1. Introduction",
                                "snippet": "Video Snapshot Compressive Imaging (SCI) technology has been developed for high-speed imaging.",
                            },
                        },
                        {
                            "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
                            "source_name": "CVPR-2024-SCINeRF.pdf",
                            "heading_path": "Abstract",
                            "summary_line": "The paper repeatedly mentions Snapshot Compressive Imaging (SCI).",
                            "primary_evidence": {
                                "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
                                "source_name": "CVPR-2024-SCINeRF.pdf",
                                "heading_path": "Abstract",
                                "snippet": "In this paper, we explore the potential of Snapshot Compressive Imaging (SCI).",
                            },
                        },
                        {
                            "source_path": r"db\OE-2007\OE-2007.en.md",
                            "source_name": "OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture.pdf",
                            "heading_path": "5. Conclusions",
                            "summary_line": "This early single-shot compressive spectral imaging paper is retained as an SCI predecessor.",
                            "primary_evidence": {
                                "source_path": r"db\OE-2007\OE-2007.en.md",
                                "source_name": "OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture.pdf",
                                "heading_path": "5. Conclusions",
                                "snippet": "This manuscript describes a new, single-shot spectral imager based on compressive sensing ideas.",
                            },
                        },
                    ]
                }
            },
        },
    ]
    store = _DocListStore({"mode": "chat"}, refs, messages)

    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(references_router, "_warm_conversation_refs_payload_async", lambda **kwargs: None)
    monkeypatch.setattr(references_router, "_compact_reader_open_text", lambda text, max_len=360: str(text or "").strip())

    out = references_router.get_conversation_refs("conv-pending-doc-list")

    pack = out[15]
    titles = [((hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}) or {}).get("display_name") for hit in list(pack.get("hits") or [])]
    assert pack["pending"] is True
    assert titles[:2] == [
        "ICIP-2025-SCIGS.pdf",
        "CVPR-2024-SCINeRF.pdf",
    ]
    assert titles[2] in {
        "OE-2007.pdf",
        "OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture.pdf",
    }
    assert "NatPhoton-2019" not in " ".join(str(item or "") for item in titles)
    assert all(bool(((hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}) or {}).get("score_pending")) for hit in list(pack.get("hits") or []))


def test_build_pending_conversation_refs_payload_stabilizes_multi_paper_identity_before_authoritative_doc_list(monkeypatch):
    prompt = "Which papers in my library mention SCI (Snapshot Compressive Imaging)?"
    hits = [
        {
            "text": "single-shot compressive spectral imaging from the conclusion",
            "meta": {
                "source_path": r"db\OE-2007\OE-2007.en.md",
                "ref_pack_state": "pending",
                "ref_best_heading_path": "5. Conclusions",
            },
        },
        {
            "text": "single-pixel imaging review article",
            "meta": {
                "source_path": r"db\NatPhoton-2019\NatPhoton-2019.en.md",
                "ref_pack_state": "pending",
                "ref_best_heading_path": "Introduction",
            },
        },
        {
            "text": "Snapshot Compressive Imaging (SCI) is used to recover a 3D scene from a single temporal compressed image.",
            "meta": {
                "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
                "ref_pack_state": "pending",
                "ref_best_heading_path": "2. Related Work",
            },
        },
        {
            "text": "Video Snapshot Compressive Imaging (SCI) technology decodes the compressed image into high frame rate images.",
            "meta": {
                "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
                "ref_pack_state": "pending",
                "ref_best_heading_path": "1. Introduction",
            },
        },
    ]

    monkeypatch.setattr(
        references_router,
        "_references_build_multi_paper_doc_list_contract",
        lambda **kwargs: [
            {
                "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
                "source_name": "CVPR-2024-SCINeRF.pdf",
                "heading_path": "2. Related Work",
            },
            {
                "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
                "source_name": "ICIP-2025-SCIGS.pdf",
                "heading_path": "1. Introduction",
            },
            {
                "source_path": r"db\OE-2007\OE-2007.en.md",
                "source_name": "OE-2007.pdf",
                "heading_path": "5. Conclusions",
            },
        ],
    )

    out = references_router._build_pending_conversation_refs_payload(
        {
            18: {
                "prompt": prompt,
                "hits": hits,
            }
        },
        doc_list_by_user={},
    )

    pack = out[18]
    titles = [
        ((hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}) or {}).get("display_name")
        for hit in list(pack.get("hits") or [])
    ]

    assert titles == [
        "CVPR-2024-SCINeRF.pdf",
        "ICIP-2025-SCIGS.pdf",
        "OE-2007.pdf",
    ]
    assert "NatPhoton-2019" not in " ".join(str(title or "") for title in titles)
    assert pack["payload_mode"] == "pending"
    assert pack["pending"] is True


def test_get_conversation_refs_full_payload_prefers_authoritative_doc_list_over_non_authoritative_rendered_payload(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()

    class _DocListStore(_FakeStore):
        def __init__(self, conversation: dict, refs: dict, messages: list[dict]) -> None:
            super().__init__(conversation, refs)
            self._messages = list(messages)
            self.persisted: list[dict] = []

        def get_messages(self, conv_id: str):
            del conv_id
            return list(self._messages)

        def set_message_refs_rendered_payload(self, **kwargs):
            self.persisted.append(dict(kwargs))

    prompt = "Which papers in my library mention SCI (Snapshot Compressive Imaging)?"
    refs = {
        21: {
            "prompt": prompt,
            "prompt_sig": "sig-21",
            "used_query": "SCI",
            "used_translation": False,
            "hits": [
                {
                    "text": "broad single-pixel review",
                    "meta": {
                        "source_path": r"db\NatPhoton-2019\NatPhoton-2019.en.md",
                        "ref_pack_state": "ready",
                    },
                }
            ],
            "scores": [7.4],
            "render_status": "full",
            "rendered_payload": {
                "pipeline_debug": {},
                "hits": [
                    {
                        "ui_meta": {
                            "display_name": "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf",
                            "summary_line": "wrong cached full payload",
                            "source_path": r"db\NatPhoton-2019\NatPhoton-2019.en.md",
                        }
                    }
                ],
            },
        }
    }
    refs[21]["rendered_payload_sig"] = references_router._refs_pack_render_signature(
        user_msg_id=21,
        pack=refs[21],
        guide_mode=False,
        guide_source_path="",
        guide_source_name="",
    )
    messages = [
        {"id": 21, "role": "user", "content": prompt},
        {
            "id": 22,
            "role": "assistant",
            "content": "According to the library hits, these SCI papers are directly relevant.",
            "meta": {
                "paper_guide_contracts": {
                    "doc_list": [
                        {
                            "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
                            "source_name": "ICIP-2025-SCIGS.pdf",
                            "heading_path": "1. Introduction",
                        },
                        {
                            "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
                            "source_name": "CVPR-2024-SCINeRF.pdf",
                            "heading_path": "2. Related Work",
                        },
                        {
                            "source_path": r"db\OE-2007\OE-2007.en.md",
                            "source_name": "OE-2007.pdf",
                            "heading_path": "5. Conclusions",
                        },
                    ]
                }
            },
        },
    ]
    store = _DocListStore({"mode": "chat"}, refs, messages)
    calls: dict[str, object] = {}

    def fake_build_doc_list_refs_payload(*, user_msg_id, pack, doc_list, **kwargs):
        del pack
        calls["kwargs"] = dict(kwargs)
        hits = []
        for item in list(doc_list or []):
            hits.append(
                {
                    "ui_meta": {
                        "display_name": str(item.get("source_name") or "").strip(),
                        "summary_line": f"authoritative::{item.get('source_name')}",
                        "source_path": str(item.get("source_path") or "").strip(),
                    },
                    "meta": {
                        "source_path": str(item.get("source_path") or "").strip(),
                        "ref_pack_state": "ready",
                    },
                }
            )
        return {
            "user_msg_id": int(user_msg_id),
            "payload_mode": "full",
            "pipeline_debug": {"doc_list_authoritative": True},
            "hits": hits,
        }

    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(references_router, "_warm_conversation_refs_payload_async", lambda **kwargs: None)
    monkeypatch.setattr(references_router, "build_doc_list_refs_payload", fake_build_doc_list_refs_payload)

    out = references_router.get_conversation_refs("conv-full-doc-list")

    pack = out[21]
    titles = [((hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}) or {}).get("display_name") for hit in list(pack.get("hits") or [])]
    assert titles == [
        "ICIP-2025-SCIGS.pdf",
        "CVPR-2024-SCINeRF.pdf",
        "OE-2007.pdf",
    ]
    assert "NatPhoton-2019" not in " ".join(str(item or "") for item in titles)
    assert pack["render_status"] == "full"
    assert dict(calls.get("kwargs") or {}).get("allow_exact_locate") is True
    assert dict(calls.get("kwargs") or {}).get("allow_expensive_llm") is False
    assert dict(calls.get("kwargs") or {}).get("apply_copy_polish") is True
    assert store.persisted
    assert store.persisted[-1]["rendered_payload"]["pipeline_debug"]["doc_list_authoritative"] is True


def test_build_pending_conversation_refs_payload_uses_empty_authoritative_doc_list_and_forwards_guide(monkeypatch):
    refs = {
        51: {
            "prompt": "Besides this paper, what other papers in my library discuss ADMM?",
            "hits": [
                {
                    "text": "Stale self paper hit.",
                    "meta": {
                        "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
                        "ref_pack_state": "pending",
                    },
                }
            ],
        }
    }
    calls: dict[str, object] = {}

    def fake_build_doc_list_refs_payload(*, user_msg_id, pack, doc_list, **kwargs):
        calls["user_msg_id"] = int(user_msg_id)
        calls["prompt"] = str(pack.get("prompt") or "")
        calls["doc_list"] = list(doc_list or [])
        calls["kwargs"] = dict(kwargs)
        return {
            "user_msg_id": int(user_msg_id),
            "payload_mode": "full",
            "pipeline_debug": {"doc_list_authoritative": True},
            "guide_filter": {"active": True, "hidden_self_source": True, "filtered_hit_count": 1},
            "hits": [],
        }

    monkeypatch.setattr(references_router, "build_doc_list_refs_payload", fake_build_doc_list_refs_payload)

    out = references_router._build_pending_conversation_refs_payload(
        refs,
        doc_list_by_user={51: []},
        guide_mode=True,
        guide_source_path=r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
        guide_source_name="CVPR-2024-SCINeRF.pdf",
    )

    assert calls["doc_list"] == []
    assert dict(calls.get("kwargs") or {}).get("guide_mode") is True
    assert dict(calls.get("kwargs") or {}).get("allow_exact_locate") is False
    assert dict(calls.get("kwargs") or {}).get("allow_expensive_llm") is False
    assert dict(calls.get("kwargs") or {}).get("apply_copy_polish") is True
    assert dict(calls.get("kwargs") or {}).get("guide_source_name") == "CVPR-2024-SCINeRF.pdf"
    pack = dict(out.get(51) or {})
    assert list(pack.get("hits") or []) == []
    assert bool(pack.get("pending")) is True
    assert str(pack.get("display_state") or "") == "pending"


def test_get_conversation_refs_empty_authoritative_doc_list_overrides_stale_full_payload(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()

    class _EmptyDocListStore(_FakeStore):
        def __init__(self, conversation: dict, refs: dict, messages: list[dict]) -> None:
            super().__init__(conversation, refs)
            self._messages = list(messages)
            self.persisted: list[dict] = []

        def get_messages(self, conv_id: str):
            del conv_id
            return list(self._messages)

        def set_message_refs_rendered_payload(self, **kwargs):
            self.persisted.append(dict(kwargs))

    prompt = "Besides this paper, what other papers in my library discuss ADMM?"
    refs = {
        61: {
            "prompt": prompt,
            "prompt_sig": "sig-61",
            "hits": [
                {
                    "text": "self paper",
                    "meta": {
                        "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
                        "ref_pack_state": "ready",
                    },
                }
            ],
            "rendered_payload": {
                "payload_mode": "full",
                "render_status": "full",
                "hits": [
                    {
                        "ui_meta": {
                            "display_name": "CVPR-2024-SCINeRF.pdf",
                            "summary_line": "stale self card",
                            "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
                        },
                        "meta": {"source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md"},
                    }
                ],
            },
            "rendered_payload_sig": "stale-sig",
            "render_status": "full",
        }
    }
    messages = [
        {"id": 61, "role": "user", "content": prompt},
        {
            "id": 62,
            "role": "assistant",
            "content": "No other retrieved paper explicitly discusses ADMM.",
            "meta": {
                "paper_guide_contracts": {
                    "doc_list": []
                }
            },
        },
    ]
    store = _EmptyDocListStore(
        {
            "mode": "paper_guide",
            "bound_source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
            "bound_source_name": "CVPR-2024-SCINeRF.pdf",
        },
        refs,
        messages,
    )

    def fake_build_doc_list_refs_payload(*, user_msg_id, pack, doc_list, **kwargs):
        del pack
        assert list(doc_list or []) == []
        assert kwargs.get("guide_mode") is True
        return {
            "user_msg_id": int(user_msg_id),
            "payload_mode": "full",
            "pipeline_debug": {"doc_list_authoritative": True, "raw_hit_count": 0, "final_hit_count": 0},
            "guide_filter": {"active": True, "hidden_self_source": True, "filtered_hit_count": 1},
            "hits": [],
        }

    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(references_router, "_warm_conversation_refs_payload_async", lambda **kwargs: None)
    monkeypatch.setattr(references_router, "build_doc_list_refs_payload", fake_build_doc_list_refs_payload)

    out = references_router.get_conversation_refs("conv-empty-doc-list")

    pack = dict(out.get(61) or {})
    assert list(pack.get("hits") or []) == []
    assert str(pack.get("display_state") or "") == "hidden_by_guide"
    assert str(pack.get("render_status") or "") == "full"
    assert store.persisted
    assert list(store.persisted[-1]["rendered_payload"].get("hits") or []) == []


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
            "display_state": "ready",
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
            "display_state": "ready",
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
