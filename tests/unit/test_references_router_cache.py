from __future__ import annotations

import sqlite3

import api.routers.references as references_router
from fastapi import Response


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


def test_get_conversation_refs_exposes_route_timing_headers(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()
    refs = {
        7: {
            "prompt": "Which paper discusses dynamic supersampling?",
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
    monkeypatch.setattr(
        references_router,
        "enrich_refs_payload",
        lambda *args, **kwargs: {7: {"payload_mode": "fast", "hits": [{"ui_meta": {"summary_line": "fast"}}]}},
    )

    response = Response()
    out = references_router.get_conversation_refs("conv-timing", response=response)

    assert out[7]["display_state"] == "ready"
    assert "total;dur=" in str(response.headers.get("server-timing") or "")
    assert response.headers.get("x-kb-refs-mode") == "fast"
    counts = str(response.headers.get("x-kb-refs-counts") or "")
    assert "packs=1" in counts
    assert "hits=1" in counts


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

    assert out1[12]["display_state"] == "ready"
    assert out1[12]["enrichment_pending"] is True
    assert out1[12]["payload_mode"] == "fast"
    assert out1[12]["hits"][0]["ui_meta"]["summary_line"] == "fast-only"
    assert out1[12]["hits"][0]["ui_meta"]["polish_status"] == "heuristic"

    assert out2[12]["display_state"] == "ready"
    assert out2[12]["hits"][0]["ui_meta"]["summary_line"] == "full-persisted"
    assert out2[12]["hits"][0]["ui_meta"]["polish_status"] == "heuristic"
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


def test_get_conversation_refs_treats_stale_pending_pack_as_fast_ready(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()
    refs = {
        17: {
            "prompt": "Which paper discusses NeRF?",
            "updated_at": 1.0,
            "hits": [
                {
                    "text": "NeRF discusses neural radiance fields.",
                    "meta": {
                        "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
                        "ref_pack_state": "pending",
                    },
                }
            ],
        }
    }
    store = _FakeStore({"mode": "chat"}, refs)
    warm_calls = {"n": 0}

    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(references_router, "_warm_conversation_refs_payload_async", lambda **kwargs: warm_calls.__setitem__("n", warm_calls["n"] + 1))
    monkeypatch.setattr(references_router, "_refs_pending_stale_after_s", lambda: 0.1)

    def fake_enrich_refs_payload(*args, **kwargs):
        del args, kwargs
        return {
            17: {
                "payload_mode": "fast",
                "hits": [{"ui_meta": {"summary_line": "fast-ready"}}],
            }
        }

    monkeypatch.setattr(references_router, "enrich_refs_payload", fake_enrich_refs_payload)

    out = references_router.get_conversation_refs("conv-stale-pending")

    assert out[17]["display_state"] == "ready"
    assert out[17]["payload_mode"] == "fast"
    assert out[17]["hits"][0]["ui_meta"]["summary_line"] == "fast-ready"
    assert out[17]["hits"][0]["ui_meta"]["polish_status"] == "heuristic"
    assert warm_calls["n"] == 1


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
    assert dict(calls.get("kwargs") or {}).get("allow_expensive_llm") is True
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


def test_get_conversation_refs_rebuilds_empty_authoritative_doc_list_for_plain_multi_paper_chat(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()

    class _EmptyDocListChatStore(_FakeStore):
        def __init__(self, conversation: dict, refs: dict, messages: list[dict]) -> None:
            super().__init__(conversation, refs)
            self._messages = list(messages)
            self.persisted: list[dict] = []

        def get_messages(self, conv_id: str):
            del conv_id
            return list(self._messages)

        def set_message_refs_rendered_payload(self, **kwargs):
            self.persisted.append(dict(kwargs))

    prompt = "哪几篇文章里提到了NeRF"
    refs = {
        71: {
            "prompt": prompt,
            "prompt_sig": "sig-71",
            "hits": [
                {
                    "text": "SCINeRF exploits NeRF as its underlying scene representation.",
                    "meta": {
                        "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
                        "ref_pack_state": "ready",
                    },
                },
                {
                    "text": "This paper focuses on 3D Gaussian splatting.",
                    "meta": {
                        "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
                        "ref_pack_state": "ready",
                    },
                },
            ],
            "rendered_payload": {
                "payload_mode": "full",
                "render_status": "full",
                "hits": [],
                "pipeline_debug": {"doc_list_authoritative": True, "raw_hit_count": 0, "final_hit_count": 0},
            },
            "rendered_payload_sig": "stale-empty-sig",
            "render_status": "full",
        }
    }
    messages = [
        {"id": 71, "role": "user", "content": prompt},
        {
            "id": 72,
            "role": "assistant",
            "content": "DOC-1 mentions NeRF.",
            "meta": {
                "paper_guide_contracts": {
                    "doc_list": []
                }
            },
        },
    ]
    store = _EmptyDocListChatStore({"mode": "chat"}, refs, messages)
    calls: dict[str, object] = {}

    def fake_build_doc_list_refs_payload(*, user_msg_id, pack, doc_list, **kwargs):
        del pack, kwargs
        calls["doc_list"] = list(doc_list or [])
        return {
            "user_msg_id": int(user_msg_id),
            "payload_mode": "full",
            "pipeline_debug": {"doc_list_authoritative": True, "raw_hit_count": 1, "final_hit_count": 1},
            "hits": [
                {
                    "ui_meta": {"display_name": "CVPR-2024-SCINeRF.pdf", "summary_line": "mentions NeRF"},
                    "meta": {"source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md"},
                }
            ],
        }

    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(references_router, "_warm_conversation_refs_payload_async", lambda **kwargs: None)
    monkeypatch.setattr(references_router, "build_doc_list_refs_payload", fake_build_doc_list_refs_payload)

    out = references_router.get_conversation_refs("conv-empty-doc-list-chat")

    assert [str(item.get("source_path") or "") for item in list(calls.get("doc_list") or [])] == [
        r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md"
    ]
    assert str(out[71]["display_state"] or "") == "ready"
    assert list(out[71]["hits"] or [])
    assert store.persisted


def test_get_conversation_refs_drops_empty_doc_list_contract_when_rebuild_has_no_rows(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()

    class _EmptyDocListChatStore(_FakeStore):
        def __init__(self, conversation: dict, refs: dict, messages: list[dict]) -> None:
            super().__init__(conversation, refs)
            self._messages = list(messages)

        def get_messages(self, conv_id: str):
            del conv_id
            return list(self._messages)

    prompt = "哪些文献讨论了单像素成像中的深度学习？请概括它解决了什么问题，又有哪些挑战。"
    refs = {
        81: {
            "prompt": prompt,
            "prompt_sig": "sig-81",
            "hits": [
                {
                    "text": "Deep learning has been used for single-pixel imaging reconstruction.",
                    "meta": {
                        "source_path": r"db\LPR-2025\LPR-2025.en.md",
                        "ref_pack_state": "ready",
                    },
                }
            ],
            "rendered_payload": {
                "payload_mode": "full",
                "display_state": "empty",
                "suppression_reason": "no_candidate_hits",
                "hits": [],
                "pipeline_debug": {
                    "doc_list_authoritative": True,
                    "raw_hit_count": 0,
                    "final_hit_count": 0,
                },
            },
            "render_status": "full",
        }
    }
    refs[81]["rendered_payload_sig"] = references_router._refs_pack_render_signature(
        user_msg_id=81,
        pack=refs[81],
        guide_mode=False,
        guide_source_path="",
        guide_source_name="",
    )
    messages = [
        {"id": 81, "role": "user", "content": prompt},
        {
            "id": 82,
            "role": "assistant",
            "content": "以下文献讨论了单像素成像中的深度学习。",
            "meta": {"paper_guide_contracts": {"doc_list": []}},
        },
    ]
    store = _EmptyDocListChatStore({"mode": "chat"}, refs, messages)
    warm_calls: list[dict] = []

    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(references_router, "_pdf_dir", lambda: None)
    monkeypatch.setattr(references_router, "_md_dir", lambda: None)
    monkeypatch.setattr(references_router, "_lib_store", lambda: None)
    monkeypatch.setattr(references_router, "_rebuild_authoritative_doc_list_from_pack", lambda **kwargs: [])
    monkeypatch.setattr(references_router, "build_doc_list_refs_payload", lambda **kwargs: (_ for _ in ()).throw(AssertionError("empty plain-chat doc_list should not rebuild authoritative payload")))
    monkeypatch.setattr(references_router, "_warm_conversation_refs_payload_async", lambda **kwargs: warm_calls.append(dict(kwargs)))

    def fake_enrich_refs_payload(*args, **kwargs):
        del args, kwargs
        return {
            81: {
                "payload_mode": "fast",
                "hits": [{"ui_meta": {"summary_line": "deep-learning card"}}],
            }
        }

    monkeypatch.setattr(references_router, "enrich_refs_payload", fake_enrich_refs_payload)

    out = references_router.get_conversation_refs("conv-empty-doc-list-no-rebuild")

    assert out[81]["display_state"] == "ready"
    assert out[81]["payload_mode"] == "fast"
    assert out[81]["hits"][0]["ui_meta"]["summary_line"] == "deep-learning card"
    assert warm_calls


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
    monkeypatch.setenv("KB_REFS_BACKGROUND_LLM_POLISH", "0")
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

    assert out[11]["display_state"] == "ready"
    assert out[11]["render_status"] == "full"
    assert out[11]["render_attempts"] == 1
    assert out[11]["hits"][0]["ui_meta"]["summary_line"] == "full-persisted"
    assert out[11]["hits"][0]["ui_meta"]["polish_status"] == "heuristic"


def test_stored_rendered_payload_is_stale_when_answer_source_disappears():
    pack = {
        "hits": [
            {"text": "answer evidence", "meta": {"source_path": r"db\DL-SPI\DL-SPI.en.md"}},
            {"text": "other evidence", "meta": {"source_path": r"db\Other\Other.en.md"}},
        ]
    }
    payload = {
        "hits": [
            {"ui_meta": {"source_path": r"db\Other\Other.en.md", "summary_line": "old card"}},
        ]
    }

    assert references_router._stored_rendered_pack_payload_lost_current_hits(
        payload=payload,
        pack=pack,
    )


def test_get_conversation_refs_ignores_stale_persisted_full_payload_and_rebuilds(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()
    refs = {
        15: {
            "prompt": "NeRF是什么",
            "prompt_sig": "sig-15",
            "used_query": "NeRF是什么",
            "used_translation": False,
            "hits": [
                {
                    "text": "SCINeRF exploits neural radiance fields as its underlying scene representation.",
                    "meta": {
                        "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
                        "ref_pack_state": "ready",
                    },
                },
                {
                    "text": "SCIGS discusses the bottlenecks of NeRF-based reconstruction.",
                    "meta": {
                        "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
                        "ref_pack_state": "ready",
                    },
                },
            ],
            "scores": [7.8, 6.4],
            "render_status": "full",
            "rendered_payload": {
                "hits": [{"ui_meta": {"summary_line": "stale-single-card"}}],
                "pipeline_debug": {"final_hit_count": 1},
            },
            "rendered_payload_sig": "schema-v3-stale-sig",
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
        if str(kwargs.get("render_variant") or "") != "fast":
            raise AssertionError("stale stored payload should fall back to fast rebuild path first")
        return {
            15: {
                "payload_mode": "fast",
                "hits": [
                    {"ui_meta": {"summary_line": "SCINeRF first"}},
                    {"ui_meta": {"summary_line": "SCIGS second"}},
                ],
            }
        }

    monkeypatch.setattr(references_router, "enrich_refs_payload", fake_enrich_refs_payload)

    out = references_router.get_conversation_refs("conv-stale-full-payload")

    assert out[15]["display_state"] == "ready"
    assert out[15]["payload_mode"] == "fast"
    assert [item["ui_meta"]["summary_line"] for item in out[15]["hits"]] == ["SCINeRF first", "SCIGS second"]
    assert warm_calls


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
    fast_kwargs: dict = {}

    def fake_enrich_refs_payload(*args, **kwargs):
        del args
        if bool(kwargs.get("allow_exact_locate")) is False:
            fast_kwargs.update(dict(kwargs))
            return {3: {"mode": "fast", "hits": [{"ui_meta": {"summary_line": "fast"}}]}}
        return {3: {"mode": "full", "hits": [{"ui_meta": {"summary_line": "full"}}]}}

    monkeypatch.setattr(references_router, "enrich_refs_payload", fake_enrich_refs_payload)

    out = references_router.get_conversation_refs("conv-fast")

    assert out[3]["mode"] == "fast"
    assert out[3]["payload_mode"] == "fast"
    assert out[3]["render_status"] == "failed"
    assert out[3]["render_error"] == "render_payload_empty"
    assert out[3]["display_state"] == "ready"
    assert out[3]["hits"][0]["ui_meta"]["summary_line"] == "fast"
    assert out[3]["hits"][0]["ui_meta"]["polish_status"] == "failed"
    assert warm_calls == []
    assert fast_kwargs.get("render_variant") == "fast"
    assert fast_kwargs.get("allow_expensive_llm_for_ready") is False
    assert fast_kwargs.get("allow_exact_locate") is False


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
    monkeypatch.delenv("KB_REFS_BACKGROUND_LLM_POLISH", raising=False)
    monkeypatch.setenv("KB_REFS_CARD_POLISH_USE_LLM", "0")

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


def test_warm_conversation_refs_payload_async_allows_background_llm_when_enabled(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()
    calls: dict[str, object] = {}
    monkeypatch.setenv("KB_REFS_BACKGROUND_LLM_POLISH", "1")

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
    monkeypatch.setattr(references_router, "_persist_rendered_refs_payloads", lambda **_kwargs: None)
    monkeypatch.setattr(references_router, "_store_cached_conversation_refs_payload", lambda **_kwargs: None)

    references_router._warm_conversation_refs_payload_async(
        conv_id="conv-warm-llm",
        signature="sig-warm-llm",
        refs={13: {"prompt": "Which paper compares Hadamard and Fourier SPI?", "hits": []}},
        guide_mode=False,
        guide_source_path="",
        guide_source_name="",
    )

    kwargs = dict(calls.get("kwargs") or {})
    assert kwargs.get("allow_expensive_llm_for_ready") is True


def test_background_llm_polish_follows_card_polish_flag_when_unset(monkeypatch):
    monkeypatch.delenv("KB_REFS_BACKGROUND_LLM_POLISH", raising=False)
    monkeypatch.setattr(references_router, "_refs_card_polish_llm_enabled", lambda: True)

    assert references_router._refs_background_llm_polish_enabled() is True


def test_fast_exact_refs_disable_background_llm_polish():
    refs = {
        13: {
            "hits": [
                {
                    "text": "Most existing methods employ ADMM [4].",
                    "meta": {"paper_guide_fast_exact": True},
                }
            ]
        }
    }

    assert references_router._refs_payload_has_fast_exact_hit(refs) is True
    assert references_router._refs_payload_has_fast_exact_hit({13: {"hits": []}}) is False


def test_background_llm_polish_env_override_can_disable_card_polish(monkeypatch):
    monkeypatch.setenv("KB_REFS_BACKGROUND_LLM_POLISH", "0")
    monkeypatch.setattr(references_router, "_refs_card_polish_llm_enabled", lambda: True)

    assert references_router._refs_background_llm_polish_enabled() is False


def test_stored_full_payload_without_llm_copy_is_ignored_when_polish_is_enabled(monkeypatch):
    monkeypatch.delenv("KB_REFS_BACKGROUND_LLM_POLISH", raising=False)
    monkeypatch.setattr(references_router, "_refs_card_polish_llm_enabled", lambda: True)
    monkeypatch.setattr(references_router, "_refs_pack_render_signature", lambda **kwargs: "sig-stored")

    pack = {
        "prompt": "深度学习给单像素成像带来的好处和坑分别是什么？",
        "rendered_payload_sig": "sig-stored",
        "hits": [
            {
                "text": "Deep learning improves reconstruction quality.",
                "meta": {"source_path": r"db\LPR-2025\LPR-2025.en.md"},
            }
        ],
        "rendered_payload": {
            "hits": [
                {
                    "ui_meta": {
                        "source_path": r"db\LPR-2025\LPR-2025.en.md",
                        "summary_kind": "guide",
                        "summary_generation": "section_grounded",
                        "why_generation": "deterministic_grounded",
                    }
                }
            ]
        },
    }

    out = references_router._get_stored_rendered_pack_payload(
        user_msg_id=13,
        pack=pack,
        guide_mode=False,
        guide_source_path="",
        guide_source_name="",
    )

    assert out is None


def test_stored_full_payload_with_llm_copy_is_reused_when_polish_is_enabled(monkeypatch):
    monkeypatch.delenv("KB_REFS_BACKGROUND_LLM_POLISH", raising=False)
    monkeypatch.setattr(references_router, "_refs_card_polish_llm_enabled", lambda: True)
    monkeypatch.setattr(references_router, "_refs_pack_render_signature", lambda **kwargs: "sig-stored")

    payload = {
        "hits": [
            {
                "ui_meta": {
                    "source_path": r"db\LPR-2025\LPR-2025.en.md",
                    "summary_kind": "guide",
                    "summary_generation": "llm_grounded",
                    "why_generation": "llm_grounded",
                }
            }
        ]
    }
    pack = {
        "prompt": "深度学习给单像素成像带来的好处和坑分别是什么？",
        "rendered_payload_sig": "sig-stored",
        "hits": [
            {
                "text": "Deep learning improves reconstruction quality.",
                "meta": {"source_path": r"db\LPR-2025\LPR-2025.en.md"},
            }
        ],
        "rendered_payload": payload,
    }

    out = references_router._get_stored_rendered_pack_payload(
        user_msg_id=13,
        pack=pack,
        guide_mode=False,
        guide_source_path="",
        guide_source_name="",
    )

    assert out == payload


def test_stored_full_payload_from_previous_render_schema_is_ignored(monkeypatch):
    monkeypatch.setattr(references_router, "_refs_card_polish_llm_enabled", lambda: False)
    pack = {
        "prompt": "what helps SPI reconstruction?",
        "prompt_sig": "sig-prev-schema",
        "used_query": "SPI reconstruction",
        "used_translation": False,
        "hits": [
            {
                "text": "Deep learning improves reconstruction quality.",
                "meta": {"source_path": r"db\LPR-2025\LPR-2025.en.md"},
            }
        ],
    }
    current_schema = int(references_router._REFS_RENDER_PAYLOAD_SCHEMA_VERSION)
    monkeypatch.setattr(references_router, "_REFS_RENDER_PAYLOAD_SCHEMA_VERSION", current_schema - 1)
    old_sig = references_router._refs_pack_render_signature(
        user_msg_id=23,
        pack=pack,
        guide_mode=False,
        guide_source_path="",
        guide_source_name="",
    )
    monkeypatch.setattr(references_router, "_REFS_RENDER_PAYLOAD_SCHEMA_VERSION", current_schema)
    pack["rendered_payload_sig"] = old_sig
    pack["rendered_payload"] = {
        "hits": [
            {
                "ui_meta": {
                    "source_path": r"db\LPR-2025\LPR-2025.en.md",
                    "summary_line": "old full payload",
                    "summary_generation": "llm_grounded",
                    "why_generation": "llm_grounded",
                }
            }
        ]
    }

    out = references_router._get_stored_rendered_pack_payload(
        user_msg_id=23,
        pack=pack,
        guide_mode=False,
        guide_source_path="",
        guide_source_name="",
    )

    assert out is None


def test_stored_full_payload_with_dirty_card_copy_is_ignored(monkeypatch):
    monkeypatch.setattr(references_router, "_refs_card_polish_llm_enabled", lambda: False)
    monkeypatch.setattr(references_router, "_refs_pack_render_signature", lambda **kwargs: "sig-current")
    pack = {
        "prompt": "what helps SPI reconstruction?",
        "rendered_payload_sig": "sig-current",
        "hits": [
            {
                "text": "Deep learning improves reconstruction quality.",
                "meta": {"source_path": r"db\LPR-2025\LPR-2025.en.md"},
            }
        ],
        "rendered_payload": {
            "hits": [
                {
                    "ui_meta": {
                        "source_path": r"db\LPR-2025\LPR-2025.en.md",
                        "summary_line": "## Benefits\nDeep learning improves reconstruction quality.",
                        "why_line": "This hit is directly relevant to the user's question.",
                        "summary_generation": "llm_grounded",
                        "why_generation": "llm_grounded",
                    }
                }
            ]
        },
    }

    out = references_router._get_stored_rendered_pack_payload(
        user_msg_id=24,
        pack=pack,
        guide_mode=False,
        guide_source_path="",
        guide_source_name="",
    )

    assert out is None


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
