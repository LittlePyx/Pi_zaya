from __future__ import annotations

import re
import sqlite3
from pathlib import Path

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

    def get_messages(self, conv_id: str):
        del conv_id
        return []


def test_state_validated_cache_skips_rendered_payload_json_loader(monkeypatch) -> None:
    references_router._REFS_CONVERSATION_CACHE.clear()
    conversation = {
        "mode": "normal",
        "updated_at": 123.0,
    }
    refs_state = {
        "rows": [
            {
                "user_msg_id": 7,
                "prompt_sig": "prompt-7",
                "rendered_payload_sig": "render-7",
                "render_status": "pending",
                "updated_at": 124.0,
                "rendered_payload_json_chars": 67_000_000,
            }
        ],
        "messages": {
            "message_count": 0,
            "max_message_id": 0,
            "content_chars": 0,
            "meta_chars": 0,
        },
    }
    state_signature = references_router._refs_conversation_state_signature(
        conversation=conversation,
        refs_state=refs_state,
    )
    cached_payload = {
        7: {
            "render_status": "pending",
            "payload_mode": "pending",
            "hits": [],
        }
    }
    references_router._store_cached_conversation_refs_payload(
        conv_id="conv-validated",
        signature="full-signature-not-needed",
        state_signature=state_signature,
        payload=cached_payload,
        mode="pending",
    )

    class Store:
        def get_conversation(self, conv_id: str, *, timeout_s=None):
            del conv_id, timeout_s
            return dict(conversation)

        def list_message_refs_state(self, conv_id: str, *, timeout_s=None):
            del conv_id, timeout_s
            return dict(refs_state)

        def list_message_refs(self, conv_id: str, *, timeout_s=None):
            raise AssertionError(
                f"validated cache must not deserialize rendered payloads for {conv_id}"
            )

        def get_messages(self, conv_id: str):
            del conv_id
            return []

    monkeypatch.setattr(references_router, "get_chat_store", lambda: Store())
    monkeypatch.setattr(references_router, "_reference_asset_roots", lambda: [])

    response = Response()
    out = references_router.get_conversation_refs(
        "conv-validated",
        response=response,
    )

    assert out == cached_payload
    assert response.headers["x-kb-refs-mode"] == "cache_validated_pending"


def test_reference_cards_follow_grounded_answer_citations(monkeypatch) -> None:
    source_path = r"F:\db\DL-SPI\DL-SPI.en.md"
    public_source_path = "kb-source/0/DL-SPI/DL-SPI.en.md"

    class Store:
        def get_messages(self, conv_id: str):
            assert conv_id == "conv"
            return [
                {"id": 10, "role": "user", "content": "深度学习给单像素成像带来的好处和坑是什么？"},
                {
                    "id": 11,
                    "role": "assistant",
                    "content": "answer",
                    "meta": {
                        "paper_guide_contracts": {
                            "render_packet": {
                                "cite_details": [
                                    {
                                        "citation_route": "system_a",
                                        "source_path": source_path,
                                        "source_name": "DL-SPI.pdf",
                                        "heading_path": "Abstract",
                                        "evidence_quote": "Deep learning provides exceptional reconstruction quality and fast reconstruction speed.",
                                        "card_takeaway": "深度学习提升重建质量和速度。",
                                    },
                                    {
                                        "citation_route": "system_a",
                                        "source_path": source_path,
                                        "source_name": "DL-SPI.pdf",
                                        "heading_path": "Challenges",
                                        "evidence_quote": "Training is prolonged and generalization is limited.",
                                        "answer_claim": "训练时间长，且泛化能力有限。",
                                    },
                                ]
                            }
                        }
                    },
                },
            ]

    payload = {
        10: {
            "prompt": "深度学习给单像素成像带来的好处和坑是什么？",
            "render_locale": "zh",
            "hits": [
                {
                    "text": "An unrelated neural-network definition.",
                    "meta": {"source_path": public_source_path},
                    "ui_meta": {"source_path": public_source_path, "display_name": "DL-SPI.pdf"},
                },
                {
                    "text": "A duplicate hit.",
                    "meta": {"source_path": public_source_path},
                    "ui_meta": {"source_path": public_source_path, "display_name": "DL-SPI.pdf"},
                },
                {
                    "text": "An unused retrieval candidate.",
                    "meta": {"source_path": "kb-source/0/Other/Other.en.md"},
                    "ui_meta": {
                        "source_path": "kb-source/0/Other/Other.en.md",
                        "display_name": "Other.pdf",
                    },
                },
            ],
        }
    }

    monkeypatch.setattr(references_router, "_ref_card_user_locale", lambda prompt: "zh")
    out = references_router._overlay_refs_payload_with_answer_citations(
        store=Store(),
        conv_id="conv",
        payload=payload,
    )

    hits = out[10]["hits"]
    assert len(hits) == 1
    assert len(out[10]["retrieval_hits"]) == 3
    assert all("Other.pdf" not in str(hit) for hit in hits)
    ui = hits[0]["ui_meta"]
    assert "重建质量和速度" in ui["summary_line"]
    assert "训练时间长" in ui["summary_line"]
    assert "重建质量与速度" in ui["why_line"]
    assert "训练耗时" in ui["why_line"]
    assert "泛化能力" in ui["why_line"]
    assert ui["summary_display_role"] == "guide"
    assert ui["summary_label"] == "导读"
    assert ui["summary_title"] == "这条证据说明什么"
    assert ui["polish_status"] in {"heuristic", "full"}
    assert ui["summary_polish_status"] in {"heuristic", "full"}
    assert ui["why_polish_status"] in {"heuristic", "full"}
    assert [section["id"] for section in ui["card_view"]["sections"]] == ["summary", "why", "location"]
    assert out[10]["render_status"] == "full"
    assert out[10]["payload_mode"] == "full"
    assert out[10]["display_state"] == "ready"
    assert out[10]["answer_aligned_citation_cards"] is True
    assert "enrichment_pending" not in out[10]


def test_completed_answer_citation_overlays_do_not_need_background_warm() -> None:
    class Store:
        def get_messages(self, conv_id: str):
            assert conv_id == "conv"
            return [
                {"id": 10, "role": "user", "content": "How did the method evolve?"},
                {
                    "id": 11,
                    "role": "assistant",
                    "content": "The first paper established the coded acquisition model [1].",
                    "meta": {
                        "paper_guide_contracts": {
                            "render_packet": {
                                "cite_details": [
                                    {
                                        "citation_route": "system_a",
                                        "source_path": r"F:\db\CASSI\CASSI.en.md",
                                        "source_name": "CASSI.pdf",
                                        "heading_path": "Method",
                                        "answer_claim": "The first paper established the coded acquisition model.",
                                        "evidence_quote": "The camera uses coded aperture snapshot spectral imaging.",
                                    }
                                ]
                            }
                        }
                    },
                },
            ]

    refs = {
        10: {
            "prompt": "How did the method evolve?",
            "hits": [{"meta": {"source_path": "kb-source/0/CASSI/CASSI.en.md"}}],
        },
        20: {
            "prompt": "What remains uncertain?",
            "hits": [{"meta": {"source_path": "kb-source/0/Other/Other.en.md"}}],
        },
    }

    remaining = references_router._refs_without_completed_answer_citation_overlays(
        store=Store(),
        conv_id="conv",
        refs=refs,
    )

    assert list(remaining) == [20]


def test_reading_route_cards_use_user_language_and_distinct_source_roles() -> None:
    prompt = "我刚开始看单像素成像，想先建立主线，应该先读哪几篇？"
    cards = [
        references_router._answer_citation_card_copy(
            [
                {
                    "source_name": "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf",
                    "heading_path": "Acquisition and image reconstruction strategies",
                    "evidence_quote": (
                        "Compressed sensing recovers images when the number of measurements "
                        "is fewer than the total number of unknown pixels."
                    ),
                }
            ],
            prefer_zh=True,
            prompt=prompt,
        ),
        references_router._answer_citation_card_copy(
            [
                {
                    "source_name": "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
                    "heading_path": "Introduction",
                    "evidence_quote": (
                        "HSI uses Hadamard basis patterns while FSI uses Fourier basis patterns; "
                        "the paper compares imaging efficiency and noise robustness."
                    ),
                }
            ],
            prefer_zh=True,
            prompt=prompt,
        ),
        references_router._answer_citation_card_copy(
            [
                {
                    "source_name": "LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.pdf",
                    "heading_path": "Abstract",
                    "evidence_quote": (
                        "Deep learning brings exceptional reconstruction quality and fast "
                        "reconstruction speed to single-pixel imaging."
                    ),
                }
            ],
            prefer_zh=True,
            prompt=prompt,
        ),
    ]

    assert all(re.search(r"[\u4e00-\u9fff]", summary) for summary, _why in cards)
    assert "采集与重建基础" in cards[0][1]
    assert "两种经典调制方案" in cards[1][1]
    assert "学习型方法" in cards[2][1]
    assert len({why for _summary, why in cards}) == 3


def test_denoising_method_map_card_uses_distinct_grounded_relevance_copy() -> None:
    summary, why = references_router._answer_citation_card_copy(
        [
            {
                "source_name": "Brief review of image denoising techniques.pdf",
                "heading_path": "Classical denoising method",
                "answer_claim": "经典去噪方法分为空间域方法与变换域方法。",
                "evidence_quote": (
                    "Image denoising methods are classified as spatial domain methods and "
                    "transform domain methods. Spatial methods use correlations between "
                    "pixels or image patches."
                ),
            }
        ],
        prefer_zh=True,
        prompt="请给我一张经典去噪方法阅读路线图。",
    )

    assert summary
    assert "空间域" in why
    assert "变换域" in why
    assert why != summary


def test_answer_citation_overlay_uses_source_identity_for_scinerf_relevance(monkeypatch) -> None:
    source_path = r"F:\db\SCINeRF\SCINeRF.en.md"

    class Store:
        def get_messages(self, conv_id: str):
            assert conv_id == "conv"
            return [
                {
                    "id": 30,
                    "role": "user",
                    "content": "SCI 是怎么走到 3D 场景重建的？",
                },
                {
                    "id": 31,
                    "role": "assistant",
                    "content": "answer",
                    "meta": {
                        "paper_guide_contracts": {
                            "render_packet": {
                                "cite_details": [
                                    {
                                        "citation_route": "system_a",
                                        "source_path": source_path,
                                        "source_name": "CVPR-2024-SCINeRF.pdf",
                                        "heading_path": "SCINeRF / Abstract",
                                        "answer_claim": "SCINeRF 将 SCI 物理成像过程纳入 NeRF 联合优化。",
                                        "evidence_quote": (
                                            "we formulate the physical imaging process of SCI as part "
                                            "of the training of NeRF"
                                        ),
                                    }
                                ]
                            }
                        }
                    },
                },
            ]

    payload = {
        30: {
            "prompt": "SCI 是怎么走到 3D 场景重建的？",
            "render_locale": "zh",
            "hits": [
                {
                    "meta": {"source_path": "kb-source/0/SCINeRF/SCINeRF.en.md"},
                    "ui_meta": {
                        "source_path": "kb-source/0/SCINeRF/SCINeRF.en.md",
                        "display_name": "CVPR-2024-SCINeRF.pdf",
                    },
                }
            ],
        }
    }

    monkeypatch.setattr(references_router, "_ref_card_user_locale", lambda prompt: "zh")
    out = references_router._overlay_refs_payload_with_answer_citations(
        store=Store(),
        conv_id="conv",
        payload=payload,
    )

    why = out[30]["hits"][0]["ui_meta"]["why_line"]
    assert all(term in why for term in ("SCI", "NeRF", "物理成像", "三维"))
    assert "逐项核对" not in why


def test_answer_citation_overlay_explains_admm_is_prior_work(monkeypatch) -> None:
    source_path = r"F:\db\SCINeRF\SCINeRF.en.md"

    class Store:
        def get_messages(self, conv_id: str):
            assert conv_id == "conv"
            return [
                {
                    "id": 35,
                    "role": "user",
                    "content": "ADMM 是作者自己发明的吗？我应该把它当成这篇论文的新东西吗？",
                },
                {
                    "id": 36,
                    "role": "assistant",
                    "content": "不是，ADMM 在这里属于已有方法。",
                    "meta": {
                        "paper_guide_contracts": {
                            "render_packet": {
                                "cite_details": [
                                    {
                                        "citation_route": "system_a",
                                        "source_path": source_path,
                                        "source_name": "CVPR-2024-SCINeRF.pdf",
                                        "heading_path": "SCINeRF / 2. Related Work",
                                        "answer_claim": "ADMM 是已有方法，不是本文原创。",
                                        "evidence_quote": (
                                            "Most existing methods employ ADMM for iterative optimization."
                                        ),
                                    }
                                ]
                            }
                        }
                    },
                },
            ]

    monkeypatch.setattr(references_router, "_ref_card_user_locale", lambda prompt: "zh")
    out = references_router._overlay_refs_payload_with_answer_citations(
        store=Store(),
        conv_id="conv",
        payload={
            35: {
                "prompt": "ADMM 是作者自己发明的吗？我应该把它当成这篇论文的新东西吗？",
                "hits": [
                    {
                        "meta": {"source_path": "kb-source/0/SCINeRF/SCINeRF.en.md"},
                        "ui_meta": {
                            "source_path": "kb-source/0/SCINeRF/SCINeRF.en.md",
                            "display_name": "CVPR-2024-SCINeRF.pdf",
                        },
                    }
                ],
            }
        },
    )

    ui = out[35]["hits"][0]["ui_meta"]
    assert all(term in ui["why_line"] for term in ("ADMM", "已有方法", "不是本文新提出"))
    assert "逐项核对" not in ui["why_line"]
    assert ui["card_view"]["sections"][1]["text"] == ui["why_line"]
    assert out[35]["render_status"] == "full"


def test_answer_citation_overlay_uses_saved_locale_when_pack_has_no_locale(monkeypatch) -> None:
    source_path = r"F:\db\SCINeRF\SCINeRF.en.md"

    class Store:
        def get_messages(self, conv_id: str):
            assert conv_id == "conv"
            return [
                {
                    "id": 40,
                    "role": "user",
                    "content": "SCI 是怎么走到 3D 场景重建的？",
                },
                {
                    "id": 41,
                    "role": "assistant",
                    "content": "answer",
                    "meta": {
                        "paper_guide_contracts": {
                            "render_packet": {
                                "cite_details": [
                                    {
                                        "citation_route": "system_a",
                                        "source_path": source_path,
                                        "source_name": "CVPR-2024-SCINeRF.pdf",
                                        "heading_path": "SCINeRF / Abstract",
                                        "answer_claim": "SCINeRF adds the SCI physical imaging process to NeRF training.",
                                        "evidence_quote": (
                                            "we formulate the physical imaging process of SCI as part "
                                            "of the training of NeRF"
                                        ),
                                    }
                                ]
                            }
                        }
                    },
                },
            ]

    monkeypatch.setattr(references_router, "_ref_card_user_locale", lambda prompt: "en")
    out = references_router._overlay_refs_payload_with_answer_citations(
        store=Store(),
        conv_id="conv",
        payload={
            40: {
                "prompt": "SCI 是怎么走到 3D 场景重建的？",
                "hits": [
                    {
                        "meta": {"source_path": "kb-source/0/SCINeRF/SCINeRF.en.md"},
                        "ui_meta": {
                            "source_path": "kb-source/0/SCINeRF/SCINeRF.en.md",
                            "display_name": "CVPR-2024-SCINeRF.pdf",
                        },
                    }
                ],
            }
        },
    )

    ui = out[40]["hits"][0]["ui_meta"]
    assert ui["render_locale"] == "en"
    assert ui["summary_label"] == "Guide"
    assert ui["card_view"]["sections"][1]["label"] == "Relevance"
    assert "SCINeRF incorporates the physical SCI imaging process" in ui["summary_line"]
    assert "lineage evidence" in ui["why_line"]
    assert "neural scene representation" in ui["why_line"]
    assert not re.search(r"[\u4e00-\u9fff]", ui["why_line"])


def test_reference_cards_stay_pending_until_planned_answer_citations_are_ready() -> None:
    source_path = r"F:\db\Paper\Paper.en.md"

    class Store:
        def get_messages(self, conv_id: str):
            assert conv_id == "conv"
            return [
                {"id": 20, "role": "user", "content": "What does the paper show?"},
                {
                    "id": 21,
                    "role": "assistant",
                    "content": "Answer still converging.",
                    "meta": {
                        "answer_quality": {
                            "citation_plan": {
                                "slots": [
                                    {
                                        "preferred_system": "system_a",
                                        "source_path": source_path,
                                    }
                                ]
                            }
                        },
                        "paper_guide_contracts": {"render_packet": {"cite_details": []}},
                    },
                },
            ]

    payload = {
        20: {
            "render_status": "full",
            "payload_mode": "full",
            "hits": [
                {
                    "meta": {"source_path": "kb-source/0/Paper/Paper.en.md"},
                    "ui_meta": {
                        "display_name": "Paper.pdf",
                        "summary_line": "Generic card copy that should not end polling.",
                        "why_line": "",
                    },
                }
            ],
        }
    }

    out = references_router._overlay_refs_payload_with_answer_citations(
        store=Store(),
        conv_id="conv",
        payload=payload,
    )

    assert out[20]["enrichment_pending"] is True
    assert out[20]["answer_citation_overlay_pending"] is True


def test_reference_cards_use_evidence_bearing_system_a_plan_while_render_packet_converges() -> None:
    source_path = r"F:\db\Paper\Paper.en.md"

    class Store:
        def get_messages(self, conv_id: str):
            assert conv_id == "conv"
            return [
                {"id": 22, "role": "user", "content": "What does the paper show?"},
                {
                    "id": 23,
                    "role": "assistant",
                    "content": "The method improves reconstruction quality [1].",
                    "meta": {
                        "answer_quality": {
                            "citation_plan": {
                                "slots": [
                                    {
                                        "preferred_system": "system_a",
                                        "source_path": source_path,
                                        "source_name": "Paper.pdf",
                                        "heading_path": "Results",
                                        "evidence_quote": (
                                            "The proposed method improves reconstruction quality "
                                            "without increasing acquisition time."
                                        ),
                                        "candidate_hits": [1],
                                        "block_id": "blk-results",
                                    }
                                ]
                            }
                        },
                        "paper_guide_contracts": {"render_packet": {"cite_details": []}},
                    },
                },
            ]

    payload = {
        22: {
            "prompt": "What does the paper show?",
            "render_status": "pending",
            "payload_mode": "pending",
            "hits": [
                {
                    "meta": {"source_path": source_path},
                    "ui_meta": {
                        "source_path": source_path,
                        "display_name": "Paper.pdf",
                    },
                }
            ],
        }
    }

    out = references_router._overlay_refs_payload_with_answer_citations(
        store=Store(),
        conv_id="conv",
        payload=payload,
    )

    pack = out[22]
    assert pack["display_state"] == "ready"
    assert pack["payload_mode"] == "full"
    assert pack["answer_aligned_citation_cards"] is True
    assert "enrichment_pending" not in pack
    assert len(pack["hits"]) == 1
    hit = pack["hits"][0]
    assert hit["ui_meta"]["summary_line"]
    assert hit["ui_meta"]["why_line"]
    assert hit["ui_meta"]["primary_evidence"]["block_id"] == "blk-results"


def test_evidence_bearing_plan_does_not_schedule_redundant_refs_warm() -> None:
    source_path = r"F:\db\Paper\Paper.en.md"

    class Store:
        def get_messages(self, conv_id: str):
            assert conv_id == "conv"
            return [
                {"id": 24, "role": "user", "content": "What does the paper show?"},
                {
                    "id": 25,
                    "role": "assistant",
                    "content": "The method improves quality [1].",
                    "meta": {
                        "answer_quality": {
                            "citation_plan": {
                                "slots": [
                                    {
                                        "preferred_system": "system_a",
                                        "source_path": source_path,
                                        "evidence_quote": (
                                            "The proposed method improves reconstruction quality "
                                            "without increasing acquisition time."
                                        ),
                                    }
                                ]
                            }
                        }
                    },
                },
            ]

    refs = {
        24: {
            "prompt": "What does the paper show?",
            "hits": [{"meta": {"source_path": source_path}}],
        }
    }

    assert (
        references_router._refs_without_completed_answer_citation_overlays(
            store=Store(),
            conv_id="conv",
            refs=refs,
        )
        == {}
    )


def test_persist_rendered_refs_payload_drops_previous_nested_render(monkeypatch):
    class Store:
        def __init__(self):
            self.saved = None

        def set_message_refs_rendered_payload(self, **kwargs):
            self.saved = dict(kwargs)

    store = Store()
    refs = {1: {"prompt": "question", "prompt_sig": "prompt-sig", "hits": []}}
    payload = {
        1: {
            "hits": [],
            "rendered_payload": {"hits": [{"text": "stale"}]},
            "rendered_payload_sig": "stale-sig",
        }
    }
    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)

    references_router._persist_rendered_refs_payloads(
        refs=refs,
        payload=payload,
        guide_mode=False,
        guide_source_path="",
        guide_source_name="",
    )

    saved_payload = dict((store.saved or {}).get("rendered_payload") or {})
    assert "rendered_payload" not in saved_payload
    assert "rendered_payload_sig" not in saved_payload
    assert saved_payload["hits"] == []


def test_first_read_rebuilds_only_latest_pack_and_reuses_shallow_historical_snapshot(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()
    refs = {
        1: {
            "prompt": "older question",
            "prompt_sig": "old-prompt",
            "hits": [{"text": "old raw", "meta": {"source_path": "old.md", "ref_pack_state": "ready"}}],
            "rendered_payload": {
                "hits": [
                    {
                        "text": "older ready card",
                        "ui_meta": {
                            "display_name": "Old paper.pdf",
                            "summary_line": "Older evidence-grounded guide.",
                            "why_line": "It supports the older question with a specific result.",
                        },
                    }
                ],
                "rendered_payload": {"hits": [{"text": "recursively nested stale card"}]},
            },
            "rendered_payload_sig": "old-schema-sig",
        },
        2: {
            "prompt": "latest question",
            "prompt_sig": "latest-prompt",
            "hits": [{"text": "latest raw", "meta": {"source_path": "latest.md", "ref_pack_state": "ready"}}],
        },
    }
    store = _FakeStore({"mode": "chat"}, refs)
    enriched_keys: list[list[int]] = []
    warmed_keys: list[list[int]] = []

    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(references_router, "_pdf_dir", lambda: None)
    monkeypatch.setattr(references_router, "_md_dir", lambda: None)
    monkeypatch.setattr(references_router, "_lib_store", lambda: None)
    monkeypatch.setattr(
        references_router,
        "_warm_conversation_refs_payload_async",
        lambda **kwargs: warmed_keys.append(sorted(int(key) for key in dict(kwargs.get("refs") or {}))),
    )

    def fake_enrich(payload, **kwargs):
        del kwargs
        keys = sorted(int(key) for key in dict(payload or {}))
        enriched_keys.append(keys)
        return {
            2: {
                "hits": [
                    {
                        "text": "latest fast card",
                        "ui_meta": {
                            "display_name": "Latest paper.pdf",
                            "summary_line": "Latest evidence-grounded guide.",
                            "why_line": "It supports the latest question with a specific result.",
                        },
                    }
                ]
            }
        }

    monkeypatch.setattr(references_router, "enrich_refs_payload", fake_enrich)

    out = references_router.get_conversation_refs("conv-latest-first")

    assert enriched_keys == [[2]]
    assert warmed_keys == [[1, 2]]
    assert out[1]["hits"][0]["text"] == "older ready card"
    assert out[2]["hits"][0]["text"] == "latest fast card"
    assert "rendered_payload" not in out[1]


def test_attach_assistant_answers_to_refs_keeps_alignment_text_internal():
    class Store:
        def get_messages(self, conv_id: str):
            assert conv_id == "conv-answer"
            return [
                {"id": 10, "role": "user", "content": "question"},
                {"id": 11, "role": "assistant", "content": "grounded final answer"},
                {"id": 12, "role": "user", "content": "other"},
                {"id": 13, "role": "assistant", "content": "other answer"},
            ]

    refs = {10: {"prompt": "question", "hits": []}}
    attached = references_router._attach_assistant_answers_to_refs(
        store=Store(),
        conv_id="conv-answer",
        refs=refs,
    )

    assert attached[10]["answer_text"] == "grounded final answer"
    assert len(attached[10]["answer_sig"]) == 40
    assert "answer_text" not in references_router.public_refs_payload_projection(attached)[10]
    assert "answer_sig" not in references_router.public_refs_payload_projection(attached)[10]


def test_attach_assistant_answers_recovers_canonical_cited_hits(tmp_path: Path):
    realtime = tmp_path / "realtime.en.md"
    realtime.write_text(
        "<!-- kb_page: 3 -->\n## Results\nThe system reconstructs real-time single-pixel video "
        "at a frame rate of 30 Hz using 333 illumination patterns.\n",
        encoding="utf-8",
    )

    class Store:
        def get_messages(self, conv_id: str):
            assert conv_id == "conv-canonical-answer"
            return [
                {"id": 20, "role": "user", "content": "question"},
                {
                    "id": 21,
                    "role": "assistant",
                    "content": "该方法以 333 个图案实现 30 Hz 实时成像。[1]",
                    "meta": {"canonical_hit_paths": [str(realtime)]},
                },
            ]

    attached = references_router._attach_assistant_answers_to_refs(
        store=Store(),
        conv_id="conv-canonical-answer",
        refs={20: {"prompt": "question", "hits": [{"text": "unrelated seed", "meta": {"source_path": "seed.md"}}]}},
    )

    recovered = next(
        hit for hit in attached[20]["hits"] if hit.get("meta", {}).get("ref_answer_citation_num") == 1
    )
    assert len(attached[20]["hits"]) == 1
    assert recovered["meta"]["source_path"] == str(realtime)
    assert "30 Hz" in recovered["text"]


def test_attach_assistant_answers_orders_cited_hits_by_answer_occurrence(monkeypatch):
    first = {
        "text": "First cited evidence.",
        "meta": {"source_path": "first.md", "ref_answer_citation_num": 1},
    }
    second = {
        "text": "Second cited evidence.",
        "meta": {"source_path": "second.md", "ref_answer_citation_num": 2},
    }

    class Store:
        def get_messages(self, conv_id: str):
            assert conv_id == "conv-citation-order"
            return [
                {"id": 30, "role": "user", "content": "question"},
                {
                    "id": 31,
                    "role": "assistant",
                    "content": "先讨论第二篇。[2] 再讨论第一篇。[1]",
                    "meta": {"canonical_hit_paths": ["first.md", "second.md"]},
                },
            ]

    monkeypatch.setattr(
        "api.chat_render._augment_hits_with_canonical_answer_citations",
        lambda *_args, **_kwargs: [first, second],
    )

    attached = references_router._attach_assistant_answers_to_refs(
        store=Store(),
        conv_id="conv-citation-order",
        refs={30: {"prompt": "question", "hits": []}},
    )

    assert [hit["meta"]["ref_answer_citation_num"] for hit in attached[30]["hits"]] == [2, 1]


def test_attach_assistant_answers_aligns_only_latest_turn_on_read(monkeypatch):
    references_router._CANONICAL_ANSWER_HITS_CACHE.clear()
    calls: list[str] = []

    class Store:
        def get_messages(self, conv_id: str):
            assert conv_id == "conv-latest-align"
            return [
                {"id": 1, "role": "user", "content": "older"},
                {
                    "id": 2,
                    "role": "assistant",
                    "content": "older answer [1]",
                    "meta": {"canonical_hit_paths": ["older-cited.md"]},
                },
                {"id": 3, "role": "user", "content": "latest"},
                {
                    "id": 4,
                    "role": "assistant",
                    "content": "latest answer [1]",
                    "meta": {"canonical_hit_paths": ["latest-cited.md"]},
                },
            ]

    def fake_augment(hits, *, canonical_paths, answer_text):
        del hits
        calls.append(answer_text)
        return [
            {
                "text": canonical_paths[0],
                "meta": {"source_path": canonical_paths[0], "ref_answer_citation_num": 1},
            }
        ]

    monkeypatch.setattr("api.chat_render._augment_hits_with_canonical_answer_citations", fake_augment)
    attached = references_router._attach_assistant_answers_to_refs(
        store=Store(),
        conv_id="conv-latest-align",
        refs={
            1: {"prompt": "older", "hits": [{"text": "older seed", "meta": {"source_path": "old.md"}}]},
            3: {"prompt": "latest", "hits": [{"text": "latest seed", "meta": {"source_path": "new.md"}}]},
        },
    )
    attached_again = references_router._attach_assistant_answers_to_refs(
        store=Store(),
        conv_id="conv-latest-align",
        refs={
            1: {"prompt": "older", "hits": [{"text": "older seed", "meta": {"source_path": "old.md"}}]},
            3: {"prompt": "latest", "hits": [{"text": "latest seed", "meta": {"source_path": "new.md"}}]},
        },
    )

    assert calls == ["latest answer [1]"]
    assert attached[1]["hits"][0]["text"] == "older seed"
    assert attached[1]["_canonical_answer_paths"] == ["older-cited.md"]
    assert attached[3]["hits"][0]["text"] == "latest-cited.md"
    assert attached[3]["_canonical_answer_paths_aligned"] is True
    assert attached_again[3]["hits"][0]["text"] == "latest-cited.md"


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


def test_get_conversation_refs_applies_public_payload_projection(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()
    source_path = r"F:\private\library\Paper.en.md"
    refs = {
        1: {
            "prompt": "Where is the method described?",
            "hits": [
                {"text": "hit", "meta": {"source_path": source_path, "ref_pack_state": "ready"}}
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
        "hydrate_refs_payload_citation_meta",
        lambda payload, **kwargs: dict(payload or {}),
    )
    monkeypatch.setattr(
        references_router,
        "enrich_refs_payload",
        lambda *args, **kwargs: {
            1: {
                "pipeline_debug": {"reranker": "internal"},
                "scores": [8.0],
                "rendered_payload": {"private": True},
                "hits": [
                    {
                        "meta": {
                            "source_path": source_path,
                            "explicit_doc_match_score": 15.0,
                        },
                        "ui_meta": {
                            "source_path": source_path,
                            "polish_detail": "internal",
                            "primary_evidence": {
                                "source_path": source_path,
                                "heading_path": "Methods",
                            },
                        },
                    }
                ],
            }
        },
    )

    out = references_router.get_conversation_refs("conv-public-projection")

    pack = out[1]
    assert "pipeline_debug" not in pack
    assert "scores" not in pack
    assert "rendered_payload" not in pack
    assert "explicit_doc_match_score" not in pack["hits"][0]["meta"]
    assert "polish_detail" not in pack["hits"][0]["ui_meta"]
    assert pack["hits"][0]["ui_meta"]["source_path"] == "Paper.en.md"
    assert "source_path" not in pack["hits"][0]["ui_meta"]["primary_evidence"]


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


def test_build_pending_conversation_refs_payload_localizes_provisional_evidence(monkeypatch):
    monkeypatch.setattr(references_router, "_ref_card_user_locale", lambda prompt: "zh")
    monkeypatch.setattr(
        references_router,
        "_filter_pending_refs_hits_by_prompt_focus",
        lambda prompt, hits: list(hits),
    )

    out = references_router._build_pending_conversation_refs_payload(
        {
            19: {
                "prompt": "How does SCINeRF connect snapshot compressive imaging to NeRF?",
                "hits": [
                    {
                        "text": "We model the physical imaging process of SCI into NeRF training.",
                        "meta": {
                            "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
                            "ref_pack_state": "pending",
                            "ref_best_heading_path": "Abstract",
                        },
                    }
                ],
            }
        },
        doc_list_by_user={},
    )

    ui = out[19]["hits"][0]["ui_meta"]
    assert ui["render_locale"] == "zh"
    assert ui["summary_kind"] == "evidence"
    assert ui["summary_display_role"] == "source_evidence"
    assert ui["summary_label"] == "原文证据"
    assert "核验" in ui["why_line"]
    assert "This pending match" not in ui["why_line"]


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
    monkeypatch.setattr(
        references_router,
        "_warm_conversation_refs_payload_async",
        lambda **kwargs: calls.setdefault("warm", dict(kwargs)),
    )
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
    assert pack["render_status"] == "fast"
    assert pack["payload_mode"] == "fast"
    assert pack["enrichment_pending"] is True
    assert dict(calls.get("kwargs") or {}).get("allow_exact_locate") is False
    assert dict(calls.get("kwargs") or {}).get("allow_expensive_llm") is False
    assert dict(calls.get("kwargs") or {}).get("allow_citation_prefetch") is False
    assert dict(calls.get("kwargs") or {}).get("apply_copy_polish") is False
    assert dict(calls.get("kwargs") or {}).get("seed_only") is True
    assert store.persisted == []
    warm = dict(calls.get("warm") or {})
    assert list((warm.get("authoritative_doc_list_by_user") or {}).get(21) or []) == messages[1]["meta"]["paper_guide_contracts"]["doc_list"]


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
    assert dict(calls.get("kwargs") or {}).get("apply_copy_polish") is False
    assert dict(calls.get("kwargs") or {}).get("seed_only") is True
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
    monkeypatch.setattr(
        references_router,
        "_warm_conversation_refs_payload_async",
        lambda **kwargs: calls.setdefault("warm", dict(kwargs)),
    )
    monkeypatch.setattr(references_router, "build_doc_list_refs_payload", fake_build_doc_list_refs_payload)

    out = references_router.get_conversation_refs("conv-empty-doc-list-chat")

    assert [str(item.get("source_path") or "") for item in list(calls.get("doc_list") or [])] == [
        r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md"
    ]
    assert str(out[71]["display_state"] or "") == "ready"
    assert list(out[71]["hits"] or [])
    assert str(out[71]["payload_mode"] or "") == "fast"
    assert out[71]["enrichment_pending"] is True
    assert store.persisted == []
    assert 71 in dict((calls.get("warm") or {}).get("authoritative_doc_list_by_user") or {})


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

    source_path = r"F:\private\cached\Paper.en.md"
    cached_payload = {
        9: {
            "render_error": "cached_render_failure",
            "render_error_detail": r"failed at F:\private\cached\trace.json",
            "render_attempts": 2,
            "render_evidence_sig": "cached-private-signature",
            "hits": [
                {
                    "ui_meta": {
                        "summary_line": "cached",
                        "source_path": source_path,
                        "primary_evidence": {
                            "source_path": source_path,
                            "heading_path": "Methods",
                        },
                    }
                }
            ],
        }
    }
    references_router._store_cached_conversation_refs_payload(
        conv_id="conv-busy",
        signature="sig",
        payload=cached_payload,
    )

    monkeypatch.setattr(references_router, "get_chat_store", lambda: _BusyStore())

    out = references_router.get_conversation_refs("conv-busy")

    pack = out[9]
    assert pack["hits"][0]["ui_meta"]["summary_line"] == "cached"
    assert pack["hits"][0]["ui_meta"]["source_path"] == "Paper.en.md"
    assert "source_path" not in pack["hits"][0]["ui_meta"]["primary_evidence"]
    assert "render_error" not in pack
    assert "render_error_detail" not in pack
    assert "render_attempts" not in pack
    assert "render_evidence_sig" not in pack


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
    source_path = r"db\SciAdv-2017\SciAdv-2017.en.md"
    nested_private_path = r"F:\private\stored\SciAdv-2017.en.md"
    rendered_payload = {
        11: {
            "hits": [
                {
                    "ui_meta": {
                        "summary_line": "full-persisted",
                        "source_path": source_path,
                        "primary_evidence": {
                            "source_path": nested_private_path,
                            "heading_path": "Introduction",
                        },
                    }
                }
            ]
        }
    }
    refs[11]["rendered_payload"] = dict(rendered_payload[11])
    refs[11]["render_status"] = "full"
    refs[11]["render_error"] = "stale-private-error"
    refs[11]["render_error_detail"] = r"failed at F:\private\stored\trace.json"
    refs[11]["render_attempts"] = 1
    refs[11]["render_evidence_sig"] = "stored-private-signature"
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
    assert "render_error" not in out[11]
    assert "render_error_detail" not in out[11]
    assert "render_attempts" not in out[11]
    assert "render_evidence_sig" not in out[11]
    assert out[11]["hits"][0]["ui_meta"]["summary_line"] == "full-persisted"
    assert out[11]["hits"][0]["ui_meta"]["source_path"] == source_path
    assert "source_path" not in out[11]["hits"][0]["ui_meta"]["primary_evidence"]
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


def test_multi_source_rendered_payload_is_stale_when_one_lineage_source_disappears():
    pack = {
        "prompt": "SCI 这条线是怎么从光谱成像走到 3D 场景重建的？",
        "hits": [
            {"meta": {"source_path": r"db\CASSI\CASSI.en.md"}},
            {"meta": {"source_path": r"db\SCINeRF\SCINeRF.en.md"}},
            {"meta": {"source_path": r"db\SCIGS\SCIGS.en.md"}},
        ],
    }
    payload = {
        "hits": [
            {"ui_meta": {"source_path": r"db\SCINeRF\SCINeRF.en.md"}},
            {"ui_meta": {"source_path": r"db\SCIGS\SCIGS.en.md"}},
            {"ui_meta": {"source_path": r"db\Unrelated\Unrelated.en.md"}},
        ],
    }

    assert references_router._stored_rendered_pack_payload_lost_current_hits(
        payload=payload,
        pack=pack,
    )


def test_rendered_payload_same_source_with_mixed_slashes_is_not_stale():
    pack = {
        "hits": [
            {"meta": {"source_path": "F:/repo/db/Paper/Paper.en.md"}},
        ]
    }
    payload = {
        "hits": [
            {"ui_meta": {"source_path": r"F:\repo\db\Paper\Paper.en.md"}},
        ]
    }

    assert not references_router._stored_rendered_pack_payload_lost_current_hits(
        payload=payload,
        pack=pack,
    )


def test_six_paper_payload_is_stale_when_fifth_source_disappears():
    sources = [rf"db\Paper-{idx}\Paper-{idx}.en.md" for idx in range(1, 7)]
    pack = {
        "prompt": "请列出并比较这六篇文献",
        "hits": [
            {"meta": {"source_path": source_path}}
            for source_path in sources
        ],
    }
    payload = {
        "hits": [
            {"ui_meta": {"source_path": source_path}}
            for source_path in [*sources[:4], sources[5]]
        ],
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
            "render_error_detail": r"failed at F:\private\failed\trace.json",
            "render_attempts": 4,
            "render_evidence_sig": "failed-private-signature",
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
            return {
                3: {
                    "mode": "fast",
                    "hits": [
                        {
                            "ui_meta": {
                                "summary_line": "fast",
                                "primary_evidence": {
                                    "source_path": r"F:\private\failed\Paper.en.md",
                                    "heading_path": "Methods",
                                },
                            }
                        }
                    ],
                }
            }
        return {3: {"mode": "full", "hits": [{"ui_meta": {"summary_line": "full"}}]}}

    monkeypatch.setattr(references_router, "enrich_refs_payload", fake_enrich_refs_payload)

    out = references_router.get_conversation_refs("conv-fast")

    assert out[3]["mode"] == "fast"
    assert out[3]["payload_mode"] == "fast"
    assert out[3]["render_status"] == "failed"
    assert "render_error" not in out[3]
    assert "render_error_detail" not in out[3]
    assert "render_attempts" not in out[3]
    assert "render_evidence_sig" not in out[3]
    assert out[3]["display_state"] == "ready"
    assert out[3]["hits"][0]["ui_meta"]["summary_line"] == "fast"
    assert "source_path" not in out[3]["hits"][0]["ui_meta"]["primary_evidence"]
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
    assert kwargs.get("allow_exact_locate") is False
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


def test_warm_conversation_refs_payload_async_skips_llm_for_answer_citation_cards(monkeypatch):
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
        return {13: {"hits": [{"ui_meta": {"summary_line": "grounded"}}]}}

    monkeypatch.setattr(references_router, "enrich_refs_payload", fake_enrich_refs_payload)
    monkeypatch.setattr(references_router, "_persist_rendered_refs_payloads", lambda **_kwargs: None)
    monkeypatch.setattr(references_router, "_store_cached_conversation_refs_payload", lambda **_kwargs: None)

    references_router._warm_conversation_refs_payload_async(
        conv_id="conv-warm-answer-citations",
        signature="sig-warm-answer-citations",
        refs={
            13: {
                "prompt": "Compare the three methods.",
                "hits": [
                    {
                        "text": "Direct evidence.",
                        "meta": {
                            "source_path": "paper.en.md",
                            "ref_answer_citation_num": 1,
                        },
                    }
                ],
            }
        },
        guide_mode=False,
        guide_source_path="",
        guide_source_name="",
    )

    kwargs = dict(calls.get("kwargs") or {})
    assert kwargs.get("allow_expensive_llm_for_ready") is False
    assert kwargs.get("allow_exact_locate") is False


def test_warm_conversation_refs_payload_async_polishes_authoritative_doc_list_and_merges_cache(monkeypatch):
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
    references_router._store_cached_conversation_refs_payload(
        conv_id="conv-authoritative-warm",
        signature="sig-authoritative-warm",
        payload={5: {"hits": [{"ui_meta": {"summary_line": "existing full card"}}]}},
        mode="fast",
    )

    doc_list = [
        {"source_path": r"db\A\A.en.md", "source_name": "A.pdf"},
        {"source_path": r"db\B\B.en.md", "source_name": "B.pdf"},
    ]

    def fake_build_doc_list_refs_payload(*, user_msg_id, pack, doc_list, **kwargs):
        del pack
        calls["user_msg_id"] = int(user_msg_id)
        calls["doc_list"] = list(doc_list or [])
        calls["kwargs"] = dict(kwargs)
        return {"hits": [{"ui_meta": {"summary_generation": "llm_grounded"}}]}

    monkeypatch.setattr(references_router, "build_doc_list_refs_payload", fake_build_doc_list_refs_payload)
    monkeypatch.setattr(
        references_router,
        "enrich_refs_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("authoritative packs use the doc-list renderer")),
    )
    monkeypatch.setattr(
        references_router,
        "_persist_rendered_refs_payloads",
        lambda **kwargs: calls.setdefault("persisted", dict(kwargs)),
    )
    monkeypatch.setattr(
        references_router,
        "_store_cached_conversation_refs_payload",
        lambda **kwargs: calls.setdefault("cached", dict(kwargs)),
    )

    references_router._warm_conversation_refs_payload_async(
        conv_id="conv-authoritative-warm",
        signature="sig-authoritative-warm",
        refs={13: {"prompt": "Which papers discuss SCI?", "hits": []}},
        guide_mode=False,
        guide_source_path="",
        guide_source_name="",
        authoritative_doc_list_by_user={13: doc_list},
    )

    assert calls["user_msg_id"] == 13
    assert calls["doc_list"] == doc_list
    assert dict(calls.get("kwargs") or {}).get("allow_expensive_llm") is True
    assert dict(calls.get("kwargs") or {}).get("allow_citation_prefetch") is False
    assert dict(calls.get("persisted") or {}).get("payload") == {
        13: {"hits": [{"ui_meta": {"summary_generation": "llm_grounded"}}]}
    }
    cached = dict(calls.get("cached") or {})
    assert cached.get("mode") == "full"
    assert set(dict(cached.get("payload") or {})) == {5, 13}


def test_stored_authoritative_payload_can_replace_raw_retrieval_source_set():
    prompt = "我刚开始看单像素成像，应该先读哪几篇？"
    raw_source = r"db\LPR\LPR.en.md"
    authoritative_source = r"db\NatPhoton\NatPhoton.en.md"
    payload = {
        "hits": [
            {
                "meta": {"source_path": authoritative_source},
                "ui_meta": {
                    "source_path": authoritative_source,
                    "summary_line": "LLM summary",
                    "why_line": "LLM reason",
                    "summary_generation": "llm_grounded",
                    "why_generation": "llm_grounded",
                },
            }
        ]
    }
    pack = {
        "prompt": prompt,
        "hits": [{"meta": {"source_path": raw_source, "ref_pack_state": "ready"}}],
        "rendered_payload": payload,
    }
    pack["rendered_payload_sig"] = references_router._refs_pack_render_signature(
        user_msg_id=77,
        pack=pack,
        guide_mode=False,
        guide_source_path="",
        guide_source_name="",
    )

    assert references_router._get_stored_rendered_pack_payload(
        user_msg_id=77,
        pack=pack,
        guide_mode=False,
        guide_source_path="",
        guide_source_name="",
    ) is None
    assert references_router._get_stored_rendered_pack_payload(
        user_msg_id=77,
        pack=pack,
        guide_mode=False,
        guide_source_path="",
        guide_source_name="",
        allow_authoritative_source_override=True,
    ) == payload


def test_background_llm_polish_is_opt_in_when_unset(monkeypatch):
    monkeypatch.delenv("KB_REFS_BACKGROUND_LLM_POLISH", raising=False)
    monkeypatch.setattr(references_router, "_refs_card_polish_llm_enabled", lambda: True)

    assert references_router._refs_background_llm_polish_enabled() is False


def test_fast_exact_refs_are_detected_for_two_stage_rendering():
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


def test_full_cache_accepts_matching_authoritative_doc_list_only():
    doc_list = [
        {"source_path": r"db\A\A.en.md", "source_name": "A.pdf"},
        {"source_path": r"db\B\B.en.md", "source_name": "B.pdf"},
    ]
    cached_payload = {
        13: {
            "pipeline_debug": {"doc_list_authoritative": True},
            "hits": [
                {"ui_meta": {"source_path": r"db\A\A.en.md"}},
                {"ui_meta": {"source_path": r"db\B\B.en.md"}},
            ],
        }
    }

    assert references_router._cached_payload_matches_authoritative_doc_lists(
        cached_payload,
        {13: doc_list},
    )
    assert not references_router._cached_payload_matches_authoritative_doc_lists(
        cached_payload,
        {13: list(reversed(doc_list))},
    )


def test_fast_refs_snapshot_does_not_rebuild_message_render_packets():
    class _Store:
        def get_messages(self, conv_id: str):
            raise AssertionError(f"fast snapshot must not load messages for {conv_id}")

    references_router._sync_message_render_packets_with_refs_payload(
        store=_Store(),
        conv_id="conv-fast-snapshot",
        payload={
            13: {
                "hits": [
                    {"ui_meta": {"primary_evidence": {"snippet": "grounded evidence"}}},
                ]
            }
        },
        mode="fast",
    )


def test_get_conversation_refs_returns_fast_exact_card_then_kicks_llm_warm(monkeypatch):
    references_router._REFS_CONVERSATION_CACHE.clear()
    references_router._REFS_CONVERSATION_WARMING.clear()
    refs = {
        13: {
            "prompt": "ADMM 是作者自己发明的吗？",
            "hits": [
                {
                    "text": "Most existing methods employ ADMM [4].",
                    "meta": {
                        "source_path": r"db\SCINeRF\SCINeRF.en.md",
                        "ref_pack_state": "ready",
                        "paper_guide_fast_exact": True,
                    },
                }
            ],
        }
    }
    store = _FakeStore(
        {
            "mode": "paper_guide",
            "bound_source_path": r"db\SCINeRF\SCINeRF.en.md",
            "bound_source_name": "SCINeRF",
        },
        refs,
    )
    warm_calls: list[dict] = []
    initial_kwargs: dict = {}

    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(references_router, "_pdf_dir", lambda: None)
    monkeypatch.setattr(references_router, "_md_dir", lambda: None)
    monkeypatch.setattr(references_router, "_lib_store", lambda: None)
    monkeypatch.setattr(
        references_router,
        "_warm_conversation_refs_payload_async",
        lambda **kwargs: warm_calls.append(dict(kwargs)),
    )
    monkeypatch.setattr(
        references_router,
        "_persist_rendered_refs_payloads",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("heuristic exact card must not be persisted as full")),
    )

    def fake_enrich_refs_payload(*args, **kwargs):
        del args
        initial_kwargs.update(dict(kwargs))
        return {
            13: {
                "hits": [
                    {
                        "meta": dict(refs[13]["hits"][0]["meta"]),
                        "ui_meta": {
                            "summary_line": "ADMM 是本文采用的已有优化方法。",
                            "summary_generation": "section_grounded",
                            "why_line": "该段明确把 ADMM 列为 existing methods。",
                            "why_generation": "section_grounded",
                        },
                    }
                ]
            }
        }

    monkeypatch.setattr(references_router, "enrich_refs_payload", fake_enrich_refs_payload)

    out = references_router.get_conversation_refs("conv-fast-exact-two-stage")

    assert out[13]["payload_mode"] == "fast"
    assert out[13]["enrichment_pending"] is True
    assert out[13]["hits"][0]["ui_meta"]["summary_line"] == "ADMM 是本文采用的已有优化方法。"
    assert out[13]["hits"][0]["ui_meta"]["polish_status"] == "heuristic"
    assert initial_kwargs.get("render_variant") == "fast"
    assert initial_kwargs.get("allow_expensive_llm_for_ready") is False
    assert initial_kwargs.get("allow_exact_locate") is False
    assert len(warm_calls) == 1
    assert warm_calls[0]["refs"] == refs


def test_warm_conversation_refs_payload_async_allows_llm_for_fast_exact_hits(monkeypatch):
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
        del args
        calls["kwargs"] = dict(kwargs)
        return {13: {"hits": [{"ui_meta": {"summary_generation": "llm_grounded", "why_generation": "llm_grounded"}}]}}

    monkeypatch.setattr(references_router, "enrich_refs_payload", fake_enrich_refs_payload)
    monkeypatch.setattr(references_router, "_persist_rendered_refs_payloads", lambda **_kwargs: None)
    monkeypatch.setattr(references_router, "_store_cached_conversation_refs_payload", lambda **_kwargs: None)

    references_router._warm_conversation_refs_payload_async(
        conv_id="conv-warm-fast-exact",
        signature="sig-warm-fast-exact",
        refs={
            13: {
                "prompt": "ADMM 是作者自己发明的吗？",
                "hits": [
                    {
                        "text": "Most existing methods employ ADMM [4].",
                        "meta": {"paper_guide_fast_exact": True},
                    }
                ],
            }
        },
        guide_mode=True,
        guide_source_path="SCINeRF.en.md",
        guide_source_name="SCINeRF",
    )

    kwargs = dict(calls.get("kwargs") or {})
    assert kwargs.get("render_variant") == "bounded_full"
    assert kwargs.get("allow_expensive_llm_for_ready") is True
    assert kwargs.get("allow_exact_locate") is False


def test_background_llm_polish_env_override_can_disable_card_polish(monkeypatch):
    monkeypatch.setenv("KB_REFS_BACKGROUND_LLM_POLISH", "0")
    monkeypatch.setattr(references_router, "_refs_card_polish_llm_enabled", lambda: True)

    assert references_router._refs_background_llm_polish_enabled() is False


def test_stored_full_payload_without_llm_copy_is_reused_when_polish_is_enabled(monkeypatch):
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

    assert out == pack["rendered_payload"]


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
