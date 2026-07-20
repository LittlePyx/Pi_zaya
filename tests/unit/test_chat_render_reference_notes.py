import json
from pathlib import Path

import pytest

from kb.chat_store import ChatStore
from api.chat_render import (
    _enrich_provenance_segments_for_display,
    _normalize_chat_markdown_for_display,
    _normalize_equation_source_notes,
    enrich_messages_with_reference_render,
)
from tests._paper_guide_fixtures import build_scinerf_like_fixture


MOJIBAKE_REFERENCE_LOCATOR = "\u9359\u509d\u20ac\u51a8\u757e\u6d63"
MOJIBAKE_REFERENCE_SOURCE_PREFIX = "\u93c9\u30e8\u569c" + MOJIBAKE_REFERENCE_LOCATOR + "?#1\u951b\u6b5a"


def test_equation_source_note_does_not_reference_removed_refs_ui():
    messages = [
        {"id": 1, "role": "user", "content": "NatPhoton 公式 8 是什么？"},
        {
            "id": 2,
            "role": "assistant",
            "content": "$$\nI_{TC}=x \\tag{8}\n$$",
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "Equation (8) defines the total-curvature objective.",
                    "meta": {
                        "source_path": r"db\NatPhoton-2019-Principles and prospects for single-pixel imaging\NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md",
                    },
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-test")
    body = str(rendered[-1].get("rendered_body") or "")

    assert "Open/Page" not in body
    assert MOJIBAKE_REFERENCE_LOCATOR not in body
    assert "库内文献" in body
    assert "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf" in body


def test_research_basket_synthetic_citation_uses_friendly_non_openable_detail():
    synthetic_path = "__research_basket__/item_1_deadbeef"
    messages = [
        {"id": 1, "role": "user", "content": "Use selected item"},
        {
            "id": 2,
            "role": "assistant",
            "content": "This is supported by the selected item [1].",
            "meta": {"canonical_hit_paths": [synthetic_path]},
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "Title: A hard to find preprint\nDOI: 10.1234/example.1\nSummary: selected metadata",
                    "score": 999.0,
                    "meta": {
                        "source_path": synthetic_path,
                        "source_name": "Research basket: A hard to find preprint",
                        "title": "A hard to find preprint",
                        "doi": "10.1234/example.1",
                        "ref_pack_state": "ready",
                        "research_basket_evidence": True,
                        "basket_source_role": "synthetic_basket_item",
                    },
                    "ui_meta": {
                        "display_name": "Research basket: A hard to find preprint",
                        "can_open": False,
                    },
                }
            ],
            "display_state": "ready",
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-basket")
    detail = rendered[-1]["cite_details"][0]

    assert detail["source_name"] == "Research basket: A hard to find preprint"
    assert detail["source_path"] == ""
    assert detail["citation_route"] == "research_basket"
    assert detail["routing_reason"] == "research_basket_evidence"
    assert detail["location_label"] == "Research basket"
    assert "item_1_deadbeef" not in json.dumps(detail, ensure_ascii=False)


def test_normalize_chat_markdown_cleans_empty_example_connectors_and_duplicate_terms():
    raw = (
        "This review (for example or this survey [2]) is a good entry point.\n\n"
        "The topic includes single-pixel imaging, single-pixel imaging."
    )

    rendered = _normalize_chat_markdown_for_display(raw)

    assert "for example or" not in rendered
    assert "single-pixel imaging, single-pixel imaging" not in rendered
    assert "This review (this survey [2]) is a good entry point." in rendered
    assert "The topic includes single-pixel imaging." in rendered


def test_equation_source_note_is_not_added_without_hits():
    messages = [
        {"id": 1, "role": "user", "content": "NatPhoton 公式 8 是什么？"},
        {
            "id": 2,
            "role": "assistant",
            "content": "$$\nI_{TC}=x \\tag{8}\n$$",
        },
    ]

    rendered = enrich_messages_with_reference_render(messages, refs_by_user={}, conv_id="conv-test")
    body = str(rendered[-1].get("rendered_body") or "")

    assert "库内文献" not in body


def test_normalize_equation_source_notes_strips_mojibake_prefix_from_pdf_label():
    raw = (
        "*（式(1) 对应命中的库内文献："
        f"`1) {MOJIBAKE_REFERENCE_SOURCE_PREFIX}CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf`）*"
    )

    out = _normalize_equation_source_notes(raw)

    assert MOJIBAKE_REFERENCE_LOCATOR not in out
    assert "CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf" in out
    assert "`1) " not in out


def test_copy_outputs_and_rendered_content_are_consistent():
    messages = [
        {"id": 1, "role": "user", "content": "请解释这个结论？"},
        {
            "id": 2,
            "role": "assistant",
            "content": "结论见 [[CITE:s1a2b3c4:12]]，并可对比 [CITE:s1a2b3c4:13]。",
        },
    ]

    rendered = enrich_messages_with_reference_render(messages, refs_by_user={}, conv_id="conv-test")
    msg = rendered[-1]
    rendered_content = str(msg.get("rendered_content") or "")
    copy_markdown = str(msg.get("copy_markdown") or "")
    copy_text = str(msg.get("copy_text") or "")

    assert "[[CITE:" not in rendered_content
    assert "[CITE:" not in rendered_content
    assert "[[CITE:" not in copy_markdown
    assert "[CITE:" not in copy_markdown
    assert "结论见" in copy_text


def test_rendered_body_falls_back_to_content_when_no_notice():
    messages = [
        {"id": 1, "role": "user", "content": "hello"},
        {"id": 2, "role": "assistant", "content": ""},
    ]

    rendered = enrich_messages_with_reference_render(messages, refs_by_user={}, conv_id="conv-test")
    msg = rendered[-1]
    assert str(msg.get("notice") or "") == ""
    assert str(msg.get("rendered_body") or "") == str(msg.get("rendered_content") or "")


def test_render_packet_contract_is_backfilled_from_rendered_message():
    messages = [
        {"id": 1, "role": "user", "content": "explain this"},
        {
            "id": 2,
            "role": "assistant",
            "content": "APR uses phase correlation [[CITE:s1234abcd:3]].",
            "provenance": {
                "source_path": "demo.md",
                "source_name": "demo.pdf",
                "segments": [
                    {
                        "segment_id": "seg-1",
                        "text": "APR uses phase correlation for registration.",
                        "locate_policy": "required",
                        "primary_heading_path": "Methods / APR",
                        "primary_block_id": "b-7",
                        "primary_anchor_id": "a-7",
                        "anchor_kind": "paragraph",
                        "claim_type": "method_claim",
                    }
                ],
            },
            "meta": {
                "paper_guide_contracts": {
                    "version": 1,
                    "intent": {"family": "method"},
                    "render_packet": {"citation_validation": {"kept": 1}},
                }
            },
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "dummy",
                    "meta": {"source_path": r"db\doc\doc.en.md"},
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-test")
    msg = rendered[-1]
    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})

    assert packet["answer_markdown"] == "APR uses phase correlation [[CITE:s1234abcd:3]]."
    assert packet["rendered_body"]
    assert packet["copy_text"]
    assert packet["citation_validation"] == {"kept": 1}
    assert packet["locate_target"]["segmentId"] == "seg-1"
    assert packet["reader_open"]["blockId"] == "b-7"
    assert packet["segment_ids"] == ["seg-1"]
    assert packet["visible_segment_ids"] == ["seg-1"]


def test_existing_render_packet_citation_cards_are_refreshed() -> None:
    messages = [
        {"id": 1, "role": "user", "content": "How should I read these papers?"},
        {
            "id": 2,
            "role": "assistant",
            "content": "A reading route is available.",
            "meta": {
                "answer_quality": {"output_mode": "citation"},
                "paper_guide_contracts": {
                    "version": 1,
                    "render_packet": {
                        "answer_markdown": "A reading route is available.",
                        "rendered_body": "A reading route is available.",
                        "copy_markdown": "A reading route is available.",
                        "copy_text": "A reading route is available.",
                        "cite_details": [
                            {
                                "num": 1,
                                "anchor": "roadmap-a1",
                                "source_name": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
                                "source_path": "hsi-fsi.en.md",
                                "heading_path": "Experiment design / Coding choice",
                                "answer_claim": (
                                    "\u518d\u8bfb\u65b9\u6cd5\u5bf9\u6bd4\uff1a"
                                    "\u300aHadamard single-pixel imaging versus Fourier single-pixel imaging\u300b "
                                    "(Optics Express, 2017)"
                                ),
                                "evidence_quote": (
                                    "Hadamard basis patterns are binary, which makes HSI naturally suitable "
                                    "for single-pixel imaging systems based on digital micromirror devices."
                                ),
                                "location_label": "Experiment design / Coding choice",
                            }
                        ],
                    },
                },
            },
        },
    ]

    rendered = enrich_messages_with_reference_render(messages, refs_by_user={}, conv_id="conv-refresh-card")
    packet = (((rendered[-1].get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    detail = packet["cite_details"][0]

    assert detail["answer_claim"] == ""
    assert detail["card_claim"] == ""
    assert detail["evidence_quote"] == detail["card_evidence"]
    assert "low_value_answer_claim" in detail["card_quality_flags"]


def test_non_paper_guide_message_preserves_minimal_primary_evidence_contract():
    messages = [
        {"id": 1, "role": "user", "content": "Which paper compares Hadamard and Fourier single-pixel imaging?"},
        {
            "id": 2,
            "role": "assistant",
            "content": "OE-2017 directly compares Hadamard and Fourier single-pixel imaging.",
            "meta": {
                "paper_guide_contracts": {
                    "version": 1,
                    "primary_evidence": {
                        "source_name": "OE-2017.pdf",
                        "block_id": "blk_22",
                        "anchor_id": "a_22",
                        "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                        "snippet": "Section 2.2 explicitly compares the two methods in terms of basis pattern properties.",
                    },
                    "render_packet": {
                        "answer_markdown": "OE-2017 directly compares Hadamard and Fourier single-pixel imaging.",
                        "primary_evidence": {
                            "source_name": "OE-2017.pdf",
                            "block_id": "blk_22",
                            "anchor_id": "a_22",
                            "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                            "snippet": "Section 2.2 explicitly compares the two methods in terms of basis pattern properties.",
                        },
                    },
                }
            },
        },
    ]

    rendered = enrich_messages_with_reference_render(messages, refs_by_user={}, conv_id="conv-normal")
    msg = rendered[-1]
    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})

    assert packet["answer_markdown"] == "OE-2017 directly compares Hadamard and Fourier single-pixel imaging."
    assert packet["primary_evidence"]["block_id"] == "blk_22"
    assert packet["primary_evidence"]["heading_path"] == "2. Comparison of theory / 2.2 Basis patterns generation"


def test_enrich_messages_uses_rendered_payload_primary_evidence_from_stored_refs_row():
    messages = [
        {"id": 1, "role": "user", "content": "Besides this paper, what other papers discuss Fourier single-pixel imaging?"},
        {
            "id": 2,
            "role": "assistant",
            "content": "A coarse answer seeded from the bound paper.",
            "meta": {
                "paper_guide_contracts": {
                    "version": 1,
                    "primary_evidence": {
                        "source_name": "NatPhoton-2019.pdf",
                        "heading_path": "Abstract / Camera architecture",
                        "selection_reason": "answer_hit_top",
                    },
                    "render_packet": {},
                }
            },
        },
    ]
    refs_by_user = {
        1: {
            "hits": [],
            "rendered_payload": {
                "hits": [
                    {
                        "ui_meta": {
                            "reader_open": {
                                "sourcePath": "oe2017.md",
                                "headingPath": "2. Comparison of theory / 2.2 Basis patterns generation",
                                "blockId": "blk_22",
                            }
                        }
                    }
                ],
                "primary_evidence": {
                    "source_path": "oe2017.md",
                    "source_name": "OE-2017.pdf",
                    "block_id": "blk_22",
                    "anchor_id": "a_22",
                    "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                    "selection_reason": "prompt_aligned",
                },
                "render_status": "full",
            },
        }
    }

    rendered = enrich_messages_with_reference_render(
        messages,
        refs_by_user=refs_by_user,
        conv_id="conv-cross-paper",
        render_packet_only=True,
    )
    msg = rendered[-1]
    contracts = (((msg.get("meta") or {}).get("paper_guide_contracts")) or {})
    packet = contracts.get("render_packet") or {}

    assert (contracts.get("primary_evidence") or {}).get("source_name") == "OE-2017.pdf"
    assert (contracts.get("primary_evidence") or {}).get("block_id") == "blk_22"
    assert (packet.get("primary_evidence") or {}).get("source_name") == "OE-2017.pdf"
    assert (packet.get("primary_evidence") or {}).get("heading_path") == "2. Comparison of theory / 2.2 Basis patterns generation"


def test_existing_render_packet_preserves_compat_render_fields_when_current_render_degrades():
    messages = [
        {"id": 1, "role": "user", "content": "explain this"},
        {
            "id": 2,
            "role": "assistant",
            "content": "SPI relies on compressive sensing [[CITE:s1234abcd:1]].",
            "created_at": 1,
            "meta": {
                "paper_guide_contracts": {
                    "version": 1,
                    "intent": {"family": "citation_lookup"},
                    "render_packet": {
                        "answer_markdown": "SPI relies on compressive sensing [[CITE:s1234abcd:1]].",
                        "rendered_body": "SPI relies on compressive sensing [1](#kb-cite-demo-1).",
                        "rendered_content": "SPI relies on compressive sensing [1](#kb-cite-demo-1).",
                        "copy_markdown": "SPI relies on compressive sensing [1](#kb-cite-demo-1).",
                        "copy_text": "SPI relies on compressive sensing [1].",
                        "cite_details": [
                            {
                                "num": 1,
                                "anchor": "kb-cite-demo-1",
                                "source_name": "demo.pdf",
                                "source_path": "demo.md",
                                "raw": "Demo reference [1]",
                            }
                        ],
                    },
                }
            },
        },
    ]

    rendered = enrich_messages_with_reference_render(messages, refs_by_user={}, conv_id="conv-test")
    msg = rendered[-1]
    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})

    assert str(msg.get("rendered_body") or "") == "SPI relies on compressive sensing [1](#kb-cite-demo-1)."
    assert str(msg.get("rendered_content") or "") == "SPI relies on compressive sensing [1](#kb-cite-demo-1)."
    assert str(msg.get("copy_markdown") or "") == "SPI relies on compressive sensing [1](#kb-cite-demo-1)."
    assert str(msg.get("copy_text") or "") == "SPI relies on compressive sensing [1]."
    assert len(msg.get("cite_details") or []) == 1
    assert packet["rendered_body"] == "SPI relies on compressive sensing [1](#kb-cite-demo-1)."
    assert len(packet["cite_details"]) == 1


def test_render_packet_replaces_stale_primary_jump_target_when_current_provenance_is_better():
    messages = [
        {"id": 1, "role": "user", "content": "Which ADMM citation is this?"},
        {
            "id": 2,
            "role": "assistant",
            "content": "The paper cites [4] for this point.\n> most of the existing methods employ alternating direction method of multipliers (ADMM) [4],",
            "provenance": {
                "source_path": "demo.md",
                "source_name": "demo.pdf",
                "segments": [
                    {
                        "segment_id": "seg-2",
                        "text": "most of the existing methods employ alternating direction method of multipliers (ADMM) [4],",
                        "locate_policy": "required",
                        "primary_heading_path": "Related Work / Snapshot Compressive Imaging",
                        "primary_block_id": "b-right",
                        "primary_anchor_id": "a-right",
                        "anchor_kind": "blockquote",
                        "claim_type": "prior_work",
                        "support_slot_claim_type": "prior_work",
                        "support_locate_anchor": "most of the existing methods employ alternating direction method of multipliers (ADMM) [4],",
                        "resolved_ref_num": 4,
                    }
                ],
            },
            "meta": {
                "paper_guide_contracts": {
                    "version": 1,
                    "intent": {"family": "citation_lookup"},
                    "render_packet": {
                        "answer_markdown": "The paper cites [4] for this point.",
                        "rendered_body": "The paper cites [4] for this point.",
                        "rendered_content": "The paper cites [4] for this point.",
                        "copy_markdown": "The paper cites [4] for this point.",
                        "copy_text": "The paper cites [4] for this point.",
                        "locate_target": {
                            "segmentId": "seg-wrong",
                            "headingPath": "Method / Wrong Section",
                            "snippet": "A generic method sentence unrelated to this citation.",
                            "anchorText": "A generic method sentence unrelated to this citation.",
                            "blockId": "b-wrong",
                            "anchorId": "a-wrong",
                        },
                        "reader_open": {
                            "sourcePath": "demo.md",
                            "headingPath": "Method / Wrong Section",
                            "snippet": "A generic method sentence unrelated to this citation.",
                            "blockId": "b-wrong",
                            "anchorId": "a-wrong",
                            "strictLocate": True,
                        },
                    },
                }
            },
        },
    ]

    rendered = enrich_messages_with_reference_render(messages, refs_by_user={}, conv_id="conv-test")
    msg = rendered[-1]
    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})

    assert packet["locate_target"]["segmentId"] == "seg-2"
    assert packet["locate_target"]["blockId"] == "b-right"
    assert "alternating direction method of multipliers" in str(packet["locate_target"]["snippet"]).lower()
    assert packet["reader_open"]["blockId"] == "b-right"


def test_sid_markers_are_removed_from_rendered_outputs():
    messages = [
        {"id": 1, "role": "user", "content": "解释单像素成像？"},
        {
            "id": 2,
            "role": "assistant",
            "content": "[SID:s50f9c165] 这是内部标记，不应该展示给用户。",
        },
    ]

    rendered = enrich_messages_with_reference_render(messages, refs_by_user={}, conv_id="conv-test")
    msg = rendered[-1]
    rendered_content = str(msg.get("rendered_content") or "")
    copy_markdown = str(msg.get("copy_markdown") or "")
    copy_text = str(msg.get("copy_text") or "")

    assert "[SID:" not in rendered_content
    assert "[SID:" not in copy_markdown
    assert "[SID:" not in copy_text


def test_structured_cite_fallback_does_not_relink_after_safe_downgrade(monkeypatch):
    from api import chat_render

    def fake_primary(_md, _hits, *, anchor_ns="", canonical_paths=None):
        del _md, _hits, anchor_ns, canonical_paths
        # Simulate safety downgrade result from primary annotator:
        # CITE token resolved to plain numeric marker and no details.
        return "Gehm et al. (2007) [24].", []

    def fake_fallback(*args, **kwargs):
        raise AssertionError("fallback should not run after safe downgrade")

    monkeypatch.setattr(chat_render, "_annotate_inpaper_citations_with_hover_meta", fake_primary)
    monkeypatch.setattr(chat_render, "_fallback_render_structured_citations", fake_fallback)

    messages = [
        {"id": 1, "role": "user", "content": "test"},
        {
            "id": 2,
            "role": "assistant",
            "content": "Gehm et al. (2007) [[CITE:s1234abcd:24]].",
            "meta": {"answer_quality": {"prompt_family": "citation_lookup", "output_mode": "citation_lookup"}},
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "dummy",
                    "meta": {
                        "source_path": r"db\doc\doc.en.md",
                    },
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-test")
    msg = rendered[-1]
    assert "[24]" in str(msg.get("rendered_body") or "")
    assert msg.get("cite_details") == []


def test_structured_cite_fallback_recovers_links_when_primary_strips_tokens(monkeypatch):
    from api import chat_render

    def fake_primary(_md, _hits, *, anchor_ns="", canonical_paths=None):
        del _md, _hits, anchor_ns, canonical_paths
        return "SPI relies on compressive sensing.", []

    def fake_fallback(_md, _hits, *, anchor_ns=""):
        del _md, _hits, anchor_ns
        return (
            "SPI relies on compressive sensing [1](#kb-cite-demo-1).",
            [{"num": 1, "anchor": "kb-cite-demo-1", "source_name": "demo.pdf"}],
        )

    monkeypatch.setattr(chat_render, "_annotate_inpaper_citations_with_hover_meta", fake_primary)
    monkeypatch.setattr(chat_render, "_fallback_render_structured_citations", fake_fallback)

    messages = [
        {"id": 1, "role": "user", "content": "test"},
        {
            "id": 2,
            "role": "assistant",
            "content": "SPI relies on compressive sensing [[CITE:s1234abcd:1]].",
            "meta": {"answer_quality": {"prompt_family": "citation_lookup", "output_mode": "citation_lookup"}},
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "dummy",
                    "meta": {
                        "source_path": r"db\doc\doc.en.md",
                    },
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-test")
    msg = rendered[-1]
    assert "[1](#kb-cite-demo-1)" in str(msg.get("rendered_body") or "")
    assert len(msg.get("cite_details") or []) == 1


def test_normal_answer_does_not_auto_link_freeform_numeric_markers_from_refs_hits():
    messages = [
        {"id": 1, "role": "user", "content": "test"},
        {
            "id": 2,
            "role": "assistant",
            "content": (
                "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf [2].\n"
                "Section 2.2 compares the two methods [2]."
            ),
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "dummy",
                    "meta": {
                        "source_path": r"db\doc\doc.en.md",
                    },
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-test")
    msg = rendered[-1]
    assert "[2]" not in str(msg.get("rendered_body") or "")
    assert "[2]" not in str(msg.get("rendered_content") or "")
    assert msg.get("cite_details") == []


def test_normal_answer_strips_structured_cite_markers_without_linking():
    messages = [
        {"id": 1, "role": "user", "content": "test"},
        {
            "id": 2,
            "role": "assistant",
            "content": (
                "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf [[CITE:demo:2]].\n"
                "Section 2.2 compares the two methods [[CITE:demo:2]]."
            ),
            "meta": {
                "answer_quality": {
                    "prompt_family": "overview",
                    "output_mode": "reading_guide",
                }
            },
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "dummy",
                    "meta": {
                        "source_path": r"db\doc\doc.en.md",
                    },
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-test")
    msg = rendered[-1]
    rendered_body = str(msg.get("rendered_body") or "")
    assert "[[CITE:" not in rendered_body
    assert "[2]" not in rendered_body
    assert msg.get("cite_details") == []


def test_normal_answer_preserves_validated_system_b_marker(monkeypatch):
    from api import chat_render

    def fake_primary(md, hits, *, anchor_ns="", canonical_paths=None):
        del hits, anchor_ns, canonical_paths
        assert "[[CITE:s1234abcd:4]]" in md
        return (
            "ADMM is prior optimization machinery [4](#kb-cite-demo-4).",
            [
                {
                    "num": 4,
                    "anchor": "kb-cite-demo-4",
                    "source_name": "SCINeRF.pdf",
                    "source_path": r"db\demo\scinerf.en.md",
                    "title": "Distributed optimization and statistical learning via ADMM",
                    "is_inpaper": True,
                }
            ],
        )

    monkeypatch.setattr(chat_render, "_annotate_inpaper_citations_with_hover_meta", fake_primary)
    messages = [
        {"id": 1, "role": "user", "content": "ADMM 是作者自己发明的吗？"},
        {
            "id": 2,
            "role": "assistant",
            "content": "ADMM is prior optimization machinery [[CITE:s1234abcd:4]].",
            "meta": {
                "answer_quality": {
                    "prompt_family": "overview",
                    "output_mode": "reading_guide",
                    "reference_opportunities": {"count": 1, "mode": "inline", "refs": [4]},
                    "citation_validation": {"raw_count": 1, "kept": 1, "rewritten": 0},
                }
            },
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "Most existing methods employ ADMM [4].",
                    "meta": {"source_path": r"db\demo\scinerf.en.md"},
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-normal-sysb")
    msg = rendered[-1]

    assert "[[CITE:" not in str(msg.get("rendered_body") or "")
    assert "#kb-cite-demo-4" in str(msg.get("rendered_body") or "")
    assert (msg.get("cite_details") or [])[0]["is_inpaper"] is True


def test_normal_upstream_question_can_route_structured_marker_to_system_b_without_validation(monkeypatch):
    from api import chat_render

    calls = []

    def fake_primary(md, hits, *, anchor_ns="", canonical_paths=None):
        del hits, anchor_ns, canonical_paths
        calls.append(md)
        return (
            "ADMM comes from prior optimization work [4](#kb-cite-demo-4).",
            [
                {
                    "num": 4,
                    "anchor": "kb-cite-demo-4",
                    "source_name": "SCINeRF.pdf",
                    "source_path": r"db\demo\scinerf.en.md",
                    "title": "Distributed optimization and statistical learning via ADMM",
                    "is_inpaper": True,
                    "citation_route": "system_b",
                }
            ],
        )

    monkeypatch.setattr(chat_render, "_annotate_inpaper_citations_with_hover_meta", fake_primary)
    messages = [
        {"id": 1, "role": "user", "content": "ADMM 是怎么来的？作者是不是借鉴了前人的方法？"},
        {
            "id": 2,
            "role": "assistant",
            "content": "ADMM comes from prior optimization work [[CITE:s1234abcd:4]].",
            "meta": {"answer_quality": {"prompt_family": "overview", "output_mode": "reading_guide"}},
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "Most existing methods employ ADMM [4].",
                    "meta": {"source_path": r"db\demo\scinerf.en.md"},
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-normal-sysb-unvalidated")
    msg = rendered[-1]

    assert calls
    assert "[[CITE:" not in str(msg.get("rendered_body") or "")
    assert "#kb-cite-demo-4" in str(msg.get("rendered_body") or "")
    assert (msg.get("cite_details") or [])[0]["is_inpaper"] is True


def test_citation_plan_allows_normal_question_to_render_system_b_without_validation(monkeypatch):
    from api import chat_render

    calls = []

    def fake_primary(md, hits, *, anchor_ns="", canonical_paths=None, citation_plan=None):
        del hits, anchor_ns, canonical_paths
        calls.append({"md": md, "citation_plan": citation_plan})
        return (
            "ADMM is the optimization background [4](#kb-cite-demo-4).",
            [
                {
                    "num": 4,
                    "anchor": "kb-cite-demo-4",
                    "source_name": "SCINeRF.pdf",
                    "source_path": r"db\demo\scinerf.en.md",
                    "title": "Distributed optimization and statistical learning via ADMM",
                    "is_inpaper": True,
                    "citation_route": "system_b",
                    "routing_reason": "citation_plan",
                }
            ],
        )

    monkeypatch.setattr(chat_render, "_annotate_inpaper_citations_with_hover_meta", fake_primary)
    plan = {
        "intent": "beginner_overview",
        "budget": {"system_a": 2, "system_b": 1},
        "system_b_enabled": True,
        "slots": [{"preferred_system": "system_b", "candidate_refs": [4]}],
    }
    messages = [
        {"id": 1, "role": "user", "content": "ADMM 这个东西我不太懂，简单说一下。"},
        {
            "id": 2,
            "role": "assistant",
            "content": "ADMM is the optimization background [[CITE:s1234abcd:4]].",
            "meta": {"answer_quality": {"prompt_family": "overview", "citation_plan": plan}},
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "Most existing methods employ ADMM [4].",
                    "meta": {"source_path": r"db\demo\scinerf.en.md"},
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-citation-plan-sysb")
    msg = rendered[-1]

    assert calls
    assert calls[0]["citation_plan"]["intent"] == "beginner_overview"
    assert "[[CITE:" not in str(msg.get("rendered_body") or "")
    assert "#kb-cite-demo-4" in str(msg.get("rendered_body") or "")
    assert (msg.get("cite_details") or [])[0]["routing_reason"] == "citation_plan"


def test_named_upstream_title_is_linked_from_current_reference_index(monkeypatch):
    from api import chat_render
    from ui import refs_renderer

    source_path = r"db\paper\paper.en.md"
    source_key = chat_render._render_norm_source_key(source_path)
    index_data = {
        "docs": {
            source_key: {
                "path": source_path,
                "name": "paper.pdf",
                "sha1": "abc",
                "refs": {
                    "24": {
                        "authors": "Jiang X, Li Z, Du G",
                        "venue": "Optics Express",
                        "year": "2022",
                        "doi": "10.1364/oe.458742",
                        "title": "Fast hyperspectral single-pixel imaging via frequency-division multiplexed illumination",
                        "raw": (
                            "[24] Jiang X, Li Z, Du G. Fast hyperspectral single-pixel imaging via "
                            "frequency-division multiplexed illumination. Optics Express, 2022. "
                            "doi:10.1364/oe.458742"
                        ),
                    }
                },
            }
        }
    }

    monkeypatch.setattr(chat_render, "_load_reference_index_cached", lambda: index_data)
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: index_data)

    messages = [
        {"id": 1, "role": "user", "content": "What else should I read?"},
        {
            "id": 2,
            "role": "assistant",
            "content": (
                "You can compare against Fast hyperspectral single-pixel imaging via "
                "frequency-division multiplexed illumination for real-time SPI context."
            ),
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "The current paper cites fast hyperspectral SPI reconstruction [24].",
                    "meta": {"source_path": source_path, "source_sha1": "abc"},
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-title-sysb")
    msg = rendered[-1]
    body = str(msg.get("rendered_body") or "")
    details = list(msg.get("cite_details") or [])

    assert "[[CITE:" not in body
    assert "[24](#kb-cite-" in body
    assert len(details) == 1
    assert details[0]["is_inpaper"] is True
    assert details[0]["doi"] == "10.1364/oe.458742"
    assert "frequency-division multiplexed illumination" in details[0]["title"]


def test_named_upstream_title_repair_does_not_link_short_venue_mentions(monkeypatch):
    from api import chat_render

    source_path = r"db\paper\paper.en.md"
    index_data = {
        "docs": {
            chat_render._render_norm_source_key(source_path): {
                "path": source_path,
                "name": "paper.pdf",
                "refs": {
                    "24": {
                        "title": "Fast hyperspectral single-pixel imaging via frequency-division multiplexed illumination",
                        "raw": "[24] Jiang X. Fast hyperspectral single-pixel imaging via frequency-division multiplexed illumination. Optics Express, 2022.",
                    }
                },
            }
        }
    }
    monkeypatch.setattr(chat_render, "_load_reference_index_cached", lambda: index_data)

    repaired, changed = chat_render._repair_named_system_b_citation_markers(
        "For comparison, the Optica 2024 work is also relevant.",
        [{"text": "hit", "meta": {"source_path": source_path}}],
        {"budget": {"system_b": 2}},
    )

    assert changed is False
    assert "[[CITE:" not in repaired


def test_named_upstream_title_repair_does_not_duplicate_a_current_library_source(monkeypatch):
    from api import chat_render

    citing_path = r"db\video\Journal of Optics-2016-3D single-pixel video.en.md"
    current_path = (
        r"db\review\NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md"
    )
    index_data = {
        "docs": {
            chat_render._render_norm_source_key(citing_path): {
                "path": citing_path,
                "name": "3D single-pixel video.pdf",
                "refs": {
                    "11": {
                        "title": "Principles and prospects for single-pixel imaging",
                        "raw": "[11] Principles and prospects for single-pixel imaging.",
                    }
                },
            }
        }
    }
    monkeypatch.setattr(chat_render, "_load_reference_index_cached", lambda: index_data)

    repaired, changed = chat_render._repair_named_system_b_citation_markers(
        "Read Principles and prospects for single-pixel imaging first.",
        [
            {"text": "citing hit", "meta": {"source_path": citing_path}},
            {"text": "review hit", "meta": {"source_path": current_path}},
        ],
        {"budget": {"system_b": 2}},
    )

    assert changed is False
    assert "[[CITE:" not in repaired


@pytest.mark.parametrize(
    "case",
    [
        {
            "name": "zh_overview_freeform_numeric",
            "user": "这篇论文的核心方法是什么？",
            "assistant": "该方法把快照压缩成像和 NeRF 训练结合起来 [1]，用于从单帧压缩观测恢复三维表示。",
            "meta": {"answer_quality": {"prompt_family": "overview", "output_mode": "reading_guide"}},
        },
        {
            "name": "en_comparison_numeric_range",
            "user": "Which paper compares Hadamard and Fourier single-pixel imaging?",
            "assistant": "OE-2017 compares Hadamard and Fourier single-pixel imaging [2, 3], but this is a normal library answer.",
            "meta": {"answer_quality": {"prompt_family": "compare", "output_mode": "reading_guide"}},
        },
        {
            "name": "zh_method_structured_marker",
            "user": "它是怎么训练 NeRF 的？",
            "assistant": "论文把物理成像过程写进训练目标 [[CITE:s1234abcd:4]]，但普通方法问答不应保留文内参考链接。",
            "meta": {"answer_quality": {"prompt_family": "method", "output_mode": "reading_guide"}},
        },
        {
            "name": "no_meta_source_like_numeric",
            "user": "给我正常概括一下这篇文献。",
            "assistant": "The answer mentions a source-like marker [5] but has no citation intent metadata.",
            "meta": {},
        },
    ],
    ids=lambda case: str(case.get("name") or "case"),
)
def test_normal_question_variants_do_not_trigger_inpaper_reference_links(case):
    messages = [
        {"id": 1, "role": "user", "content": case["user"]},
        {
            "id": 2,
            "role": "assistant",
            "content": case["assistant"],
            "meta": case["meta"],
        },
    ]
    name = str(case.get("name") or "")
    hit_text = "retrieved evidence"
    hit_meta = {"source_path": r"db\doc\doc.en.md"}
    if name == "zh_overview_freeform_numeric":
        hit_text = (
            "Snapshot Compressive Imaging (SCI) is combined with NeRF training "
            "to recover a 3D scene representation from a compressed observation."
        )
        hit_meta = {
            "source_path": r"db\doc\scinerf.en.md",
            "heading_path": "Abstract",
            "evidence_quote": hit_text,
        }
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": hit_text,
                    "meta": hit_meta,
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id=f"conv-{name}")
    msg = rendered[-1]
    rendered_body = str(msg.get("rendered_body") or "")
    rendered_content = str(msg.get("rendered_content") or "")
    copy_markdown = str(msg.get("copy_markdown") or "")
    cite_details = list(msg.get("cite_details") or [])

    # Structured [[CITE:...]] markers are always stripped in non-paper-guide mode.
    assert "[[CITE:" not in rendered_body
    assert "[CITE:" not in rendered_body
    # Cases with resolvable [n] markers (n <= hit count) get linked.
    # Cases with unresolvable [2,3] or [5] (only 1 hit) or structured markers get stripped.
    if name == "zh_overview_freeform_numeric":
        assert "#kb-cite-" in rendered_body
        assert len(cite_details) > 0
    else:
        assert "#kb-cite-" not in rendered_body
        assert cite_details == []


@pytest.mark.parametrize(
    "meta",
    [
        {"paper_guide_contracts": {"version": 1, "intent": {"family": "citation_lookup"}}},
        {"answer_quality": {"prompt_family": "citation_lookup", "output_mode": "reading_guide"}},
        {"answer_quality": {"prompt_family": "overview", "output_mode": "citation_lookup"}},
    ],
    ids=["contract_intent", "answer_prompt_family", "answer_output_mode"],
)
def test_citation_lookup_variants_trigger_and_preserve_inpaper_reference_links(monkeypatch, meta):
    from api import chat_render

    calls = []

    def fake_primary(md, hits, *, anchor_ns="", canonical_paths=None):
        calls.append({"md": md, "hits": hits, "anchor_ns": anchor_ns, "canonical_paths": canonical_paths})
        return (
            "SCI relies on compressive sensing [1](#kb-cite-demo-1).",
            [
                {
                    "num": 1,
                    "anchor": "kb-cite-demo-1",
                    "source_name": "demo.pdf",
                    "source_path": "demo.md",
                    "raw": "Demo reference [1]",
                }
            ],
        )

    monkeypatch.setattr(chat_render, "_annotate_inpaper_citations_with_hover_meta", fake_primary)

    messages = [
        {"id": 1, "role": "user", "content": "Which in-paper reference supports SCI?"},
        {
            "id": 2,
            "role": "assistant",
            "content": "SCI relies on compressive sensing [[CITE:s1234abcd:1]].",
            "meta": meta,
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "dummy",
                    "meta": {"source_path": r"db\doc\doc.en.md"},
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-citation-lookup")
    msg = rendered[-1]
    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})

    assert calls
    assert "[1](#kb-cite-demo-1)" in str(msg.get("rendered_body") or "")
    assert "[[CITE:" not in str(msg.get("rendered_body") or "")
    assert len(msg.get("cite_details") or []) == 1
    if (meta.get("paper_guide_contracts") or {}).get("intent"):
        assert packet.get("rendered_body") == msg.get("rendered_body")
        assert len(packet.get("cite_details") or []) == 1
    else:
        assert packet == {}


def test_citation_lookup_rendered_link_points_to_validated_target_reference(monkeypatch):
    from ui import refs_renderer

    source_path = r"db\doc\paper.en.md"
    sid = refs_renderer._source_cite_id(source_path)

    refs = {
        1: {
            "authors": "Wrong A",
            "year": "2020",
            "doi": "10.1000/wrong",
            "title": "Wrong Reference",
            "raw": "[1] Wrong A. Wrong Reference. 2020. doi:10.1000/wrong",
        },
        24: {
            "authors": "Gehm M, Brady D",
            "year": "2007",
            "doi": "10.1364/OE.15.014013",
            "title": "Single-shot compressive spectral imaging with a dual-disperser architecture",
            "raw": (
                "[24] Gehm M, Brady D. Single-shot compressive spectral imaging with "
                "a dual-disperser architecture. Optics Express, 2007. doi:10.1364/OE.15.014013"
            ),
        },
    }

    def fake_resolve(_index_data, src, ref_num, *, source_sha1=""):
        del _index_data, source_sha1
        if str(src) != source_path:
            return None
        ref = refs.get(int(ref_num))
        return {
            "source_path": source_path,
            "source_name": "paper.pdf",
            "ref_num": int(ref_num),
            "ref": dict(ref),
        } if isinstance(ref, dict) else None

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "paper.pdf")

    messages = [
        {"id": 1, "role": "user", "content": "Which in-paper reference supports this DOI?"},
        {
            "id": 2,
            "role": "assistant",
            "content": f"This follows DOI 10.1364/OE.15.014013 [[CITE:{sid}:24]].",
            "meta": {
                "paper_guide_contracts": {
                    "version": 1,
                    "intent": {"family": "citation_lookup"},
                }
            },
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "Evidence mentions prior work [24].",
                    "meta": {
                        "source_path": source_path,
                        "source_sha1": "abc",
                    },
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-target-ref")
    msg = rendered[-1]
    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    details = list(msg.get("cite_details") or [])

    assert len(details) == 1
    detail = details[0]
    assert detail["num"] == 24
    assert detail["doi"] == "10.1364/OE.15.014013"
    assert "dual-disperser architecture" in detail["title"]
    assert "Wrong Reference" not in str(detail)
    assert f"[24](#{detail['anchor']}" in str(msg.get("rendered_body") or "")
    assert packet["rendered_body"] == msg.get("rendered_body")
    assert packet["cite_details"][0]["doi"] == "10.1364/OE.15.014013"


def test_structured_cite_fallback_uses_local_answer_line_for_system_b_context(monkeypatch, tmp_path: Path):
    from api import chat_render

    source_file = tmp_path / "paper.en.md"
    source_file.write_text(
        "\n".join(
            [
                "# Paper",
                "This body intentionally leaves mention choice to the structured asset.",
                "",
                "## References",
                "[3] Example A. Detector-array reconstruction benchmark. Journal, 2024.",
            ]
        ),
        encoding="utf-8",
    )
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "reference_index.json").write_text(
        json.dumps(
            {
                "references": [
                    {
                        "ref_num": 3,
                        "citation_mentions": [
                            {
                                "citation_context": "The introduction briefly names earlier optical sectioning work [3].",
                                "heading_path": "Paper / Introduction",
                                "location_label": "Paper / Introduction / p. 1",
                                "page_start": 1,
                                "page_end": 1,
                                "line_start": 8,
                                "line_end": 8,
                            },
                            {
                                "citation_context": "The benchmark compares detector-array reconstruction accuracy against prior work [3].",
                                "heading_path": "Paper / Benchmark",
                                "location_label": "Paper / Benchmark / p. 5",
                                "page_start": 5,
                                "page_end": 5,
                                "line_start": 88,
                                "line_end": 88,
                            },
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    source_path = str(source_file)
    sid = chat_render._source_cite_id(source_path)
    monkeypatch.setattr(chat_render, "_load_reference_index_cached", lambda: {})

    md = f"Intro sentence without a citation.\nFor the benchmark, open prior work [[CITE:{sid}:3]]."
    rendered, details = chat_render._fallback_render_structured_citations(
        md,
        [{"text": "hit", "meta": {"source_path": source_path, "source_sha1": "abc"}}],
        anchor_ns="local-line-test",
    )

    assert "[3](#kb-cite-" in rendered
    assert len(details) == 1
    detail = details[0]
    assert detail["is_inpaper"] is True
    assert detail["answer_claim"] == "For the benchmark, open prior work."
    assert detail["citation_context_source"] == "structured_reference_index"
    assert "detector-array reconstruction accuracy" in detail["citation_context"]
    assert "briefly names earlier" not in detail["citation_context"]
    assert detail["heading_path"].endswith("Benchmark")
    assert detail["page_start"] == 5


def test_structured_cite_fallback_marks_answer_context_only_when_source_context_missing(monkeypatch, tmp_path: Path):
    from api import chat_render

    source_file = tmp_path / "paper.en.md"
    source_file.write_text(
        "\n".join(
            [
                "# Paper",
                "No body mention is available for this reference.",
                "",
                "## References",
                "[6] Example B. Unlocated upstream method. 2023.",
            ]
        ),
        encoding="utf-8",
    )
    source_path = str(source_file)
    sid = chat_render._source_cite_id(source_path)
    monkeypatch.setattr(chat_render, "_load_reference_index_cached", lambda: {})

    md = f"This answer mentions an upstream method [[CITE:{sid}:6]]."
    rendered, details = chat_render._fallback_render_structured_citations(
        md,
        [{"text": "hit", "meta": {"source_path": source_path, "source_sha1": "abc"}}],
        anchor_ns="answer-only-test",
    )

    assert "[6](#kb-cite-" in rendered
    assert len(details) == 1
    detail = details[0]
    assert detail["citation_context_source"] == "answer_context"
    assert detail["card_evidence_label"] == "回答里的线索"
    assert "answer_context_only" in detail["card_quality_flags"]


def test_non_citation_message_does_not_preserve_stale_existing_render_packet_links():
    messages = [
        {"id": 1, "role": "user", "content": "test"},
        {
            "id": 2,
            "role": "assistant",
            "content": "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf [2].",
            "meta": {
                "paper_guide_contracts": {
                    "version": 1,
                    "intent": {"family": "overview"},
                    "render_packet": {
                        "rendered_body": "Existing rendered body [2](#kb-cite-demo-2).",
                        "rendered_content": "Existing rendered body [2](#kb-cite-demo-2).",
                        "copy_markdown": "Existing rendered body [2](#kb-cite-demo-2).",
                        "copy_text": "Existing rendered body [2].",
                        "cite_details": [{"num": 2, "anchor": "kb-cite-demo-2", "source_name": "demo.pdf"}],
                    },
                }
            },
        },
    ]

    rendered = enrich_messages_with_reference_render(
        messages,
        refs_by_user={},
        conv_id="conv-test",
        render_packet_only=True,
    )
    msg = rendered[-1]
    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})

    assert "[2](" not in str(packet.get("rendered_body") or "")
    assert "[2]" not in str(packet.get("rendered_body") or "")
    assert packet.get("cite_details") == []


def test_enrich_provenance_segments_for_display_loads_md_blocks_for_quote_rebind(tmp_path: Path):
    fixture = build_scinerf_like_fixture(tmp_path)
    md_main = fixture["md_main"]
    wrong_method_block = fixture["wrong_method_block"]
    conclusion_block = fixture["conclusion_block"]

    provenance = {
        "md_path": str(md_main),
        "source_path": str(tmp_path / "dummy.pdf"),
        "source_name": "SCINeRF.pdf",
        "block_map": {
            str(wrong_method_block.get("block_id") or ""): dict(wrong_method_block),
        },
        "segments": [
            {
                "segment_id": "seg_004",
                "segment_index": 4,
                "kind": "blockquote",
                "segment_type": "evidence",
                "text": (
                    "SCINeRF exploits neural radiance fields as its underlying scene representation [...] "
                    "Physical image formation process of an SCI image is exploited to formulate the training objective "
                    "for jointly NeRF training and camera poses optimization."
                ),
                "raw_markdown": (
                    '*"SCINeRF exploits neural radiance fields as its underlying scene representation [...] '
                    "Physical image formation process of an SCI image is exploited to formulate the training objective "
                    'for jointly NeRF training and camera poses optimization."*'
                ),
                "evidence_mode": "direct",
                "claim_type": "blockquote_claim",
                "must_locate": True,
                "anchor_kind": "blockquote",
                "primary_block_id": str(wrong_method_block.get("block_id") or ""),
                "primary_anchor_id": str(wrong_method_block.get("anchor_id") or ""),
                "primary_heading_path": str(wrong_method_block.get("heading_path") or ""),
                "evidence_block_ids": [str(wrong_method_block.get("block_id") or "")],
                "support_block_ids": [],
                "anchor_text": str(wrong_method_block.get("text") or ""),
                "evidence_quote": str(wrong_method_block.get("text") or ""),
            }
        ],
    }

    enriched = _enrich_provenance_segments_for_display(provenance, hits=[], anchor_ns="conv:test")

    assert isinstance(enriched, dict)
    segments = enriched.get("segments") or []
    assert len(segments) == 1
    seg = segments[0]
    assert str(seg.get("primary_block_id") or "") == str(conclusion_block.get("block_id") or "")
    block_map = enriched.get("block_map") or {}
    assert str(conclusion_block.get("block_id") or "") in block_map


def test_enrich_provenance_segments_for_display_preserves_figure_scope_heading():
    provenance = {
        "block_map": {},
        "segments": [
            {
                "segment_id": "seg_001",
                "segment_index": 0,
                "text": "Panel (f) corresponds to methane imaging using SPC.",
                "raw_markdown": "Panel (f) corresponds to methane imaging using SPC.",
                "evidence_mode": "direct",
                "claim_type": "figure_claim",
                "must_locate": True,
                "anchor_kind": "figure",
                "anchor_text": "(f) methane imaging using SPC$^{15}$",
                "primary_heading_path": "Applications and future potential for single-pixel imaging",
                "support_slot_claim_type": "figure_panel",
                "support_slot_figure_number": 3,
                "support_slot_panel_letters": ["f"],
                "support_locate_anchor": "(f) methane imaging using SPC$^{15}$",
                "locate_policy": "required",
            }
        ],
    }

    enriched = _enrich_provenance_segments_for_display(provenance, hits=[], anchor_ns="conv:test")

    assert isinstance(enriched, dict)
    segments = enriched.get("segments") or []
    assert len(segments) == 1
    assert str(segments[0].get("primary_heading_path") or "") == (
        "Applications and future potential for single-pixel imaging / Figure 3"
    )


def test_enrich_provenance_segments_for_display_preserves_box_only_heading():
    provenance = {
        "block_map": {},
        "segments": [
            {
                "segment_id": "seg_001",
                "segment_index": 0,
                "text": "It can be shown that when the number of sampling patterns used M >= O(K log(N/K))...",
                "raw_markdown": "It can be shown that when the number of sampling patterns used M >= O(K log(N/K))...",
                "evidence_mode": "direct",
                "claim_type": "own_result",
                "must_locate": False,
                "anchor_kind": "sentence",
                "anchor_text": "It can be shown that when the number of sampling patterns used M >= O(K log(N/K))...",
                "primary_heading_path": "Acquisition and image reconstruction strategies",
                "support_slot_claim_type": "own_result",
                "support_slot_box_number": 1,
                "support_slot_panel_letters": [],
                "support_locate_anchor": "It can be shown that when the number of sampling patterns used $M \\ge O(K \\log(N/K))$...",
                "locate_policy": "hidden",
            }
        ],
    }

    enriched = _enrich_provenance_segments_for_display(provenance, hits=[], anchor_ns="conv:test")

    assert isinstance(enriched, dict)
    segments = enriched.get("segments") or []
    assert len(segments) == 1
    assert str(segments[0].get("primary_heading_path") or "") == "Box 1"
    assert str(segments[0].get("support_locate_anchor") or "") == (
        "It can be shown that when the number of sampling patterns used M >= O(K log(N/K))..."
    )
    assert str(segments[0].get("locate_policy") or "") == "required"


def test_enrich_provenance_segments_for_display_preserves_exact_method_detail_heading():
    provenance = {
        "block_map": {
            "blk_setup": {
                "block_id": "blk_setup",
                "anchor_id": "p_00035",
                "heading_path": "ARTICLE / Methods / Principle of high-throughput SPH",
                "kind": "paragraph",
                "text": (
                    "**Experimental setup.** Thus, the beat frequency of these two beams is 62,500 Hz. "
                    "The data acquisition card uses a sampling rate of 1.25 Ms/s."
                ),
                "raw_text": (
                    "**Experimental setup.** Thus, the beat frequency of these two beams is 62,500 Hz. "
                    "The data acquisition card uses a sampling rate of 1.25 Ms/s."
                ),
            }
        },
        "segments": [
            {
                "segment_id": "seg_001",
                "segment_index": 1,
                "text": "The paper states this explicitly in ARTICLE / Methods / Principle of high-throughput SPH / Experimental setup:",
                "raw_markdown": "The paper states this explicitly in ARTICLE / Methods / Principle of high-throughput SPH / Experimental setup:",
                "evidence_mode": "synthesis",
                "claim_type": "critical_fact_claim",
                "anchor_kind": "sentence",
                "anchor_text": "The paper states this explicitly in ARTICLE / Methods / Principle of high-throughput SPH / Experimental setup:",
                "locate_policy": "hidden",
            },
            {
                "segment_id": "seg_002",
                "segment_index": 2,
                "text": "Thus, the beat frequency of these two beams is 62,500 Hz. The data acquisition card uses a sampling rate of 1.25 Ms/s.",
                "raw_markdown": "Thus, the beat frequency of these two beams is 62,500 Hz. The data acquisition card uses a sampling rate of 1.25 Ms/s.",
                "evidence_mode": "direct",
                "claim_type": "method_detail",
                "must_locate": True,
                "anchor_kind": "sentence",
                "anchor_text": "Thus, the beat frequency of these two beams is 62,500 Hz. The data acquisition card uses a sampling rate of 1.25 Ms/s.",
                "primary_block_id": "blk_setup",
                "primary_anchor_id": "p_00035",
                "primary_heading_path": "ARTICLE / Methods / Principle of high-throughput SPH / Experimental setup",
                "evidence_block_ids": ["blk_setup"],
                "support_block_ids": [],
                "support_slot_claim_type": "method_detail",
                "support_locate_anchor": "Thus, the beat frequency of these two beams is 62,500 Hz. The data acquisition card uses a sampling rate of 1.25 Ms/s.",
                "locate_policy": "required",
            },
        ],
    }

    enriched = _enrich_provenance_segments_for_display(provenance, hits=[], anchor_ns="conv:test")

    assert isinstance(enriched, dict)
    segments = enriched.get("segments") or []
    assert len(segments) == 2
    assert str(segments[1].get("primary_heading_path") or "") == (
        "ARTICLE / Methods / Principle of high-throughput SPH / Experimental setup"
    )
    assert str(segments[1].get("support_slot_claim_type") or "") == "method_detail"


def test_enrich_provenance_segments_for_display_rebinds_formula_claim_using_equation_index(tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    assets_dir = md_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    formula = "$$Y = \\\\sum_{i=1}^{N} X_i \\\\odot M_i + Z \\\\tag{1}$$"
    method_line = "This paragraph explains the measurement process before the formal equation."
    md_main.write_text(
        (
            "# Method\n\n"
            f"{method_line}\n\n"
            f"{formula}\n"
        ),
        encoding="utf-8",
    )

    blocks = task_runtime.load_source_blocks(md_main)
    method_block = next(
        block for block in blocks
        if "measurement process" in str(block.get("text") or "").lower()
    )
    equation_block = next(
        block for block in blocks
        if str(block.get("kind") or "").strip().lower() == "equation"
    )
    (assets_dir / "equation_index.json").write_text(
        json.dumps(
            {
                "equations": [
                    {
                        "equation_number": 1,
                        "equation_markdown": str(equation_block.get("raw_text") or equation_block.get("text") or ""),
                        "normalized_tex": "Y = sum_i X_i odot M_i + Z tag(1)",
                        "context_before": method_line,
                        "context_after": "",
                        "block_id": str(equation_block.get("block_id") or ""),
                        "anchor_id": str(equation_block.get("anchor_id") or ""),
                        "heading_path": str(equation_block.get("heading_path") or ""),
                        "line_start": int(equation_block.get("line_start") or 0),
                        "line_end": int(equation_block.get("line_end") or 0),
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    provenance = {
        "md_path": str(md_main),
        "source_path": str(source_pdf),
        "source_name": "DemoPaper.pdf",
        "block_map": {
            str(method_block.get("block_id") or ""): dict(method_block),
        },
        "segments": [
            {
                "segment_id": "seg_formula_display_fix",
                "segment_index": 1,
                "kind": "paragraph",
                "segment_type": "equation",
                "text": "Equation (1) defines the coded measurement.",
                "raw_markdown": formula,
                "evidence_mode": "direct",
                "claim_type": "formula_claim",
                "must_locate": True,
                "anchor_kind": "equation",
                "anchor_text": formula,
                "equation_number": 1,
                "primary_block_id": str(method_block.get("block_id") or ""),
                "primary_anchor_id": str(method_block.get("anchor_id") or ""),
                "primary_heading_path": str(method_block.get("heading_path") or ""),
                "evidence_block_ids": [str(method_block.get("block_id") or "")],
                "support_block_ids": [],
                "evidence_quote": formula,
            }
        ],
    }

    enriched = _enrich_provenance_segments_for_display(provenance, hits=[], anchor_ns="conv:test")

    assert isinstance(enriched, dict)
    segments = enriched.get("segments") or []
    assert len(segments) == 1
    seg = segments[0]
    assert str(seg.get("primary_block_id") or "") == str(equation_block.get("block_id") or "")
    assert str(seg.get("primary_anchor_id") or "") == str(equation_block.get("anchor_id") or "")
    assert str(seg.get("anchor_kind") or "") == "equation"
    assert str(seg.get("hit_level") or "") == "exact"
    locate_target = seg.get("locate_target") or {}
    assert str(locate_target.get("blockId") or "") == str(equation_block.get("block_id") or "")
    assert str(locate_target.get("anchorId") or "") == str(equation_block.get("anchor_id") or "")
    assert str(locate_target.get("anchorKind") or "") == "equation"
    assert str(locate_target.get("hitLevel") or "") == "exact"
    reader_open = seg.get("reader_open") or {}
    assert str(reader_open.get("sourcePath") or "") == str(source_pdf)
    assert str(reader_open.get("blockId") or "") == str(equation_block.get("block_id") or "")
    assert str(reader_open.get("anchorId") or "") == str(equation_block.get("anchor_id") or "")
    assert bool(reader_open.get("strictLocate")) is True
    assert str(((reader_open.get("locateTarget") or {}).get("anchorKind")) or "") == "equation"
    block_map = enriched.get("block_map") or {}
    assert str(equation_block.get("block_id") or "") in block_map


def test_enrich_provenance_segments_for_display_backfills_anchor_only_segment_using_anchor_index(tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    assets_dir = md_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "# Abstract\n\n"
            "APR improves coherent reconstruction quality.\n\n"
            "# Methods\n\n"
            "APR was performed using image registration based on phase correlation of the off-axis raw images.\n"
        ),
        encoding="utf-8",
    )

    blocks = task_runtime.load_source_blocks(md_main)
    method_block = next(
        block for block in blocks
        if "phase correlation" in str(block.get("text") or "").lower()
    )
    (assets_dir / "anchor_index.json").write_text(
        json.dumps(
            {
                "anchors": [
                    {
                        "anchor_id": str(method_block.get("anchor_id") or ""),
                        "block_id": str(method_block.get("block_id") or ""),
                        "kind": str(method_block.get("kind") or ""),
                        "heading_path": str(method_block.get("heading_path") or ""),
                        "order_index": int(method_block.get("order_index") or 0),
                        "line_start": int(method_block.get("line_start") or 0),
                        "line_end": int(method_block.get("line_end") or 0),
                        "text": str(method_block.get("text") or ""),
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    provenance = {
        "md_path": str(md_main),
        "source_path": str(source_pdf),
        "source_name": "DemoPaper.pdf",
        "block_map": {},
        "segments": [
            {
                "segment_id": "seg_anchor_only_display_fix",
                "segment_index": 1,
                "kind": "paragraph",
                "segment_type": "evidence",
                "text": "APR uses phase correlation to align the off-axis raw images.",
                "raw_markdown": "APR uses phase correlation to align the off-axis raw images.",
                "evidence_mode": "direct",
                "claim_type": "method_detail",
                "must_locate": True,
                "locate_policy": "required",
                "primary_block_id": "",
                "primary_anchor_id": str(method_block.get("anchor_id") or ""),
                "primary_heading_path": str(method_block.get("heading_path") or ""),
                "evidence_block_ids": [],
                "support_block_ids": [],
            }
        ],
    }

    enriched = _enrich_provenance_segments_for_display(provenance, hits=[], anchor_ns="conv:test")

    assert isinstance(enriched, dict)
    segments = enriched.get("segments") or []
    assert len(segments) == 1
    seg = segments[0]
    assert str(seg.get("primary_block_id") or "") == str(method_block.get("block_id") or "")
    assert str(seg.get("primary_anchor_id") or "") == str(method_block.get("anchor_id") or "")
    assert str(seg.get("primary_heading_path") or "") == str(method_block.get("heading_path") or "")
    assert str(seg.get("anchor_kind") or "") == "sentence"
    assert str(seg.get("hit_level") or "") == "exact"
    locate_target = seg.get("locate_target") or {}
    assert str(locate_target.get("blockId") or "") == str(method_block.get("block_id") or "")
    assert str(locate_target.get("anchorId") or "") == str(method_block.get("anchor_id") or "")
    assert str(locate_target.get("anchorKind") or "") == "sentence"
    assert str(locate_target.get("hitLevel") or "") == "exact"
    reader_open = seg.get("reader_open") or {}
    assert str(reader_open.get("sourcePath") or "") == str(source_pdf)
    assert str(reader_open.get("blockId") or "") == str(method_block.get("block_id") or "")
    assert str(reader_open.get("anchorId") or "") == str(method_block.get("anchor_id") or "")
    assert str(((reader_open.get("locateTarget") or {}).get("anchorKind")) or "") == "sentence"
    block_map = enriched.get("block_map") or {}
    assert str(method_block.get("block_id") or "") in block_map


def test_enrich_provenance_segments_for_display_rebinds_figure_claim_using_figure_index(tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "VisionPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "VisionPaper"
    assets_dir = md_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "fig1.png").write_bytes(b"fake")
    md_main = md_dir / "VisionPaper.en.md"
    figure_caption = (
        "Figure 1. Given a single snapshot compressed image, our method is able to recover "
        "the underlying 3D scene representation."
    )
    method_para = (
        "Our method takes a single compressed image and encoding masks as input, and recovers "
        "the underlying 3D scene representation as well as camera poses."
    )
    md_main.write_text(
        (
            "# VisionPaper\n\n"
            "![Figure 1](./assets/fig1.png)\n"
            f"*{figure_caption}*\n\n"
            "## Method\n\n"
            f"{method_para}\n"
        ),
        encoding="utf-8",
    )

    blocks = task_runtime.load_source_blocks(md_main)
    figure_block = next(block for block in blocks if str(block.get("kind") or "") == "figure")
    caption_block = next(
        block for block in blocks
        if str(block.get("kind") or "") == "paragraph"
        and "single snapshot compressed image" in str(block.get("text") or "").lower()
    )
    method_block = next(
        block for block in blocks
        if str(block.get("kind") or "") == "paragraph"
        and "encoding masks as input" in str(block.get("text") or "").lower()
    )
    (assets_dir / "figure_index.json").write_text(
        json.dumps(
            {
                "figures": [
                    {
                        "paper_figure_number": 1,
                        "figure_id": str(figure_block.get("figure_id") or ""),
                        "figure_block_id": str(figure_block.get("block_id") or ""),
                        "caption_block_id": str(caption_block.get("block_id") or ""),
                        "caption_anchor_id": str(caption_block.get("anchor_id") or ""),
                        "anchor_id": str(figure_block.get("anchor_id") or ""),
                        "heading_path": str(figure_block.get("heading_path") or ""),
                        "caption": figure_caption,
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    provenance = {
        "md_path": str(md_main),
        "source_path": str(source_pdf),
        "source_name": "VisionPaper.pdf",
        "block_map": {
            str(method_block.get("block_id") or ""): dict(method_block),
        },
        "segments": [
            {
                "segment_id": "seg_figure_display_fix",
                "segment_index": 1,
                "kind": "paragraph",
                "segment_type": "evidence",
                "text": "Figure 1 shows recovery from a single snapshot compressed image.",
                "raw_markdown": "Figure 1 shows recovery from a single snapshot compressed image.",
                "evidence_mode": "direct",
                "claim_type": "figure_claim",
                "must_locate": True,
                "anchor_kind": "figure",
                "anchor_text": "Figure 1",
                "support_slot_figure_number": 1,
                "primary_block_id": str(method_block.get("block_id") or ""),
                "primary_anchor_id": str(method_block.get("anchor_id") or ""),
                "primary_heading_path": str(method_block.get("heading_path") or ""),
                "evidence_block_ids": [str(method_block.get("block_id") or "")],
                "support_block_ids": [],
                "evidence_quote": "Given a single snapshot compressed image, our method is able to recover the underlying 3D scene representation.",
            }
        ],
    }

    enriched = _enrich_provenance_segments_for_display(provenance, hits=[], anchor_ns="conv:test")

    assert isinstance(enriched, dict)
    segments = enriched.get("segments") or []
    assert len(segments) == 1
    seg = segments[0]
    # For figure claims, prefer landing on the caption when it exists (more informative than the figure placeholder).
    assert str(seg.get("primary_block_id") or "") == str(caption_block.get("block_id") or "")
    assert str(seg.get("primary_anchor_id") or "") == str(caption_block.get("anchor_id") or "")
    assert str(seg.get("primary_heading_path") or "") == str(figure_block.get("heading_path") or "")
    assert str(seg.get("anchor_kind") or "") == "figure"
    assert str(seg.get("hit_level") or "") == "exact"
    assert str(caption_block.get("block_id") or "") in list(seg.get("evidence_block_ids") or [])
    locate_target = seg.get("locate_target") or {}
    assert str(locate_target.get("blockId") or "") == str(caption_block.get("block_id") or "")
    assert str(locate_target.get("anchorId") or "") == str(caption_block.get("anchor_id") or "")
    assert str(locate_target.get("anchorKind") or "") == "figure"
    assert int(locate_target.get("anchorNumber") or 0) == 1
    reader_open = seg.get("reader_open") or {}
    assert str(reader_open.get("sourcePath") or "") == str(source_pdf)
    assert str(reader_open.get("blockId") or "") == str(caption_block.get("block_id") or "")
    assert str(reader_open.get("anchorId") or "") == str(caption_block.get("anchor_id") or "")
    assert int(reader_open.get("anchorNumber") or 0) == 1
    assert str(((reader_open.get("locateTarget") or {}).get("anchorKind")) or "") == "figure"
    alternatives = list(reader_open.get("alternatives") or [])
    assert len(alternatives) >= 1
    assert isinstance(alternatives[0], dict)
    assert list(reader_open.get("visibleAlternatives") or []) == alternatives
    assert list(reader_open.get("evidenceAlternatives") or []) == alternatives
    block_map = enriched.get("block_map") or {}
    assert str(figure_block.get("block_id") or "") in block_map


def test_enrich_messages_reuses_persisted_render_cache(monkeypatch, tmp_path: Path):
    from api import chat_render

    calls = {"primary": 0}

    def fake_primary(_md, _hits, *, anchor_ns="", canonical_paths=None):
        del _hits, anchor_ns, canonical_paths
        calls["primary"] += 1
        return (
            f"cached::{_md}",
            [{"num": 1, "anchor": "kb-cite-demo-1", "source_name": "demo.pdf"}],
        )

    monkeypatch.setattr(chat_render, "_annotate_inpaper_citations_with_hover_meta", fake_primary)

    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("cache test")
    user_id = store.append_message(conv_id, "user", "test")
    assistant_id = store.append_message(conv_id, "assistant", "SPI relies on compressive sensing [[CITE:s1234abcd:1]].")
    refs_by_user = {
        user_id: {
            "prompt_sig": "sig-1",
            "updated_at": 1.0,
            "used_query": "test",
            "used_translation": False,
            "hits": [
                {
                    "text": "dummy",
                    "meta": {"source_path": r"db\doc\doc.en.md"},
                }
            ],
        }
    }

    store.merge_message_meta(
        assistant_id,
        {"answer_quality": {"prompt_family": "citation_lookup", "output_mode": "citation_lookup"}},
    )

    first = enrich_messages_with_reference_render(store.get_messages(conv_id), refs_by_user, conv_id=conv_id, chat_store=store)
    second = enrich_messages_with_reference_render(store.get_messages(conv_id), refs_by_user, conv_id=conv_id, chat_store=store)
    persisted = store.get_messages(conv_id)[-1]
    render_cache = ((persisted.get("meta") or {}).get("render_cache") or {})

    assert calls["primary"] == 1
    assert str(first[-1].get("rendered_content") or "") == str(second[-1].get("rendered_content") or "")
    assert str(second[-1].get("copy_text") or "").strip()
    assert isinstance(render_cache.get("render_packet"), dict)


def test_render_cache_persists_render_packet_when_contracts_present(monkeypatch, tmp_path: Path):
    from api import chat_render

    def fake_primary(_md, _hits, *, anchor_ns="", canonical_paths=None):
        del _hits, anchor_ns, canonical_paths
        return (
            f"cached::{_md}",
            [{"num": 1, "anchor": "kb-cite-demo-1", "source_name": "demo.pdf"}],
        )

    monkeypatch.setattr(chat_render, "_annotate_inpaper_citations_with_hover_meta", fake_primary)

    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("cache contract test")
    user_id = store.append_message(conv_id, "user", "test")
    store.append_message(
        conv_id,
        "assistant",
        "SPI relies on compressive sensing [[CITE:s1234abcd:1]].",
        meta={"paper_guide_contracts": {"version": 1, "intent": {"family": "citation_lookup"}}},
    )
    refs_by_user = {
        user_id: {
            "prompt_sig": "sig-1",
            "updated_at": 1.0,
            "used_query": "test",
            "used_translation": False,
            "hits": [{"text": "dummy", "meta": {"source_path": r"db\doc\doc.en.md"}}],
        }
    }

    enrich_messages_with_reference_render(store.get_messages(conv_id), refs_by_user, conv_id=conv_id, chat_store=store)
    persisted = store.get_messages(conv_id)[-1]
    render_cache = ((persisted.get("meta") or {}).get("render_cache") or {})
    render_packet = render_cache.get("render_packet")

    assert isinstance(render_packet, dict)
    assert str(render_packet.get("rendered_content") or "").strip()


def test_enrich_messages_rebuilds_degraded_numeric_citation_cache(tmp_path: Path):
    from api import chat_render

    content = (
        "成像质量提升：深度学习能够改善单像素成像的重建质量 [1]。\n\n"
        "降低采样率：端到端模型可以在更少测量下恢复目标图像 [2]。"
    )
    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("degraded citation cache")
    user_id = store.append_message(conv_id, "user", "深度学习对单像素成像有什么好处？")
    assistant_id = store.append_message(conv_id, "assistant", content)
    refs_by_user = {
        user_id: {
            "prompt_sig": "sig-1",
            "updated_at": 1.0,
            "used_query": "single pixel imaging deep learning",
            "used_translation": False,
            "hits": [
                {
                    "text": "Deep learning improves reconstruction quality in single-pixel imaging.",
                    "meta": {
                        "source_path": r"db\LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.en.md",
                        "heading_path": "Benefits / Image quality",
                    },
                },
                {
                    "text": "Learning based image reconstruction can reduce the sampling ratio.",
                    "meta": {
                        "source_path": r"db\Optics-2024-Part-based image-loop network for single-pixel imaging.en.md",
                        "heading_path": "Method / Sampling ratio",
                    },
                },
            ],
        }
    }
    cache_key = chat_render._build_message_render_cache_key(
        conv_id=conv_id,
        msg_id=assistant_id,
        role="assistant",
        content=content,
        refs_user_msg_id=user_id,
        ref_pack=refs_by_user[user_id],
        provenance=None,
    )
    store.merge_message_meta(
        assistant_id,
        {
            "render_cache": chat_render._build_render_cache_payload(
                cache_key=cache_key,
                notice="",
                rendered_body=content,
                rendered_content=content,
                copy_markdown=content,
                copy_text=content,
                cite_details=[],
                refs_user_msg_id=user_id,
                render_packet={"rendered_content": content, "cite_details": []},
            )
        },
    )

    rendered = enrich_messages_with_reference_render(
        store.get_messages(conv_id),
        refs_by_user,
        conv_id=conv_id,
        chat_store=store,
    )
    msg = rendered[-1]
    persisted = store.get_messages(conv_id)[-1]
    persisted_cache = ((persisted.get("meta") or {}).get("render_cache") or {})

    assert "](#kb-cite-" in str(msg.get("rendered_content") or "")
    assert len(msg.get("cite_details") or []) == 2
    assert all(item.get("is_inpaper") is False for item in (msg.get("cite_details") or []))
    assert len(persisted_cache.get("cite_details") or []) == 2


def test_enrich_messages_rebuilds_degraded_structured_citation_cache(monkeypatch, tmp_path: Path):
    from api import chat_render
    from ui import refs_renderer

    source_path = r"db\paper-one.en.md"
    sid = chat_render._source_cite_id(source_path)
    content = f"Prior work is cited as [[CITE:{sid}:35]]."
    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("structured citation cache")
    user_id = store.append_message(conv_id, "user", "which prior work is cited?")
    assistant_id = store.append_message(conv_id, "assistant", content)
    refs_by_user = {
        user_id: {
            "prompt_sig": "sig-structured-cache",
            "updated_at": 1.0,
            "used_query": "prior work cited",
            "used_translation": False,
            "hits": [
                {
                    "text": "The paper cites compressive sensing as prior work [35].",
                    "meta": {
                        "source_path": source_path,
                        "heading_path": "Related work",
                    },
                }
            ],
        }
    }

    def fake_resolve(_index, src, ref_num, *, source_sha1=""):
        del _index, source_sha1
        if str(src) != source_path or int(ref_num) != 35:
            return None
        return {
            "source_path": source_path,
            "source_name": "paper-one.pdf",
            "ref_num": 35,
            "ref": {
                "raw": "[35] Candes et al. Compressive sensing. 2006.",
                "title": "Compressive sensing",
                "authors": "Candes et al.",
                "year": "2006",
            },
        }

    monkeypatch.setattr(chat_render, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(chat_render, "resolve_reference_entry", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)

    cache_key = chat_render._build_message_render_cache_key(
        conv_id=conv_id,
        msg_id=assistant_id,
        role="assistant",
        content=content,
        refs_user_msg_id=user_id,
        ref_pack=refs_by_user[user_id],
        provenance=None,
    )
    store.merge_message_meta(
        assistant_id,
        {
            "render_cache": chat_render._build_render_cache_payload(
                cache_key=cache_key,
                notice="",
                rendered_body="Prior work is cited as .",
                rendered_content="Prior work is cited as .",
                copy_markdown="Prior work is cited as .",
                copy_text="Prior work is cited as .",
                cite_details=[],
                refs_user_msg_id=user_id,
                render_packet={"rendered_content": "Prior work is cited as .", "cite_details": []},
            )
        },
    )

    rendered = enrich_messages_with_reference_render(
        store.get_messages(conv_id),
        refs_by_user,
        conv_id=conv_id,
        chat_store=store,
    )
    msg = rendered[-1]
    persisted_cache = ((store.get_messages(conv_id)[-1].get("meta") or {}).get("render_cache") or {})

    assert "[35](#kb-cite-" in str(msg.get("rendered_content") or "")
    assert len(msg.get("cite_details") or []) == 1
    assert (msg.get("cite_details") or [{}])[0].get("is_inpaper") is True
    assert len(persisted_cache.get("cite_details") or []) == 1


def test_enrich_messages_ignores_previous_schema_render_cache(tmp_path: Path):
    from api import chat_render

    content = "Learning-based SPI improves reconstruction quality [1]."
    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("previous schema cache")
    user_id = store.append_message(conv_id, "user", "what helps SPI reconstruction?")
    assistant_id = store.append_message(conv_id, "assistant", content)
    refs_by_user = {
        user_id: {
            "prompt_sig": "sig-prev-cache",
            "updated_at": 1.0,
            "used_query": "SPI reconstruction",
            "used_translation": False,
            "hits": [
                {
                    "text": "Deep learning improves reconstruction quality in single-pixel imaging.",
                    "meta": {
                        "source_path": r"db\LPR-2025\LPR-2025.en.md",
                        "heading_path": "Benefits / Reconstruction quality",
                    },
                }
            ],
        }
    }
    cache_key = chat_render._build_message_render_cache_key(
        conv_id=conv_id,
        msg_id=assistant_id,
        role="assistant",
        content=content,
        refs_user_msg_id=user_id,
        ref_pack=refs_by_user[user_id],
        provenance=None,
    )
    old_cache = chat_render._build_render_cache_payload(
        cache_key=cache_key,
        notice="",
        rendered_body="stale plain [1]",
        rendered_content="stale plain [1]",
        copy_markdown="stale plain [1]",
        copy_text="stale plain [1]",
        cite_details=[],
        refs_user_msg_id=user_id,
        render_packet={"rendered_content": "stale plain [1]", "cite_details": []},
    )
    old_cache["schema"] = int(chat_render._RENDER_CACHE_SCHEMA_VERSION) - 1
    store.merge_message_meta(assistant_id, {"render_cache": old_cache})

    rendered = enrich_messages_with_reference_render(
        store.get_messages(conv_id),
        refs_by_user,
        conv_id=conv_id,
        chat_store=store,
    )
    msg = rendered[-1]
    persisted_cache = ((store.get_messages(conv_id)[-1].get("meta") or {}).get("render_cache") or {})

    assert str(msg.get("rendered_content") or "") != "stale plain [1]"
    assert "](#kb-cite-" in str(msg.get("rendered_content") or "")
    assert int(persisted_cache.get("schema") or 0) == int(chat_render._RENDER_CACHE_SCHEMA_VERSION)
    assert len(persisted_cache.get("cite_details") or []) == 1


def test_render_packet_only_rebuilds_legacy_answer_markdown_citations_when_content_empty(tmp_path: Path):
    from api import chat_render

    answer = "Learning-based SPI improves reconstruction quality [1] and reduces sampling demand [2]."
    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("legacy packet repair")
    user_id = store.append_message(conv_id, "user", "what are the benefits of deep learning for SPI?")
    assistant_id = store.append_message(
        conv_id,
        "assistant",
        "",
        meta={
            "paper_guide_contracts": {
                "version": 1,
                "intent": {"family": "overview"},
                "render_packet": {
                    "answer_markdown": answer,
                    "rendered_body": "",
                    "rendered_content": "",
                    "copy_markdown": "",
                    "copy_text": "",
                    "cite_details": [],
                },
            }
        },
    )
    refs_by_user = {
        user_id: {
            "prompt_sig": "sig-legacy",
            "updated_at": 1.0,
            "used_query": "single pixel imaging deep learning benefits",
            "used_translation": False,
            "hits": [
                {
                    "text": "Deep learning improves reconstruction quality in single-pixel imaging.",
                    "meta": {
                        "source_path": r"db\paper-a.en.md",
                        "heading_path": "Benefits / Reconstruction quality",
                    },
                },
                {
                    "text": "Learning based reconstruction can reduce sampling demand.",
                    "meta": {
                        "source_path": r"db\paper-b.en.md",
                        "heading_path": "Benefits / Sampling demand",
                    },
                },
            ],
        }
    }
    stale_packet = (((store.get_messages(conv_id)[-1].get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    cache_key = chat_render._build_message_render_cache_key(
        conv_id=conv_id,
        msg_id=assistant_id,
        role="assistant",
        content=answer,
        refs_user_msg_id=user_id,
        ref_pack=refs_by_user[user_id],
        provenance=None,
    )
    store.merge_message_meta(
        assistant_id,
        {
            "render_cache": chat_render._build_render_cache_payload(
                cache_key=cache_key,
                notice="",
                rendered_body="",
                rendered_content="",
                copy_markdown="",
                copy_text="",
                cite_details=[],
                refs_user_msg_id=user_id,
                render_packet=stale_packet,
            )
        },
    )

    rendered = enrich_messages_with_reference_render(
        store.get_messages(conv_id),
        refs_by_user,
        conv_id=conv_id,
        chat_store=store,
        render_packet_only=True,
    )
    msg = rendered[-1]
    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    persisted = store.get_messages(conv_id)[-1]
    persisted_packet = ((((persisted.get("meta") or {}).get("render_cache") or {}).get("render_packet") or {}))

    assert "rendered_body" not in msg
    assert len(msg.get("cite_details") or []) == 2
    assert "](#kb-cite-" in str(packet.get("rendered_body") or "")
    assert "](#kb-cite-" in str(packet.get("rendered_content") or "")
    assert len(packet.get("cite_details") or []) == 2
    assert str(packet.get("answer_markdown") or "") == answer
    assert len(persisted_packet.get("cite_details") or []) == 2


def test_enrich_provenance_surfaces_hidden_derived_formula_source_anchor():
    provenance = {
        "source_path": "paper.pdf",
        "segments": [
            {
                "segment_id": "seg-formula",
                "claim_type": "formula_claim",
                "formula_origin": "derived",
                "evidence_mode": "direct",
                "locate_policy": "hidden",
                "locate_surface_policy": "hidden",
                "primary_heading_path": "How a single-pixel camera works",
                "primary_block_id": "blk-12",
                "primary_anchor_id": "p-12",
                "support_locate_anchor": "The single-pixel camera consists of two main components.",
                "locate_target": {
                    "segmentId": "seg-formula",
                    "headingPath": "How a single-pixel camera works",
                    "snippet": "The single-pixel camera consists of two main components.",
                    "blockId": "blk-12",
                    "anchorId": "p-12",
                    "anchorKind": "equation",
                    "locatePolicy": "hidden",
                    "locateSurfacePolicy": "hidden",
                },
                "reader_open": {
                    "sourcePath": "paper.pdf",
                    "blockId": "blk-12",
                    "anchorId": "p-12",
                    "anchorKind": "equation",
                    "strictLocate": False,
                    "locateTarget": {
                        "blockId": "blk-12",
                        "anchorId": "p-12",
                        "anchorKind": "equation",
                        "locatePolicy": "hidden",
                        "locateSurfacePolicy": "hidden",
                    },
                },
            }
        ],
        "block_map": {
            "blk-12": {
                "block_id": "blk-12",
                "anchor_id": "p-12",
                "heading_path": "How a single-pixel camera works",
                "text": "The single-pixel camera consists of two main components.",
                "kind": "paragraph",
            }
        },
    }

    out = _enrich_provenance_segments_for_display(provenance, [], anchor_ns="conv:1:2:test")
    seg = (out.get("segments") or [])[0]

    assert seg.get("locate_policy") == "required"
    assert seg.get("locate_surface_policy") == "primary"
    assert seg.get("locate_target", {}).get("locatePolicy") == "required"
    assert seg.get("locate_target", {}).get("anchorKind") in {"paragraph", "sentence"}
    assert seg.get("reader_open", {}).get("strictLocate") is True


def test_enrich_messages_refreshes_stale_cached_render_packet_from_provenance(tmp_path: Path):
    from api import chat_render

    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("stale cache packet test")
    user_id = store.append_message(conv_id, "user", "Explain the SPI workflow.")
    content = "Grounded answer."
    good_locate_target = {
        "segmentId": "seg_001",
        "headingPath": "Abstract / Acquisition and image reconstruction strategies.",
        "snippet": "Unlike the raster-scan strategy...",
        "blockId": "blk-good-26",
        "anchorId": "p-good-19",
        "anchorKind": "sentence",
        "hitLevel": "exact",
        "locatePolicy": "required",
        "locateSurfacePolicy": "primary",
    }
    provenance = {
        "status": "ready",
        "strict_identity_ready": True,
        "must_locate_count": 1,
        "strict_identity_count": 1,
        "segments": [
            {
                "segment_id": "seg_001",
                "source_segment_id": "seg_001",
                "claim_type": "critical_fact_claim",
                "must_locate": True,
                "locate_policy": "required",
                "locate_surface_policy": "primary",
                "evidence_mode": "direct",
                "primary_block_id": "blk-good-26",
                "primary_anchor_id": "p-good-19",
                "locate_target": good_locate_target,
                "reader_open": {
                    "sourcePath": "demo.en.md",
                    "headingPath": "Abstract / Acquisition and image reconstruction strategies.",
                    "blockId": "blk-good-26",
                    "anchorId": "p-good-19",
                    "anchorKind": "sentence",
                    "strictLocate": True,
                    "locateTarget": good_locate_target,
                },
            }
        ],
    }
    stale_packet = {
        "answer_markdown": content,
        "rendered_body": content,
        "rendered_content": content,
        "copy_markdown": content,
        "copy_text": content,
        "locate_target": {
            "segmentId": "seg_001",
            "snippet": "Grounded answer.",
            "hitLevel": "none",
            "locatePolicy": "hidden",
            "locateSurfacePolicy": "hidden",
        },
        "reader_open": {},
        "visible_segment_ids": [],
    }
    assistant_id = store.append_message(
        conv_id,
        "assistant",
        content,
        meta={
            "provenance": provenance,
            "paper_guide_contracts": {
                "version": 1,
                "intent": {"family": "paper_guide"},
                "render_packet": stale_packet,
            },
        },
    )
    cache_key = chat_render._build_message_render_cache_key(
        conv_id=conv_id,
        msg_id=assistant_id,
        role="assistant",
        content=content,
        refs_user_msg_id=user_id,
        ref_pack=None,
        provenance=provenance,
    )
    store.merge_message_meta(
        assistant_id,
        {
            "render_cache": chat_render._build_render_cache_payload(
                cache_key=cache_key,
                notice="",
                rendered_body=content,
                rendered_content=content,
                copy_markdown=content,
                copy_text=content,
                cite_details=[],
                refs_user_msg_id=user_id,
                render_packet=stale_packet,
            )
        },
    )

    rendered = enrich_messages_with_reference_render(
        store.get_messages(conv_id),
        refs_by_user={},
        conv_id=conv_id,
        chat_store=store,
        render_packet_only=True,
    )
    packet = (((rendered[-1].get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    persisted = store.get_messages(conv_id)[-1]
    cache_packet = (((persisted.get("meta") or {}).get("render_cache") or {}).get("render_packet") or {})

    assert packet.get("locate_target", {}).get("blockId") == "blk-good-26"
    assert packet.get("visible_segment_ids") == ["seg_001"]
    assert cache_packet.get("locate_target", {}).get("blockId") == "blk-good-26"
    assert cache_packet.get("visible_segment_ids") == ["seg_001"]

    second = enrich_messages_with_reference_render(
        store.get_messages(conv_id),
        refs_by_user={},
        conv_id=conv_id,
        chat_store=store,
        render_packet_only=True,
    )
    second_packet = (((second[-1].get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    assert second_packet.get("locate_target", {}).get("blockId") == "blk-good-26"


def test_render_packet_only_env_strips_legacy_render_fields(monkeypatch):
    from api import chat_render

    monkeypatch.setenv("KB_CHAT_RENDER_PACKET_ONLY", "1")
    messages = [
        {"id": 1, "role": "user", "content": "explain"},
        {
            "id": 2,
            "role": "assistant",
            "content": "SPI relies on compressive sensing [[CITE:s1234abcd:1]].",
            "meta": {"paper_guide_contracts": {"version": 1, "intent": {"family": "citation_lookup"}}},
        },
    ]
    refs_by_user = {
        1: {
            "hits": [{"text": "dummy", "meta": {"source_path": r"db\doc\doc.en.md"}}],
        }
    }

    rendered = chat_render.enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-test")
    msg = rendered[-1]

    assert "rendered_body" not in msg
    assert "rendered_content" not in msg
    assert "copy_text" not in msg
    assert "copy_markdown" not in msg
    assert isinstance(msg.get("cite_details"), list)
    assert "notice" not in msg

    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    assert str(packet.get("rendered_body") or "").strip()
    assert isinstance(packet.get("cite_details"), list)


def test_render_packet_only_flag_strips_legacy_render_fields(monkeypatch):
    from api import chat_render

    # No env needed; flag should be enough.
    monkeypatch.delenv("KB_CHAT_RENDER_PACKET_ONLY", raising=False)
    messages = [
        {"id": 1, "role": "user", "content": "explain"},
        {
            "id": 2,
            "role": "assistant",
            "content": "SPI relies on compressive sensing [[CITE:s1234abcd:1]].",
            "meta": {"paper_guide_contracts": {"version": 1, "intent": {"family": "citation_lookup"}}},
        },
    ]
    refs_by_user = {
        1: {
            "hits": [{"text": "dummy", "meta": {"source_path": r"db\doc\doc.en.md"}}],
        }
    }

    rendered = chat_render.enrich_messages_with_reference_render(
        messages,
        refs_by_user,
        conv_id="conv-test",
        render_packet_only=True,
    )
    msg = rendered[-1]

    assert "rendered_body" not in msg
    assert "rendered_content" not in msg
    assert "copy_text" not in msg
    assert "copy_markdown" not in msg
    assert isinstance(msg.get("cite_details"), list)
    assert "notice" not in msg


def test_figure_claim_segments_can_reach_exact_hit_level_after_required_coverage_contract():
    from api import chat_render

    messages = [
        {"id": 1, "role": "user", "content": "show me figure 6"},
        {
            "id": 2,
            "role": "assistant",
            "content": "Figure 6 shows the pipeline.",
            "provenance": {
                "source_path": "demo.md",
                "source_name": "demo.pdf",
                "md_path": "demo.md",
                "segments": [
                    {
                        "segment_id": "seg-fig-1",
                        "text": "Figure 6 shows the pipeline.",
                        "evidence_mode": "direct",
                        "claim_type": "figure_claim",
                        "must_locate": True,
                        "locate_policy": "required",
                        "locate_surface_policy": "primary",
                        "primary_heading_path": "Methods / Figure 6",
                        "primary_block_id": "blk_demo_00001",
                        "primary_anchor_id": "fg_00006",
                        # anchor_kind intentionally omitted; contract should fill it.
                    }
                ],
                "block_map": {
                    "blk_demo_00001": {
                        "block_id": "blk_demo_00001",
                        "anchor_id": "fg_00006",
                        "kind": "figure",
                        "heading_path": "Methods / Figure 6",
                        "text": "Figure 6",
                        "line_start": 1,
                        "line_end": 1,
                        "number": 6,
                    }
                },
            },
            "meta": {"paper_guide_contracts": {"version": 1, "intent": {"family": "figure_walkthrough"}}},
        },
    ]

    rendered = chat_render.enrich_messages_with_reference_render(
        messages,
        refs_by_user={},
        conv_id="conv-test",
        render_packet_only=True,
    )
    msg = rendered[-1]
    prov = msg.get("provenance") or {}
    segs = prov.get("segments") or []
    seg = segs[0] if isinstance(segs, list) and segs else {}

    assert str(seg.get("hit_level") or "") == "exact"


def test_figure_panel_segments_can_reach_exact_hit_level_after_required_coverage_contract():
    from api import chat_render

    messages = [
        {"id": 1, "role": "user", "content": "what does panel (b) show"},
        {
            "id": 2,
            "role": "assistant",
            "content": "Panel (b) compares the enhancement performance.",
            "provenance": {
                "source_path": "demo.md",
                "source_name": "demo.pdf",
                "md_path": "demo.md",
                "segments": [
                    {
                        "segment_id": "seg-figp-1",
                        "text": "Panel (b) compares the enhancement performance.",
                        "evidence_mode": "direct",
                        "claim_type": "figure_panel",
                        "must_locate": True,
                        "locate_policy": "required",
                        "locate_surface_policy": "primary",
                        "primary_heading_path": "Methods / Figure 6",
                        "primary_block_id": "blk_demo_00002",
                        "primary_anchor_id": "p_00068",
                        # anchor_kind intentionally omitted; contract should fill it.
                        "support_slot_figure_number": 6,
                        "support_slot_panel_letters": ["b"],
                    }
                ],
                "block_map": {
                    "blk_demo_00002": {
                        "block_id": "blk_demo_00002",
                        "anchor_id": "p_00068",
                        "kind": "paragraph",
                        "heading_path": "Methods / Figure 6",
                        "text": "Figure 6 ... b The enhancement comparison ...",
                        "line_start": 1,
                        "line_end": 1,
                        "number": 0,
                        "paper_figure_number": 6,
                    }
                },
            },
            "meta": {"paper_guide_contracts": {"version": 1, "intent": {"family": "figure_walkthrough"}}},
        },
    ]

    rendered = chat_render.enrich_messages_with_reference_render(
        messages,
        refs_by_user={},
        conv_id="conv-test",
        render_packet_only=True,
    )
    msg = rendered[-1]
    prov = msg.get("provenance") or {}
    segs = prov.get("segments") or []
    seg = segs[0] if isinstance(segs, list) and segs else {}

    assert str(seg.get("hit_level") or "") == "exact"


def test_figure_claim_prefers_caption_block_as_primary_locate_target_when_available():
    from api import chat_render

    messages = [
        {"id": 1, "role": "user", "content": "what does figure 6 show"},
        {
            "id": 2,
            "role": "assistant",
            "content": "Figure 6 shows the pipeline.",
            "provenance": {
                "source_path": "demo.md",
                "source_name": "demo.pdf",
                "md_path": "demo.md",
                "segments": [
                    {
                        "segment_id": "seg-fig-prim-1",
                        "text": "Figure 6 shows the pipeline.",
                        "evidence_mode": "direct",
                        "claim_type": "figure_claim",
                        "must_locate": True,
                        "locate_policy": "required",
                        "locate_surface_policy": "primary",
                        "primary_heading_path": "Methods / Figure 6",
                        "primary_block_id": "blk_demo_fig",
                        "primary_anchor_id": "fg_00006",
                        "paper_figure_number": 6,
                    }
                ],
                "block_map": {
                    "blk_demo_fig": {
                        "block_id": "blk_demo_fig",
                        "anchor_id": "fg_00006",
                        "kind": "figure",
                        "heading_path": "Methods / Figure 6",
                        "text": "Figure 6",
                        "line_start": 1,
                        "line_end": 1,
                        "number": 6,
                        "paper_figure_number": 6,
                    },
                    "blk_demo_cap": {
                        "block_id": "blk_demo_cap",
                        "anchor_id": "p_00068",
                        "kind": "paragraph",
                        "figure_role": "caption",
                        "paper_figure_number": 6,
                        "heading_path": "Methods / Figure 6",
                        "text": "**Figure 6.** Caption text for the pipeline.",
                        "raw_text": "**Figure 6.** Caption text for the pipeline.",
                        "line_start": 2,
                        "line_end": 2,
                    },
                },
            },
            "meta": {"paper_guide_contracts": {"version": 1, "intent": {"family": "figure_walkthrough"}}},
        },
    ]

    rendered = chat_render.enrich_messages_with_reference_render(
        messages,
        refs_by_user={},
        conv_id="conv-test",
        render_packet_only=True,
    )
    msg = rendered[-1]
    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    reader_open = packet.get("reader_open") or {}

    assert str(reader_open.get("blockId") or "") == "blk_demo_cap"
    assert str(reader_open.get("anchorId") or "") == "p_00068"


def test_render_packet_notice_is_not_dropped_when_preserving_existing_render(monkeypatch):
    from api import chat_render

    monkeypatch.delenv("KB_CHAT_RENDER_PACKET_ONLY", raising=False)
    messages = [
        {"id": 1, "role": "user", "content": "explain"},
        {
            "id": 2,
            "role": "assistant",
            # This prefix triggers _split_kb_miss_notice() and produces a non-empty notice.
            "content": "未命中知识库片段\nBody that cannot be re-rendered without hits.",
            "meta": {
                "paper_guide_contracts": {
                    "version": 1,
                    "intent": {"family": "citation_lookup"},
                    # Existing contract has cite_details but no notice; preserving existing render
                    # should still pick up the current notice extracted from content.
                    "render_packet": {
                        "notice": "",
                        "rendered_body": "Existing rendered body [1](#kb-cite-demo-1).",
                        "rendered_content": "Existing rendered body [1](#kb-cite-demo-1).",
                        "copy_markdown": "Existing rendered body [1](#kb-cite-demo-1).",
                        "copy_text": "Existing rendered body [1].",
                        "cite_details": [{"num": 1, "anchor": "kb-cite-demo-1", "source_name": "demo.pdf"}],
                    },
                }
            },
        },
    ]

    rendered = chat_render.enrich_messages_with_reference_render(
        messages,
        refs_by_user={},
        conv_id="conv-test",
        render_packet_only=True,
    )
    msg = rendered[-1]

    assert "notice" not in msg
    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    assert "未命中知识库片段" in str(packet.get("notice") or "")
    assert isinstance(packet.get("cite_details"), list)


def test_merge_render_packet_contract_meta_drops_stale_negative_locate_when_no_current_identity():
    from api import chat_render

    rec = {
        "id": 2,
        "role": "assistant",
        "content": "The paper does not mention ADMM in the retrieved context.",
        "rendered_body": "The paper does not mention ADMM in the retrieved context.",
        "rendered_content": "The paper does not mention ADMM in the retrieved context.",
        "copy_markdown": "The paper does not mention ADMM in the retrieved context.",
        "copy_text": "The paper does not mention ADMM in the retrieved context.",
        "meta": {
            "paper_guide_contracts": {
                "version": 1,
                "intent": {"family": "overview"},
                "render_packet": {
                    "rendered_body": "The paper does not mention ADMM in the retrieved context.",
                    "rendered_content": "The paper does not mention ADMM in the retrieved context.",
                    "copy_markdown": "The paper does not mention ADMM in the retrieved context.",
                    "copy_text": "The paper does not mention ADMM in the retrieved context.",
                    "locate_target": {
                        "segmentId": "seg-neg",
                        "headingPath": "Discussion",
                        "snippet": "The paper does not mention ADMM in the retrieved context.",
                        "anchorText": "The paper does not mention ADMM in the retrieved context.",
                        "blockId": "b-neg",
                        "anchorId": "a-neg",
                        "anchorKind": "sentence",
                        "locatePolicy": "required",
                        "locateSurfacePolicy": "primary",
                    },
                    "reader_open": {
                        "sourcePath": "demo.md",
                        "headingPath": "Discussion",
                        "snippet": "The paper does not mention ADMM in the retrieved context.",
                        "blockId": "b-neg",
                        "anchorId": "a-neg",
                        "anchorKind": "sentence",
                        "strictLocate": True,
                    },
                },
            }
        },
    }

    chat_render._merge_render_packet_contract_meta(
        rec=rec,
        msg_id=2,
        enriched_provenance={"segments": []},
        chat_store=None,
    )

    packet = (((rec.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    assert packet.get("locate_target") == {}
    assert packet.get("reader_open") == {}


def test_merge_render_packet_contract_meta_surfaces_primary_evidence_from_provenance():
    from api import chat_render

    rec = {
        "content": "Grounded answer.",
        "rendered_body": "Grounded answer.",
        "rendered_content": "Grounded answer.",
        "copy_markdown": "Grounded answer.",
        "copy_text": "Grounded answer.",
        "notice": "",
        "cite_details": [],
        "meta": {
            "paper_guide_contracts": {
                "primary_evidence": {
                    "source_name": "demo.pdf",
                    "heading_path": "Methods / APR",
                },
                "render_packet": {},
            }
        },
    }

    chat_render._merge_render_packet_contract_meta(
        rec=rec,
        msg_id=2,
        enriched_provenance={
            "segments": [
                {
                    "segment_id": "seg-1",
                    "locate_policy": "required",
                    "locate_target": {
                        "segmentId": "seg-1",
                        "headingPath": "Methods / APR",
                        "blockId": "b-7",
                    },
                    "reader_open": {
                        "sourcePath": "demo.md",
                        "headingPath": "Methods / APR",
                        "blockId": "b-7",
                    },
                }
            ],
            "primary_evidence": {
                "source_path": "demo.md",
                "source_name": "demo.pdf",
                "block_id": "b-7",
                "anchor_id": "a-7",
                "heading_path": "Methods / APR",
                "snippet": "APR uses phase correlation for registration.",
                "selection_reason": "provenance_segment",
            },
        },
        chat_store=None,
    )

    packet = (((rec.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    assert packet.get("primary_evidence", {}).get("block_id") == "b-7"
    assert packet.get("primary_evidence", {}).get("heading_path") == "Methods / APR"
    assert packet.get("reader_open", {}).get("blockId") == "b-7"


def test_merge_render_packet_contract_meta_prefers_shared_primary_identity_over_drifting_provenance():
    from api import chat_render

    rec = {
        "content": "Grounded answer.",
        "rendered_body": "Grounded answer.",
        "rendered_content": "Grounded answer.",
        "copy_markdown": "Grounded answer.",
        "copy_text": "Grounded answer.",
        "notice": "",
        "cite_details": [],
        "meta": {
            "paper_guide_contracts": {
                "primary_evidence": {
                    "source_path": "oe.md",
                    "source_name": "OE-2017.pdf",
                    "block_id": "b-22",
                    "anchor_id": "a-22",
                    "heading_path": "2. Comparison / 2.2 Basis patterns generation",
                    "snippet": "Fourier basis patterns are strictly periodical.",
                },
                "render_packet": {},
            }
        },
    }

    chat_render._merge_render_packet_contract_meta(
        rec=rec,
        msg_id=3,
        enriched_provenance={
            "segments": [
                {
                    "segment_id": "seg-1",
                    "locate_policy": "required",
                    "locate_target": {
                        "segmentId": "seg-1",
                        "headingPath": "2. Comparison / 2.2 Basis patterns generation",
                        "blockId": "b-22",
                    },
                    "reader_open": {
                        "sourcePath": "oe.md",
                        "headingPath": "2. Comparison / 2.2 Basis patterns generation",
                        "blockId": "b-22",
                    },
                }
            ],
            "primary_evidence": {
                "source_path": "natphoton.md",
                "source_name": "NatPhoton-2019.pdf",
                "block_id": "b-nat",
                "anchor_id": "a-nat",
                "heading_path": "Abstract / Acquisition and image reconstruction strategies.",
                "snippet": "A broader overview paragraph.",
                "selection_reason": "provenance_segment",
            },
        },
        chat_store=None,
    )

    packet = (((rec.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    assert packet.get("primary_evidence", {}).get("source_name") == "OE-2017.pdf"
    assert packet.get("primary_evidence", {}).get("block_id") == "b-22"
    assert packet.get("primary_evidence", {}).get("heading_path") == "2. Comparison / 2.2 Basis patterns generation"


def test_merge_render_packet_contract_meta_refreshes_contract_primary_from_refs_pack():
    from api import chat_render

    rec = {
        "content": "Grounded answer.",
        "rendered_body": "Grounded answer.",
        "rendered_content": "Grounded answer.",
        "copy_markdown": "Grounded answer.",
        "copy_text": "Grounded answer.",
        "notice": "",
        "cite_details": [],
        "meta": {
            "paper_guide_contracts": {
                "primary_evidence": {
                    "source_path": "sciadv.md",
                    "source_name": "SciAdv-2017.pdf",
                    "heading_path": "INTRODUCTION",
                    "snippet": "A broad answer-hit snippet.",
                    "selection_reason": "answer_hit_top",
                },
                "render_packet": {},
            }
        },
    }

    chat_render._merge_render_packet_contract_meta(
        rec=rec,
        msg_id=4,
        enriched_provenance={"segments": []},
        ref_pack={
            "primary_evidence": {
                "source_path": "sciadv.md",
                "source_name": "SciAdv-2017.pdf",
                "block_id": "blk_30",
                "anchor_id": "a_30",
                "heading_path": "INTRODUCTION / Spatially variant digital supersampling",
                "snippet": "dynamic supersampling is defined here.",
                "selection_reason": "prompt_aligned",
            }
        },
        chat_store=None,
    )

    contracts = ((rec.get("meta") or {}).get("paper_guide_contracts") or {})
    packet = contracts.get("render_packet") or {}
    assert (contracts.get("primary_evidence") or {}).get("block_id") == "blk_30"
    assert (contracts.get("primary_evidence") or {}).get("heading_path") == "INTRODUCTION / Spatially variant digital supersampling"
    assert (packet.get("primary_evidence") or {}).get("block_id") == "blk_30"


def test_merge_render_packet_contract_meta_backfills_system_a_card_from_ref_primary_evidence():
    from api import chat_render

    rec = {
        "content": "Light-field microscopy solves the depth-of-field trade-off [1].",
        "rendered_body": "Light-field microscopy solves the depth-of-field trade-off [1](#kb-cite-demo-1).",
        "rendered_content": "Light-field microscopy solves the depth-of-field trade-off [1](#kb-cite-demo-1).",
        "copy_markdown": "Light-field microscopy solves the depth-of-field trade-off [1].",
        "copy_text": "Light-field microscopy solves the depth-of-field trade-off [1].",
        "notice": "",
        "cite_details": [
            {
                "num": 1,
                "anchor": "kb-cite-demo-1",
                "source_path": "db/qclfm/qclfm.en.md",
                "source_name": "QCLFM.pdf",
                "citation_route": "system_a",
                "is_inpaper": False,
                "heading_path": "I. INTRODUCTION",
                "answer_claim": "Light-field microscopy solves the depth-of-field trade-off.",
                "evidence_quote": (
                    "# Quantum correlation light-field microscope with extreme depth of field\n"
                    "Yingwen Zhang,$^{1,2,*}$ Duncan England"
                ),
                "raw": (
                    "# Quantum correlation light-field microscope with extreme depth of field\n"
                    "Yingwen Zhang,$^{1,2,*}$ Duncan England"
                ),
                "card_quality_flags": ["evidence_quote_filtered", "missing_evidence_quote"],
            }
        ],
        "meta": {"paper_guide_contracts": {"render_packet": {}}},
    }

    chat_render._merge_render_packet_contract_meta(
        rec=rec,
        msg_id=6,
        enriched_provenance={"segments": []},
        ref_pack={
            "hits": [
                {
                    "text": "rough title text",
                    "meta": {"source_path": "db/qclfm/qclfm.en.md"},
                    "ui_meta": {
                        "primary_evidence": {
                            "source_path": "db/qclfm/qclfm.en.md",
                            "source_name": "QCLFM.pdf",
                            "block_id": "blk_light_field",
                            "anchor_id": "p_light_field",
                            "heading_path": "I. INTRODUCTION / Light-field microscopy",
                            "snippet": (
                                "Light-field microscopy is a 3D microscopy technique whereby volumetric "
                                "information of a sample is gained in a single shot."
                            ),
                            "highlight_snippet": (
                                "Light-field microscopy is a 3D microscopy technique whereby volumetric "
                                "information of a sample is gained in a single shot."
                            ),
                            "anchor_kind": "paragraph",
                            "selection_reason": "prompt_aligned",
                        }
                    },
                }
            ]
        },
        chat_store=None,
    )

    packet = (((rec.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    details = packet.get("cite_details") or []
    assert len(details) == 1
    detail = details[0]
    assert detail["block_id"] == "blk_light_field"
    assert detail["anchor_id"] == "p_light_field"
    assert detail["heading_path"] == "I. INTRODUCTION / Light-field microscopy"
    assert "volumetric information" in detail["card_evidence"]
    assert "Yingwen Zhang" not in detail["card_evidence"]
    assert "##" not in detail["card_evidence"]


def test_chat_messages_merge_cached_reference_payload_prefers_enriched_hits():
    from api.routers import chat, references

    raw_refs = {101: {"prompt": "raw prompt", "hits": [{"text": "raw only"}]}}
    references._store_cached_conversation_refs_payload(
        conv_id="conv-cached-refs",
        signature="conversation-sig",
        refs=raw_refs,
        payload={
            101: {
                "prompt": "cached prompt",
                "hits": [
                    {
                        "text": "enriched",
                        "ui_meta": {
                            "primary_evidence": {
                                "snippet": "Precise cached evidence.",
                                "block_id": "blk_cached",
                            }
                        },
                    }
                ],
            }
        },
    )

    merged = chat._merge_cached_reference_render_payload(
        "conv-cached-refs",
        raw_refs,
    )

    assert merged[101]["prompt"] == "raw prompt"
    assert merged[101]["hits"][0]["text"] == "raw only"
    assert merged[101]["rendered_payload"]["prompt"] == "cached prompt"
    assert merged[101]["rendered_payload"]["hits"][0]["text"] == "enriched"
    assert merged[101]["rendered_payload"]["hits"][0]["ui_meta"]["primary_evidence"]["block_id"] == "blk_cached"


def test_chat_messages_ignore_stale_cached_refs_and_rerender_current_primary_evidence():
    from api.chat_render import enrich_messages_with_reference_render
    from api.routers import chat, references

    source_path = "db/paper/current.en.md"
    old_refs = {
        101: {
            "prompt": "compare the evidence",
            "prompt_sig": "prompt-sig",
            "updated_at": 10.0,
            "hits": [
                {
                    "text": "Old result evidence.",
                    "meta": {"source_path": source_path, "heading_path": "Old results"},
                }
            ],
        }
    }
    references._store_cached_conversation_refs_payload(
        conv_id="conv-stale-cached-refs",
        signature="old-conversation-sig",
        refs=old_refs,
        payload={
            101: {
                "hits": [
                    {
                        "text": "Old result evidence.",
                        "meta": {"source_path": source_path, "heading_path": "Old results"},
                        "ui_meta": {
                            "primary_evidence": {
                                "source_path": source_path,
                                "heading_path": "Old results",
                                "snippet": "Old result evidence.",
                                "block_id": "blk_old",
                                "anchor_id": "p_old",
                            }
                        },
                    }
                ]
            }
        },
    )
    current_primary = {
        "source_path": source_path,
        "source_name": "Current Paper.pdf",
        "heading_path": "Abstract",
        "snippet": "Current primary evidence supports the answer.",
        "highlight_snippet": "Current primary evidence supports the answer.",
        "block_id": "blk_current",
        "anchor_id": "p_current",
        "anchor_kind": "paragraph",
        "strict_locate": True,
    }
    current_refs = {
        101: {
            "prompt": "compare the evidence",
            "prompt_sig": "prompt-sig",
            "updated_at": 20.0,
            "hits": [
                {
                    "text": current_primary["snippet"],
                    "meta": {
                        "source_path": source_path,
                        "source_name": current_primary["source_name"],
                        "heading_path": current_primary["heading_path"],
                    },
                    "ui_meta": {"primary_evidence": current_primary},
                }
            ],
            "primary_evidence": current_primary,
        }
    }

    merged = chat._merge_cached_reference_render_payload(
        "conv-stale-cached-refs",
        current_refs,
    )

    assert "rendered_payload" not in merged[101]
    rendered = enrich_messages_with_reference_render(
        [
            {"id": 101, "role": "user", "content": "compare the evidence"},
            {
                "id": 102,
                "role": "assistant",
                "content": "The current evidence supports this claim [1].",
                "meta": {"canonical_hit_paths": [source_path]},
            },
        ],
        merged,
        conv_id="conv-stale-cached-refs",
    )[-1]
    details = list(rendered.get("cite_details") or [])
    assert len(details) == 1
    assert details[0]["block_id"] == "blk_current"
    assert details[0]["heading_path"] == "Abstract"
    assert "Current primary evidence" in details[0]["evidence_quote"]


def test_chat_messages_do_not_overlay_unverifiable_authoritative_doc_list_cache():
    from api.routers import chat, references

    raw_refs = {
        101: {
            "prompt": "compare selected papers",
            "prompt_sig": "same-prompt",
            "updated_at": 10.0,
            "hits": [{"text": "selected", "meta": {"source_path": "selected.en.md"}}],
        }
    }
    references._store_cached_conversation_refs_payload(
        conv_id="conv-authoritative-stale",
        signature="cached-conversation",
        refs=raw_refs,
        payload={
            101: {
                "hits": [
                    {"text": "selected", "meta": {"source_path": "selected.en.md"}},
                    {"text": "stale extra", "meta": {"source_path": "extra.en.md"}},
                ],
                "pipeline_debug": {"doc_list_authoritative": True},
            }
        },
    )

    merged = chat._merge_cached_reference_render_payload(
        "conv-authoritative-stale",
        raw_refs,
    )

    assert "rendered_payload" not in merged[101]
    assert [hit["meta"]["source_path"] for hit in merged[101]["hits"]] == [
        "selected.en.md"
    ]


def test_effective_reference_pack_keeps_raw_hit_order_and_exposes_enriched_hits():
    from api.chat_render import _effective_reference_render_pack

    pack = {
        "hits": [{"text": "raw generation hit", "meta": {"source_path": "raw.md"}}],
        "rendered_payload": {
            "hits": [
                {
                    "text": "enriched reference hit",
                    "ui_meta": {"primary_evidence": {"snippet": "precise"}},
                }
            ]
        },
    }

    effective = _effective_reference_render_pack(pack)

    assert effective["hits"][0]["text"] == "raw generation hit"
    assert effective["enriched_hits"][0]["text"] == "enriched reference hit"


def test_effective_reference_pack_prefers_authoritative_doc_list_hits():
    from api.chat_render import _effective_reference_render_pack

    pack = {
        "hits": [{"text": "raw generation hit", "meta": {"source_path": "paper.md"}}],
        "rendered_payload": {
            "hits": [
                {
                    "text": "authoritative card hit",
                    "meta": {"source_path": "paper.md"},
                    "ui_meta": {"citation_meta": {"doi": "10.1000/example", "journal_if": 12.3}},
                }
            ],
            "pipeline_debug": {"doc_list_authoritative": True},
        },
    }

    effective = _effective_reference_render_pack(pack)

    assert effective["hits"][0]["text"] == "authoritative card hit"
    assert effective["hits"][0]["ui_meta"]["citation_meta"]["doi"] == "10.1000/example"
    assert effective["retrieval_hits"][0]["text"] == "raw generation hit"
    assert "enriched_hits" not in effective


def test_effective_reference_pack_keeps_newer_top_level_authoritative_hits():
    from api.chat_render import _effective_reference_render_pack

    pack = {
        "hits": [
            {
                "text": "new authoritative hit",
                "ui_meta": {"citation_meta": {"doi": "10.1000/new"}},
            }
        ],
        "pipeline_debug": {"doc_list_authoritative": True},
        "rendered_payload": {
            "hits": [{"text": "stale nested hit", "ui_meta": {"citation_meta": {}}}],
            "pipeline_debug": {"doc_list_authoritative": True},
        },
    }

    effective = _effective_reference_render_pack(pack)

    assert effective["hits"][0]["text"] == "new authoritative hit"
    assert effective["hits"][0]["ui_meta"]["citation_meta"]["doi"] == "10.1000/new"


def test_enrich_messages_does_not_mutate_reference_pack() -> None:
    from api.chat_render import enrich_messages_with_reference_render

    refs_by_user = {
        1: {
            "hits": [{"text": "new hit", "meta": {"source_path": "new.md"}}],
            "pipeline_debug": {"doc_list_authoritative": True},
            "rendered_payload": {
                "hits": [{"text": "stale hit", "meta": {"source_path": "stale.md"}}],
                "pipeline_debug": {"doc_list_authoritative": True},
            },
        }
    }

    enrich_messages_with_reference_render(
        [
            {"id": 1, "role": "user", "content": "question"},
            {"id": 2, "role": "assistant", "content": "answer"},
        ],
        refs_by_user,
        conv_id="conv-no-ref-mutation",
    )

    assert refs_by_user[1]["hits"][0]["text"] == "new hit"
    assert refs_by_user[1]["rendered_payload"]["hits"][0]["text"] == "stale hit"


def test_answer_aligned_pack_primary_replaces_stale_precise_system_a_detail():
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    source_path = "scinerf.en.md"
    details = [
        {
            "num": 1,
            "anchor": "cite-a",
            "source_path": source_path,
            "source_name": "SCINeRF.pdf",
            "citation_route": "system_a",
            "is_inpaper": False,
            "heading_path": "3. Method / 3.3. Proposed Framework",
            "title": "3. Method / 3.3. Proposed Framework",
            "block_id": "blk_method",
            "anchor_id": "p_method",
            "anchor_kind": "paragraph",
            "evidence_quote": "The camera poses cannot be estimated directly.",
            "summary_line": "The camera poses cannot be estimated directly.",
            "answer_claim": "ADMM is prior work, not an original contribution.",
        }
    ]
    pack = {
        "primary_evidence": {
            "source_path": source_path,
            "source_name": "SCINeRF.pdf",
            "heading_path": "2. Related Work",
            "block_id": "blk_related",
            "anchor_id": "p_related",
            "anchor_kind": "paragraph",
            "snippet": "Most existing methods employ ADMM [4].",
            "highlight_snippet": "Most existing methods employ ADMM [4].",
            "selection_reason": "answer_aligned_block",
            "strict_locate": True,
        }
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, pack, render_locale="en")

    assert out[0]["heading_path"] == "2. Related Work"
    assert out[0]["block_id"] == "blk_related"
    assert out[0]["anchor_id"] == "p_related"
    assert "existing methods employ ADMM" in out[0]["evidence_quote"]


def test_answer_aligned_primary_matches_same_path_across_different_display_names():
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    source_path = "F:/db/SCINeRF/SCINeRF.en.md"
    details = [
        {
            "num": 1,
            "anchor": "cite-a",
            "source_path": source_path,
            "source_name": "SCINeRF.pdf",
            "citation_route": "system_a",
            "is_inpaper": False,
            "heading_path": "3. Method",
            "block_id": "blk_method",
            "anchor_id": "p_method",
            "evidence_quote": "A stale method excerpt.",
            "answer_claim": "ADMM is prior work, not an original contribution.",
        }
    ]
    pack = {
        "primary_evidence": {
            "source_path": source_path,
            "source_name": "2024 IEEE CVPR - SCINeRF.pdf",
            "heading_path": "2. Related Work",
            "block_id": "blk_related",
            "anchor_id": "p_related",
            "snippet": "Most existing methods employ ADMM [4].",
            "selection_reason": "answer_aligned_block",
            "strict_locate": True,
        }
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, pack, render_locale="en")

    assert out[0]["heading_path"] == "2. Related Work"
    assert out[0]["block_id"] == "blk_related"


def test_system_a_primary_backfill_selects_claim_aligned_abstracts_without_relabeling_results(
    tmp_path: Path,
):
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    scigs_path = tmp_path / "scigs.en.md"
    scigs_path.write_text(
        "# SCIGS\n\n"
        "## Abstract\n\n"
        "The proposed SCIGS is the first to reconstruct a 3D explicit scene from a "
        "single compressed image, extending its application to dynamic 3D scenes.\n\n"
        "## 4.2 Result and Analysis\n\n"
        "The proposed method is evaluated on static datasets.\n",
        encoding="utf-8",
    )
    scinerf_path = tmp_path / "scinerf.en.md"
    scinerf_path.write_text(
        "# SCINeRF\n\n"
        "## Abstract\n\n"
        "Specifically, we formulate the physical imaging process of SCI as part of "
        "the training of NeRF, allowing recovery of complex scene structures.\n\n"
        "## 5. Conclusion\n\n"
        "SCINeRF exploits neural radiance fields as its scene representation.\n",
        encoding="utf-8",
    )
    details = [
        {
            "num": 1,
            "source_path": str(scigs_path),
            "source_name": "SCIGS.pdf",
            "citation_route": "system_a",
            "citation_plan_slot": True,
            "answer_claim": "SCIGS 从单张压缩图像恢复动态 3D 场景表示。",
            "evidence_quote": "Title: SCIGS",
        },
        {
            "num": 2,
            "source_path": str(scinerf_path),
            "source_name": "SCINeRF.pdf",
            "citation_route": "system_a",
            "citation_plan_slot": True,
            "answer_claim": "SCINeRF 基于 NeRF 隐式表示。",
            "evidence_quote": "Title: SCINeRF",
        },
        {
            "num": 3,
            "source_path": str(scigs_path),
            "source_name": "SCIGS.pdf",
            "citation_route": "system_a",
            "answer_claim": "SCIGS 在静态数据集上的性能超过多种方法。",
            "heading_path": "4.2 Result and Analysis",
            "evidence_quote": "The proposed method is evaluated on static datasets.",
            "block_id": "blk_results",
            "anchor_id": "p_results",
        },
        {
            "num": 4,
            "source_path": str(scinerf_path),
            "source_name": "SCINeRF.pdf",
            "citation_route": "system_a",
            "citation_plan_slot": True,
            "answer_claim": "SCINeRF jointly optimizes NeRF parameters and camera poses.",
            "heading_path": "3. Method / 3.3 Proposed Framework",
            "evidence_quote": "The camera poses and NeRF parameters are jointly optimized.",
            "block_id": "blk_camera_pose",
            "anchor_id": "p_camera_pose",
        },
    ]
    pack = {
        "hits": [
            {
                "text": "The proposed method is evaluated on static datasets.",
                "meta": {"source_path": str(scigs_path), "source_name": "SCIGS.pdf"},
                "ui_meta": {
                    "primary_evidence": {
                        "source_path": str(scigs_path),
                        "source_name": "SCIGS.pdf",
                        "heading_path": "4.2 Result and Analysis",
                        "snippet": "The proposed method is evaluated on static datasets.",
                    }
                },
            },
            {
                "text": "SCINeRF exploits neural radiance fields as its scene representation.",
                "meta": {"source_path": str(scinerf_path), "source_name": "SCINeRF.pdf"},
                "ui_meta": {
                    "primary_evidence": {
                        "source_path": str(scinerf_path),
                        "source_name": "SCINeRF.pdf",
                        "heading_path": "5. Conclusion",
                        "snippet": "SCINeRF exploits neural radiance fields as its scene representation.",
                    }
                },
            },
        ]
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, pack, render_locale="en")

    scigs_abstract = next(
        detail
        for detail in out
        if Path(detail["source_path"]) == scigs_path and "Abstract" in detail["heading_path"]
    )
    scinerf_abstract = next(
        detail
        for detail in out
        if Path(detail["source_path"]) == scinerf_path and "Abstract" in detail["heading_path"]
    )
    scigs_results = next(
        detail
        for detail in out
        if Path(detail["source_path"]) == scigs_path
        and detail["heading_path"] == "4.2 Result and Analysis"
    )
    scinerf_camera_pose = next(
        detail
        for detail in out
        if Path(detail["source_path"]) == scinerf_path
        and detail["heading_path"] == "3. Method / 3.3 Proposed Framework"
    )
    assert "dynamic" in scigs_abstract["evidence_quote"]
    assert "3D" in scigs_abstract["evidence_quote"]
    assert scigs_abstract["block_id"] and scigs_abstract["anchor_id"]
    assert "physical imaging process" in scinerf_abstract["evidence_quote"]
    assert "NeRF" in scinerf_abstract["evidence_quote"]
    assert scinerf_abstract["block_id"] and scinerf_abstract["anchor_id"]
    assert scigs_results["evidence_quote"] == "The proposed method is evaluated on static datasets."
    assert scinerf_camera_pose["block_id"] == "blk_camera_pose"
    assert scinerf_camera_pose["anchor_id"] == "p_camera_pose"
    assert scinerf_camera_pose["evidence_quote"] == (
        "The camera poses and NeRF parameters are jointly optimized."
    )


def test_system_a_primary_backfill_selects_direct_s2ism_capability_evidence(
    tmp_path: Path,
):
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    source_path = tmp_path / "s2ism.en.md"
    source_path.write_text(
        "# Structured detection for simultaneous super-resolution and optical sectioning\n\n"
        "## Abstract\n\n"
        "From single-plane acquisition, we reconstruct an image with digital and optical "
        "super-resolution, high signal-to-noise ratio and enhanced optical sectioning.\n\n"
        "## Introduction\n\n"
        "Since super-resolution and optical sectioning are achieved simultaneously, "
        "we named our technique s$^2$ISM (super-resolution sectioning ISM).\n\n"
        "## Results\n\n"
        "More specifically, s2ISM can be applied to any LSM equipped with a detector array.\n",
        encoding="utf-8",
    )
    details = [
        {
            "num": 3,
            "source_path": str(source_path),
            "source_name": "NatPhoton s2ISM.pdf",
            "citation_route": "system_a",
            "citation_plan_slot": True,
            "answer_claim": "s2ISM 能够同时实现超分辨和光学切片。",
            "heading_path": "Results / Versatility of s2ISM",
            "evidence_quote": (
                "More specifically, s2ISM can be applied to any LSM equipped with a detector array."
            ),
            "block_id": "blk_weak",
            "anchor_id": "p_weak",
        }
    ]
    pack = {
        "hits": [
            {
                "text": "More specifically, s2ISM can be applied to any LSM equipped with a detector array.",
                "meta": {"source_path": str(source_path), "source_name": "NatPhoton s2ISM.pdf"},
            }
        ]
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, pack, render_locale="zh")

    assert len(out) == 1
    assert out[0]["heading_path"].endswith("Abstract")
    assert "digital and optical super-resolution" in out[0]["evidence_quote"]
    assert "enhanced optical sectioning" in out[0]["evidence_quote"]
    assert out[0]["block_id"] != "blk_weak"
    assert out[0]["anchor_id"] != "p_weak"


def test_system_a_primary_backfill_sets_scigs_dynamic_relation_after_replacement(
    tmp_path: Path,
):
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    source_path = tmp_path / "scigs.en.md"
    source_path.write_text(
        "# SCIGS\n\n## Abstract\n\n"
        "The proposed SCIGS is the first to reconstruct a 3D explicit scene from a single "
        "compressed image, extending its application to dynamic 3D scenes.\n",
        encoding="utf-8",
    )
    details = [
        {
            "num": 4,
            "source_path": str(source_path),
            "source_name": "SCIGS.pdf",
            "citation_route": "system_a",
            "citation_plan_slot": True,
            "answer_claim": "SCIGS 从单张压缩图像重建动态 3D 场景。",
            "evidence_quote": "Title: SCIGS: 3D Gaussians Splatting from a Snapshot Compressive Image",
        }
    ]
    pack = {
        "hits": [
            {
                "text": "The method is evaluated on static datasets.",
                "meta": {"source_path": str(source_path), "source_name": "SCIGS.pdf"},
            }
        ]
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, pack, render_locale="zh")

    assert "dynamic 3D scenes" in out[0]["evidence_quote"]
    assert "SCIGS" in out[0]["support_relation"]
    assert "动态 3D" in out[0]["support_relation"]


def test_system_a_primary_relations_do_not_rewrite_unrelated_or_risk_claims():
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    scigs_path = "F:/library/scigs.en.md"
    dl_spi_path = "F:/library/dl-spi-review.en.md"
    details = [
        {
            "num": 1,
            "source_path": scigs_path,
            "source_name": "SCIGS.pdf",
            "citation_route": "system_a",
            "citation_plan_slot": True,
            "answer_claim": "SCIGS 的静态数据集评测设置仍需要进一步核对。",
            "evidence_quote": "The method is evaluated on static datasets.",
            "support_relation": "保留静态评测说明。",
        },
        {
            "num": 2,
            "source_path": dl_spi_path,
            "source_name": "Deep-learning SPI review.pdf",
            "citation_route": "system_a",
            "citation_plan_slot": True,
            "answer_claim": "深度学习单像素成像的风险是训练时间长且泛化能力有限。",
            "evidence_quote": "Data-driven methods have prolonged training and limited generalization.",
            "support_relation": "保留训练与泛化风险说明。",
        },
        {
            "num": 3,
            "source_path": dl_spi_path,
            "source_name": "Deep-learning SPI review.pdf",
            "citation_route": "system_a",
            "citation_plan_slot": True,
            "answer_claim": "深度学习单像素成像能同时提高重建质量和重建速度。",
            "evidence_quote": "Deep learning provides exceptional reconstruction quality and reconstruction speed.",
        },
    ]
    pack = {
        "hits": [
            {
                "text": "The method is evaluated on static datasets.",
                "meta": {"source_path": scigs_path, "source_name": "SCIGS.pdf"},
                "ui_meta": {
                    "primary_evidence": {
                        "heading_path": "Abstract",
                        "snippet": (
                            "SCIGS reconstructs a 3D explicit scene from one compressed image and "
                            "extends its application to dynamic 3D scenes."
                        ),
                    }
                },
            },
            {
                "text": "Data-driven methods have prolonged training and limited generalization.",
                "meta": {
                    "source_path": dl_spi_path,
                    "source_name": "Deep-learning SPI review.pdf",
                },
                "ui_meta": {
                    "primary_evidence": {
                        "heading_path": "Abstract",
                        "snippet": (
                            "Deep learning provides exceptional reconstruction quality and "
                            "reconstruction speed."
                        ),
                    }
                },
            },
        ]
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, pack, render_locale="zh")

    assert out[0]["support_relation"] == "保留静态评测说明。"
    assert out[1]["support_relation"] == "保留训练与泛化风险说明。"
    assert "重建质量" in out[2]["support_relation"]
    assert "重建速度" in out[2]["support_relation"]


def test_abstract_primary_evidence_refreshes_after_markdown_repair(tmp_path: Path):
    from api.chat_render import _abstract_primary_evidence_from_source

    source_path = tmp_path / "paper.en.md"
    source_path.write_text(
        "# Paper\n\n## Abstract\n\nThe original abstract describes static 3D scenes.\n",
        encoding="utf-8",
    )
    first = _abstract_primary_evidence_from_source(str(source_path))

    source_path.write_text(
        "# Paper\n\n## Abstract\n\n"
        "The repaired abstract now describes explicit dynamic 3D scenes in detail.\n",
        encoding="utf-8",
    )
    second = _abstract_primary_evidence_from_source(str(source_path))

    assert "original abstract" in first["snippet"]
    assert "repaired abstract" in second["snippet"]
    assert first["snippet"] != second["snippet"]


def test_system_a_primary_backfill_does_not_relabel_repeated_citation_claim(tmp_path: Path):
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    scigs_path = tmp_path / "scigs.en.md"
    scigs_path.write_text(
        "# SCIGS\n\n"
        "## Abstract\n\n"
        "The proposed SCIGS is the first to reconstruct a 3D explicit scene from a "
        "single compressed image, extending its application to dynamic 3D scenes.\n\n"
        "## 4.2 Result and Analysis\n\n"
        "The proposed method is evaluated on static datasets.\n",
        encoding="utf-8",
    )
    details = [
        {
            "num": 4,
            "source_path": str(scigs_path),
            "source_name": "SCIGS.pdf",
            "citation_route": "system_a",
            "answer_claim": "SCIGS performs well on several static datasets.",
            "heading_path": "4.2 Result and Analysis",
            "evidence_quote": "The proposed method is evaluated on static datasets.",
        }
    ]
    pack = {
        "hits": [
            {
                "text": "The proposed method is evaluated on static datasets.",
                "meta": {"source_path": str(scigs_path), "source_name": "SCIGS.pdf"},
            }
        ]
    }
    answer_text = (
        "2. **Dynamic scenes**: SCIGS can reconstruct an explicit dynamic 3D scene "
        "from a snapshot compressive image [4].\n\n"
        "- SCIGS is competitive on static datasets [4]."
    )

    out = _backfill_system_a_cite_details_from_ref_pack(
        details,
        pack,
        render_locale="en",
        answer_text=answer_text,
    )

    assert out[0]["answer_claim"] == "SCIGS performs well on several static datasets."
    assert out[0]["heading_path"] == "4.2 Result and Analysis"
    assert out[0]["evidence_quote"] == "The proposed method is evaluated on static datasets."


def test_system_a_primary_backfill_describes_quantitative_measurement_support():
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    source_path = "hsi-fsi.en.md"
    details = [
        {
            "num": 1,
            "source_path": source_path,
            "source_name": "Hadamard versus Fourier.pdf",
            "citation_route": "system_a",
            "answer_claim": "Hadamard 和 Fourier 的选择取决于实验目标。",
        }
    ]
    pack = {
        "primary_evidence": {
            "source_path": source_path,
            "source_name": "Hadamard versus Fourier.pdf",
            "heading_path": "3. Comparison of experiment / 3.1 Numerical simulations",
            "block_id": "blk_compare",
            "anchor_id": "p_compare",
            "snippet": (
                "The sampling ratio increases across experiments. "
                "PSNR and SSIM show that FSI converges faster than HSI."
            ),
            "selection_reason": "section_intent_rescue",
            "strict_locate": True,
        }
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, pack, render_locale="zh")

    assert "测量指标" in out[0]["support_relation"]
    assert "采样率" in out[0]["support_relation"]
    assert "PSNR" in out[0]["support_relation"]
    assert "SSIM" in out[0]["support_relation"]


def test_system_a_primary_backfill_describes_device_scope_boundary():
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    source_path = "perovskite-laser.en.md"
    details = [
        {
            "num": 1,
            "source_path": source_path,
            "source_name": "Perovskite laser.pdf",
            "citation_route": "system_a",
            "answer_claim": "这是一篇器件论文，与单像素成像几乎没有交集。",
        }
    ]
    pack = {
        "primary_evidence": {
            "source_path": source_path,
            "source_name": "Perovskite laser.pdf",
            "heading_path": "Abstract",
            "snippet": "We demonstrate lasing from an electrically driven dual-cavity perovskite device.",
            "selection_reason": "prompt_aligned",
        }
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, pack, render_locale="zh")

    assert "perovskite" in out[0]["support_relation"]
    assert "器件" in out[0]["support_relation"]
    assert "不是" in out[0]["support_relation"]


def test_reading_guide_repair_adds_missing_system_a_source_to_matching_paragraph():
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    answer = (
        "建议按以下顺序阅读：\n\n"
        "1. **先读探测器综述**：快速了解单光子探测器、SPAD、暗计数和死时间。\n\n"
        "2. **再读 Physics-informed deep learning 论文**：看它如何建立 SPAD 噪声模型 [1]。"
    )
    hits = [
        {
            "text": "High-resolution single-photon imaging with physics-informed deep learning.",
            "meta": {"source_path": "pidl.md"},
        },
        {
            "text": "Emerging single-photon detection technique for high-performance photodetector.",
            "meta": {"source_path": "spd-review.md"},
        },
    ]
    plan = {
        "slots": [
            {"preferred_system": "system_a", "candidate_hits": [1], "source_name": "physics-informed deep learning"},
            {"preferred_system": "system_a", "candidate_hits": [2], "source_name": "single-photon detection photodetector review"},
        ]
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
    )

    assert "死时间 [2]。" in repaired
    assert "噪声模型 [1]" in repaired


def test_reading_guide_budget_counts_only_bound_comparison_citations():
    source_path = "hsi-fsi.en.md"
    comparison_heading = (
        "Hadamard single-pixel imaging versus Fourier single-pixel imaging / "
        "3. Comparison of experiment / 3.1 Numerical simulations"
    )
    comparison_evidence = (
        "The coefficients are corrected gradually as the sampling ratio increases. "
        "As indicated by the curves of PSNR, SSIM, and RMSE, the convergence of HSI "
        "is lower than that of FSI."
    )
    otf_heading = (
        "Hadamard single-pixel imaging versus Fourier single-pixel imaging / "
        "2. Comparison of theory / 2.4 Efficiency"
    )
    otf_evidence = (
        "The optical transfer function (OTF), defined as the Fourier transform of "
        "the point spread function, shows how different spatial frequencies are "
        "handled by the system and explains the practical efficiency tradeoff "
        "between Hadamard and Fourier imaging."
    )
    answer = (
        "## 核心对比\n\n"
        "Hadamard 全采样需要 $2N^2$ 次测量，Fourier 需要 $4N^2$ 次；"
        "实验还在不同采样率下比较了 PSNR 与 SSIM。\n\n"
        "## 实用建议\n\n"
        "追求速度时选 Hadamard。需要分析系统的 OTF 和空间频率响应时选 Fourier。"
    )
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "2. Comparison of theory / 2.1 Principle of HSI and FSI",
                "evidence_quote": "Computational ghost imaging uses a bucket detector.",
                "candidate_hits": [],
            },
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "2. Comparison of theory / 2.1 Principle of HSI and FSI",
                "evidence_quote": "The image is reconstructed by applying an inverse transform.",
                "candidate_hits": [],
            },
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "2. Comparison of theory / 2.4 Efficiency",
                "evidence_quote": (
                    "The optical transfer function (OTF), defined as the Fourier transform "
                    "of the point spread function, shows how different spatial frequencies "
                    "are handled by the system."
                ),
                "candidate_hits": [],
            },
        ],
    }
    primary_evidence = {
        "source_path": source_path,
        "source_name": "Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
        "heading_path": comparison_heading,
        "snippet": comparison_evidence,
        "highlight_snippet": comparison_evidence,
        "block_id": "blk_comparison",
        "anchor_id": "p_comparison",
        "anchor_kind": "paragraph",
        "strict_locate": True,
    }
    messages = [
        {"id": 1, "role": "user", "content": "Hadamard 和 Fourier 到底该怎么选？"},
        {
            "id": 2,
            "role": "assistant",
            "content": answer,
            "meta": {
                # Reserve the model's full citation-number range. The comparison
                # rescue must use a new exact-evidence hit, not alias canonical [1].
                "canonical_hit_paths": [source_path] * 6,
                "answer_quality": {
                    "output_mode": "reading_guide",
                    "prompt_family": "compare",
                    "citation_plan": plan,
                },
                "paper_guide_contracts": {"citation_plan": plan},
            },
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    # This mirrors the fully enriched shape: the general hit surface
                    # favors the fluent OTF passage, while primary_evidence carries
                    # the strict quantitative Comparison/3.1 locate target.
                    "text": otf_evidence,
                    "meta": {
                        "source_path": source_path,
                        "source_name": primary_evidence["source_name"],
                        "heading_path": otf_heading,
                    },
                    "ui_meta": {
                        "display_name": primary_evidence["source_name"],
                        "heading_path": otf_heading,
                        "summary_line": "OTF and spatial-frequency efficiency comparison.",
                        "primary_evidence": primary_evidence,
                    },
                }
            ],
            "primary_evidence": primary_evidence,
        }
    }

    rendered = enrich_messages_with_reference_render(
        messages,
        refs_by_user,
        conv_id="conv-hadamard-fourier",
    )[-1]

    assert rendered["content"] == answer
    assert "#kb-cite-" in rendered["rendered_content"]
    details = rendered["cite_details"]
    assert len(details) == 2
    assert all(detail["citation_route"] == "system_a" for detail in details)
    assert all("Comparison" in detail["heading_path"] for detail in details)
    comparison_detail = next(
        detail
        for detail in details
        if "PSNR" in detail["evidence_quote"] and "SSIM" in detail["evidence_quote"]
    )
    assert comparison_detail["num"] > 6
    assert comparison_detail["citation_plan_slot"] is True
    assert comparison_detail["block_id"] == "blk_comparison"
    assert comparison_detail["anchor_id"] == "p_comparison"
    assert any(term in comparison_detail["answer_claim"] for term in ("测量", "采样"))


def test_comparison_rescue_adds_grounded_bridge_when_model_omits_all_citations():
    source_path = "hsi-fsi.en.md"
    heading = (
        "Hadamard single-pixel imaging versus Fourier single-pixel imaging / "
        "3. Comparison of experiment / 3.1 Numerical simulations"
    )
    evidence = (
        "The coefficients are corrected gradually as the sampling ratio increases. "
        "As indicated by the curves of PSNR, SSIM, and RMSE, the convergence of HSI "
        "is lower than that of FSI."
    )
    answer = (
        "## 核心结论\n\n"
        "追求速度时选 Hadamard；追求物理可解释性时选 Fourier。\n\n"
        "## 证据支撑的权衡\n\n"
        "Hadamard 的二值图案更适合高速 DMD。Fourier 更适合分析空间频率响应。\n\n"
        "## 一句话建议\n\n"
        "快速采集选 Hadamard，需要 OTF 解释时选 Fourier。"
    )
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
                "heading_path": "2. Comparison of theory",
                "evidence_quote": "A theoretical comparison of the two methods.",
                "candidate_hits": [],
            }
        ],
    }
    primary_evidence = {
        "source_path": source_path,
        "source_name": "Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
        "heading_path": heading,
        "snippet": evidence,
        "highlight_snippet": evidence,
        "block_id": "blk_comparison",
        "anchor_id": "p_comparison",
        "anchor_kind": "paragraph",
        "strict_locate": True,
    }
    messages = [
        {"id": 1, "role": "user", "content": "Hadamard 和 Fourier 到底该怎么选？"},
        {
            "id": 2,
            "role": "assistant",
            "content": answer,
            "meta": {
                "canonical_hit_paths": [source_path] * 6,
                "answer_quality": {
                    "output_mode": "reading_guide",
                    "prompt_family": "compare",
                    "citation_plan": plan,
                },
                "paper_guide_contracts": {"citation_plan": plan},
            },
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "Fourier OTF and spatial-frequency interpretation.",
                    "meta": {"source_path": source_path},
                    "ui_meta": {"primary_evidence": primary_evidence},
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(
        messages,
        refs_by_user,
        conv_id="conv-hadamard-no-model-citations",
    )[-1]

    assert rendered["content"] == answer
    assert "定量对比依据" in rendered["rendered_content"]
    assert rendered["rendered_content"].index("定量对比依据") < rendered["rendered_content"].index("一句话建议")
    detail = next(item for item in rendered["cite_details"] if item["citation_route"] == "system_a")
    assert detail["num"] > 6
    assert detail["heading_path"] == heading
    assert "sampling ratio" in detail["answer_claim"]
    assert "PSNR" in detail["evidence_quote"]
    assert "SSIM" in detail["evidence_quote"]


def test_reading_guide_repairs_uncited_source_definition_from_abstract(tmp_path: Path):
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    scigs_path = tmp_path / "scigs.en.md"
    scigs_path.write_text(
        "# SCIGS\n\n## Abstract\n\n"
        "SCIGS reconstructs a 3D explicit scene and extends the task to dynamic 3D scenes.\n",
        encoding="utf-8",
    )
    scinerf_path = tmp_path / "scinerf.en.md"
    scinerf_path.write_text(
        "# SCINeRF\n\n## Abstract\n\n"
        "We formulate the physical imaging process of SCI as part of the training of NeRF.\n",
        encoding="utf-8",
    )
    answer = (
        "1. SCIGS can recover a dynamic 3D scene [1].\n"
        "2. Both methods reconstruct a scene [2].\n\n"
        "**Representation**: SCINeRF uses an implicit NeRF representation."
    )
    hits = [
        {"text": "SCIGS title", "meta": {"source_path": str(scigs_path)}},
        {"text": "SCINeRF title", "meta": {"source_path": str(scinerf_path)}},
    ]
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": str(scigs_path),
                "source_name": "ICIP SCIGS",
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": str(scinerf_path),
                "source_name": "CVPR SCINeRF",
                "candidate_hits": [2],
            },
        ],
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[str(scigs_path), str(scinerf_path)],
    )

    assert repaired.startswith("**Direct evidence from the abstracts:**")
    assert "SCIGS** reconstructs an explicit 3D scene" in repaired
    assert "dynamic 3D scenes [3]" in repaired
    assert "SCINeRF** incorporates the SCI physical imaging process into NeRF training [4]" in repaired
    assert "SCIGS can recover a dynamic 3D scene [1]" in repaired
    assert "SCINeRF uses an implicit NeRF representation." in repaired
    assert len(hits) == 4
    assert all(hit["meta"]["citation_plan_claim_abstract"] is True for hit in hits[2:])
    assert all("Abstract" in hit["meta"]["heading_path"] for hit in hits[2:])


def test_reading_guide_lineage_rebinds_cassi_and_scinerf_to_direct_evidence():
    from api.chat_render import _reading_guide_repair_lineage_scinerf_evidence

    hits = [
        {"text": "Generic CASSI conclusion.", "meta": {"source_path": "cassi.en.md"}},
        {"text": "Generic SCINeRF conclusion.", "meta": {"source_path": "scinerf.en.md"}},
        {"text": "SCIGS dynamic 3D scene.", "meta": {"source_path": "scigs.en.md"}},
    ]
    plan = {
        "intent": "origin_lookup",
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "cassi.en.md",
                "source_name": "CASSI dual-disperser spectral imaging",
                "heading_path": "Abstract",
                "candidate_hits": [1],
                "evidence_quote": (
                    "The system design uses two dispersive elements arranged in opposition "
                    "around a binary-valued aperture code."
                ),
            },
            {
                "preferred_system": "system_a",
                "source_path": "scinerf.en.md",
                "source_name": "SCINeRF",
                "heading_path": "Conclusion",
                "candidate_hits": [2],
                "evidence_quote": (
                    "SCINeRF learns a 3D scene representation with NeRF from a single "
                    "snapshot compressed image."
                ),
            },
        ],
    }
    answer = (
        "### 1. Dual-disperser spectral imaging\nCASSI is an early spectral system [1].\n\n"
        "### 3. Key transition\nSCINeRF uses NeRF for 3D scenes [2].\n"
        "SCIGS uses a dynamic 3D scene [3]."
    )

    repaired = _reading_guide_repair_lineage_scinerf_evidence(answer, hits, plan)

    assert "two dispersive elements around a binary-valued aperture code" in repaired
    assert "spectral projections [4]" in repaired
    assert "SCINeRF** learns a 3D scene representation" in repaired
    assert "using NeRF [5]" in repaired
    assert "[1]" not in repaired
    assert "[2]" not in repaired
    assert "SCIGS uses a dynamic 3D scene [3]" in repaired
    assert hits[3]["meta"]["citation_plan_lineage_cassi"] is True
    assert hits[4]["meta"]["citation_plan_lineage_scinerf"] is True


def test_reading_guide_keeps_only_planned_system_b_marker_within_budget():
    from api.chat_render import _reading_guide_enforce_system_b_plan_budget

    sid = "s7f6b9404"
    answer = (
        f"Background [[CITE:{sid}:5]][[CITE:{sid}:8]] and selected [[CITE:{sid}:50]].\n"
        f"The selected reference is repeated here [[CITE:{sid}:50]]."
    )
    plan = {
        "budget": {"system_a": 3, "system_b": 1},
        "slots": [
            {
                "preferred_system": "system_b",
                "candidate_refs": [50],
                "candidate_cite_examples": [f"[[CITE:{sid}:50]]"],
            }
        ],
    }

    repaired = _reading_guide_enforce_system_b_plan_budget(answer, plan)

    assert f"[[CITE:{sid}:5]]" not in repaired
    assert f"[[CITE:{sid}:8]]" not in repaired
    assert repaired.count(f"[[CITE:{sid}:50]]") == 1


def test_reading_guide_keeps_canonical_marker_when_abstract_loses_claim_alignment(tmp_path: Path):
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    scigs_path = tmp_path / "scigs.en.md"
    scigs_path.write_text(
        "# SCIGS\n\n## Abstract\n\n"
        "The proposed SCIGS is the first to reconstruct a 3D explicit scene from a single "
        "compressed image, extending its application to dynamic 3D scenes.\n",
        encoding="utf-8",
    )
    answer = (
        "SCIGS extends SCI to dynamic scenes and uses 3DGS as its explicit representation [1].\n\n"
        "SCIGS reconstructs a dynamic 3D scene from one compressed image [1]."
    )
    hits = [
        {
            "text": (
                "SCIGS reconstructs dynamic 3D scenes and uses a transformation network "
                "with pre-trained 3DGS representations."
            ),
            "meta": {"source_path": str(scigs_path), "heading_path": "5. Conclusion"},
        }
    ]
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": str(scigs_path),
                "source_name": "ICIP-2025-SCIGS-3D Gaussians Splatting",
                "candidate_hits": [1],
            }
        ],
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[str(scigs_path)],
    )

    assert repaired == answer
    assert len(hits) == 1


def test_reading_guide_replaces_weak_s2ism_marker_with_claim_aligned_abstract(
    tmp_path: Path,
):
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    source_path = tmp_path / "NatPhoton-Structured detection for simultaneous super-resolution and optical sectioning in laser scanning microscopy.en.md"
    source_path.write_text(
        "# Structured detection for simultaneous super-resolution and optical sectioning\n\n"
        "## Abstract\n\n"
        "From single-plane acquisition, we reconstruct an image with digital and optical "
        "super-resolution, high signal-to-noise ratio and enhanced optical sectioning.\n\n"
        "## Results\n\n### Versatility\n\n"
        "More specifically, the method can be applied to any LSM equipped with a detector array.\n",
        encoding="utf-8",
    )
    other_a = str(tmp_path / "iism.en.md")
    other_b = str(tmp_path / "light-field.en.md")
    answer = "s2ISM 能够同时实现超分辨和光学切片 [3]。"
    hits = [
        {"text": "iISM evidence", "meta": {"source_path": other_a}},
        {"text": "Light-field evidence", "meta": {"source_path": other_b}},
        {
            "text": "More specifically, the method can be applied to any LSM equipped with a detector array.",
            "meta": {
                "source_path": str(source_path),
                "heading_path": (
                    "Structured detection for simultaneous super-resolution and optical "
                    "sectioning in laser scanning microscopy / Results / Versatility"
                ),
            },
        },
    ]
    plan = {
        "intent": "method_explain",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": str(source_path),
                "source_name": (
                    "NatPhoton-2025-Structured detection for simultaneous super-resolution "
                    "and optical sectioning in laser scanning microscopy"
                ),
                "candidate_hits": [3],
            }
        ],
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[other_a, other_b, str(source_path)],
    )

    assert repaired == "s2ISM 能够同时实现超分辨和光学切片 [4]。"
    assert len(hits) == 4
    assert hits[3]["meta"]["citation_plan_claim_abstract"] is True
    assert hits[3]["meta"]["heading_path"].endswith("Abstract")
    assert "enhanced optical sectioning" in hits[3]["text"]


def test_reading_guide_repairs_s2ism_tradeoff_answer_and_binds_exact_abstract(
    tmp_path: Path,
):
    from api.chat_render import _reading_guide_repair_missing_system_a_citations
    from ui.refs_renderer import _annotate_inpaper_citations_with_hover_meta

    source_path = tmp_path / "NatPhoton-Structured detection in laser scanning microscopy.en.md"
    source_path.write_text(
        "# Structured detection for laser scanning microscopy\n\n"
        "## Abstract\n\n"
        "Fast detector arrays overcome the trade-off between spatial resolution and "
        "signal-to-noise ratio. However, current image scanning microscopy approaches "
        "do not provide optical sectioning and fail with thick samples unless the detector "
        "size is limited, introducing a trade-off between optical sectioning and "
        "signal-to-noise ratio.\n\n"
        "## Results\n\nThe method is versatile.\n",
        encoding="utf-8",
    )
    answer = (
        "s2ISM 的核心 trade-off 是分辨率提升与噪声放大之间的平衡。\n\n"
        "关于厚样本，算法假设光学像差可以忽略。"
    )
    hits = [
        {
            "text": "The method can be applied to any LSM equipped with a detector array.",
            "meta": {
                "source_path": str(source_path),
                "heading_path": "Results / Versatility of s2ISM",
            },
        }
    ]
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": str(source_path),
                "source_name": "NatPhoton Structured detection for s2ISM",
                "heading_path": "Results / Versatility of s2ISM",
                "candidate_hits": [1],
            }
        ],
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[str(source_path)],
    )
    repaired_twice = _reading_guide_repair_missing_system_a_citations(
        repaired,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[str(source_path)],
    )
    assert "空间分辨率与 SNR" in repaired
    assert "光学切片（optical sectioning）与 SNR" in repaired
    assert "限制探测器尺寸" in repaired
    assert "迭代次数与噪声放大”的单一权衡" in repaired
    assert repaired_twice == repaired
    assert hits[-1]["meta"]["citation_plan_s2ism_tradeoff"] is True
    public_source_path = "F:/library/NatPhoton-s2ism.en.md"
    for hit in hits:
        hit["meta"]["source_path"] = public_source_path
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        if ui_meta:
            ui_meta["source_path"] = public_source_path
            primary = ui_meta.get("primary_evidence") if isinstance(ui_meta.get("primary_evidence"), dict) else {}
            if primary:
                primary["source_path"] = public_source_path
    plan["slots"][0]["source_path"] = public_source_path
    _rendered, details = _annotate_inpaper_citations_with_hover_meta(
        repaired,
        hits,
        canonical_paths=[public_source_path],
        citation_plan=plan,
    )
    detail = next(item for item in details if "s2ism" in str(item.get("source_path") or "").lower())
    assert detail["citation_route"] == "system_a"
    assert "Abstract" in detail["heading_path"]
    assert "thick samples" in detail["evidence_quote"]


def test_s2ism_name_detection_accepts_superscript_and_subscript_two():
    from api.chat_render import _mentions_s2ism

    assert _mentions_s2ism("s2ISM")
    assert _mentions_s2ism("s²ISM")
    assert _mentions_s2ism("s₂ISM")


def test_s2ism_tradeoff_repair_checks_correct_terms_only_in_target_paragraph(tmp_path: Path):
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    source_path = tmp_path / "s2ism-mixed-paragraphs.en.md"
    source_path.write_text(
        "# Structured detection for laser scanning microscopy\n\n"
        "## Abstract\n\n"
        "Fast detector arrays overcome the trade-off between spatial resolution and "
        "signal-to-noise ratio. Current image scanning microscopy approaches do not "
        "provide optical sectioning and fail with thick samples unless the detector size "
        "is limited, introducing a trade-off between optical sectioning and "
        "signal-to-noise ratio.\n",
        encoding="utf-8",
    )
    answer = (
        "The main s2ISM trade-off is iteration count versus noise amplification.\n\n"
        "Spatial resolution, SNR, and optical sectioning are general microscopy terms "
        "mentioned elsewhere in this answer.\n\n"
        "Thick samples require special care."
    )
    hits = [
        {
            "text": "The method applies to laser scanning microscopy.",
            "meta": {"source_path": str(source_path), "heading_path": "Results"},
        }
    ]
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": str(source_path),
                "source_name": "Structured detection for s2ISM",
                "heading_path": "Results",
                "candidate_hits": [1],
            }
        ],
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[str(source_path)],
    )

    assert "iteration count versus noise amplification" not in repaired
    assert "two coupled trade-offs" in repaired
    assert "spatial resolution versus SNR" in repaired
    assert "optical sectioning versus SNR" in repaired
    assert "mentioned elsewhere in this answer" in repaired
    assert hits[-1]["meta"]["citation_plan_s2ism_tradeoff"] is True


def test_s2ism_tradeoff_uses_canonical_source_number_when_refs_are_reordered():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )
    from api.reference_rendering import _annotate_inpaper_citations_with_hover_meta

    iism_path = "db/iism.en.md"
    s2ism_path = "db/s2ism.en.md"
    other_path = "db/other.en.md"
    exact_evidence = (
        "Fast detector arrays overcome the trade-off between spatial resolution and "
        "signal-to-noise ratio. Current image scanning microscopy approaches do not "
        "provide optical sectioning and fail with thick samples unless the detector size "
        "is limited, introducing a trade-off between optical sectioning and signal-to-noise ratio."
    )
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": s2ism_path,
                "source_name": "Structured detection for s2ISM",
                "heading_path": "Abstract",
                "evidence_quote": exact_evidence,
                "candidate_hits": [2],
            },
            {
                "preferred_system": "system_a",
                "source_path": iism_path,
                "source_name": "Interferometric image scanning microscopy",
                "heading_path": "Methods",
                "evidence_quote": "An unrelated interferometric microscope setup.",
            },
            {
                "preferred_system": "system_a",
                "source_path": other_path,
                "source_name": "Other comparison",
                "heading_path": "Results",
                "evidence_quote": "An unrelated comparison passage.",
            },
        ],
    }
    # Refs cards are reordered with s2ISM first, while answer markers follow
    # canonical retrieval order where s2ISM is number 2.
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {"text": exact_evidence, "meta": {"source_path": s2ism_path}},
            {"text": "iISM setup.", "meta": {"source_path": iism_path}},
            {"text": "Other passage.", "meta": {"source_path": other_path}},
        ],
        plan,
        reserved_count=6,
    )
    canonical_paths = [iism_path, s2ism_path, other_path, "extra4", "extra5", "extra6"]
    answer = (
        "## s2ISM trade-off and thick samples\n"
        "The claimed trade-off is spatial resolution versus SNR [2].\n\n"
        "Thick samples are difficult because optical sectioning is limited [2]."
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=canonical_paths,
    )

    assert "detector size improves sectioning only by sacrificing SNR [7]" in repaired
    assert "[2]" not in repaired
    assert "[8]" not in repaired
    assert "[9]" not in repaired
    _rendered, details = _annotate_inpaper_citations_with_hover_meta(
        repaired,
        hits,
        canonical_paths=canonical_paths,
        citation_plan=plan,
    )
    assert len(details) == 1
    assert details[0]["citation_route"] == "system_a"
    assert details[0]["source_path"] == s2ism_path
    assert "thick samples" in details[0]["evidence_quote"]


def test_reading_guide_rebinds_foveated_claim_to_plan_passage():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _backfill_system_a_cite_details_from_ref_pack,
        _reading_guide_repair_missing_system_a_citations,
    )
    from api.reference_rendering import _annotate_inpaper_citations_with_hover_meta

    source_path = "foveated-spi.en.md"
    slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "source_name": "Adaptive foveated single-pixel imaging with dynamic supersampling",
        "heading_path": "INTRODUCTION",
        "evidence_quote": (
            "This speeds up the frame rate of the vision system. Here, we demonstrate how "
            "an adaptive foveated imaging approach enhances useful data gathering."
        ),
        "candidate_hits": [1],
    }
    plan = {"intent": "answer_grounding", "slots": [slot]}
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {
                "text": "Successive frames sample different subsets for dynamic supersampling.",
                "meta": {"source_path": source_path, "heading_path": "Spatially variant supersampling"},
            }
        ],
        plan,
        reserved_count=1,
    )
    answer = (
        "1. 自适应中心凹成像把更多采样资源放在重要区域，从而减少数据量并提高帧率 [1]。\n\n"
        "2. Dynamic supersampling 融合连续帧来补充外围细节 [1]。"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[source_path],
    )

    assert "提高帧率 [2]" in repaired
    assert "外围细节 [1]" in repaired


def test_comparison_rescue_reads_strict_source_block_before_async_ref_enrichment(
    tmp_path: Path,
):
    from api.chat_render import _augment_hits_with_system_a_plan_slots

    source_path = tmp_path / "hsi-fsi.en.md"
    source_path.write_text(
        "# Hadamard versus Fourier\n\n"
        "## 2. Comparison of theory\n\n"
        "The OTF is an ideal low-pass filter.\n\n"
        "## 3. Comparison of experiment\n\n"
        "### 3.1 Numerical simulations\n\n"
        "The coefficients are corrected gradually as the sampling ratio increases. "
        "As indicated by the curves of PSNR, SSIM, and RMSE, the convergence of HSI "
        "is lower than that of FSI.\n",
        encoding="utf-8",
    )
    weak_slot = {
        "preferred_system": "system_a",
        "source_path": str(source_path),
        "source_name": "Hadamard versus Fourier.pdf",
        "heading_path": "2. Comparison of theory",
        "evidence_quote": "The OTF is an ideal low-pass filter.",
        "candidate_hits": [],
    }
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [weak_slot],
    }
    raw_hits = [
        {
            "text": "The OTF is an ideal low-pass filter.",
            "meta": {
                "source_path": str(source_path),
                "source_name": "Hadamard versus Fourier.pdf",
                "heading_path": "2. Comparison of theory",
            },
        }
    ]

    augmented = _augment_hits_with_system_a_plan_slots(
        raw_hits,
        plan,
        reserved_count=6,
    )

    rescue = augmented[6]
    assert rescue["meta"]["citation_plan_comparison_rescue"] is True
    assert "3. Comparison of experiment" in rescue["meta"]["heading_path"]
    assert "sampling ratio" in rescue["text"]
    assert "PSNR" in rescue["text"]
    assert "SSIM" in rescue["text"]
    assert rescue["meta"]["primary_block_id"]
    assert rescue["meta"]["primary_anchor_id"]


def test_comparison_rescue_does_not_select_unplanned_retrieval_source(tmp_path: Path):
    from api.chat_render import _augment_hits_with_system_a_plan_slots

    target_path = tmp_path / "target.en.md"
    target_path.write_text(
        "# Target\n\n## 3. Comparison of experiment\n\n"
        "At each sampling ratio, PSNR and SSIM compare the two target methods.\n",
        encoding="utf-8",
    )
    extra_path = tmp_path / "extra.en.md"
    extra_path.write_text(
        "# Extra\n\n## 9. Comparison of experiment\n\n"
        "Sampling ratio, measurements, PSNR, SSIM, and RMSE describe an unrelated study.\n",
        encoding="utf-8",
    )
    plan = {
        "intent": "comparison",
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": str(target_path),
                "source_name": "Target.pdf",
                "heading_path": "2. Comparison",
                "evidence_quote": "Target overview.",
                "candidate_hits": [],
            }
        ],
    }
    hits = [
        {"text": "Unrelated", "meta": {"source_path": str(extra_path)}},
        {"text": "Target overview", "meta": {"source_path": str(target_path)}},
    ]

    augmented = _augment_hits_with_system_a_plan_slots(hits, plan)

    rescue = next(
        hit
        for hit in augmented
        if bool((hit.get("meta") or {}).get("citation_plan_comparison_rescue"))
    )
    assert rescue["meta"]["source_path"] == str(target_path)
    assert "target methods" in rescue["text"]
    assert "unrelated study" not in rescue["text"]


def test_system_a_plan_slots_create_distinct_same_paper_evidence_hits():
    from api.chat_render import _augment_hits_with_system_a_plan_slots, _reading_slot_hit_nums

    source_path = "dl-spi-review.en.md"
    hits = [{"text": "Paper overview.", "meta": {"source_path": source_path}}]
    benefit_slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "source_name": "DL SPI review",
        "heading_path": "Abstract",
        "evidence_quote": "Deep learning provides exceptional reconstruction quality and fast reconstruction speed.",
    }
    risk_slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "source_name": "DL SPI review",
        "heading_path": "4. Strategy and Advantages",
        "evidence_quote": "Data-driven training has limited generalization across imaging scenes.",
    }
    augmented = _augment_hits_with_system_a_plan_slots(
        hits,
        {"slots": [benefit_slot, risk_slot]},
    )

    assert len(augmented) == 3
    assert _reading_slot_hit_nums(benefit_slot, augmented) == [2]
    assert _reading_slot_hit_nums(risk_slot, augmented) == [3]

    reserved = _augment_hits_with_system_a_plan_slots(
        hits,
        {"slots": [benefit_slot, risk_slot]},
        reserved_count=3,
    )
    assert _reading_slot_hit_nums(benefit_slot, reserved, canonical_paths=[source_path] * 3) == [4]
    assert _reading_slot_hit_nums(risk_slot, reserved, canonical_paths=[source_path] * 3) == [5]


def test_reading_guide_does_not_add_duplicate_plan_slot_citations_to_multi_source_answer():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    source_paths = ["review.md", "comparison.md", "frontier.md"]
    hits = [
        {"text": f"Core evidence {idx}.", "meta": {"source_path": source_path}}
        for idx, source_path in enumerate(source_paths, start=1)
    ]
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_paths[0],
                "heading_path": "Abstract",
                "evidence_quote": "The review establishes the field overview.",
            },
            {
                "preferred_system": "system_a",
                "source_path": source_paths[1],
                "heading_path": "Principles",
                "evidence_quote": "The comparison explains the acquisition strategies.",
            },
        ],
    }
    augmented = _augment_hits_with_system_a_plan_slots(hits, plan, reserved_count=6)
    answer = "Review evidence [1].\n\nComparison evidence [2].\n\nFrontier evidence [3]."

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        augmented,
        plan,
        output_mode="reading_guide",
        canonical_paths=source_paths + ["extra-4.md", "extra-5.md", "extra-6.md"],
    )

    assert repaired == answer
    assert "[7]" not in repaired
    assert "[8]" not in repaired


def test_reading_guide_roadmap_keeps_review_evidence_without_inserting_bridge():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )
    from api.reference_rendering import _annotate_inpaper_citations_with_hover_meta

    source_paths = ["dl-review.en.md", "hsi-fsi.en.md", "spi-prospects.en.md"]
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_paths[0],
                "source_name": "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
                "heading_path": "Abstract",
                "evidence_quote": "Deep learning improves reconstruction quality and speed.",
            },
            {
                "preferred_system": "system_a",
                "source_path": source_paths[1],
                "source_name": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
                "heading_path": "Introduction",
                "evidence_quote": "The paper compares HSI and FSI in imaging efficiency and noise robustness.",
            },
            {
                "preferred_system": "system_a",
                "source_path": source_paths[2],
                "source_name": "Principles and prospects for single-pixel imaging",
                "heading_path": "Acquisition and image reconstruction strategies",
                "evidence_quote": (
                    "A single-pixel camera can recover images when the number of measurements is "
                    "fewer than the total number of unknown pixels, also known as under-sampling."
                ),
            },
        ],
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {"text": "Generic DL overview.", "meta": {"source_path": source_paths[0]}},
            {"text": "Generic HSI comparison.", "meta": {"source_path": source_paths[1]}},
            {"text": "Generic SPI review.", "meta": {"source_path": source_paths[2]}},
        ],
        plan,
        reserved_count=6,
    )
    answer = (
        "### Principles and prospects for single-pixel imaging\n"
        "Compressive sensing enables undersampled reconstruction [3].\n\n"
        "### Hadamard versus Fourier\nThe two bases are compared [2].\n\n"
        "### Deep learning review\nQuality and speed are reviewed [1]."
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=source_paths + ["extra-4.md", "extra-5.md", "extra-6.md"],
    )

    assert "### Principles and prospects for single-pixel imaging [9]" in repaired
    assert "Compressive sensing enables undersampled reconstruction [3]." in repaired
    assert repaired.count("[9]") == 1
    assert "The review states that" not in repaired
    _rendered, details = _annotate_inpaper_citations_with_hover_meta(
        repaired,
        hits,
        canonical_paths=source_paths + ["extra-4.md", "extra-5.md", "extra-6.md"],
        citation_plan=plan,
    )
    detail = next(item for item in details if int(item.get("num") or 0) == 9)
    assert "number of measurements is fewer" in detail["evidence_quote"]
    assert "unknown pixels" in detail["evidence_quote"]


def test_reading_guide_system_a_plan_enables_linking_without_existing_marker():
    from api.chat_render import _should_link_inpaper_citations_for_message

    rec = {
        "meta": {
            "answer_quality": {
                "output_mode": "reading_guide",
                "citation_plan": {
                    "slots": [
                        {
                            "preferred_system": "system_a",
                            "candidate_hits": [],
                            "source_path": "hsi-fsi.md",
                        }
                    ]
                },
            }
        }
    }

    assert _should_link_inpaper_citations_for_message(
        rec=rec,
        content="追求采集速度选 Hadamard，追求物理可解释性选 Fourier。",
        hits=[{"text": "Hadamard and Fourier comparison", "meta": {"source_path": "hsi-fsi.md"}}],
    ) is True


def test_reading_guide_repair_dedupes_system_a_slots_for_same_source():
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    answer = (
        "结论：追求采集速度选 Hadamard，追求物理可解释性选 Fourier。\n\n"
        "| 维度 | Hadamard | Fourier |\n"
        "|:---|:---|:---|\n"
        "| 采集 | 快 | 慢 |\n\n"
        "选择建议：\n"
        "1. Hadamard 适合高速 DMD。\n"
        "2. Fourier 适合分析空间频率。\n"
        "3. 两者都属于全局变换。"
    )
    hits = [
        {
            "text": "Hadamard and Fourier single-pixel imaging are compared experimentally.",
            "meta": {"source_path": "hsi-fsi.md"},
        }
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "hsi-fsi.md",
                "source_name": "Hadamard Fourier comparison",
                "evidence_quote": "Hadamard and Fourier comparison.",
            },
            {
                "preferred_system": "system_a",
                "source_path": "hsi-fsi.md",
                "source_name": "Hadamard Fourier comparison",
                "evidence_quote": "Fourier spatial frequency comparison.",
            },
        ]
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
    )

    assert repaired.count("[1]") == 1
    citation_line = next(line for line in repaired.splitlines() if "[1]" in line)
    assert "Hadamard" in citation_line or "Fourier" in citation_line
    assert "两者都属于全局变换 [1]" not in repaired
    assert "| [1]" not in repaired


def test_reading_guide_repair_does_not_add_ranked_sources_when_every_step_is_already_cited():
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    answer = (
        "## 1. 综述\n\n**论文：** Review paper [2]\n\n为什么读它：建立全局认识。\n\n"
        "## 2. 实时系统\n\n**论文：** Real-time paper [6]\n\n为什么读它：理解工程实现。"
    )
    hits = [
        {"text": "Unselected ranked source", "meta": {"source_path": "rank-1.md"}},
        {"text": "Review paper", "meta": {"source_path": "review.md"}},
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "rank-1.md",
                "source_name": "Unselected ranked source",
            }
        ]
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
    )

    assert repaired == answer
    assert "[1]" not in repaired


def test_reading_guide_repair_resolves_stale_candidate_hit_by_source_path():
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    answer = (
        "Overview: SCI moves from spectral data cubes toward 3D scene reconstruction.\n\n"
        "Stage 1: early SCI dual-disperser spectral imaging compresses a spectral data cube [1].\n\n"
        "Stage 2: SCINeRF and SCIGS extend SCI to 3D scene reconstruction [2]."
    )
    hits = [
        {"text": "SCINeRF uses snapshot compressive imaging for 3D scene representation.", "meta": {"source_path": "scinerf.md"}},
        {"text": "SCIGS reconstructs dynamic 3D scenes from snapshot compressive images.", "meta": {"source_path": "scigs.md"}},
        {"text": "Single-shot compressive spectral imaging uses a dual-disperser architecture.", "meta": {"source_path": "cassi.md"}},
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "cassi.md",
                "source_name": "Single-shot compressive spectral imaging with a dual-disperser architecture",
                "evidence_quote": "Single-shot compressive spectral imaging uses a dual-disperser architecture.",
            }
        ]
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
    )

    assert "spectral data cube [1] [3]." in repaired
    assert "Overview: SCI moves from spectral data cubes toward 3D scene reconstruction. [3]" not in repaired


def test_reading_guide_repair_prefers_canonical_number_for_source_path():
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    answer = (
        "1. Detector review covers single-photon detectors, SPAD hardware, and photodetector applications.\n\n"
        "2. Physics-informed deep learning models SPAD noise [1]."
    )
    hits = [
        {"text": "Physics-informed deep learning models SPAD noise.", "meta": {"source_path": "pidl.md"}},
        {"text": "A denoising review mentions physics-informed methods.", "meta": {"source_path": "denoise.md"}},
        {"text": "Single-photon detector review covers SPAD devices and applications.", "meta": {"source_path": "spd-review.md"}},
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [2],
                "source_path": "spd-review.md",
                "source_name": "Emerging single-photon detection technique for high-performance photodetector",
                "evidence_quote": "Single-photon detector review covers SPAD devices and applications.",
            }
        ]
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=["pidl.md", "spd-review.md", "piln.md"],
    )

    assert "photodetector applications [2]." in repaired
    assert "photodetector applications [3]." not in repaired


def test_merge_render_packet_contract_meta_allows_refs_pack_to_replace_coarse_cross_paper_seed():
    from api import chat_render

    rec = {
        "content": "Grounded answer.",
        "rendered_body": "Grounded answer.",
        "rendered_content": "Grounded answer.",
        "copy_markdown": "Grounded answer.",
        "copy_text": "Grounded answer.",
        "notice": "",
        "cite_details": [],
        "meta": {
            "paper_guide_contracts": {
                "primary_evidence": {
                    "source_path": "natphoton.md",
                    "source_name": "NatPhoton-2019.pdf",
                    "heading_path": "Abstract / Acquisition and image reconstruction strategies.",
                    "snippet": "A broad answer-hit snippet.",
                    "selection_reason": "answer_hit_top",
                },
                "render_packet": {},
            }
        },
    }

    chat_render._merge_render_packet_contract_meta(
        rec=rec,
        msg_id=5,
        enriched_provenance={"segments": []},
        ref_pack={
            "primary_evidence": {
                "source_path": "oe2017.md",
                "source_name": "OE-2017.pdf",
                "block_id": "blk_22",
                "anchor_id": "a_22",
                "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                "snippet": "Section 2.2 explicitly compares Hadamard and Fourier basis patterns.",
                "selection_reason": "prompt_aligned",
            }
        },
        chat_store=None,
    )

    contracts = ((rec.get("meta") or {}).get("paper_guide_contracts") or {})
    packet = contracts.get("render_packet") or {}
    assert (contracts.get("primary_evidence") or {}).get("source_name") == "OE-2017.pdf"
    assert (contracts.get("primary_evidence") or {}).get("block_id") == "blk_22"
    assert (packet.get("primary_evidence") or {}).get("source_name") == "OE-2017.pdf"
    assert (packet.get("primary_evidence") or {}).get("heading_path") == "2. Comparison of theory / 2.2 Basis patterns generation"


def test_enrich_messages_invalidates_render_cache_when_refs_change(monkeypatch, tmp_path: Path):
    from api import chat_render

    calls = {"primary": 0}

    def fake_primary(_md, _hits, *, anchor_ns="", canonical_paths=None):
        del _hits, anchor_ns, canonical_paths
        calls["primary"] += 1
        return (
            f"render-{calls['primary']}::{_md}",
            [
                {
                    "num": calls["primary"],
                    "anchor": f"kb-cite-demo-{calls['primary']}",
                    "source_name": "demo.pdf",
                    "is_inpaper": True,
                }
            ],
        )

    monkeypatch.setattr(chat_render, "_annotate_inpaper_citations_with_hover_meta", fake_primary)

    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("cache invalidation test")
    user_id = store.append_message(conv_id, "user", "test")
    assistant_id = store.append_message(conv_id, "assistant", "SPI relies on compressive sensing [[CITE:s1234abcd:1]].")

    refs_v1 = {
        user_id: {
            "prompt_sig": "sig-1",
            "updated_at": 1.0,
            "used_query": "test",
            "used_translation": False,
            "hits": [{"text": "dummy", "meta": {"source_path": r"db\doc\doc.en.md"}}],
        }
    }
    refs_v2 = {
        user_id: {
            "prompt_sig": "sig-2",
            "updated_at": 2.0,
            "used_query": "test-updated",
            "used_translation": False,
            "hits": [{"text": "dummy-updated", "meta": {"source_path": r"db\doc\doc.en.md"}}],
        }
    }

    store.merge_message_meta(
        assistant_id,
        {"answer_quality": {"prompt_family": "citation_lookup", "output_mode": "citation_lookup"}},
    )

    first = enrich_messages_with_reference_render(store.get_messages(conv_id), refs_v1, conv_id=conv_id, chat_store=store)
    second = enrich_messages_with_reference_render(store.get_messages(conv_id), refs_v2, conv_id=conv_id, chat_store=store)

    assert calls["primary"] == 2
    assert str(first[-1].get("rendered_content") or "") != str(second[-1].get("rendered_content") or "")


def test_unlinked_reference_candidates_find_unique_venue_year(monkeypatch):
    from api import chat_render

    monkeypatch.setattr(
        chat_render,
        "_load_reference_index_cached",
        lambda: {
            "docs": {
                "demo": {
                    "path": "current-paper.md",
                    "name": "current-paper.en.md",
                    "refs": {
                        "7": {
                            "num": 7,
                            "raw": "Smith J. Fast rotation-shearing single-pixel imaging. Optica. 2024.",
                            "title": "Fast rotation-shearing single-pixel imaging",
                            "authors": "Smith J",
                            "venue": "Optica",
                            "year": "2024",
                            "doi": "10.1364/optica.demo",
                        }
                    },
                }
            }
        },
    )

    candidates = chat_render._build_unlinked_reference_candidates(
        answer_markdown="For real-time imaging, the Optica 2024 work is a better comparison point.",
        rendered_body="",
        copy_text="",
        cite_details=[],
        ref_pack={"hits": [{"meta": {"source_path": "current-paper.md"}}]},
        provenance_segments=[],
        render_locale="en",
        anchor_ns="test",
    )

    assert len(candidates) == 1
    assert candidates[0]["match_method"] == "unique_venue_year_mention"
    assert candidates[0]["ref_num"] == 7
    assert candidates[0]["title"] == "Fast rotation-shearing single-pixel imaging"
    assert candidates[0]["cite_detail"]["citation_route"] == "system_b"
    assert "answer_context_only" in candidates[0]["cite_detail"]["card_quality_flags"]
    assert candidates[0]["cite_detail"]["system_b_trace_complete"] is False
    assert "answer_context_only" in candidates[0]["cite_detail"]["system_b_trace_flags"]


def test_unlinked_reference_candidate_promotes_retrieved_library_document(monkeypatch):
    from api import chat_render

    local_source = r"F:\kb\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md"
    parent_source = r"F:\kb\NatCommun-2021-Imaging biological tissue.en.md"
    second_parent_source = r"F:\kb\NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md"
    monkeypatch.setattr(
        chat_render,
        "_load_reference_index_cached",
        lambda: {
            "docs": {
                "demo": {
                    "path": parent_source,
                    "name": "NatCommun-2021-Imaging biological tissue.en.md",
                    "refs": {
                        "12": {
                            "num": 12,
                            "raw": "Zhang Z et al. Hadamard single-pixel imaging versus Fourier single-pixel imaging. Opt. Express. 2017.",
                            "title": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
                            "authors": "Zhang Z, Wang X, Zheng G, et al",
                            "venue": "Opt. Express",
                            "year": "2017",
                            "doi": "10.1364/oe.25.019619",
                        }
                    },
                },
                "demo-duplicate": {
                    "path": second_parent_source,
                    "name": "NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md",
                    "refs": {
                        "41": {
                            "num": 41,
                            "raw": "Zhang Z et al. Hadamard single-pixel imaging versus Fourier single-pixel imaging. Opt. Express. 2017.",
                            "title": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
                            "authors": "Zhang Z, Wang X, Zheng G, et al",
                            "venue": "Opt. Express",
                            "year": "2017",
                            "doi": "10.1364/oe.25.019619-duplicate-index-row",
                        }
                    },
                },
            }
        },
    )

    candidates = chat_render._build_unlinked_reference_candidates(
        answer_markdown="The best match is Hadamard single-pixel imaging versus Fourier single-pixel imaging.",
        rendered_body="",
        copy_text="",
        cite_details=[],
        ref_pack={
            "hits": [
                {"meta": {"source_path": parent_source}},
                {"meta": {"source_path": second_parent_source}},
                {"meta": {"source_path": local_source}},
            ]
        },
        provenance_segments=[],
        render_locale="en",
        anchor_ns="test",
    )

    assert len(candidates) == 1
    detail = candidates[0]["cite_detail"]
    assert candidates[0]["source_path"] == local_source
    assert candidates[0]["ref_num"] == 0
    assert detail["source_path"] == local_source
    assert detail["is_inpaper"] is False
    assert detail["citation_route"] == "system_a"
    assert detail["library_match_status"] == "in_library"
    assert detail["reference_source_path"] == parent_source
    assert detail["reference_ref_num"] == 12


def test_reading_guide_repair_binds_benefit_and_risk_evidence_to_distinct_claims():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    source_path = "dl-spi-review.en.md"
    risk_slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "heading_path": "4. Strategy and Advantages",
        "evidence_quote": "Data-driven strategies have prolonged training and limited generalization across imaging scenes.",
    }
    benefit_slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "heading_path": "Abstract",
        "evidence_quote": "Deep learning provides exceptional reconstruction quality and fast reconstruction speed.",
    }
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [risk_slot, benefit_slot],
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [{"text": "Paper overview.", "meta": {"source_path": source_path}}],
        plan,
    )
    answer = (
        "深度学习给单像素成像带来了更高的重建质量和更快的重建速度。\n\n"
        "主要风险：\n"
        "- 数据驱动方法训练时间长，而且泛化能力有限，难以适应多样场景。"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
    )

    assert "重建速度 [3]。" in repaired
    assert "多样场景 [2]。" in repaired


def test_reading_guide_repair_combines_adjacent_risks_supported_by_one_evidence_sentence():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )
    from api.reference_rendering import _annotate_inpaper_citations_with_hover_meta

    source_path = "dl-spi-review.en.md"
    risk_slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "source_name": "DL-SPI review",
        "heading_path": "4. Strategy and Advantages",
        "evidence_quote": (
            "Data-driven strategies have prolonged training duration and limited generalization, "
            "which makes them hard to adapt to diverse imaging scenes."
        ),
    }
    benefit_slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "source_name": "DL-SPI review",
        "heading_path": "Abstract",
        "evidence_quote": "Deep learning provides exceptional reconstruction quality and fast reconstruction speed.",
    }
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [risk_slot, benefit_slot],
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [{"text": "Paper overview.", "meta": {"source_path": source_path}}],
        plan,
        reserved_count=6,
    )
    answer = (
        "深度学习能够提高重建质量和重建速度。\n\n"
        "主要风险包括：\n"
        "- 训练时间长：数据驱动策略的训练周期较长。\n"
        "- 泛化能力有限：难以有效适应多样化的成像场景。\n"
        "- 依赖大量数据集：需要大量训练数据。"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[source_path] * 6,
    )
    _rendered, details = _annotate_inpaper_citations_with_hover_meta(
        repaired,
        hits,
        canonical_paths=[source_path] * 6,
        citation_plan=plan,
    )

    assert "训练时间长：数据驱动策略的训练周期较长；泛化能力有限：难以有效适应多样化的成像场景 [7]。" in repaired
    assert "\n- 泛化能力有限" not in repaired
    risk_detail = next(item for item in details if int(item.get("num") or 0) == 7)
    assert "训练" in str(risk_detail.get("answer_claim") or "")
    assert "泛化" in str(risk_detail.get("answer_claim") or "")
    assert "prolonged training" in str(risk_detail.get("evidence_quote") or "")
    assert "limited generalization" in str(risk_detail.get("evidence_quote") or "")


def test_reading_guide_repair_combines_separated_data_training_and_generalization_claims():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )
    from api.reference_rendering import _annotate_inpaper_citations_with_hover_meta

    source_path = "dl-spi-review.en.md"
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "DL-SPI review",
                "heading_path": "4. Strategy and Advantages",
                "evidence_quote": (
                    "Data-driven strategies have prolonged training duration and limited generalization, "
                    "which makes them hard to adapt to diverse imaging scenes."
                ),
            },
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "DL-SPI review",
                "heading_path": "Abstract",
                "evidence_quote": (
                    "Deep learning provides exceptional reconstruction quality and fast reconstruction speed."
                ),
            },
        ],
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [{"text": "Paper overview.", "meta": {"source_path": source_path}}],
        plan,
        reserved_count=6,
    )
    answer = (
        "深度学习能够提高重建质量和重建速度。\n\n"
        "主要风险包括：\n"
        "- 依赖大规模数据集：训练需要大量标注数据。\n"
        "- 泛化能力有限：难以有效适应多样化的成像场景。\n\n"
        "- 可解释性差：模型的决策过程难以理解。\n"
        "- 容易过拟合：在未见过的数据上可能表现不佳。\n\n"
        "此外，数据驱动策略的训练时间较长，这也是实际应用中的挑战。"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[source_path] * 6,
    )
    _rendered, details = _annotate_inpaper_citations_with_hover_meta(
        repaired,
        hits,
        canonical_paths=[source_path] * 6,
        citation_plan=plan,
    )

    assert "数据驱动策略的训练时间较长" in repaired
    assert "泛化能力有限" in repaired
    risk_detail = next(
        item
        for item in details
        if "limited generalization" in str(item.get("evidence_quote") or "")
    )
    assert "数据" in str(risk_detail.get("answer_claim") or "")
    assert "泛化" in str(risk_detail.get("answer_claim") or "")


def test_reading_guide_repair_combines_numbered_risks_supported_by_one_evidence_sentence():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )
    from api.reference_rendering import _annotate_inpaper_citations_with_hover_meta

    source_path = "dl-spi-review.en.md"
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "DL-SPI review",
                "heading_path": "4. Strategy and Advantages",
                "evidence_quote": (
                    "Data-driven strategies have prolonged training duration and limited generalization, "
                    "which makes them hard to adapt to diverse imaging scenes."
                ),
            },
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "DL-SPI review",
                "heading_path": "Abstract",
                "evidence_quote": (
                    "Deep learning provides exceptional reconstruction quality and fast reconstruction speed."
                ),
            },
        ],
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [{"text": "Paper overview.", "meta": {"source_path": source_path}}],
        plan,
        reserved_count=6,
    )
    answer = (
        "Deep learning improves reconstruction quality and speed.\n\n"
        "Main risks:\n"
        "1. Prolonged training: data-driven strategies take a long time to train.\n"
        "2. Limited generalization: they struggle with diverse imaging scenes.\n"
        "3. Large datasets: they require substantial training data."
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[source_path] * 6,
    )
    _rendered, details = _annotate_inpaper_citations_with_hover_meta(
        repaired,
        hits,
        canonical_paths=[source_path] * 6,
        citation_plan=plan,
    )

    assert "1. Prolonged training: data-driven strategies take a long time to train; Limited generalization: they struggle with diverse imaging scenes [7]." in repaired
    assert "\n2. Limited generalization" not in repaired
    risk_detail = next(item for item in details if int(item.get("num") or 0) == 7)
    assert "Prolonged training" in str(risk_detail.get("answer_claim") or "")
    assert "Limited generalization" in str(risk_detail.get("answer_claim") or "")
    assert "prolonged training" in str(risk_detail.get("evidence_quote") or "")
    assert "limited generalization" in str(risk_detail.get("evidence_quote") or "")


def test_reading_guide_rebinds_three_source_markers_to_dedicated_plan_hits():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    source_paths = ["paper-a.en.md", "paper-b.en.md", "paper-c.en.md"]
    slots = [
        {
            "preferred_system": "system_a",
            "source_path": source_path,
            "source_name": f"Paper {idx}",
            "heading_path": "Abstract",
            "evidence_quote": f"Direct evidence for paper {idx}.",
        }
        for idx, source_path in enumerate(source_paths, start=1)
    ]
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": slots,
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {"text": "Raw C", "meta": {"source_path": source_paths[2]}},
            {"text": "Raw B", "meta": {"source_path": source_paths[1]}},
            {"text": "Raw A", "meta": {"source_path": source_paths[0]}},
        ],
        plan,
        reserved_count=3,
    )
    answer = "1. Paper A overview [1].\n2. Paper B overview [2].\n3. Paper C overview [3]."

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=source_paths,
    )

    assert "Paper A overview [4]" in repaired
    assert "Paper B overview [5]" in repaired
    assert "Paper C overview [6]" in repaired


def test_reading_guide_adds_one_plan_citation_to_each_named_paper_heading():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    titles = [
        "Principles and prospects for single-pixel imaging",
        "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
        "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
    ]
    source_paths = [f"paper-{idx}.en.md" for idx in range(1, 4)]
    slots = [
        {
            "preferred_system": "system_a",
            "source_path": source_path,
            "source_name": title,
            "topic": f"{title} / Abstract",
            "heading_path": f"{title} / Abstract",
            "evidence_quote": f"Direct source evidence for {title}.",
        }
        for title, source_path in zip(titles, source_paths)
    ]
    plan = {"budget": {"system_a": 3, "system_b": 0}, "slots": slots}
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {"text": f"Raw evidence {idx}.", "meta": {"source_path": source_path}}
            for idx, source_path in enumerate(source_paths, start=1)
        ],
        plan,
        reserved_count=3,
    )
    answer = "\n\n".join(
        f"### {idx}. {title}\n\nMain point [{idx}]."
        for idx, title in enumerate(titles, start=1)
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=source_paths,
    )

    for idx, title in enumerate(titles, start=4):
        assert f"{title} [{idx}]" in repaired


def test_reading_guide_keeps_occurrence_markers_when_source_has_multiple_plan_slots():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    source_paths = ["paper-a.en.md", "paper-b.en.md", "paper-c.en.md"]
    slots = [
        {
            "preferred_system": "system_a",
            "source_path": source_paths[0],
            "heading_path": "Method",
            "evidence_quote": "Paper A method evidence.",
        },
        {
            "preferred_system": "system_a",
            "source_path": source_paths[0],
            "heading_path": "Limitations",
            "evidence_quote": "Paper A limitation evidence.",
        },
        {
            "preferred_system": "system_a",
            "source_path": source_paths[1],
            "heading_path": "Abstract",
            "evidence_quote": "Paper B evidence.",
        },
        {
            "preferred_system": "system_a",
            "source_path": source_paths[2],
            "heading_path": "Abstract",
            "evidence_quote": "Paper C evidence.",
        },
    ]
    plan = {"budget": {"system_a": 3, "system_b": 0}, "slots": slots}
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {"text": f"Raw {idx}.", "meta": {"source_path": source_path}}
            for idx, source_path in enumerate(source_paths, start=1)
        ],
        plan,
        reserved_count=3,
    )
    answer = "Paper A method [1]. Paper A limitation [1]. Paper B [2]. Paper C [3]."

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=source_paths,
    )

    assert repaired.count("[1]") == 2
    assert "Paper B [6]" in repaired
    assert "Paper C [7]" in repaired


def test_reading_guide_rebinds_only_the_locally_aligned_same_source_occurrence():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    source_paths = ["paper-a.en.md", "paper-b.en.md", "paper-c.en.md"]
    slots = [
        {
            "preferred_system": "system_a",
            "source_path": source_path,
            "source_name": f"Paper {letter}",
            "heading_path": "Abstract",
            "evidence_quote": f"Paper {letter} directly supports its overview.",
        }
        for source_path, letter in zip(source_paths, ("A", "B", "C"))
    ]
    plan = {"budget": {"system_a": 3, "system_b": 0}, "slots": slots}
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {"text": f"Raw {letter}.", "meta": {"source_path": source_path}}
            for source_path, letter in zip(source_paths, ("A", "B", "C"))
        ],
        plan,
        reserved_count=3,
    )
    answer = (
        "Paper A overview [1].\n"
        "A general deployment warning with no support in the selected passage [1].\n"
        "Paper B overview [2].\n"
        "Paper C overview [3]."
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=source_paths,
    )

    assert "Paper A overview [4]" in repaired
    assert "deployment warning with no support in the selected passage [1]" in repaired
    assert "Paper B overview [5]" in repaired
    assert "Paper C overview [6]" in repaired


def test_reading_guide_cassi_lineage_keeps_three_system_a_cards_and_cleans_system_b_prose():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    source_paths = ["cassi.en.md", "scinerf.en.md", "scigs.en.md"]
    system_a_slots = [
        {
            "preferred_system": "system_a",
            "source_path": source_paths[0],
            "source_name": "Single-shot compressive spectral imaging with a dual-disperser architecture",
            "heading_path": "Abstract",
            "evidence_quote": "Two dispersive elements surround a binary aperture code.",
        },
        {
            "preferred_system": "system_a",
            "source_path": source_paths[1],
            "source_name": "SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
            "heading_path": "Methods",
            "evidence_quote": "SCINeRF uses a neural radiance field and the SCI physical image formation process.",
        },
        {
            "preferred_system": "system_a",
            "source_path": source_paths[2],
            "source_name": "SCIGS: 3D Gaussians Splatting from a Snapshot Compressive Image",
            "heading_path": "Abstract",
            "evidence_quote": "SCIGS reconstructs a dynamic 3D scene from one compressed image.",
        },
    ]
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 3, "system_b": 1},
        "slots": [
            {
                "preferred_system": "system_b",
                "source_path": source_paths[1],
                "topic": "snapshot compressive imaging",
            },
            *system_a_slots,
        ],
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {"text": "Raw CASSI.", "meta": {"source_path": source_paths[0]}},
            {"text": "Raw SCINeRF.", "meta": {"source_path": source_paths[1]}},
            {"text": "Raw SCIGS.", "meta": {"source_path": source_paths[2]}},
        ],
        plan,
        reserved_count=3,
    )
    answer = (
        "CASSI starts with a dual-disperser architecture [1].\n"
        "Video SCI is an upstream step [ [[CITE:sid:50]] ].\n"
        "SCINeRF uses NeRF with the SCI physical image formation process.\n"
        "SCIGS reconstructs a dynamic 3D scene [3].\n"
        "如需细节，请查阅原始论文（如文献[[CITE:sid:50]]）。"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=source_paths,
    )

    assert "CASSI starts with a dual-disperser architecture [4]" in repaired
    assert "SCINeRF uses NeRF with the SCI physical image formation process [5]" in repaired
    assert "SCIGS reconstructs a dynamic 3D scene [6]" in repaired
    assert "[ [[CITE:sid:50]] ]" not in repaired
    assert "upstream step [[CITE:sid:50]]" in repaired
    assert "原始论文" not in repaired
    assert "上游文献或背景入口（如文献[[CITE:sid:50]]）" in repaired


def test_reading_guide_names_ilnet_and_binds_method_plus_strategy_evidence():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _backfill_system_a_cite_details_from_ref_pack,
        _reading_guide_repair_missing_system_a_citations,
    )
    from api.reference_rendering import _annotate_inpaper_citations_with_hover_meta

    method_path = "part-based-image-loop.en.md"
    review_path = "dl-spi-review.en.md"
    other_path = "unrelated.en.md"
    slots = [
        {
            "preferred_system": "system_a",
            "source_path": method_path,
            "source_name": "Part-based image-loop network for single-pixel imaging",
            "heading_path": "Methods / ILNet architecture",
            "evidence_quote": (
                "We propose a self-supervised image-loop neural network (ILNet) with a "
                "part-based model; detector signals are labels for optimization."
            ),
        },
        {
            "preferred_system": "system_a",
            "source_path": review_path,
            "source_name": "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
            "heading_path": "4.1.2 Model-Driven Strategy",
            "evidence_quote": (
                "Model-driven strategy is an unsupervised learning mode that integrates the "
                "physical process of SPI with neural networks."
            ),
        },
        {
            "preferred_system": "system_a",
            "source_path": other_path,
            "source_name": "Other SPI paper",
            "heading_path": "Methods",
            "evidence_quote": "An unrelated detector model.",
        },
    ]
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": slots,
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {"text": "Raw ILNet.", "meta": {"source_path": method_path}},
            {"text": "Raw review.", "meta": {"source_path": review_path}},
            {"text": "Raw other.", "meta": {"source_path": other_path}},
        ],
        plan,
        reserved_count=3,
    )
    answer = (
        "## PILN 与主线的关系\n\n"
        "PILN（Part-based Image-Loop Network）属于模型驱动策略，这是两条主线之一 [2]。\n\n"
        "### 深度学习单像素成像的两条主线\n\n"
        "### 不适合解决的问题\n\n"
        "| 实时成像任务 | 迭代需要大量计算时间 |\n"
        "| 高帧率视频成像 | 难以恢复高帧率图像 [5] |\n\n"
        "### 关键权衡\n\n"
        "代价是 **计算时间**，这限制了它在实时应用中的部署。"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[method_path, review_path, other_path],
    )

    assert "论文原文将该方法称为 **ILNet**" in repaired
    assert "part-based model" in repaired
    assert "问题中称 PILN" in repaired
    assert "[4]" in repaired
    assert "[5]" not in repaired
    assert "model-driven strategy" in repaired
    assert "用于定位的两类策略" in repaired
    assert "两条主线之一" not in repaired
    assert "实时成像" not in repaired
    assert "高帧率" not in repaired
    assert "[2]" not in repaired
    assert hits[3]["meta"]["heading_path"].endswith("Methods / ILNet architecture")
    assert "self-supervised image-loop neural network (ILNet)" in hits[3]["text"]
    assert "physical process of SPI" in hits[4]["text"]
    pinned_review_hits = [
        (idx, hit)
        for idx, hit in enumerate(hits, start=1)
        if isinstance(hit, dict)
        and isinstance(hit.get("meta"), dict)
        and hit["meta"].get("citation_plan_ilnet_review") is True
    ]
    assert len(pinned_review_hits) == 1
    pinned_review_num, pinned_review_hit = pinned_review_hits[0]
    assert f"[{pinned_review_num}]" in repaired
    assert "model-driven strategy" in pinned_review_hit["text"].lower()
    assert "physical process of SPI" in pinned_review_hit["text"]

    _rendered, details = _annotate_inpaper_citations_with_hover_meta(
        repaired,
        hits,
        canonical_paths=[method_path, review_path, other_path],
        citation_plan=plan,
    )
    review_detail = next(
        item for item in details if int(item.get("num") or 0) == pinned_review_num
    )
    assert review_detail.get("citation_plan_slot") is True
    generic_abstract = (
        "Single-pixel imaging technology can capture images at wavelengths outside conventional "
        "detectors, while deep learning improves reconstruction quality and speed."
    )
    backfilled = _backfill_system_a_cite_details_from_ref_pack(
        [review_detail],
        {
            "primary_evidence": {
                "source_path": review_path,
                "source_name": "DL-SPI review",
                "heading_path": "Abstract",
                "snippet": generic_abstract,
                "selection_reason": "pending_section_seed",
            }
        },
    )
    assert "model-driven strategy" in backfilled[0]["evidence_quote"].lower()
    assert "physical process of SPI" in backfilled[0]["evidence_quote"]
    assert backfilled[0]["heading_path"] == "4.1.2. Model-Driven Strategy"


def test_s2ism_tradeoff_whole_paragraph_rewrite_requires_focused_comparison_plan():
    from api.chat_render import _reading_guide_repair_s2ism_tradeoff_answer

    answer = (
        "This method map mentions the s2ISM trade-off in thick samples, then compares "
        "it with two unrelated microscopy methods."
    )
    plan = {
        "intent": "method_explain",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "s2ism.en.md",
                "source_name": "Structured detection for s2ISM",
            }
        ],
    }

    repaired = _reading_guide_repair_s2ism_tradeoff_answer(answer, [], plan)

    assert repaired == answer


def test_s2ism_repair_continues_binding_other_planned_sources():
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    s2ism_evidence = (
        "Spatial resolution and signal-to-noise trade-off. Current image scanning "
        "microscopy approaches do not provide optical sectioning in thick samples "
        "unless detector size is limited, sacrificing signal-to-noise."
    )
    method_x_evidence = "Method X improves axial resolution by phase diversity."
    hits = [
        {
            "text": s2ism_evidence,
            "meta": {"source_path": "s2ism.en.md", "heading_path": "Abstract"},
        },
        {
            "text": method_x_evidence,
            "meta": {"source_path": "method-x.en.md", "heading_path": "Results"},
        },
    ]
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "s2ism.en.md",
                "source_name": "Structured detection s2ISM",
                "heading_path": "Abstract",
                "evidence_quote": s2ism_evidence,
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": "method-x.en.md",
                "source_name": "Method X",
                "heading_path": "Results",
                "evidence_quote": method_x_evidence,
                "candidate_hits": [2],
            },
        ],
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        "s2ISM trade-off in thick samples.\n\n"
        "Method X improves axial resolution by phase diversity.",
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=["s2ism.en.md", "method-x.en.md"],
    )

    assert "Method X improves axial resolution" in repaired
    assert "[1]" in repaired
    assert "[2]" in repaired


def test_normal_answer_binds_three_planned_sources_without_inserting_topic_specific_prose():
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    answer = (
        "Paper Alpha establishes the measurement model.\n\n"
        "Paper Beta parallelizes the hardware acquisition.\n\n"
        "Paper Gamma adds learned reconstruction."
    )
    hits = [
        {"text": "Alpha evidence.", "meta": {"source_path": "alpha.en.md"}},
        {"text": "Beta evidence.", "meta": {"source_path": "beta.en.md"}},
        {"text": "Gamma evidence.", "meta": {"source_path": "gamma.en.md"}},
    ]
    plan = {
        "intent": "multi_source_synthesis",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "alpha.en.md",
                "source_name": "Paper Alpha",
                "evidence_quote": "Alpha evidence.",
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": "beta.en.md",
                "source_name": "Paper Beta",
                "evidence_quote": "Beta evidence.",
                "candidate_hits": [2],
            },
            {
                "preferred_system": "system_a",
                "source_path": "gamma.en.md",
                "source_name": "Paper Gamma",
                "evidence_quote": "Gamma evidence.",
                "candidate_hits": [3],
            },
        ],
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="answer",
        canonical_paths=["alpha.en.md", "beta.en.md", "gamma.en.md"],
    )

    assert "Paper Alpha establishes the measurement model [1]." in repaired
    assert "Paper Beta parallelizes the hardware acquisition [2]." in repaired
    assert "Paper Gamma adds learned reconstruction [3]." in repaired
    assert "single-pixel camera" not in repaired


def test_multi_source_plan_normalizes_duplicate_canonical_markers_before_budgeting():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    canonical_paths = [
        "distractor-a.en.md",
        "distractor-b.en.md",
        "review.en.md",
        "distractor-c.en.md",
        "deep-review.en.md",
        "hardware.en.md",
    ]
    hits = [
        {"text": "Hardware evidence.", "meta": {"source_path": "hardware.en.md"}},
        {"text": "Deep review evidence.", "meta": {"source_path": "deep-review.en.md"}},
        {"text": "Foundation review evidence.", "meta": {"source_path": "review.en.md"}},
    ]
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "deep-review.en.md",
                "source_name": "Deep Review",
                "evidence_quote": "Deep review evidence.",
                "candidate_hits": [5],
            },
            {
                "preferred_system": "system_a",
                "source_path": "hardware.en.md",
                "source_name": "Hardware Paper",
                "evidence_quote": "Hardware evidence.",
                "candidate_hits": [6],
            },
            {
                "preferred_system": "system_a",
                "source_path": "review.en.md",
                "source_name": "Foundation Review",
                "evidence_quote": "Foundation review evidence.",
                "candidate_hits": [3],
            },
        ],
    }
    augmented = _augment_hits_with_system_a_plan_slots(
        hits,
        plan,
        reserved_count=len(canonical_paths),
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        "Foundation Review establishes the model.\n\n"
        "Hardware Paper accelerates acquisition [6] and improves throughput [6].\n\n"
        "Deep Review covers learned reconstruction [5] and deployment [5].",
        augmented,
        plan,
        output_mode="answer",
        canonical_paths=canonical_paths,
    )

    assert "[5]" not in repaired
    assert "[6]" not in repaired
    assert repaired.count("[7]") == 2
    assert repaired.count("[8]") == 2
    assert repaired.count("[9]") == 1


def test_microscopy_method_map_repair_preserves_unrelated_numeric_citations(
    tmp_path: Path,
    monkeypatch,
):
    from api import chat_render

    sources = [
        (
            "s2ism.en.md",
            "Structured detection for s2ISM",
            "Structured detection provides simultaneous super-resolution and optical sectioning.",
        ),
        (
            "iism.en.md",
            "Interferometric image scanning microscopy",
            "Interferometric detection enables live-cell imaging at 120 nm lateral resolution.",
        ),
        (
            "light-field.en.md",
            "Light-field microscopy",
            "Light-field microscopy records position and angular information for volumetric reconstruction.",
        ),
    ]
    hits: list[dict] = []
    slots: list[dict] = []
    for index, (filename, source_name, evidence) in enumerate(sources, start=1):
        source_path = tmp_path / filename
        source_path.write_text(f"# {source_name}\n\n## Abstract\n\n{evidence}\n", encoding="utf-8")
        hits.append(
            {
                "text": evidence,
                "meta": {"source_path": str(source_path), "source_name": source_name},
            }
        )
        slots.append(
            {
                "preferred_system": "system_a",
                "source_path": str(source_path),
                "source_name": source_name,
                "candidate_hits": [index],
            }
        )
    evidence_by_source = {source_name: evidence for _, source_name, evidence in sources}
    monkeypatch.setattr(
        chat_render,
        "_claim_aligned_abstract_primary_evidence",
        lambda _pack, item: {
            "snippet": evidence_by_source[str(item.get("source_name") or "")],
            "heading_path": "Abstract",
        },
    )
    hits.append(
        {
            "text": "An unrelated paper supports the acquisition-system claim.",
            "meta": {"source_path": str(tmp_path / "unrelated.en.md")},
        }
    )
    answer = (
        "s2ISM uses structured detection [1].\n\n"
        "iISM uses interferometric detection [2].\n\n"
        "Light-field microscopy records angular information [3].\n\n"
        "The acquisition system has an independently supported property [4]."
    )

    repaired = chat_render._reading_guide_repair_microscopy_method_map_evidence(
        answer,
        hits,
        {"slots": slots},
    )

    assert "structured detection [1]" not in repaired
    assert "interferometric detection [2]" not in repaired
    assert "angular information [3]" not in repaired
    assert "independently supported property [4]" in repaired
    assert all(f"[{num}]" in repaired for num in (5, 6, 7))


def test_perovskite_scope_bridge_does_not_rewrite_answer_without_boundary_claim():
    from api.chat_render import _reading_guide_repair_scope_boundary_citation

    answer = "The perovskite laser uses a dual-cavity device and we should inspect its materials stack."
    plan = {
        "intent": "scope_boundary",
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "perovskite.en.md",
                "evidence_quote": "We demonstrate lasing from a dual-cavity perovskite device.",
                "candidate_hits": [1],
            }
        ],
    }
    hits = [{"text": "Device evidence.", "meta": {"source_path": "perovskite.en.md"}}]

    repaired = _reading_guide_repair_scope_boundary_citation(answer, hits, plan)

    assert repaired == answer


def test_reading_guide_repair_bridges_perovskite_device_scope_to_chinese_claim():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _backfill_system_a_cite_details_from_ref_pack,
        _reading_guide_repair_missing_system_a_citations,
        _should_link_inpaper_citations_for_message,
    )
    from ui.refs_renderer import _annotate_inpaper_citations_with_hover_meta

    source_path = "F:/library/perovskite-laser.en.md"
    slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "heading_path": "Abstract",
        "evidence_quote": "We demonstrate electrically driven lasing from a dual-cavity perovskite device.",
    }
    plan = {
        "intent": "scope_boundary",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [slot],
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [{"text": "Paper overview.", "meta": {"source_path": source_path}}],
        plan,
    )
    answer = (
        "直接回答：关系不大，不是当前主线的核心文献。\n\n"
        "这篇论文研究电驱动钙钛矿激光器的器件结构。"
        "你的单像素成像主线属于计算成像，两者几乎没有交集。"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
    )

    assert "dual-cavity perovskite" in repaired
    assert "lasing 研究，而不是单像素成像方法 [2]" in repaired
    _rendered, details = _annotate_inpaper_citations_with_hover_meta(
        repaired,
        hits,
        citation_plan=plan,
    )
    details = _backfill_system_a_cite_details_from_ref_pack(
        details,
        {
            "primary_evidence": {
                "source_path": source_path,
                "source_name": "Perovskite laser.pdf",
                "heading_path": "Abstract",
                "snippet": slot["evidence_quote"],
                "selection_reason": "prompt_aligned",
            }
        },
        render_locale="zh",
    )
    assert len(details) == 1
    detail = details[0]
    assert detail["citation_route"] == "system_a"
    assert detail["binding_status"] == "grounded"
    assert all(term in detail["answer_claim"] for term in ("perovskite", "器件", "不是"))
    assert all(term in detail["evidence_quote"] for term in ("dual-cavity perovskite", "lasing"))
    assert "Abstract" in detail["heading_path"]

    concise_repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="concise_answer",
    )
    rec = {
        "content": answer,
        "meta": {
            "answer_quality": {
                "output_mode": "concise_answer",
                "citation_plan": plan,
            }
        },
    }
    assert "dual-cavity perovskite" in concise_repaired
    assert _should_link_inpaper_citations_for_message(rec=rec, content=answer, hits=hits)


def test_reading_guide_repair_ignores_unrelated_same_paper_method_slot():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    source_path = "qclfm.en.md"
    refocus_slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "heading_path": "A. Concept",
        "evidence_quote": (
            "Digital refocusing uses two steps. First, ray tracing reconstructs photon trajectories. "
            "The second step applies wave propagation to reverse diffraction."
        ),
    }
    unrelated_slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "heading_path": "Figure 1",
        "evidence_quote": "Type-II spontaneous parametric down-conversion produces orthogonally polarized photon pairs.",
    }
    plan = {
        "intent": "method_explain",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [refocus_slot, unrelated_slot],
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [{"text": "Paper overview.", "meta": {"source_path": source_path}}],
        plan,
    )
    answer = "数字重聚焦分为两步：先用 ray tracing 重建轨迹，再用 wave propagation 反演衍射。"

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
    )

    assert "反演衍射 [2]。" in repaired
    assert "[3]" not in repaired


def test_system_a_render_backfills_public_bibliography_without_primary_evidence(monkeypatch):
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack
    from kb.citation_card import compose_citation_card

    monkeypatch.setattr(
        "api.chat_render.load_local_source_citation_meta",
        lambda *_args, **_kwargs: {},
    )
    source_path = r"db\Nature-2024-Useful paper\Nature-2024-Useful paper.en.md"
    details = [
        compose_citation_card({
            "num": 1,
            "anchor": "kb-cite-1",
            "source_name": "Nature-2024-Useful paper.pdf",
            "source_path": source_path,
            "title": "3. Results",
            "heading_path": "3. Results",
            "is_inpaper": False,
            "citation_route": "system_a",
            "answer_claim": "The method improves reconstruction quality.",
            "evidence_quote": "The method improves reconstruction quality.",
        })
    ]
    ref_pack = {
        "hits": [
            {
                "text": "The method improves reconstruction quality.",
                "meta": {"source_path": source_path},
                "ui_meta": {
                    "citation_meta": {
                        "title": "Useful Paper",
                        "authors": "Ada Lovelace; Grace Hopper",
                        "venue": "Nature Methods",
                        "year": "2024",
                        "doi": "10.1234/useful.paper",
                        "doi_url": "https://doi.org/10.9999/wrong-url",
                        "venue_kind": "journal",
                        "metadata_quality": {"score": 99},
                    }
                },
            }
        ]
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, ref_pack)

    assert len(out) == 1
    detail = out[0]
    assert detail["title"] == "Useful Paper"
    assert detail["bibliographic_title"] == "Useful Paper"
    assert detail["authors"] == "Ada Lovelace; Grace Hopper"
    assert detail["venue"] == "Nature Methods"
    assert detail["year"] == "2024"
    assert detail["doi"] == "10.1234/useful.paper"
    assert detail["doi_url"] == "https://doi.org/10.1234/useful.paper"
    assert detail["venue_kind"] == "journal"
    assert "metadata_quality" not in detail
    assert detail["heading_path"] == "3. Results"
    assert detail["card_view"]["header"]["subtitle"] == "3. Results"


def test_reading_guide_does_not_bind_piln_evidence_to_pidl_retrieval_notice():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    source_path = "F:/library/Part-based image-loop network for single-pixel imaging.en.md"
    evidence = (
        "Researchers embed an untrained neural network into the physical model for "
        "single-pixel image reconstruction."
    )
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "Part-based image-loop network for single-pixel imaging.pdf",
                "heading_path": "1. Introduction",
                "evidence_quote": evidence,
            }
        ],
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {
                "text": "PILN is an untrained network for single-pixel imaging.",
                "meta": {"source_path": source_path},
            }
        ],
        plan,
    )
    answer = (
        "根据检索到的文献，PIDL 的相关内容未出现在本次检索结果中，"
        "因此以下比较仅基于检索到的 PILN 信息。\n\n"
        "PILN 将单像素成像物理模型嵌入未训练神经网络 [1]。"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
    )

    retrieval_notice, piln_claim = repaired.split("\n\n", 1)
    assert "[2]" not in retrieval_notice
    assert "[2]" in piln_claim


def test_system_a_bibliography_priority_is_existing_then_ref_pack_then_local(monkeypatch):
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    monkeypatch.setattr(
        "api.chat_render.load_local_source_citation_meta",
        lambda *_args, **_kwargs: {
            "title": "Stale Local Title",
            "authors": "Local Author",
            "venue": "Local Venue",
            "year": "2020",
            "doi": "10.1000/local",
        },
    )
    source_path = "db/paper.en.md"
    details = [
        {
            "num": 1,
            "source_path": source_path,
            "source_name": "paper.pdf",
            "citation_route": "system_a",
            "title": "Existing Detail Title",
            "bibliographic_title": "Existing Detail Title",
            "authors": "Existing Detail Author",
            "heading_path": "3. Results",
            "evidence_quote": "Grounded evidence.",
        }
    ]
    ref_pack = {
        "hits": [
            {
                "meta": {"source_path": source_path},
                "ui_meta": {
                    "citation_meta": {
                        "title": "New Ref Pack Title",
                        "authors": "Ref Pack Author",
                        "venue": "Ref Pack Venue",
                        "year": "2025",
                    }
                },
            }
        ]
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, ref_pack)

    assert out[0]["title"] == "Existing Detail Title"
    assert out[0]["bibliographic_title"] == "Existing Detail Title"
    assert out[0]["authors"] == "Existing Detail Author"
    assert out[0]["venue"] == "Ref Pack Venue"
    assert out[0]["year"] == "2025"
    assert out[0]["doi"] == "10.1000/local"


def test_system_a_bibliography_keeps_same_basename_paths_and_dois_separate(monkeypatch):
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    monkeypatch.setattr(
        "api.chat_render.load_local_source_citation_meta",
        lambda *_args, **_kwargs: {},
    )
    source_a = r"db\collection-a\Repeated Paper.en.md"
    source_b = r"db\collection-b\Repeated Paper.en.md"
    details = [
        {
            "num": 1,
            "source_path": source_a,
            "source_name": "Repeated Paper.pdf",
            "citation_route": "system_a",
            "title": "Methods A",
            "heading_path": "Methods A",
            "evidence_quote": "Evidence from collection A.",
        },
        {
            "num": 2,
            "source_path": source_b,
            "source_name": "Repeated Paper.pdf",
            "citation_route": "system_a",
            "title": "Methods B",
            "heading_path": "Methods B",
            "evidence_quote": "Evidence from collection B.",
        },
        {
            "num": 3,
            "source_name": "Repeated Paper.pdf",
            "citation_route": "system_a",
            "title": "Ambiguous Methods",
            "heading_path": "Ambiguous Methods",
            "evidence_quote": "Evidence without a source path.",
        },
    ]
    ref_pack = {
        "hits": [
            {
                "meta": {"source_path": source_a},
                "ui_meta": {
                    "citation_meta": {
                        "title": "Collection A Paper",
                        "doi": "10.1234/collection-a",
                    }
                },
            },
            {
                "meta": {"source_path": source_b},
                "ui_meta": {
                    "citation_meta": {
                        "title": "Collection B Paper",
                        "doi": "10.1234/collection-b",
                    }
                },
            },
        ]
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, ref_pack)

    assert out[0]["bibliographic_title"] == "Collection A Paper"
    assert out[0]["doi"] == "10.1234/collection-a"
    assert out[1]["bibliographic_title"] == "Collection B Paper"
    assert out[1]["doi"] == "10.1234/collection-b"
    assert "bibliographic_title" not in out[2]
    assert "doi" not in out[2]


def test_system_a_primary_evidence_backfill_preserves_page_locator():
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    source_path = "db/Paper/Paper.en.md"
    details = [
        {
            "num": 1,
            "citation_route": "system_a",
            "source_path": source_path,
            "answer_claim": "SCINeRF embeds the physical imaging process in NeRF training.",
            "evidence_quote": "NeRF training.",
        }
    ]
    primary = {
        "source_path": source_path,
        "source_name": "Paper",
        "heading_path": "Abstract",
        "snippet": "We formulate the physical imaging process of SCI as part of the training of NeRF.",
        "highlight_snippet": "We formulate the physical imaging process of SCI as part of the training of NeRF.",
        "block_id": "blk-1",
        "anchor_id": "p-1",
        "anchor_kind": "paragraph",
        "page_start": 1,
        "page_end": 1,
    }
    pack = {
        "primary_evidence": primary,
        "hits": [
            {
                "text": primary["snippet"],
                "meta": {"source_path": source_path},
                "ui_meta": {"source_path": source_path, "primary_evidence": primary},
            }
        ],
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, pack, render_locale="en")

    assert out[0]["page_start"] == 1
    assert out[0]["page_end"] == 1
    assert "p. 1" in out[0]["location_label"]


def test_answer_aligned_ref_primary_becomes_page_aware_citation_plan_hit():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _citation_plan_with_ref_primary,
    )

    primary = {
        "source_path": "db/Paper/Paper.en.md",
        "source_name": "Paper",
        "heading_path": "Abstract",
        "snippet": "The supported answer evidence.",
        "block_id": "blk-2",
        "anchor_id": "p-2",
        "anchor_kind": "paragraph",
        "page_start": 2,
        "page_end": 2,
        "strict_locate": True,
    }
    plan = _citation_plan_with_ref_primary(
        {
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "db/Paper/Paper.en.md",
                    "heading_path": "Introduction",
                    "evidence_quote": "A generic same-paper passage.",
                },
                {
                    "preferred_system": "system_a",
                    "source_path": "db/Other/Other.en.md",
                    "heading_path": "Results",
                    "evidence_quote": "A relevant passage from another paper.",
                },
            ],
        },
        {"primary_evidence": primary},
    )
    hits = _augment_hits_with_system_a_plan_slots([], plan)

    assert plan["slots"][0]["selection_reason"] == "answer_aligned_reference_primary"
    assert len(plan["slots"]) == 2
    assert plan["slots"][1]["source_path"] == "db/Other/Other.en.md"
    assert hits[0]["meta"]["page_start"] == 2
    assert hits[0]["meta"]["primary_block_id"] == "blk-2"
    assert hits[0]["ui_meta"]["primary_evidence"]["page_start"] == 2


def test_final_answer_alignment_runs_before_citation_repair(tmp_path, monkeypatch):
    monkeypatch.setattr("ui.refs_renderer._is_temp_source_path", lambda _path: False)
    source_path = tmp_path / "three-d-video.en.md"
    source_path.write_text(
        "\n".join(
            [
                "<!-- kb_page: 2 -->",
                "## Abstract",
                (
                    "Performing high-speed structured illumination and sensing reflected light with four "
                    "spatially-separated, single-pixel detectors, our system reconstructs real-time 3D video "
                    "at 8 frames per second for image resolutions of 64 by 64 pixels."
                ),
                "<!-- kb_page: 5 -->",
                "## Methods",
                "Hadamard patterns are projected by the spatial light modulator.",
            ]
        ),
        encoding="utf-8",
    )
    answer = (
        "The system uses four spatially-separated single-pixel detectors and reconstructs real-time 3D video "
        "at 8 frames per second for 64 by 64 pixels."
    )
    messages = [
        {"id": 1, "role": "user", "content": "How many detectors are used, and what is the speed?"},
        {
            "id": 2,
            "role": "assistant",
            "content": answer,
            "meta": {
                "answer_quality": {
                    "output_mode": "reading_guide",
                    "citation_plan": {"budget": {"system_a": 1, "system_b": 0}, "slots": []},
                }
            },
        },
    ]
    refs_by_user = {
        1: {
            "prompt": messages[0]["content"],
            "hits": [
                {
                    "text": "Hadamard patterns are projected by the spatial light modulator.",
                    "meta": {"source_path": str(source_path), "heading_path": "Methods"},
                    "ui_meta": {
                        "source_path": str(source_path),
                        "display_name": "3D single-pixel video",
                        "heading_path": "Methods",
                    },
                }
            ],
        }
    }

    from api.chat_render import (
        _answer_aligned_reference_render_pack,
        _augment_hits_with_system_a_plan_slots,
        _citation_plan_with_ref_primary,
        _reading_guide_repair_missing_system_a_citations,
    )

    aligned_pack = _answer_aligned_reference_render_pack(refs_by_user[1], answer)
    assert aligned_pack["primary_evidence"]["heading_path"] == "Abstract"
    plan = _citation_plan_with_ref_primary(messages[1]["meta"]["answer_quality"]["citation_plan"], aligned_pack)
    assert plan["slots"][0]["heading_path"] == "Abstract"
    citation_hits = _augment_hits_with_system_a_plan_slots(aligned_pack["hits"], plan)
    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        citation_hits,
        plan,
        output_mode="reading_guide",
    )
    assert repaired != answer

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="answer-align")

    details = rendered[-1]["cite_details"]
    assert len(details) == 1
    assert details[0]["heading_path"] == "Abstract"
    assert details[0]["page_start"] == 2
    assert "four spatially-separated" in details[0]["evidence_quote"]
    assert "8 frames per second" in details[0]["evidence_quote"]


def test_system_a_binding_bridges_chinese_spad_noise_claim_to_english_evidence():
    from ui.refs_renderer import _assess_system_a_hit_binding

    evidence = (
        "The multi-source physical noise model of SPAD arrays consists of dark count rate, "
        "afterpulsing and crosstalk noise."
    )
    binding = _assess_system_a_hit_binding(
        answer_claim="后脉冲和串扰噪声会产生额外的虚假事件。",
        hit={"text": evidence},
        meta={},
        heading="Introduction / Figure 1",
        evidence_quote=evidence,
        source_name="High-resolution single-photon imaging with physics-informed deep learning",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False
    assert "crosstalk noise" in binding["overlap_terms"]
