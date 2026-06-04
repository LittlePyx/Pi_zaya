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
    assert "鍙傝€冨畾浣" not in body
    assert "库内文献" in body
    assert "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf" in body


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
        "`1) 鏉ヨ嚜鍙傝€冨畾浣?#1锛歚CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf`）*"
    )

    out = _normalize_equation_source_notes(raw)

    assert "鍙傝€冨畾浣" not in out
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
    assert "cite_details" not in msg
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
    assert "cite_details" not in msg
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
    assert "cite_details" not in msg
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


def test_chat_messages_merge_cached_reference_payload_prefers_enriched_hits(monkeypatch):
    from api.routers import chat, references

    monkeypatch.setitem(
        references._REFS_CONVERSATION_CACHE,
        "conv-cached-refs",
        {
            "payload": {
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
            }
        },
    )

    merged = chat._merge_cached_reference_render_payload(
        "conv-cached-refs",
        {101: {"prompt": "raw prompt", "hits": [{"text": "raw only"}]}},
    )

    assert merged[101]["prompt"] == "raw prompt"
    assert merged[101]["hits"][0]["text"] == "raw only"
    assert merged[101]["rendered_payload"]["prompt"] == "cached prompt"
    assert merged[101]["rendered_payload"]["hits"][0]["text"] == "enriched"
    assert merged[101]["rendered_payload"]["hits"][0]["ui_meta"]["primary_evidence"]["block_id"] == "blk_cached"


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
