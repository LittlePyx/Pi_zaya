import json
from pathlib import Path

from kb.chat_store import ChatStore
from api.chat_render import (
    _enrich_provenance_segments_for_display,
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

    def fake_primary(_md, _hits, *, anchor_ns=""):
        del _md, _hits, anchor_ns
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

    def fake_primary(_md, _hits, *, anchor_ns=""):
        del _md, _hits, anchor_ns
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

    def fake_primary(_md, _hits, *, anchor_ns=""):
        del _hits, anchor_ns
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

    def fake_primary(_md, _hits, *, anchor_ns=""):
        del _hits, anchor_ns
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

    def fake_primary(_md, _hits, *, anchor_ns=""):
        del _hits, anchor_ns
        calls["primary"] += 1
        return (
            f"render-{calls['primary']}::{_md}",
            [{"num": calls["primary"], "anchor": f"kb-cite-demo-{calls['primary']}", "source_name": "demo.pdf"}],
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
