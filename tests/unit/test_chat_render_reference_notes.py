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
        {"id": 2, "role": "assistant", "content": "Gehm et al. (2007) [[CITE:s1234abcd:24]]."},
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
        {"id": 2, "role": "assistant", "content": "SPI relies on compressive sensing [[CITE:s1234abcd:1]]."},
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
    store.append_message(conv_id, "assistant", "SPI relies on compressive sensing [[CITE:s1234abcd:1]].")
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

    first = enrich_messages_with_reference_render(store.get_messages(conv_id), refs_by_user, conv_id=conv_id, chat_store=store)
    second = enrich_messages_with_reference_render(store.get_messages(conv_id), refs_by_user, conv_id=conv_id, chat_store=store)

    assert calls["primary"] == 1
    assert str(first[-1].get("rendered_content") or "") == str(second[-1].get("rendered_content") or "")
    assert str(second[-1].get("copy_text") or "").strip()


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
    store.append_message(conv_id, "assistant", "SPI relies on compressive sensing [[CITE:s1234abcd:1]].")

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

    first = enrich_messages_with_reference_render(store.get_messages(conv_id), refs_v1, conv_id=conv_id, chat_store=store)
    second = enrich_messages_with_reference_render(store.get_messages(conv_id), refs_v2, conv_id=conv_id, chat_store=store)

    assert calls["primary"] == 2
    assert str(first[-1].get("rendered_content") or "") != str(second[-1].get("rendered_content") or "")
