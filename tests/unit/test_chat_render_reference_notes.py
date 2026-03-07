from api.chat_render import enrich_messages_with_reference_render


def test_equation_source_note_does_not_reference_removed_refs_ui():
    messages = [
        {"id": 1, "role": "user", "content": "NatPhoton 公式8是什么"},
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
    assert "参考定位" not in body
    assert "库内文献" in body


def test_equation_source_note_is_not_added_without_hits():
    messages = [
        {"id": 1, "role": "user", "content": "NatPhoton 公式8是什么"},
        {
            "id": 2,
            "role": "assistant",
            "content": "$$\nI_{TC}=x \\tag{8}\n$$",
        },
    ]

    rendered = enrich_messages_with_reference_render(messages, refs_by_user={}, conv_id="conv-test")
    body = str(rendered[-1].get("rendered_body") or "")

    assert "库内文献" not in body


def test_copy_outputs_and_rendered_content_are_consistent():
    messages = [
        {"id": 1, "role": "user", "content": "请解释这个结论"},
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
        {"id": 1, "role": "user", "content": "解释单像素成像"},
        {
            "id": 2,
            "role": "assistant",
            "content": "[SID:s50f9c165] 这是内部标签，不应该展示给用户。",
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
