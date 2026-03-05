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
