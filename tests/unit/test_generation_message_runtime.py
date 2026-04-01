from pathlib import Path

from kb.generation_message_runtime import (
    _build_generation_messages,
    _build_multimodal_user_content,
    _filter_history_for_multimodal_turn,
)


def test_filter_history_for_multimodal_turn_skips_old_image_roundtrip():
    history = [
        {
            "id": 1,
            "role": "user",
            "content": "[Image attachment x1]",
            "attachments": [{"mime": "image/png", "path": "a.png"}],
        },
        {
            "id": 2,
            "role": "assistant",
            "content": "old image reply",
        },
        {
            "id": 3,
            "role": "user",
            "content": "follow-up question",
            "attachments": [],
        },
        {
            "id": 4,
            "role": "assistant",
            "content": "follow-up answer",
        },
    ]

    out = _filter_history_for_multimodal_turn(
        history,
        cur_user_msg_id=9,
        cur_assistant_msg_id=10,
        has_current_images=True,
        is_live_assistant_text=lambda text: False,
    )

    assert [int(item["id"]) for item in out] == [3, 4]


def test_build_multimodal_user_content_embeds_local_image(tmp_path: Path):
    img = tmp_path / "demo.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    out = _build_multimodal_user_content(
        "inspect this image",
        [{"path": str(img), "mime": "image/png"}],
        vision_image_mime_by_suffix={".png": "image/png"},
    )

    assert isinstance(out, list)
    assert out[0]["type"] == "text"
    assert out[1]["type"] == "image_url"
    assert str(out[1]["image_url"]["url"]).startswith("data:image/png;base64,")


def test_build_generation_messages_wraps_system_history_and_user():
    out = _build_generation_messages(
        system="system",
        hist=[{"role": "assistant", "content": "prev"}],
        user_content="user",
    )

    assert out == [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "prev"},
        {"role": "user", "content": "user"},
    ]
