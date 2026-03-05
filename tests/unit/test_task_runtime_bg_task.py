from pathlib import Path

from kb.chat_store import ChatStore
from kb.task_runtime import (
    _augment_prompt_with_source_hint,
    _build_bg_task,
    _filter_history_for_multimodal_turn,
    _has_anchor_grounded_answer_hits,
    _needs_conversational_source_hint,
    _pick_recent_source_hint,
)


def test_ultra_fast_does_not_force_no_llm():
    task = _build_bg_task(
        pdf_path=Path("a.pdf"),
        out_root=Path("out"),
        db_dir=Path("db"),
        no_llm=False,
        replace=False,
        speed_mode="ultra_fast",
    )
    assert task["speed_mode"] == "ultra_fast"
    assert task["no_llm"] is False


def test_no_llm_mode_respects_flag():
    task = _build_bg_task(
        pdf_path=Path("a.pdf"),
        out_root=Path("out"),
        db_dir=Path("db"),
        no_llm=True,
        replace=False,
        speed_mode="no_llm",
    )
    assert task["speed_mode"] == "no_llm"
    assert task["no_llm"] is True


def test_conversational_source_hint_detection_and_augmentation():
    assert _needs_conversational_source_hint("那这篇文章里的公式8写的是什么")
    assert not _needs_conversational_source_hint("NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf里的公式8写的是什么")
    assert (
        _augment_prompt_with_source_hint(
            "那这篇文章里的公式8写的是什么",
            "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf",
        )
        == "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf 那这篇文章里的公式8写的是什么"
    )


def test_pick_recent_source_hint_from_message_refs(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation()
    user1 = store.append_message(conv_id, "user", "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf这篇文章的第二张图是什么")
    store.append_message(conv_id, "assistant", "...")
    store.upsert_message_refs(
        user_msg_id=user1,
        conv_id=conv_id,
        prompt="NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf这篇文章的第二张图是什么",
        prompt_sig="sig1",
        hits=[
            {
                "text": "Fig. 2 | Evaluating regularized and non-regularized image reconstructions with compressive sensing",
                "meta": {
                    "source_path": str(
                        Path("db")
                        / "NatPhoton-2019-Principles and prospects for single-pixel imaging"
                        / "NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md"
                    ),
                },
            }
        ],
        scores=[9.0],
        used_query="NatPhoton-2019 ... figure 2",
        used_translation=False,
    )
    user2 = store.append_message(conv_id, "user", "那这篇文章里的公式8写的是什么")
    hint = _pick_recent_source_hint(conv_id=conv_id, user_msg_id=user2, chat_store=store)
    assert hint == "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf"


def test_has_anchor_grounded_answer_hits_detects_numbered_item_match():
    hits = [
        {
            "meta": {
                "anchor_target_kind": "equation",
                "anchor_target_number": 8,
                "anchor_match_score": 14.5,
            }
        }
    ]
    assert _has_anchor_grounded_answer_hits(hits) is True


def test_has_anchor_grounded_answer_hits_ignores_unmatched_items():
    hits = [
        {
            "meta": {
                "anchor_target_kind": "equation",
                "anchor_target_number": 8,
                "anchor_match_score": 0.0,
            }
        }
    ]
    assert _has_anchor_grounded_answer_hits(hits) is False


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
            "content": "上一张图里是一个流程图。",
        },
        {
            "id": 3,
            "role": "user",
            "content": "继续说说方法细节",
            "attachments": [],
        },
        {
            "id": 4,
            "role": "assistant",
            "content": "这里的方法用了 NeRF。",
        },
    ]

    out = _filter_history_for_multimodal_turn(
        history,
        cur_user_msg_id=9,
        cur_assistant_msg_id=10,
        has_current_images=True,
    )

    assert [int(item["id"]) for item in out] == [3, 4]


def test_filter_history_for_multimodal_turn_keeps_old_history_without_current_images():
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
            "content": "上一张图里是一个流程图。",
        },
    ]

    out = _filter_history_for_multimodal_turn(
        history,
        cur_user_msg_id=9,
        cur_assistant_msg_id=10,
        has_current_images=False,
    )

    assert [int(item["id"]) for item in out] == [1, 2]
