from tools.manual_regression import run_paper_guide_lsa_regression as runner


def test_last_assistant_message_prefers_non_live_message_after_min_id():
    messages = [
        {"id": 8, "role": "assistant", "content": "older turn"},
        {"id": 12, "role": "assistant", "content": "__KB_LIVE_TASK__:demo"},
        {"id": 13, "role": "assistant", "content": "final answer"},
    ]

    picked = runner._last_assistant_message(messages, min_message_id=10)

    assert int(picked["id"]) == 13
    assert picked["content"] == "final answer"


def test_fetch_answer_bundle_waits_for_current_non_live_message(monkeypatch):
    calls = [
        [
            {"id": 8, "role": "assistant", "content": "older turn"},
            {"id": 12, "role": "assistant", "content": "__KB_LIVE_TASK__:demo"},
        ],
        [
            {"id": 8, "role": "assistant", "content": "older turn"},
            {"id": 12, "role": "assistant", "content": "__KB_LIVE_TASK__:demo"},
            {"id": 13, "role": "assistant", "content": "final answer"},
        ],
    ]

    def _fake_list_messages(base_url: str, *, conv_id: str, timeout_s: float):
        assert base_url == "http://demo"
        assert conv_id == "conv-1"
        assert timeout_s == 5.0
        return calls.pop(0)

    monkeypatch.setattr(runner, "_list_messages", _fake_list_messages)

    message, answer = runner._fetch_answer_bundle(
        "http://demo",
        conv_id="conv-1",
        payload={},
        timeout_s=5.0,
        min_message_id=10,
        attempts=2,
        delay_s=0.0,
    )

    assert int(message["id"]) == 13
    assert answer == "final answer"


def test_evaluate_case_counts_raw_structured_citations_without_render_cache():
    result = runner._evaluate_case(
        {"id": "Q2", "title": "method_apr", "prompt": "Explain APR.", "cite_required": False, "jump_required": False},
        answer_text="Evidence:\n- Implementation detail [[CITE:sabc1234:35]].",
        answer_quality={"minimum_ok": True},
        render_cache={},
        provenance={},
        message={},
    )

    assert result["structured_cite_count"] == 1
    assert result["raw_structured_cite_count"] == 1
    assert result["cite_count"] == 1


def test_has_jump_asset_accepts_direct_provenance_locate_entry_without_asset_link():
    assert runner._has_jump_asset(
        "Conclusion:\n- The paper reports a clear trade-off.",
        {
            "source_path": r"F:\papers\demo.pdf",
            "segments": [
                {
                    "evidence_mode": "direct",
                    "locate_policy": "optional",
                    "primary_block_id": "blk_demo_1",
                    "anchor_text": "The paper reports a clear trade-off.",
                    "text": "The paper reports a clear trade-off.",
                }
            ],
        },
    ) is True


def test_has_jump_asset_ignores_hidden_or_non_direct_segments_without_asset_link():
    assert runner._has_jump_asset(
        "Conclusion:\n- Summary only.",
        {
            "source_path": r"F:\papers\demo.pdf",
            "segments": [
                {
                    "evidence_mode": "synthesis",
                    "locate_policy": "optional",
                    "primary_block_id": "blk_demo_1",
                    "anchor_text": "Summary only.",
                },
                {
                    "evidence_mode": "direct",
                    "locate_policy": "hidden",
                    "primary_block_id": "blk_demo_2",
                    "anchor_text": "Summary only.",
                },
            ],
        },
    ) is False


def test_minimum_ok_label_returns_na_when_probe_missing():
    assert runner._minimum_ok_label(None) == "n/a"
    assert runner._minimum_ok_label({}) == "n/a"
    assert runner._minimum_ok_label({"minimum_ok": True}) == "true"
