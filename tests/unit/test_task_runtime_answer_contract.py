from __future__ import annotations


def test_detect_answer_intent_prefers_explicit_hint():
    from kb import task_runtime

    intent = task_runtime._detect_answer_intent(
        "Can you compare method A and method B?",
        answer_mode_hint="writing",
    )
    assert intent == "writing"


def test_detect_answer_intent_by_prompt():
    from kb import task_runtime

    assert task_runtime._detect_answer_intent("compare transformer and cnn for this task") == "compare"
    assert task_runtime._detect_answer_intent("这个 idea 可行吗，风险是什么") == "idea"
    assert task_runtime._detect_answer_intent("如何设计实验和对照组") == "experiment"
    assert task_runtime._detect_answer_intent("训练报错了，帮我排查") == "troubleshoot"
    assert task_runtime._detect_answer_intent("帮我润色 related work 这一段") == "writing"


def test_detect_answer_depth_auto_and_fixed():
    from kb import task_runtime

    assert task_runtime._detect_answer_depth("ok?", intent="reading", auto_depth=True) == "L1"
    assert task_runtime._detect_answer_depth(
        "idea feasibility with experiment design and evaluation metrics for a new algorithm",
        intent="idea",
        auto_depth=True,
    ) == "L3"
    assert task_runtime._detect_answer_depth("any prompt", intent="reading", auto_depth=False) == "L2"


def test_apply_answer_contract_with_hits():
    from kb import task_runtime

    raw = (
        "This method can reduce reconstruction noise in low-light capture.\n\n"
        "The retrieved snippet reports lower MAE and higher PSNR than baseline [1].\n\n"
        "Another paragraph with details [2]."
    )
    out = task_runtime._apply_answer_contract_v1(
        raw,
        prompt="what is the key contribution?",
        has_hits=True,
        intent="reading",
        depth="L2",
    )
    assert "Conclusion:" in out
    assert "Evidence:" in out
    assert "Next Steps:" in out
    assert "[1]" in out or "[2]" in out


def test_apply_answer_contract_without_hits_adds_limits():
    from kb import task_runtime

    raw = "No direct paper snippet is available for this query. Here is a general answer."
    out = task_runtime._apply_answer_contract_v1(
        raw,
        prompt="how to start",
        has_hits=False,
        intent="reading",
        depth="L2",
    )
    assert "Conclusion:" in out
    assert "Limits:" in out
    assert "general guidance" in out
    assert "Next Steps:" in out


def test_apply_answer_contract_is_idempotent_on_structured_text():
    from kb import task_runtime

    raw = "Conclusion: done.\n\nEvidence:\n1. snippet [1]\n\nNext Steps:\n1. verify"
    out = task_runtime._apply_answer_contract_v1(
        raw,
        prompt="q",
        has_hits=True,
        intent="reading",
        depth="L2",
    )
    assert out == raw


def test_apply_answer_contract_repairs_partial_structured_text():
    from kb import task_runtime

    raw = "Next Steps:\n1. Check the cited section."
    out = task_runtime._apply_answer_contract_v1(
        raw,
        prompt="what should I do next?",
        has_hits=True,
        intent="reading",
        depth="L2",
    )
    assert "Conclusion:" in out
    assert out.count("Next Steps:") == 1


def test_apply_answer_contract_uses_chinese_locale_for_chinese_prompt():
    from kb import task_runtime

    raw = "这项方法在低照度下重建质量更高。"
    out = task_runtime._apply_answer_contract_v1(
        raw,
        prompt="这篇论文核心贡献是什么？",
        has_hits=False,
        intent="reading",
        depth="L2",
    )
    assert "结论:" in out
    assert "下一步:" in out
    assert "general guidance" not in out


def test_enhance_kb_miss_fallback_appends_next_steps():
    from kb import task_runtime

    raw = "未命中知识库片段。\n\n这是一个通用说明。"
    out = task_runtime._enhance_kb_miss_fallback(
        raw,
        has_hits=False,
        intent="reading",
        depth="L2",
    )
    assert "未命中知识库片段" in out
    assert "下一步建议" in out
    assert "1." in out


def test_enhance_kb_miss_fallback_does_not_duplicate_next_steps():
    from kb import task_runtime

    raw = "未命中知识库片段。\n\n说明。\n\nNext Steps:\n1. keep"
    out = task_runtime._enhance_kb_miss_fallback(
        raw,
        has_hits=False,
        intent="reading",
        depth="L2",
    )
    assert out.count("Next Steps:") == 1


def test_sanitize_structured_tokens_removes_sid_markers():
    from kb import task_runtime

    raw = "[SID:s50f9c165] text\n[1] [SID:s50f9c165] source | section\nanswer body"
    out = task_runtime._sanitize_structured_cite_tokens(raw)
    assert "[SID:" not in out
    assert "answer body" in out
