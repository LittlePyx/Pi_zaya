from kb.answer_presentation import clean_assistant_answer_presentation_text


def test_clean_assistant_answer_presentation_text_removes_trace_heading_suffix():
    answer = "\n\n".join(
        [
            "The useful answer stays here.",
            "Research Agent Trace\nPlan\n- retrieve_evidence debug",
        ]
    )

    assert clean_assistant_answer_presentation_text(answer) == "The useful answer stays here."


def test_clean_assistant_answer_presentation_text_removes_agent_trace_json_fence():
    answer = "\n".join(
        [
            "The useful answer stays here.",
            "",
            "```json",
            '{"agent_trace": {"mode": "research_agent"}, "debug": true}',
            "```",
        ]
    )

    assert clean_assistant_answer_presentation_text(answer) == "The useful answer stays here."


def test_clean_assistant_answer_presentation_text_keeps_non_trace_json():
    answer = "\n".join(
        [
            "Example:",
            "```json",
            '{"mode": "research_agent", "enabled": true}',
            "```",
        ]
    )

    assert clean_assistant_answer_presentation_text(answer) == answer
