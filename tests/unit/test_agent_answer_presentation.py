from kb.agent import tools


def test_generate_grounded_answer_cleans_llm_trace_suffix(monkeypatch):
    class _Settings:
        text_api_key = "test-key"

    class _FakeDeepSeekChat:
        def __init__(self, settings):
            self.settings = settings

        def chat(self, messages, *, temperature=0.2, max_tokens=1200):
            return "Grounded answer [1].\n\nResearch Agent Trace\nPlan\n- retrieve_evidence debug"

    monkeypatch.setattr(tools, "DeepSeekChat", _FakeDeepSeekChat)

    result = tools.generate_grounded_answer(
        "What is supported?",
        [{"text": "Evidence supports the answer.", "meta": {"source_name": "Paper A"}}],
        settings=_Settings(),
    )

    assert result["llm_used"] is True
    assert result["answer"] == "Grounded answer [1]."
