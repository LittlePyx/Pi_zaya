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


def test_generate_grounded_answer_skips_llm_without_evidence(monkeypatch):
    class _Settings:
        text_api_key = "test-key"

    class _FakeDeepSeekChat:
        def __init__(self, settings):
            raise AssertionError("LLM should not be called without evidence")

    monkeypatch.setattr(tools, "DeepSeekChat", _FakeDeepSeekChat)

    result = tools.generate_grounded_answer("What is supported?", [], settings=_Settings())

    assert result["llm_used"] is False
    assert "No relevant indexed evidence" in result["answer"]


def test_generate_grounded_answer_allows_general_llm_without_evidence(monkeypatch):
    captured = {}

    class _Settings:
        text_api_key = "test-key"

    class _FakeDeepSeekChat:
        def __init__(self, settings):
            self.settings = settings

        def chat(self, messages, *, temperature=0.2, max_tokens=1200):
            captured["messages"] = messages
            return "Python lists are mutable ordered sequences."

    monkeypatch.setattr(tools, "DeepSeekChat", _FakeDeepSeekChat)

    result = tools.generate_grounded_answer(
        "How do Python lists work?",
        [],
        settings=_Settings(),
        agent_notes={"evidence_gate": {"answer_mode": "general_llm"}},
    )

    assert result["llm_used"] is True
    assert result["answer_mode"] == "general_llm"
    assert result["answer"] == "Python lists are mutable ordered sequences."
    assert "Retrieved snippets" not in captured["messages"][-1]["content"]
