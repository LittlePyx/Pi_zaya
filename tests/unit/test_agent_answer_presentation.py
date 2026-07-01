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
        [{"text": "Grounded answer. Evidence supports the answer.", "meta": {"source_name": "Paper A"}}],
        settings=_Settings(),
    )

    assert result["llm_used"] is True
    assert result["answer"] == "Grounded answer [1]."
    assert result["quality_gate"]["status"] == "passed"


def test_generate_grounded_answer_can_blend_local_evidence_with_external_context(monkeypatch):
    captured = {}

    class _Settings:
        text_api_key = "test-key"

    class _FakeDeepSeekChat:
        def __init__(self, settings):
            self.settings = settings

        def chat(self, messages, *, temperature=0.2, max_tokens=1200):
            captured["messages"] = messages
            return "Local evidence says retrieval is used [1]. Background: retrieval can improve grounding."

    monkeypatch.setattr(tools, "DeepSeekChat", _FakeDeepSeekChat)

    result = tools.generate_grounded_answer(
        "How does the paper use retrieval?",
        [{"text": "The paper uses retrieval before generation.", "meta": {"source_name": "Paper A"}}],
        settings=_Settings(),
        agent_notes={
            "evidence_gate": {"answer_mode": "hybrid_local_external"},
            "research_run": {
                "status": "synthesizing",
                "source_policy": "local_plus_external_background",
                "metrics": {"evidence_matrix_rows": 1, "local_evidence_hit_count": 1},
            },
            "evidence_matrix": [
                {
                    "paper": "Paper A",
                    "method": "retrieval before generation",
                    "key_result": "improves grounding",
                    "limitation": "latency not evaluated",
                    "evidence_quote": "The paper uses retrieval before generation.",
                    "citation": "[1]",
                    "support_status": "needs_review",
                }
            ],
        },
    )

    assert result["llm_used"] is True
    assert result["answer_mode"] == "hybrid_local_external"
    assert result["quality_gate"]["status"] == "passed"
    assert "local citations [n] come from the knowledge base" in result["answer"]
    assert "Local evidence says retrieval is used [1]." in result["answer"]
    assert "Hybrid answer source policy" in captured["messages"][-1]["content"]
    assert "`evidence_matrix` as the synthesis scaffold" in captured["messages"][-1]["content"]
    assert '"evidence_matrix"' in captured["messages"][-1]["content"]
    assert "latency not evaluated" in captured["messages"][-1]["content"]
    assert "Compact answer shape" in captured["messages"][-1]["content"]
    assert "External context" in captured["messages"][-1]["content"]


def test_generate_grounded_answer_repairs_missing_local_citation(monkeypatch):
    captured = {"calls": 0}

    class _Settings:
        text_api_key = "test-key"

    class _FakeDeepSeekChat:
        def __init__(self, settings):
            self.settings = settings

        def chat(self, messages, *, temperature=0.2, max_tokens=1200):
            captured["calls"] += 1
            captured["messages"] = messages
            if captured["calls"] == 1:
                return "The paper uses retrieval before generation."
            return "The paper uses retrieval before generation [1]."

    monkeypatch.setattr(tools, "DeepSeekChat", _FakeDeepSeekChat)

    result = tools.generate_grounded_answer(
        "How does the paper use retrieval?",
        [{"text": "The paper uses retrieval before generation.", "meta": {"source_name": "Paper A"}}],
        settings=_Settings(),
    )

    assert captured["calls"] == 2
    assert result["answer"] == "The paper uses retrieval before generation [1]."
    assert result["quality_gate"]["status"] == "repaired"
    assert "missing_local_citation" in result["quality_gate"]["reasons"]
    assert "Quality gate reasons" in captured["messages"][-1]["content"]


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
    assert "not a knowledge-base-grounded answer" in result["answer"]
    assert "Python lists are mutable ordered sequences." in result["answer"]
    assert "Retrieved snippets" not in captured["messages"][-1]["content"]


def test_generate_grounded_answer_uses_web_search_for_external_academic_no_hit(monkeypatch):
    class _Settings:
        text_api_key = "test-key"
        agent_web_search_enabled = True
        agent_web_search_api_key = "web-key"
        agent_web_search_model = "gpt-5-search-api"

    class _FakeDeepSeekChat:
        def __init__(self, settings):
            self.settings = settings

        def chat_with_web_search(self, messages, *, temperature=0.2, max_tokens=1200):
            return {
                "content": "Diffusion models learn to reverse a noising process.",
                "annotations": [{"url": "https://example.org/paper", "title": "Example"}],
                "model": "gpt-5-search-api",
            }

    monkeypatch.setattr(tools, "DeepSeekChat", _FakeDeepSeekChat)

    result = tools.generate_grounded_answer(
        "Why do diffusion models work?",
        [],
        settings=_Settings(),
        agent_notes={"evidence_gate": {"answer_mode": "external_academic_llm"}},
    )

    assert result["llm_used"] is True
    assert result["web_search_used"] is True
    assert result["answer_mode"] == "external_academic_llm"
    assert "not a knowledge-base-grounded answer" in result["answer"]
    assert result["web_citations"][0]["url"] == "https://example.org/paper"


def test_generate_grounded_answer_uses_web_search_for_hybrid_answer(monkeypatch):
    class _Settings:
        text_api_key = "test-key"
        agent_web_search_enabled = True
        agent_web_search_api_key = "web-key"
        agent_web_search_model = "gpt-5-search-api"

    class _FakeDeepSeekChat:
        def __init__(self, settings):
            self.settings = settings

        def chat_with_web_search(self, messages, *, temperature=0.2, max_tokens=1200):
            return {
                "content": "The local snippet supports retrieval use [1]. External context: RAG often uses retrieval to reduce unsupported generation.",
                "annotations": [{"url": "https://example.org/rag", "title": "RAG"}],
                "model": "gpt-5-search-api",
            }

    monkeypatch.setattr(tools, "DeepSeekChat", _FakeDeepSeekChat)

    result = tools.generate_grounded_answer(
        "How does the paper use RAG?",
        [{"text": "The paper uses retrieval before generation.", "meta": {"source_name": "Paper A"}}],
        settings=_Settings(),
        agent_notes={"evidence_gate": {"answer_mode": "hybrid_local_external"}},
    )

    assert result["llm_used"] is True
    assert result["answer_mode"] == "hybrid_local_external"
    assert result["web_search_used"] is True
    assert "external model and web context" in result["answer"]
    assert result["web_citations"][0]["url"] == "https://example.org/rag"
