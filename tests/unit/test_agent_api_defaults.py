from api.routers.generate import GenerateBody, _generation_user_meta
from api.contracts.research_agent import ResearchAgentResponse
from api.routers import chat as chat_router
from api.routers.chat import ResearchAgentBody


def test_generate_body_keeps_agent_mode_disabled_by_default():
    body = GenerateBody(conv_id="conv-1", prompt="What is the method?")

    assert body.agent_mode is False


def test_generate_body_accepts_agent_mode_without_requiring_it():
    body = GenerateBody(conv_id="conv-1", prompt="Compare papers", agent_mode=True)

    assert body.agent_mode is True


def test_generation_user_meta_omits_agent_fields_by_default():
    meta = _generation_user_meta(None, "library", False)

    assert meta == {"query_scope": "library"}


def test_generation_user_meta_records_explicit_agent_request():
    meta = _generation_user_meta({"items": []}, "basket", True)

    assert meta["agent_mode"] == "research_agent"
    assert meta["agent_mode_requested"] is True
    assert meta["query_scope"] == "basket"
    assert meta["prompt_context"] == {"items": []}


def test_research_agent_body_accepts_scope_without_requiring_it():
    body = ResearchAgentBody(
        prompt="Compare selected papers",
        query_scope="basket",
        prompt_context={"items": [{"title": "Paper A", "sourcePath": "paper-a.md"}]},
    )

    assert body.query_scope == "basket"
    assert body.prompt_context["items"][0]["title"] == "Paper A"


def test_research_agent_route_returns_typed_public_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(chat_router, "get_settings", lambda: type("Settings", (), {"db_dir": tmp_path})())
    monkeypatch.setattr(
        chat_router,
        "run_research_agent",
        lambda *args, **kwargs: {"answer": "A concise answer.", "agent_trace": {}, "hits": []},
    )

    result = chat_router.run_chat_research_agent(ResearchAgentBody(query="What does the paper show?"))

    assert isinstance(result, ResearchAgentResponse)
    assert result.answer == "A concise answer."
    assert result.agent_trace == {}
    assert result.hits == []
