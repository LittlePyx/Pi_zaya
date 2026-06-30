from api.routers.generate import GenerateBody
from api.routers.chat import ResearchAgentBody


def test_generate_body_keeps_agent_mode_disabled_by_default():
    body = GenerateBody(conv_id="conv-1", prompt="What is the method?")

    assert body.agent_mode is False


def test_generate_body_accepts_agent_mode_without_requiring_it():
    body = GenerateBody(conv_id="conv-1", prompt="Compare papers", agent_mode=True)

    assert body.agent_mode is True


def test_research_agent_body_accepts_scope_without_requiring_it():
    body = ResearchAgentBody(
        prompt="Compare selected papers",
        query_scope="basket",
        prompt_context={"items": [{"title": "Paper A", "sourcePath": "paper-a.md"}]},
    )

    assert body.query_scope == "basket"
    assert body.prompt_context["items"][0]["title"] == "Paper A"
