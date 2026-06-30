from api.routers.generate import GenerateBody


def test_generate_body_keeps_agent_mode_disabled_by_default():
    body = GenerateBody(conv_id="conv-1", prompt="What is the method?")

    assert body.agent_mode is False


def test_generate_body_accepts_agent_mode_without_requiring_it():
    body = GenerateBody(conv_id="conv-1", prompt="Compare papers", agent_mode=True)

    assert body.agent_mode is True
