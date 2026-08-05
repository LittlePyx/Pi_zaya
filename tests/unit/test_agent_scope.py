from kb.agent import runner


def _hit(source_path: str, source_name: str, text: str) -> dict:
    return {
        "text": text,
        "score": 1.0,
        "meta": {
            "source_path": source_path,
            "source_name": source_name,
            "title": source_name,
            "heading_path": "Results",
        },
    }


def test_research_agent_current_paper_scope_filters_hits(monkeypatch, tmp_path):
    captured: dict[str, list[dict]] = {}

    def fake_retrieve(*args, **kwargs):
        return {
            "observation": "retrieved",
            "hits": [
                _hit("papers/paper-a.md", "Paper A", "Paper A evidence"),
                _hit("papers/paper-b.md", "Paper B", "Paper B evidence"),
            ],
        }

    def fake_answer(query, hits, **kwargs):
        captured["hits"] = list(hits)
        return {"observation": "answered", "answer": "Paper A evidence is relevant [1]."}

    monkeypatch.setattr(runner, "retrieve_evidence", fake_retrieve)
    monkeypatch.setattr(runner, "generate_grounded_answer", fake_answer)

    result = runner.run_research_agent(
        "What does this paper show?",
        db_dir=tmp_path,
        query_scope="current_paper",
        current_source_path="papers/paper-a.md",
    )

    assert [hit["meta"]["source_name"] for hit in captured["hits"]] == ["Paper A"]
    assert result["agent_trace"]["context"]["query_scope"] == "current_paper"
    assert result["agent_trace"]["context"]["retrieved_hit_count"] == 2
    assert result["agent_trace"]["context"]["scoped_hit_count"] == 1
    assert result["agent_trace"]["steps"][0]["output"]["scope_filter"]["after"] == 1


def test_research_agent_basket_scope_filters_hits(monkeypatch, tmp_path):
    captured: dict[str, list[dict]] = {}

    def fake_retrieve(*args, **kwargs):
        captured["source_paths"] = list(kwargs.get("source_paths") or [])
        return {
            "observation": "retrieved",
            "hits": [
                _hit("papers/paper-a.md", "Paper A", "Paper A evidence"),
                _hit("papers/paper-b.md", "Paper B", "Paper B evidence"),
            ],
        }

    def fake_answer(query, hits, **kwargs):
        captured["hits"] = list(hits)
        return {"observation": "answered", "answer": "Paper B evidence is relevant [1]."}

    monkeypatch.setattr(runner, "retrieve_evidence", fake_retrieve)
    monkeypatch.setattr(runner, "generate_grounded_answer", fake_answer)

    result = runner.run_research_agent(
        "Summarize selected work",
        db_dir=tmp_path,
        query_scope="basket",
        selected_research_context={"items": [{"sourcePath": "papers/paper-b.md", "title": "Paper B"}]},
    )

    assert [hit["meta"]["source_name"] for hit in captured["hits"]] == ["Paper B"]
    assert captured["source_paths"] == ["papers/paper-b.md"]
    assert result["agent_trace"]["context"]["query_scope"] == "basket"
    assert result["agent_trace"]["steps"][0]["output"]["scope_filter"]["after"] == 1


def test_retrieve_evidence_balances_explicit_basket_sources(monkeypatch, tmp_path):
    from kb.agent import tools

    chunks = [
        {
            "id": f"a-{index}",
            "text": f"Common reconstruction mechanism result number {index} for Paper A.",
            "meta": {"source_path": "papers/paper-a.md", "evidence_ready": True},
        }
        for index in range(12)
    ]
    chunks.append(
        {
            "id": "b-1",
            "text": "Paper B reports a common reconstruction mechanism with a separate detector.",
            "meta": {"source_path": "papers/paper-b.md", "evidence_ready": True},
        }
    )
    monkeypatch.setattr(tools, "load_all_chunks", lambda _path: chunks)

    result = tools.retrieve_evidence(
        "Compare the common reconstruction mechanism and detector result.",
        db_dir=tmp_path,
        top_k=4,
        source_paths=["papers/paper-a.md", "papers/paper-b.md"],
    )

    sources = [hit["meta"]["source_path"] for hit in result["hits"]]
    assert sources.count("papers/paper-a.md") == 3
    assert sources.count("papers/paper-b.md") == 1
    assert result["requested_source_count"] == 2
