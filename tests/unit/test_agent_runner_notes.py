from kb.agent import runner


def test_runner_passes_structured_comparison_notes_to_answer_tool(monkeypatch, tmp_path):
    captured = {}

    def fake_retrieve_evidence(query, *, db_dir, settings=None, top_k=6):
        return {
            "hits": [
                {
                    "text": "Paper A method uses retrieval.",
                    "score": 2.0,
                    "meta": {"source_name": "Paper A", "source_path": "paper-a.md"},
                }
            ],
            "observation": "ok",
        }

    def fake_compare_papers(query, hits):
        return {
            "comparisons": [
                {
                    "paper": "Paper A",
                    "source_name": "Paper A",
                    "source_path": "paper-a.md",
                    "method": "retrieval",
                    "evidence": [],
                    "limitation": "Not identified in the retrieved snippets.",
                    "relation_to_question": "Matches retrieval.",
                    "supporting_headings": [],
                    "hit_count": 1,
                }
            ],
            "observation": "structured",
        }

    def fake_generate_grounded_answer(query, hits, *, settings=None, history=None, agent_notes=None, temperature=0.2, max_tokens=1200):
        captured["agent_notes"] = agent_notes
        return {"answer": "Paper A uses retrieval [1].", "observation": "answered"}

    monkeypatch.setattr(runner, "retrieve_evidence", fake_retrieve_evidence)
    monkeypatch.setattr(runner, "compare_papers", fake_compare_papers)
    monkeypatch.setattr(runner, "generate_grounded_answer", fake_generate_grounded_answer)

    result = runner.run_research_agent("Compare retrieval methods", db_dir=tmp_path)

    assert result["agent_trace"]["question_type"] == "multi_paper_comparison"
    assert captured["agent_notes"]["comparisons"][0]["paper"] == "Paper A"


def test_runner_passes_reference_notes_to_answer_tool(monkeypatch, tmp_path):
    captured = {}

    def fake_retrieve_evidence(query, *, db_dir, settings=None, top_k=6):
        return {
            "hits": [
                {
                    "text": "Paper A cites the upstream method [7].",
                    "score": 2.0,
                    "meta": {"source_name": "Paper A", "source_path": "paper-a.md"},
                }
            ],
            "observation": "ok",
        }

    def fake_retrieve_references(query, hits, *, db_dir=None, settings=None, top_k=6):
        captured["db_dir"] = db_dir
        return {
            "references": [
                {
                    "source_paper": "Paper A",
                    "source_path": "paper-a.md",
                    "ref_num": 7,
                    "title": "Upstream Method Paper",
                    "why_relevant": "Matches upstream method.",
                    "reference_index_available": True,
                }
            ],
            "observation": "resolved",
        }

    def fake_generate_grounded_answer(query, hits, *, settings=None, history=None, agent_notes=None, temperature=0.2, max_tokens=1200):
        captured["agent_notes"] = agent_notes
        return {"answer": "Read the upstream method paper [7].", "observation": "answered"}

    monkeypatch.setattr(runner, "retrieve_evidence", fake_retrieve_evidence)
    monkeypatch.setattr(runner, "retrieve_references", fake_retrieve_references)
    monkeypatch.setattr(runner, "generate_grounded_answer", fake_generate_grounded_answer)

    result = runner.run_research_agent("Which upstream reference should I read?", db_dir=tmp_path)

    assert result["agent_trace"]["question_type"] == "reference_followup"
    assert captured["db_dir"] == tmp_path
    assert captured["agent_notes"]["references"][0]["title"] == "Upstream Method Paper"
