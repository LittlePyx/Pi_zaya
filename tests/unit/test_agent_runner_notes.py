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
        return {
            "answer": "Paper A uses retrieval [1].",
            "observation": "answered",
            "quality_gate": {"status": "repaired", "reasons": ["missing_local_citation"], "warnings": []},
        }

    monkeypatch.setattr(runner, "retrieve_evidence", fake_retrieve_evidence)
    monkeypatch.setattr(runner, "compare_papers", fake_compare_papers)
    monkeypatch.setattr(runner, "generate_grounded_answer", fake_generate_grounded_answer)

    result = runner.run_research_agent("Compare retrieval methods", db_dir=tmp_path)

    assert result["agent_trace"]["question_type"] == "multi_paper_comparison"
    assert result["agent_trace"]["research_run"]["source_policy"] == "local_plus_external_background"
    assert result["agent_trace"]["research_run"]["metrics"]["evidence_matrix_rows"] == 1
    assert captured["agent_notes"]["comparisons"][0]["paper"] == "Paper A"
    assert captured["agent_notes"]["research_run"]["status"] == "synthesizing"
    assert captured["agent_notes"]["research_run"]["source_policy"] == "local_plus_external_background"
    assert captured["agent_notes"]["evidence_matrix"][0]["paper"] == "Paper A"
    assert captured["agent_notes"]["evidence_gate"]["evidence_status"] == "needs_review"
    assert captured["agent_notes"]["evidence_gate"]["answer_mode"] == "hybrid_local_external"
    assert captured["agent_notes"]["evidence_gate"]["source_blend"] == "hybrid_local_external"
    assert captured["agent_notes"]["evidence_gate"]["source_policy"] == "local_plus_external_background"
    assert result["agent_trace"]["summary"]["answer_source_blend"] == "hybrid_local_external"
    assert result["agent_trace"]["summary"]["quality_gate_status"] == "repaired"
    assert result["agent_trace"]["summary"]["quality_gate_reasons"] == ["missing_local_citation"]


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


def test_runner_marks_academic_no_hit_as_external_answer(monkeypatch, tmp_path):
    captured = {}

    def fake_retrieve_evidence(query, *, db_dir, settings=None, top_k=6):
        return {"hits": [], "observation": "empty"}

    def fake_generate_grounded_answer(query, hits, *, settings=None, history=None, agent_notes=None, temperature=0.2, max_tokens=1200):
        captured["agent_notes"] = agent_notes
        return {
            "answer": "Note: no matching local knowledge-base evidence was found; this is an external model answer.",
            "answer_mode": "external_academic_llm",
            "observation": "answered",
        }

    monkeypatch.setattr(runner, "retrieve_evidence", fake_retrieve_evidence)
    monkeypatch.setattr(runner, "generate_grounded_answer", fake_generate_grounded_answer)

    result = runner.run_research_agent("What does the paper prove?", db_dir=tmp_path)

    assert "external model answer" in result["answer"]
    assert captured["agent_notes"]["evidence_gate"]["answer_mode"] == "external_academic_llm"
    assert captured["agent_notes"]["evidence_gate"]["source_blend"] == "external_academic"
    assert captured["agent_notes"]["evidence_gate"]["source_notice"] == "external"
    assert "not_based_on_local_knowledge_base" in captured["agent_notes"]["evidence_gate"]["reasons"]
    assert result["agent_trace"]["verification"]["evidence_status"] == "not_applicable"
    assert result["agent_trace"]["summary"]["evidence_status"] == "not_applicable"
    assert result["agent_trace"]["summary"]["evidence_hit_count"] == 0
    assert result["agent_trace"]["research_run"]["source_policy"] == "external_allowed_with_notice"
    assert result["agent_trace"]["summary"]["evidence_matrix_rows"] == 0
    assert captured["agent_notes"]["research_run"]["status"] == "synthesizing"
    assert captured["agent_notes"]["research_run"]["source_policy"] == "external_allowed_with_notice"
    assert captured["agent_notes"]["evidence_matrix"] == []
    assert result["agent_trace"]["steps"][-1]["status"] == "skipped"


def test_runner_surfaces_tool_errors_even_when_a_partial_answer_exists(monkeypatch, tmp_path):
    def fake_retrieve_evidence(query, *, db_dir, settings=None, top_k=6):
        return {
            "hits": [
                {
                    "text": "The paper uses retrieval before generation to improve grounding.",
                    "score": 3.0,
                    "meta": {"source_name": "Paper A", "source_path": "paper-a.md"},
                }
            ],
            "observation": "retrieved",
        }

    def failed_retrieve_references(*args, **kwargs):
        raise RuntimeError("reference index unavailable")

    def fake_generate_grounded_answer(query, hits, **kwargs):
        return {"answer": "The paper uses retrieval before generation [1].", "observation": "answered"}

    monkeypatch.setattr(runner, "retrieve_evidence", fake_retrieve_evidence)
    monkeypatch.setattr(runner, "retrieve_references", failed_retrieve_references)
    monkeypatch.setattr(runner, "generate_grounded_answer", fake_generate_grounded_answer)

    result = runner.run_research_agent("Which upstream reference should I read?", db_dir=tmp_path)

    assert result["agent_trace"]["status"] == "error"
    assert "retrieve_references" in result["agent_trace"]["errors"][0]
    assert result["agent_trace"]["research_run"]["status"] == "failed"


def test_runner_allows_general_llm_answer_when_query_is_not_about_library(monkeypatch, tmp_path):
    captured = {}

    def fake_retrieve_evidence(query, *, db_dir, settings=None, top_k=6):
        return {"hits": [], "observation": "empty"}

    def fake_generate_grounded_answer(query, hits, *, settings=None, history=None, agent_notes=None, temperature=0.2, max_tokens=1200):
        captured["agent_notes"] = agent_notes
        return {"answer": "Python lists are mutable ordered sequences.", "answer_mode": "general_llm", "observation": "answered"}

    monkeypatch.setattr(runner, "retrieve_evidence", fake_retrieve_evidence)
    monkeypatch.setattr(runner, "generate_grounded_answer", fake_generate_grounded_answer)

    result = runner.run_research_agent("Compare Python lists and tuples.", db_dir=tmp_path)

    assert result["answer"] == "Python lists are mutable ordered sequences."
    assert captured["agent_notes"]["evidence_gate"]["answer_mode"] == "general_llm"
    assert captured["agent_notes"]["evidence_gate"]["source_blend"] == "general_llm"
    assert captured["agent_notes"]["evidence_gate"]["source_notice"] == "none"
    assert captured["agent_notes"]["evidence_gate"]["evidence_status"] == "not_applicable"
    assert result["agent_trace"]["verification"]["evidence_status"] == "not_applicable"
    assert result["agent_trace"]["summary"]["evidence_status"] == "not_applicable"
    assert result["agent_trace"]["steps"][-1]["tool"] == "verify_answer_citations"
    assert result["agent_trace"]["steps"][-1]["status"] == "skipped"


def test_runner_uses_external_academic_mode_for_academic_no_hit(monkeypatch, tmp_path):
    captured = {}

    def fake_retrieve_evidence(query, *, db_dir, settings=None, top_k=6):
        return {"hits": [], "observation": "empty"}

    def fake_generate_grounded_answer(query, hits, *, settings=None, history=None, agent_notes=None, temperature=0.2, max_tokens=1200):
        captured["agent_notes"] = agent_notes
        return {"answer": "External academic answer.", "answer_mode": "external_academic_llm", "observation": "answered"}

    monkeypatch.setattr(runner, "retrieve_evidence", fake_retrieve_evidence)
    monkeypatch.setattr(runner, "generate_grounded_answer", fake_generate_grounded_answer)

    result = runner.run_research_agent("Why do diffusion models work?", db_dir=tmp_path)

    assert result["answer"] == "External academic answer."
    assert captured["agent_notes"]["evidence_gate"]["answer_mode"] == "external_academic_llm"
    assert captured["agent_notes"]["evidence_gate"]["source_blend"] == "external_academic"
    assert captured["agent_notes"]["evidence_gate"]["evidence_status"] == "not_applicable"
    assert result["agent_trace"]["steps"][-1]["status"] == "skipped"


def test_runner_filters_low_confidence_hits_before_external_academic_answer(monkeypatch, tmp_path):
    captured = {}

    def fake_retrieve_evidence(query, *, db_dir, settings=None, top_k=6):
        return {
            "hits": [
                {
                    "text": "A cooking note about tomatoes and basil.",
                    "score": 0.1,
                    "meta": {"source_name": "Recipe", "source_path": "recipe.md"},
                },
                {
                    "text": "A travel checklist for packing bags.",
                    "score": 0.2,
                    "meta": {"source_name": "Travel", "source_path": "travel.md"},
                },
            ],
            "observation": "weak",
        }

    def fake_generate_grounded_answer(query, hits, *, settings=None, history=None, agent_notes=None, temperature=0.2, max_tokens=1200):
        captured["hits"] = hits
        captured["agent_notes"] = agent_notes
        return {"answer": "External academic answer.", "answer_mode": "external_academic_llm", "observation": "answered"}

    monkeypatch.setattr(runner, "retrieve_evidence", fake_retrieve_evidence)
    monkeypatch.setattr(runner, "generate_grounded_answer", fake_generate_grounded_answer)

    result = runner.run_research_agent("What does the paper prove?", db_dir=tmp_path)

    gate = captured["agent_notes"]["evidence_gate"]
    assert captured["hits"] == []
    assert gate["answer_mode"] == "external_academic_llm"
    assert gate["source_blend"] == "external_academic"
    assert gate["evidence_status"] == "not_applicable"
    assert gate["candidate_hit_count"] == 2
    assert gate["retrieval_confidence"] == "low"
    assert "low_retrieval_confidence" in gate["reasons"]
    assert result["agent_trace"]["context"]["scoped_hit_count"] == 2
    assert result["agent_trace"]["context"]["usable_hit_count"] == 0
    assert result["agent_trace"]["context"]["retrieval_confidence"] == "low"
    assert result["agent_trace"]["steps"][0]["output"]["candidate_hits"]


def test_retrieval_confidence_keeps_high_score_semantic_hit():
    confidence = runner._assess_retrieval_confidence(
        "Why do diffusion models work?",
        [
            {
                "text": "The denoising objective learns a reverse process.",
                "score": 9.0,
                "meta": {"source_name": "Diffusion Paper"},
            }
        ],
    )

    assert confidence["level"] == "medium"
    assert confidence["usable_hit_count"] == 1
    assert confidence["usable_hits"][0]["meta"]["source_name"] == "Diffusion Paper"


def test_retrieval_confidence_keeps_scoped_current_paper_hit():
    confidence = runner._assess_retrieval_confidence(
        "What does this paper prove?",
        [
            {
                "text": "The abstract describes the proposed microscope and evaluation.",
                "score": 0.8,
                "meta": {"source_name": "Current Paper", "source_path": "paper.md"},
            }
        ],
        scope_context={"query_scope": "current_paper"},
    )

    assert confidence["level"] == "medium"
    assert confidence["usable_hit_count"] == 1
    assert confidence["signals"][0]["scope_signal"] is True
