from kb.agent.research_run import build_evidence_matrix, build_research_run, infer_source_policy


def test_research_run_builds_evidence_matrix_from_comparison_notes():
    run = build_research_run(
        "Compare the two methods.",
        question_type="multi_paper_comparison",
        hits=[
            {
                "text": "Paper A reports retrieval-augmented answering.",
                "score": 2.0,
                "meta": {"source_name": "Paper A", "source_path": "paper-a.md"},
            }
        ],
        agent_notes={
            "evidence_gate": {"answer_mode": "hybrid_local_external"},
            "comparisons": [
                {
                    "paper": "Paper A",
                    "source_name": "Paper A",
                    "source_path": "paper-a.md",
                    "method": "retrieval-augmented answering",
                    "relation_to_question": "Directly compares retrieval behavior.",
                    "limitation": "No latency analysis in the retrieved evidence.",
                    "evidence": [
                        {
                            "text_preview": "Paper A reports retrieval-augmented answering.",
                            "heading_path": "Paper A / Method",
                        }
                    ],
                }
            ],
        },
        scope_context={"query_scope": "library"},
        verification_status="needs_review",
    )

    payload = run.to_dict()

    assert payload["status"] == "verified"
    assert payload["source_policy"] == "local_plus_external_background"
    assert payload["metrics"]["evidence_matrix_rows"] == 1
    assert payload["evidence_matrix"][0]["paper"] == "Paper A"
    assert payload["evidence_matrix"][0]["method"] == "retrieval-augmented answering"
    assert payload["evidence_matrix"][0]["support_status"] == "needs_review"
    assert payload["subtasks"][2]["tool"] == "compare_papers"


def test_research_run_marks_external_policy_when_no_local_hits():
    policy = infer_source_policy(
        hits=[],
        agent_notes={"evidence_gate": {"answer_mode": "external_academic_llm"}},
    )

    assert policy == "external_allowed_with_notice"


def test_research_run_prefers_explicit_source_policy_from_gate():
    policy = infer_source_policy(
        hits=[{"text": "Local evidence", "meta": {"source_name": "Paper A"}}],
        agent_notes={
            "evidence_gate": {
                "answer_mode": "hybrid_local_external",
                "source_blend": "hybrid_local_external",
                "source_policy": "local_plus_external_background",
            }
        },
    )

    assert policy == "local_plus_external_background"


def test_build_evidence_matrix_can_be_used_before_answer_generation():
    matrix = build_evidence_matrix(
        hits=[
            {
                "text": "The method uses sparse coded illumination and reports faster capture.",
                "score": 4.0,
                "meta": {
                    "source_name": "Paper B",
                    "source_path": "paper-b.md",
                    "heading_path": "Results",
                },
            }
        ],
        verification_status="needs_review",
    )
    run = build_research_run(
        "Summarize Paper B.",
        question_type="single_paper_qa",
        hits=[],
        agent_notes={"evidence_gate": {"answer_mode": "hybrid_local_external"}},
        verification_status="needs_review",
        status="synthesizing",
    )

    assert matrix[0].paper == "Paper B"
    assert matrix[0].support_status == "needs_review"
    assert run.status == "synthesizing"


def test_research_run_id_uses_runtime_identity_and_evidence_fallback():
    common = {
        "query": "Summarize the paper.",
        "question_type": "single_paper_qa",
        "agent_notes": {},
        "verification_status": "needs_review",
    }
    first = build_research_run(
        **common,
        hits=[{"text": "Evidence A", "meta": {"source_path": "paper-a.md", "block_id": "a"}}],
        scope_context={"query_scope": "library", "task_id": "task-1"},
    )
    same_runtime = build_research_run(
        **common,
        hits=[{"text": "Different evidence", "meta": {"source_path": "paper-b.md", "block_id": "b"}}],
        scope_context={"query_scope": "library", "task_id": "task-1"},
    )
    second_runtime = build_research_run(
        **common,
        hits=[{"text": "Evidence A", "meta": {"source_path": "paper-a.md", "block_id": "a"}}],
        scope_context={"query_scope": "library", "task_id": "task-2"},
    )
    evidence_a = build_research_run(
        **common,
        hits=[{"text": "Evidence A", "meta": {"source_path": "paper-a.md", "block_id": "a"}}],
        scope_context={"query_scope": "library"},
    )
    evidence_b = build_research_run(
        **common,
        hits=[{"text": "Evidence B", "meta": {"source_path": "paper-b.md", "block_id": "b"}}],
        scope_context={"query_scope": "library"},
    )

    assert first.run_id == same_runtime.run_id
    assert first.run_id != second_runtime.run_id
    assert evidence_a.run_id != evidence_b.run_id
    assert first.run_id.startswith("rr_")
