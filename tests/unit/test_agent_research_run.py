from kb.agent.research_run import build_research_run, infer_source_policy


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
