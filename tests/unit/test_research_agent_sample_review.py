from __future__ import annotations

import json

from tools.research_qa.review_research_agent_samples import (
    merge_review_labels,
    prepare_review_labels,
)
from tools.research_qa.run_agent_trace_eval import build_eval_report, evaluate_quality_cases


def _write_sample(path):
    sample = {
        "id": "chat-101",
        "sample_kind": "real_chat_replay",
        "query": "What does Paper C say about citation grounding?",
        "answer": "Paper C shows anchored citations improve citation grounding [1].",
        "answer_mode": "evidence_grounded",
        "source_blend": "local_grounded",
        "agent_trace": {
            "mode": "research_agent",
            "question_type": "single_paper_qa",
            "summary": {
                "answer_mode": "evidence_grounded",
                "answer_source_blend": "local_grounded",
            },
        },
        "evidence_hits": [
            {
                "text": "Paper C shows anchored citations improve citation grounding.",
                "score": 0.9,
                "meta": {"source_name": "paper-c.md", "source_path": "paper-c.md"},
            }
        ],
        "expected_retrieval_hit": True,
        "should_use_local_evidence": False,
        "external_fallback_allowed": False,
        "expected_answer_points": [],
        "expected_source_keywords": [],
        "expected_user_notice": "none",
        "replay_unlabeled": True,
    }
    path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")


def test_review_labels_merge_into_eval_ready_dataset(tmp_path):
    samples_path = tmp_path / "samples.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    reviewed_path = tmp_path / "reviewed.jsonl"
    _write_sample(samples_path)

    prepare_summary = prepare_review_labels(samples_path=samples_path, labels_path=labels_path)

    assert prepare_summary["ok"] is True
    label = json.loads(labels_path.read_text(encoding="utf-8").strip())
    assert label["review_status"] == "needs_review"
    assert label["source_blend_observed"] == "local_grounded"
    assert label["evidence_preview"][0]["source_path"] == "paper-c.md"

    label.update(
        {
            "review_status": "accepted",
            "expected_source_blend": "local_grounded",
            "expected_answer_points": ["anchored citations", "citation grounding"],
            "expected_source_keywords": ["paper-c.md"],
            "should_use_local_evidence": True,
        }
    )
    labels_path.write_text(json.dumps(label, ensure_ascii=False) + "\n", encoding="utf-8")

    merge_summary = merge_review_labels(
        samples_path=samples_path,
        labels_path=labels_path,
        out_path=reviewed_path,
    )

    assert merge_summary["ok"] is True, merge_summary["errors"]
    assert merge_summary["reviewed_case_count"] == 1
    row = json.loads(reviewed_path.read_text(encoding="utf-8").strip())
    assert row["sample_kind"] == "real_chat_reviewed"
    assert row["replay_unlabeled"] is False
    assert row["expected_source_blend"] == "local_grounded"
    assert row["should_use_local_evidence"] is True

    quality = evaluate_quality_cases(reviewed_path)
    assert quality["ok"] is True, quality["errors"]
    assert quality["real_replay_case_count"] == 0
    assert quality["real_reviewed_case_count"] == 1
    assert quality["source_blend_accuracy"] == 1.0
    assert quality["expected_answer_point_coverage"] == 1.0
    assert quality["expected_source_hit_rate"] == 1.0

    report = build_eval_report(
        {"case_count": 0, "ok": True, "planning_errors": [], "schema_errors": [], "question_types": {}},
        quality_summary=quality,
        commit="test-commit",
        date="2026-01-01T00:00:00+00:00",
    )
    assert report["num_real_reviewed_cases"] == 1
    assert report["num_real_replay_cases"] == 0


def test_merge_review_labels_rejects_accepted_label_without_answer_points(tmp_path):
    samples_path = tmp_path / "samples.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    reviewed_path = tmp_path / "reviewed.jsonl"
    _write_sample(samples_path)
    prepare_review_labels(samples_path=samples_path, labels_path=labels_path)
    label = json.loads(labels_path.read_text(encoding="utf-8").strip())
    label.update({"review_status": "accepted", "expected_source_blend": "local_grounded"})
    labels_path.write_text(json.dumps(label, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = merge_review_labels(samples_path=samples_path, labels_path=labels_path, out_path=reviewed_path)

    assert summary["ok"] is False
    assert summary["reviewed_case_count"] == 0
    assert any("expected_answer_points" in error for error in summary["errors"])
