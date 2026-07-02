from __future__ import annotations

import json

from tools.research_qa.run_reviewed_replay_eval import run_reviewed_replay_eval


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _reviewed_case(case_id: str = "chat-reviewed-1") -> dict:
    return {
        "id": case_id,
        "sample_kind": "real_chat_reviewed",
        "review_status": "accepted",
        "replay_unlabeled": False,
        "query": "What does Paper D say about source policy?",
        "answer": "Paper D says source policy separates local evidence from external model context [1].",
        "answer_mode": "evidence_grounded",
        "source_blend": "local_grounded",
        "expected_source_blend": "local_grounded",
        "evidence_hits": [
            {
                "text": "Paper D says source policy separates local evidence from external model context.",
                "score": 0.95,
                "meta": {"source_name": "paper-d.md", "source_path": "paper-d.md"},
            }
        ],
        "expected_retrieval_hit": True,
        "should_use_local_evidence": True,
        "external_fallback_allowed": False,
        "expected_answer_points": ["source policy", "local evidence", "external model context"],
        "expected_source_keywords": ["paper-d.md"],
        "expected_user_notice": "none",
    }


def test_reviewed_replay_eval_skips_missing_default_paths(tmp_path):
    summary = run_reviewed_replay_eval(paths=[tmp_path / "missing.jsonl"])

    assert summary["ok"] is True
    assert summary["reviewed_case_count"] == 0
    assert summary["evaluated_dataset_count"] == 0
    assert summary["skipped"][0]["reason"] == "missing"


def test_reviewed_replay_eval_runs_committed_deidentified_fixture():
    summary = run_reviewed_replay_eval()

    assert summary["ok"] is True, summary["errors"]
    assert summary["reviewed_case_count"] >= 2
    assert any(
        item["path"].replace("\\", "/") == "docs/research_agent_reviewed_replay.jsonl"
        for item in summary["evaluated"]
    )


def test_reviewed_replay_eval_can_require_reviewed_cases(tmp_path):
    summary = run_reviewed_replay_eval(paths=[tmp_path / "missing.jsonl"], require_reviewed=True)

    assert summary["ok"] is False
    assert any("no reviewed replay cases" in error for error in summary["errors"])


def test_reviewed_replay_eval_runs_strict_quality_gate(tmp_path):
    reviewed_path = tmp_path / "reviewed.jsonl"
    _write_jsonl(reviewed_path, [_reviewed_case()])

    summary = run_reviewed_replay_eval(paths=[reviewed_path])

    assert summary["ok"] is True, summary["errors"]
    assert summary["reviewed_case_count"] == 1
    assert summary["evaluated_dataset_count"] == 1
    report = summary["evaluated"][0]["report"]
    assert report["quality_eval_ok"] is True
    assert report["num_real_reviewed_cases"] == 1
    assert report["source_blend_accuracy"] == 1.0


def test_reviewed_replay_eval_rejects_mixed_unreviewed_dataset(tmp_path):
    reviewed_path = tmp_path / "reviewed.jsonl"
    unreviewed = dict(_reviewed_case("chat-unreviewed-1"))
    unreviewed["sample_kind"] = "real_chat_replay"
    unreviewed["review_status"] = "needs_review"
    unreviewed["replay_unlabeled"] = True
    _write_jsonl(reviewed_path, [_reviewed_case(), unreviewed])

    summary = run_reviewed_replay_eval(paths=[reviewed_path])

    assert summary["ok"] is False
    assert any("expects only accepted reviewed cases" in error for error in summary["errors"])
