from __future__ import annotations

import json
import sqlite3

from tools.research_qa.export_research_agent_samples import export_research_agent_samples
from tools.research_qa.run_agent_trace_eval import evaluate_quality_cases


def _create_chat_db(path):
    trace = {
        "mode": "research_agent",
        "question_type": "single_paper_qa",
        "summary": {
            "answer_mode": "hybrid_local_external",
            "answer_source_blend": "hybrid_local_external",
        },
        "steps": [
            {
                "tool": "retrieve_evidence",
                "output": {
                    "hits": [
                        {
                            "text": "Paper A reports that retrieval improves citation coverage.",
                            "score": 0.92,
                            "meta": {
                                "source_name": "paper-a.md",
                                "source_path": "paper-a.md",
                                "heading_path": ["Results"],
                            },
                        }
                    ]
                },
            }
        ],
    }
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                conv_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                attachments_json TEXT NOT NULL DEFAULT '[]',
                meta_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO messages (id, conv_id, role, content, attachments_json, meta_json, created_at)
            VALUES (?, ?, ?, ?, '[]', '{}', ?)
            """,
            (1, "conv-1", "user", "What does Paper A say about citation coverage?", 1.0),
        )
        conn.execute(
            """
            INSERT INTO messages (id, conv_id, role, content, attachments_json, meta_json, created_at)
            VALUES (?, ?, ?, ?, '[]', ?, ?)
            """,
            (
                2,
                "conv-1",
                "assistant",
                (
                    "Using local citations from the knowledge base plus external model context, "
                    "Paper A reports retrieval improves citation coverage [1]."
                ),
                json.dumps({"agent_trace": trace}),
                2.0,
            ),
        )


def test_export_research_agent_samples_can_be_replayed_by_quality_eval(tmp_path):
    db_path = tmp_path / "chat.sqlite3"
    out_path = tmp_path / "real_samples.jsonl"
    _create_chat_db(db_path)

    summary = export_research_agent_samples(db_path=db_path, out_path=out_path, limit=10)

    assert summary["ok"] is True
    assert summary["sample_count"] == 1
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["sample_kind"] == "real_chat_replay"
    assert rows[0]["source_blend"] == "hybrid_local_external"
    assert rows[0]["should_use_local_evidence"] is False
    assert rows[0]["evidence_hits"][0]["meta"]["source_path"] == "paper-a.md"

    quality = evaluate_quality_cases(out_path)

    assert quality["ok"] is True, quality["errors"]
    assert quality["case_count"] == 1
    assert quality["real_replay_case_count"] == 1
    assert quality["required_notice_accuracy"] == 1.0
    assert quality["source_blend_accuracy"] is None


def test_export_research_agent_samples_uses_message_refs_fallback(tmp_path):
    db_path = tmp_path / "chat.sqlite3"
    out_path = tmp_path / "real_samples.jsonl"
    trace = {
        "mode": "research_agent",
        "question_type": "single_paper_qa",
        "summary": {
            "answer_mode": "evidence_grounded",
            "answer_source_blend": "local_grounded",
        },
    }
    message_ref_hit = {
        "text": "Paper B reports that the retrieval index is deterministic.",
        "score": 0.81,
        "meta": {"source_name": "paper-b.md", "source_path": "paper-b.md"},
    }
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                conv_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                attachments_json TEXT NOT NULL DEFAULT '[]',
                meta_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute("CREATE TABLE message_refs (user_msg_id INTEGER PRIMARY KEY, hits_json TEXT NOT NULL)")
        conn.execute(
            """
            INSERT INTO messages (id, conv_id, role, content, attachments_json, meta_json, created_at)
            VALUES (10, 'conv-2', 'user', 'What does Paper B say?', '[]', '{}', 1.0)
            """
        )
        conn.execute(
            """
            INSERT INTO messages (id, conv_id, role, content, attachments_json, meta_json, created_at)
            VALUES (?, ?, ?, ?, '[]', ?, ?)
            """,
            (
                11,
                "conv-2",
                "assistant",
                "Paper B reports that the retrieval index is deterministic [1].",
                json.dumps({"agent_trace": trace}),
                2.0,
            ),
        )
        conn.execute(
            "INSERT INTO message_refs (user_msg_id, hits_json) VALUES (?, ?)",
            (10, json.dumps([message_ref_hit])),
        )

    summary = export_research_agent_samples(db_path=db_path, out_path=out_path, limit=10)

    assert summary["sample_count"] == 1
    row = json.loads(out_path.read_text(encoding="utf-8").strip())
    assert row["expected_retrieval_hit"] is True
    assert row["evidence_hits"][0]["meta"]["source_path"] == "paper-b.md"


def test_export_research_agent_samples_handles_missing_db(tmp_path):
    out_path = tmp_path / "real_samples.jsonl"

    summary = export_research_agent_samples(db_path=tmp_path / "missing.sqlite3", out_path=out_path)

    assert summary["ok"] is True
    assert summary["reason"] == "missing_db"
    assert summary["sample_count"] == 0
    assert out_path.read_text(encoding="utf-8") == ""
