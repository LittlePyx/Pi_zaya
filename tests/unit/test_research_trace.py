from __future__ import annotations

from kb.research_trace import add_event, compact_trace, finish_trace, new_trace, summarize_hits


def test_research_trace_records_timings_and_compacts_sources():
    trace = new_trace(
        session_id="s1",
        task_id="t1",
        conv_id="c1",
        user_msg_id=10,
        assistant_msg_id=11,
        trace_id="trace-1",
        prompt_sig="abc",
    )
    trace = add_event(trace, "retrieve", elapsed_s=0.123, hit_count=2)
    trace["retrieval"] = {
        "top_hits": summarize_hits(
            [
                {"score": 1.5, "meta": {"source_path": "F:/papers/a.md", "heading_path": "Intro"}},
                {"score": 1.2, "meta": {"source_path": "F:/papers/b.md", "heading_path": "Method"}},
            ],
            limit=5,
        )
    }
    out = compact_trace(finish_trace(trace, status="done", total_elapsed_s=1.25), max_sources=1)
    assert out["trace_id"] == "trace-1"
    assert out["status"] == "done"
    assert out["timings_ms"]["retrieve"] == 123.0
    assert out["timings_ms"]["total"] == 1250.0
    assert len(out["retrieval"]["top_hits"]) == 1
    assert out["retrieval"]["top_hits"][0]["source_name"] == "a.md"
