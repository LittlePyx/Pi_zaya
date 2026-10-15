from __future__ import annotations

import json

from kb.answer_finalization_outcome import build_answer_finalization_outcome


def test_answer_finalization_outcome_records_pass_without_content() -> None:
    outcome = build_answer_finalization_outcome({
        "minimum_ok": True,
        "claim_evidence": {
            "final_gate_applied": True,
            "minimum_ok": True,
            "pass_timings_ms": {"initial": 4.2, "recheck": 0.0},
        },
    })

    assert outcome["status"] == "passed"
    assert outcome["gate_pass_count"] == 1
    assert outcome["duration_ms"] == 4.2
    assert outcome["repair_required"] is False
    assert outcome["repair_succeeded"] is None
    assert outcome["action_count"] == 0


def test_answer_finalization_outcome_counts_successful_repairs_and_narrowing() -> None:
    outcome = build_answer_finalization_outcome({
        "minimum_ok": True,
        "claim_evidence": {
            "final_gate_applied": True,
            "minimum_ok": True,
            "repaired_citations": 2,
            "dropped_hard_mismatch_claims": 1,
            "citation_mismatch_claims": 1,
            "post_gate_citation_rebinding": True,
            "pass_timings_ms": {"initial": 4.2, "recheck": 2.1},
            "repairs": [{"claim": "private answer text"}],
        },
    })

    assert outcome["status"] == "repaired_and_narrowed"
    assert outcome["gate_pass_count"] == 2
    assert outcome["duration_ms"] == 6.3
    assert outcome["repair_required"] is True
    assert outcome["repair_succeeded"] is True
    assert outcome["action_count"] == 3
    assert outcome["detected_issue_count"] == 1
    assert "private answer text" not in json.dumps(outcome)


def test_answer_finalization_outcome_records_failed_and_not_applicable() -> None:
    failed = build_answer_finalization_outcome({
        "minimum_ok": False,
        "claim_evidence": {"final_gate_applied": True, "minimum_ok": False},
    })
    skipped = build_answer_finalization_outcome({"minimum_ok": True})

    assert failed["status"] == "failed"
    assert failed["final_ok"] is False
    assert skipped["status"] == "not_applicable"
    assert skipped["final_ok"] is None
    assert skipped["gate_pass_count"] == 0
