from __future__ import annotations

import json
from pathlib import Path

from tools.research_qa.build_quality_outcome_report import (
    _visible_unresolved_rows,
    build_report,
    validate_report,
)


def _write_report(root: Path, row: dict) -> Path:
    root.mkdir(parents=True)
    raw_path = root / "raw_results.jsonl"
    raw_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "total": 1,
        "passed": int(bool(row["quality"]["ok"])),
        "failed": int(not bool(row["quality"]["ok"])),
        "timing": {},
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return root


def _quality_row() -> dict:
    check_names = {
        "answer_directly_addresses_question",
        "answer_contains_required_terms",
        "answer_contains_required_term_groups",
        "claims_have_matching_evidence",
        "answer_avoids_forbidden_claims",
        "primary_evidence_avoids_forbidden_terms",
        "answer_claims_have_inline_citations",
        "answer_high_risk_claims_are_cited",
        "citations_include_required_docs",
        "citations_avoid_unexpected_docs",
        "citations_match_required_routes",
        "citation_card_quality",
        "system_b_audit",
        "citations_have_expected_locators",
        "answer_no_unresolved_citation_markers",
        "refs_card_copy_quality",
    }
    return {
        "id": "case-1",
        "quality": {
            "ok": True,
            "checks": [
                {"name": name, "ok": True, "detail": {}}
                for name in sorted(check_names)
            ],
        },
        "assistant_message": {
            "content": "Grounded answer [1].",
            "meta": {
                "answer_quality": {
                    "claim_evidence": {
                        "repaired_citations": 2,
                        "dropped_hard_mismatch_claims": 1,
                        "uncited_high_risk_claims": 0,
                    }
                }
            },
        },
        "final_payload": {"answer": "Grounded answer [1].", "answer_runtime_check": {}},
        "refs_payload": {"1": {"cite_details": [{"card_warning": ""}]}},
    }


def test_build_report_aggregates_quality_actions_and_observability(tmp_path: Path) -> None:
    report_dir = _write_report(tmp_path / "report", _quality_row())

    report = build_report([("fixture", report_dir)])

    assert report["live_outcomes"]["case_count"] == 1
    assert report["live_outcomes"]["failed"] == 0
    assert all(
        category["failed"] == 0
        for category in report["live_outcomes"]["categories"].values()
    )
    resolution = report["automatic_resolution"]
    assert resolution["observed_finalization_action_count"] == 3
    assert resolution["observed_finalization_action_cases"] == 1
    assert resolution["attempt_count"] is None
    assert resolution["final_user_visible_unresolved_count"] == 0
    assert report["validation"]["ok"] is True


def test_visible_scan_finds_rendered_pending_copy_but_ignores_internal_status() -> None:
    row = _quality_row()
    row["refs_payload"] = {
        "status": "pending verification",
        "cards": [{"summary_title": "待核验的原文片段"}],
    }

    matches = _visible_unresolved_rows(row)

    assert matches == [{"field": "refs.cards[0].summary_title", "matched_text": "待核验"}]


def test_validation_rejects_visible_unresolved_state(tmp_path: Path) -> None:
    row = _quality_row()
    row["assistant_message"]["content"] = "Verification failed; try later."
    report_dir = _write_report(tmp_path / "report", row)
    report = build_report([("fixture", report_dir)])

    errors = validate_report(report, strict_committed=False)

    assert "final user-visible unresolved verifier copy remains" in errors


def test_build_report_uses_stable_finalization_outcome_when_every_case_has_it(tmp_path: Path) -> None:
    row = _quality_row()
    row["assistant_message"]["meta"]["answer_quality"]["finalization_outcome"] = {
        "schema_version": 1,
        "status": "repaired",
        "gate_pass_count": 2,
        "repair_required": True,
        "repair_succeeded": True,
    }
    report_dir = _write_report(tmp_path / "report", row)

    report = build_report([("fixture", report_dir)])
    resolution = report["automatic_resolution"]

    assert resolution["stable_outcome_observed_cases"] == 1
    assert resolution["stable_outcome_status_counts"] == {"repaired": 1}
    assert resolution["attempt_count"] == 2
    assert resolution["attempt_observability"] == "answer_finalization_outcome_v1"
    assert resolution["stable_repair_success_rate"] == 1.0
    assert report["validation"]["ok"] is True
