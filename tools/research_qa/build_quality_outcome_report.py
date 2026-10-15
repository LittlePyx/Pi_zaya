from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kb.research_note_evidence import auto_resolve_research_note_source_state
from tools.research_qa.run_agent_trace_eval import evaluate_quality_cases
from tools.research_qa.run_research_qa_eval import (
    DEFAULT_FIXTURE,
    DEFAULT_REPLAY,
    evaluate_replay_rows,
    load_fixture,
    load_replay,
)
from tools.research_qa.run_reviewed_replay_eval import run_reviewed_replay_eval


DEFAULT_OUTPUT = Path("docs/research_qa_quality_outcomes_v1.json")
DEFAULT_INPUTS = (
    (
        "main_56",
        Path("test_results/research_qa_long_tail_full_56_release_verified/20260811_031858"),
    ),
    (
        "unseen_v2_22",
        Path(
            "F:/research-papers/2026/Jan/else/kb_chat_eval/unseen_v2/results/"
            "full_22_release/20260814_070530"
        ),
    ),
)

COMMITTED_SOURCE_CONTRACTS = {
    "main_56": {
        "case_count": 56,
        "raw_sha256": "64b46e4136e168c29b9aa65105e3eb43c92d1f6c524a8a92618e6b1df64dee86",
        "summary_sha256": "20511de463567203df0882fdd9a521166eed1ab48b5139f6701d35fb1ce608e3",
    },
    "unseen_v2_22": {
        "case_count": 22,
        "raw_sha256": "52be5477728fc196ce7f62a9c44dfc1b4fdb5be1c75db5aebd2bd809cc4ec933",
        "summary_sha256": "ace4fee311abf1b2ac7b081aefc0f199adeab496bdf9722a1e0f178fa12e7c74",
    },
}

CATEGORY_CHECKS = {
    "answer_correctness": {
        "answer_directly_addresses_question",
        "answer_contains_required_terms",
        "answer_contains_required_term_groups",
    },
    "faithfulness": {
        "claims_have_matching_evidence",
        "answer_avoids_forbidden_claims",
        "primary_evidence_avoids_forbidden_terms",
    },
    "citation_completeness": {
        "answer_claims_have_inline_citations",
        "answer_high_risk_claims_are_cited",
        "citations_include_required_docs",
    },
    "citation_precision_and_routing": {
        "citations_avoid_unexpected_docs",
        "citations_match_required_routes",
        "citation_card_quality",
        "system_b_audit",
    },
    "locator_accuracy": {"citations_have_expected_locators"},
    "contradiction_boundary": {
        "answer_avoids_forbidden_claims",
        "primary_evidence_avoids_forbidden_terms",
    },
    "user_visible_integrity": {
        "answer_no_unresolved_citation_markers",
        "refs_card_copy_quality",
    },
}

FINALIZATION_ACTION_FIELDS = (
    "repaired_citations",
    "rebound_citations",
    "section_source_rebound_citations",
    "repaired_modality_boundaries",
    "relocated_midphrase_citations",
    "restored_prompt_terms",
    "restored_evidence_numbers",
    "restored_source_facts",
    "added_requested_facet_boundaries",
    "trimmed_unsupported_inferences",
    "dropped_cross_source_claims",
    "dropped_hard_mismatch_claims",
    "stripped_weak_citations",
    "renderer_rejected_citations",
    "removed_unplanned_citations",
    "removed_heading_citations",
    "dropped_unsupported_unplanned_claims",
    "dropped_unsupported_inferences",
    "dropped_placeholder_sections",
)

DETECTION_FIELDS = (
    "uncited_high_risk_claims",
    "citation_mismatch_claims",
)

VISIBLE_TEXT_KEYS = {
    "card_title",
    "card_subtitle",
    "card_takeaway_label",
    "card_takeaway",
    "card_claim_label",
    "card_claim",
    "card_locator_label",
    "card_locator",
    "card_evidence_label",
    "card_evidence",
    "card_context_summary",
    "card_reference_label",
    "card_reference_entry",
    "card_support_label",
    "card_support_explanation",
    "card_quality_label",
    "card_warning",
    "summary_title",
    "summary_basis",
    "why_basis",
    "summary_line",
    "why_line",
    "location_label",
    "kicker",
    "label",
    "text",
    "hint",
    "warning",
    "title",
    "subtitle",
    "summary",
}

UNRESOLVED_VISIBLE_RE = re.compile(
    r"待核验|核验失败|pending\s+verification|verification\s+failed|"
    r"source\s+passage\s+being\s+checked|provisional\s+source",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected an object")
        rows.append(row)
    return rows


def _visible_strings(value: Any, *, path: str = "refs") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in VISIBLE_TEXT_KEYS and isinstance(child, str) and child.strip():
                yield child_path, child
            elif isinstance(child, (dict, list)):
                yield from _visible_strings(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, (dict, list)):
                yield from _visible_strings(child, path=f"{path}[{index}]")


def _visible_unresolved_rows(row: dict[str, Any]) -> list[dict[str, str]]:
    candidates: list[tuple[str, str]] = []
    assistant = row.get("assistant_message")
    if isinstance(assistant, dict) and isinstance(assistant.get("content"), str):
        candidates.append(("assistant_message.content", assistant["content"]))
    final_payload = row.get("final_payload")
    if isinstance(final_payload, dict) and isinstance(final_payload.get("answer"), str):
        candidates.append(("final_payload.answer", final_payload["answer"]))
    candidates.extend(_visible_strings(row.get("refs_payload")))

    matches: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for field, text in candidates:
        match = UNRESOLVED_VISIBLE_RE.search(text)
        if match is None:
            continue
        key = (field, match.group(0).lower())
        if key in seen:
            continue
        seen.add(key)
        matches.append({"field": field, "matched_text": match.group(0)})
    return matches


def _claim_evidence(row: dict[str, Any]) -> dict[str, Any]:
    assistant = row.get("assistant_message")
    meta = assistant.get("meta") if isinstance(assistant, dict) else None
    answer_quality = meta.get("answer_quality") if isinstance(meta, dict) else None
    claim_evidence = answer_quality.get("claim_evidence") if isinstance(answer_quality, dict) else None
    return claim_evidence if isinstance(claim_evidence, dict) else {}


def _finalization_outcome(row: dict[str, Any]) -> dict[str, Any]:
    assistant = row.get("assistant_message")
    meta = assistant.get("meta") if isinstance(assistant, dict) else None
    answer_quality = meta.get("answer_quality") if isinstance(meta, dict) else None
    outcome = answer_quality.get("finalization_outcome") if isinstance(answer_quality, dict) else None
    if not isinstance(outcome, dict) or outcome.get("schema_version") != 1:
        return {}
    return outcome


def _runtime_gate_status(row: dict[str, Any]) -> str:
    final_payload = row.get("final_payload")
    runtime_check = final_payload.get("answer_runtime_check") if isinstance(final_payload, dict) else None
    if not isinstance(runtime_check, dict):
        return ""
    return str(runtime_check.get("status") or "").strip()


def summarize_live_dataset(dataset_id: str, report_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_path = report_dir / "raw_results.jsonl"
    summary_path = report_dir / "summary.json"
    rows = _load_jsonl(raw_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"{summary_path}: expected an object")

    passed = sum(1 for row in rows if bool((row.get("quality") or {}).get("ok")))
    failed = len(rows) - passed
    if int(summary.get("total") or 0) != len(rows):
        raise ValueError(f"{dataset_id}: summary/raw case count mismatch")
    if int(summary.get("passed") or 0) != passed or int(summary.get("failed") or 0) != failed:
        raise ValueError(f"{dataset_id}: summary/raw pass counts mismatch")

    return (
        {
            "id": dataset_id,
            "case_count": len(rows),
            "passed": passed,
            "failed": failed,
            "raw_sha256": _sha256(raw_path),
            "summary_sha256": _sha256(summary_path),
            "timing": deepcopy(summary.get("timing") or {}),
        },
        rows,
    )


def _note_auto_resolution_contract() -> dict[str, Any]:
    saved_quote = "The detector remained linear across the calibrated range."
    cases = (
        ("source_change_auto_verified", "exact_block", "block-old", "anchor-new"),
        ("stable_locator_auto_aligned", "exact_anchor", "block-new", "anchor-old"),
        ("exact_quote_auto_relocated", "quote_relocated", "block-new", "anchor-new"),
        ("saved_context_auto_disambiguated", "context_disambiguated", "block-new", "anchor-new"),
    )
    results: list[dict[str, Any]] = []
    for action, match_reason, block_id, anchor_id in cases:
        state = {
            "links": [{
                "kind": "source",
                "source_path": "library/example.pdf",
                "block_id": "block-old",
                "anchor_id": "anchor-old",
                "anchor_kind": "paragraph",
                "evidence_quote": saved_quote,
                "evidence_baseline_check": {
                    "version": 1,
                    "status": "skipped",
                    "reason": "source_changed",
                    "checked_at": 1_799_999_900,
                },
            }]
        }
        target = {
            "block_id": block_id,
            "anchor_id": anchor_id,
            "kind": "paragraph",
            "heading_path": "Results",
            "page_start": 6,
            "text": f"In repeated trials, {saved_quote} No saturation was observed.",
        }
        snapshot = {
            "version": 1,
            "captured_at": 1_800_000_000,
            "source_fingerprint": "sha256:new",
            "reason": match_reason,
            "before": [],
            "target": target,
            "after": [],
        }
        try:
            resolved = auto_resolve_research_note_source_state(
                state,
                source_index=0,
                source_path="LIBRARY\\example.pdf",
                target=target,
                source_fingerprint="sha256:new",
                context_snapshot=snapshot,
                match_reason=match_reason,
                action=action,
                resolution_evidence={"mode": "strict_saved_context"},
                resolved_at=1_800_000_000,
            )
            link = resolved["links"][0]
            ok = bool(
                link.get("evidence_quote") == saved_quote
                and (link.get("evidence_verification") or {}).get("status") == "exact"
                and (link.get("evidence_verification") or {}).get("reason") == action
                and (link.get("evidence_auto_resolution_history") or [{}])[-1].get("action") == action
                and "evidence_baseline_check" not in link
            )
            detail = "resolved_exact_without_rewriting_quote" if ok else "contract_mismatch"
        except Exception as exc:  # pragma: no cover - reported as a failed contract
            ok = False
            detail = f"{type(exc).__name__}: {exc}"
        results.append({"id": action, "ok": ok, "detail": detail})

    unsafe_state = {
        "links": [{
            "kind": "source",
            "source_path": "library/example.pdf",
            "block_id": "old",
            "evidence_quote": saved_quote,
        }]
    }
    unsafe_target = {"block_id": "new", "kind": "paragraph", "text": "Unrelated source text."}
    unsafe_snapshot = {
        "version": 1,
        "captured_at": 1_800_000_000,
        "source_fingerprint": "sha256:new",
        "target": unsafe_target,
    }
    try:
        auto_resolve_research_note_source_state(
            unsafe_state,
            source_index=0,
            source_path="library/example.pdf",
            target=unsafe_target,
            source_fingerprint="sha256:new",
            context_snapshot=unsafe_snapshot,
            match_reason="quote_relocated",
            action="exact_quote_auto_relocated",
            resolved_at=1_800_000_000,
        )
    except ValueError:
        unsafe_ok = True
    else:  # pragma: no cover - this would be a safety regression
        unsafe_ok = False
    results.append({
        "id": "unsafe_fuzzy_candidate_rejected",
        "ok": unsafe_ok,
        "detail": "requires_review" if unsafe_ok else "unsafe_auto_resolution",
    })
    passed = sum(1 for row in results if row["ok"])
    return {
        "case_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "cases": results,
        "scope": "pure deterministic note-evidence state transitions; API orchestration remains in sanity gates",
    }


def _deterministic_contracts() -> dict[str, Any]:
    agent_quality = evaluate_quality_cases()
    fixture = load_fixture(DEFAULT_FIXTURE)
    grounded = evaluate_replay_rows(fixture, load_replay(DEFAULT_REPLAY))
    reviewed = run_reviewed_replay_eval(require_reviewed=True)
    return {
        "agent_answer_quality": {
            "ok": bool(agent_quality.get("ok")),
            "case_count": int(agent_quality.get("case_count") or 0),
            "expected_answer_point_coverage": agent_quality.get("expected_answer_point_coverage"),
            "citation_precision": agent_quality.get("citation_precision"),
            "claim_support_rate": agent_quality.get("claim_support_rate"),
            "unsupported_claim_rate": agent_quality.get("unsupported_claim_rate"),
            "no_evidence_refusal_accuracy": agent_quality.get("no_evidence_refusal_accuracy"),
            "quality_gate_observed_count": int(agent_quality.get("quality_gate_observed_count") or 0),
        },
        "grounded_replay": {
            "ok": bool(grounded.get("ok")),
            "case_count": int(grounded.get("total") or 0),
            "passed": int(grounded.get("passed") or 0),
            "failed": int(grounded.get("failed") or 0),
        },
        "reviewed_replay": {
            "ok": bool(reviewed.get("ok")),
            "case_count": int(reviewed.get("reviewed_case_count") or 0),
            "dataset_count": int(reviewed.get("evaluated_dataset_count") or 0),
        },
        "note_evidence_auto_resolution": _note_auto_resolution_contract(),
    }


def build_report(inputs: list[tuple[str, Path]]) -> dict[str, Any]:
    source_reports: list[dict[str, Any]] = []
    all_rows: list[tuple[str, dict[str, Any]]] = []
    for dataset_id, report_dir in inputs:
        source, rows = summarize_live_dataset(dataset_id, report_dir)
        source_reports.append(source)
        all_rows.extend((dataset_id, row) for row in rows)

    category_rows: dict[str, dict[str, Any]] = {}
    for category, names in CATEGORY_CHECKS.items():
        check_counts: Counter[str] = Counter()
        passed_counts: Counter[str] = Counter()
        covered_cases: set[str] = set()
        for dataset_id, row in all_rows:
            case_id = f"{dataset_id}:{row.get('id')}"
            quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
            for check in quality.get("checks") or []:
                if not isinstance(check, dict):
                    continue
                name = str(check.get("name") or "")
                if name not in names:
                    continue
                check_counts[name] += 1
                passed_counts[name] += int(bool(check.get("ok")))
                covered_cases.add(case_id)
        applied = sum(check_counts.values())
        passed = sum(passed_counts.values())
        category_rows[category] = {
            "case_coverage": len(covered_cases),
            "checks_applied": applied,
            "passed": passed,
            "failed": applied - passed,
            "pass_rate": round(passed / applied, 6) if applied else None,
            "check_counts": {
                name: {"applied": check_counts[name], "passed": passed_counts[name]}
                for name in sorted(check_counts)
            },
        }

    action_counts: Counter[str] = Counter()
    detection_counts: Counter[str] = Counter()
    cases_with_actions: set[str] = set()
    claim_evidence_observed = 0
    runtime_gate_statuses: Counter[str] = Counter()
    finalization_statuses: Counter[str] = Counter()
    stable_gate_pass_count = 0
    stable_repair_required_cases = 0
    stable_repair_succeeded_cases = 0
    visible_unresolved: list[dict[str, str]] = []
    for dataset_id, row in all_rows:
        case_id = f"{dataset_id}:{row.get('id')}"
        claim_evidence = _claim_evidence(row)
        if claim_evidence:
            claim_evidence_observed += 1
        before = sum(action_counts.values())
        for field in FINALIZATION_ACTION_FIELDS:
            value = claim_evidence.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                action_counts[field] += value
        if sum(action_counts.values()) > before:
            cases_with_actions.add(case_id)
        for field in DETECTION_FIELDS:
            value = claim_evidence.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                detection_counts[field] += value
        status = _runtime_gate_status(row)
        if status:
            runtime_gate_statuses[status] += 1
        finalization_outcome = _finalization_outcome(row)
        if finalization_outcome:
            finalization_statuses[str(finalization_outcome.get("status") or "unknown")] += 1
            stable_gate_pass_count += int(finalization_outcome.get("gate_pass_count") or 0)
            if bool(finalization_outcome.get("repair_required")):
                stable_repair_required_cases += 1
                if bool(finalization_outcome.get("repair_succeeded")):
                    stable_repair_succeeded_cases += 1
        for match in _visible_unresolved_rows(row):
            visible_unresolved.append({"case_id": case_id, **match})

    total_cases = len(all_rows)
    total_passed = sum(source["passed"] for source in source_reports)
    deterministic = _deterministic_contracts()
    report = {
        "schema_version": 1,
        "report_id": "research_qa_quality_outcomes_v1",
        "product_boundary": (
            "reading library papers and finding, comparing, verifying, locating, and organizing traceable evidence; "
            "not manuscript writing or submission preparation"
        ),
        "source_reports": source_reports,
        "live_outcomes": {
            "case_count": total_cases,
            "passed": total_passed,
            "failed": total_cases - total_passed,
            "categories": category_rows,
        },
        "automatic_resolution": {
            "claim_evidence_observed_cases": claim_evidence_observed,
            "observed_finalization_action_cases": len(cases_with_actions),
            "observed_finalization_action_count": sum(action_counts.values()),
            "observed_finalization_actions": {
                field: action_counts[field] for field in FINALIZATION_ACTION_FIELDS
            },
            "residual_detection_counts": {
                field: detection_counts[field] for field in DETECTION_FIELDS
            },
            "runtime_gate_status_counts": dict(sorted(runtime_gate_statuses.items())),
            "runtime_gate_observed_count": sum(runtime_gate_statuses.values()),
            "stable_outcome_observed_cases": sum(finalization_statuses.values()),
            "stable_outcome_status_counts": dict(sorted(finalization_statuses.items())),
            "stable_gate_pass_count": stable_gate_pass_count,
            "stable_repair_required_cases": stable_repair_required_cases,
            "stable_repair_succeeded_cases": stable_repair_succeeded_cases,
            "stable_repair_success_rate": (
                round(stable_repair_succeeded_cases / stable_repair_required_cases, 6)
                if stable_repair_required_cases
                else None
            ),
            "attempt_count": (
                stable_gate_pass_count
                if sum(finalization_statuses.values()) == total_cases
                else None
            ),
            "attempt_observability": (
                "answer_finalization_outcome_v1"
                if sum(finalization_statuses.values()) == total_cases
                else "partially_recorded_answer_finalization_outcome_v1"
                if finalization_statuses
                else "not_recorded_in_historical_live_traces"
            ),
            "final_user_visible_unresolved_count": len(visible_unresolved),
            "final_user_visible_unresolved_examples": visible_unresolved[:20],
        },
        "deterministic_contracts": deterministic,
        "interpretation_boundaries": {
            "correctness": "expected answer points and direct-answer fixture contracts; not open-domain human correctness",
            "faithfulness": "claim-to-retrieved-evidence contracts over labeled cases",
            "contradiction": (
                "forbidden-claim and forbidden-evidence boundary checks only; this is not semantic entailment or NLI"
            ),
            "repair_rate": (
                "finalization actions are observed, but historical traces do not persist total attempts; no repair rate is claimed"
            ),
            "visible_state_scan": (
                "scans final answers and rendered citation-card copy; internal diagnostic statuses are intentionally excluded"
            ),
        },
    }
    errors = validate_report(report, strict_committed=False)
    report["validation"] = {"ok": not errors, "errors": errors}
    return report


def validate_report(report: dict[str, Any], *, strict_committed: bool) -> list[str]:
    errors: list[str] = []
    if report.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if report.get("report_id") != "research_qa_quality_outcomes_v1":
        errors.append("unexpected report_id")

    sources = report.get("source_reports")
    sources = sources if isinstance(sources, list) else []
    if not sources:
        errors.append("source_reports must not be empty")
    if strict_committed:
        by_id = {str(source.get("id") or ""): source for source in sources if isinstance(source, dict)}
        if set(by_id) != set(COMMITTED_SOURCE_CONTRACTS):
            errors.append("committed source report IDs changed")
        for dataset_id, expected in COMMITTED_SOURCE_CONTRACTS.items():
            actual = by_id.get(dataset_id) or {}
            for field, value in expected.items():
                if actual.get(field) != value:
                    errors.append(f"{dataset_id}: {field} changed")

    live = report.get("live_outcomes") if isinstance(report.get("live_outcomes"), dict) else {}
    if int(live.get("case_count") or 0) <= 0:
        errors.append("live outcome cases are missing")
    if int(live.get("failed") or 0) != 0:
        errors.append("live outcome failures remain")
    if strict_committed and (live.get("case_count"), live.get("passed")) != (78, 78):
        errors.append("committed live outcome coverage changed")
    categories = live.get("categories") if isinstance(live.get("categories"), dict) else {}
    for category in CATEGORY_CHECKS:
        outcome = categories.get(category) if isinstance(categories.get(category), dict) else {}
        if int(outcome.get("checks_applied") or 0) <= 0:
            errors.append(f"{category}: no checks applied")
        if int(outcome.get("failed") or 0) != 0:
            errors.append(f"{category}: failures remain")

    resolution = (
        report.get("automatic_resolution")
        if isinstance(report.get("automatic_resolution"), dict)
        else {}
    )
    if int(resolution.get("final_user_visible_unresolved_count") or 0) != 0:
        errors.append("final user-visible unresolved verifier copy remains")
    stable_observed = int(resolution.get("stable_outcome_observed_cases") or 0)
    live_case_count = int(live.get("case_count") or 0)
    if stable_observed < live_case_count and resolution.get("attempt_count") is not None:
        errors.append("historical repair attempt count must remain unknown")
    if stable_observed == 0 and resolution.get("attempt_observability") != "not_recorded_in_historical_live_traces":
        errors.append("repair attempt observability boundary changed")
    if stable_observed == live_case_count and live_case_count > 0:
        if resolution.get("attempt_observability") != "answer_finalization_outcome_v1":
            errors.append("stable repair attempt observability is missing")
        if int(resolution.get("attempt_count") or 0) <= 0:
            errors.append("stable repair attempt count is missing")
    if int(resolution.get("observed_finalization_action_count") or 0) <= 0:
        errors.append("no automatic finalization actions were observed")

    deterministic = (
        report.get("deterministic_contracts")
        if isinstance(report.get("deterministic_contracts"), dict)
        else {}
    )
    expected_counts = {
        "agent_answer_quality": 5,
        "grounded_replay": 6,
        "reviewed_replay": 5,
    }
    for name, expected_count in expected_counts.items():
        outcome = deterministic.get(name) if isinstance(deterministic.get(name), dict) else {}
        if not bool(outcome.get("ok")):
            errors.append(f"{name}: failed")
        if strict_committed and int(outcome.get("case_count") or 0) != expected_count:
            errors.append(f"{name}: committed case coverage changed")
    note_resolution = deterministic.get("note_evidence_auto_resolution")
    note_resolution = note_resolution if isinstance(note_resolution, dict) else {}
    if int(note_resolution.get("case_count") or 0) != 5:
        errors.append("note auto-resolution coverage changed")
    if int(note_resolution.get("failed") or 0) != 0:
        errors.append("note auto-resolution contract failed")
    return errors


def _parse_input(value: str) -> tuple[str, Path]:
    dataset_id, separator, raw_path = value.partition("=")
    if not separator or not dataset_id.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("input must be DATASET_ID=REPORT_DIR")
    return dataset_id.strip(), Path(raw_path.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate the deterministic Research QA outcome report."
    )
    parser.add_argument(
        "--input",
        action="append",
        type=_parse_input,
        default=None,
        help="Live report source as DATASET_ID=REPORT_DIR. Repeatable.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.validate_only:
        report = json.loads(output.read_text(encoding="utf-8"))
        errors = validate_report(report, strict_committed=True)
        if errors:
            for error in errors:
                print(f"[ERROR] {error}", file=sys.stderr)
            return 1
        print("[OK] Research QA outcome report: 78 live cases plus deterministic repair contracts")
        return 0

    inputs = args.input or list(DEFAULT_INPUTS)
    resolved_inputs = [
        (dataset_id, path if path.is_absolute() else ROOT / path)
        for dataset_id, path in inputs
    ]
    report = build_report(resolved_inputs)
    errors = list((report.get("validation") or {}).get("errors") or [])
    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "[OK] wrote Research QA outcome report: "
        f"{report['live_outcomes']['passed']}/{report['live_outcomes']['case_count']} live cases"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
