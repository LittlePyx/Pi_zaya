from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from kb.retrieval_semantic_eval_fixture import load_retrieval_benchmark_fixture


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_cases_v1.json"
DEFAULT_BENCHMARK = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_benchmark_v1.json"
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_semantic_representation_diagnostic_v1.json"
MODEL_ID = "intfloat/multilingual-e5-base"
REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"


def _focus_ids(benchmark: dict[str, Any]) -> list[str]:
    analysis = benchmark.get("failure_analysis")
    analysis = analysis if isinstance(analysis, dict) else {}
    return list(
        dict.fromkeys(
            [
                str(case_id)
                for key in (
                    "missed_by_product_bm25_and_raw_dense_at_10_in_both_modes",
                    "raw_dense_regressions_at_10",
                )
                for case_id in analysis.get(key) or []
            ]
        )
    )


def validate_report(
    report: dict[str, Any],
    fixture_path: Path = DEFAULT_FIXTURE,
    benchmark_path: Path = DEFAULT_BENCHMARK,
) -> list[str]:
    fixture = load_retrieval_benchmark_fixture(fixture_path, repo_root=REPO_ROOT)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if report.get("report_kind") != "semantic_chunk_representation_diagnostic":
        errors.append("report kind does not match")
    source = report.get("fixture") if isinstance(report.get("fixture"), dict) else {}
    if source.get("id") != fixture.get("id") or source.get("sha256") != sha256_file(fixture_path):
        errors.append("fixture identity or SHA-256 does not match")
    measured = report.get("benchmark") if isinstance(report.get("benchmark"), dict) else {}
    if measured.get("sha256") != sha256_file(benchmark_path):
        errors.append("semantic benchmark SHA-256 does not match")

    candidate = report.get("candidate") if isinstance(report.get("candidate"), dict) else {}
    if candidate.get("model_id") != MODEL_ID or candidate.get("revision") != REVISION:
        errors.append("candidate identity does not match")
    if candidate.get("max_length") != 512 or candidate.get("passage_prefix") != "passage: ":
        errors.append("candidate passage window changed")
    if candidate.get("local_files_only") is not True or candidate.get("trust_remote_code") is not False:
        errors.append("diagnostic must remain cache-only without remote code")
    if candidate.get("product_enabled") is not False:
        errors.append("diagnostic candidate must remain product-disabled")

    expected_labels = {
        (str(case.get("id") or ""), str(label.get("chunk_id") or "")): label
        for case in fixture.get("cases") or []
        for label in case.get("labels") or []
    }
    labels = report.get("labels") if isinstance(report.get("labels"), list) else []
    actual_keys = {
        (str(item.get("case_id") or ""), str(item.get("chunk_id") or ""))
        for item in labels
        if isinstance(item, dict)
    }
    if actual_keys != set(expected_labels):
        errors.append("reported label coverage does not match the fixture")
    for item in labels:
        if not isinstance(item, dict):
            errors.append("label rows must be objects")
            continue
        expected = expected_labels.get((str(item.get("case_id") or ""), str(item.get("chunk_id") or "")))
        if expected is None:
            continue
        if item.get("text_sha256") != expected.get("text_sha256"):
            errors.append(f"{item.get('chunk_id')} text SHA-256 does not match")
        expected_terms = [str(term) for term in expected.get("matched_evidence_terms") or []]
        terms = item.get("terms") if isinstance(item.get("terms"), list) else []
        if [term.get("term") for term in terms if isinstance(term, dict)] != expected_terms:
            errors.append(f"{item.get('chunk_id')} evidence terms do not match")
        if int(item.get("resolved_term_count") or 0) != len(expected_terms):
            errors.append(f"{item.get('chunk_id')} has unresolved evidence terms")
        if item.get("would_truncate") is not False or item.get("all_evidence_inside_window") is not True:
            errors.append(f"{item.get('chunk_id')} no longer supports the frozen no-truncation finding")

    population = report.get("population") if isinstance(report.get("population"), dict) else {}
    label_count = len(expected_labels)
    if population.get("label_count") != label_count or population.get("truncated_relevant_label_count") != 0:
        errors.append("relevant-label truncation aggregate is stale")
    if population.get("evidence_outside_window_label_count") != 0:
        errors.append("evidence-window aggregate is stale")
    if population.get("max_relevant_token_count") != max(
        (int(item.get("token_count_with_prefix") or 0) for item in labels), default=0
    ):
        errors.append("maximum relevant token count is stale")

    focus = report.get("focus") if isinstance(report.get("focus"), dict) else {}
    expected_focus = _focus_ids(benchmark)
    if focus.get("case_ids") != expected_focus or focus.get("case_count") != len(expected_focus):
        errors.append("focus failures do not match the frozen benchmark")
    focus_rows = [item for item in labels if item.get("case_id") in expected_focus]
    if focus.get("label_count") != len(focus_rows) or focus.get("truncated_label_count") != 0:
        errors.append("focus truncation aggregate is stale")
    if focus.get("evidence_outside_window_label_count") != 0:
        errors.append("focus evidence-window aggregate is stale")

    decision = report.get("decision") if isinstance(report.get("decision"), dict) else {}
    if decision.get("truncation_explains_frozen_failures") is not False:
        errors.append("unsupported truncation attribution was introduced")
    if decision.get("rechunk_stored_citation_units") is not False:
        errors.append("stored citation units must not be rechunked from this evidence")
    experiment = str(decision.get("only_permitted_subchunk_experiment") or "")
    if "parent" not in experiment or "chunk ID/page locator" not in experiment:
        errors.append("locator-preserving parent collapse contract is missing")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the semantic representation diagnosis.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = json.loads(args.report.resolve().read_text(encoding="utf-8"))
    errors = validate_report(report, args.fixture.resolve(), args.benchmark.resolve())
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print(
        "[OK] semantic representation diagnostic: "
        f"{report['population']['label_count']} labels, "
        f"{report['focus']['case_count']} focus failures, no relevant truncation"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
