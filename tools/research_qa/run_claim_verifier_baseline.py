from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.agent.verifier import verify_answer_citations
from kb.claim_entailment_benchmark import evaluate_binary_claim_support
from kb.claim_entailment_eval_fixture import validate_claim_entailment_fixture
from kb.retrieval_eval_fixture import sha256_file


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "claim_entailment_eval_v1.json"
DEFAULT_QRELS = REPO_ROOT / "docs" / "retrieval_eval_unseen_v2.json"
DEFAULT_REPORT = REPO_ROOT / "docs" / "claim_entailment_lexical_baseline_v1.json"
VERIFIER_SOURCE = REPO_ROOT / "kb" / "agent" / "verifier.py"


def build_report(fixture_path: Path, qrels_path: Path) -> dict:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    errors = validate_claim_entailment_fixture(
        fixture,
        qrels,
        qrels_sha256=sha256_file(qrels_path),
    )
    if errors:
        raise ValueError("invalid claim entailment fixture: " + "; ".join(errors))
    return {
        "schema_version": 1,
        "report_kind": "lexical_claim_verifier_baseline",
        "fixture": {
            "path": fixture_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(fixture_path),
        },
        "verifier": {
            "contract": "citation_binding_plus_two_term_overlap",
            "source_path": VERIFIER_SOURCE.relative_to(REPO_ROOT).as_posix(),
            "source_sha256": sha256_file(VERIFIER_SOURCE),
            "semantic_entailment": False,
        },
        "result": evaluate_binary_claim_support(fixture, verify_answer_citations),
        "interpretation": (
            "Diagnostic baseline only. Passing citation syntax plus lexical overlap is not "
            "evidence of entailment; contradiction false-support is a release-blocking metric "
            "for any future semantic verifier candidate."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure the current lexical citation verifier on the semantic gold set."
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    report = build_report(args.fixture.resolve(), args.qrels.resolve())
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.validate_only:
        report_path = args.report.resolve()
        if not report_path.is_file() or report_path.read_text(encoding="utf-8") != rendered:
            print(f"[FAIL] lexical verifier baseline differs from {report_path}")
            return 1
        metrics = report["result"]["binary_metrics"]
        print(
            f"[OK] lexical verifier diagnostic: accuracy={metrics['accuracy']:.3f}, "
            f"contradiction false-support={metrics['contradiction_false_support_rate']:.3f}"
        )
        return 0
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
