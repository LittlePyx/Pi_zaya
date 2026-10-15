from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.claim_entailment_eval_fixture import (
    claim_entailment_fixture_summary,
    validate_claim_entailment_fixture,
)
from kb.retrieval_eval_fixture import sha256_file


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "claim_entailment_eval_v1.json"
DEFAULT_QRELS = REPO_ROOT / "docs" / "retrieval_eval_unseen_v2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the pre-registered claim entailment/contradiction gold set."
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.resolve().read_text(encoding="utf-8"))
    qrels = json.loads(args.qrels.resolve().read_text(encoding="utf-8"))
    errors = validate_claim_entailment_fixture(
        fixture,
        qrels,
        qrels_sha256=sha256_file(args.qrels.resolve()),
    )
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    summary = claim_entailment_fixture_summary(fixture)
    print(
        f"[OK] claim entailment fixture: {summary['case_count']} cases, "
        f"labels={summary['label_counts']}, locales={summary['locale_counts']}, "
        f"lexical_adversaries={summary['lexical_adversary_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
