from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.run_portable_optional_e5 import (
    DEFAULT_PROTOCOL,
    DEFAULT_REPORT,
    validate_report,
)
from tools.retrieval_eval.validate_portable_optional_e5_protocol import validate_protocol


EXPECTED_REPORT_SHA256 = "a938b1130c600aa8d9b1ccb11630848bba54ecef204d593348fde10764f42a89"
EXPECTED_RESULT_FINGERPRINT = (
    "2120f98ec523d98c0e890a66be008d71a65a67f04e468b3edf61ada48625df7d"
)


def main() -> int:
    try:
        protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
        report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] portable optional E5 report unreadable: {type(exc).__name__}")
        return 1
    errors = validate_protocol(protocol) + validate_report(
        report,
        protocol=protocol,
        expected_result_fingerprint=EXPECTED_RESULT_FINGERPRINT,
    )
    if sha256_file(DEFAULT_REPORT) != EXPECTED_REPORT_SHA256:
        errors.append("portable optional E5 report file changed")
    if report.get("decision", {}).get("status") != protocol.get("decision", {}).get(
        "failure_status"
    ):
        errors.append("portable optional E5 rejection verdict changed")
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[OK] portable optional E5 rejection report is pinned and valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
