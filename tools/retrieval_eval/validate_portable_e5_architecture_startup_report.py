from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.run_portable_e5_architecture_startup import (
    DEFAULT_PROTOCOL,
    DEFAULT_REPORT,
    validate_report,
)
from tools.retrieval_eval.validate_portable_e5_architecture_startup_protocol import (
    validate_protocol,
)


EXPECTED_REPORT_SHA256 = "1843fe12db7290cbcb07e892187f0882cee7f24c791790354b6a79eee1c6bc4e"
EXPECTED_RESULT_FINGERPRINT = (
    "07b64dd0541c3c6bf3205f95aeee96c8bd70dd68f55760f13f4a4cabc3f91384"
)


def main() -> int:
    try:
        protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
        report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] portable E5 architecture-startup report unreadable: {type(exc).__name__}")
        return 1
    errors = validate_protocol(protocol) + validate_report(
        report,
        protocol=protocol,
        expected_result_fingerprint=EXPECTED_RESULT_FINGERPRINT,
    )
    if sha256_file(DEFAULT_REPORT) != EXPECTED_REPORT_SHA256:
        errors.append("portable E5 architecture-startup report file changed")
    if report.get("decision", {}).get("status") != protocol.get("decision", {}).get(
        "pass_status"
    ):
        errors.append("portable E5 architecture-startup passing verdict changed")
    failed = [
        key for key, value in report.get("gate", {}).get("checks", {}).items() if not value
    ]
    if failed:
        errors.append("portable E5 architecture-startup acquired a failed gate")
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[OK] portable E5 architecture-startup passing report is pinned and valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
