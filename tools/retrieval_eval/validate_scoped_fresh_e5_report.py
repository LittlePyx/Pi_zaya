from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.run_scoped_fresh_e5 import (
    DEFAULT_PROTOCOL,
    DEFAULT_REPORT,
    validate_report,
)
from tools.retrieval_eval.validate_scoped_fresh_e5_protocol import validate_protocol


EXPECTED_REPORT_SHA256 = "90db9c97eef1196ab1e93f5531f662735daeef53c28adbb8a2a54cf5d0bb7fd8"


def main() -> int:
    try:
        protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
        report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] scoped fresh E5 report unreadable: {type(exc).__name__}")
        return 1
    errors = validate_protocol(protocol) + validate_report(report, protocol=protocol)
    if sha256_file(DEFAULT_REPORT) != EXPECTED_REPORT_SHA256:
        errors.append("scoped fresh E5 report file changed")
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[OK] scoped fresh E5 passing report is pinned and valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
