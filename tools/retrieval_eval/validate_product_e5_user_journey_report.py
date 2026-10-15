from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.retrieval_eval.run_product_e5_user_journey import (
    DEFAULT_PROTOCOL,
    DEFAULT_REPORT,
    validate_report,
)
from tools.retrieval_eval.validate_product_e5_user_journey_protocol import (
    validate_protocol,
)


def main() -> int:
    try:
        protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
        report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] product E5 user-journey report unreadable: {type(exc).__name__}")
        return 1
    errors = validate_protocol(protocol) + validate_report(report, protocol=protocol)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[OK] product E5 user-journey rejection report is pinned and valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
