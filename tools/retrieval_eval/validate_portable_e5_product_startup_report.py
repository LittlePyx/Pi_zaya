from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.run_portable_optional_e5_product_startup import (
    DEFAULT_PROTOCOL,
    DEFAULT_REPORT,
    validate_report,
)
from tools.retrieval_eval.validate_portable_e5_product_startup_protocol import (
    validate_protocol,
)


EXPECTED_REPORT_SHA256 = "5f8216155254b9b5fe61d46cc8375ed3b023ea68cbc81e27bdf338c9cf21c5ca"
EXPECTED_RESULT_FINGERPRINT = (
    "7789569cae9397c475472d0d0a63d312b8789f06de7531b357e4521cd5d48d85"
)


def main() -> int:
    try:
        protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
        report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] portable E5 product-startup report unreadable: {type(exc).__name__}")
        return 1
    errors = validate_protocol(protocol) + validate_report(
        report,
        protocol=protocol,
        expected_result_fingerprint=EXPECTED_RESULT_FINGERPRINT,
    )
    if sha256_file(DEFAULT_REPORT) != EXPECTED_REPORT_SHA256:
        errors.append("portable E5 product-startup report file changed")
    if report.get("decision", {}).get("status") != protocol.get("decision", {}).get(
        "failure_status"
    ):
        errors.append("portable E5 product-startup rejection verdict changed")
    failed = [
        key for key, value in report.get("gate", {}).get("checks", {}).items() if not value
    ]
    if failed != ["component_tree_reproduced", "product_import_time"]:
        errors.append("portable E5 product-startup failed-gate set changed")
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[OK] portable E5 product-startup rejection report is pinned and valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
