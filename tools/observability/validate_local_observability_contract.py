from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.local_observability import (
    LocalJsonlObservationRecorder,
    local_observability_contract,
    sanitize_observation_attributes,
)


DEFAULT_REPORT = REPO_ROOT / "docs" / "local_observability_contract_v2.json"
_CANARY = "PI_ZAYA_PRIVATE_CONTENT_CANARY"


def _representative_attributes(contract: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    types = contract["attribute_types"]
    stable_ids = set(types["stable_id"])
    booleans = set(types["boolean"])
    counts = set(types["non_negative_count"])
    return {
        field: (
            "redacted_id"
            if field in stable_ids
            else True
            if field in booleans
            else 1
            if field in counts
            else 1.25
        )
        for field in fields
    }


def validate_contract(payload: dict[str, Any]) -> list[str]:
    expected = local_observability_contract()
    errors: list[str] = []
    if payload != expected:
        errors.append("frozen contract does not match the runtime allowlist or retention policy")
    typed_fields = {
        field
        for fields in expected["attribute_types"].values()
        for field in fields
    }
    allowed_fields = {
        field for fields in expected["events"].values() for field in fields
    }
    if not allowed_fields <= typed_fields:
        errors.append("one or more allowed attributes have no approved type")
    prohibited = set(expected["privacy"]["prohibited_attribute_names"])
    if allowed_fields & prohibited:
        errors.append("raw-content attribute entered an event allowlist")

    with tempfile.TemporaryDirectory(prefix="pi-zaya-observation-contract-") as temp_dir:
        path = Path(temp_dir) / "privacy-replay.jsonl"
        recorder = LocalJsonlObservationRecorder(path)
        for event_name, fields in expected["events"].items():
            clean = _representative_attributes(expected, fields)
            recorder.emit(event_name, clean)
            for field in prohibited:
                try:
                    sanitize_observation_attributes(event_name, {field: _CANARY})
                except ValueError:
                    continue
                errors.append(f"{event_name} accepted prohibited attribute {field}")
        if _CANARY in path.read_text(encoding="utf-8"):
            errors.append("privacy canary reached the local observation sink")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze and validate local observation privacy contract.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report_path = args.report.expanduser().resolve()
    if args.write:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(local_observability_contract(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if not report_path.is_file():
        print(f"Missing local observation contract: {report_path}", file=sys.stderr)
        return 2
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Invalid local observation contract: {exc}", file=sys.stderr)
        return 2
    errors = validate_contract(payload)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "Local observation contract valid: "
        f"{len(payload['events'])} redacted event types; bounded local retention; external export off."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
