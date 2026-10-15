from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.run_product_shadow_operational_replay import (
    DEFAULT_FIXTURE,
    DEFAULT_PROTOCOL as DEFAULT_PARENT_PROTOCOL,
    DEFAULT_QWEN_INDEX,
    _result_fingerprint,
    _run_replay,
    validate_report as validate_parent_report,
)
from tools.retrieval_eval.validate_product_shadow_protocol import (
    sha256_file_variants,
    validate_protocol as validate_parent_protocol,
)
from tools.retrieval_eval.validate_product_shadow_vectorized_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


DEFAULT_PARENT_REPORT = (
    REPO_ROOT / "docs" / "retrieval_product_shadow_operational_v1.json"
)
DEFAULT_REPORT = (
    REPO_ROOT / "docs" / "retrieval_product_shadow_vectorized_operational_v1.json"
)
DEFAULT_TELEMETRY = (
    REPO_ROOT
    / "test_results"
    / "retrieval_product_shadow_vectorized_v1"
    / "observations.jsonl"
)
EXPECTED_RESULT_FINGERPRINT = (
    "b63ec97dc7b5562c1c85e4c532a442725d4b55842caa9a172343dd94992600d0"
)


def build_child_report(
    base: dict[str, Any], child_protocol: dict[str, Any]
) -> dict[str, Any]:
    report = copy.deepcopy(base)
    report["report_kind"] = "retrieval_product_shadow_vectorized_operational_v1"
    report["protocol_sha256"] = sha256_file(DEFAULT_PROTOCOL)
    report["parent_rejected_report_sha256"] = sha256_file(DEFAULT_PARENT_REPORT)
    changes = child_protocol["allowed_implementation_changes"]
    report["performance_candidate"] = {
        key: changes[key]
        for key in (
            "scoring_backend",
            "dense_retriever_sha256",
            "provider_sha256",
            "shadow_runtime_sha256",
        )
    }
    passed = bool((report.get("operational_gate") or {}).get("passes_all"))
    report["decision"] = {
        "status": (
            child_protocol["execution"]["pass_decision"]
            if passed
            else child_protocol["execution"]["reject_decision"]
        ),
        "explicit_opt_in_design_allowed": passed,
        "shadow_enabled_by_default": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }
    return report


def _as_parent_report(report: dict[str, Any]) -> dict[str, Any]:
    parent = copy.deepcopy(report)
    parent["report_kind"] = "retrieval_product_shadow_operational_v1"
    parent["protocol_sha256"] = sha256_file(DEFAULT_PARENT_PROTOCOL)
    parent.pop("parent_rejected_report_sha256", None)
    parent.pop("performance_candidate", None)
    passed = bool((parent.get("operational_gate") or {}).get("passes_all"))
    parent["decision"] = {
        "status": "passed_for_opt_in_design"
        if passed
        else "rejected_keep_shadow_disabled",
        "opt_in_design_allowed": passed,
        "shadow_enabled_by_default": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }
    return parent


def validate_report(
    report: dict[str, Any],
    *,
    child_protocol: dict[str, Any],
    parent_protocol: dict[str, Any],
    fixture: dict[str, Any],
    expected_result_fingerprint: str | None = EXPECTED_RESULT_FINGERPRINT,
) -> list[str]:
    errors = validate_parent_report(
        _as_parent_report(report),
        protocol=parent_protocol,
        fixture=fixture,
        expected_result_fingerprint=None,
    )
    if (
        report.get("report_kind")
        != "retrieval_product_shadow_vectorized_operational_v1"
    ):
        errors.append("vectorized shadow report identity changed")
    if report.get("protocol_sha256") not in sha256_file_variants(DEFAULT_PROTOCOL):
        errors.append("vectorized shadow protocol identity changed")
    if report.get("parent_rejected_report_sha256") not in sha256_file_variants(DEFAULT_PARENT_REPORT):
        errors.append("vectorized shadow rejected parent changed")
    changes = child_protocol["allowed_implementation_changes"]
    expected_candidate = {
        key: changes[key]
        for key in (
            "scoring_backend",
            "dense_retriever_sha256",
            "provider_sha256",
            "shadow_runtime_sha256",
        )
    }
    if report.get("performance_candidate") != expected_candidate:
        errors.append("vectorized shadow performance candidate changed")
    fingerprint = _result_fingerprint(
        list(report.get("results") or []), report.get("operational_gate") or {}
    )
    if report.get("result_fingerprint") != fingerprint:
        errors.append("vectorized shadow result fingerprint is inconsistent")
    if expected_result_fingerprint and fingerprint != expected_result_fingerprint:
        errors.append("pinned vectorized shadow result changed")
    passed = bool((report.get("operational_gate") or {}).get("passes_all"))
    expected_decision = {
        "status": (
            child_protocol["execution"]["pass_decision"]
            if passed
            else child_protocol["execution"]["reject_decision"]
        ),
        "explicit_opt_in_design_allowed": passed,
        "shadow_enabled_by_default": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }
    if report.get("decision") != expected_decision:
        errors.append("vectorized shadow decision changed")
    return list(dict.fromkeys(errors))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run or validate the frozen vectorized shadow replacement."
    )
    parser.add_argument("--db-root", type=Path, default=REPO_ROOT / "db")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_QWEN_INDEX)
    parser.add_argument("--telemetry", type=Path, default=DEFAULT_TELEMETRY)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    child_protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    parent_protocol = json.loads(DEFAULT_PARENT_PROTOCOL.read_text(encoding="utf-8"))
    fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    contract_errors = validate_protocol(child_protocol) + validate_parent_protocol(
        parent_protocol
    )
    if contract_errors:
        parser.error(
            "invalid vectorized shadow contract: " + "; ".join(contract_errors)
        )
    if args.validate_only:
        report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
        errors = validate_report(
            report,
            child_protocol=child_protocol,
            parent_protocol=parent_protocol,
            fixture=fixture,
        )
        if errors:
            for error in errors:
                print(f"[FAIL] {error}")
            return 1
        print(f"[OK] vectorized shadow replay: {report['decision']['status']}")
        return 0
    if DEFAULT_REPORT.exists():
        parser.error("vectorized shadow report already exists; use --validate-only")
    base = _run_replay(
        db_root=args.db_root.expanduser().resolve(),
        index_dir=args.index_dir.expanduser().resolve(),
        telemetry_path=args.telemetry.expanduser().resolve(),
        protocol=parent_protocol,
        fixture=fixture,
    )
    report = build_child_report(base, child_protocol)
    errors = validate_report(
        report,
        child_protocol=child_protocol,
        parent_protocol=parent_protocol,
        fixture=fixture,
        expected_result_fingerprint=None,
    )
    if errors:
        raise ValueError("invalid vectorized shadow report: " + "; ".join(errors))
    DEFAULT_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"summary": report["summary"], "decision": report["decision"]}, indent=2
        )
    )
    print(f"[OK] wrote {DEFAULT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
