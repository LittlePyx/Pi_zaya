from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file


DEFAULT_INVALIDATION = (
    REPO_ROOT
    / "docs"
    / "retrieval_user_journey_fusion_v3_1_semantic_preflight_invalidation.json"
)
OLD_REPORT = (
    REPO_ROOT / "docs" / "retrieval_user_journey_fusion_v3_1_semantic_acceptance.json"
)


def validate_invalidation(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("report_kind") != "fusion_v3_1_semantic_preflight_invalidation"
        or payload.get("status") != "closed_before_model_or_query_measurement"
    ):
        errors.append("semantic preflight invalidation identity changed")
    protocol = REPO_ROOT / str(payload.get("protocol_path") or "")
    if (
        not protocol.is_file()
        or payload.get("protocol_sha256") != sha256_file(protocol)
        or payload.get("attempted_runner_sha256")
        != "64d6b58e65805a8548ac344db2873220cc9ab8ffd2fda0f989ffaa001524f7e6"
    ):
        errors.append("semantic preflight invalidation source changed")
    if payload.get("execution_boundary") != {
        "corpus_preflight_executed": True,
        "model_loaded": False,
        "document_embeddings_created": 0,
        "query_embeddings_created": 0,
        "candidate_result_observed": False,
        "canonical_report_written": False,
        "product_enabled": False,
    }:
        errors.append("semantic preflight execution boundary changed")
    failure = payload.get("failure") or {}
    if (
        failure.get("message") != "semantic corpus has 0 chunks, expected 1673"
        or "isolated 10-document corpus" not in str(failure.get("root_cause") or "")
        or failure.get("quality_conclusion_allowed") is not False
    ):
        errors.append("semantic preflight root cause changed")
    decision = payload.get("effective_decision") or {}
    if (
        decision.get("protocol_closed") is not True
        or decision.get("semantic_result") != "none"
        or decision.get("retry_under_same_protocol") is not False
        or "one dedicated Qwen index"
        not in str(decision.get("replacement_allowed") or "")
        or decision.get("candidate_or_gate_change_allowed") is not False
        or decision.get("product_shadow_allowed") is not False
    ):
        errors.append("semantic preflight no-result decision changed")
    if OLD_REPORT.exists():
        errors.append("closed semantic preflight unexpectedly wrote a report")
    return errors


def main() -> int:
    payload = json.loads(DEFAULT_INVALIDATION.read_text(encoding="utf-8"))
    errors = validate_invalidation(payload)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print(
        "[OK] semantic preflight closed before model, query, result, or product access"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
