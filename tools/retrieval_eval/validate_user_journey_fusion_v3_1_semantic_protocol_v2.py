from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file


DEFAULT_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_user_journey_fusion_v3_1_semantic_protocol_v2.json"
)
DEFAULT_PREDECESSOR = (
    REPO_ROOT / "docs" / "retrieval_user_journey_fusion_v3_1_semantic_protocol.json"
)
DEFAULT_INVALIDATION = (
    REPO_ROOT
    / "docs"
    / "retrieval_user_journey_fusion_v3_1_semantic_preflight_invalidation.json"
)
INHERITED_KEYS = (
    "eligibility",
    "unchanged_candidate",
    "acceptance_gate",
    "failure_policy",
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_protocol(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("id")
        != "retrieval_user_journey_fusion_v3_1_semantic_acceptance_protocol_v2"
        or payload.get("curation_status")
        != "frozen_after_zero-query_preflight_invalidation_before_index_build_or_measurement"
    ):
        errors.append("semantic-v2 protocol identity changed")
    predecessor = payload.get("predecessor") or {}
    if (
        predecessor.get("protocol_path")
        != "docs/retrieval_user_journey_fusion_v3_1_semantic_protocol.json"
        or predecessor.get("protocol_sha256") != sha256_file(DEFAULT_PREDECESSOR)
        or predecessor.get("invalidation_path")
        != "docs/retrieval_user_journey_fusion_v3_1_semantic_preflight_invalidation.json"
        or predecessor.get("invalidation_sha256") != sha256_file(DEFAULT_INVALIDATION)
    ):
        errors.append("semantic-v2 predecessor identity changed")
    old = json.loads(DEFAULT_PREDECESSOR.read_text(encoding="utf-8"))
    expected_hashes = {key: _canonical_sha256(old[key]) for key in INHERITED_KEYS}
    if payload.get("inherited_contract_sha256") != expected_hashes:
        errors.append(
            "semantic-v2 changed the candidate, gate, eligibility, or failure policy"
        )
    corpus = payload.get("isolated_corpus") or {}
    fixture_path = REPO_ROOT / str(corpus.get("fixture_path") or "")
    if (
        corpus.get("runtime_db_argument_required") is not True
        or not fixture_path.is_file()
        or corpus.get("fixture_sha256") != sha256_file(fixture_path)
        or corpus.get("document_count") != 10
        or corpus.get("chunk_count") != 1673
        or corpus.get("qrel_corpus_fingerprint")
        != "d1602fb0d3a7320806dc1cf1a9c3615d7fc1864471e93270f2135f8f44f27516"
        or corpus.get(
            "must_validate_chunk_ids_text_hashes_and_logical_documents_before_queries"
        )
        is not True
    ):
        errors.append("semantic-v2 isolated corpus contract changed")
    if payload.get("dedicated_index") != {
        "relative_path": "test_results/retrieval_user_journey_fusion_v3_1_semantic/qwen_index_v1",
        "provider_identity": "55229348d5bbe4b1e1935aeb8ffe57cc03c527685b1d08d44f11a56a54525cb8",
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "maximum_initial_builds": 1,
        "build_must_complete_before_query_encoding": True,
        "allow_download": False,
        "trust_remote_code": False,
        "product_index_unchanged": True,
    }:
        errors.append("semantic-v2 dedicated index contract changed")
    if payload.get("execution_contract") != {
        "maximum_query_measurement_runs": 1,
        "ranking_depth": 10,
        "reported_metric_depths": [5, 10],
        "expected_unique_query_embeddings": 12,
        "product_control_must_match_predecessor_for": [
            "recall@5",
            "recall@10",
            "wrong_source_rate@5",
        ],
        "candidate_or_gate_changes": 0,
        "model_or_prompt_variants": 0,
        "product_enabled": False,
    }:
        errors.append("semantic-v2 execution contract changed")
    return errors


def effective_protocol(payload: dict[str, Any]) -> dict[str, Any]:
    errors = validate_protocol(payload)
    if errors:
        raise ValueError("invalid semantic-v2 protocol: " + "; ".join(errors))
    old = json.loads(DEFAULT_PREDECESSOR.read_text(encoding="utf-8"))
    return {
        **old,
        "id": payload["id"],
        "curation_status": payload["curation_status"],
        "isolated_corpus": payload["isolated_corpus"],
        "dedicated_index": payload["dedicated_index"],
        "execution_contract": payload["execution_contract"],
        "replacement_protocol": payload,
    }


def main() -> int:
    payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    errors = validate_protocol(payload)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] semantic-v2 dedicated-index replacement frozen before build or query")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
