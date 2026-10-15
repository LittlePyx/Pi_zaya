from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.validate_lighter_runtime_stage0_e5_small import (
    DEFAULT_REPORT as DEFAULT_STAGE0_REPORT,
)


DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "retrieval_e5_small_window_census_protocol_v1.json"
DEFAULT_UMBRELLA = REPO_ROOT / "docs" / "retrieval_lighter_runtime_screen_protocol_v1.json"
DEFAULT_QWEN_RESULT = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage1_qwen_v1.json"
DEFAULT_STRESS_MANIFEST = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stress_manifest_v1.json"
EXPECTED_PROTOCOL_SHA256 = "ca464188ca3b6d8ec89e61f07fd7d0b9b4c08f79b17481951fe883f5030d3fe0"


def validate_protocol(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id") != "retrieval_e5_small_window_census_v1"
        or payload.get("status") != "frozen_before_tokenizer_load"
    ):
        errors.append("E5-small window-census protocol identity changed")

    authorization = payload.get("authorization") or {}
    if (
        authorization.get("umbrella_protocol_sha256") != sha256_file(DEFAULT_UMBRELLA)
        or authorization.get("qwen_stage_1_report_sha256")
        != sha256_file(DEFAULT_QWEN_RESULT)
        or authorization.get("stage_0_report_sha256") != sha256_file(DEFAULT_STAGE0_REPORT)
        or authorization.get("stress_manifest_sha256")
        != sha256_file(DEFAULT_STRESS_MANIFEST)
        or authorization.get("qwen_decision") != "rejected_stage_1"
        or authorization.get("stage_0_decision") != "passed_stage_0"
    ):
        errors.append("E5-small window-census authorization changed")

    source = payload.get("source_corpus") or {}
    if source != {
        "runtime_db_argument_required": True,
        "tree_sha256": "deb60a97bdfc50b0273de07ebe1c775352ffb2c5e4adeb3e925d48f7f25cafaf",
        "document_count": 10,
        "parent_chunk_count": 1673,
        "writes_allowed": False,
    }:
        errors.append("E5-small window-census source contract changed")

    tokenizer = payload.get("tokenizer") or {}
    if tokenizer != {
        "model_id": "intfloat/multilingual-e5-small",
        "revision": "03415a4be176a1620747c692ed433219fabc3def",
        "tokenizer_json_bytes": 17082730,
        "tokenizer_json_sha256": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
        "padding_side": "right",
        "maximum_model_tokens": 512,
        "bos_token_id": 0,
        "pad_token_id": 1,
        "eos_token_id": 2,
        "local_files_only": True,
    }:
        errors.append("E5-small window-census tokenizer contract changed")

    representation = payload.get("representation") or {}
    if (
        representation.get("content_window_tokens") != 448
        or representation.get("minimum_content_overlap_tokens") != 64
        or representation.get("nominal_content_stride_tokens") != 384
        or representation.get("query_prefix") != "query: "
        or representation.get("passage_prefix") != "passage: "
        or representation.get("full_parent_content_coverage_required") is not True
        or representation.get("silent_tail_loss_allowed") is not False
        or representation.get("parent_score_collapse")
        != "maximum_window_cosine_then_parent_id"
        or representation.get("returned_result_identity")
        != "complete_immutable_parent_chunk"
        or representation.get("child_window_text_returned") is not False
    ):
        errors.append("E5-small window-census representation changed")

    measurement = payload.get("measurement") or {}
    if (
        measurement.get("tokenize_all_parent_chunks") is not True
        or measurement.get("tokenize_all_12_semantic_queries") is not True
        or measurement.get("count_64_stress_parent_windows") is not True
        or measurement.get("model_load_allowed") is not False
        or measurement.get("onnx_session_allowed") is not False
        or measurement.get("model_inference_allowed") is not False
        or measurement.get("vectors_allowed") is not False
        or measurement.get("paper_text_retained_in_report") is not False
        or measurement.get("query_text_retained_in_report") is not False
    ):
        errors.append("E5-small window-census execution boundary changed")

    gates = payload.get("gates") or {}
    if gates != {
        "maximum_input_tokens_per_window": 512,
        "maximum_truncated_parent_count": 0,
        "maximum_uncovered_content_token_count": 0,
        "maximum_truncated_query_count": 0,
        "maximum_window_count_multiplier": 2.0,
        "maximum_projected_index_bytes": 10000000,
        "embedding_dimension": 384,
        "float_bytes": 4,
        "all_required": True,
        "pass_decision": "window_census_passed_freeze_stage_1",
        "reject_decision": "e5_small_rejected_before_inference",
    }:
        errors.append("E5-small window-census gates changed")

    boundary = payload.get("product_boundary") or {}
    if (
        boundary.get("library_papers_only") is not True
        or boundary.get("evidence_retrieval_only") is not True
        or boundary.get("writing_or_submission_workflow_allowed") is not False
        or boundary.get("product_enabled") is not False
        or boundary.get("ordinary_user_surface_exposed") is not False
    ):
        errors.append("E5-small window-census product boundary changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] E5-small window-census protocol cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_protocol(payload)
    if sha256_file(DEFAULT_PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        errors.append("E5-small window-census protocol SHA-256 changed")
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] E5-small full-coverage window census protocol is frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
