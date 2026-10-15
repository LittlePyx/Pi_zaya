from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.prepare_lighter_runtime_candidate import (
    DEFAULT_PROTOCOL,
    DEFAULT_REQUIREMENTS,
    _locked_requirements,
)


DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage0_qwen_v1.json"
EXPECTED_REPORT_SHA256 = "1c7b2eed6469e002b229337ce5e90bd4770f202818002b4c3192bf8488f75fb9"


def validate_report(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1 or payload.get("report_kind") != "retrieval_lighter_runtime_stage0_v1":
        errors.append("lighter runtime stage-0 report identity changed")
    if payload.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL):
        errors.append("lighter runtime stage-0 protocol identity changed")
    if payload.get("requirements_sha256") != sha256_file(DEFAULT_REQUIREMENTS):
        errors.append("lighter runtime stage-0 requirements identity changed")

    candidate = payload.get("candidate") or {}
    if (
        candidate.get("candidate_id")
        != "qwen3_embedding_0_6b_official_quint8_avx2_onnx_v1"
        or candidate.get("evaluation_order") != 1
        or candidate.get("model_revision") != "bd58e9fd4b0467770da53678d8da25481c0f959b"
        or candidate.get("artifact_bytes") != 600127788
        or candidate.get("artifact_sha256")
        != "a7eda29cc374b01ae75eb1af1ab65c3c61a64c43f39f16e08c6e37c075dcbf4d"
        or candidate.get("artifact_reused") is not False
        or not isinstance(candidate.get("downloaded_bytes_this_run"), int)
        or candidate.get("download_attempts_this_run") != 1
    ):
        errors.append("lighter runtime stage-0 candidate evidence changed")

    environment = payload.get("environment") or {}
    distribution_versions = {
        str(row.get("name") or ""): str(row.get("version") or "")
        for row in environment.get("distributions") or []
    }
    if (
        environment.get("python_version") != "3.10.11"
        or environment.get("virtual_environment") is not True
        or environment.get("user_site_enabled") is not False
        or environment.get("locked_distribution_count") != 24
        or environment.get("installed_distribution_count_excluding_bootstrap") != 24
        or distribution_versions != _locked_requirements()
        or environment.get("installed_distribution_bytes") != 204279163
        or {"torch", "transformers", "optimum"} & set(distribution_versions)
    ):
        errors.append("lighter runtime stage-0 isolated environment changed")

    onnx = payload.get("onnx") or {}
    if (
        onnx.get("onnx_version") != "1.18.0"
        or onnx.get("onnxruntime_version") != "1.23.2"
        or onnx.get("session_providers") != ["CPUExecutionProvider"]
        or onnx.get("inputs")
        != [
            {
                "name": "input_ids",
                "type": "tensor(int64)",
                "shape": ["batch_size", "sequence_length"],
            },
            {
                "name": "attention_mask",
                "type": "tensor(int64)",
                "shape": ["batch_size", "sequence_length"],
            },
        ]
        or onnx.get("outputs")
        != [
            {
                "name": "token_embeddings",
                "type": "tensor(float)",
                "shape": ["batch_size", "sequence_length", 1024],
            },
            {
                "name": "sentence_embedding",
                "type": "tensor(float)",
                "shape": ["batch_size", 1024],
            },
        ]
        or onnx.get("session_run_calls") != 0
    ):
        errors.append("lighter runtime stage-0 ONNX graph evidence changed")

    execution = payload.get("execution") or {}
    if execution != {
        "paper_text_accessed": False,
        "queries_encoded": 0,
        "documents_embedded": 0,
        "vectors_created": 0,
        "model_inference_runs": 0,
        "product_requirements_modified": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }:
        errors.append("lighter runtime stage-0 execution boundary changed")
    if payload.get("decision") != {
        "status": "passed_stage_0",
        "stage_1_allowed": True,
        "product_enabled": False,
    }:
        errors.append("lighter runtime stage-0 decision changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] lighter runtime stage-0 report cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_report(payload)
    if sha256_file(DEFAULT_REPORT) != EXPECTED_REPORT_SHA256:
        errors.append("lighter runtime stage-0 report SHA-256 changed")
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] Qwen official AVX2 ONNX candidate passed pinned zero-inference stage 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
