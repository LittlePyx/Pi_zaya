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


DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage0_e5_small_v1.json"
EXPECTED_REPORT_SHA256 = "5d901b01af8201004adde29119321b60fb82a781b54e93688b22f25d3fde963d"
EXPECTED_TOKENIZER_FILES = [
    {
        "repository_path": "onnx/config.json",
        "local_name": "config.json",
        "bytes": 653,
        "sha256": "bbb7c1333fc4b3e27fbc9cd5d2070aabcc1d4dfb99917c3633e772f97545a6b6",
        "downloaded_bytes_this_run": 653,
        "reused": False,
    },
    {
        "repository_path": "onnx/sentencepiece.bpe.model",
        "local_name": "sentencepiece.bpe.model",
        "bytes": 5069051,
        "sha256": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
        "downloaded_bytes_this_run": 5069051,
        "reused": False,
    },
    {
        "repository_path": "onnx/special_tokens_map.json",
        "local_name": "special_tokens_map.json",
        "bytes": 167,
        "sha256": "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7",
        "downloaded_bytes_this_run": 167,
        "reused": False,
    },
    {
        "repository_path": "onnx/tokenizer.json",
        "local_name": "tokenizer.json",
        "bytes": 17082730,
        "sha256": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
        "downloaded_bytes_this_run": 17082730,
        "reused": False,
    },
    {
        "repository_path": "onnx/tokenizer_config.json",
        "local_name": "tokenizer_config.json",
        "bytes": 443,
        "sha256": "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b",
        "downloaded_bytes_this_run": 443,
        "reused": False,
    },
]


def validate_report(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("report_kind") != "retrieval_lighter_runtime_stage0_v1"
    ):
        errors.append("E5-small Stage-0 report identity changed")
    if payload.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL):
        errors.append("E5-small Stage-0 protocol identity changed")
    if payload.get("requirements_sha256") != sha256_file(DEFAULT_REQUIREMENTS):
        errors.append("E5-small Stage-0 requirements identity changed")

    candidate = payload.get("candidate") or {}
    if (
        candidate.get("candidate_id") != "multilingual_e5_small_official_onnx_v1"
        or candidate.get("evaluation_order") != 2
        or candidate.get("model_revision") != "03415a4be176a1620747c692ed433219fabc3def"
        or candidate.get("artifact_bytes") != 470268510
        or candidate.get("artifact_sha256")
        != "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665"
        or candidate.get("artifact_reused") is not False
        or candidate.get("downloaded_bytes_this_run") != 470268510
        or candidate.get("download_attempts_this_run") != 1
    ):
        errors.append("E5-small Stage-0 candidate evidence changed")

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
        errors.append("E5-small Stage-0 isolated environment changed")

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
            {
                "name": "token_type_ids",
                "type": "tensor(int64)",
                "shape": ["batch_size", "sequence_length"],
            },
        ]
        or onnx.get("outputs")
        != [
            {
                "name": "last_hidden_state",
                "type": "tensor(float)",
                "shape": ["batch_size", "sequence_length", 384],
            }
        ]
        or onnx.get("session_run_calls") != 0
    ):
        errors.append("E5-small Stage-0 ONNX graph evidence changed")

    tokenizer = payload.get("tokenizer") or {}
    if (
        tokenizer.get("repository_revision")
        != "03415a4be176a1620747c692ed433219fabc3def"
        or tokenizer.get("files") != EXPECTED_TOKENIZER_FILES
        or tokenizer.get("tokenizer_load_calls") != 0
    ):
        errors.append("E5-small Stage-0 tokenizer evidence changed")

    if payload.get("execution") != {
        "paper_text_accessed": False,
        "queries_encoded": 0,
        "documents_embedded": 0,
        "vectors_created": 0,
        "model_inference_runs": 0,
        "product_requirements_modified": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }:
        errors.append("E5-small Stage-0 execution boundary changed")
    if payload.get("decision") != {
        "status": "passed_stage_0",
        "stage_1_allowed": True,
        "product_enabled": False,
    }:
        errors.append("E5-small Stage-0 decision changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] E5-small Stage-0 report cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_report(payload)
    if sha256_file(DEFAULT_REPORT) != EXPECTED_REPORT_SHA256:
        errors.append("E5-small Stage-0 report SHA-256 changed")
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] E5-small official ONNX candidate passed pinned zero-inference Stage 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
