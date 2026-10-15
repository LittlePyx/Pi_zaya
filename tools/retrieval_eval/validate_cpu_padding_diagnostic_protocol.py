from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "retrieval_cpu_padding_diagnostic_protocol_v1.json"


def _sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def validate_protocol(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id") != "retrieval_cpu_padding_diagnostic_v1"
    ):
        errors.append("CPU padding protocol identity changed")
    parent = payload.get("parent") or {}
    parent_path = REPO_ROOT / str(parent.get("report_path") or "")
    if (
        parent
        != {
            "report_path": "docs/retrieval_cpu_full_index_feasibility_v1.json",
            "report_sha256": "866e2abb58566cb0f569e5eb4c3dca20e6f7764d60793dd68c2ea4e8f85f282a",
            "result_fingerprint": "3085c4ed8cf73b2e68fa071b569610ef1812902c726f1af7b6e91e7ddaae00af",
            "decision": "rejected_cpu_full_index_runtime",
            "full_build_ms": 4952500.783,
            "peak_process_rss_mib": 15751.07,
            "max_observed_tokens": 4712,
        }
        or not parent_path.is_file()
        or parent.get("report_sha256") != _sha256(parent_path)
    ):
        errors.append("CPU padding parent changed")
    if payload.get("corpus") != {
        "runtime_db_argument_required": True,
        "tree_sha256": "deb60a97bdfc50b0273de07ebe1c775352ffb2c5e4adeb3e925d48f7f25cafaf",
        "document_count": 10,
        "chunk_count": 1673,
    }:
        errors.append("CPU padding corpus changed")
    if payload.get("tokenizer") != {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "local_files_only": True,
        "trust_remote_code": False,
        "add_special_tokens": True,
        "maximum_effective_length": 8192,
        "model_load_allowed": False,
        "embedding_allowed": False,
    }:
        errors.append("CPU padding tokenizer boundary changed")
    if payload.get("strategies") != [
        {
            "id": "document_order_fixed_8",
            "sort_within_document": False,
            "maximum_batch_size": 8,
            "maximum_padded_tokens_per_batch": None,
            "oversize_singleton": False,
        },
        {
            "id": "length_bucketed_fixed_8",
            "sort_within_document": True,
            "maximum_batch_size": 8,
            "maximum_padded_tokens_per_batch": None,
            "oversize_singleton": False,
        },
        {
            "id": "length_bucketed_token_budget_4096",
            "sort_within_document": True,
            "maximum_batch_size": 8,
            "maximum_padded_tokens_per_batch": 4096,
            "oversize_singleton": True,
        },
    ]:
        errors.append("CPU padding strategies changed")
    if payload.get("branch_gate") != {
        "candidate_strategy": "length_bucketed_token_budget_4096",
        "minimum_attention_work_proxy_speedup": 4.0,
        "minimum_peak_padded_tokens_reduction": 0.5,
        "maximum_projected_build_ms_for_bounded_followup": 900000.0,
        "pass_decision": "bounded_length_aware_qwen_sample_allowed",
        "reject_decision": "skip_native_qwen_and_screen_lighter_runtime",
    }:
        errors.append("CPU padding branch gate changed")
    execution = payload.get("execution") or {}
    if (
        execution.get("maximum_runs") != 1
        or execution.get("queries_encoded") != 0
        or execution.get("documents_embedded") != 0
        or execution.get("vectors_created") != 0
        or execution.get("product_enabled") is not False
        or execution.get("ordinary_user_surface_exposed") is not False
    ):
        errors.append("CPU padding execution boundary changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] CPU padding protocol cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_protocol(payload)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] CPU padding diagnostic is frozen before tokenizer load")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
