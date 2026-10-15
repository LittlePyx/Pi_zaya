from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.e5_small_embedding_runtime import E5SmallOnnxExecutor
from kb.store import load_all_chunks
from tools.retrieval_eval.lighter_runtime_e5_small_worker import (
    build_document_window_ids as benchmark_document_window_ids,
)


DEFAULT_MODULE = REPO_ROOT / "kb" / "e5_small_embedding_runtime.py"
DEFAULT_STAGE2_REPORT = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage2_e5_small_v1.json"
DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_cases_v1.json"
DEFAULT_WORKER_RESULT = (
    REPO_ROOT
    / "test_results"
    / "retrieval_lighter_runtime_v1"
    / "stage2_e5_small_worker_v1.json"
)
DEFAULT_MODEL = (
    REPO_ROOT
    / "test_results"
    / "retrieval_lighter_runtime_v1"
    / "artifacts"
    / "multilingual_e5_small_official_onnx_v1"
    / "model.onnx"
)
DEFAULT_TOKENIZER = (
    REPO_ROOT
    / "test_results"
    / "retrieval_lighter_runtime_v1"
    / "artifacts"
    / "multilingual_e5_small_official_onnx_v1"
    / "tokenizer"
    / "tokenizer.json"
)
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_product_e5_runtime_parity_v1.json"
EXPECTED_MODULE_SHA256 = "25713a37f3c00f075bb076a0c31a8eab491a46c43d32d46fe677ccd000e60e3d"
EXPECTED_REPORT_SHA256: str | None = (
    "f900750d20f5fbe01a169efab6883feeb8743be08b5334b025be21c9817e16a6"
)
EXPECTED_QUERY_MATRIX_SHA256 = "6374e52e225e65333843be7aedfa5683414a60f3465f63b43434e5e54c245eed"
EXPECTED_WINDOWS_SHA256 = "562265dab68376fb2083057c05927dc92f0bc4f98c6c70876e0575c78ee883f9"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix_sha256(vectors: list[tuple[float, ...]]) -> str:
    import numpy as np

    return hashlib.sha256(
        np.asarray(vectors, dtype=np.float32).tobytes(order="C")
    ).hexdigest()


def _windows_sha256(windows: list[dict[str, object]]) -> str:
    value = json.dumps(windows, ensure_ascii=False, separators=(",", ":")) + "\n"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_report(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("report_kind") != "retrieval_product_e5_runtime_parity_v1"
    ):
        errors.append("product E5 parity report identity changed")
    authorization = payload.get("authorization") or {}
    if (
        authorization.get("stage_2_report_sha256") != _sha256(DEFAULT_STAGE2_REPORT)
        or authorization.get("product_runtime_source_sha256") != _sha256(DEFAULT_MODULE)
        or authorization.get("product_runtime_source_sha256") != EXPECTED_MODULE_SHA256
    ):
        errors.append("product E5 parity authorization changed")
    query = payload.get("query_parity") or {}
    if (
        query.get("query_count") != 12
        or query.get("expected_matrix_sha256") != EXPECTED_QUERY_MATRIX_SHA256
        or query.get("observed_matrix_sha256") != EXPECTED_QUERY_MATRIX_SHA256
        or query.get("maximum_absolute_difference") != 0.0
        or query.get("exact_float32_parity") is not True
    ):
        errors.append("product E5 query-vector parity changed")
    representation = payload.get("representation_parity") or {}
    if (
        representation.get("parent_count") != 2251
        or representation.get("product_window_count") != 2663
        or representation.get("benchmark_window_count") != 2663
        or representation.get("maximum_input_tokens") != 453
        or representation.get("frozen_stage_2_windows_sha256") != EXPECTED_WINDOWS_SHA256
        or representation.get("product_windows_sha256")
        != representation.get("benchmark_windows_sha256")
        or representation.get("exact_current_corpus_input_and_mapping_parity") is not True
        or representation.get("parent_or_window_text_retained") is not False
    ):
        errors.append("product E5 representation parity changed")
    execution = payload.get("execution") or {}
    if (
        execution.get("document_vectors_built") != 0
        or execution.get("index_generations_built") != 0
        or execution.get("product_enabled") is not False
        or execution.get("ordinary_user_surface_exposed") is not False
    ):
        errors.append("product E5 parity execution boundary changed")
    decision = payload.get("decision") or {}
    if (
        decision.get("passes_all") is not True
        or decision.get("status") != "product_executor_parity_passed"
        or decision.get("index_implementation_allowed") is not True
        or decision.get("product_enablement_allowed") is not False
    ):
        errors.append("product E5 parity decision changed")
    return errors


def run_parity(*, db_root: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise RuntimeError("product E5 parity report already exists")
    if _sha256(DEFAULT_MODULE) != EXPECTED_MODULE_SHA256:
        raise RuntimeError("product E5 runtime source changed before parity")
    stage2 = json.loads(DEFAULT_STAGE2_REPORT.read_text(encoding="utf-8"))
    worker = json.loads(DEFAULT_WORKER_RESULT.read_text(encoding="utf-8"))
    fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    executor = E5SmallOnnxExecutor(
        model_path=DEFAULT_MODEL,
        tokenizer_path=DEFAULT_TOKENIZER,
    )
    observed_vectors = [executor.embed_query(str(case["query"])) for case in fixture["cases"]]
    expected_vectors = [
        worker["query"]["vectors"][str(case["id"])] for case in fixture["cases"]
    ]
    import numpy as np

    observed_matrix = np.asarray(observed_vectors, dtype=np.float32)
    expected_matrix = np.asarray(expected_vectors, dtype=np.float32)
    maximum_difference = float(np.max(np.abs(observed_matrix - expected_matrix)))
    chunks = load_all_chunks(db_root)
    windows = executor.build_windows(chunks)
    public_windows = [window.public_metadata() for window in windows]
    benchmark_windows: list[dict[str, object]] = []
    exact_input_parity = True
    product_offset = 0
    prefix_ids = executor.tokenizer.encode("passage: ", add_special_tokens=False).ids
    for parent_position, chunk in enumerate(chunks):
        content_ids = executor.tokenizer.encode(
            str(chunk.get("text") or ""),
            add_special_tokens=False,
        ).ids
        rows = benchmark_document_window_ids(
            content_ids,
            prefix_ids=prefix_ids,
            bos_token_id=0,
            eos_token_id=2,
            window_tokens=448,
            stride_tokens=384,
        )
        for ordinal, row in enumerate(rows):
            product = windows[product_offset]
            exact_input_parity = exact_input_parity and tuple(row) == product.input_ids
            benchmark_windows.append(
                {
                    "parent_id": str(chunk["id"]),
                    "parent_position": parent_position,
                    "window_ordinal": ordinal,
                }
            )
            product_offset += 1
    observed_query_sha = _matrix_sha256(observed_vectors)
    product_windows_sha = _windows_sha256(public_windows)
    benchmark_windows_sha = _windows_sha256(benchmark_windows)
    passes = (
        observed_query_sha == EXPECTED_QUERY_MATRIX_SHA256
        and maximum_difference == 0.0
        and product_offset == len(windows)
        and max(len(window.input_ids) for window in windows) == 453
        and exact_input_parity
        and product_windows_sha == benchmark_windows_sha
    )
    payload = {
        "schema_version": 1,
        "report_kind": "retrieval_product_e5_runtime_parity_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "authorization": {
            "stage_2_report_sha256": _sha256(DEFAULT_STAGE2_REPORT),
            "stage_2_result_fingerprint": stage2["result_fingerprint"],
            "product_runtime_source_sha256": _sha256(DEFAULT_MODULE),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "onnxruntime_version": __import__("onnxruntime").__version__,
            "tokenizers_version": __import__("tokenizers").__version__,
            "session_providers": executor.session.get_providers(),
            "artifact_sha256": _sha256(DEFAULT_MODEL),
            "tokenizer_json_sha256": _sha256(DEFAULT_TOKENIZER),
        },
        "query_parity": {
            "query_count": len(observed_vectors),
            "expected_matrix_sha256": EXPECTED_QUERY_MATRIX_SHA256,
            "observed_matrix_sha256": observed_query_sha,
            "maximum_absolute_difference": maximum_difference,
            "exact_float32_parity": maximum_difference == 0.0,
        },
        "representation_parity": {
            "parent_count": len(chunks),
            "product_window_count": len(windows),
            "benchmark_window_count": len(benchmark_windows),
            "maximum_input_tokens": max(len(window.input_ids) for window in windows),
            "frozen_stage_2_windows_sha256": EXPECTED_WINDOWS_SHA256,
            "product_windows_sha256": product_windows_sha,
            "benchmark_windows_sha256": benchmark_windows_sha,
            "exact_current_corpus_input_and_mapping_parity": (
                exact_input_parity and product_windows_sha == benchmark_windows_sha
            ),
            "parent_or_window_text_retained": False,
        },
        "execution": {
            "query_vectors_built": len(observed_vectors),
            "document_vectors_built": 0,
            "index_generations_built": 0,
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
        },
        "decision": {
            "passes_all": passes,
            "status": (
                "product_executor_parity_passed"
                if passes
                else "product_executor_parity_failed"
            ),
            "index_implementation_allowed": passes,
            "product_enablement_allowed": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate product E5 executor parity")
    parser.add_argument("--db-root", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.validate_only:
            payload = json.loads(args.report.read_text(encoding="utf-8"))
        else:
            if args.db_root is None:
                raise ValueError("--db-root is required for parity execution")
            payload = run_parity(
                db_root=args.db_root.resolve(),
                output_path=args.report.resolve(),
            )
        errors = validate_report(payload)
        if EXPECTED_REPORT_SHA256 and _sha256(args.report) != EXPECTED_REPORT_SHA256:
            errors.append("product E5 parity report SHA-256 changed")
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[FAIL] product E5 runtime parity: {type(exc).__name__}")
        return 2
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] product E5 executor exactly matches the retained Stage-2 representation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
