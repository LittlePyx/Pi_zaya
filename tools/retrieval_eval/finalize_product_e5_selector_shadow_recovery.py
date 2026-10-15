from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.local_semantic_window_index import LocalSemanticWindowIndexStore
from kb.retrieval_eval_fixture import sha256_file
from kb.store import atomic_write_json
from tools.retrieval_eval.lighter_runtime_onnx_worker import (
    _raw_sha256,
    _tree_fingerprint,
)
from tools.retrieval_eval.run_product_e5_selector_shadow import (
    DEFAULT_PROTOCOL,
    _decision,
    _result_fingerprint,
    _utc_now,
    evaluate_quality_gate,
    evaluate_runtime_gate,
    validate_report,
)
from tools.retrieval_eval.validate_product_e5_selector_shadow_protocol import (
    validate_protocol,
)


DEFAULT_RECOVERY = (
    REPO_ROOT
    / "docs"
    / "retrieval_product_e5_selector_shadow_measurement_recovery_v1.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


def validate_recovery(payload: dict[str, Any], *, protocol_sha256: str) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("recovery_id")
        != "retrieval_product_e5_selector_shadow_measurement_recovery_v1"
        or payload.get("protocol_sha256") != protocol_sha256
        or payload.get("failure_stage")
        != "after_atomic_build_readback_and_selector_replay_before_worker_payload_serialization"
        or payload.get("reason_id") != "windows_peak_rss_handle_signature_missing"
    ):
        errors.append("product E5 selector measurement recovery identity changed")
    build = payload.get("completed_build") or {}
    if (
        build.get("count") != 1
        or build.get("parent_count") != 1673
        or build.get("window_count") != 2016
        or build.get("vectors_sha256")
        != "6c2d100ccbf6d1ea50c8f59237b8f8d7f723ad04c999086a2f0edbfa2d10d73f"
        or float(build.get("build_elapsed_upper_bound_ms") or 0.0) <= 0.0
        or build.get("atomic_readback_passed") is not True
    ):
        errors.append("product E5 selector completed-build recovery changed")
    fix = payload.get("measurement_fix") or {}
    if (
        fix.get("model_index_or_selector_changed") is not False
        or fix.get("gate_changed") is not False
        or fix.get("second_full_build_allowed") is not False
    ):
        errors.append("product E5 selector measurement fix boundary changed")
    replay = payload.get("recovery_replay") or {}
    if (
        replay.get("index_build_was_no_op") is not True
        or replay.get("embedded_parent_count") != 0
        or replay.get("embedded_window_count") != 0
        or replay.get("selection_call_count") != 24
        or replay.get("fallback_count") != 0
        or replay.get("source_escape_count") != 0
        or replay.get("protected_prefix_rate") != 1.0
        or replay.get("product_sparse_input_unchanged_rate") != 1.0
    ):
        errors.append("product E5 selector recovery replay changed")
    integrity = payload.get("integrity") or {}
    if (
        not integrity.get("artifact_tree_sha256")
        or integrity.get("source_db_changed") is not False
        or integrity.get("working_db_changed") is not False
        or integrity.get("model_or_tokenizer_changed") is not False
        or integrity.get("product_enabled") is not False
        or integrity.get("ordinary_user_surface_exposed") is not False
        or integrity.get("network_used") is not False
        or payload.get("report_recovery_authorized") is not True
    ):
        errors.append("product E5 selector recovery integrity changed")
    return errors


def finalize(source_db: Path) -> tuple[dict[str, Any], list[str]]:
    protocol = _load_json(DEFAULT_PROTOCOL)
    protocol_errors = validate_protocol(protocol)
    if protocol_errors:
        raise RuntimeError("product E5 selector protocol invalid: " + "; ".join(protocol_errors))
    protocol_sha = sha256_file(DEFAULT_PROTOCOL)
    recovery = _load_json(DEFAULT_RECOVERY)
    recovery_errors = validate_recovery(recovery, protocol_sha256=protocol_sha)
    if recovery_errors:
        raise RuntimeError("product E5 selector recovery invalid: " + "; ".join(recovery_errors))

    execution = protocol["execution"]
    root = REPO_ROOT / execution["working_root"]
    working_db = REPO_ROOT / execution["working_db_path"]
    index_dir = REPO_ROOT / execution["product_index_path"]
    attempt_path = REPO_ROOT / execution["attempt_receipt_path"]
    report_path = REPO_ROOT / execution["report_path"]
    worker_path = root / "worker_recovery_v1.json"
    if not root.is_dir() or report_path.exists() or not worker_path.is_file():
        raise RuntimeError("product E5 selector recovery inputs are unavailable")
    if _raw_sha256(attempt_path) != recovery["attempt_receipt_sha256"]:
        raise RuntimeError("product E5 selector recovery attempt changed")
    if _raw_sha256(worker_path) != recovery["recovery_replay"]["worker_result_sha256"]:
        raise RuntimeError("product E5 selector recovery worker changed")

    pointer_path = index_dir / "current.json"
    pointer = _load_json(pointer_path)
    build_id = str(pointer.get("build_id") or "")
    build = recovery["completed_build"]
    manifest_path = index_dir / "generations" / build_id / "manifest.json"
    manifest = _load_json(manifest_path)
    if (
        build_id != build["build_id"]
        or _raw_sha256(pointer_path) != build["current_pointer_sha256"]
        or _raw_sha256(manifest_path) != build["manifest_sha256"]
        or manifest.get("vectors_sha256") != build["vectors_sha256"]
        or _tree_fingerprint(index_dir) != build["index_tree_sha256_after_recovery"]
    ):
        raise RuntimeError("product E5 selector recovered index changed")
    status = LocalSemanticWindowIndexStore(working_db, index_dir=index_dir).status()
    if (
        status.state != "ready"
        or status.build_id != build_id
        or status.parent_count != build["parent_count"]
        or status.window_count != build["window_count"]
    ):
        raise RuntimeError("product E5 selector recovered index is not ready")

    candidate = protocol["candidate"]
    model_path = REPO_ROOT / candidate["model_path"]
    tokenizer_path = REPO_ROOT / candidate["tokenizer_path"]
    artifact_root = model_path.parent
    artifact_tree = _tree_fingerprint(artifact_root)
    if (
        artifact_tree != recovery["integrity"]["artifact_tree_sha256"]
        or _raw_sha256(model_path) != candidate["model_sha256"]
        or _raw_sha256(tokenizer_path) != candidate["tokenizer_sha256"]
    ):
        raise RuntimeError("product E5 selector recovery artifacts changed")
    source_sha = _tree_fingerprint(source_db)
    working_sha = _tree_fingerprint(working_db)
    if (
        source_sha != protocol["source_corpus"]["source_tree_sha256"]
        or working_sha != source_sha
    ):
        raise RuntimeError("product E5 selector recovery corpus changed")

    worker = _load_json(worker_path)
    replay_index = worker.get("index") or {}
    if (
        replay_index.get("no_op") is not True
        or replay_index.get("embedded_parent_count") != 0
        or replay_index.get("embedded_window_count") != 0
    ):
        raise RuntimeError("product E5 selector recovery performed another build")
    recovered_worker = {
        **worker,
        "runtime": {
            **worker["runtime"],
            "full_build_ms": build["build_elapsed_upper_bound_ms"],
            "full_build_time_evidence": "conservative_attempt_to_atomic_pointer_upper_bound",
            "peak_process_rss_evidence": "direct_same_runtime_graph_recovery_replay",
            "recovery_no_op_ms": worker["runtime"]["full_build_ms"],
        },
        "index": {
            **replay_index,
            "embedded_parent_count": build["parent_count"],
            "embedded_window_count": build["window_count"],
            "reused_parent_count": 0,
            "reused_window_count": 0,
            "no_op": False,
            "recovery_readback_no_op": True,
        },
        "execution": {
            **worker["execution"],
            "full_build_count": 1,
            "measurement_recovery_used": True,
        },
    }
    integrity = {
        "source_before_sha256": source_sha,
        "source_after_sha256": source_sha,
        "working_db_before_sha256": source_sha,
        "working_db_after_sha256": working_sha,
        "artifact_tree_before_sha256": artifact_tree,
        "artifact_tree_after_sha256": artifact_tree,
        "model_before_sha256": candidate["model_sha256"],
        "model_after_sha256": _raw_sha256(model_path),
        "tokenizer_before_sha256": candidate["tokenizer_sha256"],
        "tokenizer_after_sha256": _raw_sha256(tokenizer_path),
    }
    runtime_gate = evaluate_runtime_gate(recovered_worker, protocol, integrity)
    quality_gate = evaluate_quality_gate(recovered_worker["channels"], protocol)
    decision = _decision(protocol, runtime_gate, quality_gate)
    report = {
        "schema_version": 1,
        "report_kind": "retrieval_product_e5_selector_shadow_v1",
        "generated_at": _utc_now(),
        "protocol_sha256": protocol_sha,
        "measurement_recovery": {
            "path": DEFAULT_RECOVERY.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(DEFAULT_RECOVERY),
        },
        "source": {
            "document_count": protocol["source_corpus"]["document_count"],
            "parent_chunk_count": protocol["source_corpus"]["parent_chunk_count"],
            "semantic_case_count": protocol["source_corpus"]["semantic_case_count"],
            "corpus_fingerprint": protocol["source_corpus"]["corpus_fingerprint"],
        },
        "runtime": recovered_worker["runtime"],
        "index": recovered_worker["index"],
        "selector_trace": recovered_worker["selector_trace"],
        "channels": recovered_worker["channels"],
        "execution": recovered_worker["execution"],
        "integrity": integrity,
        "runtime_gate": runtime_gate,
        "quality_gate": quality_gate,
        "decision": decision,
    }
    report["result_fingerprint"] = _result_fingerprint(report)
    atomic_write_json(report_path, report)
    return report, validate_report(report, protocol=protocol)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        report, errors = finalize(args.source_db_root.resolve())
    except Exception as exc:
        print(f"[FAIL] product E5 selector recovery: {type(exc).__name__}: {exc}")
        return 1
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print(
        "[OK] product E5 selector recovered report: "
        f"{report['decision']['status']} {report['result_fingerprint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
