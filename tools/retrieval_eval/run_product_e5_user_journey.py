from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.embedding_index import _snapshot_from_db as load_embedding_corpus_snapshot
from kb.retrieval_benchmark import validate_indexed_corpus
from kb.retrieval_eval_fixture import sha256_file
from kb.retrieval_semantic_eval_fixture import load_retrieval_benchmark_fixture
from kb.store import atomic_write_json, load_all_chunks
from tools.retrieval_eval.lighter_runtime_onnx_worker import (
    _raw_sha256,
    _tree_fingerprint,
)
from tools.retrieval_eval.run_lighter_runtime_stage1_e5_small import DEFAULT_PYTHON
from tools.retrieval_eval.run_product_e5_selector_shadow import (
    CANDIDATE_CHANNEL,
    DEFAULT_WORKER,
    PRODUCT_CHANNEL,
    _load_json,
    _result_fingerprint,
    _utc_now,
    evaluate_runtime_gate,
)
from tools.retrieval_eval.validate_product_e5_user_journey_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_product_e5_user_journey_v1.json"
EXPECTED_RESULT_FINGERPRINT: str | None = (
    "4bedbf5ae2b76f800474aee13ab190c2268dd3948a2c369eddf9d6feb39d7fcf"
)


def retrieval_snapshot_sha256(db_dir: Path) -> str:
    files = [db_dir / "docs.json", *sorted((db_dir / "chunks").glob("*.jsonl"))]
    if not files or any(not path.is_file() for path in files):
        raise RuntimeError("retrieval snapshot files are missing")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(db_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def copy_retrieval_snapshot(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / "docs.json", destination / "docs.json")
    shutil.copytree(source / "chunks", destination / "chunks")


def _case_regressions(
    product: dict[str, Any],
    candidate: dict[str, Any],
    *,
    metric: str,
    increase_is_regression: bool = False,
) -> list[str]:
    product_rows = {str(row["id"]): row for row in product.get("cases") or []}
    candidate_rows = {str(row["id"]): row for row in candidate.get("cases") or []}
    if set(product_rows) != set(candidate_rows):
        raise ValueError("product/candidate case membership changed")
    out: list[str] = []
    for case_id in sorted(product_rows):
        baseline = float(product_rows[case_id]["metrics"][metric])
        observed = float(candidate_rows[case_id]["metrics"][metric])
        regressed = observed > baseline if increase_is_regression else observed < baseline
        if regressed:
            out.append(case_id)
    return out


def evaluate_quality_gate(
    channels: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    contract = protocol["quality_gates"]
    product = channels[PRODUCT_CHANNEL]
    candidate = channels[CANDIDATE_CHANNEL]
    modes: dict[str, Any] = {}
    for mode in ("scoped", "full_library"):
        product_mode = product["modes"][mode]
        candidate_mode = candidate["modes"][mode]
        product_metrics = product_mode["metrics"]
        candidate_metrics = candidate_mode["metrics"]
        aggregate_checks = {
            f"{metric}_no_regression": float(candidate_metrics[metric])
            >= float(product_metrics[metric])
            for metric in contract["required_no_regression_each_mode"]
        }
        aggregate_checks.update(
            {
                f"{metric}_no_increase": float(candidate_metrics[metric])
                <= float(product_metrics[metric])
                for metric in contract["wrong_source_rate_no_regression_each_mode"]
            }
        )
        case_regressions = {
            "recall@10": _case_regressions(
                product_mode,
                candidate_mode,
                metric="recall@10",
            ),
            "contract_recall@10": _case_regressions(
                product_mode,
                candidate_mode,
                metric="contract_recall@10",
            ),
            "mrr": _case_regressions(product_mode, candidate_mode, metric="mrr"),
            "wrong_source_rate@10_increase": _case_regressions(
                product_mode,
                candidate_mode,
                metric="wrong_source_rate@10",
                increase_is_regression=True,
            ),
        }
        case_checks = {
            f"maximum_{metric}_case_regressions": len(case_regressions[metric])
            <= int(maximum)
            for metric, maximum in contract["maximum_case_regressions_each_mode"].items()
        }
        checks = {**aggregate_checks, **case_checks}
        modes[mode] = {
            "recall_at_10_gain": round(
                float(candidate_metrics["recall@10"])
                - float(product_metrics["recall@10"]),
                6,
            ),
            "case_regressions": case_regressions,
            "checks": checks,
            "passes_all": all(checks.values()),
        }
    minimum_gain = float(contract["minimum_recall_at_10_gain_in_at_least_one_mode"])
    gain_check = any(item["recall_at_10_gain"] >= minimum_gain for item in modes.values())
    return {
        "modes": modes,
        "minimum_gain_in_at_least_one_mode": gain_check,
        "passes_all": gain_check and all(item["passes_all"] for item in modes.values()),
    }


def _decision(
    protocol: dict[str, Any],
    runtime_gate: dict[str, Any],
    quality_gate: dict[str, Any],
) -> dict[str, Any]:
    contract = protocol["decision"]
    if not runtime_gate["passes_all"]:
        status = contract["runtime_failure_status"]
    elif not quality_gate["passes_all"]:
        status = contract["quality_failure_status"]
    else:
        status = contract["pass_status"]
    return {
        "status": status,
        "product_enabled": False,
        "user_mode_control_visible": False,
        "candidate_tuning_allowed": False,
        "next_action": (
            "run_portable_packaging_experiment"
            if status == contract["pass_status"]
            else "keep_bm25_and_record_non_tunable_failure"
        ),
    }


def _preflight(source_db: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    execution = protocol["execution"]
    root = REPO_ROOT / execution["working_root"]
    report_path = REPO_ROOT / execution["report_path"]
    if root.exists() or report_path.exists():
        raise RuntimeError("product E5 user-journey execution artifacts already exist")
    source_contract = protocol["source_corpus"]
    snapshot_sha = retrieval_snapshot_sha256(source_db)
    if snapshot_sha != source_contract["retrieval_snapshot_sha256"]:
        raise RuntimeError("product E5 user-journey retrieval snapshot changed")
    snapshot = load_embedding_corpus_snapshot(source_db)
    if (
        snapshot.committed_docs_sha256 != source_contract["committed_docs_sha256"]
        or snapshot.corpus_fingerprint != source_contract["index_corpus_fingerprint"]
        or len(snapshot.chunks) != source_contract["parent_chunk_count"]
    ):
        raise RuntimeError("product E5 user-journey product corpus identity changed")
    fixture_path = REPO_ROOT / source_contract["fixture_path"]
    fixture = load_retrieval_benchmark_fixture(fixture_path, repo_root=REPO_ROOT)
    _mapping, errors = validate_indexed_corpus(fixture, load_all_chunks(source_db))
    if errors:
        raise RuntimeError("product E5 user-journey qrels mismatch: " + "; ".join(errors))

    candidate = protocol["candidate"]
    model_path = REPO_ROOT / candidate["model_path"]
    tokenizer_path = REPO_ROOT / candidate["tokenizer_path"]
    for path, size, sha in (
        (model_path, candidate["model_bytes"], candidate["model_sha256"]),
        (tokenizer_path, candidate["tokenizer_bytes"], candidate["tokenizer_sha256"]),
    ):
        if not path.is_file() or path.stat().st_size != int(size) or _raw_sha256(path) != sha:
            raise RuntimeError(f"product E5 user-journey artifact changed: {path.name}")

    runtime = protocol["runtime_environment"]
    model_python = REPO_ROOT / runtime["model_runtime_python"]
    dependency_root = REPO_ROOT / runtime["product_dependency_site_packages"]
    rank_path = dependency_root / "rank_bm25.py"
    if (
        not DEFAULT_WORKER.is_file()
        or not model_python.is_file()
        or model_python.resolve() != DEFAULT_PYTHON.resolve()
        or not rank_path.is_file()
        or _raw_sha256(rank_path) != runtime["rank_bm25_source_sha256"].lower()
    ):
        raise RuntimeError("product E5 user-journey runtime dependency changed")
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(dependency_root)
    check = subprocess.run(
        [
            str(model_python),
            "-c",
            (
                "import importlib.metadata as m,sys;"
                "assert sys.version_info[:3]==(3,10,11);"
                "assert m.version('numpy')=='2.2.6';"
                "assert m.version('onnxruntime')=='1.23.2';"
                "assert m.version('tokenizers')=='0.21.4';"
                "assert m.version('rank-bm25')=='0.2.2';"
                "import rank_bm25"
            ),
        ],
        cwd=str(REPO_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if check.returncode:
        raise RuntimeError("product E5 user-journey isolated runtime changed")
    return {
        "snapshot_sha256": snapshot_sha,
        "fixture_path": fixture_path,
        "artifact_root": model_path.parent,
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "artifact_tree_sha256": _tree_fingerprint(model_path.parent),
        "dependency_root": dependency_root,
    }


def validate_report(
    report: dict[str, Any],
    *,
    protocol: dict[str, Any],
    expected_result_fingerprint: str | None = EXPECTED_RESULT_FINGERPRINT,
) -> list[str]:
    errors: list[str] = []
    if (
        report.get("schema_version") != 1
        or report.get("report_kind") != "retrieval_product_e5_user_journey_v1"
        or report.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL)
    ):
        errors.append("product E5 user-journey report identity changed")
    worker = {
        "runtime": report.get("runtime") or {},
        "index": report.get("index") or {},
        "selector_trace": report.get("selector_trace") or [],
        "channels": report.get("channels") or {},
        "execution": report.get("execution") or {},
    }
    expected_runtime = evaluate_runtime_gate(worker, protocol, report.get("integrity") or {})
    if report.get("runtime_gate") != expected_runtime:
        errors.append("product E5 user-journey runtime gate is inconsistent")
    try:
        expected_quality = evaluate_quality_gate(worker["channels"], protocol)
    except (KeyError, TypeError, ValueError):
        expected_quality = {}
        errors.append("product E5 user-journey channels are malformed")
    if report.get("quality_gate") != expected_quality:
        errors.append("product E5 user-journey quality gate is inconsistent")
    expected_decision = _decision(protocol, expected_runtime, expected_quality)
    if report.get("decision") != expected_decision:
        errors.append("product E5 user-journey decision is inconsistent")
    try:
        fingerprint = _result_fingerprint(report)
    except (KeyError, TypeError):
        fingerprint = ""
    if report.get("result_fingerprint") != fingerprint:
        errors.append("product E5 user-journey result fingerprint is inconsistent")
    if expected_result_fingerprint and fingerprint != expected_result_fingerprint:
        errors.append("pinned product E5 user-journey result changed")
    return errors


def run(source_db: Path) -> tuple[dict[str, Any], list[str]]:
    protocol = _load_json(DEFAULT_PROTOCOL)
    errors = validate_protocol(protocol, inspect_implementation_files=True)
    if errors:
        raise RuntimeError("product E5 user-journey protocol invalid: " + "; ".join(errors))
    preflight = _preflight(source_db, protocol)
    execution = protocol["execution"]
    root = REPO_ROOT / execution["working_root"]
    working_db = REPO_ROOT / execution["working_db_path"]
    index_dir = REPO_ROOT / execution["product_index_path"]
    attempt_path = REPO_ROOT / execution["attempt_receipt_path"]
    worker_path = REPO_ROOT / execution["worker_result_path"]
    report_path = REPO_ROOT / execution["report_path"]

    root.mkdir(parents=True, exist_ok=False)
    copy_retrieval_snapshot(source_db, working_db)
    working_before = retrieval_snapshot_sha256(working_db)
    source_after_copy = retrieval_snapshot_sha256(source_db)
    if working_before != preflight["snapshot_sha256"] or source_after_copy != working_before:
        raise RuntimeError("product E5 user-journey source changed during snapshot copy")
    atomic_write_json(
        attempt_path,
        {
            "schema_version": 1,
            "attempt_kind": "retrieval_product_e5_user_journey_v1",
            "started_at": _utc_now(),
            "maximum_completed_full_builds": 1,
            "source_snapshot_sha256": working_before,
            "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        },
    )

    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(preflight["dependency_root"])
    command = [
        str(DEFAULT_PYTHON),
        str(DEFAULT_WORKER),
        "--db-dir",
        str(working_db),
        "--index-dir",
        str(index_dir),
        "--artifact-root",
        str(preflight["artifact_root"]),
        "--fixture",
        str(preflight["fixture_path"]),
        "--output",
        str(worker_path),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=environment,
            capture_output=True,
            text=True,
            timeout=int(execution["hard_subprocess_timeout_seconds"]),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("product E5 user-journey worker exceeded hard timeout") from exc
    wall_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode:
        lines = (completed.stdout + "\n" + completed.stderr).strip().splitlines()
        failures = [line for line in lines if "[FAIL]" in line]
        detail = failures[-1:] or lines[-1:]
        raise RuntimeError(detail[0][:240] if detail else "product E5 user-journey worker failed")
    worker = _load_json(worker_path)
    if worker.get("worker_kind") != "retrieval_product_e5_selector_shadow_worker_v1":
        raise RuntimeError("product E5 user-journey worker identity changed")

    source_after = retrieval_snapshot_sha256(source_db)
    working_after = retrieval_snapshot_sha256(working_db)
    artifact_after = _tree_fingerprint(preflight["artifact_root"])
    integrity = {
        "source_before_sha256": preflight["snapshot_sha256"],
        "source_after_sha256": source_after,
        "working_db_before_sha256": working_before,
        "working_db_after_sha256": working_after,
        "artifact_tree_before_sha256": preflight["artifact_tree_sha256"],
        "artifact_tree_after_sha256": artifact_after,
        "model_before_sha256": protocol["candidate"]["model_sha256"],
        "model_after_sha256": _raw_sha256(preflight["model_path"]),
        "tokenizer_before_sha256": protocol["candidate"]["tokenizer_sha256"],
        "tokenizer_after_sha256": _raw_sha256(preflight["tokenizer_path"]),
    }
    runtime_gate = evaluate_runtime_gate(worker, protocol, integrity)
    quality_gate = evaluate_quality_gate(worker["channels"], protocol)
    decision = _decision(protocol, runtime_gate, quality_gate)
    report = {
        "schema_version": 1,
        "report_kind": "retrieval_product_e5_user_journey_v1",
        "generated_at": _utc_now(),
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "source": {
            "document_count": protocol["source_corpus"]["document_count"],
            "parent_chunk_count": protocol["source_corpus"]["parent_chunk_count"],
            "case_count": protocol["source_corpus"]["case_count"],
            "corpus_fingerprint": protocol["source_corpus"]["corpus_fingerprint"],
            "evaluation_status": protocol["evaluation_status"],
        },
        "runtime": {**worker["runtime"], "subprocess_wall_ms": round(wall_ms, 3)},
        "index": worker["index"],
        "selector_trace": worker["selector_trace"],
        "channels": worker["channels"],
        "execution": worker["execution"],
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
        report, errors = run(args.source_db_root.resolve())
    except Exception as exc:
        print(f"[FAIL] product E5 user-journey replay: {type(exc).__name__}: {exc}")
        return 1
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    status = report["decision"]["status"]
    if status != "current_domain_suite_passed_authorize_packaging_experiment":
        print(f"[REJECT] product E5 user-journey replay: {status}")
        return 2
    print(f"[OK] product E5 user-journey replay passed: {report['result_fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
