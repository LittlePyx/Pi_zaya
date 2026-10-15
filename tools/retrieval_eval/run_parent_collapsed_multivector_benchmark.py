from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.dense_retriever import DenseRetriever
from kb.embedding_index import EmbeddingIndexStore
from kb.hybrid_retriever import HybridCandidateRetriever
from kb.parent_collapsed_dense_retriever import (
    ParentCollapsedDenseRetriever,
    focused_child_representations,
)
from kb.retrieval_benchmark import evaluate_candidate_channel, validate_indexed_corpus
from kb.retrieval_eval_fixture import sha256_file
from kb.retrieval_semantic_eval_fixture import load_retrieval_benchmark_fixture
from kb.store import load_all_chunks, save_docs_index, write_doc_chunks
from tools.retrieval_eval.product_query_sparse_retriever import (
    ProductQuerySparseRetriever,
)
from tools.retrieval_eval.run_dense_research_qa_benchmark import (
    _compact_channel,
    _index_size_bytes,
)
from tools.retrieval_eval.transformers_embedding_provider import (
    TransformersMeanPoolEmbeddingProvider,
)
from tools.retrieval_eval.validate_parent_collapsed_multivector_candidate import (
    DEFAULT_REGISTRY,
    validate_candidate,
)


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_eval_development_v1.json"
DEFAULT_BASELINE = REPO_ROOT / "docs" / "retrieval_bm25_baseline_development_v1.json"
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_parent_collapsed_multivector_development_v1.json"
DEFAULT_PARENT_INDEX = (
    REPO_ROOT
    / "test_results"
    / "retrieval_eval_development_v1"
    / "multilingual_e5_base_rrf_index"
)
DEFAULT_OUT_ROOT = REPO_ROOT / "test_results" / "retrieval_parent_collapsed_multivector"
PARENT_DENSE_CHANNEL = "parent_dense_multilingual_e5_base"
MULTIVECTOR_CHANNEL = "parent_collapsed_multivector_dense"
SAFE_RRF_CHANNEL = "parent_collapsed_multivector_safe_rrf"
_EXPECTED_MEASURED_METRICS = {
    PARENT_DENSE_CHANNEL: {
        "scoped": {"recall@10": 0.609756, "mrr": 0.299119, "ndcg@10": 0.364578},
        "full_library": {"recall@10": 0.414634, "mrr": 0.188353, "ndcg@10": 0.233349},
    },
    MULTIVECTOR_CHANNEL: {
        "scoped": {"recall@10": 0.634146, "mrr": 0.309413, "ndcg@10": 0.379426},
        "full_library": {"recall@10": 0.329268, "mrr": 0.180286, "ndcg@10": 0.205809},
    },
    SAFE_RRF_CHANNEL: {
        "scoped": {"recall@10": 0.695122, "mrr": 0.400011, "ndcg@10": 0.461845},
        "full_library": {"recall@10": 0.634146, "mrr": 0.289791, "ndcg@10": 0.359362},
    },
}


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return ""


def _metrics(channel: dict[str, Any], mode: str) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in dict((channel.get("modes") or {}).get(mode, {}).get("metrics") or {}).items()
    }


def _latency_p95(channel: dict[str, Any], mode: str) -> float:
    return float(
        (channel.get("modes") or {}).get(mode, {}).get("latency_ms", {}).get("p95")
        or 0.0
    )


def evaluate_development_gates(
    *,
    parent_dense: dict[str, Any],
    multivector: dict[str, Any],
    safe_rrf: dict[str, Any],
    product_baseline: dict[str, Any],
    representation_count_multiplier: float,
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode in ("scoped", "full_library"):
        parent_metrics = _metrics(parent_dense, mode)
        multi_metrics = _metrics(multivector, mode)
        safe_metrics = _metrics(safe_rrf, mode)
        product_metrics = _metrics(product_baseline, mode)
        parent_p95 = _latency_p95(parent_dense, mode)
        multi_p95 = _latency_p95(multivector, mode)
        latency_limit = max(parent_p95 * 2.0, parent_p95 + 15.0)
        modes[mode] = {
            "raw_recall_at_10_gain": round(
                multi_metrics.get("recall@10", 0.0) - parent_metrics.get("recall@10", 0.0),
                6,
            ),
            "raw_recall_at_10_no_regression": multi_metrics.get("recall@10", 0.0)
            >= parent_metrics.get("recall@10", 0.0),
            "safe_recall_at_10_gain_vs_product": round(
                safe_metrics.get("recall@10", 0.0) - product_metrics.get("recall@10", 0.0),
                6,
            ),
            "safe_preserves_product_recall_at_5": safe_metrics.get("recall@5")
            == product_metrics.get("recall@5"),
            "safe_preserves_product_wrong_source_at_5": safe_metrics.get(
                "wrong_source_rate@5"
            )
            == product_metrics.get("wrong_source_rate@5"),
            "safe_no_regression_vs_product": all(
                safe_metrics.get(metric, 0.0) >= product_metrics.get(metric, 0.0)
                for metric in ("recall@10", "mrr", "ndcg@10")
            ),
            "parent_dense_p95_ms": round(parent_p95, 3),
            "multivector_p95_ms": round(multi_p95, 3),
            "multivector_p95_limit_ms": round(latency_limit, 3),
            "latency_pass": multi_p95 <= latency_limit,
        }
    checks = {
        "raw_recall_at_10_no_regression_both_modes": all(
            mode["raw_recall_at_10_no_regression"] for mode in modes.values()
        ),
        "raw_recall_at_10_gain_any_mode": max(
            mode["raw_recall_at_10_gain"] for mode in modes.values()
        )
        >= 0.02,
        "safe_preserves_product_recall_at_5_both_modes": all(
            mode["safe_preserves_product_recall_at_5"] for mode in modes.values()
        ),
        "safe_preserves_product_wrong_source_at_5_both_modes": all(
            mode["safe_preserves_product_wrong_source_at_5"] for mode in modes.values()
        ),
        "safe_no_regression_vs_product_both_modes": all(
            mode["safe_no_regression_vs_product"] for mode in modes.values()
        ),
        "safe_recall_at_10_gain_any_mode": max(
            mode["safe_recall_at_10_gain_vs_product"] for mode in modes.values()
        )
        >= 0.02,
        "representation_count_multiplier_pass": representation_count_multiplier <= 2.0,
        "latency_pass_both_modes": all(mode["latency_pass"] for mode in modes.values()),
    }
    return {
        "modes": modes,
        "representation_count_multiplier": round(representation_count_multiplier, 6),
        "checks": checks,
        "passes_all": all(checks.values()),
    }


def _write_child_representation_db(
    path: Path,
    representations,
) -> None:
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in representations:
        document_id = str(row.parent_chunk_id).split(":", 1)[0]
        by_document[document_id].append(
            {
                "text": row.text,
                "meta": {
                    "parent_chunk_id": row.parent_chunk_id,
                    "representation_ordinal": row.ordinal,
                    "token_start": row.token_start,
                    "token_end": row.token_end,
                    "evidence_ready": True,
                },
            }
        )
    docs: dict[str, Any] = {}
    for document_id in sorted(by_document):
        rows = by_document[document_id]
        write_doc_chunks(path, document_id, rows)
        docs[document_id] = {
            "index_status": "ready",
            "num_chunks": len(rows),
            "representation_only": True,
        }
    save_docs_index(path, docs)


def _child_vectors_by_parent(loaded) -> dict[str, list[tuple[int, tuple[float, ...]]]]:
    grouped: dict[str, list[tuple[int, tuple[float, ...]]]] = defaultdict(list)
    for chunk, vector in zip(loaded.chunks, loaded.embeddings):
        meta = chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}
        parent_id = str(meta.get("parent_chunk_id") or "")
        if parent_id:
            grouped[parent_id].append((int(meta.get("representation_ordinal") or 0), vector))
    return grouped


def _report_validation_errors(
    report: dict[str, Any],
    *,
    registry: dict[str, Any],
    fixture_path: Path,
) -> list[str]:
    errors: list[str] = []
    if report.get("report_kind") != "parent_collapsed_multivector_development_benchmark":
        errors.append("report kind changed")
    if report.get("candidate_registry_sha256") != sha256_file(DEFAULT_REGISTRY):
        errors.append("candidate registry SHA-256 changed")
    fixture = report.get("fixture") if isinstance(report.get("fixture"), dict) else {}
    if fixture.get("sha256") != sha256_file(fixture_path):
        errors.append("development fixture SHA-256 changed")
    if report.get("fixture_id") != registry["development_fixture"]["id"]:
        errors.append("development fixture ID changed")
    candidate = report.get("candidate") if isinstance(report.get("candidate"), dict) else {}
    if candidate.get("product_enabled") is not False:
        errors.append("candidate must remain product-disabled")
    if candidate.get("allow_download") is not False or candidate.get("trust_remote_code") is not False:
        errors.append("candidate must remain cache-only without remote code")
    channels = report.get("channels") if isinstance(report.get("channels"), dict) else {}
    for name in (PARENT_DENSE_CHANNEL, MULTIVECTOR_CHANNEL, SAFE_RRF_CHANNEL):
        channel = channels.get(name) if isinstance(channels.get(name), dict) else {}
        for mode in ("scoped", "full_library"):
            metrics = _metrics(channel, mode)
            if not all(metric in metrics for metric in ("recall@5", "recall@10", "mrr", "ndcg@10", "wrong_source_rate@5")):
                errors.append(f"{name} {mode} metrics are incomplete")
            for metric, expected in _EXPECTED_MEASURED_METRICS[name][mode].items():
                if metrics.get(metric) != expected:
                    errors.append(f"{name} {mode} measured {metric} changed")
    representations = report.get("representations") if isinstance(report.get("representations"), dict) else {}
    if representations.get("parent_count") != 2251 or representations.get("child_count") != 1672:
        errors.append("measured parent or child representation count changed")
    if representations.get("count_multiplier") != 1.742781:
        errors.append("measured representation multiplier changed")
    if representations.get("parent_chunks_changed") is not False or representations.get("child_ids_returned") != 0:
        errors.append("parent identity preservation result changed")
    baseline = report.get("product_baseline") if isinstance(report.get("product_baseline"), dict) else {}
    measured = evaluate_development_gates(
        parent_dense=channels.get(PARENT_DENSE_CHANNEL) or {},
        multivector=channels.get(MULTIVECTOR_CHANNEL) or {},
        safe_rrf=channels.get(SAFE_RRF_CHANNEL) or {},
        product_baseline=baseline,
        representation_count_multiplier=float(
            (report.get("representations") or {}).get("count_multiplier") or 0.0
        ),
    )
    if report.get("development_gate") != measured:
        errors.append("development gate arithmetic changed")
    expected_decision = "eligible_for_frozen_replay" if measured["passes_all"] else "rejected"
    decision = report.get("decision") if isinstance(report.get("decision"), dict) else {}
    if decision.get("status") != expected_decision:
        errors.append("candidate decision does not match development gates")
    if decision.get("frozen_replay_allowed") is not bool(measured["passes_all"]):
        errors.append("frozen replay permission does not match development gates")
    return errors


def _validate_report(report_path: Path, fixture_path: Path) -> list[str]:
    try:
        registry = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"report or registry cannot be read: {type(exc).__name__}"]
    errors = validate_candidate(registry)
    errors.extend(_report_validation_errors(report, registry=registry, fixture_path=fixture_path))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure frozen parent-collapsed multi-vector retrieval.")
    parser.add_argument("--db-root", type=Path)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--parent-index-dir", type=Path, default=DEFAULT_PARENT_INDEX)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    fixture_path = args.fixture.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if args.validate_only:
        errors = _validate_report(report_path, fixture_path)
        if errors:
            print("[FAIL] " + "; ".join(errors))
            return 1
        print("[OK] parent-collapsed multi-vector development report")
        return 0
    if report_path.exists():
        parser.error("development report already exists; the frozen one-run measurement cannot be overwritten")
    if args.db_root is None:
        parser.error("--db-root is required unless --validate-only is used")

    registry = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
    registry_errors = validate_candidate(registry)
    if registry_errors:
        raise ValueError("candidate registry invalid: " + "; ".join(registry_errors))
    fixture = load_retrieval_benchmark_fixture(fixture_path, repo_root=REPO_ROOT)
    db_root = args.db_root.expanduser().resolve()
    chunks = load_all_chunks(db_root)
    _chunk_to_doc, corpus_errors = validate_indexed_corpus(fixture, chunks)
    if corpus_errors:
        raise ValueError("indexed corpus does not match development fixture: " + "; ".join(corpus_errors))

    candidate = registry["embedding_candidate"]
    provider = TransformersMeanPoolEmbeddingProvider(
        model_id=candidate["model_id"],
        revision=candidate["revision"],
        dimension=candidate["dimension"],
        query_prefix=candidate["query_prefix"],
        passage_prefix=candidate["passage_prefix"],
        max_length=candidate["encoder_max_length"],
        batch_size=max(1, int(args.batch_size)),
        device=args.device,
        dtype=candidate["dtype"],
        allow_download=False,
    )
    provider.prepare()
    parent_store = EmbeddingIndexStore(
        db_root,
        index_dir=args.parent_index_dir.expanduser().resolve(),
    )
    parent_loaded = parent_store.load(provider)
    parent_vectors_by_id = {
        str(chunk.get("id") or ""): vector
        for chunk, vector in zip(parent_loaded.chunks, parent_loaded.embeddings)
    }
    representation = registry["representation_contract"]
    children = focused_child_representations(
        chunks,
        token_offsets=provider.content_token_offsets,
        window_tokens=representation["child_window_tokens"],
        stride_tokens=representation["child_stride_tokens"],
        minimum_parent_tokens=representation["minimum_parent_tokens_for_children"],
    )
    out_root = args.out_root.expanduser().resolve()
    child_db = out_root / "development_child_representation_db_v1"
    _write_child_representation_db(child_db, children)
    child_store = EmbeddingIndexStore(child_db, index_dir=out_root / "development_child_index_v1")
    child_build = child_store.build(provider)
    child_loaded = child_store.load(provider)
    child_vectors = _child_vectors_by_parent(child_loaded)

    def parent_factory(selected_chunks: list[dict]) -> DenseRetriever:
        return DenseRetriever(
            selected_chunks,
            [parent_vectors_by_id[str(chunk.get("id") or "")] for chunk in selected_chunks],
            provider.embed_query,
            min_score=-1.0,
        )

    def multivector_factory(selected_chunks: list[dict]) -> ParentCollapsedDenseRetriever:
        selected_ids = {str(chunk.get("id") or "") for chunk in selected_chunks}
        return ParentCollapsedDenseRetriever(
            selected_chunks,
            [parent_vectors_by_id[str(chunk.get("id") or "")] for chunk in selected_chunks],
            {parent_id: rows for parent_id, rows in child_vectors.items() if parent_id in selected_ids},
            provider.embed_query,
        )

    def safe_factory(selected_chunks: list[dict], whole_library: bool) -> HybridCandidateRetriever:
        return HybridCandidateRetriever(
            ProductQuerySparseRetriever(selected_chunks, whole_library=whole_library),
            multivector_factory(selected_chunks),
            sparse_weight=2.0,
            dense_weight=1.0,
            rrf_k=20.0,
            candidate_multiplier=4,
            protected_sparse_prefix=5,
            dense_source_gate_depth=5,
        )

    parent_channel = evaluate_candidate_channel(
        fixture,
        chunks,
        channel_name=PARENT_DENSE_CHANNEL,
        retriever_factory=parent_factory,
    )
    multivector_channel = evaluate_candidate_channel(
        fixture,
        chunks,
        channel_name=MULTIVECTOR_CHANNEL,
        retriever_factory=multivector_factory,
    )
    safe_channel = evaluate_candidate_channel(
        fixture,
        chunks,
        channel_name=SAFE_RRF_CHANNEL,
        mode_retriever_factory=safe_factory,
    )
    baseline = json.loads(args.baseline.expanduser().resolve().read_text(encoding="utf-8"))
    product_baseline = (baseline.get("channels") or {}).get("bm25_product_queries") or {}
    representation_count = len(chunks) + len(children)
    multiplier = representation_count / max(1, len(chunks))
    development_gate = evaluate_development_gates(
        parent_dense=parent_channel,
        multivector=multivector_channel,
        safe_rrf=safe_channel,
        product_baseline=product_baseline,
        representation_count_multiplier=multiplier,
    )
    passes = bool(development_gate["passes_all"])
    report = {
        "schema_version": 1,
        "report_kind": "parent_collapsed_multivector_development_benchmark",
        "fixture_id": str(fixture.get("id") or ""),
        "candidate_registry_sha256": sha256_file(DEFAULT_REGISTRY),
        "fixture": {
            "path": fixture_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(fixture_path),
            "case_count": len(list(fixture.get("cases") or [])),
        },
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            **provider.runtime_info(),
            "trust_remote_code": False,
            "product_enabled": False,
        },
        "representation_contract": representation,
        "representations": {
            "parent_count": len(chunks),
            "child_count": len(children),
            "total_count": representation_count,
            "count_multiplier": round(multiplier, 6),
            "parent_chunks_changed": False,
            "child_ids_returned": 0,
        },
        "indexes": {
            "parent": asdict(parent_store.status(provider)),
            "child": {
                **asdict(child_build),
                "size_bytes": _index_size_bytes(child_store.index_dir),
                "status": asdict(child_store.status(provider)),
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "transformers": _package_version("transformers"),
        },
        "product_baseline": _compact_channel(product_baseline),
        "channels": {
            PARENT_DENSE_CHANNEL: _compact_channel(parent_channel),
            MULTIVECTOR_CHANNEL: _compact_channel(multivector_channel),
            SAFE_RRF_CHANNEL: _compact_channel(safe_channel),
        },
        "development_gate": development_gate,
        "decision": {
            "status": "eligible_for_frozen_replay" if passes else "rejected",
            "frozen_replay_allowed": passes,
            "product_enabled": False,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"representations": report["representations"], "development_gate": development_gate, "decision": report["decision"]}, ensure_ascii=False, indent=2))
    print(f"[OK] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
