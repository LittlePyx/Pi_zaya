from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
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
from kb.retrieval_benchmark import evaluate_candidate_channel, validate_indexed_corpus
from kb.retrieval_eval_fixture import sha256_file
from kb.retrieval_semantic_eval_fixture import load_retrieval_benchmark_fixture
from kb.store import load_all_chunks
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
from tools.retrieval_eval.validate_e5_large_instruct_candidate import (
    DEFAULT_REGISTRY,
    BASE_MODEL_ID,
    BASE_REVISION,
    MODEL_ID,
    PROVIDER_IDENTITY,
    QUERY_PREFIX,
    REVISION,
    validate_candidate,
)


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_eval_development_v1.json"
DEFAULT_BASELINE = REPO_ROOT / "docs" / "retrieval_bm25_baseline_development_v1.json"
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_multilingual_e5_large_instruct_development_v1.json"
DEFAULT_BASE_INDEX = (
    REPO_ROOT
    / "test_results"
    / "retrieval_eval_development_v1"
    / "multilingual_e5_base_rrf_index"
)
DEFAULT_LARGE_INDEX = (
    REPO_ROOT
    / "test_results"
    / "retrieval_e5_large_instruct"
    / "development_index_v1"
)
BASE_CHANNEL = "base_e5_single_vector"
LARGE_CHANNEL = "large_instruct_e5_single_vector"
SAFE_CHANNEL = "large_instruct_e5_safe_rrf"
_EXPECTED_MEASURED_METRICS = {
    BASE_CHANNEL: {
        "scoped": {
            "recall@5": 0.439024,
            "recall@10": 0.609756,
            "mrr": 0.299119,
            "ndcg@10": 0.364578,
            "wrong_source_rate@5": 0.0,
        },
        "full_library": {
            "recall@5": 0.256098,
            "recall@10": 0.414634,
            "mrr": 0.188353,
            "ndcg@10": 0.233349,
            "wrong_source_rate@5": 0.531707,
        },
    },
    LARGE_CHANNEL: {
        "scoped": {
            "recall@5": 0.5,
            "recall@10": 0.560976,
            "mrr": 0.302337,
            "ndcg@10": 0.357645,
            "wrong_source_rate@5": 0.0,
        },
        "full_library": {
            "recall@5": 0.280488,
            "recall@10": 0.329268,
            "mrr": 0.132955,
            "ndcg@10": 0.172974,
            "wrong_source_rate@5": 0.580488,
        },
    },
    SAFE_CHANNEL: {
        "scoped": {
            "recall@5": 0.597561,
            "recall@10": 0.670732,
            "mrr": 0.396369,
            "ndcg@10": 0.450935,
            "wrong_source_rate@5": 0.0,
        },
        "full_library": {
            "recall@5": 0.5,
            "recall@10": 0.609756,
            "mrr": 0.28467,
            "ndcg@10": 0.35004,
            "wrong_source_rate@5": 0.297561,
        },
    },
}


def _sha256_binary(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _p95(channel: dict[str, Any], mode: str) -> float:
    return float((channel.get("modes") or {}).get(mode, {}).get("latency_ms", {}).get("p95") or 0.0)


def evaluate_development_gates(
    *,
    base_dense: dict[str, Any],
    large_dense: dict[str, Any],
    safe_rrf: dict[str, Any],
    product_baseline: dict[str, Any],
    index_size_ratio: float,
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode in ("scoped", "full_library"):
        base_metrics = _metrics(base_dense, mode)
        large_metrics = _metrics(large_dense, mode)
        safe_metrics = _metrics(safe_rrf, mode)
        product_metrics = _metrics(product_baseline, mode)
        base_p95 = _p95(base_dense, mode)
        large_p95 = _p95(large_dense, mode)
        latency_limit = max(base_p95 * 2.5, base_p95 + 50.0)
        modes[mode] = {
            "raw_recall_at_10_gain_vs_base": round(
                large_metrics.get("recall@10", 0.0) - base_metrics.get("recall@10", 0.0),
                6,
            ),
            "raw_recall_at_10_no_regression": large_metrics.get("recall@10", 0.0)
            >= base_metrics.get("recall@10", 0.0),
            "safe_recall_at_10_gain_vs_product": round(
                safe_metrics.get("recall@10", 0.0) - product_metrics.get("recall@10", 0.0),
                6,
            ),
            "safe_recall_at_10_gain_pass": (
                safe_metrics.get("recall@10", 0.0) - product_metrics.get("recall@10", 0.0)
            )
            >= 0.02,
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
            "base_p95_ms": round(base_p95, 3),
            "large_p95_ms": round(large_p95, 3),
            "large_p95_limit_ms": round(latency_limit, 3),
            "latency_pass": large_p95 <= latency_limit,
        }
    checks = {
        "raw_recall_at_10_no_regression_both_modes": all(
            row["raw_recall_at_10_no_regression"] for row in modes.values()
        ),
        "raw_recall_at_10_gain_any_mode": max(
            row["raw_recall_at_10_gain_vs_base"] for row in modes.values()
        )
        >= 0.05,
        "safe_recall_at_10_gain_both_modes": all(
            row["safe_recall_at_10_gain_pass"] for row in modes.values()
        ),
        "safe_preserves_product_recall_at_5_both_modes": all(
            row["safe_preserves_product_recall_at_5"] for row in modes.values()
        ),
        "safe_preserves_product_wrong_source_at_5_both_modes": all(
            row["safe_preserves_product_wrong_source_at_5"] for row in modes.values()
        ),
        "safe_no_regression_vs_product_both_modes": all(
            row["safe_no_regression_vs_product"] for row in modes.values()
        ),
        "index_size_ratio_pass": index_size_ratio <= 1.5,
        "latency_pass_both_modes": all(row["latency_pass"] for row in modes.values()),
    }
    return {
        "modes": modes,
        "index_size_ratio_vs_base": round(index_size_ratio, 6),
        "checks": checks,
        "passes_all": all(checks.values()),
    }


def _prepare_model(registry: dict[str, Any], cache_dir: Path | None) -> int:
    from huggingface_hub import snapshot_download

    for label, model in (
        ("comparison baseline", registry["comparison_baseline"]),
        ("E5-large-instruct candidate", registry["selected_candidate"]),
    ):
        snapshot = Path(
            snapshot_download(
                model["model_id"],
                revision=model["revision"],
                cache_dir=str(cache_dir.expanduser().resolve()) if cache_dir else None,
                allow_patterns=[
                    "config.json",
                    "sentencepiece.bpe.model",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "config_sentence_transformers.json",
                    "modules.json",
                    "1_Pooling/config.json",
                    "model.safetensors",
                ],
            )
        )
        weight = snapshot / "model.safetensors"
        if weight.stat().st_size != int(model["model_safetensors_bytes"]):
            raise RuntimeError(f"{label} safe weight byte size mismatch")
        if _sha256_binary(weight) != str(model["model_safetensors_sha256"]):
            raise RuntimeError(f"{label} safe weight SHA-256 mismatch")
        print(f"[OK] cached and verified pinned {label}: {snapshot}")
    return 0


def _release_provider(provider: Any) -> None:
    try:
        provider._model = None
        provider._tokenizer = None
        provider._torch = None
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _report_errors(report_path: Path, fixture_path: Path) -> list[str]:
    try:
        registry = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"report or registry cannot be read: {type(exc).__name__}"]
    errors = validate_candidate(registry)
    if report.get("report_kind") != "e5_large_instruct_development_benchmark":
        errors.append("report kind changed")
    if report.get("candidate_registry_sha256") != sha256_file(DEFAULT_REGISTRY):
        errors.append("candidate registry SHA-256 changed")
    fixture = report.get("fixture") if isinstance(report.get("fixture"), dict) else {}
    if fixture.get("sha256") != sha256_file(fixture_path):
        errors.append("development fixture SHA-256 changed")
    candidate = report.get("candidate") if isinstance(report.get("candidate"), dict) else {}
    if candidate.get("model_id") != MODEL_ID or candidate.get("revision") != REVISION:
        errors.append("measured candidate identity changed")
    if candidate.get("provider_identity") != PROVIDER_IDENTITY:
        errors.append("measured provider identity changed")
    if (
        candidate.get("allow_download") is not False
        or candidate.get("trust_remote_code") is not False
        or candidate.get("product_enabled") is not False
    ):
        errors.append("measured candidate must be cache-only and product-disabled")
    base_provider = (
        report.get("base_provider") if isinstance(report.get("base_provider"), dict) else {}
    )
    if (
        base_provider.get("model_id") != BASE_MODEL_ID
        or base_provider.get("revision") != BASE_REVISION
        or base_provider.get("allow_download") is not False
    ):
        errors.append("measured comparison baseline identity changed")
    channels = report.get("channels") if isinstance(report.get("channels"), dict) else {}
    for name in (BASE_CHANNEL, LARGE_CHANNEL, SAFE_CHANNEL):
        for mode in ("scoped", "full_library"):
            metrics = _metrics(channels.get(name) or {}, mode)
            if not all(key in metrics for key in ("recall@5", "recall@10", "mrr", "ndcg@10", "wrong_source_rate@5")):
                errors.append(f"{name} {mode} metrics incomplete")
            for metric, expected in _EXPECTED_MEASURED_METRICS[name][mode].items():
                if metrics.get(metric) != expected:
                    errors.append(f"{name} {mode} measured {metric} changed")
    try:
        baseline = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))
        expected_product = _compact_channel(
            (baseline.get("channels") or {}).get("bm25_product_queries") or {}
        )
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"product baseline cannot be read: {type(exc).__name__}")
        expected_product = None
    if expected_product is not None and report.get("product_baseline") != expected_product:
        errors.append("embedded product baseline changed")
    index = report.get("index") if isinstance(report.get("index"), dict) else {}
    if (
        index.get("chunk_count") != 2251
        or index.get("size_bytes") != 9801713
        or index.get("base_size_bytes") != 7496677
        or index.get("size_ratio_vs_base") != 1.307474
    ):
        errors.append("measured index contract changed")
    measured_gate = evaluate_development_gates(
        base_dense=channels.get(BASE_CHANNEL) or {},
        large_dense=channels.get(LARGE_CHANNEL) or {},
        safe_rrf=channels.get(SAFE_CHANNEL) or {},
        product_baseline=report.get("product_baseline") or {},
        index_size_ratio=float(index.get("size_ratio_vs_base") or 0.0),
    )
    if report.get("development_gate") != measured_gate:
        errors.append("development gate arithmetic changed")
    expected_status = "eligible_for_frozen_replay" if measured_gate["passes_all"] else "rejected"
    decision = report.get("decision") if isinstance(report.get("decision"), dict) else {}
    if decision.get("status") != expected_status:
        errors.append("decision does not match development gates")
    if decision.get("frozen_replay_allowed") is not bool(measured_gate["passes_all"]):
        errors.append("frozen replay permission does not match gates")
    if decision.get("product_enabled") is not False:
        errors.append("development report cannot activate product retrieval")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare, measure, or validate E5-large-instruct retrieval.")
    parser.add_argument("--db-root", type=Path)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--base-index-dir", type=Path, default=DEFAULT_BASE_INDEX)
    parser.add_argument("--large-index-dir", type=Path, default=DEFAULT_LARGE_INDEX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prepare-model", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    registry = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
    registry_errors = validate_candidate(registry)
    if registry_errors:
        raise ValueError("candidate registry invalid: " + "; ".join(registry_errors))
    if args.prepare_model:
        if not args.allow_download:
            parser.error("--prepare-model requires explicit --allow-download")
        return _prepare_model(registry, args.cache_dir)
    if args.allow_download:
        parser.error("downloads are allowed only during --prepare-model")
    fixture_path = args.fixture.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if args.validate_only:
        errors = _report_errors(report_path, fixture_path)
        if errors:
            print("[FAIL] " + "; ".join(errors))
            return 1
        print("[OK] E5-large-instruct development report")
        return 0
    if report_path.exists():
        parser.error("development report already exists; the frozen one-run measurement cannot be overwritten")
    if args.db_root is None:
        parser.error("--db-root is required for measurement")
    db_root = args.db_root.expanduser().resolve()
    fixture = load_retrieval_benchmark_fixture(fixture_path, repo_root=REPO_ROOT)
    chunks = load_all_chunks(db_root)
    _mapping, corpus_errors = validate_indexed_corpus(fixture, chunks)
    if corpus_errors:
        raise ValueError("indexed corpus does not match development fixture: " + "; ".join(corpus_errors))

    base_provider = TransformersMeanPoolEmbeddingProvider(
        model_id=BASE_MODEL_ID,
        revision=BASE_REVISION,
        dimension=768,
        query_prefix="query: ",
        passage_prefix="passage: ",
        max_length=512,
        batch_size=max(1, int(args.batch_size)),
        device=args.device,
        dtype="float16",
        allow_download=False,
    )
    base_provider.prepare()
    base_store = EmbeddingIndexStore(db_root, index_dir=args.base_index_dir.expanduser().resolve())
    base_loaded = base_store.load(base_provider)
    base_vectors = {
        str(chunk.get("id") or ""): vector
        for chunk, vector in zip(base_loaded.chunks, base_loaded.embeddings)
    }

    def base_factory(selected_chunks: list[dict]) -> DenseRetriever:
        return DenseRetriever(
            selected_chunks,
            [base_vectors[str(chunk.get("id") or "")] for chunk in selected_chunks],
            base_provider.embed_query,
            min_score=-1.0,
        )

    base_channel = evaluate_candidate_channel(
        fixture,
        chunks,
        channel_name=BASE_CHANNEL,
        retriever_factory=base_factory,
    )
    base_runtime = base_provider.runtime_info()
    base_index_size = _index_size_bytes(base_store.index_dir)
    _release_provider(base_provider)

    selected = registry["selected_candidate"]
    large_provider = TransformersMeanPoolEmbeddingProvider(
        model_id=MODEL_ID,
        revision=REVISION,
        dimension=1024,
        query_prefix=QUERY_PREFIX,
        passage_prefix="",
        max_length=512,
        batch_size=max(1, int(args.batch_size)),
        device=args.device,
        dtype="float16",
        allow_download=False,
        cache_dir=args.cache_dir,
    )
    large_provider.prepare()
    large_store = EmbeddingIndexStore(db_root, index_dir=args.large_index_dir.expanduser().resolve())
    large_build = large_store.build(large_provider)
    large_loaded = large_store.load(large_provider)
    large_vectors = {
        str(chunk.get("id") or ""): vector
        for chunk, vector in zip(large_loaded.chunks, large_loaded.embeddings)
    }

    def large_factory(selected_chunks: list[dict]) -> DenseRetriever:
        return DenseRetriever(
            selected_chunks,
            [large_vectors[str(chunk.get("id") or "")] for chunk in selected_chunks],
            large_provider.embed_query,
            min_score=-1.0,
        )

    def safe_factory(selected_chunks: list[dict], whole_library: bool) -> HybridCandidateRetriever:
        return HybridCandidateRetriever(
            ProductQuerySparseRetriever(selected_chunks, whole_library=whole_library),
            large_factory(selected_chunks),
            sparse_weight=2.0,
            dense_weight=1.0,
            rrf_k=20.0,
            candidate_multiplier=4,
            protected_sparse_prefix=5,
            dense_source_gate_depth=5,
        )

    large_channel = evaluate_candidate_channel(
        fixture,
        chunks,
        channel_name=LARGE_CHANNEL,
        retriever_factory=large_factory,
    )
    safe_channel = evaluate_candidate_channel(
        fixture,
        chunks,
        channel_name=SAFE_CHANNEL,
        mode_retriever_factory=safe_factory,
    )
    baseline = json.loads(args.baseline.expanduser().resolve().read_text(encoding="utf-8"))
    product_baseline = (baseline.get("channels") or {}).get("bm25_product_queries") or {}
    large_index_size = _index_size_bytes(large_store.index_dir)
    size_ratio = large_index_size / max(1, base_index_size)
    gate = evaluate_development_gates(
        base_dense=base_channel,
        large_dense=large_channel,
        safe_rrf=safe_channel,
        product_baseline=product_baseline,
        index_size_ratio=size_ratio,
    )
    passes = bool(gate["passes_all"])
    report = {
        "schema_version": 1,
        "report_kind": "e5_large_instruct_development_benchmark",
        "fixture_id": str(fixture.get("id") or ""),
        "candidate_registry_sha256": sha256_file(DEFAULT_REGISTRY),
        "fixture": {
            "path": fixture_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(fixture_path),
            "case_count": len(list(fixture.get("cases") or [])),
        },
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            **large_provider.runtime_info(),
            "model_safetensors_bytes": selected["model_safetensors_bytes"],
            "model_safetensors_sha256": selected["model_safetensors_sha256"],
            "trust_remote_code": False,
            "product_enabled": False,
        },
        "base_provider": base_runtime,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "transformers": _package_version("transformers"),
        },
        "index": {
            **asdict(large_build),
            "size_bytes": large_index_size,
            "base_size_bytes": base_index_size,
            "size_ratio_vs_base": round(size_ratio, 6),
            "status": asdict(large_store.status(large_provider)),
        },
        "product_baseline": _compact_channel(product_baseline),
        "channels": {
            BASE_CHANNEL: _compact_channel(base_channel),
            LARGE_CHANNEL: _compact_channel(large_channel),
            SAFE_CHANNEL: _compact_channel(safe_channel),
        },
        "development_gate": gate,
        "decision": {
            "status": "eligible_for_frozen_replay" if passes else "rejected",
            "frozen_replay_allowed": passes,
            "product_enabled": False,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"index": report["index"], "development_gate": gate, "decision": report["decision"]}, ensure_ascii=False, indent=2))
    print(f"[OK] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
