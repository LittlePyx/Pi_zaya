from __future__ import annotations

import argparse
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
from kb.retriever import BM25Retriever
from kb.store import load_all_chunks
from tools.retrieval_eval.product_query_sparse_retriever import (
    ProductQuerySparseRetriever,
)
from tools.retrieval_eval.transformers_embedding_provider import (
    TransformersMeanPoolEmbeddingProvider,
)


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_eval_unseen_v2.json"
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_dense_multilingual_e5_base_unseen_v2.json"
DEFAULT_BASELINE = REPO_ROOT / "docs" / "retrieval_bm25_baseline_unseen_v2.json"
DEFAULT_OUT_ROOT = REPO_ROOT / "test_results" / "retrieval_eval_unseen_v2_dense"
DEFAULT_MODEL_ID = "intfloat/multilingual-e5-base"
DEFAULT_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
_CASE_METRICS = {
    "mrr",
    "recall@5",
    "recall@10",
    "recall@20",
    "ndcg@10",
    "contract_recall@10",
    "contract_recall@20",
    "wrong_source_rate@5",
}
_SOURCE_SAFE_GRID = (
    (2.0, 5),
    (2.0, 10),
    (2.0, 20),
    (4.0, 5),
    (4.0, 10),
    (4.0, 20),
    (8.0, 5),
    (8.0, 10),
    (8.0, 20),
)


def _source_safe_channel_name(sparse_weight: float, gate_depth: int) -> str:
    weight_label = str(float(sparse_weight)).replace(".", "p")
    return f"hybrid_product_bm25_dense_safe_sw{weight_label}_gate{int(gate_depth)}"


def _parse_source_safe_config(value: str) -> tuple[float, int]:
    try:
        raw_weight, raw_depth = str(value).split(":", 1)
        sparse_weight = float(raw_weight)
        gate_depth = int(raw_depth)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected SPARSE_WEIGHT:GATE_DEPTH") from exc
    if sparse_weight <= 0.0 or gate_depth <= 0:
        raise argparse.ArgumentTypeError("source-safe weight and gate depth must be positive")
    return sparse_weight, gate_depth


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return ""


def _compact_channel(channel: dict[str, Any]) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode_name, mode in dict(channel.get("modes") or {}).items():
        groups = {
            dimension: {
                value: {
                    "case_count": group.get("case_count"),
                    "metrics": group.get("metrics"),
                }
                for value, group in dict(dimension_groups or {}).items()
            }
            for dimension, dimension_groups in dict(mode.get("groups") or {}).items()
        }
        cases = []
        for case in list(mode.get("cases") or []):
            cases.append(
                {
                    "id": case.get("id"),
                    "locale": case.get("locale"),
                    "evaluation_focus": case.get("evaluation_focus"),
                    "coverage_tags": case.get("coverage_tags"),
                    "first_relevant_rank": case.get("first_relevant_rank"),
                    "label_count": case.get("label_count"),
                    "contract_count": case.get("contract_count"),
                    "metrics": {
                        key: value
                        for key, value in dict(case.get("metrics") or {}).items()
                        if key in _CASE_METRICS
                    },
                }
            )
        modes[mode_name] = {
            key: mode.get(key)
            for key in (
                "case_count",
                "label_count",
                "zero_hit_cases",
                "translated_cases",
                "metrics",
                "latency_ms",
                "ks",
            )
        }
        modes[mode_name]["groups"] = groups
        modes[mode_name]["cases"] = cases
    return {"channel": channel.get("channel"), "modes": modes}


def _index_size_bytes(index_dir: Path) -> int:
    return sum(path.stat().st_size for path in index_dir.rglob("*") if path.is_file())


def _validate_committed_report(
    fixture: dict[str, Any],
    fixture_path: Path,
    report_path: Path,
    baseline_path: Path,
) -> list[str]:
    if not report_path.is_file():
        return [f"Dense candidate report does not exist: {report_path}"]
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Dense candidate report cannot be read: {type(exc).__name__}"]
    errors: list[str] = []
    if report.get("report_kind") != "dense_candidate_benchmark":
        errors.append("report_kind does not identify a Dense candidate benchmark")
    if str(report.get("fixture_id") or "") != str(fixture.get("id") or ""):
        errors.append("fixture_id does not match")
    source = report.get("fixture") if isinstance(report.get("fixture"), dict) else {}
    if str(source.get("sha256") or "") != sha256_file(fixture_path):
        errors.append("fixture SHA-256 does not match")
    fixture_corpus = fixture.get("corpus") if isinstance(fixture.get("corpus"), dict) else {}
    report_corpus = report.get("corpus") if isinstance(report.get("corpus"), dict) else {}
    expected_case_count = len(list(fixture.get("cases") or []))
    expected_chunk_count = int(fixture_corpus.get("chunk_count") or 0)
    if int(report.get("case_count") or 0) != expected_case_count:
        errors.append("case count does not match")
    if str(report_corpus.get("fingerprint") or "") != str(fixture_corpus.get("fingerprint") or ""):
        errors.append("corpus fingerprint does not match")
    if int(report_corpus.get("chunk_count") or 0) != expected_chunk_count:
        errors.append("corpus chunk count does not match")

    expected_provider = TransformersMeanPoolEmbeddingProvider(
        model_id=DEFAULT_MODEL_ID,
        revision=DEFAULT_REVISION,
        dimension=768,
        query_prefix="query: ",
        passage_prefix="passage: ",
        max_length=512,
        batch_size=8,
        device="cuda",
        dtype="float16",
        allow_download=False,
    )
    candidate = report.get("candidate") if isinstance(report.get("candidate"), dict) else {}
    if str(candidate.get("model_id") or "") != DEFAULT_MODEL_ID:
        errors.append("candidate model ID does not match the preselected model")
    if str(candidate.get("revision") or "") != DEFAULT_REVISION:
        errors.append("candidate revision does not match the pinned revision")
    if str(candidate.get("provider_identity") or "") != expected_provider.descriptor.identity:
        errors.append("candidate provider identity does not match its encoding contract")
    if candidate.get("product_enabled") is not False:
        errors.append("candidate report must record product_enabled=false")
    if candidate.get("trust_remote_code") is not False:
        errors.append("candidate report must record trust_remote_code=false")
    if candidate.get("allow_download") is not False:
        errors.append("committed candidate report must come from a cache-only run")
    if int(candidate.get("document_count") or 0) != expected_chunk_count:
        errors.append("candidate encoded-document count does not match the corpus")

    index = report.get("index") if isinstance(report.get("index"), dict) else {}
    status = index.get("status") if isinstance(index.get("status"), dict) else {}
    if int(index.get("chunk_count") or 0) != expected_chunk_count:
        errors.append("embedding index chunk count does not match")
    if status.get("state") != "ready":
        errors.append("committed embedding index status was not ready")

    channels = report.get("channels") if isinstance(report.get("channels"), dict) else {}
    expected_case_ids = {str(case.get("id") or "") for case in list(fixture.get("cases") or [])}
    for channel_name in (
        "dense_multilingual_e5_base",
        "hybrid_bm25_dense_rrf_predeclared",
        "hybrid_product_bm25_dense_safe_sw2p0_gate5",
    ):
        channel = channels.get(channel_name)
        modes = channel.get("modes") if isinstance(channel, dict) else {}
        for mode_name in ("scoped", "full_library"):
            mode = modes.get(mode_name) if isinstance(modes, dict) else None
            if not isinstance(mode, dict):
                errors.append(f"{channel_name} report is missing {mode_name} results")
                continue
            cases = list(mode.get("cases") or [])
            case_ids = {
                str(case.get("id") or "")
                for case in cases
                if isinstance(case, dict)
            }
            if case_ids != expected_case_ids:
                errors.append(f"{channel_name} {mode_name} case IDs do not match the fixture")
            metrics = mode.get("metrics") if isinstance(mode.get("metrics"), dict) else {}
            for metric in ("mrr", "recall@5", "recall@10", "recall@20", "ndcg@10"):
                if not isinstance(metrics.get(metric), (int, float)):
                    errors.append(f"{channel_name} {mode_name} is missing numeric {metric}")

    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"BM25 baseline cannot be read: {type(exc).__name__}")
        return errors
    if str(baseline.get("fixture_id") or "") != str(fixture.get("id") or ""):
        errors.append("BM25 baseline fixture_id does not match")
        return errors
    baseline_channels = (
        baseline.get("channels") if isinstance(baseline.get("channels"), dict) else {}
    )
    product_channel = baseline_channels.get("bm25_product_queries")
    safe_channel = channels.get("hybrid_product_bm25_dense_safe_sw2p0_gate5")
    for mode_name in ("scoped", "full_library"):
        product_mode = (
            (product_channel.get("modes") or {}).get(mode_name)
            if isinstance(product_channel, dict)
            else None
        )
        safe_mode = (
            (safe_channel.get("modes") or {}).get(mode_name)
            if isinstance(safe_channel, dict)
            else None
        )
        if not isinstance(product_mode, dict) or not isinstance(safe_mode, dict):
            continue
        product_metrics = dict(product_mode.get("metrics") or {})
        safe_metrics = dict(safe_mode.get("metrics") or {})
        for metric in ("recall@5", "wrong_source_rate@5"):
            if safe_metrics.get(metric) != product_metrics.get(metric):
                errors.append(
                    f"source-safe {mode_name} {metric} does not preserve product BM25"
                )
        for metric in ("recall@10", "recall@20", "mrr", "ndcg@10"):
            if float(safe_metrics.get(metric) or 0.0) < float(
                product_metrics.get(metric) or 0.0
            ):
                errors.append(
                    f"source-safe {mode_name} {metric} regresses product BM25"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a pinned local embedding index and measure raw Dense retrieval "
            "against the frozen unseen-paper qrels. This does not enable Dense in the product."
        )
    )
    parser.add_argument("--db-root", type=Path)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--model-license", default="mit")
    parser.add_argument("--dimension", type=int, default=768)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Explicitly allow the benchmark process to download the pinned model.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Omit ranked chunk lists while retaining per-case metrics.",
    )
    parser.add_argument(
        "--source-safe-grid",
        action="store_true",
        help=(
            "Evaluate the fixed 3x3 source-safe development grid. Use only on an "
            "explicit development fixture, never to iterate on unseen-v2."
        ),
    )
    parser.add_argument(
        "--source-safe-config",
        action="append",
        type=_parse_source_safe_config,
        default=[],
        metavar="SPARSE_WEIGHT:GATE_DEPTH",
        help=(
            "Also evaluate one source-safe product-query fusion with a protected "
            "sparse top-5 and the specified sparse-source gate."
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    fixture_path = args.fixture.expanduser().resolve()
    fixture = load_retrieval_benchmark_fixture(fixture_path, repo_root=REPO_ROOT)
    if args.validate_only:
        errors = _validate_committed_report(
            fixture,
            fixture_path,
            args.report.expanduser().resolve(),
            args.baseline.expanduser().resolve(),
        )
        if errors:
            print("[FAIL] " + "; ".join(errors))
            return 1
        print(
            f"[OK] pinned Dense candidate report: {len(fixture['cases'])} cases, "
            f"{fixture['corpus']['chunk_count']} chunks, {DEFAULT_MODEL_ID}@{DEFAULT_REVISION[:12]}"
        )
        return 0
    if args.db_root is None:
        parser.error("--db-root is required unless --validate-only is used")
    db_root = args.db_root.expanduser().resolve()
    chunks = load_all_chunks(db_root)
    if not chunks:
        raise ValueError(f"no ready chunks found under {db_root}")
    _chunk_to_doc, corpus_errors = validate_indexed_corpus(fixture, chunks)
    if corpus_errors:
        raise ValueError(
            "indexed corpus does not match frozen qrels: " + "; ".join(corpus_errors)
        )

    provider = TransformersMeanPoolEmbeddingProvider(
        model_id=args.model_id,
        revision=args.revision,
        dimension=args.dimension,
        query_prefix="query: ",
        passage_prefix="passage: ",
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        allow_download=args.allow_download,
        cache_dir=args.cache_dir,
    )
    provider.prepare()
    if args.index_dir is not None:
        index_dir = args.index_dir.expanduser().resolve()
    else:
        index_dir = (
            args.out_root.expanduser().resolve()
            / "indexes"
            / provider.descriptor.identity
        )
    store = EmbeddingIndexStore(db_root, index_dir=index_dir)
    build_result = store.build(provider)
    loaded = store.load(provider)
    vectors_by_id = {
        str(chunk.get("id") or ""): vector
        for chunk, vector in zip(loaded.chunks, loaded.embeddings)
    }

    def dense_factory(selected_chunks: list[dict]) -> DenseRetriever:
        return DenseRetriever(
            selected_chunks,
            [vectors_by_id[str(chunk.get("id") or "")] for chunk in selected_chunks],
            provider.embed_query,
            min_score=-1.0,
        )

    channel_name = "dense_multilingual_e5_base"
    channel = evaluate_candidate_channel(
        fixture,
        chunks,
        channel_name=channel_name,
        retriever_factory=dense_factory,
    )

    def hybrid_factory(selected_chunks: list[dict]) -> HybridCandidateRetriever:
        dense = dense_factory(selected_chunks)
        return HybridCandidateRetriever(
            BM25Retriever(selected_chunks),
            dense,
            sparse_weight=2.0,
            dense_weight=1.0,
            rrf_k=20.0,
            candidate_multiplier=4,
        )

    hybrid_channel_name = "hybrid_bm25_dense_rrf_predeclared"
    hybrid_channel = evaluate_candidate_channel(
        fixture,
        chunks,
        channel_name=hybrid_channel_name,
        retriever_factory=hybrid_factory,
    )
    measured_channels = {
        channel_name: channel,
        hybrid_channel_name: hybrid_channel,
    }
    source_safe_configs = list(args.source_safe_config)
    if args.source_safe_grid:
        source_safe_configs.extend(_SOURCE_SAFE_GRID)
    for sparse_weight, gate_depth in dict.fromkeys(source_safe_configs):
        safe_channel_name = _source_safe_channel_name(sparse_weight, gate_depth)

        def source_safe_factory(
            selected_chunks: list[dict],
            whole_library: bool,
            *,
            weight: float = sparse_weight,
            source_depth: int = gate_depth,
        ) -> HybridCandidateRetriever:
            return HybridCandidateRetriever(
                ProductQuerySparseRetriever(
                    selected_chunks,
                    whole_library=whole_library,
                ),
                dense_factory(selected_chunks),
                sparse_weight=weight,
                dense_weight=1.0,
                rrf_k=20.0,
                candidate_multiplier=4,
                protected_sparse_prefix=5,
                dense_source_gate_depth=source_depth,
            )

        measured_channels[safe_channel_name] = evaluate_candidate_channel(
            fixture,
            chunks,
            channel_name=safe_channel_name,
            mode_retriever_factory=source_safe_factory,
        )

    corpus = fixture.get("corpus") if isinstance(fixture.get("corpus"), dict) else {}
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_kind": "dense_candidate_benchmark",
        "fixture_id": str(fixture.get("id") or ""),
        "case_count": len(list(fixture.get("cases") or [])),
        "corpus": {
            "document_count": int(corpus.get("document_count") or 0),
            "chunk_count": int(corpus.get("chunk_count") or 0),
            "fingerprint": str(corpus.get("fingerprint") or ""),
        },
        "fixture": {
            "path": fixture_path.relative_to(REPO_ROOT).as_posix()
            if fixture_path.is_relative_to(REPO_ROOT)
            else fixture_path.name,
            "sha256": sha256_file(fixture_path),
            "sha256_newlines": "normalized_lf",
        },
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            **provider.runtime_info(),
            "declared_model_license": str(args.model_license or ""),
            "trust_remote_code": False,
            "product_enabled": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "transformers": _package_version("transformers"),
        },
        "index": {
            **asdict(build_result),
            "size_bytes": _index_size_bytes(index_dir),
            "status": asdict(store.status(provider)),
        },
        "channels": {
            name: _compact_channel(candidate_channel) if args.compact else candidate_channel
            for name, candidate_channel in measured_channels.items()
        },
    }
    summary = {
        "fixture_id": report["fixture_id"],
        "case_count": report["case_count"],
        "candidate": report["candidate"],
        "index": report["index"],
        "channels": {
            name: {
                mode_name: {
                    "metrics": mode.get("metrics"),
                    "latency_ms": mode.get("latency_ms"),
                }
                for mode_name, mode in candidate_channel["modes"].items()
            }
            for name, candidate_channel in measured_channels.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output is not None:
        output = args.output.expanduser().resolve()
    else:
        output = (
            args.out_root.expanduser().resolve()
            / datetime.now().strftime("%Y%m%d_%H%M%S")
            / "report.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
