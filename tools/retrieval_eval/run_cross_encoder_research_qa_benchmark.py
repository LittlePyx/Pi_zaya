from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.cross_encoder_reranker import CrossEncoderReranker, CrossEncoderRerankingRetriever
from kb.dense_retriever import DenseRetriever
from kb.embedding_index import EmbeddingIndexStore
from kb.hybrid_retriever import HybridCandidateRetriever
from kb.retrieval_benchmark import evaluate_candidate_channel, validate_indexed_corpus
from kb.retrieval_eval_fixture import load_qrels_fixture, sha256_file
from kb.store import load_all_chunks
from tools.retrieval_eval.run_dense_research_qa_benchmark import (
    DEFAULT_MODEL_ID as EMBEDDING_MODEL_ID,
)
from tools.retrieval_eval.run_dense_research_qa_benchmark import (
    DEFAULT_REVISION as EMBEDDING_REVISION,
)
from tools.retrieval_eval.run_dense_research_qa_benchmark import (
    _compact_channel,
)
from tools.retrieval_eval.product_query_sparse_retriever import (
    ProductQuerySparseRetriever,
)
from tools.retrieval_eval.transformers_cross_encoder_provider import (
    TransformersSequenceClassificationReranker,
)
from tools.retrieval_eval.transformers_embedding_provider import (
    TransformersMeanPoolEmbeddingProvider,
)


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_eval_development_v1.json"
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_cross_encoder_mmarco_development_v1.json"
DEFAULT_MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_REVISION = "8008f0154ca2387468013ed439df09d70c122412"
SAFE_CHANNEL = "hybrid_product_bm25_dense_safe_sw2p0_gate5"
RERANKED_CHANNEL = "hybrid_product_bm25_dense_safe_mmarco_tail_6_20"


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return ""


def _safe_factory(
    selected_chunks: list[dict],
    whole_library: bool,
    *,
    vectors_by_id: dict[str, tuple[float, ...]],
    embed_query,
) -> HybridCandidateRetriever:
    dense = DenseRetriever(
        selected_chunks,
        [vectors_by_id[str(chunk.get("id") or "")] for chunk in selected_chunks],
        embed_query,
        min_score=-1.0,
    )
    return HybridCandidateRetriever(
        ProductQuerySparseRetriever(
            selected_chunks,
            whole_library=whole_library,
        ),
        dense,
        sparse_weight=2.0,
        dense_weight=1.0,
        rrf_k=20.0,
        candidate_multiplier=4,
        protected_sparse_prefix=5,
        dense_source_gate_depth=5,
    )


def _validate_committed_report(
    fixture: dict[str, Any],
    fixture_path: Path,
    report_path: Path,
    safe_report_path: Path | None,
) -> list[str]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cross-encoder report cannot be read: {type(exc).__name__}"]
    errors: list[str] = []
    if report.get("report_kind") != "cross_encoder_candidate_benchmark":
        errors.append("report_kind does not identify a cross-encoder candidate benchmark")
    if str(report.get("fixture_id") or "") != str(fixture.get("id") or ""):
        errors.append("fixture_id does not match")
    source = report.get("fixture") if isinstance(report.get("fixture"), dict) else {}
    if str(source.get("sha256") or "") != sha256_file(fixture_path):
        errors.append("fixture SHA-256 does not match")
    corpus = fixture.get("corpus") if isinstance(fixture.get("corpus"), dict) else {}
    report_corpus = report.get("corpus") if isinstance(report.get("corpus"), dict) else {}
    if str(report_corpus.get("fingerprint") or "") != str(corpus.get("fingerprint") or ""):
        errors.append("corpus fingerprint does not match")
    if int(report_corpus.get("chunk_count") or 0) != int(corpus.get("chunk_count") or 0):
        errors.append("corpus chunk count does not match")
    candidate = report.get("candidate") if isinstance(report.get("candidate"), dict) else {}
    expected = TransformersSequenceClassificationReranker(
        model_id=DEFAULT_MODEL_ID,
        revision=DEFAULT_REVISION,
        max_length=512,
        batch_size=8,
        device="cuda",
        dtype="float16",
        allow_download=False,
    )
    if str(candidate.get("model_id") or "") != DEFAULT_MODEL_ID:
        errors.append("cross-encoder model ID does not match")
    if str(candidate.get("revision") or "") != DEFAULT_REVISION:
        errors.append("cross-encoder revision does not match")
    if str(candidate.get("contract_identity") or "") != expected.contract_identity:
        errors.append("cross-encoder scoring contract does not match")
    if candidate.get("allow_download") is not False:
        errors.append("committed cross-encoder report must come from a cache-only run")
    if candidate.get("trust_remote_code") is not False:
        errors.append("committed cross-encoder report must record trust_remote_code=false")
    if candidate.get("product_enabled") is not False:
        errors.append("cross-encoder candidate must not be product-enabled")
    ranking = report.get("ranking_contract") if isinstance(report.get("ranking_contract"), dict) else {}
    if ranking != {
        "upstream_channel": SAFE_CHANNEL,
        "candidate_limit": 20,
        "protected_prefix": 5,
        "reranked_positions": [6, 20],
    }:
        errors.append("cross-encoder ranking contract is not the frozen tail-only policy")

    channels = report.get("channels") if isinstance(report.get("channels"), dict) else {}
    expected_case_ids = {str(case.get("id") or "") for case in list(fixture.get("cases") or [])}
    for channel_name in (SAFE_CHANNEL, RERANKED_CHANNEL):
        channel = channels.get(channel_name)
        modes = channel.get("modes") if isinstance(channel, dict) else {}
        for mode_name in ("scoped", "full_library"):
            mode = modes.get(mode_name) if isinstance(modes, dict) else None
            if not isinstance(mode, dict):
                errors.append(f"{channel_name} report is missing {mode_name}")
                continue
            case_ids = {
                str(case.get("id") or "")
                for case in list(mode.get("cases") or [])
                if isinstance(case, dict)
            }
            if case_ids != expected_case_ids:
                errors.append(f"{channel_name} {mode_name} case IDs do not match")
            metrics = mode.get("metrics") if isinstance(mode.get("metrics"), dict) else {}
            for metric in ("mrr", "recall@5", "recall@10", "recall@20", "ndcg@10"):
                if not isinstance(metrics.get(metric), (int, float)):
                    errors.append(f"{channel_name} {mode_name} is missing numeric {metric}")

    measured_safe = channels.get(SAFE_CHANNEL, {})
    reranked = channels.get(RERANKED_CHANNEL, {})
    frozen_safe = measured_safe
    if safe_report_path is not None:
        try:
            safe_report = json.loads(safe_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"safe-RRF report cannot be read: {type(exc).__name__}")
            return errors
        frozen_safe = (safe_report.get("channels") or {}).get(SAFE_CHANNEL, {})
    for mode_name in ("scoped", "full_library"):
        frozen_metrics = dict(((frozen_safe.get("modes") or {}).get(mode_name) or {}).get("metrics") or {})
        measured_metrics = dict(((measured_safe.get("modes") or {}).get(mode_name) or {}).get("metrics") or {})
        reranked_metrics = dict(((reranked.get("modes") or {}).get(mode_name) or {}).get("metrics") or {})
        for metric in ("recall@5", "recall@10", "recall@20", "mrr", "ndcg@10", "wrong_source_rate@5"):
            if measured_metrics.get(metric) != frozen_metrics.get(metric):
                errors.append(f"safe-RRF {mode_name} {metric} does not match its frozen report")
        for metric in ("recall@5", "wrong_source_rate@5"):
            if reranked_metrics.get(metric) != measured_metrics.get(metric):
                errors.append(f"tail reranker changed protected {mode_name} {metric}")
        if reranked_metrics.get("recall@20") != measured_metrics.get("recall@20"):
            errors.append(f"tail reranker changed {mode_name} Recall@20 candidate coverage")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure a pinned local cross-encoder over positions 6-20 of the frozen "
            "source-safe hybrid candidate pool. This does not enable product reranking."
        )
    )
    parser.add_argument("--db-root", type=Path)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--embedding-index-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--safe-report", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--model-license", default="apache-2.0")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    fixture_path = args.fixture.expanduser().resolve()
    fixture = load_qrels_fixture(fixture_path)
    if args.validate_only:
        errors = _validate_committed_report(
            fixture,
            fixture_path,
            args.report.expanduser().resolve(),
            args.safe_report.expanduser().resolve() if args.safe_report else None,
        )
        if errors:
            print("[FAIL] " + "; ".join(errors))
            return 1
        print(
            f"[OK] pinned cross-encoder report: {len(fixture['cases'])} cases, "
            f"{fixture['corpus']['chunk_count']} chunks, {DEFAULT_MODEL_ID}@{DEFAULT_REVISION[:12]}"
        )
        return 0
    if args.db_root is None or args.embedding_index_dir is None:
        parser.error("--db-root and --embedding-index-dir are required")
    db_root = args.db_root.expanduser().resolve()
    chunks = load_all_chunks(db_root)
    _chunk_to_doc, corpus_errors = validate_indexed_corpus(fixture, chunks)
    if corpus_errors:
        raise ValueError("indexed corpus does not match qrels: " + "; ".join(corpus_errors))

    embedding_provider = TransformersMeanPoolEmbeddingProvider(
        model_id=EMBEDDING_MODEL_ID,
        revision=EMBEDDING_REVISION,
        dimension=768,
        query_prefix="query: ",
        passage_prefix="passage: ",
        max_length=512,
        batch_size=8,
        device=args.device,
        dtype=args.dtype,
        allow_download=False,
    )
    embedding_provider.prepare()
    index_store = EmbeddingIndexStore(
        db_root,
        index_dir=args.embedding_index_dir.expanduser().resolve(),
    )
    loaded = index_store.load(embedding_provider)
    vectors_by_id = {
        str(chunk.get("id") or ""): vector
        for chunk, vector in zip(loaded.chunks, loaded.embeddings)
    }
    cross_encoder = TransformersSequenceClassificationReranker(
        model_id=args.model_id,
        revision=args.revision,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        allow_download=args.allow_download,
        cache_dir=args.cache_dir,
    )
    cross_encoder.prepare()

    def safe_factory(selected_chunks: list[dict], whole_library: bool):
        return _safe_factory(
            selected_chunks,
            whole_library,
            vectors_by_id=vectors_by_id,
            embed_query=embedding_provider.embed_query,
        )

    def reranked_factory(selected_chunks: list[dict], whole_library: bool):
        return CrossEncoderRerankingRetriever(
            safe_factory(selected_chunks, whole_library),
            CrossEncoderReranker(
                cross_encoder.score_pairs,
                candidate_limit=20,
                protected_prefix=5,
            ),
        )

    safe_channel = evaluate_candidate_channel(
        fixture,
        chunks,
        channel_name=SAFE_CHANNEL,
        mode_retriever_factory=safe_factory,
    )
    reranked_channel = evaluate_candidate_channel(
        fixture,
        chunks,
        channel_name=RERANKED_CHANNEL,
        mode_retriever_factory=reranked_factory,
    )
    corpus = fixture.get("corpus") if isinstance(fixture.get("corpus"), dict) else {}
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_kind": "cross_encoder_candidate_benchmark",
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
            **cross_encoder.runtime_info(),
            "declared_model_license": str(args.model_license or ""),
            "product_enabled": False,
        },
        "embedding_provider": embedding_provider.runtime_info(),
        "ranking_contract": {
            "upstream_channel": SAFE_CHANNEL,
            "candidate_limit": 20,
            "protected_prefix": 5,
            "reranked_positions": [6, 20],
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "transformers": _package_version("transformers"),
            "sentencepiece": _package_version("sentencepiece"),
            "safetensors": _package_version("safetensors"),
        },
        "channels": {
            SAFE_CHANNEL: _compact_channel(safe_channel) if args.compact else safe_channel,
            RERANKED_CHANNEL: (
                _compact_channel(reranked_channel) if args.compact else reranked_channel
            ),
        },
    }
    summary = {
        "fixture_id": report["fixture_id"],
        "candidate": report["candidate"],
        "ranking_contract": report["ranking_contract"],
        "channels": {
            name: {
                mode: {
                    "metrics": values.get("metrics"),
                    "latency_ms": values.get("latency_ms"),
                }
                for mode, values in channel["modes"].items()
            }
            for name, channel in (
                (SAFE_CHANNEL, safe_channel),
                (RERANKED_CHANNEL, reranked_channel),
            )
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
