from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_benchmark import evaluate_bm25_baseline
from kb.product_query_policy import search_hits_with_fallback
from kb.retrieval_eval_fixture import sha256_file
from kb.retrieval_semantic_eval_fixture import load_retrieval_benchmark_fixture
from kb.retriever import BM25Retriever
from kb.store import load_all_chunks


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_eval_unseen_v2.json"
DEFAULT_BASELINE = REPO_ROOT / "docs" / "retrieval_bm25_baseline_unseen_v2.json"
DEFAULT_OUT_ROOT = REPO_ROOT / "test_results" / "retrieval_eval_unseen_v2"
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


def _product_bm25_search(
    query: str,
    retriever: BM25Retriever,
    top_k: int,
    whole_library: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = SimpleNamespace(api_key=None, query_expansion_enabled=False)
    hits, _scores, _used_query, used_translation, query_variants = search_hits_with_fallback(
        query,
        retriever,
        top_k,
        settings,
        allow_translate=True,
        allow_expand=False,
        whole_library=whole_library,
    )
    return list(hits[:top_k]), {
        "used_translation": bool(used_translation),
        "query_variant_count": len(query_variants),
    }


def _validate_baseline(
    fixture: dict[str, Any],
    fixture_path: Path,
    baseline_path: Path,
) -> list[str]:
    errors: list[str] = []
    if not baseline_path.is_file():
        return [f"baseline report does not exist: {baseline_path}"]
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if str(baseline.get("fixture_id") or "") != str(fixture.get("id") or ""):
        errors.append("baseline fixture_id does not match")
    source = baseline.get("fixture") if isinstance(baseline.get("fixture"), dict) else {}
    if str(source.get("sha256") or "") != sha256_file(fixture_path):
        errors.append("baseline fixture SHA-256 does not match")
    fixture_corpus = fixture.get("corpus") if isinstance(fixture.get("corpus"), dict) else {}
    baseline_corpus = baseline.get("corpus") if isinstance(baseline.get("corpus"), dict) else {}
    if str(baseline_corpus.get("fingerprint") or "") != str(fixture_corpus.get("fingerprint") or ""):
        errors.append("baseline corpus fingerprint does not match")
    if int(baseline.get("case_count") or 0) != len(list(fixture.get("cases") or [])):
        errors.append("baseline case count does not match")
    channels = baseline.get("channels") if isinstance(baseline.get("channels"), dict) else {}
    for channel in ("bm25_raw", "bm25_product_queries"):
        modes = (channels.get(channel) or {}).get("modes") if isinstance(channels.get(channel), dict) else {}
        if not isinstance(modes, dict) or not {"scoped", "full_library"}.issubset(modes):
            errors.append(f"baseline is missing {channel} scoped/full_library results")
    return errors


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    channels: dict[str, Any] = {}
    for name, channel in dict(report.get("channels") or {}).items():
        channels[name] = {
            mode: {
                "case_count": data.get("case_count"),
                "zero_hit_cases": data.get("zero_hit_cases"),
                "translated_cases": data.get("translated_cases"),
                "metrics": data.get("metrics"),
                "latency_ms": data.get("latency_ms"),
            }
            for mode, data in dict(channel.get("modes") or {}).items()
        }
    return {
        "fixture_id": report.get("fixture_id"),
        "case_count": report.get("case_count"),
        "corpus": report.get("corpus"),
        "channels": channels,
    }


def _committed_summary(report: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: report.get(key)
        for key in (
            "schema_version",
            "fixture_id",
            "case_count",
            "corpus",
            "fixture",
            "measured_at",
        )
    }
    compact["report_kind"] = "committed_baseline_summary"
    compact_channels: dict[str, Any] = {}
    for channel_name, channel in dict(report.get("channels") or {}).items():
        compact_modes: dict[str, Any] = {}
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
                metrics = {
                    key: value
                    for key, value in dict(case.get("metrics") or {}).items()
                    if key in _CASE_METRICS
                }
                cases.append(
                    {
                        "id": case.get("id"),
                        "locale": case.get("locale"),
                        "evaluation_focus": case.get("evaluation_focus"),
                        "coverage_tags": case.get("coverage_tags"),
                        "first_relevant_rank": case.get("first_relevant_rank"),
                        "label_count": case.get("label_count"),
                        "contract_count": case.get("contract_count"),
                        "used_translation": case.get("used_translation"),
                        "query_variant_count": case.get("query_variant_count"),
                        "metrics": metrics,
                    }
                )
            compact_modes[mode_name] = {
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
            compact_modes[mode_name]["groups"] = groups
            compact_modes[mode_name]["cases"] = cases
        compact_channels[channel_name] = {"channel": channel_name, "modes": compact_modes}
    compact["channels"] = compact_channels
    return compact


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure the frozen unseen-paper chunk qrels with raw and product-query BM25."
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--db-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--verbose-json", action="store_true")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write a compact committed summary without top-K result lists.",
    )
    args = parser.parse_args()

    fixture_path = args.fixture.resolve()
    fixture = load_retrieval_benchmark_fixture(fixture_path, repo_root=REPO_ROOT)
    if args.validate_only:
        errors = _validate_baseline(fixture, fixture_path, args.baseline.resolve())
        if errors:
            print("[FAIL] " + "; ".join(errors))
            return 1
        print(
            f"[OK] frozen real-paper qrels and baseline metadata: "
            f"{len(fixture['cases'])} cases, {fixture['corpus']['chunk_count']} chunks"
        )
        return 0
    if args.db_root is None:
        parser.error("--db-root is required unless --validate-only is used")

    chunks = load_all_chunks(args.db_root.resolve())
    if not chunks:
        raise ValueError(f"no ready chunks found under {args.db_root.resolve()}")
    report = evaluate_bm25_baseline(fixture, chunks, product_search=_product_bm25_search)
    report["fixture"] = {
        "path": fixture_path.relative_to(REPO_ROOT).as_posix()
        if fixture_path.is_relative_to(REPO_ROOT)
        else fixture_path.name,
        "sha256": sha256_file(fixture_path),
        "sha256_newlines": "normalized_lf",
    }
    report["measured_at"] = datetime.now(timezone.utc).isoformat()

    if args.compact:
        report = _committed_summary(report)

    if args.verbose_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(_summary(report), ensure_ascii=False, indent=2))

    if args.output is not None:
        output = args.output.resolve()
    else:
        output = args.out_root.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S") / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
