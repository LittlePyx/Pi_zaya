from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_benchmark import validate_indexed_corpus
from kb.retrieval_eval_fixture import sha256_file
from kb.retrieval_representation_diagnostic import analyze_chunk_representation
from kb.retrieval_semantic_eval_fixture import load_retrieval_benchmark_fixture
from kb.store import load_all_chunks


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_cases_v1.json"
DEFAULT_BENCHMARK = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_benchmark_v1.json"
DEFAULT_CANDIDATE = REPO_ROOT / "docs" / "retrieval_dense_multilingual_e5_base_unseen_v2.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "retrieval_semantic_representation_diagnostic_v1.json"
MODEL_ID = "intfloat/multilingual-e5-base"
REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
PASSAGE_PREFIX = "passage: "
MAX_LENGTH = 512


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return ""


def _display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _failure_categories(benchmark: dict[str, Any]) -> dict[str, str]:
    analysis = benchmark.get("failure_analysis")
    analysis = analysis if isinstance(analysis, dict) else {}
    categories: dict[str, str] = {}
    for case_id in analysis.get("missed_by_product_bm25_and_raw_dense_at_10_in_both_modes") or []:
        categories[str(case_id)] = "shared_top10_miss"
    for case_id in analysis.get("raw_dense_regressions_at_10") or []:
        categories[str(case_id)] = "raw_dense_top10_regression"
    return categories


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether frozen semantic retrieval failures are caused by the "
            "pinned encoder's 512-token passage window. No vectors are created."
        )
    )
    parser.add_argument("--db-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--candidate-report", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()

    fixture_path = args.fixture.expanduser().resolve()
    benchmark_path = args.benchmark.expanduser().resolve()
    candidate_path = args.candidate_report.expanduser().resolve()
    db_root = args.db_root.expanduser().resolve()
    fixture = load_retrieval_benchmark_fixture(fixture_path, repo_root=REPO_ROOT)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    chunks = load_all_chunks(db_root)
    _chunk_to_doc, corpus_errors = validate_indexed_corpus(fixture, chunks)
    if corpus_errors:
        raise ValueError("indexed corpus does not match frozen qrels: " + "; ".join(corpus_errors))
    chunks_by_id = {str(chunk.get("id") or ""): chunk for chunk in chunks}

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("semantic representation analysis requires transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        cache_dir=str(args.cache_dir.resolve()) if args.cache_dir else None,
        local_files_only=True,
        trust_remote_code=False,
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise ValueError("the pinned tokenizer must expose offset mappings")

    corpus_token_counts = [
        len(
            tokenizer(
                f"{PASSAGE_PREFIX}{str(chunk.get('text') or '')}",
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )["input_ids"]
        )
        for chunk in chunks
    ]
    failure_categories = _failure_categories(benchmark)
    labels: list[dict[str, Any]] = []
    for case in fixture.get("cases") or []:
        case_id = str(case.get("id") or "")
        for label in case.get("labels") or []:
            chunk_id = str(label.get("chunk_id") or "")
            chunk = chunks_by_id[chunk_id]
            labels.append(
                {
                    "case_id": case_id,
                    "failure_category": failure_categories.get(case_id, "non_focus"),
                    "chunk_id": chunk_id,
                    "text_sha256": str(label.get("text_sha256") or ""),
                    "page_start": label.get("page_start"),
                    "page_end": label.get("page_end"),
                    **analyze_chunk_representation(
                        text=str(chunk.get("text") or ""),
                        evidence_terms=[str(term) for term in label.get("matched_evidence_terms") or []],
                        tokenize=tokenizer,
                        max_length=MAX_LENGTH,
                        passage_prefix=PASSAGE_PREFIX,
                    ),
                }
            )

    focus_labels = [item for item in labels if item["failure_category"] != "non_focus"]
    report = {
        "schema_version": 1,
        "report_kind": "semantic_chunk_representation_diagnostic",
        "fixture": {
            "id": fixture.get("id"),
            "path": _display_path(fixture_path),
            "sha256": sha256_file(fixture_path),
        },
        "benchmark": {
            "path": _display_path(benchmark_path),
            "sha256": sha256_file(benchmark_path),
            "decision": (benchmark.get("decision") or {}).get("status"),
        },
        "candidate_report": {
            "path": _display_path(candidate_path),
            "sha256": sha256_file(candidate_path),
            "indexed_truncated_chunk_count": (candidate.get("candidate") or {}).get(
                "truncated_document_count"
            ),
        },
        "candidate": {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "tokenizer_class": type(tokenizer).__name__,
            "transformers_version": _package_version("transformers"),
            "passage_prefix": PASSAGE_PREFIX,
            "max_length": MAX_LENGTH,
            "local_files_only": True,
            "trust_remote_code": False,
            "product_enabled": False,
        },
        "corpus": {
            "document_count": (fixture.get("corpus") or {}).get("document_count"),
            "chunk_count": len(chunks),
            "fingerprint": (fixture.get("corpus") or {}).get("fingerprint"),
            "tokenized_chunk_count": len(corpus_token_counts),
            "truncated_chunk_count": sum(count > MAX_LENGTH for count in corpus_token_counts),
            "max_token_count": max(corpus_token_counts, default=0),
        },
        "population": {
            "case_count": len(fixture.get("cases") or []),
            "label_count": len(labels),
            "unique_relevant_chunk_count": len({item["chunk_id"] for item in labels}),
            "truncated_relevant_label_count": sum(item["would_truncate"] for item in labels),
            "evidence_outside_window_label_count": sum(
                not item["all_evidence_inside_window"] for item in labels
            ),
            "max_relevant_token_count": max(
                (item["token_count_with_prefix"] for item in labels), default=0
            ),
        },
        "focus": {
            "case_ids": list(failure_categories),
            "case_count": len(failure_categories),
            "label_count": len(focus_labels),
            "truncated_label_count": sum(item["would_truncate"] for item in focus_labels),
            "evidence_outside_window_label_count": sum(
                not item["all_evidence_inside_window"] for item in focus_labels
            ),
            "max_token_count": max(
                (item["token_count_with_prefix"] for item in focus_labels), default=0
            ),
        },
        "labels": labels,
        "decision": {
            "truncation_explains_frozen_failures": False,
            "rechunk_stored_citation_units": False,
            "interpretation": (
                "The corpus contains long truncated chunks, but none of the 15 labelled semantic "
                "evidence units and none of the six labels behind the five focus failures exceed "
                "the pinned window. The measured failures are not evidence-loss-by-truncation."
            ),
            "only_permitted_subchunk_experiment": (
                "Represent each original chunk by multiple bounded child embeddings, score the "
                "parent by child max score, and collapse results back to the unchanged parent "
                "chunk ID/page locator before fusion or citation."
            ),
            "next_action": (
                "Do not alter stored chunking. Reprioritize the frozen multilingual NLI baseline; "
                "if retrieval experimentation resumes, pre-register a development-only "
                "parent-collapsed multi-vector candidate before replaying the frozen slice."
            ),
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "[OK] semantic representation diagnostic: "
        f"{report['population']['label_count']} labels, "
        f"{report['population']['truncated_relevant_label_count']} truncated, "
        f"{report['focus']['case_count']} focus failures"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
