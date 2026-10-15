from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from kb.retrieval_semantic_eval_fixture import load_retrieval_benchmark_fixture


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_cases_v1.json"
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_benchmark_v1.json"
MODEL_ID = "intfloat/multilingual-e5-base"
REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
PROVIDER_IDENTITY = "b6f461d0da1959a877bc465919b54dd2121149e360a8c2c973ddfcda1e0ae99b"


def _number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def validate_report(report: dict[str, Any], fixture_path: Path) -> list[str]:
    fixture = load_retrieval_benchmark_fixture(fixture_path, repo_root=REPO_ROOT)
    errors: list[str] = []
    if report.get("report_kind") != "semantic_paraphrase_retrieval_benchmark":
        errors.append("report kind does not match")
    source = report.get("fixture") if isinstance(report.get("fixture"), dict) else {}
    if source.get("id") != fixture.get("id") or source.get("sha256") != sha256_file(fixture_path):
        errors.append("fixture identity or SHA-256 does not match")
    expected_ids = [str(case.get("id") or "") for case in fixture.get("cases") or []]
    if source.get("case_ids") != expected_ids or int(source.get("case_count") or 0) != len(expected_ids):
        errors.append("fixture case coverage does not match")
    corpus = report.get("corpus") if isinstance(report.get("corpus"), dict) else {}
    fixture_corpus = fixture.get("corpus") if isinstance(fixture.get("corpus"), dict) else {}
    for key in ("document_count", "chunk_count", "fingerprint"):
        if corpus.get(key) != fixture_corpus.get(key):
            errors.append(f"corpus {key} does not match")

    measurements = report.get("raw_measurements") if isinstance(report.get("raw_measurements"), dict) else {}
    for key in ("bm25_sha256", "dense_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(measurements.get(key) or "")):
            errors.append(f"raw measurement {key} is not pinned")
    candidate = report.get("candidate") if isinstance(report.get("candidate"), dict) else {}
    if candidate.get("model_id") != MODEL_ID or candidate.get("revision") != REVISION:
        errors.append("Dense candidate identity does not match")
    if candidate.get("provider_identity") != PROVIDER_IDENTITY:
        errors.append("Dense provider encoding contract does not match")
    if candidate.get("product_enabled") is not False:
        errors.append("rejected candidate must remain product-disabled")
    if candidate.get("allow_download") is not False or candidate.get("trust_remote_code") is not False:
        errors.append("measurement must remain cache-only with trust_remote_code disabled")
    index = candidate.get("index") if isinstance(candidate.get("index"), dict) else {}
    if int(index.get("chunk_count") or 0) != int(fixture_corpus.get("chunk_count") or 0):
        errors.append("Dense index chunk count does not match")
    if index.get("no_op") is not True or int(index.get("reused_chunk_count") or 0) != int(index.get("chunk_count") or 0):
        errors.append("semantic run must reuse the already accepted pinned index")

    channels = report.get("channels") if isinstance(report.get("channels"), dict) else {}
    for channel_name in ("product_bm25", "raw_dense", "safe_rrf"):
        channel = channels.get(channel_name) if isinstance(channels.get(channel_name), dict) else {}
        for mode in ("scoped", "full_library"):
            metrics = channel.get(mode) if isinstance(channel.get(mode), dict) else {}
            for metric in (
                "recall@5",
                "recall@10",
                "recall@20",
                "mrr",
                "ndcg@10",
                "wrong_source_rate@5",
                "latency_p50_ms",
            ):
                if _number(metrics.get(metric)) is None:
                    errors.append(f"{channel_name} {mode} is missing numeric {metric}")

    acceptance = report.get("acceptance") if isinstance(report.get("acceptance"), dict) else {}
    minimum_gain = float(acceptance.get("minimum_gain") or 0.0)
    if acceptance.get("metric") != "recall@10" or minimum_gain != 0.15:
        errors.append("semantic acceptance metric changed")
    for candidate_name, gain_key in (("raw_dense", "raw_dense_gain"), ("safe_rrf", "safe_rrf_gain")):
        gains = acceptance.get(gain_key) if isinstance(acceptance.get(gain_key), dict) else {}
        passes = True
        for mode in ("scoped", "full_library"):
            product_value = _number(((channels.get("product_bm25") or {}).get(mode) or {}).get("recall@10"))
            candidate_value = _number(((channels.get(candidate_name) or {}).get(mode) or {}).get("recall@10"))
            if product_value is None or candidate_value is None:
                continue
            actual_gain = round(candidate_value - product_value, 6)
            if _number(gains.get(mode)) != actual_gain:
                errors.append(f"{candidate_name} {mode} gain is stale")
            passes = passes and actual_gain >= minimum_gain
        if gains.get("passes_all_required_modes") is not passes:
            errors.append(f"{candidate_name} acceptance decision is stale")

    for metric, flag in (
        ("recall@5", "safe_rrf_preserves_product_recall_at_5"),
        ("wrong_source_rate@5", "safe_rrf_preserves_product_wrong_source_at_5"),
    ):
        preserved = all(
            ((channels.get("product_bm25") or {}).get(mode) or {}).get(metric)
            == ((channels.get("safe_rrf") or {}).get(mode) or {}).get(metric)
            for mode in ("scoped", "full_library")
        )
        if acceptance.get(flag) is not preserved:
            errors.append(f"safe RRF {metric} preservation decision is stale")
    decision = report.get("decision") if isinstance(report.get("decision"), dict) else {}
    if decision.get("status") != "rejected_for_product_promotion":
        errors.append("safe RRF must remain rejected after missing its frozen threshold")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the frozen semantic retrieval outcome.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = json.loads(args.report.resolve().read_text(encoding="utf-8"))
    errors = validate_report(report, args.fixture.resolve())
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    gains = report["acceptance"]
    print(
        "[OK] semantic retrieval report: "
        f"raw Dense gain={gains['raw_dense_gain']}, safe RRF gain={gains['safe_rrf_gain']}, "
        f"decision={report['decision']['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
