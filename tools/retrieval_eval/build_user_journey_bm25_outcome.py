from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.validate_user_journey_bm25_baseline import (
    DEFAULT_BASELINE,
    DEFAULT_FIXTURE,
    EXPECTED_BASELINE_SHA256,
    validate_baseline,
)


DEFAULT_OUTPUT = REPO_ROOT / "docs" / "retrieval_user_journey_bm25_outcome_v1.json"
METRIC_KEYS = (
    "mrr",
    "recall@5",
    "recall@10",
    "ndcg@10",
    "contract_recall@10",
    "contract_recall@20",
    "wrong_source_rate@5",
)


def _mean_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        key: round(
            sum(float((row.get("metrics") or {}).get(key) or 0.0) for row in rows)
            / len(rows),
            6,
        )
        for key in METRIC_KEYS
    }


def _rank_bucket(rank: Any) -> str:
    if rank is None:
        return "miss_top20"
    value = int(rank)
    if value == 1:
        return "rank_1"
    if value <= 5:
        return "ranks_2_5"
    if value <= 10:
        return "ranks_6_10"
    return "ranks_11_20"


def _result_fingerprint(baseline: dict[str, Any]) -> str:
    projection: dict[str, Any] = {}
    for channel_name, channel in (baseline.get("channels") or {}).items():
        projection[channel_name] = {}
        for mode_name, mode in (channel.get("modes") or {}).items():
            projection[channel_name][mode_name] = {
                "metrics": mode.get("metrics"),
                "cases": [
                    {
                        "id": row.get("id"),
                        "first_relevant_rank": row.get("first_relevant_rank"),
                        "used_translation": row.get("used_translation"),
                        "query_variant_count": row.get("query_variant_count"),
                        "metrics": row.get("metrics"),
                    }
                    for row in mode.get("cases") or []
                ],
            }
    canonical = json.dumps(
        projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_outcome(fixture: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    errors = validate_baseline(baseline, fixture)
    if errors:
        raise ValueError("invalid user-journey BM25 baseline: " + "; ".join(errors))
    split = fixture["split"]
    product = baseline["channels"]["bm25_product_queries"]["modes"]
    split_summary: dict[str, Any] = {}
    for split_name, member_key in (
        ("calibration", "calibration_case_ids"),
        ("holdout", "holdout_case_ids"),
    ):
        members = set(split[member_key])
        split_summary[split_name] = {}
        for mode_name, mode in product.items():
            rows = [row for row in mode["cases"] if row["id"] in members]
            split_summary[split_name][mode_name] = {
                "case_count": len(rows),
                "metrics": _mean_rows(rows),
                "first_relevant_rank_buckets": dict(
                    sorted(
                        Counter(
                            _rank_bucket(row["first_relevant_rank"]) for row in rows
                        ).items()
                    )
                ),
            }

    calibration = set(split["calibration_case_ids"])
    diagnosis: dict[str, Any] = {}
    for mode_name, mode in product.items():
        rows = [row for row in mode["cases"] if row["id"] in calibration]
        focus_summary: dict[str, Any] = {}
        for focus in sorted({str(row["evaluation_focus"]) for row in rows}):
            focus_rows = [row for row in rows if row["evaluation_focus"] == focus]
            focus_summary[focus] = {
                "case_count": len(focus_rows),
                "recall@10": _mean_rows(focus_rows)["recall@10"],
                "contract_recall@20": _mean_rows(focus_rows)["contract_recall@20"],
                "miss_top20_count": sum(
                    row["first_relevant_rank"] is None for row in focus_rows
                ),
            }
        raw_by_id = {
            row["id"]: row
            for row in baseline["channels"]["bm25_raw"]["modes"][mode_name]["cases"]
            if row["id"] in calibration
        }
        effects = Counter()
        for row in rows:
            product_value = float(row["metrics"]["recall@10"])
            raw_value = float(raw_by_id[row["id"]]["metrics"]["recall@10"])
            effects[
                "improved"
                if product_value > raw_value
                else "regressed"
                if product_value < raw_value
                else "unchanged"
            ] += 1
        diagnosis[mode_name] = {
            "by_reading_journey": focus_summary,
            "product_query_vs_raw_recall@10_case_counts": {
                key: effects.get(key, 0)
                for key in ("improved", "unchanged", "regressed")
            },
        }

    return {
        "schema_version": 1,
        "report_kind": "retrieval_user_journey_bm25_outcome_v1",
        "fixture": {
            "path": "docs/retrieval_user_journey_expansion_v1.json",
            "sha256": sha256_file(DEFAULT_FIXTURE),
        },
        "baseline": {
            "path": "docs/retrieval_user_journey_bm25_baseline_v1.json",
            "sha256": EXPECTED_BASELINE_SHA256,
            "result_fingerprint": _result_fingerprint(baseline),
        },
        "depth_contract": {
            "retrieval_ranking_depth": 20,
            "reported_metric_depths": [1, 3, 5, 10, 20],
            "note": "This BM25 baseline has no fusion layout; future fusion ranking and reporting depths must be separate fields.",
        },
        "split_summary": split_summary,
        "calibration_only_diagnosis": diagnosis,
        "decision": {
            "product_bm25_recall_is_sufficient": False,
            "reason": "Most frozen reading journeys do not retrieve all required evidence in the first 10 or 20 results.",
            "candidate_design_may_use": "calibration cases and aggregates only",
            "holdout_case_details_may_be_used_for_candidate_design": False,
            "next_step": "freeze and run one calibration-only Qwen dense diagnostic before designing fusion v3",
            "product_behavior_changed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or check the frozen BM25 outcome."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    baseline = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))
    outcome = build_outcome(fixture, baseline)
    serialized = json.dumps(outcome, ensure_ascii=False, indent=2) + "\n"
    output = args.output.expanduser().resolve()
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != serialized:
            print("[FAIL] user-journey BM25 outcome differs from frozen baseline")
            return 1
        print("[OK] user-journey BM25 outcome matches frozen baseline")
        return 0
    if output.exists():
        parser.error("BM25 outcome already exists; use --check")
    output.write_text(serialized, encoding="utf-8")
    print(f"[OK] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
