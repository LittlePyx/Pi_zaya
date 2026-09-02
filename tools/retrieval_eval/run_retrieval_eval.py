from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_metrics import evaluate_ranking, mean_metrics
from kb.retriever import BM25Retriever


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_eval_v1.json"
DEFAULT_OUT_ROOT = REPO_ROOT / "test_results" / "retrieval_eval"


def _load_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version") or 0) != 1:
        raise ValueError("retrieval fixture schema_version must be 1")
    if not isinstance(payload.get("corpus"), list) or not payload["corpus"]:
        raise ValueError("retrieval fixture corpus cannot be empty")
    if not isinstance(payload.get("cases"), list) or not payload["cases"]:
        raise ValueError("retrieval fixture cases cannot be empty")
    return payload


def _acceptance_failures(report: dict[str, Any], acceptance: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    groups = report.get("groups") if isinstance(report.get("groups"), dict) else {}
    for group, thresholds in acceptance.items():
        metrics = (groups.get(group) or {}).get("metrics") if isinstance(groups.get(group), dict) else {}
        if not isinstance(thresholds, dict) or not isinstance(metrics, dict):
            failures.append(f"missing metrics for acceptance group {group}")
            continue
        for raw_name, raw_limit in thresholds.items():
            name = str(raw_name)
            if not name.startswith("min_"):
                continue
            metric_name = name[4:]
            actual = float(metrics.get(metric_name, 0.0) or 0.0)
            limit = float(raw_limit)
            if actual + 1e-12 < limit:
                failures.append(f"{group}.{metric_name}={actual:.4f} is below {limit:.4f}")
    return failures


def run_fixture(payload: dict[str, Any]) -> dict[str, Any]:
    corpus = [dict(item) for item in payload["corpus"] if isinstance(item, dict)]
    retriever = BM25Retriever(corpus)
    ks = tuple(max(1, int(value)) for value in list(payload.get("ks") or [1, 3, 5]))
    max_k = max(ks)
    rows: list[dict[str, Any]] = []
    grouped_metric_rows: dict[str, list[dict[str, float]]] = {}
    for case in payload["cases"]:
        if not isinstance(case, dict):
            continue
        relevance = {
            str(key): float(value)
            for key, value in dict(case.get("relevance") or {}).items()
            if float(value) > 0.0
        }
        hits = retriever.search(str(case.get("query") or ""), top_k=max_k)
        ranked_ids = [str(hit.get("id") or "") for hit in hits]
        metrics = evaluate_ranking(ranked_ids, relevance, ks=ks)
        group = str(case.get("group") or "default")
        grouped_metric_rows.setdefault(group, []).append(metrics)
        rows.append(
            {
                "id": str(case.get("id") or ""),
                "group": group,
                "expected_sparse_gap": bool(case.get("expected_sparse_gap")),
                "ranked_ids": ranked_ids,
                "metrics": {key: round(value, 6) for key, value in metrics.items()},
            }
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "fixture_id": str(payload.get("id") or ""),
        "retriever": "bm25",
        "case_count": len(rows),
        "metrics": {key: round(value, 6) for key, value in mean_metrics([row["metrics"] for row in rows]).items()},
        "groups": {
            group: {
                "case_count": len(metric_rows),
                "metrics": {key: round(value, 6) for key, value in mean_metrics(metric_rows).items()},
            }
            for group, metric_rows in sorted(grouped_metric_rows.items())
        },
        "cases": rows,
    }
    report["acceptance_failures"] = _acceptance_failures(report, dict(payload.get("acceptance") or {}))
    report["ok"] = not report["acceptance_failures"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate deterministic candidate retrieval metrics.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-quality", action="store_true")
    parser.add_argument("--verbose-json", action="store_true")
    args = parser.parse_args()

    payload = _load_fixture(args.fixture.resolve())
    report = run_fixture(payload)
    if args.verbose_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            json.dumps(
                {
                    "ok": report["ok"],
                    "fixture_id": report["fixture_id"],
                    "retriever": report["retriever"],
                    "case_count": report["case_count"],
                    "metrics": report["metrics"],
                    "groups": report["groups"],
                    "acceptance_failures": report["acceptance_failures"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    if not args.dry_run:
        output_dir = args.out_root.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.fail_on_quality and not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
