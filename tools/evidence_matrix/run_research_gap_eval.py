from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kb.evidence_watch import source_identity
from kb.research_gap import build_project_research_gaps, find_research_gap_candidates
from kb.store import load_all_chunks


DEFAULT_FIXTURE = ROOT / "test_results" / "evidence_matrix" / "20260806_012452" / "deterministic_report.json"


def _normalized(value: object) -> str:
    return " ".join(str(value or "").split())


def _write_report(root: Path, payload: dict[str, Any]) -> Path:
    folder = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "report.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _expected_missing(quality: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(item.get("row_id") or ""), str(item.get("field") or ""))
        for item in list(quality.get("missing_cells") or [])
        if isinstance(item, dict) and str(item.get("row_id") or "") and str(item.get("field") or "")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay project research gaps on the five reviewed real evidence matrices.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--db-root", type=Path, default=ROOT / "db")
    parser.add_argument("--out-root", type=Path, default=ROOT / "test_results" / "research_gap")
    parser.add_argument("--candidate-limit", type=int, default=3)
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    cases = [item for item in list(fixture.get("cases") or []) if isinstance(item, dict)]
    if len(cases) != 5:
        raise RuntimeError(f"reviewed fixture must contain five real matrices, got {len(cases)}")
    chunks = [item for item in load_all_chunks(args.db_root) if isinstance(item, dict)]
    chunks_by_id = {str(item.get("id") or ""): item for item in chunks if str(item.get("id") or "")}

    results: list[dict[str, Any]] = []
    build_times: list[float] = []
    candidate_times: list[float] = []
    total_candidates = 0
    exact_candidates = 0
    for case in cases:
        case_id = str(case.get("id") or "")
        matrix = dict(case.get("matrix") or {})
        quality = dict(case.get("quality") or {})
        matrix.update(
            {
                "project_id": f"real-gap-{case_id}",
                "title": f"Reviewed real matrix: {case_id}",
                "objective": str(case.get("objective") or ""),
                "quality": quality,
            }
        )
        started = time.perf_counter()
        gaps = build_project_research_gaps(project_id=matrix["project_id"], matrices=[matrix])
        build_ms = round((time.perf_counter() - started) * 1000, 3)
        build_times.append(build_ms)

        missing = [item for item in gaps if str(item.get("kind") or "") == "missing_cell"]
        observed_missing = {(str(item.get("row_id") or ""), str(item.get("field") or "")) for item in missing}
        expected_missing = _expected_missing(quality)
        excluded = [
            str(row.get("source_path") or "")
            for row in list(matrix.get("rows") or [])
            if isinstance(row, dict) and str(row.get("source_path") or "")
        ]
        excluded_identities = {source_identity(path) for path in excluded if source_identity(path)}

        case_candidates: list[dict[str, Any]] = []
        search_started = time.perf_counter()
        for gap in missing:
            candidates = find_research_gap_candidates(
                gap,
                db_dir=args.db_root,
                excluded_source_paths=excluded,
                limit=args.candidate_limit,
                chunks=chunks,
            )
            for candidate in candidates:
                source_path = str(candidate.get("source_path") or "")
                chunk = chunks_by_id.get(str(candidate.get("chunk_id") or ""), {})
                quote = _normalized(candidate.get("evidence_quote"))
                exact = bool(
                    source_identity(source_path)
                    and source_identity(source_path) not in excluded_identities
                    and quote
                    and quote in _normalized(chunk.get("text"))
                    and source_identity((chunk.get("meta") or {}).get("source_path")) == source_identity(source_path)
                    and (
                        str(candidate.get("anchor_id") or "")
                        or str(candidate.get("block_id") or "")
                        or str(candidate.get("heading_path") or "")
                        or candidate.get("page_start") is not None
                    )
                )
                case_candidates.append(
                    {
                        "gap_key": str(gap.get("gap_key") or ""),
                        "field": str(gap.get("field") or ""),
                        "source_path": source_path,
                        "chunk_id": str(candidate.get("chunk_id") or ""),
                        "exact_same_source_with_locator": exact,
                    }
                )
                total_candidates += 1
                exact_candidates += int(exact)
        candidate_ms = round((time.perf_counter() - search_started) * 1000, 3)
        candidate_times.append(candidate_ms)
        passed = observed_missing == expected_missing and all(
            bool(item.get("exact_same_source_with_locator")) for item in case_candidates
        )
        results.append(
            {
                "id": case_id,
                "real_source_count": len(excluded),
                "expected_missing_count": len(expected_missing),
                "observed_missing_count": len(observed_missing),
                "missing_contract_exact": observed_missing == expected_missing,
                "candidate_count": len(case_candidates),
                "candidate_exact_count": sum(1 for item in case_candidates if item["exact_same_source_with_locator"]),
                "build_ms": build_ms,
                "candidate_search_ms": candidate_ms,
                "passed": passed,
                "candidates": case_candidates,
            }
        )

    payload = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "fixture": str(args.fixture.resolve()),
        "db_root": str(args.db_root.resolve()),
        "indexed_chunk_count": len(chunks),
        "cases": results,
        "summary": {
            "passed": sum(1 for item in results if item["passed"]),
            "total": len(results),
            "missing_gap_count": sum(int(item["observed_missing_count"]) for item in results),
            "candidate_count": total_candidates,
            "candidate_exact_count": exact_candidates,
            "build_median_ms": round(statistics.median(build_times), 3),
            "build_max_ms": round(max(build_times), 3),
            "candidate_search_median_ms": round(statistics.median(candidate_times), 3),
            "candidate_search_max_ms": round(max(candidate_times), 3),
        },
    }
    report_path = _write_report(args.out_root, payload)
    print(json.dumps({"report": str(report_path), **payload["summary"]}, ensure_ascii=False))
    return 0 if payload["summary"]["passed"] == payload["summary"]["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
