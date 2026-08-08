from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kb.evidence_watch import build_evidence_watch_events, source_identity, source_watch_snapshot
from kb.store import load_docs_index


DEFAULT_FIXTURE = ROOT / "test_results" / "evidence_matrix" / "20260806_012452" / "deterministic_report.json"


def _write_report(root: Path, payload: dict[str, Any]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = root / stamp
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "report.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _source_map(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("identity") or ""): item
        for item in list(snapshot.get("sources") or [])
        if isinstance(item, dict) and str(item.get("identity") or "")
    }


def _mutated_snapshot(snapshot: dict[str, Any], identity: str, *, kind: str) -> dict[str, Any]:
    out = deepcopy(snapshot)
    source = _source_map(out)[identity]
    if kind == "content":
        source["content_fingerprint"] = "changed-" + str(source.get("content_fingerprint") or "")
        source.setdefault("content", {})["sha256"] = source["content_fingerprint"]
    elif kind == "metadata":
        source["metadata_fingerprint"] = "changed-" + str(source.get("metadata_fingerprint") or "")
        source["title"] = str(source.get("title") or source.get("source_name") or "") + " (corrected)"
    elif kind == "unavailable":
        source["content_fingerprint"] = ""
        source["content"] = {"exists": False, "size": 0, "mtime_ns": 0, "sha256": ""}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay five real-corpus evidence-change scenarios.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--db-root", type=Path, default=ROOT / "db")
    parser.add_argument("--out-root", type=Path, default=ROOT / "test_results" / "evidence_change_watch")
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    case = next(item for item in list(fixture.get("cases") or []) if isinstance(item, dict))
    selected = [item for item in list(case.get("selected_items") or []) if isinstance(item, dict)]
    if len(selected) < 2:
        raise RuntimeError("fixture must contain at least two real sources")
    matrix = dict(case.get("matrix") or {})
    matrix.update({"project_id": "real-change-watch-eval", "title": "Real change-watch replay"})
    baseline = source_watch_snapshot(selected, shelf_revision=1)
    identities = [source_identity(item.get("sourcePath")) for item in selected]
    if any(not identity for identity in identities):
        raise RuntimeError("fixture contains an invalid source path")

    docs = load_docs_index(args.db_root)
    selected_identities = set(identities)
    added_record = next(
        (
            record
            for record in docs.values()
            if isinstance(record, dict)
            and source_identity(record.get("path")) not in selected_identities
            and Path(str(record.get("path") or "")).is_file()
            and int(record.get("num_chunks") or 0) > 0
        ),
        None,
    )
    if not isinstance(added_record, dict):
        raise RuntimeError("no additional indexed real source is available")
    added_path = str(added_record.get("path") or "")
    added_item = {
        "key": "real-added-source",
        "title": Path(added_path).stem,
        "sourceName": Path(added_path).name,
        "sourcePath": added_path,
    }
    briefs = [
        {
            "id": "real-brief-impact",
            "title": "Real matrix-backed brief",
            "revision": 1,
            "quality": {"source_matrix_id": str(matrix.get("id") or "")},
            "evidence": [{"source_path": str(selected[0].get("sourcePath") or ""), "citation_number": 1}],
        }
    ]
    scenarios = [
        ("content_changed", "source_content_changed", _mutated_snapshot(baseline, identities[0], kind="content")),
        ("metadata_changed", "source_metadata_changed", _mutated_snapshot(baseline, identities[0], kind="metadata")),
        ("source_added", "source_added", source_watch_snapshot([*selected, added_item], shelf_revision=2)),
        ("source_removed", "source_removed", source_watch_snapshot(selected[:1], shelf_revision=2)),
        ("source_unavailable", "source_unavailable", _mutated_snapshot(baseline, identities[0], kind="unavailable")),
    ]

    results: list[dict[str, Any]] = []
    elapsed_values: list[float] = []
    for scenario_id, expected_kind, current in scenarios:
        started = time.perf_counter()
        events = build_evidence_watch_events(
            matrix,
            baseline=baseline,
            current=current,
            briefs=briefs,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        elapsed_values.append(elapsed_ms)
        matching = [item for item in events if str(item.get("kind") or "") == expected_kind]
        event = matching[0] if len(matching) == 1 else {}
        impact = event.get("impact") if isinstance(event.get("impact"), dict) else {}
        passed = len(matching) == 1
        if expected_kind == "source_metadata_changed":
            passed = passed and not bool(event.get("actionable"))
        else:
            passed = passed and bool(event.get("actionable"))
        if expected_kind in {"source_content_changed", "source_removed", "source_unavailable"}:
            passed = passed and bool(impact.get("affected_row_ids")) and bool(impact.get("affected_fields"))
        if expected_kind == "source_content_changed":
            passed = passed and int(impact.get("affected_brief_count") or 0) == 1
            passed = passed and int(impact.get("affected_citation_count") or 0) == 1
        if expected_kind == "source_added":
            passed = passed and len(list(impact.get("candidate_fields") or [])) == 5
        results.append(
            {
                "id": scenario_id,
                "expected_kind": expected_kind,
                "observed_kinds": [str(item.get("kind") or "") for item in events],
                "passed": bool(passed),
                "actionable": bool(event.get("actionable")),
                "affected_rows": len(list(impact.get("affected_row_ids") or [])),
                "affected_fields": len(list(impact.get("affected_fields") or [])),
                "affected_comparisons": len(list(impact.get("affected_comparison_ids") or [])),
                "affected_briefs": int(impact.get("affected_brief_count") or 0),
                "affected_citations": int(impact.get("affected_citation_count") or 0),
                "elapsed_ms": elapsed_ms,
            }
        )

    payload = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "fixture": str(args.fixture.resolve()),
        "db_root": str(args.db_root.resolve()),
        "real_source_count": len(selected) + 1,
        "source_fingerprints_present": all(
            bool(item.get("content_fingerprint")) and bool((item.get("content") or {}).get("exists"))
            for item in list(baseline.get("sources") or [])
            if isinstance(item, dict)
        ),
        "cases": results,
        "summary": {
            "passed": sum(1 for item in results if bool(item.get("passed"))),
            "total": len(results),
            "median_ms": round(statistics.median(elapsed_values), 3),
            "max_ms": round(max(elapsed_values), 3),
        },
    }
    report_path = _write_report(args.out_root, payload)
    print(json.dumps({"report": str(report_path), **payload["summary"]}, ensure_ascii=False))
    return 0 if payload["summary"]["passed"] == payload["summary"]["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
