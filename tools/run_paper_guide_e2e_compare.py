from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = str(ln or "").strip()
        if not s or s.startswith("#"):
            continue
        try:
            rec = json.loads(s)
        except Exception:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _print_counter(title: str, c: Counter) -> None:
    items = ", ".join([f"{k}={v}" for k, v in c.most_common()])
    print(f"{title}: {items}")


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    merged = dict(os.environ)
    merged.setdefault("PYTHONIOENCODING", "utf-8")
    if env:
        merged.update({k: str(v) for k, v in env.items()})
    subprocess.run(cmd, check=True, env=merged)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="One-shot: collect failure pool then run end-to-end compare old vs new DB.")
    ap.add_argument("--new-db-dir", required=True, help="New KB_DB_DIR root (reconverted).")
    ap.add_argument("--old-db-dir", default="db", help="Old KB_DB_DIR root (default: db).")
    ap.add_argument("--max-cases", type=int, default=300, help="Max cases to collect from chat db.")
    ap.add_argument("--failures-only", action="store_true", help="Only collect failure candidates.")
    ap.add_argument("--include-archived", action="store_true", help="Include archived conversations.")
    ap.add_argument("--top-k", type=int, default=6, help="Retrieval top_k for end-to-end compare.")
    ap.add_argument("--pool-out", default="", help="Pool jsonl output path (default: tmp_e2e_pool.jsonl).")
    ap.add_argument("--report-out", default="", help="Report json output path (default: tmp_e2e_compare_report.json).")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    collect_script = root / "tests" / "replay" / "collect_paper_guide_replay_cases.py"
    compare_script = root / "tools" / "compare_paper_guide_end_to_end.py"

    pool_out = Path(args.pool_out or "tmp_e2e_pool.jsonl").expanduser().resolve()
    report_out = Path(args.report_out or "tmp_e2e_compare_report.json").expanduser().resolve()

    collect_cmd = [sys.executable, str(collect_script), "--out", str(pool_out), "--max-cases", str(int(args.max_cases))]
    if bool(args.failures_only):
        collect_cmd.append("--failures-only")
    collect_cmd.append("--summary")
    if bool(args.include_archived):
        collect_cmd.append("--include-archived")

    print(f"[1/2] collect pool -> {pool_out}")
    _run(collect_cmd)

    print(f"[2/2] compare end-to-end old={args.old_db_dir} new={args.new_db_dir} -> {report_out}")
    compare_cmd = [
        sys.executable,
        str(compare_script),
        "--pool",
        str(pool_out),
        "--old-db-dir",
        str(args.old_db_dir),
        "--new-db-dir",
        str(args.new_db_dir),
        "--top-k",
        str(int(args.top_k)),
        "--out",
        str(report_out),
    ]
    _run(compare_cmd)

    # Lightweight console diff view.
    payload = json.loads(report_out.read_text(encoding="utf-8"))
    old_sum = payload.get("old") or {}
    new_sum = payload.get("new") or {}
    print("=== DIFF (new - old) ===")
    print("old:", old_sum)
    print("new:", new_sum)
    try:
        _print_counter("old.tag", Counter(old_sum.get("tag") or {}))
        _print_counter("new.tag", Counter(new_sum.get("tag") or {}))
        _print_counter("old.hit_level", Counter(old_sum.get("hit_level") or {}))
        _print_counter("new.hit_level", Counter(new_sum.get("hit_level") or {}))
        _print_counter("old.claim_type", Counter(old_sum.get("claim_type") or {}))
        _print_counter("new.claim_type", Counter(new_sum.get("claim_type") or {}))
    except Exception:
        pass

    # Print top worst cases by must_fallback then missing locate.
    try:
        old_cases = list((payload.get("cases") or {}).get("old") or [])
        new_cases = list((payload.get("cases") or {}).get("new") or [])
        # join by id
        new_by_id = {str(c.get("case_id") or ""): c for c in new_cases if isinstance(c, dict)}
        scored = []
        for oc in old_cases:
            if not isinstance(oc, dict):
                continue
            cid = str(oc.get("case_id") or "")
            nc = new_by_id.get(cid) or {}
            scored.append(
                (
                    int(oc.get("must_fallback") or 0),
                    int(oc.get("must_locate") or 0),
                    int(0 if oc.get("has_locate") else 1),
                    str(oc.get("tag") or ""),
                    cid,
                    oc,
                    nc,
                )
            )
        scored.sort(key=lambda x: (x[2], x[0], x[1]), reverse=True)
        print("=== TOP WORST (old) ===")
        for row in scored[:20]:
            _must_fb, _must_loc, miss, tag, cid, oc, nc = row
            print(
                f"id={cid} tag={tag} miss_locate={miss} must_fallback={_must_fb}/{_must_loc} "
                f"old_hit={oc.get('hit_level')} new_hit={nc.get('hit_level')} "
                f"old_claim={oc.get('claim_type')} new_claim={nc.get('claim_type')}"
            )
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
