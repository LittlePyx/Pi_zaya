from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from api.chat_render import enrich_messages_with_reference_render
except ModuleNotFoundError:  # pragma: no cover
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from api.chat_render import enrich_messages_with_reference_render  # type: ignore


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = str(ln or "").strip()
        if not s or s.startswith("#"):
            continue
        rec = json.loads(s)
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _swap_db_root(p: str, new_db_root: Path) -> str:
    s = str(p or "").strip()
    if not s:
        return s
    norm = s.replace("/", "\\")
    marker = "\\db\\"
    idx = norm.lower().find(marker)
    if idx < 0:
        # Allow already pointing at other roots, but still try to map by leaf folder.
        return str(new_db_root / Path(norm).parent.name / Path(norm).name)
    suffix = norm[idx + len(marker) :]
    return str(Path(new_db_root) / suffix)


def _remap_case_paths(case: dict[str, Any], *, new_db_root: Path) -> dict[str, Any]:
    out = dict(case)
    prov = out.get("provenance")
    if isinstance(prov, dict):
        prov2 = dict(prov)
        for k in ["md_path", "source_path"]:
            prov2[k] = _swap_db_root(str(prov2.get(k) or ""), new_db_root)
        prov2["block_map"] = prov2.get("block_map")  # keep
        out["provenance"] = prov2
    hits = []
    for h in list(out.get("hits") or []):
        if not isinstance(h, dict):
            continue
        h2 = dict(h)
        meta = h2.get("meta")
        if isinstance(meta, dict):
            meta2 = dict(meta)
            meta2["source_path"] = _swap_db_root(str(meta2.get("source_path") or ""), new_db_root)
            h2["meta"] = meta2
        hits.append(h2)
    out["hits"] = hits
    return out


@dataclass
class RunResult:
    has_locate: bool
    hit_level: str
    tag: str
    claim_type: str


def _run_case(case: dict[str, Any]) -> RunResult:
    prompt = str(case.get("prompt") or "").strip()
    assistant_content = str(case.get("assistant_content") or "").strip()
    provenance = case.get("provenance") if isinstance(case.get("provenance"), dict) else None
    hits = [dict(item) for item in list(case.get("hits") or []) if isinstance(item, dict)]
    tag = str(case.get("tag") or "").strip() or "unknown"

    messages = [
        {"id": 1, "role": "user", "content": prompt or "test"},
        {
            "id": 2,
            "role": "assistant",
            "content": assistant_content or "answer",
            "provenance": provenance,
            "meta": {"paper_guide_contracts": {"version": 1, "intent": {"family": str(case.get("intent_family") or "").strip()}}},
        },
    ]
    refs_by_user = {1: {"hits": hits}}
    rendered = enrich_messages_with_reference_render(
        messages,
        refs_by_user,
        conv_id="conv-test",
        render_packet_only=True,
    )
    packet = (((rendered[-1].get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    reader_open = packet.get("reader_open") or {}
    locate_target = packet.get("locate_target") or (reader_open.get("locateTarget") or {})

    has_locate = bool(str(reader_open.get("blockId") or "").strip() and str(reader_open.get("anchorId") or "").strip())
    hit_level = str(locate_target.get("hitLevel") or "").strip() or "unknown"
    claim_type = str(locate_target.get("claimType") or "").strip() or "unknown"
    return RunResult(has_locate=has_locate, hit_level=hit_level, tag=tag, claim_type=claim_type)


def _summarize(results: list[RunResult]) -> dict[str, Any]:
    by_tag: dict[str, dict[str, Any]] = {}
    for tag in sorted({r.tag for r in results}):
        group = [r for r in results if r.tag == tag]
        by_tag[tag] = {
            "n": len(group),
            "missing_locate": sum(1 for r in group if not r.has_locate),
            "hit_level": dict(Counter(r.hit_level for r in group)),
            "claim_type": dict(Counter(r.claim_type for r in group)),
        }
    overall = {
        "n": len(results),
        "missing_locate": sum(1 for r in results if not r.has_locate),
        "hit_level": dict(Counter(r.hit_level for r in results)),
        "tag": dict(Counter(r.tag for r in results)),
    }
    return {"overall": overall, "by_tag": by_tag}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare paper_guide render_packet locate behavior across two KB_DB_DIR roots.")
    ap.add_argument("--pool", required=True, help="Input jsonl pool file (cases from collect_paper_guide_replay_cases.py).")
    ap.add_argument("--new-db-dir", required=True, help="New KB_DB_DIR root to compare against default/old.")
    ap.add_argument("--out", default="", help="Optional output json path for summary.")
    args = ap.parse_args(argv)

    pool_path = Path(str(args.pool)).expanduser()
    new_db_dir = Path(str(args.new_db_dir)).expanduser().resolve()
    cases = _read_jsonl(pool_path)

    old_results: list[RunResult] = []
    new_results: list[RunResult] = []

    os.environ.pop("KB_DB_DIR", None)
    for c in cases:
        old_results.append(_run_case(c))

    os.environ["KB_DB_DIR"] = str(new_db_dir)
    for c in cases:
        mapped = _remap_case_paths(c, new_db_root=new_db_dir)
        new_results.append(_run_case(mapped))

    old_summary = _summarize(old_results)
    new_summary = _summarize(new_results)

    # Diff view: only show tags present in pool.
    print("OLD overall:", old_summary["overall"])
    print("NEW overall:", new_summary["overall"])
    for tag in sorted(old_summary["by_tag"].keys()):
        print(f"TAG={tag}")
        print("  OLD", old_summary["by_tag"][tag])
        print("  NEW", new_summary["by_tag"].get(tag))

    if str(args.out or "").strip():
        out_path = Path(str(args.out)).expanduser()
        out_path.write_text(
            json.dumps({"old": old_summary, "new": new_summary}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("wrote", str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
