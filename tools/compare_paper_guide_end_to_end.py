from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from kb.config import load_settings
    from kb.store import load_all_chunks
    from kb.retriever import BM25Retriever
    from kb.retrieval_engine import _search_hits_with_fallback
    from kb.paper_guide_provenance import _build_paper_guide_answer_provenance
    from kb.paper_guide_focus import _extract_caption_panel_letters
    from api.chat_render import enrich_messages_with_reference_render
except ModuleNotFoundError:  # pragma: no cover
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from kb.config import load_settings  # type: ignore
    from kb.store import load_all_chunks  # type: ignore
    from kb.retriever import BM25Retriever  # type: ignore
    from kb.retrieval_engine import _search_hits_with_fallback  # type: ignore
    from kb.paper_guide_provenance import _build_paper_guide_answer_provenance  # type: ignore
    from kb.paper_guide_focus import _extract_caption_panel_letters  # type: ignore
    from api.chat_render import enrich_messages_with_reference_render  # type: ignore

import re

_FIG_RE = re.compile(r"(?:figure|fig\\.)\\s*(\\d{1,3})", flags=re.I)


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
        out["provenance"] = prov2
    hits2: list[dict[str, Any]] = []
    for h in list(out.get("hits") or []):
        if not isinstance(h, dict):
            continue
        h2 = dict(h)
        meta = h2.get("meta")
        if isinstance(meta, dict):
            meta2 = dict(meta)
            meta2["source_path"] = _swap_db_root(str(meta2.get("source_path") or ""), new_db_root)
            h2["meta"] = meta2
        hits2.append(h2)
    out["hits"] = hits2
    return out


@dataclass
class CaseResult:
    case_id: str
    tag: str
    has_locate: bool
    hit_level: str
    claim_type: str
    source_path: str
    must_locate: int
    must_fallback: int
    requested_figure: int
    requested_panels: list[str]
    target_match: bool


def _summarize(results: list[CaseResult]) -> dict[str, Any]:
    return {
        "n": len(results),
        "missing_locate": sum(1 for r in results if not r.has_locate),
        "tag": dict(Counter(r.tag for r in results)),
        "hit_level": dict(Counter(r.hit_level for r in results)),
        "claim_type": dict(Counter(r.claim_type for r in results)),
        "must_locate": int(sum(r.must_locate for r in results)),
        "must_fallback": int(sum(r.must_fallback for r in results)),
        "target_miss": int(sum(1 for r in results if (r.requested_figure > 0 or r.requested_panels) and (not r.target_match))),
    }


def _retrieve_hits(prompt: str, *, db_dir: Path, top_k: int) -> list[dict[str, Any]]:
    settings = load_settings()
    chunks = load_all_chunks(db_dir)
    retriever = BM25Retriever(chunks)
    hits_raw, _scores, _used_query, _used_translation = _search_hits_with_fallback(
        prompt,
        retriever,
        top_k=top_k,
        settings=settings,
    )
    hits: list[dict[str, Any]] = []
    for h in list(hits_raw or []):
        if isinstance(h, dict):
            hits.append(h)
    return hits


def _render_packet(prompt: str, *, answer: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    bound_source_path = ""
    bound_source_name = ""
    if hits:
        meta0 = hits[0].get("meta") if isinstance(hits[0].get("meta"), dict) else {}
        bound_source_path = str((meta0 or {}).get("source_path") or "").strip()
        bound_source_name = Path(bound_source_path).name if bound_source_path else ""

    provenance = _build_paper_guide_answer_provenance(
        answer=answer,
        answer_hits=hits,
        bound_source_path=bound_source_path,
        bound_source_name=bound_source_name,
        db_dir=None,
        llm_rerank=False,
    )

    messages = [
        {"id": 1, "role": "user", "content": prompt or "test"},
        {
            "id": 2,
            "role": "assistant",
            "content": answer or "answer",
            "provenance": provenance,
            "meta": {"paper_guide_contracts": {"version": 1, "intent": {"family": ""}}},
        },
    ]
    refs_by_user = {1: {"hits": hits}}
    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-e2e", render_packet_only=True)
    return (((rendered[-1].get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})


def _eval_pool(pool: list[dict[str, Any]], *, db_dir: Path, top_k: int) -> list[CaseResult]:
    os.environ["KB_DB_DIR"] = str(db_dir)
    out: list[CaseResult] = []
    for case in pool:
        case_id = str(case.get("id") or "").strip() or "case"
        prompt = str(case.get("prompt") or "").strip()
        answer = str(case.get("assistant_content") or "").strip() or "answer"
        tag = str(case.get("tag") or "").strip() or "unknown"
        m = _FIG_RE.search(prompt or "")
        requested_figure = int(m.group(1)) if m else 0
        requested_panels = sorted(_extract_caption_panel_letters(prompt or ""))

        hits = _retrieve_hits(prompt, db_dir=db_dir, top_k=top_k)
        packet = _render_packet(prompt, answer=answer, hits=hits)
        reader_open = packet.get("reader_open") or {}
        locate_target = packet.get("locate_target") or (reader_open.get("locateTarget") or {})
        has_locate = bool(str(reader_open.get("blockId") or "").strip() and str(reader_open.get("anchorId") or "").strip())
        hit_level = str(locate_target.get("hitLevel") or "").strip() or "unknown"
        claim_type = str(locate_target.get("claimType") or "").strip() or "unknown"
        src = str(reader_open.get("sourcePath") or "").strip()
        anchor_number = int(locate_target.get("anchorNumber") or reader_open.get("anchorNumber") or 0)
        target_match = True
        if requested_figure > 0:
            target_match = anchor_number == requested_figure
            if not target_match:
                for alt in list(reader_open.get("alternatives") or []):
                    if not isinstance(alt, dict):
                        continue
                    try:
                        if int(alt.get("anchorNumber") or 0) == requested_figure:
                            target_match = True
                            break
                    except Exception:
                        continue
        # Compute "must-locate fallback" on provenance segments (same signal used by replay collector).
        # We intentionally compute this on the runtime-built provenance (from retrieval hits),
        # not the captured provenance in the pool file.
        bound_source_path = ""
        bound_source_name = ""
        if hits:
            meta0 = hits[0].get("meta") if isinstance(hits[0].get("meta"), dict) else {}
            bound_source_path = str((meta0 or {}).get("source_path") or "").strip()
            bound_source_name = Path(bound_source_path).name if bound_source_path else ""
        provenance = _build_paper_guide_answer_provenance(
            answer=answer,
            answer_hits=hits,
            bound_source_path=bound_source_path,
            bound_source_name=bound_source_name,
            db_dir=None,
            llm_rerank=False,
        )
        must_locate = 0
        must_fallback = 0
        if isinstance(provenance, dict):
            for seg in list(provenance.get("segments") or []):
                if not isinstance(seg, dict):
                    continue
                must = bool(seg.get("must_locate")) or str(seg.get("locate_policy") or "").strip().lower() == "required"
                if not must:
                    continue
                must_locate += 1
                hl = str(seg.get("hit_level") or "").strip().lower()
                if hl and hl != "exact":
                    must_fallback += 1
        out.append(
            CaseResult(
                case_id=case_id,
                tag=tag,
                has_locate=has_locate,
                hit_level=hit_level,
                claim_type=claim_type,
                source_path=src,
                must_locate=must_locate,
                must_fallback=must_fallback,
                requested_figure=requested_figure,
                requested_panels=list(requested_panels),
                target_match=bool(target_match),
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="End-to-end compare paper_guide retrieval+provenance+render across DB dirs.")
    ap.add_argument("--pool", required=True, help="Input jsonl pool file (from collect_paper_guide_replay_cases.py).")
    ap.add_argument("--old-db-dir", default="db", help="Old KB_DB_DIR root (default: db).")
    ap.add_argument("--new-db-dir", required=True, help="New KB_DB_DIR root.")
    ap.add_argument("--top-k", type=int, default=6, help="Retrieval top_k.")
    ap.add_argument("--out", default="", help="Optional output json path for summary+per-case.")
    args = ap.parse_args(argv)

    pool_path = Path(str(args.pool)).expanduser()
    old_db_dir = Path(str(args.old_db_dir)).expanduser().resolve()
    new_db_dir = Path(str(args.new_db_dir)).expanduser().resolve()
    top_k = max(1, int(args.top_k))

    pool = _read_jsonl(pool_path)
    pool_new = [_remap_case_paths(c, new_db_root=new_db_dir) for c in pool]

    old_results = _eval_pool(pool, db_dir=old_db_dir, top_k=top_k)
    new_results = _eval_pool(pool_new, db_dir=new_db_dir, top_k=top_k)

    old_summary = _summarize(old_results)
    new_summary = _summarize(new_results)
    print("OLD", str(old_db_dir), old_summary)
    print("NEW", str(new_db_dir), new_summary)

    if str(args.out or "").strip():
        out_path = Path(str(args.out)).expanduser()
        out_payload = {
            "old_db_dir": str(old_db_dir),
            "new_db_dir": str(new_db_dir),
            "top_k": top_k,
            "old": old_summary,
            "new": new_summary,
            "cases": {
                "old": [r.__dict__ for r in old_results],
                "new": [r.__dict__ for r in new_results],
            },
        }
        out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
