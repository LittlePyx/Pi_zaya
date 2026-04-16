from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from kb.config import load_settings
    from kb.paper_guide_structured_index_runtime import load_paper_guide_figure_index
    from kb.paper_guide_focus import _extract_caption_panel_letters
    from kb.store import load_all_chunks
    from kb.retriever import BM25Retriever
    from kb.retrieval_engine import _search_hits_with_fallback
    from kb.paper_guide_provenance import _build_paper_guide_answer_provenance
    from api.chat_render import enrich_messages_with_reference_render
except ModuleNotFoundError:  # pragma: no cover
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from kb.config import load_settings  # type: ignore
    from kb.paper_guide_structured_index_runtime import load_paper_guide_figure_index  # type: ignore
    from kb.paper_guide_focus import _extract_caption_panel_letters  # type: ignore
    from kb.store import load_all_chunks  # type: ignore
    from kb.retriever import BM25Retriever  # type: ignore
    from kb.retrieval_engine import _search_hits_with_fallback  # type: ignore
    from kb.paper_guide_provenance import _build_paper_guide_answer_provenance  # type: ignore
    from api.chat_render import enrich_messages_with_reference_render  # type: ignore


_PANEL_RE = re.compile(r"\bpanel\s*\(?([a-z])\)?", flags=re.I)


@dataclass
class BenchCase:
    md_path: str
    figure_number: int
    panel: str
    caption: str
    prompt: str


def _iter_md_files(db_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in db_dir.rglob("*.en.md"):
        if not p.is_file():
            continue
        if any(part.lower() == "temp" for part in p.parts):
            continue
        out.append(p)
    out.sort(key=lambda x: str(x))
    return out


def _build_cases(db_dir: Path, *, limit: int) -> list[BenchCase]:
    cases: list[BenchCase] = []
    for md in _iter_md_files(db_dir):
        rows = load_paper_guide_figure_index(md)
        for r in list(rows or []):
            if not isinstance(r, dict):
                continue
            try:
                fig_no = int(r.get("paper_figure_number") or 0)
            except Exception:
                fig_no = 0
            caption = str(r.get("caption") or "").strip()
            if fig_no <= 0 or not caption:
                continue
            panels = sorted(_extract_caption_panel_letters(caption))
            if not panels:
                continue
            # create one case per letter (cap at 3 letters per figure to keep the suite small)
            for ch in panels[:3]:
                prompt = f"Explain Figure {fig_no} panel ({ch}) in simple terms. What does it show?"
                cases.append(BenchCase(md_path=str(md), figure_number=fig_no, panel=str(ch), caption=caption, prompt=prompt))
                if len(cases) >= limit:
                    return cases
    return cases


def _retrieve_hits(prompt: str, *, db_dir: Path, top_k: int) -> list[dict[str, Any]]:
    settings = load_settings()
    chunks = load_all_chunks(db_dir)
    retriever = BM25Retriever(chunks)
    hits_raw, _scores, _used_query, _used_translation = _search_hits_with_fallback(prompt, retriever, top_k=top_k, settings=settings)
    return [dict(h) for h in list(hits_raw or []) if isinstance(h, dict)]


def _run_one(case: BenchCase, *, db_dir: Path, top_k: int) -> dict[str, Any]:
    # Force doc scoping: prefix with the source filename and filter hits to that md_path.
    source_hint = Path(case.md_path).name
    scoped_prompt = f"{source_hint} {case.prompt}".strip()
    hits = _retrieve_hits(scoped_prompt, db_dir=db_dir, top_k=top_k)
    hits = [
        h
        for h in hits
        if str((h.get("meta") or {}).get("source_path") or "").strip() == str(case.md_path)
    ]
    bound_source_path = ""
    bound_source_name = ""
    if hits:
        meta0 = hits[0].get("meta") if isinstance(hits[0].get("meta"), dict) else {}
        bound_source_path = str((meta0 or {}).get("source_path") or "").strip()
        bound_source_name = Path(bound_source_path).name if bound_source_path else ""
    # Use a simple synthetic answer containing the panel reference so provenance tries to bind to a figure.
    answer = f"Figure {case.figure_number} panel ({case.panel}) shows: (see caption and the paper)."
    provenance = _build_paper_guide_answer_provenance(
        answer=answer,
        answer_hits=hits,
        bound_source_path=bound_source_path,
        bound_source_name=bound_source_name,
        db_dir=None,
        llm_rerank=False,
    )
    messages = [
        {"id": 1, "role": "user", "content": case.prompt},
        {"id": 2, "role": "assistant", "content": answer, "provenance": provenance, "meta": {"paper_guide_contracts": {"version": 1, "intent": {"family": "figure_walkthrough"}}}},
    ]
    refs_by_user = {1: {"hits": hits}}
    packet = (((enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-panel", render_packet_only=True)[-1].get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    reader_open = packet.get("reader_open") or {}
    locate_target = packet.get("locate_target") or (reader_open.get("locateTarget") or {})
    alts = list(reader_open.get("visibleAlternatives") or reader_open.get("alternatives") or [])
    # Weak panel-satisfaction heuristic: locate target or a visible alternative should contain the panel marker.
    panel_ok = False
    lt_hay = str(locate_target.get("snippet") or "") + "\n" + str(locate_target.get("highlightSnippet") or "")
    if case.panel in _extract_caption_panel_letters(lt_hay):
        panel_ok = True
    for alt in alts:
        if not isinstance(alt, dict):
            continue
        hay = str(alt.get("snippet") or "") + "\n" + str(alt.get("highlightSnippet") or "")
        if case.panel in _extract_caption_panel_letters(hay):
            panel_ok = True
            break
    return {
        "md_path": case.md_path,
        "figure_number": case.figure_number,
        "panel": case.panel,
        "panel_ok": panel_ok,
        "has_locate": bool(str(reader_open.get("blockId") or "").strip()),
        "hit_level": str(locate_target.get("hitLevel") or "").strip() or "unknown",
        "source_path": str(reader_open.get("sourcePath") or "").strip(),
        "claim_type": str(locate_target.get("claimType") or "").strip() or "unknown",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Synthetic panel benchmark based on figure_index captions.")
    ap.add_argument("--db-dir", default="", help="KB_DB_DIR root (default: settings.db_dir)")
    ap.add_argument("--limit", type=int, default=60, help="Max synthetic panel cases")
    ap.add_argument("--top-k", type=int, default=8, help="Retrieval top_k")
    ap.add_argument("--out", default="", help="Optional output json path")
    args = ap.parse_args(argv)

    settings = load_settings()
    db_dir = Path(str(args.db_dir or settings.db_dir)).expanduser().resolve()
    os.environ["KB_DB_DIR"] = str(db_dir)

    cases = _build_cases(db_dir, limit=max(1, int(args.limit)))
    rows: list[dict[str, Any]] = []
    for c in cases:
        rows.append(_run_one(c, db_dir=db_dir, top_k=max(1, int(args.top_k))))

    c_panel_ok = Counter([bool(r.get("panel_ok")) for r in rows])
    c_hit = Counter([str(r.get("hit_level") or "") for r in rows])
    c_claim = Counter([str(r.get("claim_type") or "") for r in rows])
    print("n", len(rows))
    print("panel_ok", dict(c_panel_ok))
    print("hit_level", dict(c_hit))
    print("claim_type", dict(c_claim))

    if str(args.out or "").strip():
        out_path = Path(str(args.out)).expanduser().resolve()
        out_path.write_text(json.dumps({"db_dir": str(db_dir), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
