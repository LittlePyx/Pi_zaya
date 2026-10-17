from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kb.source_blocks import load_source_blocks_cached, normalize_match_text


def audit(report: dict) -> dict:
    rows = []
    for mode in ("bm25", "hybrid_local"):
        for case in report[mode]["cases"]:
            for index, card in enumerate(
                case.get("assistant_message", {}).get("cite_details", [])
            ):
                if card.get("is_inpaper") or not card.get("source_path"):
                    continue
                source = Path(card["source_path"])
                blocks = load_source_blocks_cached(source) if source.is_file() else []
                block = next(
                    (b for b in blocks if b["block_id"] == card.get("block_id")), {}
                )
                quote = str(card.get("evidence_quote") or "")
                fields = ("anchor_id", "heading_path", "page_start", "page_end")
                locator_ok = bool(
                    block and all(card.get(k) == block.get(k) for k in fields)
                )
                # Whole continuous quote, not independently matched fragments.
                # This is stricter than the original r52 ellipsis-split audit.
                contained = bool(
                    quote.strip()
                    and block
                    and normalize_match_text(quote)
                    in normalize_match_text(block["raw_text"])
                )
                evidence_section = next(
                    (
                        s
                        for s in card.get("card_view", {}).get("sections", [])
                        if s.get("id") == "evidence"
                    ),
                    {},
                )
                projection_ok = bool(
                    quote
                    and card.get("card_evidence") == quote
                    and evidence_section.get("text") == quote
                )
                rows.append(
                    {
                        "mode": mode,
                        "case": case["id"],
                        "card_index": index,
                        "anchor": card.get("anchor"),
                        "source_path": str(source),
                        "block_id": card.get("block_id"),
                        "anchor_id": card.get("anchor_id"),
                        "page": card.get("page_start"),
                        "quote": quote,
                        "locator_matches_source": locator_ok,
                        "continuous_quote_contained": contained,
                        "display_projections_equal": projection_ok,
                        "ok": locator_ok and contained and projection_ok,
                    }
                )
    return {
        "card_count": len(rows),
        "passed": sum(r["ok"] for r in rows),
        "results": rows,
        "limits": "Exact Markdown source occurrence and display contract only; no entailment or PDF conversion quality claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.report.read_text(encoding="utf8")))
    with args.output.open("x", encoding="utf8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"Source card excerpts: {result['passed']}/{result['card_count']}")
    return int(result["passed"] != result["card_count"])


if __name__ == "__main__":
    raise SystemExit(main())
