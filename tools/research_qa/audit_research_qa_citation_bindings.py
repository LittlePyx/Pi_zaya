from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kb.citation_marker_context import numeric_marker_is_content
from kb.source_blocks import load_source_blocks


MARKER = re.compile(r'\[(\d{1,4})\](?:\(#(kb-cite-[^\s)]+)(?:\s+"[^"]*")?\))?')


def audit(report: dict, fixture: dict) -> dict:
    """Audit final display binding and physical source identity, not entailment."""
    scopes = {case["id"]: set(case["docIds"]) for case in fixture["cases"]}
    required = {
        case["id"]: set(case.get("expected", {}).get("requiredCitationDocIds", []))
        for case in fixture["cases"]
    }
    results = []
    for mode in ("bm25", "hybrid_local"):
        for case in report.get(mode, {}).get("cases", []):
            message = case.get("assistant_message") or {}
            packet = (message.get("meta", {}).get("paper_guide_contracts") or {}).get(
                "render_packet"
            ) or {}
            details = packet.get("cite_details", message.get("cite_details", []))
            anchors = {
                str(card.get("anchor") or ""): int(card.get("num") or 0)
                for card in details
            }
            numbers = set(anchors.values())
            orphan_markers = []
            for field in (
                "rendered_body",
                "rendered_content",
                "copy_markdown",
                "copy_text",
            ):
                body = str(packet.get(field) or message.get(field) or "")
                for match in MARKER.finditer(body):
                    if numeric_marker_is_content(body, match.start()):
                        continue
                    number, anchor = int(match[1]), match[2]
                    if not anchor and body[match.end() :].startswith("("):
                        continue
                    bound = (
                        anchors.get(anchor) == number
                        if anchor
                        else field.startswith("copy") and number in numbers
                    )
                    if not bound and (anchor or 1 <= number < 1000):
                        orphan_markers.append({"field": field, "marker": match[0]})
            cards = []
            for card in details:
                if card.get("is_inpaper") or not card.get("source_path"):
                    continue
                source = Path(card["source_path"])
                check = {
                    "source_path": str(source),
                    "block_id": card.get("block_id"),
                    "page_start": card.get("page_start"),
                }
                check["scope_ok"] = source.parent.name in scopes[case["id"]]
                try:
                    blocks = load_source_blocks(source)
                    block = next(
                        (
                            b
                            for b in blocks
                            if b.get("block_id") == card.get("block_id")
                        ),
                        None,
                    )
                    check["block_exists"] = (
                        bool(block) if card.get("block_id") else None
                    )
                    check["block_page_matches"] = (
                        bool(
                            block
                            and int(block.get("page_start") or 0) > 0
                            and int(block.get("page_start") or 0)
                            == int(card.get("page_start") or 0)
                        )
                        if card.get("block_id")
                        else None
                    )
                    check["locator_kind"] = (
                        "block" if card.get("block_id") else "phrase"
                    )
                except (OSError, ValueError, TypeError):
                    check.update(block_exists=False, block_page_matches=False)
                cards.append(check)
            results.append(
                {
                    "mode": mode,
                    "id": case["id"],
                    "orphan_markers": orphan_markers,
                    "local_cards": cards,
                    "missing_required_local_sources": sorted(
                        required[case["id"]]
                        - {Path(card["source_path"]).parent.name for card in cards}
                    ),
                }
            )
    return {
        "limits": "Final display/copy markers and local source/block/page/scope consistency only; does not measure answer completeness or entailment. In-paper bibliography cards are excluded from local-block checks.",
        "case_count": len(results),
        "orphan_marker_count": sum(len(r["orphan_markers"]) for r in results),
        "local_card_count": sum(len(r["local_cards"]) for r in results),
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("docs/research_qa_independent_denoising_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        json.loads(args.report.read_text(encoding="utf-8")),
        json.loads(args.fixture.read_text(encoding="utf-8")),
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Cases: {result['case_count']}; orphan markers: {result['orphan_marker_count']}; local cards: {result['local_card_count']}"
    )
