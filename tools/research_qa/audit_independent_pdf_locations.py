from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import unicodedata

import fitz


def _words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2", normalized)
    return re.findall(r"[a-z]{3,}", normalized.lower())


def audit(
    root: Path, fixture: Path, output: Path, report_path: Path | None = None
) -> dict:
    spec = json.loads(fixture.read_text(encoding="utf-8"))
    report_path = report_path or root / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for mode in ("bm25", "hybrid_local"):
        for row in report[mode]["cases"]:
            cards = row.get("assistant_message", {}).get("cite_details", [])
            card = next(
                (
                    c
                    for c in cards
                    if not c.get("is_inpaper")
                    and c.get("source_path")
                    and c.get("evidence_quote")
                ),
                None,
            )
            if not card:
                results.append(
                    {"mode": mode, "id": row["id"], "status": "no_local_quote"}
                )
                continue
            doc_id = Path(card["source_path"]).parent.name
            page = int(card.get("page_start") or 0)
            pdf_path = root / "pdfs" / (doc_id + ".pdf")
            quote = str(card.get("evidence_quote") or "")
            check = {
                "mode": mode,
                "id": row["id"],
                "doc_id": doc_id,
                "declared_pdf_page": page,
                "evidence_quote": quote,
                "pdf": str(pdf_path),
            }
            if not pdf_path.is_file():
                check["status"] = "pdf_missing"
                results.append(check)
                continue
            with fitz.open(pdf_path) as pdf:
                if not 1 <= page <= len(pdf):
                    check["status"] = "invalid_page"
                    results.append(check)
                    continue
                tokens = set(_words(quote))
                page_tokens = set(_words(pdf[page - 1].get_text()))
                check["quote_token_coverage"] = round(
                    len(tokens & page_tokens) / max(1, len(tokens)), 4
                )
                # Exact PDF text rectangles are only physical location evidence.
                # They do not establish that the answer is entailed by the quote.
                phrase = ""
                quads = []
                normalized_quote = unicodedata.normalize("NFKC", quote)
                for match in re.finditer(r"[A-Za-z][A-Za-z -]{25,}", normalized_quote):
                    words = match.group().split()
                    for start in range(max(0, len(words) - 5)):
                        candidate = " ".join(words[start : start + 6])
                        quads = pdf[page - 1].search_for(candidate, quads=True)
                        if quads:
                            phrase = candidate
                            break
                    if quads:
                        break
                check["exact_pdf_phrase"] = phrase
                check["highlight_rectangles"] = [list(quad.rect) for quad in quads]
                check["status"] = "exact_pdf_text" if quads else "no_exact_phrase"
                if quads:
                    page_copy = fitz.open()
                    page_copy.insert_pdf(pdf, from_page=page - 1, to_page=page - 1)
                    page_copy[0].add_highlight_annot(quads)
                    target = output / f"{mode}-{row['id']}-page-{page:02d}.pdf"
                    page_copy.save(target)
                    page_copy.close()
                    check["annotated_page_pdf"] = str(target)
            results.append(check)
    result = {
        "schema_version": 1,
        "fixture_id": spec["id"],
        "limits": "One local citation per case; exact physical PDF phrase/page checks, not entailment or human faithfulness. Annotated page extracts are audit artifacts, not product UI screenshots.",
        "results": results,
    }
    (output / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("F:/pz-independent-20261013"))
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("docs/research_qa_independent_denoising_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = audit(args.root, args.fixture, args.output, args.report)
    print(
        "Original PDF exact phrase/page checks:",
        sum(r["status"] == "exact_pdf_text" for r in result["results"]),
        "/",
        len(result["results"]),
    )
