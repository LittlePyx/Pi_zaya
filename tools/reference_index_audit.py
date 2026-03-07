from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_index(index_path: Path) -> dict:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    docs = data.get("docs")
    if not isinstance(docs, dict):
        return {}
    return data


def _doc_stats(doc: dict) -> dict[str, int]:
    refs_map = doc.get("refs") if isinstance(doc.get("refs"), dict) else {}
    refs = [r for r in refs_map.values() if isinstance(r, dict)]
    total = len(refs)
    with_doi = sum(1 for r in refs if str(r.get("doi") or "").strip())
    empty_title = sum(1 for r in refs if not str(r.get("title") or "").strip())
    blank_match = sum(1 for r in refs if not str(r.get("match_method") or "").strip())
    unresolved = sum(
        1 for r in refs if (not str(r.get("doi") or "").strip()) and (not bool(r.get("crossref_ok")))
    )
    missing_doi = total - with_doi
    # Higher score means the document needs repair with higher priority.
    repair_score = (missing_doi * 3) + (empty_title * 2) + blank_match + (unresolved * 2)
    return {
        "refs_total": total,
        "refs_with_doi": with_doi,
        "missing_doi": missing_doi,
        "empty_title": empty_title,
        "blank_match": blank_match,
        "unresolved": unresolved,
        "repair_score": repair_score,
    }


def _summary(docs: dict[str, dict]) -> dict[str, int]:
    s = {
        "docs": 0,
        "refs_total": 0,
        "refs_with_doi": 0,
        "missing_doi": 0,
        "empty_title": 0,
        "blank_match": 0,
        "unresolved": 0,
    }
    for doc in docs.values():
        if not isinstance(doc, dict):
            continue
        st = _doc_stats(doc)
        s["docs"] += 1
        s["refs_total"] += st["refs_total"]
        s["refs_with_doi"] += st["refs_with_doi"]
        s["missing_doi"] += st["missing_doi"]
        s["empty_title"] += st["empty_title"]
        s["blank_match"] += st["blank_match"]
        s["unresolved"] += st["unresolved"]
    return s


def _print_table(rows: list[dict], top_n: int) -> None:
    head = [
        "rank",
        "score",
        "refs",
        "missing_doi",
        "empty_title",
        "blank_match",
        "unresolved",
        "name",
    ]
    print("\t".join(head))
    for i, row in enumerate(rows[:top_n], start=1):
        print(
            "\t".join(
                [
                    str(i),
                    str(row["repair_score"]),
                    str(row["refs_total"]),
                    str(row["missing_doi"]),
                    str(row["empty_title"]),
                    str(row["blank_match"]),
                    str(row["unresolved"]),
                    str(row["name"]),
                ]
            )
        )


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Audit reference index quality and rank documents that need DOI/title repair."
    )
    parser.add_argument(
        "--index",
        type=str,
        default="db/references_index.json",
        help="Path to references_index.json",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top documents to print",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a TSV table",
    )
    args = parser.parse_args()

    index_path = Path(args.index).expanduser().resolve()
    if not index_path.exists():
        raise SystemExit(f"index file not found: {index_path}")

    data = _load_index(index_path)
    docs = data.get("docs") if isinstance(data.get("docs"), dict) else {}
    rows: list[dict] = []
    for _, doc in docs.items():
        if not isinstance(doc, dict):
            continue
        st = _doc_stats(doc)
        rows.append(
            {
                "name": str(doc.get("name") or ""),
                "path": str(doc.get("path") or ""),
                "crossref_enriched": bool(doc.get("crossref_enriched")),
                **st,
            }
        )
    rows.sort(
        key=lambda r: (
            int(r["repair_score"]),
            int(r["missing_doi"]),
            int(r["empty_title"]),
            int(r["blank_match"]),
        ),
        reverse=True,
    )

    payload = {
        "index": str(index_path),
        "summary": _summary(docs),
        "top": rows[: max(0, int(args.top))],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("summary:", payload["summary"])
        _print_table(rows, max(0, int(args.top)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
