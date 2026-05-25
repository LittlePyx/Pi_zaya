from __future__ import annotations

import argparse
from pathlib import Path

from kb.chunking import chunk_markdown
from kb.converter.structured_index_batch import rebuild_structured_indices_for_root
from kb.store import (
    compute_doc_id,
    compute_file_sha1,
    load_docs_index,
    prune_missing_docs,
    save_docs_index,
    write_doc_chunks,
)


def _iter_md_files(src: Path, glob: str, exclude_dirs: set[str], exclude_names: set[str]) -> list[Path]:
    if src.is_file():
        return [src]

    files: list[Path] = []
    for p in src.rglob(glob):
        if not p.is_file():
            continue
        if p.name in exclude_names:
            continue
        # Skip any path that contains excluded directory names (e.g. temp page dumps)
        if any(part in exclude_dirs for part in p.parts):
            continue
        files.append(p)

    return sorted(files)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest markdown files into a lightweight KB (BM25).")
    ap.add_argument("--src", required=True, help="Source markdown file or directory.")
    ap.add_argument("--db", required=True, help="DB directory (will be created).")
    ap.add_argument("--glob", default="*.md", help="Glob pattern when --src is a directory. Default: *.md")
    ap.add_argument("--exclude-dir", action="append", default=["temp"], help="Exclude directory name. Can be repeated. Default: temp")
    ap.add_argument(
        "--exclude-name",
        action="append",
        default=["assets_manifest.md"],
        help="Exclude filename. Can be repeated. Default: assets_manifest.md",
    )
    ap.add_argument("--incremental", action="store_true", help="Skip unchanged docs (by sha1).")
    ap.add_argument("--prune", action="store_true", help="Remove docs from DB if source file is missing.")
    ap.add_argument("--chunk-size", type=int, default=1400, help="Chunk size in characters. Default: 1400")
    ap.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in characters. Default: 200")
    ap.add_argument(
        "--rebuild-structured-indices",
        action="store_true",
        help="Rebuild per-document structured indices under each markdown assets/ directory.",
    )
    ap.add_argument(
        "--force-structured-indices",
        action="store_true",
        help="Rebuild structured indices even when the current index version is already present.",
    )
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    db_dir = Path(args.db).expanduser().resolve()
    db_dir.mkdir(parents=True, exist_ok=True)

    docs_index = load_docs_index(db_dir)
    md_files = _iter_md_files(src, args.glob, set(args.exclude_dir), set(args.exclude_name))
    if not md_files:
        raise SystemExit(f"No markdown files found under: {src}")

    structured_stats: dict | None = None
    if bool(args.rebuild_structured_indices):
        structured_stats = rebuild_structured_indices_for_root(
            src,
            glob=str(args.glob or "*.md"),
            exclude_dirs=set(args.exclude_dir),
            exclude_names=set(args.exclude_name),
            force=bool(args.force_structured_indices),
        )

    changed = 0
    skipped = 0
    total_chunks = 0

    for p in md_files:
        sha1 = compute_file_sha1(p)
        doc_id = compute_doc_id(p)
        prev = docs_index.get(doc_id)

        if args.incremental and prev and prev.get("sha1") == sha1:
            skipped += 1
            continue

        text = p.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_markdown(
            text,
            source_path=str(p),
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
        )

        write_doc_chunks(db_dir, doc_id, chunks)

        docs_index[doc_id] = {
            "doc_id": doc_id,
            "path": str(p),
            "sha1": sha1,
            "mtime": p.stat().st_mtime,
            "num_chunks": len(chunks),
        }

        changed += 1
        total_chunks += len(chunks)

    if args.prune:
        removed = prune_missing_docs(db_dir, docs_index)
    else:
        removed = 0

    save_docs_index(db_dir, docs_index)

    print(f"Docs: {len(md_files)} | updated: {changed} | skipped: {skipped} | removed: {removed}")
    if structured_stats is not None:
        print(
            "Structured indices: "
            f"scanned={int(structured_stats.get('scanned') or 0)} "
            f"rebuilt={int(structured_stats.get('rebuilt') or 0)} "
            f"skipped={int(structured_stats.get('skipped') or 0)} "
            f"failed={int(structured_stats.get('failed') or 0)} "
            f"citation_mentions={int(structured_stats.get('citation_mention_count') or 0)}"
        )
    if changed:
        print(f"New/updated chunks written: {total_chunks}")
    print(f"DB: {db_dir}")


if __name__ == "__main__":
    main()
