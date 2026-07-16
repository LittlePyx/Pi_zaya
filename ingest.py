from __future__ import annotations

import argparse
import json
from pathlib import Path

from kb.chunking import CHUNK_SCHEMA_VERSION, chunk_markdown
from kb.converter.quality_gate import index_quality_document_fields, prepare_markdown_for_index
from kb.converter.structured_index_batch import rebuild_structured_indices_for_root, structured_indices_need_rebuild
from kb.converter.structured_indices import STRUCTURED_INDEX_VERSION, rebuild_structured_indices_for_markdown
from kb.source_filters import is_excluded_source_path
from kb.store import (
    compute_doc_id,
    compute_file_sha1,
    db_write_lock,
    delete_doc_chunks,
    doc_chunks_path,
    load_docs_index,
    save_docs_index,
    write_doc_chunks,
)


def _incremental_chunks_are_usable(db_dir: Path, doc_id: str, record: dict | None) -> bool:
    if not isinstance(record, dict):
        return False
    if int(record.get("chunk_schema_version") or 0) != int(CHUNK_SCHEMA_VERSION):
        return False
    expected_count = int(record.get("num_chunks") or 0)
    if expected_count <= 0:
        return False
    path = doc_chunks_path(db_dir, doc_id)
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        actual_count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    return False
                chunk = json.loads(raw)
                if not isinstance(chunk, dict) or not str(chunk.get("text") or "").strip():
                    return False
                if int((chunk.get("meta") or {}).get("chunk_schema_version") or 0) != int(CHUNK_SCHEMA_VERSION):
                    return False
                actual_count += 1
        return actual_count == expected_count
    except Exception:
        return False


def _iter_md_files(src: Path, glob: str, exclude_dirs: set[str], exclude_names: set[str]) -> list[Path]:
    if src.is_file():
        return [src]

    files: list[Path] = []
    for p in src.rglob(glob):
        if not p.is_file():
            continue
        if p.name in exclude_names:
            continue
        if is_excluded_source_path(str(p)):
            continue
        # Skip any path that contains excluded directory names (e.g. temp page dumps)
        if any(part in exclude_dirs for part in p.parts):
            continue
        files.append(p)

    return sorted(files)


def _annotate_chunks_with_quality(chunks: list[dict], assessment: dict) -> list[dict]:
    status = str((assessment or {}).get("status") or "ready")
    action = str((assessment or {}).get("action") or "none")
    issue_codes = [str(item) for item in list((assessment or {}).get("issue_codes") or []) if str(item or "").strip()][:30]
    out: list[dict] = []
    for chunk in list(chunks or []):
        row = dict(chunk)
        meta = dict(row.get("meta") or {})
        meta["conversion_quality_status"] = status
        meta["conversion_quality_action"] = action
        if issue_codes:
            meta["conversion_quality_issue_codes"] = issue_codes
        row["meta"] = meta
        out.append(row)
    return out


def _doc_ids_to_prune(docs_index: dict) -> set[str]:
    to_delete: set[str] = set()
    for doc_id, rec in docs_index.items():
        if not isinstance(rec, dict):
            continue
        path = str(rec.get("path") or "")
        if (path and not Path(path).exists()) or (path and is_excluded_source_path(path)):
            to_delete.add(str(doc_id))
    return to_delete


def _empty_structured_stats() -> dict:
    return {
        "scanned": 0,
        "rebuilt": 0,
        "skipped": 0,
        "failed": 0,
        "citation_mention_count": 0,
        "errors": [],
        "version": int(STRUCTURED_INDEX_VERSION),
    }


def _rebuild_structured_for_markdown(md_path: Path, *, force: bool, stats: dict) -> None:
    stats["scanned"] = int(stats.get("scanned") or 0) + 1
    assets_dir = md_path.parent / "assets"
    if (not force) and (not structured_indices_need_rebuild(md_path, assets_dir=assets_dir)):
        stats["skipped"] = int(stats.get("skipped") or 0) + 1
        return
    try:
        out = rebuild_structured_indices_for_markdown(md_path, assets_dir=assets_dir)
        ref_payload = out.get("reference_index") if isinstance(out, dict) else {}
        try:
            stats["citation_mention_count"] = int(stats.get("citation_mention_count") or 0) + int(
                (ref_payload or {}).get("citation_mention_count") or 0
            )
        except Exception:
            pass
        stats["rebuilt"] = int(stats.get("rebuilt") or 0) + 1
    except Exception as exc:
        stats["failed"] = int(stats.get("failed") or 0) + 1
        errors = stats.get("errors")
        if not isinstance(errors, list):
            errors = []
            stats["errors"] = errors
        if len(errors) < 20:
            errors.append({"path": str(md_path), "error": str(exc)})


def _prepare_ingest_document(
    args: argparse.Namespace,
    *,
    db_dir: Path,
    path: Path,
    previous: dict | None,
    allow_incremental_shortcut: bool,
) -> dict:
    quality_assessment: dict = {
        "indexable": True,
        "status": "ready",
        "action": "none",
        "issue_codes": [],
    }
    if bool(args.quality_gate):
        quality_assessment = prepare_markdown_for_index(
            path,
            auto_repair=bool(args.quality_autofix),
            allow_blocked=bool(args.allow_blocked_quality),
        )

    doc_id = compute_doc_id(path)
    sha1 = compute_file_sha1(path)
    quality_fields = index_quality_document_fields(quality_assessment) if bool(args.quality_gate) else {}
    blocked = bool(args.quality_gate) and not bool(quality_assessment.get("indexable"))
    chunks: list[dict] | None = None
    if not blocked:
        incremental_ready = (
            allow_incremental_shortcut
            and bool(args.incremental)
            and bool(previous)
            and previous.get("sha1") == sha1
            and _incremental_chunks_are_usable(db_dir, doc_id, previous)
        )
        if not incremental_ready:
            text = path.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_markdown(
                text,
                source_path=str(path),
                chunk_size=args.chunk_size,
                overlap=args.chunk_overlap,
            )
            if bool(args.quality_gate):
                chunks = _annotate_chunks_with_quality(chunks, quality_assessment)

    record = {
        "doc_id": doc_id,
        "path": str(path),
        "sha1": sha1,
        "mtime": path.stat().st_mtime,
        "num_chunks": 0 if blocked else len(chunks or []),
        "chunk_schema_version": int(CHUNK_SCHEMA_VERSION),
        **quality_fields,
    }
    return {
        "path": path,
        "doc_id": doc_id,
        "sha1": sha1,
        "quality_fields": quality_fields,
        "blocked": blocked,
        "chunks": chunks,
        "record": record,
    }


def _prepare_ingest_documents(args: argparse.Namespace, *, db_dir: Path, md_files: list[Path]) -> list[dict]:
    snapshot = load_docs_index(db_dir)
    return [
        _prepare_ingest_document(
            args,
            db_dir=db_dir,
            path=path,
            previous=snapshot.get(compute_doc_id(path)),
            allow_incremental_shortcut=True,
        )
        for path in md_files
    ]


def _commit_prepared_ingest(
    args: argparse.Namespace,
    *,
    db_dir: Path,
    prepared_documents: list[dict],
) -> tuple[int, int, int, int, int, list[Path]]:
    """Merge prepared documents into the latest index while holding the writer lock."""

    docs_index = load_docs_index(db_dir)
    changed = 0
    skipped = 0
    quality_blocked = 0
    total_chunks = 0
    indexed_paths: list[Path] = []
    chunks_to_delete_after_commit: set[str] = set()

    for prepared in prepared_documents:
        item = prepared
        path = Path(item["path"])
        doc_id = str(item["doc_id"])
        previous = docs_index.get(doc_id)

        # Quality repair or a concurrent converter can change the source while
        # this process waits for the commit lock. Re-prepare only that rare case.
        if compute_file_sha1(path) != str(item["sha1"]):
            item = _prepare_ingest_document(
                args,
                db_dir=db_dir,
                path=path,
                previous=previous,
                allow_incremental_shortcut=True,
            )

        if bool(item["blocked"]):
            docs_index[doc_id] = dict(item["record"])
            chunks_to_delete_after_commit.add(doc_id)
            quality_blocked += 1
            continue

        if (
            args.incremental
            and previous
            and previous.get("sha1") == item["sha1"]
            and _incremental_chunks_are_usable(db_dir, doc_id, previous)
        ):
            quality_fields = dict(item["quality_fields"])
            if quality_fields:
                merged = dict(previous)
                merged.update(quality_fields)
                docs_index[doc_id] = merged
            indexed_paths.append(path)
            skipped += 1
            continue

        chunks = item.get("chunks")
        if chunks is None:
            item = _prepare_ingest_document(
                args,
                db_dir=db_dir,
                path=path,
                previous=None,
                allow_incremental_shortcut=False,
            )
            if bool(item["blocked"]):
                docs_index[doc_id] = dict(item["record"])
                chunks_to_delete_after_commit.add(doc_id)
                quality_blocked += 1
                continue
            chunks = item["chunks"]

        write_doc_chunks(db_dir, doc_id, chunks)
        docs_index[doc_id] = dict(item["record"])
        indexed_paths.append(path)
        changed += 1
        total_chunks += len(chunks)

    if args.prune:
        prune_doc_ids = _doc_ids_to_prune(docs_index)
        for doc_id in prune_doc_ids:
            docs_index.pop(doc_id, None)
        chunks_to_delete_after_commit.update(prune_doc_ids)
        removed = len(prune_doc_ids)
    else:
        removed = 0

    save_docs_index(db_dir, docs_index)
    for doc_id in sorted(chunks_to_delete_after_commit):
        delete_doc_chunks(db_dir, doc_id)
    return changed, skipped, quality_blocked, total_chunks, removed, indexed_paths


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
    ap.add_argument(
        "--no-quality-gate",
        action="store_false",
        dest="quality_gate",
        default=True,
        help="Disable conversion-quality gate before chunking/indexing.",
    )
    ap.add_argument(
        "--no-quality-autofix",
        action="store_false",
        dest="quality_autofix",
        default=True,
        help="Do not run safe deterministic Markdown repairs before indexing.",
    )
    ap.add_argument(
        "--allow-blocked-quality",
        action="store_true",
        help="Index documents even when the conversion-quality gate recommends reconversion or manual review.",
    )
    ap.add_argument(
        "--lock-timeout-s",
        type=float,
        default=None,
        help="Maximum seconds to wait for another knowledge-base writer. Default: KB_DB_WRITE_LOCK_TIMEOUT_S or 600.",
    )
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    db_dir = Path(args.db).expanduser().resolve()
    db_dir.mkdir(parents=True, exist_ok=True)

    md_files = _iter_md_files(src, args.glob, set(args.exclude_dir), set(args.exclude_name))
    if not md_files:
        raise SystemExit(f"No markdown files found under: {src}")

    structured_stats: dict | None = None
    defer_structured_rebuild = bool(args.rebuild_structured_indices) and bool(args.quality_gate)
    if bool(args.rebuild_structured_indices) and not defer_structured_rebuild:
        structured_stats = rebuild_structured_indices_for_root(
            src,
            glob=str(args.glob or "*.md"),
            exclude_dirs=set(args.exclude_dir),
            exclude_names=set(args.exclude_name),
            force=bool(args.force_structured_indices),
        )
    elif defer_structured_rebuild:
        structured_stats = _empty_structured_stats()

    prepared_documents = _prepare_ingest_documents(args, db_dir=db_dir, md_files=md_files)
    try:
        with db_write_lock(db_dir, timeout_s=args.lock_timeout_s):
            changed, skipped, quality_blocked, total_chunks, removed, deferred_structured_paths = (
                _commit_prepared_ingest(
                    args,
                    db_dir=db_dir,
                    prepared_documents=prepared_documents,
                )
            )
    except TimeoutError as exc:
        raise SystemExit(f"Knowledge-base index is busy: {exc}") from exc

    if defer_structured_rebuild and structured_stats is not None:
        for p in deferred_structured_paths:
            _rebuild_structured_for_markdown(
                p,
                force=bool(args.force_structured_indices),
                stats=structured_stats,
            )

    print(f"Docs: {len(md_files)} | updated: {changed} | skipped: {skipped} | quality_blocked: {quality_blocked} | removed: {removed}")
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
    if quality_blocked > 0 and changed <= 0 and skipped <= 0 and not bool(args.allow_blocked_quality):
        raise SystemExit(f"Quality gate blocked {quality_blocked} document(s); no indexable markdown was written.")


if __name__ == "__main__":
    main()
