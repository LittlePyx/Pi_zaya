from __future__ import annotations

import hashlib
import json
from pathlib import Path


def compute_file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_doc_id(path: Path) -> str:
    # Stable ID based on absolute path.
    s = str(path.resolve()).encode("utf-8", errors="ignore")
    return hashlib.sha1(s).hexdigest()[:16]


def _docs_index_path(db_dir: Path) -> Path:
    return db_dir / "docs.json"


def _chunks_dir(db_dir: Path) -> Path:
    return db_dir / "chunks"


def doc_chunks_path(db_dir: Path, doc_id: str) -> Path:
    return _chunks_dir(db_dir) / f"{doc_id}.jsonl"


def load_docs_index(db_dir: Path) -> dict:
    p = _docs_index_path(db_dir)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_docs_index(db_dir: Path, docs: dict) -> None:
    p = _docs_index_path(db_dir)
    p.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")


def write_doc_chunks(db_dir: Path, doc_id: str, chunks: list[dict]) -> None:
    d = _chunks_dir(db_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = doc_chunks_path(db_dir, doc_id)
    with p.open("w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            c = dict(c)
            c["id"] = f"{doc_id}:{i}"
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def delete_doc_chunks(db_dir: Path, doc_id: str) -> bool:
    p = doc_chunks_path(db_dir, doc_id)
    if not p.exists():
        return False
    p.unlink()
    return True


def _doc_index_is_ready(rec: dict | None) -> bool:
    if not isinstance(rec, dict):
        return False
    status = str(rec.get("index_status") or "").strip().lower()
    if status and status != "ready":
        return False
    if not status and int(rec.get("num_chunks") or 0) <= 0:
        return False
    if status == "ready" and int(rec.get("num_chunks") or 0) <= 0:
        return False
    return True


def load_all_chunks(db_dir: Path, *, include_non_ready: bool = False) -> list[dict]:
    chunks: list[dict] = []
    d = _chunks_dir(db_dir)
    if not d.exists():
        return chunks
    docs_index = load_docs_index(db_dir)
    enforce_index = bool(docs_index) and not bool(include_non_ready)
    for p in sorted(d.glob("*.jsonl")):
        if enforce_index:
            doc_id = p.stem
            rec = docs_index.get(doc_id)
            if not _doc_index_is_ready(rec):
                continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunks.append(json.loads(line))
    return chunks


def prune_missing_docs(db_dir: Path, docs_index: dict) -> int:
    removed = 0
    to_delete: list[str] = []
    for doc_id, rec in docs_index.items():
        path = Path(rec.get("path", ""))
        if not path.exists():
            to_delete.append(doc_id)

    for doc_id in to_delete:
        docs_index.pop(doc_id, None)
        p = doc_chunks_path(db_dir, doc_id)
        if p.exists():
            p.unlink()
        removed += 1
    return removed
