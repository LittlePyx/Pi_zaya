from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from kb.store import compute_doc_id, doc_chunks_path, load_all_chunks, load_docs_index, save_docs_index, write_doc_chunks


def _good_markdown() -> str:
    return "\n".join(
        [
            "<!-- kb_page: 1 -->",
            "",
            "# Good Paper",
            "",
            "## Abstract",
            "",
            "This paper has a usable abstract and cites prior work [1].",
            "",
            "## Method",
            "",
            "The method section is clear enough for retrieval chunks.",
            "",
            "## References",
            "",
            "[1] Ada Lovelace. Example reference. Journal of Testing, 2024.",
        ]
    )


def test_ingest_quality_gate_blocks_bad_markdown_before_chunks(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    src = tmp_path / "src"
    db_dir = tmp_path / "db"
    good_dir = src / "good"
    bad_dir = src / "bad"
    good_dir.mkdir(parents=True)
    bad_dir.mkdir(parents=True)
    good_md = good_dir / "good.en.md"
    bad_md = bad_dir / "bad.en.md"
    quality_report = good_dir / "quality_report.md"
    legacy_output = good_dir / "output.md"
    good_md.write_text(_good_markdown(), encoding="utf-8")
    bad_md.write_text("# Bad Paper\n\n![missing](assets/missing.png)\n\n\\u951b\n", encoding="utf-8")
    quality_report.write_text("# Markdown Quality Analysis Report\n\nThis is not a paper.", encoding="utf-8")
    legacy_output.write_text("# Legacy output\n\nThis duplicate should not be indexed.", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "ingest.py", "--src", str(src), "--db", str(db_dir)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    docs = load_docs_index(db_dir)
    good_id = compute_doc_id(good_md)
    bad_id = compute_doc_id(bad_md)
    quality_report_id = compute_doc_id(quality_report)
    legacy_output_id = compute_doc_id(legacy_output)
    assert docs[good_id]["index_status"] == "ready"
    assert docs[bad_id]["index_status"] == "quality_blocked"
    assert quality_report_id not in docs
    assert legacy_output_id not in docs
    assert doc_chunks_path(db_dir, good_id).exists()
    assert not doc_chunks_path(db_dir, bad_id).exists()
    first_chunk = json.loads(doc_chunks_path(db_dir, good_id).read_text(encoding="utf-8").splitlines()[0])
    assert first_chunk["meta"]["conversion_quality_status"] == "ready"
    write_doc_chunks(db_dir, bad_id, [{"text": "stale blocked chunk", "meta": {"source_path": str(bad_md)}}])
    loaded = load_all_chunks(db_dir)
    assert all(chunk["meta"]["source_path"] != str(bad_md) for chunk in loaded)
    assert any(chunk["meta"]["source_path"] == str(good_md) for chunk in loaded)
    assert "quality_blocked: 1" in proc.stdout


def test_incremental_ingest_rebuilds_unchanged_doc_when_chunk_artifact_is_missing(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    src = tmp_path / "src"
    db_dir = tmp_path / "db"
    src.mkdir(parents=True)
    md_path = src / "paper.en.md"
    md_path.write_text(_good_markdown(), encoding="utf-8")

    first = subprocess.run(
        [sys.executable, "ingest.py", "--src", str(src), "--db", str(db_dir)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert first.returncode == 0, first.stderr

    doc_id = compute_doc_id(md_path)
    docs = load_docs_index(db_dir)
    docs[doc_id]["num_chunks"] = 0
    save_docs_index(db_dir, docs)
    doc_chunks_path(db_dir, doc_id).unlink()

    second = subprocess.run(
        [sys.executable, "ingest.py", "--src", str(src), "--db", str(db_dir), "--incremental"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert second.returncode == 0, second.stderr
    rebuilt = load_docs_index(db_dir)[doc_id]
    assert int(rebuilt["num_chunks"]) > 0
    assert doc_chunks_path(db_dir, doc_id).stat().st_size > 0
    assert "updated: 1" in second.stdout
    assert "skipped: 0" in second.stdout

    # A non-empty but truncated/corrupt JSONL artifact must also be rebuilt.
    doc_chunks_path(db_dir, doc_id).write_text('{"text": "truncated"', encoding="utf-8")
    third = subprocess.run(
        [sys.executable, "ingest.py", "--src", str(src), "--db", str(db_dir), "--incremental"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert third.returncode == 0, third.stderr
    assert "updated: 1" in third.stdout
    assert "skipped: 0" in third.stdout
    rebuilt_lines = doc_chunks_path(db_dir, doc_id).read_text(encoding="utf-8").splitlines()
    assert len(rebuilt_lines) == int(load_docs_index(db_dir)[doc_id]["num_chunks"])
    assert all(isinstance(json.loads(line), dict) for line in rebuilt_lines)


def test_ingest_prune_removes_existing_nonpaper_artifact_docs(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    src = tmp_path / "src"
    db_dir = tmp_path / "db"
    paper_dir = src / "paper"
    paper_dir.mkdir(parents=True)
    db_dir.mkdir(parents=True)
    good_md = paper_dir / "paper.en.md"
    quality_report = paper_dir / "quality_report.md"
    legacy_output = paper_dir / "output.md"
    good_md.write_text(_good_markdown(), encoding="utf-8")
    quality_report.write_text("# Markdown Quality Analysis Report\n\nThis is not a paper.", encoding="utf-8")
    legacy_output.write_text("# Legacy output\n\nThis duplicate should not be indexed.", encoding="utf-8")

    quality_report_id = compute_doc_id(quality_report)
    legacy_output_id = compute_doc_id(legacy_output)
    save_docs_index(
        db_dir,
        {
            quality_report_id: {
                "doc_id": quality_report_id,
                "path": str(quality_report),
                "num_chunks": 1,
                "index_status": "ready",
            },
            legacy_output_id: {
                "doc_id": legacy_output_id,
                "path": str(legacy_output),
                "num_chunks": 1,
                "index_status": "ready",
            },
        },
    )
    write_doc_chunks(db_dir, quality_report_id, [{"text": "stale quality report", "meta": {"source_path": str(quality_report)}}])
    write_doc_chunks(db_dir, legacy_output_id, [{"text": "stale output", "meta": {"source_path": str(legacy_output)}}])

    proc = subprocess.run(
        [sys.executable, "ingest.py", "--src", str(src), "--db", str(db_dir), "--prune"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    docs = load_docs_index(db_dir)
    assert compute_doc_id(good_md) in docs
    assert quality_report_id not in docs
    assert legacy_output_id not in docs
    assert not doc_chunks_path(db_dir, quality_report_id).exists()
    assert not doc_chunks_path(db_dir, legacy_output_id).exists()
    assert "removed: 2" in proc.stdout
