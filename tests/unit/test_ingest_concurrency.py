from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from kb.store import compute_doc_id, db_write_lock, doc_chunks_path, load_docs_index


def _markdown(title: str) -> str:
    return f"# {title}\n\n## Abstract\n\n{title} provides enough content for a retrieval chunk.\n"


def test_concurrent_ingest_processes_preserve_both_documents(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    db_dir = tmp_path / "db"
    first_md = tmp_path / "first.md"
    second_md = tmp_path / "second.md"
    first_md.write_text(_markdown("First paper"), encoding="utf-8")
    second_md.write_text(_markdown("Second paper"), encoding="utf-8")

    processes: list[subprocess.Popen] = []
    try:
        with db_write_lock(db_dir, timeout_s=10):
            processes = [
                subprocess.Popen(
                    [
                        sys.executable,
                        "ingest.py",
                        "--src",
                        str(md_path),
                        "--db",
                        str(db_dir),
                        "--no-quality-gate",
                        "--lock-timeout-s",
                        "10",
                    ],
                    cwd=repo_root,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                for md_path in (first_md, second_md)
            ]
            for process in processes:
                assert process.poll() is None

        results = [process.communicate(timeout=20) for process in processes]
        for process, (stdout, stderr) in zip(processes, results):
            assert process.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)

    docs = load_docs_index(db_dir)
    expected = {compute_doc_id(first_md), compute_doc_id(second_md)}
    assert set(docs) == expected
    for md_path in (first_md, second_md):
        doc_id = compute_doc_id(md_path)
        rows = [json.loads(line) for line in doc_chunks_path(db_dir, doc_id).read_text(encoding="utf-8").splitlines()]
        assert len(rows) == int(docs[doc_id]["num_chunks"])
        assert [row["id"] for row in rows] == [f"{doc_id}:{index}" for index in range(len(rows))]
        assert all(row["meta"]["source_path"] == str(md_path) for row in rows)
