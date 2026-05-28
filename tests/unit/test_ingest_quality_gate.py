from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from kb.store import compute_doc_id, doc_chunks_path, load_docs_index


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
    good_md.write_text(_good_markdown(), encoding="utf-8")
    bad_md.write_text("# Bad Paper\n\n![missing](assets/missing.png)\n\n\\u951b\n", encoding="utf-8")

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
    assert docs[good_id]["index_status"] == "ready"
    assert docs[bad_id]["index_status"] == "quality_blocked"
    assert doc_chunks_path(db_dir, good_id).exists()
    assert not doc_chunks_path(db_dir, bad_id).exists()
    first_chunk = json.loads(doc_chunks_path(db_dir, good_id).read_text(encoding="utf-8").splitlines()[0])
    assert first_chunk["meta"]["conversion_quality_status"] == "ready"
    assert "quality_blocked: 1" in proc.stdout
