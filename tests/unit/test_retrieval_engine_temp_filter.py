from __future__ import annotations

import os
from pathlib import Path

from kb.retrieval_engine import _is_temp_source_path


def test_is_temp_source_path_allows_paths_under_kb_db_dir_even_if_tmp_named(tmp_path: Path, monkeypatch):
    db_dir = tmp_path / "tmp_reconvert_all_20260408"
    db_dir.mkdir(parents=True, exist_ok=True)
    md_path = db_dir / "Demo.en.md"
    md_path.write_text("# Demo\n", encoding="utf-8")

    monkeypatch.setenv("KB_DB_DIR", str(db_dir))
    assert _is_temp_source_path(str(md_path)) is False


def test_is_temp_source_path_still_filters_tmp_paths_outside_db_dir(tmp_path: Path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "tmp_other" / "Doc.en.md"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("# Doc\n", encoding="utf-8")

    monkeypatch.setenv("KB_DB_DIR", str(db_dir))
    assert _is_temp_source_path(str(outside)) is True

