from __future__ import annotations

from pathlib import Path

from kb import reference_index as ref_index


def _good_markdown() -> str:
    return "\n".join(
        [
            "<!-- kb_page: 1 -->",
            "",
            "# Indexed Paper",
            "",
            "## Abstract",
            "",
            "This paper cites a relevant prior work [1].",
            "",
            "## Method",
            "",
            "The method text is usable for downstream citation metadata.",
            "",
            "## References",
            "",
            "[1] Ada Lovelace. Example reference. Journal of Testing, 2024.",
        ]
    )


def test_reference_index_quality_gate_skips_blocked_sources(tmp_path: Path, monkeypatch):
    src = tmp_path / "src"
    db_dir = tmp_path / "db"
    good_dir = src / "good"
    bad_dir = src / "bad"
    good_dir.mkdir(parents=True)
    bad_dir.mkdir(parents=True)
    good_md = good_dir / "good.en.md"
    bad_md = bad_dir / "bad.en.md"
    good_md.write_text(_good_markdown(), encoding="utf-8")
    bad_md.write_text("# Bad\n\n![missing](assets/missing.png)\n", encoding="utf-8")

    monkeypatch.setattr(ref_index, "_crossref_preflight_ok", lambda **kwargs: False)
    monkeypatch.setattr(ref_index, "_iter_md_files", lambda *args, **kwargs: [good_md, bad_md])

    stats = ref_index.build_reference_index(
        src_root=src,
        db_dir=db_dir,
        incremental=False,
        enable_title_lookup=False,
        quality_gate=True,
    )

    data = ref_index.load_reference_index(db_dir)
    assert int(stats.get("docs_quality_blocked") or 0) == 1
    assert int(stats.get("docs_indexed") or 0) == 1
    assert len(data.get("docs") or {}) == 1
    doc = next(iter((data.get("docs") or {}).values()))
    assert doc["index_status"] == "ready"
    assert str(doc.get("path") or "").endswith("good.en.md")
