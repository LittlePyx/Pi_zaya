from __future__ import annotations

from pathlib import Path

from tools.evidence_matrix import run_project_research_journey_eval as journey_eval


def test_source_records_reuses_canonical_indexed_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_root = tmp_path / "copied_db"
    copied_source = db_root / "paper" / "source.en.md"
    copied_source.parent.mkdir(parents=True)
    copied_source.write_text("copied evidence", encoding="utf-8")

    canonical_source = tmp_path / "canonical" / "paper" / "source.en.md"
    canonical_source.parent.mkdir(parents=True)
    canonical_source.write_text("canonical evidence", encoding="utf-8")
    monkeypatch.setattr(
        journey_eval,
        "load_all_chunks",
        lambda _root: [
            {
                "text": "canonical evidence",
                "meta": {"source_path": str(canonical_source)},
            }
        ],
    )

    records = journey_eval._source_records(
        {"sources": {"paper": "db/paper/source.en.md"}},
        fixture_path=tmp_path / "docs" / "fixture.json",
        db_root=db_root,
    )

    assert records[0]["sourcePath"] == str(canonical_source.resolve())
    assert records[0]["libraryMatchPath"] == str(canonical_source.resolve())
