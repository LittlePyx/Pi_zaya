from __future__ import annotations

import json
from pathlib import Path

from api import reference_ui


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_hydrate_regular_system_a_payload_from_local_source_cache_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = str(tmp_path / "Nature-2024-Cached source paper.en.md")
    doi = "10.1234/cached.source"
    _write_json(
        tmp_path / "references_index.json",
        {
            "docs": {
                source_path.lower(): {
                    "path": source_path,
                    "name": Path(source_path).name,
                    "stem": Path(source_path).stem.lower(),
                    "source_doi": doi,
                    "refs": {},
                }
            }
        },
    )
    _write_json(
        tmp_path / "crossref_cache.json",
        {
            "doi": {
                doi: {
                    "title": "Cached Source Paper",
                    "authors": "Ada Lovelace; Grace Hopper",
                    "venue": "Journal of Offline Metadata",
                    "year": "2024",
                    "volume": "7",
                    "issue": "2",
                    "pages": "10-19",
                    "doi": doi,
                    "citation_count": 18,
                    "metadata_repair_source": "internal-cache-detail",
                    "metadata_quality": {"score": 99},
                }
            }
        },
    )
    monkeypatch.setattr(
        reference_ui,
        "ensure_source_citation_meta",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("network enrichment must not run")),
    )

    out = reference_ui.hydrate_refs_payload_citation_meta(
        {
            "hits": [
                {
                    "meta": {"source_path": source_path},
                    "ui_meta": {
                        "display_name": Path(source_path).name,
                        "citation_meta": {"title": "Existing Public Title"},
                    },
                }
            ]
        },
        pdf_root=None,
        lib_store=None,
        db_dir=tmp_path,
    )

    citation_meta = out["hits"][0]["ui_meta"]["citation_meta"]
    assert citation_meta == {
        "title": "Existing Public Title",
        "authors": "Ada Lovelace; Grace Hopper",
        "venue": "Journal of Offline Metadata",
        "year": "2024",
        "volume": "7",
        "issue": "2",
        "pages": "10-19",
        "doi": doi,
        "doi_url": f"https://doi.org/{doi}",
        "citation_count": 18,
    }


def test_hydrate_regular_system_a_payload_matches_library_record_without_pdf_resolution(
    tmp_path: Path,
) -> None:
    class FakeLibraryStore:
        def list_citation_records(self) -> list[dict]:
            return [
                {
                    "path": str(tmp_path / "Library Paper.pdf"),
                    "citation_meta": {
                        "title": "Library Paper",
                        "authors": "Lin Chen",
                        "venue": "Applied Optics",
                        "year": "2022",
                        "doi": "10.5678/library.paper",
                        "metadata_quality": {"score": 88},
                    },
                }
            ]

    source_path = str(tmp_path / "moved" / "Library Paper.en.md")
    out = reference_ui.hydrate_refs_payload_citation_meta(
        {"hits": [{"meta": {"source_path": source_path}, "ui_meta": {}}]},
        pdf_root=None,
        lib_store=FakeLibraryStore(),
        db_dir=tmp_path,
    )

    citation_meta = out["hits"][0]["ui_meta"]["citation_meta"]
    assert citation_meta["title"] == "Library Paper"
    assert citation_meta["authors"] == "Lin Chen"
    assert citation_meta["venue"] == "Applied Optics"
    assert citation_meta["year"] == "2022"
    assert citation_meta["doi"] == "10.5678/library.paper"
    assert citation_meta["doi_url"] == "https://doi.org/10.5678/library.paper"
    assert "metadata_quality" not in citation_meta


def test_library_basename_fallback_rejects_duplicate_records_even_when_metadata_matches(
    tmp_path: Path,
) -> None:
    shared_meta = {
        "title": "Ambiguous Same-name Paper",
        "doi": "10.5678/ambiguous.same-name",
    }

    class FakeLibraryStore:
        def list_citation_records(self) -> list[dict]:
            return [
                {
                    "path": str(tmp_path / "collection-a" / "Repeated Paper.pdf"),
                    "citation_meta": dict(shared_meta),
                },
                {
                    "path": str(tmp_path / "collection-b" / "Repeated Paper.pdf"),
                    "citation_meta": dict(shared_meta),
                },
            ]

    source_path = str(tmp_path / "moved" / "Repeated Paper.en.md")
    out = reference_ui.hydrate_refs_payload_citation_meta(
        {"hits": [{"meta": {"source_path": source_path}, "ui_meta": {}}]},
        pdf_root=None,
        lib_store=FakeLibraryStore(),
        db_dir=tmp_path,
    )

    assert "citation_meta" not in out["hits"][0]["ui_meta"]


def test_library_basename_fallback_counts_namesake_without_metadata_as_ambiguous(
    tmp_path: Path,
) -> None:
    class FakeLibraryStore:
        def list_citation_records(self) -> list[dict]:
            return [
                {
                    "path": str(tmp_path / "collection-a" / "Repeated Paper.pdf"),
                    "citation_meta": {},
                },
                {
                    "path": str(tmp_path / "collection-b" / "Repeated Paper.pdf"),
                    "citation_meta": {
                        "title": "Wrong Same-name Paper",
                        "doi": "10.5678/wrong.only-annotated-namesake",
                    },
                },
            ]

    source_path = str(tmp_path / "moved" / "Repeated Paper.en.md")
    out = reference_ui.hydrate_refs_payload_citation_meta(
        {"hits": [{"meta": {"source_path": source_path}, "ui_meta": {}}]},
        pdf_root=None,
        lib_store=FakeLibraryStore(),
        db_dir=tmp_path,
    )

    assert "citation_meta" not in out["hits"][0]["ui_meta"]


def test_library_basename_fallback_rejects_conflicting_records_for_same_path(
    tmp_path: Path,
) -> None:
    repeated_path = str(tmp_path / "collection-a" / "Repeated Paper.pdf")

    class FakeLibraryStore:
        def list_citation_records(self) -> list[dict]:
            return [
                {
                    "path": repeated_path,
                    "citation_meta": {
                        "title": "Older Conflicting Title",
                        "doi": "10.5678/older.same-path",
                    },
                },
                {
                    "path": repeated_path,
                    "citation_meta": {
                        "title": "Newer Conflicting Title",
                        "doi": "10.5678/newer.same-path",
                    },
                },
            ]

    source_path = str(tmp_path / "moved" / "Repeated Paper.en.md")
    out = reference_ui.hydrate_refs_payload_citation_meta(
        {"hits": [{"meta": {"source_path": source_path}, "ui_meta": {}}]},
        pdf_root=None,
        lib_store=FakeLibraryStore(),
        db_dir=tmp_path,
    )

    assert "citation_meta" not in out["hits"][0]["ui_meta"]


def test_library_basename_fallback_uses_unique_matching_parent_path(
    tmp_path: Path,
) -> None:
    class FakeLibraryStore:
        def list_citation_records(self) -> list[dict]:
            return [
                {
                    "path": str(tmp_path / "collection-a" / "Repeated Paper.pdf"),
                    "citation_meta": {
                        "title": "Correct Same-name Paper",
                        "doi": "10.5678/correct.same-name",
                    },
                },
                {
                    "path": str(tmp_path / "collection-b" / "Repeated Paper.pdf"),
                    "citation_meta": {
                        "title": "Wrong Same-name Paper",
                        "doi": "10.5678/wrong.same-name",
                    },
                },
            ]

    source_path = str(tmp_path / "collection-a" / "Repeated Paper.en.md")
    out = reference_ui.hydrate_refs_payload_citation_meta(
        {"hits": [{"meta": {"source_path": source_path}, "ui_meta": {}}]},
        pdf_root=None,
        lib_store=FakeLibraryStore(),
        db_dir=tmp_path,
    )

    assert out["hits"][0]["ui_meta"]["citation_meta"] == {
        "title": "Correct Same-name Paper",
        "doi": "10.5678/correct.same-name",
        "doi_url": "https://doi.org/10.5678/correct.same-name",
    }


def test_hydrate_regular_system_a_payload_filters_exact_store_and_existing_ui_meta(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = str(tmp_path / "Exact Paper.en.md")
    pdf_path = tmp_path / "Exact Paper.pdf"

    class FakeLibraryStore:
        def get_citation_meta(self, requested_path: Path) -> dict:
            assert requested_path == pdf_path
            return {
                "title": "Stored Public Title",
                "authors": "Stored Author",
                "metadata_quality": {"score": 91},
                "match_score": 0.99,
                "_kb_doc_id": "private-doc-id",
            }

    monkeypatch.setattr(
        reference_ui,
        "_resolve_pdf_for_source",
        lambda _root, requested_source: pdf_path if requested_source == source_path else None,
    )
    out = reference_ui.hydrate_refs_payload_citation_meta(
        {
            "hits": [
                {
                    "meta": {"source_path": source_path},
                    "ui_meta": {
                        "citation_meta": {
                            "title": "Newer UI Title",
                            "year": "2026",
                            "metadata_quality": {"score": 72},
                            "match_score": 0.75,
                            "_kb_doc_id": "another-private-id",
                        }
                    },
                },
                {
                    "meta": {"source_path": str(tmp_path / "Unstored.en.md")},
                    "ui_meta": {
                        "citation_meta": {
                            "venue": "Public Venue",
                            "metadata_quality": {"score": 50},
                            "match_score": 0.5,
                            "_kb_doc_id": "private-without-store",
                        }
                    },
                },
            ]
        },
        pdf_root=tmp_path,
        lib_store=FakeLibraryStore(),
        db_dir=tmp_path,
    )

    exact_meta = out["hits"][0]["ui_meta"]["citation_meta"]
    assert exact_meta == {
        "title": "Newer UI Title",
        "authors": "Stored Author",
        "year": "2026",
    }
    assert out["hits"][1]["ui_meta"]["citation_meta"] == {"venue": "Public Venue"}


def test_public_refs_payload_projection_hides_internal_fields_and_nested_absolute_paths() -> None:
    source_path = r"F:\private\library\Paper.en.md"
    out = reference_ui.public_refs_payload_projection(
        {
            9: {
                "prompt_sig": "public-cache-version",
                "updated_at": 1784040000.0,
                "pipeline_debug": {"reranker": "internal"},
                "scores": [9.5],
                "render_error": "render_payload_empty",
                "render_error_detail": r"failed at F:\private\render\trace.json",
                "render_attempts": 3,
                "render_evidence_sig": "private-evidence-signature",
                "rendered_payload": {"instruction": "internal rendering prompt"},
                "hits": [
                    {
                        "meta": {
                            "source_path": source_path,
                            "explicit_doc_match_score": 15.0,
                            "primary_evidence": {
                                "source_path": source_path,
                                "heading_path": "Methods",
                            },
                        },
                        "ui_meta": {
                            "display_name": source_path,
                            "source_path": source_path,
                            "polish_detail": "summary:llm->full",
                            "reader_open": {
                                "sourcePath": source_path,
                                "sourceName": source_path,
                                "primaryEvidence": {
                                    "source_path": source_path,
                                    "heading_path": "Methods",
                                },
                                "evidenceAlternatives": [
                                    {"sourcePath": source_path, "headingPath": "Results"}
                                ],
                            },
                        },
                    }
                ],
            }
        }
    )

    pack = out[9]
    assert pack["prompt_sig"] == "public-cache-version"
    assert pack["updated_at"] == 1784040000.0
    assert "pipeline_debug" not in pack
    assert "scores" not in pack
    assert "render_error" not in pack
    assert "render_error_detail" not in pack
    assert "render_attempts" not in pack
    assert "render_evidence_sig" not in pack
    assert "rendered_payload" not in pack
    hit = pack["hits"][0]
    assert hit["meta"]["source_path"] == "Paper.en.md"
    assert "explicit_doc_match_score" not in hit["meta"]
    assert "source_path" not in hit["meta"]["primary_evidence"]
    ui_meta = hit["ui_meta"]
    assert ui_meta["display_name"] == "Paper.en.md"
    assert ui_meta["source_path"] == "Paper.en.md"
    assert "polish_detail" not in ui_meta
    assert ui_meta["reader_open"]["sourcePath"] == "Paper.en.md"
    assert ui_meta["reader_open"]["sourceName"] == "Paper.en.md"
    assert "source_path" not in ui_meta["reader_open"]["primaryEvidence"]
    assert "sourcePath" not in ui_meta["reader_open"]["evidenceAlternatives"][0]
    assert source_path not in str(out)


def test_local_source_identity_and_name_hints_accept_windows_paths_cross_platform() -> None:
    from api.reference_local_source_meta import _source_name_hints, source_identity_key

    windows_path = r"F:\private\library\Nature-2026-Portable Paper.en.md"

    assert source_identity_key(windows_path) == source_identity_key(
        "/srv/library/Nature-2026-Portable Paper.pdf"
    )
    assert _source_name_hints(windows_path) == (
        "Nature",
        "2026",
        "Portable Paper",
    )


def test_local_source_meta_returns_offline_filename_identity_with_cached_doi(tmp_path: Path) -> None:
    from api.reference_local_source_meta import load_local_source_citation_meta

    source_path = str(tmp_path / "NatPhoton-2025-Structured detection.en.md")
    doi = "10.1038/example.structured"
    _write_json(
        tmp_path / "references_index.json",
        {
            "docs": {
                source_path.lower(): {
                    "path": source_path,
                    "name": Path(source_path).name,
                    "stem": Path(source_path).stem.lower(),
                    "source_doi": doi,
                    "refs": {},
                }
            }
        },
    )
    _write_json(tmp_path / "crossref_cache.json", {"doi": {}, "source_work": {}})

    meta = load_local_source_citation_meta(
        source_path,
        source_name="NatPhoton-2025-Structured detection.pdf",
        db_dir=tmp_path,
    )

    assert meta == {
        "title": "Structured detection",
        "venue": "NatPhoton",
        "year": "2025",
        "doi": doi,
        "doi_url": f"https://doi.org/{doi}",
    }
