from __future__ import annotations

from pathlib import Path

from api import reference_source_citation_meta as source_meta


class Store:
    def __init__(self, stored: dict | None = None) -> None:
        self.stored = stored
        self.saved: list[tuple[Path, dict]] = []

    def get_citation_meta(self, path: Path) -> dict | None:
        return self.stored

    def set_citation_meta(self, path: Path, meta: dict) -> None:
        self.saved.append((path, dict(meta)))


def _base_kwargs(tmp_path: Path) -> dict:
    pdf_path = tmp_path / "paper.pdf"
    return {
        "source_path": "paper.en.md",
        "pdf_root": tmp_path,
        "md_root": tmp_path / "md",
        "lib_store": Store(),
        "resolve_pdf_for_source": lambda pdf_root, source_path: pdf_path,
        "has_metrics_payload": lambda meta: False,
        "parse_filename_meta": lambda source_path: ("Optics Express", "2024", ""),
        "source_filename": lambda source_path: "paper.en.md",
        "infer_title_from_source_text": lambda source_path, fallback_title, **kwargs: "Adaptive sampling for imaging",
        "fetch_crossref_meta": lambda *args, **kwargs: {
            "title": "Adaptive sampling for imaging",
            "authors": "Gehm M, Brady D",
            "doi": "10.1000/demo",
        },
        "is_weak_meta_value": lambda key, value: False,
        "fetch_best_crossref_meta": lambda **kwargs: None,
        "merge_meta_prefer_richer": lambda base, incoming: {**dict(base or {}), **dict(incoming or {})},
        "enrich_bibliometrics": lambda meta: {**dict(meta or {}), "citation_count": 12},
        "ensure_summary_line": lambda meta, **kwargs: {**dict(meta or {}), "summary_line": "Summary"},
    }


def test_source_citation_meta_uses_cached_metrics_without_crossref(tmp_path: Path) -> None:
    store = Store({"title": "Cached paper", "citation_count": 5})
    called: list[str] = []

    out = source_meta.ensure_source_citation_meta(
        **{
            **_base_kwargs(tmp_path),
            "lib_store": store,
            "has_metrics_payload": lambda meta: True,
            "fetch_crossref_meta": lambda *args, **kwargs: called.append("crossref") or {},
        }
    )

    assert out["title"] == "Cached paper"
    assert out["summary_line"] == "Summary"
    assert called == []


def test_source_citation_meta_merges_crossref_and_persists_to_store(tmp_path: Path) -> None:
    store = Store()
    kwargs = _base_kwargs(tmp_path)
    kwargs["lib_store"] = store

    out = source_meta.ensure_source_citation_meta(**kwargs)

    assert out["title"] == "Adaptive sampling for imaging"
    assert out["venue"] == "Optics Express"
    assert out["year"] == "2024"
    assert out["authors"] == "Gehm M, Brady D"
    assert out["citation_count"] == 12
    assert out["summary_line"] == "Summary"
    assert store.saved
    assert store.saved[0][1]["doi"] == "10.1000/demo"


def test_source_citation_meta_uses_best_crossref_when_primary_lookup_empty(tmp_path: Path) -> None:
    calls: list[dict] = []

    def best(**kwargs):
        calls.append(kwargs)
        return {"title": "Best title", "doi": "10.1000/best"}

    out = source_meta.ensure_source_citation_meta(
        **{
            **_base_kwargs(tmp_path),
            "fetch_crossref_meta": lambda *args, **kwargs: None,
            "fetch_best_crossref_meta": best,
        }
    )

    assert out["title"] == "Best title"
    assert out["doi"] == "10.1000/best"
    assert calls
    assert calls[0]["allow_title_only"] is True


def test_source_citation_meta_keeps_running_when_store_write_fails(tmp_path: Path) -> None:
    class FailingStore(Store):
        def set_citation_meta(self, path: Path, meta: dict) -> None:
            raise OSError("read only")

    out = source_meta.ensure_source_citation_meta(
        **{
            **_base_kwargs(tmp_path),
            "lib_store": FailingStore(),
        }
    )

    assert out["doi"] == "10.1000/demo"
    assert out["summary_line"] == "Summary"
