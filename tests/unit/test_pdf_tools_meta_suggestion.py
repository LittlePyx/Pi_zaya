from pathlib import Path

from kb import pdf_tools


class _FakeDoc:
    metadata = {}
    page_count = 0

    def load_page(self, _index: int):
        raise RuntimeError("no page")


def test_extract_pdf_meta_suggestion_preserves_filename_year_when_crossref_missing(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "ICIP-2025-SCIGS-3D Gaussians Splatting from A Snapshot Compressive Image.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 demo")

    monkeypatch.setattr(pdf_tools.fitz, "open", lambda _path: _FakeDoc())
    monkeypatch.setattr(pdf_tools, "fetch_best_crossref_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(pdf_tools, "extract_first_doi", lambda *_args, **_kwargs: "")

    out = pdf_tools.extract_pdf_meta_suggestion(pdf_path, settings=None)

    assert out.venue == "ICIP"
    assert out.year == "2025"
    assert "SCIGS" in out.title

