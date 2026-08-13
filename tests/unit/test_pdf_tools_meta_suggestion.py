from pathlib import Path

from kb import pdf_tools


class _FakeDoc:
    metadata = {}
    page_count = 0

    def load_page(self, _index: int):
        raise RuntimeError("no page")


class _FakeTitlePage:
    class _Rect:
        width = 612.0
        height = 792.0

    rect = _Rect()

    def get_text(self, kind: str):
        assert kind == "dict"

        def block(y0: float, size: float, text: str) -> dict:
            return {
                "type": 0,
                "bbox": [20.0, y0, 592.0, y0 + size],
                "lines": [{"spans": [{"text": text, "size": size}]}],
            }

        return {
            "blocks": [
                block(27.0, 10.0, "Published as a conference paper at ICLR 2024"),
                block(80.0, 17.2, "ITRANSFORMER: INVERTED TRANSFORMERS ARE"),
                block(106.0, 17.2, "EFFECTIVE FOR TIME SERIES FORECASTING"),
                block(209.0, 20.0, "arXiv:2310.06625v4 [cs.LG] 14 Mar 2024"),
            ]
        }


def test_font_title_ignores_large_arxiv_stamp_and_conference_header() -> None:
    title = pdf_tools._title_from_font_spans(_FakeTitlePage())

    assert title == (
        "ITRANSFORMER: INVERTED TRANSFORMERS ARE "
        "EFFECTIVE FOR TIME SERIES FORECASTING"
    )


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
