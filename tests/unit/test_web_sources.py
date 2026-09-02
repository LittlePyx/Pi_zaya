from __future__ import annotations

import pytest

from kb import web_sources


class _FakeResponse:
    def __init__(self, body: bytes, *, content_type: str = "text/html", status: int = 200, headers=None):
        self._body = body
        self.status_code = status
        self.headers = {"content-type": content_type, **dict(headers or {})}
        self.closed = False

    def iter_content(self, chunk_size: int = 65536):
        for offset in range(0, len(self._body), max(1, chunk_size)):
            yield self._body[offset:offset + chunk_size]

    def close(self):
        self.closed = True


def test_verify_web_citations_checks_title_binds_claim_and_dedupes(monkeypatch):
    monkeypatch.setattr(web_sources, "_host_is_public", lambda hostname: True)
    answer = "Diffusion models reverse a noising process and improve sample quality."
    start = answer.index("Diffusion")
    end = answer.index(" quality")

    def getter(url, **kwargs):
        assert url == "https://example.org/diffusion"
        return _FakeResponse(
            b"<html><head><title>Diffusion Models Reverse the Noising Process</title></head>"
            b"<body>doi:10.1000/diffusion.2026</body></html>"
        )

    result = web_sources.verify_web_citations(
        answer,
        [
            {
                "url": "https://example.org/diffusion#results",
                "title": "Diffusion Models Reverse the Noising Process",
                "start_index": start,
                "end_index": end,
            },
            {"url": "https://example.org/diffusion#other", "title": "duplicate"},
        ],
        getter=getter,
    )

    assert result["checked_count"] == 1
    assert result["rejected_count"] == 0
    source = result["sources"][0]
    assert source["verification_status"] == "verified"
    assert source["verification_level"] == "content_match"
    assert source["doi"] == "10.1000/diffusion.2026"
    assert source["scholarly_kind"] == "doi"
    assert source["claim_text"].startswith("Diffusion models reverse")
    assert source["import_eligible"] is False


def test_verify_web_citations_recognizes_arxiv_pdf_as_import_candidate(monkeypatch):
    monkeypatch.setattr(web_sources, "_host_is_public", lambda hostname: True)

    result = web_sources.verify_web_citations(
        "An external paper reports this result.",
        [{"url": "https://arxiv.org/pdf/2601.12345", "title": "External paper"}],
        getter=lambda url, **kwargs: _FakeResponse(b"%PDF-1.7\nfixture", content_type="application/pdf"),
    )

    source = result["sources"][0]
    assert source["arxiv_id"] == "2601.12345"
    assert source["doi"] == "10.48550/arXiv.2601.12345"
    assert source["pdf_url"] == "https://arxiv.org/pdf/2601.12345"
    assert source["import_eligible"] is True


def test_verify_web_citations_drops_private_and_mismatched_sources(monkeypatch):
    monkeypatch.setattr(web_sources, "_host_is_public", lambda hostname: hostname != "127.0.0.1")

    def getter(url, **kwargs):
        return _FakeResponse(b"<title>Completely Different Landing Page</title>")

    result = web_sources.verify_web_citations(
        "Answer",
        [
            {"url": "https://127.0.0.1/secret", "title": "Secret"},
            {"url": "https://example.org/paper", "title": "Diffusion Training Dynamics"},
        ],
        getter=getter,
    )

    assert result["sources"] == []
    assert {row["reason"] for row in result["rejected"]} == {"unsafe_host", "content_mismatch"}


def test_fetch_external_pdf_enforces_pdf_signature_and_size(monkeypatch):
    monkeypatch.setattr(web_sources, "_host_is_public", lambda hostname: True)
    fetched = web_sources.fetch_external_pdf(
        "https://example.org/files/paper.pdf",
        max_bytes=64,
        getter=lambda url, **kwargs: _FakeResponse(b"%PDF-1.7 fixture", content_type="application/pdf"),
    )
    assert fetched["file_name"] == "paper.pdf"
    assert fetched["data"].startswith(b"%PDF-")

    with pytest.raises(web_sources.ExternalSourceError, match="direct PDF"):
        web_sources.fetch_external_pdf(
            "https://example.org/files/paper",
            max_bytes=64,
            getter=lambda url, **kwargs: _FakeResponse(b"<html>not a pdf</html>"),
        )

    with pytest.raises(web_sources.ExternalSourceError) as exc_info:
        web_sources.fetch_external_pdf(
            "https://example.org/files/large.pdf",
            max_bytes=8,
            getter=lambda url, **kwargs: _FakeResponse(b"%PDF-1.7 fixture", content_type="application/pdf"),
        )
    assert exc_info.value.code == "too_large"

    declared_large = _FakeResponse(
        b"%PDF-1.7",
        content_type="application/pdf",
        headers={"content-length": "1024"},
    )
    with pytest.raises(web_sources.ExternalSourceError) as declared_exc:
        web_sources.fetch_external_pdf(
            "https://example.org/files/declared-large.pdf",
            max_bytes=64,
            getter=lambda url, **kwargs: declared_large,
        )
    assert declared_exc.value.code == "too_large"
    assert declared_large.closed is True
