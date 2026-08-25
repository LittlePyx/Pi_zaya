from __future__ import annotations

from io import BytesIO

from docx import Document
from fastapi.testclient import TestClient

from api.main import app


def test_chat_research_note_export_returns_real_docx():
    response = TestClient(app).post(
        "/api/chat/research-note/export",
        json={
            "title": "可追溯研究笔记",
            "content_markdown": (
                "## 研究问题 1\n\n问题正文\n\n"
                "### 本回答引用\n\n"
                "- DOI: [10.1000/test](https://doi.org/10.1000/test)"
            ),
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    disposition = response.headers["content-disposition"]
    assert 'filename="Pi_zaya research note.docx"' in disposition
    assert "filename*=UTF-8''%E5%8F%AF%E8%BF%BD%E6%BA%AF%E7%A0%94%E7%A9%B6%E7%AC%94%E8%AE%B0.docx" in disposition
    document = Document(BytesIO(response.content))
    assert document.core_properties.title == "可追溯研究笔记"
    assert any("研究问题 1" in paragraph.text for paragraph in document.paragraphs)


def test_chat_research_note_export_rejects_blank_or_oversized_content():
    client = TestClient(app)
    blank = client.post(
        "/api/chat/research-note/export",
        json={"title": "note", "content_markdown": "   "},
    )
    oversized = client.post(
        "/api/chat/research-note/export",
        json={"title": "note", "content_markdown": "x" * 160_001},
    )

    assert blank.status_code == 422
    assert oversized.status_code == 422
