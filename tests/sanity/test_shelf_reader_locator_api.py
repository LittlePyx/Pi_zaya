from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from api.routers import references


def test_batch_maintains_locators_once_per_source_and_recovers_renamed_path(tmp_path, monkeypatch) -> None:
    md_path = tmp_path / "renamed.en.md"
    md_path.write_text(
        "# Paper\n\n"
        "## Methods\n\n"
        "The coded measurement is decoded by a physics-informed reconstruction model.\n\n"
        "A second evidence sentence describes the calibrated detector response.\n",
        encoding="utf-8",
    )
    real_load_source_blocks = references.load_source_blocks
    calls = {"blocks": 0}

    def resolve_reader_path(source_path: str):
        return md_path if str(source_path) == "renamed.pdf" else None

    def counted_load_source_blocks(*args, **kwargs):
        calls["blocks"] += 1
        return real_load_source_blocks(*args, **kwargs)

    monkeypatch.setattr(references, "_resolve_reader_md_path", resolve_reader_path)
    monkeypatch.setattr(references, "load_source_blocks", counted_load_source_blocks)

    response = TestClient(app).post(
        "/api/references/shelf/reader-locators/maintain",
        json={
            "items": [
                {
                    "key": "evidence-1",
                    "source_path": "old-name.pdf",
                    "fallback_source_path": "renamed.pdf",
                    "block_id": "stale-block-1",
                    "anchor_id": "stale-anchor-1",
                    "anchor_kind": "paragraph",
                    "heading_path": "Paper / Methods",
                    "evidence_quote": "decoded by a physics-informed reconstruction model",
                },
                {
                    "key": "evidence-2",
                    "source_path": "old-name.pdf",
                    "fallback_source_path": "renamed.pdf",
                    "block_id": "stale-block-2",
                    "anchor_id": "stale-anchor-2",
                    "anchor_kind": "paragraph",
                    "heading_path": "Paper / Methods",
                    "evidence_quote": "describes the calibrated detector response",
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["requested"] == 2
    assert payload["ready"] == 2
    assert payload["changed"] == 2
    assert payload["unresolved"] == 0
    assert payload["model_calls"] == 0
    assert calls["blocks"] == 1
    assert all(item["source_relocated"] is True for item in payload["items"])
    assert all(item["source_path"] == "renamed.pdf" for item in payload["items"])
    assert payload["items"][0]["locator"]["block_id"] != "stale-block-1"
    assert payload["items"][1]["locator"]["block_id"] != "stale-block-2"


def test_batch_locator_maintenance_rejects_oversized_item_list() -> None:
    response = TestClient(app).post(
        "/api/references/shelf/reader-locators/maintain",
        json={
            "items": [
                {"key": f"item-{index}", "source_path": "paper.pdf", "evidence_quote": "evidence"}
                for index in range(121)
            ],
        },
    )

    assert response.status_code == 422
