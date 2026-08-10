from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app
from kb.chat_store import ChatStore
from kb.research_brief_lineage import matrix_contract_fingerprint


def _metric_chunk(
    chunk_id: str,
    source_path: str,
    *,
    protocol: str,
    target: str,
    result: str,
) -> dict:
    return {
        "id": chunk_id,
        "text": (
            f"Table 1. Quantitative SCI image reconstruction comparisons on the {protocol}. "
            f"Cozy2room LPIPS (lower is better): baseline = .6031; {target} = {result}"
        ),
        "meta": {
            "source_path": source_path,
            "heading_path": "Experiments / Table 1",
            "page_start": 6,
            "block_id": f"block-{chunk_id}",
            "structured_kind": "table_metric",
            "table_metric": "LPIPS",
            "table_metric_direction": "lower",
        },
    }


def test_comparison_candidate_api_requires_mapping_and_refreshes_downstream_lineage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from api.routers import research_gaps as gap_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Comparison candidates")
    left_path = str(tmp_path / "SCIGS.md")
    right_path = str(tmp_path / "SCINeRF.md")
    rows = [
        {
            "id": "row-left",
            "source_item_key": "left",
            "paper": "SCIGS",
            "source_name": "SCIGS",
            "source_path": left_path,
            "source_status": "active",
            "cells": {},
        },
        {
            "id": "row-right",
            "source_item_key": "right",
            "paper": "SCINeRF",
            "source_name": "SCINeRF",
            "source_path": right_path,
            "source_status": "active",
            "cells": {},
        },
    ]
    matrix = store.create_evidence_matrix(
        project_id=project_id,
        title="SCI comparison",
        rows=rows,
        source_items=[
            {"key": "left", "title": "SCIGS", "sourcePath": left_path},
            {"key": "right", "title": "SCINeRF", "sourcePath": right_path},
        ],
        quality_status="verified",
        quality={"contract_version": 2, "reasons": []},
    )
    assert matrix is not None
    brief = store.create_research_brief(
        project_id=project_id,
        title="Living comparison brief",
        content_markdown="# Brief",
        quality_status="verified",
        quality={
            "source_matrix_id": matrix["id"],
            "source_matrix_title": matrix["title"],
            "source_matrix_revision": matrix["revision"],
            "source_matrix_quality_status": "verified",
            "source_matrix_fingerprint": matrix_contract_fingerprint(matrix),
        },
    )
    assert brief is not None
    chunks = [
        _metric_chunk(
            "left-table",
            left_path,
            protocol="static datasets",
            target="SCIGS(ours)",
            result=".0423",
        ),
        _metric_chunk(
            "right-table",
            right_path,
            protocol="synthetic datasets",
            target="ours",
            result=".0445",
        ),
    ]
    monkeypatch.setattr(gap_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(gap_router, "get_settings", lambda: SimpleNamespace(db_dir=tmp_path))
    monkeypatch.setattr(gap_router, "load_all_chunks", lambda _db_dir: chunks)
    monkeypatch.setattr(gap_router, "_indexed_source_is_fresh", lambda _source_path: True)
    monkeypatch.setattr(
        gap_router,
        "_scan_project_evidence_changes",
        lambda _project_id: {"items": []},
    )
    client = TestClient(app)

    response = client.get(
        f"/api/projects/{project_id}/evidence-matrices/{matrix['id']}/comparison-candidates"
    )

    assert response.status_code == 200
    listed = response.json()
    assert listed["matrix_revision"] == 1
    assert listed["examined_row_pairs"] == 1
    assert len(listed["items"]) == 1
    candidate = listed["items"][0]
    assert candidate["required_confirmations"] == ["evaluation_protocol"]
    assert candidate["left_result"] == ".0423"
    assert candidate["right_result"] == ".0445"

    missing_confirmation = client.post(
        f"/api/projects/{project_id}/evidence-matrices/{matrix['id']}"
        f"/comparison-candidates/{candidate['id']}/audit",
        json={"expected_revision": 1, "confirmed_mappings": []},
    )
    assert missing_confirmation.status_code == 400
    assert "evaluation_protocol" in missing_confirmation.json()["detail"]

    audited = client.post(
        f"/api/projects/{project_id}/evidence-matrices/{matrix['id']}"
        f"/comparison-candidates/{candidate['id']}/audit",
        json={
            "expected_revision": 1,
            "confirmed_mappings": ["evaluation_protocol"],
        },
    )

    assert audited.status_code == 200
    payload = audited.json()
    assert payload["audit"]["status"] == "verified"
    assert payload["audit"]["preferred_side"] == "left"
    assert payload["matrix"]["revision"] == 2
    assert payload["matrix"]["quality"]["verified_comparison_count"] == 1
    assert payload["affected_briefs"][0]["id"] == brief["id"]
    assert payload["affected_briefs"][0]["update_ready"] is True
    assert any(
        item["kind"] == "brief_stale"
        for item in payload["research_gaps"]["items"]
    )

    rescanned = client.get(
        f"/api/projects/{project_id}/evidence-matrices/{matrix['id']}/comparison-candidates"
    )
    assert rescanned.status_code == 200
    assert rescanned.json()["items"] == []
