from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app
from kb.chat_store import ChatStore


def _generated_matrix(source_path: str) -> tuple[list[dict], list[dict], list[dict]]:
    evidence = {
        "id": "ev-method",
        "field": "method",
        "source_item_key": "paper-a",
        "source_path": source_path,
        "source_name": "Paper A",
        "title": "Paper A",
        "heading_path": "Method / Architecture",
        "page_start": 3,
        "block_id": "block-method",
        "evidence_quote": "The method uses a coded optical network.",
        "score": 8.0,
    }
    row = {
        "id": "row-a",
        "source_item_key": "paper-a",
        "paper": "Paper A",
        "source_name": "Paper A",
        "source_path": source_path,
        "notes": "",
        "source_status": "active",
        "cells": {
            "method": {
                "field": "method",
                "value": evidence["evidence_quote"],
                "support_status": "grounded",
                "evidence_ids": [evidence["id"]],
                "manual_override": False,
            }
        },
    }
    return [row], [evidence], []


def test_evidence_matrix_api_generates_versions_edits_and_exports(monkeypatch, tmp_path: Path) -> None:
    from api.routers import evidence_matrices as matrix_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Matrix project")
    conv_id = store.create_conversation("Evidence", project_id=project_id)
    source_path = str(tmp_path / "paper-a.md")
    store.save_citation_shelf(
        project_id=project_id,
        scope="project",
        items=[{"key": "paper-a", "title": "Paper A", "sourcePath": source_path}],
        open=True,
    )
    monkeypatch.setattr(matrix_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(matrix_router, "get_settings", lambda: SimpleNamespace(db_dir=tmp_path))
    def fake_build(*args, **kwargs):
        rows, evidence, flags = _generated_matrix(source_path)
        existing = list(kwargs.get("existing_rows") or [])
        if existing:
            rows[0]["notes"] = str(existing[0].get("notes") or "")
        return rows, evidence, flags

    monkeypatch.setattr(matrix_router, "build_project_evidence_matrix", fake_build)
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{project_id}/evidence-matrices/generate",
        json={
            "title": "Optical comparison",
            "objective": "Compare methods.",
            "item_keys": ["paper-a"],
            "source_conv_id": conv_id,
        },
    )
    assert response.status_code == 200
    record = response.json()
    assert record["quality_status"] == "verified"
    assert record["quality"]["covered_source_count"] == 1
    assert record["rows"][0]["cells"]["method"]["support_status"] == "grounded"
    assert record["evidence"][0]["heading_path"] == "Method / Architecture"
    matrix_id = record["id"]

    edited = client.patch(
        f"/api/evidence-matrices/{matrix_id}",
        json={
            "expected_revision": 1,
            "row_updates": [
                {
                    "row_id": "row-a",
                    "notes": "Keep this note on refresh.",
                    "cells": [{"field": "method", "value": "Manual method summary."}],
                }
            ],
        },
    )
    assert edited.status_code == 200
    edited_record = edited.json()
    assert edited_record["revision"] == 2
    assert edited_record["quality_status"] == "needs_review"
    assert edited_record["rows"][0]["cells"]["method"]["manual_override"] is True
    assert "edited_after_verification" in edited_record["quality"]["reasons"]

    conflict = client.patch(
        f"/api/evidence-matrices/{matrix_id}",
        json={"expected_revision": 1, "title": "Stale title"},
    )
    assert conflict.status_code == 409

    refreshed = client.post(
        f"/api/projects/{project_id}/evidence-matrices/generate",
        json={
            "title": "Optical comparison",
            "objective": "Compare methods.",
            "item_keys": ["paper-a"],
            "matrix_id": matrix_id,
            "expected_revision": 2,
        },
    )
    assert refreshed.status_code == 200
    assert refreshed.json()["revision"] == 3
    assert refreshed.json()["quality_status"] == "verified"
    assert refreshed.json()["rows"][0]["notes"] == "Keep this note on refresh."

    revisions = client.get(f"/api/evidence-matrices/{matrix_id}/revisions")
    assert revisions.status_code == 200
    assert [item["revision"] for item in revisions.json()] == [3, 2, 1]

    markdown = client.get(f"/api/evidence-matrices/{matrix_id}/export?format=markdown")
    assert markdown.status_code == 200
    assert "Evidence appendix" in markdown.text
    csv_response = client.get(f"/api/evidence-matrices/{matrix_id}/export?format=csv")
    assert csv_response.status_code == 200
    assert "Paper A" in csv_response.content.decode("utf-8-sig")
    xlsx = client.get(f"/api/evidence-matrices/{matrix_id}/export?format=xlsx")
    assert xlsx.status_code == 200
    assert xlsx.content.startswith(b"PK")


def test_evidence_change_scan_and_apply_refreshes_only_affected_source(monkeypatch, tmp_path: Path) -> None:
    from api.routers import evidence_matrices as matrix_router
    from kb.evidence_watch import source_watch_snapshot

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Living matrix")
    source_a = tmp_path / "paper-a.md"
    source_b = tmp_path / "paper-b.md"
    source_a.write_text("Paper A evidence", encoding="utf-8")
    source_b.write_text("Paper B evidence", encoding="utf-8")
    item_a = {"key": "paper-a", "title": "Paper A", "sourceName": "Paper A", "sourcePath": str(source_a)}
    item_b = {"key": "paper-b", "title": "Paper B", "sourceName": "Paper B", "sourcePath": str(source_b)}
    store.save_citation_shelf(
        project_id=project_id,
        scope="project",
        items=[item_a],
        open=True,
    )
    rows_a, evidence_a, flags_a = _generated_matrix(str(source_a))
    rows_a[0]["notes"] = "Preserve this reviewed note."
    baseline = source_watch_snapshot([item_a], shelf_revision=1)
    matrix = store.create_evidence_matrix(
        project_id=project_id,
        title="Living evidence",
        objective="Compare methods.",
        rows=rows_a,
        evidence=evidence_a,
        source_items=[item_a],
        comparison_flags=flags_a,
        quality_status="verified",
        quality={"contract_version": 2, "reasons": [], "source_watch_snapshot": baseline},
    )
    assert matrix is not None
    manual_draft = store.create_evidence_matrix(project_id=project_id, title="Unbound manual draft")
    assert manual_draft is not None
    store.set_evidence_watch_baseline(
        matrix["id"],
        project_id=project_id,
        matrix_revision=1,
        snapshot=baseline,
    )
    store.save_citation_shelf(
        project_id=project_id,
        scope="project",
        items=[item_a, item_b],
        open=True,
    )

    def fake_build(selected_items, **_kwargs):
        assert len(selected_items) == 1
        assert str(selected_items[0]["sourcePath"]) == str(source_b)
        rows, evidence, flags = _generated_matrix(str(source_b))
        rows[0].update(
            {
                "id": "row-b",
                "source_item_key": "paper-b",
                "paper": "Paper B",
                "source_name": "Paper B",
            }
        )
        evidence[0].update(
            {
                "id": "ev-method-b",
                "source_item_key": "paper-b",
                "source_name": "Paper B",
                "title": "Paper B",
                "evidence_quote": "Paper B uses a distinct optical network.",
            }
        )
        rows[0]["cells"]["method"].update(
            {
                "value": evidence[0]["evidence_quote"],
                "evidence_ids": [evidence[0]["id"]],
            }
        )
        return rows, evidence, flags

    monkeypatch.setattr(matrix_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(matrix_router, "get_settings", lambda: SimpleNamespace(db_dir=tmp_path))
    monkeypatch.setattr(matrix_router, "build_project_evidence_matrix", fake_build)
    monkeypatch.setattr(matrix_router, "_indexed_source_is_fresh", lambda _source_path: True)
    client = TestClient(app)

    scanned = client.post(f"/api/projects/{project_id}/evidence-changes/scan")
    assert scanned.status_code == 200
    events = scanned.json()["items"]
    assert len(events) == 1
    assert events[0]["kind"] == "source_added"
    assert events[0]["actionable"] is True
    assert events[0]["matrix_id"] == matrix["id"]

    applied = client.post(
        f"/api/evidence-matrices/{matrix['id']}/evidence-changes/apply",
        json={"expected_revision": 1, "event_ids": [events[0]["id"]]},
    )
    assert applied.status_code == 200
    payload = applied.json()
    record = payload["record"]
    assert record["revision"] == 2
    assert payload["refreshed_source_count"] == 1
    assert payload["preserved_row_count"] == 1
    assert [row["id"] for row in record["rows"]] == ["row-a", "row-b"]
    assert record["rows"][0] == rows_a[0]
    assert record["rows"][1]["cells"]["method"]["evidence_ids"] == ["ev-method-b"]
    assert record["quality"]["last_evidence_change_application"]["refreshed_row_count"] == 1
    assert client.get(f"/api/projects/{project_id}/evidence-changes").json()["items"] == []

    store.save_citation_shelf(
        project_id=project_id,
        scope="project",
        items=[{**item_a, "title": "Paper A corrected"}, item_b],
        open=True,
    )
    source_b.write_text("Paper B changed evidence", encoding="utf-8")
    metadata_scan = client.post(f"/api/projects/{project_id}/evidence-changes/scan")
    assert metadata_scan.status_code == 200
    metadata_events = metadata_scan.json()["items"]
    assert {event["kind"] for event in metadata_events} == {
        "source_content_changed",
        "source_metadata_changed",
    }
    content_event = next(event for event in metadata_events if event["kind"] == "source_content_changed")
    metadata_event = next(event for event in metadata_events if event["kind"] == "source_metadata_changed")
    metadata_blocked = client.post(
        f"/api/evidence-matrices/{matrix['id']}/evidence-changes/apply",
        json={"expected_revision": 2, "event_ids": [content_event["id"]]},
    )
    assert metadata_blocked.status_code == 409
    assert "metadata-only" in metadata_blocked.json()["detail"]
    acknowledged = client.post(
        f"/api/projects/{project_id}/evidence-changes/{metadata_event['id']}/ignore",
        json={},
    )
    assert acknowledged.status_code == 200

    source_a.unlink()
    rescanned = client.post(f"/api/projects/{project_id}/evidence-changes/scan")
    assert rescanned.status_code == 200
    next_events = rescanned.json()["items"]
    assert {event["kind"] for event in next_events} == {
        "source_content_changed",
        "source_unavailable",
    }
    changed = next(event for event in next_events if event["kind"] == "source_content_changed")
    unavailable = next(event for event in next_events if event["kind"] == "source_unavailable")

    partial = client.post(
        f"/api/evidence-matrices/{matrix['id']}/evidence-changes/apply",
        json={"expected_revision": 2, "event_ids": [changed["id"]]},
    )
    assert partial.status_code == 409
    assert "all open actionable" in partial.json()["detail"]
    ignored = client.post(
        f"/api/projects/{project_id}/evidence-changes/{unavailable['id']}/ignore",
        json={},
    )
    assert ignored.status_code == 400
    blocked = client.post(
        f"/api/evidence-matrices/{matrix['id']}/evidence-changes/apply",
        json={"expected_revision": 2, "event_ids": [event["id"] for event in next_events]},
    )
    assert blocked.status_code == 409
    assert "unavailable" in blocked.json()["detail"]


def test_research_brief_can_use_only_a_verified_matrix(monkeypatch, tmp_path: Path) -> None:
    from api.routers import research_briefs as brief_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Matrix-backed brief")
    source_path = str(tmp_path / "paper-a.md")
    rows, evidence, _flags = _generated_matrix(source_path)
    matrix = store.create_evidence_matrix(
        project_id=project_id,
        title="Verified matrix",
        objective="Compare methods.",
        rows=rows,
        evidence=evidence,
        source_items=[{"key": "paper-a", "title": "Paper A", "sourcePath": source_path}],
        quality_status="verified",
        quality={"supported_cell_count": 1},
    )
    assert matrix is not None
    monkeypatch.setattr(brief_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(brief_router, "get_settings", lambda: SimpleNamespace(db_dir=tmp_path))
    monkeypatch.setattr(
        brief_router,
        "generate_research_brief_from_matrix",
        lambda *args, **kwargs: {
            "answer": "The method uses a coded optical network [1].",
            "hits": [
                {
                    "text": "The method uses a coded optical network.",
                    "score": 8.0,
                    "meta": {
                        "source_path": source_path,
                        "source_name": "Paper A",
                        "heading_path": "Method / Architecture",
                        "matrix_field": "method",
                    },
                }
            ],
            "agent_trace": {
                "status": "done",
                "errors": [],
                "verification": {
                    "total_claims": 1,
                    "supported_claims": 1,
                    "unsupported_claims": 0,
                    "support_ratio": 1.0,
                    "evidence_status": "grounded",
                },
                "summary": {"query_scope": "basket", "quality_gate_status": "passed"},
            },
        },
    )
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{project_id}/research-briefs/generate",
        json={"title": "Matrix brief", "objective": "Compare methods.", "matrix_id": matrix["id"]},
    )
    assert response.status_code == 200
    brief = response.json()
    assert brief["quality_status"] == "verified"
    assert brief["quality"]["source_matrix_id"] == matrix["id"]
    assert brief["quality"]["source_matrix_revision"] == 1
    assert brief["lineage"]["status"] == "current"
    assert brief["lineage"]["latest_verified"] is True

    refreshed_rows = deepcopy(rows)
    refreshed_evidence = deepcopy(evidence)
    refreshed_rows[0]["cells"]["method"]["value"] = "The method uses a revised coded optical network."
    refreshed_rows[0]["cells"]["method"]["evidence_ids"] = ["ev-method-r2"]
    refreshed_evidence[0]["id"] = "ev-method-r2"
    refreshed_evidence[0]["evidence_quote"] = "The method uses a revised coded optical network."
    refreshed, conflict = store.update_evidence_matrix(
        matrix["id"],
        expected_revision=1,
        rows=refreshed_rows,
        evidence=refreshed_evidence,
        quality_status="verified",
        quality={"supported_cell_count": 1},
    )
    assert conflict is False
    assert refreshed is not None
    assert refreshed["revision"] == 2

    stale_brief = client.get(f"/api/research-briefs/{brief['id']}")
    assert stale_brief.status_code == 200
    stale_lineage = stale_brief.json()["lineage"]
    assert stale_lineage["status"] == "matrix_updated"
    assert stale_lineage["historical_verified"] is True
    assert stale_lineage["latest_verified"] is False
    assert stale_lineage["impact"]["changed_field_count"] == 1
    assert stale_lineage["impact"]["affected_citation_numbers"] == [1]

    historical_export = client.get(f"/api/research-briefs/{brief['id']}/export?format=markdown")
    assert historical_export.status_code == 200
    assert "brief source revision: 1" in historical_export.text
    assert "current matrix revision: 2" in historical_export.text
    assert "freshness: matrix_updated" in historical_export.text

    full_replace = client.post(
        f"/api/projects/{project_id}/research-briefs/generate",
        json={
            "title": "Matrix brief",
            "objective": "Compare methods.",
            "matrix_id": matrix["id"],
            "brief_id": brief["id"],
            "expected_revision": 1,
        },
    )
    assert full_replace.status_code == 409
    assert "incremental update plan" in full_replace.json()["detail"]

    monkeypatch.setattr(
        brief_router,
        "generate_grounded_answer",
        lambda *args, **kwargs: {
            "answer": "- Paper A uses a revised coded optical network [1].",
        },
    )
    planned = client.post(
        f"/api/research-briefs/{brief['id']}/update-plans",
        json={"expected_revision": 1, "locale": "en"},
    )
    assert planned.status_code == 200
    update_plan = planned.json()
    assert len(update_plan["items"]) == 1
    assert "revised coded optical network" in update_plan["items"][0]["proposed_markdown"]
    regenerated = client.post(
        f"/api/research-briefs/{brief['id']}/update-plans/{update_plan['id']}/apply",
        json={
            "expected_revision": 1,
            "decisions": [
                {"item_id": update_plan["items"][0]["id"], "decision": "accept"},
            ],
        },
    )
    assert regenerated.status_code == 200
    regenerated_brief = regenerated.json()
    assert regenerated_brief["revision"] == 2
    assert regenerated_brief["quality"]["source_matrix_revision"] == 2
    assert regenerated_brief["lineage"]["status"] == "current"

    other = store.create_evidence_matrix(
        project_id=project_id,
        title="Other verified matrix",
        rows=refreshed_rows,
        evidence=refreshed_evidence,
        source_items=[{"key": "paper-a", "title": "Paper A", "sourcePath": source_path}],
        quality_status="verified",
        quality={"supported_cell_count": 1},
    )
    assert other is not None
    switched = client.post(
        f"/api/projects/{project_id}/research-briefs/generate",
        json={
            "title": "Unsafe switch",
            "matrix_id": other["id"],
            "brief_id": brief["id"],
            "expected_revision": 2,
        },
    )
    assert switched.status_code == 400
    assert "cannot switch evidence matrices" in switched.json()["detail"]

    assert store.delete_evidence_matrix(matrix["id"]) is True
    missing_lineage = client.get(f"/api/research-briefs/{brief['id']}")
    assert missing_lineage.status_code == 200
    assert missing_lineage.json()["lineage"]["status"] == "matrix_missing"
    blocked_export = client.get(f"/api/research-briefs/{brief['id']}/export?format=markdown")
    assert blocked_export.status_code == 409
    assert "lineage cannot be verified" in blocked_export.json()["detail"]

    draft = store.create_evidence_matrix(project_id=project_id, title="Draft matrix")
    assert draft is not None
    rejected = client.post(
        f"/api/projects/{project_id}/research-briefs/generate",
        json={"title": "Rejected", "matrix_id": draft["id"]},
    )
    assert rejected.status_code == 400
    assert "verified evidence matrix" in rejected.json()["detail"]


def test_comparison_audit_api_versions_persists_and_deletes_audited_result(monkeypatch, tmp_path: Path) -> None:
    from api.routers import evidence_matrices as matrix_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Comparison audit")
    rows = [
        {
            "id": "row-left",
            "source_item_key": "left",
            "paper": "Paper Left",
            "source_name": "Paper Left",
            "source_path": "F:/papers/left.md",
            "source_status": "active",
            "cells": {},
        },
        {
            "id": "row-right",
            "source_item_key": "right",
            "paper": "Paper Right",
            "source_name": "Paper Right",
            "source_path": "F:/papers/right.md",
            "source_status": "active",
            "cells": {},
        },
    ]
    matrix = store.create_evidence_matrix(
        project_id=project_id,
        title="Audited comparison",
        rows=rows,
        source_items=[
            {"key": "left", "sourcePath": "F:/papers/left.md"},
            {"key": "right", "sourcePath": "F:/papers/right.md"},
        ],
        quality_status="verified",
        quality={"contract_version": 2, "reasons": []},
    )
    assert matrix is not None
    chunks = [
        {
            "id": "left-table",
            "text": (
                "Quantitative SCI image reconstruction comparisons on the static datasets. "
                "Cozy2room LPIPS ↓ (lower is better): SCIGS(ours) = .0423"
            ),
            "meta": {"source_path": "F:/papers/left.md", "page": 6, "heading_path": "Table 1"},
        },
        {
            "id": "right-table",
            "text": (
                "Quantitative SCI image reconstruction comparisons on the synthetic datasets. "
                "Cozy2room LPIPS ↓ (lower is better): ours = .0445"
            ),
            "meta": {"source_path": "F:/papers/right.md", "page": 6, "heading_path": "Table 1"},
        },
    ]
    monkeypatch.setattr(matrix_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(matrix_router, "get_settings", lambda: SimpleNamespace(db_dir=tmp_path))
    monkeypatch.setattr("kb.evidence_matrix.load_all_chunks", lambda _db_dir: chunks)
    client = TestClient(app)
    body = {
        "expected_revision": 1,
        "mode": "ranking",
        "left_row_id": "row-left",
        "right_row_id": "row-right",
        "dimensions": [
            {"dimension": "task", "left_value": "SCI image reconstruction", "right_value": "SCI image reconstruction"},
            {"dimension": "dataset", "left_value": "Cozy2room", "right_value": "Cozy2room"},
            {
                "dimension": "evaluation_protocol",
                "left_value": "static datasets",
                "right_value": "synthetic datasets",
                "mapping_confirmed": True,
            },
            {"dimension": "metric", "left_value": "LPIPS", "right_value": "LPIPS"},
        ],
        "left_target": "SCIGS(ours)",
        "right_target": "ours",
        "left_result": ".0423",
        "right_result": ".0445",
    }

    response = client.post(f"/api/evidence-matrices/{matrix['id']}/comparison-audits", json=body)

    assert response.status_code == 200
    audited = response.json()
    assert audited["revision"] == 2
    assert audited["comparison_audits"][0]["status"] == "verified"
    assert audited["comparison_audits"][0]["preferred_side"] == "left"
    assert audited["quality"]["verified_comparison_count"] == 1
    stale = client.post(f"/api/evidence-matrices/{matrix['id']}/comparison-audits", json=body)
    assert stale.status_code == 409

    comparison_id = audited["comparison_audits"][0]["id"]
    deleted = client.delete(
        f"/api/evidence-matrices/{matrix['id']}/comparison-audits/{comparison_id}?expected_revision=2"
    )
    assert deleted.status_code == 200
    assert deleted.json()["revision"] == 3
    assert deleted.json()["comparison_audits"] == []
