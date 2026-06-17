from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api.main import app
from kb.library_store import LibraryStore
from kb.store import compute_doc_id, save_docs_index, write_doc_chunks


def test_library_files_route_classifies_queue_and_reconvert(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf_a = pdf_dir / "a.pdf"
    pdf_b = pdf_dir / "b.pdf"
    pdf_c = pdf_dir / "c.pdf"
    for p in (pdf_a, pdf_b, pdf_c):
        p.write_bytes(b"%PDF-1.4 test")

    md_b = md_dir / "b" / "b.en.md"
    md_b.parent.mkdir(parents=True, exist_ok=True)
    md_b.write_text("# b\n", encoding="utf-8")

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(
        library_router,
        "_bg_snapshot",
        lambda: {
            "running": True,
            "current": "a.pdf",
            "cur_task_replace": False,
            "queue": [
                {"pdf": str(pdf_c), "name": "c.pdf", "replace": False, "_tid": "q1"},
                {"pdf": str(pdf_b), "name": "b.pdf", "replace": True, "_tid": "q2"},
            ],
            "done": 0,
            "total": 2,
        },
    )

    client = TestClient(app)
    response = client.get("/api/library/files", params={"scope": "all"})
    assert response.status_code == 200
    payload = response.json()

    by_name = {str(item.get("name") or ""): item for item in list(payload.get("items") or [])}
    assert by_name["a.pdf"]["task_state"] == "running"
    assert by_name["a.pdf"]["category"] == "pending"
    assert by_name["b.pdf"]["task_state"] == "queued"
    assert by_name["b.pdf"]["replace_task"] is True
    assert by_name["b.pdf"]["category"] == "pending"
    assert by_name["c.pdf"]["queue_pos"] == 1
    assert int((payload.get("counts") or {}).get("pending") or 0) == 3
    assert int((payload.get("counts") or {}).get("converted") or 0) == 0


def test_library_files_route_classifies_multiple_active_tasks(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf_a = pdf_dir / "a.pdf"
    pdf_b = pdf_dir / "b.pdf"
    pdf_c = pdf_dir / "c.pdf"
    for p in (pdf_a, pdf_b, pdf_c):
        p.write_bytes(b"%PDF-1.4 test")

    md_b = md_dir / "b" / "b.en.md"
    md_b.parent.mkdir(parents=True, exist_ok=True)
    md_b.write_text("# b\n", encoding="utf-8")

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(
        library_router,
        "_bg_snapshot",
        lambda: {
            "running": True,
            "current": "a.pdf",
            "cur_task_replace": False,
            "active_count": 2,
            "active_tasks": [
                {
                    "_tid": "r1",
                    "pdf": str(pdf_a),
                    "name": "a.pdf",
                    "replace": False,
                    "cur_page_done": 1,
                    "cur_page_total": 4,
                    "cur_page_msg": "page 1",
                },
                {
                    "_tid": "r2",
                    "pdf": str(pdf_b),
                    "name": "b.pdf",
                    "replace": True,
                    "cur_page_done": 2,
                    "cur_page_total": 6,
                    "cur_page_msg": "page 2",
                },
            ],
            "queue": [
                {"pdf": str(pdf_c), "name": "c.pdf", "replace": False, "_tid": "q1"},
            ],
            "done": 0,
            "total": 3,
        },
    )

    client = TestClient(app)
    response = client.get("/api/library/files", params={"scope": "all"})
    assert response.status_code == 200
    payload = response.json()

    by_name = {str(item.get("name") or ""): item for item in list(payload.get("items") or [])}
    assert by_name["a.pdf"]["task_state"] == "running"
    assert by_name["b.pdf"]["task_state"] == "running"
    assert by_name["b.pdf"]["replace_task"] is True
    assert by_name["b.pdf"]["category"] == "pending"
    assert by_name["a.pdf"]["cur_page_done"] == 1
    assert by_name["b.pdf"]["cur_page_total"] == 6
    assert by_name["c.pdf"]["task_state"] == "queued"
    assert by_name["c.pdf"]["queue_pos"] == 1
    counts = payload.get("counts") or {}
    assert int(counts.get("running") or 0) == 2
    assert int(counts.get("queued") or 0) == 1
    assert int(counts.get("pending") or 0) == 3
    queue_meta = payload.get("queue") or {}
    assert int(queue_meta.get("active_count") or 0) == 2
    assert len(list(queue_meta.get("active_tasks") or [])) == 2


def test_library_files_route_includes_conversion_quality(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router
    from kb.converter.quality_repair import append_conversion_repair_attempt, write_conversion_quality_result

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    good_pdf = pdf_dir / "good.pdf"
    broken_pdf = pdf_dir / "broken.pdf"
    pending_pdf = pdf_dir / "pending.pdf"
    for p in (good_pdf, broken_pdf, pending_pdf):
        p.write_bytes(b"%PDF-1.4 test")

    good_assets = md_dir / "good" / "assets"
    good_assets.mkdir(parents=True, exist_ok=True)
    (good_assets / "fig.png").write_bytes(b"png")
    good_md = md_dir / "good" / "good.en.md"
    good_md.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "# Good Paper",
                "## Abstract",
                "This paper cites prior work [1-2].",
                "## Method",
                "![Figure 1](assets/fig.png)",
                "**Figure 1.** Diagram.",
                "$$",
                "x=y",
                "$$",
                "## References",
                "[1] First reference.",
                "[2] Second reference.",
            ]
        ),
        encoding="utf-8",
    )
    write_conversion_quality_result(
        good_md,
        auto_repair_result={
            "changed": True,
            "applied": ["ensure_page_anchor"],
            "issue_codes_before": ["missing_page_markers"],
            "issue_codes_after": [],
            "remaining_issue_codes": [],
        },
    )
    append_conversion_repair_attempt(
        good_md,
        event="ingest_finished",
        status="success",
        action="autofix",
        source="test",
        detail="indexed",
    )

    broken_folder = md_dir / "broken"
    broken_folder.mkdir(parents=True, exist_ok=True)
    (broken_folder / "broken.en.md").write_text(
        "\n".join(
            [
                "# Broken Paper",
                "![missing](assets/missing.png)",
                "$$",
                "x=y",
                "\u951b",
            ]
        ),
        encoding="utf-8",
    )

    class FakeStore:
        def list_records_by_paths(self, paths):
            return {}

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_library_store", lambda: FakeStore())
    monkeypatch.setattr(library_router, "_bg_snapshot", lambda: {"running": False, "current": "", "queue": []})

    client = TestClient(app)
    response = client.get("/api/library/files", params={"scope": "all"})
    assert response.status_code == 200
    payload = response.json()

    by_name = {str(item.get("name") or ""): item for item in list(payload.get("items") or [])}
    good_quality = by_name["good.pdf"]["conversion_quality"]
    assert good_quality["status"] == "good"
    assert good_quality["score"] >= 90
    assert good_quality["metrics"]["references"] == 2
    assert good_quality["metrics"]["missing_images"] == 0
    assert good_quality["conversion_report"]["auto_repair_changed"] is True
    assert good_quality["conversion_report"]["auto_repair_applied"] == ["ensure_page_anchor"]
    assert good_quality["conversion_report"]["repair_plan"]["action"] == "none"
    assert good_quality["conversion_report"]["latest_repair_attempt"]["event"] == "ingest_finished"
    assert good_quality["conversion_report"]["repair_attempt_count"] == 1
    assert good_quality["conversion_report"]["recommended_action"] == "none"

    broken_quality = by_name["broken.pdf"]["conversion_quality"]
    assert broken_quality["status"] == "error"
    assert broken_quality["has_review_issue"] is True
    issue_codes = {str(item.get("code") or "") for item in list(broken_quality.get("issues") or [])}
    assert {"missing_images", "unclosed_display_math", "mojibake"}.issubset(issue_codes)
    assert by_name["pending.pdf"]["conversion_quality"] is None
    assert int((payload.get("counts") or {}).get("quality_review") or 0) == 1
    assert int((payload.get("counts") or {}).get("quality_ready") or 0) == 1

    monkeypatch.setattr(
        library_router,
        "_latest_research_qa_quality_summary",
        lambda: {
            "available": True,
            "status": "error",
            "summary": {"total": 4, "passed": 3, "failed": 1},
            "top_failures": [{"name": "refs_include_required_docs", "count": 1}],
            "latest_path": str(tmp_path / "research_qa_eval" / "latest"),
            "report_path": str(tmp_path / "research_qa_eval" / "latest" / "report.md"),
            "updated_at": 1,
        },
    )
    monkeypatch.setattr(
        library_router,
        "_latest_citation_card_quality_summary",
        lambda: {
            "available": True,
            "status": "error",
            "summary": {
                "tracked_checks": 8,
                "failed_checks": 2,
                "citation_card_failed": 1,
                "shelf_failed": 1,
                "ref_card_failed": 0,
                "system_b_failed": 0,
                "shelf_item_count": 2,
                "shelf_metadata_ready_count": 1,
                "shelf_export_ready_count": 1,
                "shelf_summary_export_ready_count": 1,
                "shelf_doi_count": 1,
                "shelf_source_clickable_count": 2,
                "shelf_review_count": 1,
            },
            "top_failures": [{"name": "citation_card_quality", "count": 1}],
            "latest_path": str(tmp_path / "research_qa_eval" / "latest"),
            "updated_at": 1,
        },
    )

    overview_response = client.get("/api/library/quality/overview", params={"scope": "all"})
    assert overview_response.status_code == 200
    overview = overview_response.json()
    assert overview["ok"] is True
    assert overview["status"] == "error"
    summary = overview["summary"]
    assert summary["converted"] == 2
    assert summary["assessed"] == 2
    assert summary["review"] == 1
    assert summary["good"] == 1
    assert summary["avg_score"] > 0
    top_codes = {str(item.get("code") or "") for item in list(overview.get("top_issues") or [])}
    assert "missing_images" in top_codes
    recommended = list(overview.get("recommended") or [])
    assert recommended
    assert recommended[0]["name"] == "broken.pdf"
    assert recommended[0]["score"] == broken_quality["score"]
    domains = overview["domains"]
    assert domains["conversion"]["summary"]["review"] == 1
    assert domains["research_qa"]["summary"]["failed"] == 1
    assert domains["citation_cards"]["summary"]["failed_checks"] == 2
    assert domains["citation_cards"]["summary"]["shelf_export_ready_count"] == 1
    assert domains["citation_cards"]["summary"]["shelf_summary_export_ready_count"] == 1
    priority_domains = {str(item.get("domain") or "") for item in list(overview.get("priority_actions") or [])}
    assert {"conversion", "research_qa", "citation_cards"}.issubset(priority_domains)
    full_chain = overview["full_chain"]
    assert full_chain["status"] == "error"
    assert full_chain["score"] < 100
    stage_keys = {str(item.get("key") or "") for item in list(full_chain.get("stages") or [])}
    assert {"conversion", "research_qa", "retrieval", "citations", "shelf", "repair_loop"}.issubset(stage_keys)
    root_codes = {str(item.get("code") or "") for item in list(full_chain.get("root_causes") or [])}
    assert "missing_images" in root_codes
    feature_health = overview["feature_health"]
    assert feature_health["status"] == "error"
    feature_keys = {str(item.get("key") or "") for item in list(feature_health.get("items") or [])}
    assert {"pdf_conversion", "general_qa", "paper_guide", "citation_cards", "literature_basket", "reader_locate", "repair_loop"}.issubset(feature_keys)


def test_figure_asset_refresh_queues_problem_markdown(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    Image = pytest.importorskip("PIL.Image")
    ImageDraw = pytest.importorskip("PIL.ImageDraw")

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    assets = md_dir / "paper" / "assets"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")
    md_path = md_dir / "paper" / "paper.en.md"
    md_path.write_text("![Figure](./assets/page_1_fig_1.png)\n", encoding="utf-8")

    img = Image.new("RGB", (160, 160), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 140, 140], outline="black", width=3)
    img.save(assets / "page_1_fig_1.png")
    (assets / "figure_index.json").write_text(
        json.dumps(
            {
                "figures": [
                    {
                        "page": 1,
                        "index": 1,
                        "asset_name": "page_1_fig_1.png",
                        "crop_bbox": [0, 0, 72, 72],
                        "bbox": [0, 0, 72, 72],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    queued: list[dict] = []
    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_bg_snapshot", lambda: {"running": False, "current": "", "queue": [], "active_tasks": []})
    monkeypatch.setattr(library_router, "_bg_enqueue", lambda task: queued.append(dict(task)))

    client = TestClient(app)
    response = client.post("/api/library/quality/figure-assets/refresh", json={"limit": 10, "target_dpi": 320})

    assert response.status_code == 200
    payload = response.json()
    assert payload["enqueued"] == 1
    assert payload["issue_counts"]["low_resolution"] == 1
    assert len(queued) == 1
    assert queued[0]["pdf"] == str(pdf_path)
    assert queued[0]["replace"] is True
    assert queued[0]["repair_context"]["source"] == "figure_asset_quality_refresh"


def test_library_files_route_exposes_authoritative_index_state(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    db_dir = tmp_path / "db"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    ready_pdf = pdf_dir / "ready.pdf"
    blocked_pdf = pdf_dir / "blocked.pdf"
    stale_pdf = pdf_dir / "stale.pdf"
    for p in (ready_pdf, blocked_pdf, stale_pdf):
        p.write_bytes(b"%PDF-1.4 test")

    ready_md = md_dir / "ready" / "ready.en.md"
    blocked_md = md_dir / "blocked" / "blocked.en.md"
    stale_md = md_dir / "stale" / "stale.en.md"
    for md in (ready_md, blocked_md, stale_md):
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text("# Paper\n\n## Abstract\n\ncontent\n\n## References\n\n[1] Ref.", encoding="utf-8")

    ready_id = compute_doc_id(ready_md)
    blocked_id = compute_doc_id(blocked_md)
    stale_id = compute_doc_id(stale_md)
    save_docs_index(
        db_dir,
        {
            ready_id: {
                "doc_id": ready_id,
                "path": str(ready_md),
                "sha1": "ready-sha",
                "num_chunks": 1,
                "index_status": "ready",
                "quality_gate": {"status": "ready", "action": "none"},
            },
            blocked_id: {
                "doc_id": blocked_id,
                "path": str(blocked_md),
                "sha1": "blocked-sha",
                "num_chunks": 0,
                "index_status": "quality_blocked",
                "quality_gate": {"status": "blocked", "action": "reconvert", "issue_codes": ["missing_references"]},
            },
            stale_id: {
                "doc_id": stale_id,
                "path": str(stale_md),
                "sha1": "stale-sha",
                "num_chunks": 2,
                "index_status": "ready",
                "quality_gate": {"status": "ready", "action": "none"},
            },
        },
    )
    write_doc_chunks(db_dir, ready_id, [{"text": "ready chunk", "meta": {"source_path": str(ready_md)}}])

    class FakeStore:
        def list_records_by_paths(self, paths):
            return {}

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_library_store", lambda: FakeStore())
    monkeypatch.setattr(library_router, "_bg_snapshot", lambda: {"running": False, "current": "", "queue": []})
    monkeypatch.setattr(
        library_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=str(db_dir), library_db_path=str(tmp_path / "library.db")),
    )

    client = TestClient(app)
    response = client.get("/api/library/files", params={"scope": "all"})

    assert response.status_code == 200
    payload = response.json()
    by_name = {str(item.get("name") or ""): item for item in list(payload.get("items") or [])}
    assert by_name["ready.pdf"]["index_state"] == "ready"
    assert by_name["ready.pdf"]["index_ready"] is True
    assert by_name["ready.pdf"]["index_chunk_exists"] is True
    assert by_name["blocked.pdf"]["index_state"] == "quality_blocked"
    assert by_name["blocked.pdf"]["index_ready"] is False
    assert by_name["blocked.pdf"]["quality_gate"]["action"] == "reconvert"
    assert by_name["stale.pdf"]["index_state"] == "index_stale"
    assert by_name["stale.pdf"]["index_ready"] is False
    counts = payload.get("counts") or {}
    assert counts["index_ready"] == 1
    assert counts["index_quality_blocked"] == 1
    assert counts["index_stale"] == 1


def test_library_source_quality_route_resolves_pdf_and_md_sources(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdf_dir / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    md_folder = md_dir / "source"
    md_folder.mkdir(parents=True, exist_ok=True)
    md = md_folder / "source.en.md"
    md.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "# Source",
                "## Abstract",
                "A clean source with citation [1].",
                "## References",
                "[1] Reference.",
            ]
        ),
        encoding="utf-8",
    )

    broken_md = md_dir / "broken" / "broken.en.md"
    broken_md.parent.mkdir(parents=True, exist_ok=True)
    broken_md.write_text(
        "\n".join(
            [
                "# Broken",
                "![missing](assets/missing.png)",
                "$$",
                "x=y",
                "\u951b",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)

    client = TestClient(app)
    response = client.post(
        "/api/library/quality/sources",
        json={
            "sources": [
                {"source_path": str(pdf), "source_name": "source.pdf"},
                {"source_path": str(broken_md), "source_name": "broken.en.md"},
                {"source_path": str(tmp_path / "outside.md"), "source_name": "outside.md"},
            ]
        },
    )
    assert response.status_code == 200
    payload = response.json()
    items = list(payload.get("items") or [])
    assert len(items) == 3

    by_name = {str(item.get("source_name") or ""): item for item in items}
    assert by_name["source.pdf"]["md_exists"] is True
    assert by_name["source.pdf"]["conversion_quality"]["status"] == "good"
    assert by_name["broken.en.md"]["conversion_quality"]["status"] == "error"
    assert by_name["outside.md"]["conversion_quality"] is None
    assert int(payload.get("review_count") or 0) == 1


def test_library_conversion_quality_batch_route_scans_markdown(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdf_dir / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    md = md_dir / "source" / "source.en.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "# Source",
                "## Abstract",
                "A clean source with citation [1].",
                "## Method",
                "The method section has enough text for a stable quality scan.",
                "## References",
                "[1] Reference.",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)

    client = TestClient(app)
    response = client.post(
        "/api/library/quality/conversion/batch",
        json={"repair": False, "limit": 5},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "scan"
    assert payload["target_count"] == 1
    assert payload["scanned"] == 1
    assert payload["failed"] == 0
    assert sum(int(payload.get(key) or 0) for key in ("ready", "autofix", "reconvert", "review", "unknown")) == 1


def test_library_reader_locate_quality_events_feed_overview(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdf_dir / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    md = md_dir / "source" / "source.en.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "# Source",
                "## Abstract",
                "This source contains the exact evidence [1].",
                "## Method",
                "The method has a stable anchor.",
                "## References",
                "[1] Reference.",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_reader_locate_events_path", lambda: tmp_path / "reader_locate_events.jsonl")
    monkeypatch.setattr(library_router, "_bg_snapshot", lambda: {"running": False, "current": "", "queue": []})
    monkeypatch.setattr(
        library_router,
        "_latest_research_qa_quality_summary",
        lambda: {
            "available": True,
            "status": "good",
            "summary": {"total": 1, "passed": 1, "failed": 0},
            "top_failures": [],
        },
    )
    monkeypatch.setattr(
        library_router,
        "_latest_citation_card_quality_summary",
        lambda: {
            "available": True,
            "status": "good",
            "summary": {
                "tracked_checks": 2,
                "failed_checks": 0,
                "citation_card_failed": 0,
                "shelf_failed": 0,
                "ref_card_failed": 0,
                "system_b_failed": 0,
                "shelf_item_count": 1,
                "shelf_metadata_ready_count": 1,
                "shelf_export_ready_count": 1,
                "shelf_summary_export_ready_count": 1,
                "shelf_doi_count": 1,
                "shelf_source_clickable_count": 1,
                "shelf_review_count": 0,
            },
            "top_failures": [],
        },
    )
    monkeypatch.setattr(library_router, "_research_qa_rerun_history_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(library_router, "_latest_research_qa_failure_cases", lambda *args, **kwargs: [])

    client = TestClient(app)
    exact_response = client.post(
        "/api/library/quality/reader-locate",
        json={
            "source_path": str(pdf),
            "source_name": "source.pdf",
            "locate_feedback_key": "cite-1",
            "locate_request_id": 1,
            "status": "exact",
            "precision": "phrase",
            "ok": True,
            "repairable": False,
            "strict_locate": True,
            "block_id": "p-1",
            "anchor_id": "a-p-1",
            "heading_path": "Method",
        },
    )
    assert exact_response.status_code == 200
    assert exact_response.json()["item"]["md_exists"] is True

    failed_response = client.post(
        "/api/library/quality/reader-locate",
        json={
            "source_path": str(md),
            "source_name": "source.en.md",
            "locate_feedback_key": "cite-2",
            "locate_request_id": 2,
            "status": "failed",
            "precision": "failed",
            "ok": False,
            "repairable": True,
            "strict_locate": True,
            "reason": "anchor not found",
            "heading_path": "Method",
        },
    )
    assert failed_response.status_code == 200
    failed_item = failed_response.json()["item"]
    assert failed_item["recommended_action"] == "repair_conversion_and_reindex"
    assert failed_item["pdf_path"] == str(pdf)

    overview_response = client.get("/api/library/quality/overview", params={"scope": "all"})
    assert overview_response.status_code == 200
    overview = overview_response.json()
    reader_locate = overview["reader_locate"]
    assert reader_locate["available"] is True
    assert reader_locate["status"] == "error"
    assert reader_locate["summary"]["total"] == 2
    assert reader_locate["summary"]["exact"] == 1
    assert reader_locate["summary"]["failed"] == 1
    assert reader_locate["summary"]["repairable"] == 1
    assert reader_locate["recommended_sources"][0]["failed"] == 1
    assert reader_locate["recommended_sources"][0]["recommended_action"] == "repair_conversion_and_reindex"
    assert overview["domains"]["reader_locate"]["summary"]["failed"] == 1
    priority_domains = {str(item.get("domain") or "") for item in list(overview.get("priority_actions") or [])}
    assert "reader_locate" in priority_domains
    feature_reader = next(
        item for item in list(overview["feature_health"]["items"] or [])
        if str(item.get("key") or "") == "reader_locate"
    )
    assert feature_reader["status"] == "error"
    assert feature_reader["target_stage"] == "reader_locate"
    assert feature_reader["metrics"]["failed"] == 1

    rows_response = client.get("/api/library/quality/reader-locate", params={"limit": 5})
    assert rows_response.status_code == 200
    rows_payload = rows_response.json()
    assert len(rows_payload["items"]) == 2
    assert rows_payload["summary"]["summary"]["failed"] == 1


def test_library_reader_locate_ignores_removed_source_events(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_reader_locate_events_path", lambda: tmp_path / "reader_locate_events.jsonl")
    monkeypatch.setattr(library_router, "_bg_snapshot", lambda: {"running": False, "current": "", "queue": []})
    monkeypatch.setattr(
        library_router,
        "_latest_research_qa_quality_summary",
        lambda: {
            "available": True,
            "status": "good",
            "summary": {"total": 1, "passed": 1, "failed": 0},
            "top_failures": [],
        },
    )
    monkeypatch.setattr(
        library_router,
        "_latest_citation_card_quality_summary",
        lambda: {
            "available": True,
            "status": "good",
            "summary": {
                "tracked_checks": 1,
                "failed_checks": 0,
                "citation_card_failed": 0,
                "shelf_failed": 0,
                "ref_card_failed": 0,
                "system_b_failed": 0,
                "shelf_item_count": 0,
                "shelf_metadata_ready_count": 0,
                "shelf_export_ready_count": 0,
                "shelf_summary_export_ready_count": 0,
                "shelf_doi_count": 0,
                "shelf_source_clickable_count": 0,
                "shelf_review_count": 0,
            },
            "top_failures": [],
        },
    )
    monkeypatch.setattr(library_router, "_research_qa_rerun_history_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(library_router, "_latest_research_qa_failure_cases", lambda *args, **kwargs: [])

    client = TestClient(app)
    stale_response = client.post(
        "/api/library/quality/reader-locate",
        json={
            "source_path": "Reference / 8 / Removed Source",
            "source_name": "Removed Source",
            "locate_feedback_key": "stale-cite",
            "locate_request_id": 2,
            "status": "failed",
            "precision": "failed",
            "ok": False,
            "repairable": True,
            "strict_locate": False,
            "reason": "404 Not Found: {\"detail\":\"markdown not found for source\"}",
            "heading_path": "文献篮 / 已确认文献",
        },
    )
    assert stale_response.status_code == 200
    assert stale_response.json()["item"]["source_available"] is False
    assert stale_response.json()["item"]["recommended_action"] == "source_removed"

    rows_response = client.get("/api/library/quality/reader-locate", params={"limit": 5})
    assert rows_response.status_code == 200
    rows_payload = rows_response.json()
    assert rows_payload["items"] == []
    assert rows_payload["summary"]["available"] is False
    assert rows_payload["summary"]["summary"]["failed"] == 0

    overview_response = client.get("/api/library/quality/overview", params={"scope": "all"})
    assert overview_response.status_code == 200
    overview = overview_response.json()
    assert overview["reader_locate"]["available"] is False
    assert overview["reader_locate"]["summary"]["failed"] == 0
    priority_domains = {str(item.get("domain") or "") for item in list(overview.get("priority_actions") or [])}
    assert "reader_locate" not in priority_domains


def test_library_reader_locate_non_strict_fuzzy_match_is_informational(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdf_dir / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    md = md_dir / "source" / "source.en.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "# Source",
                "## Abstract",
                "This source contains fuzzy but usable evidence [1].",
                "## Discussion",
                "The reader locate event found a nearby paragraph.",
                "## References",
                "[1] Reference.",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_reader_locate_events_path", lambda: tmp_path / "reader_locate_events.jsonl")
    monkeypatch.setattr(library_router, "_bg_snapshot", lambda: {"running": False, "current": "", "queue": []})
    monkeypatch.setattr(
        library_router,
        "_latest_research_qa_quality_summary",
        lambda: {
            "available": True,
            "status": "good",
            "summary": {"total": 1, "passed": 1, "failed": 0},
            "top_failures": [],
        },
    )
    monkeypatch.setattr(
        library_router,
        "_latest_citation_card_quality_summary",
        lambda: {
            "available": True,
            "status": "good",
            "summary": {
                "tracked_checks": 1,
                "failed_checks": 0,
                "citation_card_failed": 0,
                "shelf_failed": 0,
                "ref_card_failed": 0,
                "system_b_failed": 0,
                "shelf_item_count": 0,
                "shelf_metadata_ready_count": 0,
                "shelf_export_ready_count": 0,
                "shelf_summary_export_ready_count": 0,
                "shelf_doi_count": 0,
                "shelf_source_clickable_count": 0,
                "shelf_review_count": 0,
            },
            "top_failures": [],
        },
    )
    monkeypatch.setattr(library_router, "_research_qa_rerun_history_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(library_router, "_latest_research_qa_failure_cases", lambda *args, **kwargs: [])

    client = TestClient(app)
    response = client.post(
        "/api/library/quality/reader-locate",
        json={
            "source_path": str(md),
            "source_name": "source.en.md",
            "locate_feedback_key": "fuzzy-cite",
            "locate_request_id": 2,
            "status": "fuzzy",
            "precision": "fuzzy",
            "ok": True,
            "repairable": False,
            "strict_locate": False,
            "reason": "Reader locate matched.",
            "heading_path": "Discussion",
        },
    )
    assert response.status_code == 200
    assert response.json()["item"]["source_available"] is True

    overview_response = client.get("/api/library/quality/overview", params={"scope": "all"})
    assert overview_response.status_code == 200
    overview = overview_response.json()
    reader_locate = overview["reader_locate"]
    assert reader_locate["available"] is True
    assert reader_locate["status"] == "good"
    assert reader_locate["summary"]["degraded"] == 1
    assert reader_locate["recommended_sources"] == []
    assert overview["domains"]["reader_locate"]["status"] == "good"
    priority_domains = {str(item.get("domain") or "") for item in list(overview.get("priority_actions") or [])}
    assert "reader_locate" not in priority_domains


def test_library_reader_locate_repair_run_verifies_anchor_targets(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    db_dir = tmp_path / "db"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdf_dir / "target.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    md = md_dir / "target" / "target.en.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(
        "\n".join(
            [
                '<!-- kb_page: 1 -->',
                '# Target Paper',
                '## Method',
                '<span id="a-method-1"></span>',
                '<p id="p-method-1">The repaired method paragraph is now addressable.</p>',
                '## References',
                '[1] Reference.',
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_reader_locate_events_path", lambda: tmp_path / "reader_locate_events.jsonl")
    monkeypatch.setattr(library_router, "_quality_repair_runs_path", lambda: tmp_path / "repair_runs.jsonl")
    monkeypatch.setattr(library_router, "_latest_research_qa_failure_cases", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        library_router,
        "_run_library_reindex",
        lambda: {
            "ok": True,
            "stdout": "ingest ok",
            "stderr": "",
            "structured_indices": {"version": 2, "scanned": 1, "rebuilt": 1, "skipped": 0, "failed": 0, "citation_mention_count": 1, "errors": []},
            "structured_indices_error": "",
            "refsync": {"started": True, "run_id": 99},
            "refsync_error": "",
        },
    )

    library_router._append_reader_locate_event(
        {
            "created_at": 100,
            "source_path": str(md),
            "source_name": "target.en.md",
            "locate_feedback_key": "cite-target-1",
            "locate_request_id": 1,
            "status": "failed",
            "precision": "failed",
            "ok": False,
            "repairable": True,
            "strict_locate": True,
            "reason": "target anchor was missing before repair",
            "block_id": "p-method-1",
            "anchor_id": "a-method-1",
            "heading_path": "Method",
        }
    )
    library_router._append_quality_repair_run(
        {
            "run_id": "reader-locate-run",
            "status": "reindex_pending",
            "phase": "reindex_pending",
            "created_at": 200,
            "updated_at": 200,
            "requested": 1,
            "enqueued": 0,
            "repaired": 1,
            "failed": 0,
            "skipped_busy": 0,
            "needs_reindex": True,
            "target_names": ["target.en.md"],
            "target_sources": [str(md)],
            "impact": {"requested": 1, "repaired": 1, "enqueued": 0, "needs_reindex": True},
            "detail": "Markdown source repair completed; index refresh is pending.",
        }
    )

    client = TestClient(app)
    response = client.post("/api/library/quality/repair-runs/reader-locate-run/advance")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["advanced"] is True
    assert payload["item"]["status"] == "completed"
    assert payload["item"]["phase"] == "verification_passed"
    assert payload["item"]["reindexed"] is True
    verification = payload["item"]["verification"]
    assert verification["type"] == "reader_locate_repair"
    assert verification["status"] == "passed"
    assert verification["quality_ok"] is True
    assert verification["target_count"] == 1
    assert verification["passed"] == 1
    assert verification["checked"][0]["status"] == "passed"
    assert verification["checked"][0]["checks"]["found_ids"] == ["a-method-1", "p-method-1"]


def test_library_quality_repair_reindexes_reader_locate_failure_when_markdown_is_good(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    db_dir = tmp_path / "db"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    md = md_dir / "ready" / "ready.en.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "# Ready Paper",
                "## Abstract",
                "This paper has enough text for a good conversion-quality assessment and source indexing.",
                "## Method",
                '<span id="a-method-1"></span>',
                '<p id="p-method-1">The method paragraph is already present but the reader locate index was stale.</p>',
                "## References",
                "[1] Ada Lovelace. Example reference. Journal of Testing, 2024.",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_reader_locate_events_path", lambda: tmp_path / "reader_locate_events.jsonl")
    monkeypatch.setattr(library_router, "_quality_repair_runs_path", lambda: tmp_path / "repair_runs.jsonl")
    monkeypatch.setattr(
        library_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=str(db_dir), library_db_path=str(tmp_path / "library.db")),
    )
    monkeypatch.setattr(library_router, "_bg_snapshot", lambda: {"running": False, "current": "", "queue": []})
    enqueued: list[dict] = []
    monkeypatch.setattr(library_router, "_bg_enqueue", lambda task: enqueued.append(dict(task or {})))
    monkeypatch.setattr(
        library_router,
        "_run_library_reindex",
        lambda: {
            "ok": True,
            "stdout": "ingest ok",
            "stderr": "",
            "structured_indices": {"version": 2, "scanned": 1, "rebuilt": 1, "skipped": 0, "failed": 0, "citation_mention_count": 0, "errors": []},
            "structured_indices_error": "",
            "refsync": {"started": True, "run_id": 7},
            "refsync_error": "",
        },
    )

    library_router._append_reader_locate_event(
        {
            "created_at": 100,
            "source_path": str(md),
            "source_name": "ready.en.md",
            "locate_feedback_key": "cite-ready-1",
            "locate_request_id": 1,
            "status": "failed",
            "precision": "failed",
            "ok": False,
            "repairable": True,
            "strict_locate": True,
            "reason": "source block index was stale",
            "block_id": "p-method-1",
            "anchor_id": "a-method-1",
            "heading_path": "Ready Paper / Method",
        }
    )

    client = TestClient(app)
    response = client.post(
        "/api/library/quality/repair",
        json={"sources": [{"source_path": str(md), "source_name": "ready.en.md"}], "md_autofix": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["enqueued"] == 0
    assert payload["repaired"] == 0
    assert payload["needs_reindex"] is True
    assert payload["impact"]["reader_locate_reindex"] == 1
    assert payload["repair_run"]["status"] == "reindex_pending"
    assert payload["repair_run"]["phase"] == "reindex_pending"
    assert enqueued == []
    item = payload["items"][0]
    assert item["ok"] is True
    assert item["reader_locate_reindex_required"] is True
    assert item["reader_locate_problem_count"] == 1
    assert item["planned_action"] == "reindex"
    assert item["planned_scope"] == "source_blocks"
    assert item["repair_attempt"]["event"] == "reader_locate_reindex_required"
    assert item["repair_attempt"]["status"] == "reindex_pending"

    advance = client.post(f"/api/library/quality/repair-runs/{payload['repair_run_id']}/advance")
    assert advance.status_code == 200
    advanced = advance.json()
    assert advanced["item"]["status"] == "completed"
    assert advanced["item"]["phase"] == "verification_passed"
    assert advanced["item"]["verification"]["type"] == "reader_locate_repair"
    assert advanced["item"]["verification"]["status"] == "passed"
    assert advanced["item"]["verification"]["checked"][0]["checks"]["found_ids"] == ["a-method-1", "p-method-1"]


def test_library_quality_repair_ignores_reader_locate_failure_after_newer_exact_open(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    db_dir = tmp_path / "db"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    md = md_dir / "ready" / "ready.en.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "# Ready Paper",
                "## Abstract",
                "This paper has enough text for a good conversion-quality assessment and source indexing.",
                "## Method",
                '<span id="a-method-1"></span>',
                '<p id="p-method-1">The method paragraph is already present and was later located exactly.</p>',
                "## References",
                "[1] Ada Lovelace. Example reference. Journal of Testing, 2024.",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_reader_locate_events_path", lambda: tmp_path / "reader_locate_events.jsonl")
    monkeypatch.setattr(library_router, "_quality_repair_runs_path", lambda: tmp_path / "repair_runs.jsonl")
    monkeypatch.setattr(
        library_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=str(db_dir), library_db_path=str(tmp_path / "library.db")),
    )
    monkeypatch.setattr(library_router, "_bg_snapshot", lambda: {"running": False, "current": "", "queue": []})
    enqueued: list[dict] = []
    monkeypatch.setattr(library_router, "_bg_enqueue", lambda task: enqueued.append(dict(task or {})))

    base_event = {
        "source_path": str(md),
        "source_name": "ready.en.md",
        "locate_feedback_key": "cite-ready-1",
        "locate_request_id": 1,
        "strict_locate": True,
        "block_id": "p-method-1",
        "anchor_id": "a-method-1",
        "heading_path": "Ready Paper / Method",
    }
    library_router._append_reader_locate_event(
        {
            **base_event,
            "created_at": 100,
            "status": "failed",
            "precision": "failed",
            "ok": False,
            "repairable": True,
            "reason": "source block index was stale",
        }
    )
    library_router._append_reader_locate_event(
        {
            **base_event,
            "created_at": 200,
            "status": "exact",
            "precision": "exact_anchor",
            "ok": True,
            "repairable": False,
            "reason": "reader reopen verified the exact anchor",
        }
    )

    client = TestClient(app)
    response = client.post(
        "/api/library/quality/repair",
        json={"sources": [{"source_path": str(md), "source_name": "ready.en.md"}], "md_autofix": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["enqueued"] == 0
    assert payload["repaired"] == 0
    assert payload["needs_reindex"] is False
    assert payload["impact"]["reader_locate_reindex"] == 0
    assert payload["repair_run"]["status"] == "completed"
    assert payload["repair_run"]["phase"] == "repair_complete"
    assert enqueued == []
    item = payload["items"][0]
    assert item["ok"] is True
    assert "reader_locate_reindex_required" not in item
    assert "reader_locate_problem_count" not in item
    assert item["repair_attempt"]["event"] == "repair_closed"


def test_library_quality_repair_route_enqueues_resolved_sources(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    direct_pdf = pdf_dir / "direct.pdf"
    source_pdf = pdf_dir / "source.pdf"
    for path in (direct_pdf, source_pdf):
        path.write_bytes(b"%PDF-1.4 test")

    source_md = md_dir / "source" / "source.en.md"
    source_md.parent.mkdir(parents=True, exist_ok=True)
    source_md.write_text("# Source\n", encoding="utf-8")

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(
        library_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=str(tmp_path / "db"), library_db_path=str(tmp_path / "library.db")),
    )
    monkeypatch.setattr(library_router, "_bg_snapshot", lambda: {"running": False, "current": "", "queue": []})
    monkeypatch.setattr(
        library_router,
        "_build_bg_task",
        lambda **kwargs: {
            "_tid": f"task-{Path(kwargs['pdf_path']).name}",
            "name": Path(kwargs["pdf_path"]).name,
            "pdf": str(kwargs["pdf_path"]),
            "replace": kwargs.get("replace"),
            "speed_mode": kwargs.get("speed_mode"),
            "no_llm": kwargs.get("no_llm"),
            "repair_context": kwargs.get("repair_context"),
        },
    )
    enqueued: list[dict] = []
    monkeypatch.setattr(library_router, "_bg_enqueue", lambda task: enqueued.append(dict(task or {})))

    client = TestClient(app)
    response = client.post(
        "/api/library/quality/repair",
        json={
            "pdf_names": ["direct.pdf"],
            "sources": [{"source_path": str(source_md), "source_name": "source.en.md"}],
            "speed_mode": "no_llm",
            "replace": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["repair_run_id"]
    assert payload["repair_run"]["status"] == "queued"
    assert payload["repair_run"]["phase"] == "source_reconversion_queued"
    assert payload["requested"] == 2
    assert payload["enqueued"] == 2
    assert payload["repaired"] == 1
    assert payload["needs_reindex"] is True
    assert payload["impact"]["repaired"] == 1
    assert payload["impact"]["enqueued"] == 2
    assert payload["impact"]["before_avg_score"] < payload["impact"]["after_avg_score"]
    assert payload["skipped_busy"] == 0
    assert {item.get("pdf_name") for item in payload["items"] if item.get("enqueued")} == {"direct.pdf", "source.pdf"}
    source_item = next(item for item in payload["items"] if item.get("pdf_name") == "source.pdf")
    assert source_item["repair_changed"] is True
    assert "ensure_page_anchor" in source_item["repair_applied"]
    assert source_item["repair_before_score"] < source_item["repair_after_score"]
    assert "missing_page_markers" in source_item["fixed_issue_codes"]
    assert source_item["quality_before"]["status"] == "warning"
    assert source_item["quality_after"]["status"] == "warning"
    assert source_item["remaining_issue_codes"]
    assert source_item["repair_plan"]["action"] == "reconvert"
    assert source_item["planned_scope"] == "document"
    assert source_item["planned_speed_mode"] == "no_llm"
    assert source_item["planned_no_llm"] is True
    assert source_item["repair_attempt"]["event"] == "reconvert_queued"
    assert source_item["repair_attempt"]["status"] == "queued"
    assert source_md.read_text(encoding="utf-8").lstrip().startswith("<!-- kb_page: 1 -->")
    assert len(enqueued) == 2
    assert {str(task.get("name") or "") for task in enqueued} == {"direct.pdf", "source.pdf"}
    assert {str(task.get("speed_mode") or "") for task in enqueued} == {"no_llm"}
    assert all(bool(task.get("no_llm")) for task in enqueued)
    assert all(isinstance(task.get("repair_context"), dict) for task in enqueued)
    assert {str((task.get("repair_context") or {}).get("repair_run_id") or "") for task in enqueued} == {payload["repair_run_id"]}

    run_response = client.get(f"/api/library/quality/repair-runs/{payload['repair_run_id']}")
    assert run_response.status_code == 200
    assert run_response.json()["item"]["enqueued"] == 2


def test_library_quality_repair_route_autofixes_markdown_without_pdf(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    md_path = md_dir / "standalone" / "standalone.en.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        "\n".join(
            [
                "# Standalone Markdown",
                "",
                "## Abstract",
                "",
                "This converted paper already has enough structure, references, and readable text for indexing.",
                "",
                "## Method",
                "",
                "The method section remains usable after a deterministic source-level repair.",
                "",
                "## References",
                "",
                "[1] Ada Lovelace. Example reference. Journal of Testing, 2024.",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(
        library_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=str(tmp_path / "db"), library_db_path=str(tmp_path / "library.db")),
    )
    monkeypatch.setattr(library_router, "_bg_snapshot", lambda: {"running": False, "current": "", "queue": []})
    enqueued: list[dict] = []
    monkeypatch.setattr(library_router, "_bg_enqueue", lambda task: enqueued.append(dict(task or {})))

    client = TestClient(app)
    response = client.post(
        "/api/library/quality/repair",
        json={
            "sources": [{"source_path": str(md_path), "source_name": "standalone.en.md"}],
            "md_autofix": True,
            "replace": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["repair_run_id"]
    assert payload["repair_run"]["status"] == "reindex_pending"
    assert payload["repair_run"]["phase"] == "reindex_pending"
    assert payload["requested"] == 1
    assert payload["enqueued"] == 0
    assert payload["repaired"] == 1
    assert payload["needs_reindex"] is True
    assert payload["impact"]["repaired"] == 1
    assert payload["impact"]["enqueued"] == 0
    assert enqueued == []

    item = payload["items"][0]
    assert item["ok"] is True
    assert item["pdf_path"] == ""
    assert item["md_path"] == str(md_path)
    assert item["repair_changed"] is True
    assert "ensure_page_anchor" in item["repair_applied"]
    assert "missing_page_markers" in item["fixed_issue_codes"]
    assert item["quality_before"]["status"] == "warning"
    assert item["quality_after"]["status"] == "good"
    assert item["repair_plan"]["action"] == "none"
    assert item["repair_attempt"]["event"] == "repair_closed"
    assert md_path.read_text(encoding="utf-8").lstrip().startswith("<!-- kb_page: 1 -->")

    list_response = client.get("/api/library/quality/repair-runs", params={"limit": 5})
    assert list_response.status_code == 200
    rows = list_response.json()["items"]
    assert any(row["run_id"] == payload["repair_run_id"] for row in rows)

    update_response = client.post(
        f"/api/library/quality/repair-runs/{payload['repair_run_id']}",
        json={"status": "completed", "phase": "reindex_complete", "reindexed": True},
    )
    assert update_response.status_code == 200
    updated = update_response.json()["item"]
    assert updated["status"] == "completed"
    assert updated["phase"] == "reindex_complete"
    assert updated["reindexed"] is True


def test_library_quality_repair_run_advance_reindexes_pending_run(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    md_dir = tmp_path / "md_output"
    pdf_dir = tmp_path / "pdfs"
    db_dir = tmp_path / "db"
    ingest_py = tmp_path / "ingest.py"
    md_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)
    ingest_py.write_text("print('ingest ok')\n", encoding="utf-8")

    class FakeProc:
        returncode = 0
        stdout = "ingest ok\n"
        stderr = ""

    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_ingest_py_path", lambda: ingest_py)
    monkeypatch.setattr(library_router, "_quality_repair_runs_path", lambda: tmp_path / "repair_runs.jsonl")
    monkeypatch.setattr(
        library_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=str(db_dir), library_db_path=str(tmp_path / "library.db")),
    )
    monkeypatch.setattr(library_router.subprocess, "run", lambda *args, **kwargs: FakeProc())
    monkeypatch.setattr(library_router, "_bg_snapshot", lambda: {"running": False, "current": "", "queue": []})
    monkeypatch.setattr(
        library_router,
        "rebuild_structured_indices_for_root",
        lambda src_root, **kwargs: {
            "version": 2,
            "scanned": 1,
            "rebuilt": 1,
            "skipped": 0,
            "failed": 0,
            "citation_mention_count": 2,
            "errors": [],
        },
    )
    monkeypatch.setattr(library_router, "start_reference_sync", lambda **kwargs: {"started": True, "run_id": 12})

    library_router._append_quality_repair_run({
        "run_id": "run-md-advance",
        "status": "reindex_pending",
        "phase": "reindex_pending",
        "requested": 1,
        "enqueued": 0,
        "repaired": 1,
        "failed": 0,
        "skipped_busy": 0,
        "needs_reindex": True,
        "target_names": ["standalone.en.md"],
        "target_sources": [str(md_dir / "standalone" / "standalone.en.md")],
        "impact": {"requested": 1, "repaired": 1, "enqueued": 0, "needs_reindex": True},
        "detail": "Markdown source repair completed; index refresh is pending.",
    })

    client = TestClient(app)
    response = client.post("/api/library/quality/repair-runs/run-md-advance/advance")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["advanced"] is True
    assert payload["waiting"] is False
    assert payload["item"]["status"] == "completed"
    assert payload["item"]["phase"] == "reindex_complete"
    assert payload["item"]["reindexed"] is True
    assert payload["item"]["impact"]["reindexed"] is True
    assert payload["reindex"]["ok"] is True
    assert payload["reindex"]["structured_indices"]["rebuilt"] == 1
    assert payload["reindex"]["refsync"]["started"] is True


def test_library_quality_repair_run_advance_waits_for_conversion_then_reindexes(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    md_dir = tmp_path / "md_output"
    pdf_dir = tmp_path / "pdfs"
    db_dir = tmp_path / "db"
    ingest_py = tmp_path / "ingest.py"
    md_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)
    ingest_py.write_text("print('ingest ok')\n", encoding="utf-8")
    pdf_path = pdf_dir / "queued.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")
    snap_state = {"active": True}

    class FakeProc:
        returncode = 0
        stdout = "ingest ok\n"
        stderr = ""

    def fake_snapshot():
        if snap_state["active"]:
            return {
                "running": False,
                "current": "",
                "queue": [{"pdf": str(pdf_path), "name": "queued.pdf", "replace": True, "_tid": "task-1"}],
            }
        return {"running": False, "current": "", "queue": []}

    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_ingest_py_path", lambda: ingest_py)
    monkeypatch.setattr(library_router, "_quality_repair_runs_path", lambda: tmp_path / "repair_runs.jsonl")
    monkeypatch.setattr(
        library_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=str(db_dir), library_db_path=str(tmp_path / "library.db")),
    )
    monkeypatch.setattr(library_router.subprocess, "run", lambda *args, **kwargs: FakeProc())
    monkeypatch.setattr(library_router, "_bg_snapshot", fake_snapshot)
    monkeypatch.setattr(
        library_router,
        "rebuild_structured_indices_for_root",
        lambda src_root, **kwargs: {
            "version": 2,
            "scanned": 1,
            "rebuilt": 1,
            "skipped": 0,
            "failed": 0,
            "citation_mention_count": 2,
            "errors": [],
        },
    )
    monkeypatch.setattr(library_router, "start_reference_sync", lambda **kwargs: {"started": True, "run_id": 13})

    library_router._append_quality_repair_run({
        "run_id": "run-queued-advance",
        "status": "queued",
        "phase": "source_reconversion_queued",
        "requested": 1,
        "enqueued": 1,
        "repaired": 0,
        "failed": 0,
        "skipped_busy": 0,
        "needs_reindex": True,
        "target_names": ["queued.pdf"],
        "target_sources": [str(pdf_path)],
        "impact": {"requested": 1, "repaired": 0, "enqueued": 1, "needs_reindex": True},
        "detail": "Source reconversion queued; index refresh should run after conversion.",
    })

    client = TestClient(app)
    waiting_response = client.post("/api/library/quality/repair-runs/run-queued-advance/advance")
    assert waiting_response.status_code == 200
    waiting_payload = waiting_response.json()
    assert waiting_payload["ok"] is True
    assert waiting_payload["waiting"] is True
    assert waiting_payload["advanced"] is False
    assert waiting_payload["item"]["phase"] == "source_reconversion_queued"
    assert waiting_payload["reindex"] is None

    md_path = md_dir / "queued" / "queued.en.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "# Queued Paper",
                "",
                "## Abstract",
                "",
                "This converted paper has a usable abstract and cites prior work [1].",
                "",
                "## Method",
                "",
                "The method section remains indexable after reconversion.",
                "",
                "## References",
                "",
                "[1] Ada Lovelace. Example reference. Journal, 2024.",
            ]
        ),
        encoding="utf-8",
    )
    snap_state["active"] = False
    resume_response = client.post("/api/library/quality/repair-runs/run-queued-advance/advance")
    assert resume_response.status_code == 200
    resume_payload = resume_response.json()
    assert resume_payload["ok"] is True
    assert resume_payload["advanced"] is True
    assert resume_payload["waiting"] is False
    assert resume_payload["item"]["status"] == "completed"
    assert resume_payload["item"]["phase"] == "reindex_complete"
    assert resume_payload["item"]["reindexed"] is True
    assert resume_payload["item"]["verification"]["type"] == "conversion_source_quality"
    assert resume_payload["item"]["verification"]["quality_ok"] is True
    assert resume_payload["item"]["verification"]["ready"] == 1
    assert resume_payload["reindex"]["ok"] is True


def test_library_quality_repair_run_advance_verifies_matching_qa_case(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    md_dir = tmp_path / "md_output"
    pdf_dir = tmp_path / "pdfs"
    db_dir = tmp_path / "db"
    ingest_py = tmp_path / "ingest.py"
    md_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)
    ingest_py.write_text("print('ingest ok')\n", encoding="utf-8")
    md_path = md_dir / "scinerf" / "scinerf.en.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("# SCINeRF\n", encoding="utf-8")

    class FakeProc:
        returncode = 0
        stdout = "ingest ok\n"
        stderr = ""

    rerun_calls: list[str] = []

    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_ingest_py_path", lambda: ingest_py)
    monkeypatch.setattr(library_router, "_quality_repair_runs_path", lambda: tmp_path / "repair_runs.jsonl")
    monkeypatch.setattr(
        library_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=str(db_dir), library_db_path=str(tmp_path / "library.db")),
    )
    monkeypatch.setattr(library_router.subprocess, "run", lambda *args, **kwargs: FakeProc())
    monkeypatch.setattr(library_router, "_bg_snapshot", lambda: {"running": False, "current": "", "queue": []})
    monkeypatch.setattr(
        library_router,
        "rebuild_structured_indices_for_root",
        lambda src_root, **kwargs: {
            "version": 2,
            "scanned": 1,
            "rebuilt": 1,
            "skipped": 0,
            "failed": 0,
            "citation_mention_count": 2,
            "errors": [],
        },
    )
    monkeypatch.setattr(library_router, "start_reference_sync", lambda **kwargs: {"started": True, "run_id": 14})
    monkeypatch.setattr(
        library_router,
        "_latest_research_qa_failure_cases",
        lambda **kwargs: [
            {
                "id": "scinerf-admm-origin",
                "source_diagnostics": [
                    {
                        "source_name": "SCINeRF",
                        "md_path": str(md_path),
                        "source_path": str(md_path),
                    }
                ],
            }
        ],
    )

    def fake_run_research_qa_case(**kwargs):
        rerun_calls.append(str(kwargs.get("case_id") or ""))
        return {
            "ok": True,
            "case_id": str(kwargs.get("case_id") or ""),
            "status": "passed",
            "quality_ok": True,
            "failures": [],
            "report_path": str(tmp_path / "report.md"),
            "raw_path": str(tmp_path / "raw_results.jsonl"),
            "finished_at": 1790000700,
        }

    monkeypatch.setattr(library_router, "_run_research_qa_case", fake_run_research_qa_case)

    library_router._append_quality_repair_run({
        "run_id": "run-verify-advance",
        "status": "reindex_pending",
        "phase": "reindex_pending",
        "requested": 1,
        "enqueued": 0,
        "repaired": 1,
        "failed": 0,
        "skipped_busy": 0,
        "needs_reindex": True,
        "target_names": ["scinerf.en.md"],
        "target_sources": [str(md_path)],
        "impact": {"requested": 1, "repaired": 1, "enqueued": 0, "needs_reindex": True},
        "detail": "Markdown source repair completed; index refresh is pending.",
    })

    client = TestClient(app)
    response = client.post("/api/library/quality/repair-runs/run-verify-advance/advance")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["advanced"] is True
    assert payload["item"]["status"] == "completed"
    assert payload["item"]["phase"] == "verification_passed"
    assert payload["item"]["reindexed"] is True
    assert payload["item"]["verification"]["case_id"] == "scinerf-admm-origin"
    assert payload["item"]["verification"]["quality_ok"] is True
    assert rerun_calls == ["scinerf-admm-origin"]


def test_library_quality_repair_route_skips_busy_pdf(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    busy_pdf = pdf_dir / "busy.pdf"
    busy_pdf.write_bytes(b"%PDF-1.4 test")

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(
        library_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=str(tmp_path / "db"), library_db_path=str(tmp_path / "library.db")),
    )
    monkeypatch.setattr(
        library_router,
        "_bg_snapshot",
        lambda: {
            "running": False,
            "current": "",
            "queue": [{"pdf": str(busy_pdf), "name": "busy.pdf", "replace": True, "_tid": "q1"}],
        },
    )
    enqueued: list[dict] = []
    monkeypatch.setattr(library_router, "_bg_enqueue", lambda task: enqueued.append(dict(task or {})))

    client = TestClient(app)
    response = client.post("/api/library/quality/repair", json={"pdf_names": ["busy.pdf"]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["requested"] == 1
    assert payload["enqueued"] == 0
    assert payload["skipped_busy"] == 1
    assert payload["items"][0]["skipped_busy"] is True
    assert enqueued == []


def test_convert_pending_enqueues_only_idle_pending(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf_a = pdf_dir / "a.pdf"
    pdf_b = pdf_dir / "b.pdf"
    pdf_c = pdf_dir / "c.pdf"
    for p in (pdf_a, pdf_b, pdf_c):
        p.write_bytes(b"%PDF-1.4 test")

    md_b = md_dir / "b" / "b.en.md"
    md_b.parent.mkdir(parents=True, exist_ok=True)
    md_b.write_text("# b\n", encoding="utf-8")

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(
        library_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=str(tmp_path / "db"), library_db_path=str(tmp_path / "library.db")),
    )
    monkeypatch.setattr(
        library_router,
        "_bg_snapshot",
        lambda: {
            "running": False,
            "current": "",
            "queue": [
                {"pdf": str(pdf_a), "name": "a.pdf", "replace": False, "_tid": "q1"},
            ],
            "done": 0,
            "total": 1,
        },
    )

    enqueued: list[dict] = []
    monkeypatch.setattr(library_router, "_bg_enqueue", lambda task: enqueued.append(dict(task or {})))

    client = TestClient(app)
    response = client.post("/api/library/convert/pending", json={"speed_mode": "ultra_fast"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["enqueued"] == 1
    assert payload["skipped_busy"] == 1
    assert len(enqueued) == 1
    assert str(enqueued[0].get("name") or "") == "c.pdf"
    assert str(enqueued[0].get("speed_mode") or "") == "ultra_fast"
    assert bool(enqueued[0].get("replace")) is True


def test_delete_library_file_route_deletes_pdf_and_md(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf_d = pdf_dir / "d.pdf"
    pdf_d.write_bytes(b"%PDF-1.4 test")
    md_d = md_dir / "d" / "d.en.md"
    md_d.parent.mkdir(parents=True, exist_ok=True)
    md_d.write_text("# d\n", encoding="utf-8")

    class FakeStore:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        def delete_by_path(self, path: Path) -> int:
            self.deleted.append(str(path))
            return 1

    fake_store = FakeStore()

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_library_store", lambda: fake_store)
    monkeypatch.setattr(library_router, "_bg_snapshot", lambda: {"running": False, "current": ""})
    monkeypatch.setattr(library_router, "_bg_remove_queued_tasks_for_pdf", lambda path: 2)

    client = TestClient(app)
    response = client.post("/api/library/file/delete", json={"pdf_name": "d.pdf", "also_md": True})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["pdf_deleted"] is True
    assert payload["md_deleted"] is True
    assert payload["removed_queued"] == 2
    assert not pdf_d.exists()
    assert not md_d.parent.exists()
    assert fake_store.deleted == [str(pdf_d)]


def test_delete_library_file_route_blocks_any_active_task(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf_d = pdf_dir / "d.pdf"
    pdf_d.write_bytes(b"%PDF-1.4 test")

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(
        library_router,
        "_bg_snapshot",
        lambda: {
            "running": True,
            "current": "other.pdf",
            "active_count": 2,
            "active_tasks": [
                {"_tid": "r1", "pdf": str(pdf_d), "name": "d.pdf", "replace": False},
                {"_tid": "r2", "pdf": str(pdf_dir / "x.pdf"), "name": "x.pdf", "replace": False},
            ],
            "queue": [],
        },
    )

    client = TestClient(app)
    response = client.post("/api/library/file/delete", json={"pdf_name": "d.pdf", "also_md": True})
    assert response.status_code == 409
    assert "currently converting" in response.text


def test_open_library_file_route_opens_md_target(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf_e = pdf_dir / "e.pdf"
    pdf_e.write_bytes(b"%PDF-1.4 test")
    md_e = md_dir / "e" / "e.en.md"
    md_e.parent.mkdir(parents=True, exist_ok=True)
    md_e.write_text("# e\n", encoding="utf-8")

    opened: list[str] = []
    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "open_in_explorer", lambda path: opened.append(str(path)))

    client = TestClient(app)
    response = client.post("/api/library/file/open", json={"pdf_name": "e.pdf", "target": "md"})
    assert response.status_code == 200
    assert opened == [str(md_e)]


def test_open_library_file_route_opens_dir_without_pdf_name(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    opened: list[str] = []
    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "open_in_explorer", lambda path: opened.append(str(path)))

    client = TestClient(app)
    response = client.post("/api/library/file/open", json={"pdf_name": "", "target": "pdf_dir"})
    assert response.status_code == 200
    assert opened == [str(pdf_dir)]


def test_open_quality_artifact_route_opens_latest_report(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    eval_root = tmp_path / "test_results" / "research_qa_eval"
    run_dir = eval_root / "20260101-120000"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = run_dir / "summary.json"
    raw = run_dir / "raw_results.jsonl"
    report = run_dir / "report.md"
    summary.write_text(json.dumps({"total": 1, "passed": 0, "failed": 1}), encoding="utf-8")
    raw.write_text('{"id":"case-1","quality":{"ok":false,"failures":[{"name":"citation_card_quality"}]}}\n', encoding="utf-8")
    report.write_text("# report\n", encoding="utf-8")

    opened: list[str] = []
    monkeypatch.setattr(library_router, "_RESEARCH_QA_EVAL_ROOT", eval_root)
    monkeypatch.setattr(library_router, "open_in_explorer", lambda path: opened.append(str(path)))

    client = TestClient(app)
    response = client.post("/api/library/quality/artifact/open", json={"domain": "research_qa", "target": "report"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["target"] == "report"
    assert payload["path"] == str(report)
    assert opened == [str(report)]


def test_quality_overview_includes_research_qa_failure_cases(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    eval_root = tmp_path / "test_results" / "research_qa_eval"
    run_dir = eval_root / "20260101-120000"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"total": 2, "passed": 1, "failed": 1}),
        encoding="utf-8",
    )
    (eval_root / "rerun_history.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case_id": "case-a",
                        "status": "failed",
                        "quality_ok": False,
                        "failures": [{"name": "refs_include_required_docs", "detail": "paper-a"}],
                        "finished_at": 1790000100,
                        "latency_ms": 2100,
                        "report_path": str(run_dir / "report.md"),
                        "raw_path": str(run_dir / "raw_results.jsonl"),
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "case_id": "case-a",
                        "status": "passed",
                        "quality_ok": True,
                        "failures": [],
                        "finished_at": 1790000000,
                        "latency_ms": 1900,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "raw_results.jsonl").write_text(
        json.dumps(
            {
                "id": "case-a",
                "question": "Why does the citation fail?",
                "status": "done",
                "latency_ms": 123.4,
                "expected": {
                    "requiredRefDocIds": ["paper-a"],
                    "requiredCitationDocIds": ["paper-b"],
                },
                "assistant_message": {
                    "cite_details": [
                        {
                            "num": 1,
                            "anchor": "cite-a1",
                            "source_name": "Paper B",
                            "source_path": "paper-b.md",
                            "title": "Paper B title",
                            "heading_path": "Introduction",
                            "evidence_quote": "The answer quotes Paper B.",
                        },
                        {
                            "num": 2,
                            "anchor": "cite-b1",
                            "is_inpaper": True,
                            "source_name": "Paper A",
                            "source_path": "paper-a.md",
                            "title": "Paper A upstream",
                            "heading_path": "Related work",
                            "citation_context": "Paper A is discussed by Paper B.",
                            "raw": "Author A. Paper A upstream. Journal A, 2024. doi:10.1234/paper-a",
                            "summary_line": "Paper A is the upstream reference discussed by Paper B for this answer.",
                        },
                    ],
                },
                "refs_payload": {
                    "hits": [
                        {
                            "text": "Paper C retrieval snippet.",
                            "meta": {
                                "source_path": "paper-c.md",
                                "heading_path": "Methods",
                                "ref_pack_state": "ready",
                            },
                            "ui_meta": {
                                "display_name": "Paper C",
                                "score": 9.1,
                                "summary_line": "Paper C summary.",
                                "why_line": "Paper C was retrieved for the question.",
                                "polish_status": "full",
                            },
                        }
                    ]
                },
                "quality": {
                    "ok": False,
                    "failures": [
                        {"name": "citation_card_quality", "detail": ["missing source title"]},
                        {"name": "refs_include_required_docs", "detail": ["paper-a"]},
                    ],
                    "citation_quality": {
                        "failures": [
                            {
                                "index": 2,
                                "name": "system_b_missing_citing_context",
                                "field": "citation_context",
                                "detail": "missing context",
                                "severity": "error",
                            }
                        ],
                        "warnings": [],
                    },
                    "citation_shelf_quality": {
                        "count": 2,
                        "metadata_ready_count": 0,
                        "export_ready_count": 0,
                        "summary_export_ready_count": 1,
                        "doi_count": 1,
                        "source_clickable_count": 2,
                        "review_count": 1,
                        "failures": [
                            {
                                "index": 1,
                                "name": "shelf_template_phrase_visible",
                                "field": "summary",
                                "detail": "No summary available",
                                "severity": "error",
                            }
                        ],
                        "warnings": [
                            {
                                "index": 2,
                                "name": "shelf_missing_doi",
                                "field": "doi",
                                "severity": "warning",
                            }
                        ],
                    },
                    "ref_card_quality": {
                        "failures": [
                            {
                                "index": 1,
                                "name": "ref_card_summary_too_short",
                                "field": "summary_line",
                                "detail": "Paper C summary.",
                                "severity": "error",
                            }
                        ],
                        "warnings": [],
                    },
                    "system_b_audit": {
                        "needs_review_count": 1,
                        "answer_context_only_count": 0,
                        "reference_index_fallback_count": 0,
                    },
                    "ref_doc_ids": ["paper-c"],
                    "citation_doc_ids": ["paper-b"],
                    "citation_count": 2,
                    "system_b_count": 1,
                    "ref_hit_count": 3,
                    "answer_preview": "preview",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (eval_root / "action_history.jsonl").write_text(
        json.dumps(
            {
                "id": "hist-1",
                "stage_key": "retrieval",
                "stage_label": "Retrieval coverage",
                "action": "rebuild_index",
                "status": "success",
                "summary": "Rebuilt retrieval index",
                "detail": "Next QA check: case-a",
                "target_ids": ["case-a"],
                "metrics": {"target_count": 1},
                "created_at": 1790000200,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeStore:
        def list_records_by_paths(self, paths):
            return {}

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_library_store", lambda: FakeStore())
    monkeypatch.setattr(library_router, "_bg_snapshot", lambda: {"running": False, "current": "", "queue": []})
    monkeypatch.setattr(library_router, "_RESEARCH_QA_EVAL_ROOT", eval_root)

    client = TestClient(app)
    response = client.get("/api/library/quality/overview", params={"scope": "all"})
    assert response.status_code == 200
    payload = response.json()
    failure_cases = list(payload.get("failure_cases") or [])
    assert len(failure_cases) == 1
    case = failure_cases[0]
    assert case["id"] == "case-a"
    assert case["question"] == "Why does the citation fail?"
    assert case["doc_ids"] == ["paper-a", "paper-b", "paper-c"]
    assert case["missing_expected_doc_ids"] == ["paper-a"]
    assert case["citation_count"] == 2
    assert case["diagnostic_summary"]["citation_routes"] == {"system_a": 1, "system_b": 1}
    assert case["diagnostic_summary"]["citation_card_failure_count"] == 1
    assert case["diagnostic_summary"]["shelf_failure_count"] == 1
    assert case["diagnostic_summary"]["shelf_warning_count"] == 1
    assert case["diagnostic_summary"]["shelf_metadata_ready_count"] == 0
    assert case["diagnostic_summary"]["shelf_export_ready_count"] == 0
    assert case["diagnostic_summary"]["shelf_summary_export_ready_count"] == 1
    assert case["diagnostic_summary"]["shelf_doi_count"] == 1
    assert case["diagnostic_summary"]["shelf_source_clickable_count"] == 2
    assert case["diagnostic_summary"]["shelf_review_count"] == 1
    assert case["diagnostic_summary"]["shelf_metadata_repair_target_count"] == 1
    assert case["diagnostic_summary"]["shelf_missing_export_fields"] == [{"name": "doi", "count": 1}]
    assert case["diagnostic_summary"]["ref_card_failure_count"] == 1
    assert case["diagnostic_summary"]["system_b_needs_review_count"] == 1
    assert payload["domains"]["citation_cards"]["summary"]["shelf_item_count"] == 2
    assert payload["domains"]["citation_cards"]["summary"]["shelf_export_ready_count"] == 0
    assert payload["domains"]["citation_cards"]["summary"]["shelf_summary_export_ready_count"] == 1
    assert case["citation_diagnostics"][0]["route"] == "system_a"
    assert case["citation_diagnostics"][0]["title"] == "Paper B title"
    assert case["citation_diagnostics"][0]["shelf_quality_issues"][0]["name"] == "shelf_template_phrase_visible"
    assert case["citation_diagnostics"][1]["route"] == "system_b"
    assert case["citation_diagnostics"][1]["quality_issues"][0]["name"] == "system_b_missing_citing_context"
    assert case["citation_diagnostics"][1]["shelf_quality_issues"][0]["name"] == "shelf_missing_doi"
    assert case["citation_diagnostics"][1]["metadata_missing_fields"] == ["doi"]
    assert case["citation_diagnostics"][1]["metadata_repairable"] is True
    assert "10.1234/paper-a" in case["citation_diagnostics"][1]["raw"]
    assert case["shelf_metadata_missing_fields"] == [{"name": "doi", "count": 1}]
    assert case["shelf_metadata_repair_targets"][0]["repair_target_kind"] == "system_b_citation"
    assert case["shelf_metadata_repair_targets"][0]["metadata_missing_fields"] == ["doi"]
    assert case["ref_diagnostics"][0]["title"] == "Paper C"
    assert case["ref_diagnostics"][0]["summary_line"] == "Paper C summary."
    assert case["ref_diagnostics"][0]["quality_issues"][0]["name"] == "ref_card_summary_too_short"
    assert case["rerun_status"]["available"] is True
    assert case["rerun_status"]["last_status"] == "failed"
    assert case["rerun_status"]["last_passed_at"] == 1790000000
    assert case["rerun_status"]["consecutive_failed"] == 1
    assert payload["rerun_summary"]["total"] == 2
    assert payload["rerun_summary"]["passed"] == 1
    assert payload["rerun_summary"]["failed"] == 1
    full_chain = payload["full_chain"]
    assert full_chain["status"] == "error"
    stage_keys = {str(item.get("key") or "") for item in list(full_chain.get("stages") or [])}
    assert {"research_qa", "retrieval", "citations", "repair_loop"}.issubset(stage_keys)
    full_chain_root_codes = {str(item.get("code") or "") for item in list(full_chain.get("root_causes") or [])}
    assert "retrieval_missing_expected_docs" in full_chain_root_codes
    assert "citation_card_quality" in full_chain_root_codes
    assert full_chain["action_history"][0]["summary"] == "Rebuilt retrieval index"
    feature_health = payload["feature_health"]
    feature_keys = {str(item.get("key") or "") for item in list(feature_health.get("items") or [])}
    assert "paper_guide" in feature_keys
    assert "literature_basket" in feature_keys
    source_titles = {str(item.get("title") or item.get("source_name") or "") for item in case.get("source_diagnostics") or []}
    assert {"Paper B title", "Paper A upstream", "Paper C"}.issubset(source_titles)
    root_codes = {str(item.get("code") or "") for item in case.get("root_causes") or []}
    assert "retrieval_missing_expected_docs" in root_codes
    assert "citation_card_quality" in root_codes
    assert "shelf_metadata_export_fields" in root_codes
    action_kinds = {str(item.get("kind") or "") for item in case.get("repair_actions") or []}
    assert {"apply_repair_plan", "open_replay", "rebuild_index", "open_artifact"}.issubset(action_kinds)
    repair_plan = next(item for item in case.get("repair_actions") or [] if item.get("kind") == "apply_repair_plan")
    repair_step_kinds = [str(item.get("kind") or "") for item in repair_plan.get("steps") or []]
    assert repair_step_kinds == ["repair_shelf_metadata", "rebuild_index", "rerun_case"]
    assert repair_plan["steps"][0]["target_count"] == 1
    assert repair_plan["steps"][0]["missing_fields"] == [{"name": "doi", "count": 1}]
    assert "Rerun QA acceptance" in str(repair_plan.get("detail") or "")
    assert "metadata fields doi x1" in str(repair_plan.get("detail") or "")
    failures = list(case.get("failures") or [])
    assert failures[0]["name"] == "citation_card_quality"
    assert failures[0]["domain"] == "citation_cards"
    assert failures[1]["domain"] == "research_qa"


def test_research_qa_rerun_route_runs_single_case(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    eval_root = tmp_path / "test_results" / "research_qa_eval"
    run_dir = eval_root / "20260101_130000"
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["cwd"] = kwargs.get("cwd")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(
            json.dumps({"total": 1, "passed": 1, "failed": 0, "output_dir": str(run_dir)}),
            encoding="utf-8",
        )
        (run_dir / "raw_results.jsonl").write_text(
            json.dumps(
                {
                    "id": "case-a",
                    "quality": {"ok": True, "failures": []},
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "report.md").write_text("# report\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout=f"[OK] research QA eval finished: {run_dir}\n", stderr="")

    monkeypatch.setattr(library_router, "_RESEARCH_QA_EVAL_ROOT", eval_root)
    monkeypatch.setattr(library_router.subprocess, "run", fake_run)

    client = TestClient(app)
    response = client.post("/api/library/quality/research-qa/rerun", json={"case_id": "case-a", "base_url": "http://127.0.0.1:8005"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["status"] == "passed"
    assert payload["quality_ok"] is True
    assert payload["case_id"] == "case-a"
    assert payload["report_path"] == str(run_dir / "report.md")
    history_path = eval_root / "rerun_history.jsonl"
    history_rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert history_rows[-1]["case_id"] == "case-a"
    assert history_rows[-1]["status"] == "passed"
    assert history_rows[-1]["quality_ok"] is True
    assert "--case-id" in captured["cmd"]
    assert "case-a" in captured["cmd"]
    assert "--base-url" in captured["cmd"]
    assert "http://127.0.0.1:8005" in captured["cmd"]


def test_research_qa_rerun_route_classifies_connection_error(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    eval_root = tmp_path / "test_results" / "research_qa_eval"

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="requests.exceptions.ConnectionError: Failed to establish a new connection: [Errno 111] Connection refused",
        )

    monkeypatch.setattr(library_router, "_RESEARCH_QA_EVAL_ROOT", eval_root)
    monkeypatch.setattr(library_router.subprocess, "run", fake_run)

    client = TestClient(app)
    response = client.post("/api/library/quality/research-qa/rerun", json={"case_id": "case-a"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["error_kind"] == "connection"
    assert "Connection refused" in payload["error_detail"]
    history_path = eval_root / "rerun_history.jsonl"
    history_rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert history_rows[-1]["error_kind"] == "connection"


def test_quality_action_history_route_persists_stage_results(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    eval_root = tmp_path / "test_results" / "research_qa_eval"
    monkeypatch.setattr(library_router, "_RESEARCH_QA_EVAL_ROOT", eval_root)

    client = TestClient(app)
    response = client.post(
        "/api/library/quality/action-history",
        json={
            "stage_key": "retrieval",
            "stage_label": "Retrieval coverage",
            "action": "rebuild_index",
            "status": "success",
            "summary": "Rebuilt retrieval index",
            "detail": "Next QA check: case-a",
            "target_ids": ["case-a"],
            "metrics": {"target_count": 1},
            "before": {"status": "error", "score": 42, "count": 1, "detail": "1 missed doc"},
            "after": {"status": "good", "score": 96, "count": 0, "detail": "coverage passed"},
            "delta": {"improved": True, "score_delta": 54, "count_delta": 1, "summary": "Improved: error -> good"},
            "improved": True,
            "verification": {"type": "research_qa_rerun", "case_id": "case-a", "quality_ok": True},
            "created_at": 1790000200,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["item"]["stage_key"] == "retrieval"
    history_path = eval_root / "action_history.jsonl"
    history_rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert history_rows[-1]["summary"] == "Rebuilt retrieval index"
    assert history_rows[-1]["target_ids"] == ["case-a"]
    assert history_rows[-1]["before"]["status"] == "error"
    assert history_rows[-1]["after"]["count"] == 0
    assert history_rows[-1]["delta"]["improved"] is True
    assert history_rows[-1]["improved"] is True
    assert history_rows[-1]["verification"]["case_id"] == "case-a"

    list_response = client.get("/api/library/quality/action-history", params={"limit": 5})
    assert list_response.status_code == 200
    rows = list_response.json()["items"]
    assert rows[0]["stage_label"] == "Retrieval coverage"
    assert rows[0]["metrics"]["target_count"] == 1
    assert rows[0]["delta"]["score_delta"] == 54
    assert rows[0]["verification"]["quality_ok"] is True


def test_start_convert_route_infers_no_llm_from_mode(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    (pdf_dir / "z.pdf").write_bytes(b"%PDF-1.4 test")

    captured: dict = {}
    enqueued: list[dict] = []

    def fake_build_bg_task(**kwargs):
        captured.update(kwargs)
        return {"_tid": "task-1", "name": "z.pdf"}

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(
        library_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=str(tmp_path / "db"), library_db_path=str(tmp_path / "library.db")),
    )
    monkeypatch.setattr(library_router, "_build_bg_task", fake_build_bg_task)
    monkeypatch.setattr(library_router, "_bg_enqueue", lambda task: enqueued.append(dict(task or {})))

    client = TestClient(app)
    response = client.post("/api/library/convert", json={"pdf_name": "z.pdf", "speed_mode": "no_llm"})
    assert response.status_code == 200
    assert bool(captured.get("no_llm")) is True
    assert bool(captured.get("replace")) is True
    assert len(enqueued) == 1


def test_reindex_route_starts_reference_sync_on_success(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    md_dir = tmp_path / "md_output"
    pdf_dir = tmp_path / "pdfs"
    md_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    ingest_py = tmp_path / "ingest.py"
    ingest_py.write_text("print('ok')\n", encoding="utf-8")

    class FakeProc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    captured: dict = {}
    structured_captured: dict = {}

    def fake_start_reference_sync(**kwargs):
        captured.update(kwargs)
        return {"started": True, "run_id": 7}

    def fake_rebuild_structured_indices_for_root(src_root, **kwargs):
        structured_captured["src_root"] = src_root
        structured_captured.update(kwargs)
        return {
            "version": 2,
            "scanned": 3,
            "rebuilt": 2,
            "skipped": 1,
            "failed": 0,
            "citation_mention_count": 5,
            "errors": [],
        }

    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_ingest_py_path", lambda: ingest_py)
    monkeypatch.setattr(library_router, "get_settings", lambda: SimpleNamespace(db_dir=str(tmp_path / "db"), library_db_path=str(tmp_path / "library.db")))
    monkeypatch.setattr(library_router.subprocess, "run", lambda *args, **kwargs: FakeProc())
    monkeypatch.setattr(library_router, "start_reference_sync", fake_start_reference_sync)
    monkeypatch.setattr(library_router, "rebuild_structured_indices_for_root", fake_rebuild_structured_indices_for_root)
    monkeypatch.setenv("KB_CROSSREF_BUDGET_S", "55")
    monkeypatch.setenv("KB_REFSYNC_WORKERS", "8")

    client = TestClient(app)
    response = client.post("/api/library/reindex")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert (payload.get("structured_indices") or {}).get("rebuilt") == 2
    assert (payload.get("structured_indices") or {}).get("citation_mention_count") == 5
    assert structured_captured.get("src_root") == md_dir
    assert structured_captured.get("force") is False
    assert (payload.get("refsync") or {}).get("started") is True
    assert captured.get("src_root") == md_dir
    assert captured.get("pdf_root") == pdf_dir
    assert float(captured.get("crossref_time_budget_s") or 0.0) == 55.0
    assert int(captured.get("doi_prefetch_workers") or 0) == 8


def test_references_sync_route_passes_workers_and_budget(monkeypatch, tmp_path: Path):
    from api.routers import references as references_router

    md_dir = tmp_path / "md_output"
    pdf_dir = tmp_path / "pdfs"
    md_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    captured: dict = {}

    def fake_start_reference_sync(**kwargs):
        captured.update(kwargs)
        return {"started": True, "run_id": 11}

    monkeypatch.setattr(references_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(references_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(
        references_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=str(tmp_path / "db"), library_db_path=str(tmp_path / "library.db")),
    )
    monkeypatch.setattr(references_router, "start_reference_sync", fake_start_reference_sync)
    monkeypatch.setenv("KB_REFSYNC_WORKERS", "7")
    monkeypatch.setenv("KB_CROSSREF_BUDGET_S", "70")

    client = TestClient(app)
    response = client.post("/api/references/sync")
    assert response.status_code == 200
    payload = response.json()
    assert payload.get("started") is True
    assert int(captured.get("doi_prefetch_workers") or 0) == 7
    assert float(captured.get("crossref_time_budget_s") or 0.0) == 70.0


def test_rename_suggestions_route_returns_items(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    pdf_a = pdf_dir / "paper_a.pdf"
    pdf_a.write_bytes(b"%PDF-1.4 test")

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(
        library_router,
        "extract_pdf_meta_suggestion",
        lambda path, settings=None: SimpleNamespace(venue="CVPR", year="2024", title="Vision Paper", crossref_meta={}),
    )

    client = TestClient(app)
    response = client.get("/api/library/rename/suggestions", params={"scope": "all", "use_llm": "false"})
    assert response.status_code == 200
    payload = response.json()
    assert int(payload.get("total_scanned") or 0) == 1
    items = list(payload.get("items") or [])
    assert len(items) == 1
    assert str(items[0].get("name") or "") == "paper_a.pdf"
    assert isinstance(items[0].get("suggested_name"), str)
    meta = dict(items[0].get("meta") or {})
    assert isinstance(meta.get("basis_label"), str)
    assert isinstance(meta.get("basis_detail"), str)


def test_apply_rename_suggestions_route_runs_selected(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    (pdf_dir / "a.pdf").write_bytes(b"%PDF-1.4 test")
    (pdf_dir / "b.pdf").write_bytes(b"%PDF-1.4 test")

    called: list[tuple[str, str, bool, bool]] = []

    def fake_auto_rename_saved_pdf_in_library(*, pdf_path, base_name="", use_llm=True, also_md=True):
        called.append((str(Path(pdf_path).name), str(base_name), bool(use_llm), bool(also_md)))
        return {"ok": True, "renamed": True, "name": f"{Path(pdf_path).stem}-new.pdf"}

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "auto_rename_saved_pdf_in_library", fake_auto_rename_saved_pdf_in_library)

    client = TestClient(app)
    response = client.post(
        "/api/library/rename/apply",
        json={
            "pdf_names": ["a.pdf", "b.pdf"],
            "base_overrides": {"a.pdf": "a-custom"},
            "use_llm": False,
            "also_md": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["renamed"] == 2
    assert payload["failed"] == 0
    assert ("a.pdf", "a-custom", False, True) in called
    assert ("b.pdf", "", False, True) in called


def test_auto_rename_saved_pdf_keeps_current_normalized_name(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdf_dir / "CVPR-2024-Vision Paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")

    class FakeStore:
        def __init__(self):
            self.upserts = []

        def upsert(self, sha1, path, citation_meta=None):
            self.upserts.append((sha1, Path(path).name, dict(citation_meta or {})))

    fake_store = FakeStore()
    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_library_store", lambda: fake_store)
    monkeypatch.setattr(
        library_router,
        "extract_pdf_meta_suggestion",
        lambda path, settings=None: SimpleNamespace(venue="CVPR", year="2024", title="Vision Paper", crossref_meta={}),
    )

    result = library_router.auto_rename_saved_pdf_in_library(pdf_path=pdf, use_llm=False, also_md=True)

    assert result["ok"] is True
    assert result["renamed"] is False
    assert result["name"] == "CVPR-2024-Vision Paper.pdf"
    assert not (pdf_dir / "CVPR-2024-Vision Paper-2.pdf").exists()
    assert fake_store.upserts[-1][1] == "CVPR-2024-Vision Paper.pdf"


def test_auto_rename_saved_pdf_rolls_back_when_md_sync_fails(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdf_dir / "old.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    (md_dir / "old").mkdir(parents=True, exist_ok=True)

    class FakeStore:
        def upsert(self, *args, **kwargs):
            raise AssertionError("store should not be updated after failed rename")

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_library_store", lambda: FakeStore())
    monkeypatch.setattr(
        library_router,
        "extract_pdf_meta_suggestion",
        lambda path, settings=None: SimpleNamespace(venue="Nature", year="2024", title="New Paper", crossref_meta={}),
    )
    monkeypatch.setattr(
        library_router,
        "_sync_md_after_pdf_rename_basic",
        lambda **kwargs: {"ok": False, "msg": "forced md failure"},
    )

    result = library_router.auto_rename_saved_pdf_in_library(pdf_path=pdf, use_llm=False, also_md=True)

    assert result["ok"] is False
    assert result["renamed"] is False
    assert result["rollback"]["pdf"] is True
    assert pdf.exists()
    assert not (pdf_dir / "Nature-2024-New Paper.pdf").exists()


def test_sync_md_after_pdf_rename_merges_existing_target_folder(tmp_path: Path):
    from api.routers import library as library_router

    md_dir = tmp_path / "md_output"
    old_folder = md_dir / "old"
    new_folder = md_dir / "new"
    old_folder.mkdir(parents=True, exist_ok=True)
    new_folder.mkdir(parents=True, exist_ok=True)
    (old_folder / "old.en.md").write_text("# old\n", encoding="utf-8")
    (new_folder / "assets_manifest.md").write_text("asset\n", encoding="utf-8")

    result = library_router._sync_md_after_pdf_rename_basic(
        md_root=md_dir,
        src_pdf=tmp_path / "old.pdf",
        dest_pdf=tmp_path / "new.pdf",
    )

    assert result["ok"] is True
    assert not old_folder.exists()
    assert (new_folder / "new.en.md").read_text(encoding="utf-8") == "# old\n"
    assert (new_folder / "assets_manifest.md").exists()


def test_rename_destination_preserves_dots_inside_base_name(tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    current_pdf = pdf_dir / "old.pdf"
    current_pdf.write_bytes(b"%PDF-1.4 test")

    dest = library_router._suggest_dest_for_base(
        pdf_dir=pdf_dir,
        current_pdf=current_pdf,
        base_name="Long...Title.v2",
        md_dir=md_dir,
    )

    assert dest.name == "Long...Title.v2.pdf"


def test_upload_inspect_route_returns_suggestion(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(
        library_router,
        "extract_pdf_meta_suggestion",
        lambda path, settings=None: SimpleNamespace(venue="Nature", year="2018", title="Great Paper", crossref_meta={}),
    )

    client = TestClient(app)
    response = client.post(
        "/api/library/upload/inspect",
        data={"use_llm": "false"},
        files={"file": ("draft.pdf", b"%PDF-1.4 demo", "application/pdf")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "draft.pdf"
    assert payload["duplicate"] is False
    assert isinstance(payload.get("suggested_stem"), str)
    assert isinstance(payload.get("display_full_name"), str)
    meta = dict(payload.get("meta") or {})
    assert isinstance(meta.get("basis_label"), str)
    assert isinstance(meta.get("basis_detail"), str)


def test_upload_inspect_route_rejects_oversized_pdf(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(
        library_router,
        "get_settings",
        lambda: SimpleNamespace(
            db_dir=str(tmp_path / "db"),
            library_db_path=str(tmp_path / "library.db"),
            max_pdf_upload_bytes=8,
        ),
    )

    client = TestClient(app)
    response = client.post(
        "/api/library/upload/inspect",
        data={"use_llm": "false"},
        files={"file": ("draft.pdf", b"%PDF-1.4 demo", "application/pdf")},
    )

    assert response.status_code == 413


def test_upload_commit_route_can_enqueue_convert(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    enqueued: list[dict] = []

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(
        library_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=str(tmp_path / "db"), library_db_path=str(tmp_path / "library.db")),
    )
    monkeypatch.setattr(library_router, "_bg_enqueue", lambda task: enqueued.append(dict(task or {})))

    client = TestClient(app)
    response = client.post(
        "/api/library/upload/commit",
        data={
            "base_name": "custom-base",
            "convert_now": "true",
            "speed_mode": "balanced",
            "allow_duplicate": "false",
        },
        files={"file": ("draft.pdf", b"%PDF-1.4 demo", "application/pdf")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["duplicate"] is False
    assert payload["enqueued"] is True
    assert len(enqueued) == 1


def test_save_pdf_to_library_respects_explicit_base_name_even_with_llm_title(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    class FakeStore:
        def __init__(self) -> None:
            self.saved: list[tuple[str, str]] = []

        def upsert(self, sha1: str, path: Path, citation_meta: dict | None = None) -> None:
            self.saved.append((sha1, str(path)))

    fake_store = FakeStore()

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_library_store", lambda: fake_store)
    monkeypatch.setattr(
        library_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=str(tmp_path / "db"), library_db_path=str(tmp_path / "library.db")),
    )
    monkeypatch.setattr(
        library_router,
        "extract_pdf_meta_suggestion",
        lambda *args, **kwargs: SimpleNamespace(title="LLM Preferred Title", venue="", year="", crossref_meta=None),
    )

    payload = library_router.save_pdf_to_library(
        file_name="draft.pdf",
        data=b"%PDF-1.4 demo",
        base_name="custom-upload-name",
    )

    assert payload["duplicate"] is False
    assert payload["name"] == "custom-upload-name.pdf"
    assert (pdf_dir / "custom-upload-name.pdf").exists()
    assert fake_store.saved


def test_library_meta_update_route_persists_user_meta(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = pdf_dir / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test meta")

    store = LibraryStore(tmp_path / "library.db")

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_library_store", lambda: store)

    client = TestClient(app)
    response = client.post(
        "/api/library/meta/update",
        json={
            "pdf_name": "paper.pdf",
            "paper_category": "SCI",
            "reading_status": "reading",
            "note": "important paper",
            "user_tags": ["pose-free", "Pose-Free", "single-image"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["paper_category"] == "SCI"
    assert payload["reading_status"] == "reading"
    assert payload["note"] == "important paper"
    assert payload["user_tags"] == ["pose-free", "single-image"]

    meta = store.get_paper_user_meta(path=pdf_path)
    assert meta is not None
    assert meta["paper_category"] == "SCI"
    assert meta["reading_status"] == "reading"
    assert meta["note"] == "important paper"
    assert meta["user_tags"] == ["pose-free", "single-image"]


def test_library_files_route_includes_paper_meta_fields(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = pdf_dir / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test meta")

    store = LibraryStore(tmp_path / "library.db")
    store.upsert_paper_user_meta(
        path=pdf_path,
        paper_category="NeRF",
        reading_status="done",
        note="core reference",
        user_tags=["baseline", "view-synthesis"],
    )

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_library_store", lambda: store)
    monkeypatch.setattr(library_router, "_bg_snapshot", lambda: {"running": False, "current": "", "queue": []})

    client = TestClient(app)
    response = client.get("/api/library/files", params={"scope": "all"})
    assert response.status_code == 200
    payload = response.json()
    by_name = {str(item.get("name") or ""): item for item in list(payload.get("items") or [])}
    assert by_name["paper.pdf"]["paper_category"] == "NeRF"
    assert by_name["paper.pdf"]["reading_status"] == "done"
    assert by_name["paper.pdf"]["note"] == "core reference"
    assert by_name["paper.pdf"]["user_tags"] == ["baseline", "view-synthesis"]


def test_library_meta_batch_update_route_persists_batch_changes(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf_a = pdf_dir / "a.pdf"
    pdf_b = pdf_dir / "b.pdf"
    pdf_a.write_bytes(b"%PDF-1.4 a")
    pdf_b.write_bytes(b"%PDF-1.4 b")

    store = LibraryStore(tmp_path / "library.db")
    store.upsert_paper_user_meta(
        path=pdf_a,
        paper_category="NeRF",
        reading_status="unread",
        note="a note",
        user_tags=["baseline", "view-synthesis"],
    )
    store.upsert_paper_user_meta(
        path=pdf_b,
        paper_category="SCI",
        reading_status="reading",
        note="b note",
        user_tags=["baseline", "single-image"],
    )

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_library_store", lambda: store)

    client = TestClient(app)
    response = client.post(
        "/api/library/meta/batch_update",
        json={
            "pdf_names": ["a.pdf", "b.pdf"],
            "apply_paper_category": True,
            "paper_category": "SCI",
            "apply_reading_status": True,
            "reading_status": "done",
            "add_tags": ["pose-free"],
            "remove_tags": ["baseline"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["requested"] == 2
    assert payload["updated"] == 2

    meta_a = store.get_paper_user_meta(path=pdf_a)
    meta_b = store.get_paper_user_meta(path=pdf_b)
    assert meta_a is not None and meta_b is not None
    assert meta_a["paper_category"] == "SCI"
    assert meta_b["paper_category"] == "SCI"
    assert meta_a["reading_status"] == "done"
    assert meta_b["reading_status"] == "done"
    assert meta_a["user_tags"] == ["pose-free", "view-synthesis"]
    assert meta_b["user_tags"] == ["pose-free", "single-image"]


def test_library_suggestions_regenerate_and_apply(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router
    monkeypatch.setenv("KB_LIBRARY_SUGGEST_USE_LLM", "0")

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = pdf_dir / "scinerf.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")

    db_path = tmp_path / "library.db"
    store = LibraryStore(db_path)
    store.upsert(
        "sha1-scinerf",
        pdf_path,
        citation_meta={
            "title": "Pose-Free Single-Image Neural Radiance Fields from Snapshot Compressive Sensing",
            "venue": "CVPR",
        },
    )

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_library_store", lambda: store)

    client = TestClient(app)

    regen_response = client.post(
        "/api/library/meta/suggestions/regenerate",
        json={"pdf_names": ["scinerf.pdf"]},
    )
    assert regen_response.status_code == 200
    regen_payload = regen_response.json()
    assert regen_payload["updated"] == 1
    item = regen_payload["items"][0]
    assert item["suggested_category"] == "SCI"
    assert "pose-free" in item["suggested_tags"]
    assert "single-image" in item["suggested_tags"]

    files_response = client.get("/api/library/files", params={"scope": "all"})
    assert files_response.status_code == 200
    by_name = {str(file_item.get("name") or ""): file_item for file_item in list(files_response.json().get("items") or [])}
    assert by_name["scinerf.pdf"]["has_suggestions"] is True

    apply_response = client.post(
        "/api/library/meta/suggestions/apply",
        json={
            "pdf_name": "scinerf.pdf",
            "category_action": "accept",
            "accept_tags": ["pose-free"],
            "dismiss_tags": ["single-image"],
        },
    )
    assert apply_response.status_code == 200
    applied = apply_response.json()
    assert applied["paper_category"] == "SCI"
    assert "pose-free" in applied["user_tags"]
    assert "pose-free" not in applied["suggested_tags"]
    assert "single-image" not in applied["suggested_tags"]

    regen_again = client.post(
        "/api/library/meta/suggestions/regenerate",
        json={"pdf_names": ["scinerf.pdf"]},
    )
    assert regen_again.status_code == 200
    refreshed = regen_again.json()["items"][0]
    assert refreshed["suggested_category"] == ""
    assert "single-image" not in refreshed["suggested_tags"]


def test_library_suggestions_can_auto_apply_empty_fields(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router
    monkeypatch.setenv("KB_LIBRARY_SUGGEST_USE_LLM", "0")

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = pdf_dir / "scinerf-auto.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")

    store = LibraryStore(tmp_path / "library.db")
    store.upsert(
        "sha1-scinerf-auto",
        pdf_path,
        citation_meta={
            "title": "Pose-Free Single-Image Neural Radiance Fields from Snapshot Compressive Sensing",
            "venue": "CVPR",
        },
    )

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_library_store", lambda: store)

    client = TestClient(app)
    regen_response = client.post(
        "/api/library/meta/suggestions/regenerate",
        json={"pdf_names": ["scinerf-auto.pdf"], "auto_apply_empty": True},
    )
    assert regen_response.status_code == 200
    item = regen_response.json()["items"][0]
    assert item["paper_category"] == "SCI"
    assert "pose-free" in item["user_tags"]
    assert "single-image" in item["user_tags"]
    assert item["suggested_category"] == ""
    assert item["suggested_tags"] == []
    assert item["has_suggestions"] is False


def test_library_suggestions_use_markdown_and_user_taxonomy(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router
    monkeypatch.setenv("KB_LIBRARY_SUGGEST_USE_LLM", "0")

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    seed_pdf = pdf_dir / "seed.pdf"
    target_pdf = pdf_dir / "target.pdf"
    seed_pdf.write_bytes(b"%PDF-1.4 seed")
    target_pdf.write_bytes(b"%PDF-1.4 target")

    target_md = md_dir / "target" / "target.en.md"
    target_md.parent.mkdir(parents=True, exist_ok=True)
    target_md.write_text(
        "\n".join(
            [
                "# Robust Reconstruction",
                "",
                "## Abstract",
                "We present a physics-informed inverse imaging framework for snapshot reconstruction.",
                "",
                "## Introduction",
                "The inverse imaging pipeline uses physics informed regularization to stabilize training.",
                "",
                "## Method",
                "Our method solves an inverse imaging objective under sparse measurements.",
            ]
        ),
        encoding="utf-8",
    )

    db_path = tmp_path / "library.db"
    store = LibraryStore(db_path)
    store.upsert("sha1-seed", seed_pdf, citation_meta={"title": "Seed paper"})
    store.upsert_paper_user_meta(
        path=seed_pdf,
        paper_category="Inverse Imaging",
        reading_status="done",
        note="seed taxonomy",
        user_tags=["physics-informed"],
    )
    store.upsert(
        "sha1-target",
        target_pdf,
        citation_meta={
            "title": "Robust Reconstruction Pipeline",
            "venue": "ICCV",
        },
    )

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_library_store", lambda: store)
    monkeypatch.setenv("KB_MD_DIR", str(md_dir))

    client = TestClient(app)
    regen_response = client.post(
        "/api/library/meta/suggestions/regenerate",
        json={"pdf_names": ["target.pdf"]},
    )
    assert regen_response.status_code == 200
    item = regen_response.json()["items"][0]
    assert item["suggested_category"] == "Inverse Imaging"
    assert "physics-informed" in item["suggested_tags"]


def test_library_suggestions_can_use_llm(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router
    monkeypatch.setenv("KB_LIBRARY_SUGGEST_USE_LLM", "1")

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = pdf_dir / "semantic-paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 semantic")

    store = LibraryStore(tmp_path / "library.db")
    store.upsert(
        "sha1-semantic",
        pdf_path,
        citation_meta={
            "title": "Robust Scene Priors for Reconstruction",
            "abstract": "This abstract is intentionally vague so heuristic matching stays weak.",
        },
    )

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_library_store", lambda: store)
    monkeypatch.setattr(LibraryStore, "_build_suggestion_llm", lambda self, total_targets: object())
    monkeypatch.setattr(
        LibraryStore,
        "_generate_llm_suggestions_for_row",
        lambda self, **kwargs: ("Inverse Imaging", ["physics-informed", "sparse-reconstruction"]),
    )

    client = TestClient(app)
    regen_response = client.post(
        "/api/library/meta/suggestions/regenerate",
        json={"pdf_names": ["semantic-paper.pdf"]},
    )
    assert regen_response.status_code == 200
    item = regen_response.json()["items"][0]
    assert item["suggested_category"] == "Inverse Imaging"
    assert item["suggested_tags"][:2] == ["physics-informed", "sparse-reconstruction"]


def test_library_suggestions_block_generic_doc_types_without_explicit_evidence(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router
    monkeypatch.setenv("KB_LIBRARY_SUGGEST_USE_LLM", "0")

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = pdf_dir / "single-photon.pdf"
    seed_pdf = pdf_dir / "seed.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 single-photon")
    seed_pdf.write_bytes(b"%PDF-1.4 seed")

    store = LibraryStore(tmp_path / "library.db")
    store.upsert(
        "sha1-single-photon",
        pdf_path,
        citation_meta={
            "title": "High-resolution single-photon imaging with physics-informed deep learning",
            "abstract": (
                "We propose a physics-informed reconstruction method for single-photon imaging. "
                "The experiments compare against prior methods on a public dataset."
            ),
        },
    )
    store.upsert_paper_user_meta(
        path=pdf_path,
        paper_category="Dataset",
        reading_status="",
        note="",
        user_tags=["dataset", "physics-informed"],
    )
    store.upsert_paper_user_meta(
        path=seed_pdf,
        paper_category="Dataset",
        reading_status="",
        note="",
        user_tags=["dataset"],
    )

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_library_store", lambda: store)

    # Reset target paper user meta after seeding library taxonomy so it stays unclassified.
    store.upsert_paper_user_meta(
        path=pdf_path,
        paper_category="",
        reading_status="",
        note="",
        user_tags=["physics-informed"],
    )

    client = TestClient(app)
    regen_response = client.post(
        "/api/library/meta/suggestions/regenerate",
        json={"pdf_names": ["single-photon.pdf"]},
    )
    assert regen_response.status_code == 200
    item = regen_response.json()["items"][0]
    assert item["suggested_category"] != "Dataset"
    assert "dataset" not in item["suggested_tags"]


def test_library_suggestions_prefer_domain_category_and_facet_tags(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router
    monkeypatch.setenv("KB_LIBRARY_SUGGEST_USE_LLM", "0")

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = pdf_dir / "single-photon-physics.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 single-photon-physics")

    store = LibraryStore(tmp_path / "library.db")
    store.upsert(
        "sha1-single-photon-physics",
        pdf_path,
        citation_meta={
            "title": "High-resolution single-photon imaging with physics-informed deep learning",
            "abstract": (
                "We present a physics-informed method for single-photon imaging under low-light conditions. "
                "The method reconstructs high-resolution images from photon-limited measurements."
            ),
        },
    )

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_library_store", lambda: store)

    client = TestClient(app)
    regen_response = client.post(
        "/api/library/meta/suggestions/regenerate",
        json={"pdf_names": ["single-photon-physics.pdf"]},
    )
    assert regen_response.status_code == 200
    item = regen_response.json()["items"][0]
    assert item["suggested_category"] == "Single-Photon Imaging"
    assert "physics-informed" in item["suggested_tags"]
    assert "single-photon" in item["suggested_tags"]


def test_library_suggestions_do_not_overclassify_single_pixel_detector_mentions(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router
    monkeypatch.setenv("KB_LIBRARY_SUGGEST_USE_LLM", "0")

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = pdf_dir / "interferometric-ism.pdf"
    seed_pdf = pdf_dir / "single-pixel-seed.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 ism")
    seed_pdf.write_bytes(b"%PDF-1.4 seed")

    store = LibraryStore(tmp_path / "library.db")
    store.upsert("sha1-seed-spi", seed_pdf, citation_meta={"title": "Seed single-pixel imaging paper"})
    store.upsert_paper_user_meta(
        path=seed_pdf,
        paper_category="Single-Pixel Imaging",
        reading_status="",
        note="",
        user_tags=["high-resolution", "single-pixel"],
    )
    store.upsert(
        "sha1-ism",
        pdf_path,
        citation_meta={
            "title": "Interferometric image scanning microscopy for label-free live cell imaging",
            "abstract": (
                "Image scanning microscopy can replace the single pixel detector with an array detector. "
                "This microscopy paper focuses on pixel reassignment and live-cell label-free contrast."
            ),
        },
    )
    store.upsert_paper_user_meta(
        path=pdf_path,
        paper_category="",
        reading_status="",
        note="",
        user_tags=["high-resolution"],
    )

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_library_store", lambda: store)

    client = TestClient(app)
    regen_response = client.post(
        "/api/library/meta/suggestions/regenerate",
        json={"pdf_names": ["interferometric-ism.pdf"]},
    )
    assert regen_response.status_code == 200
    item = regen_response.json()["items"][0]
    assert item["suggested_category"] != "Single-Pixel Imaging"
    assert "single-pixel" not in item["suggested_tags"]


def test_library_llm_suggestions_default_to_candidate_vocab(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    class DummyLLM:
        def chat(self, messages, temperature=0.0, max_tokens=420):
            return json.dumps(
                {
                    "suggested_category": "Computational Imaging",
                    "suggested_tags": ["physics informed", "deep-learning", "custom odd phrase"],
                    "category_confidence": 0.95,
                    "tag_confidence": 0.95,
                    "reason": "test",
                }
            )

    monkeypatch.setenv("KB_LIBRARY_SUGGEST_USE_LLM", "1")
    monkeypatch.setenv("KB_LIBRARY_SUGGEST_ALLOW_NEW_CATEGORY", "0")
    monkeypatch.setenv("KB_LIBRARY_SUGGEST_ALLOW_NEW_TAGS", "0")

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = pdf_dir / "single-photon-llm.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 single-photon-llm")

    store = LibraryStore(tmp_path / "library.db")
    store.upsert(
        "sha1-single-photon-llm",
        pdf_path,
        citation_meta={
            "title": "High-resolution single-photon imaging with physics-informed deep learning",
            "abstract": "A physics-informed single-photon imaging method for photon-limited reconstruction.",
        },
    )

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_library_store", lambda: store)
    monkeypatch.setattr(LibraryStore, "_build_suggestion_llm", lambda self, total_targets: DummyLLM())

    client = TestClient(app)
    regen_response = client.post(
        "/api/library/meta/suggestions/regenerate",
        json={"pdf_names": ["single-photon-llm.pdf"]},
    )
    assert regen_response.status_code == 200
    item = regen_response.json()["items"][0]
    assert item["suggested_category"] == "Single-Photon Imaging"
    assert "physics-informed" in item["suggested_tags"]
    assert "deep-learning" not in item["suggested_tags"]
    assert "custom odd phrase" not in item["suggested_tags"]
