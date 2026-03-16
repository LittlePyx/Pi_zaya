from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app
from kb.library_store import LibraryStore


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


def test_start_convert_route_infers_no_llm_from_mode(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

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

    def fake_start_reference_sync(**kwargs):
        captured.update(kwargs)
        return {"started": True, "run_id": 7}

    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_ingest_py_path", lambda: ingest_py)
    monkeypatch.setattr(library_router, "get_settings", lambda: SimpleNamespace(db_dir=str(tmp_path / "db"), library_db_path=str(tmp_path / "library.db")))
    monkeypatch.setattr(library_router.subprocess, "run", lambda *args, **kwargs: FakeProc())
    monkeypatch.setattr(library_router, "start_reference_sync", fake_start_reference_sync)
    monkeypatch.setenv("KB_CROSSREF_BUDGET_S", "55")
    monkeypatch.setenv("KB_REFSYNC_WORKERS", "8")

    client = TestClient(app)
    response = client.post("/api/library/reindex")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
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
