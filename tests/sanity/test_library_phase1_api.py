from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app


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
