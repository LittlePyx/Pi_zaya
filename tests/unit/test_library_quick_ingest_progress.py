from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_quick_ingest_pdf_reports_progress(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")
    md_root = tmp_path / "md"
    md_root.mkdir(parents=True, exist_ok=True)
    md_main = md_root / "paper.en.md"
    md_main.write_text("# paper\n", encoding="utf-8")
    ingest_py = tmp_path / "ingest.py"
    ingest_py.write_text("print('ok')\n", encoding="utf-8")

    monkeypatch.setattr(library_router, "get_settings", lambda: SimpleNamespace(db_dir=str(tmp_path / "db")))
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(library_router, "run_pdf_to_md", lambda **kwargs: (True, str(md_root / "paper")))
    monkeypatch.setattr(library_router, "_resolve_md_output_paths", lambda *args, **kwargs: (None, md_main, True))
    monkeypatch.setattr(library_router, "_ingest_py_path", lambda: ingest_py)

    class FakeCompletedProcess:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(library_router.subprocess, "run", lambda *args, **kwargs: FakeCompletedProcess())

    stages: list[str] = []
    result = library_router.quick_ingest_pdf(
        pdf_path=pdf_path,
        speed_mode="ultra_fast",
        progress_cb=stages.append,
    )

    assert result["ready"] is True
    assert stages == ["converting", "ingesting"]
