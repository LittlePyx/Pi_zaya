from __future__ import annotations

import threading
import time
import os
from pathlib import Path
from typing import Any

from kb.reference_index import build_reference_index


_LOCK = threading.Lock()
_THREAD: threading.Thread | None = None
_STATE: dict[str, Any] = {
    "running": False,
    "status": "idle",  # idle | running | done | error
    "stage": "",
    "message": "",
    "error": "",
    "run_id": 0,
    "current": "",
    "docs_done": 0,
    "docs_total": 0,
    "refs_total": 0,
    "refs_with_doi": 0,
    "refs_crossref_ok": 0,
    "refs_source_map_ok": 0,
    "stats": {},
    "started_at": 0.0,
    "finished_at": 0.0,
    "updated_at": 0.0,
}


def _now() -> float:
    return float(time.time())


def _set_state_if_current(run_id: int, **patch: Any) -> bool:
    with _LOCK:
        if int(_STATE.get("run_id", 0) or 0) != int(run_id):
            return False
        _STATE.update(patch)
        _STATE["updated_at"] = _now()
    return True


def _fmt_done_message(stats: dict[str, Any]) -> str:
    docs = int(stats.get("docs_indexed", 0) or 0)
    refs = int(stats.get("refs_total", 0) or 0)
    doi = int(stats.get("refs_with_doi", 0) or 0)
    src = int(stats.get("refs_source_map_ok", 0) or 0)
    cross = int(stats.get("refs_crossref_ok", 0) or 0)
    return (
        f"参考文献索引已更新: 文档 {docs}, 条目 {refs}, "
        f"含 DOI {doi}, 源文献映射 {src}, Crossref 匹配 {cross}."
    )


def _pdf_md_coverage(src_root: Path, pdf_root: Path | None) -> dict[str, Any]:
    def _to_os_path(path_like: Path | str) -> str:
        p = Path(path_like).expanduser()
        try:
            s = str(p.resolve(strict=False))
        except Exception:
            s = str(p)
        if os.name != "nt":
            return s
        if s.startswith("\\\\?\\"):
            return s
        if s.startswith("\\\\"):
            return "\\\\?\\UNC\\" + s.lstrip("\\")
        return "\\\\?\\" + s

    def _is_file(path_like: Path | str) -> bool:
        try:
            return bool(os.path.isfile(_to_os_path(path_like)))
        except Exception:
            try:
                return Path(path_like).is_file()
            except Exception:
                return False

    if pdf_root is None:
        return {}
    try:
        pdfs = [p for p in Path(pdf_root).glob("*.pdf") if _is_file(p)]
    except Exception:
        pdfs = []
    if not pdfs:
        return {"pdf_total": 0, "md_total": 0, "missing_pdf_count": 0}

    md_paths: list[Path] = []
    try:
        for p in Path(src_root).rglob("*.md"):
            n = str(p.name or "").lower()
            if n in {"assets_manifest.md", "quality_report.md", "output.md"}:
                continue
            if _is_file(p):
                md_paths.append(p)
    except Exception:
        md_paths = []

    md_stems: set[str] = set()
    for m in md_paths:
        s = str(m.stem or "")
        if s.lower().endswith(".en"):
            s = s[:-3]
        s = s.strip().lower()
        if s:
            md_stems.add(s)

    missing: list[str] = []
    for pdf in pdfs:
        if str(pdf.stem or "").strip().lower() not in md_stems:
            missing.append(pdf.name)

    return {
        "pdf_total": int(len(pdfs)),
        "md_total": int(len(md_paths)),
        "missing_pdf_count": int(len(missing)),
        "missing_pdf_examples": list(missing[:4]),
    }


def _worker(
    run_id: int,
    *,
    src_root: Path,
    db_dir: Path,
    pdf_root: Path | None,
    library_db_path: Path | None,
    incremental: bool,
    enable_title_lookup: bool,
    crossref_time_budget_s: float,
) -> None:
    def _on_progress(payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        _set_state_if_current(
            run_id,
            stage=str(payload.get("stage") or "").strip(),
            current=str(payload.get("current") or "").strip(),
            docs_done=int(payload.get("docs_done", 0) or 0),
            docs_total=int(payload.get("docs_total", 0) or 0),
            refs_total=int(payload.get("refs_total", 0) or 0),
            refs_with_doi=int(payload.get("refs_with_doi", 0) or 0),
            refs_crossref_ok=int(payload.get("refs_crossref_ok", 0) or 0),
            refs_source_map_ok=int(payload.get("refs_source_map_ok", 0) or 0),
        )

    try:
        stats = build_reference_index(
            src_root=src_root,
            db_dir=db_dir,
            incremental=bool(incremental),
            enable_title_lookup=bool(enable_title_lookup),
            crossref_time_budget_s=float(max(5.0, crossref_time_budget_s)),
            pdf_root=pdf_root,
            library_db_path=library_db_path,
            progress_cb=_on_progress,
        )
        coverage = _pdf_md_coverage(src_root=src_root, pdf_root=pdf_root)
        merged_stats = dict(stats if isinstance(stats, dict) else {})
        if isinstance(coverage, dict):
            merged_stats.update(coverage)

        done_msg = _fmt_done_message(stats if isinstance(stats, dict) else {})
        try:
            pdf_total = int((coverage or {}).get("pdf_total", 0) or 0)
            missing_n = int((coverage or {}).get("missing_pdf_count", 0) or 0)
            if pdf_total > 0:
                done_msg = done_msg + f"（MD 文档 {int(stats.get('docs_indexed', 0) or 0)}/{pdf_total} 篇 PDF）"
            if missing_n > 0:
                done_msg = done_msg + f"；另有 {missing_n} 篇 PDF 尚未生成主 MD。"
        except Exception:
            pass
        _set_state_if_current(
            run_id,
            running=False,
            status="done",
            stage="done",
            stats=merged_stats,
            message=done_msg,
            error="",
            finished_at=_now(),
        )
    except Exception as e:
        _set_state_if_current(
            run_id,
            running=False,
            status="error",
            stage="error",
            error=str(e),
            message=f"参考文献索引后台同步失败: {e}",
            finished_at=_now(),
        )


def start_reference_sync(
    *,
    src_root: Path,
    db_dir: Path,
    pdf_root: Path | None = None,
    library_db_path: Path | None = None,
    incremental: bool = True,
    enable_title_lookup: bool = True,
    crossref_time_budget_s: float = 45.0,
) -> dict[str, Any]:
    global _THREAD
    src = Path(src_root).expanduser().resolve()
    db = Path(db_dir).expanduser().resolve()
    pdf = Path(pdf_root).expanduser().resolve() if pdf_root else None
    lib_db = Path(library_db_path).expanduser().resolve() if library_db_path else None

    with _LOCK:
        if bool(_STATE.get("running")):
            return {"started": False, "reason": "running", "run_id": int(_STATE.get("run_id", 0) or 0)}

        run_id = int(_STATE.get("run_id", 0) or 0) + 1
        _STATE.update(
            {
                "running": True,
                "status": "running",
                "stage": "starting",
                "message": "正在后台同步参考文献索引 (Crossref)...",
                "error": "",
                "run_id": run_id,
                "current": "",
                "docs_done": 0,
                "docs_total": 0,
                "refs_total": 0,
                "refs_with_doi": 0,
                "refs_crossref_ok": 0,
                "refs_source_map_ok": 0,
                "stats": {},
                "started_at": _now(),
                "finished_at": 0.0,
                "updated_at": _now(),
            }
        )

        t = threading.Thread(
            target=_worker,
            kwargs={
                "run_id": run_id,
                "src_root": src,
                "db_dir": db,
                "pdf_root": pdf,
                "library_db_path": lib_db,
                "incremental": bool(incremental),
                "enable_title_lookup": bool(enable_title_lookup),
                "crossref_time_budget_s": float(max(5.0, crossref_time_budget_s)),
            },
            daemon=True,
            name=f"kb-ref-sync-{run_id}",
        )
        _THREAD = t
        t.start()
        return {"started": True, "run_id": run_id}


def snapshot() -> dict[str, Any]:
    with _LOCK:
        out = dict(_STATE)
        out["stats"] = dict(_STATE.get("stats") or {})
        return out


def is_running_snapshot(snap: dict[str, Any]) -> bool:
    if not isinstance(snap, dict):
        return False
    return bool(snap.get("running"))
