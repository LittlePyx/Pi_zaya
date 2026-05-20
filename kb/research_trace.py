from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _source_label(source_path: str) -> str:
    raw = str(source_path or "").strip()
    if not raw:
        return ""
    try:
        return Path(raw).name or raw
    except Exception:
        return raw.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] or raw


def summarize_hit(hit: dict | None) -> dict:
    if not isinstance(hit, dict):
        return {}
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    source_path = str((meta or {}).get("source_path") or "").strip()
    out = {
        "source_path": source_path,
        "source_name": _source_label(source_path),
        "heading_path": str((meta or {}).get("heading_path") or "").strip(),
        "score": round(_safe_float(hit.get("score") or (meta or {}).get("score"), 0.0), 4),
    }
    if (meta or {}).get("ref_display_reason"):
        out["ref_display_reason"] = str((meta or {}).get("ref_display_reason") or "").strip()
    return {k: v for k, v in out.items() if v not in ("", 0.0, None)}


def summarize_hits(hits: list[dict] | None, *, limit: int = 6) -> list[dict]:
    out: list[dict] = []
    for hit in list(hits or [])[: max(0, int(limit or 0))]:
        item = summarize_hit(hit if isinstance(hit, dict) else None)
        if item:
            out.append(item)
    return out


def new_trace(
    *,
    session_id: str,
    task_id: str,
    conv_id: str,
    user_msg_id: int = 0,
    assistant_msg_id: int = 0,
    trace_id: str = "",
    prompt_sig: str = "",
    mode: str = "normal",
    started_at: float | None = None,
) -> dict:
    started = float(started_at if started_at is not None else time.time())
    return {
        "version": 1,
        "trace_id": str(trace_id or uuid.uuid4().hex[:16]),
        "session_id": str(session_id or ""),
        "task_id": str(task_id or ""),
        "conv_id": str(conv_id or ""),
        "user_msg_id": _safe_int(user_msg_id),
        "assistant_msg_id": _safe_int(assistant_msg_id),
        "prompt_sig": str(prompt_sig or ""),
        "mode": str(mode or "normal"),
        "status": "running",
        "started_at": started,
        "updated_at": started,
        "timings_ms": {},
        "events": [],
    }


def add_event(trace: dict | None, stage: str, *, elapsed_s: float | None = None, **payload: Any) -> dict:
    tr = dict(trace or {})
    now = time.time()
    tr["updated_at"] = now
    stage_name = str(stage or "").strip()
    if elapsed_s is not None and stage_name:
        timings = dict(tr.get("timings_ms") or {})
        timings[stage_name] = round(max(0.0, _safe_float(elapsed_s)) * 1000.0, 2)
        tr["timings_ms"] = timings
    event = {"stage": stage_name, "at": now}
    if elapsed_s is not None:
        event["elapsed_ms"] = round(max(0.0, _safe_float(elapsed_s)) * 1000.0, 2)
    for key, value in payload.items():
        if value is None:
            continue
        event[str(key)] = value
    events = list(tr.get("events") or [])
    if stage_name:
        events.append(event)
    tr["events"] = events[-40:]
    return tr


def merge_section(trace: dict | None, section: str, payload: dict | None) -> dict:
    tr = dict(trace or {})
    key = str(section or "").strip()
    if not key:
        return tr
    current = tr.get(key)
    merged = dict(current or {}) if isinstance(current, dict) else {}
    if isinstance(payload, dict):
        merged.update(payload)
    tr[key] = merged
    tr["updated_at"] = time.time()
    return tr


def finish_trace(trace: dict | None, *, status: str, total_elapsed_s: float | None = None, error: str = "") -> dict:
    tr = dict(trace or {})
    now = time.time()
    tr["status"] = str(status or "done")
    tr["finished_at"] = now
    tr["updated_at"] = now
    if total_elapsed_s is not None:
        timings = dict(tr.get("timings_ms") or {})
        timings["total"] = round(max(0.0, _safe_float(total_elapsed_s)) * 1000.0, 2)
        tr["timings_ms"] = timings
    if error:
        tr["error"] = str(error)[:500]
    return compact_trace(tr)


def compact_trace(trace: dict | None, *, max_events: int = 30, max_sources: int = 8) -> dict:
    tr = dict(trace or {})
    events = list(tr.get("events") or [])
    tr["events"] = events[-max(0, int(max_events or 0)) :]
    for section_name in ("retrieval", "answer", "refs", "citation_systems"):
        section = tr.get(section_name)
        if not isinstance(section, dict):
            continue
        for key in ("top_hits", "answer_sources", "final_display_sources", "seed_sources"):
            value = section.get(key)
            if isinstance(value, list):
                section[key] = value[: max(0, int(max_sources or 0))]
        tr[section_name] = section
    return tr
