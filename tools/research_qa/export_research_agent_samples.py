from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from kb.agent.source_summary import build_agent_source_summary


DEFAULT_DB_PATH = Path(os.getenv("KB_CHAT_DB") or "chat.sqlite3")
DEFAULT_OUT_PATH = Path("test_results/research_agent_answer_samples.jsonl")

VALID_SOURCE_BLENDS = {
    "local_grounded",
    "hybrid_local_external",
    "external_academic",
    "general_llm",
}

ANSWER_MODE_TO_SOURCE_BLEND = {
    "evidence_grounded": "local_grounded",
    "hybrid_local_external": "hybrid_local_external",
    "external_academic_llm": "external_academic",
    "general_llm": "general_llm",
}


def _json_loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _first_str(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _clip(value: Any, max_len: int = 900) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "..."


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _agent_trace_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    trace = meta.get("agent_trace")
    if isinstance(trace, dict):
        return trace
    trace = meta.get("agentTrace")
    if isinstance(trace, dict):
        return trace
    return {}


def _agent_source_summary_from_meta(meta: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    summary = meta.get("agent_source_summary")
    if isinstance(summary, dict):
        return summary
    summary = meta.get("agentSourceSummary")
    if isinstance(summary, dict):
        return summary
    return build_agent_source_summary(trace)


def _previous_user_message(
    conn: sqlite3.Connection,
    conv_id: str,
    assistant_message_id: int,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT id, content, meta_json, created_at
        FROM messages
        WHERE conv_id = ? AND role = 'user' AND id < ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (conv_id, assistant_message_id),
    ).fetchone()


def _trace_summary(trace: dict[str, Any]) -> dict[str, Any]:
    summary = _as_dict(trace.get("summary"))
    if summary:
        return summary
    context = _as_dict(trace.get("context"))
    return _as_dict(context.get("summary"))


def _answer_mode(trace: dict[str, Any], meta: dict[str, Any]) -> str:
    summary = _trace_summary(trace)
    context = _as_dict(trace.get("context"))
    return _first_str(
        summary.get("answer_mode"),
        context.get("answer_mode"),
        meta.get("answer_mode"),
        meta.get("agent_answer_mode"),
    )


def _source_blend(trace: dict[str, Any], meta: dict[str, Any]) -> str:
    summary = _trace_summary(trace)
    context = _as_dict(trace.get("context"))
    blend = _first_str(
        summary.get("answer_source_blend"),
        summary.get("source_blend"),
        context.get("answer_source_blend"),
        context.get("source_blend"),
        meta.get("answer_source_blend"),
        meta.get("source_blend"),
    )
    if blend in VALID_SOURCE_BLENDS:
        return blend
    mode = _answer_mode(trace, meta)
    return ANSWER_MODE_TO_SOURCE_BLEND.get(mode, "")


def _normalize_hit(hit: Any) -> dict[str, Any] | None:
    if not isinstance(hit, dict):
        return None
    meta = _as_dict(hit.get("meta"))
    text = _first_str(
        hit.get("text"),
        hit.get("content"),
        hit.get("snippet"),
        hit.get("text_preview"),
        hit.get("evidence_quote"),
        hit.get("quote"),
    )
    if not text:
        return None
    source_name = _first_str(
        hit.get("source_name"),
        hit.get("paper_title"),
        hit.get("title"),
        meta.get("source_name"),
        meta.get("paper_title"),
        meta.get("title"),
    )
    source_path = _first_str(hit.get("source_path"), meta.get("source_path"), meta.get("path"))
    heading_path = hit.get("heading_path", meta.get("heading_path", []))
    if isinstance(heading_path, str):
        heading_path = [heading_path]
    if not isinstance(heading_path, list):
        heading_path = []
    normalized: dict[str, Any] = {
        "text": _clip(text),
        "score": hit.get("score", hit.get("rerank_score", hit.get("similarity", 0.0))),
        "meta": {
            "source_name": source_name,
            "source_path": source_path,
            "heading_path": heading_path,
        },
    }
    page = hit.get("page", meta.get("page"))
    if page is not None:
        normalized["meta"]["page"] = page
    return normalized


def _hits_from_trace_steps(trace: dict[str, Any]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for step in _as_list(trace.get("steps")):
        if not isinstance(step, dict):
            continue
        output = _as_dict(step.get("output"))
        candidates = output.get("hits")
        if candidates is None:
            candidates = output.get("evidence_hits")
        for hit in _as_list(candidates):
            normalized = _normalize_hit(hit)
            if normalized:
                hits.append(normalized)
    return hits


def _hits_from_research_run(trace: dict[str, Any]) -> list[dict[str, Any]]:
    research_run = _as_dict(trace.get("research_run"))
    hits: list[dict[str, Any]] = []
    for item in _as_list(research_run.get("evidence_matrix")):
        if not isinstance(item, dict):
            continue
        normalized = _normalize_hit(
            {
                "text": item.get("evidence_quote") or item.get("claim"),
                "source_name": item.get("source_title") or item.get("source_name"),
                "source_path": item.get("source_path"),
                "heading_path": item.get("heading_path", []),
                "page": item.get("page"),
            }
        )
        if normalized:
            hits.append(normalized)
    return hits


def _hits_from_cite_details(meta: dict[str, Any]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for item in _as_list(meta.get("cite_details")):
        if not isinstance(item, dict):
            continue
        normalized = _normalize_hit(item)
        if normalized:
            hits.append(normalized)
    return hits


def _hits_from_message_refs(conn: sqlite3.Connection, user_msg_id: int) -> list[dict[str, Any]]:
    if user_msg_id <= 0 or not _table_exists(conn, "message_refs"):
        return []
    row = conn.execute(
        "SELECT hits_json FROM message_refs WHERE user_msg_id = ?",
        (user_msg_id,),
    ).fetchone()
    if row is None:
        return []
    raw_hits = _json_loads(row["hits_json"], [])
    hits: list[dict[str, Any]] = []
    for hit in _as_list(raw_hits):
        normalized = _normalize_hit(hit)
        if normalized:
            hits.append(normalized)
    return hits


def _dedupe_hits(hits: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for hit in hits:
        meta = _as_dict(hit.get("meta"))
        key = (str(meta.get("source_path") or meta.get("source_name") or ""), hit["text"][:160])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hit)
        if len(deduped) >= limit:
            break
    return deduped


def _evidence_hits(trace: dict[str, Any], meta: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    hits = []
    hits.extend(_hits_from_trace_steps(trace))
    hits.extend(_hits_from_research_run(trace))
    hits.extend(_hits_from_cite_details(meta))
    return _dedupe_hits(hits, limit)


def _expected_notice(source_blend: str) -> str:
    if source_blend == "hybrid_local_external":
        return "hybrid_notice"
    if source_blend == "external_academic":
        return "external_not_local"
    return "none"


def _compact_trace(trace: dict[str, Any], source_blend: str, answer_mode: str) -> dict[str, Any]:
    summary = dict(_trace_summary(trace))
    if source_blend:
        summary.setdefault("answer_source_blend", source_blend)
    if answer_mode:
        summary.setdefault("answer_mode", answer_mode)
    compact = {
        "mode": trace.get("mode", "research_agent"),
        "question_type": trace.get("question_type", "unknown"),
        "summary": summary,
    }
    plan = _as_list(trace.get("plan"))
    if plan:
        compact["plan"] = plan[:8]
    return compact


def _sample_from_message(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    check_local_support: bool,
    evidence_limit: int,
) -> dict[str, Any] | None:
    meta = _as_dict(_json_loads(row["meta_json"], {}))
    trace = _agent_trace_from_meta(meta)
    if _first_str(trace.get("mode"), meta.get("agent_mode")) != "research_agent":
        return None

    source_blend = _source_blend(trace, meta)
    if not source_blend:
        return None

    user_row = _previous_user_message(conn, row["conv_id"], int(row["id"]))
    if user_row is None:
        return None

    answer = _clip(row["content"], 6000)
    hits = _evidence_hits(trace, meta, evidence_limit)
    if len(hits) < evidence_limit:
        hits = _dedupe_hits(
            [*hits, *_hits_from_message_refs(conn, int(user_row["id"]))],
            evidence_limit,
        )
    answer_mode = _answer_mode(trace, meta)
    if not answer_mode:
        answer_mode = next(
            (mode for mode, blend in ANSWER_MODE_TO_SOURCE_BLEND.items() if blend == source_blend),
            source_blend,
        )
    should_check_local = (
        check_local_support and source_blend in {"local_grounded", "hybrid_local_external"} and bool(hits)
    )
    source_summary = _agent_source_summary_from_meta(meta, trace)

    return {
        "id": f"chat-{row['id']}",
        "sample_kind": "real_chat_replay",
        "conv_id": row["conv_id"],
        "assistant_message_id": int(row["id"]),
        "user_message_id": int(user_row["id"]),
        "created_at": row["created_at"],
        "query": _clip(user_row["content"], 1600),
        "answer": answer,
        "answer_mode": answer_mode,
        "source_blend": source_blend,
        "agent_source_summary": source_summary,
        "agent_trace": _compact_trace(trace, source_blend, answer_mode),
        "evidence_hits": hits,
        "expected_retrieval_hit": bool(hits),
        "should_use_local_evidence": should_check_local,
        "external_fallback_allowed": source_blend in {"hybrid_local_external", "external_academic", "general_llm"},
        "expected_answer_points": [],
        "expected_user_notice": _expected_notice(source_blend),
        "expected_source_keywords": [],
        "replay_unlabeled": True,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def export_research_agent_samples(
    *,
    db_path: Path,
    out_path: Path,
    limit: int = 50,
    scan_limit: int | None = None,
    check_local_support: bool = False,
    evidence_limit: int = 8,
) -> dict[str, Any]:
    db_path = Path(db_path)
    out_path = Path(out_path)
    if limit <= 0:
        raise ValueError("limit must be positive")
    if evidence_limit <= 0:
        raise ValueError("evidence_limit must be positive")
    if not db_path.exists():
        _write_jsonl(out_path, [])
        return {
            "ok": True,
            "reason": "missing_db",
            "db_path": str(db_path),
            "out_path": str(out_path),
            "sample_count": 0,
            "scanned_messages": 0,
        }

    effective_scan_limit = scan_limit or max(limit * 5, 100)
    samples: list[dict[str, Any]] = []
    scanned = 0
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if not _table_exists(conn, "messages"):
            _write_jsonl(out_path, [])
            return {
                "ok": True,
                "reason": "missing_messages_table",
                "db_path": str(db_path),
                "out_path": str(out_path),
                "sample_count": 0,
                "scanned_messages": 0,
            }
        rows = conn.execute(
            """
            SELECT id, conv_id, role, content, meta_json, created_at
            FROM messages
            WHERE role = 'assistant'
            ORDER BY id DESC
            LIMIT ?
            """,
            (effective_scan_limit,),
        ).fetchall()
        for row in rows:
            scanned += 1
            sample = _sample_from_message(
                conn,
                row,
                check_local_support=check_local_support,
                evidence_limit=evidence_limit,
            )
            if not sample:
                continue
            samples.append(sample)
            if len(samples) >= limit:
                break

    _write_jsonl(out_path, samples)
    return {
        "ok": True,
        "db_path": str(db_path),
        "out_path": str(out_path),
        "sample_count": len(samples),
        "scanned_messages": scanned,
        "check_local_support": check_local_support,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export recent real Research Agent chat answers as JSONL eval samples."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to chat.sqlite3")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=50, help="Maximum exported samples")
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=None,
        help="Maximum assistant messages to scan before filtering Research Agent answers",
    )
    parser.add_argument(
        "--check-local-support",
        action="store_true",
        help="Enable claim-support checks for local/hybrid real samples; off by default for unlabeled replay.",
    )
    parser.add_argument("--evidence-limit", type=int, default=8, help="Max evidence snippets per sample")
    args = parser.parse_args()

    summary = export_research_agent_samples(
        db_path=args.db,
        out_path=args.out,
        limit=args.limit,
        scan_limit=args.scan_limit,
        check_local_support=args.check_local_support,
        evidence_limit=args.evidence_limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
