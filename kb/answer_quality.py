from __future__ import annotations

import os
import time

from kb import runtime_state as RUNTIME


def _gen_record_answer_quality(
    *,
    session_id: str,
    task_id: str,
    conv_id: str,
    answer_quality: dict | None,
) -> None:
    q = dict(answer_quality or {})
    if not q:
        return
    try:
        max_keep = int(str(os.environ.get("KB_ANSWER_QUALITY_KEEP", "800") or "800"))
    except Exception:
        max_keep = 800
    max_keep = max(100, min(5000, max_keep))
    sample = {
        "ts": float(time.time()),
        "session_id": str(session_id or ""),
        "task_id": str(task_id or ""),
        "conv_id": str(conv_id or ""),
        "intent": str(q.get("intent") or ""),
        "depth": str(q.get("depth") or ""),
        "contract_enabled": bool(q.get("contract_enabled", False)),
        "has_hits": bool(q.get("has_hits", False)),
        "has_conclusion": bool(q.get("has_conclusion", False)),
        "has_evidence": bool(q.get("has_evidence", False)),
        "has_next_steps": bool(q.get("has_next_steps", False)),
        "evidence_required": bool(q.get("evidence_required", False)),
        "evidence_ok": bool(q.get("evidence_ok", False)),
        "minimum_ok": bool(q.get("minimum_ok", False)),
        "core_section_coverage": float(q.get("core_section_coverage") or 0.0),
        "char_count": int(q.get("char_count") or 0),
    }
    with RUNTIME.GEN_LOCK:
        events = getattr(RUNTIME, "GEN_QUALITY_EVENTS", None)
        if not isinstance(events, list):
            events = []
            RUNTIME.GEN_QUALITY_EVENTS = events
        events.append(sample)
        overflow = len(events) - max_keep
        if overflow > 0:
            del events[:overflow]


def _gen_answer_quality_summary(
    *,
    limit: int = 200,
    intent: str = "",
    depth: str = "",
    only_failed: bool = False,
) -> dict:
    try:
        n_limit = int(limit)
    except Exception:
        n_limit = 200
    n_limit = max(20, min(2000, n_limit))
    intent_filter = str(intent or "").strip().lower()
    depth_filter = str(depth or "").strip().upper()
    failed_only = bool(only_failed)
    with RUNTIME.GEN_LOCK:
        events0 = getattr(RUNTIME, "GEN_QUALITY_EVENTS", None)
        events = list(events0) if isinstance(events0, list) else []
    if intent_filter:
        events = [x for x in events if str(x.get("intent") or "").strip().lower() == intent_filter]
    if depth_filter:
        events = [x for x in events if str(x.get("depth") or "").strip().upper() == depth_filter]
    if failed_only:
        events = [x for x in events if not bool(x.get("minimum_ok", False))]
    if n_limit and len(events) > n_limit:
        events = events[-n_limit:]
    total = len(events)
    if total <= 0:
        return {
            "limit": n_limit,
            "filters": {
                "intent": intent_filter,
                "depth": depth_filter,
                "only_failed": failed_only,
            },
            "total": 0,
            "failed_count": 0,
            "failed_rate": 0.0,
            "structure_complete_rate": 0.0,
            "evidence_coverage_rate": 0.0,
            "next_steps_coverage_rate": 0.0,
            "minimum_ok_rate": 0.0,
            "avg_core_section_coverage": 0.0,
            "by_intent": {},
            "by_depth": {},
            "fail_reasons": {},
        }

    structure_ok = 0
    evidence_ok = 0
    next_steps_ok = 0
    minimum_ok = 0
    failed_count = 0
    core_cov_sum = 0.0
    by_intent: dict[str, dict] = {}
    by_depth: dict[str, dict] = {}
    fail_reasons: dict[str, int] = {}

    for rec in events:
        has_conclusion = bool(rec.get("has_conclusion", False))
        has_next_steps = bool(rec.get("has_next_steps", False))
        has_evidence = bool(rec.get("has_evidence", False))
        evidence_required = bool(rec.get("evidence_required", False))
        rec_structure_ok = bool(has_conclusion and has_next_steps)
        rec_evidence_ok = bool((not evidence_required) or has_evidence)
        rec_next_ok = bool(has_next_steps)
        rec_minimum_ok = bool(rec.get("minimum_ok", False))
        rec_core_cov = float(rec.get("core_section_coverage") or 0.0)

        structure_ok += int(rec_structure_ok)
        evidence_ok += int(rec_evidence_ok)
        next_steps_ok += int(rec_next_ok)
        minimum_ok += int(rec_minimum_ok)
        failed_count += int(not rec_minimum_ok)
        core_cov_sum += rec_core_cov

        intent_key = str(rec.get("intent") or "unknown").strip().lower() or "unknown"
        bucket = by_intent.setdefault(
            intent_key,
            {
                "count": 0,
                "structure_complete_rate": 0.0,
                "evidence_coverage_rate": 0.0,
                "next_steps_coverage_rate": 0.0,
                "minimum_ok_rate": 0.0,
            },
        )
        bucket["count"] += 1
        bucket["_structure_ok"] = int(bucket.get("_structure_ok", 0)) + int(rec_structure_ok)
        bucket["_evidence_ok"] = int(bucket.get("_evidence_ok", 0)) + int(rec_evidence_ok)
        bucket["_next_ok"] = int(bucket.get("_next_ok", 0)) + int(rec_next_ok)
        bucket["_minimum_ok"] = int(bucket.get("_minimum_ok", 0)) + int(rec_minimum_ok)

        depth_key = str(rec.get("depth") or "unknown").strip().upper() or "UNKNOWN"
        d_bucket = by_depth.setdefault(
            depth_key,
            {
                "count": 0,
                "minimum_ok_rate": 0.0,
                "avg_char_count": 0.0,
            },
        )
        d_bucket["count"] += 1
        d_bucket["_minimum_ok"] = int(d_bucket.get("_minimum_ok", 0)) + int(rec_minimum_ok)
        d_bucket["_char_sum"] = int(d_bucket.get("_char_sum", 0)) + int(rec.get("char_count") or 0)

        if not rec_minimum_ok:
            reasons: list[str] = []
            if not has_conclusion:
                reasons.append("missing_conclusion")
            if not has_next_steps:
                reasons.append("missing_next_steps")
            if evidence_required and (not has_evidence):
                reasons.append("missing_evidence")
            if not reasons:
                reasons.append("other")
            for reason in reasons:
                fail_reasons[reason] = int(fail_reasons.get(reason, 0)) + 1

    def _ratio(ok: int, n: int) -> float:
        if n <= 0:
            return 0.0
        return round(float(ok) / float(n), 3)

    for bucket in by_intent.values():
        c = int(bucket.get("count") or 0)
        bucket["structure_complete_rate"] = _ratio(int(bucket.get("_structure_ok") or 0), c)
        bucket["evidence_coverage_rate"] = _ratio(int(bucket.get("_evidence_ok") or 0), c)
        bucket["next_steps_coverage_rate"] = _ratio(int(bucket.get("_next_ok") or 0), c)
        bucket["minimum_ok_rate"] = _ratio(int(bucket.get("_minimum_ok") or 0), c)
        bucket.pop("_structure_ok", None)
        bucket.pop("_evidence_ok", None)
        bucket.pop("_next_ok", None)
        bucket.pop("_minimum_ok", None)

    for d_bucket in by_depth.values():
        c = int(d_bucket.get("count") or 0)
        d_bucket["minimum_ok_rate"] = _ratio(int(d_bucket.get("_minimum_ok") or 0), c)
        if c > 0:
            d_bucket["avg_char_count"] = round(float(int(d_bucket.get("_char_sum") or 0)) / float(c), 1)
        else:
            d_bucket["avg_char_count"] = 0.0
        d_bucket.pop("_minimum_ok", None)
        d_bucket.pop("_char_sum", None)

    return {
        "limit": n_limit,
        "filters": {
            "intent": intent_filter,
            "depth": depth_filter,
            "only_failed": failed_only,
        },
        "total": total,
        "failed_count": failed_count,
        "failed_rate": _ratio(failed_count, total),
        "structure_complete_rate": _ratio(structure_ok, total),
        "evidence_coverage_rate": _ratio(evidence_ok, total),
        "next_steps_coverage_rate": _ratio(next_steps_ok, total),
        "minimum_ok_rate": _ratio(minimum_ok, total),
        "avg_core_section_coverage": round(core_cov_sum / float(total), 3),
        "by_intent": by_intent,
        "by_depth": by_depth,
        "fail_reasons": dict(sorted(fail_reasons.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))),
    }
