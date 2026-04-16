from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parent
_DEFAULT_CAPTURED_SOURCES = [
    _ROOT / "paper_guide_failure_pool_captured_failures.jsonl",
    _ROOT / "paper_guide_failure_pool_captured_promoted.jsonl",
    _ROOT / "paper_guide_failure_pool_captured.jsonl",
]
_DEFAULT_POOL = _ROOT / "paper_guide_failure_pool_v1.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8-sig").splitlines():
        s = str(ln or "").strip()
        if not s or s.startswith("#"):
            continue
        rec = json.loads(s)
        if isinstance(rec, dict):
            out.append(dict(rec))
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text((payload + "\n") if payload else "", encoding="utf-8")


def _load_jsonl_many(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        out.extend(_load_jsonl(path))
    return out


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _looks_like_beginner_component_role_prompt(prompt: str) -> bool:
    p = _normalize_text(prompt)
    if not p:
        return False
    if any(
        token in p
        for token in (
            "figure",
            "fig.",
            "panel",
            "reference",
            "citation",
            "图",
            "子图",
            "引用",
            "参考文献",
            "equation",
            "eq.",
            "eq(",
            "公式",
            "式",
        )
    ):
        return False
    has_beginner = any(
        token in p
        for token in (
            "beginner",
            "new to this paper",
            "just getting started",
            "getting started",
            "in simple terms",
            "通俗",
            "入门",
            "小白",
        )
    )
    has_role = bool(re.search(r"\bwhat\s+(?:is|are)\b.{0,80}\bdoing here\b", p)) or "what role do they play" in p or "what role does it play" in p
    return bool(has_beginner and has_role)


def _looks_like_discussion_only_prompt(prompt: str) -> bool:
    p = _normalize_text(prompt)
    if not p:
        return False
    return bool(
        re.search(r"\b(?:from\s+the\s+)?discussion(?:\s+section)?\s+only\b", p, flags=re.IGNORECASE)
        or re.search(r"\b(?:from\s+the\s+)?future(?:\s+work)?(?:\s+section)?\s+only\b", p, flags=re.IGNORECASE)
        or ("discussion section" in p and "future" in p)
        or ("discussion" in p and "future direction" in p)
        or ("discussion" in p and "future work" in p)
        or ("future work section" in p and "only" in p)
    )


def _looks_like_strength_limits_prompt(prompt: str) -> bool:
    p = _normalize_text(prompt)
    if not p:
        return False
    return bool(
        re.search(
            r"\b(?:limitation|limitations|weakness|weaknesses|trade[\s-]?off|trade[\s-]?offs|what is missing|strongest evidence)\b",
            p,
            flags=re.IGNORECASE,
        )
        or any(token in p for token in ("局限", "不足", "缺点", "薄弱", "证据"))
    )


def _classify_case(case: dict[str, Any]) -> str:
    raw = str(case.get("tag") or case.get("intent_family") or "").strip().lower()
    if "discussion" in raw:
        return "discussion_only"
    if "strength" in raw or "limitation" in raw:
        return "strength_limits"
    if raw:
        return raw
    prompt = _normalize_text(case.get("prompt"))
    if _looks_like_beginner_component_role_prompt(prompt):
        return "overview"
    if _looks_like_discussion_only_prompt(prompt):
        return "discussion_only"
    if _looks_like_strength_limits_prompt(prompt):
        return "strength_limits"
    answer = _normalize_text(case.get("assistant_content"))
    if any(token in prompt for token in ("equation", "eq.", "eq(", "式", "公式")):
        return "equation"
    if any(token in prompt for token in ("figure", "fig.", "panel", "图", "子图")):
        return "figure"
    if any(token in prompt for token in ("reference", "citation", "引用", "参考文献")):
        return "citation_lookup"
    if any(token in prompt for token in ("method", "pipeline", "训练", "实现")):
        return "method"
    if "[[cite:" in answer or "[cite:" in answer:
        return "citation_lookup"
    return "overview"


def _case_source_key(case: dict[str, Any]) -> str:
    provenance = case.get("provenance") if isinstance(case.get("provenance"), dict) else {}
    source_path = str(provenance.get("source_path") or "").strip()
    if source_path:
        return source_path.lower()
    hits = case.get("hits") if isinstance(case.get("hits"), list) else []
    for item in hits:
        if not isinstance(item, dict):
            continue
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        source_path = str(meta.get("source_path") or "").strip()
        if source_path:
            return source_path.lower()
    return ""


def _case_doc_id(case: dict[str, Any]) -> str:
    provenance = case.get("provenance") if isinstance(case.get("provenance"), dict) else {}
    doc_id = str(provenance.get("doc_id") or "").strip().lower()
    if doc_id:
        return doc_id
    source_key = _case_source_key(case)
    if source_key:
        stem = Path(source_key).stem.lower()
        stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
        if stem:
            return stem[:24]
    return "case"


def _interesting_score(case: dict[str, Any]) -> tuple[float, int, str]:
    score = 0.0
    if bool(case.get("expect_locate")):
        score += 3.0
    if str(case.get("expect_block_id") or "").strip():
        score += 1.0
    if str(case.get("expect_anchor_id") or "").strip():
        score += 0.8
    answer = str(case.get("assistant_content") or "")
    if "[[CITE:" in answer or "[SID:" in answer:
        score += 1.5
    answer_len = len(answer.strip())
    if 120 <= answer_len <= 1600:
        score += 0.8
    elif answer_len > 2200:
        score -= 0.4
    provenance = case.get("provenance") if isinstance(case.get("provenance"), dict) else {}
    segments = provenance.get("segments") if isinstance(provenance.get("segments"), list) else []
    must_fallback = 0
    must_exact = 0
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        must = bool(seg.get("must_locate")) or str(seg.get("locate_policy") or "").strip().lower() == "required"
        if not must:
            continue
        hit_level = str(seg.get("hit_level") or "").strip().lower()
        if hit_level == "exact":
            must_exact += 1
        elif hit_level:
            must_fallback += 1
    score += min(2.0, 0.5 * float(must_exact))
    score += min(3.0, 1.2 * float(must_fallback))
    try:
        msg_id = int(case.get("msg_id") or 0)
    except Exception:
        msg_id = 0
    return score, msg_id, str(case.get("id") or "")


def _case_signature(case: dict[str, Any]) -> str:
    return "::".join(
        [
            _classify_case(case),
            _case_source_key(case),
            _normalize_text(case.get("prompt")),
        ]
    )


def _best_cases_by_signature(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for case in cases:
        sig = _case_signature(case)
        prev = best.get(sig)
        if prev is None or _interesting_score(case) > _interesting_score(prev):
            best[sig] = case
    return list(best.values())


def _parse_limit_overrides(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in values:
        text = str(raw or "").strip()
        if not text or "=" not in text:
            continue
        tag, limit = text.split("=", 1)
        tag_norm = str(tag or "").strip().lower()
        try:
            limit_num = int(limit)
        except Exception:
            continue
        if tag_norm and limit_num > 0:
            out[tag_norm] = limit_num
    return out


def _curated_ref_row(case: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case.get("id") or "").strip()
    tag = _classify_case(case)
    try:
        msg_id = int(case.get("msg_id") or 0)
    except Exception:
        msg_id = 0
    ref_id = f"captured_{tag}_{_case_doc_id(case)}_{msg_id or 0}"
    return {
        "id": ref_id,
        "fixture": "captured_ref",
        "captured_case_id": case_id,
        "tag": tag,
    }


def curate_cases(
    *,
    captured_cases: list[dict[str, Any]],
    existing_pool: list[dict[str, Any]] | None = None,
    per_tag_limits: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    captured_index = {
        str(case.get("id") or "").strip(): dict(case)
        for case in captured_cases
        if str(case.get("id") or "").strip()
    }
    existing_case_ids = {
        str(row.get("captured_case_id") or row.get("source_case_id") or "").strip()
        for row in list(existing_pool or [])
        if isinstance(row, dict)
    }
    existing_signatures = {
        _case_signature(captured_index[case_id])
        for case_id in existing_case_ids
        if case_id in captured_index
    }
    unique_cases = [
        case
        for case in _best_cases_by_signature(captured_cases)
        if str(case.get("id") or "").strip()
        and str(case.get("id") or "").strip() not in existing_case_ids
        and _case_signature(case) not in existing_signatures
    ]
    ranked = sorted(unique_cases, key=_interesting_score, reverse=True)
    limits = dict(per_tag_limits or {})
    tag_counts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    for case in ranked:
        tag = _classify_case(case)
        limit = int(limits.get(tag) or 0)
        if limit > 0 and int(tag_counts.get(tag) or 0) >= limit:
            continue
        tag_counts[tag] = int(tag_counts.get(tag) or 0) + 1
        selected.append(case)
    return selected


def main() -> int:
    ap = argparse.ArgumentParser(description="Curate lightweight captured_ref rows for paper guide replay pool v1.")
    ap.add_argument(
        "--captured",
        action="append",
        default=[],
        help="Captured jsonl path. Can be repeated; defaults to failures + promoted + general captured pools.",
    )
    ap.add_argument("--pool", default=str(_DEFAULT_POOL), help="Existing replay pool v1 path")
    ap.add_argument("--out", default="", help="Optional output jsonl path; omit to print selected rows")
    ap.add_argument(
        "--limit",
        action="append",
        default=[],
        help="Optional per-tag cap, e.g. citation_lookup=4. Can be repeated.",
    )
    ap.add_argument(
        "--include-existing",
        action="store_true",
        help="When used with --out, write existing pool rows followed by selected captured_ref rows.",
    )
    args = ap.parse_args()

    captured_paths = [Path(str(item)) for item in list(args.captured or []) if str(item).strip()]
    if not captured_paths:
        captured_paths = list(_DEFAULT_CAPTURED_SOURCES)
    pool_path = Path(str(args.pool))
    captured_rows = _load_jsonl_many(captured_paths)
    pool_rows = _load_jsonl(pool_path) if pool_path.exists() else []
    selected_cases = curate_cases(
        captured_cases=captured_rows,
        existing_pool=pool_rows,
        per_tag_limits=_parse_limit_overrides(list(args.limit or [])),
    )
    selected_rows = [_curated_ref_row(case) for case in selected_cases]

    summary: dict[str, int] = {}
    for row in selected_rows:
        tag = str(row.get("tag") or "").strip().lower() or "overview"
        summary[tag] = int(summary.get(tag) or 0) + 1

    captured_desc = ",".join(str(path) for path in captured_paths)
    print(f"selected_cases={len(selected_rows)} captured={captured_desc}")
    for tag in sorted(summary.keys()):
        print(f"summary tag={tag} count={summary[tag]}")

    if args.out:
        out_path = Path(str(args.out))
        rows = [dict(item) for item in pool_rows] if args.include_existing else []
        rows.extend(selected_rows)
        _write_jsonl(out_path, rows)
        print(f"wrote={len(rows)} out={out_path}")
    else:
        for row in selected_rows:
            print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
