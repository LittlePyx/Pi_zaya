from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    from kb.chat_store import ChatStore
    from kb.config import load_settings
except ModuleNotFoundError:  # pragma: no cover
    # Support running the script directly without installing the project as a package.
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    from kb.chat_store import ChatStore  # type: ignore
    from kb.config import load_settings  # type: ignore


def _jsonl_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_get(d: object, key: str, default=None):
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


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


def _build_case(
    *,
    conv: dict,
    user_msg: dict | None,
    assistant_msg: dict,
    refs_by_user: dict[int, dict],
) -> dict | None:
    conv_id = str(conv.get("id") or "").strip()
    mode = str(conv.get("mode") or "").strip().lower()
    if mode != "paper_guide":
        return None

    msg_id = int(assistant_msg.get("id") or 0)
    if msg_id <= 0:
        return None

    prompt = str((user_msg or {}).get("content") or "").strip()
    assistant_content = str(assistant_msg.get("content") or "").strip()

    meta = _safe_get(assistant_msg, "meta", {}) or {}
    contracts = _safe_get(meta, "paper_guide_contracts", {}) or {}
    intent = _safe_get(contracts, "intent", {}) or {}
    intent_family = str(_safe_get(intent, "family", "") or "").strip()

    # Capture provenance as-is (it is already a JSON-serializable dict in ChatStore).
    provenance = _safe_get(assistant_msg, "provenance", None)
    if not isinstance(provenance, dict):
        provenance = None

    # Try to use the last known render_packet location as expected targets.
    render_packet = _safe_get(contracts, "render_packet", {}) or {}
    locate_target = _safe_get(render_packet, "locate_target", {}) or {}
    reader_open = _safe_get(render_packet, "reader_open", {}) or {}

    expect_block_id = str(_safe_get(reader_open, "blockId", "") or _safe_get(locate_target, "blockId", "") or "").strip()
    expect_anchor_id = str(_safe_get(reader_open, "anchorId", "") or _safe_get(locate_target, "anchorId", "") or "").strip()

    strict_locate = bool(_safe_get(reader_open, "strictLocate", False))
    locate_policy = str(_safe_get(locate_target, "locatePolicy", "") or "").strip().lower()
    locate_surface = str(_safe_get(locate_target, "locateSurfacePolicy", "") or "").strip().lower()
    expect_locate = bool(strict_locate or locate_policy == "required" or locate_surface == "primary")

    # Backfill expectations from provenance when render_packet is missing targets.
    if isinstance(provenance, dict):
        segs = provenance.get("segments")
        if isinstance(segs, list) and segs:
            # Match the contract builder's primary segment selection as closely as possible.
            # In _build_paper_guide_render_packet_model(): visible_segments are those
            # with locate_policy != hidden, and primary_segment defaults to visible_segments[0].
            candidate = None
            for seg in segs:
                if not isinstance(seg, dict):
                    continue
                lp = str(seg.get("locate_policy") or "").strip().lower()
                if lp and lp == "hidden":
                    continue
                candidate = seg
                break
            if candidate is None:
                candidate = next((seg for seg in segs if isinstance(seg, dict)), None)
            if isinstance(candidate, dict):
                if not expect_block_id:
                    expect_block_id = str(candidate.get("primary_block_id") or "").strip()
                if not expect_anchor_id:
                    expect_anchor_id = str(candidate.get("primary_anchor_id") or "").strip()
            if not expect_locate:
                # If the provenance explicitly marks any segment as required/primary, treat it as expected locate.
                for seg in segs:
                    if not isinstance(seg, dict):
                        continue
                    lp = str(seg.get("locate_policy") or "").strip().lower()
                    sp = str(seg.get("locate_surface_policy") or "").strip().lower()
                    if lp == "required" or sp == "primary" or bool(seg.get("must_locate")):
                        expect_locate = True
                        break

    user_msg_id = int((user_msg or {}).get("id") or 0)
    hits = []
    if user_msg_id > 0:
        ref_pack = refs_by_user.get(user_msg_id) if isinstance(refs_by_user, dict) else None
        if isinstance(ref_pack, dict):
            hits_raw = ref_pack.get("hits")
            if isinstance(hits_raw, list):
                hits = [dict(item) for item in hits_raw if isinstance(item, dict)]

    case_id = f"captured::{conv_id}::{msg_id}"
    return {
        "id": case_id,
        "fixture": "captured",
        "conv_id": conv_id,
        "msg_id": msg_id,
        "prompt": prompt,
        "intent_family": intent_family,
        "assistant_content": assistant_content,
        "expect_locate": bool(expect_locate),
        "expect_block_id": expect_block_id,
        "expect_anchor_id": expect_anchor_id,
        "provenance": provenance,
        "hits": hits,
    }


def _classify_case(case: dict) -> str:
    # Heuristic tag used for triage dashboards. Keep stable and low-effort.
    prompt = _normalize_text(case.get("prompt") or "")
    answer = str(case.get("assistant_content") or "").lower()
    intent_family = str(case.get("intent_family") or "").strip().lower()
    if _looks_like_beginner_component_role_prompt(prompt):
        return "overview"
    if "discussion" in intent_family:
        return "discussion_only"
    if "strength" in intent_family or "limitation" in intent_family:
        return "strength_limits"
    if intent_family:
        return intent_family
    if _looks_like_discussion_only_prompt(prompt):
        return "discussion_only"
    if _looks_like_strength_limits_prompt(prompt):
        return "strength_limits"
    if "equation" in prompt or "eq." in prompt or "eq(" in prompt or "式" in prompt or "公式" in prompt:
        return "equation"
    if "figure" in prompt or "fig." in prompt or "panel" in prompt or "图" in prompt or "子图" in prompt:
        return "figure"
    if "reference" in prompt or "citation" in prompt or "ref" in prompt or "引用" in prompt or "参考文献" in prompt:
        return "citation_lookup"
    if "method" in prompt or "pipeline" in prompt or "实现" in prompt or "训练" in prompt:
        return "method"
    if "beginner" in prompt or "通俗" in prompt or "入门" in prompt or "小白" in prompt:
        return "overview"
    if "[[cite:" in answer or "[cite:" in answer:
        return "citation_lookup"
    return "overview"


def _is_case_interesting(case: dict) -> bool:
    # "Interesting" == worth adding to the failure replay pool.
    # Prefer: must-locate segments, strict locate missing, fallback hit levels,
    # or any leakage signals in content.
    provenance = case.get("provenance")
    if not isinstance(provenance, dict):
        return False
    segments = provenance.get("segments")
    if not isinstance(segments, list):
        return False
    any_must = False
    any_must_fallback = False
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        must = bool(seg.get("must_locate")) or str(seg.get("locate_policy") or "").strip().lower() == "required"
        if must:
            any_must = True
            hit_level = str(seg.get("hit_level") or "").strip().lower()
            if hit_level and hit_level != "exact":
                any_must_fallback = True
    if any_must:
        # If must-locate but we did not get an explicit target, it's a failure candidate.
        if not str(case.get("expect_block_id") or "").strip():
            return True
        # Or strict locate isn't satisfied.
        if not bool(case.get("expect_locate")):
            return True
    if any_must_fallback:
        return True
    # Leakage guards based on stored raw assistant content.
    hay = (str(case.get("assistant_content") or "") + "\n" + str(case.get("prompt") or "")).upper()
    if "[[CITE:" in hay or "[SID:" in hay:
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect paper_guide replay cases from chat sqlite db.")
    ap.add_argument("--out", default=str(Path("tests/replay/paper_guide_failure_pool_captured.jsonl")), help="Output jsonl path")
    ap.add_argument("--limit", type=int, default=80, help="Max conversations to scan")
    ap.add_argument("--max-cases", type=int, default=120, help="Max cases to append")
    ap.add_argument("--failures-only", action="store_true", help="Only collect interesting failure candidates")
    ap.add_argument("--summary", action="store_true", help="Print a small summary by tag and failure signals")
    ap.add_argument("--include-archived", action="store_true", help="Include archived conversations")
    args = ap.parse_args()

    settings = load_settings()
    store = ChatStore(settings.chat_db_path)

    conversations = store.list_conversations(limit=max(1, int(args.limit)), include_archived=bool(args.include_archived))
    cases: list[dict] = []
    summary: dict[str, dict[str, int]] = {}
    for conv in conversations:
        if str(conv.get("mode") or "").strip().lower() != "paper_guide":
            continue
        conv_id = str(conv.get("id") or "").strip()
        if not conv_id:
            continue
        msgs = store.get_messages(conv_id)
        refs_by_user = store.list_message_refs(conv_id) or {}
        prev_user: dict | None = None
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip().lower()
            if role == "user":
                prev_user = msg
                continue
            if role != "assistant":
                continue
            case = _build_case(conv=conv, user_msg=prev_user, assistant_msg=msg, refs_by_user=refs_by_user)
            if not case:
                continue
            if args.failures_only and (not _is_case_interesting(case)):
                continue
            tag = _classify_case(case)
            case["tag"] = tag
            cases.append(case)
            if args.summary:
                bucket = summary.setdefault(tag, {"count": 0, "must_locate": 0, "no_target": 0, "fallback": 0})
                bucket["count"] += 1
                prov = case.get("provenance") if isinstance(case.get("provenance"), dict) else {}
                segs = prov.get("segments") if isinstance(prov.get("segments"), list) else []
                must = any(
                    (bool(s.get("must_locate")) or str(s.get("locate_policy") or "").strip().lower() == "required")
                    for s in segs
                    if isinstance(s, dict)
                )
                if must:
                    bucket["must_locate"] += 1
                if must and not str(case.get("expect_block_id") or "").strip():
                    bucket["no_target"] += 1
                if any(
                    (bool(s.get("must_locate")) or str(s.get("locate_policy") or "").strip().lower() == "required")
                    and str(s.get("hit_level") or "").strip().lower() not in {"", "exact"}
                    for s in segs
                    if isinstance(s, dict)
                ):
                    bucket["fallback"] += 1
            if len(cases) >= int(args.max_cases):
                break
        if len(cases) >= int(args.max_cases):
            break

    out_path = Path(str(args.out))
    _jsonl_write(out_path, cases)
    print(f"appended_cases={len(cases)} out={out_path}")
    if args.summary:
        for tag in sorted(summary.keys()):
            bucket = summary[tag]
            print(
                "summary"
                + f" tag={tag}"
                + f" count={bucket.get('count', 0)}"
                + f" must_locate={bucket.get('must_locate', 0)}"
                + f" no_target={bucket.get('no_target', 0)}"
                + f" fallback={bucket.get('fallback', 0)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
