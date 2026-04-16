from __future__ import annotations

import hashlib
import time
from pathlib import Path

from kb.inpaper_citation_grounding import (
    extract_candidate_ref_cue_texts,
    extract_candidate_ref_nums_from_hits,
)
from kb.paper_guide_citation_surfacing import (
    _collect_paper_guide_candidate_refs_by_source,
)
from kb.paper_guide_contracts import (
    _build_paper_guide_retrieval_bundle_model,
    _build_paper_guide_support_pack_model,
    _paper_guide_model_dump,
)
from kb.paper_guide_focus import _build_paper_guide_special_focus_block
from kb.paper_guide.grounder import (
    _build_paper_guide_support_slots,
    _build_paper_guide_support_slots_block,
)
from kb.paper_guide_prompting import (
    _build_paper_guide_citation_grounding_block,
    _build_paper_guide_evidence_cards_block,
    _merge_paper_guide_deepread_context,
    _paper_guide_allows_citeless_answer,
)
from kb.paper_guide.router import _resolve_paper_guide_intent
from kb.paper_guide_retrieval_runtime import _select_paper_guide_deepread_extras
from kb.paper_guide_shared import _cite_source_id, _source_name_from_md_path
from kb.paper_guide_target_scope import _build_paper_guide_target_scope
from kb.retrieval_engine import _deep_read_md_for_context, _top_heading
from kb.retrieval_heuristics import _is_probably_bad_heading


def _hit_source_path(hit: dict) -> str:
    if not isinstance(hit, dict):
        return ""
    meta = hit.get("meta", {}) or {}
    return str(meta.get("source_path") or "").strip()


def _positive_int(value) -> int | None:
    try:
        iv = int(value)
    except Exception:
        return None
    return iv if iv > 0 else None


def _first_answer_hit_snippet(hit: dict) -> str:
    if not isinstance(hit, dict):
        return ""
    meta = hit.get("meta", {}) or {}
    for key in ("ref_show_snippets", "ref_snippets"):
        raw = meta.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            text = str(item or "").strip()
            if text:
                return text
    for key in ("text", "snippet"):
        text = str(hit.get(key) or "").strip()
        if text:
            return text
    return ""


def _build_answer_hit_primary_evidence(
    hit: dict,
    *,
    source_path: str,
    source_name: str,
    heading: str,
    snippet: str = "",
) -> dict:
    if not isinstance(hit, dict):
        return {}
    meta = hit.get("meta", {}) or {}
    snippet_text = str(snippet or "").strip() or _first_answer_hit_snippet(hit)
    out = {
        "source_path": str(source_path or "").strip() or None,
        "source_name": str(source_name or "").strip() or None,
        "block_id": str(meta.get("block_id") or meta.get("ref_block_id") or "").strip() or None,
        "anchor_id": str(meta.get("anchor_id") or meta.get("ref_anchor_id") or "").strip() or None,
        "heading_path": str(
            heading
            or meta.get("ref_best_heading_path")
            or meta.get("heading_path")
            or meta.get("top_heading")
            or ""
        ).strip() or None,
        "snippet": snippet_text or None,
        "anchor_kind": str(meta.get("anchor_target_kind") or meta.get("anchor_kind") or "").strip().lower() or None,
        "anchor_number": _positive_int(meta.get("anchor_target_number") or meta.get("anchor_number")),
        "selection_reason": "answer_hit_top",
    }
    return {
        key: value
        for key, value in out.items()
        if value not in (None, "", [], {})
    }


def _select_seed_primary_evidence(evidence_cards: list[dict] | None) -> dict:
    for card in list(evidence_cards or []):
        if not isinstance(card, dict):
            continue
        primary = card.get("primary_evidence")
        if isinstance(primary, dict) and primary:
            return dict(primary)
    return {}


def _build_paper_guide_context_records(
    answer_hits: list[dict],
    *,
    paper_guide_mode: bool,
) -> dict:
    ctx_parts: list[str] = []
    doc_first_idx: dict[str, int] = {}
    paper_guide_evidence_cards: list[dict] = []
    paper_guide_card_by_doc_idx: dict[int, dict] = {}

    for i, hit in enumerate(answer_hits or [], start=1):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta", {}) or {}
        src = _hit_source_path(hit)
        if src and src not in doc_first_idx:
            doc_first_idx[src] = i
        src_name = _source_name_from_md_path(src) if src else "unknown"
        focus_heading = (
            str(meta.get("ref_best_heading_path") or "").strip()
            or str(meta.get("top_heading") or "").strip()
            or str(_top_heading(meta.get("heading_path", "")) or "").strip()
        )
        top = "" if _is_probably_bad_heading(focus_heading) else focus_heading
        sid = _cite_source_id(src)
        header = f"DOC-{i} [SID:{sid}] {src_name or 'unknown'}" + (f" | {top}" if top else "")

        candidate_refs: list[int] = []
        cue_texts: list[str] = []
        if paper_guide_mode and src:
            candidate_refs = extract_candidate_ref_nums_from_hits([hit], source_path=src, max_candidates=6)
            cue_texts = extract_candidate_ref_cue_texts(hit, max_cues=1, max_chars=160)
            if candidate_refs:
                refs_txt = ", ".join(str(int(n)) for n in candidate_refs[:6])
                header += f" | candidate refs: {refs_txt}"

        body = ""
        ref_show_snippets = meta.get("ref_show_snippets")
        if isinstance(ref_show_snippets, list):
            parts: list[str] = []
            seen_parts: set[str] = set()
            for raw_part in ref_show_snippets[:2]:
                part = str(raw_part or "").strip()
                if not part:
                    continue
                key = hashlib.sha1(part.encode("utf-8", "ignore")).hexdigest()[:12]
                if key in seen_parts:
                    continue
                seen_parts.add(key)
                parts.append(part)
            if parts:
                body = "\n\n".join(parts)
        if not body:
            body = str(hit.get("text") or "")

        ctx_parts.append(header + "\n" + body)
        if src:
            primary_evidence = _build_answer_hit_primary_evidence(
                hit,
                source_path=src,
                source_name=src_name,
                heading=(
                    str(meta.get("ref_best_heading_path") or "").strip()
                    or str(meta.get("heading_path") or "").strip()
                    or focus_heading
                    or top
                ),
                snippet=body,
            )
            card = {
                "doc_idx": int(i),
                "sid": sid,
                "source_path": src,
                "heading": top,
                "candidate_refs": [int(n) for n in candidate_refs[:6]],
                "cue": str(cue_texts[0] or "").strip() if cue_texts else "",
                "snippet": body,
                "deepread_texts": [],
            }
            if primary_evidence:
                card["primary_evidence"] = dict(primary_evidence)
            paper_guide_evidence_cards.append(card)
            paper_guide_card_by_doc_idx[int(i)] = card

    return {
        "ctx_parts": ctx_parts,
        "doc_first_idx": doc_first_idx,
        "paper_guide_evidence_cards": paper_guide_evidence_cards,
        "paper_guide_card_by_doc_idx": paper_guide_card_by_doc_idx,
    }


def _apply_paper_guide_deepread_context(
    *,
    ctx_parts: list[str],
    doc_first_idx: dict[str, int],
    paper_guide_card_by_doc_idx: dict[int, dict],
    prompt: str,
    retrieval_prompt: str,
    used_query: str,
    prompt_family: str,
    deep_read: bool,
    answer_hits: list[dict],
    should_cancel=None,
    on_stage=None,
    deep_read_fn=_deep_read_md_for_context,
    select_extras_fn=_select_paper_guide_deepread_extras,
    merge_context_fn=_merge_paper_guide_deepread_context,
    allows_citeless_answer_fn=_paper_guide_allows_citeless_answer,
    deep_budget_s: float = 9.0,
) -> dict:
    deep_added = 0
    deep_docs = 0
    updated_ctx_parts = list(ctx_parts or [])
    if (not deep_read) or (not answer_hits):
        return {
            "ctx_parts": updated_ctx_parts,
            "deep_added": 0,
            "deep_docs": 0,
        }

    deep_begin = time.monotonic()
    fine_query = str(used_query or retrieval_prompt or prompt or "").strip()
    items = list((doc_first_idx or {}).items())[: min(3, max(1, len(doc_first_idx or {})))]
    total = len(items)
    for n, (src, idx0) in enumerate(items, start=1):
        if callable(should_cancel) and should_cancel():
            raise RuntimeError("canceled")
        if (time.monotonic() - deep_begin) >= float(deep_budget_s):
            if callable(on_stage):
                on_stage("deep-read skipped (timeout)")
            break
        if callable(on_stage):
            on_stage(f"deep-read {n}/{total}")
        extras: list[dict] = []
        if fine_query:
            deep_snippet_cap = 4 if str(prompt_family or "").strip().lower() == "abstract" else 2
            extras.extend(deep_read_fn(Path(src), fine_query, max_snippets=deep_snippet_cap, snippet_chars=1000))
        if not extras:
            continue
        deep_docs += 1
        extras2 = select_extras_fn(
            extras,
            prompt=(prompt or retrieval_prompt or fine_query),
            prompt_family=prompt_family,
            limit=1 if allows_citeless_answer_fn(prompt_family) else 2,
        )
        if not extras2:
            continue
        try:
            base = updated_ctx_parts[int(idx0) - 1]
        except Exception:
            continue
        for text in extras2:
            merged = merge_context_fn(
                base,
                text,
                prompt=(prompt or retrieval_prompt or fine_query),
                prompt_family=prompt_family,
            )
            if merged == base:
                continue
            base = merged
            deep_added += 1
            card = paper_guide_card_by_doc_idx.get(int(idx0))
            if isinstance(card, dict):
                texts = card.setdefault("deepread_texts", [])
                if isinstance(texts, list) and (text not in texts):
                    texts.append(text)
                if str(prompt_family or "").strip().lower() in {"abstract", "figure_walkthrough"}:
                    card["snippet"] = text
        updated_ctx_parts[int(idx0) - 1] = base

    return {
        "ctx_parts": updated_ctx_parts,
        "deep_added": int(deep_added),
        "deep_docs": int(deep_docs),
    }


def _prepare_paper_guide_prompt_context(
    *,
    paper_guide_mode: bool,
    paper_guide_bound_source_ready: bool,
    answer_hits: list[dict],
    paper_guide_evidence_cards: list[dict],
    prompt: str,
    retrieval_prompt: str,
    used_query: str,
    prompt_family: str,
    paper_guide_bound_source_path: str,
    db_dir,
) -> dict:
    paper_guide_evidence_cards_block = ""
    paper_guide_support_slots_block = ""
    paper_guide_special_focus_block = ""
    paper_guide_citation_grounding_block = ""
    paper_guide_candidate_refs_by_source: dict[str, list[int]] = {}
    paper_guide_support_slots: list[dict] = []
    paper_guide_target_scope = _build_paper_guide_target_scope(
        prompt or retrieval_prompt or used_query,
        prompt_family=prompt_family,
    )
    paper_guide_direct_source_path = str(paper_guide_bound_source_path or "").strip()
    paper_guide_focus_source_path = str(paper_guide_bound_source_path or "").strip()

    if paper_guide_mode and _paper_guide_allows_citeless_answer(prompt_family):
        for hit in answer_hits or []:
            src_hit = _hit_source_path(hit)
            if src_hit:
                paper_guide_direct_source_path = src_hit
                break
    for hit in answer_hits or []:
        src_hit = _hit_source_path(hit)
        if src_hit:
            paper_guide_focus_source_path = src_hit
            break

    if paper_guide_mode and paper_guide_bound_source_ready:
        prompt_text = prompt or retrieval_prompt or used_query
        paper_guide_support_slots = _build_paper_guide_support_slots(
            paper_guide_evidence_cards,
            prompt=prompt_text,
            prompt_family=prompt_family,
            db_dir=db_dir,
            target_scope=paper_guide_target_scope,
        )
        paper_guide_evidence_cards_block = _build_paper_guide_evidence_cards_block(
            paper_guide_evidence_cards,
            prompt=prompt_text,
            prompt_family=prompt_family,
        )
        paper_guide_support_slots_block = _build_paper_guide_support_slots_block(
            paper_guide_support_slots,
        )
        paper_guide_special_focus_block = _build_paper_guide_special_focus_block(
            paper_guide_evidence_cards,
            prompt=prompt_text,
            prompt_family=prompt_family,
            source_path=paper_guide_focus_source_path or paper_guide_direct_source_path or paper_guide_bound_source_path,
            db_dir=db_dir,
            answer_hits=answer_hits,
        )
        paper_guide_candidate_refs_by_source = _collect_paper_guide_candidate_refs_by_source(
            paper_guide_evidence_cards,
            focus_source_path=paper_guide_focus_source_path or paper_guide_direct_source_path or paper_guide_bound_source_path,
            special_focus_block=paper_guide_special_focus_block,
            prompt_family=prompt_family,
            prompt=prompt_text,
            db_dir=db_dir,
        )

    if (
        paper_guide_mode
        and paper_guide_bound_source_ready
        and (not _paper_guide_allows_citeless_answer(prompt_family))
    ):
        paper_guide_citation_grounding_block = _build_paper_guide_citation_grounding_block(answer_hits)

    prompt_text = prompt or retrieval_prompt or used_query
    paper_guide_contracts_seed = {}
    if paper_guide_mode:
        resolved_intent = _resolve_paper_guide_intent(
            prompt_text,
            prompt_family=prompt_family,
            answer_hits=answer_hits,
        )
        support_pack_model = _build_paper_guide_support_pack_model(
            family=str(getattr(resolved_intent, "family", "") or "").strip(),
            answer_markdown="",
            support_records=list(paper_guide_support_slots or []),
            needs_supplement=False,
        )
        prompt_context = {
            "target_scope": dict(paper_guide_target_scope or {}) if isinstance(paper_guide_target_scope, dict) else {},
            "direct_source_path": paper_guide_direct_source_path,
            "focus_source_path": paper_guide_focus_source_path,
            "bound_source_path": str(paper_guide_bound_source_path or "").strip(),
        }
        paper_guide_contracts_seed = {
            "version": 1,
            "intent": _paper_guide_model_dump(resolved_intent),
            "retrieval_bundle": _paper_guide_model_dump(
                _build_paper_guide_retrieval_bundle_model(
                    prompt_family=str(getattr(resolved_intent, "family", "") or "").strip(),
                    target_scope=paper_guide_target_scope,
                    evidence_cards=list(paper_guide_evidence_cards or []),
                    candidate_refs_by_source=dict(paper_guide_candidate_refs_by_source or {}),
                    direct_source_path=paper_guide_direct_source_path,
                    focus_source_path=paper_guide_focus_source_path,
                    bound_source_path=str(paper_guide_bound_source_path or "").strip(),
                )
            ),
            "support_pack": _paper_guide_model_dump(support_pack_model),
            "prompt_context": {
                key: value
                for key, value in prompt_context.items()
                if value not in (None, "", [], {})
            },
        }
        seed_primary_evidence = _select_seed_primary_evidence(paper_guide_evidence_cards)
        if seed_primary_evidence:
            paper_guide_contracts_seed["primary_evidence"] = dict(seed_primary_evidence)

    return {
        "paper_guide_evidence_cards_block": paper_guide_evidence_cards_block,
        "paper_guide_support_slots_block": paper_guide_support_slots_block,
        "paper_guide_special_focus_block": paper_guide_special_focus_block,
        "paper_guide_citation_grounding_block": paper_guide_citation_grounding_block,
        "paper_guide_candidate_refs_by_source": paper_guide_candidate_refs_by_source,
        "paper_guide_support_slots": paper_guide_support_slots,
        "paper_guide_target_scope": paper_guide_target_scope,
        "paper_guide_direct_source_path": paper_guide_direct_source_path,
        "paper_guide_focus_source_path": paper_guide_focus_source_path,
        "paper_guide_contracts_seed": paper_guide_contracts_seed,
    }
