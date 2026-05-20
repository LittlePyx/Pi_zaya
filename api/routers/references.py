from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from urllib.parse import quote
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.deps import get_chat_store, get_settings, load_prefs
from api.reference_ui import (
    _attach_pack_display_contract,
    _compact_reader_open_text,
    _filter_pending_refs_hits_by_prompt_focus,
    _refs_card_polish_llm_enabled,
    _refs_prompt_focus_terms,
    build_doc_list_refs_payload,
    enrich_citation_detail_meta,
    enrich_refs_payload,
    ensure_source_citation_meta,
    open_reference_source,
)
from api.reference_card_quality import refs_pack_has_full_llm_copy
from kb.generation_answer_finalize_runtime import (
    _build_multi_paper_doc_list_contract as _references_build_multi_paper_doc_list_contract,
)
from kb.file_ops import _resolve_md_output_paths
from kb.library_store import LibraryStore
from kb.reference_query_family import (
    prompt_explicitly_requests_multi_paper_list,
    prompt_reference_focus_action,
)
from kb.paper_guide_shared import _source_name_from_md_path
from kb.source_blocks import load_source_blocks, source_blocks_to_reader_anchors
from api.sse import sse_generator, sse_response
from kb.reference_sync import (
    start_reference_sync,
    snapshot as refsync_snapshot,
)

router = APIRouter(prefix="/api/references", tags=["references"])

_REFS_CONVERSATION_CACHE: dict[str, dict] = {}
_REFS_CONVERSATION_WARMING: set[str] = set()
_REFS_CONVERSATION_WARMING_LOCK = threading.Lock()
_REFS_RENDER_PAYLOAD_SCHEMA_VERSION = 9


def _md_dir() -> Path:
    from api.routers.library import _md_dir
    return _md_dir()


def _pdf_dir() -> Path:
    from api.routers.library import _pdf_dir
    return _pdf_dir()


def _lib_store() -> LibraryStore:
    return LibraryStore(get_settings().library_db_path)


def _project_root() -> Path:
    s = get_settings()
    return Path(s.db_dir).expanduser().resolve().parent


def _reference_asset_roots() -> list[Path]:
    roots: list[Path] = []
    for raw in (_md_dir(), _project_root() / "tmp"):
        try:
            resolved = Path(raw).expanduser().resolve(strict=False)
        except Exception:
            continue
        if resolved in roots:
            continue
        roots.append(resolved)
    return roots


def _path_within_roots(path_obj: Path, roots: list[Path]) -> bool:
    p = Path(path_obj)
    for root in roots:
        try:
            p.relative_to(root)
            return True
        except Exception:
            continue
    return False


def _refs_conversation_cache_ttl_s() -> float:
    try:
        raw = float(str(os.environ.get("KB_REFS_CONVERSATION_CACHE_TTL_S", "6") or "6"))
    except Exception:
        raw = 6.0
    return max(0.0, min(30.0, raw))


def _refs_conversation_cache_signature(
    *,
    refs: dict,
    guide_mode: bool,
    guide_source_path: str,
    guide_source_name: str,
    authoritative_doc_list_by_user: dict[int, list[dict]] | None = None,
) -> str:
    try:
        prefs = load_prefs()
    except Exception:
        prefs = {}
    authoritative_map: dict[int, list[dict]] = {}
    for key, value in dict(authoritative_doc_list_by_user or {}).items():
        try:
            user_msg_key = int(key)
        except Exception:
            continue
        authoritative_map[user_msg_key] = [dict(item) for item in list(value or []) if isinstance(item, dict)]
    refs_digest: list[dict] = []
    for user_msg_id, pack in sorted((refs or {}).items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0])):
        if not isinstance(pack, dict):
            continue
        try:
            user_msg_key = int(user_msg_id)
        except Exception:
            user_msg_key = 0
        hits = list(pack.get("hits") or [])
        pending_count = 0
        source_keys: list[str] = []
        for hit in hits[:4]:
            if not isinstance(hit, dict):
                continue
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            if str((meta or {}).get("ref_pack_state") or "").strip().lower() == "pending":
                pending_count += 1
            source_path = str((meta or {}).get("source_path") or "").strip()
            if source_path:
                source_keys.append(source_path)
        doc_list_sig: list[dict] = []
        for item in list(authoritative_map.get(user_msg_key, []) or [])[:4]:
            source_path = str(item.get("source_path") or "").strip()
            source_name = str(item.get("source_name") or "").strip()
            heading_path = str(item.get("heading_path") or "").strip()
            if source_path or source_name:
                doc_list_sig.append(
                    {
                        "source_path": source_path,
                        "source_name": source_name,
                        "heading_path": heading_path,
                    }
                )
        payload = {
            "user_msg_id": user_msg_key if user_msg_key > 0 else str(user_msg_id),
            "prompt_sig": str(pack.get("prompt_sig") or "").strip(),
            "used_query": str(pack.get("used_query") or "").strip(),
            "used_translation": bool(pack.get("used_translation")),
            "updated_at": float(pack.get("updated_at") or 0.0),
            "render_status": str(pack.get("render_status") or "").strip().lower(),
            "rendered_payload_sig": str(pack.get("rendered_payload_sig") or "").strip(),
            "hit_count": len(hits),
            "pending_count": pending_count,
            "top_sources": source_keys,
            "authoritative_doc_list": doc_list_sig,
        }
        refs_digest.append(payload)
    payload = {
        "render_schema": _REFS_RENDER_PAYLOAD_SCHEMA_VERSION,
        "guide_mode": bool(guide_mode),
        "guide_source_path": str(guide_source_path or "").strip(),
        "guide_source_name": str(guide_source_name or "").strip(),
        "refs_background_llm_polish": bool(_refs_background_llm_polish_enabled()),
        "refs_card_locale": str((prefs or {}).get("refs_card_locale") or "").strip().lower(),
        "ui_locale": str((prefs or {}).get("ui_locale") or "").strip().lower(),
        "refs_digest": refs_digest,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _refs_pack_render_signature(
    *,
    user_msg_id: int | str,
    pack: dict,
    guide_mode: bool,
    guide_source_path: str,
    guide_source_name: str,
) -> str:
    try:
        prefs = load_prefs()
    except Exception:
        prefs = {}
    payload = {
        "render_schema": _REFS_RENDER_PAYLOAD_SCHEMA_VERSION,
        "user_msg_id": int(user_msg_id) if str(user_msg_id).isdigit() else str(user_msg_id),
        "guide_mode": bool(guide_mode),
        "guide_source_path": str(guide_source_path or "").strip(),
        "guide_source_name": str(guide_source_name or "").strip(),
        "refs_background_llm_polish": bool(_refs_background_llm_polish_enabled()),
        "refs_card_locale": str((prefs or {}).get("refs_card_locale") or "").strip().lower(),
        "ui_locale": str((prefs or {}).get("ui_locale") or "").strip().lower(),
        "prompt": str((pack or {}).get("prompt") or "").strip(),
        "prompt_sig": str((pack or {}).get("prompt_sig") or "").strip(),
        "used_query": str((pack or {}).get("used_query") or "").strip(),
        "used_translation": bool((pack or {}).get("used_translation")),
        "hits": list((pack or {}).get("hits") or []),
        "scores": list((pack or {}).get("scores") or []),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _get_cached_conversation_refs_record(*, conv_id: str, signature: str) -> dict | None:
    ttl_s = _refs_conversation_cache_ttl_s()
    if ttl_s <= 0:
        return None
    rec = _REFS_CONVERSATION_CACHE.get(str(conv_id or "").strip())
    if not isinstance(rec, dict):
        return None
    if str(rec.get("signature") or "") != str(signature or ""):
        return None
    try:
        cached_at = float(rec.get("cached_at") or 0.0)
    except Exception:
        cached_at = 0.0
    if cached_at <= 0 or (time.time() - cached_at) > ttl_s:
        return None
    payload = rec.get("payload")
    if not isinstance(payload, dict):
        return None
    return rec


def _get_cached_conversation_refs_payload(*, conv_id: str, signature: str) -> dict | None:
    rec = _get_cached_conversation_refs_record(conv_id=conv_id, signature=signature)
    if not isinstance(rec, dict):
        return None
    payload = rec.get("payload")
    return payload if isinstance(payload, dict) else None


def _store_cached_conversation_refs_payload(*, conv_id: str, signature: str, payload: dict, mode: str = "full") -> None:
    _REFS_CONVERSATION_CACHE[str(conv_id or "").strip()] = {
        "signature": str(signature or ""),
        "cached_at": time.time(),
        "mode": str(mode or "full").strip().lower() or "full",
        "payload": dict(payload or {}),
    }


def _get_any_cached_conversation_refs_payload(*, conv_id: str) -> dict | None:
    rec = _REFS_CONVERSATION_CACHE.get(str(conv_id or "").strip())
    if not isinstance(rec, dict):
        return None
    payload = rec.get("payload")
    return payload if isinstance(payload, dict) else None


def _refs_perf_ms(started_at: float) -> float:
    return max(0.0, (time.perf_counter() - float(started_at or time.perf_counter())) * 1000.0)


def _refs_payload_counts_for_header(payload: dict | None) -> str:
    packs = 0
    hits = 0
    pending = 0
    fast = 0
    ready = 0
    for pack in list((payload or {}).values()):
        if not isinstance(pack, dict):
            continue
        packs += 1
        pack_hits = [hit for hit in list(pack.get("hits") or []) if isinstance(hit, dict)]
        hits += len(pack_hits)
        mode = str(pack.get("payload_mode") or "").strip().lower()
        if mode == "pending" or bool(pack.get("enrichment_pending")):
            pending += 1
        elif mode == "fast":
            fast += 1
        elif pack_hits:
            ready += 1
    return f"packs={packs};hits={hits};pending={pending};fast={fast};ready={ready}"


def _set_refs_timing_headers(
    response: Response | None,
    *,
    timings: list[tuple[str, float]],
    total_ms: float,
    mode: str,
    payload: dict | None,
) -> None:
    if response is None:
        return
    seen: dict[str, int] = {}
    parts: list[str] = []
    for raw_name, raw_duration in list(timings or []):
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw_name or "").strip())[:36] or "phase"
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}_{seen[name]}"
        try:
            duration = float(raw_duration)
        except Exception:
            duration = 0.0
        parts.append(f"{name};dur={max(0.0, duration):.1f}")
    parts.append(f"total;dur={max(0.0, float(total_ms or 0.0)):.1f}")
    response.headers["Server-Timing"] = ", ".join(parts)
    response.headers["X-KB-Refs-Mode"] = str(mode or "").strip().lower() or "unknown"
    response.headers["X-KB-Refs-Counts"] = _refs_payload_counts_for_header(payload)


def _refs_conversation_read_timeout_s() -> float:
    try:
        raw = float(str(os.environ.get("KB_REFS_CONVERSATION_READ_TIMEOUT_S", "0.35") or "0.35"))
    except Exception:
        raw = 0.35
    return max(0.05, min(2.0, raw))


def _refs_ready_budget_s() -> float:
    try:
        raw = float(str(os.environ.get("KB_REFS_READY_BUDGET_S", "1.8") or "1.8"))
    except Exception:
        raw = 1.8
    return max(0.25, min(8.0, raw))


def _refs_pending_stale_after_s() -> float:
    try:
        raw = float(str(os.environ.get("KB_REFS_PENDING_STALE_AFTER_S", "20") or "20"))
    except Exception:
        raw = 20.0
    return max(5.0, min(120.0, raw))


def _refs_pack_is_stale_pending(pack: dict) -> bool:
    if not isinstance(pack, dict):
        return False
    has_pending = False
    for hit in list(pack.get("hits") or []):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        if str((meta or {}).get("ref_pack_state") or "").strip().lower() == "pending":
            has_pending = True
            break
    if not has_pending:
        return False
    try:
        updated_at = float(pack.get("updated_at") or 0.0)
    except Exception:
        updated_at = 0.0
    if updated_at <= 0:
        return False
    return (time.time() - updated_at) >= _refs_pending_stale_after_s()


def _refs_payload_has_pending(refs: dict, *, include_stale: bool = True) -> bool:
    for pack in list((refs or {}).values()):
        if not isinstance(pack, dict):
            continue
        if _refs_pack_has_pending(pack, include_stale=include_stale):
            return True
    return False


def _refs_pack_has_pending(pack: dict, *, include_stale: bool = True) -> bool:
    if not isinstance(pack, dict):
        return False
    has_pending = False
    for hit in list(pack.get("hits") or []):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        if str((meta or {}).get("ref_pack_state") or "").strip().lower() == "pending":
            has_pending = True
            break
    if (not has_pending) or include_stale:
        return has_pending
    return not _refs_pack_is_stale_pending(pack)


def _stored_rendered_pack_payload_lost_current_hits(*, payload: dict, pack: dict) -> bool:
    if not isinstance(payload, dict) or not isinstance(pack, dict):
        return False
    raw_hits = [hit for hit in list(pack.get("hits") or []) if isinstance(hit, dict)]
    if not raw_hits:
        return False
    payload_hits = [hit for hit in list(payload.get("hits") or []) if isinstance(hit, dict)]
    if payload_hits:
        raw_sources: list[str] = []
        for hit in raw_hits[:4]:
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            source_path = str((meta or {}).get("source_path") or "").strip()
            if source_path:
                raw_sources.append(source_path)
        payload_sources: list[str] = []
        for hit in payload_hits[:4]:
            ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            source_path = str((ui_meta or {}).get("source_path") or (meta or {}).get("source_path") or "").strip()
            if source_path:
                payload_sources.append(source_path)
        # A persisted rendered payload is stale if it no longer contains the
        # answer-leading source stored in message_refs.  This can happen when
        # the fast reference route was cached before final answer provenance
        # rewrote the refs pack.
        if raw_sources and payload_sources and raw_sources[0] not in payload_sources:
            return True
        return False
    display_state = str(payload.get("display_state") or "").strip().lower()
    suppression_reason = str(payload.get("suppression_reason") or "").strip().lower()
    if display_state == "hidden_by_guide" or suppression_reason == "guide_self_source_only":
        return False
    pipeline_debug = payload.get("pipeline_debug") if isinstance(payload.get("pipeline_debug"), dict) else {}
    try:
        debug_raw_hit_count = int((pipeline_debug or {}).get("raw_hit_count") or 0)
    except Exception:
        debug_raw_hit_count = 0
    doc_list_authoritative = bool((pipeline_debug or {}).get("doc_list_authoritative"))
    if doc_list_authoritative and debug_raw_hit_count <= 0:
        return True
    return bool(display_state == "empty" and suppression_reason == "no_candidate_hits" and debug_raw_hit_count <= 0)


def _get_stored_rendered_pack_payload(
    *,
    user_msg_id: int | str,
    pack: dict,
    guide_mode: bool,
    guide_source_path: str,
    guide_source_name: str,
) -> dict | None:
    if not isinstance(pack, dict):
        return None
    payload = pack.get("rendered_payload")
    if not isinstance(payload, dict) or (not payload):
        return None
    stored_sig = str(pack.get("rendered_payload_sig") or "").strip()
    expected_sig = _refs_pack_render_signature(
        user_msg_id=user_msg_id,
        pack=pack,
        guide_mode=guide_mode,
        guide_source_path=guide_source_path,
        guide_source_name=guide_source_name,
    )
    if (not stored_sig) or (stored_sig != expected_sig):
        return None
    if _stored_rendered_pack_payload_lost_current_hits(payload=payload, pack=pack):
        return None
    if (
        _refs_background_llm_polish_enabled()
        and (not prompt_explicitly_requests_multi_paper_list(str((pack or {}).get("prompt") or "").strip()))
        and (not _payload_refs_card_copy_has_llm_result(payload))
    ):
        return None
    return dict(payload)


def _extract_doc_list_contract_from_message_meta(meta: dict | None) -> list[dict]:
    if not isinstance(meta, dict):
        return []
    contracts = meta.get("paper_guide_contracts") if isinstance(meta.get("paper_guide_contracts"), dict) else {}
    return [dict(item) for item in list((contracts or {}).get("doc_list") or []) if isinstance(item, dict)]


def _load_authoritative_doc_list_contracts(
    *,
    store,
    conv_id: str,
    user_msg_ids: set[int],
) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    if not user_msg_ids:
        return out
    get_messages = getattr(store, "get_messages", None)
    if not callable(get_messages):
        return out
    try:
        messages = list(get_messages(str(conv_id or "").strip()) or [])
    except sqlite3.OperationalError:
        return out
    except Exception:
        return out
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        try:
            msg_id = int(msg.get("id") or 0)
        except Exception:
            msg_id = 0
        if msg_id not in user_msg_ids:
            continue
        if str(msg.get("role") or "").strip().lower() != "user":
            continue
        for nxt in messages[idx + 1 :]:
            if not isinstance(nxt, dict):
                continue
            role = str(nxt.get("role") or "").strip().lower()
            if role == "user":
                break
            if role != "assistant":
                continue
            content = str(nxt.get("content") or "")
            if content.startswith("__KB_LIVE_TASK__:"):
                continue
            meta = nxt.get("meta") if isinstance(nxt.get("meta"), dict) else {}
            contracts = meta.get("paper_guide_contracts") if isinstance(meta.get("paper_guide_contracts"), dict) else {}
            if "doc_list" in contracts:
                out[msg_id] = _extract_doc_list_contract_from_message_meta(meta)
            break
    return out


def _load_pending_doc_list_contracts(
    *,
    store,
    conv_id: str,
    pending_user_msg_ids: set[int],
) -> dict[int, list[dict]]:
    return _load_authoritative_doc_list_contracts(
        store=store,
        conv_id=conv_id,
        user_msg_ids=pending_user_msg_ids,
    )


def _mark_doc_list_pending_pack(*, payload_pack: dict, pending_count: int) -> dict:
    pack2 = dict(payload_pack or {})
    hits_out: list[dict] = []
    for raw_hit in list(pack2.get("hits") or []):
        if not isinstance(raw_hit, dict):
            continue
        hit = dict(raw_hit)
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        meta2 = dict(meta or {})
        meta2["ref_pack_state"] = "pending"
        hit["meta"] = meta2
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        if isinstance(ui_meta, dict):
            ui_meta2 = dict(ui_meta)
            ui_meta2["score_pending"] = True
            ui_meta2["score"] = None
            ui_meta2["score_tier"] = ""
            hit["ui_meta"] = ui_meta2
        hits_out.append(hit)
    pack2["hits"] = hits_out
    pack2["pending"] = True
    pack2["pending_hit_count"] = int(max(0, int(pending_count or 0)))
    pack2["payload_mode"] = "pending"
    pack2["enrichment_pending"] = True
    return _attach_pack_display_contract(pack2)


def _doc_list_source_paths(doc_list: list[dict] | None) -> list[str]:
    out: list[str] = []
    for item in list(doc_list or []):
        if not isinstance(item, dict):
            continue
        source_path = str(item.get("source_path") or "").strip()
        if source_path:
            out.append(source_path)
    return out


def _payload_source_paths(payload_pack: dict | None) -> list[str]:
    out: list[str] = []
    if not isinstance(payload_pack, dict):
        return out
    for hit in list(payload_pack.get("hits") or []):
        if not isinstance(hit, dict):
            continue
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        source_path = str(
            (ui_meta or {}).get("source_path")
            or (meta or {}).get("source_path")
            or ""
        ).strip()
        if source_path:
            out.append(source_path)
    return out


def _payload_refs_card_copy_has_llm_result(payload_pack: dict | None) -> bool:
    return refs_pack_has_full_llm_copy(payload_pack)


def _payload_is_authoritative_doc_list_pack(payload_pack: dict | None, authoritative_doc_list: list[dict] | None) -> bool:
    if not isinstance(payload_pack, dict):
        return False
    pipeline_debug = payload_pack.get("pipeline_debug") if isinstance(payload_pack.get("pipeline_debug"), dict) else {}
    if not bool((pipeline_debug or {}).get("doc_list_authoritative")):
        return False
    if not _payload_refs_card_copy_has_llm_result(payload_pack):
        return False
    expected_paths = _doc_list_source_paths(authoritative_doc_list)
    actual_paths = _payload_source_paths(payload_pack)
    if not expected_paths:
        return True
    return bool(actual_paths) and actual_paths == expected_paths


def _rebuild_authoritative_doc_list_from_pack(*, prompt: str, pack: dict, guide_mode: bool) -> list[dict]:
    prompt_text = str(prompt or "").strip()
    if guide_mode or (not prompt_explicitly_requests_multi_paper_list(prompt_text)):
        return []
    rows = [dict(hit) for hit in list((pack or {}).get("hits") or []) if isinstance(hit, dict)]
    if not rows:
        return []
    try:
        rebuilt = _references_build_multi_paper_doc_list_contract(
            prompt=prompt_text,
            seed_docs=list(rows),
            answer_hits=list(rows),
            evidence_cards=[],
        )
    except Exception:
        rebuilt = []
    return [dict(item) for item in list(rebuilt or []) if isinstance(item, dict)]


def _normalize_authoritative_doc_list_contracts_for_refs(
    *,
    refs: dict,
    doc_lists: dict[int, list[dict]],
    guide_mode: bool,
) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for raw_user_msg_id, raw_rows in dict(doc_lists or {}).items():
        try:
            user_msg_id = int(raw_user_msg_id)
        except Exception:
            continue
        rows = [dict(item) for item in list(raw_rows or []) if isinstance(item, dict)]
        if rows:
            out[user_msg_id] = rows
            continue
        if guide_mode:
            # In guide mode an empty cross-paper contract is meaningful: it hides self-only refs.
            out[user_msg_id] = []
            continue
        pack = None
        for key in (user_msg_id, str(user_msg_id)):
            candidate = (refs or {}).get(key)
            if isinstance(candidate, dict):
                pack = candidate
                break
        if not isinstance(pack, dict):
            continue
        rebuilt = _rebuild_authoritative_doc_list_from_pack(
            prompt=str(pack.get("prompt") or "").strip(),
            pack=pack,
            guide_mode=False,
        )
        if rebuilt:
            out[user_msg_id] = rebuilt
    return out


def _filter_pending_multi_paper_hits_for_display(prompt: str, hits: list[dict] | None) -> list[dict]:
    rows = [dict(hit) for hit in list(hits or []) if isinstance(hit, dict)]
    if (not rows) or (not prompt_explicitly_requests_multi_paper_list(prompt)):
        return rows
    try:
        doc_list_seed = _references_build_multi_paper_doc_list_contract(
            prompt=prompt,
            seed_docs=list(rows),
            answer_hits=list(rows),
            evidence_cards=[],
        )
    except Exception:
        return rows
    source_order = _doc_list_source_paths(doc_list_seed)
    if not source_order:
        return rows
    rows_by_source: dict[str, dict] = {}
    for row in rows:
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        source_path = str((meta or {}).get("source_path") or "").strip()
        if source_path and source_path not in rows_by_source:
            rows_by_source[source_path] = row
    filtered: list[dict] = []
    seen: set[str] = set()
    for source_path in source_order:
        row = rows_by_source.get(source_path)
        if not isinstance(row, dict) or source_path in seen:
            continue
        filtered.append(row)
        seen.add(source_path)
    target_count = min(3, len(rows))
    if len(filtered) < target_count:
        for row in rows:
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            source_path = str((meta or {}).get("source_path") or "").strip()
            if (not source_path) or source_path in seen:
                continue
            filtered.append(row)
            seen.add(source_path)
            if len(filtered) >= target_count:
                break
    return filtered or rows


def _render_authoritative_doc_list_pack(
    *,
    user_msg_id: int,
    pack: dict,
    doc_list: list[dict],
    guide_mode: bool,
    guide_source_path: str,
    guide_source_name: str,
    pending: bool,
) -> dict:
    # Keep pending/full on the same authoritative paper set and only defer strict locate until full render.
    return build_doc_list_refs_payload(
        user_msg_id=int(user_msg_id),
        pack=pack,
        doc_list=doc_list,
        allow_expensive_llm=not pending,
        allow_exact_locate=not pending,
        apply_copy_polish=True,
        guide_mode=bool(guide_mode),
        guide_source_path=str(guide_source_path or "").strip(),
        guide_source_name=str(guide_source_name or "").strip(),
    )


def _build_pending_conversation_refs_payload(
    refs: dict,
    *,
    doc_list_by_user: dict[int, list[dict]] | None = None,
    guide_mode: bool = False,
    guide_source_path: str = "",
    guide_source_name: str = "",
) -> dict[int, dict]:
    out: dict[int, dict] = {}
    authoritative_map = {
        int(key): [dict(item) for item in list(value or []) if isinstance(item, dict)]
        for key, value in dict(doc_list_by_user or {}).items()
        if str(key).isdigit() or isinstance(key, int)
    }
    for user_msg_id, pack in (refs or {}).items():
        if not isinstance(pack, dict):
            continue
        prompt = str(pack.get("prompt") or "").strip()
        focus_terms = [str(term or "").strip() for term in _refs_prompt_focus_terms(prompt) if str(term or "").strip()]
        focus_action = prompt_reference_focus_action(prompt)
        raw_hits = [hit for hit in list(pack.get("hits") or []) if isinstance(hit, dict)]
        if prompt_explicitly_requests_multi_paper_list(prompt):
            filtered_hits = _filter_pending_multi_paper_hits_for_display(prompt, raw_hits)
            if not filtered_hits:
                filtered_hits = _filter_pending_refs_hits_by_prompt_focus(prompt, raw_hits)
            filtered_hits = filtered_hits[:3]
        else:
            filtered_hits = _filter_pending_refs_hits_by_prompt_focus(prompt, raw_hits)[:2]
        pending_count = 0
        hits_out: list[dict] = []
        for hit in raw_hits:
            if not isinstance(hit, dict):
                continue
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            if str((meta or {}).get("ref_pack_state") or "").strip().lower() == "pending":
                pending_count += 1
        authoritative_doc_list_present = int(user_msg_id) in authoritative_map
        authoritative_doc_list = [dict(item) for item in list(authoritative_map.get(int(user_msg_id), []) or []) if isinstance(item, dict)]
        if authoritative_doc_list_present:
            authoritative_pack = _render_authoritative_doc_list_pack(
                user_msg_id=int(user_msg_id),
                pack=pack,
                doc_list=authoritative_doc_list,
                guide_mode=bool(guide_mode),
                guide_source_path=str(guide_source_path or "").strip(),
                guide_source_name=str(guide_source_name or "").strip(),
                pending=True,
            )
            out[int(user_msg_id)] = _mark_doc_list_pending_pack(
                payload_pack=authoritative_pack,
                pending_count=pending_count,
            )
            continue
        for hit in filtered_hits:
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            source_path = str((meta or {}).get("source_path") or "").strip()
            heading_path = str((meta or {}).get("ref_best_heading_path") or (meta or {}).get("top_heading") or "").strip()
            display_name = _source_name_from_md_path(source_path) if source_path else "Reference"
            snippet_seed = ""
            for key in ("ref_show_snippets", "ref_snippets", "ref_overview_snippets"):
                raw = (meta or {}).get(key)
                if isinstance(raw, list):
                    snippet_seed = next((str(item or "").strip() for item in raw if str(item or "").strip()), "")
                if snippet_seed:
                    break
            if not snippet_seed:
                snippet_seed = str(hit.get("text") or "").strip()
            summary_line = _compact_reader_open_text(snippet_seed)
            focus_text = " and ".join(focus_terms[:2]) if len(focus_terms) >= 2 else (focus_terms[0] if focus_terms else "the requested concept")
            if focus_action == "compare":
                why_line = f"This pending match most directly compares {focus_text} in {heading_path or 'the matched section'}."
            elif focus_action == "define":
                why_line = f"This pending match most directly defines {focus_text} in {heading_path or 'the matched section'}."
            else:
                why_line = f"This pending match most directly discusses {focus_text} in {heading_path or 'the matched section'}."
            reader_open = {
                "sourcePath": source_path,
                "sourceName": display_name,
                "headingPath": heading_path or None,
                "snippet": summary_line or None,
                "highlightSnippet": summary_line or None,
                "strictLocate": False,
            }
            primary_evidence = {
                "source_path": source_path,
                "source_name": display_name,
                "heading_path": heading_path or None,
                "snippet": summary_line or None,
                "highlight_snippet": summary_line or None,
                "selection_reason": "pending_section_seed",
                "strict_locate": False,
            }
            hit2 = dict(hit)
            hit2["ui_meta"] = {
                "display_name": display_name,
                "heading_path": heading_path,
                "summary_line": summary_line,
                "summary_kind": "guide",
                "summary_label": "Guide",
                "summary_title": "What This Matched Section Covers",
                "summary_generation": "pending_section_seed",
                "summary_basis": "Provisional summary from pending matched section evidence",
                "why_line": why_line,
                "why_generation": "pending_focus_seed",
                "why_basis": "Provisional relevance note from pending matched section and focus-term alignment",
                "score": None,
                "score_pending": True,
                "score_tier": "",
                "primary_evidence": {key: value for key, value in primary_evidence.items() if value not in (None, "", [], {})},
                "source_path": source_path,
                "reader_open": {key: value for key, value in reader_open.items() if value not in (None, "", [], {})},
            }
            if isinstance(hit2["ui_meta"].get("reader_open"), dict) and hit2["ui_meta"].get("primary_evidence"):
                hit2["ui_meta"]["reader_open"] = dict(hit2["ui_meta"]["reader_open"])
                hit2["ui_meta"]["reader_open"]["primaryEvidence"] = dict(hit2["ui_meta"]["primary_evidence"])
            hits_out.append(hit2)
        pack2 = dict(pack)
        pack2["hits"] = hits_out
        pack2["pending"] = True
        pack2["pending_hit_count"] = int(pending_count)
        pack2["payload_mode"] = "pending"
        pack2["enrichment_pending"] = True
        out[int(user_msg_id)] = _attach_pack_display_contract(pack2)
    return out


def _annotate_refs_payload_refresh_state(payload: dict, *, mode: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    mode_norm = str(mode or "").strip().lower() or "full"
    needs_enrichment = mode_norm in {"fast", "pending"}
    for user_msg_id, pack in (payload or {}).items():
        if not isinstance(pack, dict):
            continue
        pack2 = _attach_pack_display_contract(pack)
        pack2["payload_mode"] = mode_norm
        if needs_enrichment:
            pack2["enrichment_pending"] = True
        else:
            pack2.pop("enrichment_pending", None)
        out[int(user_msg_id)] = _attach_pack_display_contract(pack2)
    return out


def _attach_pack_render_state(payload_pack: dict, *, source_pack: dict | None, default_status: str = "") -> dict:
    out = _attach_pack_display_contract(payload_pack)
    src = source_pack if isinstance(source_pack, dict) else {}
    render_status = str((src or {}).get("render_status") or default_status or "").strip().lower()
    render_error = str((src or {}).get("render_error") or "").strip()
    render_error_detail = str((src or {}).get("render_error_detail") or "").strip()
    try:
        render_attempts = int((src or {}).get("render_attempts") or 0)
    except Exception:
        render_attempts = 0
    try:
        render_built_at = float((src or {}).get("render_built_at") or 0.0)
    except Exception:
        render_built_at = 0.0
    render_evidence_sig = str((src or {}).get("render_evidence_sig") or "").strip()
    render_locale = str((src or {}).get("render_locale") or "").strip()
    if render_status:
        out["render_status"] = render_status
    if render_error:
        out["render_error"] = render_error
    if render_error_detail:
        out["render_error_detail"] = render_error_detail
    if render_attempts > 0:
        out["render_attempts"] = render_attempts
    if render_built_at > 0:
        out["render_built_at"] = render_built_at
    if render_evidence_sig:
        out["render_evidence_sig"] = render_evidence_sig
    if render_locale:
        out["render_locale"] = render_locale
    if str(out.get("render_status") or "").strip().lower() == "failed":
        out.pop("enrichment_pending", None)
    return _attach_pack_display_contract(out)


def _build_fast_ready_conversation_refs_payload(
    *,
    refs: dict,
    guide_mode: bool,
    guide_source_path: str,
    guide_source_name: str,
    deadline_at: float | None = None,
) -> dict[int, dict]:
    return _annotate_refs_payload_refresh_state(
        enrich_refs_payload(
            refs,
            pdf_root=_pdf_dir(),
            md_root=_md_dir(),
            lib_store=_lib_store(),
            guide_mode=guide_mode,
            guide_source_path=guide_source_path,
            guide_source_name=guide_source_name,
            render_variant="fast",
            allow_expensive_llm_for_ready=False,
            allow_exact_locate=False,
            deadline_at=deadline_at,
        ),
        mode="fast",
    )


def _persist_rendered_refs_payloads(
    *,
    refs: dict,
    payload: dict,
    guide_mode: bool,
    guide_source_path: str,
    guide_source_name: str,
) -> None:
    if not isinstance(refs, dict) or not isinstance(payload, dict):
        return
    try:
        store = get_chat_store()
    except Exception:
        return
    for user_msg_id, pack in refs.items():
        if not isinstance(pack, dict):
            continue
        payload_pack = payload.get(user_msg_id)
        if not isinstance(payload_pack, dict):
            payload_pack = payload.get(str(user_msg_id))
        if not isinstance(payload_pack, dict) or (not payload_pack):
            continue
        sig = _refs_pack_render_signature(
            user_msg_id=user_msg_id,
            pack=pack,
            guide_mode=guide_mode,
            guide_source_path=guide_source_path,
            guide_source_name=guide_source_name,
        )
        try:
            store.set_message_refs_rendered_payload(
                user_msg_id=int(user_msg_id),
                rendered_payload=payload_pack,
                rendered_payload_sig=sig,
                render_status="full",
                render_error="",
                render_error_detail="",
                render_built_at=time.time(),
                render_evidence_sig=str(sig or "").strip(),
            )
        except Exception:
            continue


def _refs_background_llm_polish_enabled() -> bool:
    raw = str(os.environ.get("KB_REFS_BACKGROUND_LLM_POLISH", "") or "").strip().lower()
    if raw:
        return raw in {"1", "true", "on", "yes"}
    return bool(_refs_card_polish_llm_enabled())


def _warm_conversation_refs_payload_async(
    *,
    conv_id: str,
    signature: str,
    refs: dict,
    guide_mode: bool,
    guide_source_path: str,
    guide_source_name: str,
) -> None:
    conv_key = str(conv_id or "").strip()
    sig_key = str(signature or "").strip()
    if (not conv_key) or (not sig_key):
        return
    warm_key = f"{conv_key}:{sig_key}"
    with _REFS_CONVERSATION_WARMING_LOCK:
        if warm_key in _REFS_CONVERSATION_WARMING:
            return
        _REFS_CONVERSATION_WARMING.add(warm_key)

    def _run() -> None:
        try:
            payload = enrich_refs_payload(
                refs,
                pdf_root=_pdf_dir(),
                md_root=_md_dir(),
                lib_store=_lib_store(),
                guide_mode=guide_mode,
                guide_source_path=guide_source_path,
                guide_source_name=guide_source_name,
                render_variant="bounded_full",
                allow_expensive_llm_for_ready=_refs_background_llm_polish_enabled(),
                allow_exact_locate=True,
            )
            if not isinstance(payload, dict):
                return
            current = _REFS_CONVERSATION_CACHE.get(conv_key)
            if isinstance(current, dict):
                current_sig = str(current.get("signature") or "").strip()
                if current_sig and current_sig != sig_key:
                    return
            _persist_rendered_refs_payloads(
                refs=refs,
                payload=payload,
                guide_mode=guide_mode,
                guide_source_path=guide_source_path,
                guide_source_name=guide_source_name,
            )
            _store_cached_conversation_refs_payload(
                conv_id=conv_key,
                signature=sig_key,
                payload=payload,
                mode="full",
            )
        except Exception as exc:
            try:
                store = get_chat_store()
            except Exception:
                store = None
            if store is not None:
                for user_msg_id, pack in (refs or {}).items():
                    if not isinstance(pack, dict):
                        continue
                    try:
                        store.set_message_refs_render_state(
                            user_msg_id=int(user_msg_id),
                            render_status="failed",
                            render_error="route_warm_failed",
                            render_error_detail=f"{type(exc).__name__}: {str(exc or '').strip()}"[:500],
                        )
                    except Exception:
                        continue
        finally:
            with _REFS_CONVERSATION_WARMING_LOCK:
                _REFS_CONVERSATION_WARMING.discard(warm_key)

    try:
        threading.Thread(target=_run, daemon=True, name="kb_refs_conv_warm").start()
    except Exception:
        with _REFS_CONVERSATION_WARMING_LOCK:
            _REFS_CONVERSATION_WARMING.discard(warm_key)


@router.post("/sync")
def start_sync(workers: int | None = None, crossref_budget_s: float | None = None):
    s = get_settings()
    try:
        workers_default = int(os.environ.get("KB_REFSYNC_WORKERS", "6") or 6)
    except Exception:
        workers_default = 6
    if workers is None:
        workers = workers_default
    workers_final = int(max(1, min(16, int(workers))))

    try:
        budget_default = float(os.environ.get("KB_CROSSREF_BUDGET_S", "45") or 45.0)
    except Exception:
        budget_default = 45.0
    if crossref_budget_s is None:
        crossref_budget_s = budget_default
    budget_final = float(max(5.0, min(180.0, float(crossref_budget_s))))

    result = start_reference_sync(
        src_root=_md_dir(),
        db_dir=s.db_dir,
        pdf_root=_pdf_dir(),
        library_db_path=s.library_db_path,
        crossref_time_budget_s=budget_final,
        doi_prefetch_workers=workers_final,
    )
    return result


@router.get("/sync/status")
async def sync_status():
    def poll():
        snap = refsync_snapshot()
        return {
            **snap,
            "done": snap.get("status") in ("done", "error", "idle"),
        }
    return sse_response(sse_generator(poll, interval=0.5))


def _compute_diagnose_suggestion(suppression_reason: str) -> str:
    suggestions = {
        "no_candidate_hits": (
            "No documents matched the query. Try rephrasing with different keywords, "
            "or check that relevant documents are ingested in the knowledge base."
        ),
        "score_gate_removed_all": (
            "All BM25 scores were below the relevance threshold. "
            "Try a more specific query."
        ),
        "focus_filter_removed_all": (
            "All hits were filtered out because they did not match the prompt's "
            "focus terms. Try broadening the question or removing specific constraints."
        ),
        "llm_filter_removed_all": (
            "The LLM relevance filter judged all hits as irrelevant. "
            "This may indicate a vocabulary mismatch between the query and documents."
        ),
        "guide_self_source_only": (
            "Guide mode hides the bound source paper. "
            "Disable guide mode or ask about other papers."
        ),
        "render_failed": (
            "The reference card rendering pipeline failed unexpectedly. "
            "Check server logs for error details."
        ),
        "pending_enrichment": (
            "Results are still being computed. "
            "Try again in a few seconds."
        ),
        "no_renderable_hits": (
            "Hits entered the pipeline but none could be rendered as reference cards. "
            "Check the pipeline stage counts for details."
        ),
    }
    return suggestions.get(suppression_reason, "No specific suggestion available for this state.")


def _build_diagnostic_report(*, store, conv_id: str, refs: dict) -> dict:
    """Build a diagnostic report for all refs packs in a conversation."""
    packs: dict[int, dict] = {}
    total_packs = 0
    empty_packs = 0
    suppressed_packs = 0

    for key, pack in (refs or {}).items():
        try:
            user_msg_id = int(key)
        except (ValueError, TypeError):
            continue
        total_packs += 1
        if not isinstance(pack, dict):
            packs[user_msg_id] = {"parse_error": "pack is not a dict"}
            continue

        try:
            contract = _attach_pack_display_contract(pack)
        except Exception as exc:
            packs[user_msg_id] = {"parse_error": str(exc)[:200]}
            continue

        display_state = str(contract.get("display_state") or "unknown")
        suppression_reason = str(contract.get("suppression_reason") or "").strip()
        pipeline_debug = contract.get("pipeline_debug") if isinstance(contract.get("pipeline_debug"), dict) else {}
        retrieval_diag = pipeline_debug.get("retrieval_diag") if isinstance(pipeline_debug.get("retrieval_diag"), dict) else {}
        prompt_raw = str(pack.get("prompt") or pack.get("question") or "").strip()
        used_query = str(pack.get("used_query") or pipeline_debug.get("used_query") or retrieval_diag.get("used_query") or "").strip()
        used_translation = bool(pack.get("used_translation") or retrieval_diag.get("query_translated") or False)

        # Compute top BM25 scores from hits.
        top_scores: list[dict] = []
        hits = [h for h in list(contract.get("hits") or []) if isinstance(h, dict)]
        scored = []
        for h in hits:
            try:
                bm25_score = float(h.get("score") or 0.0)
            except (ValueError, TypeError):
                bm25_score = 0.0
            meta = h.get("meta") if isinstance(h.get("meta"), dict) else {}
            source_path = str(meta.get("source_path") or "").strip()
            source_name = str(meta.get("source_name") or "").strip()
            if not source_name:
                source_name = str(Path(source_path).stem if source_path else "unknown")
            heading = str(meta.get("heading_path") or "").strip()[:120]
            scored.append({
                "score": round(bm25_score, 2),
                "doc_name": source_name[:80],
                "source_path": source_path,
                "heading_path": heading,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        top_scores = scored[:5]

        if display_state == "empty":
            empty_packs += 1
        elif display_state in ("suppressed", "hidden_by_guide"):
            suppressed_packs += 1

        suggestion = _compute_diagnose_suggestion(suppression_reason) if suppression_reason else ""

        packs[user_msg_id] = {
            "prompt": prompt_raw[:500],
            "display_state": display_state,
            "suppression_reason": suppression_reason,
            "pipeline_debug": pipeline_debug,
            "retrieval_diag": retrieval_diag,
            "used_query": used_query,
            "used_translation": used_translation,
            "top_scores": top_scores,
            "has_pending": bool(contract.get("pending")),
            "suggestion": suggestion,
        }

    return {
        "conv_id": conv_id,
        "total_packs": total_packs,
        "empty_packs": empty_packs,
        "suppressed_packs": suppressed_packs,
        "packs": packs,
    }


def get_conversation_refs(conv_id: str, response: Response | None = None):
    route_started_at = time.perf_counter()
    route_deadline_at = route_started_at + _refs_ready_budget_s()
    timings: list[tuple[str, float]] = []

    def _record(name: str, started_at: float) -> None:
        timings.append((name, _refs_perf_ms(started_at)))

    def _finish(payload: dict | None, mode: str) -> dict:
        payload_out = payload if isinstance(payload, dict) else {}
        _set_refs_timing_headers(
            response,
            timings=timings,
            total_ms=_refs_perf_ms(route_started_at),
            mode=mode,
            payload=payload_out,
        )
        return payload_out

    store = get_chat_store()
    read_timeout_s = _refs_conversation_read_timeout_s()
    phase_started_at = time.perf_counter()
    try:
        conversation = store.get_conversation(conv_id, timeout_s=read_timeout_s) or {}
    except TypeError:
        conversation = store.get_conversation(conv_id) or {}
    except sqlite3.OperationalError:
        _record("conversation", phase_started_at)
        cached_any = _get_any_cached_conversation_refs_payload(conv_id=conv_id)
        return _finish(cached_any if isinstance(cached_any, dict) else {}, "cache_fallback")
    _record("conversation", phase_started_at)
    guide_mode = str(conversation.get("mode") or "").strip().lower() == "paper_guide"
    guide_source_path = str(conversation.get("bound_source_path") or "").strip()
    guide_source_name = str(conversation.get("bound_source_name") or "").strip()
    phase_started_at = time.perf_counter()
    try:
        refs = store.list_message_refs(conv_id, timeout_s=read_timeout_s)
    except TypeError:
        refs = store.list_message_refs(conv_id)
    except sqlite3.OperationalError:
        _record("list_refs", phase_started_at)
        cached_any = _get_any_cached_conversation_refs_payload(conv_id=conv_id)
        return _finish(cached_any if isinstance(cached_any, dict) else {}, "cache_fallback")
    _record("list_refs", phase_started_at)
    refs_norm = refs if isinstance(refs, dict) else {}
    all_user_msg_ids: set[int] = set()
    for key in refs_norm.keys():
        try:
            all_user_msg_ids.add(int(key))
        except Exception:
            continue
    phase_started_at = time.perf_counter()
    authoritative_doc_lists = _load_authoritative_doc_list_contracts(
        store=store,
        conv_id=conv_id,
        user_msg_ids=all_user_msg_ids,
    )
    authoritative_doc_lists = _normalize_authoritative_doc_list_contracts_for_refs(
        refs=refs_norm,
        doc_lists=authoritative_doc_lists,
        guide_mode=bool(guide_mode),
    )
    _record("doc_list_contracts", phase_started_at)
    phase_started_at = time.perf_counter()
    signature = _refs_conversation_cache_signature(
        refs=refs_norm,
        guide_mode=guide_mode,
        guide_source_path=guide_source_path,
        guide_source_name=guide_source_name,
        authoritative_doc_list_by_user=authoritative_doc_lists,
    )
    _record("signature", phase_started_at)
    has_pending = _refs_payload_has_pending(refs_norm, include_stale=False)
    has_authoritative_doc_list = bool(authoritative_doc_lists)
    phase_started_at = time.perf_counter()
    cached_rec = _get_cached_conversation_refs_record(conv_id=conv_id, signature=signature)
    _record("cache_lookup", phase_started_at)
    cached_payload = cached_rec.get("payload") if isinstance(cached_rec, dict) else None
    cached_mode = str(cached_rec.get("mode") or "").strip().lower() if isinstance(cached_rec, dict) else ""
    if isinstance(cached_payload, dict) and cached_mode == "full" and (not has_authoritative_doc_list):
        return _finish(cached_payload, "cache_full")

    stored_full_payload: dict[int, dict] = {}
    pending_refs: dict[int, dict] = {}
    failed_ready_refs: dict[int, dict] = {}
    ready_missing_refs: dict[int, dict] = {}
    authoritative_full_payloads: dict[int, dict] = {}
    authoritative_full_refs: dict[int, dict] = {}
    phase_started_at = time.perf_counter()
    for user_msg_id, pack in refs_norm.items():
        if not isinstance(pack, dict):
            continue
        prompt_text = str(pack.get("prompt") or "").strip()
        authoritative_doc_list_present = int(user_msg_id) in authoritative_doc_lists
        authoritative_doc_list = [
            dict(item)
            for item in list(authoritative_doc_lists.get(int(user_msg_id), []) or [])
            if isinstance(item, dict)
        ]
        if authoritative_doc_list_present and (not authoritative_doc_list):
            rebuilt_doc_list = _rebuild_authoritative_doc_list_from_pack(
                prompt=prompt_text,
                pack=pack,
                guide_mode=bool(guide_mode),
            )
            if rebuilt_doc_list:
                authoritative_doc_list = rebuilt_doc_list
        pack_full = _get_stored_rendered_pack_payload(
            user_msg_id=user_msg_id,
            pack=pack,
            guide_mode=guide_mode,
            guide_source_path=guide_source_path,
            guide_source_name=guide_source_name,
        )
        if authoritative_doc_list_present and _refs_pack_has_pending(pack, include_stale=False):
            pending_refs[int(user_msg_id)] = pack
            continue
        if authoritative_doc_list_present:
            if isinstance(pack_full, dict) and _payload_is_authoritative_doc_list_pack(pack_full, authoritative_doc_list):
                stored_full_payload[int(user_msg_id)] = _attach_pack_render_state(
                    pack_full,
                    source_pack=pack,
                    default_status="full",
                )
                continue
            authoritative_payload = _render_authoritative_doc_list_pack(
                user_msg_id=int(user_msg_id),
                pack=pack,
                doc_list=authoritative_doc_list,
                guide_mode=bool(guide_mode),
                guide_source_path=str(guide_source_path or "").strip(),
                guide_source_name=str(guide_source_name or "").strip(),
                pending=False,
            )
            if isinstance(authoritative_payload, dict) and authoritative_payload:
                authoritative_full_payloads[int(user_msg_id)] = authoritative_payload
                authoritative_full_refs[int(user_msg_id)] = pack
                stored_full_payload[int(user_msg_id)] = _attach_pack_render_state(
                    authoritative_payload,
                    source_pack=pack,
                    default_status="full",
                )
                continue
        if isinstance(pack_full, dict):
            stored_full_payload[int(user_msg_id)] = _attach_pack_render_state(
                pack_full,
                source_pack=pack,
                default_status="full",
            )
            continue
        if _refs_pack_has_pending(pack, include_stale=False):
            pending_refs[int(user_msg_id)] = pack
        elif str((pack or {}).get("render_status") or "").strip().lower() == "failed":
            failed_ready_refs[int(user_msg_id)] = pack
        else:
            ready_missing_refs[int(user_msg_id)] = pack
    _record("render_state_scan", phase_started_at)

    if authoritative_full_payloads:
        phase_started_at = time.perf_counter()
        _persist_rendered_refs_payloads(
            refs=authoritative_full_refs,
            payload=authoritative_full_payloads,
            guide_mode=guide_mode,
            guide_source_path=guide_source_path,
            guide_source_name=guide_source_name,
        )
        _record("persist_authoritative", phase_started_at)

    if refs_norm and (not pending_refs) and (not failed_ready_refs) and (not ready_missing_refs) and stored_full_payload:
        _store_cached_conversation_refs_payload(
            conv_id=conv_id,
            signature=signature,
            payload=stored_full_payload,
            mode="full",
        )
        return _finish(stored_full_payload, "stored_full")

    if isinstance(cached_payload, dict) and (not stored_full_payload) and (not has_authoritative_doc_list):
        if (not has_pending) and (not failed_ready_refs) and cached_mode != "full":
            _warm_conversation_refs_payload_async(
                conv_id=conv_id,
                signature=signature,
                refs=refs_norm,
                guide_mode=guide_mode,
                guide_source_path=guide_source_path,
                guide_source_name=guide_source_name,
            )
        annotated_cached = _annotate_refs_payload_refresh_state(
            cached_payload,
            mode=cached_mode or ("pending" if has_pending else "fast"),
        )
        return _finish(annotated_cached, f"cache_{cached_mode or ('pending' if has_pending else 'fast')}")

    payload: dict[int, dict] = dict(stored_full_payload)
    if pending_refs:
        phase_started_at = time.perf_counter()
        pending_payload = _build_pending_conversation_refs_payload(
            pending_refs,
            doc_list_by_user=authoritative_doc_lists,
            guide_mode=bool(guide_mode),
            guide_source_path=str(guide_source_path or "").strip(),
            guide_source_name=str(guide_source_name or "").strip(),
        )
        for user_msg_id, pack in pending_refs.items():
            payload_pack = pending_payload.get(int(user_msg_id))
            if isinstance(payload_pack, dict):
                payload[int(user_msg_id)] = _attach_pack_render_state(
                    payload_pack,
                    source_pack=pack,
                    default_status="pending",
                )
        _record("pending_render", phase_started_at)
    if failed_ready_refs:
        phase_started_at = time.perf_counter()
        failed_payload = _build_fast_ready_conversation_refs_payload(
            refs=failed_ready_refs,
            guide_mode=guide_mode,
            guide_source_path=guide_source_path,
            guide_source_name=guide_source_name,
            deadline_at=route_deadline_at,
        )
        for user_msg_id, pack in failed_ready_refs.items():
            payload_pack = failed_payload.get(int(user_msg_id))
            if isinstance(payload_pack, dict):
                payload[int(user_msg_id)] = _attach_pack_render_state(
                    payload_pack,
                    source_pack=pack,
                    default_status="failed",
                )
        _record("failed_fast_render", phase_started_at)
    if ready_missing_refs:
        phase_started_at = time.perf_counter()
        fast_payload = _build_fast_ready_conversation_refs_payload(
            refs=ready_missing_refs,
            guide_mode=guide_mode,
            guide_source_path=guide_source_path,
            guide_source_name=guide_source_name,
            deadline_at=route_deadline_at,
        )
        for user_msg_id, pack in ready_missing_refs.items():
            payload_pack = fast_payload.get(int(user_msg_id))
            if isinstance(payload_pack, dict):
                payload[int(user_msg_id)] = _attach_pack_render_state(
                    payload_pack,
                    source_pack=pack,
                    default_status="fast",
                )
        _record("fast_render", phase_started_at)
        _warm_conversation_refs_payload_async(
            conv_id=conv_id,
            signature=signature,
            refs=refs_norm,
            guide_mode=guide_mode,
            guide_source_path=guide_source_path,
            guide_source_name=guide_source_name,
        )

    cache_mode = "full"
    if ready_missing_refs:
        cache_mode = "fast"
    elif failed_ready_refs:
        cache_mode = "fast"
    elif pending_refs:
        cache_mode = "pending"
    if isinstance(payload, dict):
        _store_cached_conversation_refs_payload(
            conv_id=conv_id,
            signature=signature,
            payload=payload,
            mode=cache_mode,
        )
    return _finish(payload, cache_mode)


@router.get("/conversation/{conv_id}")
def get_conversation_refs_route(conv_id: str, response: Response):
    return get_conversation_refs(conv_id, response=response)


@router.get("/diagnose/{conv_id}")
def get_refs_diagnose(conv_id: str):
    """Return a diagnostic report for why reference cards are empty/suppressed."""
    store = get_chat_store()
    try:
        refs = store.list_message_refs(conv_id, timeout_s=10.0)
    except Exception:
        refs = None
    if refs is None:
        raise HTTPException(404, f"Conversation {conv_id} not found or has no refs data")
    return _build_diagnostic_report(store=store, conv_id=conv_id, refs=refs)


class OpenReferenceBody(BaseModel):
    source_path: str
    page: int | None = None


class CitationMetaBody(BaseModel):
    source_path: str


class BibliometricsBody(BaseModel):
    meta: dict


class ReaderDocBody(BaseModel):
    source_path: str


_ASSET_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
_MD_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_MD_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
_MD_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
_MD_FENCE_RE = re.compile(r"^\s*(```+|~~~+)\s*")
_EQ_NUMBER_RE = re.compile(r"(?:\b(?:eq|equation|公式)\s*[#(（]?\s*|[\(（])(\d{1,4})(?:\s*[)）])", re.IGNORECASE)
_INLINE_EQ_RE = re.compile(r"\$[^$]{1,280}\$")
_TEX_CMD_RE = re.compile(r"\\[a-zA-Z]{2,}")


@router.post("/open")
def open_reference(body: OpenReferenceBody):
    ok, message = open_reference_source(
        source_path=body.source_path,
        pdf_root=_pdf_dir(),
        page=body.page,
    )
    if not ok:
        raise HTTPException(404, message)
    return {"ok": True, "message": message}


@router.post("/citation-meta")
def get_reference_citation_meta(body: CitationMetaBody):
    return ensure_source_citation_meta(
        source_path=body.source_path,
        pdf_root=_pdf_dir(),
        md_root=_md_dir(),
        lib_store=_lib_store(),
    )


@router.post("/bibliometrics")
def get_bibliometrics(body: BibliometricsBody):
    return enrich_citation_detail_meta(body.meta or {})


def _resolve_reader_md_path(source_path: str) -> Path | None:
    raw = str(source_path or "").strip()
    if not raw:
        return None
    src = Path(raw).expanduser()
    if src.suffix.lower().endswith(".md"):
        try:
            if src.exists() and src.is_file():
                return src.resolve(strict=False)
        except Exception:
            return None
        return None

    pdf_root = _pdf_dir()
    md_root = _md_dir()

    pdf_candidate = src
    try:
        if (not pdf_candidate.is_absolute()) and (Path(pdf_candidate).name == str(pdf_candidate)):
            pdf_candidate = pdf_root / pdf_candidate
    except Exception:
        pass

    try:
        if not (pdf_candidate.exists() and pdf_candidate.is_file()):
            return None
    except Exception:
        return None

    try:
        _md_folder, md_main, md_exists = _resolve_md_output_paths(md_root, pdf_candidate)
    except Exception:
        return None
    if (not md_exists) or (not md_main.exists()) or (not md_main.is_file()):
        return None
    try:
        return md_main.resolve(strict=False)
    except Exception:
        return md_main


def _rewrite_md_asset_links(md_text: str, *, md_path: Path, asset_roots: list[Path]) -> str:
    text = str(md_text or "")
    if not text:
        return text

    def _replace(m: re.Match[str]) -> str:
        alt = str(m.group(1) or "")
        raw = str(m.group(2) or "").strip()
        if not raw:
            return m.group(0)
        url = raw.strip().strip("<>").split()[0].strip()
        low = url.lower()
        if low.startswith(("http://", "https://", "data:", "#", "/api/")):
            return m.group(0)
        try:
            cand = Path(url).expanduser()
            if not cand.is_absolute():
                cand = (md_path.parent / cand).resolve(strict=False)
            else:
                cand = cand.resolve(strict=False)
            if (not cand.exists()) or (not cand.is_file()):
                return m.group(0)
            if not _path_within_roots(cand, asset_roots):
                return m.group(0)
            asset_url = f"/api/references/asset?path={quote(str(cand), safe='')}"
            return f"![{alt}]({asset_url})"
        except Exception:
            return m.group(0)

    return _MD_IMAGE_RE.sub(_replace, text)


def _strip_md_inline_for_anchor(input_text: str) -> str:
    text = str(input_text or "")
    if not text:
        return ""
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _has_equation_signal(text: str) -> bool:
    src = str(text or "")
    if not src:
        return False
    if "$$" in src:
        return True
    low = src.lower()
    if "\\begin{equation" in low or "\\[" in src:
        return True
    if _INLINE_EQ_RE.search(src):
        return True
    if _TEX_CMD_RE.search(src) and re.search(r"[=^_]", src):
        return True
    return False


def _extract_equation_number(text: str) -> int:
    src = str(text or "")
    if not src:
        return 0
    m = _EQ_NUMBER_RE.search(src)
    if not m:
        return 0
    try:
        v = int(str(m.group(1) or "0"))
    except Exception:
        return 0
    return v if v > 0 else 0


def _anchor_id(kind: str, index: int) -> str:
    prefix_map = {
        "heading": "hd",
        "paragraph": "p",
        "equation": "eq",
        "list_item": "li",
        "blockquote": "bq",
        "code": "cd",
        "table": "tb",
    }
    prefix = prefix_map.get(str(kind or "").strip().lower(), "a")
    return f"{prefix}_{int(max(1, index)):05d}"


def _build_reader_anchors(md_text: str, *, md_path: Path) -> tuple[list[dict], list[dict]]:
    blocks = load_source_blocks(md_path, md_text=md_text)
    anchors = source_blocks_to_reader_anchors(blocks)
    return anchors, blocks


@router.post("/reader/doc")
def get_reader_doc(body: ReaderDocBody):
    source_path = str(body.source_path or "").strip()
    if not source_path:
        raise HTTPException(400, "source_path required")
    md_path = _resolve_reader_md_path(source_path)
    if md_path is None:
        raise HTTPException(404, "markdown not found for source")
    try:
        md_text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        raise HTTPException(500, "failed to read markdown")

    md_render = _rewrite_md_asset_links(
        md_text,
        md_path=md_path,
        asset_roots=_reference_asset_roots(),
    )
    anchors, blocks = _build_reader_anchors(md_text, md_path=md_path)
    source_name = md_path.name
    low = source_name.lower()
    if low.endswith(".en.md"):
        source_name = source_name[:-6] + ".pdf"
    elif low.endswith(".md"):
        source_name = source_name[:-3] + ".pdf"

    return {
        "ok": True,
        "source_path": source_path,
        "source_name": source_name,
        "md_path": str(md_path),
        "markdown": md_render,
        "anchors": anchors,
        "blocks": blocks,
    }


@router.get("/asset")
def get_reference_asset(path: str):
    raw = str(path or "").strip()
    if not raw:
        raise HTTPException(404, "asset not found")
    try:
        resolved = Path(raw).expanduser().resolve()
    except Exception:
        raise HTTPException(404, "asset not found")
    if (not resolved.exists()) or (not resolved.is_file()):
        raise HTTPException(404, "asset not found")
    if resolved.suffix.lower() not in _ASSET_IMAGE_EXTS:
        raise HTTPException(404, "asset not found")
    if not _path_within_roots(resolved, _reference_asset_roots()):
        raise HTTPException(404, "asset not found")
    media_type = str(mimetypes.guess_type(str(resolved))[0] or "application/octet-stream")
    return FileResponse(str(resolved), media_type=media_type, filename=resolved.name)
