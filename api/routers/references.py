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
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.deps import get_chat_store, get_settings, load_prefs
from api.reference_ui import (
    _attach_pack_primary_ref_evidence,
    _compact_reader_open_text,
    _filter_pending_refs_hits_by_prompt_focus,
    _refs_prompt_focus_terms,
    enrich_citation_detail_meta,
    enrich_refs_payload,
    ensure_source_citation_meta,
    open_reference_source,
)
from kb.file_ops import _resolve_md_output_paths
from kb.library_store import LibraryStore
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
_REFS_RENDER_PAYLOAD_SCHEMA_VERSION = 3


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
) -> str:
    try:
        prefs = load_prefs()
    except Exception:
        prefs = {}
    refs_digest: list[dict] = []
    for user_msg_id, pack in sorted((refs or {}).items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0])):
        if not isinstance(pack, dict):
            continue
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
        payload = {
            "user_msg_id": int(user_msg_id) if str(user_msg_id).isdigit() else str(user_msg_id),
            "prompt_sig": str(pack.get("prompt_sig") or "").strip(),
            "used_query": str(pack.get("used_query") or "").strip(),
            "used_translation": bool(pack.get("used_translation")),
            "updated_at": float(pack.get("updated_at") or 0.0),
            "hit_count": len(hits),
            "pending_count": pending_count,
            "top_sources": source_keys,
        }
        refs_digest.append(payload)
    payload = {
        "render_schema": _REFS_RENDER_PAYLOAD_SCHEMA_VERSION,
        "guide_mode": bool(guide_mode),
        "guide_source_path": str(guide_source_path or "").strip(),
        "guide_source_name": str(guide_source_name or "").strip(),
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


def _refs_conversation_read_timeout_s() -> float:
    try:
        raw = float(str(os.environ.get("KB_REFS_CONVERSATION_READ_TIMEOUT_S", "0.35") or "0.35"))
    except Exception:
        raw = 0.35
    return max(0.05, min(2.0, raw))


def _refs_payload_has_pending(refs: dict) -> bool:
    for pack in list((refs or {}).values()):
        if not isinstance(pack, dict):
            continue
        if _refs_pack_has_pending(pack):
            return True
    return False


def _refs_pack_has_pending(pack: dict) -> bool:
    if not isinstance(pack, dict):
        return False
    for hit in list(pack.get("hits") or []):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        if str((meta or {}).get("ref_pack_state") or "").strip().lower() == "pending":
            return True
    return False


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
    return dict(payload)


def _build_pending_conversation_refs_payload(refs: dict) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for user_msg_id, pack in (refs or {}).items():
        if not isinstance(pack, dict):
            continue
        prompt = str(pack.get("prompt") or "").strip()
        prompt_low = prompt.lower()
        focus_terms = [str(term or "").strip() for term in _refs_prompt_focus_terms(prompt) if str(term or "").strip()]
        raw_hits = [hit for hit in list(pack.get("hits") or []) if isinstance(hit, dict)]
        filtered_hits = _filter_pending_refs_hits_by_prompt_focus(prompt, raw_hits)[:2]
        pending_count = 0
        hits_out: list[dict] = []
        for hit in raw_hits:
            if not isinstance(hit, dict):
                continue
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            if str((meta or {}).get("ref_pack_state") or "").strip().lower() == "pending":
                pending_count += 1
        for hit in filtered_hits:
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            source_path = str((meta or {}).get("source_path") or "").strip()
            heading_path = str((meta or {}).get("ref_best_heading_path") or (meta or {}).get("top_heading") or "").strip()
            display_name = Path(source_path).name.replace(".en.md", ".pdf") if source_path else "Reference"
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
            if re.search(r"\b(compare|compares|compared|comparison|versus|vs\.?)\b", prompt_low):
                why_line = f"This pending match most directly compares {focus_text} in {heading_path or 'the matched section'}."
            elif re.search(r"\b(define|defined|definition)\b", prompt_low):
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
        out[int(user_msg_id)] = _attach_pack_primary_ref_evidence(pack2)
    return out


def _annotate_refs_payload_refresh_state(payload: dict, *, mode: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    mode_norm = str(mode or "").strip().lower() or "full"
    needs_enrichment = mode_norm in {"fast", "pending"}
    for user_msg_id, pack in (payload or {}).items():
        if not isinstance(pack, dict):
            continue
        pack2 = _attach_pack_primary_ref_evidence(pack)
        pack2["payload_mode"] = mode_norm
        if needs_enrichment:
            pack2["enrichment_pending"] = True
        else:
            pack2.pop("enrichment_pending", None)
        out[int(user_msg_id)] = pack2
    return out


def _attach_pack_render_state(payload_pack: dict, *, source_pack: dict | None, default_status: str = "") -> dict:
    out = _attach_pack_primary_ref_evidence(payload_pack)
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
    return out


def _build_fast_ready_conversation_refs_payload(
    *,
    refs: dict,
    guide_mode: bool,
    guide_source_path: str,
    guide_source_name: str,
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
                allow_expensive_llm_for_ready=False,
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


@router.get("/conversation/{conv_id}")
def get_conversation_refs(conv_id: str):
    store = get_chat_store()
    read_timeout_s = _refs_conversation_read_timeout_s()
    try:
        conversation = store.get_conversation(conv_id, timeout_s=read_timeout_s) or {}
    except TypeError:
        conversation = store.get_conversation(conv_id) or {}
    except sqlite3.OperationalError:
        cached_any = _get_any_cached_conversation_refs_payload(conv_id=conv_id)
        return cached_any if isinstance(cached_any, dict) else {}
    guide_mode = str(conversation.get("mode") or "").strip().lower() == "paper_guide"
    guide_source_path = str(conversation.get("bound_source_path") or "").strip()
    guide_source_name = str(conversation.get("bound_source_name") or "").strip()
    try:
        refs = store.list_message_refs(conv_id, timeout_s=read_timeout_s)
    except TypeError:
        refs = store.list_message_refs(conv_id)
    except sqlite3.OperationalError:
        cached_any = _get_any_cached_conversation_refs_payload(conv_id=conv_id)
        return cached_any if isinstance(cached_any, dict) else {}
    signature = _refs_conversation_cache_signature(
        refs=refs if isinstance(refs, dict) else {},
        guide_mode=guide_mode,
        guide_source_path=guide_source_path,
        guide_source_name=guide_source_name,
    )
    refs_norm = refs if isinstance(refs, dict) else {}
    has_pending = _refs_payload_has_pending(refs_norm)
    cached_rec = _get_cached_conversation_refs_record(conv_id=conv_id, signature=signature)
    cached_payload = cached_rec.get("payload") if isinstance(cached_rec, dict) else None
    cached_mode = str(cached_rec.get("mode") or "").strip().lower() if isinstance(cached_rec, dict) else ""
    if isinstance(cached_payload, dict) and cached_mode == "full":
        return cached_payload

    stored_full_payload: dict[int, dict] = {}
    pending_refs: dict[int, dict] = {}
    failed_ready_refs: dict[int, dict] = {}
    ready_missing_refs: dict[int, dict] = {}
    for user_msg_id, pack in refs_norm.items():
        if not isinstance(pack, dict):
            continue
        pack_full = _get_stored_rendered_pack_payload(
            user_msg_id=user_msg_id,
            pack=pack,
            guide_mode=guide_mode,
            guide_source_path=guide_source_path,
            guide_source_name=guide_source_name,
        )
        if isinstance(pack_full, dict):
            stored_full_payload[int(user_msg_id)] = _attach_pack_render_state(
                pack_full,
                source_pack=pack,
                default_status="full",
            )
            continue
        if _refs_pack_has_pending(pack):
            pending_refs[int(user_msg_id)] = pack
        elif str((pack or {}).get("render_status") or "").strip().lower() == "failed":
            failed_ready_refs[int(user_msg_id)] = pack
        else:
            ready_missing_refs[int(user_msg_id)] = pack

    if refs_norm and (not pending_refs) and (not failed_ready_refs) and (not ready_missing_refs) and stored_full_payload:
        _store_cached_conversation_refs_payload(
            conv_id=conv_id,
            signature=signature,
            payload=stored_full_payload,
            mode="full",
        )
        return stored_full_payload

    if isinstance(cached_payload, dict) and (not stored_full_payload):
        if (not has_pending) and (not failed_ready_refs) and cached_mode != "full":
            _warm_conversation_refs_payload_async(
                conv_id=conv_id,
                signature=signature,
                refs=refs_norm,
                guide_mode=guide_mode,
                guide_source_path=guide_source_path,
                guide_source_name=guide_source_name,
            )
        return _annotate_refs_payload_refresh_state(
            cached_payload,
            mode=cached_mode or ("pending" if has_pending else "fast"),
        )

    payload: dict[int, dict] = dict(stored_full_payload)
    if pending_refs:
        pending_payload = _build_pending_conversation_refs_payload(pending_refs)
        for user_msg_id, pack in pending_refs.items():
            payload_pack = pending_payload.get(int(user_msg_id))
            if isinstance(payload_pack, dict):
                payload[int(user_msg_id)] = _attach_pack_render_state(
                    payload_pack,
                    source_pack=pack,
                    default_status="pending",
                )
    if failed_ready_refs:
        failed_payload = _build_fast_ready_conversation_refs_payload(
            refs=failed_ready_refs,
            guide_mode=guide_mode,
            guide_source_path=guide_source_path,
            guide_source_name=guide_source_name,
        )
        for user_msg_id, pack in failed_ready_refs.items():
            payload_pack = failed_payload.get(int(user_msg_id))
            if isinstance(payload_pack, dict):
                payload[int(user_msg_id)] = _attach_pack_render_state(
                    payload_pack,
                    source_pack=pack,
                    default_status="failed",
                )
    if ready_missing_refs:
        fast_payload = _build_fast_ready_conversation_refs_payload(
            refs=ready_missing_refs,
            guide_mode=guide_mode,
            guide_source_path=guide_source_path,
            guide_source_name=guide_source_name,
        )
        for user_msg_id, pack in ready_missing_refs.items():
            payload_pack = fast_payload.get(int(user_msg_id))
            if isinstance(payload_pack, dict):
                payload[int(user_msg_id)] = _attach_pack_render_state(
                    payload_pack,
                    source_pack=pack,
                    default_status="fast",
                )
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
    return payload


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
