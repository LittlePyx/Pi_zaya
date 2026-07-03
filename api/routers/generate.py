from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any
from fastapi import APIRouter, Body, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.deps import get_settings, get_chat_store, load_prefs
from api.internal_access import internal_api_allowed, require_internal_api
from api.routers.chat import _normalize_chat_image_attachment, _resolve_allowed_paper_guide_source_path
from api.sse import sse_generator, sse_response
from kb.answer_presentation import clean_assistant_answer_presentation_text
from kb.generation_state_runtime import _strip_internal_generation_markers
from kb.path_safety import resolve_verified_chat_image_upload_path
from kb.task_runtime import (
    generation_start_failed_message,
    _gen_has_running_for_conversation,
    _gen_start_task,
    _gen_get_task,
    _gen_mark_cancel,
    _gen_answer_quality_summary,
    _live_assistant_text,
)


def _normalize_query_scope(value: object) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in {"current", "paper", "current_paper", "source", "reader"}:
        return "current_paper"
    if raw in {"basket", "shelf", "citation_shelf", "selected"}:
        return "basket"
    if raw in {"library", "all", "all_library", "full_library"}:
        return "library"
    return ""


def _generation_user_meta(prompt_context: object, query_scope: str, agent_mode: bool) -> dict[str, object]:
    meta: dict[str, object] = {}
    if prompt_context:
        meta["prompt_context"] = prompt_context
    if query_scope:
        meta["query_scope"] = query_scope
    if agent_mode:
        meta["agent_mode"] = "research_agent"
        meta["agent_mode_requested"] = True
    return meta


def _strip_internal_structured_markers(text: str) -> str:
    """Final safety net: never leak internal grounding markers in /api/generate output.

    Strips [[SUPPORT:...]] tokens which are internal grounding metadata that should
    never be user-visible.  Preserves [[CITE:...]] tokens because they are intentionally
    generated citation markers that the renderer converts to user-visible links.
    """
    return _strip_internal_generation_markers(text)


_TITLE_LEADING_NOISE_RE = re.compile(
    r"^(?:请问|帮我看看|帮我分析|帮我总结|帮我解释|帮我|麻烦你?|请你?|请|能不能|可以帮我|我想知道|想问一下|看一下|解释一下|总结一下)[\s,，:：;；-]*",
    flags=re.IGNORECASE,
)
_TITLE_EN_LEADING_NOISE_RE = re.compile(
    r"^(?:(?:please\s+)?(?:can|could)\s+you\s+|please\s+|help\s+me\s+|tell\s+me\s+|explain\s+|summarize\s+)[\s,;:-]*",
    flags=re.IGNORECASE,
)
_TITLE_FOCUS_CUE_RE = re.compile(
    r"(为什么|是什么|如何|怎样|怎么|区别|对比|影响|原因|机制|where|why|how|what|which|compare|difference|effect|mechanism)",
    flags=re.IGNORECASE,
)


def _title_locale(current_title: str, prompt: str) -> str:
    text = f"{current_title} {prompt}"
    return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"


def _strip_title_noise(text: str) -> str:
    s = str(text or "")
    s = re.sub(r"\[\[\s*(?:SUPPORT|CITE)\s*:[^\]]+\]\]", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<!--[\s\S]*?-->", " ", s)
    s = re.sub(r"!\[[^\]\n]*\]\([^)]+\)", " ", s)
    s = re.sub(r"\[([^\]\n]{1,160})\]\([^)]+\)", r"\1", s)
    s = re.sub(r"`([^`\n]{1,160})`", r"\1", s)
    s = re.sub(r"^\s{0,3}#{1,6}\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s{0,3}(?:[-*+]|\d+[.)])\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"[\r\n]+", " ", s)
    s = re.sub(r"[*_~<>|]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for _ in range(2):
        next_s = _TITLE_LEADING_NOISE_RE.sub("", s).strip()
        next_s = _TITLE_EN_LEADING_NOISE_RE.sub("", next_s).strip()
        if next_s == s:
            break
        s = next_s
    return s.strip(" \t\r\n-–—:：,，.。?？!！;；")


def _focus_title_clause(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    for sep in ("。", "？", "！", "?", "!", ";", "；"):
        idx = s.find(sep)
        if 8 <= idx <= 80:
            return s[:idx].strip()
    for sep in ("，", ","):
        idx = s.find(sep)
        before = s[:idx].strip() if idx >= 0 else ""
        if 8 <= idx <= 80 and _TITLE_FOCUS_CUE_RE.search(before):
            return before
    return s


def _clip_conversation_title(text: str, *, locale: str) -> str:
    s = str(text or "").strip(" \t\r\n-–—:：,，.。?？!！;；")
    if not s:
        return ""
    max_chars = 34 if locale == "zh" else 58
    min_cut = 8 if locale == "zh" else 16
    if len(s) <= max_chars:
        return s
    window = s[: max_chars + 8]
    cuts = [window.rfind(ch) for ch in ("。", "？", "！", "?", "!", "；", ";", "，", ",", ":", "：")]
    cut = max(cuts)
    if cut >= min_cut:
        return window[:cut].strip(" \t\r\n-–—:：,，.。?？!！;；")
    if locale != "zh":
        space = window.rfind(" ", 0, max_chars + 1)
        if space >= min_cut:
            return window[:space].strip(" \t\r\n-–—:：,，.。?？!！;；")
    return s[:max_chars].strip(" \t\r\n-–—:：,，.。?？!！;；")


def _conversation_title_candidate(prompt: str, *, image_count: int = 0, current_title: str = "") -> str:
    locale = _title_locale(current_title, prompt)
    if not str(prompt or "").strip():
        if image_count > 0:
            return (f"图片提问 x{image_count}" if locale == "zh" else f"Image question x{image_count}")[:80]
        return ""
    cleaned = _strip_title_noise(prompt)
    focused = _focus_title_clause(cleaned)
    title = _clip_conversation_title(focused, locale=locale)
    if not title and image_count > 0:
        title = f"图片提问 x{image_count}" if locale == "zh" else f"Image question x{image_count}"
    return title[:80]


router = APIRouter(prefix="/api/generate", tags=["generate"])

_PROMPT_CONTEXT_MAX_ITEMS = 8
_PROMPT_CONTEXT_MAX_TEXT = 900
_PROMPT_CONTEXT_MAX_TOTAL = 4200
_GENERATE_CONV_ID_MAX_CHARS = 120
_GENERATE_PROMPT_MAX_CHARS = 80_000
_GENERATE_SOURCE_PATH_MAX_CHARS = 1_200
_GENERATE_SOURCE_NAME_MAX_CHARS = 500
_GENERATE_QUERY_SCOPE_MAX_CHARS = 40
_GENERATE_MAX_IMAGE_ATTACHMENTS = 4
_GENERATE_IMAGE_ATTACHMENT_MAX_JSON_CHARS = 40_000
_GENERATE_IMAGE_ATTACHMENTS_MAX_JSON_CHARS = 90_000
_GENERATE_MAX_PREFERRED_SOURCES = 12
_GENERATE_PREFERRED_SOURCE_MAX_CHARS = 1_200
_GENERATE_PROMPT_CONTEXT_MAX_JSON_CHARS = 260_000


def _json_size(value: Any, *, name: str, max_json_chars: int) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    except Exception as exc:
        raise ValueError(f"{name} must be JSON serializable") from exc
    if len(encoded) > int(max_json_chars):
        raise ValueError(f"{name} is too large; max {int(max_json_chars)} JSON chars")
    return value


def _json_dict(value: Any, *, name: str, max_json_chars: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return _json_size(value, name=name, max_json_chars=max_json_chars)


def _json_dict_list(value: Any, *, name: str, max_items: int, max_json_chars: int, item_max_json_chars: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    if len(value) > int(max_items):
        raise ValueError(f"{name} has too many items; max {int(max_items)}")
    for item in value:
        _json_dict(item, name=f"{name} item", max_json_chars=item_max_json_chars)
    return _json_size(value, name=name, max_json_chars=max_json_chars)


def _clip_prompt_context_text(value, max_chars: int = _PROMPT_CONTEXT_MAX_TEXT) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _sanitize_prompt_context(value) -> dict:
    if not isinstance(value, dict):
        return {}
    raw_items = value.get("items")
    if not isinstance(raw_items, list):
        return {}
    items: list[dict] = []
    total_chars = 0
    allowed_text_fields = {
        "title": 240,
        "sourceName": 240,
        "sourcePath": 520,
        "locationLabel": 240,
        "doi": 180,
        "authors": 240,
        "year": 24,
        "summary": 900,
        "excerpt": 900,
        "note": 520,
        "kind": 40,
        "key": 160,
    }
    for raw in raw_items[:_PROMPT_CONTEXT_MAX_ITEMS]:
        if not isinstance(raw, dict):
            continue
        item: dict = {}
        for field, limit in allowed_text_fields.items():
            text = _clip_prompt_context_text(raw.get(field), limit)
            if text:
                item[field] = text
        try:
            ref_num = int(raw.get("refNum") or 0)
        except Exception:
            ref_num = 0
        if ref_num > 0:
            item["refNum"] = ref_num
        if not any(str(item.get(k) or "").strip() for k in ("title", "summary", "excerpt", "note")):
            continue
        for field in ("title", "summary", "excerpt", "note"):
            total_chars += len(str(item.get(field) or ""))
        if total_chars > _PROMPT_CONTEXT_MAX_TOTAL:
            overflow = total_chars - _PROMPT_CONTEXT_MAX_TOTAL
            for field in ("note", "excerpt", "summary"):
                text = str(item.get(field) or "")
                if not text or overflow <= 0:
                    continue
                keep = max(0, len(text) - overflow)
                item[field] = (text[: max(0, keep - 3)].rstrip() + "...") if keep > 8 else text[:keep]
                overflow = max(0, overflow - (len(text) - len(str(item.get(field) or ""))))
            total_chars = _PROMPT_CONTEXT_MAX_TOTAL
        items.append(item)
        if total_chars >= _PROMPT_CONTEXT_MAX_TOTAL:
            break
    if not items:
        return {}
    token_estimate = max(1, min(1600, int(value.get("tokenEstimate") or ((total_chars + 3) // 4))))
    return {
        "version": 1,
        "source": "citation_shelf",
        "id": _clip_prompt_context_text(value.get("id"), 120),
        "createdAt": value.get("createdAt") if isinstance(value.get("createdAt"), (int, float)) else None,
        "conversationId": _clip_prompt_context_text(value.get("conversationId"), 120),
        "guideSourcePath": _clip_prompt_context_text(value.get("guideSourcePath"), 520),
        "guideSourceName": _clip_prompt_context_text(value.get("guideSourceName"), 240),
        "itemCount": len(items),
        "tokenEstimate": token_estimate,
        "items": items,
    }


def _safe_generation_image_attachments(items: list[dict], *, db_dir) -> list[dict]:
    out: list[dict] = []
    for raw in list(items or []):
        if not isinstance(raw, dict):
            continue
        rec = dict(_normalize_chat_image_attachment(raw))
        verified = resolve_verified_chat_image_upload_path(rec.get("path"), db_dir=db_dir)
        if verified is None:
            raise HTTPException(400, "invalid image attachment")
        image_path, mime = verified
        safe_rec = {
            "sha1": str(rec.get("sha1") or "").strip().lower(),
            "path": str(image_path),
            "name": str(rec.get("name") or image_path.name),
            "mime": mime,
        }
        out.append(_normalize_chat_image_attachment(safe_rec))
    return out[:4]


class GenerateBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    conv_id: str = Field(..., max_length=_GENERATE_CONV_ID_MAX_CHARS)
    prompt: str = Field("", max_length=_GENERATE_PROMPT_MAX_CHARS)
    top_k: int = Field(6, ge=1, le=20)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(1216, ge=1, le=8192)
    deep_read: bool = False
    agent_mode: bool = False
    image_attachments: list[dict[str, Any]] = Field(default_factory=list, max_length=_GENERATE_MAX_IMAGE_ATTACHMENTS)
    preferred_sources: list[str] = Field(default_factory=list, max_length=_GENERATE_MAX_PREFERRED_SOURCES)
    source_lock_path: str = Field("", max_length=_GENERATE_SOURCE_PATH_MAX_CHARS)
    source_lock_name: str = Field("", max_length=_GENERATE_SOURCE_NAME_MAX_CHARS)
    query_scope: str = Field("", max_length=_GENERATE_QUERY_SCOPE_MAX_CHARS)
    prompt_context: dict | None = None

    @field_validator("image_attachments")
    @classmethod
    def _check_image_attachments(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return _json_dict_list(
            value,
            name="image attachments",
            max_items=_GENERATE_MAX_IMAGE_ATTACHMENTS,
            max_json_chars=_GENERATE_IMAGE_ATTACHMENTS_MAX_JSON_CHARS,
            item_max_json_chars=_GENERATE_IMAGE_ATTACHMENT_MAX_JSON_CHARS,
        )

    @field_validator("preferred_sources")
    @classmethod
    def _check_preferred_sources(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for raw in list(value or []):
            text = str(raw or "").strip()
            if len(text) > _GENERATE_PREFERRED_SOURCE_MAX_CHARS:
                raise ValueError(f"preferred source is too long; max {_GENERATE_PREFERRED_SOURCE_MAX_CHARS} chars")
            out.append(text)
        return out

    @field_validator("prompt_context")
    @classmethod
    def _check_prompt_context(cls, value: dict | None) -> dict | None:
        if value is None:
            return None
        return _json_dict(value, name="prompt context", max_json_chars=_GENERATE_PROMPT_CONTEXT_MAX_JSON_CHARS)


class CancelGenerationBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_id: str = Field("", max_length=160)


@router.post("")
def start_generation(body: GenerateBody):
    settings = get_settings()
    chat_store = get_chat_store()
    prefs = load_prefs()
    session_id = uuid.uuid4().hex
    task_id = uuid.uuid4().hex
    trace_id = uuid.uuid4().hex[:16]
    prompt = str(body.prompt or "").strip()
    max_tokens = max(256, min(4096, int(body.max_tokens or 1216)))
    image_attachments = _safe_generation_image_attachments(
        [it for it in list(body.image_attachments or []) if isinstance(it, dict)],
        db_dir=settings.db_dir,
    )
    prompt_context = _sanitize_prompt_context(body.prompt_context)
    query_scope = _normalize_query_scope(body.query_scope)
    if (not prompt) and (not image_attachments):
        raise HTTPException(400, "prompt or image_attachments required")
    conv_meta = chat_store.get_conversation(body.conv_id)
    if conv_meta is None:
        raise HTTPException(404, "conversation not found")
    if _gen_has_running_for_conversation(body.conv_id, chat_db_path=settings.chat_db_path):
        raise HTTPException(409, "generation already running for this conversation")
    conv_mode = str(conv_meta.get("mode") or "normal").strip().lower()
    bound_source_path = str(conv_meta.get("bound_source_path") or "").strip()
    bound_source_name = str(conv_meta.get("bound_source_name") or "").strip()
    try:
        bound_source_ready = bool(int(conv_meta.get("bound_source_ready") or 0))
    except Exception:
        bound_source_ready = False
    source_lock_path = str(body.source_lock_path or "").strip()
    source_lock_name = str(body.source_lock_name or "").strip()
    if source_lock_path:
        source_lock_path = _resolve_allowed_paper_guide_source_path(source_lock_path)
        conv_mode = "paper_guide"
        bound_source_path = source_lock_path
        if source_lock_name:
            bound_source_name = source_lock_name
        bound_source_ready = True
    elif conv_mode == "paper_guide" and bound_source_path:
        bound_source_path = _resolve_allowed_paper_guide_source_path(bound_source_path)

    user_store_text = prompt if prompt else f"[Image attachment x{len(image_attachments)}]"
    agent_mode = bool(body.agent_mode)
    user_meta = _generation_user_meta(prompt_context, query_scope, agent_mode)
    user_msg_id = chat_store.append_message(
        body.conv_id,
        "user",
        user_store_text,
        attachments=image_attachments,
        meta=user_meta or None,
    )
    assistant_msg_id = chat_store.append_message(
        body.conv_id,
        "assistant",
        _live_assistant_text(task_id),
        meta={"trace_id": trace_id, **({"agent_mode": "research_agent"} if agent_mode else {})},
    )
    preferred_sources = [str(x or "").strip() for x in list(body.preferred_sources or []) if str(x or "").strip()]
    if conv_mode == "paper_guide":
        for hint in (bound_source_path, bound_source_name):
            if (not hint) or (hint in preferred_sources):
                continue
            preferred_sources.insert(0, hint)
    preferred_sources = preferred_sources[:4]

    task = {
        "id": task_id,
        "trace_id": trace_id,
        "session_id": session_id,
        "conv_id": body.conv_id,
        "prompt": prompt,
        "prompt_sig": hashlib.sha1(prompt.encode("utf-8", "ignore")).hexdigest()[:12] if prompt else "",
        "image_attachments": image_attachments,
        "selected_research_context": prompt_context,
        "query_scope": query_scope,
        "preferred_sources": preferred_sources,
        "paper_guide_mode": conv_mode == "paper_guide",
        "paper_guide_bound_source_path": bound_source_path,
        "paper_guide_bound_source_name": bound_source_name,
        "paper_guide_bound_source_ready": bool(bound_source_ready and bound_source_path),
        "chat_db": str(settings.chat_db_path),
        "db_dir": str(settings.db_dir),
        "top_k": body.top_k,
        "temperature": body.temperature,
        "max_tokens": max_tokens,
        "deep_read": body.deep_read,
        "agent_mode": agent_mode,
        "settings_obj": settings,
        "user_msg_id": user_msg_id,
        "assistant_msg_id": assistant_msg_id,
        "ui_locale": str(prefs.get("ui_locale") or "").strip(),
        "answer_contract_v1": bool(prefs.get("answer_contract_v1", False)),
        "answer_depth_auto": bool(prefs.get("answer_depth_auto", True)),
        "answer_mode_hint": str(prefs.get("answer_mode_hint") or "").strip()[:32],
        "answer_output_mode": str(prefs.get("answer_output_mode") or "").strip()[:32],
    }
    started = bool(_gen_start_task(task))
    start_error = ""
    if (not started) and _gen_has_running_for_conversation(body.conv_id, chat_db_path=settings.chat_db_path):
        for message_id in (assistant_msg_id, user_msg_id):
            try:
                chat_store.delete_message(message_id)
            except Exception:
                pass
        raise HTTPException(409, "generation already running for this conversation")

    title_candidate = _conversation_title_candidate(
        prompt,
        image_count=len(image_attachments),
        current_title=str(conv_meta.get("title") or ""),
    )
    title_changed = bool(chat_store.set_title_if_default(body.conv_id, title_candidate))
    latest_conv_meta = chat_store.get_conversation(body.conv_id) or conv_meta
    conversation_title = title_candidate if title_changed else str(latest_conv_meta.get("title") or "").strip()
    if not started:
        start_error = "generation_start_failed"
        failure_message = generation_start_failed_message(prefs.get("ui_locale"))
        try:
            chat_store.update_message_content(assistant_msg_id, failure_message)
        except Exception:
            pass
    return {
        "session_id": session_id,
        "task_id": task_id,
        "trace_id": trace_id,
        "user_msg_id": user_msg_id,
        "assistant_msg_id": assistant_msg_id,
        "conversation_title": conversation_title,
        "started": started,
        "start_error": start_error,
    }


@router.get("/{session_id}/stream")
async def stream_generation(session_id: str, request: Request):
    include_internal_debug = internal_api_allowed(request)

    def poll():
        t = _gen_get_task(session_id)
        if t is None:
            failure_message = generation_start_failed_message(load_prefs().get("ui_locale"))
            return {
                "stream_schema_version": 2,
                "stage": "error",
                "partial": failure_message,
                "char_count": len(failure_message),
                "done": True,
                "status": "error",
                "answer": failure_message,
                "error": "not_found",
                "answer_intent": "",
                "answer_depth": "",
                "answer_output_mode": "",
                "answer_contract_v1": False,
                "answer_quality": {},
                "paper_guide_debug": {},
                "research_trace": {},
                "agent_trace": {},
                "agent_source_summary": {},
                "answer_runtime_check": {},
                "answer_contract": {},
            }
        partial = _strip_internal_structured_markers(str(t.get("partial", "") or ""))
        answer = _strip_internal_structured_markers(str(t.get("answer", "") or ""))
        if bool(t.get("agent_mode")):
            partial = clean_assistant_answer_presentation_text(partial)
            answer = clean_assistant_answer_presentation_text(answer)
        visible_text = partial or answer
        return {
            "stream_schema_version": 2,
            "stage": t.get("stage", ""),
            "partial": partial,
            "char_count": len(visible_text) if visible_text else t.get("char_count", 0),
            "done": t.get("status") in ("done", "error", "canceled"),
            "status": t.get("status", ""),
            "answer": answer,
            "error": t.get("error", ""),
            "answer_intent": t.get("answer_intent", ""),
            "answer_depth": t.get("answer_depth", ""),
            "answer_output_mode": t.get("answer_output_mode", ""),
            "answer_contract_v1": bool(t.get("answer_contract_v1", False)),
            "answer_quality": t.get("answer_quality", {}),
            "paper_guide_debug": t.get("paper_guide_debug", {}) if include_internal_debug else {},
            "research_trace": t.get("research_trace", {}) if include_internal_debug else {},
            "agent_trace": t.get("agent_trace", {}) if bool(t.get("agent_mode")) else {},
            "agent_source_summary": t.get("agent_source_summary", {}) if bool(t.get("agent_mode")) else {},
            "answer_runtime_check": t.get("answer_runtime_check", {}) if bool(t.get("agent_mode")) else {},
            "answer_contract": t.get("answer_contract", {}) if bool(t.get("agent_mode")) else {},
        }

    return sse_response(sse_generator(poll, interval=0.15))


@router.get("/{session_id}/trace")
def generation_trace(session_id: str, request: Request):
    require_internal_api(request)
    t = _gen_get_task(session_id)
    if t is None:
        raise HTTPException(404, "generation session not found")
    return t.get("research_trace", {}) or {}


@router.post("/{session_id}/cancel")
def cancel_generation(
    session_id: str,
    body: CancelGenerationBody | None = Body(default=None),
    task_id: str = Query(default=""),
):
    task_id_final = str(task_id or "").strip()
    if not task_id_final and body is not None:
        task_id_final = str(body.task_id or "").strip()
    ok = _gen_mark_cancel(session_id, task_id_final)
    return {"ok": ok}


@router.get("/quality/summary")
def generation_quality_summary(
    request: Request,
    limit: int = 200,
    intent: str = "",
    depth: str = "",
    only_failed: bool = False,
):
    require_internal_api(request)
    return _gen_answer_quality_summary(
        limit=limit,
        intent=intent,
        depth=depth,
        only_failed=only_failed,
    )
