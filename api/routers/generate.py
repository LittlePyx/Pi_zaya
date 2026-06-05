from __future__ import annotations

import hashlib
import re
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.deps import get_settings, get_chat_store, load_prefs
from api.routers.chat import _normalize_chat_image_attachment
from api.sse import sse_generator, sse_response
from kb.task_runtime import (
    _gen_start_task,
    _gen_get_task,
    _gen_mark_cancel,
    _gen_answer_quality_summary,
    _live_assistant_text,
)


def _strip_internal_structured_markers(text: str) -> str:
    """Final safety net: never leak internal grounding markers in /api/generate output.

    Strips [[SUPPORT:...]] tokens which are internal grounding metadata that should
    never be user-visible.  Preserves [[CITE:...]] tokens because they are intentionally
    generated citation markers that the renderer converts to user-visible links.
    """
    s = str(text or "")
    if not s:
        return s
    s = re.sub(r"\[\[\s*SUPPORT\s*:[^\]]+\]\]", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


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


class GenerateBody(BaseModel):
    conv_id: str
    prompt: str = ""
    top_k: int = 6
    temperature: float = 0.2
    max_tokens: int = 1216
    deep_read: bool = False
    image_attachments: list[dict] = Field(default_factory=list)
    preferred_sources: list[str] = Field(default_factory=list)
    source_lock_path: str = ""
    source_lock_name: str = ""


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
    image_attachments = [_normalize_chat_image_attachment(it) for it in list(body.image_attachments or []) if isinstance(it, dict)]
    if (not prompt) and (not image_attachments):
        raise HTTPException(400, "prompt or image_attachments required")
    conv_meta = chat_store.get_conversation(body.conv_id) or {}

    user_store_text = prompt if prompt else f"[Image attachment x{len(image_attachments)}]"
    user_msg_id = chat_store.append_message(body.conv_id, "user", user_store_text, attachments=image_attachments)
    assistant_msg_id = chat_store.append_message(
        body.conv_id,
        "assistant",
        _live_assistant_text(task_id),
        meta={"trace_id": trace_id},
    )
    title_candidate = _conversation_title_candidate(
        prompt,
        image_count=len(image_attachments),
        current_title=str(conv_meta.get("title") or ""),
    )
    title_changed = bool(chat_store.set_title_if_default(body.conv_id, title_candidate))
    conversation_title = title_candidate if title_changed else str(conv_meta.get("title") or "").strip()
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
        conv_mode = "paper_guide"
        bound_source_path = source_lock_path
        if source_lock_name:
            bound_source_name = source_lock_name
        bound_source_ready = True
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
        "settings_obj": settings,
        "user_msg_id": user_msg_id,
        "assistant_msg_id": assistant_msg_id,
        "answer_contract_v1": bool(prefs.get("answer_contract_v1", False)),
        "answer_depth_auto": bool(prefs.get("answer_depth_auto", True)),
        "answer_mode_hint": str(prefs.get("answer_mode_hint") or "").strip()[:32],
        "answer_output_mode": str(prefs.get("answer_output_mode") or "").strip()[:32],
    }
    _gen_start_task(task)
    return {
        "session_id": session_id,
        "task_id": task_id,
        "trace_id": trace_id,
        "user_msg_id": user_msg_id,
        "assistant_msg_id": assistant_msg_id,
        "conversation_title": conversation_title,
    }


@router.get("/{session_id}/stream")
async def stream_generation(session_id: str):
    def poll():
        t = _gen_get_task(session_id)
        if t is None:
            return {"done": True, "error": "not_found"}
        answer = _strip_internal_structured_markers(str(t.get("answer", "") or ""))
        return {
            "stream_schema_version": 2,
            "stage": t.get("stage", ""),
            "partial": t.get("partial", ""),
            "char_count": t.get("char_count", 0),
            "done": t.get("status") in ("done", "error", "canceled"),
            "status": t.get("status", ""),
            "answer": answer,
            "answer_intent": t.get("answer_intent", ""),
            "answer_depth": t.get("answer_depth", ""),
            "answer_output_mode": t.get("answer_output_mode", ""),
            "answer_contract_v1": bool(t.get("answer_contract_v1", False)),
            "answer_quality": t.get("answer_quality", {}),
            "paper_guide_debug": t.get("paper_guide_debug", {}),
            "research_trace": t.get("research_trace", {}),
        }

    return sse_response(sse_generator(poll, interval=0.15))


@router.get("/{session_id}/trace")
def generation_trace(session_id: str):
    t = _gen_get_task(session_id)
    if t is None:
        raise HTTPException(404, "generation session not found")
    return t.get("research_trace", {}) or {}


@router.post("/{session_id}/cancel")
def cancel_generation(session_id: str, task_id: str):
    ok = _gen_mark_cancel(session_id, task_id)
    return {"ok": ok}


@router.get("/quality/summary")
def generation_quality_summary(
    limit: int = 200,
    intent: str = "",
    depth: str = "",
    only_failed: bool = False,
):
    return _gen_answer_quality_summary(
        limit=limit,
        intent=intent,
        depth=depth,
        only_failed=only_failed,
    )
