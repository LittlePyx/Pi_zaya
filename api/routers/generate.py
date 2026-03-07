from __future__ import annotations

import hashlib
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


@router.post("")
def start_generation(body: GenerateBody):
    settings = get_settings()
    chat_store = get_chat_store()
    prefs = load_prefs()
    session_id = uuid.uuid4().hex
    task_id = uuid.uuid4().hex
    prompt = str(body.prompt or "").strip()
    max_tokens = max(256, min(4096, int(body.max_tokens or 1216)))
    image_attachments = [_normalize_chat_image_attachment(it) for it in list(body.image_attachments or []) if isinstance(it, dict)]
    if (not prompt) and (not image_attachments):
        raise HTTPException(400, "prompt or image_attachments required")

    user_store_text = prompt if prompt else f"[Image attachment x{len(image_attachments)}]"
    user_msg_id = chat_store.append_message(body.conv_id, "user", user_store_text, attachments=image_attachments)
    assistant_msg_id = chat_store.append_message(
        body.conv_id, "assistant", _live_assistant_text(task_id)
    )
    chat_store.set_title_if_default(body.conv_id, user_store_text[:60])

    task = {
        "id": task_id,
        "session_id": session_id,
        "conv_id": body.conv_id,
        "prompt": prompt,
        "prompt_sig": hashlib.sha1(prompt.encode("utf-8", "ignore")).hexdigest()[:12] if prompt else "",
        "image_attachments": image_attachments,
        "preferred_sources": [str(x or "").strip() for x in list(body.preferred_sources or []) if str(x or "").strip()][:4],
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
    }
    _gen_start_task(task)
    return {
        "session_id": session_id,
        "task_id": task_id,
        "user_msg_id": user_msg_id,
        "assistant_msg_id": assistant_msg_id,
    }


@router.get("/{session_id}/stream")
async def stream_generation(session_id: str):
    def poll():
        t = _gen_get_task(session_id)
        if t is None:
            return {"done": True, "error": "not_found"}
        return {
            "stage": t.get("stage", ""),
            "partial": t.get("partial", ""),
            "char_count": t.get("char_count", 0),
            "done": t.get("status") in ("done", "error", "canceled"),
            "status": t.get("status", ""),
            "answer": t.get("answer", ""),
            "answer_intent": t.get("answer_intent", ""),
            "answer_depth": t.get("answer_depth", ""),
            "answer_contract_v1": bool(t.get("answer_contract_v1", False)),
            "answer_quality": t.get("answer_quality", {}),
        }

    return sse_response(sse_generator(poll, interval=0.15))


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
