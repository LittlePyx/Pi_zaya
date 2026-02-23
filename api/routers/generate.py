from __future__ import annotations

import uuid
from fastapi import APIRouter
from pydantic import BaseModel

from api.deps import get_settings, get_chat_store
from api.sse import sse_generator, sse_response
from kb.task_runtime import (
    _gen_start_task,
    _gen_get_task,
    _gen_mark_cancel,
    _live_assistant_text,
)

router = APIRouter(prefix="/api/generate", tags=["generate"])


class GenerateBody(BaseModel):
    conv_id: str
    prompt: str
    top_k: int = 6
    temperature: float = 0.2
    max_tokens: int = 1216
    deep_read: bool = False


@router.post("")
def start_generation(body: GenerateBody):
    settings = get_settings()
    chat_store = get_chat_store()
    session_id = uuid.uuid4().hex
    task_id = uuid.uuid4().hex

    user_msg_id = chat_store.append_message(body.conv_id, "user", body.prompt)
    assistant_msg_id = chat_store.append_message(
        body.conv_id, "assistant", _live_assistant_text(task_id)
    )
    chat_store.set_title_if_default(body.conv_id, body.prompt[:60])

    task = {
        "id": task_id,
        "session_id": session_id,
        "conv_id": body.conv_id,
        "prompt": body.prompt,
        "prompt_sig": "",
        "chat_db": str(settings.chat_db_path),
        "db_dir": str(settings.db_dir),
        "top_k": body.top_k,
        "temperature": body.temperature,
        "max_tokens": body.max_tokens,
        "deep_read": body.deep_read,
        "settings_obj": settings,
        "user_msg_id": user_msg_id,
        "assistant_msg_id": assistant_msg_id,
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
        }

    return sse_response(sse_generator(poll, interval=0.15))


@router.post("/{session_id}/cancel")
def cancel_generation(session_id: str, task_id: str):
    ok = _gen_mark_cancel(session_id, task_id)
    return {"ok": ok}
