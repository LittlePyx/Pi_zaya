from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.deps import get_chat_store

router = APIRouter(prefix="/api", tags=["chat"])


class CreateConvBody(BaseModel):
    title: str = "新对话"


class AppendMsgBody(BaseModel):
    role: str = "user"
    content: str


class UpdateMsgBody(BaseModel):
    content: str


class UpdateTitleBody(BaseModel):
    title: str


@router.get("/conversations")
def list_conversations(limit: int = 50):
    return get_chat_store().list_conversations(limit=limit)


@router.post("/conversations")
def create_conversation(body: CreateConvBody):
    conv_id = get_chat_store().create_conversation(body.title)
    return {"id": conv_id}


@router.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    get_chat_store().delete_conversation(conv_id)
    return {"ok": True}


@router.get("/conversations/{conv_id}/messages")
def get_messages(conv_id: str, limit: int | None = None):
    return get_chat_store().get_messages(conv_id, limit=limit)


@router.post("/conversations/{conv_id}/messages")
def append_message(conv_id: str, body: AppendMsgBody):
    msg_id = get_chat_store().append_message(conv_id, body.role, body.content)
    return {"id": msg_id}


@router.patch("/messages/{msg_id}")
def update_message(msg_id: int, body: UpdateMsgBody):
    ok = get_chat_store().update_message_content(msg_id, body.content)
    if not ok:
        raise HTTPException(404, "message not found")
    return {"ok": True}


@router.delete("/messages/{msg_id}")
def delete_message(msg_id: int):
    ok = get_chat_store().delete_message(msg_id)
    if not ok:
        raise HTTPException(404, "message not found")
    return {"ok": True}


@router.get("/conversations/{conv_id}/refs")
def list_refs(conv_id: str):
    return get_chat_store().list_message_refs(conv_id)


@router.patch("/conversations/{conv_id}/title")
def update_title(conv_id: str, body: UpdateTitleBody):
    get_chat_store().set_title_if_default(conv_id, body.title)
    return {"ok": True}
