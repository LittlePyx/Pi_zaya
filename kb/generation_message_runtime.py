from __future__ import annotations

import base64
from pathlib import Path


def _filter_history_for_multimodal_turn(
    history: list[dict],
    *,
    cur_user_msg_id: int,
    cur_assistant_msg_id: int,
    has_current_images: bool,
    is_live_assistant_text,
) -> list[dict]:
    hist: list[dict] = []
    suppress_followup_assistant = False

    for msg in history or []:
        if msg.get("role") not in ("user", "assistant"):
            continue
        try:
            mid = int(msg.get("id") or 0)
        except Exception:
            mid = 0
        if mid and mid in {cur_user_msg_id, cur_assistant_msg_id}:
            continue
        if callable(is_live_assistant_text) and is_live_assistant_text(str(msg.get("content") or "")):
            continue

        attachments = msg.get("attachments") if isinstance(msg.get("attachments"), list) else []
        has_message_images = any(
            isinstance(item, dict) and str(item.get("mime") or "").strip().lower().startswith("image/")
            for item in attachments
        )

        if has_current_images and str(msg.get("role") or "") == "user":
            suppress_followup_assistant = bool(has_message_images)
            if has_message_images:
                continue
        elif has_current_images and str(msg.get("role") or "") == "assistant" and suppress_followup_assistant:
            continue
        else:
            suppress_followup_assistant = False

        hist.append(msg)

    return hist


def _build_multimodal_user_content(
    user: str,
    image_attachments: list[dict] | None,
    *,
    vision_image_mime_by_suffix: dict[str, str],
) -> str | list[dict]:
    user_content: str | list[dict] = str(user or "")
    if not image_attachments:
        return user_content

    mm_parts: list[dict] = [{"type": "text", "text": user_content}]
    for item in image_attachments or []:
        try:
            path = Path(str(item.get("path") or "")).expanduser()
            if (not path.exists()) or (not path.is_file()):
                continue
            data = path.read_bytes()
            if (not data) or (len(data) > 8 * 1024 * 1024):
                continue
            mime = str(item.get("mime") or "").strip().lower() or vision_image_mime_by_suffix.get(path.suffix.lower(), "image/png")
            b64 = base64.b64encode(data).decode("ascii")
            mm_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        except Exception:
            continue
    if len(mm_parts) > 1:
        return mm_parts
    return user_content


def _build_generation_messages(*, system: str, hist: list[dict], user_content: str | list[dict]) -> list[dict]:
    return [{"role": "system", "content": system}, *(hist or []), {"role": "user", "content": user_content}]
