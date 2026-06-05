from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pydantic import BaseModel

from api.deps import get_settings, load_prefs, save_prefs
from kb.file_ops import _pick_directory_dialog

router = APIRouter(prefix="/api", tags=["settings"])
_PATH_PREF_KEYS = {"pdf_dir", "md_dir"}
_API_KEY_PREF_KEYS = {"text_api_key", "vision_api_key"}
_LLM_PREF_KEYS = {"text_base_url", "text_model", "vision_base_url", "vision_model"}


def _normalize_pref_value(key: str, value):
    if key in _API_KEY_PREF_KEYS:
        raw = str(value or "").replace("\x00", "").strip()
        if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
            raw = raw[1:-1].strip()
        return raw
    if key in _LLM_PREF_KEYS:
        raw = str(value or "").replace("\x00", "").strip()
        if key.endswith("_base_url"):
            raw = raw.rstrip("/")
        return raw
    if key not in _PATH_PREF_KEYS:
        return value
    raw = str(value or "").strip()
    if not raw:
        raise HTTPException(400, f"{key} cannot be empty")
    try:
        path = Path(raw).expanduser().resolve(strict=False)
    except Exception as exc:
        raise HTTPException(400, f"invalid {key}: {exc}") from exc
    if path.exists() and not path.is_dir():
        raise HTTPException(400, f"{key} must be a directory")
    return str(path)


def _connection_status(s) -> dict:
    return {
        "text": {
            "has_api_key": bool(s.text_api_key),
            "base_url": s.text_base_url,
            "model": s.text_model,
        },
        "vision": {
            "has_api_key": bool(s.vision_api_key),
            "base_url": s.vision_base_url,
            "model": s.vision_model,
            "uses_text_fallback": bool(getattr(s, "vision_uses_text_fallback", False)),
        },
        "auto_route": bool(s.auto_route),
    }


def _public_prefs(prefs: dict) -> dict:
    return {k: v for k, v in prefs.items() if k not in _API_KEY_PREF_KEYS}


@router.get("/settings")
def get_all_settings():
    s = get_settings()
    prefs = load_prefs()
    return {
        "model": s.model,
        "base_url": s.base_url,
        "has_api_key": bool(s.api_key),
        "connection": _connection_status(s),
        "db_dir": str(s.db_dir),
        "prefs": _public_prefs(prefs),
    }


class PrefsPatch(BaseModel):
    top_k: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    deep_read: bool | None = None
    show_context: bool | None = None
    theme: str | None = None
    pdf_dir: str | None = None
    md_dir: str | None = None
    answer_contract_v1: bool | None = None
    answer_depth_auto: bool | None = None
    answer_mode_hint: str | None = None
    answer_output_mode: str | None = None
    refs_card_locale: str | None = None
    ui_locale: str | None = None
    sidebar_collapsed: bool | None = None
    text_api_key: str | None = None
    text_base_url: str | None = None
    text_model: str | None = None
    vision_api_key: str | None = None
    vision_base_url: str | None = None
    vision_model: str | None = None


@router.patch("/settings")
def update_settings(body: PrefsPatch):
    prefs = load_prefs()
    for k, v in body.model_dump(exclude_none=True).items():
        normalized = _normalize_pref_value(k, v)
        if k in (_API_KEY_PREF_KEYS | _LLM_PREF_KEYS) and not normalized:
            prefs.pop(k, None)
        else:
            prefs[k] = normalized
    save_prefs(prefs)
    try:
        get_settings.cache_clear()
    except Exception:
        pass
    return {"ok": True}


class PickDirRequest(BaseModel):
    target: Literal["pdf", "md"]
    initial_dir: str | None = None


@router.post("/settings/pick-dir")
def pick_dir(body: PickDirRequest):
    prefs = load_prefs()
    key = "pdf_dir" if body.target == "pdf" else "md_dir"
    initial = (body.initial_dir or "").strip() or str(prefs.get(key) or "").strip()
    picked = _pick_directory_dialog(initial)
    if not picked:
        return {"ok": False, "path": None}
    return {"ok": True, "path": picked}


class ConnectionTestBody(BaseModel):
    target: Literal["text", "vision"] = "text"
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None


def _test_chat_completion(*, api_key: str | None, base_url: str, model: str, timeout_s: float) -> dict:
    if not api_key:
        return {"ok": False, "error": "API key is missing"}
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Hi, reply OK in one word."}],
        temperature=0.0,
        max_tokens=16,
        timeout=timeout_s,
    )
    reply = (resp.choices[0].message.content or "").strip()
    return {"ok": True, "reply": reply}


@router.post("/settings/test-llm")
def test_llm(body: ConnectionTestBody | None = None):
    try:
        s = get_settings()
        target = (body.target if body else "text")
        override_api_key = _normalize_pref_value("text_api_key", body.api_key) if body and body.api_key is not None else ""
        override_base_url = _normalize_pref_value("text_base_url", body.base_url) if body and body.base_url is not None else ""
        override_model = _normalize_pref_value("text_model", body.model) if body and body.model is not None else ""
        if target == "vision":
            return _test_chat_completion(
                api_key=override_api_key or s.vision_api_key,
                base_url=override_base_url or s.vision_base_url,
                model=override_model or s.vision_model,
                timeout_s=s.timeout_s,
            )
        return _test_chat_completion(
            api_key=override_api_key or s.text_api_key,
            base_url=override_base_url or s.text_base_url,
            model=override_model or s.text_model,
            timeout_s=s.timeout_s,
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/health")
def health():
    return {"status": "ok"}
