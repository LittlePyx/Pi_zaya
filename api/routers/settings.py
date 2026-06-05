from __future__ import annotations

import hashlib
import time
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
_LLM_TEST_RESULTS: dict[str, dict] = {}


def _classify_connection_error(error: object) -> str:
    text = str(error or "").lower()
    if "api key" in text or "authentication" in text or "unauthorized" in text or "401" in text:
        return "auth"
    if "forbidden" in text or "403" in text:
        return "permission"
    if "model" in text and ("not found" in text or "does not exist" in text or "invalid" in text):
        return "model"
    if "base_url" in text or "invalid url" in text or "unsupported protocol" in text:
        return "base_url"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "connection" in text or "network" in text or "name resolution" in text or "connect" in text:
        return "network"
    return "unknown"


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


def _provider_fingerprint(*, api_key: str | None, base_url: str, model: str) -> str:
    raw = "|".join([str(api_key or ""), str(base_url or ""), str(model or "")])
    return hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest()[:16]


def _provider_readiness(s, target: Literal["text", "vision"]) -> dict:
    if target == "vision":
        has_key = bool(s.vision_api_key)
        base_url = s.vision_base_url
        model = s.vision_model
        uses_text_fallback = bool(getattr(s, "vision_uses_text_fallback", False))
        fingerprint = _provider_fingerprint(api_key=s.vision_api_key, base_url=base_url, model=model)
    else:
        has_key = bool(s.text_api_key)
        base_url = s.text_base_url
        model = s.text_model
        uses_text_fallback = False
        fingerprint = _provider_fingerprint(api_key=s.text_api_key, base_url=base_url, model=model)

    last_test = _LLM_TEST_RESULTS.get(target)
    if last_test and last_test.get("fingerprint") != fingerprint:
        last_test = None

    if not has_key:
        status = "missing"
        severity = "error"
        reason = "missing_api_key"
    elif uses_text_fallback:
        status = "fallback"
        severity = "warning"
        reason = "vision_uses_text_fallback"
    elif last_test and bool(last_test.get("ok")):
        status = "ok"
        severity = "ok"
        reason = "last_test_ok"
    elif last_test:
        status = "failed"
        severity = "error"
        reason = str(last_test.get("error_type") or "unknown")
    else:
        status = "configured"
        severity = "ok"
        reason = "configured_not_tested"

    public_last_test = None
    if last_test:
        public_last_test = {
            "ok": bool(last_test.get("ok")),
            "checked_at": float(last_test.get("checked_at") or 0.0),
            "error": str(last_test.get("error") or ""),
            "error_type": str(last_test.get("error_type") or ""),
            "reply": str(last_test.get("reply") or ""),
        }

    return {
        "target": target,
        "has_api_key": has_key,
        "base_url": base_url,
        "model": model,
        "uses_text_fallback": uses_text_fallback,
        "status": status,
        "severity": severity,
        "reason": reason,
        "last_test": public_last_test,
    }


def _readiness_payload(s) -> dict:
    text = _provider_readiness(s, "text")
    vision = _provider_readiness(s, "vision")
    if text["severity"] == "error":
        overall = {"status": "error", "reason": text["reason"], "target": "text"}
    elif vision["severity"] == "error":
        overall = {"status": "error", "reason": vision["reason"], "target": "vision"}
    elif vision["severity"] == "warning":
        overall = {"status": "warning", "reason": vision["reason"], "target": "vision"}
    else:
        overall = {"status": "ok", "reason": "ready", "target": ""}
    return {"providers": {"text": text, "vision": vision}, "overall": overall}


@router.get("/settings")
def get_all_settings():
    s = get_settings()
    prefs = load_prefs()
    return {
        "model": s.model,
        "base_url": s.base_url,
        "has_api_key": bool(s.api_key),
        "connection": _connection_status(s),
        "readiness": _readiness_payload(s),
        "db_dir": str(s.db_dir),
        "prefs": _public_prefs(prefs),
    }


@router.get("/settings/readiness")
def get_llm_readiness():
    return _readiness_payload(get_settings())


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
        return {"ok": False, "error": "API key is missing", "error_type": "auth"}
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
    checked_at = time.time()
    target = body.target if body else "text"
    api_key: str | None = None
    base_url = ""
    model = ""
    try:
        s = get_settings()
        override_api_key = _normalize_pref_value("text_api_key", body.api_key) if body and body.api_key is not None else ""
        override_base_url = _normalize_pref_value("text_base_url", body.base_url) if body and body.base_url is not None else ""
        override_model = _normalize_pref_value("text_model", body.model) if body and body.model is not None else ""
        if target == "vision":
            api_key = override_api_key or s.vision_api_key
            base_url = override_base_url or s.vision_base_url
            model = override_model or s.vision_model
            result = _test_chat_completion(
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout_s=s.timeout_s,
            )
        else:
            api_key = override_api_key or s.text_api_key
            base_url = override_base_url or s.text_base_url
            model = override_model or s.text_model
            result = _test_chat_completion(
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout_s=s.timeout_s,
            )
        error_type = str(result.get("error_type") or "")
        if not result.get("ok") and not error_type:
            error_type = _classify_connection_error(result.get("error"))
            result["error_type"] = error_type
        result["checked_at"] = checked_at
        _LLM_TEST_RESULTS[target] = {
            "ok": bool(result.get("ok")),
            "reply": str(result.get("reply") or ""),
            "error": str(result.get("error") or ""),
            "error_type": error_type,
            "checked_at": checked_at,
            "fingerprint": _provider_fingerprint(api_key=api_key, base_url=base_url, model=model),
        }
        return result
    except Exception as e:
        error_type = _classify_connection_error(e)
        _LLM_TEST_RESULTS[target] = {
            "ok": False,
            "reply": "",
            "error": str(e),
            "error_type": error_type,
            "checked_at": checked_at,
            "fingerprint": _provider_fingerprint(api_key=api_key, base_url=base_url, model=model),
        }
        return {"ok": False, "error": str(e), "error_type": error_type, "checked_at": checked_at}


@router.get("/health")
def health():
    return {"status": "ok"}
