from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.deps import get_settings, load_prefs, save_prefs
from kb.llm import DeepSeekChat

router = APIRouter(prefix="/api", tags=["settings"])


@router.get("/settings")
def get_all_settings():
    s = get_settings()
    prefs = load_prefs()
    return {
        "model": s.model,
        "base_url": s.base_url,
        "has_api_key": bool(s.api_key),
        "db_dir": str(s.db_dir),
        "prefs": prefs,
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


@router.patch("/settings")
def update_settings(body: PrefsPatch):
    prefs = load_prefs()
    for k, v in body.model_dump(exclude_none=True).items():
        prefs[k] = v
    save_prefs(prefs)
    return {"ok": True}


@router.post("/settings/test-llm")
def test_llm():
    try:
        s = get_settings()
        ds = DeepSeekChat(s)
        reply = ds.chat(
            [{"role": "user", "content": "Hi, reply OK in one word."}],
            temperature=0.0, max_tokens=16,
        )
        return {"ok": True, "reply": reply}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/health")
def health():
    return {"status": "ok"}
