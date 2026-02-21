from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str
    model: str
    db_dir: Path
    chat_db_path: Path
    library_db_path: Path
    timeout_s: float
    max_retries: int


def load_settings() -> Settings:
    # Prefer Qwen (vision-capable) when configured; fall back to DeepSeek/OpenAI.
    api_key = (
        os.environ.get("QWEN_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()
    # Users often set env vars with quotes (e.g. cmd.exe: set DEEPSEEK_API_KEY="sk-...").
    if (api_key.startswith('"') and api_key.endswith('"')) or (api_key.startswith("'") and api_key.endswith("'")):
        api_key = api_key[1:-1].strip()
    api_key = api_key or None

    # Base URL: prefer Qwen OpenAI-compatible endpoint when QWEN_API_KEY is set.
    if os.environ.get("QWEN_API_KEY"):
        base_url = (os.environ.get("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1").strip().rstrip("/")
        model = (os.environ.get("QWEN_MODEL") or os.environ.get("OPENAI_MODEL") or "qwen3-vl-plus").strip()
    else:
        base_url = (os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.deepseek.com/v1").strip().rstrip("/")
        # Be forgiving: many people set DEEPSEEK_BASE_URL=https://api.deepseek.com
        # but the OpenAI-compatible endpoint is under /v1.
        if "api.deepseek.com" in base_url and not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
        model = (os.environ.get("DEEPSEEK_MODEL") or os.environ.get("OPENAI_MODEL") or "deepseek-chat").strip()

    here = Path(__file__).resolve().parent.parent
    db_dir = Path(os.environ.get("KB_DB_DIR", str(here / "db"))).expanduser().resolve()
    chat_db_path = Path(os.environ.get("KB_CHAT_DB", str(here / "chat.sqlite3"))).expanduser().resolve()
    library_db_path = Path(os.environ.get("KB_LIBRARY_DB", str(here / "library.sqlite3"))).expanduser().resolve()

    timeout_s = float(os.environ.get("KB_LLM_TIMEOUT_S", os.environ.get("DEEPSEEK_TIMEOUT_S", "60")))
    max_retries = int(os.environ.get("KB_LLM_MAX_RETRIES", os.environ.get("DEEPSEEK_MAX_RETRIES", "2")))

    return Settings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        db_dir=db_dir,
        chat_db_path=chat_db_path,
        library_db_path=library_db_path,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )
