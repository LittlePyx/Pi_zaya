from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


# Offset added to snippet numbers in classic RAG context so that hit citations
# (System A: [10001], [10002], ...) and in-paper bibliography references
# (System B: [1], [2], ...) use disjoint numeric ranges.  No post-processing
# heuristic needed — the number itself tells you which system it belongs to.
CITATION_OFFSET = 10000


def _clean_env_key(raw: str) -> str | None:
    v = str(raw or "").strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1].strip()
    return v or None


def _load_runtime_prefs() -> dict:
    configured = str(os.environ.get("KB_USER_PREFS_PATH") or "").strip()
    prefs_path = Path(configured).expanduser() if configured else Path(__file__).resolve().parent.parent / "user_prefs.json"
    if not prefs_path.exists():
        return {}
    try:
        data = json.loads(prefs_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _clean_pref_str(value: object) -> str | None:
    raw = str(value or "").replace("\x00", "").strip()
    return raw or None


def _clean_base_url(value: object) -> str | None:
    raw = _clean_pref_str(value)
    return raw.rstrip("/") if raw else None


def _clean_pref_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return None


@dataclass(frozen=True)
class Settings:
    # Text model (cheaper, faster — e.g. DeepSeek).
    text_api_key: str | None
    text_base_url: str
    text_model: str
    # Vision model (multimodal — e.g. Qwen VL).  When not configured, text model is used for
    # everything (including image-bearing requests, which will likely fail or degrade).
    vision_api_key: str | None
    vision_base_url: str
    vision_model: str
    # Shared settings.
    db_dir: Path
    chat_db_path: Path
    library_db_path: Path
    timeout_s: float
    max_retries: int
    user_issues_db_path: Path = field(default_factory=lambda: Path("user_issues.sqlite3"))
    user_issues_remote_enabled: bool = field(default=False)
    user_issues_remote_url: str = field(default="")
    user_issues_remote_token: str | None = field(default=None, repr=False)
    user_issues_ingest_token: str | None = field(default=None, repr=False)
    # Whether auto-routing is active (both text *and* vision keys are set).
    auto_route: bool = field(default=False)
    vision_uses_text_fallback: bool = field(default=False)
    # Whether LLM-based query expansion is enabled for BM25 retrieval.
    query_expansion_enabled: bool = field(default=False)
    # Runtime environment and API access protection.
    app_env: str = field(default="development")
    production: bool = field(default=False)
    access_token: str | None = field(default=None, repr=False)
    access_token_sha256: str | None = field(default=None, repr=False)
    auth_required: bool = field(default=False)
    auth_cookie_secure: bool = field(default=False)
    # Management operations can remain protected even when the user-facing
    # chat surface is intentionally public.
    management_auth_required: bool = field(default=False)
    management_access_token: str | None = field(default=None, repr=False)
    management_access_token_sha256: str | None = field(default=None, repr=False)
    auto_backup_enabled: bool | None = field(default=None)
    max_pdf_upload_bytes: int = field(default=80 * 1024 * 1024)
    max_image_upload_bytes: int = field(default=8 * 1024 * 1024)
    max_chat_upload_files: int = field(default=12)
    agent_web_search_enabled: bool = field(default=False)
    agent_web_search_api_key: str | None = field(default=None, repr=False)
    agent_web_search_base_url: str = field(default="https://api.openai.com/v1")
    agent_web_search_model: str = field(default="gpt-5-search-api")
    agent_web_search_context_size: str = field(default="low")

    # ------------------------------------------------------------------
    # Backward-compatible accessors (so existing callers that read
    # .api_key / .base_url / .model still work).
    # ------------------------------------------------------------------
    @property
    def api_key(self) -> str | None:
        return self.text_api_key

    @property
    def base_url(self) -> str:
        return self.text_base_url

    @property
    def model(self) -> str:
        return self.text_model


def load_settings() -> Settings:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

    _env = os.environ.get
    prefs = _load_runtime_prefs()
    stored_text_api_key = _clean_env_key(str(prefs.get("text_api_key") or ""))
    stored_text_base_url = _clean_base_url(prefs.get("text_base_url"))
    stored_text_model = _clean_pref_str(prefs.get("text_model"))
    stored_vision_api_key = _clean_env_key(str(prefs.get("vision_api_key") or ""))
    stored_vision_base_url = _clean_base_url(prefs.get("vision_base_url"))
    stored_vision_model = _clean_pref_str(prefs.get("vision_model"))
    stored_auto_backup_enabled = _clean_pref_bool(prefs.get("auto_backup_enabled"))

    # --- text model ---------------------------------------------------
    # Prefer DeepSeek (cheapest / fastest for text).  Fall back to Qwen,
    # then OpenAI.
    env_text_api_key = _clean_env_key(
        _env("DEEPSEEK_API_KEY") or _env("QWEN_API_KEY") or _env("OPENAI_API_KEY") or ""
    )
    text_api_key = env_text_api_key or stored_text_api_key
    if _env("DEEPSEEK_API_KEY"):
        text_base_url = (
            _env("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"
        ).strip().rstrip("/")
        if "api.deepseek.com" in text_base_url and not text_base_url.endswith("/v1"):
            text_base_url = text_base_url + "/v1"
        raw_model = (
            _env("DEEPSEEK_MODEL") or _env("OPENAI_MODEL") or "deepseek-chat"
        ).strip()
        # Auto-upgrade old deprecated model IDs to the current series.
        if raw_model in ("deepseek-reasoner",):
            raw_model = "deepseek-chat"
        text_model = raw_model
    elif _env("QWEN_API_KEY"):
        text_base_url = (
            _env("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).strip().rstrip("/")
        text_model = (
            _env("QWEN_MODEL") or _env("OPENAI_MODEL") or "qwen3-vl-plus"
        ).strip()
    elif stored_text_api_key:
        text_base_url = (
            stored_text_base_url or _env("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).strip().rstrip("/")
        text_model = (stored_text_model or _env("OPENAI_MODEL") or "gpt-4o").strip()
    else:
        text_base_url = (
            stored_text_base_url or _env("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).strip().rstrip("/")
        text_model = (stored_text_model or _env("OPENAI_MODEL") or "gpt-4o").strip()

    # --- vision model -------------------------------------------------
    # Qwen VL is the primary vision model.  DeepSeek does not support
    # image inputs through its API at time of writing.
    env_vision_api_key = _clean_env_key(_env("QWEN_API_KEY") or "")
    dedicated_vision_api_key = env_vision_api_key or stored_vision_api_key
    vision_api_key = dedicated_vision_api_key or text_api_key
    vision_uses_text_fallback = not bool(dedicated_vision_api_key)
    if _env("QWEN_API_KEY"):
        vision_base_url = (
            _env("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).strip().rstrip("/")
        vision_model = (
            _env("QWEN_MODEL") or _env("OPENAI_MODEL") or "qwen3-vl-plus"
        ).strip()
    elif stored_vision_api_key:
        vision_base_url = (
            stored_vision_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).strip().rstrip("/")
        vision_model = (stored_vision_model or "qwen3-vl-plus").strip()
    else:
        # No dedicated vision key — fall back to text model for everything.
        vision_base_url = text_base_url
        vision_model = text_model

    # --- shared -------------------------------------------------------
    here = Path(__file__).resolve().parent.parent
    db_dir = Path(_env("KB_DB_DIR", str(here / "db"))).expanduser().resolve()
    chat_db_path = Path(_env("KB_CHAT_DB", str(here / "chat.sqlite3"))).expanduser().resolve()
    library_db_path = Path(_env("KB_LIBRARY_DB", str(here / "library.sqlite3"))).expanduser().resolve()
    user_issues_db_path = Path(_env("KB_USER_ISSUES_DB", str(here / "user_issues.sqlite3"))).expanduser().resolve()
    user_issues_remote_enabled = str(_env("KB_USER_ISSUES_REMOTE_ENABLED", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    user_issues_remote_url = str(_env("KB_USER_ISSUES_REMOTE_URL", "") or "").strip()
    user_issues_remote_token = _clean_env_key(_env("KB_USER_ISSUES_REMOTE_TOKEN") or "")
    user_issues_ingest_token = _clean_env_key(_env("KB_USER_ISSUES_INGEST_TOKEN") or "")

    timeout_s = float(_env("KB_LLM_TIMEOUT_S", _env("DEEPSEEK_TIMEOUT_S", "60")))
    max_retries = int(_env("KB_LLM_MAX_RETRIES", _env("DEEPSEEK_MAX_RETRIES", "2")))

    # Auto-routing is active when both text and vision keys differ
    # (i.e. the operator intentionally set up two providers).
    auto_route = bool(
        text_api_key
        and vision_api_key
        and not vision_uses_text_fallback
        and (text_api_key != vision_api_key or text_base_url != vision_base_url)
    )
    query_expansion_enabled = _env("KB_QUERY_EXPANSION_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
    app_env = (_env("KB_APP_ENV") or _env("KB_ENV") or "development").strip().lower() or "development"
    production = app_env in {"prod", "production"}
    access_token = _clean_env_key(
        _env("KB_ACCESS_TOKEN") or _env("KB_API_TOKEN") or _env("KB_AUTH_TOKEN") or ""
    )
    access_token_sha256 = _clean_env_key(
        _env("KB_ACCESS_TOKEN_SHA256") or _env("KB_API_TOKEN_SHA256") or _env("KB_AUTH_TOKEN_SHA256") or ""
    )
    auth_gate_enabled = str(_env("KB_ENABLE_AUTH_GATE", "") or "").strip().lower() in {"1", "true", "yes", "on"}
    private_instance_auth = str(_env("KB_PRIVATE_INSTANCE_AUTH", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    auth_raw = _env("KB_REQUIRE_AUTH")
    if auth_raw is None or str(auth_raw).strip() == "":
        auth_requested = False
    else:
        auth_requested = str(auth_raw).strip().lower() in {"1", "true", "yes", "on"}
    local_auth_gate_allowed = str(_env("KB_ALLOW_LOCAL_AUTH_GATE", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    auth_required = private_instance_auth and auth_gate_enabled and auth_requested and (
        production or local_auth_gate_allowed
    )
    management_access_token = _clean_env_key(_env("KB_MANAGEMENT_ACCESS_TOKEN") or "")
    management_access_token_sha256 = _clean_env_key(_env("KB_MANAGEMENT_ACCESS_TOKEN_SHA256") or "")
    management_auth_raw = _env("KB_REQUIRE_MANAGEMENT_AUTH")
    if management_auth_raw is None or str(management_auth_raw).strip() == "":
        management_auth_required = production
    else:
        management_auth_required = str(management_auth_raw).strip().lower() in {"1", "true", "yes", "on"}
    cookie_secure_raw = _env("KB_AUTH_COOKIE_SECURE")
    if cookie_secure_raw is None or str(cookie_secure_raw).strip() == "":
        auth_cookie_secure = production
    else:
        auth_cookie_secure = str(cookie_secure_raw).strip().lower() in {"1", "true", "yes", "on"}
    try:
        max_pdf_upload_bytes = max(
            1,
            int(
                _env("KB_MAX_PDF_UPLOAD_BYTES")
                or int(float(_env("KB_MAX_PDF_UPLOAD_MB", "80")) * 1024 * 1024)
            ),
        )
    except Exception:
        max_pdf_upload_bytes = 80 * 1024 * 1024
    try:
        max_image_upload_bytes = max(
            1,
            int(
                _env("KB_MAX_IMAGE_UPLOAD_BYTES")
                or int(float(_env("KB_MAX_IMAGE_UPLOAD_MB", "8")) * 1024 * 1024)
            ),
        )
    except Exception:
        max_image_upload_bytes = 8 * 1024 * 1024
    try:
        max_chat_upload_files = max(1, int(_env("KB_MAX_CHAT_UPLOAD_FILES", "12")))
    except Exception:
        max_chat_upload_files = 12
    agent_web_search_api_key = _clean_env_key(
        _env("KB_AGENT_WEB_SEARCH_API_KEY") or _env("OPENAI_API_KEY") or ""
    )
    agent_web_search_base_url = (
        _env("KB_AGENT_WEB_SEARCH_BASE_URL")
        or _env("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    ).strip().rstrip("/")
    agent_web_search_model = (_env("KB_AGENT_WEB_SEARCH_MODEL") or "gpt-5-search-api").strip()
    raw_agent_web_search_enabled = str(_env("KB_AGENT_WEB_SEARCH_ENABLED", "") or "").strip().lower()
    if raw_agent_web_search_enabled:
        agent_web_search_enabled = raw_agent_web_search_enabled in {"1", "true", "yes", "on"}
    else:
        agent_web_search_enabled = bool(agent_web_search_api_key)
    raw_context_size = str(_env("KB_AGENT_WEB_SEARCH_CONTEXT_SIZE", "low") or "low").strip().lower()
    agent_web_search_context_size = raw_context_size if raw_context_size in {"low", "medium", "high"} else "low"

    return Settings(
        text_api_key=text_api_key,
        text_base_url=text_base_url,
        text_model=text_model,
        vision_api_key=vision_api_key,
        vision_base_url=vision_base_url,
        vision_model=vision_model,
        db_dir=db_dir,
        chat_db_path=chat_db_path,
        library_db_path=library_db_path,
        user_issues_db_path=user_issues_db_path,
        user_issues_remote_enabled=user_issues_remote_enabled,
        user_issues_remote_url=user_issues_remote_url,
        user_issues_remote_token=user_issues_remote_token,
        user_issues_ingest_token=user_issues_ingest_token,
        timeout_s=timeout_s,
        max_retries=max_retries,
        auto_route=auto_route,
        vision_uses_text_fallback=vision_uses_text_fallback,
        query_expansion_enabled=query_expansion_enabled,
        app_env=app_env,
        production=production,
        access_token=access_token,
        access_token_sha256=access_token_sha256,
        auth_required=auth_required,
        auth_cookie_secure=auth_cookie_secure,
        management_auth_required=management_auth_required,
        management_access_token=management_access_token,
        management_access_token_sha256=management_access_token_sha256,
        auto_backup_enabled=stored_auto_backup_enabled,
        max_pdf_upload_bytes=max_pdf_upload_bytes,
        max_image_upload_bytes=max_image_upload_bytes,
        max_chat_upload_files=max_chat_upload_files,
        agent_web_search_enabled=agent_web_search_enabled,
        agent_web_search_api_key=agent_web_search_api_key,
        agent_web_search_base_url=agent_web_search_base_url,
        agent_web_search_model=agent_web_search_model,
        agent_web_search_context_size=agent_web_search_context_size,
    )
