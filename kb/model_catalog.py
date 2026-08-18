from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlsplit


_PROVIDERS: dict[str, dict[str, object]] = {
    "qwen": {
        "label": "Qwen / 通义千问",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "targets": ("text", "vision"),
        "models": {
            "text": (
                "qwen3.7-plus-2026-05-26",
                "qwen3-max",
                "qwen-plus",
                "qwen-turbo",
            ),
            "vision": (
                "qwen3-vl-plus",
                "qwen-vl-max",
                "qwen-vl-plus",
            ),
        },
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "targets": ("text",),
        "models": {
            "text": (
                "deepseek-v4-flash",
                "deepseek-v4-pro",
            ),
            "vision": (),
        },
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "targets": ("text", "vision"),
        "models": {
            "text": (
                "gpt-5",
                "gpt-5-mini",
                "gpt-4.1",
                "gpt-4.1-mini",
                "gpt-4o",
            ),
            "vision": (
                "gpt-5",
                "gpt-4.1",
                "gpt-4o",
            ),
        },
    },
}

_NON_CHAT_MARKERS = (
    "embedding",
    "rerank",
    "whisper",
    "transcribe",
    "moderation",
    "tts",
    "dall-e",
    "image-generation",
)
_VISION_MARKERS = ("vision", "-vl", "vl-", "omni", "gpt-4o", "gpt-4.1", "gpt-5")


def provider_catalog(*, target: str | None = None) -> list[dict[str, object]]:
    requested = str(target or "").strip().lower()
    items: list[dict[str, object]] = []
    for provider_id, spec in _PROVIDERS.items():
        targets = tuple(str(item) for item in spec["targets"])
        if requested and requested not in targets:
            continue
        items.append(
            {
                "id": provider_id,
                "label": str(spec["label"]),
                "base_url": str(spec["base_url"]),
                "targets": list(targets),
            }
        )
    items.append(
        {
            "id": "custom",
            "label": "OpenAI-compatible / 自定义",
            "base_url": "",
            "targets": ["text", "vision"],
        }
    )
    return items


def provider_base_url(provider: str) -> str:
    spec = _PROVIDERS.get(str(provider or "").strip().lower())
    return str(spec["base_url"]) if spec else ""


def provider_supports_target(provider: str, target: str) -> bool:
    clean_provider = str(provider or "").strip().lower()
    clean_target = str(target or "").strip().lower()
    if clean_provider == "custom":
        return clean_target in {"text", "vision"}
    spec = _PROVIDERS.get(clean_provider)
    return bool(spec and clean_target in spec["targets"])


def infer_provider(*, api_key: str | None = None, base_url: str = "") -> tuple[str, str]:
    raw_url = str(base_url or "").strip()
    if raw_url:
        try:
            host = (urlsplit(raw_url).hostname or "").lower()
        except ValueError:
            host = ""
        if host == "api.openai.com" or host.endswith(".openai.com"):
            return "openai", "base_url"
        if host == "api.deepseek.com" or host.endswith(".deepseek.com"):
            return "deepseek", "base_url"
        if host == "dashscope.aliyuncs.com" or host.endswith(".dashscope.aliyuncs.com"):
            return "qwen", "base_url"
        return "custom", "base_url"

    key = str(api_key or "").strip()
    # Only prefixes that identify one provider with high confidence belong
    # here. Generic ``sk-`` credentials are shared by many providers and must
    # never be tried against several third-party endpoints.
    if key.startswith(("sk-proj-", "sk-svcacct-")):
        return "openai", "key_prefix"
    return "", "ambiguous"


def fallback_models(provider: str, target: str) -> list[dict[str, object]]:
    clean_provider = str(provider or "").strip().lower()
    clean_target = str(target or "").strip().lower()
    spec = _PROVIDERS.get(clean_provider)
    if not spec:
        return []
    models = dict(spec["models"]).get(clean_target, ())
    return [
        {
            "id": str(model_id),
            "label": str(model_id),
            "capabilities": [clean_target],
            "source": "catalog",
        }
        for model_id in models
    ]


def filter_discovered_models(model_ids: Iterable[object], *, target: str) -> list[dict[str, object]]:
    clean_target = str(target or "").strip().lower()
    seen: set[str] = set()
    items: list[dict[str, object]] = []
    for raw in model_ids:
        model_id = str(raw or "").replace("\x00", "").strip()
        low = model_id.lower()
        if not model_id or len(model_id) > 200 or low in seen:
            continue
        if any(marker in low for marker in _NON_CHAT_MARKERS):
            continue
        if clean_target == "vision" and not any(marker in low for marker in _VISION_MARKERS):
            continue
        seen.add(low)
        capabilities = ["text"]
        if any(marker in low for marker in _VISION_MARKERS):
            capabilities.append("vision")
        items.append(
            {
                "id": model_id,
                "label": model_id,
                "capabilities": capabilities,
                "source": "provider",
            }
        )
    return sorted(items, key=lambda item: str(item["id"]).lower())[:200]


def recommended_model(models: Iterable[dict[str, object]], *, provider: str, target: str) -> str:
    available = [str(item.get("id") or "").strip() for item in models]
    available_set = {item.lower() for item in available if item}
    for item in fallback_models(provider, target):
        candidate = str(item.get("id") or "").strip()
        if candidate.lower() in available_set:
            return candidate
    return available[0] if available else ""
