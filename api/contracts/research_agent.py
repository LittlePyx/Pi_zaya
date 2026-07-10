from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


_MAX_MESSAGE_CHARS = 80_000
_MAX_SOURCE_PATH_CHARS = 1_200
_MAX_SOURCE_NAME_CHARS = 500
_MAX_QUERY_SCOPE_CHARS = 40
_MAX_PROMPT_CONTEXT_JSON_CHARS = 260_000


def _bounded_prompt_context(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    except Exception as exc:
        raise ValueError("prompt context must be JSON serializable") from exc
    if len(encoded) > _MAX_PROMPT_CONTEXT_JSON_CHARS:
        raise ValueError(f"prompt context is too large; max {_MAX_PROMPT_CONTEXT_JSON_CHARS} JSON chars")
    return value


class ResearchAgentRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str = Field("", max_length=_MAX_MESSAGE_CHARS)
    query: str = Field("", max_length=_MAX_MESSAGE_CHARS)
    top_k: int = Field(6, ge=1, le=20)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(1200, ge=1, le=8192)
    query_scope: str = Field("", max_length=_MAX_QUERY_SCOPE_CHARS)
    prompt_context: dict[str, Any] | None = None
    source_lock_path: str = Field("", max_length=_MAX_SOURCE_PATH_CHARS)
    source_lock_name: str = Field("", max_length=_MAX_SOURCE_NAME_CHARS)

    @field_validator("prompt_context")
    @classmethod
    def _check_prompt_context(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        return _bounded_prompt_context(value)


class ResearchAgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    agent_trace: dict[str, Any] = Field(default_factory=dict)
    hits: list[dict[str, Any]] = Field(default_factory=list)
