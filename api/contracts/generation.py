from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except Exception:  # pragma: no cover - pydantic v1 compatibility
    ConfigDict = None


GENERATION_STREAM_SCHEMA_VERSION: Literal[2] = 2
GENERATION_STREAM_EVENT_FIELDS = (
    "stream_schema_version",
    "stage",
    "partial",
    "char_count",
    "done",
    "status",
    "answer",
    "error",
    "answer_intent",
    "answer_depth",
    "answer_output_mode",
    "answer_contract_v1",
    "answer_quality",
    "paper_guide_debug",
    "research_trace",
    "agent_trace",
    "agent_source_summary",
    "answer_runtime_check",
    "answer_contract",
)


class _GenerationContractModel(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:  # pragma: no cover - pydantic v1 compatibility
        class Config:
            extra = "forbid"


def _text(value: Any) -> str:
    return str(value or "")


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _record(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


class GenerationStreamEvent(_GenerationContractModel):
    stream_schema_version: Literal[2] = GENERATION_STREAM_SCHEMA_VERSION
    stage: str = ""
    partial: str = ""
    char_count: int = 0
    done: bool = False
    status: str = ""
    answer: str = ""
    error: str = ""
    answer_intent: str = ""
    answer_depth: str = ""
    answer_output_mode: str = ""
    answer_contract_v1: bool = False
    answer_quality: dict[str, Any] = Field(default_factory=dict)
    paper_guide_debug: dict[str, Any] = Field(default_factory=dict)
    research_trace: dict[str, Any] = Field(default_factory=dict)
    agent_trace: dict[str, Any] = Field(default_factory=dict)
    agent_source_summary: dict[str, Any] = Field(default_factory=dict)
    answer_runtime_check: dict[str, Any] = Field(default_factory=dict)
    answer_contract: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def missing_task(cls, failure_message: str) -> GenerationStreamEvent:
        message = _text(failure_message)
        return cls(
            stage="error",
            partial=message,
            char_count=len(message),
            done=True,
            status="error",
            answer=message,
            error="not_found",
        )

    @classmethod
    def from_task(
        cls,
        task: dict[str, Any],
        *,
        partial: str,
        answer: str,
        visible_text: str,
        include_internal_debug: bool,
    ) -> GenerationStreamEvent:
        status = _text(task.get("status"))
        agent_mode = bool(task.get("agent_mode"))
        visible = _text(visible_text)
        return cls(
            stage=_text(task.get("stage")),
            partial=_text(partial),
            char_count=len(visible) if visible else _int(task.get("char_count")),
            done=status in {"done", "error", "canceled"},
            status=status,
            answer=_text(answer),
            error=_text(task.get("error")),
            answer_intent=_text(task.get("answer_intent")),
            answer_depth=_text(task.get("answer_depth")),
            answer_output_mode=_text(task.get("answer_output_mode")),
            answer_contract_v1=bool(task.get("answer_contract_v1", False)),
            answer_quality=_record(task.get("answer_quality")) if include_internal_debug else {},
            paper_guide_debug=_record(task.get("paper_guide_debug")) if include_internal_debug else {},
            research_trace=_record(task.get("research_trace")) if include_internal_debug else {},
            agent_trace=_record(task.get("agent_trace")) if agent_mode else {},
            agent_source_summary=_record(task.get("agent_source_summary")) if agent_mode else {},
            answer_runtime_check=_record(task.get("answer_runtime_check")) if agent_mode else {},
            answer_contract=_record(task.get("answer_contract")) if agent_mode else {},
        )

    def as_dict(self) -> dict[str, Any]:
        try:
            return dict(self.model_dump(mode="python"))
        except Exception:  # pragma: no cover - pydantic v1 compatibility
            return dict(self.dict())
