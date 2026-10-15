from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final


LOCAL_SEMANTIC_MODE: Final = "hybrid_local"
LOCAL_SEMANTIC_STATES: Final = (
    "unavailable_in_build",
    "not_prepared",
    "downloading",
    "verifying",
    "ready_to_index",
    "indexing",
    "cancelling",
    "interrupted",
    "stale",
    "repairable_error",
    "ready",
)
_ACTION_BY_STATE: Final = {
    "unavailable_in_build": "update_app",
    "not_prepared": "prepare",
    "downloading": "cancel",
    "verifying": "cancel",
    "ready_to_index": "resume",
    "indexing": "cancel",
    "cancelling": "wait",
    "interrupted": "resume",
    "stale": "rebuild",
    "repairable_error": "retry",
    "ready": "none",
}
_ACTIVE_STATES: Final = {"downloading", "verifying", "indexing", "cancelling"}
_RESTART_STATE: Final = {
    "downloading": "interrupted",
    "verifying": "interrupted",
    "indexing": "interrupted",
    "cancelling": "interrupted",
}


def normalize_preparation_state(value: object) -> str:
    state = str(value or "").strip().lower()
    return state if state in LOCAL_SEMANTIC_STATES else "repairable_error"


def preparation_action(state: object) -> str:
    return _ACTION_BY_STATE[normalize_preparation_state(state)]


def reconcile_preparation_state_after_restart(state: object) -> str:
    normalized = normalize_preparation_state(state)
    return _RESTART_STATE.get(normalized, normalized)


def preparation_is_active(state: object) -> bool:
    return normalize_preparation_state(state) in _ACTIVE_STATES


@dataclass(frozen=True)
class LocalSemanticPreparationStatus:
    state: str
    reason_id: str = ""
    task_id: str = ""
    phase: str = ""
    completed_units: int = 0
    total_units: int = 0
    completed_bytes: int = 0
    total_bytes: int = 0
    eta_seconds: int | None = None
    document_count: int = 0
    chunk_count: int = 0

    def __post_init__(self) -> None:
        normalized = normalize_preparation_state(self.state)
        if normalized != self.state:
            raise ValueError("local semantic preparation state is invalid")
        for field_name in (
            "completed_units",
            "total_units",
            "completed_bytes",
            "total_bytes",
            "document_count",
            "chunk_count",
        ):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if self.completed_units > self.total_units and self.total_units > 0:
            raise ValueError("completed_units cannot exceed total_units")
        if self.completed_bytes > self.total_bytes and self.total_bytes > 0:
            raise ValueError("completed_bytes cannot exceed total_bytes")
        if self.eta_seconds is not None and int(self.eta_seconds) < 0:
            raise ValueError("eta_seconds cannot be negative")

    @property
    def available(self) -> bool:
        return self.state == "ready"

    @property
    def action_id(self) -> str:
        return preparation_action(self.state)

    def public_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "available": self.available,
                "mode": LOCAL_SEMANTIC_MODE,
                "action_id": self.action_id,
            }
        )
        return payload
