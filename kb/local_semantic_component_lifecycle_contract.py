from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Final


COMPONENT_LIFECYCLE_STATES: Final = (
    "not_installed",
    "checking_capacity",
    "downloading",
    "verifying_archive",
    "installing",
    "health_checking",
    "ready_to_index",
    "indexing",
    "ready",
    "cancelling",
    "paused_by_user",
    "recovering",
    "needs_space",
    "degraded_bm25",
    "removing",
    "unavailable_in_build",
)
COMPONENT_ACTIVE_STATES: Final = frozenset(
    {
        "checking_capacity",
        "downloading",
        "verifying_archive",
        "installing",
        "health_checking",
        "ready_to_index",
        "indexing",
        "cancelling",
        "recovering",
        "removing",
    }
)
_PUBLIC_STATE_BY_INTERNAL: Final = {
    "not_installed": "not_installed",
    "checking_capacity": "preparing",
    "downloading": "preparing",
    "verifying_archive": "preparing",
    "installing": "preparing",
    "health_checking": "preparing",
    "ready_to_index": "preparing",
    "indexing": "preparing",
    "ready": "ready",
    "cancelling": "preparing",
    "paused_by_user": "paused",
    "recovering": "preparing",
    "needs_space": "needs_space",
    "degraded_bm25": "bm25_available",
    "removing": "removing",
    "unavailable_in_build": "unavailable",
}
_ACTION_BY_INTERNAL: Final = {
    "not_installed": "enable",
    "checking_capacity": "cancel",
    "downloading": "cancel",
    "verifying_archive": "cancel",
    "installing": "cancel",
    "health_checking": "cancel",
    "ready_to_index": "cancel",
    "indexing": "cancel",
    "ready": "remove",
    "cancelling": "none",
    "paused_by_user": "resume",
    "recovering": "cancel",
    "needs_space": "free_space",
    "degraded_bm25": "none",
    "removing": "none",
    "unavailable_in_build": "update_app",
}
_PUBLIC_REASON_BY_INTERNAL: Final = {
    "not_installed": "semantic_component_not_installed",
    "checking_capacity": "semantic_component_preparing",
    "downloading": "semantic_component_preparing",
    "verifying_archive": "semantic_component_preparing",
    "installing": "semantic_component_preparing",
    "health_checking": "semantic_component_preparing",
    "ready_to_index": "semantic_index_preparing",
    "indexing": "semantic_index_preparing",
    "ready": "semantic_component_ready",
    "cancelling": "semantic_component_pausing",
    "paused_by_user": "semantic_component_paused",
    "recovering": "semantic_component_recovering",
    "needs_space": "semantic_component_needs_space",
    "degraded_bm25": "semantic_component_temporarily_unavailable",
    "removing": "semantic_component_removing",
    "unavailable_in_build": "semantic_component_requires_app_update",
}
_REASON_RE = re.compile(r"^[a-z0-9_]{0,80}$")


def normalize_component_lifecycle_state(value: object) -> str:
    state = str(value or "").strip().lower()
    return state if state in COMPONENT_LIFECYCLE_STATES else "degraded_bm25"


def component_lifecycle_is_active(value: object) -> bool:
    return normalize_component_lifecycle_state(value) in COMPONENT_ACTIVE_STATES


def normalize_internal_reason(value: object) -> str:
    reason = str(value or "").strip().lower()
    return reason if _REASON_RE.fullmatch(reason) else "component_internal_failure"


@dataclass(frozen=True)
class LocalSemanticComponentLifecycleRecord:
    state: str
    reason_id: str = ""
    task_id: str = ""
    operation: str = ""
    manifest_identity: str = ""
    owner_session: str = ""
    attempt: int = 0
    consented: bool = False
    completed_bytes: int = 0
    total_bytes: int = 0
    eta_seconds: int | None = None
    required_additional_bytes: int = 0

    def __post_init__(self) -> None:
        if normalize_component_lifecycle_state(self.state) != self.state:
            raise ValueError("local semantic component lifecycle state is invalid")
        for name in (
            "attempt",
            "completed_bytes",
            "total_bytes",
            "required_additional_bytes",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.total_bytes and self.completed_bytes > self.total_bytes:
            raise ValueError("completed_bytes cannot exceed total_bytes")
        if self.eta_seconds is not None and int(self.eta_seconds) < 0:
            raise ValueError("eta_seconds cannot be negative")

    @property
    def public_state(self) -> str:
        return _PUBLIC_STATE_BY_INTERNAL[self.state]

    @property
    def available(self) -> bool:
        return self.state == "ready"

    @property
    def action_id(self) -> str:
        return _ACTION_BY_INTERNAL[self.state]

    def public_payload(self) -> dict[str, object]:
        return {
            "state": self.public_state,
            "available": self.available,
            "mode": "hybrid_local",
            "reason_id": _PUBLIC_REASON_BY_INTERNAL[self.state],
            "action_id": self.action_id,
            "task_id": self.task_id if component_lifecycle_is_active(self.state) else "",
            "completed_bytes": self.completed_bytes,
            "total_bytes": self.total_bytes,
            "eta_seconds": self.eta_seconds,
            "required_additional_bytes": (
                self.required_additional_bytes if self.state == "needs_space" else 0
            ),
            "bm25_available": True,
        }

    def internal_payload(self) -> dict[str, object]:
        return asdict(self)
