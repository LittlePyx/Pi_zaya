from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


QuestionType = Literal[
    "single_paper_qa",
    "multi_paper_comparison",
    "reading_guide",
    "reference_followup",
    "unknown",
]

ToolName = Literal[
    "retrieve_evidence",
    "retrieve_references",
    "build_reading_guide",
    "compare_papers",
    "generate_grounded_answer",
    "verify_answer_citations",
]

StepStatus = Literal["pending", "running", "done", "error", "skipped", "canceled"]


@dataclass
class AgentPlanStep:
    goal: str
    tool: ToolName
    status: StepStatus = "pending"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentExecutionStep:
    tool: ToolName
    status: StepStatus
    observation: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    elapsed_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentVerification:
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    support_ratio: float = 0.0
    claims: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentTrace:
    mode: Literal["research_agent"] = "research_agent"
    question_type: QuestionType = "unknown"
    context: dict[str, Any] = field(default_factory=dict)
    plan: list[AgentPlanStep] = field(default_factory=list)
    steps: list[AgentExecutionStep] = field(default_factory=list)
    verification: AgentVerification = field(default_factory=AgentVerification)
    status: StepStatus = "pending"
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "question_type": self.question_type,
            "context": dict(self.context),
            "plan": [step.to_dict() for step in self.plan],
            "steps": [step.to_dict() for step in self.steps],
            "verification": self.verification.to_dict(),
            "status": self.status,
            "errors": list(self.errors),
        }
