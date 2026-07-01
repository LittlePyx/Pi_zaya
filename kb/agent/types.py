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

EvidenceStatus = Literal["grounded", "needs_review", "insufficient", "not_applicable"]
EvidenceNeed = Literal["low", "medium", "high"]
ResearchRunStatus = Literal["planning", "retrieving", "extracting", "synthesizing", "verified", "failed"]
SourcePolicy = Literal["local_only", "local_plus_external_background", "external_allowed_with_notice", "trusted_sites_only"]

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
class AgentIntent:
    task_type: QuestionType = "unknown"
    target_papers: list[str] = field(default_factory=list)
    required_tools: list[ToolName] = field(default_factory=list)
    evidence_need: EvidenceNeed = "medium"
    confidence: float = 0.0
    routing_signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["confidence"] = round(max(0.0, min(1.0, float(self.confidence or 0.0))), 4)
        return data


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
    local_claims: int = 0
    external_background_claims: int = 0
    source_notice_count: int = 0
    support_ratio: float = 0.0
    evidence_status: EvidenceStatus = "insufficient"
    evidence_hit_count: int = 0
    evidence_status_reasons: list[str] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchSubtask:
    goal: str
    status: StepStatus = "pending"
    tool: ToolName | str = ""
    observation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceMatrixRow:
    paper: str = ""
    source_name: str = ""
    source_path: str = ""
    method: str = ""
    dataset_or_experiment: str = ""
    key_result: str = ""
    limitation: str = ""
    evidence_quote: str = ""
    citation: str = ""
    heading_path: str = ""
    support_status: EvidenceStatus = "needs_review"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchRun:
    run_id: str
    status: ResearchRunStatus = "planning"
    source_policy: SourcePolicy = "local_only"
    query_scope: str = ""
    question: str = ""
    subtasks: list[ResearchSubtask] = field(default_factory=list)
    evidence_matrix: list[EvidenceMatrixRow] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "source_policy": self.source_policy,
            "query_scope": self.query_scope,
            "question": self.question,
            "subtasks": [item.to_dict() for item in self.subtasks],
            "evidence_matrix": [item.to_dict() for item in self.evidence_matrix],
            "metrics": dict(self.metrics),
        }


@dataclass
class AgentTrace:
    mode: Literal["research_agent"] = "research_agent"
    question_type: QuestionType = "unknown"
    context: dict[str, Any] = field(default_factory=dict)
    plan: list[AgentPlanStep] = field(default_factory=list)
    steps: list[AgentExecutionStep] = field(default_factory=list)
    verification: AgentVerification = field(default_factory=AgentVerification)
    research_run: ResearchRun | None = None
    status: StepStatus = "pending"
    errors: list[str] = field(default_factory=list)

    def _quality_gate_summary(self) -> dict[str, Any]:
        valid_statuses = {"passed", "repaired", "fallback"}
        status = ""
        reasons: list[str] = []
        warnings: list[str] = []
        for step in reversed(self.steps):
            if step.tool != "generate_grounded_answer" or not isinstance(step.output, dict):
                continue
            gate = step.output.get("quality_gate")
            if not isinstance(gate, dict):
                continue
            candidate_status = str(gate.get("status") or "").strip().lower()
            if candidate_status in valid_statuses:
                status = candidate_status
            reasons = [
                str(item).strip()
                for item in list(gate.get("reasons") or [])
                if str(item).strip()
            ][:4]
            warnings = [
                str(item).strip()
                for item in list(gate.get("warnings") or [])
                if str(item).strip()
            ][:4]
            break
        return {
            "quality_gate_status": status,
            "quality_gate_reasons": reasons,
            "quality_gate_warnings": warnings,
        }

    def summary_dict(self) -> dict[str, Any]:
        total_claims = int(self.verification.total_claims or 0)
        supported_claims = int(self.verification.supported_claims or 0)
        unsupported_claims = int(self.verification.unsupported_claims or 0)
        ratio = float(self.verification.support_ratio or 0.0)
        if total_claims > 0 and ratio <= 0:
            ratio = supported_claims / total_claims
        planner_intent = self.context.get("planner_intent") if isinstance(self.context.get("planner_intent"), dict) else {}
        try:
            planner_confidence = float(
                planner_intent.get("confidence", self.context.get("planner_confidence", 0.0)) or 0.0
            )
        except Exception:
            planner_confidence = 0.0
        research_run = self.research_run.to_dict() if self.research_run is not None else {}
        metrics = research_run.get("metrics") if isinstance(research_run.get("metrics"), dict) else {}
        quality_gate = self._quality_gate_summary()
        return {
            "question_type": self.question_type,
            "status": self.status,
            "research_run_status": str(research_run.get("status") or ""),
            "source_policy": str(research_run.get("source_policy") or ""),
            "subtask_count": int(metrics.get("subtask_count") or 0),
            "evidence_matrix_rows": int(metrics.get("evidence_matrix_rows") or 0),
            "evidence_status": self.verification.evidence_status,
            "evidence_hit_count": int(self.verification.evidence_hit_count or 0),
            "evidence_status_reasons": list(self.verification.evidence_status_reasons or [])[:4],
            "query_scope": str(self.context.get("query_scope") or ""),
            "requested_query_scope": str(self.context.get("requested_query_scope") or ""),
            "planner_confidence": round(max(0.0, min(1.0, planner_confidence)), 4),
            "evidence_need": str(planner_intent.get("evidence_need") or self.context.get("evidence_need") or ""),
            "retrieval_confidence": str(self.context.get("retrieval_confidence") or ""),
            "usable_hit_count": int(self.context.get("usable_hit_count") or 0),
            "total_claims": total_claims,
            "supported_claims": supported_claims,
            "unsupported_claims": unsupported_claims,
            "local_claims": int(self.verification.local_claims or 0),
            "external_background_claims": int(self.verification.external_background_claims or 0),
            "source_notice_count": int(self.verification.source_notice_count or 0),
            "support_ratio": round(max(0.0, min(1.0, ratio)), 4),
            "plan_step_count": len(self.plan),
            "tool_call_count": len(self.steps),
            "has_errors": bool(self.errors),
            **quality_gate,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "question_type": self.question_type,
            "context": dict(self.context),
            "plan": [step.to_dict() for step in self.plan],
            "steps": [step.to_dict() for step in self.steps],
            "verification": self.verification.to_dict(),
            "research_run": self.research_run.to_dict() if self.research_run is not None else {},
            "summary": self.summary_dict(),
            "status": self.status,
            "errors": list(self.errors),
        }
