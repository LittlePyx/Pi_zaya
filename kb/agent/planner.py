from __future__ import annotations

import re

from .types import AgentPlanStep, QuestionType


_COMPARISON_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference|different|contrast|trade-?off|benchmark)\b"
    r"|(?:比较|对比|区别|差异|不同|相比|相较|取舍|权衡)"
    r"|(?:\bA\s+vs\.?\s+B\b)",
    flags=re.IGNORECASE,
)
_READING_GUIDE_RE = re.compile(
    r"\b(how\s+to\s+read|reading\s+guide|reading\s+map|roadmap|study\s+plan|where\s+should\s+i\s+start)\b"
    r"|(?:怎么读|如何读|阅读路线|阅读地图|读哪|先读|阅读指南|学习路线)",
    flags=re.IGNORECASE,
)
_REFERENCE_FOLLOWUP_RE = re.compile(
    r"\b(reference|references|citation|cites?|cited|bibliography|upstream|prior\s+work|follow[- ]?up)\b"
    r"|(?:参考文献|引用|被引|上游|前作|前人工作|追溯|出处|来源)",
    flags=re.IGNORECASE,
)


def classify_question_type(query: str) -> QuestionType:
    text = str(query or "").strip()
    if not text:
        return "unknown"
    if _COMPARISON_RE.search(text):
        return "multi_paper_comparison"
    if _READING_GUIDE_RE.search(text):
        return "reading_guide"
    if _REFERENCE_FOLLOWUP_RE.search(text):
        return "reference_followup"
    return "single_paper_qa"


def plan_research_question(query: str) -> tuple[QuestionType, list[AgentPlanStep]]:
    question_type = classify_question_type(query)
    steps: list[AgentPlanStep] = [
        AgentPlanStep(goal="Retrieve evidence from the indexed literature library.", tool="retrieve_evidence"),
    ]
    if question_type == "reference_followup":
        steps.append(AgentPlanStep(goal="Retrieve bibliography and upstream reference context.", tool="retrieve_references"))
    elif question_type == "reading_guide":
        steps.append(AgentPlanStep(goal="Build a section-aware reading guide from retrieved sources.", tool="build_reading_guide"))
    elif question_type == "multi_paper_comparison":
        steps.append(AgentPlanStep(goal="Compare retrieved papers by source and evidence focus.", tool="compare_papers"))
    steps.extend(
        [
            AgentPlanStep(goal="Generate an evidence-grounded answer.", tool="generate_grounded_answer"),
            AgentPlanStep(goal="Verify citation support at sentence level.", tool="verify_answer_citations"),
        ]
    )
    return question_type, steps
