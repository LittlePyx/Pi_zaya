from __future__ import annotations

import re

from .types import AgentIntent, AgentPlanStep, EvidenceNeed, QuestionType, ToolName


_COMPARISON_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference|differences|different|contrast|trade-?off|benchmark)\b"
    r"|(?:\u6bd4\u8f83|\u5bf9\u6bd4|\u533a\u522b|\u5dee\u5f02|\u4e0d\u540c|\u76f8\u6bd4|\u76f8\u8f83|\u53d6\u820d|\u6743\u8861)"
    r"|(?:\bA\s+vs\.?\s+B\b)",
    flags=re.IGNORECASE,
)
_ANSWER_AUDIT_RE = re.compile(
    r"\b(?:audit|review|check|verify|critique)\s+(?:the\s+)?(?:previous|last|prior)\s+answer\b"
    r"|(?:\u5ba1\u67e5|\u6838\u5bf9|\u68c0\u67e5|\u9a8c\u8bc1)(?:\u4e0a\u4e00\u6761|\u4e0a\u4e2a|\u524d\u4e00\u6761|\u8be5)?\u56de\u7b54",
    flags=re.IGNORECASE,
)
_READING_GUIDE_RE = re.compile(
    r"\b(how\s+(?:should\s+i\s+)?read|how\s+to\s+read|reading\s+guide|reading\s+map|roadmap|study\s+plan|where\s+should\s+i\s+start)\b"
    r"|(?:\u600e\u4e48\u8bfb|\u5982\u4f55\u8bfb|\u9605\u8bfb\u8def\u7ebf|\u9605\u8bfb\u5730\u56fe|\u8bfb\u54ea|\u5148\u8bfb|\u9605\u8bfb\u6307\u5357|\u5b66\u4e60\u8def\u7ebf)",
    flags=re.IGNORECASE,
)
_REFERENCE_FOLLOWUP_RE = re.compile(
    r"\b(reference|references|citation|cites?|cited|bibliography|upstream|prior\s+work|follow[- ]?up)\b"
    r"|(?:\u53c2\u8003\u6587\u732e|\u5f15\u7528|\u88ab\u5f15|\u4e0a\u6e38|\u524d\u4f5c|\u524d\u4eba\u5de5\u4f5c|\u8ffd\u6eaf|\u51fa\u5904|\u6765\u6e90)",
    flags=re.IGNORECASE,
)
_METHOD_RE = re.compile(
    r"\b(methods?|approaches?|algorithm|model|architecture|pipeline|framework|implementation|training)\b"
    r"|(?:\u65b9\u6cd5|\u7b97\u6cd5|\u6a21\u578b|\u67b6\u6784|\u6d41\u7a0b|\u6846\u67b6|\u5b9e\u73b0|\u8bad\u7ec3)",
    flags=re.IGNORECASE,
)
_LIMITATION_RE = re.compile(
    r"\b(limitations?|challenges?|weakness(?:es)?|failure|fails?|future\s+work|open\s+problem)\b"
    r"|(?:\u5c40\u9650|\u9650\u5236|\u6311\u6218|\u5931\u8d25|\u4e0d\u8db3|\u672a\u6765|\u5f00\u653e\u95ee\u9898)",
    flags=re.IGNORECASE,
)
_EXPERIMENT_RE = re.compile(
    r"\b(experiments?|results?|datasets?|ablation|metrics?|benchmark|evaluation|tables?|figures?)\b"
    r"|(?:\u5b9e\u9a8c|\u7ed3\u679c|\u6570\u636e\u96c6|\u6d88\u878d|\u6307\u6807|\u57fa\u51c6|\u8bc4\u4f30|\u8868|\u56fe)",
    flags=re.IGNORECASE,
)
_QUOTED_TARGET_RE = re.compile(r"[\"'\u201c\u201d\u2018\u2019]([^\"'\u201c\u201d\u2018\u2019]{4,140})[\"'\u201c\u201d\u2018\u2019]")
_FILE_TARGET_RE = re.compile(r"\b([A-Za-z0-9][A-Za-z0-9 _.\-:]{3,140}\.(?:pdf|md))\b", flags=re.IGNORECASE)

_PLAN_GOALS: dict[ToolName, str] = {
    "retrieve_evidence": "Retrieve evidence from the indexed literature library.",
    "retrieve_references": "Retrieve bibliography and upstream reference context.",
    "build_reading_guide": "Build a section-aware reading guide from retrieved sources.",
    "compare_papers": "Compare retrieved papers by source and evidence focus.",
    "generate_grounded_answer": "Generate an evidence-grounded answer.",
    "verify_answer_citations": "Verify citation support at sentence level.",
}


def _classify_question_type_from_text(text: str) -> QuestionType:
    if not text:
        return "unknown"
    if _ANSWER_AUDIT_RE.search(text):
        return "multi_paper_comparison"
    if _COMPARISON_RE.search(text):
        return "multi_paper_comparison"
    if _READING_GUIDE_RE.search(text):
        return "reading_guide"
    if _REFERENCE_FOLLOWUP_RE.search(text):
        return "reference_followup"
    return "single_paper_qa"


def classify_question_type(query: str) -> QuestionType:
    return _classify_question_type_from_text(str(query or "").strip())


def _tools_for_question_type(question_type: QuestionType) -> list[ToolName]:
    tools: list[ToolName] = ["retrieve_evidence"]
    if question_type == "reference_followup":
        tools.append("retrieve_references")
    elif question_type == "reading_guide":
        tools.append("build_reading_guide")
    elif question_type == "multi_paper_comparison":
        tools.append("compare_papers")
    tools.extend(["generate_grounded_answer", "verify_answer_citations"])
    return tools


def _extract_target_papers(text: str) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    for match in list(_QUOTED_TARGET_RE.finditer(text)) + list(_FILE_TARGET_RE.finditer(text)):
        title = " ".join(str(match.group(1) or "").split()).strip(" .,:;")
        if len(title) < 4:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        targets.append(title)
    return targets[:6]


def _routing_signals(text: str, question_type: QuestionType) -> list[str]:
    signals: list[str] = []
    if _ANSWER_AUDIT_RE.search(text):
        signals.append("answer_audit")
    elif question_type == "multi_paper_comparison":
        signals.append("comparison_keyword")
    elif question_type == "reading_guide":
        signals.append("reading_guide_keyword")
    elif question_type == "reference_followup":
        signals.append("reference_followup_keyword")
    if _METHOD_RE.search(text):
        signals.append("method_extraction")
    if _LIMITATION_RE.search(text):
        signals.append("limitation_analysis")
    if _EXPERIMENT_RE.search(text):
        signals.append("experiment_summary")
    if not signals and question_type == "single_paper_qa":
        signals.append("default_single_paper_qa")
    if question_type == "unknown":
        signals.append("empty_or_unknown_query")
    return signals


def _evidence_need(question_type: QuestionType, signals: list[str]) -> EvidenceNeed:
    if question_type in {"multi_paper_comparison", "reference_followup"}:
        return "high"
    if any(signal in {"method_extraction", "limitation_analysis", "experiment_summary"} for signal in signals):
        return "high"
    if question_type == "reading_guide":
        return "medium"
    if question_type == "unknown":
        return "low"
    return "medium"


def _confidence(text: str, question_type: QuestionType, signals: list[str], targets: list[str]) -> float:
    if question_type == "unknown":
        return 0.0
    score = 0.55
    if question_type != "single_paper_qa":
        score = 0.82
    if any(signal in {"method_extraction", "limitation_analysis", "experiment_summary"} for signal in signals):
        score = max(score, 0.72)
    if targets:
        score += 0.05
    if len(text) < 8:
        score = min(score, 0.35)
    return max(0.0, min(0.95, score))


def classify_agent_intent(query: str) -> AgentIntent:
    text = str(query or "").strip()
    question_type = _classify_question_type_from_text(text)
    targets = _extract_target_papers(text)
    signals = _routing_signals(text, question_type)
    return AgentIntent(
        task_type=question_type,
        target_papers=targets,
        required_tools=_tools_for_question_type(question_type),
        evidence_need=_evidence_need(question_type, signals),
        confidence=_confidence(text, question_type, signals, targets),
        routing_signals=signals,
    )


def plan_research_intent(query: str) -> AgentIntent:
    return classify_agent_intent(query)


def plan_research_question(query: str) -> tuple[QuestionType, list[AgentPlanStep]]:
    intent = classify_agent_intent(query)
    return intent.task_type, [AgentPlanStep(goal=_PLAN_GOALS[tool], tool=tool) for tool in intent.required_tools]
