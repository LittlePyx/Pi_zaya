from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ResearchAnswerPlan:
    kind: str
    evidence_need: str
    answer_shape: str
    avoid: str

    def to_prompt_block(self) -> str:
        return (
            "Research answer plan:\n"
            f"- Plan type: {self.kind}.\n"
            f"- Evidence need: {self.evidence_need}\n"
            f"- Answer shape: {self.answer_shape}\n"
            f"- Avoid: {self.avoid}\n"
            "- Keep paper-grounded claims separate from inference or general research advice."
        )


_PLAN_CARDS: dict[str, ResearchAnswerPlan] = {
    "paper_summary": ResearchAnswerPlan(
        kind="paper_summary",
        evidence_need="problem statement, core method, main result, and stated limitation or scope.",
        answer_shape="direct takeaway first; then contribution, key evidence, limits, and one reading/use suggestion.",
        avoid="generic literature-summary prose that does not say what this paper changes.",
    ),
    "method_explain": ResearchAnswerPlan(
        kind="method_explain",
        evidence_need="the problem each method solves, inputs/outputs, method steps, key equation/module, reported effects or metrics, and stated assumptions; preserve source terms and acronyms such as SNR when present.",
        answer_shape="what trouble it solves, how it works, what each key component changes, the evidence-backed trade-off, and what is not specified.",
        avoid="inventing formulas, parameters, hardware, or training details that are not in the retrieved evidence.",
    ),
    "compare": ResearchAnswerPlan(
        kind="compare",
        evidence_need="the same comparison axes for each paper/method: method, metric, assumption, result, and limitation.",
        answer_shape="verdict first; compact side-by-side comparison table if useful; then evidence-backed trade-offs and use cases.",
        avoid="one-paper-at-a-time summaries without a clear side-by-side judgment.",
    ),
    "experiment_design": ResearchAnswerPlan(
        kind="experiment_design",
        evidence_need="variables, controls, metrics, dataset/sample, baseline, and failure risks stated or implied by evidence.",
        answer_shape="minimal experiment path, controls/metrics, expected observations, and risk checks.",
        avoid="turning a paper summary into an executable protocol when required details are missing.",
    ),
    "literature_positioning": ResearchAnswerPlan(
        kind="literature_positioning",
        evidence_need="upstream references, cited methods, related papers, venue/year metadata, and relation to the user topic.",
        answer_shape="where this work sits, which prior works matter, why each is relevant, and what to read next.",
        avoid="bibliography dumps that do not explain the role of each work.",
    ),
    "critical_review": ResearchAnswerPlan(
        kind="critical_review",
        evidence_need="strongest direct evidence, weak or indirect evidence, missing controls, assumptions, and stated limitations.",
        answer_shape="bottom-line assessment; credible claims; weak claims; open questions; concrete verification step.",
        avoid="unsupported criticism or praise that cannot be tied back to retrieved evidence.",
    ),
}


def _has_any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _normalize_family(value: str) -> str:
    return str(value or "").strip().lower()


def infer_research_answer_plan(
    *,
    prompt: str,
    paper_guide_prompt_family: str = "",
    answer_intent: str = "",
    answer_output_mode: str = "",
    paper_guide_mode: bool = False,
) -> ResearchAnswerPlan:
    """Choose one small research-answer strategy card.

    This is intentionally shallow. Retrieval still decides evidence; exact
    paper-guide skills still handle exact equation/citation/figure requests.
    The plan only tells the LLM what kind of research-facing answer to produce.
    """

    q = str(prompt or "").strip()
    q_low = q.lower()
    family = _normalize_family(paper_guide_prompt_family)
    intent = _normalize_family(answer_intent)
    output_mode = _normalize_family(answer_output_mode)

    if family in {"citation_lookup"}:
        return _PLAN_CARDS["literature_positioning"]
    if family in {"equation", "method", "figure_walkthrough"}:
        return _PLAN_CARDS["method_explain"]
    if family in {"reproduce"}:
        return _PLAN_CARDS["experiment_design"]
    if family in {"compare"}:
        return _PLAN_CARDS["compare"]
    if family in {"box_only"}:
        return _PLAN_CARDS["paper_summary"]
    if family in {"discussion_only", "strength_limits"} or output_mode == "critical_review":
        return _PLAN_CARDS["critical_review"]
    if intent == "compare":
        return _PLAN_CARDS["compare"]
    if intent == "experiment":
        return _PLAN_CARDS["experiment_design"]
    if intent == "writing":
        return _PLAN_CARDS["literature_positioning"]
    if intent == "troubleshoot":
        return _PLAN_CARDS["critical_review"]
    if intent == "idea":
        return _PLAN_CARDS["critical_review"]

    if _has_any(
        (
            r"\bcompare\b",
            r"\bversus\b",
            r"\bvs\.?\b",
            r"\btrade[-\s]?off\b",
            r"\bdifference(?:s)?\b",
            r"\badvantage(?:s)?\b",
            r"\bdisadvantage(?:s)?\b",
            r"\u5bf9\u6bd4",
            r"\u6bd4\u8f83",
            r"\u533a\u522b",
            r"\u53d6\u820d",
            r"\u4f18\u52bf",
            r"\u7f3a\u70b9",
        ),
        q,
    ):
        return _PLAN_CARDS["compare"]
    if _has_any(
        (
            r"\breproduc(?:e|tion|ible)\b",
            r"\breplicat(?:e|ion)\b",
            r"\bexperiment(?:al)?\b",
            r"\bprotocol\b",
            r"\bcontrol(?:s)?\b",
            r"\bmetric(?:s)?\b",
            r"\bbaseline(?:s)?\b",
            r"\bsetup\b",
            r"\u590d\u73b0",
            r"\u5b9e\u9a8c",
            r"\u5bf9\u7167",
            r"\u6307\u6807",
            r"\u57fa\u7ebf",
            r"\u642d\u5efa",
        ),
        q,
    ):
        return _PLAN_CARDS["experiment_design"]
    if _has_any(
        (
            r"\bmethod\b",
            r"\balgorithm\b",
            r"\bmechanism\b",
            r"\bhow\s+(?:does|do|is|are)\b",
            r"\bequation\b",
            r"\bformula\b",
            r"\bmodule\b",
            r"\bcomponent\b",
            r"\u65b9\u6cd5",
            r"\u7b97\u6cd5",
            r"\u539f\u7406",
            r"\u673a\u5236",
            r"\u516c\u5f0f",
            r"\u6a21\u5757",
            r"\u600e\u4e48",
        ),
        q,
    ):
        return _PLAN_CARDS["method_explain"]
    if _has_any(
        (
            r"\bcitation(?:s)?\b",
            r"\breference(?:s)?\b",
            r"\bprior\s+work\b",
            r"\brelated\s+work\b",
            r"\borigin\b",
            r"\bsource\b",
            r"\bwhat\s+to\s+read\b",
            r"\bread\s+next\b",
            r"\u5f15\u7528",
            r"\u53c2\u8003\u6587\u732e",
            r"\u5148\u524d\u5de5\u4f5c",
            r"\u76f8\u5173\u5de5\u4f5c",
            r"\u6765\u6e90",
            r"\u8bfb\u4ec0\u4e48",
        ),
        q,
    ):
        return _PLAN_CARDS["literature_positioning"]
    if _has_any(
        (
            r"\blimit(?:ation)?s?\b",
            r"\bweak(?:ness|nesses)?\b",
            r"\brisk(?:s)?\b",
            r"\bcritic(?:al|ize|ise|ism)\b",
            r"\bconvincing\b",
            r"\bcredible\b",
            r"\bwhat\s+is\s+missing\b",
            r"\u5c40\u9650",
            r"\u4e0d\u8db3",
            r"\u98ce\u9669",
            r"\u6279\u5224",
            r"\u53ef\u4fe1",
            r"\u7f3a\u4ec0\u4e48",
            r"\baudit\b",
            r"\bverify\s+(?:the\s+)?(?:previous|last|prior)\s+answer\b",
            r"\u5ba1\u67e5",
            r"\u6838\u5bf9",
            r"\u9519\u914d",
            r"\u4e0d\u4e00\u81f4",
        ),
        q,
    ):
        return _PLAN_CARDS["critical_review"]

    if paper_guide_mode and family in {"overview", "abstract"}:
        return _PLAN_CARDS["paper_summary"]
    if "summary" in q_low or "\u603b\u7ed3" in q:
        return _PLAN_CARDS["paper_summary"]
    return _PLAN_CARDS["paper_summary"]
