"""Paper Guide router.

This module is the package-oriented home for prompt intent resolution and exact
skill dispatch. The legacy flat import path ``kb.paper_guide_router`` now acts
as a compatibility shim to this implementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .contracts import PaperGuideIntentModel, _build_paper_guide_intent_model
from .skills import (
    PaperGuideSkillResult,
    run_abstract_skill,
    run_box_target_skill,
    run_exact_citation_lookup_skill,
    run_exact_equation_skill,
    run_exact_figure_panel_skill,
    run_exact_method_skill,
    run_overview_component_role_skill,
    run_section_target_skill,
)
from ..paper_guide_prompting import (
    _paper_guide_prompt_family,
    _paper_guide_prompt_requests_exact_method_support,
    _requested_figure_number,
)
from ..paper_guide_provenance import _extract_figure_number
from ..paper_guide_target_scope import _build_paper_guide_target_scope, _extract_prompt_panel_letters
from ..source_blocks import extract_equation_number

PromptPredicateFn = Callable[[str], bool]
ResolveRecordFn = Callable[..., dict]
BuildAnswerFn = Callable[[dict], tuple[str, list[dict]]]
ExtractRefNumsFn = Callable[..., list[int]]
SanitizeAnswerFn = Callable[..., str]
ExtractFocusTermsFn = Callable[[str], list[str]]
ExtractFocusTextFn = Callable[..., str]
BuildLinesFn = Callable[..., list[str]]
SelectHitFn = Callable[..., dict]
ExtractSnippetFn = Callable[[str], str]
BuildDirectAnswerFn = Callable[..., str]
ExtractNumsFn = Callable[[str], list[int]]

_PAPER_GUIDE_CITATION_EXACT_SUPPORT_RE = re.compile(
    r"(?i)("
    r"where\s+is\s+that\s+stated\s+exactly|where\s+exactly|exact\s+supporting\s+part|"
    r"exact\s+supporting\s+sentence(?:s)?|exact\s+supporting\s+sentence\(s\)|"
    r"exact\s+sentence(?:s)?|supporting\s+sentence(?:s)?|point\s+me\s+to|"
    r"\u5f15\u7528\u7f16\u53f7|\u53c2\u8003\u6587\u732e\u7f16\u53f7|\u6587\u5185\u5f15\u7528|\u6587\u5185\u53c2\u8003|"
    r"\u539f\u6587.*?(?:\u54ea\u91cc|\u54ea\u513f).*?(?:\u5199|\u8bf4|\u63d0\u5230)|"
    r"(?:\u54ea\u91cc|\u54ea\u513f).{0,6}\u660e\u786e.{0,10}(?:\u5199|\u8bf4|\u63d0\u5230)|"
    r"\u7ed9\u51fa.*?(?:\u539f\u6587|\u539f\u53e5).*?(?:\u652f\u6301|\u8bc1\u636e)|"
    r"\u53ef\u5b9a\u4f4d.*?(?:\u539f\u6587|\u652f\u6301|\u8bc1\u636e)"
    r")"
)
_PAPER_GUIDE_FIGURE_CAPTION_EXACT_SUPPORT_RE = re.compile(
    r"(?i)("
    r"exact\s+supporting\s+caption\s+clause|caption\s+clause|exact\s+caption|"
    r"\u56fe\u6ce8.*?(?:\u539f\u6587|\u539f\u53e5)|"
    r"(?:\u539f\u6587|\u539f\u53e5).*?\u56fe\u6ce8|"
    r"\u56fe\u6ce8.*?\u53e5"
    r")"
)
_PAPER_GUIDE_BEGINNER_HINT_RE = re.compile(
    r"(?i)("
    r"\bbeginner(?:s)?\b|\bnew\s+to\b|\bintro(?:duction)?\b|\bhigh[- ]level\b|\bin simple terms\b|"
    r"\bsimply explain\b|\bsimple explanation\b|\bquick overview\b|"
    r"\u521d\u5b66\u8005|\u5165\u95e8|\u5c0f\u767d|\u901a\u4fd7|\u7b80\u5355\u8bf4|\u7b80\u5355\u8bb2|\u5148\u8bb2\u4e00\u4e0b"
    r")"
)
_PAPER_GUIDE_BROAD_OVERVIEW_RE = re.compile(
    r"(?i)("
    r"\bwhat\s+problem\b|\bwhat\s+does\s+this\s+paper\s+do\b|\bmain\s+idea\b|\bcore\s+idea\b|"
    r"\bkey\s+contribution\b|\bsummary\b|\bwhy\s+is\s+it\s+interesting\b|"
    r"\u89e3\u51b3.*\u95ee\u9898|\u8fd9\u7bc7.*\u8bb2.*\u4ec0\u4e48|\u6838\u5fc3\u8d21\u732e|\u4e3b\u8981\u8d21\u732e|\u603b\u7ed3"
    r")"
)


@dataclass(frozen=True)
class PaperGuideExactSkillDeps:
    resolve_exact_method_support: ResolveRecordFn
    resolve_exact_equation_support: ResolveRecordFn
    build_exact_equation_answer: BuildAnswerFn
    resolve_exact_citation_lookup_support: ResolveRecordFn
    extract_inline_reference_numbers: ExtractRefNumsFn
    resolve_exact_figure_panel_caption_support: ResolveRecordFn
    extract_caption_clause_superscript_ref_nums: ExtractRefNumsFn
    sanitize_answer: SanitizeAnswerFn


@dataclass(frozen=True)
class PaperGuideBroadSkillDeps:
    build_direct_abstract_answer: BuildDirectAnswerFn
    prompt_requests_component_role_explanation: PromptPredicateFn
    extract_method_focus_terms: ExtractFocusTermsFn
    extract_component_role_focus: ExtractFocusTextFn
    build_overview_role_lines: BuildLinesFn
    select_section_target_hit: SelectHitFn
    extract_discussion_future_snippet: ExtractSnippetFn
    extract_box_numbers: ExtractNumsFn
    sanitize_answer: SanitizeAnswerFn | None = None


def _paper_guide_prompt_requests_exact_equation_support(prompt: str) -> bool:
    q = str(prompt or "").strip()
    if not q:
        return False
    has_equation_marker = bool(
        re.search(r"(?i)\b(?:equation|eq\.?|formula)\b", q) or ("\u516c\u5f0f" in q) or ("\u65b9\u7a0b" in q)
    )
    if not has_equation_marker:
        return False
    if extract_equation_number(q) > 0:
        return True
    return bool(
        re.search(
            r"(?i)(?:\bvariable(?:s)?\b|\bdefine(?:s|d)?\b|\bdefinition\b|\bdenote(?:s|d)?\b|\brepresent(?:s|ed)?\b|"
            r"\bwhat\s+does\b|\bpoint\s+me\s+to\b|\bexact\s+supporting\s+part\b|\bexact\s+supporting\s+sentence\b|"
            r"\u53d8\u91cf|\u7b26\u53f7|\u5b9a\u4e49|\u539f\u6587|\u652f\u6301|\u54ea\u91cc)",
            q,
        )
    )


def _paper_guide_prompt_requests_exact_citation_support(prompt: str) -> bool:
    q = str(prompt or "").strip()
    if not q:
        return False
    return bool(_PAPER_GUIDE_CITATION_EXACT_SUPPORT_RE.search(q))


def _paper_guide_prompt_requests_exact_figure_caption_support(prompt: str) -> bool:
    q = str(prompt or "").strip()
    if not q:
        return False
    return bool(_PAPER_GUIDE_FIGURE_CAPTION_EXACT_SUPPORT_RE.search(q))


def _paper_guide_beginner_mode(prompt: str, *, family: str = "") -> bool:
    q = str(prompt or "").strip()
    family_norm = str(family or "").strip().lower()
    if not q:
        return False
    if _PAPER_GUIDE_BEGINNER_HINT_RE.search(q):
        return True
    return bool(family_norm == "overview" and _PAPER_GUIDE_BROAD_OVERVIEW_RE.search(q))


def _resolve_paper_guide_intent(
    prompt: str,
    *,
    prompt_family: str = "",
    intent: str = "",
    answer_hits: list[dict] | None = None,
) -> PaperGuideIntentModel:
    q = str(prompt or "").strip()
    family = str(prompt_family or "").strip().lower() or str(_paper_guide_prompt_family(q, intent=intent) or "").strip().lower()
    target_scope = _build_paper_guide_target_scope(q, prompt_family=family)
    target_figure = int(_requested_figure_number(q, list(answer_hits or [])) or 0)
    if target_figure <= 0:
        target_figure = int(_extract_figure_number(q) or 0)
    target_panels = _extract_prompt_panel_letters(q)
    target_equation = int(extract_equation_number(q) or 0)

    exact_support = False
    if family in {"method", "reproduce"}:
        exact_support = bool(_paper_guide_prompt_requests_exact_method_support(q))
    elif family == "equation":
        exact_support = bool(_paper_guide_prompt_requests_exact_equation_support(q))
    elif family == "citation_lookup":
        exact_support = bool(_paper_guide_prompt_requests_exact_citation_support(q))
    elif family == "figure_walkthrough":
        exact_support = bool(_paper_guide_prompt_requests_exact_figure_caption_support(q))

    return _build_paper_guide_intent_model(
        prompt=q,
        family=family,
        exact_support=exact_support,
        beginner_mode=_paper_guide_beginner_mode(q, family=family),
        target_figure=target_figure,
        target_panels=target_panels,
        target_equation=target_equation,
        target_scope=target_scope,
    )


def _dispatch_paper_guide_exact_support_skill(
    *,
    prompt_text: str,
    resolved_intent: PaperGuideIntentModel | None,
    source_path: str,
    db_dir: Path | None,
    has_hits: bool,
    deps: PaperGuideExactSkillDeps,
) -> PaperGuideSkillResult | None:
    prompt = str(prompt_text or "").strip()
    family = str((resolved_intent.family if isinstance(resolved_intent, PaperGuideIntentModel) else "") or "").strip().lower()
    if (not prompt) or (not family):
        return None

    if family == "equation":
        return run_exact_equation_skill(
            prompt_text=prompt,
            prompt_family=family,
            source_path=source_path,
            db_dir=db_dir,
            has_hits=has_hits,
            prompt_requests_exact_support=_paper_guide_prompt_requests_exact_equation_support,
            resolve_exact_support=deps.resolve_exact_equation_support,
            build_exact_answer=deps.build_exact_equation_answer,
            sanitize_answer=deps.sanitize_answer,
        )
    if family == "citation_lookup":
        return run_exact_citation_lookup_skill(
            prompt_text=prompt,
            prompt_family=family,
            source_path=source_path,
            db_dir=db_dir,
            has_hits=has_hits,
            prompt_requests_exact_support=_paper_guide_prompt_requests_exact_citation_support,
            resolve_exact_support=deps.resolve_exact_citation_lookup_support,
            extract_inline_reference_numbers=deps.extract_inline_reference_numbers,
            sanitize_answer=deps.sanitize_answer,
        )
    if family == "figure_walkthrough":
        return run_exact_figure_panel_skill(
            prompt_text=prompt,
            prompt_family=family,
            source_path=source_path,
            db_dir=db_dir,
            has_hits=has_hits,
            prompt_requests_exact_support=_paper_guide_prompt_requests_exact_figure_caption_support,
            resolve_exact_support=deps.resolve_exact_figure_panel_caption_support,
            extract_clause_reference_numbers=deps.extract_caption_clause_superscript_ref_nums,
            sanitize_answer=deps.sanitize_answer,
        )
    if family in {"method", "reproduce"}:
        return run_exact_method_skill(
            prompt_text=prompt,
            prompt_family=family,
            source_path=source_path,
            db_dir=db_dir,
            has_hits=has_hits,
            prompt_requests_exact_support=_paper_guide_prompt_requests_exact_method_support,
            resolve_exact_support=deps.resolve_exact_method_support,
            sanitize_answer=deps.sanitize_answer,
        )
    return None


def _dispatch_paper_guide_broad_skill(
    *,
    prompt_text: str,
    resolved_intent: PaperGuideIntentModel | None,
    source_path: str,
    db_dir: Path | None,
    has_hits: bool,
    has_non_ref_target: bool,
    llm=None,
    deps: PaperGuideBroadSkillDeps,
) -> PaperGuideSkillResult | None:
    prompt = str(prompt_text or "").strip()
    family = str((resolved_intent.family if isinstance(resolved_intent, PaperGuideIntentModel) else "") or "").strip().lower()
    if (not prompt) or (not family):
        return None

    if family == "abstract":
        return run_abstract_skill(
            prompt_text=prompt,
            prompt_family=family,
            source_path=source_path,
            db_dir=db_dir,
            llm=llm,
            build_direct_abstract_answer=deps.build_direct_abstract_answer,
            sanitize_answer=deps.sanitize_answer,
        )
    if family == "box_only":
        return run_box_target_skill(
            prompt_text=prompt,
            prompt_family=family,
            source_path=source_path,
            db_dir=db_dir,
            has_hits=has_hits,
            has_non_ref_target=has_non_ref_target,
            select_target_hit=deps.select_section_target_hit,
            extract_box_numbers=deps.extract_box_numbers,
            sanitize_answer=deps.sanitize_answer,
        )
    if family in {"overview", "method"}:
        return run_overview_component_role_skill(
            prompt_text=prompt,
            prompt_family=family,
            source_path=source_path,
            db_dir=db_dir,
            has_hits=has_hits,
            prompt_requests_component_role_explanation=deps.prompt_requests_component_role_explanation,
            extract_method_focus_terms=deps.extract_method_focus_terms,
            extract_component_role_focus=deps.extract_component_role_focus,
            build_overview_role_lines=deps.build_overview_role_lines,
            sanitize_answer=deps.sanitize_answer,
        )
    if family in {"strength_limits", "discussion_only"}:
        return run_section_target_skill(
            prompt_text=prompt,
            prompt_family=family,
            source_path=source_path,
            db_dir=db_dir,
            has_hits=has_hits,
            has_non_ref_target=has_non_ref_target,
            select_target_hit=deps.select_section_target_hit,
            extract_discussion_future_snippet=deps.extract_discussion_future_snippet,
            sanitize_answer=deps.sanitize_answer,
        )
    return None


# Back-compat aliases for the old flat module symbol names.
_paper_guide_prompt_requests_exact_equation_support_router = _paper_guide_prompt_requests_exact_equation_support
_paper_guide_prompt_requests_exact_citation_support_router = _paper_guide_prompt_requests_exact_citation_support
_paper_guide_prompt_requests_exact_figure_caption_support_router = _paper_guide_prompt_requests_exact_figure_caption_support


__all__ = [
    "PaperGuideBroadSkillDeps",
    "PaperGuideExactSkillDeps",
    "PaperGuideIntentModel",
    "PaperGuideSkillResult",
    "_dispatch_paper_guide_broad_skill",
    "_dispatch_paper_guide_exact_support_skill",
    "_paper_guide_beginner_mode",
    "_paper_guide_prompt_requests_exact_citation_support",
    "_paper_guide_prompt_requests_exact_citation_support_router",
    "_paper_guide_prompt_requests_exact_equation_support",
    "_paper_guide_prompt_requests_exact_equation_support_router",
    "_paper_guide_prompt_requests_exact_figure_caption_support",
    "_paper_guide_prompt_requests_exact_figure_caption_support_router",
    "_resolve_paper_guide_intent",
]
