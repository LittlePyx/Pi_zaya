"""Paper Guide exact-support skill orchestration helpers.

This module intentionally stays thin: the legacy resolver logic still lives in
``kb.paper_guide_answer_post_runtime`` and related flat modules, while the
package-oriented skills layer owns only exact-support orchestration and answer
formatting.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Callable

from ..paper_guide_prompting import _paper_guide_prompt_requests_naive_source_trace


PromptPredicateFn = Callable[[str], bool]
ResolveRecordFn = Callable[..., dict]
BuildAnswerFn = Callable[[dict], tuple[str, list[dict]]]
SanitizeAnswerFn = Callable[..., str]
ExtractRefNumsFn = Callable[..., list[int]]
ExtractFocusTermsFn = Callable[[str], list[str]]
ExtractFocusTextFn = Callable[..., str]
BuildLinesFn = Callable[..., list[str]]
SelectHitFn = Callable[..., dict]
ExtractSnippetFn = Callable[[str], str]
BuildDirectAnswerFn = Callable[..., str]
ExtractNumsFn = Callable[[str], list[int]]


@dataclass(frozen=True)
class PaperGuideSkillResult:
    answer_text: str
    support_resolution: list[dict]


def _identity_sanitize_answer(answer: str, **_kwargs) -> str:
    return str(answer or "").strip()


def _positive_ints(values: list[int] | tuple[int, ...], *, limit: int) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for item in list(values or []):
        try:
            value = int(item)
        except Exception:
            continue
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= limit:
            break
    return out


def _source_location_label(heading_path: str, *, prefer_zh: bool) -> str:
    heading = str(heading_path or "").strip()
    if prefer_zh:
        return f"原文位置：{heading}。" if heading else "原文位置：当前绑定论文的匹配段落。"
    return f"Source location: {heading}." if heading else "Source location: matched passage in the bound paper."


def _finalize_skill_result(
    answer_text: str,
    *,
    support_resolution: list[dict],
    has_hits: bool,
    prompt_text: str,
    prompt_family: str,
    sanitize_answer: SanitizeAnswerFn | None,
) -> PaperGuideSkillResult:
    sanitizer = sanitize_answer or _identity_sanitize_answer
    out = sanitizer(
        str(answer_text or "").strip(),
        has_hits=bool(has_hits),
        prompt=prompt_text,
        prompt_family=prompt_family,
    )
    return PaperGuideSkillResult(
        answer_text=str(out or "").strip(),
        support_resolution=list(support_resolution or []),
    )


def _build_skill_support_record(
    *,
    source_path: str,
    heading_path: str = "",
    locate_anchor: str = "",
    claim_type: str = "",
    extra: dict | None = None,
) -> dict:
    out = dict(extra or {})
    src = str(source_path or "").strip()
    heading = str(heading_path or "").strip()
    anchor = str(locate_anchor or "").strip()
    family = str(claim_type or "").strip()
    if src:
        out.setdefault("source_path", src)
    if heading:
        out["heading_path"] = heading
    if anchor:
        out["locate_anchor"] = anchor
        out.setdefault("segment_text", anchor)
    if family:
        out.setdefault("claim_type", family)
    out.setdefault("segment_index", -1)
    return out


def _extract_block_quote(answer_text: str, *, label: str) -> str:
    text = str(answer_text or "").strip()
    target = str(label or "").strip()
    if (not text) or (not target):
        return ""
    pattern = re.compile(rf"(?ms){re.escape(target)}\s*\n>\s*(.+?)(?:\n\s*\n|$)")
    match = pattern.search(text)
    if not match:
        return ""
    return str(match.group(1) or "").strip()


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def _source_trace_target_label(prompt: str) -> str:
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9_-]{2,24}\b", str(prompt or "")):
        label = str(match.group(0) or "").strip()
        if label and label not in {"SCI", "SPI", "PDF"}:
            return label
    return "这个方法" if _contains_cjk(prompt) else "This method"


def _structured_reference_label(source_path: str, ref_nums: list[int]) -> str:
    source = str(source_path or "").strip()
    if not source:
        return ", ".join(f"[{int(n)}]" for n in ref_nums if int(n) > 0)
    sid = "s" + hashlib.sha1(source.encode("utf-8", "ignore")).hexdigest()[:8]
    return ", ".join(f"[[CITE:{sid}:{int(n)}]]" for n in ref_nums if int(n) > 0)


def run_exact_equation_skill(
    *,
    prompt_text: str,
    prompt_family: str,
    source_path: str,
    db_dir: Path | None,
    has_hits: bool,
    prompt_requests_exact_support: PromptPredicateFn,
    resolve_exact_support: ResolveRecordFn,
    build_exact_answer: BuildAnswerFn,
    sanitize_answer: SanitizeAnswerFn,
) -> PaperGuideSkillResult | None:
    prompt = str(prompt_text or "").strip()
    if (not prompt) or (not prompt_requests_exact_support(prompt)):
        return None

    rec = resolve_exact_support(
        str(source_path or "").strip(),
        prompt=prompt,
        db_dir=db_dir,
    )
    equation_markdown = str((rec or {}).get("equation_markdown") or "").strip()
    if not equation_markdown:
        return None

    out, support_resolution = build_exact_answer(rec)
    return _finalize_skill_result(
        out,
        support_resolution=list(support_resolution or []),
        has_hits=has_hits,
        prompt_text=prompt,
        prompt_family=prompt_family,
        sanitize_answer=sanitize_answer,
    )


def run_exact_citation_lookup_skill(
    *,
    prompt_text: str,
    prompt_family: str,
    source_path: str,
    db_dir: Path | None,
    has_hits: bool,
    prompt_requests_exact_support: PromptPredicateFn,
    resolve_exact_support: ResolveRecordFn,
    extract_inline_reference_numbers: ExtractRefNumsFn,
    sanitize_answer: SanitizeAnswerFn,
) -> PaperGuideSkillResult | None:
    prompt = str(prompt_text or "").strip()
    if (not prompt) or (not prompt_requests_exact_support(prompt)):
        return None

    rec = resolve_exact_support(
        str(source_path or "").strip(),
        prompt=prompt,
        db_dir=db_dir,
    )
    if bool((rec or {}).get("not_found")):
        requested_title = str((rec or {}).get("requested_title") or "").strip()
        if _contains_cjk(prompt):
            answer = f"本文参考文献中没有《{requested_title}》。" if requested_title else "本文参考文献中没有找到这篇文献。"
        else:
            answer = (
                f'The paper\'s reference list does not contain "{requested_title}".'
                if requested_title
                else "The paper's reference list does not contain that work."
            )
        return _finalize_skill_result(
            answer,
            support_resolution=[],
            has_hits=has_hits,
            prompt_text=prompt,
            prompt_family=prompt_family,
            sanitize_answer=sanitize_answer,
        )
    locate_anchor = str((rec or {}).get("locate_anchor") or "").strip()
    heading_path = str((rec or {}).get("heading_path") or "").strip()
    ref_nums = _positive_ints(list((rec or {}).get("ref_nums") or []), limit=4)
    if locate_anchor and (not ref_nums):
        ref_nums = _positive_ints(
            extract_inline_reference_numbers(locate_anchor, max_candidates=4),
            limit=4,
        )
    if (not locate_anchor) or (not ref_nums):
        return None

    ref_label = _structured_reference_label(str(source_path or ""), ref_nums[:4])
    reference_title = str((rec or {}).get("reference_title") or "").strip()
    naive_source_trace = _paper_guide_prompt_requests_naive_source_trace(prompt)
    if _contains_cjk(prompt) and reference_title:
        if re.search(r"(?i)compress(?:ed|ive)\s+sensing", locate_anchor) and re.search(
            r"(?i)reduce(?:d|s)?\s+measurements?", locate_anchor
        ):
            role_line = "上游作用：它被本文列为用压缩感知减少测量次数的已有工作；该句同时指出代价是计算时间。"
        else:
            role_line = "上游作用：它是当前这句话指向的已有工作，而不是本文自身提出的结果。"
        lines = [
            f"这篇文献是本文参考文献 {ref_label}：{reference_title}。",
            _source_location_label(heading_path, prefer_zh=True),
            role_line,
            "正文引用语境：",
        ]
    elif _contains_cjk(prompt) and naive_source_trace:
        target_label = _source_trace_target_label(prompt)
        lines = [
            f"不是。{target_label} 在这里是当前论文引用的已有方法背景 {ref_label}，不是本文原创。",
            _source_location_label(heading_path, prefer_zh=True),
            "原文引用语境：",
        ]
    elif _contains_cjk(prompt):
        lines = [
            f"这条线索在原文里可以定位到下面这句；句中的文内来源是 {ref_label}。",
            _source_location_label(heading_path, prefer_zh=True),
        ]
    elif reference_title:
        lines = [
            f'Reference {ref_label} is "{reference_title}".',
            _source_location_label(heading_path, prefer_zh=False),
            "In-text citation context:",
        ]
    elif naive_source_trace:
        target_label = _source_trace_target_label(prompt)
        lines = [
            f"No. {target_label} is cited here as existing prior work {ref_label}, not introduced as an original contribution of the current paper.",
            _source_location_label(heading_path, prefer_zh=False),
            "In-text citation context:",
        ]
    else:
        lines = [
            f"Use {ref_label} as the cited source for this passage.",
            _source_location_label(heading_path, prefer_zh=False),
        ]
    lines.append(f"> {locate_anchor}")

    rec_out = dict(rec or {})
    if ref_nums and (not list(rec_out.get("candidate_refs") or [])):
        rec_out["candidate_refs"] = list(ref_nums[:4])
    if ref_nums and int(ref_nums[0]) > 0 and int(rec_out.get("resolved_ref_num") or 0) <= 0:
        rec_out["resolved_ref_num"] = int(ref_nums[0])

    return _finalize_skill_result(
        "\n".join(lines).strip(),
        support_resolution=[rec_out],
        has_hits=has_hits,
        prompt_text=prompt,
        prompt_family=prompt_family,
        sanitize_answer=sanitize_answer,
    )


def run_exact_figure_panel_skill(
    *,
    prompt_text: str,
    prompt_family: str,
    source_path: str,
    db_dir: Path | None,
    has_hits: bool,
    prompt_requests_exact_support: PromptPredicateFn,
    resolve_exact_support: ResolveRecordFn,
    extract_clause_reference_numbers: ExtractRefNumsFn,
    sanitize_answer: SanitizeAnswerFn,
) -> PaperGuideSkillResult | None:
    prompt = str(prompt_text or "").strip()
    if (not prompt) or (not prompt_requests_exact_support(prompt)):
        return None

    rec = resolve_exact_support(
        str(source_path or "").strip(),
        prompt=prompt,
        db_dir=db_dir,
    )
    locate_anchor = str((rec or {}).get("locate_anchor") or "").strip()
    heading_path = str((rec or {}).get("heading_path") or "").strip()
    panel_letters = [
        str(ch or "").strip().lower()
        for ch in list((rec or {}).get("panel_letters") or [])
        if str(ch or "").strip()
    ]
    try:
        fig_num = int((rec or {}).get("figure_number") or 0)
    except Exception:
        fig_num = 0
    if not locate_anchor:
        return None

    prefer_zh = _contains_cjk(prompt)
    if prefer_zh:
        if fig_num > 0 and panel_letters:
            prefix = f"图 {int(fig_num)}({panel_letters[0]}) 的图注原文片段："
        elif fig_num > 0:
            prefix = f"图 {int(fig_num)} 的图注原文片段："
        else:
            prefix = "图注原文片段："
    else:
        if fig_num > 0 and panel_letters:
            prefix = f"Caption clause for Figure {int(fig_num)}({panel_letters[0]}):"
        elif fig_num > 0:
            prefix = f"Caption clause for Figure {int(fig_num)}:"
        else:
            prefix = "Caption clause:"

    lines = [prefix]
    if heading_path:
        lines.append(_source_location_label(heading_path, prefer_zh=prefer_zh))
    lines.append(f"> {locate_anchor}")

    ref_nums = _positive_ints(
        extract_clause_reference_numbers(locate_anchor, max_nums=4),
        limit=4,
    )
    if ref_nums:
        label = ", ".join(f"[{int(n)}]" for n in ref_nums if int(n) > 0)
        if label:
            lines.append(f"该片段里的文内来源：{label}" if prefer_zh else f"References in this clause: {label}")

    rec_out = dict(rec or {})
    if ref_nums and (not list(rec_out.get("candidate_refs") or [])):
        rec_out["candidate_refs"] = list(ref_nums)
        if len(ref_nums) == 1:
            rec_out["resolved_ref_num"] = int(ref_nums[0])

    return _finalize_skill_result(
        "\n".join(lines).strip(),
        support_resolution=[rec_out],
        has_hits=has_hits,
        prompt_text=prompt,
        prompt_family=prompt_family,
        sanitize_answer=sanitize_answer,
    )


def run_exact_method_skill(
    *,
    prompt_text: str,
    prompt_family: str,
    source_path: str,
    db_dir: Path | None,
    has_hits: bool,
    prompt_requests_exact_support: PromptPredicateFn,
    resolve_exact_support: ResolveRecordFn,
    sanitize_answer: SanitizeAnswerFn,
) -> PaperGuideSkillResult | None:
    prompt = str(prompt_text or "").strip()
    if (not prompt) or (not prompt_requests_exact_support(prompt)):
        return None

    rec = resolve_exact_support(
        str(source_path or "").strip(),
        prompt=prompt,
        db_dir=db_dir,
    )
    locate_anchor = str((rec or {}).get("locate_anchor") or "").strip()
    heading_path = str((rec or {}).get("heading_path") or "").strip()
    if not locate_anchor:
        return None

    if _contains_cjk(prompt):
        out = f"{_source_location_label(heading_path, prefer_zh=True)}\n> {locate_anchor}"
    else:
        out = f"{_source_location_label(heading_path, prefer_zh=False)}\n> {locate_anchor}"

    rec_out = dict(rec or {})
    rec_out.setdefault("segment_text", locate_anchor)
    rec_out["segment_index"] = -1

    return _finalize_skill_result(
        out,
        support_resolution=[rec_out],
        has_hits=has_hits,
        prompt_text=prompt,
        prompt_family=prompt_family,
        sanitize_answer=sanitize_answer,
    )


def run_overview_component_role_skill(
    *,
    prompt_text: str,
    prompt_family: str,
    source_path: str,
    db_dir: Path | None,
    has_hits: bool,
    prompt_requests_component_role_explanation: PromptPredicateFn,
    extract_method_focus_terms: ExtractFocusTermsFn,
    extract_component_role_focus: ExtractFocusTextFn,
    build_overview_role_lines: BuildLinesFn,
    sanitize_answer: SanitizeAnswerFn | None = None,
) -> PaperGuideSkillResult | None:
    prompt = str(prompt_text or "").strip()
    family = str(prompt_family or "").strip().lower()
    if family not in {"overview", "method"}:
        return None
    if (not prompt) or (not prompt_requests_component_role_explanation(prompt)):
        return None

    focus_terms = list(extract_method_focus_terms(prompt) or [])
    focus_excerpt = str(
        extract_component_role_focus(
            str(source_path or "").strip(),
            db_dir=db_dir,
            focus_terms=focus_terms,
        )
        or ""
    ).strip()
    if not focus_excerpt:
        return None

    role_lines = [
        str(line or "").strip()
        for line in list(
            build_overview_role_lines(
                focus_excerpt,
                focus_terms=focus_terms,
            )
            or []
        )
        if str(line or "").strip()
    ]
    if len(role_lines) < 2:
        return None

    intro = "简单说：" if _contains_cjk(prompt) else "In simple terms:"
    return _finalize_skill_result(
        intro + "\n- " + "\n- ".join(role_lines),
        support_resolution=[
            _build_skill_support_record(
                source_path=source_path,
                locate_anchor=focus_excerpt,
                claim_type="overview_component_role",
            )
        ],
        has_hits=has_hits,
        prompt_text=prompt,
        prompt_family=family,
        sanitize_answer=sanitize_answer,
    )


def run_section_target_skill(
    *,
    prompt_text: str,
    prompt_family: str,
    source_path: str,
    db_dir: Path | None,
    has_hits: bool,
    has_non_ref_target: bool,
    select_target_hit: SelectHitFn,
    extract_discussion_future_snippet: ExtractSnippetFn,
    sanitize_answer: SanitizeAnswerFn | None = None,
) -> PaperGuideSkillResult | None:
    prompt = str(prompt_text or "").strip()
    family = str(prompt_family or "").strip().lower()
    if family not in {"strength_limits", "discussion_only"}:
        return None
    if (not prompt) or (not has_non_ref_target):
        return None

    target_hit = dict(
        select_target_hit(
            str(source_path or "").strip(),
            prompt=prompt,
            prompt_family=family,
            db_dir=db_dir,
        )
        or {}
    )
    if not target_hit:
        return None

    meta = target_hit.get("meta", {}) or {}
    locate_anchor = str(target_hit.get("text") or "").strip()
    heading_path = str(meta.get("heading_path") or "").strip()
    heading_leaf = str(heading_path.split(" / ")[-1] or "").strip() or "target section"
    if not locate_anchor:
        return None

    support_anchor = locate_anchor
    if family == "discussion_only":
        support_anchor = str(extract_discussion_future_snippet(locate_anchor) or locate_anchor).strip()
        answer_text = support_anchor
    elif family == "strength_limits":
        if _contains_cjk(prompt):
            answer_text = f"{heading_leaf} 这一节里的关键限制/取舍是：\n> {locate_anchor}"
        else:
            answer_text = f"Key limitation or trade-off from {heading_leaf}:\n> {locate_anchor}"
    else:
        if _contains_cjk(prompt):
            answer_text = f"{heading_leaf} 这一节的相关原文是：\n> {locate_anchor}"
        else:
            answer_text = f"Relevant passage from {heading_leaf}:\n> {locate_anchor}"

    return _finalize_skill_result(
        answer_text,
        support_resolution=[
            _build_skill_support_record(
                source_path=source_path,
                heading_path=heading_path,
                locate_anchor=support_anchor,
                claim_type=family,
                extra=meta if isinstance(meta, dict) else {},
            )
        ],
        has_hits=has_hits,
        prompt_text=prompt,
        prompt_family=family,
        sanitize_answer=sanitize_answer,
    )


def run_abstract_skill(
    *,
    prompt_text: str,
    prompt_family: str,
    source_path: str,
    db_dir: Path | None,
    llm=None,
    build_direct_abstract_answer: BuildDirectAnswerFn,
    sanitize_answer: SanitizeAnswerFn | None = None,
) -> PaperGuideSkillResult | None:
    prompt = str(prompt_text or "").strip()
    family = str(prompt_family or "").strip().lower()
    if family != "abstract" or (not prompt):
        return None

    answer_text = str(
        build_direct_abstract_answer(
            prompt=prompt,
            source_path=str(source_path or "").strip(),
            db_dir=db_dir,
            llm=llm,
        )
        or ""
    ).strip()
    if not answer_text:
        return None

    anchor = _extract_block_quote(answer_text, label="Anchor sentence for locate jump:")
    return _finalize_skill_result(
        answer_text,
        support_resolution=[
            _build_skill_support_record(
                source_path=source_path,
                locate_anchor=anchor,
                claim_type="abstract",
            )
        ],
        has_hits=False,
        prompt_text=prompt,
        prompt_family=family,
        sanitize_answer=sanitize_answer,
    )


def run_box_target_skill(
    *,
    prompt_text: str,
    prompt_family: str,
    source_path: str,
    db_dir: Path | None,
    has_hits: bool,
    has_non_ref_target: bool,
    select_target_hit: SelectHitFn,
    extract_box_numbers: ExtractNumsFn,
    sanitize_answer: SanitizeAnswerFn | None = None,
) -> PaperGuideSkillResult | None:
    prompt = str(prompt_text or "").strip()
    family = str(prompt_family or "").strip().lower()
    if family != "box_only" or (not prompt) or (not has_non_ref_target):
        return None

    box_nums = _positive_ints(extract_box_numbers(prompt), limit=3)
    if not box_nums:
        return None

    target_hit = dict(
        select_target_hit(
            str(source_path or "").strip(),
            prompt=prompt,
            prompt_family=family,
            db_dir=db_dir,
        )
        or {}
    )
    if not target_hit:
        return None

    meta = target_hit.get("meta", {}) or {}
    locate_anchor = str(target_hit.get("text") or "").strip()
    heading_path = str(meta.get("heading_path") or "").strip()
    if not locate_anchor:
        return None

    label = f"Box {int(box_nums[0])}"
    if _contains_cjk(prompt):
        answer_text = f"{label} 里的相关原文是：\n> {locate_anchor}"
    else:
        answer_text = f"Relevant passage from {label}:\n> {locate_anchor}"
    return _finalize_skill_result(
        answer_text,
        support_resolution=[
            _build_skill_support_record(
                source_path=source_path,
                heading_path=heading_path,
                locate_anchor=locate_anchor,
                claim_type="box_only",
                extra=meta if isinstance(meta, dict) else {},
            )
        ],
        has_hits=has_hits,
        prompt_text=prompt,
        prompt_family=family,
        sanitize_answer=sanitize_answer,
    )


__all__ = [
    "PaperGuideSkillResult",
    "run_exact_citation_lookup_skill",
    "run_exact_equation_skill",
    "run_exact_figure_panel_skill",
    "run_exact_method_skill",
    "run_abstract_skill",
    "run_box_target_skill",
    "run_overview_component_role_skill",
    "run_section_target_skill",
]
