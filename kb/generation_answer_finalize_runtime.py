from __future__ import annotations

import os
import re
from pathlib import Path

from kb.answer_contract import (
    _apply_answer_contract_v1,
    _build_answer_quality_probe,
    _enhance_kb_miss_fallback,
    _reconcile_kb_notice,
)
from kb.paper_guide_contracts import (
    _build_paper_guide_render_packet_model,
    _build_paper_guide_retrieval_bundle_model,
    _build_paper_guide_support_pack_model,
    _paper_guide_grounding_trace_segment_model_from_raw,
    _paper_guide_model_dump,
)
from kb.paper_guide.router import _resolve_paper_guide_intent
from kb.paper_guide_postprocess import (
    _sanitize_paper_guide_answer_for_user,
    _sanitize_structured_cite_tokens,
    _strip_model_ref_section,
)
from kb.source_blocks import normalize_inline_markdown
from ui.chat_widgets import _normalize_math_markdown

_CITE_CANON_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*(\d{1,4})\s*\]\]",
    re.IGNORECASE,
)
_STRUCT_CITE_SINGLE_RE = re.compile(
    r"(?<!\[)\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})(?:\s*:\s*(\d{1,4}))?\s*\](?!\])",
    re.IGNORECASE,
)
_STRUCT_CITE_SID_ONLY_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*\]\]",
    re.IGNORECASE,
)
_STRUCT_CITE_GARBAGE_RE = re.compile(r"\[\[?\s*CITE\s*:[^\]\n]*\]?\]", re.IGNORECASE)
_SID_INLINE_RE = re.compile(r"\[\s*SID\s*:\s*[A-Za-z0-9_-]{4,24}\s*\]", re.IGNORECASE)
_SID_RE = re.compile(r"^[A-Za-z0-9_-]{4,24}$")
_INLINE_REF_NUM_RE = re.compile(r"\[(\d{1,4})\]")
_FREEFORM_NUMERIC_CITE_RE = re.compile(
    r"(?<![!\\])\[(\d{1,4}(?:\s*(?:-|–|—|,)\s*\d{1,4})*)\](?!\()"
)
_PAPER_GUIDE_NEGATIVE_SHELL_RE = re.compile(
    r"(?i)\b(?:not stated|does not state|do not state|does not specify|do not specify|"
    r"does not discuss|do not discuss|does not mention|do not mention|makes no statement|"
    r"cannot be determined from the retrieved)\b"
)
_PAPER_GUIDE_SUPPLEMENT_BLOCK_MARKER_RE = re.compile(
    r"(?i)(补充说明（通用知识，非检索片段内容|supplementary note \(generic knowledge, non-retrieved content\))"
)
_PAPER_GUIDE_SUPPLEMENT_OPTOUT_RE = re.compile(
    r"(?i)(只基于原文|仅基于原文|不要补充|不要扩展|不要通用知识|only from the paper|paper-only|no supplement|no general knowledge)"
)
_PAPER_GUIDE_SUPPLEMENT_DISCLAIMER_RE = re.compile(
    r"(?i)(以下内容是\s*AI\s*基于通用知识的补充|"
    r"不代表论文原文明确陈述|"
    r"the notes below are ai supplemental context|"
    r"not explicit claims from the paper)"
)
_STRUCTURED_ANSWER_SECTION_RE = re.compile(
    r"(?im)^\s*(Conclusion|Evidence|Limits|Next Steps|结论|依据|证据|边界|限制|局限|下一步建议|下一步)\s*[:：]"
)
_CROSS_PAPER_QUERY_RE = re.compile(
    r"(\bwhich other papers?\b|\bother papers?\b|\bbesides this paper\b|\banother paper\b|"
    r"除此之外|除(?:了)?这篇|其他论文|别的论文|还有哪些论文|另一篇论文)",
    flags=re.IGNORECASE,
)


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def _as_positive_int(value: object) -> int:
    try:
        n = int(value)
    except Exception:
        return 0
    return n if n > 0 else 0


def _collect_low_confidence_candidate_refs(
    *,
    support_resolution: list[dict] | None,
    candidate_refs_by_source: dict[str, list[int]] | None,
    retrieval_confidence_hint: dict[str, object] | None,
    max_items: int = 6,
) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()

    def _add(value: object) -> None:
        n = _as_positive_int(value)
        if n <= 0 or n in seen:
            return
        seen.add(n)
        out.append(n)

    for rec in list(support_resolution or []):
        if not isinstance(rec, dict):
            continue
        _add(rec.get("resolved_ref_num"))
        for key in ("candidate_refs", "support_ref_candidates", "ref_nums"):
            for item in list(rec.get(key) or []):
                _add(item)

    for refs in list((candidate_refs_by_source or {}).values()):
        for item in list(refs or []):
            _add(item)

    hint = dict(retrieval_confidence_hint or {})
    for item in list(hint.get("candidate_refs") or []):
        _add(item)
    for key in ("resolved_ref_num", "top_ref_num"):
        _add(hint.get(key))

    return [int(n) for n in out[: max(1, int(max_items or 6))] if int(n) > 0]


def _has_structured_cite_marker(text: str) -> bool:
    return bool(_CITE_CANON_RE.search(str(text or "")))


def _collect_inline_reference_numbers(text: str, *, max_items: int = 6) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for m in _INLINE_REF_NUM_RE.finditer(str(text or "")):
        n = _as_positive_int(m.group(1))
        if n <= 0 or n in seen:
            continue
        seen.add(n)
        out.append(n)
        if len(out) >= max(1, int(max_items or 6)):
            break
    return out


def _prompt_explicitly_requests_citation_lookup(prompt: str) -> bool:
    text = str(prompt or "").strip().lower()
    if not text:
        return False
    patterns = (
        "citation",
        "cited",
        "cite",
        "reference number",
        "reference numbers",
        "which reference",
        "which references",
        "what in-paper citation",
        "prior work is",
        "attributed to",
        "引用",
        "引文",
        "参考文献",
        "编号",
    )
    return any(pattern in text for pattern in patterns)


def _should_preserve_final_answer_numeric_citations(
    *,
    prompt: str,
    answer_output_mode: str,
    paper_guide_mode: bool,
    prompt_family: str,
) -> bool:
    if str(prompt_family or "").strip().lower() == "citation_lookup":
        return True
    if "citation" in str(answer_output_mode or "").strip().lower():
        return True
    if paper_guide_mode and _prompt_explicitly_requests_citation_lookup(prompt):
        return True
    return False


def _strip_final_answer_citation_markers(answer: str, *, preserve_numeric_markers: bool) -> str:
    text = str(answer or "")
    if not text:
        return text
    out = _sanitize_structured_cite_tokens(text)
    out = _CITE_CANON_RE.sub("", out)
    out = _STRUCT_CITE_SINGLE_RE.sub("", out)
    out = _STRUCT_CITE_SID_ONLY_RE.sub("", out)
    out = _STRUCT_CITE_GARBAGE_RE.sub("", out)
    out = _SID_INLINE_RE.sub("", out)
    if not preserve_numeric_markers:
        out = _FREEFORM_NUMERIC_CITE_RE.sub("", out)
    out = re.sub(r"[ \t]+([,.;:!?])", r"\1", out)
    out = re.sub(r"(?m)[ \t]{2,}", " ", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _select_minimum_paper_guide_ref_num(
    *,
    answer: str,
    support_resolution: list[dict] | None,
    candidate_refs_by_source: dict[str, list[int]] | None,
    retrieval_confidence_hint: dict[str, object] | None,
) -> int:
    inline_refs = _collect_inline_reference_numbers(answer, max_items=6)
    if inline_refs:
        return int(inline_refs[0])
    refs = _collect_low_confidence_candidate_refs(
        support_resolution=support_resolution,
        candidate_refs_by_source=candidate_refs_by_source,
        retrieval_confidence_hint=retrieval_confidence_hint,
        max_items=6,
    )
    return int(refs[0]) if refs else 0


def _select_minimum_paper_guide_sid(
    *,
    support_resolution: list[dict] | None,
    locked_citation_source: dict | None,
) -> str:
    for rec in list(support_resolution or []):
        if not isinstance(rec, dict):
            continue
        sid = str(rec.get("sid") or "").strip()
        if sid and _SID_RE.match(sid):
            return sid
    locked_sid = str((locked_citation_source or {}).get("sid") or "").strip()
    if locked_sid and _SID_RE.match(locked_sid):
        return locked_sid
    return ""


def _maybe_ensure_minimum_paper_guide_citation(
    answer: str,
    *,
    paper_guide_mode: bool,
    prompt_family: str = "",
    has_hits: bool,
    support_resolution: list[dict] | None = None,
    candidate_refs_by_source: dict[str, list[int]] | None = None,
    retrieval_confidence_hint: dict[str, object] | None = None,
    locked_citation_source: dict | None = None,
) -> str:
    text = str(answer or "").strip()
    family = str(prompt_family or "").strip().lower()
    if not text:
        return text
    if not paper_guide_mode or not has_hits:
        return text
    if family and family not in {"citation_lookup"}:
        return text
    if _has_structured_cite_marker(text):
        return text
    # Keep negative shells citation-free to avoid implying unsupported absence claims.
    if _PAPER_GUIDE_NEGATIVE_SHELL_RE.search(text):
        return text
    sid = _select_minimum_paper_guide_sid(
        support_resolution=support_resolution,
        locked_citation_source=locked_citation_source,
    )
    if not sid:
        return text
    ref_num = _select_minimum_paper_guide_ref_num(
        answer=text,
        support_resolution=support_resolution,
        candidate_refs_by_source=candidate_refs_by_source,
        retrieval_confidence_hint=retrieval_confidence_hint,
    )
    if ref_num <= 0:
        return text
    return f"{text} [[CITE:{sid}:{int(ref_num)}]]"


def _maybe_prepend_paper_guide_low_confidence_notice(
    answer: str,
    *,
    paper_guide_mode: bool,
    prompt_text: str,
    prompt_family: str,
    retrieval_confidence_hint: dict[str, object] | None,
    support_resolution: list[dict] | None = None,
    candidate_refs_by_source: dict[str, list[int]] | None = None,
) -> str:
    text = str(answer or "").strip()
    if not text:
        return text
    if not paper_guide_mode:
        return text
    hint = dict(retrieval_confidence_hint or {})
    if not hint:
        return text
    if not bool(hint.get("low_confidence")):
        return text
    try:
        enabled = bool(int(str(os.environ.get("KB_PAPER_GUIDE_LOW_CONF_NOTICE", "1") or "1")))
    except Exception:
        enabled = True
    if not enabled:
        return text
    lowered = text.lower()
    if ("low confidence" in lowered) or ("低置信" in text):
        return text
    reason = str(hint.get("low_confidence_reason") or hint.get("force_rescue_reason") or "").strip()
    if not reason:
        reason = "weak_evidence_alignment"
    reason_map_en = {
        "empty_hits": "no scoped evidence was retrieved",
        "target_miss": "the requested target section was not matched directly",
        "reference_only_hits": "retrieval mostly returned reference-like snippets",
        "weak_signal": "retrieval signal is weak for the requested claim",
        "strict_family_without_targeted_support": "strict question type lacks targeted support",
        "strict_family_weak_overlap": "strict question type has weak lexical overlap",
        "strict_family_sparse_hits": "strict question type has sparse evidence hits",
        "broad_family_weak_overlap": "broad summary question has weak evidence overlap",
    }
    reason_map_zh = {
        "empty_hits": "未检索到同文证据片段",
        "target_miss": "未直接命中你指定的目标段落",
        "reference_only_hits": "检索结果主要是参考文献样式片段",
        "weak_signal": "针对该问题的证据信号偏弱",
        "strict_family_without_targeted_support": "严格问题类型缺少定向证据支撑",
        "strict_family_weak_overlap": "严格问题类型与证据词重叠较弱",
        "strict_family_sparse_hits": "严格问题类型命中证据过少",
        "broad_family_weak_overlap": "概览类问题与证据重叠较弱",
    }
    family = str(prompt_family or "").strip().lower()
    if family in {"abstract"}:
        return text
    is_zh = _contains_cjk(prompt_text)
    if is_zh:
        reason_msg = reason_map_zh.get(reason, reason)
        notice = f"提示：当前回答基于低置信证据匹配（{reason_msg}）。建议点击“定位到原文证据”核对关键句。"
    else:
        reason_msg = reason_map_en.get(reason, reason.replace("_", " "))
        notice = (
            f"Note: this answer is based on lower-confidence evidence matching ({reason_msg}). "
            f"Please verify key claims via locate-to-source evidence."
        )
    candidate_refs = _collect_low_confidence_candidate_refs(
        support_resolution=support_resolution,
        candidate_refs_by_source=candidate_refs_by_source,
        retrieval_confidence_hint=hint,
        max_items=6,
    )
    if candidate_refs:
        refs_text = ", ".join(f"[{int(n)}]" for n in candidate_refs if int(n) > 0)
        if refs_text:
            if is_zh:
                notice += f" 候选参考文献：{refs_text}（供交叉核对）。"
            else:
                notice += f" Candidate refs for cross-check: {refs_text}."
    return f"{notice}\n\n{text}"


def _build_paper_guide_supplement_lines(*, prompt_family: str, prefer_zh: bool) -> list[str]:
    family = str(prompt_family or "").strip().lower()
    if prefer_zh:
        if family == "citation_lookup":
            return [
                "引用问题应以文内编号与参考文献列表为准，通用背景不能替代原始引用链。",
                "若仍不稳定，建议继续追问“具体术语 + 句子位置”以触发更窄范围定位。",
            ]
        if family in {"method", "reproduce"}:
            return [
                "方法理解通常要把“输入/输出、关键模块、训练设定、适用边界”分开核对。",
                "用于实验前，建议把本段补充与可定位原文逐条对照后再采用。",
            ]
        if family in {"equation", "figure_walkthrough", "box_only"}:
            return [
                "公式/图示解读常依赖上下文定义，单句解释可能遗漏符号约束与实验条件。",
                "若要用于结论，请优先以可定位的原文片段为准。",
            ]
        return [
            "以下内容用于帮助理解领域背景，不等同于论文原文已明确陈述。",
            "需要用于结论时，请以可定位的原文证据为准。",
        ]
    if family == "citation_lookup":
        return [
            "Reference questions should be decided by in-paper numbering and the reference list, not by generic background.",
            "If grounding is still weak, ask with exact terms plus sentence scope to trigger narrower locate matching.",
        ]
    if family in {"method", "reproduce"}:
        return [
            "Method understanding is more reliable when input/output, key modules, training setup, and failure boundaries are checked separately.",
            "Before applying this in experiments, map each supplemental point to a locate-able source sentence.",
        ]
    if family in {"equation", "figure_walkthrough", "box_only"}:
        return [
            "Equation/figure interpretation often depends on nearby definitions; a single sentence can miss constraints.",
            "Use locate-able paper evidence as the final authority for decisions.",
        ]
    return [
        "The notes below are general background to aid understanding, not explicit paper-verified claims.",
        "For final conclusions, prioritize locate-able source evidence.",
    ]


def _normalize_paper_guide_supplement_lines(
    raw_lines: object,
    *,
    max_items: int = 3,
) -> list[str]:
    if isinstance(raw_lines, (list, tuple)):
        text = "\n".join(str(item or "") for item in raw_lines)
    else:
        text = str(raw_lines or "")
    text = str(text or "").strip()
    if not text:
        return []

    text = re.sub(r"```(?:markdown|md|text)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    text = _PAPER_GUIDE_SUPPLEMENT_BLOCK_MARKER_RE.sub("", text)
    text = _PAPER_GUIDE_SUPPLEMENT_DISCLAIMER_RE.sub("", text)

    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        s = str(line or "").strip()
        if not s:
            continue
        s = re.sub(r"^\s*>\s*", "", s)
        s = re.sub(r"^\s*#{1,6}\s*", "", s)
        s = re.sub(r"^\s*\*\*(.*?)\*\*\s*$", r"\1", s)
        s = re.sub(r"^\s*\d+[.)]\s*", "- ", s)
        if re.match(r"^\s*[*-]\s+", s):
            s = "- " + re.sub(r"^\s*[*-]\s+", "", s).strip()
        s = _CITE_CANON_RE.sub("", s)
        s = re.sub(r"\[(\d{1,4})\]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        if (not s) or _PAPER_GUIDE_SUPPLEMENT_DISCLAIMER_RE.search(s):
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max(1, int(max_items or 3)):
            break

    if out:
        return out

    flat = re.sub(r"\s+", " ", text).strip()
    if not flat:
        return []
    flat = _CITE_CANON_RE.sub("", flat)
    flat = re.sub(r"\[(\d{1,4})\]", "", flat)
    flat = re.sub(r"\s+", " ", flat).strip()
    if not flat:
        return []
    return [flat[:280].rstrip()]


def _count_paper_guide_supportive_segments(support_resolution: list[dict] | None) -> int:
    count = 0
    for rec in list(support_resolution or []):
        if not isinstance(rec, dict):
            continue
        if any(
            str(rec.get(key) or "").strip()
            for key in ("locate_anchor", "evidence_quote", "segment_text", "anchor_text", "primary_block_id")
        ) or _as_positive_int(rec.get("resolved_ref_num")) > 0:
            count += 1
    return count


def _should_append_paper_guide_supplement(
    *,
    answer: str,
    prompt_family: str,
    retrieval_confidence_hint: dict[str, object] | None,
    support_resolution: list[dict] | None,
) -> bool:
    hint = dict(retrieval_confidence_hint or {})
    if bool(hint.get("low_confidence")):
        return True
    family = str(prompt_family or "").strip().lower()
    support_count = _count_paper_guide_supportive_segments(support_resolution)
    explanation_family = family in {
        "method",
        "reproduce",
        "equation",
        "figure_walkthrough",
        "overview",
        "compare",
        "strength_limits",
        "box_only",
        "discussion_only",
    }
    if explanation_family and support_count <= 1 and _PAPER_GUIDE_NEGATIVE_SHELL_RE.search(str(answer or "")):
        return True
    return False


def _maybe_append_paper_guide_supplement_block(
    answer: str,
    *,
    paper_guide_mode: bool,
    has_hits: bool,
    prompt_text: str,
    prompt_family: str,
    retrieval_confidence_hint: dict[str, object] | None,
    grounded_answer: str = "",
    support_resolution: list[dict] | None = None,
    build_paper_guide_supplement_lines=None,
) -> str:
    text = str(answer or "").strip()
    if not text:
        return text
    if not paper_guide_mode:
        return text
    if not has_hits:
        return text
    try:
        enabled = bool(int(str(os.environ.get("KB_PAPER_GUIDE_SUPPLEMENT_BLOCK", "1") or "1")))
    except Exception:
        enabled = True
    if not enabled:
        return text
    if _PAPER_GUIDE_SUPPLEMENT_BLOCK_MARKER_RE.search(text):
        return text
    if _PAPER_GUIDE_SUPPLEMENT_OPTOUT_RE.search(str(prompt_text or "")):
        return text
    if _STRUCTURED_ANSWER_SECTION_RE.search(text):
        return text
    if _CROSS_PAPER_QUERY_RE.search(str(prompt_text or "")):
        return text
    # When the grounded answer is explicitly a "not stated / does not specify" response,
    # avoid adding generic supplement blocks. Users asking for a concrete paper detail
    # are better served by a short negative answer plus actionable paper-only next steps.
    grounded_norm = normalize_inline_markdown(str(grounded_answer or text)).lower()
    if re.search(r"(?i)\b(?:does not specify|does not mention|not stated|cannot be determined)\b", grounded_norm):
        q = str(prompt_text or "").strip().lower()
        # Skip for "hardware/compute spec" questions where generic supplement is usually noise.
        # Apply regardless of family inference because intent classifiers can vary.
        if any(
            tok in q
            for tok in (
                "gpu",
                "cuda",
                "nvidia",
                "rtx",
                "a100",
                "v100",
                "3090",
                "4090",
                "hardware",
                "compute",
                "device",
            )
        ):
            return text
    hint = dict(retrieval_confidence_hint or {})
    if not _should_append_paper_guide_supplement(
        answer=str(grounded_answer or text),
        prompt_family=str(prompt_family or ""),
        retrieval_confidence_hint=hint,
        support_resolution=list(support_resolution or []),
    ):
        return text
    prefer_zh = _contains_cjk(prompt_text)
    lines: list[str] = []
    if callable(build_paper_guide_supplement_lines):
        try:
            lines = _normalize_paper_guide_supplement_lines(
                build_paper_guide_supplement_lines(
                    prompt_text=str(prompt_text or ""),
                    grounded_answer=str(grounded_answer or text),
                    prompt_family=str(prompt_family or ""),
                    prefer_zh=bool(prefer_zh),
                    retrieval_confidence_hint=dict(hint),
                    support_resolution=list(support_resolution or []),
                ),
                max_items=3,
            )
        except Exception:
            lines = []
    if not lines:
        lines = _build_paper_guide_supplement_lines(prompt_family=prompt_family, prefer_zh=prefer_zh)
    if not lines:
        return text
    if prefer_zh:
        header = "> 补充说明（通用知识，非检索片段内容 / Supplementary note (generic knowledge, non-retrieved content)）："
        disclaimer = "> 以下内容是 AI 基于通用知识的补充，不代表论文原文明确陈述。"
    else:
        header = "> Supplementary note (generic knowledge, non-retrieved content / 补充说明（通用知识，非检索片段内容）):"
        disclaimer = "> The notes below are AI supplemental context and are not explicit claims from the paper."
    block = [header, disclaimer]
    block.extend(f"> - {line}" for line in lines[:3] if str(line or "").strip())
    return f"{text}\n\n" + "\n".join(block).strip()


def _build_paper_guide_contract_snapshot(
    *,
    paper_guide_mode: bool,
    intent_model,
    answer_markdown: str,
    final_answer_markdown: str,
    evidence_cards: list[dict] | None,
    candidate_refs_by_source: dict[str, list[int]] | None,
    support_slots: list[dict] | None,
    support_resolution: list[dict] | None,
    needs_supplement: bool,
    citation_validation: dict | None,
    paper_guide_contracts_seed: dict | None = None,
) -> dict:
    seed = dict(paper_guide_contracts_seed or {})
    primary_evidence = _pick_shared_primary_evidence(
        paper_guide_contracts_seed=paper_guide_contracts_seed,
        evidence_cards=evidence_cards,
    )
    render_packet_seed = seed.get("render_packet") if isinstance(seed.get("render_packet"), dict) else {}
    if (not paper_guide_mode) and (not primary_evidence) and (not render_packet_seed):
        return {}

    snapshot = {"version": 1}
    if not paper_guide_mode:
        render_packet_model = _build_paper_guide_render_packet_model(
            answer_markdown=str(render_packet_seed.get("answer_markdown") or final_answer_markdown or "").strip(),
            notice=str(render_packet_seed.get("notice") or "").strip(),
            rendered_body=str(render_packet_seed.get("rendered_body") or "").strip(),
            rendered_content=str(render_packet_seed.get("rendered_content") or "").strip(),
            copy_markdown=str(render_packet_seed.get("copy_markdown") or "").strip(),
            copy_text=str(render_packet_seed.get("copy_text") or "").strip(),
            cite_details=list(render_packet_seed.get("cite_details") or []),
            citation_validation=(
                render_packet_seed.get("citation_validation")
                if isinstance(render_packet_seed.get("citation_validation"), dict)
                else citation_validation
            ),
            locate_target=render_packet_seed.get("locate_target") if isinstance(render_packet_seed.get("locate_target"), dict) else {},
            reader_open=render_packet_seed.get("reader_open") if isinstance(render_packet_seed.get("reader_open"), dict) else {},
            provenance_segments=list(render_packet_seed.get("provenance_segments") or []),
            primary_evidence=primary_evidence,
        )
        render_packet_dump = _paper_guide_model_dump(render_packet_model)
        if any(render_packet_dump.values()):
            snapshot["render_packet"] = render_packet_dump
        if primary_evidence:
            snapshot["primary_evidence"] = dict(primary_evidence)
        return {
            key: value
            for key, value in snapshot.items()
            if value not in (None, "", [], {})
        }

    pack_records = list(support_resolution or []) or list(support_slots or [])
    support_pack_model = _build_paper_guide_support_pack_model(
        family=str(getattr(intent_model, "family", "") or "").strip(),
        answer_markdown=str(answer_markdown or "").strip(),
        support_records=pack_records,
        needs_supplement=bool(needs_supplement),
    )
    grounding_trace = [
        _paper_guide_model_dump(_paper_guide_grounding_trace_segment_model_from_raw(item))
        for item in list(support_resolution or [])
        if isinstance(item, dict)
    ]
    snapshot = {
        "version": 1,
        "intent": _paper_guide_model_dump(intent_model),
        "support_pack": _paper_guide_model_dump(support_pack_model),
        "grounding_trace": grounding_trace,
    }
    retrieval_bundle = seed.get("retrieval_bundle") if isinstance(seed.get("retrieval_bundle"), dict) else {}
    if retrieval_bundle:
        snapshot["retrieval_bundle"] = dict(retrieval_bundle)
    else:
        prompt_context_seed = seed.get("prompt_context") if isinstance(seed.get("prompt_context"), dict) else {}
        retrieval_bundle_model = _build_paper_guide_retrieval_bundle_model(
            prompt_family=str(getattr(intent_model, "family", "") or "").strip(),
            target_scope=prompt_context_seed.get("target_scope") if isinstance(prompt_context_seed.get("target_scope"), dict) else {},
            evidence_cards=list(evidence_cards or []),
            candidate_refs_by_source=dict(candidate_refs_by_source or {}),
            direct_source_path=str(prompt_context_seed.get("direct_source_path") or "").strip(),
            focus_source_path=str(prompt_context_seed.get("focus_source_path") or "").strip(),
            bound_source_path=str(prompt_context_seed.get("bound_source_path") or "").strip(),
        )
        retrieval_bundle_dump = _paper_guide_model_dump(retrieval_bundle_model)
        if any(retrieval_bundle_dump.values()):
            snapshot["retrieval_bundle"] = retrieval_bundle_dump
    prompt_context = seed.get("prompt_context") if isinstance(seed.get("prompt_context"), dict) else {}
    if prompt_context:
        snapshot["prompt_context"] = dict(prompt_context)
    render_packet_model = _build_paper_guide_render_packet_model(
        answer_markdown=str(render_packet_seed.get("answer_markdown") or final_answer_markdown or "").strip(),
        notice=str(render_packet_seed.get("notice") or "").strip(),
        rendered_body=str(render_packet_seed.get("rendered_body") or "").strip(),
        rendered_content=str(render_packet_seed.get("rendered_content") or "").strip(),
        copy_markdown=str(render_packet_seed.get("copy_markdown") or "").strip(),
        copy_text=str(render_packet_seed.get("copy_text") or "").strip(),
        cite_details=list(render_packet_seed.get("cite_details") or []),
        citation_validation=(
            render_packet_seed.get("citation_validation")
            if isinstance(render_packet_seed.get("citation_validation"), dict)
            else citation_validation
        ),
        locate_target=render_packet_seed.get("locate_target") if isinstance(render_packet_seed.get("locate_target"), dict) else {},
        reader_open=render_packet_seed.get("reader_open") if isinstance(render_packet_seed.get("reader_open"), dict) else {},
        provenance_segments=list(render_packet_seed.get("provenance_segments") or []),
        primary_evidence=primary_evidence,
    )
    render_packet_dump = _paper_guide_model_dump(render_packet_model)
    if any(render_packet_dump.values()):
        snapshot["render_packet"] = render_packet_dump
    if primary_evidence:
        snapshot["primary_evidence"] = dict(primary_evidence)
    return {
        key: value
        for key, value in snapshot.items()
        if value not in (None, "", [], {})
    }


def _pick_shared_primary_evidence(
    *,
    paper_guide_contracts_seed: dict | None,
    evidence_cards: list[dict] | None,
) -> dict:
    seed = dict(paper_guide_contracts_seed or {})
    primary = seed.get("primary_evidence")
    if isinstance(primary, dict) and primary:
        return dict(primary)
    for card in list(evidence_cards or []):
        if not isinstance(card, dict):
            continue
        primary = card.get("primary_evidence")
        if isinstance(primary, dict) and primary:
            return dict(primary)
    return {}


def _finalize_generation_answer(
    partial: str,
    *,
    prompt: str,
    prompt_for_user: str,
    answer_hits: list[dict],
    db_dir: Path | None,
    locked_citation_source: dict | None,
    answer_intent: str,
    answer_depth: str,
    answer_output_mode: str,
    paper_guide_mode: bool,
    paper_guide_contract_enabled: bool,
    paper_guide_prompt_family: str,
    paper_guide_special_focus_block: str,
    paper_guide_focus_source_path: str,
    paper_guide_direct_source_path: str,
    paper_guide_bound_source_path: str,
    paper_guide_candidate_refs_by_source: dict[str, list[int]] | None,
    paper_guide_support_slots: list[dict] | None,
    paper_guide_evidence_cards: list[dict] | None,
    paper_guide_contracts_seed: dict | None = None,
    paper_guide_retrieval_confidence_hint: dict[str, object] | None = None,
    apply_paper_guide_answer_postprocess,
    maybe_append_library_figure_markdown,
    validate_structured_citations,
    build_paper_guide_supplement_lines=None,
) -> dict:
    resolved_paper_guide_intent = _resolve_paper_guide_intent(
        prompt_for_user or prompt,
        prompt_family=paper_guide_prompt_family,
    )
    effective_paper_guide_family = str(getattr(resolved_paper_guide_intent, "family", "") or "").strip()
    sanitize_paper_guide_family = effective_paper_guide_family or "overview"
    answer = _normalize_math_markdown(
        _strip_model_ref_section(_sanitize_structured_cite_tokens(partial or ""))
    ).strip() or "(No text returned)"
    answer = _reconcile_kb_notice(answer, has_hits=bool(answer_hits))
    shared_primary_evidence = _pick_shared_primary_evidence(
        paper_guide_contracts_seed=dict(paper_guide_contracts_seed or {}),
        evidence_cards=list(paper_guide_evidence_cards or []),
    )
    if paper_guide_contract_enabled:
        answer = _apply_answer_contract_v1(
            answer,
            prompt=prompt,
            has_hits=bool(answer_hits),
            answer_hits=answer_hits,
            primary_evidence=shared_primary_evidence,
            intent=answer_intent,
            depth=answer_depth,
            output_mode=answer_output_mode,
        )
    answer = _enhance_kb_miss_fallback(
        answer,
        has_hits=bool(answer_hits),
        intent=answer_intent,
        depth=answer_depth,
        contract_enabled=bool(paper_guide_contract_enabled),
        output_mode=answer_output_mode,
    )
    answer, paper_guide_support_resolution = apply_paper_guide_answer_postprocess(
        answer,
        paper_guide_mode=paper_guide_mode,
        prompt=prompt,
        prompt_for_user=prompt_for_user,
        prompt_family=paper_guide_prompt_family,
        special_focus_block=paper_guide_special_focus_block,
        focus_source_path=paper_guide_focus_source_path,
        direct_source_path=paper_guide_direct_source_path,
        bound_source_path=paper_guide_bound_source_path,
        db_dir=db_dir,
        answer_hits=answer_hits,
        support_slots=list(paper_guide_support_slots or []),
        cards=list(paper_guide_evidence_cards or []),
        locked_citation_source=locked_citation_source,
    )
    answer = maybe_append_library_figure_markdown(
        answer,
        prompt=prompt,
        answer_hits=answer_hits,
        bound_source_path=paper_guide_bound_source_path,
    )
    answer, citation_validation = validate_structured_citations(
        answer,
        answer_hits=answer_hits,
        db_dir=db_dir,
        locked_source=locked_citation_source,
        paper_guide_mode=bool(paper_guide_mode),
        paper_guide_candidate_refs_by_source=dict(paper_guide_candidate_refs_by_source or {}),
        paper_guide_support_slots=list(paper_guide_support_slots or []),
        paper_guide_support_resolution=list(paper_guide_support_resolution or []),
    )
    # Citation validation may legitimately rewrite or inject structured cite markers for grounding,
    # but the final user-facing paper-guide answer still needs the same family-aware sanitization pass.
    if paper_guide_mode:
        answer = _sanitize_paper_guide_answer_for_user(
            answer,
            has_hits=bool(answer_hits),
            prompt=prompt_for_user or prompt,
            prompt_family=sanitize_paper_guide_family,
        )
        answer = _maybe_ensure_minimum_paper_guide_citation(
            answer,
            paper_guide_mode=bool(paper_guide_mode),
            prompt_family=sanitize_paper_guide_family,
            has_hits=bool(answer_hits),
            support_resolution=list(paper_guide_support_resolution or []),
            candidate_refs_by_source=dict(paper_guide_candidate_refs_by_source or {}),
            retrieval_confidence_hint=dict(paper_guide_retrieval_confidence_hint or {}),
            locked_citation_source=locked_citation_source,
        )
        # The minimum-citation helper may append structured markers for internal grounding.
        # Always re-sanitize for the final user-facing string so raw tokens never leak.
        answer = _sanitize_paper_guide_answer_for_user(
            answer,
            has_hits=bool(answer_hits),
            prompt=prompt_for_user or prompt,
            prompt_family=sanitize_paper_guide_family,
        )
    preserve_numeric_citations = _should_preserve_final_answer_numeric_citations(
        prompt=prompt_for_user or prompt,
        answer_output_mode=answer_output_mode,
        paper_guide_mode=bool(paper_guide_mode),
        prompt_family=sanitize_paper_guide_family,
    )
    answer = _strip_final_answer_citation_markers(
        answer,
        preserve_numeric_markers=preserve_numeric_citations,
    )
    grounded_answer = str(answer or "")
    answer = _maybe_prepend_paper_guide_low_confidence_notice(
        answer,
        paper_guide_mode=bool(paper_guide_mode),
        prompt_text=prompt_for_user or prompt,
        prompt_family=sanitize_paper_guide_family,
        retrieval_confidence_hint=dict(paper_guide_retrieval_confidence_hint or {}),
        support_resolution=list(paper_guide_support_resolution or []),
        candidate_refs_by_source=dict(paper_guide_candidate_refs_by_source or {}),
    )
    answer = _maybe_append_paper_guide_supplement_block(
        answer,
        paper_guide_mode=bool(paper_guide_mode),
        has_hits=bool(answer_hits),
        prompt_text=prompt_for_user or prompt,
        prompt_family=sanitize_paper_guide_family,
        retrieval_confidence_hint=dict(paper_guide_retrieval_confidence_hint or {}),
        grounded_answer=grounded_answer,
        support_resolution=list(paper_guide_support_resolution or []),
        build_paper_guide_supplement_lines=build_paper_guide_supplement_lines,
    )
    paper_guide_contracts = _build_paper_guide_contract_snapshot(
        paper_guide_mode=bool(paper_guide_mode),
        intent_model=resolved_paper_guide_intent,
        answer_markdown=grounded_answer,
        final_answer_markdown=answer,
        evidence_cards=list(paper_guide_evidence_cards or []),
        candidate_refs_by_source=dict(paper_guide_candidate_refs_by_source or {}),
        support_slots=list(paper_guide_support_slots or []),
        support_resolution=list(paper_guide_support_resolution or []),
        needs_supplement=bool(_PAPER_GUIDE_SUPPLEMENT_BLOCK_MARKER_RE.search(answer)),
        citation_validation=dict(citation_validation or {}),
        paper_guide_contracts_seed=dict(paper_guide_contracts_seed or {}),
    )
    answer_quality = _build_answer_quality_probe(
        answer,
        has_hits=bool(answer_hits),
        contract_enabled=bool(paper_guide_contract_enabled),
        intent=answer_intent,
        depth=answer_depth,
        output_mode=answer_output_mode,
        paper_guide_mode=bool(paper_guide_mode),
        prompt_family=sanitize_paper_guide_family,
    )
    retrieval_confidence = dict(paper_guide_retrieval_confidence_hint or {})
    if bool(retrieval_confidence.get("low_confidence")):
        refs_for_notice = _collect_low_confidence_candidate_refs(
            support_resolution=list(paper_guide_support_resolution or []),
            candidate_refs_by_source=dict(paper_guide_candidate_refs_by_source or {}),
            retrieval_confidence_hint=retrieval_confidence,
            max_items=6,
        )
        if refs_for_notice:
            retrieval_confidence["candidate_refs_for_notice"] = list(refs_for_notice)
    answer_quality["retrieval_confidence"] = retrieval_confidence
    return {
        "answer": answer,
        "paper_guide_support_resolution": list(paper_guide_support_resolution or []),
        "paper_guide_contracts": paper_guide_contracts,
        "citation_validation": citation_validation,
        "answer_quality": answer_quality,
    }
