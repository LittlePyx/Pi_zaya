from __future__ import annotations

from pathlib import Path
import re

from kb.paper_guide_shared import _source_name_from_md_path


def _split_kb_miss_notice(text: str) -> tuple[str, str]:
    s = str(text or "").lstrip()
    prefix = "未命中知识库片段"
    if not s.startswith(prefix):
        return "", str(text or "")

    nl = s.find("\n")
    if nl != -1:
        return s[:nl].strip(), s[nl + 1 :].lstrip("\n")

    for sep in ("。", ".", "！", "!", "？", "?", ";", "；"):
        idx = s.find(sep)
        if 0 <= idx <= 80:
            return s[: idx + 1].strip(), s[idx + 1 :].lstrip()

    return prefix, s[len(prefix) :].lstrip("：: \t")


def _reconcile_kb_notice(answer: str, *, has_hits: bool) -> str:
    notice, body = _split_kb_miss_notice(answer)
    body = str(body or "").strip()
    if has_hits:
        return body or str(answer or "").strip()

    if notice:
        return str(answer or "").strip()
    if not body:
        return "未命中知识库片段"
    return f"未命中知识库片段。\n\n{body}"


_ANSWER_INTENT_COMPARE_RE = re.compile(
    r"(\bcompare\b|\bcomparison\b|\bversus\b|\bvs\.?\b|\bdifference\b|\btrade[\s-]?off\b|"
    r"对比|区别|优劣|哪个好|怎么选|选型)",
    flags=re.IGNORECASE,
)
_ANSWER_INTENT_IDEA_RE = re.compile(
    r"(\bidea\b|\bnovelty\b|\binnovation\b|\bfeasib(?:le|ility)\b|\bhypothesis\b|\bbrainstorm\b|"
    r"想法|创新|可行性|可行|值得做|是否可做|研究点子)",
    flags=re.IGNORECASE,
)
_ANSWER_INTENT_EXPERIMENT_RE = re.compile(
    r"(\bexperiment\b|\bablation\b|\bbaseline\b|\bmetric\b|\bevaluation\b|\bprotocol\b|\breproduc(?:e|ibility)\b|"
    r"实验|对照|指标|评估|消融|复现|验证方案)",
    flags=re.IGNORECASE,
)
_ANSWER_INTENT_TROUBLESHOOT_RE = re.compile(
    r"(\bdebug\b|\btroubleshoot\b|\berror\b|\bissue\b|\bfail(?:ed|ure)?\b|\bwhy not\b|"
    r"报错|排查|卡住|失败|不收敛|跑不通)",
    flags=re.IGNORECASE,
)
_ANSWER_INTENT_WRITING_RE = re.compile(
    r"(\bwrite\b|\bwriting\b|\brewrite\b|\bedit\b|\bwording\b|\brelated work\b|\babstract\b|"
    r"写作|润色|改写|表达|摘要|相关工作)",
    flags=re.IGNORECASE,
)
_ANSWER_CITE_HINT_RE = re.compile(r"(\[\d+\]|\[\[CITE:)", flags=re.IGNORECASE)
_ANSWER_LIMITS_HINT_RE = re.compile(
    r"(\bmay\b|\bmight\b|\buncertain\b|\bunknown\b|\bnot shown\b|\binsufficient\b|\bassum(?:e|ption)\b|"
    r"可能|不确定|未知|未给出|证据不足|假设)",
    flags=re.IGNORECASE,
)
_ANSWER_SECTION_PREFIX_RE = re.compile(
    r"(?im)^\s*(Conclusion|Evidence|Limits|Next Steps|结论|依据|证据|边界|限制|局限|下一步建议|下一步)\s*[:：]"
)
_ANSWER_NEXT_STEPS_HEADER_RE = re.compile(r"(?im)^\s*(Next Steps|Next Step|下一步建议|下一步|建议)\s*[:：]")
_ANSWER_ORDERED_LIST_RE = re.compile(r"(?m)^\s*1\.\s+\S+")
_ANSWER_CROSS_PAPER_QUERY_RE = re.compile(
    r"(\bother papers?\b|\bwhich papers?\b|\bwhich paper\b|\bbesides this paper\b|\bin my library\b|"
    r"哪篇|哪些论文|别的论文|其他论文|库里|文库里)",
    flags=re.IGNORECASE,
)
_ANSWER_INTENTS = {"reading", "compare", "idea", "experiment", "troubleshoot", "writing"}
_ANSWER_DEPTHS = {"L1", "L2", "L3"}
_ANSWER_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_ANSWER_NEGATIVE_SHELL_RE = re.compile(
    r"(\bnone of the retrieved\b|\bno direct paper snippet\b|\bno papers? in (?:my|your) library\b|"
    r"\bdoes not mention\b|\bdo not mention\b|\bdoes not discuss\b|\bdo not discuss\b|"
    r"\bnot stated\b|\bnot mentioned\b|未命中知识库片段|没有直接证据|未直接提到|没有提到|未讨论|没有讨论|未明确给出|未说明)",
    flags=re.IGNORECASE,
)
_ANSWER_OUTPUT_MODES = {"fact_answer", "reading_guide", "critical_review"}
_ANSWER_OUTPUT_MODE_FACT_RE = re.compile(
    r"(\bquote\b|\bexact\b|\bverif(?:y|ication)\b|\bfact[\s-]?check\b|\bwhich section\b|"
    r"\bwhich paragraph\b|\bwhere in (?:the )?paper\b|\baccording to the paper\b|"
    r"\bdoes the paper say\b|\bstated in the paper\b|\bfigure\s*\d+\b|\bfig\.\s*\d+\b|"
    r"\bequation\s*\d+\b|\beq\.\s*\d+\b|\btheorem\s*\d+\b|\blocate\b|"
    r"原文|引用|哪一节|哪一段|定位|证实|是否写到|公式|图|表|定理)",
    flags=re.IGNORECASE,
)
_ANSWER_OUTPUT_MODE_CRITICAL_RE = re.compile(
    r"(\bcrit(?:ical|ique)\b|\breview\b|\blimitation\b|\bweakness\b|\bconcern\b|\brisk\b|"
    r"\bmissing\b|\bflaw\b|\bdrawback\b|局限|不足|缺点|批判|质疑|风险|问题)",
    flags=re.IGNORECASE,
)
_ANSWER_OUTPUT_MODE_OVERVIEW_RE = re.compile(
    r"(\bwhat problem\b|\bsolve(?:s|d)?\b|\bmain contribution\b|\bcore contribution\b|\bkey contribution\b|"
    r"\bmain idea\b|\bsummary\b|\bwhat does this paper do\b|解决.*问题|核心贡献|主要贡献|这篇.*讲了什么)",
    flags=re.IGNORECASE,
)


def _normalize_answer_mode_hint(answer_mode_hint: str) -> str:
    hint = str(answer_mode_hint or "").strip().lower()
    alias = {
        "read": "reading",
        "reading": "reading",
        "literature": "reading",
        "compare": "compare",
        "comparison": "compare",
        "idea": "idea",
        "experiment": "experiment",
        "exp": "experiment",
        "debug": "troubleshoot",
        "troubleshoot": "troubleshoot",
        "writing": "writing",
    }
    return alias.get(hint, "")


def _normalize_answer_output_mode_hint(answer_output_mode_hint: str) -> str:
    hint = str(answer_output_mode_hint or "").strip().lower()
    alias = {
        "fact": "fact_answer",
        "fact_answer": "fact_answer",
        "factanswer": "fact_answer",
        "verify": "fact_answer",
        "verification": "fact_answer",
        "reading": "reading_guide",
        "reading_guide": "reading_guide",
        "readingguide": "reading_guide",
        "guide": "reading_guide",
        "paper_guide": "reading_guide",
        "critical": "critical_review",
        "critical_review": "critical_review",
        "criticalreview": "critical_review",
        "review": "critical_review",
        "critique": "critical_review",
    }
    return alias.get(hint, "")


def _normalize_answer_output_mode(answer_output_mode: str) -> str:
    mode = _normalize_answer_output_mode_hint(answer_output_mode)
    return mode or "reading_guide"


def _answer_output_mode_requires_next_steps(answer_output_mode: str) -> bool:
    return _normalize_answer_output_mode(answer_output_mode) != "fact_answer"


def _detect_answer_output_mode(
    prompt: str,
    *,
    answer_output_mode_hint: str = "",
    answer_mode_hint: str = "",
    paper_guide_mode: bool = False,
    intent: str = "",
    anchor_grounded: bool = False,
) -> str:
    explicit_mode = _normalize_answer_output_mode_hint(answer_output_mode_hint)
    if explicit_mode:
        return explicit_mode
    hinted_mode = _normalize_answer_output_mode_hint(answer_mode_hint)
    if hinted_mode:
        return hinted_mode

    q = str(prompt or "").strip()
    if not q:
        return "reading_guide"
    if re.search(r"(解决.*问题|核心贡献|主要贡献|这篇.*讲了什么)", q, flags=re.IGNORECASE):
        return "reading_guide"
    if re.search(r"(局限|不足|缺点|批判|质疑|风险|问题)", q, flags=re.IGNORECASE):
        return "critical_review"
    if paper_guide_mode and (anchor_grounded or re.search(r"(原文|哪一段|定位|证实|是否写到|公式|图|表|定理)", q, flags=re.IGNORECASE)):
        return "fact_answer"
    if paper_guide_mode and _ANSWER_OUTPUT_MODE_OVERVIEW_RE.search(q):
        return "reading_guide"
    if _ANSWER_OUTPUT_MODE_CRITICAL_RE.search(q):
        return "critical_review"
    if paper_guide_mode and (anchor_grounded or _ANSWER_OUTPUT_MODE_FACT_RE.search(q)):
        return "fact_answer"
    if intent in {"compare", "idea", "experiment", "troubleshoot", "writing"}:
        return "reading_guide"
    return "reading_guide"


def _detect_answer_intent(prompt: str, *, answer_mode_hint: str = "") -> str:
    mode_hint = _normalize_answer_mode_hint(answer_mode_hint)
    if mode_hint:
        return mode_hint
    q = str(prompt or "").strip()
    if not q:
        return "reading"
    if re.search(r"(对比|区别|优劣|哪个好|怎么选)", q, flags=re.IGNORECASE):
        return "compare"
    if re.search(r"(想法|创新|可行性|值得做|是否可做|研究点子)", q, flags=re.IGNORECASE):
        return "idea"
    if re.search(r"(实验|对照|指标|评估|消融|复现|验证方案)", q, flags=re.IGNORECASE):
        return "experiment"
    if re.search(r"(报错|排查|卡住|失败|不收敛|跑不通)", q, flags=re.IGNORECASE):
        return "troubleshoot"
    if re.search(r"(写作|润色|改写|表达|摘要|相关工作)", q, flags=re.IGNORECASE):
        return "writing"
    if _ANSWER_INTENT_COMPARE_RE.search(q):
        return "compare"
    if _ANSWER_INTENT_IDEA_RE.search(q):
        return "idea"
    if _ANSWER_INTENT_EXPERIMENT_RE.search(q):
        return "experiment"
    if _ANSWER_INTENT_TROUBLESHOOT_RE.search(q):
        return "troubleshoot"
    if _ANSWER_INTENT_WRITING_RE.search(q):
        return "writing"
    return "reading"


def _detect_answer_depth(prompt: str, *, intent: str, auto_depth: bool) -> str:
    if not auto_depth:
        return "L2"
    q = str(prompt or "")
    q_len = len(q)
    technical_markers = len(
        re.findall(
            r"(\bfig(?:ure)?\b|\beq(?:uation)?\b|\bmethod\b|\balgorithm\b|\bproof\b|"
            r"实验|公式|定理|算法|模型|指标|对照|复杂度|损失函数)",
            q,
            flags=re.IGNORECASE,
        )
    )
    if intent in {"idea", "experiment", "compare"} and (q_len >= 48 or technical_markers >= 2):
        return "L3"
    if intent in {"reading", "troubleshoot", "writing"} and q_len <= 24 and technical_markers == 0:
        return "L1"
    return "L2"


def _prefer_zh_locale(*texts: str) -> bool:
    parts = [str(t or "") for t in texts if str(t or "")]
    if not parts:
        return False
    first = str(parts[0] or "")
    if first:
        first_cjk = len(_ANSWER_CJK_CHAR_RE.findall(first))
        first_latin = len(re.findall(r"[A-Za-z]", first))
        if first_cjk >= 4:
            return True
        if first_cjk >= 2 and (
            first_cjk >= max(2, first_latin // 5)
            or bool(re.search(r"[。！？；，、】【]|(这篇|论文|文献|库里|还有哪些|哪篇|讨论|比较|定义|为什么|怎么)", first))
        ):
            return True
    joined = " ".join(parts)
    cjk = len(_ANSWER_CJK_CHAR_RE.findall(joined))
    if cjk <= 0:
        return False
    latin = len(re.findall(r"[A-Za-z]", joined))
    return cjk >= 4 or cjk >= max(2, latin // 3)


def _normalize_answer_section_name(raw: str) -> str:
    s = str(raw or "").strip().lower().replace(" ", "")
    if s in {"conclusion", "结论"}:
        return "conclusion"
    if s in {"evidence", "依据", "证据"}:
        return "evidence"
    if s in {"limits", "边界", "限制", "局限"}:
        return "limits"
    if s in {"nextsteps", "下一步", "下一步建议"}:
        return "next_steps"
    return ""


def _extract_answer_section_keys(text: str) -> list[str]:
    keys: list[str] = []
    for m in _ANSWER_SECTION_PREFIX_RE.finditer(str(text or "")):
        key = _normalize_answer_section_name(str(m.group(1) or ""))
        if key:
            keys.append(key)
    return keys


def _extract_answer_sections(text: str) -> dict[str, str]:
    src = str(text or "")
    matches = list(_ANSWER_SECTION_PREFIX_RE.finditer(src))
    if not matches:
        return {}
    out: dict[str, str] = {}
    for idx, match in enumerate(matches):
        key = _normalize_answer_section_name(str(match.group(1) or ""))
        if not key:
            continue
        start = match.end()
        end = matches[idx + 1].start() if (idx + 1) < len(matches) else len(src)
        body = src[start:end].strip()
        if body:
            out[key] = body
    return out


def _normalize_contract_compare_text(text: str) -> str:
    s = str(text or "").strip().lower()
    if not s:
        return ""
    s = s.replace(".en.md", " ").replace(".pdf", " ")
    s = re.sub(r"[\*_`#>\[\]\(\)\-–—:;,.]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _split_contract_ordered_items(body: str) -> list[str]:
    src = str(body or "").strip()
    if not src:
        return []
    matches = list(re.finditer(r"(?m)^\s*\d+\.\s+", src))
    if not matches:
        return [src]
    out: list[str] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if (idx + 1) < len(matches) else len(src)
        item = src[start:end].strip()
        if item:
            out.append(item)
    return out


def _evidence_section_is_too_thin(body: str, *, answer_hits: list[dict] | None = None) -> bool:
    raw = str(body or "").strip()
    if not raw:
        return True
    if _ANSWER_CITE_HINT_RE.search(raw):
        return False
    if re.search(r"\b(states?|shows?|reports?|defines?|discuss(?:es|ed)?|compare(?:s|d)?|indicat(?:es|ed)|describes?|notes?)\b", raw, flags=re.IGNORECASE):
        return False
    if re.search(r"(指出|表明|显示|定义|讨论|比较|说明|描述)", raw):
        return False
    plain = re.sub(r"(?m)^\s*(?:[-*]|\d+\.)\s*", "", raw)
    plain = plain.replace("**", "").replace("__", "")
    plain = re.sub(r"\s+", " ", plain).strip(" .;:-")
    if not plain:
        return True
    plain_norm = _normalize_contract_compare_text(plain)
    if not plain_norm:
        return True
    source_norms = {
        norm
        for norm in (_normalize_contract_compare_text(_contract_source_name_from_hit(hit)) for hit in list(answer_hits or []))
        if norm
    }
    if plain_norm in source_norms:
        return True
    return len(plain_norm) < 80


def _classify_contract_extra_paragraph(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return "extra"
    if re.search(r"(?im)^\s*(?:\*\*)?(?:next\s*step|next\s*steps|下一步建议|下一步|建议)(?:\*\*)?\s*[:：]?", s):
        return "step"
    if re.search(r"(?im)^\s*(?:\*\*)?(?:evidence|依据|证据)(?:\*\*)?\s*[:：]?", s):
        return "evidence"
    if re.search(r"(?im)^\s*(?:\*\*)?(?:limits|边界|限制|局限)(?:\*\*)?\s*[:：]?", s):
        return "limits"
    if re.search(
        r"(\bno other retrieved\b|\bnone of the retrieved\b|\bdoes not mention\b|\bdo not mention\b|"
        r"\bdoes not discuss\b|\bdo not discuss\b|\bnot supersampling\b|没有提到|未讨论|未直接提到|没有直接证据)",
        s,
        flags=re.IGNORECASE,
    ):
        return "limits"
    if re.search(
        r"(?im)(^\s*[-*]\s+\*\*(definition location|key definition|operational example|matched section)\*\*|"
        r"^\s*>\s+|\bsection titled\b|\bmatched section\b|\bfirst paragraph\b|定义位置|关键定义|操作示例|命中章节)",
        s,
        flags=re.IGNORECASE,
    ):
        return "evidence"
    if re.search(
        r"(\bsearch (?:my|your|the) library\b|\bsearch for keywords\b|\bkeyword search\b|"
        r"\bcheck the likely methods sections\b|\blook under another name\b|"
        r"搜索(?:你|您的|库内)?(?:论文|文献|知识库)|检索关键词|同义词|缩写|别名)",
        s,
        flags=re.IGNORECASE,
    ):
        return "step"
    return "extra"


def _strip_contract_header_prefix(text: str, *labels: str) -> str:
    s = str(text or "").strip()
    parts = [re.escape(str(label or "").strip()) for label in labels if str(label or "").strip()]
    if (not s) or (not parts):
        return s
    pattern = r"(?im)^\s*(?:\*\*)?(?:" + "|".join(parts) + r")(?:\*\*)?\s*[:：]?\s*"
    return re.sub(pattern, "", s, count=1).strip()


def _normalize_contract_evidence_item(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    s = re.sub(r"(?m)^\s*\d+\.\s+", "", s)
    s = re.sub(r"(?m)^\s*[-*]\s+", "", s)
    s = _strip_contract_header_prefix(s, "Evidence", "依据", "证据")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _normalize_contract_limits_item(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    s = re.sub(r"(?m)^\s*\d+\.\s+", "", s)
    s = re.sub(r"(?m)^\s*[-*]\s+", "", s)
    s = _strip_contract_header_prefix(s, "Limits", "边界", "限制", "局限")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip(" -")


def _normalize_contract_step_item(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    s = _strip_contract_header_prefix(s, "Next Steps", "Next Step", "下一步建议", "下一步", "建议")
    s = re.sub(r"(?m)^\s*\d+\.\s+", "", s)
    s = re.sub(r"(?m)^\s*[-*]\s+", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _is_contract_title_only_evidence_item(text: str) -> bool:
    s = _normalize_contract_evidence_item(text)
    if not s:
        return False
    s = re.sub(r"(?im)^\s*(Evidence|依据|证据)\s*[:：]\s*", "", s).strip()
    return bool(re.fullmatch(r"\*\*[^*\n]{6,}\*\*\.?", s))


def _cleanup_structured_evidence_body(body: str, *, answer_hits: list[dict] | None = None) -> str:
    items = _split_contract_ordered_items(body)
    if not items:
        return str(body or "").strip()
    normalized = [_normalize_contract_evidence_item(item) for item in items]
    normalized = [item for item in normalized if item]
    if len(normalized) <= 1:
        return "\n".join(f"{idx}. {item}" for idx, item in enumerate(normalized, start=1)) if normalized else ""
    substantive = [
        item
        for item in normalized
        if (not _is_contract_title_only_evidence_item(item))
        and (not _evidence_section_is_too_thin(item, answer_hits=answer_hits))
    ]
    if substantive:
        normalized = substantive
    return "\n".join(f"{idx}. {item}" for idx, item in enumerate(normalized, start=1))


def _cleanup_structured_answer_sections(
    answer: str,
    *,
    prompt: str,
    has_hits: bool,
    answer_hits: list[dict] | None = None,
) -> str:
    src = str(answer or "").strip()
    if not src:
        return src
    notice, body0 = _split_kb_miss_notice(src)
    body = str(body0 if notice else src).strip()
    sections = _extract_answer_sections(body)
    if not sections:
        return src
    changed = False
    if has_hits and sections.get("evidence"):
        cleaned_evidence = _cleanup_structured_evidence_body(
            sections.get("evidence", ""),
            answer_hits=answer_hits,
        )
        if cleaned_evidence and cleaned_evidence != str(sections.get("evidence") or "").strip():
            sections["evidence"] = cleaned_evidence
            changed = True
    if not changed:
        return src
    prefer_zh = _prefer_zh_locale(prompt, src)
    labels = {
        "conclusion": "结论" if prefer_zh else "Conclusion",
        "evidence": "依据" if prefer_zh else "Evidence",
        "limits": "边界" if prefer_zh else "Limits",
        "next_steps": "下一步" if prefer_zh else "Next Steps",
    }
    order = ["conclusion", "evidence", "limits", "next_steps"]
    parts: list[str] = []
    if notice:
        parts.append(notice)
    for key in order:
        content = str(sections.get(key) or "").strip()
        if not content:
            continue
        if key == "conclusion":
            parts.append(f"{labels[key]}: {content}")
        else:
            parts.append(f"{labels[key]}:\n{content}")
    return "\n\n".join(parts).strip()


def _has_sufficient_answer_sections(
    text: str,
    *,
    has_hits: bool,
    output_mode: str = "reading_guide",
    answer_hits: list[dict] | None = None,
) -> bool:
    keys = set(_extract_answer_section_keys(text))
    sections = _extract_answer_sections(text)
    next_steps_required = _answer_output_mode_requires_next_steps(output_mode)
    if "conclusion" not in keys:
        return False
    if next_steps_required and ("next_steps" not in keys):
        return False
    if bool(has_hits) and ("evidence" not in keys):
        return False
    if bool(has_hits) and _evidence_section_is_too_thin(sections.get("evidence", ""), answer_hits=answer_hits):
        return False
    return len(keys) >= 2


def _build_answer_quality_probe(
    answer: str,
    *,
    has_hits: bool,
    contract_enabled: bool,
    intent: str = "",
    depth: str = "",
    output_mode: str = "reading_guide",
    paper_guide_mode: bool = False,
    prompt_family: str = "",
) -> dict:
    text = str(answer or "").strip()
    keys = set(_extract_answer_section_keys(text))
    has_conclusion = "conclusion" in keys
    has_evidence = "evidence" in keys
    has_limits = "limits" in keys
    has_next_steps = "next_steps" in keys
    has_citations = bool(_ANSWER_CITE_HINT_RE.search(text) or re.search(r"\[\d{1,3}\]", text))
    prompt_family_norm = str(prompt_family or "").strip().lower()
    output_mode_norm = _normalize_answer_output_mode(output_mode)
    abstract_mode = bool(paper_guide_mode and prompt_family_norm == "abstract")
    locate_required = bool(paper_guide_mode and prompt_family_norm == "figure_walkthrough" and has_hits)
    structured_required = bool(contract_enabled)
    has_locate_hint = bool(
        re.search(r"(/api/references/asset|(?:^|\b)(?:fig(?:ure)?\.?\s*\d+|图\s*\d+|panel\s*[a-z]))", text, flags=re.IGNORECASE)
    )
    has_locate_hint = bool(
        has_locate_hint
        or re.search(r"(/api/references/asset|(?:^|\b)(?:fig(?:ure)?\.?\s*\d+|图\s*\d+|panel\s*[a-z]))", text, flags=re.IGNORECASE)
    )
    abstract_style_ok = bool(
        (not abstract_mode)
        or (
            len(text) >= 40
            and (not bool(keys.intersection({"conclusion", "evidence", "limits", "next_steps"})))
        )
    )
    evidence_required = bool(has_hits and (not abstract_mode))
    next_steps_required = bool(_answer_output_mode_requires_next_steps(output_mode_norm) and (not abstract_mode))
    if not structured_required:
        evidence_required = False
        next_steps_required = False
    expected_core_sections = max(1, 1 + int(evidence_required) + int(next_steps_required))
    core_sections = int(has_conclusion)
    if evidence_required:
        core_sections += int(has_evidence)
    if next_steps_required:
        core_sections += int(has_next_steps)
    core_ratio = round(float(core_sections) / float(expected_core_sections), 3)
    evidence_ok = (not evidence_required) or bool(has_evidence)
    locate_ok = (not locate_required) or bool(has_locate_hint)
    if abstract_mode:
        minimum_ok = bool(abstract_style_ok)
    elif not structured_required:
        minimum_ok = bool(len(text) >= 40 and locate_ok)
    else:
        minimum_ok = bool(has_conclusion and evidence_ok and ((not next_steps_required) or has_next_steps) and locate_ok)
    return {
        "contract_enabled": bool(contract_enabled),
        "intent": str(intent or ""),
        "depth": str(depth or ""),
        "output_mode": output_mode_norm,
        "char_count": len(text),
        "has_hits": bool(has_hits),
        "has_conclusion": bool(has_conclusion),
        "has_evidence": bool(has_evidence),
        "has_limits": bool(has_limits),
        "has_next_steps": bool(has_next_steps),
        "has_citations": bool(has_citations),
        "paper_guide_mode": bool(paper_guide_mode),
        "prompt_family": prompt_family_norm,
        "has_locate_hint": bool(has_locate_hint),
        "locate_required": bool(locate_required),
        "locate_ok": bool(locate_ok),
        "abstract_style_ok": bool(abstract_style_ok),
        "evidence_required": bool(evidence_required),
        "next_steps_required": bool(next_steps_required),
        "evidence_ok": bool(evidence_ok),
        "core_section_coverage": core_ratio,
        "minimum_ok": bool(minimum_ok),
    }


def _build_answer_contract_system_rules(
    *,
    intent: str,
    depth: str,
    has_hits: bool,
    output_mode: str = "reading_guide",
) -> str:
    output_mode_norm = _normalize_answer_output_mode(output_mode)
    lines = [
        "Answer Contract v1 (enabled):",
        "- Keep the response compact but structured.",
        "- Keep the answer in the same language as the user's query.",
        "- Conclusion should answer the user's core question directly in 1-3 sentences.",
        "- Avoid redundancy; keep total bullet/action lines concise (usually <= 8).",
    ]
    if output_mode_norm == "fact_answer":
        lines.append("- Use this section order when possible: Conclusion, Evidence, Limits.")
        lines.append("- Do not add a Next Steps section unless the user explicitly asks for follow-up actions.")
    else:
        lines.append("- Use this section order when possible: Conclusion, Evidence, Limits, Next Steps.")
    if has_hits:
        lines.append("- Evidence should be grounded in retrieved snippets and include citations when available.")
    else:
        lines.append("- If retrieval has no hits, avoid fabrication and clearly mark the answer as general guidance.")
    if output_mode_norm == "fact_answer":
        if depth == "L1":
            lines.append("- Depth=L1: answer directly with the smallest sufficient evidence span.")
        elif depth == "L3":
            lines.append("- Depth=L3: include assumptions/boundaries, but stay focused on resolving the exact question.")
        else:
            lines.append("- Depth=L2: include at least one evidence item and keep the response tightly scoped to the asked fact.")
    elif depth == "L1":
        lines.append("- Depth=L1: concise response, at most one next-step action.")
    elif depth == "L3":
        lines.append("- Depth=L3: include assumptions/boundaries and 2-3 concrete follow-up actions.")
    else:
        lines.append("- Depth=L2: include at least one evidence item and 1-2 concrete next-step actions.")

    intent_rules = {
        "reading": "- Intent=reading: focus on contribution, key evidence, and where to read next.",
        "compare": "- Intent=compare: emphasize differences, applicability boundary, and trade-offs.",
        "idea": "- Intent=idea: include feasibility, risk, and a minimum validation path.",
        "experiment": "- Intent=experiment: include variables/controls, metrics, and expected outcomes.",
        "troubleshoot": "- Intent=troubleshoot: prioritize likely causes, diagnosis steps, and fix order.",
        "writing": "- Intent=writing: provide structure edits and directly reusable wording suggestions.",
    }
    lines.append(intent_rules.get(intent, intent_rules["reading"]))
    if output_mode_norm == "critical_review":
        lines.append("- Output mode=critical_review: separate paper-grounded criticism from speculative concerns.")
    elif output_mode_norm == "fact_answer":
        lines.append("- Output mode=fact_answer: prefer exact paper-grounded resolution over broad study advice.")
    return "\n" + "\n".join(lines) + "\n"


def _answer_contract_enabled(task: dict | None) -> bool:
    return bool((task or {}).get("answer_contract_v1", False))


def _build_paper_guide_grounding_rules(
    *,
    answer_contract_v1: bool,
    output_mode: str = "reading_guide",
    prompt_family: str = "",
) -> str:
    output_mode_norm = _normalize_answer_output_mode(output_mode)
    prompt_family_norm = str(prompt_family or "").strip().lower()
    lines = [
        "Paper-guide formula grounding:",
        "- When the question asks about a formula/model, quote the equation from retrieved context as-is, including equation tag/number when available.",
        "- If exact equation text is not present in retrieved context, say it is not explicitly available; do not invent a generic replacement formula.",
        "- When quoting a retrieved long equation or numbered equation, render the equation itself as display math and keep the explanation sentence separate.",
        "- Do not compress a retrieved long equation plus its explanation into one mixed prose sentence if that would blur the locate target.",
        "- Do not introduce display-math formulas for figure explanations, mechanism summaries, or optimization sketches unless that exact display formula was retrieved from the paper.",
        "- Do not introduce hardware models, acquisition parameters, baseline numbers, or modality-specific claims that are not explicitly present in retrieved context.",
        "- For broad or generic questions, synthesize across retrieved sections, but mark any missing quantity as not stated instead of filling it with background knowledge.",
        "- Keep entity names exact: do not swap sample type, cell line, nanoparticle type, hardware model, or dataset identity with a plausible nearby alternative.",
        "- When the paper separately supports a claim with test objects and live-cell demonstrations, keep those evidence scopes separate instead of merging them into one stronger claim.",
        "- If a 'Paper-guide method focus' or 'Paper-guide figure focus' block is provided, resolve that focal sub-question from the focus block before using broader context.",
        "- If a 'Paper-guide support slots' block is provided, end each paper-grounded claim with that slot's exact support_example marker instead of guessing a paper reference number directly.",
        "- Treat [[SUPPORT:...]] as the primary grounding marker for paper-guide mode; runtime will resolve it into the final citation or locate-only support.",
        "- If a slot says cite_policy=locate_only, still use its support_example marker but do not invent a paper reference number.",
        "- If a 'Paper-guide evidence cards' block is provided, treat each DOC-k card as a hard boundary for paper-grounded claims.",
        "- Follow the card's use= instruction and stay inside the snippet scope instead of merging multiple cards into a stronger unsupported claim.",
        "- If a DOC-k card shows cite_example=[[CITE:<sid>:<ref_num>]], reuse that exact marker on the claim derived from the card unless DOI or author-year text clearly identifies a different ref.",
        "- If a 'Paper-guide citation grounding hints' block is provided, keep each claim aligned to the same DOC-k line before choosing [[CITE:<sid>:<ref_num>]].",
        "- Prefer the ref numbers listed on that DOC-k line; do not borrow a ref number from another DOC-k line unless DOI or author-year text explicitly identifies it.",
    ]
    if prompt_family_norm == "abstract":
        lines.extend(
            [
                "- Prompt family=abstract: use the abstract span itself before any translation or explanation.",
                "- Do not prepend title, authors, or reading-guide shells such as Conclusion/Evidence/Next Steps to an abstract-only answer.",
            ]
        )
    elif prompt_family_norm == "figure_walkthrough":
        lines.extend(
            [
                "- Prompt family=figure_walkthrough: preserve figure numbers and panel letters exactly as shown in the evidence cards/snippets.",
                "- Do not infer unlabeled panels, hardware details, or cross-panel relationships that are not explicitly stated in the retrieved figure caption/result text.",
            ]
        )
    elif prompt_family_norm == "method":
        lines.extend(
            [
                "- Prompt family=method: when the question explicitly names APR or another sub-module, prefer the exact implementation detail from the narrowest matching snippet over a broader overview summary.",
                "- If the retrieved method snippet names phase correlation, image registration, or RVT, keep those terms exact instead of replacing them with a generic mechanism paraphrase.",
            ]
        )
    elif prompt_family_norm == "equation":
        lines.extend(
            [
                "- Prompt family=equation: keep the equation itself and the nearby variable-definition sentence together when both are available.",
                "- Do not replace math symbols or variable names with a generic prose paraphrase if the retrieved equation text already states them explicitly.",
            ]
        )
    elif prompt_family_norm == "citation_lookup":
        lines.extend(
            [
                "- Prompt family=citation_lookup: extract the exact in-paper reference numbers or reference-list entries from the narrowest matching snippet before adding interpretation.",
                "- If the retrieved snippet already shows the explicit refs, do not answer 'not stated' or 'not mentioned'.",
            ]
        )
    if output_mode_norm == "fact_answer":
        lines.extend(
            [
                "- Output mode=fact_answer: answer the exact paper-grounded question first and avoid generic reading advice.",
                "- Prefer the smallest evidence span that resolves the question; if the exact span is unavailable, say that explicitly.",
            ]
        )
    elif output_mode_norm == "critical_review":
        lines.append("- Output mode=critical_review: keep critique tied to retrieved evidence before adding synthesis.")
    if answer_contract_v1:
        lines.extend(
            [
                "- Keep the answer compact: 3-4 sections, avoid repetitive paragraphs.",
                "- Only keep claims with direct support under Evidence; move unsupported general knowledge to Limits.",
            ]
        )
    else:
        lines.extend(
            [
                "- Keep the answer compact and direct, but do not force a fixed section template.",
                "- Distinguish retrieved evidence from supplemental general knowledge in prose; do not present unsupported context as direct paper evidence.",
            ]
        )
    return "\n" + "\n".join(lines) + "\n"


def _extract_cited_sentences(text: str, *, limit: int = 2) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for seg in re.split(r"(?<=[。！？.!?])\s*", str(text or "")):
        s = str(seg or "").strip()
        if (not s) or (not _ANSWER_CITE_HINT_RE.search(s)):
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max(1, int(limit)):
            break
    return out
def _contract_prompt_keywords(prompt: str, *, max_items: int = 6) -> list[str]:
    stopwords = {
        "which",
        "what",
        "where",
        "who",
        "when",
        "why",
        "how",
        "the",
        "this",
        "that",
        "these",
        "those",
        "my",
        "your",
        "our",
        "their",
        "me",
        "us",
        "you",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "from",
        "about",
        "into",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "paper",
        "papers",
        "library",
        "directly",
        "most",
        "please",
        "point",
        "section",
        "source",
        "other",
        "besides",
        "define",
        "defines",
        "defined",
        "discuss",
        "discusses",
        "compare",
        "compares",
        "comparison",
        "single",
        "pixel",
        "imaging",
        "哪篇",
        "哪个",
        "什么",
        "论文",
        "文献",
        "文章",
        "库里",
        "直接",
        "定义",
        "比较",
        "对比",
        "是否",
        "有没有",
        "提到",
        "讨论",
    }
    out: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", str(prompt or "")):
        key = str(raw or "").strip().lower()
        if (not key) or key.isdigit() or key in stopwords or key in seen:
            continue
        seen.add(key)
        out.append(str(raw).strip())
        if len(out) >= max(1, int(max_items or 6)):
            break
    return out


def _contract_focus_label(prompt: str) -> str:
    keywords = _contract_prompt_keywords(prompt, max_items=3)
    if not keywords:
        return ""
    if len(keywords) >= 2 and (len(keywords[0]) + len(keywords[1])) <= 36:
        return f"{keywords[0]} {keywords[1]}"
    return keywords[0]


def _contract_prompt_requests_definition(prompt: str) -> bool:
    return bool(
        re.search(
            r"(\bdefine\b|\bdefines\b|\bdefinition\b|\bwhat is\b|\bmost directly defines\b|"
            r"\bpoint me to the source section\b|\bwhere is .*defined\b|定义|是什么|出处|哪一节|哪一段)",
            str(prompt or ""),
            flags=re.IGNORECASE,
        )
    )


def _contract_prompt_requests_comparison(prompt: str) -> bool:
    return bool(
        re.search(
            r"(\bcompare\b|\bcomparison\b|\bcompares\b|\bversus\b|\bvs\.?\b|对比|比较|区别|差异)",
            str(prompt or ""),
            flags=re.IGNORECASE,
        )
    )


def _contract_source_name_from_hit(hit: dict) -> str:
    if not isinstance(hit, dict):
        return ""
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    source_path = str((meta or {}).get("source_path") or "").strip()
    if not source_path:
        return ""
    return _source_name_from_md_path(source_path)


def _contract_heading_from_hit(hit: dict) -> str:
    if not isinstance(hit, dict):
        return ""
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    for key in ("ref_best_heading_path", "heading_path", "ref_subsection", "top_heading", "ref_section"):
        value = str((meta or {}).get(key) or "").strip()
        if value:
            return value
    return ""


def _contract_source_name_from_primary_evidence(primary_evidence: dict | None) -> str:
    if not isinstance(primary_evidence, dict):
        return ""
    source_name = str(
        primary_evidence.get("source_name")
        or primary_evidence.get("sourceName")
        or ""
    ).strip()
    if source_name:
        return source_name
    source_path = str(
        primary_evidence.get("source_path")
        or primary_evidence.get("sourcePath")
        or ""
    ).strip()
    if not source_path:
        return ""
    return _source_name_from_md_path(source_path)


def _contract_heading_from_primary_evidence(primary_evidence: dict | None) -> str:
    if not isinstance(primary_evidence, dict):
        return ""
    for key in ("heading_path", "headingPath", "heading", "section_label", "sectionLabel"):
        value = str(primary_evidence.get(key) or "").strip()
        if value:
            return value
    return ""


def _contract_snippet_candidates_from_primary_evidence(primary_evidence: dict | None) -> list[str]:
    if not isinstance(primary_evidence, dict):
        return []
    out: list[str] = []
    for key in ("snippet", "highlight_snippet", "highlightSnippet"):
        text = str(primary_evidence.get(key) or "").strip()
        if text:
            out.append(text)
    return out


def _contract_requests_cross_paper_query(prompt: str) -> bool:
    return bool(
        re.search(
            r"(\bwhich other papers?\b|\bother papers?\b|\bbesides this paper\b|\banother paper\b|"
            r"除此之外|除(?:了)?这篇|其他论文|别的论文|还有哪些论文|另一篇论文)",
            str(prompt or ""),
            flags=re.IGNORECASE,
        )
    )


def _contract_unique_source_names(answer_hits: list[dict] | None, *, limit: int = 4) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for hit in list(answer_hits or []):
        name = _contract_source_name_from_hit(hit)
        key = _normalize_contract_compare_text(name)
        if (not name) or (not key) or key in seen:
            continue
        seen.add(key)
        out.append(name)
        if len(out) >= max(1, int(limit or 4)):
            break
    return out


def _contract_cross_paper_topic_label(prompt: str) -> str:
    q = str(prompt or "").strip()
    if not q:
        return ""
    patterns = [
        r"(?i)\b(?:discuss(?:es|ed)?|mention(?:s|ed)?|define(?:s|d)?|compare(?:s|d)?|cite(?:s|d)?|use(?:s|d)?|cover(?:s|ed)?)\s+(.+?)(?:[?!.]\s*$|$)",
        r"(?i)\babout\s+(.+?)(?:[?!.]\s*$|$)",
        r"(?:讨论|提到|定义|比较|对比|引用|涉及|使用)\s*[“\"']?(.+?)[”\"']?(?:[？?。！!]\s*$|$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, q)
        if not m:
            continue
        text = str(m.group(1) or "").strip()
        text = re.sub(
            r"(?i)\b(?:please\s+point\s+me(?:\s+to)?|point\s+me(?:\s+to)?|show\s+me|source\s+section(?:s)?|cited\s+section(?:s)?|those\s+sources|source\s+too)\b.*$",
            "",
            text,
        ).strip()
        text = re.sub(r"(?i)^(?:the|this|that)\s+", "", text).strip()
        text = re.sub(r"(?i)\s+(?:in|from)\s+(?:my|your|the)\s+library\s*$", "", text).strip()
        text = re.sub(r"(?i)\s+(?:paper|papers)\s*$", "", text).strip()
        text = re.sub(r"(?:的|吗|呢|呀|啊)\s*$", "", text).strip()
        text = text.strip(" \"'“”‘’.,;:!?")
        if text:
            return text
    return _contract_focus_label(prompt)


def _contract_cross_paper_relation(prompt: str) -> tuple[str, str, str]:
    q = str(prompt or "")
    if re.search(r"(\bcompare\b|\bcomparison\b|\bcompares\b|\bversus\b|\bvs\.?\b|对比|比较|区别|差异)", q, flags=re.IGNORECASE):
        return "directly compares", "directly compare", "直接比较"
    if re.search(r"(\bdefine\b|\bdefines\b|\bdefinition\b|定义)", q, flags=re.IGNORECASE):
        return "directly defines", "directly define", "直接定义"
    if re.search(r"(\bmention\b|\bmentions\b|提到)", q, flags=re.IGNORECASE):
        return "mentions", "mention", "提到"
    if re.search(r"(\bcite\b|\bcites\b|\bcited\b|引用)", q, flags=re.IGNORECASE):
        return "cites", "cite", "引用"
    if re.search(r"(\buse\b|\buses\b|\bused\b|使用)", q, flags=re.IGNORECASE):
        return "uses", "use", "使用"
    if re.search(r"(\bdiscuss\b|\bdiscusses\b|\bdiscussed\b|讨论|涉及)", q, flags=re.IGNORECASE):
        return "discusses", "discuss", "讨论"
    return "matches", "match", "命中"


def _contract_requests_single_paper_lookup(prompt: str) -> bool:
    return bool(
        re.search(
            r"(\bwhich paper\b|\bwhat paper\b|\bpaper in (?:my|your) library\b|哪篇|哪一篇)",
            str(prompt or ""),
            flags=re.IGNORECASE,
        )
    )


def _best_contract_source_name(
    answer_hits: list[dict] | None,
    *,
    primary_evidence: dict | None = None,
) -> str:
    source_name = _contract_source_name_from_primary_evidence(primary_evidence)
    if source_name:
        return source_name
    names = _contract_unique_source_names(answer_hits, limit=1)
    return names[0] if names else ""


def _select_best_contract_hit_snippet(
    answer_hits: list[dict] | None,
    *,
    prompt: str,
    primary_evidence: dict | None = None,
) -> tuple[str, str, str]:
    primary_source = _contract_source_name_from_primary_evidence(primary_evidence)
    primary_heading = _contract_heading_from_primary_evidence(primary_evidence)
    for raw in _contract_snippet_candidates_from_primary_evidence(primary_evidence):
        snippet = _clean_contract_snippet(raw)
        if snippet:
            return primary_source, primary_heading, snippet
    best_source = ""
    best_heading = ""
    best_snippet = ""
    best_score = float("-inf")
    for hit in list(answer_hits or []):
        if not isinstance(hit, dict):
            continue
        source_name = _contract_source_name_from_hit(hit)
        heading = _contract_heading_from_hit(hit)
        heading_lower = heading.lower()
        for raw in _contract_snippet_candidates(hit):
            snippet = _clean_contract_snippet(raw)
            if not snippet:
                continue
            score = _score_contract_snippet(prompt, snippet, heading=heading)
            for keyword in _contract_prompt_keywords(prompt, max_items=4):
                if str(keyword or "").strip().lower() in heading_lower:
                    score += 0.75
            if score > best_score:
                best_score = score
                best_source = source_name
                best_heading = heading
                best_snippet = snippet
    return best_source, best_heading, best_snippet


def _contract_cross_paper_conclusion_needs_rewrite(conclusion: str, *, source_names: list[str] | None = None) -> bool:
    text = str(conclusion or "").strip()
    if not text:
        return True
    low = _normalize_contract_compare_text(text)
    if not low:
        return True
    if re.search(
        r"(\bretrieved context\b|\bonly includes\b|\bonly include\b|\bonly one paper\b|\bthere is only\b|\bboth from\b|"
        r"当前检索结果|只包括|只有一篇|仅包括)",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    for name in list(source_names or []):
        if _normalize_contract_compare_text(name) in low:
            return False
    return bool(source_names)


def _build_cross_paper_contract_conclusion(prompt: str, *, answer_hits: list[dict] | None, prefer_zh: bool) -> str:
    names = _contract_unique_source_names(answer_hits, limit=4)
    if not names:
        return ""
    topic = _contract_cross_paper_topic_label(prompt)
    singular_rel, plural_rel, zh_rel = _contract_cross_paper_relation(prompt)
    if prefer_zh:
        relation_prefix = f"{zh_rel}“{topic}”的" if topic else "相关"
        if len(names) == 1:
            return f"除当前论文外，库内另外命中 1 篇{relation_prefix}论文：{names[0]}。"
        display = "；".join(names[:3])
        if len(names) <= 3:
            return f"除当前论文外，库内另外命中 {len(names)} 篇{relation_prefix}论文：{display}。"
        return f"除当前论文外，库内另外命中 {len(names)} 篇{relation_prefix}论文，前几篇包括：{display} 等。"
    if topic:
        if len(names) == 1:
            return f"Besides the current paper, one additional library paper {singular_rel} {topic}: {names[0]}."
        display = ", ".join(names[:3])
        if len(names) <= 3:
            return f"Besides the current paper, {len(names)} additional library papers {plural_rel} {topic}: {display}."
        return f"Besides the current paper, {len(names)} additional library papers {plural_rel} {topic}; the top matches include {display}."
    if len(names) == 1:
        return f"Besides the current paper, one additional library paper matches this topic: {names[0]}."
    display = ", ".join(names[:3])
    if len(names) <= 3:
        return f"Besides the current paper, {len(names)} additional library papers match this topic: {display}."
    return f"Besides the current paper, {len(names)} additional library papers match this topic; the top matches include {display}."


def _build_negative_cross_paper_contract_conclusion(prompt: str, *, prefer_zh: bool) -> str:
    topic = _contract_cross_paper_topic_label(prompt)
    singular_rel, plural_rel, zh_rel = _contract_cross_paper_relation(prompt)
    if prefer_zh:
        if topic:
            return f"除当前论文外，当前检索结果里没有其他库内论文明确{zh_rel}“{topic}”。"
        return "除当前论文外，当前检索结果里没有其他库内论文明确命中这个主题。"
    if topic:
        return f"Besides the current paper, no other retrieved library paper explicitly {singular_rel} {topic}."
    return "Besides the current paper, no other retrieved library paper explicitly matches this topic."


def _contract_snippet_candidates(hit: dict) -> list[str]:
    if not isinstance(hit, dict):
        return []
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    out: list[str] = []
    for key in ("ref_show_snippets", "ref_snippets"):
        raw = (meta or {}).get(key)
        if not isinstance(raw, list):
            continue
        for item in raw[:3]:
            text = str(item or "").strip()
            if text:
                out.append(text)
    for key in ("text", "snippet"):
        text = str(hit.get(key) or "").strip()
        if text:
            out.append(text)
    return out


def _looks_contract_math_heavy(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    if "$$" in s or "\\tag{" in s:
        return True
    math_tokens = len(re.findall(r"(\\[A-Za-z]+|\b(?:sigma|mathbf|mathcal|frac|int)\b)", s))
    return math_tokens >= 3


def _clean_contract_snippet(text: str, *, max_len: int = 240) -> str:
    raw = str(text or "").strip()
    if (not raw) or _looks_contract_math_heavy(raw):
        return ""
    lines = [str(line or "").strip() for line in raw.splitlines() if str(line or "").strip()]
    while lines and lines[0].startswith("#"):
        lines = lines[1:]
    cleaned = " ".join(lines).strip()
    cleaned = re.sub(r"^\s*#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"\[\[CITE:[^\]]+\]\]|\[\d{1,4}\]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,.")
    if not cleaned:
        return ""
    parts = [part.strip() for part in re.split(r"(?<=[.!?。！？;；])\s+", cleaned) if part.strip()]
    chosen: list[str] = []
    total = 0
    for part in parts or [cleaned]:
        if not chosen and len(part) > max_len:
            chosen.append(part[:max_len].rstrip(" ,;:") + "...")
            break
        if chosen and (total + len(part) + 1) > max_len:
            break
        chosen.append(part)
        total += len(part) + 1
        if len(chosen) >= 2:
            break
    out = " ".join(chosen).strip()
    out = re.sub(r"\s+", " ", out).strip(" ,;:")
    return out


def _score_contract_snippet(prompt: str, snippet: str, *, heading: str = "") -> float:
    text = str(snippet or "").strip()
    if not text:
        return float("-inf")
    score = 0.0
    lower = text.lower()
    heading_text = str(heading or "").strip()
    heading_lower = heading_text.lower()
    combined = f"{heading_lower} {lower}".strip()
    prompt_lower = str(prompt or "").strip().lower()
    wants_definition = _contract_prompt_requests_definition(prompt)
    wants_comparison = _contract_prompt_requests_comparison(prompt)
    definition_like = bool(
        re.search(r"\b(defin(?:e|es|ed|ition)|known as|refers to|is called|means|term)\b", lower)
        or re.search(r"(定义|称为|指的是|即|是指)", text)
    )
    comparison_like = bool(
        re.search(r"\b(compare|compares|comparison|versus|vs\.?|contrast|difference)\b", lower)
        or re.search(r"(比较|对比|差异|区别)", text)
    )
    for keyword in _contract_prompt_keywords(prompt, max_items=6):
        key = str(keyword or "").strip().lower()
        if key in combined:
            score += 2.0
        if key and key in heading_lower:
            score += 0.75
    if 48 <= len(text) <= 240:
        score += 1.5
    if definition_like:
        score += 1.0
    if comparison_like:
        score += 1.0
    focus_label = _normalize_contract_compare_text(_contract_focus_label(prompt))
    if focus_label and focus_label in _normalize_contract_compare_text(combined):
        score += 3.0
    if wants_definition:
        if definition_like:
            score += 2.5
        if focus_label and focus_label in _normalize_contract_compare_text(heading_text):
            score += 2.0
        if heading_lower and re.search(r"\b(introduction|overview|definition|background)\b", heading_lower):
            score += 0.5
        if re.search(r"\b(however|alternatively|instead|would reduce|factor of)\b", lower) and (not definition_like):
            score -= 1.75
        if "point me to the source section" in prompt_lower and heading_text:
            score += 0.5
    if wants_comparison:
        if comparison_like:
            score += 2.5
        if heading_lower and re.search(r"\b(compare|comparison|versus|difference)\b", heading_lower):
            score += 1.5
    return score


def _build_hit_grounded_evidence_fallback(
    answer_hits: list[dict] | None,
    *,
    prompt: str,
    prefer_zh: bool,
    primary_evidence: dict | None = None,
) -> str:
    best_source, best_heading, best_snippet = _select_best_contract_hit_snippet(
        answer_hits,
        prompt=prompt,
        primary_evidence=primary_evidence,
    )
    if not best_snippet:
        return ""
    topic = _contract_cross_paper_topic_label(prompt) or _contract_focus_label(prompt)
    singular_rel, _plural_rel, zh_rel = _contract_cross_paper_relation(prompt)
    if prefer_zh:
        if _contract_text_is_english_heavy(best_snippet):
            return _build_hit_grounded_evidence_bridge_zh(
                prompt=prompt,
                best_source=best_source,
                best_heading=best_heading,
                best_snippet=best_snippet,
            )
        if best_source and best_heading and topic:
            return f"《{best_source}》的“{best_heading}”部分直接{zh_rel}了“{topic}”：{best_snippet}"
        if best_heading and topic:
            return f"命中章节“{best_heading}”直接{zh_rel}了“{topic}”：{best_snippet}"
        if best_source and topic:
            return f"《{best_source}》直接{zh_rel}了“{topic}”：{best_snippet}"
        if best_source and best_heading:
            return f"《{best_source}》的“{best_heading}”部分指出：{best_snippet}"
        if best_heading:
            return f"命中章节“{best_heading}”指出：{best_snippet}"
        if best_source:
            return f"《{best_source}》的命中段落指出：{best_snippet}"
        return best_snippet
    if best_source and best_heading and topic:
        return f'In {best_source}, the section "{best_heading}" {singular_rel} {topic}: {best_snippet}'
    if best_heading and topic:
        return f'The matched section "{best_heading}" {singular_rel} {topic}: {best_snippet}'
    if best_source and topic:
        return f"In {best_source}, the matched passage {singular_rel} {topic}: {best_snippet}"
    if best_source and best_heading:
        return f'In {best_source}, the section "{best_heading}" states: {best_snippet}'
    if best_heading:
        return f'The matched section "{best_heading}" states: {best_snippet}'
    if best_source:
        return f"In {best_source}, the matched passage states: {best_snippet}"
    return best_snippet


def _conclusion_needs_locale_bridge(conclusion: str, *, prefer_zh: bool) -> bool:
    text = str(conclusion or "").strip()
    if (not prefer_zh) or (not text):
        return False
    cjk = len(_ANSWER_CJK_CHAR_RE.findall(text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return latin >= 12 and cjk * 3 < latin


def _contract_text_is_english_heavy(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    cjk = len(_ANSWER_CJK_CHAR_RE.findall(s))
    latin = len(re.findall(r"[A-Za-z]", s))
    return latin >= 12 and cjk * 3 < latin


def _build_hit_grounded_conclusion_bridge(
    answer_hits: list[dict] | None,
    *,
    prompt: str,
    prefer_zh: bool,
    primary_evidence: dict | None = None,
) -> str:
    if not prefer_zh:
        return ""
    best_source, best_heading, best_snippet = _select_best_contract_hit_snippet(
        answer_hits,
        prompt=prompt,
        primary_evidence=primary_evidence,
    )
    topic = _contract_cross_paper_topic_label(prompt) or _contract_focus_label(prompt)
    _singular_rel, _plural_rel, zh_rel = _contract_cross_paper_relation(prompt)
    if best_source and best_heading and topic:
        return f"命中的论文《{best_source}》在“{best_heading}”里直接{zh_rel}了“{topic}”。"
    if best_heading and topic:
        return f"命中的论文在“{best_heading}”里直接{zh_rel}了“{topic}”。"
    if best_source and topic:
        return f"命中的论文《{best_source}》直接{zh_rel}了“{topic}”。"
    if best_heading:
        return f"命中的论文在“{best_heading}”里给出了直接相关的说明。"
    if best_source:
        return f"命中的论文《{best_source}》给出了直接相关的说明。"
    if best_snippet:
        return f"命中的论文直接指出：{best_snippet}"
    return ""


def _build_positive_contract_conclusion(
    prompt: str,
    *,
    answer_hits: list[dict] | None,
    primary_evidence: dict | None,
    prefer_zh: bool,
) -> str:
    source_name = _best_contract_source_name(answer_hits, primary_evidence=primary_evidence)
    topic = _contract_cross_paper_topic_label(prompt) or _contract_focus_label(prompt)
    if not source_name:
        return ""
    if prefer_zh:
        if _contract_prompt_requests_definition(prompt) and topic:
            return f"你库里最直接定义“{topic}”的论文是《{source_name}》。"
        if _contract_prompt_requests_comparison(prompt) and topic:
            return f"你库里直接比较“{topic}”的论文是《{source_name}》。"
        if topic:
            return f"你库里直接讨论“{topic}”的论文是《{source_name}》。"
        return f"当前命中的库内论文是《{source_name}》。"
    if _contract_prompt_requests_definition(prompt) and topic:
        return f"The paper in your library that most directly defines {topic} is {source_name}."
    if _contract_prompt_requests_comparison(prompt) and topic:
        return f"The paper in your library that directly compares {topic} is {source_name}."
    if topic:
        return f"The paper in your library that directly discusses {topic} is {source_name}."
    return f"The matched paper in your library is {source_name}."


def _positive_contract_conclusion_needs_rewrite(
    conclusion: str,
    *,
    prompt: str,
    source_name: str,
) -> bool:
    text = str(conclusion or "").strip()
    if not source_name:
        return False
    source_norm = _normalize_contract_compare_text(source_name)
    low = _normalize_contract_compare_text(text)
    if source_norm and source_norm in low:
        return False
    if not text:
        return True
    if text.endswith(":"):
        return True
    if _contract_requests_single_paper_lookup(prompt):
        if re.search(r"\b(this paper|the paper|matched paper|paper in (?:my|your) library)\b", text, flags=re.IGNORECASE):
            return True
        if _contract_prompt_requests_definition(prompt) and re.search(r"\bdefin(?:e|es|ed|ition)\b", text, flags=re.IGNORECASE):
            return True
        if _contract_prompt_requests_comparison(prompt) and re.search(r"\b(compare|comparison|versus|vs\.?)\b", text, flags=re.IGNORECASE):
            return True
    return False


def _build_hit_grounded_evidence_bridge_zh(
    *,
    prompt: str,
    best_source: str,
    best_heading: str,
    best_snippet: str,
) -> str:
    topic = _contract_cross_paper_topic_label(prompt) or _contract_focus_label(prompt)
    _, _, zh_rel = _contract_cross_paper_relation(prompt)
    if best_source and best_heading and topic:
        return f"《{best_source}》的“{best_heading}”部分直接{zh_rel}了“{topic}”；原文片段：{best_snippet}"
    if best_heading and topic:
        return f"命中章节“{best_heading}”直接{zh_rel}了“{topic}”；原文片段：{best_snippet}"
    if best_source and best_heading:
        return f"《{best_source}》的“{best_heading}”部分给出了直接相关的说明；原文片段：{best_snippet}"
    if best_heading:
        return f"命中章节“{best_heading}”给出了直接相关的说明；原文片段：{best_snippet}"
    if best_source:
        return f"《{best_source}》的命中段落给出了直接相关的说明；原文片段：{best_snippet}"
    return f"命中片段的原文为：{best_snippet}"


def _build_negative_contract_evidence_fallback(prompt: str, *, prefer_zh: bool) -> str:
    focus = _contract_focus_label(prompt) or ("该概念" if prefer_zh else "the requested concept")
    if prefer_zh:
        return f"命中的库内片段覆盖了相关上下文，但没有直接定义或明确讨论“{focus}”。"
    return f"The matched snippets cover adjacent material, but none explicitly define or directly discuss {focus}."


def _build_negative_cross_paper_contract_evidence_fallback(prompt: str, *, prefer_zh: bool) -> str:
    topic = _contract_cross_paper_topic_label(prompt) or _contract_focus_label(prompt) or ("该主题" if prefer_zh else "the requested topic")
    singular_rel, plural_rel, zh_rel = _contract_cross_paper_relation(prompt)
    if prefer_zh:
        return f"当前额外检索到的库内论文片段里，没有哪篇明确{zh_rel}“{topic}”。"
    return f"Among the additional retrieved library papers, none explicitly {plural_rel} {topic}."


def _build_negative_contract_next_steps(prompt: str, *, prefer_zh: bool) -> list[str]:
    focus = _contract_focus_label(prompt) or ("该概念" if prefer_zh else "the requested concept")
    if prefer_zh:
        return [
            f"在库内用“{focus}”及其全称、缩写或常见别名再检索一次，确认是否只是换了术语表达。",
            "优先检查最相关论文的方法、重建或优化章节，看这个概念是否以近义说法出现。",
            "如果你预期应该命中，补搜 1 到 2 个同义关键词后，再回看最接近的那篇论文。",
        ]
    return [
        f'Search your library for "{focus}" plus its full name, abbreviation, or close variants to see whether the concept appears under another term.',
        "Check the methods, reconstruction, or optimization sections of the nearest paper to see whether the idea appears without the exact keyword.",
        "If you expected a hit, try one or two synonym keywords and recheck the closest paper before concluding it is absent.",
    ]


def _build_cross_paper_contract_next_steps(
    prompt: str,
    *,
    answer_hits: list[dict] | None,
    prefer_zh: bool,
    negative: bool,
) -> list[str]:
    focus = _contract_cross_paper_topic_label(prompt) or _contract_focus_label(prompt) or ("该主题" if prefer_zh else "the topic")
    source_names = _contract_unique_source_names(answer_hits, limit=3)
    first_source = source_names[0] if source_names else ""
    if prefer_zh:
        if negative or (not first_source):
            return [
                f"先用“{focus}”及其同义词、缩写或相邻术语再检索一次，确认是否还有遗漏的库内论文。",
                "优先查看最接近当前主题的方法、结果或讨论章节，确认是否只是换了表达方式而没有直接用这个词。",
                "如果仍然没有额外命中，把这次结果记成“当前库内暂未找到其他相关论文”，避免过度外推。",
            ]
        return [
            f"先打开《{first_source}》的命中章节，对照它如何处理“{focus}”，并和当前论文比较证据位置与方法差异。",
            f"再用“{focus}”及其同义词或缩写扩检库内文献，看看是否还能补出第二层相关论文。",
            "把当前论文和新增命中文献做一张并排对照，记录主题、方法和定位到的证据段落。",
        ]
    if negative or (not first_source):
        return [
            f'Search your library again for "{focus}" plus close variants, abbreviations, or neighboring terms to check for missed papers.',
            "Inspect the nearest methods, results, or discussion sections to see whether another paper uses a different phrase for the same topic.",
            "If no extra hit appears, record this as 'no additional library paper found for now' instead of overgeneralizing beyond the current library.",
        ]
    return [
        f"Open the matched section in {first_source} and compare how it treats {focus} against the current paper, including method scope and evidence location.",
        f'Search your library again with "{focus}" plus close variants or abbreviations to expand beyond the first extra hit.',
        "Keep a short side-by-side note of the current paper and the additional matches, including topic scope, method, and locate-able evidence.",
    ]


def _build_positive_contract_next_steps(
    prompt: str,
    *,
    answer_hits: list[dict] | None,
    primary_evidence: dict | None,
    prefer_zh: bool,
    intent: str,
) -> list[str]:
    best_source, best_heading, _best_snippet = _select_best_contract_hit_snippet(
        answer_hits,
        prompt=prompt,
        primary_evidence=primary_evidence,
    )
    focus = _contract_cross_paper_topic_label(prompt) or _contract_focus_label(prompt) or ("该主题" if prefer_zh else "the topic")
    wants_definition = _contract_prompt_requests_definition(prompt)
    wants_comparison = _contract_prompt_requests_comparison(prompt)
    if prefer_zh:
        if wants_definition:
            steps: list[str] = []
            if best_source and best_heading:
                steps.append(f"先打开《{best_source}》的“{best_heading}”部分，核对它如何直接定义“{focus}”。")
            elif best_heading:
                steps.append(f"先回到“{best_heading}”这一节，核对它如何直接定义“{focus}”。")
            else:
                steps.append(f"先回到当前命中的原文段落，核对它如何直接定义“{focus}”。")
            steps.append(f"再用“{focus}”及其全称、缩写或近义说法在库内补检一次，确认是否还有论文用相邻术语讨论同一概念。")
            steps.append("把这条定义句和对应小节记成 1 条工作定义，后续引用时优先回到这段原文。")
            return steps
        if wants_comparison or intent == "compare":
            steps = []
            if best_source and best_heading:
                steps.append(f"先打开《{best_source}》的“{best_heading}”部分，逐条记下它比较“{focus}”时用到的维度和证据位置。")
            elif best_heading:
                steps.append(f"先打开“{best_heading}”这一节，逐条记下它比较“{focus}”时用到的维度和证据位置。")
            else:
                steps.append(f"先回到当前命中的原文段落，确认它比较“{focus}”时到底比较了哪些维度。")
            steps.append("做一张并排对照，记录方法差异、假设条件、效率或实验差异，以及各自对应的小节。")
            steps.append("再检查相邻章节是否补充了限制条件或实验边界，避免只凭一段比较就过度外推。")
            return steps
        steps = []
        if best_source and best_heading:
            steps.append(f"先打开《{best_source}》的“{best_heading}”部分，核对主回答引用的原文句子和定位是否一致。")
        elif best_heading:
            steps.append(f"先打开“{best_heading}”这一节，核对主回答引用的原文句子和定位是否一致。")
        else:
            steps.append("先回到当前命中的原文段落，核对主回答引用的原文句子和定位是否一致。")
        steps.append(f"再围绕“{focus}”查看相邻小节或图表，确认有没有补充说明、限制条件或实现细节。")
        steps.append("把论文名、小节名和关键证据句记成一条可回溯笔记，后续引用时优先链接到这条定位证据。")
        return steps

    if wants_definition:
        steps = []
        if best_source and best_heading:
            steps.append(f'Open "{best_heading}" in {best_source} and verify the exact sentence that defines {focus}.')
        elif best_heading:
            steps.append(f'Open the matched section "{best_heading}" and verify the exact sentence that defines {focus}.')
        else:
            steps.append(f"Return to the matched passage and verify the exact sentence that defines {focus}.")
        steps.append(f'Search your library again for "{focus}" plus full-name, abbreviation, or close-variant terms to see whether nearby papers use a different label for the same concept.')
        steps.append("Save one canonical definition sentence with its section heading so later notes stay tied to the original evidence.")
        return steps
    if wants_comparison or intent == "compare":
        steps = []
        if best_source and best_heading:
            steps.append(f'Open "{best_heading}" in {best_source} and list the exact comparison dimensions used for {focus}.')
        elif best_heading:
            steps.append(f'Open the matched section "{best_heading}" and list the exact comparison dimensions used for {focus}.')
        else:
            steps.append(f"Return to the matched passage and confirm which dimensions it actually compares for {focus}.")
        steps.append("Build a short side-by-side note covering method differences, assumptions, efficiency or experiment deltas, and the evidence location for each point.")
        steps.append("Check the neighboring section for limits or boundary conditions before generalizing from a single comparison passage.")
        return steps
    steps = []
    if best_source and best_heading:
        steps.append(f'Open "{best_heading}" in {best_source} and verify that the cited sentence matches the locate target.')
    elif best_heading:
        steps.append(f'Open the matched section "{best_heading}" and verify that the cited sentence matches the locate target.')
    else:
        steps.append("Return to the matched passage and verify that the cited sentence matches the locate target.")
    steps.append(f'Read one neighboring section or figure around {focus} to capture caveats, implementation detail, or scope that the first snippet may omit.')
    steps.append("Save the paper name, section heading, and one key evidence sentence as a short traceable note for reuse.")
    return steps


def _build_default_next_steps(*, intent: str, has_hits: bool, locale: str = "en") -> list[str]:
    by_intent_en = {
        "reading": [
            "Check the cited section/figure to verify the conclusion details.",
            "Compare this result against one baseline paper from the same period.",
            "Write a 3-line takeaway linked to the cited evidence for your notes.",
        ],
        "compare": [
            "Build a side-by-side table with assumptions, compute cost, and expected gains.",
            "Choose one metric where methods diverge most and run a small pilot test.",
            "Document the boundary conditions where each method is likely to fail.",
        ],
        "idea": [
            "Define the minimum viable experiment that can falsify this idea within one week.",
            "List the top 2 technical risks and one mitigation for each risk.",
            "Search for one recent paper that tests a similar hypothesis and compare setup.",
        ],
        "experiment": [
            "Fix the control group and only vary one key factor in the first run.",
            "Predefine evaluation metrics and stopping criteria before training starts.",
            "Record reproducibility settings (seed, environment, data split) for each run.",
        ],
        "troubleshoot": [
            "Reproduce the issue on a minimal case and log exact error/output deltas.",
            "Check environment, dependency, and data preprocessing mismatches first.",
            "Apply one change at a time and validate with a short regression test.",
        ],
        "writing": [
            "Rewrite one paragraph using claim -> evidence -> implication order.",
            "Replace broad statements with measurable or cited wording.",
            "Run a final pass for logical transitions between adjacent paragraphs.",
        ],
    }
    by_intent_zh = {
        "reading": [
            "优先核对被引用的具体段落/图表，确认结论对应的原文证据。",
            "找一篇同阶段基线文献做并行对比，判断结论是否稳健。",
            "把本次结论整理成 3 行读书卡片，并附上对应引文编号。",
        ],
        "compare": [
            "做一张并排对比表：假设条件、计算代价、预期收益各列一行。",
            "挑一个差异最大的指标先做小样本验证，快速判断优劣。",
            "补充两种方法各自的失效边界，避免误用到不适配场景。",
        ],
        "idea": [
            "定义一个一周内可执行的最小可证伪实验来验证这个想法。",
            "列出前 2 个技术风险，并给出每个风险的缓解方案。",
            "补查 1 篇近期相似假设论文，重点对比实验设置与结果。",
        ],
        "experiment": [
            "先固定对照组，只改变一个关键变量完成首轮实验。",
            "训练前预先写清评价指标与停止条件，减少后验偏差。",
            "逐次记录复现配置（随机种子、环境、数据切分）便于回溯。",
        ],
        "troubleshoot": [
            "先构造最小复现样例，并记录精确报错与输出差异。",
            "优先检查环境、依赖版本和数据预处理是否一致。",
            "每次只改一个变量，并配合短回归测试验证修复效果。",
        ],
        "writing": [
            "按“结论-证据-意义”重写一段，先提升主线清晰度。",
            "把泛化表述替换为可量化或可引用的具体表述。",
            "最后做一轮段间衔接检查，确保上下文过渡自然。",
        ],
    }
    use_zh = str(locale or "").strip().lower().startswith("zh")
    by_intent = by_intent_zh if use_zh else by_intent_en
    steps = list(by_intent.get(intent, by_intent["reading"]))
    if not has_hits:
        if use_zh:
            steps.insert(0, "补充 1-2 篇目标论文或具体章节，下次回答才能更好地基于证据。")
        else:
            steps.insert(0, "Add one or two target papers or sections so the next answer can be evidence-grounded.")
    return steps


def _apply_answer_contract_v1(
    answer: str,
    *,
    prompt: str,
    has_hits: bool,
    answer_hits: list[dict] | None = None,
    primary_evidence: dict | None = None,
    intent: str,
    depth: str,
    output_mode: str = "reading_guide",
) -> str:
    _ = str(prompt or "")
    src = str(answer or "").strip()
    if not src:
        return src

    notice, body0 = _split_kb_miss_notice(src)
    body = str(body0 if notice else src).strip()
    if not body:
        return src
    if _has_sufficient_answer_sections(
        body,
        has_hits=has_hits,
        output_mode=output_mode,
        answer_hits=answer_hits,
    ):
        return _cleanup_structured_answer_sections(
            src,
            prompt=prompt,
            has_hits=has_hits,
            answer_hits=answer_hits,
        )

    paras = [p.strip() for p in re.split(r"\n{2,}", body) if str(p or "").strip()]
    if not paras:
        return src

    conclusion = re.sub(_ANSWER_SECTION_PREFIX_RE, "", paras[0], count=1).strip() or paras[0]
    tail = paras[1:]
    evidence: list[str] = []
    extra: list[str] = []
    seeded_limits: list[str] = []
    seeded_steps: list[str] = []
    negative_shell = bool(_ANSWER_NEGATIVE_SHELL_RE.search(body))
    for p in tail:
        p1 = str(p or "").strip()
        if _ANSWER_NEXT_STEPS_HEADER_RE.search(p1) or re.search(
            r"(?im)^\s*\*\*(?:next\s*step|next\s*steps|下一步建议|下一步|建议)\*\*\s*[:：]?",
            p1,
        ):
            remainder = _normalize_contract_step_item(p1)
            if remainder:
                split_steps = [_normalize_contract_step_item(item) for item in _split_contract_ordered_items(remainder)]
                if split_steps:
                    seeded_steps.extend(item for item in split_steps if item)
                else:
                    seeded_steps.append(remainder)
            continue
        if re.match(r"(?im)^\s*(?:\*\*)?(?:Evidence|依据|证据)(?:\*\*)?\s*[:：]?", p1):
            remainder = _normalize_contract_evidence_item(p1)
            if remainder and (not _evidence_section_is_too_thin(remainder, answer_hits=answer_hits)):
                evidence.append(remainder)
            continue
        if re.match(r"(?im)^\s*(?:\*\*)?(?:Limits|边界|限制|局限)(?:\*\*)?\s*[:：]?", p1):
            remainder = _normalize_contract_limits_item(p1)
            if remainder:
                seeded_limits.append(remainder)
            continue
        if _ANSWER_CITE_HINT_RE.search(p):
            evidence.append(p)
            continue
        classification = _classify_contract_extra_paragraph(p1) if has_hits else "extra"
        if classification == "evidence":
            evidence.append(_normalize_contract_evidence_item(p1))
        elif classification == "limits":
            seeded_limits.append(_normalize_contract_limits_item(p1))
        elif classification == "step":
            step_item = _normalize_contract_step_item(p1)
            if step_item:
                seeded_steps.append(step_item)
        else:
            extra.append(p1)
    if has_hits and not evidence:
        evidence = _extract_cited_sentences(body, limit=2)

    prefer_zh = _prefer_zh_locale(prompt, src)
    cross_paper_query = _contract_requests_cross_paper_query(prompt)
    cross_paper_sources = _contract_unique_source_names(answer_hits, limit=4) if has_hits else []
    if cross_paper_query:
        if cross_paper_sources and _contract_cross_paper_conclusion_needs_rewrite(
            conclusion,
            source_names=cross_paper_sources,
        ):
            rewritten_conclusion = _build_cross_paper_contract_conclusion(
                prompt,
                answer_hits=answer_hits,
                prefer_zh=prefer_zh,
            )
            if rewritten_conclusion:
                conclusion = rewritten_conclusion
        elif negative_shell and (not cross_paper_sources):
            conclusion = _build_negative_cross_paper_contract_conclusion(prompt, prefer_zh=prefer_zh)
    elif has_hits and _positive_contract_conclusion_needs_rewrite(
        conclusion,
        prompt=prompt,
        source_name=_best_contract_source_name(answer_hits, primary_evidence=primary_evidence),
    ):
        rewritten_conclusion = _build_positive_contract_conclusion(
            prompt,
            answer_hits=answer_hits,
            primary_evidence=primary_evidence,
            prefer_zh=prefer_zh,
        )
        if rewritten_conclusion:
            conclusion = rewritten_conclusion
    elif has_hits and _conclusion_needs_locale_bridge(conclusion, prefer_zh=prefer_zh):
        bridged_conclusion = _build_hit_grounded_conclusion_bridge(
            answer_hits,
            prompt=prompt,
            prefer_zh=prefer_zh,
            primary_evidence=primary_evidence,
        )
        if bridged_conclusion:
            conclusion = bridged_conclusion
    steps_required = _answer_output_mode_requires_next_steps(output_mode)
    labels = {
        "conclusion": "结论" if prefer_zh else "Conclusion",
        "evidence": "依据" if prefer_zh else "Evidence",
        "limits": "边界" if prefer_zh else "Limits",
        "next_steps": "下一步" if prefer_zh else "Next Steps",
    }
    evidence = [item for item in evidence if item]
    weak_evidence = bool(evidence) and all(
        _evidence_section_is_too_thin(item, answer_hits=answer_hits) or _is_contract_title_only_evidence_item(item)
        for item in evidence
    )
    if has_hits and ((not evidence) or weak_evidence):
        fallback = ""
        if negative_shell:
            fallback = (
                _build_negative_cross_paper_contract_evidence_fallback(prompt, prefer_zh=prefer_zh)
                if cross_paper_query
                else _build_negative_contract_evidence_fallback(prompt, prefer_zh=prefer_zh)
            )
        else:
            fallback = _build_hit_grounded_evidence_fallback(
                answer_hits,
                prompt=prompt,
                prefer_zh=prefer_zh,
                primary_evidence=primary_evidence,
            )
        evidence.append(
            fallback
            or (
                "命中库内片段支持该结论，建议优先核对对应原文段落/图表。"
                if prefer_zh
                else "Retrieved library snippets support this conclusion; verify against the cited source passage/figure."
            )
        )
    normalized_evidence = [_normalize_contract_evidence_item(item) for item in evidence]
    evidence = [item for item in normalized_evidence if item]
    if len(evidence) > 1:
        substantive = [
            item
            for item in evidence
            if (not _evidence_section_is_too_thin(item, answer_hits=answer_hits))
            and (not _is_contract_title_only_evidence_item(item))
        ]
        if substantive:
            evidence = substantive

    limits: list[str] = []
    for item in seeded_limits:
        normalized_limit = _normalize_contract_limits_item(item)
        if normalized_limit:
            limits.append(normalized_limit)
    if _ANSWER_LIMITS_HINT_RE.search(body):
        limits.append(
            "部分细节依赖假设或在当前上下文中未明确给出。"
            if prefer_zh
            else "Some details may depend on assumptions or are not explicit in the available context."
        )
    if not has_hits:
        limits.append(
            "当前未检索到可直接引用的库内片段，本回答属于通用指导。"
            if prefer_zh
            else "This answer is general guidance because no direct library snippets were retrieved."
        )
    if (not limits) and depth == "L3":
        limits.append(
            "在将结论作为最终证据前，请回到原文核验关键假设。"
            if prefer_zh
            else "Validate key assumptions against the original paper before using this as final evidence."
        )

    depth_level = depth if depth in _ANSWER_DEPTHS else "L2"
    step_limit = 1 if depth_level == "L1" else (3 if depth_level == "L3" else 2)
    body_limit = 2 if depth_level == "L1" else (6 if depth_level == "L3" else 4)
    steps: list[str] = []
    if steps_required:
        if seeded_steps:
            for item in seeded_steps:
                normalized_step = _normalize_contract_step_item(item)
                if normalized_step:
                    steps.append(normalized_step)
        else:
            steps = (
                _build_cross_paper_contract_next_steps(
                    prompt,
                    answer_hits=answer_hits,
                    prefer_zh=prefer_zh,
                    negative=bool(negative_shell or (cross_paper_query and not cross_paper_sources)),
                )
                if cross_paper_query
                else (
                    _build_negative_contract_next_steps(prompt, prefer_zh=prefer_zh)
                    if negative_shell
                    else _build_positive_contract_next_steps(
                        prompt,
                        answer_hits=answer_hits,
                        primary_evidence=primary_evidence,
                        prefer_zh=prefer_zh,
                        intent=intent if intent in _ANSWER_INTENTS else "reading",
                    )
                )
            )
        steps = steps[:step_limit]

    parts: list[str] = []
    if notice:
        parts.append(notice)
    parts.append(f"{labels['conclusion']}: {conclusion}")
    if has_hits and evidence:
        parts.append(f"{labels['evidence']}:\n" + "\n".join(f"{i}. {item}" for i, item in enumerate(evidence[:3], start=1)))
    if limits:
        parts.append(f"{labels['limits']}:\n" + "\n".join(f"- {item}" for item in limits[:2]))
    if steps:
        parts.append(f"{labels['next_steps']}:\n" + "\n".join(f"{i}. {item}" for i, item in enumerate(steps, start=1)))
    if extra:
        parts.append("\n\n".join(extra[:body_limit]))

    contracted = "\n\n".join(parts).strip()
    if len(src) >= 700 and len(contracted) < int(len(src) * 0.65):
        details = "\n\n".join(extra).strip()
        if details and (details not in contracted):
            details_title = "补充细节" if prefer_zh else "Additional Details"
            contracted = f"{contracted}\n\n{details_title}:\n{details}".strip()
    return contracted


def _enhance_kb_miss_fallback(
    answer: str,
    *,
    has_hits: bool,
    intent: str,
    depth: str,
    contract_enabled: bool,
    output_mode: str = "reading_guide",
) -> str:
    src = str(answer or "").strip()
    if (not src) or has_hits:
        return src
    notice, body0 = _split_kb_miss_notice(src)
    if not notice:
        return src
    body = str(body0 or "").strip()
    if not body:
        body = "当前没有检索到可直接引用的库内片段，先给你一个可执行的通用路径。"
    if not contract_enabled:
        return f"{notice}\n\n{body}".strip()
    if _ANSWER_NEXT_STEPS_HEADER_RE.search(body):
        return f"{notice}\n\n{body}".strip()
    if _ANSWER_ORDERED_LIST_RE.search(body):
        return f"{notice}\n\n{body}".strip()
    if not _answer_output_mode_requires_next_steps(output_mode):
        return f"{notice}\n\n{body}".strip()

    depth_level = depth if depth in _ANSWER_DEPTHS else "L2"
    step_limit = 1 if depth_level == "L1" else (3 if depth_level == "L3" else 2)
    intent_norm = intent if intent in _ANSWER_INTENTS else "reading"
    prefer_zh = _prefer_zh_locale(body, notice)
    steps = _build_default_next_steps(
        intent=intent_norm,
        has_hits=False,
        locale="zh" if prefer_zh else "en",
    )[:step_limit]
    if not steps:
        return f"{notice}\n\n{body}".strip()
    step_lines = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, start=1))
    next_steps_title = "下一步建议" if prefer_zh else "Next Steps"
    return f"{notice}\n\n{body}\n\n{next_steps_title}:\n{step_lines}".strip()

