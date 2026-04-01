from __future__ import annotations

import re


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
_ANSWER_INTENTS = {"reading", "compare", "idea", "experiment", "troubleshoot", "writing"}
_ANSWER_DEPTHS = {"L1", "L2", "L3"}
_ANSWER_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
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
    joined = " ".join(str(t or "") for t in texts if str(t or ""))
    if not joined:
        return False
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


def _has_sufficient_answer_sections(text: str, *, has_hits: bool, output_mode: str = "reading_guide") -> bool:
    keys = set(_extract_answer_section_keys(text))
    next_steps_required = _answer_output_mode_requires_next_steps(output_mode)
    if "conclusion" not in keys:
        return False
    if next_steps_required and ("next_steps" not in keys):
        return False
    if bool(has_hits) and ("evidence" not in keys):
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
    if _has_sufficient_answer_sections(body, has_hits=has_hits, output_mode=output_mode):
        return src

    paras = [p.strip() for p in re.split(r"\n{2,}", body) if str(p or "").strip()]
    if not paras:
        return src

    conclusion = re.sub(_ANSWER_SECTION_PREFIX_RE, "", paras[0], count=1).strip() or paras[0]
    tail = paras[1:]
    evidence: list[str] = []
    extra: list[str] = []
    for p in tail:
        p1 = str(p or "").strip()
        if _ANSWER_NEXT_STEPS_HEADER_RE.search(p1):
            continue
        if _ANSWER_CITE_HINT_RE.search(p):
            evidence.append(p)
        else:
            extra.append(p)
    if has_hits and not evidence:
        evidence = _extract_cited_sentences(body, limit=2)

    prefer_zh = _prefer_zh_locale(prompt, src)
    steps_required = _answer_output_mode_requires_next_steps(output_mode)
    labels = {
        "conclusion": "结论" if prefer_zh else "Conclusion",
        "evidence": "依据" if prefer_zh else "Evidence",
        "limits": "边界" if prefer_zh else "Limits",
        "next_steps": "下一步" if prefer_zh else "Next Steps",
    }
    if has_hits and (not evidence):
        evidence.append(
            "命中库内片段支持该结论，建议优先核对对应原文段落/图表。"
            if prefer_zh
            else "Retrieved library snippets support this conclusion; verify against the cited source passage/figure."
        )

    limits: list[str] = []
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
        steps = _build_default_next_steps(
            intent=intent if intent in _ANSWER_INTENTS else "reading",
            has_hits=has_hits,
            locale="zh" if prefer_zh else "en",
        )[:step_limit]

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

