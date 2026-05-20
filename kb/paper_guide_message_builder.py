from __future__ import annotations

from kb.answer_contract import (
    _build_answer_contract_system_rules,
    _build_paper_guide_grounding_rules,
)
from kb.paper_guide_prompting import (
    _PAPER_GUIDE_CITATION_LOOKUP_PROMPT_RE,
    _paper_guide_allows_citeless_answer,
)


def _build_generation_prompt_bundle(
    *,
    prompt: str,
    ctx: str,
    paper_guide_mode: bool,
    paper_guide_bound_source_ready: bool,
    paper_guide_prompt_family: str,
    answer_intent: str,
    answer_depth: str,
    answer_output_mode: str,
    answer_contract_v1: bool,
    has_answer_hits: bool,
    locked_citation_source: dict | None,
    image_first_prompt: bool,
    anchor_grounded_answer: bool,
    paper_guide_special_focus_block: str,
    paper_guide_support_slots_block: str,
    paper_guide_evidence_cards_block: str,
    paper_guide_citation_grounding_block: str,
    paper_guide_reference_opportunities_block: str = "",
    image_attachment_count: int = 0,
) -> dict:
    prompt_for_user = str(prompt or "").strip() or "[Image attachment only request]"
    prompt_family = str(paper_guide_prompt_family or "").strip()
    allows_citeless = bool(_paper_guide_allows_citeless_answer(prompt_family))
    paper_guide_contract_enabled = False if paper_guide_mode else bool(answer_contract_v1)

    system = (
        "You are zaya, a personal knowledge-base assistant developed by P&I Lab.\n"
        "Answer the user's question directly and keep the response concise, concrete, and evidence-aware.\n"
        "Use retrieved snippets when they are available.\n"
        "If the retrieved evidence is missing or incomplete, say that clearly instead of fabricating paper details.\n"
        "Do not invent papers, equations, numbers, baselines, or conclusions that are not supported by the retrieved context.\n"
        "Do not output retrieval diagnostics, Top-K lists, or reference-location dumps unless the user explicitly asks for them.\n"
        "For math, use inline $...$ for short symbols and $$...$$ for longer equations; do not wrap equations in backticks.\n"
        "If the user asks for code, pseudocode, or derivation, provide directly usable output instead of only high-level discussion.\n"
    )
    if paper_guide_mode:
        system += (
            "\nStructured citation protocol:\n"
            "- Context headers contain [SID:<sid>] identifiers.\n"
            "- Retrieval block labels like DOC-1 / DOC-2 are context ids only, not paper reference numbers.\n"
            "- Never mention DOC-k retrieval labels in the user-visible answer; they are internal grounding ids only.\n"
            "- When citing paper references, MUST use [[CITE:<sid>:<ref_num>]].\n"
            "- Example: [[CITE:s1a2b3c4:24]] or [[CITE:s1a2b3c4:24]][[CITE:s1a2b3c4:25]].\n"
            "- Do NOT output free-form numeric citations like [24] / [2][4].\n"
            "- NEVER output malformed markers like [[CITE:<sid>]] or [CITE:<sid>] (missing ref_num).\n"
            "\n中文引用规范：\n"
            "- 上下文头部中的 [SID:<sid>] 是来源文献的唯一标识。\n"
            "- 当回答中提及原文已有的参考文献编号时，必须使用 [[CITE:<sid>:<ref_num>]] 格式，不可使用裸 [n] 格式。\n"
            "- 示例：\"如 Sen 等人 [[CITE:s9a2b3c4:3]] 所述\"，而不是 \"如 Sen 等人 [3] 所述\"。\n"
            "- <sid> 必须从对应文献上下文头部的 [SID:xxx] 获取。\n"
            "- 裸 [n] 格式无法区分是哪篇文献的参考文献，会导致跳转链接错误。\n"
        )
    else:
        system += (
            "\nIn-paper reference tracking — CRITICAL:\n"
            "- Retrieved snippets contain original-paper reference markers like [3,58] from the paper's own bibliography.\n"
            "- Context headers also list candidate refs explicitly (e.g. \"candidate refs: 3, 58, 4, 5\").\n"
            "- When you mention or repeat ANY such reference number from a paper's bibliography, "
            "you MUST use [[CITE:<sid>:<ref_num>]] format. NEVER output bare [n] for in-paper references.\n"
            "- The <sid> is shown in the context header as [SID:<sid>] for each document.\n"
            "- Example: write \"as shown by Sen et al. [[CITE:s9a2b3c4:3]]\" instead of \"as shown by Sen et al. [3]\".\n"
            "- When listing individual bibliographic entries (e.g. \"ref [3] is ..., ref [4] is ...\"), "
            "MUST use [[CITE:<sid>:3]] and [[CITE:<sid>:4]] for each, NOT bare [3] [4].\n"
            "- Only use [[CITE:...]] for reference numbers that belong to the original paper's bibliography; "
            "do NOT use it for your own snippet citations (see snippet rule below).\n"
            "- ⚠️ 检索片段的上下文头部标注了该文献的 SID 和 candidate refs（如 candidate refs: 3, 58, 4, 5）。\n"
            "- 当你在回答中提及原文已有的参考文献编号（如 [3]、[4]、[5]）时，必须使用 [[CITE:<sid>:<ref_num>]] 格式，"
            "绝对不允许使用裸 [3] [4] [5]。\n"
            "- 示例：\"1976 年的高光谱成像仪 [[CITE:s9a2b3c4:3]] 是早期工作之一\"，而不是全用 [2]。\n"
            "- 裸 [n] 格式无法区分是哪篇论文的参考文献，会导致跳转链接错误、用户无法查看来源。\n"
            "\nSnippet citation rule:\n"
            "- Retrieved snippets are labeled DOC-1, DOC-2, etc. in the context.\n"
            "- When citing information from a snippet, MUST use offset citation numbers: "
            "DOC-1 → [10001], DOC-2 → [10002], DOC-3 → [10003], and so on.\n"
            "- Example: \"The method achieves state-of-the-art results [10001].\"\n"
            "- CRITICAL: Include [10001] [10002] markers regardless of response language. "
            "When writing in Chinese, still use [10001][10002] markers in the same way.\n"
            "- Chinese example: \"单像素成像是一种计算成像技术 [10001]，它使用单个探测器 [10002]。\"\n"
            "- ⚠️ 绝对不要使用 [1] [2] [3] 等裸编号引用检索片段！对检索片段的引用必须始终使用 [10001] [10002] [10003] 偏移编号。\n"
            "- 即使你觉得 [10001] 可以简写为 [1]，也绝对不允许！引用检索片段只能用 [10001] 格式。\n"
            "- Never mention DOC-k labels directly in the visible answer; use [10001] [10002] markers instead.\n"
        )
    if paper_guide_mode and allows_citeless:
        system += (
            "\nPaper-guide abstract rule:\n"
            "- When the user asks for the abstract or its translation, use the paper's Abstract section itself.\n"
            "- Prefer Abstract over Introduction, Results, Discussion, Methods, and References.\n"
            "- Output the abstract body itself before adding any translation.\n"
            "- Preserve sentence order from the abstract span, and do NOT append title, author list, explanatory notes, or in-paper citation markers unless the user explicitly asks for them or the quoted abstract text itself contains them.\n"
        )
    if (
        locked_citation_source
        and (not allows_citeless)
        and (not _PAPER_GUIDE_CITATION_LOOKUP_PROMPT_RE.search(prompt_for_user))
    ):
        locked_sid = str((locked_citation_source or {}).get("sid") or "").strip()
        locked_name = str((locked_citation_source or {}).get("source_name") or "").strip()
        system += (
            "\nCitation source lock:\n"
            f"- This answer is primarily grounded in [SID:{locked_sid}] {locked_name}.\n"
            f"- Include at least one valid [[CITE:{locked_sid}:<ref_num>]] when the answer uses retrieved evidence.\n"
            "- Only switch to another SID when the same reference number cannot be verified in the locked source.\n"
        )
    if image_first_prompt:
        system += (
            "\nImage-first rule:\n"
            "- The user is asking about the attached image itself.\n"
            "- Analyze the attached image first.\n"
            "- Use retrieved paper context only as secondary background, not as a substitute for visual inspection.\n"
        )
    if anchor_grounded_answer:
        system += (
            "\nAnchor-grounded answer rule:\n"
            "- The requested numbered figure/equation/theorem is already matched in the retrieved library context.\n"
            "- Answer from the matched snippets and the same document's retrieved context.\n"
            "- Do NOT say the item is missing, unavailable, inferred only from a public version, or that later sections may possibly add details unless the retrieved context explicitly shows that.\n"
            "- If a detail is not shown in the retrieved context, say it is not shown in the retrieved context; do not speculate that it might appear later.\n"
        )
    if paper_guide_contract_enabled:
        system += _build_answer_contract_system_rules(
            intent=answer_intent,
            depth=answer_depth,
            has_hits=bool(has_answer_hits),
            output_mode=answer_output_mode,
        )
    if paper_guide_mode and paper_guide_bound_source_ready:
        system += _build_paper_guide_grounding_rules(
            answer_contract_v1=bool(paper_guide_contract_enabled),
            output_mode=answer_output_mode,
            prompt_family=prompt_family,
        )
    if paper_guide_mode and paper_guide_support_slots_block:
        system += (
            "\nPaper-guide support-slot protocol:\n"
            "- Prefer the exact support_example marker from the paper-guide support slots block for paper-grounded claims instead of guessing a paper reference number directly.\n"
            "- Runtime will resolve [[SUPPORT:...]] into the final structured citation or locate-only grounding.\n"
            "- Use direct [[CITE:<sid>:<ref_num>]] only when copying an explicit cite_example exactly.\n"
        )
    if paper_guide_reference_opportunities_block:
        system += (
            "\nUpstream-reference protocol:\n"
            "- Upstream reference opportunities are System B links to a retrieved paper's bibliography, not ordinary snippet citations.\n"
            "- For ordinary beginner, concept, origin, prior-work, or method-background questions, cite those upstream works inline on the sentence that explains them.\n"
            "- Answer the user's substantive question first; for example, say whether the method is original, borrowed, prior work, or background before adding the cite_example.\n"
            "- Never begin the final answer with locator-only shells such as 'The paper cites...', 'This is stated in...', or 'Source location...'.\n"
            "- Do not add a separate bibliography trail unless the user explicitly asks for a reading list or reference list.\n"
            "- Only use an upstream cite_example when the answer actually discusses the matching concept or prior work.\n"
        )

    cite_reminder = ""
    if (not paper_guide_mode) and has_answer_hits:
        cite_reminder = (
            "\n【强制要求】每次引用上方检索片段中的内容时，必须在对应的句尾标注 [10001] [10002] [10003] 等偏移编号标记（注意：编号从 10001 开始，不是从 1 开始！）。中文回答同样必须标注，不可省略！\n"
            "【强制要求】严禁使用 [1] [2] [3] 等裸编号引用检索片段！[1] [2] 会被误认为是原文的参考文献编号，导致引用断裂。必须用 [10001] [10002]！\n"
            "[REQUIRED] Append [10001] [10002] [10003] (offset) markers after each claim that uses retrieved context. "
            "IMPORTANT: snippet numbering starts at 10001, not 1! This applies to ALL languages.\n"
            "【强制要求】如果回答中提及原文已有的参考文献编号（如 [3,58]），必须使用 [[CITE:<sid>:<ref_num>]] 格式，不可使用裸 [n]。<sid> 从上下文头部的 [SID:xxx] 获取。\n"
            "[REQUIRED] When citing in-paper bibliography reference numbers (e.g. [3,58] from the original paper), "
            "MUST use [[CITE:<sid>:<ref_num>]] instead of bare [n]. The <sid> comes from [SID:xxx] in the context header.\n"
        )
    user = (
        f"Question:\n{prompt_for_user}\n\n{cite_reminder}"
        f"Retrieved context (with deep-read supplements):\n{ctx if ctx else '(none)'}\n"
    )
    if paper_guide_special_focus_block:
        user += f"\n{paper_guide_special_focus_block}\n"
    if paper_guide_support_slots_block:
        user += f"\n{paper_guide_support_slots_block}\n"
    if paper_guide_reference_opportunities_block:
        user += f"\n{paper_guide_reference_opportunities_block}\n"
    if paper_guide_evidence_cards_block:
        user += f"\n{paper_guide_evidence_cards_block}\n"
    if paper_guide_citation_grounding_block:
        user += f"\n{paper_guide_citation_grounding_block}\n"
    if anchor_grounded_answer:
        user += (
            "\nAnchor-grounded retrieval: the requested numbered item is already matched in the library snippets above. "
            "Resolve the answer from those snippets and any explicit follow-up context already retrieved from the same document.\n"
        )
    if int(image_attachment_count or 0) > 0:
        user += (
            f"\nAttached images: {int(image_attachment_count)}. "
            "These images are part of the current request. Inspect them directly before answering. "
            "Do not claim that no image was uploaded.\n"
        )

    return {
        "system": system,
        "user": user,
        "prompt_for_user": prompt_for_user,
        "paper_guide_contract_enabled": bool(paper_guide_contract_enabled),
    }
