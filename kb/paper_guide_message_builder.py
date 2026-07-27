from __future__ import annotations

from kb.answer_contract import (
    _build_answer_contract_system_rules,
    _build_paper_guide_grounding_rules,
)
from kb.paper_guide_prompting import (
    _PAPER_GUIDE_CITATION_LOOKUP_PROMPT_RE,
    _paper_guide_allows_citeless_answer,
)
from kb.research_answer_plan import infer_research_answer_plan


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
    citation_plan_block: str = "",
    image_attachment_count: int = 0,
) -> dict:
    prompt_for_user = str(prompt or "").strip() or "[Image attachment only request]"
    prompt_family = str(paper_guide_prompt_family or "").strip()
    allows_citeless = bool(_paper_guide_allows_citeless_answer(prompt_family))
    paper_guide_contract_enabled = False if paper_guide_mode else bool(answer_contract_v1)
    research_plan = None
    if not allows_citeless:
        research_plan = infer_research_answer_plan(
            prompt=prompt_for_user,
            paper_guide_prompt_family=prompt_family,
            answer_intent=answer_intent,
            answer_output_mode=answer_output_mode,
            paper_guide_mode=paper_guide_mode,
        )

    system = (
        "You are zaya, a personal knowledge-base assistant developed by P&I Lab.\n"
        "Answer the user's question directly and keep the response concise, concrete, and evidence-aware.\n"
        "Use retrieved snippets when they are available.\n"
        "If retrieved evidence is missing or incomplete, say that clearly instead of fabricating paper details.\n"
        "Do not invent papers, equations, numbers, baselines, or conclusions that are not supported by retrieved context.\n"
        "Retrieved context is a bounded candidate window, not a census of the user's whole library. Never present the number of DOC blocks as the total library paper count.\n"
        "If the candidate window lacks support, say 'the current retrieval did not find direct evidence'; do not claim that the whole library has no such paper.\n"
        "Do not output retrieval diagnostics, Top-K lists, DOC-k labels, or reference-location dumps unless the user explicitly asks for them.\n"
        "For math, use inline $...$ for short symbols and $$...$$ for longer equations; do not wrap equations in backticks.\n"
        "If the user asks for code, pseudocode, or derivation, provide directly usable output instead of only high-level discussion.\n"
    )
    if research_plan is not None:
        system += "\n" + research_plan.to_prompt_block() + "\n"

    if not paper_guide_mode:
        system += (
            "\nUser-facing research answer quality protocol:\n"
            "- Start with the direct answer to the user's actual question before background or reading advice.\n"
            "- Follow the Research answer plan above, but do not force unnecessary sections for a narrow factual question.\n"
            "- Evidence should pair each important claim with the retrieved snippet marker that supports that exact sentence.\n"
            "- Treat every paper-specific mechanism, result, number, comparison, and limitation as a separate claim: cite it inline from the same DOC block or remove it.\n"
            "- A citation in one sentence or paragraph does not support a later uncited claim. Do not leave factual table rows or summary bullets uncited.\n"
            "- Keep neighboring modalities separate (for example, single-photon versus single-pixel imaging). Evidence about one may be labeled as background for the other, but cannot prove a target-modality claim.\n"
            "- Never say that a paper, the whole library, or an experiment lacks validation merely because the current snippet window does not show it; scope such limits to the currently cited evidence.\n"
            "- Before finalizing, delete unsupported specifics instead of filling gaps with plausible domain knowledge. If an inference is essential, label that sentence explicitly as an inference and do not attach an unrelated citation.\n"
            "- Put missing, weak, or conflicting retrieved support under Limits instead of smoothing it over.\n"
            "- Keep the final action source-aware, such as the exact section, figure, table, experiment, or paper family to inspect next.\n"
        )
        if has_answer_hits:
            system += (
                "- Every paper-specific claim based on retrieved snippets must end with offset snippet citations such as [10001] or [10002].\n"
                "- Do not use bare [1] [2] [3] for retrieved snippets; those can collide with a paper's own bibliography numbers.\n"
            )
        else:
            system += (
                "- If no retrieved context is available, explicitly say that no matching library snippets were retrieved, then label any answer as general guidance.\n"
            )

    if paper_guide_mode:
        system += (
            "\nStructured citation protocol:\n"
            "- Context headers contain [SID:<sid>] identifiers.\n"
            "- Retrieval block labels like DOC-1 / DOC-2 are context ids only, not paper reference numbers.\n"
            "- Never mention DOC-k retrieval labels in the user-visible answer; use them only to decide support.\n"
            "- When citing paper bibliography references, MUST use [[CITE:<sid>:<ref_num>]].\n"
            "- Example: [[CITE:s1a2b3c4:24]] or [[CITE:s1a2b3c4:24]][[CITE:s1a2b3c4:25]].\n"
            "- Do NOT output free-form numeric citations like [24] / [2][4].\n"
            "- NEVER output malformed markers like [[CITE:<sid>]] or [CITE:<sid>] without a ref_num.\n"
            "\nChinese citation rule:\n"
            "- 上下文头部里的 [SID:<sid>] 是来源文献的唯一标识。\n"
            "- 回答中提到原文参考文献编号时，必须写成 [[CITE:<sid>:<ref_num>]]，不能裸写 [n]。\n"
            "- 示例：写作 `Sen et al. [[CITE:s9a2b3c4:3]]`，不要写作 `Sen et al. [3]`。\n"
        )
    else:
        system += (
            "\nIn-paper reference tracking - CRITICAL:\n"
            "- Retrieved snippets may contain original-paper reference markers like [3,58] from the paper's own bibliography.\n"
            "- Context headers may list candidate refs explicitly, for example: candidate refs: 3, 58, 4, 5.\n"
            "- When you mention or repeat any such bibliography number, use [[CITE:<sid>:<ref_num>]]. Never output bare [n] for in-paper references.\n"
            "- The <sid> is shown in the context header as [SID:<sid>] for each document.\n"
            "- Example: write `as shown by Sen et al. [[CITE:s9a2b3c4:3]]`, not `as shown by Sen et al. [3]`.\n"
            "- Only use [[CITE:...]] for original-paper bibliography numbers; do not use it for snippet citations.\n"
            "\nSnippet citation rule:\n"
            "- Retrieved snippets are labeled DOC-1, DOC-2, etc. in the context.\n"
            "- When citing information from a snippet, MUST use offset citation numbers: DOC-1 -> [10001], DOC-2 -> [10002], DOC-3 -> [10003], and so on.\n"
            "- Example: `The method achieves state-of-the-art results [10001].`\n"
            "- Include [10001] [10002] markers regardless of response language; Chinese answers must use the same markers.\n"
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
        citation_hint_text = "\n".join(
            [
                str(ctx or ""),
                str(paper_guide_reference_opportunities_block or ""),
                str(citation_plan_block or ""),
            ]
        ).lower()
        support_hint_text = "\n".join(
            [
                str(paper_guide_support_slots_block or ""),
                str(citation_plan_block or ""),
            ]
        ).lower()
        has_direct_citation_hint = bool(
            str(paper_guide_reference_opportunities_block or "").strip()
            or "cite_example" in citation_hint_text
            or "[[cite:" in citation_hint_text
            or "candidate refs:" in citation_hint_text
        )
        has_support_hint = bool(
            str(paper_guide_support_slots_block or "").strip()
            or "support_example" in support_hint_text
            or "[[support:" in support_hint_text
        )
        system += (
            "\nCitation source lock:\n"
            f"- This answer is primarily grounded in [SID:{locked_sid}] {locked_name}.\n"
        )
        if has_direct_citation_hint:
            system += (
                f"- Use a valid [[CITE:{locked_sid}:<ref_num>]] or copy a provided cite_example exactly when the answer discusses a matched bibliography reference.\n"
                "- Only switch to another SID when the same reference number cannot be verified in the locked source.\n"
            )
        elif has_support_hint:
            system += (
                "- Use provided [[SUPPORT:...]] support_example markers for paper-grounded claims; runtime will resolve them to a citation or locate-only grounding.\n"
                "- If no candidate bibliography ref or cite_example is provided, do not invent a ref_num.\n"
            )
        else:
            system += (
                "- If no candidate bibliography ref or cite_example is provided, use snippet or section grounding and do not invent a ref_num.\n"
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
            "- For beginner, concept, origin, prior-work, or method-background questions, cite those upstream works inline on the sentence that explains them.\n"
            "- Answer the user's substantive question first; for example, say whether the method is original, borrowed, prior work, or background before adding the cite_example.\n"
            "- Never begin the final answer with locator-only shells such as 'The paper cites...', 'This is stated in...', or 'Source location...'.\n"
            "- Do not add a separate bibliography trail unless the user explicitly asks for a reading list or reference list.\n"
            "- Only use an upstream cite_example when the answer actually discusses the matching concept or prior work.\n"
        )

    if citation_plan_block:
        system += (
            "\nCitation-plan protocol:\n"
            "- Follow the Citation plan block before choosing citation markers.\n"
            "- Treat SystemA citations as evidence links to retrieved paper text, and SystemB citations as links to a paper's bibliography entries.\n"
            "- Treat the citation budget as a limit on distinct evidence cards, not on how often a valid marker may be reused.\n"
            "- Every substantive paper-specific section, bullet, or table row must carry its supporting marker inline; a citation in an earlier paragraph does not support a later uncited restatement.\n"
            "- Keep one claim aligned to one evidence scope. Do not combine a supported statement with an uncited mechanism, number, comparison, or limitation in the same sentence.\n"
            "- Evidence from a neighboring modality may be presented only as clearly labeled background; it cannot support a claim about the target modality.\n"
            "- Negative claims must be bounded to the cited snippet when full-paper absence has not been verified.\n"
            "- Do not collect the citations in an evidence preamble while leaving the detailed answer body uncited.\n"
            "- Respect the distinct-card and per-paragraph budgets unless the user explicitly asks for a dense reference list.\n"
            "- If the plan provides a cite_example or support_example, copy that marker exactly instead of inventing a new number.\n"
            "- Remove unsupported specifics before finalizing, especially numbers, performance claims, mechanisms, comparisons, and limitations; if an inference is useful, label it explicitly as an inference.\n"
        )

    cite_reminder = ""
    if (not paper_guide_mode) and has_answer_hits:
        cite_reminder = (
            "\nRequired citation reminder:\n"
            "- Claims based on retrieved snippets must use offset markers [10001], [10002], [10003], starting from DOC-1.\n"
            "- Do not use bare [1], [2], [3] for snippet citations; those look like the paper's bibliography numbers.\n"
            "- If you mention an original-paper bibliography number such as [3,58], convert it to [[CITE:<sid>:<ref_num>]] using the SID in the context header.\n"
        )

    user = (
        f"Question:\n{prompt_for_user}\n\n{cite_reminder}"
        f"Retrieved context (with deep-read supplements):\n{ctx if ctx else '(none)'}\n"
    )
    if paper_guide_special_focus_block:
        user += f"\n{paper_guide_special_focus_block}\n"
    if citation_plan_block:
        user += f"\n{citation_plan_block}\n"
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
        "research_answer_plan": str(getattr(research_plan, "kind", "") or ""),
    }
