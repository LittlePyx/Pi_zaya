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
    system += (
        "\nStructured citation protocol:\n"
        "- Context headers contain [SID:<sid>] identifiers.\n"
        "- Retrieval block labels like DOC-1 / DOC-2 are context ids only, not paper reference numbers.\n"
        "- Never mention DOC-k retrieval labels in the user-visible answer; they are internal grounding ids only.\n"
        "- When citing paper references, MUST use [[CITE:<sid>:<ref_num>]].\n"
        "- Example: [[CITE:s1a2b3c4:24]] or [[CITE:s1a2b3c4:24]][[CITE:s1a2b3c4:25]].\n"
        "- Do NOT output free-form numeric citations like [24] / [2][4].\n"
        "- NEVER output malformed markers like [[CITE:<sid>]] or [CITE:<sid>] (missing ref_num).\n"
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

    user = (
        f"Question:\n{prompt_for_user}\n\n"
        f"Retrieved context (with deep-read supplements):\n{ctx if ctx else '(none)'}\n"
    )
    if paper_guide_special_focus_block:
        user += f"\n{paper_guide_special_focus_block}\n"
    if paper_guide_support_slots_block:
        user += f"\n{paper_guide_support_slots_block}\n"
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
