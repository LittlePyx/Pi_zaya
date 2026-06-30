from __future__ import annotations

import re

from kb.source_blocks import normalize_inline_markdown

_CITE_SINGLE_BRACKET_RE = re.compile(
    r"(?<!\[)\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*(\d{1,4})\s*\](?!\])",
    re.IGNORECASE,
)
_CITE_SID_ONLY_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*\]\]",
    re.IGNORECASE,
)
_CITE_NON_NUMERIC_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*[A-Za-z0-9_-]{4,24}\s*:\s*(?!\d{1,4}\s*\]\])[^]\n]+\]\]",
    re.IGNORECASE,
)
_CITE_CANON_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*(\d{1,4})\s*\]\]",
    re.IGNORECASE,
)
_DOC_CONTEXT_LABEL_RE = re.compile(
    r"\bDOC-\d{1,3}(?:-S\d{1,3})?(?:\s*(?:,|/|&|and|or)\s*DOC-\d{1,3}(?:-S\d{1,3})?)*\b",
    re.IGNORECASE,
)
_SUPPORT_MARKER_RE = re.compile(
    r"\[\[\s*SUPPORT\s*:\s*(DOC-(\d{1,3})(?:-S(\d{1,3}))?)\s*\]\]",
    re.IGNORECASE,
)
_SID_INLINE_RE = re.compile(r"\[\s*SID\s*:\s*[A-Za-z0-9_-]{4,24}\s*\]", re.IGNORECASE)
_SID_HEADER_LINE_RE = re.compile(
    r"(?im)^\s*(?:\[\d{1,3}\]|DOC-\d{1,3})\s*\[\s*SID\s*:\s*[A-Za-z0-9_-]{4,24}\s*\][^\n]*\n?",
    re.IGNORECASE,
)
_PAPER_GUIDE_INTERNAL_POLICY_LINE_RE = re.compile(
    r"(?im)^\s*(?:"
    r"根据规则.*|"
    r"规则第\s*\d+.*|"
    r"不得编造.*|"
    r"若您能提供以下任一内容.*|"
    r"如果您能提供以下任一内容.*|"
    r"请提供以下任一内容.*|"
    r"如果检索片段为空.*|"
    r"do not invent.*|"
    r"not present in retrieved context.*"
    r")\s*$"
)


def _strip_model_ref_section(answer: str) -> str:
    if not answer:
        return answer
    for marker in ("Reference locate", "参考定位"):
        idx = answer.find(marker)
        if idx >= 0:
            return answer[:idx].rstrip()
    return answer


def _sanitize_structured_cite_tokens(answer: str) -> str:
    s = str(answer or "")
    if not s:
        return s
    s = _CITE_SINGLE_BRACKET_RE.sub(lambda m: f"[[CITE:{m.group(1)}:{m.group(2)}]]", s)
    s = _CITE_SID_ONLY_RE.sub("", s)
    s = _CITE_NON_NUMERIC_RE.sub("", s)
    s = _SID_HEADER_LINE_RE.sub("", s)
    s = _SID_INLINE_RE.sub("", s)
    return s


def _should_preserve_structured_cites_for_user(*, prompt: str = "", prompt_family: str = "") -> bool:
    family = str(prompt_family or "").strip().lower()
    if family == "citation_lookup":
        return True
    text = str(prompt or "").strip().lower()
    if not text:
        return False
    return bool(
        re.search(
            r"\b(?:citation|cited|cite|reference number|reference numbers|in-paper citation|which reference|which references)\b|"
            r"(?:文内参考|文中参考|参考编号|引用编号|哪条引用|哪些引用|指出.*参考|点开.*参考)",
            text,
            flags=re.I,
        )
    )


def _canonicalize_negative_shell(answer: str) -> str:
    text = str(answer or "").strip()
    if not text:
        return text
    text = re.sub(
        r"(?i)\b(the retrieved (?:paper|paper evidence|paper evidence|paper context|paper evidence is insufficient|paper evidence|paper evidence)|the retrieved paper evidence|the retrieved paper|the retrieved context|the paper|the context)\s+does not state\b",
        lambda m: f"{str(m.group(1) or '').strip()} does not specify",
        text,
    )
    return text


def _sanitize_paper_guide_answer_for_user(
    answer: str,
    *,
    has_hits: bool,
    prompt: str = "",
    prompt_family: str = "",
    preserve_structured_cites: bool | None = None,
) -> str:
    text = str(answer or "").strip()
    if not text:
        return text

    if preserve_structured_cites is None:
        preserve_structured_cites = _should_preserve_structured_cites_for_user(
            prompt=prompt,
            prompt_family=prompt_family,
        )
    else:
        preserve_structured_cites = bool(preserve_structured_cites)

    # Normalize and strip internal structured markers. The user-facing answer should never
    # leak raw SUPPORT/SID tokens. Citation lookup intentionally keeps valid
    # [[CITE:sid:n]] tokens for the renderer to turn into hoverable chips.
    text = _sanitize_structured_cite_tokens(text)

    if "未命中知识库片段" in text:
        if has_hits:
            text = text.replace("未命中知识库片段。", "").replace("未命中知识库片段", "").strip()
        else:
            text = re.sub(
                r"^\s*未命中知识库片段[。:\s]*",
                "The retrieved paper evidence is insufficient; only the supported part is kept below.\n\n",
                text,
                count=1,
                flags=re.MULTILINE,
            ).strip()

    paras = [p.strip() for p in re.split(r"\n{2,}", text) if str(p or "").strip()]
    if not paras:
        return text

    kept: list[str] = []
    for para in paras:
        lines = [ln.rstrip() for ln in para.splitlines()]
        filtered = [ln for ln in lines if not _PAPER_GUIDE_INTERNAL_POLICY_LINE_RE.match(ln.strip())]
        para2 = "\n".join(filtered).strip()
        if not para2:
            continue
        if _PAPER_GUIDE_INTERNAL_POLICY_LINE_RE.search(para2):
            continue
        para_check = normalize_inline_markdown(para2)
        if re.search(
            r"(?i)\b(?:not stated|does not state|do not state|does not specify|do not specify|"
            r"does not discuss|do not discuss|does not mention|do not mention|makes no statement|"
            r"cannot be determined from the retrieved)\b",
            para_check,
        ):
            para2 = _CITE_CANON_RE.sub("", para2).strip()
            para2 = _canonicalize_negative_shell(para2)
        kept.append(para2)

    out = "\n\n".join(kept).strip()
    if out:
        # Strip orphaned markdown bold/italic markers: broken ****, **, __, or * at word boundaries.
        out = re.sub(r"\*{2,4}|\_{2,4}", "", out)
        out = re.sub(r"(?<!\w)\*\s+(?=\w)", "", out)
        out = re.sub(r"(?<=\w)\s+\*(?!\w)", "", out)
        # Remove empty or near-empty parenthetical brackets.
        out = re.sub(r"[（(]\s*[,，、;；:\s]*[）)]", "", out)
        # Fix sentence fragments starting with Chinese conjunctions that lack a subject.
        out = re.sub(
            r"(?<=[。])\s*(和|并且|以及|分别|同时|另外|此外|同时|进一步|最后|最终)\s*"
            r"(讨论|说明|阐述|指出|介绍|描述|提到|分析|探讨|对比|比较|实现|采用|提出|证明)",
            lambda m: f"，{m.group(1)}{m.group(2)}",
            out,
        )
        out = _SUPPORT_MARKER_RE.sub("", out)
        if not preserve_structured_cites:
            out = _CITE_CANON_RE.sub("", out)
        context_label = "原文证据" if re.search(r"[\u4e00-\u9fff]", out) else "source evidence"
        out = _DOC_CONTEXT_LABEL_RE.sub(context_label, out)
        out = re.sub(r"(?i)\bthe supporting excerpts\b", context_label, out)
        out = re.sub(r"(?i)\bsupporting excerpts\b", context_label, out)
        out = re.sub(r"(?i)\bretrieved context\b", "retrieved paper evidence", out)
        out = re.sub(r"当前\s*retrieved paper evidence", "当前已检索到的原文证据", out)
        out = re.sub(r"依据\s*retrieved paper evidence", "依据已检索到的原文证据", out)
        out = re.sub(r"依据当前\s*retrieved paper evidence", "依据当前已检索到的原文证据", out)
        out = re.sub(r"原文证据\s+的", "原文证据的", out)
        out = re.sub(r"位于\s+原文证据", "位于原文证据", out)
        out = re.sub(r"\brefs?\s+([0-9,\s]+)", lambda m: f"参考文献 {str(m.group(1) or '').strip()}", out, flags=re.IGNORECASE)
        out = re.sub(r"（\s*依据\s*[,，、;；:\s]*）", "", out)
        out = re.sub(r"\(\s*依据\s*[,，、;；:\s]*\)", "", out)
        out = re.sub(r"\s*依据\s*[,，、;；:\s]*[）)]", "", out)
        out = re.sub(r"（\s*(?:均)?基于\s*[,，、;；:\s]*）", "", out)
        out = re.sub(r"\(\s*(?:all\s+)?based\s+on\s*[,，、;；:\s]*\)", "", out, flags=re.IGNORECASE)
        out = re.sub(r"（\s*(?:共\s*\d+\s*篇)?参考文献\s*[:：]\s*[,，、;；\s]*）", "", out)
        out = re.sub(r"\(\s*(?:total\s+)?(?:\d+\s+)?references?\s*[:：]\s*[,，、;；\s]*\)", "", out, flags=re.IGNORECASE)
        out = re.sub(r"(?m)^\s*[-*+]\s*[^。\n]{1,100}[:：]\s*$\n?", "", out)
        out = re.sub(
            r"并分别标注了对应参考文献编号\s*[:：]\s*(?=(?:\n|\s)*(?:作者想表达|该句|这里))",
            "并在原句中保留了对应参考文献编号。",
            out,
        )
        out = re.sub(r"即\s*[:：]\s*(?=(?:\n|\s)*(?:作者想表达|该句|这里|证据\d))", "", out)
        out = re.sub(r"（\s*(?:如|例如)\s*）", "", out)
        out = re.sub(r"\(\s*(?:e\.g\.?|for example|see)?\s*\)", "", out, flags=re.IGNORECASE)
        # Remove truly orphaned Chinese brackets after all paired-bracket patterns have been handled.
        out = re.sub(r"""[）](?=\s*[，。；：！？、])""", "", out)
        out = re.sub(r"""^\s*[（]\s*(?=[一-鿿])""", "", out, flags=re.MULTILINE)
        out = re.sub(r"""(?<=[。！？])\s*[（]\s*(?=[一-鿿])""", "", out)
        out = re.sub(r"\s+([,.;:!?])", r"\1", out)
        out = re.sub(r"\s+([，。；：！？、）])", r"\1", out)
        out = re.sub(r"[ \t]{2,}", " ", out)
        out = re.sub(r"\(\s*原文证据\s*\)", "（原文证据）", out)
        out = re.sub(r"\(\s*source evidence\s*\)", "(source evidence)", out, flags=re.IGNORECASE)
        out = re.sub(r"\n{3,}", "\n\n", out).strip()

    # If the answer is a short negative-shell ("does not specify ..."), add a tiny next-step hint.
    # This improves user experience without inventing paper facts.
    if out:
        family = str(prompt_family or "").strip().lower()
        if family != "citation_lookup":
            out_norm = normalize_inline_markdown(out).lower()
            is_negative = bool(
                re.search(
                    r"(?i)\b(?:does not specify|does not mention|not stated|cannot be determined)\b",
                    out_norm,
                )
            )
            if is_negative and len(out.strip()) <= 240:
                q = str(prompt or "").strip().lower()
                if not q:
                    # If the caller didn't pass the original prompt, fall back to the answer text.
                    q = out_norm
                if any(tok in q for tok in ("gpu", "cuda", "nvidia", "rtx", "a100", "v100", "3090", "4090", "hardware")):
                    out = (
                        out
                        + "\n\nNext steps (paper-only): search within the paper for `GPU`, `CUDA`, `NVIDIA`, `RTX`, `hardware`, and check the Methods / Implementation / Experimental setup sections."
                    ).strip()
                elif any(tok in q for tok in ("cpu", "ram", "memory", "os", "ubuntu", "windows", "pytorch", "tensorflow", "environment")):
                    out = (
                        out
                        + "\n\nNext steps (paper-only): search within the paper for `implementation`, `code`, `training`, `environment`, and check Methods / Supplementary / Appendix for compute details."
                    ).strip()
    if out:
        return out
    if has_hits:
        return text
    return "The retrieved paper evidence is insufficient; only the supported part is kept below."
