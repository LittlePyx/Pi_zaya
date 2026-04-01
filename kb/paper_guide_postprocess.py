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
_PAPER_GUIDE_CITATION_LOOKUP_PROMPT_RE = re.compile(
    r"(?i)\b(?:which\s+(?:paper|prior work|reference)|what\s+reference|what\s+citation|cited\s+as\s+reference|attributed\s+to|which\s+reference\s+number)\b"
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
        if idx > 0:
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


def _canonicalize_negative_shell(answer: str) -> str:
    text = str(answer or "").strip()
    if not text:
        return text
    text = re.sub(
        r"(?i)\b(the retrieved (?:paper|context)|the paper|the context)\s+does not state\b",
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
) -> str:
    text = str(answer or "").strip()
    if not text:
        return text

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
        family = str(prompt_family or "").strip().lower()
        if family and family != "method":
            out = _CITE_CANON_RE.sub("", out)
        if _PAPER_GUIDE_CITATION_LOOKUP_PROMPT_RE.search(str(prompt or "").strip()):
            out = _CITE_CANON_RE.sub("", out)
        out = _SUPPORT_MARKER_RE.sub("", out)
        out = _DOC_CONTEXT_LABEL_RE.sub("the supporting excerpts", out)
        out = re.sub(r"\s+([,.;:!?])", r"\1", out)
        out = re.sub(r"[ \t]{2,}", " ", out)
        out = re.sub(r"\(\s*the supporting excerpts\s*\)", "(supporting excerpts)", out)
        out = re.sub(r"\n{3,}", "\n\n", out).strip()
    if out:
        return out
    if has_hits:
        return text
    return "The retrieved paper evidence is insufficient; only the supported part is kept below."
