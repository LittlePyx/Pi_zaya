from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

from kb.paper_guide_shared import _cite_source_id
from kb.source_blocks import normalize_inline_markdown


_TEMPLATE_ONLY_RE = re.compile(
    r"(?is)^\s*(?:"
    r"The paper cites\b|"
    r"The paper states this explicitly\b|"
    r"Equation support\s*\(|"
    r"Figure caption\s*\(|"
    r"Source location\b|"
    r"This is stated in\b|"
    r"This hit is directly relevant\b"
    r")"
)
_NUMERIC_REF_RE = re.compile(r"(?<!\[)\[(\d{1,4})\](?!\])")
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*\b")
_ORIGIN_PROMPT_RE = re.compile(
    r"(?i)\b(?:origin|original|invented|invention|come\s+from|came\s+from|prior|previous|earlier|"
    r"cited|borrowed|inspired|based\s+on)\b|"
    r"(?:怎么来的|从哪来|来源|出处|源头|原创|自己.{0,6}(?:想|做|发明|提出)|借鉴|参考|引用|前人|已有|以前|之前)",
)


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def _compact_text(text: str, *, max_len: int = 420) -> str:
    s = normalize_inline_markdown(str(text or ""))
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip(" ,;:") + "..."


def _answer_looks_template_only(answer: str) -> bool:
    text = str(answer or "").strip()
    if not text:
        return False
    if _TEMPLATE_ONLY_RE.search(text):
        return True
    first_line = text.splitlines()[0].strip()
    if re.search(r"\b(?:directly relevant|good entry point|matched section|matched passage)\b", first_line, flags=re.I):
        return len(text) < 260 or any(line.strip().startswith(">") for line in text.splitlines())
    return False


def _record_text(record: Mapping[str, object]) -> str:
    for key in (
        "locate_anchor",
        "evidence_quote",
        "anchor_text",
        "segment_text",
        "highlight_snippet",
        "snippet",
        "text",
    ):
        value = str(record.get(key) or "").strip()
        if value:
            return _compact_text(value)
    return ""


def _record_heading(record: Mapping[str, object]) -> str:
    for key in ("heading_path", "primary_heading_path", "heading", "section"):
        value = str(record.get(key) or "").strip()
        if value:
            return _compact_text(value, max_len=160)
    return ""


def _record_source_path(record: Mapping[str, object], *, fallback_source_path: str = "") -> str:
    for key in ("source_path", "md_path", "path"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return str(fallback_source_path or "").strip()


def _record_ref_nums(record: Mapping[str, object], *, answer: str) -> list[int]:
    nums: list[int] = []

    def _add(value: object) -> None:
        try:
            n = int(value)
        except Exception:
            return
        if n > 0 and n not in nums:
            nums.append(n)

    for key in ("resolved_ref_num", "ref_num", "reference_number"):
        _add(record.get(key))
    for key in ("candidate_refs", "support_ref_candidates", "inline_refs"):
        values = record.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for item in values:
                _add(item)
    for match in _NUMERIC_REF_RE.finditer(str(answer or "")):
        _add(match.group(1))
    return nums[:4]


def _best_support_record(
    support_resolution: Sequence[Mapping[str, object]] | None,
    *,
    cards: Sequence[Mapping[str, object]] | None,
) -> Mapping[str, object]:
    candidates: list[Mapping[str, object]] = []
    for rec in list(support_resolution or []):
        if isinstance(rec, Mapping):
            candidates.append(rec)
    for card in list(cards or []):
        if isinstance(card, Mapping):
            candidates.append(card)

    def _score(rec: Mapping[str, object]) -> tuple[int, int]:
        text = _record_text(rec)
        refs = _record_ref_nums(rec, answer="")
        return (int(bool(text)) + int(bool(refs)) + int(bool(_record_heading(rec))), len(text))

    candidates.sort(key=_score, reverse=True)
    return candidates[0] if candidates else {}


def _ref_markers(ref_nums: Sequence[int], *, source_path: str) -> str:
    nums = [int(n) for n in ref_nums if int(n) > 0]
    if not nums:
        return ""
    sid = _cite_source_id(str(source_path or "").strip()) if source_path else ""
    if sid:
        return ", ".join(f"[[CITE:{sid}:{n}]]" for n in nums[:4])
    return ", ".join(f"[{n}]" for n in nums[:4])


def _question_asks_origin_or_prior(prompt: str) -> bool:
    return bool(_ORIGIN_PROMPT_RE.search(str(prompt or "")))


def _label_from_prompt_or_quote(*, prompt: str, quote: str, ref_nums: Sequence[int]) -> str:
    quote_text = str(quote or "")
    for ref_num in list(ref_nums or [])[:2]:
        local_ref = re.search(rf"\[\s*{int(ref_num)}\s*\]", quote_text)
        if local_ref:
            before = quote_text[max(0, local_ref.start() - 110) : local_ref.start()]
            entities = [
                item
                for item in _ENTITY_RE.findall(before)
                if len(item) >= 3 and item.lower() not in {"the", "this", "that", "most"}
            ]
            if entities:
                return entities[-1]
    quote_low = quote_text.lower()
    for entity in _ENTITY_RE.findall(str(prompt or "")):
        if len(entity) >= 3 and entity.lower() in quote_low:
            return entity
    for entity in _ENTITY_RE.findall(quote_text):
        if len(entity) >= 3 and entity.lower() not in {"the", "this", "that", "most"}:
            return entity
    return "这条线索" if _contains_cjk(prompt) else "this thread"


def repair_template_only_paper_guide_answer(
    answer: str,
    *,
    prompt: str,
    prompt_family: str,
    support_resolution: Sequence[Mapping[str, object]] | None,
    cards: Sequence[Mapping[str, object]] | None = None,
    fallback_source_path: str = "",
) -> tuple[str, dict[str, object]]:
    """Rewrite user-visible paper-guide answers that only say where a citation is.

    Upstream helpers may produce terse locator shells such as
    "The paper cites [4] for this point." Those are useful intermediate signals,
    but poor final answers. This function turns that signal into a compact,
    evidence-grounded explanation without calling an LLM.
    """

    text = str(answer or "").strip()
    if not text or not _answer_looks_template_only(text):
        return text, {"changed": False}

    record = _best_support_record(support_resolution, cards=cards)
    quote = _record_text(record)
    if not quote:
        quote_match = re.search(r"(?m)^\s*>\s*(.+?)\s*$", text)
        quote = _compact_text(quote_match.group(1) if quote_match else "")
    if not quote:
        return text, {"changed": False, "reason": "no_support_text"}

    heading = _record_heading(record)
    source_path = _record_source_path(record, fallback_source_path=fallback_source_path)
    ref_nums = _record_ref_nums(record, answer=text)
    markers = _ref_markers(ref_nums, source_path=source_path)
    prefer_zh = _contains_cjk(prompt)
    family = str(prompt_family or "").strip().lower()
    origin_or_prior = _question_asks_origin_or_prior(prompt)
    label = _label_from_prompt_or_quote(prompt=prompt, quote=quote, ref_nums=ref_nums)

    if prefer_zh:
        where = f"在 {heading}，" if heading else ""
        if markers and family == "citation_lookup" and origin_or_prior:
            repaired = (
                f"{label} 在这里不是作为本文原创发明来讲的；{where}原文把它放在已有方法/上游工作的线索里，并对应到 {markers}。\n\n"
                f"> {quote}\n\n"
                f"所以要追“它从哪来”，优先点开 {markers}，再回到当前论文看作者怎样借用这条思路。"
            )
        elif markers and family == "citation_lookup":
            repaired = (
                f"{where}原文把这条说法连接到文内参考 {markers}。\n\n"
                f"> {quote}\n\n"
                f"如果你要追作者引用的上游来源，就从 {markers} 开始；当前论文里这更像背景依据，而不是本文新提出的贡献。"
            )
        elif markers:
            repaired = (
                f"{where}原文给出的直接依据是：\n\n"
                f"> {quote}\n\n"
                f"这段话对应到文内参考 {markers}，可以顺着它回到作者引用的上游文献。"
            )
        else:
            repaired = (
                f"{where}原文对应的关键句是：\n\n"
                f"> {quote}\n\n"
                "这段话比单纯给出位置更重要：它是回答你这个问题时应优先核对的原文依据。"
            )
    else:
        where = f"In {heading}, " if heading else ""
        if markers and family == "citation_lookup" and origin_or_prior:
            repaired = (
                f"{label} is not presented here as this paper's own invention. {where}the source sentence treats it as cited prior or upstream work and links it to {markers}.\n\n"
                f"> {quote}\n\n"
                f"To trace where that idea comes from, open {markers} first, then compare how the current paper reuses it."
            )
        elif markers and family == "citation_lookup":
            repaired = (
                f"{where}the paper links this claim to {markers}.\n\n"
                f"> {quote}\n\n"
                f"Open {markers} to inspect the upstream source; in the current paper it functions as cited prior work or background evidence, not as the paper's new contribution."
            )
        elif markers:
            repaired = (
                f"{where}the direct evidence is:\n\n"
                f"> {quote}\n\n"
                f"This passage points to {markers}, which is the upstream reference to inspect next."
            )
        else:
            repaired = (
                f"{where}the key source sentence is:\n\n"
                f"> {quote}\n\n"
                "This is the paper evidence to check first for the question, rather than only a navigation hint."
            )

    return repaired.strip(), {
        "changed": True,
        "reason": "template_only",
        "prompt_family": family,
        "ref_nums": list(ref_nums),
        "heading_path": heading,
    }
