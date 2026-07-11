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
from kb.paper_guide_prompting import _paper_guide_prompt_requests_naive_source_trace
from kb.paper_guide_postprocess import (
    _sanitize_paper_guide_answer_for_user,
    _sanitize_structured_cite_tokens,
    _strip_model_ref_section,
)
from kb.paper_guide_answer_repair import repair_template_only_paper_guide_answer as _repair_template_only_paper_guide_answer
from kb.paper_guide_reference_opportunities import (
    apply_reference_opportunities_to_answer,
    detect_paper_guide_reference_opportunities,
    detect_text_reference_opportunities,
    merge_reference_opportunity_candidate_refs,
    strip_reference_opportunity_note,
)
from kb.reference_query_family import (
    extract_requested_paper_count,
    extract_multi_paper_topic as _shared_extract_multi_paper_topic,
    prompt_explicitly_requests_multi_paper_list,
    prompt_explicitly_requests_single_paper_pick,
    prompt_likely_cross_paper_refs,
    prompt_prefers_zh,
    prompt_requests_answer_audit,
    prompt_requires_reference_focus_match as _shared_prompt_requires_reference_focus_match,
    prompt_targets_sci_topic as _shared_prompt_targets_sci_topic,
)
from kb.config import CITATION_OFFSET
from kb.paper_guide_shared import _cite_source_id
from kb.reference_index import (
    load_reference_index as _load_reference_index,
    resolve_reference_entry as _resolve_reference_entry,
)
from kb.source_blocks import normalize_inline_markdown
from kb.markdown_rendering import _normalize_math_markdown

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
_NEGATIVE_BOUNDARY_PROMPT_RE = re.compile(
    r"(?i)\b(?:worth\s+(?:reading|opening)|related|relationship|relevant|fit)\b|"
    r"(?:\u5173\u7cfb.{0,8}\u5927|\u503c\u5f97.{0,12}(?:\u8bfb|\u770b)|"
    r"\u4e00\u8d77\u8bfb|\u76f8\u5173(?:\u6027)?|\u53c2\u8003\u4ef7\u503c)"
)
_NEGATIVE_BOUNDARY_ANSWER_RE = re.compile(
    r"(?i)\b(?:not\s+worth|not\s+closely\s+related|unrelated|low\s+relevance)\b|"
    r"(?:\u5173\u7cfb\u4e0d\u5927|\u4e0d\u5efa\u8bae|\u6ca1\u6709.{0,8}\u4ea4\u96c6|"
    r"\u53c2\u8003\u4ef7\u503c.{0,8}\u4f4e|\u4ef7\u503c.{0,8}\u4f4e|\u4e0d\u503c\u5f97)"
)
_STRUCT_CITE_GARBAGE_RE = re.compile(r"\[\[?\s*CITE\s*:[^\]\n]*\]?\]", re.IGNORECASE)
_SID_INLINE_RE = re.compile(r"\[\s*SID\s*:\s*[A-Za-z0-9_-]{4,24}\s*\]", re.IGNORECASE)
_SID_RE = re.compile(r"^[A-Za-z0-9_-]{4,24}$")
_INLINE_REF_NUM_RE = re.compile(r"\[(\d{1,4})\]")
_FREEFORM_NUMERIC_CITE_RE = re.compile(
    r"(?<![!\\])\[(\d{1,5}(?:\s*(?:-|–|—|,)\s*\d{1,5})*)\](?!\()"
)
_DOC_HEADING_LINE_RE = re.compile(
    r"(?im)^\s*(?:>\s*)?(?:\*{1,2}\s*)?DOC-\d{1,3}(?:-S\d{1,3})?(?:\s*\*{1,2})?\s*[:：]\s*$"
)
_DOC_TITLE_LINE_RE = re.compile(r"(?im)^\s*(?:title|标题)\s*[:：]\s*(.+?)\s*$")
_DOC_DIAGNOSTIC_LINE_RE = re.compile(
    r"(?im)^\s*(?:>\s*)?(?:note|注意|说明)\s*[:：]?\s*DOC-\d{1,3}(?:-S\d{1,3})?[^\n]*$"
)
_DOC_RESULT_PREAMBLE_RE = re.compile(
    r"(?im)^\s*(?:based on the retrieved results|according to the retrieved results|根据提供的检索结果|根据检索结果)[^:：\n]*[:：]?\s*$"
)
_DOC_INLINE_TITLE_LINE_RE = re.compile(
    r"(?ix)^\s*(?:>\s*)?(?:[-*+]\s+|\d+[.)]\s+)?"
    r"(?:\*{1,2}\s*)?DOC-\d{1,3}(?:-S\d{1,3})?(?:\s*\*{1,2})?"
    r"(?:\s*[\(\[\{（【][^\)\]\}）】]{0,24}[\)\]\}）】])?"
    r"\s*[:：-]\s*(?P<title>\S.*)\s*$"
)
_DOC_LABEL_TOKEN_RE = re.compile(r"(?i)\*{0,2}DOC-\d{1,3}(?:-S\d{1,3})?\*{0,2}")
_DOC_LABEL_CAPTURE_RE = re.compile(r"(?i)\*{0,2}DOC-(\d{1,3})(?:-S\d{1,3})?\*{0,2}")
_ANSWER_AUDIT_CITATION_FORMAT_REQUEST_RE = re.compile(
    r"(?i)(?:citation|reference)\s+(?:format|numbering|marker|syntax)|"
    r"(?:\u5f15\u7528|\u53c2\u8003)(?:\u7f16\u53f7|\u683c\u5f0f|\u6807\u8bb0)|\u504f\u79fb\u6807\u8bb0"
)
_INTERNAL_CITATION_REVIEW_HEADING_RE = re.compile(
    r"(?i)(?:citation|reference)\s+(?:format|numbering|marker|syntax)|"
    r"(?:\u5f15\u7528|\u53c2\u8003)(?:\u7f16\u53f7|\u683c\u5f0f|\u6807\u8bb0)(?:\u95ee\u9898)?|\u504f\u79fb\u6807\u8bb0"
)
_DOC_LABEL_GROUP_IN_PARENS_RE = re.compile(
    r"(?i)[\(\[（【]\s*(?:(?:\*{0,2}DOC-\d{1,3}(?:-S\d{1,3})?\*{0,2})"
    r"\s*(?:[,/、，]|\band\b|\bor\b|及|和|与)?\s*)+[\)\]）】]"
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
_SINGLE_NUM_CITE_RE = re.compile(r"(?<![!\\])\[(\d{1,4})\](?!\()")
def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def _promote_numeric_inpaper_refs(
    answer: str,
    *,
    answer_hits: list[dict],
    db_dir: Path | None,
    paper_guide_mode: bool = False,
) -> str:
    """Convert [n] (where n < CITATION_OFFSET) to structured [[CITE:<sid>:n]].

    With the offset numbering scheme, hit citations use [OFFSET+1], [OFFSET+2],
    ... while any bare [n] with n < CITATION_OFFSET is necessarily an in-paper
    bibliography reference \u2014 there is no overlap.  Each candidate [n] is verified
    against the reference index of the source documents before promotion.

    Skipped in paper_guide mode where the LLM already emits [[CITE:...]] natively.
    """
    if paper_guide_mode:
        return answer
    hit_count = len(list(answer_hits or []))
    if hit_count == 0 or "[" not in answer:
        return answer

    # Collect unique source paths + their SIDs.
    unique_sources: list[tuple[str, str]] = []
    seen_sp: set[str] = set()
    for h in answer_hits or []:
        meta = h.get("meta", {}) or {}
        sp = str(meta.get("source_path") or "").strip()
        if sp and sp not in seen_sp:
            seen_sp.add(sp)
            sid = _cite_source_id(sp)
            unique_sources.append((sp, sid))
    if not unique_sources:
        return answer

    # Load reference index once.
    try:
        _idx = _load_reference_index(Path(db_dir).expanduser()) if db_dir else {}
    except Exception:
        _idx = {}
    if not isinstance(_idx, dict):
        _idx = {}

    # Scan answer for [n] where n < CITATION_OFFSET \u2014 these are in-paper
    # bibliography references (hit citations use OFFSET+1 etc.).
    candidates: set[int] = set()
    for m in _FREEFORM_NUMERIC_CITE_RE.finditer(answer):
        for chunk in re.findall(r"\d+", m.group(1)):
            n = int(chunk)
            if n < CITATION_OFFSET:
                candidates.add(n)
    if not candidates:
        return answer

    # Resolve each candidate ref number against each source's reference index.
    # When exactly ONE source has this ref -> promote.
    # 0 matches -> not a bibliography ref (probably a hit citation), leave as [n].
    # >1 matches -> try proximity disambiguation: check which source name
    # appears near [n] in the answer text.  If still ambiguous, leave as [n].
    ref_valid: dict[int, tuple[str, str]] = {}  # n -> (source_path, sid)

    # Pre-build display-name tokens for each unique source.
    source_name_tokens: dict[str, tuple[str, set[str]]] = {}  # sp -> (sid, tokens)
    _doc_idx_by_sp: dict[str, int] = {}  # sp -> 0-based index in unique_sources
    for idx, (sp, sid) in enumerate(unique_sources):
        stem = Path(sp).stem.lower()
        for sfx in ('.en', '.zh', '.md'):
            if stem.endswith(sfx):
                stem = stem[:-len(sfx)]
        tokens = {t for t in re.split(r'[\s\-_.,;:()\[\]{}]+', stem) if len(t) >= 4}
        source_name_tokens[sp] = (sid, tokens)
        _doc_idx_by_sp[sp] = idx

    for n in sorted(candidates):
        matched: list[tuple[str, str]] = []  # (sp, sid)
        for sp, sid in unique_sources:
            try:
                entry = _resolve_reference_entry(_idx, sp, n)
                if isinstance(entry, dict) and entry.get("ref"):
                    matched.append((sp, sid))
            except Exception:
                pass
        if len(matched) == 1:
            ref_valid[n] = matched[0]
        elif len(matched) > 1:
            # Two disambiguation strategies, tried in order:
            #
            # Strategy A \u2014 DOC-k label: the answer often refers to sources as
            # DOC-1 / DOC-2 / DOC-3 (these internal labels predate sanitization).
            # If a DOC-k label appears within 300 chars of [n], map it to the
            # k-th source in unique_sources (0-indexed: DOC-3 -> sources[2]).
            #
            # Strategy B \u2014 stem-token proximity: check which source's file-stem
            # tokens (e.g. "NatPhoton" from "NatPhoton-2025-Structured-...") appear
            # most frequently near [n] in the answer text.
            best_sid: str | None = None
            best_score = 0
            for m in _FREEFORM_NUMERIC_CITE_RE.finditer(answer):
                spec = str(m.group(1) or "")
                nums_in_spec = {int(x) for x in re.split(r"\s*(?:-|\u2013|\u2014|,)\s*", spec) if x.strip()}
                if n not in nums_in_spec:
                    continue
                ctx_start = max(0, m.start() - 300)
                ctx_end = min(len(answer), m.end() + 300)
                ctx = answer[ctx_start:ctx_end].lower()

                # Strategy A: DOC-k label
                doc_m = re.search(r'doc[-\s]*(\d+)', ctx)
                if doc_m:
                    doc_idx = int(doc_m.group(1)) - 1
                    if 0 <= doc_idx < len(unique_sources):
                        doc_sp, doc_sid = unique_sources[doc_idx]
                        if doc_sid in {sid for _, sid in matched}:
                            if best_score < 999:
                                best_score = 999
                                best_sid = doc_sid

                # Strategy B: stem-token proximity
                for sp, sid in matched:
                    _, tokens = source_name_tokens[sp]
                    score = sum(1 for t in tokens if t in ctx)
                    if score > best_score:
                        best_score = score
                        best_sid = sid
            if best_sid:
                ref_valid[n] = next((sp, sid) for sp, sid in matched if sid == best_sid)

    if not ref_valid:
        return answer

    # Replace each matched spec (single, range, or comma-separated) with
    # individual [[CITE:...]] markers when ALL numbers in the spec are
    # verified in-paper refs.  If any number is >= CITATION_OFFSET or
    # unresolvable, keep the spec unchanged (it will be stripped or
    # processed by subsequent pipeline steps).
    def _repl(m: re.Match) -> str:
        spec = str(m.group(1) or "").strip()
        nums = [int(x) for x in re.split(r"\s*(?:-|\u2013|\u2014|,)\s*", spec) if x.strip()]
        if not nums:
            return m.group(0)

        # ALL numbers must be < CITATION_OFFSET (in-paper refs).
        if any(n >= CITATION_OFFSET for n in nums):
            return m.group(0)

        # ALL numbers must be resolvable in the reference index.
        parts: list[str] = []
        for n in nums:
            pair = ref_valid.get(n)
            if not pair:
                return m.group(0)
            parts.append(f"[[CITE:{pair[1]}:{n}]]")
        return "".join(parts)

    # Protect existing [[CITE:...]] markers so inner [<n>] isn't re-processed.
    _cite_holder: dict[str, str] = {}
    _cite_counter = 0
    def _capture_cite(m: re.Match) -> str:
        nonlocal _cite_counter
        key = f"\x00C{_cite_counter}\x00"
        _cite_counter += 1
        _cite_holder[key] = m.group(0)
        return key
    protected = _CITE_CANON_RE.sub(_capture_cite, answer)
    result = _FREEFORM_NUMERIC_CITE_RE.sub(_repl, protected)
    for key, original in _cite_holder.items():
        result = result.replace(key, original)
    return result


# Regex: LaTeX superscript/subscript footnote markers that leak from paper text.
# Matches $^4$, $_n$, $^{14}$, $_{label}$ — short single-token footnotes.
_LATEX_FOOTNOTE_RE = re.compile(r"\$[\^_](?:\d{1,2}|[A-Za-z]|\{[^}]{1,12}\})\$")


def _strip_latex_footnote_markers(answer: str) -> str:
    """Strip isolated LaTeX footnote/endnote markers like $^n$ or $_{xx}$.

    These leak from the original paper text through the LLM output when the
    paper uses LaTeX superscript markers for footnotes (e.g., ``$^4$`` in
    ``Duarte et al.$^4$ showed...``).  They are NOT real math and should not
    appear in the user-visible answer.

    Only single-token markers are stripped — multi-token math expressions
    like $x^2 + y^2$ are preserved as-is.
    """
    if not answer or "$" not in answer:
        return answer
    return _LATEX_FOOTNOTE_RE.sub("", answer)


def _strip_citation_offset(
    answer: str,
) -> str:
    """Convert offset citation numbers back to 1-based for storage/rendering.

    After _promote_numeric_inpaper_refs has promoted in-paper refs to
    [[CITE:...]], this pass rewrites [OFFSET+1], [OFFSET+2], ... back to
    [1], [2], ... so the renderer's _resolve_n_from_hits works unchanged.

    Only specs where ALL numbers are >= CITATION_OFFSET are converted.
    Mixed specs (e.g. [10001,35]) are left untouched.
    """
    if not answer or "[" not in answer:
        return answer

    def _repl(m: re.Match) -> str:
        spec = str(m.group(1) or "").strip()
        nums = [int(x) for x in re.split(r"\s*(?:-|\u2013|\u2014|,)\s*", spec) if x.strip()]
        if not nums:
            return m.group(0)

        # Only convert when ALL numbers carry the offset.
        if any(n < CITATION_OFFSET for n in nums):
            return m.group(0)

        new_nums = [n - CITATION_OFFSET for n in nums]
        return "[" + ",".join(str(n) for n in new_nums) + "]"

    return _FREEFORM_NUMERIC_CITE_RE.sub(_repl, answer)


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
    try:
        if _paper_guide_prompt_requests_naive_source_trace(prompt):
            return True
    except Exception:
        pass
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


def _prompt_prefers_chinese_answer(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    if _contains_cjk(text):
        return True
    return bool(re.search(r"\b(answer|respond|reply)\s+in\s+chinese\b|\bchinese\b", text, flags=re.I))


def _sanitize_empty_markdown_label_fragments(answer: str) -> str:
    text = str(answer or "")
    if not text:
        return text
    text = re.sub(r"(?m)^\s*\*{4,}\s*[:：]\s*", "", text)
    text = re.sub(r"(?m)(^|\n)(\s*[-*+]\s*)?\*{4,}\s*[:：]\s*", r"\1", text)
    text = re.sub(r"(?<!\*)\*{4,}\s*[:：]\s*", "", text)
    text = re.sub(r"[ \t]+([,.;:!?，。；：！？])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _candidate_sources_for_inpaper_lookup(
    *,
    answer_hits: list[dict],
    locked_citation_source: dict | None,
    prompt: str,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(source_path: object, source_sha1: object = "") -> None:
        sp = str(source_path or "").strip()
        if not sp or sp in seen:
            return
        seen.add(sp)
        rows.append((sp, str(source_sha1 or "").strip().lower()))

    if isinstance(locked_citation_source, dict):
        _add(locked_citation_source.get("source_path"), locked_citation_source.get("source_sha1"))
    for hit in list(answer_hits or []):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        _add((meta or {}).get("source_path"), (meta or {}).get("source_sha1"))

    prompt_norm = re.sub(r"[^a-z0-9]+", " ", str(prompt or "").lower()).strip()
    if "scinerf" in prompt_norm:
        exact = [(sp, sha) for sp, sha in rows if "scinerf" in re.sub(r"[^a-z0-9]+", " ", sp.lower())]
        if exact:
            return exact
    return rows


def _prompt_requested_reference_targets(prompt: str) -> list[tuple[str, tuple[tuple[str, ...], ...]]]:
    low = str(prompt or "").strip().lower()
    if not low or not _prompt_explicitly_requests_citation_lookup(low):
        return []
    targets: list[tuple[str, tuple[tuple[str, ...], ...]]] = []
    has_admm_net = bool("admm-net" in low or "admm net" in low or "deep tensor" in low)
    has_standalone_admm = bool(
        re.search(r"(?<![a-z0-9])admm(?!\s*[- ]?\s*net)(?![a-z0-9])", low)
        or "alternating direction method" in low
    )
    if has_standalone_admm:
        targets.append(
            (
                "ADMM",
                (
                    ("alternating direction method of multipliers",),
                    ("distributed optimization", "multipliers"),
                ),
            )
        )
    if has_admm_net:
        targets.append(
            (
                "ADMM-Net",
                (
                    ("admm net",),
                    ("deep tensor admm",),
                    ("snapshot compressive imaging", "admm"),
                ),
            )
        )
    return targets


def _reference_surface(ref: dict) -> str:
    if not isinstance(ref, dict):
        return ""
    parts = [
        str(ref.get("title") or ""),
        str(ref.get("raw") or ""),
        str(ref.get("authors") or ""),
        str(ref.get("venue") or ""),
        str(ref.get("year") or ""),
    ]
    return re.sub(r"[^a-z0-9]+", " ", " ".join(parts).lower()).strip()


def _find_reference_num_by_terms(
    index_data: dict,
    source_path: str,
    source_sha1: str,
    alternatives: tuple[tuple[str, ...], ...],
) -> int:
    best_num = 0
    best_score = -1.0
    for n in range(1, 501):
        try:
            got = _resolve_reference_entry(index_data, source_path, n, source_sha1=source_sha1)
        except Exception:
            got = None
        ref = got.get("ref") if isinstance(got, dict) and isinstance(got.get("ref"), dict) else None
        if not isinstance(ref, dict):
            continue
        surface = _reference_surface(ref)
        if not surface:
            continue
        for alt in alternatives:
            terms = [re.sub(r"[^a-z0-9]+", " ", str(term or "").lower()).strip() for term in alt]
            terms = [term for term in terms if term]
            if not terms or not all(term in surface for term in terms):
                continue
            score = 10.0 + float(sum(len(term) for term in terms)) / 100.0
            title_surface = re.sub(r"[^a-z0-9]+", " ", str(ref.get("title") or "").lower()).strip()
            if title_surface and all(term in title_surface for term in terms):
                score += 2.0
            if score > best_score:
                best_score = score
                best_num = int(n)
    return best_num


def _strip_conflicting_missing_reference_notes(answer: str, labels: list[str]) -> str:
    text = str(answer or "")
    if not text or not labels:
        return text
    label_patterns = [re.escape(str(label or "").lower()) for label in labels if str(label or "").strip()]
    if not label_patterns:
        return text
    label_re = re.compile("|".join(label_patterns), flags=re.I)
    missing_re = re.compile(
        r"not\s+(?:appear|included|found)|not\s+in\s+the\s+(?:current\s+)?(?:retrieved|candidate)|"
        r"未出现在|没有出现在|未检索到|当前检索片段|候选列表",
        flags=re.I,
    )
    out: list[str] = []
    for line in text.splitlines():
        stripped = str(line or "").strip()
        if stripped and label_re.search(stripped) and missing_re.search(stripped):
            continue
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _maybe_append_prompt_requested_inpaper_refs(
    answer: str,
    *,
    prompt: str,
    answer_hits: list[dict],
    db_dir: Path | None,
    locked_citation_source: dict | None,
) -> str:
    text = str(answer or "").strip()
    targets = _prompt_requested_reference_targets(prompt)
    if not text or not targets:
        return text
    try:
        index_data = _load_reference_index(Path(db_dir).expanduser()) if db_dir else {}
    except Exception:
        index_data = {}
    if not isinstance(index_data, dict) or not index_data:
        return text
    sources = _candidate_sources_for_inpaper_lookup(
        answer_hits=list(answer_hits or []),
        locked_citation_source=locked_citation_source,
        prompt=prompt,
    )
    if not sources:
        return text
    existing: set[int] = set(_collect_inline_reference_numbers(text, max_items=24))
    for m in _CITE_CANON_RE.finditer(text):
        try:
            existing.add(int(m.group(2) or 0))
        except Exception:
            pass
    resolved: list[tuple[str, int, str]] = []
    seen_nums: set[int] = set()
    for label, alternatives in targets:
        for source_path, source_sha1 in sources:
            ref_num = _find_reference_num_by_terms(index_data, source_path, source_sha1, alternatives)
            if ref_num <= 0 or ref_num in seen_nums or ref_num in existing:
                continue
            sid = _cite_source_id(source_path)
            resolved.append((label, int(ref_num), sid))
            seen_nums.add(int(ref_num))
            break
    if not resolved:
        return text
    text = _strip_conflicting_missing_reference_notes(text, [label for label, _, _ in resolved])
    prefer_zh = _prompt_prefers_chinese_answer(prompt)
    cites = "\u3001".join(f"{label} [[CITE:{sid}:{num}]]" for label, num, sid in resolved)
    if prefer_zh:
        line = f"\u53ef\u4ee5\u4f18\u5148\u70b9\u5f00\u7684\u539f\u8bba\u6587\u6765\u6e90\uff1a{cites}\u3002"
    else:
        line = f"Original cited sources worth opening first: {cites}."
    if line in text:
        return text
    return f"{text}\n\n{line}".strip()


def _should_preserve_final_answer_numeric_citations(
    *,
    prompt: str,
    answer_output_mode: str,
    paper_guide_mode: bool,
    prompt_family: str,
    has_hits: bool = False,
) -> bool:
    if str(prompt_family or "").strip().lower() == "citation_lookup":
        return True
    if "citation" in str(answer_output_mode or "").strip().lower():
        return True
    if paper_guide_mode and _prompt_explicitly_requests_citation_lookup(prompt):
        return True
    # Classic RAG with hits: preserve [n] markers so the renderer can link them.
    if not paper_guide_mode and has_hits:
        return True
    return False


def _should_preserve_final_answer_structured_citations(
    *,
    prompt: str,
    answer_output_mode: str,
    paper_guide_mode: bool,
    prompt_family: str,
    allow_paper_guide_structured_refs: bool = False,
) -> bool:
    if bool(allow_paper_guide_structured_refs):
        return True
    if str(prompt_family or "").strip().lower() == "citation_lookup":
        return True
    if _prompt_explicitly_requests_citation_lookup(prompt):
        return True
    if bool(paper_guide_mode) and _prompt_explicitly_requests_citation_lookup(prompt):
        return True
    if bool(paper_guide_mode) and "citation" in str(answer_output_mode or "").strip().lower():
        return True
    return False


def _strip_final_answer_citation_markers(
    answer: str,
    *,
    preserve_numeric_markers: bool,
    preserve_structured_markers: bool = False,
) -> str:
    text = str(answer or "")
    if not text:
        return text
    out = _sanitize_structured_cite_tokens(text)
    # Always strip malformed / incomplete CITE tokens (these are never valid).
    out = _STRUCT_CITE_SINGLE_RE.sub("", out)
    out = _STRUCT_CITE_SID_ONLY_RE.sub("", out)
    out = _SID_INLINE_RE.sub("", out)
    if not preserve_structured_markers:
        out = _CITE_CANON_RE.sub("", out)
        out = _STRUCT_CITE_GARBAGE_RE.sub("", out)
    if not preserve_numeric_markers:
        out = _FREEFORM_NUMERIC_CITE_RE.sub("", out)
    out = re.sub(r"[ \t]+([,.;:!?])", r"\1", out)
    out = re.sub(r"(?m)[ \t]{2,}", " ", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _strip_internal_doc_label_mentions(text: str) -> str:
    out = str(text or "")
    if not out or ("DOC-" not in out.upper()):
        return out.strip()
    out = _DOC_LABEL_GROUP_IN_PARENS_RE.sub("", out)
    out = _DOC_LABEL_TOKEN_RE.sub("", out)
    out = re.sub(r"[\(\[（【]\s*[\)\]）】]", "", out)
    out = re.sub(r"(?m)^\s*(?:>\s*)?(?:[-*+]\s+|\d+[.)]\s+)?[:：-]\s*", "", out)
    out = re.sub(r"\s+([,.;:!?，。；：！？])", r"\1", out)
    out = re.sub(r"([(\[（【])\s+", r"\1", out)
    out = re.sub(r"\s+([)\]）】])", r"\1", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip(" \t:-")


def _replace_answer_audit_doc_labels(text: str) -> str:
    raw = str(text or "")
    if not raw or "DOC-" not in raw.upper():
        return raw
    label = "来源" if _contains_cjk(raw) else "Source"
    return _DOC_LABEL_CAPTURE_RE.sub(lambda match: f"{label} [{int(match.group(1))}]", raw)


def _strip_answer_audit_internal_citation_review(text: str, *, prompt: str) -> str:
    raw = str(text or "")
    if (
        not raw
        or not prompt_requests_answer_audit(prompt)
        or _ANSWER_AUDIT_CITATION_FORMAT_REQUEST_RE.search(prompt)
    ):
        return raw
    lines = raw.splitlines()
    out: list[str] = []
    skipped_level = 0
    for line in lines:
        heading = re.match(r"^\s*(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2)
            if _INTERNAL_CITATION_REVIEW_HEADING_RE.search(title):
                skipped_level = level
                continue
            if skipped_level and level <= skipped_level:
                skipped_level = 0
        if skipped_level:
            continue
        if _INTERNAL_CITATION_REVIEW_HEADING_RE.search(line) and (
            "|" in line or "10001" in line or "offset" in line.lower() or "\u504f\u79fb" in line
        ):
            continue
        out.append(line)
    cleaned = "\n".join(out)
    cleaned = re.sub(
        r"[\uff1b;]\s*(?:\u4e8c\u662f|second(?:ly)?,?)?[^\n\u3002.!?]{0,100}"
        r"(?:\u5f15\u7528|\u53c2\u8003)(?:\u7f16\u53f7|\u683c\u5f0f|\u6807\u8bb0)[^\n\u3002.!?]*[\u3002.!?]",
        "\u3002",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _sanitize_internal_doc_label_blocks(answer: str) -> str:
    text = str(answer or "").strip()
    if not text or ("DOC-" not in text.upper()):
        return text

    lines = [str(line or "").rstrip() for line in text.splitlines()]
    out: list[str] = []
    idx = 0
    converted = False

    def _push_block(value: str) -> None:
        block = str(value or "").strip()
        if block:
            out.append(block)

    while idx < len(lines):
        line = lines[idx].strip()
        if _DOC_RESULT_PREAMBLE_RE.match(line):
            idx += 1
            continue
        if _DOC_DIAGNOSTIC_LINE_RE.match(line):
            converted = True
            idx += 1
            continue
        inline_title_match = _DOC_INLINE_TITLE_LINE_RE.match(line)
        if not _DOC_HEADING_LINE_RE.match(line) and not inline_title_match:
            cleaned_line = _strip_internal_doc_label_mentions(lines[idx])
            if cleaned_line != lines[idx].strip():
                converted = True
            _push_block(cleaned_line)
            idx += 1
            continue

        converted = True
        idx += 1
        title = ""
        if inline_title_match:
            title = _strip_internal_doc_label_mentions(inline_title_match.group("title"))
        body_lines: list[str] = []
        while idx < len(lines):
            current = lines[idx].strip()
            if _DOC_HEADING_LINE_RE.match(current) or _DOC_INLINE_TITLE_LINE_RE.match(current):
                break
            if _DOC_DIAGNOSTIC_LINE_RE.match(current):
                idx += 1
                continue
            title_match = _DOC_TITLE_LINE_RE.match(current)
            if title_match and not title:
                title = _strip_internal_doc_label_mentions(title_match.group(1))
                idx += 1
                continue
            cleaned_current = _strip_internal_doc_label_mentions(current)
            if cleaned_current:
                body_lines.append(cleaned_current)
            idx += 1

        body = re.sub(r"\s+", " ", " ".join(body_lines)).strip()
        if title and body:
            _push_block(f"- {title}: {body}")
        elif title:
            _push_block(f"- {title}")
        elif body:
            _push_block(f"- {body}")

    if not converted:
        return text

    out_text = "\n\n".join(part for part in out if str(part or "").strip())
    out_text = re.sub(r"\n{3,}", "\n\n", out_text).strip()
    return out_text or text


def _source_name_from_path_like(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    name = Path(raw).name
    for suffix in (".en.md", ".zh.md", ".md"):
        if name.endswith(suffix):
            return name[: -len(suffix)] + ".pdf"
    return name


def _normalize_topic_identity(text: str) -> str:
    raw = str(text or "").strip().lower()
    if not raw:
        return ""
    raw = raw.replace(".en.md", " ").replace(".md", " ").replace(".pdf", " ")
    raw = re.sub(r"[_/\\]+", " ", raw)
    raw = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def _single_line_summary(text: str, *, source_name: str = "", max_chars: int = 180) -> str:
    cleaned = _normalize_math_markdown(normalize_inline_markdown(str(text or "").strip()))
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*(?:abstract|introduction|related work|conclusion|conclusions)\s*[:.-]?\s*", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*(?:\d+(?:\.\d+)*|[ivxlcdm]+)\s*[.)-]?\s*(?:abstract|introduction|related work|conclusion|conclusions)\s*[:.-]?\s*", "", cleaned)
    cleaned = re.sub(r"\$[^$\n]{1,60}\$", " ", cleaned)
    cleaned = cleaned.replace("\\sim", "~").replace("\\mum", "um").replace("\\mu", "u")
    cleaned = re.sub(r"\\[A-Za-z]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -\n\t")
    source_display = str(source_name or "").strip()
    source_stem = re.sub(r"(?i)\.pdf$", "", source_display).strip()
    if source_stem:
        cleaned = re.sub(rf"^\s*{re.escape(source_stem)}\s*", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"^[A-Z][A-Za-z.\-\s,]{24,220}(?=\bAbstract\b)", "", cleaned).strip()
    cleaned = re.sub(r"^(?:figure|table)\s+\d+\s*[:.-]?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^(?:abstract)\s*[:.-]?\s*", "", cleaned, flags=re.I)
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    trimmed = cleaned[: max_chars - 1].rstrip()
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0].rstrip()
    return trimmed + "…"


def _multi_paper_technical_markers(text: str) -> set[str]:
    raw = str(text or "")
    markers = {
        str(token or "").strip("-_").lower()
        for token in re.findall(
            r"(?<![A-Za-z0-9])(?:[A-Z]{2,}[A-Za-z0-9-]*|[A-Z][A-Z][A-Za-z]+|\d+(?:\.\d+)?\s*(?:hz|db|ms|fps|%)?)",
            raw,
        )
        if str(token or "").strip("-_")
    }
    return {marker for marker in markers if marker not in {"pdf", "doi"}}


def _multi_paper_summary_conflicts_with_evidence(summary: str, evidence: str) -> bool:
    summary_markers = _multi_paper_technical_markers(summary)
    if not summary_markers:
        return False
    evidence_low = str(evidence or "").lower()
    return any(marker not in evidence_low for marker in summary_markers)



def _sanitize_multi_paper_doc_list_entry_for_scoring(*, prompt: str, raw_item: dict) -> dict:
    entry = {k: v for k, v in dict(raw_item or {}).items() if v not in ("", None, [], {})}
    summary = str(entry.get("summary_line") or "").strip()
    if _looks_generic_multi_paper_support_text(summary, prompt=prompt):
        entry.pop("summary_line", None)
    primary = dict(entry.get("primary_evidence") or {}) if isinstance(entry.get("primary_evidence"), dict) else {}
    primary_snippet_was_generic = False
    if primary:
        snippet = str(primary.get("highlight_snippet") or primary.get("snippet") or "").strip()
        if _looks_generic_multi_paper_support_text(snippet, prompt=prompt):
            primary_snippet_was_generic = True
            primary.pop("snippet", None)
            primary.pop("highlight_snippet", None)
        if primary:
            entry["primary_evidence"] = primary
        else:
            entry.pop("primary_evidence", None)
    summary = str(entry.get("summary_line") or "").strip()
    topic = _extract_multi_paper_topic(prompt)
    topic_norm = _normalize_topic_identity(topic)
    summary_norm = _normalize_topic_identity(summary)
    if summary and topic_norm and summary_norm and _surface_has_token_sequence(summary_norm, topic_norm.split()):
        support_surface = _multi_paper_entry_surface(
            source_name=str(entry.get("source_name") or "").strip(),
            heading_path=str(entry.get("heading_path") or "").strip(),
            summary_line="",
            primary_evidence=entry.get("primary_evidence") if isinstance(entry.get("primary_evidence"), dict) else {},
        )
        support_surface_norm = _normalize_topic_identity(support_surface)
        support_has_topic = _multi_paper_segment_matches(
            segment=topic_norm,
            surface_norm=support_surface_norm,
            surface_tokens=support_surface_norm.split(),
            raw_low=str(support_surface or "").lower(),
        )
        if primary_snippet_was_generic and (not support_has_topic):
            entry.pop("summary_line", None)
    return entry


def _multi_paper_topic_segments(topic: str) -> list[str]:
    norm = _normalize_topic_identity(topic)
    if not norm:
        return []
    pieces = re.split(
        r"\b(?:for|via|using|through|with|without|about|regarding|based on|based)\b",
        norm,
        flags=re.I,
    )
    out: list[str] = []
    for piece in pieces:
        seg = re.sub(r"\s+", " ", str(piece or "").strip())
        if seg:
            out.append(seg)
    return out


def _surface_has_token_sequence(surface_norm: str, token_seq: list[str]) -> bool:
    tokens = [str(tok or "").strip() for tok in list(token_seq or []) if str(tok or "").strip()]
    if not surface_norm or not tokens:
        return False
    phrase = " ".join(tokens).strip()
    if not phrase:
        return False
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", surface_norm, flags=re.I))


def _multi_paper_term_presence_pattern(term: str) -> str:
    token = str(term or "").strip()
    if not token:
        return ""
    return rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])"


def _is_informative_multi_paper_focus_token(token: str) -> bool:
    low = str(token or "").strip().lower()
    if not low:
        return False
    generic_tokens = {
        "single",
        "pixel",
        "imaging",
        "image",
        "images",
        "paper",
        "papers",
        "library",
        "libraries",
    }
    return low not in generic_tokens


def _multi_paper_segment_matches(
    *,
    segment: str,
    surface_norm: str,
    surface_tokens: list[str],
    raw_low: str,
) -> bool:
    seg_norm = _normalize_topic_identity(segment)
    if not seg_norm:
        return False
    seg_tokens = [tok for tok in seg_norm.split() if tok and len(tok) >= 4]
    if not seg_tokens:
        return False
    surface_token_set = set(surface_tokens)
    if len(seg_tokens) == 1:
        token = str(seg_tokens[0] or "")
        return _surface_has_token_sequence(surface_norm, [token]) and (
            not _multi_paper_focus_term_only_negated(token, raw_low)
        )
    if _surface_has_token_sequence(surface_norm, seg_tokens):
        return not _multi_paper_focus_term_only_negated(" ".join(seg_tokens), raw_low)
    non_negated_tokens = [
        tok for tok in seg_tokens
        if (tok in surface_token_set) and (not _multi_paper_focus_term_only_negated(tok, raw_low))
    ]
    for width in range(min(3, len(seg_tokens)), 1, -1):
        for idx in range(0, len(seg_tokens) - width + 1):
            phrase_tokens = seg_tokens[idx : idx + width]
            if not any(_is_informative_multi_paper_focus_token(tok) for tok in phrase_tokens):
                continue
            if _surface_has_token_sequence(surface_norm, phrase_tokens):
                return True
    if len(seg_tokens) == 2:
        return len(non_negated_tokens) >= 2
    return len(non_negated_tokens) >= len(seg_tokens)


def _multi_paper_focus_match(
    *,
    prompt: str,
    source_name: str,
    heading_path: str,
    summary_line: str,
    primary_evidence: dict | None,
) -> bool:
    topic = _extract_multi_paper_topic(prompt)
    if not topic:
        return False
    surface = _multi_paper_entry_surface(
        source_name=source_name,
        heading_path=heading_path,
        summary_line=summary_line,
        primary_evidence=primary_evidence,
    )
    surface_norm = _normalize_topic_identity(surface)
    if not surface_norm:
        return False
    raw_low = str(surface or "").lower()
    surface_tokens = [tok for tok in surface_norm.split() if tok]
    segments = _multi_paper_topic_segments(topic)
    if not segments:
        return False
    for segment in segments:
        if not _multi_paper_segment_matches(
            segment=segment,
            surface_norm=surface_norm,
            surface_tokens=surface_tokens,
            raw_low=raw_low,
        ):
            return False
    return True



def _multi_paper_entry_surface(
    *,
    source_name: str,
    heading_path: str,
    summary_line: str,
    primary_evidence: dict | None,
) -> str:
    primary = dict(primary_evidence or {}) if isinstance(primary_evidence, dict) else {}
    parts = [
        str(source_name or "").strip(),
        str(heading_path or "").strip(),
        str(summary_line or "").strip(),
        str(primary.get("snippet") or "").strip(),
        str(primary.get("highlight_snippet") or "").strip(),
        str(primary.get("selection_reason") or "").strip(),
    ]
    return " ".join(part for part in parts if part)


def _multi_paper_topic_score(
    *,
    prompt: str,
    source_name: str,
    heading_path: str,
    summary_line: str,
    primary_evidence: dict | None,
) -> float:
    surface = _multi_paper_entry_surface(
        source_name=source_name,
        heading_path=heading_path,
        summary_line=summary_line,
        primary_evidence=primary_evidence,
    )
    surface_norm = _normalize_topic_identity(surface)
    raw_low = str(surface or "").lower()
    if not surface_norm:
        return 0.0

    score = 0.0
    topic = _extract_multi_paper_topic(prompt)
    focus_matched = _multi_paper_focus_match(
        prompt=prompt,
        source_name=source_name,
        heading_path=heading_path,
        summary_line=summary_line,
        primary_evidence=primary_evidence,
    )
    prompt_requires_focus = _multi_paper_prompt_requires_explicit_focus_match(prompt)
    generic_topic_stop = {
        "which", "papers", "paper", "other", "library", "libraries",
        "mention", "mentions", "mentioned", "discuss", "discusses", "discussed",
        "image", "images", "imaging", "technique", "techniques",
        "single", "pixel",
    }
    topic_tokens = [
        tok for tok in _normalize_topic_identity(topic).split()
        if tok and len(tok) >= 4 and tok not in generic_topic_stop
    ]
    if topic_tokens:
        surface_token_set = set(surface_norm.split())
        overlap_tokens = [
            tok
            for tok in topic_tokens
            if (tok in surface_token_set) or _surface_has_token_sequence(surface_norm, [tok])
        ]
        overlap = len(overlap_tokens)
        non_negated_overlap = [
            tok for tok in overlap_tokens
            if not _multi_paper_focus_term_only_negated(tok, raw_low)
        ]
        overlap = len(non_negated_overlap)
        if overlap >= 2:
            score += 1.2 * float(overlap)
        elif overlap == 1:
            token = str(non_negated_overlap[0] or "")
            min_len = 4 if len(topic_tokens) <= 1 else 6
            if len(token) >= min_len:
                score += 1.4 if len(topic_tokens) <= 1 else 1.2
    if focus_matched:
        score += 2.6
    elif prompt_requires_focus and topic and (not _prompt_targets_sci_topic(prompt)):
        return 0.0

    if _prompt_targets_sci_topic(prompt):
        sci_positive_norm = (
            "snapshot compressive imaging",
            "snapshot compressive image",
            "single shot compressive spectral imaging",
        )
        sci_positive_raw = (
            "scinerf",
            "scigs",
            "snapshot compressive imaging",
            "snapshot compressive image",
            "single-shot compressive spectral imaging",
            "single shot compressive spectral imaging",
        )
        if re.search(r"\bsci\b", raw_low):
            score += 3.5
        if any(alias in surface_norm for alias in sci_positive_norm):
            score += 3.5
        if any(alias in raw_low for alias in sci_positive_raw):
            score += 2.5
        if ("single pixel imaging" in surface_norm) and (score <= 0.0):
            score -= 2.5
        if ("single pixel compressive holography" in surface_norm) and (score <= 0.0):
            score -= 3.0
        if ("compressive sensing" in surface_norm) and (score <= 0.0):
            score -= 1.2
    return score


def _classify_multi_paper_topic_match(
    *,
    prompt: str,
    source_name: str,
    heading_path: str,
    summary_line: str,
    primary_evidence: dict | None,
) -> str:
    surface = _multi_paper_entry_surface(
        source_name=source_name,
        heading_path=heading_path,
        summary_line=summary_line,
        primary_evidence=primary_evidence,
    )
    surface_norm = _normalize_topic_identity(surface)
    raw_low = str(surface or "").lower()
    if not surface_norm:
        return ""
    topic_score = _multi_paper_topic_score(
        prompt=prompt,
        source_name=source_name,
        heading_path=heading_path,
        summary_line=summary_line,
        primary_evidence=primary_evidence,
    )
    if _prompt_targets_sci_topic(prompt):
        if re.search(r"\bsci\b", raw_low) or ("snapshot compressive imaging" in raw_low):
            return "explicit_sci_mention"
        if (
            ("single-shot compressive spectral imaging" in raw_low)
            or ("single shot compressive spectral imaging" in raw_low)
        ):
            return "sci_related_predecessor"
    return "topic_aligned" if topic_score > 0.0 else ""


def _multi_paper_topic_match_rank(match_kind: str) -> int:
    kind = str(match_kind or "").strip().lower()
    if kind == "explicit_sci_mention":
        return 2
    if kind == "sci_related_predecessor":
        return 1
    if kind:
        return 1
    return 0


def _multi_paper_topic_match_note(*, prompt: str, match_kind: str) -> str:
    kind = str(match_kind or "").strip().lower()
    if not kind:
        return ""
    prefer_zh = bool(prompt_prefers_zh(prompt))
    if kind == "explicit_sci_mention":
        if prefer_zh:
            return "\u6587\u4e2d\u660e\u786e\u63d0\u5230 Snapshot Compressive Imaging (SCI)\u3002"
        return "The paper explicitly mentions Snapshot Compressive Imaging (SCI)."
    if kind == "sci_related_predecessor":
        if prefer_zh:
            return "\u8fd9\u7bc7\u66f4\u9002\u5408\u89c6\u4e3a\u4e0e SCI \u76f8\u5173\u7684\u65e9\u671f\u524d\u8eab\u5de5\u4f5c\uff1a\u8ba8\u8bba\u7684\u662f single-shot compressive spectral imaging\uff0c\u4e0e SCI \u6982\u5ff5\u76f8\u5173\uff0c\u4f46\u4e0d\u662f\u4e25\u683c\u7684 SCI \u672f\u8bed\u547d\u4e2d\u3002"
        return "This is better treated as an early related predecessor: it discusses single-shot compressive spectral imaging, which is SCI-adjacent rather than an exact SCI term match."
    return ""


def _filter_multi_paper_doc_list_contract(*, prompt: str, doc_list: list[dict] | None) -> list[dict]:
    rows: list[dict] = []
    for idx, raw_item in enumerate(list(doc_list or [])):
        if not isinstance(raw_item, dict):
            continue
        entry = _sanitize_multi_paper_doc_list_entry_for_scoring(
            prompt=prompt,
            raw_item=raw_item,
        )
        entry["_topic_score"] = _multi_paper_topic_score(
            prompt=prompt,
            source_name=str(entry.get("source_name") or "").strip(),
            heading_path=str(entry.get("heading_path") or "").strip(),
            summary_line=str(entry.get("summary_line") or "").strip(),
            primary_evidence=entry.get("primary_evidence") if isinstance(entry.get("primary_evidence"), dict) else {},
        )
        entry["topic_match_kind"] = _classify_multi_paper_topic_match(
            prompt=prompt,
            source_name=str(entry.get("source_name") or "").strip(),
            heading_path=str(entry.get("heading_path") or "").strip(),
            summary_line=str(entry.get("summary_line") or "").strip(),
            primary_evidence=entry.get("primary_evidence") if isinstance(entry.get("primary_evidence"), dict) else {},
        )
        entry["_topic_match_rank"] = _multi_paper_topic_match_rank(str(entry.get("topic_match_kind") or ""))
        entry["_order"] = idx
        rows.append(entry)

    positive_rows = [row for row in rows if float(row.get("_topic_score") or 0.0) > 0.0]
    if positive_rows:
        rows = positive_rows
    elif _multi_paper_prompt_requires_explicit_focus_match(prompt):
        return []

    rows.sort(
        key=lambda item: (
            -int(item.get("_topic_match_rank") or 0),
            -float(item.get("_topic_score") or 0.0),
            int(item.get("_order") or 0),
        )
    )
    filtered = [
        {k: v for k, v in row.items() if not str(k).startswith("_")}
        for row in rows
    ]
    requested_count = extract_requested_paper_count(prompt)
    if requested_count is not None:
        return filtered[:requested_count]
    return filtered


def _doc_list_entry_matches_bound_source(
    entry: dict,
    *,
    bound_source_path: str,
    bound_source_name: str,
) -> bool:
    if not isinstance(entry, dict):
        return False
    target_tokens = {
        token
        for token in (
            _normalize_topic_identity(bound_source_path),
            _normalize_topic_identity(bound_source_name),
            _normalize_topic_identity(_source_name_from_path_like(bound_source_path)),
        )
        if token
    }
    if not target_tokens:
        return False
    candidate_tokens = {
        token
        for token in (
            _normalize_topic_identity(str(entry.get("source_path") or "")),
            _normalize_topic_identity(str(entry.get("source_name") or "")),
            _normalize_topic_identity(_source_name_from_path_like(str(entry.get("source_path") or ""))),
        )
        if token
    }
    if not candidate_tokens:
        return False
    if candidate_tokens.intersection(target_tokens):
        return True
    for left in candidate_tokens:
        for right in target_tokens:
            if (len(left) >= 20 and left in right) or (len(right) >= 20 and right in left):
                return True
    return False


def _exclude_bound_source_from_multi_paper_doc_list_contract(
    *,
    doc_list: list[dict] | None,
    bound_source_path: str,
    bound_source_name: str,
) -> list[dict]:
    rows = [dict(item) for item in list(doc_list or []) if isinstance(item, dict)]
    if not rows:
        return []
    out: list[dict] = []
    for item in rows:
        if _doc_list_entry_matches_bound_source(
            item,
            bound_source_path=bound_source_path,
            bound_source_name=bound_source_name,
        ):
            continue
        out.append(item)
    return out


def _multi_paper_primary_precision_score(primary_evidence: dict | None) -> tuple[int, int, int, int, int, int]:
    primary = dict(primary_evidence or {}) if isinstance(primary_evidence, dict) else {}
    if not primary:
        return (0, 0, 0, 0, 0, 0)
    reason = str(primary.get("selection_reason") or primary.get("selectionReason") or "").strip().lower()
    reason_rank = {
        "prompt_aligned_block": 8,
        "prompt_aligned": 7,
        "reader_open": 5,
        "strict_locate": 5,
        "provenance_segment": 5,
        "shared_refs_pack": 5,
        "pending_section_seed": 2,
        "shared_contract_seed": 1,
        "answer_hit_top": 0,
    }.get(reason, 3 if reason else 0)
    strict_locate = primary.get("strict_locate")
    if strict_locate is None:
        strict_locate = primary.get("strictLocate")
    return (
        1 if bool(strict_locate) else 0,
        1 if str(primary.get("block_id") or primary.get("blockId") or "").strip() else 0,
        1 if str(primary.get("anchor_id") or primary.get("anchorId") or "").strip() else 0,
        1 if str(primary.get("heading_path") or primary.get("headingPath") or "").strip() else 0,
        1
        if str(primary.get("highlight_snippet") or primary.get("snippet") or "").strip()
        else 0,
        reason_rank,
    )


def _multi_paper_primary_is_weak(primary_evidence: dict | None) -> bool:
    primary = dict(primary_evidence or {}) if isinstance(primary_evidence, dict) else {}
    if not primary:
        return True
    strict_locate = primary.get("strict_locate")
    if strict_locate is None:
        strict_locate = primary.get("strictLocate")
    if bool(strict_locate):
        return False
    if str(primary.get("block_id") or primary.get("blockId") or "").strip():
        return False
    if str(primary.get("anchor_id") or primary.get("anchorId") or "").strip():
        return False
    reason = str(primary.get("selection_reason") or primary.get("selectionReason") or "").strip().lower()
    return reason in {"", "answer_hit_top", "pending_section_seed"}


def _looks_like_multi_paper_section_heading(heading: str) -> bool:
    text = re.sub(r"\s+", " ", str(heading or "").strip())
    if not text:
        return False
    low = text.lower()
    if re.match(r"^(?:\d+(?:\.\d+)*|[ivxlcdm]+)\s*[.)-]?\s+[a-z]", low, flags=re.I):
        return True
    return bool(
        re.match(
            r"(?i)^(?:abstract|introduction|related work|background|preliminar(?:y|ies)|"
            r"method(?:s)?|approach|framework|experiments?|results?|discussion|"
            r"conclusion(?:s)?|applications?|appendix|supplementary)\b",
            text,
        )
    )


def _extract_multi_paper_surface_seed(raw_text: str) -> tuple[str, str]:
    raw = str(raw_text or "").strip()
    if not raw:
        return "", ""

    abstract_match = re.search(
        r"(?is)(?:^|\n)\s*\*\*Abstract\*\*\s*[:：]\s*(.+?)(?=(?:\n\s*#{1,6}\s+\S)|\Z)",
        raw,
    )
    if abstract_match:
        return "Abstract", str(abstract_match.group(1) or "").strip()

    heading_matches = list(re.finditer(r"(?m)^\s{0,3}#{1,6}\s*([^\n#]{1,140})\s*$", raw))
    for idx, match in enumerate(heading_matches):
        heading = re.sub(r"\s+", " ", str(match.group(1) or "").strip())
        if not _looks_like_multi_paper_section_heading(heading):
            continue
        next_match = heading_matches[idx + 1] if (idx + 1) < len(heading_matches) else None
        excerpt = raw[match.end() : (next_match.start() if next_match else len(raw))].strip()
        return heading, excerpt
    return "", raw


def _normalize_multi_paper_surface_seed(
    *,
    source_name: str,
    heading_path: str,
    raw_text: str,
) -> tuple[str, str]:
    normalized_heading = str(heading_path or "").strip()
    inferred_heading, excerpt_text = _extract_multi_paper_surface_seed(raw_text)
    if inferred_heading:
        normalized_heading = inferred_heading
    normalized_summary = _single_line_summary(
        str(excerpt_text or raw_text or "").strip(),
        source_name=source_name,
    )
    return normalized_heading, normalized_summary


def _normalize_multi_paper_contract_primary_evidence(
    *,
    source_path: str,
    source_name: str,
    heading_path: str,
    raw_text: str,
    primary_evidence: dict | None,
    selection_reason: str,
) -> dict:
    primary = dict(primary_evidence or {}) if isinstance(primary_evidence, dict) else {}
    weak_primary = _multi_paper_primary_is_weak(primary)
    normalized_heading, normalized_summary = _normalize_multi_paper_surface_seed(
        source_name=source_name,
        heading_path=heading_path,
        raw_text=raw_text,
    )
    out = {
        key: value
        for key, value in primary.items()
        if value not in ("", None, [], {})
    }
    if source_path and (not str(out.get("source_path") or "").strip()):
        out["source_path"] = source_path
    if source_name and (not str(out.get("source_name") or "").strip()):
        out["source_name"] = source_name
    if normalized_heading and (weak_primary or (not str(out.get("heading_path") or "").strip())):
        out["heading_path"] = normalized_heading
    if normalized_summary and (
        weak_primary
        or (
            not str(out.get("highlight_snippet") or out.get("snippet") or "").strip()
        )
    ):
        out["snippet"] = normalized_summary
        out["highlight_snippet"] = normalized_summary
    if selection_reason and (not str(out.get("selection_reason") or "").strip()):
        out["selection_reason"] = str(selection_reason or "").strip()
    return {
        key: value
        for key, value in out.items()
        if value not in ("", None, [], {})
    }


def _pick_multi_paper_card_raw_summary(
    *,
    prompt: str,
    card: dict,
    primary_evidence: dict | None,
) -> str:
    primary = dict(primary_evidence or {}) if isinstance(primary_evidence, dict) else {}
    primary_summary = str(primary.get("highlight_snippet") or primary.get("snippet") or "").strip()
    if primary_summary and (not _looks_generic_multi_paper_support_text(primary_summary, prompt=prompt)):
        return primary_summary

    card_summary = str(card.get("snippet") or "").strip()
    deepread_candidates = [
        str(item or "").strip()
        for item in list(card.get("deepread_texts") or [])
        if str(item or "").strip()
    ]
    deepread_summary = str(deepread_candidates[0] or "").strip() if deepread_candidates else ""

    if card_summary and (not _looks_generic_multi_paper_support_text(card_summary, prompt=prompt)):
        return card_summary
    if deepread_summary and (not _looks_generic_multi_paper_support_text(deepread_summary, prompt=prompt)):
        return deepread_summary
    return primary_summary or card_summary or deepread_summary


def _pick_multi_paper_doc_list_llm_pack_copy(
    *,
    prompt: str,
    meta: dict | None,
    source_name: str,
) -> tuple[str, str]:
    ref_pack = dict((meta or {}).get("ref_pack") or {}) if isinstance((meta or {}).get("ref_pack"), dict) else {}
    if not ref_pack:
        return "", ""
    summary_line = _single_line_summary(
        str(ref_pack.get("what") or "").strip(),
        source_name=source_name,
    )
    why_line = _single_line_summary(
        str(ref_pack.get("why") or "").strip(),
        source_name=source_name,
    )
    if summary_line and _looks_generic_multi_paper_support_text(summary_line, prompt=prompt):
        summary_line = ""
    if why_line and _looks_generic_multi_paper_support_text(why_line, prompt=prompt):
        why_line = ""
    return summary_line, why_line


def _build_multi_paper_doc_list_contract(
    *,
    prompt: str,
    seed_docs: list[dict] | None = None,
    answer_hits: list[dict] | None,
    evidence_cards: list[dict] | None,
    apply_prompt_filter: bool = True,
) -> list[dict]:
    entries: list[dict] = []
    entry_by_source: dict[str, dict] = {}

    def _merge_entry(
        *,
        source_path: str,
        source_name: str,
        heading_path: str,
        summary: str,
        summary_generation: str,
        why_line: str,
        why_generation: str,
        primary_evidence: dict | None,
        rank: int,
    ) -> None:
        src = str(source_path or "").strip()
        if not src:
            return
        entry = entry_by_source.get(src)
        if entry is None:
            entry = {
                "source_path": src,
                "source_name": str(source_name or "").strip() or _source_name_from_path_like(src),
                "heading_path": "",
                "summary_line": "",
                "_source_rank": int(rank),
            }
            entry_by_source[src] = entry
            entries.append(entry)
        else:
            entry["_source_rank"] = min(int(entry.get("_source_rank") or rank), int(rank))

        source_name_norm = str(source_name or "").strip() or _source_name_from_path_like(src)
        if source_name_norm and (not str(entry.get("source_name") or "").strip()):
            entry["source_name"] = source_name_norm

        current_primary_score = _multi_paper_primary_precision_score(
            entry.get("primary_evidence") if isinstance(entry.get("primary_evidence"), dict) else {}
        )
        incoming_primary_score = _multi_paper_primary_precision_score(primary_evidence)

        new_heading = str(heading_path or "").strip()
        cur_heading = str(entry.get("heading_path") or "").strip()
        if new_heading and (
            (not cur_heading)
            or (
                int(rank) >= 2
                and (
                    current_primary_score <= (0, 0, 0, 0, 0, 0)
                    or incoming_primary_score >= current_primary_score
                )
            )
        ):
            entry["heading_path"] = new_heading

        new_summary = str(summary or "").strip()
        cur_summary = str(entry.get("summary_line") or "").strip()
        if new_summary and (
            (not cur_summary)
            or (
                int(rank) >= 2
                and (
                    current_primary_score <= (0, 0, 0, 0, 0, 0)
                    or incoming_primary_score >= current_primary_score
                )
                and len(new_summary) >= max(24, len(cur_summary))
            )
        ):
            entry["summary_line"] = new_summary
            if str(summary_generation or "").strip():
                entry["summary_generation"] = str(summary_generation or "").strip()

        new_why = str(why_line or "").strip()
        cur_why = str(entry.get("why_line") or "").strip()
        if new_why and (
            (not cur_why)
            or (
                int(rank) >= 2
                and len(new_why) >= max(24, len(cur_why))
            )
        ):
            entry["why_line"] = new_why
            if str(why_generation or "").strip():
                entry["why_generation"] = str(why_generation or "").strip()

        if isinstance(primary_evidence, dict) and primary_evidence:
            norm_primary = {k: v for k, v in dict(primary_evidence).items() if v not in ("", None, [], {})}
            if norm_primary:
                current_primary = (
                    dict(entry.get("primary_evidence") or {})
                    if isinstance(entry.get("primary_evidence"), dict)
                    else {}
                )
                current_primary_score = _multi_paper_primary_precision_score(current_primary)
                norm_primary_score = _multi_paper_primary_precision_score(norm_primary)
                if (not current_primary) or norm_primary_score >= current_primary_score:
                    entry["primary_evidence"] = norm_primary
                    if str(norm_primary.get("heading_path") or "").strip():
                        entry["heading_path"] = str(norm_primary.get("heading_path") or "").strip()
                    snippet = _single_line_summary(
                        str(norm_primary.get("highlight_snippet") or norm_primary.get("snippet") or "").strip(),
                        source_name=str(entry.get("source_name") or ""),
                    )
                    summary_conflicts = _multi_paper_summary_conflicts_with_evidence(
                        str(entry.get("summary_line") or ""),
                        snippet,
                    )
                    if snippet and (
                        (not str(entry.get("summary_line") or "").strip())
                        or summary_conflicts
                        or (
                            norm_primary_score >= current_primary_score
                            and str(entry.get("summary_generation") or "").strip().lower() != "llm_pack"
                        )
                    ):
                        entry["summary_line"] = snippet
                        entry.pop("summary_generation", None)

    for doc in list(seed_docs or []):
        if not isinstance(doc, dict):
            continue
        meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
        source_path = str((meta or {}).get("source_path") or "").strip()
        source_name = _source_name_from_path_like(source_path)
        llm_summary, llm_why = _pick_multi_paper_doc_list_llm_pack_copy(
            prompt=prompt,
            meta=meta,
            source_name=source_name,
        )
        raw_summary = str((((meta or {}).get("ref_show_snippets") or [None])[0]) or doc.get("text") or "").strip()
        heading_path_raw = (
            str((meta or {}).get("ref_best_heading_path") or "").strip()
            or str((meta or {}).get("heading_path") or "").strip()
            or str((meta or {}).get("top_heading") or "").strip()
        )
        heading_path, summary = _normalize_multi_paper_surface_seed(
            source_name=source_name,
            heading_path=heading_path_raw,
            raw_text=raw_summary,
        )
        primary_evidence = _normalize_multi_paper_contract_primary_evidence(
            source_path=source_path,
            source_name=source_name,
            heading_path=heading_path,
            raw_text=raw_summary,
            primary_evidence=None,
            selection_reason="pending_section_seed",
        )
        _merge_entry(
            source_path=source_path,
            source_name=source_name,
            heading_path=heading_path,
            summary=llm_summary or summary,
            summary_generation="llm_pack" if llm_summary else "",
            why_line=llm_why,
            why_generation="llm_pack" if llm_why else "",
            primary_evidence=primary_evidence,
            rank=1,
        )

    for card in list(evidence_cards or []):
        if not isinstance(card, dict):
            continue
        primary = dict(card.get("primary_evidence") or {}) if isinstance(card.get("primary_evidence"), dict) else {}
        source_path = str(card.get("source_path") or primary.get("source_path") or "").strip()
        source_name = str(primary.get("source_name") or "").strip() or _source_name_from_path_like(source_path)
        raw_summary = _pick_multi_paper_card_raw_summary(
            prompt=prompt,
            card=card,
            primary_evidence=primary,
        )
        heading_path_raw = str(primary.get("heading_path") or "").strip() or str(card.get("heading") or "").strip()
        heading_path, summary = _normalize_multi_paper_surface_seed(
            source_name=source_name,
            heading_path=heading_path_raw,
            raw_text=raw_summary,
        )
        normalized_primary = _normalize_multi_paper_contract_primary_evidence(
            source_path=source_path,
            source_name=source_name,
            heading_path=heading_path,
            raw_text=raw_summary,
            primary_evidence=primary,
            selection_reason=str(primary.get("selection_reason") or "answer_hit_top").strip(),
        )
        _merge_entry(
            source_path=source_path,
            source_name=source_name,
            heading_path=heading_path,
            summary=summary,
            summary_generation="",
            why_line="",
            why_generation="",
            primary_evidence=normalized_primary,
            rank=3,
        )

    for hit_index, hit in enumerate(list(answer_hits or []), start=1):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        source_path = str((meta or {}).get("source_path") or "").strip()
        source_name = _source_name_from_path_like(source_path)
        llm_summary, llm_why = _pick_multi_paper_doc_list_llm_pack_copy(
            prompt=prompt,
            meta=meta,
            source_name=source_name,
        )
        raw_summary = str((((meta or {}).get("ref_show_snippets") or [None])[0]) or hit.get("text") or "").strip()
        heading_path_raw = (
            str((meta or {}).get("ref_best_heading_path") or "").strip()
            or str((meta or {}).get("heading_path") or "").strip()
            or str((meta or {}).get("top_heading") or "").strip()
        )
        heading_path, summary = _normalize_multi_paper_surface_seed(
            source_name=source_name,
            heading_path=heading_path_raw,
            raw_text=raw_summary,
        )
        primary_evidence = _normalize_multi_paper_contract_primary_evidence(
            source_path=source_path,
            source_name=source_name,
            heading_path=heading_path,
            raw_text=raw_summary,
            primary_evidence=None,
            selection_reason="answer_hit_top",
        )
        _merge_entry(
            source_path=source_path,
            source_name=source_name,
            heading_path=heading_path,
            summary=llm_summary or summary,
            summary_generation="llm_pack" if llm_summary else "",
            why_line=llm_why,
            why_generation="llm_pack" if llm_why else "",
            primary_evidence=primary_evidence,
            rank=2,
        )
        if source_path and source_path in entry_by_source:
            entry_by_source[source_path].setdefault("citation_num", hit_index)

    normalized_entries = [
        {
            k: v
            for k, v in dict(raw_entry or {}).items()
            if k not in {"_source_rank"} and v not in ("", None, [], {})
        }
        for raw_entry in entries
    ]
    if not apply_prompt_filter:
        return normalized_entries
    return _filter_multi_paper_doc_list_contract(prompt=prompt, doc_list=normalized_entries)



def _format_multi_paper_list_answer_v2(*, prompt: str, docs: list[dict]) -> str:
    rows = [dict(item) for item in list(docs or []) if isinstance(item, dict)]
    if not rows:
        return ""
    prefer_zh = bool(prompt_prefers_zh(prompt))
    topic = _extract_multi_paper_topic(prompt)
    paper_count = len(rows)
    if prefer_zh:
        intro = (
            f"\u6839\u636e\u547d\u4e2d\u7684\u5e93\u5185\u6587\u732e\uff0c\u4ee5\u4e0b {paper_count} \u7bc7\u6587\u7ae0\u4e0e\u201c{topic}\u201d\u76f4\u63a5\u76f8\u5173\uff1a"
            if topic
            else f"\u6839\u636e\u547d\u4e2d\u7684\u5e93\u5185\u6587\u732e\uff0c\u4ee5\u4e0b {paper_count} \u7bc7\u6587\u7ae0\u4e0e\u5f53\u524d\u95ee\u9898\u76f4\u63a5\u76f8\u5173\uff1a"
        )
        lines = [intro, ""]
        for idx, item in enumerate(rows, start=1):
            name = str(item.get("source_name") or _source_name_from_path_like(item.get("source_path") or "")).strip() or f"\u6587\u732e {idx}"
            heading = str(item.get("heading_path") or "").strip()
            summary = str(item.get("summary_line") or "").strip()
            citation_num = _as_positive_int(item.get("citation_num"))
            citation_marker = f" [{citation_num}]" if citation_num > 0 else ""
            match_note = _multi_paper_topic_match_note(
                prompt=prompt,
                match_kind=str(item.get("topic_match_kind") or ""),
            )
            lines.append(f"{idx}. **{name}**")
            if heading:
                lines.append(f"   - \u5b9a\u4f4d\uff1a{heading}")
            if summary:
                lines.append(f"   - \u4f9d\u636e\uff1a{summary}{citation_marker}")
            elif citation_marker:
                lines.append(f"   - \u6765\u6e90\uff1a{citation_marker.strip()}")
            why_line = str(item.get("why_line") or "").strip()
            if why_line:
                lines.append(f"   - \u4e3a\u4ec0\u4e48\u8bfb\uff1a{why_line}")
            if match_note:
                lines.append(f"   - \u76f8\u5173\u6027\uff1a{match_note}")
            lines.append("")
        return "\n".join(lines).strip()

    intro = (
        f"The following library paper directly relates to '{topic}':"
        if topic and paper_count == 1
        else f"The following library paper directly relates to the current query:"
        if paper_count == 1
        else f"The following {paper_count} library papers directly relate to '{topic}':"
        if topic
        else f"The following {paper_count} library papers directly relate to the current query:"
    )
    lines = [intro, ""]
    for idx, item in enumerate(rows, start=1):
        name = str(item.get("source_name") or _source_name_from_path_like(item.get("source_path") or "")).strip() or f"Paper {idx}"
        heading = str(item.get("heading_path") or "").strip()
        summary = str(item.get("summary_line") or "").strip()
        citation_num = _as_positive_int(item.get("citation_num"))
        citation_marker = f" [{citation_num}]" if citation_num > 0 else ""
        match_note = _multi_paper_topic_match_note(
            prompt=prompt,
            match_kind=str(item.get("topic_match_kind") or ""),
        )
        lines.append(f"{idx}. **{name}**")
        if heading:
            lines.append(f"   - Locate: {heading}")
        if summary:
            lines.append(f"   - Evidence: {summary}{citation_marker}")
        elif citation_marker:
            lines.append(f"   - Source: {citation_marker.strip()}")
        why_line = str(item.get("why_line") or "").strip()
        if why_line:
            lines.append(f"   - Why read it: {why_line}")
        if match_note:
            lines.append(f"   - Match: {match_note}")
        lines.append("")
    return "\n".join(lines).strip()


_MULTI_PAPER_NUMBERED_SECTION_RE = re.compile(
    r"(?m)^\s*(?:#{1,6}\s*)?(?:\u7b2c\s*)?(\d{1,2})"
    r"(?:\s*(?:\u7bc7|\u6b65|\u9879|\u90e8)\s*[:\uff1a]\s*|[.)]\s+)"
)


def _multi_paper_numbered_sections(answer: str) -> list[str]:
    text = str(answer or "")
    matches = list(_MULTI_PAPER_NUMBERED_SECTION_RE.finditer(text))
    return [
        text[match.start() : (matches[idx + 1].start() if idx + 1 < len(matches) else len(text))]
        for idx, match in enumerate(matches)
    ]


def _count_multi_paper_answer_items(answer: str) -> int:
    numbers = [
        int(match.group(1))
        for match in _MULTI_PAPER_NUMBERED_SECTION_RE.finditer(str(answer or ""))
    ]
    if not numbers:
        return 0
    expected = list(range(1, len(numbers) + 1))
    return len(numbers) if numbers == expected else len(set(numbers))


def _prompt_requests_multi_paper_source_markers(prompt: str) -> bool:
    return bool(
        re.search(
            r"\b(?:cite|citation|source\s+(?:number|marker)|evidence\s+(?:number|marker))\b|"
            r"\u6765\u6e90\u7f16\u53f7|\u6765\u6e90\u6807\u8bb0|\u5f15\u7528\u7f16\u53f7|"
            r"\u7528\u6765\u6e90|\u53ef\u70b9\u56de|\u70b9\u56de\u539f\u6587|\u6838\u5bf9\u7684\u4f9d\u636e",
            str(prompt or ""),
            flags=re.I,
        )
    )


def _section_has_citation_marker(section: str) -> bool:
    return bool(_FREEFORM_NUMERIC_CITE_RE.search(str(section or "")) or _has_structured_cite_marker(section))


def _strip_requested_multi_paper_extras(answer: str) -> str:
    text = str(answer or "").strip()
    extra_block = re.compile(
        r"(?ims)\n\s*(?:---\s*\n\s*)?(?:#{1,6}\s*)?(?:\*\*)?"
        r"(?:\u8865\u5145\u8bf4\u660e|\u8865\u5145\u9605\u8bfb|\u5ef6\u4f38\u9605\u8bfb|\u8fdb\u4e00\u6b65\u9605\u8bfb|"
        r"additional\s+reading|further\s+reading|supplementary\s+note)"
        r"(?:\s*[:\uff1a](?:\*\*)?)?.*$"
    )
    text = extra_block.sub("", text).rstrip()
    citation_chain_tail = re.compile(
        r"(?ims)\n\s*(?:\u5982\u679c\u60f3\u987a\u7740\u8bba\u6587\u7684\u5f15\u7528\u94fe|"
        r"if\s+you\s+want\s+to\s+follow\s+the\s+citation\s+chain).*$"
    )
    return citation_chain_tail.sub("", text).rstrip()


def _strip_single_paper_selection_extras(answer: str) -> str:
    text = str(answer or "").strip()
    if not text:
        return text
    candidate_section = re.compile(
        r"(?ims)^\s*#{1,6}\s*(?:"
        r"\u5176\u4ed6\u5019\u9009(?:\u8bba\u6587|\u6587\u7ae0|\u6587\u732e)?(?:\u4e3a\u4f55\u4e0d\u9009)?|"
        r"\u5176\u4ed6\u8bba\u6587\u4e3a\u4f55\u4e0d\u9009|\u672a\u9009\u5019\u9009|"
        r"other\s+candidates?|why\s+not\s+the\s+others?|alternatives?)\s*$"
        r".*?(?=^\s*#{1,6}\s|\Z)"
    )
    out = candidate_section.sub("", text)
    out = re.sub(r"(?m)(?:^\s*---\s*$\n*){2,}", "---\n\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _multi_paper_section_hit_num(section: str, answer_hits: list[dict] | None) -> int:
    section_norm = _normalize_topic_identity(section)
    if not section_norm:
        return 0
    stop_tokens = {"paper", "article", "journal", "single", "pixel", "imaging", "study", "method"}
    best_num = 0
    best_score = 0
    for hit_num, hit in enumerate(list(answer_hits or []), start=1):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        source_path = str((meta or {}).get("source_path") or hit.get("source_path") or "").strip()
        source_name = _source_name_from_path_like(source_path)
        source_norm = _normalize_topic_identity(source_name)
        tokens = {
            token
            for token in source_norm.split()
            if len(token) >= 4 and (not token.isdigit()) and token not in stop_tokens
        }
        score = sum(1 for token in tokens if token in section_norm)
        if score > best_score:
            best_num = hit_num
            best_score = score
    return best_num if best_score >= 2 else 0


def _select_multi_paper_doc_list_from_answer(
    *,
    answer: str,
    answer_hits: list[dict] | None,
    doc_list: list[dict] | None,
) -> list[dict]:
    entries_by_source: dict[str, dict] = {}
    for raw in list(doc_list or []):
        if not isinstance(raw, dict):
            continue
        source_path = str(raw.get("source_path") or "").strip()
        if source_path:
            entries_by_source[source_path.replace("\\", "/").lower()] = dict(raw)

    selected: list[dict] = []
    seen_sources: set[str] = set()
    for section in _multi_paper_numbered_sections(answer):
        hit_num = _multi_paper_section_hit_num(section, answer_hits)
        if hit_num <= 0:
            marker = _FREEFORM_NUMERIC_CITE_RE.search(section)
            try:
                hit_num = int(marker.group(1)) if marker else 0
            except Exception:
                hit_num = 0
        if not (1 <= hit_num <= len(list(answer_hits or []))):
            continue
        hit = list(answer_hits or [])[hit_num - 1]
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        source_path = str((meta or {}).get("source_path") or hit.get("source_path") or "").strip()
        source_key = source_path.replace("\\", "/").lower()
        if not source_key or source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        entry = dict(entries_by_source.get(source_key) or {})
        entry["source_path"] = source_path
        entry["source_name"] = str(entry.get("source_name") or _source_name_from_path_like(source_path)).strip()
        entry["citation_num"] = int(hit_num)
        selected.append(entry)
    return selected


def _repair_requested_multi_paper_answer(
    answer: str,
    *,
    prompt: str,
    answer_hits: list[dict] | None,
) -> str:
    requested_count = extract_requested_paper_count(prompt)
    if requested_count is None:
        return str(answer or "")
    text = _strip_requested_multi_paper_extras(answer)
    if not _prompt_requests_multi_paper_source_markers(prompt):
        return text
    matches = list(_MULTI_PAPER_NUMBERED_SECTION_RE.finditer(text))
    if len(matches) != requested_count:
        return text
    repaired: list[str] = [text[: matches[0].start()]]
    for idx, match in enumerate(matches):
        section_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section = text[match.start() : section_end]
        if not _section_has_citation_marker(section):
            hit_num = _multi_paper_section_hit_num(section, answer_hits)
            if hit_num > 0:
                trailing_separator = re.search(r"\n\s*---\s*$", section)
                if trailing_separator:
                    section = (
                        section[: trailing_separator.start()].rstrip()
                        + f" [{hit_num}]\n\n---\n\n"
                    )
                else:
                    section = section.rstrip() + f" [{hit_num}]\n\n"
        repaired.append(section)
    return "".join(repaired).rstrip()


def _multi_paper_answer_needs_contract_rebuild(*, answer: str, prompt: str) -> bool:
    text = str(answer or "").strip()
    if len(text) < 120:
        return True
    if re.search(r"\bDOC-\d{1,3}(?:-S\d{1,3})?\b", text, flags=re.I):
        return True
    requested_count = extract_requested_paper_count(prompt)
    if requested_count is None:
        return False
    actual_count = _count_multi_paper_answer_items(text)
    if actual_count != requested_count:
        return True
    if _prompt_requests_multi_paper_source_markers(prompt):
        sections = _multi_paper_numbered_sections(text)
        return len(sections) != requested_count or any(
            not _section_has_citation_marker(section)
            for section in sections
        )
    return False


def _extract_multi_paper_topic(prompt: str) -> str:
    return _shared_extract_multi_paper_topic(prompt)


def _multi_paper_prompt_requires_explicit_focus_match(prompt: str) -> bool:
    return _shared_prompt_requires_reference_focus_match(prompt)


def _looks_generic_multi_paper_support_text(text: str, *, prompt: str) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return False
    patterns = (
        "directly related to the current query",
        "directly relevant to the current query",
        "directly relevant to the current question",
        "directly responds to the user",
        "can serve as the current question",
        "matched section",
        "besides this paper, what other",
        "what other...",
        "\u4e0e\u5f53\u524d\u95ee\u9898\u76f4\u63a5\u76f8\u5173",
        "\u4e0e\u7528\u6237\u67e5\u8be2",
        "\u76f4\u63a5\u56de\u5e94\u7528\u6237",
        "\u5e93\u5185\u660e\u786e\u547d\u4e2d",
        "\u547d\u4e2d\u7ae0\u8282",
        "\u4e3b\u9898\u4e00\u81f4",
        "\u540c\u7c7b\u6280\u672f\u6587\u732e",
        "\u53ef\u4f5c\u4e3a\u5f53\u524d\u95ee\u9898",
    )
    if any(pattern in low for pattern in patterns):
        return True
    prompt_echo = str(prompt or "").strip().lower()
    if prompt_echo:
        prompt_echo = re.sub(r"\s+", " ", prompt_echo)
        if len(prompt_echo) >= 18 and prompt_echo[:32] in low:
            return True
    return False


def _multi_paper_focus_term_only_negated(term: str, surface: str) -> bool:
    token = str(term or "").strip().lower()
    normalized_surface = str(surface or "").strip().lower()
    if not token or not normalized_surface:
        return False
    term_pattern = _multi_paper_term_presence_pattern(token)
    if not term_pattern:
        return False
    matches = list(re.finditer(term_pattern, normalized_surface, flags=re.I))
    if not matches:
        return False
    english_before_re = re.compile(
        r"\b(?:without|not|no|lack(?:s|ing)?|avoid(?:s|ed|ing)?|rather than|instead of|"
        r"does not mention|doesn't mention|does not discuss|doesn't discuss)\b"
        r"[^.!?;\n]{0,32}$",
        flags=re.I,
    )
    chinese_before_re = re.compile(
        r"(?:\u672a\u63d0\u53ca|\u4e0d\u6d89\u53ca|\u6ca1\u6709|\u5e76\u672a|\u4e0d\u662f)"
        r"[^\u3002\uff01\uff1f\uff1b\n]{0,20}$",
        flags=re.I,
    )
    english_after_re = re.compile(r"^[^.!?;\n]{0,24}\b(?:not|absent|omitted)\b", flags=re.I)
    negated_count = 0
    for match in matches:
        prefix = normalized_surface[max(0, match.start() - 40) : match.start()]
        suffix = normalized_surface[match.end() : min(len(normalized_surface), match.end() + 28)]
        if (
            english_before_re.search(prefix)
            or chinese_before_re.search(prefix)
            or english_after_re.search(suffix)
        ):
            negated_count += 1
    return negated_count >= len(matches)


def _prompt_targets_sci_topic(prompt: str) -> bool:
    return _shared_prompt_targets_sci_topic(prompt)


def _format_multi_paper_list_answer(*, prompt: str, docs: list[dict]) -> str:
    return _format_multi_paper_list_answer_v2(prompt=prompt, docs=docs)


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
    if prompt_likely_cross_paper_refs(str(prompt_text or "")):
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


def _finalize_user_visible_citation_markers(
    answer: str,
    *,
    prompt: str,
    answer_output_mode: str,
    paper_guide_mode: bool,
    prompt_family: str,
    has_hits: bool,
    answer_hits: list[dict],
    db_dir: Path | None,
    locked_citation_source: dict | None,
    support_resolution: list[dict] | None,
    candidate_refs_by_source: dict[str, list[int]] | None,
    retrieval_confidence_hint: dict[str, object] | None,
    allow_paper_guide_structured_refs: bool = False,
) -> str:
    text = str(answer or "").strip()
    if bool(paper_guide_mode):
        text = _sanitize_paper_guide_answer_for_user(
            text,
            has_hits=bool(has_hits),
            prompt=prompt,
            prompt_family=prompt_family,
            preserve_structured_cites=True if allow_paper_guide_structured_refs else None,
        )
        text = _maybe_ensure_minimum_paper_guide_citation(
            text,
            paper_guide_mode=True,
            prompt_family=prompt_family,
            has_hits=bool(has_hits),
            support_resolution=list(support_resolution or []),
            candidate_refs_by_source=dict(candidate_refs_by_source or {}),
            retrieval_confidence_hint=dict(retrieval_confidence_hint or {}),
            locked_citation_source=locked_citation_source,
        )
        text = _maybe_append_prompt_requested_inpaper_refs(
            text,
            prompt=prompt,
            answer_hits=answer_hits,
            db_dir=db_dir,
            locked_citation_source=locked_citation_source,
        )
        text = _sanitize_paper_guide_answer_for_user(
            text,
            has_hits=bool(has_hits),
            prompt=prompt,
            prompt_family=prompt_family,
            preserve_structured_cites=True if allow_paper_guide_structured_refs else None,
        )

    text = _sanitize_internal_doc_label_blocks(text)
    preserve_numeric_citations = _should_preserve_final_answer_numeric_citations(
        prompt=prompt,
        answer_output_mode=answer_output_mode,
        paper_guide_mode=bool(paper_guide_mode),
        prompt_family=prompt_family,
        has_hits=bool(has_hits),
    )
    preserve_structured_citations = _should_preserve_final_answer_structured_citations(
        prompt=prompt,
        answer_output_mode=answer_output_mode,
        paper_guide_mode=bool(paper_guide_mode),
        prompt_family=prompt_family,
        allow_paper_guide_structured_refs=bool(allow_paper_guide_structured_refs),
    )
    text = _strip_final_answer_citation_markers(
        text,
        preserve_numeric_markers=preserve_numeric_citations,
        preserve_structured_markers=preserve_structured_citations,
    )
    return _sanitize_empty_markdown_label_fragments(text)


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
    doc_list_contract: list[dict] | None = None,
    paper_guide_contracts_seed: dict | None = None,
) -> dict:
    seed = dict(paper_guide_contracts_seed or {})
    doc_list = [dict(item) for item in list(doc_list_contract or []) if isinstance(item, dict)]
    primary_evidence = _pick_shared_primary_evidence(
        paper_guide_contracts_seed=paper_guide_contracts_seed,
        evidence_cards=evidence_cards,
    )
    render_packet_seed = seed.get("render_packet") if isinstance(seed.get("render_packet"), dict) else {}
    citation_plan_seed = seed.get("citation_plan") if isinstance(seed.get("citation_plan"), dict) else {}
    final_packet_answer = str(final_answer_markdown or render_packet_seed.get("answer_markdown") or "").strip()
    seed_packet_answer = str(render_packet_seed.get("answer_markdown") or "").strip()
    seed_packet_matches_final = bool(
        seed_packet_answer
        and final_packet_answer
        and seed_packet_answer == final_packet_answer
    )

    def _seed_render_text(key: str) -> str:
        if not seed_packet_matches_final:
            return ""
        return str(render_packet_seed.get(key) or "").strip()
    if (
        (not paper_guide_mode)
        and (not primary_evidence)
        and (not render_packet_seed)
        and (not doc_list)
        and (not citation_plan_seed)
    ):
        return {}

    snapshot = {"version": 1}
    if not paper_guide_mode:
        render_packet_model = _build_paper_guide_render_packet_model(
            answer_markdown=final_packet_answer,
            notice=str(render_packet_seed.get("notice") or "").strip(),
            rendered_body=_seed_render_text("rendered_body"),
            rendered_content=_seed_render_text("rendered_content"),
            copy_markdown=_seed_render_text("copy_markdown"),
            copy_text=_seed_render_text("copy_text"),
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
        if doc_list:
            snapshot["doc_list"] = doc_list
        if citation_plan_seed:
            snapshot["citation_plan"] = dict(citation_plan_seed)
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
    if citation_plan_seed:
        snapshot["citation_plan"] = dict(citation_plan_seed)
    render_packet_model = _build_paper_guide_render_packet_model(
        answer_markdown=final_packet_answer,
        notice=str(render_packet_seed.get("notice") or "").strip(),
        rendered_body=_seed_render_text("rendered_body"),
        rendered_content=_seed_render_text("rendered_content"),
        copy_markdown=_seed_render_text("copy_markdown"),
        copy_text=_seed_render_text("copy_text"),
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
    if doc_list:
        snapshot["doc_list"] = doc_list
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
    def _primary_precision_score(primary: dict | None) -> tuple[int, int, int, int, int, int]:
        if not isinstance(primary, dict) or not primary:
            return (0, 0, 0, 0, 0, 0)
        reason = str(primary.get("selection_reason") or primary.get("selectionReason") or "").strip().lower()
        reason_rank = {
            "prompt_aligned": 6,
            "reader_open": 5,
            "strict_locate": 5,
            "provenance_segment": 5,
            "shared_refs_pack": 5,
            "pending_section_seed": 2,
            "shared_contract_seed": 1,
            "answer_hit_top": 0,
        }.get(reason, 3 if reason else 0)
        return (
            reason_rank,
            1 if str(primary.get("block_id") or primary.get("blockId") or "").strip() else 0,
            1 if str(primary.get("anchor_id") or primary.get("anchorId") or "").strip() else 0,
            1 if str(primary.get("heading_path") or primary.get("headingPath") or "").strip() else 0,
            1 if str(primary.get("snippet") or "").strip() else 0,
            1
            if str(primary.get("source_path") or primary.get("sourcePath") or primary.get("source_name") or primary.get("sourceName") or "").strip()
            else 0,
        )

    best: dict = {}
    best_score = (0, 0, 0, 0, 0, 0)

    seed = dict(paper_guide_contracts_seed or {})
    candidates: list[dict] = []
    primary = seed.get("primary_evidence")
    if isinstance(primary, dict) and primary:
        candidates.append(dict(primary))
    for card in list(evidence_cards or []):
        if not isinstance(card, dict):
            continue
        primary = card.get("primary_evidence")
        if isinstance(primary, dict) and primary:
            candidates.append(dict(primary))

    for candidate in candidates:
        score = _primary_precision_score(candidate)
        if (not best) or score > best_score:
            best = dict(candidate)
            best_score = score
    return best


def _maybe_clarify_negative_boundary_answer(answer: str, *, prompt: str) -> str:
    text = str(answer or "").strip()
    prompt_text = str(prompt or "").strip()
    if not text or not prompt_text:
        return text
    if "\u4e0d\u662f" in text or re.search(r"(?i)\bnot\s+(?:a|the\s+)?(?:core|central|main|relevant)\b", text):
        return text
    if not _NEGATIVE_BOUNDARY_PROMPT_RE.search(prompt_text):
        return text
    if not _NEGATIVE_BOUNDARY_ANSWER_RE.search(text):
        return text

    replacement = "\u5173\u7cfb\u4e0d\u5927\uff0c\u4e0d\u662f\u5f53\u524d\u4e3b\u7ebf\u7684\u6838\u5fc3\u6587\u732e"
    out = re.sub(r"\u5173\u7cfb\u4e0d\u5927", replacement, text, count=1)
    if out != text:
        return out
    out = re.sub(
        r"\u4e0d\u5efa\u8bae",
        "\u4e0d\u662f\u5f53\u524d\u4e3b\u7ebf\u7684\u6838\u5fc3\u6587\u732e\uff0c\u4e0d\u5efa\u8bae",
        text,
        count=1,
    )
    if out != text:
        return out
    if text.startswith("**\u7ed3\u8bba"):
        return re.sub(
            r"^(\*\*\u7ed3\u8bba[^\n]*?\uff1a\s*)",
            lambda match: match.group(1)
            + "\u4e0d\u662f\u5f53\u524d\u4e3b\u7ebf\u7684\u6838\u5fc3\u6587\u732e\uff1b",
            text,
            count=1,
        )
    return "\u4e0d\u662f\u5f53\u524d\u4e3b\u7ebf\u7684\u6838\u5fc3\u6587\u732e\uff1b" + text


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
    research_answer_plan: str = "",
    paper_guide_contracts_seed: dict | None = None,
    paper_guide_retrieval_confidence_hint: dict[str, object] | None = None,
    apply_paper_guide_answer_postprocess,
    maybe_append_library_figure_markdown,
    validate_structured_citations,
    build_paper_guide_supplement_lines=None,
    validate_freeform_numeric_citations=None,
) -> dict:
    resolved_paper_guide_intent = _resolve_paper_guide_intent(
        prompt_for_user or prompt,
        prompt_family=paper_guide_prompt_family,
    )
    effective_paper_guide_family = str(getattr(resolved_paper_guide_intent, "family", "") or "").strip()
    sanitize_paper_guide_family = effective_paper_guide_family or "overview"
    citation_plan_seed = (
        dict((paper_guide_contracts_seed or {}).get("citation_plan") or {})
        if isinstance((paper_guide_contracts_seed or {}).get("citation_plan"), dict)
        else {}
    )
    citation_plan_budget = (
        dict(citation_plan_seed.get("budget") or {})
        if isinstance(citation_plan_seed.get("budget"), dict)
        else {}
    )
    system_b_explicitly_disabled = bool(
        citation_plan_seed
        and "system_b" in citation_plan_budget
        and int(citation_plan_budget.get("system_b") or 0) <= 0
    )
    research_answer_plan_norm = str(research_answer_plan or "").strip()
    answer_audit_requested = prompt_requests_answer_audit(prompt_for_user or prompt)
    multi_paper_list_prompt = bool(prompt_explicitly_requests_multi_paper_list(prompt_for_user or prompt))
    single_paper_pick_prompt = bool(prompt_explicitly_requests_single_paper_pick(prompt_for_user or prompt))
    library_paper_selection_prompt = bool(multi_paper_list_prompt or single_paper_pick_prompt)
    raw_answer_had_internal_doc_labels = bool(
        re.search(r"\bDOC-\d{1,3}(?:-S\d{1,3})?\b", str(partial or ""), flags=re.I)
    )
    multi_paper_doc_list = (
        _build_multi_paper_doc_list_contract(
            prompt=prompt or prompt_for_user,
            seed_docs=list((paper_guide_contracts_seed or {}).get("doc_list_seed") or []),
            answer_hits=list(answer_hits or []),
            evidence_cards=list(paper_guide_evidence_cards or []),
            apply_prompt_filter=False,
        )
        if multi_paper_list_prompt
        else []
    )
    answer = _normalize_math_markdown(
        _strip_model_ref_section(_sanitize_structured_cite_tokens(partial or ""))
    ).strip() or "(No text returned)"
    if answer_audit_requested:
        answer = _replace_answer_audit_doc_labels(answer)
        answer = _strip_answer_audit_internal_citation_review(
            answer,
            prompt=prompt_for_user or prompt,
        )
    answer = _sanitize_empty_markdown_label_fragments(answer)
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
    template_repair_meta: dict[str, object] = {"changed": False}
    if paper_guide_mode:
        answer, template_repair_meta = _repair_template_only_paper_guide_answer(
            answer,
            prompt=prompt_for_user or prompt,
            prompt_family=sanitize_paper_guide_family,
            support_resolution=list(paper_guide_support_resolution or []),
            cards=list(paper_guide_evidence_cards or []),
            fallback_source_path=str(paper_guide_bound_source_path or paper_guide_direct_source_path or paper_guide_focus_source_path or ""),
        )
    # Step 1: Promote bare [n] where n < CITATION_OFFSET to structured
    # [[CITE:<sid>:n]] — these are in-paper bibliography references (System B).
    # Hit citations use [OFFSET+1] numbers and are handled in step 2.
    if not paper_guide_mode and not answer_audit_requested:
        answer = _promote_numeric_inpaper_refs(
            answer,
            answer_hits=answer_hits,
            db_dir=db_dir,
            paper_guide_mode=False,
        )
    # Step 2: Strip the citation offset so System A markers like [10001], [10002]
    # become [1], [2] for standard rendering.  After this, all remaining [n] are
    # 1-based hit citations; System B refs are already [[CITE:...]].
    if not paper_guide_mode:
        answer = _strip_citation_offset(answer)
    # Step 3: Strip LaTeX footnote markers ($^n$, $_{xx}$) that leak from paper text.
    answer = _strip_latex_footnote_markers(answer)
    if not answer_audit_requested:
        answer = _maybe_append_prompt_requested_inpaper_refs(
            answer,
            prompt=prompt_for_user or prompt,
            answer_hits=answer_hits,
            db_dir=db_dir,
            locked_citation_source=locked_citation_source,
        )
    paper_guide_reference_opportunities: list[dict[str, object]] = [
        dict(item)
        for item in list((paper_guide_contracts_seed or {}).get("reference_opportunities") or [])
        if isinstance(item, dict)
    ]
    paper_guide_reference_apply_meta: dict[str, object] = {"mode": "none", "tail_used": False}
    paper_guide_candidate_refs_effective = (
        dict(paper_guide_candidate_refs_by_source or {}) if bool(paper_guide_mode) else {}
    )
    if bool(paper_guide_mode) and system_b_explicitly_disabled:
        paper_guide_reference_opportunities = []
    elif bool(paper_guide_mode):
        reference_source_path = str(
            paper_guide_bound_source_path
            or paper_guide_direct_source_path
            or paper_guide_focus_source_path
            or ""
        ).strip()
        paper_guide_reference_opportunities = detect_paper_guide_reference_opportunities(
            prompt=prompt_for_user or prompt,
            answer=answer,
            prompt_family=sanitize_paper_guide_family,
            source_path=reference_source_path,
            support_resolution=list(paper_guide_support_resolution or []),
            support_slots=list(paper_guide_support_slots or []),
            cards=list(paper_guide_evidence_cards or []),
            max_items=3,
        )
    elif not answer_audit_requested and not library_paper_selection_prompt:
        paper_guide_reference_opportunities = detect_text_reference_opportunities(
            prompt=prompt_for_user or prompt,
            answer=answer,
            answer_hits=answer_hits,
            db_dir=db_dir,
            max_items=3,
        )
    if paper_guide_reference_opportunities:
        answer, paper_guide_reference_apply_meta = apply_reference_opportunities_to_answer(
            answer,
            prompt=prompt_for_user or prompt,
            opportunities=paper_guide_reference_opportunities,
        )
        reference_opportunities_for_validation = paper_guide_reference_opportunities
        if not bool(paper_guide_mode):
            applied_refs: set[int] = set()
            for key in ("injected_refs", "tail_refs"):
                for raw_ref in list(paper_guide_reference_apply_meta.get(key) or []):
                    try:
                        ref_num = int(raw_ref)
                    except Exception:
                        continue
                    if ref_num > 0:
                        applied_refs.add(ref_num)
            filtered_reference_opportunities: list[dict[str, object]] = []
            for item in paper_guide_reference_opportunities:
                try:
                    item_ref_num = int(item.get("ref_num") or 0)
                except Exception:
                    item_ref_num = 0
                if item_ref_num in applied_refs:
                    filtered_reference_opportunities.append(item)
            reference_opportunities_for_validation = filtered_reference_opportunities
        paper_guide_candidate_refs_effective = merge_reference_opportunity_candidate_refs(
            paper_guide_candidate_refs_effective,
            reference_opportunities_for_validation,
        )
    answer, citation_validation = validate_structured_citations(
        answer,
        answer_hits=answer_hits,
        db_dir=db_dir,
        locked_source=locked_citation_source,
        paper_guide_mode=bool(paper_guide_mode),
        paper_guide_candidate_refs_by_source=dict(paper_guide_candidate_refs_effective or {}),
        paper_guide_support_slots=list(paper_guide_support_slots or []),
        paper_guide_support_resolution=list(paper_guide_support_resolution or []),
    )
    structured_refs_allowed = bool(
        bool(paper_guide_reference_opportunities)
        or sanitize_paper_guide_family == "citation_lookup"
        or _prompt_explicitly_requests_citation_lookup(prompt_for_user or prompt)
        or (bool(paper_guide_mode) and "citation" in str(answer_output_mode or "").strip().lower())
    )
    # Standard RAG [n] citation validation — catch hallucinated ref nums.
    paper_guide_validated_structured_refs = bool(
        structured_refs_allowed
        and _has_structured_cite_marker(answer)
        and (
            int(dict(citation_validation or {}).get("kept") or 0) > 0
            or int(dict(citation_validation or {}).get("rewritten") or 0) > 0
        )
    )
    if system_b_explicitly_disabled or (
        paper_guide_reference_opportunities
        and not paper_guide_validated_structured_refs
        and bool(paper_guide_reference_apply_meta.get("tail_used"))
    ):
        answer = strip_reference_opportunity_note(answer)
    if callable(validate_freeform_numeric_citations):
        answer, freeform_validation = validate_freeform_numeric_citations(
            answer,
            answer_hits=answer_hits,
        )
        citation_validation["freeform"] = freeform_validation
    answer = _finalize_user_visible_citation_markers(
        answer,
        prompt=prompt_for_user or prompt,
        answer_output_mode=answer_output_mode,
        paper_guide_mode=bool(paper_guide_mode),
        prompt_family=sanitize_paper_guide_family,
        has_hits=bool(answer_hits),
        answer_hits=answer_hits,
        db_dir=db_dir,
        locked_citation_source=locked_citation_source,
        support_resolution=list(paper_guide_support_resolution or []),
        candidate_refs_by_source=dict(paper_guide_candidate_refs_effective or {}),
        retrieval_confidence_hint=dict(paper_guide_retrieval_confidence_hint or {}),
        allow_paper_guide_structured_refs=bool(paper_guide_validated_structured_refs),
    )
    answer = _maybe_clarify_negative_boundary_answer(answer, prompt=prompt_for_user or prompt)
    if single_paper_pick_prompt:
        answer = _strip_single_paper_selection_extras(answer)
    if multi_paper_list_prompt:
        answer = _repair_requested_multi_paper_answer(
            answer,
            prompt=prompt_for_user or prompt,
            answer_hits=answer_hits,
        )
        selected_multi_paper_doc_list = _select_multi_paper_doc_list_from_answer(
            answer=answer,
            answer_hits=answer_hits,
            doc_list=multi_paper_doc_list,
        )
        requested_count = extract_requested_paper_count(prompt_for_user or prompt)
        if selected_multi_paper_doc_list and (
            requested_count is None or len(selected_multi_paper_doc_list) == requested_count
        ):
            multi_paper_doc_list = selected_multi_paper_doc_list
        else:
            multi_paper_doc_list = _filter_multi_paper_doc_list_contract(
                prompt=prompt_for_user or prompt,
                doc_list=multi_paper_doc_list,
            )
    if (
        multi_paper_list_prompt
        and multi_paper_doc_list
        and (
            raw_answer_had_internal_doc_labels
            or _multi_paper_answer_needs_contract_rebuild(
                answer=answer,
                prompt=prompt_for_user or prompt,
            )
        )
    ):
        formatted_multi_paper_answer = _format_multi_paper_list_answer_v2(
            prompt=prompt_for_user or prompt,
            docs=multi_paper_doc_list,
        )
        if formatted_multi_paper_answer:
            answer = formatted_multi_paper_answer
    grounded_answer = str(answer or "")
    answer = _maybe_prepend_paper_guide_low_confidence_notice(
        answer,
        paper_guide_mode=bool(paper_guide_mode),
        prompt_text=prompt_for_user or prompt,
        prompt_family=sanitize_paper_guide_family,
        retrieval_confidence_hint=dict(paper_guide_retrieval_confidence_hint or {}),
        support_resolution=list(paper_guide_support_resolution or []),
        candidate_refs_by_source=dict(paper_guide_candidate_refs_effective or {}),
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
        candidate_refs_by_source=dict(paper_guide_candidate_refs_effective or {}),
        support_slots=list(paper_guide_support_slots or []),
        support_resolution=list(paper_guide_support_resolution or []),
        needs_supplement=bool(_PAPER_GUIDE_SUPPLEMENT_BLOCK_MARKER_RE.search(answer)),
        citation_validation=dict(citation_validation or {}),
        doc_list_contract=list(multi_paper_doc_list or []),
        paper_guide_contracts_seed=dict(paper_guide_contracts_seed or {}),
    )
    if research_answer_plan_norm:
        intent_contract = (
            dict(paper_guide_contracts.get("intent") or {})
            if isinstance(paper_guide_contracts.get("intent"), dict)
            else {}
        )
        intent_contract["research_answer_plan"] = research_answer_plan_norm
        paper_guide_contracts["intent"] = intent_contract
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
    requested_paper_count = extract_requested_paper_count(prompt_for_user or prompt)
    if requested_paper_count is not None:
        actual_paper_count = (
            1
            if single_paper_pick_prompt and str(answer or "").strip()
            else _count_multi_paper_answer_items(answer)
        )
        paper_count_ok = actual_paper_count == requested_paper_count
        answer_quality["requested_paper_count"] = requested_paper_count
        answer_quality["actual_paper_count"] = actual_paper_count
        answer_quality["paper_count_ok"] = paper_count_ok
        if not paper_count_ok:
            answer_quality["minimum_ok"] = False
    if research_answer_plan_norm:
        answer_quality["research_answer_plan"] = research_answer_plan_norm
    retrieval_confidence = dict(paper_guide_retrieval_confidence_hint or {})
    citation_plan = (
        dict((paper_guide_contracts_seed or {}).get("citation_plan") or {})
        if isinstance((paper_guide_contracts_seed or {}).get("citation_plan"), dict)
        else {}
    )
    if citation_plan:
        answer_quality["citation_plan"] = dict(citation_plan)
    if bool(template_repair_meta.get("changed")):
        answer_quality["template_repair"] = dict(template_repair_meta)
    if bool(retrieval_confidence.get("low_confidence")):
        refs_for_notice = _collect_low_confidence_candidate_refs(
            support_resolution=list(paper_guide_support_resolution or []),
            candidate_refs_by_source=dict(paper_guide_candidate_refs_effective or {}),
            retrieval_confidence_hint=retrieval_confidence,
            max_items=6,
        )
        if refs_for_notice:
            retrieval_confidence["candidate_refs_for_notice"] = list(refs_for_notice)
    if paper_guide_reference_opportunities:
        opportunity_refs = [
            int(item.get("ref_num") or 0)
            for item in paper_guide_reference_opportunities
            if isinstance(item, dict) and int(item.get("ref_num") or 0) > 0
        ]
        opportunity_ref_set = set(opportunity_refs)
        rendered_refs: list[int] = []
        for match in _CITE_CANON_RE.finditer(str(answer or "")):
            try:
                n = int(match.group(2) or 0)
            except Exception:
                n = 0
            if n > 0 and n in opportunity_ref_set and n not in rendered_refs:
                rendered_refs.append(n)
        answer_quality["reference_opportunities"] = {
            "count": int(len(paper_guide_reference_opportunities)),
            "rendered_count": int(len(rendered_refs)),
            "mode": str(paper_guide_reference_apply_meta.get("mode") or "none"),
            "injected_refs": list(paper_guide_reference_apply_meta.get("injected_refs") or []),
            "rendered_refs": list(rendered_refs),
            "refs": opportunity_refs,
        }
    if dict(citation_validation or {}).get("raw_count"):
        answer_quality["citation_validation"] = dict(citation_validation or {})
    answer_quality["retrieval_confidence"] = retrieval_confidence
    return {
        "answer": answer,
        "paper_guide_support_resolution": list(paper_guide_support_resolution or []),
        "paper_guide_contracts": paper_guide_contracts,
        "citation_validation": citation_validation,
        "answer_quality": answer_quality,
    }
