from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from kb.paper_guide_shared import _cite_source_id
from kb.reference_index import load_reference_index, resolve_reference_entry
from kb.source_blocks import normalize_inline_markdown

_CITE_CANON_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*(\d{1,4})\s*\]\]",
    re.IGNORECASE,
)
_INLINE_REF_RE = re.compile(
    r"(?<!\[)\[(\d{1,4}(?:\s*(?:-|–|—|,)\s*\d{1,4})*)\](?!\])"
)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}|\d{3,}|\w+")
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*\b")
_COMMON_LABELS = {
    "SCI",
    "SPI",
    "SNR",
    "CNN",
    "RNN",
    "PDF",
    "DMD",
    "SPAD",
    "NeRF",
    "SCIGS",
    "SCINeRF",
}
_LABEL_EXPANSIONS = {
    "admm": ("alternating direction method of multipliers",),
    "admm-net": ("deep tensor admm-net", "snapshot compressive imaging admm-net"),
    "pnp": ("plug and play", "plug-and-play"),
    "pnp-ffdnet": ("plug and play fast and flexible denoising", "plug-and-play fast and flexible denoising"),
    "fastdvdnet": ("fast deep video denoising network",),
    "stformer": ("spatial temporal transformer", "spatial-temporal transformer"),
}
_UPSTREAM_INTENT_RE = re.compile(
    r"(?i)\b(?:origin|source|prior|previous|existing|earlier|classic|baseline|"
    r"reference|citation|cite|cited|invented|new|original|background|comes?\s+from|"
    r"builds?\s+on|based\s+on|inspired\s+by)\b|"
    r"(?:\u6765\u6e90|\u51fa\u5904|\u6e90\u5934|\u4e4b\u524d|\u4ee5\u524d|\u5df2\u6709|"
    r"\u73b0\u6210|\u7ecf\u5178|\u80cc\u666f|\u81ea\u5df1|\u53d1\u660e|\u539f\u521b|"
    r"\u65b0\u4e1c\u897f|\u501f\u9274|\u5f15\u7528|\u53c2\u8003\u6587\u732e)"
)
_PRIOR_WORK_CUE_RE = re.compile(
    r"(?i)("
    r"\b(?:prior|previous|existing|earlier|classic|baseline|original|source|origin|"
    r"proposed|introduced|developed|based\s+on|built\s+on|inspired\s+by|derived\s+from|"
    r"extends?|adapted|uses?|employs?|cites?|reference[sd]?)\b|"
    r"(?:前人|已有|先前|早期|经典|来源|出处|源头|原创|自创|自己|发明|提出|引入|基于|借鉴|参考|引用|发展自|改进自)"
    r")"
)
_SKIP_FAMILIES = {"abstract", "doc_map"}
_OPPORTUNITY_NOTE_RE = re.compile(
    r"(?ims)"
    r"(?:^|\n\n)"
    r"(?:"
    r"如果想顺着论文的引用链继续追，可以优先打开：.*?。|"
    r"To follow the paper's citation trail, open: .*?\."
    r")"
    r"(?=\n\n|$)"
)


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def _compact_text(text: str, *, max_len: int = 360) -> str:
    s = normalize_inline_markdown(str(text or ""))
    s = _CITE_CANON_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip(" ,;:") + "..."


def _source_path_from_record(record: Mapping[str, object], *, fallback: str = "") -> str:
    for key in ("source_path", "sourcePath", "md_path", "path"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return str(fallback or "").strip()


def _sid_from_record(record: Mapping[str, object], *, source_path: str) -> str:
    sid = str(record.get("sid") or record.get("source_id") or "").strip()
    if sid:
        return sid
    return _cite_source_id(source_path) if source_path else ""


def _heading_from_record(record: Mapping[str, object]) -> str:
    for key in ("heading_path", "headingPath", "primary_heading_path", "heading", "section"):
        value = str(record.get(key) or "").strip()
        if value:
            return _compact_text(value, max_len=160)
    return ""


def _text_from_record(record: Mapping[str, object]) -> str:
    parts: list[str] = []
    for key in (
        "locate_anchor",
        "support_locate_anchor",
        "evidence_quote",
        "evidence_atom_text",
        "segment_text",
        "snippet",
        "cue",
        "text",
    ):
        value = str(record.get(key) or "").strip()
        if value:
            parts.append(value)
    for extra in list(record.get("deepread_texts") or [])[:1]:
        value = str(extra or "").strip()
        if value:
            parts.append(value)
    return _compact_text(" ".join(parts), max_len=520)


def _append_ref_num(bucket: list[int], value: object) -> None:
    try:
        n = int(value)
    except Exception:
        return
    if n > 0 and n not in bucket:
        bucket.append(n)


def _ref_nums_from_record(record: Mapping[str, object], *, text: str) -> list[int]:
    out: list[int] = []
    for key in ("resolved_ref_num", "ref_num", "reference_number"):
        _append_ref_num(out, record.get(key))
    for key in ("candidate_refs", "support_ref_candidates", "ref_nums", "inline_refs"):
        values = record.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for item in values:
                _append_ref_num(out, item)
    for span in list(record.get("ref_spans") or []):
        if not isinstance(span, Mapping):
            continue
        for item in list(span.get("nums") or []):
            _append_ref_num(out, item)
    for match in _INLINE_REF_RE.finditer(text):
        for item in re.split(r"\s*(?:-|–|—|,)\s*", str(match.group(1) or "")):
            _append_ref_num(out, item)
    return out[:6]


def _record_has_upstream_ref_signal(record: Mapping[str, object], *, text: str) -> bool:
    if _INLINE_REF_RE.search(str(text or "")):
        return True
    cite_policy = str(record.get("cite_policy") or "").strip().lower()
    if cite_policy == "prefer_ref":
        return True
    claim_type = str(record.get("claim_type") or "").strip().lower()
    has_candidate_refs = False
    for key in ("candidate_refs", "support_ref_candidates", "ref_nums", "inline_refs"):
        values = record.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and list(values):
            has_candidate_refs = True
            break
    if has_candidate_refs and claim_type in {"prior_work", "method_detail", "component_role"}:
        return True
    return False


def _tokens(text: str) -> set[str]:
    out = {
        token.lower()
        for token in _TOKEN_RE.findall(str(text or ""))
        if len(str(token or "").strip()) >= 3
    }
    return {tok for tok in out if tok not in {"this", "that", "paper", "method", "what", "with", "from"}}


def _hit_source_path(hit: Mapping[str, object]) -> str:
    meta = hit.get("meta") if isinstance(hit.get("meta"), Mapping) else {}
    return str((meta or {}).get("source_path") or "").strip()


def _hit_source_sha1(hit: Mapping[str, object]) -> str:
    meta = hit.get("meta") if isinstance(hit.get("meta"), Mapping) else {}
    return str((meta or {}).get("source_sha1") or "").strip().lower()


def _candidate_labels_from_text(*, prompt: str, answer: str = "", max_labels: int = 5) -> list[str]:
    surface = f"{prompt}\n{answer}"
    labels: list[str] = []
    seen: set[str] = set()
    for match in _ENTITY_RE.finditer(surface):
        label = str(match.group(0) or "").strip()
        if len(label) < 3 or len(label) > 48:
            continue
        if label in _COMMON_LABELS:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        labels.append(label)
        if len(labels) >= max(1, int(max_labels)):
            break
    return labels


def _reference_surface_for_match(ref: Mapping[str, object]) -> tuple[str, str]:
    title = str(ref.get("title") or "")
    raw = str(ref.get("raw") or ref.get("cite_fmt") or "")
    authors = str(ref.get("authors") or "")
    venue = str(ref.get("venue") or "")
    return title, " ".join([title, raw, authors, venue])


def _loose_ascii_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _loose_ascii_text(text: str) -> str:
    return " ".join(_loose_ascii_words(text))


def _expansions_for_label(label: str) -> tuple[str, ...]:
    key = _loose_ascii_text(label).replace(" ", "-")
    plain_key = key.replace("-", "")
    out: list[str] = []
    for lookup in (key, plain_key, str(label or "").strip().lower()):
        for item in _LABEL_EXPANSIONS.get(lookup, ()):
            if item and item not in out:
                out.append(item)
    return tuple(out)


def _acronym_matches_expansion(label: str, surface: str) -> bool:
    label_key = "".join(_loose_ascii_words(label)).upper()
    if len(label_key) < 3 or len(label_key) > 12:
        return False
    words = _loose_ascii_words(surface)
    if len(words) < len(label_key):
        return False
    stop = {"and", "of", "the", "for", "to", "via", "with", "in", "on", "a", "an"}
    initials = "".join(word[0].upper() for word in words if word not in stop and word)
    if label_key in initials:
        return True
    span_len = len(label_key) + 4
    for start in range(0, max(0, len(words) - 1)):
        span = words[start : start + span_len]
        if not span:
            continue
        span_initials = "".join(word[0].upper() for word in span if word not in stop and word)
        if span_initials.startswith(label_key):
            return True
    return False


def _score_reference_label_match(label: str, ref: Mapping[str, object]) -> float:
    title, surface = _reference_surface_for_match(ref)
    if not surface:
        return float("-inf")
    label_norm = str(label or "").strip()
    if not label_norm:
        return float("-inf")
    score = 0.0
    if _label_matches_surface(label_norm, title):
        score += 12.0
    if _label_matches_surface(label_norm, surface):
        score += 7.0
    label_tokens = _tokens(label_norm)
    title_tokens = _tokens(title)
    surface_tokens = _tokens(surface)
    if label_tokens:
        if label_tokens.issubset(title_tokens):
            score += 5.0
        elif label_tokens.issubset(surface_tokens):
            score += 2.5
    if label_norm.isupper() and len(label_norm) <= 12 and re.search(rf"(?<![A-Za-z0-9]){re.escape(label_norm)}(?![A-Za-z0-9-])", surface):
        score += 3.0
    title_loose = _loose_ascii_text(title)
    surface_loose = _loose_ascii_text(surface)
    for expansion in _expansions_for_label(label_norm):
        expansion_loose = _loose_ascii_text(expansion)
        if not expansion_loose:
            continue
        if expansion_loose in title_loose:
            score += 14.0
        elif expansion_loose in surface_loose:
            score += 9.0
    if label_norm.isupper() and _acronym_matches_expansion(label_norm, title):
        score += 10.0
    elif label_norm.isupper() and _acronym_matches_expansion(label_norm, surface):
        score += 6.0
    return score


def _find_reference_num_for_label(
    *,
    index_data: Mapping[str, object],
    source_path: str,
    source_sha1: str,
    label: str,
) -> tuple[int, dict[str, object]]:
    best_num = 0
    best_ref: dict[str, object] = {}
    best_score = float("-inf")
    for n in range(1, 501):
        try:
            got = resolve_reference_entry(dict(index_data or {}), source_path, n, source_sha1=source_sha1)
        except Exception:
            got = None
        ref = got.get("ref") if isinstance(got, Mapping) and isinstance(got.get("ref"), Mapping) else None
        if not isinstance(ref, Mapping):
            continue
        score = _score_reference_label_match(label, ref)
        if score > best_score:
            best_score = score
            best_num = int(n)
            best_ref = dict(ref)
    if best_score < 7.0:
        return 0, {}
    return best_num, best_ref


def detect_text_reference_opportunities(
    *,
    prompt: str,
    answer: str = "",
    answer_hits: Sequence[Mapping[str, object]] | None = None,
    db_dir: str | Path | None = None,
    max_items: int = 3,
) -> list[dict[str, object]]:
    """Find upstream bibliography refs for ordinary library Q&A.

    This is the non-paper-guide counterpart to support-slot opportunities:
    when the user's question names a method/concept and asks whether it is
    new, borrowed, prior work, or background, scan the references of retrieved
    source papers for that method/concept and expose a validated System B
    candidate. The final answer still has to discuss the matching label before
    a marker is injected.
    """

    prompt_text = str(prompt or "").strip()
    if not prompt_text or not _UPSTREAM_INTENT_RE.search(prompt_text):
        return []
    labels = _candidate_labels_from_text(prompt=prompt_text, answer=answer, max_labels=5)
    if not labels:
        return []
    try:
        index_data = load_reference_index(Path(db_dir).expanduser()) if db_dir else {}
    except Exception:
        index_data = {}
    if not isinstance(index_data, Mapping) or not index_data:
        return []

    rows: list[tuple[float, dict[str, object]]] = []
    seen_source: set[str] = set()
    for hit_index, hit in enumerate(list(answer_hits or [])[:6], start=1):
        if not isinstance(hit, Mapping):
            continue
        source_path = _hit_source_path(hit)
        if not source_path or source_path in seen_source:
            continue
        seen_source.add(source_path)
        source_sha1 = _hit_source_sha1(hit)
        sid = _cite_source_id(source_path)
        if not sid:
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), Mapping) else {}
        heading = _compact_text(
            str((meta or {}).get("ref_best_heading_path") or (meta or {}).get("heading_path") or ""),
            max_len=120,
        )
        evidence = _compact_text(
            str((((meta or {}).get("ref_show_snippets") or [None])[0]) or hit.get("text") or ""),
            max_len=240,
        )
        for label in labels:
            ref_num, ref = _find_reference_num_for_label(
                index_data=index_data,
                source_path=source_path,
                source_sha1=source_sha1,
                label=label,
            )
            if ref_num <= 0:
                continue
            title, surface = _reference_surface_for_match(ref)
            score = 12.0 - (0.25 * float(hit_index)) + _score_reference_label_match(label, ref)
            rows.append(
                (
                    score,
                    {
                        "source_path": source_path,
                        "sid": sid,
                        "ref_num": int(ref_num),
                        "label": label,
                        "heading_path": heading,
                        "evidence_quote": evidence or _compact_text(surface, max_len=240),
                        "why_line": "The retrieved paper's bibliography contains the upstream work named in this question.",
                        "ref_title": _compact_text(title, max_len=160),
                    },
                )
            )

    rows.sort(key=lambda item: item[0], reverse=True)
    out: list[dict[str, object]] = []
    seen: set[tuple[str, int, str]] = set()
    for _score, row in rows:
        key = (
            str(row.get("sid") or "").lower(),
            int(row.get("ref_num") or 0),
            str(row.get("label") or "").lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= max(1, int(max_items or 3)):
            break
    return out


def _label_matches_surface(label: str, surface: str) -> bool:
    needle = str(label or "").strip()
    hay = str(surface or "")
    if not needle or len(needle) < 3 or needle.lower().startswith("ref "):
        return False
    if _contains_cjk(needle):
        return needle in hay
    if not re.search(r"[A-Za-z0-9]", needle):
        return needle.lower() in hay.lower()
    # Avoid treating ADMM as a match for ADMM-Net.  System B refs often differ
    # by short method suffixes, so hyphen-aware boundaries matter.
    pattern = rf"(?<![A-Za-z0-9]){re.escape(needle)}(?![A-Za-z0-9-])"
    return bool(re.search(pattern, hay, flags=re.I))


def _line_can_take_prompt_bound_opportunity(*, line: str, prompt: str, label: str) -> bool:
    if not _label_matches_surface(label, prompt):
        return False
    plain = _compact_text(line, max_len=360)
    if not plain:
        return False
    if _PRIOR_WORK_CUE_RE.search(plain):
        return True
    return bool(
        re.search(
            r"(?i)\b(?:not\s+(?:new|original)|comes?\s+from|came\s+from|builds?\s+on|"
            r"background|citation|reference)\b|"
            r"(?:不是.{0,12}(?:原创|新提出)|来自|沿用|上游|前面|前人|已有|借鉴|参考)",
            plain,
        )
    )


def _label_for_opportunity(*, prompt: str, text: str, ref_num: int) -> str:
    local_ref = re.search(rf"\[\s*{int(ref_num)}\s*\]", str(text or ""))
    if local_ref:
        before = str(text or "")[max(0, local_ref.start() - 100) : local_ref.start()]
        local_entities = [
            item
            for item in _ENTITY_RE.findall(before)
            if len(item) >= 3 and item.lower() not in {"the", "this", "that", "most"}
        ]
        if local_entities:
            return local_entities[-1]
    prompt_entities = [m.group(0) for m in _ENTITY_RE.finditer(str(prompt or ""))]
    text_low = str(text or "").lower()
    for entity in prompt_entities:
        value = str(entity or "").strip()
        if len(value) >= 3 and value.lower() not in {"the", "this", "that"} and value.lower() in text_low:
            return entity
    for entity in _ENTITY_RE.findall(str(text or "")):
        if len(entity) >= 3 and entity.lower() not in {"the", "this"}:
            return entity
    return f"ref {int(ref_num)}"


def _score_record(*, prompt: str, answer: str, heading: str, text: str, refs: list[int], record: Mapping[str, object]) -> float:
    if not refs or not text:
        return float("-inf")
    score = 2.0 + min(3.0, 0.6 * float(len(refs)))
    prompt_tokens = _tokens(prompt)
    answer_tokens = _tokens(answer)
    text_tokens = _tokens(text)
    if prompt_tokens:
        score += min(5.0, 1.2 * float(len(prompt_tokens.intersection(text_tokens))))
    if answer_tokens:
        score += min(4.0, 0.8 * float(len(answer_tokens.intersection(text_tokens))))
    if _PRIOR_WORK_CUE_RE.search(text):
        score += 4.0
    if re.search(r"(?i)\b(?:related\s+work|background|introduction)\b|(?:相关工作|背景|引言)", heading):
        score += 1.8
    cite_policy = str(record.get("cite_policy") or "").strip().lower()
    if cite_policy == "prefer_ref":
        score += 2.0
    elif cite_policy == "locate_only":
        score -= 3.0
    claim_type = str(record.get("claim_type") or "").strip().lower()
    if claim_type in {"prior_work", "method_detail", "component_role"}:
        score += 1.5
    return score


def _iter_candidate_records(
    *,
    support_resolution: Sequence[Mapping[str, object]] | None,
    support_slots: Sequence[Mapping[str, object]] | None,
    cards: Sequence[Mapping[str, object]] | None,
) -> list[Mapping[str, object]]:
    out: list[Mapping[str, object]] = []
    for group in (support_resolution, support_slots, cards):
        for item in list(group or []):
            if isinstance(item, Mapping):
                out.append(item)
    return out


def detect_paper_guide_reference_opportunities(
    *,
    prompt: str,
    answer: str,
    prompt_family: str,
    source_path: str = "",
    support_resolution: Sequence[Mapping[str, object]] | None = None,
    support_slots: Sequence[Mapping[str, object]] | None = None,
    cards: Sequence[Mapping[str, object]] | None = None,
    max_items: int = 3,
) -> list[dict[str, object]]:
    """Find upstream bibliography refs that should surface in ordinary answers.

    The detector only trusts refs that are already attached to the current
    paper evidence, either as explicit inline markers in a paragraph or as
    candidate refs on a support slot/card.
    """

    family = str(prompt_family or "").strip().lower()
    if family in _SKIP_FAMILIES:
        return []
    try:
        limit = max(1, min(4, int(max_items)))
    except Exception:
        limit = 3

    rows: list[tuple[float, dict[str, object]]] = []
    for record in _iter_candidate_records(
        support_resolution=support_resolution,
        support_slots=support_slots,
        cards=cards,
    ):
        record_source = _source_path_from_record(record, fallback=source_path)
        if not record_source:
            continue
        text = _text_from_record(record)
        refs = _ref_nums_from_record(record, text=text)
        if not refs:
            continue
        if not _record_has_upstream_ref_signal(record, text=text):
            continue
        heading = _heading_from_record(record)
        score = _score_record(
            prompt=prompt,
            answer=answer,
            heading=heading,
            text=text,
            refs=refs,
            record=record,
        )
        if score < 4.0:
            continue
        sid = _sid_from_record(record, source_path=record_source)
        if not sid:
            continue
        for ref_num in refs[:3]:
            rows.append(
                (
                    score - (0.05 * len(rows)),
                    {
                        "source_path": record_source,
                        "sid": sid,
                        "ref_num": int(ref_num),
                        "label": _label_for_opportunity(prompt=prompt, text=text, ref_num=int(ref_num)),
                        "heading_path": heading,
                        "evidence_quote": text,
                        "why_line": (
                            "The current paper cites this upstream work in the evidence used for the answer."
                        ),
                    },
                )
            )

    rows.sort(key=lambda item: item[0], reverse=True)
    out: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for _score, row in rows:
        key = (str(row.get("sid") or "").lower(), int(row.get("ref_num") or 0))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= limit:
            break
    return out


def _normalized_opportunities(
    opportunities: Sequence[Mapping[str, object]] | None,
    *,
    max_items: int = 3,
) -> list[dict[str, object]]:
    try:
        limit = max(1, min(4, int(max_items)))
    except Exception:
        limit = 3
    out: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for item in list(opportunities or []):
        if not isinstance(item, Mapping):
            continue
        sid = str(item.get("sid") or "").strip()
        try:
            ref_num = int(item.get("ref_num") or 0)
        except Exception:
            ref_num = 0
        if not sid or ref_num <= 0:
            continue
        key = (sid.lower(), ref_num)
        if key in seen:
            continue
        seen.add(key)
        row = dict(item)
        row["sid"] = sid
        row["ref_num"] = ref_num
        out.append(row)
        if len(out) >= limit:
            break
    return out


def build_reference_opportunities_prompt_block(
    opportunities: Sequence[Mapping[str, object]] | None,
    *,
    max_items: int = 3,
) -> str:
    """Build a generation-time hint for natural System B citation placement."""

    rows = _normalized_opportunities(opportunities, max_items=max_items)
    if not rows:
        return ""
    lines = [
        "Upstream reference opportunities:",
        "- These are bibliography references explicitly attached to current-paper evidence.",
        "- For ordinary concept, origin, prior-work, or method-background questions, place the exact cite_example inline next to the sentence that explains that upstream work.",
        "- Do not dump these as a separate bibliography list unless the user asks for a reading list.",
        "- If the answer does not discuss the listed concept, do not force the citation.",
    ]
    for row in rows:
        sid = str(row.get("sid") or "").strip()
        ref_num = int(row.get("ref_num") or 0)
        label = _compact_text(str(row.get("label") or f"ref {ref_num}"), max_len=80)
        heading = _compact_text(str(row.get("heading_path") or ""), max_len=120)
        evidence = _compact_text(str(row.get("evidence_quote") or ""), max_len=180)
        parts = [f"label={label}", f"cite_example=[[CITE:{sid}:{ref_num}]]"]
        if heading:
            parts.append(f"heading={heading}")
        if evidence:
            parts.append(f"evidence={evidence}")
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines).strip()


_ANSWER_LINE_SKIP_RE = re.compile(
    r"^\s*(?:#{1,6}\s+|[-*]\s*$|```|\|+|(?:references?|bibliography|参考文献|引用)\s*[:：]?\s*$)",
    re.IGNORECASE,
)


def _cite_marker_for_opportunity(opp: Mapping[str, object]) -> str:
    return f"[[CITE:{str(opp.get('sid') or '').strip()}:{int(opp.get('ref_num') or 0)}]]"


def _line_score_for_opportunity(*, line: str, prompt: str, opp: Mapping[str, object]) -> float:
    plain = _compact_text(line, max_len=520)
    if len(plain) < 8 or _ANSWER_LINE_SKIP_RE.search(plain):
        return float("-inf")
    if _OPPORTUNITY_NOTE_RE.search(plain):
        return float("-inf")
    label = str(opp.get("label") or "").strip()
    evidence = str(opp.get("evidence_quote") or "").strip()
    score = 0.0
    label_matches = _label_matches_surface(label, plain)
    if label_matches:
        score += 8.0
    label_tokens = _tokens(label)
    evidence_tokens = _tokens(evidence)
    prompt_tokens = _tokens(prompt)
    line_tokens = _tokens(plain)
    meaningful_label = bool(label_tokens and not str(label or "").strip().lower().startswith("ref "))
    prompt_bound_fallback = _line_can_take_prompt_bound_opportunity(
        line=plain,
        prompt=prompt,
        label=label,
    )
    if meaningful_label and (not label_matches) and not label_tokens.intersection(line_tokens):
        if not prompt_bound_fallback:
            return float("-inf")
        score += 4.0
    if label_tokens:
        score += min(4.0, 2.0 * float(len(label_tokens.intersection(line_tokens))))
    if evidence_tokens:
        score += min(3.0, 0.8 * float(len(evidence_tokens.intersection(line_tokens))))
    if prompt_tokens:
        score += min(3.0, 0.8 * float(len(prompt_tokens.intersection(line_tokens))))
    if _PRIOR_WORK_CUE_RE.search(plain):
        score += 2.0
    return score


def _insert_marker_before_terminal_punctuation(line: str, marker: str) -> str:
    if marker in line:
        return line
    trailing_len = len(line) - len(line.rstrip())
    trailing = line[len(line) - trailing_len :] if trailing_len else ""
    body = line.rstrip()
    match = re.search(r"([。！？.!?；;]+)$", body)
    if match:
        return f"{body[: match.start()].rstrip()} {marker}{match.group(1)}{trailing}"
    return f"{body} {marker}{trailing}"


def inject_reference_opportunity_citations_inline(
    answer: str,
    *,
    prompt: str,
    opportunities: Sequence[Mapping[str, object]] | None,
    min_score: float = 6.0,
) -> tuple[str, dict[str, object]]:
    """Place validated upstream reference markers on the nearest answer sentence."""

    text = str(answer or "").strip()
    rows = _normalized_opportunities(opportunities, max_items=3)
    if not text or not rows:
        return text, {"mode": "none", "injected_refs": []}

    lines = text.splitlines()
    existing = {
        (str(m.group(1) or "").strip().lower(), int(m.group(2) or 0))
        for m in _CITE_CANON_RE.finditer(text)
    }
    injected_refs: list[int] = []
    for opp in rows:
        sid = str(opp.get("sid") or "").strip()
        ref_num = int(opp.get("ref_num") or 0)
        if not sid or ref_num <= 0 or (sid.lower(), ref_num) in existing:
            continue
        marker = _cite_marker_for_opportunity(opp)
        best_idx = -1
        best_score = float("-inf")
        for idx, line in enumerate(lines):
            score = _line_score_for_opportunity(line=line, prompt=prompt, opp=opp)
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx < 0 or best_score < float(min_score):
            continue
        lines[best_idx] = _insert_marker_before_terminal_punctuation(lines[best_idx], marker)
        injected_refs.append(ref_num)
        existing.add((sid.lower(), ref_num))

    if not injected_refs:
        mode = "already_present" if len(existing) > 0 else "none"
        return text, {"mode": mode, "injected_refs": []}
    return "\n".join(lines).strip(), {"mode": "inline", "injected_refs": injected_refs}


def apply_reference_opportunities_to_answer(
    answer: str,
    *,
    prompt: str,
    opportunities: Sequence[Mapping[str, object]] | None,
) -> tuple[str, dict[str, object]]:
    """Prefer natural inline System B cites; use the old tail note only as fallback."""

    text = str(answer or "").strip()
    rows = _normalized_opportunities(opportunities, max_items=3)
    if not text or not rows:
        return text, {"mode": "none", "injected_refs": [], "tail_used": False}

    inline_text, inline_meta = inject_reference_opportunity_citations_inline(
        text,
        prompt=prompt,
        opportunities=rows,
    )
    if inline_text != text:
        meta = dict(inline_meta)
        meta["tail_used"] = False
        return inline_text, meta

    appended = append_reference_opportunity_note(text, prompt=prompt, opportunities=rows)
    if appended != text:
        return appended, {
            "mode": "tail",
            "injected_refs": [],
            "tail_refs": [int(row.get("ref_num") or 0) for row in rows],
            "tail_used": True,
        }
    return text, {"mode": str(inline_meta.get("mode") or "none"), "injected_refs": [], "tail_used": False}


def merge_reference_opportunity_candidate_refs(
    candidate_refs_by_source: Mapping[str, Sequence[int]] | None,
    opportunities: Sequence[Mapping[str, object]] | None,
) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for src, nums in dict(candidate_refs_by_source or {}).items():
        src_key = str(src or "").strip()
        if not src_key:
            continue
        bucket: list[int] = []
        for item in list(nums or []):
            _append_ref_num(bucket, item)
        if bucket:
            out[src_key] = bucket
    for opp in list(opportunities or []):
        if not isinstance(opp, Mapping):
            continue
        src = str(opp.get("source_path") or "").strip()
        if not src:
            continue
        bucket = out.setdefault(src, [])
        _append_ref_num(bucket, opp.get("ref_num"))
    return out


def append_reference_opportunity_note(
    answer: str,
    *,
    prompt: str,
    opportunities: Sequence[Mapping[str, object]] | None,
) -> str:
    text = str(answer or "").strip()
    if not text:
        return text
    opps = [dict(item) for item in list(opportunities or []) if isinstance(item, Mapping)]
    if not opps:
        return text
    existing = {
        (str(m.group(1) or "").strip().lower(), int(m.group(2) or 0))
        for m in _CITE_CANON_RE.finditer(text)
    }
    parts: list[str] = []
    for opp in opps:
        sid = str(opp.get("sid") or "").strip()
        try:
            ref_num = int(opp.get("ref_num") or 0)
        except Exception:
            ref_num = 0
        if not sid or ref_num <= 0 or (sid.lower(), ref_num) in existing:
            continue
        label = str(opp.get("label") or "").strip() or f"ref {ref_num}"
        parts.append(f"{label} [[CITE:{sid}:{ref_num}]]")
        existing.add((sid.lower(), ref_num))
        if len(parts) >= 3:
            break
    if not parts:
        return text
    prefer_zh = _contains_cjk(prompt) or _contains_cjk(text)
    if prefer_zh:
        note = "如果想顺着论文的引用链继续追，可以优先打开：" + "、".join(parts) + "。"
    else:
        note = "To follow the paper's citation trail, open: " + ", ".join(parts) + "."
    if note in text:
        return text
    return f"{text}\n\n{note}".strip()


def strip_reference_opportunity_note(answer: str) -> str:
    text = str(answer or "").strip()
    if not text:
        return text
    out = _OPPORTUNITY_NOTE_RE.sub("", text)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out
