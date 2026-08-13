from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from kb.inpaper_citation_grounding import parse_ref_num_set
from kb.paper_guide_structured_index_runtime import load_paper_guide_reference_index
from kb.paper_guide_shared import _cite_source_id
from kb.reference_index import load_reference_index, resolve_reference_entry
from kb.source_blocks import normalize_inline_markdown

_CITE_CANON_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*(\d{1,4})\s*\]\]",
    re.IGNORECASE,
)
_INLINE_REF_RE = re.compile(
    r"(?<!\[)\[(\d{1,4}(?:\s*(?:[-\u2013\u2014\u2212,])\s*\d{1,4})*)\](?!\])"
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
_ALLOW_COMMON_LABELS_FOR_OPPORTUNITY = {"SCI", "SPI", "SPAD", "NeRF", "SCIGS", "SCINeRF"}
_DOMAIN_LABEL_PATTERNS = (
    ("single-pixel imaging", r"(?i)\bsingle[-\s]?pixel imaging\b|\u5355\u50cf\u7d20\u6210\u50cf"),
    (
        "snapshot compressive imaging",
        r"(?i)\bsnapshot compressive imaging\b|\bSCI\b|\u538b\u7f29\u5feb\u7167\u6210\u50cf|\u5feb\u7167\u538b\u7f29\u6210\u50cf",
    ),
    ("physics-informed deep learning", r"(?i)\bphysics[-\s]?informed deep learning\b"),
    ("SPAD", r"(?i)\bSPADs?\b|single[-\s]?photon avalanche diode"),
    ("Structured detection", r"(?i)\bstructured detection\b|\bs2ISM\b|结构检测"),
    ("Interferometric", r"(?i)\binterferometric\b|\biSCAT\b|干涉(?:散射|检测)?"),
    ("Light-field", r"(?i)\blight[-\s]?field\b|\bplenoptic\b|光场"),
    ("ISM", r"(?i)\bimage scanning microscopy\b|\bISM\b"),
    ("dark count", r"(?i)\bdark count(?: rate)?\b|暗计数"),
    ("afterpulsing", r"(?i)\bafterpuls(?:e|ing)\b|后脉冲"),
    ("crosstalk", r"(?i)\bcrosstalk\b|串扰"),
    ("dead time", r"(?i)\bdead time\b|死时间"),
    ("Hadamard", r"(?i)\bHadamard\b"),
    ("Fourier", r"(?i)\bFourier\b"),
    ("Cosine transform", r"(?i)\bCosine transform\b|\bDCT\b"),
    ("Orbital Angular Momentum", r"(?i)\bOrbital Angular Momentum\b|\bOAM\b"),
    ("CASSI", r"(?i)\bCASSI\b|coded aperture snapshot spectral imaging"),
    ("NeRF", r"(?i)\bNeRF\b|neural radiance fields?"),
    ("3D Gaussian", r"(?i)\b3DGS\b|3D Gaussian|Gaussian splatting"),
    ("PILN", r"(?i)\bPILN\b|part-based image-loop network|image-loop network"),
)
_CONCRETE_DOMAIN_LABELS = {
    label
    for label, _pattern in _DOMAIN_LABEL_PATTERNS
    if label not in {"single-pixel imaging", "snapshot compressive imaging"}
}
_BROAD_REFERENCE_LABELS = {
    "single pixel imaging",
    "snapshot compressive imaging",
    "deep learning",
    "neural network",
}
_LOW_VALUE_REFERENCE_LABELS = {
    "fig",
    "figure",
    "table",
    "results",
    "methods",
    "method",
}
_GENERIC_SYNTHESIS_LINE_RE = re.compile(
    r"(?i)\b(?:brings?|offers?|has|provides?)\s+(?:significant\s+)?(?:benefits?|advantages?)\b.{0,80}\b(?:challenges?|risks?|limitations?)\b|"
    r"(?:\u5e26\u6765.{0,16}(?:\u597d\u5904|\u4f18\u52bf|\u63d0\u5347).{0,40}(?:\u6311\u6218|\u98ce\u9669|\u5c40\u9650|\u5751)|"
    r"\u603b\u7684\u6765\u8bf4|\u5177\u4f53\u6765\u8bf4)"
)
_SEMANTIC_SUPPORT_PATTERNS = (
    ("single_pixel_imaging", r"(?i)\bsingle[-\s]?pixel imaging\b|\bSPI\b|\u5355\u50cf\u7d20\u6210\u50cf"),
    ("snapshot_compressive_imaging", r"(?i)\bsnapshot compressive imaging\b|\bSCI\b|\u538b\u7f29\u5feb\u7167|\u5feb\u7167\u538b\u7f29"),
    ("deep_learning", r"(?i)\bdeep learning\b|\bDNNs?\b|\bneural networks?\b|\u6df1\u5ea6\u5b66\u4e60|\u795e\u7ecf\u7f51\u7edc"),
    ("reconstruction", r"(?i)\breconstruction\b|\breconstruct\b|\u91cd\u5efa"),
    ("speed", r"(?i)\bspeed\b|\breal[-\s]?time\b|\bfast\b|\u901f\u5ea6|\u5feb\u901f|\u5b9e\u65f6"),
    ("quality", r"(?i)\bquality\b|\bhigh[-\s]?quality\b|\u8d28\u91cf|\u9ad8\u8d28\u91cf"),
    ("sampling", r"(?i)\bsampling\b|\bsampling rate\b|\u91c7\u6837|\u91c7\u6837\u7387"),
    ("basis_patterns", r"(?i)\bbasis patterns?\b|\bHadamard\b|\bFourier\b|\bCosine transform\b|\bOAM\b|\bOrbital Angular Momentum\b"),
    ("hadamard", r"(?i)\bHadamard\b"),
    ("fourier", r"(?i)\bFourier\b"),
    ("oam", r"(?i)\bOAM\b|\bOrbital Angular Momentum\b"),
    ("structured_detection", r"(?i)\bstructured detection\b|\bs2ISM\b|结构检测"),
    ("image_scanning_microscopy", r"(?i)\bimage scanning microscopy\b|\bISM\b|\bAiryscan\b"),
    ("interferometric", r"(?i)\binterferometric\b|\biSCAT\b|干涉(?:散射|检测)?"),
    ("light_field", r"(?i)\blight[-\s]?field\b|\bplenoptic\b|光场"),
    ("depth_of_field", r"(?i)\bdepth of field\b|\bDOF\b|景深"),
    ("dark_count", r"(?i)\bdark count(?: rate)?\b|暗计数"),
    ("afterpulsing", r"(?i)\bafterpuls(?:e|ing)\b|后脉冲"),
    ("crosstalk", r"(?i)\bcrosstalk\b|串扰"),
    ("dead_time", r"(?i)\bdead time\b|死时间"),
    ("noise", r"(?i)\bnoise\b|\bdenois(?:e|ing)\b|\u566a\u58f0|\u53bb\u566a"),
    ("spad", r"(?i)\bSPADs?\b|single[-\s]?photon avalanche diode|\u5355\u5149\u5b50"),
    ("super_resolution", r"(?i)\bsuper[-\s]?resolution\b|\u8d85\u5206\u8fa8"),
    ("generalization", r"(?i)\bgenerali[sz]ation\b|\brobustness\b|\bunseen\b|\u6cdb\u5316|\u9c81\u68d2"),
    ("data", r"(?i)\bdatasets?\b|\btraining data\b|\u6570\u636e\u96c6|\u8bad\u7ec3\u6570\u636e"),
    ("physical_model", r"(?i)\bphysical model\b|\bphysics[-\s]?informed\b|\u7269\u7406\u6a21\u578b|\u7269\u7406\u4fe1\u606f"),
)
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
    r"reference|citation|cite|cited|invent(?:ed)?|upstream|new|original|background|comes?\s+from|"
    r"builds?\s+on|based\s+on|inspired\s+by)\b|"
    r"(?:\u6765\u6e90|\u51fa\u5904|\u6e90\u5934|\u4e4b\u524d|\u4ee5\u524d|\u5df2\u6709|"
    r"\u73b0\u6210|\u7ecf\u5178|\u80cc\u666f|\u81ea\u5df1|\u53d1\u660e|\u539f\u521b|"
    r"\u65b0\u4e1c\u897f|\u501f\u9274|\u5f15\u7528|\u53c2\u8003\u6587\u732e)"
)
_RESEARCH_READING_TRACE_RE = re.compile(
    r"(?i)\b(?:roadmap|lineage|reading\s+route|reading\s+order|how\s+to\s+read|"
    r"relationship|relate|position|background|context|pair|connect)\b|"
    r"(?:\u8bfb\u4e66\u8def\u7ebf|\u9605\u8bfb\u8def\u7ebf|\u5148\u8bfb|\u600e\u4e48\u8bfb|"
    r"\u642d\u914d\u8bfb|\u5efa\u7acb\u4e3b\u7ebf|\u4e3b\u7ebf|\u8109\u7edc|\u8fd9\u6761\u7ebf|"
    r"\u4ece.{0,40}\u5230|\u5173\u7cfb|\u5206\u522b|\u9002\u5408\u89e3\u51b3)"
)
_REFERENCE_TAIL_INTENT_RE = re.compile(
    r"(?i)\b(?:origin|source|upstream|citation\s+trail|reference\s+trail|reading\s+route|"
    r"reading\s+order|lineage|evolution|trajectory|roadmap|from\s+.+?\s+to|where\s+did|come\s+from|came\s+from|prior\s+work|"
    r"previous\s+work|who\s+proposed|who\s+introduced|invented|borrowed|inspired\s+by)\b|"
    r"(?:\u6765\u6e90|\u51fa\u5904|\u6e90\u5934|\u4e0a\u6e38|\u5f15\u7528\u94fe|"
    r"\u53c2\u8003\u6587\u732e|\u4ece\u54ea|\u600e\u4e48\u6765|\u8c01\u63d0\u51fa|"
    r"\u8c01\u53d1\u660e|\u539f\u521b|\u501f\u9274|\u8bfb\u4e66\u8def\u7ebf|"
    r"\u9605\u8bfb\u8def\u7ebf|\u5148\u8bfb|\u600e\u4e48\u8bfb|\u642d\u914d\u8bfb|"
    r"\u4e3b\u7ebf|\u8109\u7edc|\u6f14\u8fdb|\u8fdb\u5c55|\u53d1\u5c55|\u8def\u7ebf|"
    r"\u4ece.{0,60}\u5230|\u524d\u4eba|\u5df2\u6709|\u5148\u524d|\u4e4b\u524d)"
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


def _prompt_allows_reference_tail_note(prompt: str) -> bool:
    """Tail notes are only for explicit upstream/reading-route questions."""

    return bool(_REFERENCE_TAIL_INTENT_RE.search(str(prompt or "")))


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


def _inline_ref_nums_from_text(text: str, *, limit: int = 12) -> list[int]:
    out: list[int] = []
    for match in _INLINE_REF_RE.finditer(str(text or "")):
        for item in parse_ref_num_set(str(match.group(1) or ""), max_items=limit):
            _append_ref_num(out, item)
            if len(out) >= max(1, int(limit)):
                return out
    return out


def _explicit_ref_nums_from_record(record: Mapping[str, object], *, text: str) -> list[int]:
    """Refs that are explicitly present in the evidence sentence itself.

    Candidate refs can be useful for validation, but they are too broad for
    System B surfacing: a System B card should mean that the current paper
    evidence actually contains "[n]" (or an equivalent ref_span parsed from
    that same sentence/clause), not merely that a bibliography entry matched a
    label.
    """

    out: list[int] = []
    for item in _inline_ref_nums_from_text(text):
        _append_ref_num(out, item)
    for span in list(record.get("ref_spans") or []):
        if not isinstance(span, Mapping):
            continue
        scope = str(span.get("scope") or "").strip().lower()
        if scope in {"reference_entry", "bibliography", "references"}:
            continue
        span_text = str(span.get("text") or span.get("surface") or span.get("snippet") or "").strip()
        span_nums = [int(n) for n in list(span.get("nums") or []) if str(n).strip().isdigit() and int(n) > 0]
        if not span_nums:
            continue
        inline_nums = _inline_ref_nums_from_text(span_text) if span_text else []
        if inline_nums:
            for item in inline_nums:
                if item in span_nums:
                    _append_ref_num(out, item)
            continue
        if scope in {"same_sentence", "same_clause", "context_explicit_ref"}:
            for item in span_nums:
                _append_ref_num(out, item)
    return out[:6]


def _prioritize_explicit_refs_for_prompt(*, prompt: str, text: str, refs: Sequence[int]) -> list[int]:
    """Rank same-paragraph refs by the sentence that matches the user's target.

    Related-work paragraphs often enumerate broad method families first and only
    later attach the reference for the named method. Keeping raw appearance order
    can therefore surface an unrelated early ref and drop the relevant one.
    """

    ordered = [int(item) for item in refs if int(item) > 0]
    if len(ordered) <= 1:
        return ordered
    prompt_text = str(prompt or "").strip()
    labels = _candidate_labels_from_text(prompt=prompt_text, answer="", max_labels=8)
    prompt_tokens = _tokens(prompt_text)
    contexts: dict[int, list[str]] = {ref_num: [] for ref_num in ordered}
    for segment in re.split(r"(?<=[.!?。！？;；])\s+|\n+", str(text or "")):
        segment_text = str(segment or "").strip()
        if not segment_text:
            continue
        segment_refs = set(_inline_ref_nums_from_text(segment_text, limit=24))
        for ref_num in ordered:
            if ref_num in segment_refs:
                contexts.setdefault(ref_num, []).append(segment_text)

    def _score(ref_num: int) -> tuple[float, int]:
        best = 0.0
        for context in contexts.get(ref_num, []):
            context_tokens = _tokens(context)
            shared = len(prompt_tokens.intersection(context_tokens))
            label_hits = sum(1 for label in labels if _label_matches_surface(label, context))
            best = max(best, (12.0 * float(label_hits)) + (1.2 * float(shared)))
        return best, -ordered.index(ref_num)

    return sorted(ordered, key=_score, reverse=True)


def _explicit_ref_contexts_from_text(text: str, *, max_contexts: int = 12) -> list[tuple[int, str]]:
    src = str(text or "")
    if not src.strip():
        return []
    out: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for match in _INLINE_REF_RE.finditer(src):
        start = max(0, int(match.start()) - 220)
        end = min(len(src), int(match.end()) + 220)
        context = _compact_text(src[start:end], max_len=360)
        if not context:
            continue
        for ref_num in parse_ref_num_set(str(match.group(1) or ""), max_items=16):
            try:
                n = int(ref_num)
            except Exception:
                continue
            if n <= 0:
                continue
            key = (n, context)
            if key in seen:
                continue
            seen.add(key)
            out.append((n, context))
            if len(out) >= max(1, int(max_contexts)):
                return out
    return out


_AUTHOR_YEAR_CITATION_RE = re.compile(
    r"(?:\(\s*|;\s*)(?P<surname>[A-Z][A-Za-z'\u2019-]{2,})"
    r"(?:\s+et\s+al\.)?\s*,\s*(?P<year>(?:19|20)\d{2})",
    re.IGNORECASE,
)


def _author_year_ref_contexts_from_text(
    text: str,
    *,
    reference_rows: Sequence[Mapping[str, object]],
    target_refs: set[int],
    max_contexts: int = 12,
) -> list[tuple[int, str]]:
    """Resolve verified ``(Surname, year)`` body markers to structured refs."""

    source = str(text or "")
    if not source.strip() or not target_refs:
        return []
    candidates: list[tuple[int, str, str]] = []
    for row in list(reference_rows or []):
        if not isinstance(row, Mapping):
            continue
        try:
            ref_num = int(row.get("ref_num") or 0)
        except Exception:
            ref_num = 0
        if ref_num not in target_refs:
            continue
        raw = str(row.get("text") or row.get("raw") or "").strip()
        year = str(row.get("year") or "").strip()
        if raw and year:
            candidates.append((ref_num, year, raw))

    out: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for marker in _AUTHOR_YEAR_CITATION_RE.finditer(source):
        surname = str(marker.group("surname") or "").strip()
        year = str(marker.group("year") or "").strip()
        if not surname or not year:
            continue
        left = max(
            source.rfind("\n", 0, marker.start()),
            source.rfind(".", 0, marker.start()),
            source.rfind(";", 0, marker.start()),
        )
        right_candidates = [
            index
            for index in (
                source.find("\n", marker.end()),
                source.find(".", marker.end()),
                source.find(";", marker.end()),
            )
            if index >= 0
        ]
        right = min(right_candidates) + 1 if right_candidates else min(len(source), marker.end() + 260)
        context = _compact_text(source[left + 1 : right], max_len=520)
        if len(context) < 12:
            continue
        for ref_num, ref_year, raw in candidates:
            if ref_year != year or not re.search(
                rf"(?<![A-Za-z]){re.escape(surname)}(?![A-Za-z])",
                raw,
                flags=re.IGNORECASE,
            ):
                continue
            key = (ref_num, context.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append((ref_num, context))
            if len(out) >= max(1, int(max_contexts)):
                return out
    return out


def _source_body_text(source_path: str, *, max_chars: int = 1_200_000) -> str:
    path = Path(str(source_path or "")).expanduser()
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if not text:
        return ""
    text = text[: max(1, int(max_chars))]
    parts = re.split(r"(?im)^\s*#{1,6}\s+(?:references|bibliography)\b", text, maxsplit=1)
    return str(parts[0] if parts else text)


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
    return bool(_explicit_ref_nums_from_record(record, text=text))


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

    def _push(label: str) -> bool:
        value = str(label or "").strip()
        if len(value) < 3 or len(value) > 64:
            return False
        key = value.lower()
        if key in seen:
            return False
        seen.add(key)
        labels.append(value)
        return len(labels) >= max(1, int(max_labels))

    for label, pattern in _DOMAIN_LABEL_PATTERNS:
        if re.search(pattern, surface):
            if _push(label):
                return labels

    for match in _ENTITY_RE.finditer(surface):
        label = str(match.group(0) or "").strip()
        if len(label) < 3 or len(label) > 48:
            continue
        if label in _COMMON_LABELS and label not in _ALLOW_COMMON_LABELS_FOR_OPPORTUNITY:
            continue
        if _push(label):
            break
    return labels


def _reference_surface_for_match(ref: Mapping[str, object]) -> tuple[str, str]:
    title = str(ref.get("title") or "")
    raw = str(ref.get("raw") or ref.get("cite_fmt") or "")
    authors = str(ref.get("authors") or "")
    venue = str(ref.get("venue") or "")
    return title, " ".join([title, raw, authors, venue])


def _structured_ref_to_ref(row: Mapping[str, object]) -> dict[str, object]:
    ref: dict[str, object] = {}
    for src_key, dst_key in (
        ("title", "title"),
        ("text", "raw"),
        ("authors", "authors"),
        ("venue", "venue"),
        ("year", "year"),
        ("doi", "doi"),
    ):
        value = str(row.get(src_key) or "").strip()
        if value:
            ref[dst_key] = value
    return ref


def _loose_ascii_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _loose_ascii_text(text: str) -> str:
    return " ".join(_loose_ascii_words(text))


def _semantic_support_tags(text: str) -> set[str]:
    src = str(text or "")
    if not src:
        return set()
    return {tag for tag, pattern in _SEMANTIC_SUPPORT_PATTERNS if re.search(pattern, src)}


def _is_broad_reference_label(label: str) -> bool:
    key = _loose_ascii_text(label)
    if key in _BROAD_REFERENCE_LABELS:
        return True
    if str(label or "").strip().upper() in {"SCI", "SPI"}:
        return True
    return False


def _is_low_value_reference_label(label: str) -> bool:
    key = _loose_ascii_text(label)
    return (not key) or key in _LOW_VALUE_REFERENCE_LABELS


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
    label_loose = _loose_ascii_text(label_norm)
    if label_loose and title_loose == label_loose:
        score += 30.0
    elif label_loose and title_loose.startswith(label_loose + " "):
        score += 20.0
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


def _reference_label_is_relevant(
    *,
    label: str,
    ref: Mapping[str, object],
    context: str,
) -> bool:
    """Reject broad labels that only match the answer topic, not the cited work."""

    label_text = str(label or "").strip()
    if (not label_text) or _is_low_value_reference_label(label_text):
        return False
    try:
        label_score = float(_score_reference_label_match(label_text, ref))
    except Exception:
        label_score = float("-inf")
    if label_score >= 7.0:
        return True
    if not _is_broad_reference_label(label_text):
        return label_score >= 3.0 or _label_matches_surface(label_text, context)

    title, surface = _reference_surface_for_match(ref)
    ref_tags = _semantic_support_tags(f"{title}\n{surface}")
    label_tags = _semantic_support_tags(label_text)
    if label_tags and ref_tags.intersection(label_tags):
        return True
    return False


def _specific_label_from_reference_surface(
    *,
    ref: Mapping[str, object],
    context: str,
    focus_surface: str,
) -> str:
    title, surface = _reference_surface_for_match(ref)
    ref_surface = f"{title}\n{surface}"
    for label, pattern in _DOMAIN_LABEL_PATTERNS:
        if label not in _CONCRETE_DOMAIN_LABELS:
            continue
        if not re.search(pattern, ref_surface):
            continue
        if _label_matches_surface(label, focus_surface) or _label_matches_surface(label, context):
            return label
    return ""


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


def _source_key(path: object) -> str:
    return str(path or "").strip().replace("\\", "/").lower()


def _cited_source_keys_from_answer(
    answer: str,
    answer_hits: Sequence[Mapping[str, object]] | None,
) -> set[str]:
    hits = [hit for hit in list(answer_hits or []) if isinstance(hit, Mapping)]
    if not hits:
        return set()
    text = _CITE_CANON_RE.sub("", str(answer or ""))
    out: set[str] = set()
    for match in _INLINE_REF_RE.finditer(text):
        for item in parse_ref_num_set(str(match.group(1) or ""), max_items=24):
            try:
                idx = int(item)
            except Exception:
                continue
            if idx <= 0 or idx > len(hits):
                continue
            key = _source_key(_hit_source_path(hits[idx - 1]))
            if key:
                out.add(key)
    return out


def detect_text_reference_opportunities(
    *,
    prompt: str,
    answer: str = "",
    answer_hits: Sequence[Mapping[str, object]] | None = None,
    db_dir: str | Path | None = None,
    max_items: int = 3,
) -> list[dict[str, object]]:
    """Find grounded upstream refs for ordinary library Q&A.

    Ordinary Q&A does not have paper-guide support slots, so use each source
    paper's structured reference index instead.  A candidate is trusted only
    when that index contains a citation context from the current paper body
    and the context explicitly contains the same "[n]" marker.
    """

    prompt_text = str(prompt or "").strip()
    if not prompt_text:
        return []
    answer_text = str(answer or "").strip()
    focus_surface = f"{prompt_text}\n{answer_text}"
    focus_tokens = _tokens(focus_surface)
    if not focus_tokens:
        return []
    prompt_labels = _candidate_labels_from_text(prompt=prompt_text, answer="", max_labels=6)
    labels = _candidate_labels_from_text(prompt=prompt_text, answer=answer_text, max_labels=6)
    explicit_upstream_intent = bool(_UPSTREAM_INTENT_RE.search(prompt_text))
    strong_trace_intent = bool(
        explicit_upstream_intent
        or _RESEARCH_READING_TRACE_RE.search(prompt_text)
        or re.search(
            r"(?i)\b(?:lineage|roadmap|background|prior\s+work|from\s+.+\s+to)\b|"
            r"(?:\u4e3b\u7ebf|\u8109\u7edc|\u8def\u7ebf|\u4e0a\u6e38|\u524d\u4eba|\u4ece.+\u5230)",
            prompt_text,
        )
    )
    try:
        index_data = load_reference_index(Path(db_dir).expanduser()) if db_dir else {}
    except Exception:
        index_data = {}

    cited_source_keys = _cited_source_keys_from_answer(answer_text, answer_hits)
    rows: list[tuple[float, dict[str, object]]] = []
    seen_source: set[str] = set()
    for hit_index, hit in enumerate(list(answer_hits or [])[:6], start=1):
        if not isinstance(hit, Mapping):
            continue
        source_path = _hit_source_path(hit)
        source_key = _source_key(source_path)
        if not source_path or not source_key or source_key in seen_source:
            continue
        if cited_source_keys and source_key not in cited_source_keys:
            continue
        seen_source.add(source_key)
        source_sha1 = _hit_source_sha1(hit)
        sid = _cite_source_id(source_path)
        if not sid:
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), Mapping) else {}
        source_sha1 = _hit_source_sha1(hit)
        hit_surface = " ".join(
            [
                str(hit.get("text") or ""),
                " ".join(str(x or "") for x in list((meta or {}).get("ref_show_snippets") or [])[:2]),
                " ".join(str(x or "") for x in list((meta or {}).get("ref_snippets") or [])[:2]),
            ]
        )

        def _push_context_candidate(
            *,
            ref_num: int,
            context: str,
            heading: str = "",
            seed_ref: Mapping[str, object] | None = None,
            score_bonus: float = 0.0,
        ) -> None:
            try:
                n = int(ref_num or 0)
            except Exception:
                n = 0
            if n <= 0:
                return
            context_text = _compact_text(str(context or "").strip(), max_len=360)
            if not context_text or n not in _inline_ref_nums_from_text(context_text):
                return
            ref = dict(seed_ref or {})
            # Per-paper ``reference_index.json`` rows already contain the
            # bibliography fields needed by the opportunity scorer. Resolving
            # every one of those rows again through the global index can fall
            # back to a full document scan when paths moved across machines;
            # on a six-paper origin query that added several seconds before
            # the answer request even started. Keep the global lookup only for
            # inline-hit candidates that do not carry structured metadata.
            if not any(str(ref.get(key) or "").strip() for key in ("title", "raw", "doi")):
                try:
                    resolved = resolve_reference_entry(
                        dict(index_data or {}),
                        source_path,
                        n,
                        source_sha1=source_sha1,
                    )
                except Exception:
                    resolved = None
                if isinstance(resolved, Mapping) and isinstance(resolved.get("ref"), Mapping):
                    ref.update({k: v for k, v in dict(resolved.get("ref") or {}).items() if v})
            title, surface = _reference_surface_for_match(ref)
            label = _label_for_opportunity(prompt=prompt_text, text=context_text, ref_num=n)
            active_labels = list(prompt_labels if (explicit_upstream_intent and prompt_labels) else labels)
            if explicit_upstream_intent and prompt_labels:
                explicit_match = False
                for candidate in prompt_labels:
                    cand = str(candidate or "").strip()
                    if not cand:
                        continue
                    try:
                        cand_score = float(_score_reference_label_match(cand, ref))
                    except Exception:
                        cand_score = float("-inf")
                    if cand_score >= 7.0 or _label_matches_surface(cand, label):
                        explicit_match = True
                        break
                if not explicit_match:
                    return
            best_focus_label = ""
            best_focus_score = float("-inf")
            for candidate in active_labels:
                cand = str(candidate or "").strip()
                if not cand or not _label_matches_surface(cand, focus_surface):
                    continue
                try:
                    cand_score = float(_score_reference_label_match(cand, ref))
                except Exception:
                    cand_score = float("-inf")
                if cand_score > best_focus_score:
                    best_focus_score = cand_score
                    best_focus_label = cand
            try:
                current_label_score = float(_score_reference_label_match(label, ref))
            except Exception:
                current_label_score = float("-inf")
            if (
                best_focus_label
                and best_focus_score >= 7.0
                and (
                    _is_broad_reference_label(label)
                    or (current_label_score < 7.0 and label not in _CONCRETE_DOMAIN_LABELS)
                )
            ):
                label = best_focus_label
            if _is_broad_reference_label(label):
                specific_label = _specific_label_from_reference_surface(
                    ref=ref,
                    context=context_text,
                    focus_surface=focus_surface,
                )
                if specific_label:
                    label = specific_label
            if not _reference_label_is_relevant(label=label, ref=ref, context=context_text):
                return
            combined = f"{context_text}\n{title}\n{surface}"
            combined_tokens = _tokens(combined)
            shared_focus = focus_tokens.intersection(combined_tokens)
            label_matches_prompt = _label_matches_surface(label, prompt_text)
            label_matches = bool(
                label_matches_prompt
                or _label_matches_surface(label, answer_text)
                or any(_label_matches_surface(candidate, combined) for candidate in active_labels)
            )
            if len(shared_focus) < 2 and not label_matches:
                return
            if not strong_trace_intent and len(shared_focus) < 3 and not label_matches:
                return
            hit_overlap = len(_tokens(hit_surface).intersection(combined_tokens))
            reference_focus_score = max(float(current_label_score), float(best_focus_score))
            score = (
                8.0
                - (0.25 * float(hit_index))
                + min(8.0, 1.1 * float(len(shared_focus)))
                + min(3.0, 0.6 * float(hit_overlap))
                + min(12.0, max(0.0, reference_focus_score) / 3.0)
                + (4.0 if label_matches_prompt else 0.0)
                + (2.0 if strong_trace_intent else 0.0)
                + (2.0 if label_matches else 0.0)
                + float(score_bonus)
            )
            rows.append(
                (
                    score,
                    {
                        "source_path": source_path,
                        "sid": sid,
                        "ref_num": int(n),
                        "label": label,
                        "heading_path": _compact_text(heading, max_len=160),
                        "evidence_quote": context_text,
                        "context_marker_verified": True,
                        "why_line": "The retrieved source paper explicitly cites this upstream work in the cited context.",
                        "ref_title": _compact_text(title, max_len=160),
                        "ref_authors": _compact_text(
                            str(ref.get("authors") or ""), max_len=160
                        ),
                    },
                )
            )

        heading_hint = str((meta or {}).get("ref_best_heading_path") or (meta or {}).get("heading_path") or "")
        for n, context in _explicit_ref_contexts_from_text(hit_surface, max_contexts=14):
            _push_context_candidate(ref_num=n, context=context, heading=heading_hint, score_bonus=1.0)

        if explicit_upstream_intent and prompt_labels:
            body_text = _source_body_text(source_path)
            for n, context in _explicit_ref_contexts_from_text(body_text, max_contexts=32):
                _push_context_candidate(ref_num=n, context=context, heading=heading_hint, score_bonus=2.5)

        try:
            reference_rows = load_paper_guide_reference_index(source_path)
        except Exception:
            reference_rows = []
        for raw in list(reference_rows or []):
            if not isinstance(raw, Mapping):
                continue
            try:
                ref_num = int(raw.get("ref_num") or 0)
            except Exception:
                ref_num = 0
            if ref_num <= 0:
                continue
            context = _compact_text(str(raw.get("first_citation_context") or "").strip(), max_len=300)
            if not context or int(ref_num) not in _inline_ref_nums_from_text(context):
                continue
            heading = _compact_text(str(raw.get("first_citation_location") or ""), max_len=160)
            _push_context_candidate(
                ref_num=ref_num,
                context=context,
                heading=heading,
                seed_ref=_structured_ref_to_ref(raw),
            )

    rows.sort(key=lambda item: item[0], reverse=True)
    out: list[dict[str, object]] = []
    seen: set[tuple[str, int, str]] = set()
    seen_sources_out: set[str] = set()

    def _try_add(row: Mapping[str, object]) -> bool:
        key = (
            str(row.get("sid") or "").lower(),
            int(row.get("ref_num") or 0),
            str(row.get("label") or "").lower(),
        )
        if key in seen:
            return False
        seen.add(key)
        out.append(dict(row))
        source_key = _source_key(row.get("source_path"))
        if source_key:
            seen_sources_out.add(source_key)
        return True

    limit = max(1, int(max_items or 3))
    for _score, row in rows:
        source_key = _source_key(row.get("source_path"))
        if source_key and source_key in seen_sources_out:
            continue
        _try_add(row)
        if len(out) >= limit:
            return out
    for _score, row in rows:
        _try_add(row)
        if len(out) >= limit:
            break
    return out


def _label_matches_surface(label: str, surface: str) -> bool:
    needle = str(label or "").strip()
    hay = str(surface or "")
    if not needle or len(needle) < 3 or needle.lower().startswith("ref "):
        return False
    needle_loose = _loose_ascii_text(needle)
    for canonical, pattern in _DOMAIN_LABEL_PATTERNS:
        if needle_loose and needle_loose == _loose_ascii_text(canonical):
            if re.search(pattern, hay):
                return True
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


def _line_has_grounded_opportunity_context(*, line: str, prompt: str, opp: Mapping[str, object]) -> bool:
    plain = _compact_text(line, max_len=520)
    if not plain or _GENERIC_SYNTHESIS_LINE_RE.search(plain):
        return False
    if re.search(
        r"(?i)\b(?:next\s+step|if\s+you\s+want|likely|probably|may|might)\b|"
        r"(?:下一步|如果你想|可以查阅|很可能|可能引用|大概率|推测)",
        plain,
    ):
        return False
    label = str(opp.get("label") or "").strip()
    if not label or label.lower().startswith("ref "):
        return False
    label_matches_line = _label_matches_surface(label, plain)
    ref_title = str(opp.get("ref_title") or "").strip()
    ref_title_matches_line = bool(
        ref_title and _loose_ascii_text(ref_title) in _loose_ascii_text(plain)
    )
    evidence_surface = " ".join(
        [
            str(opp.get("evidence_quote") or "").strip(),
            str(opp.get("ref_title") or "").strip(),
            str(opp.get("why_line") or "").strip(),
        ]
    )
    line_tags = _semantic_support_tags(plain)
    evidence_tags = _semantic_support_tags(evidence_surface)
    noise_tags = {"noise", "dark_count", "afterpulsing", "crosstalk", "dead_time"}
    if line_tags.intersection(noise_tags) and not evidence_tags.intersection(noise_tags):
        return False
    if not _is_broad_reference_label(label):
        if label_matches_line and line_tags and evidence_tags:
            broad_only_tags = {"single_pixel_imaging", "snapshot_compressive_imaging", "deep_learning"}
            concrete_line_tags = line_tags.difference(broad_only_tags)
            if concrete_line_tags and not concrete_line_tags.intersection(evidence_tags):
                return False
        return bool(
            label_matches_line
            or ref_title_matches_line
            or _line_can_take_prompt_bound_opportunity(line=plain, prompt=prompt, label=label)
        )
    if not label_matches_line:
        return False
    shared_tags = line_tags.intersection(evidence_tags)
    broad_only_tags = {"single_pixel_imaging", "snapshot_compressive_imaging", "deep_learning"}
    concrete_shared = shared_tags.difference(broad_only_tags)
    return bool(concrete_shared or len(shared_tags) >= 3)


def _label_for_opportunity(*, prompt: str, text: str, ref_num: int) -> str:
    local_ref: re.Match[str] | None = None
    for match in _INLINE_REF_RE.finditer(str(text or "")):
        nums = parse_ref_num_set(str(match.group(1) or ""))
        if int(ref_num) in nums:
            local_ref = match
            break
    if local_ref:
        before = str(text or "")[max(0, local_ref.start() - 100) : local_ref.start()]
        nearest_label = ""
        nearest_start = -1
        for label, pattern in _DOMAIN_LABEL_PATTERNS:
            for match in re.finditer(pattern, before):
                if match.start() >= nearest_start:
                    nearest_start = match.start()
                    nearest_label = label
        if nearest_label:
            return nearest_label
        local_entities = [
            item
            for item in _ENTITY_RE.findall(before)
            if len(item) >= 3
            and item.lower() not in {"the", "this", "that", "most", "for", "example", "fig", "figure", "table"}
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
        if len(entity) >= 3 and entity.lower() not in {"the", "this", "fig", "figure", "table"}:
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
    answer_hits: Sequence[Mapping[str, object]] | None = None,
) -> list[Mapping[str, object]]:
    out: list[Mapping[str, object]] = []
    for group in (support_resolution, support_slots, cards):
        for item in list(group or []):
            if isinstance(item, Mapping):
                out.append(item)
    for hit in list(answer_hits or []):
        if not isinstance(hit, Mapping):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), Mapping) else {}
        passages = (meta or {}).get("source_passages")
        if isinstance(passages, Sequence) and not isinstance(passages, (str, bytes)):
            for passage in passages:
                if not isinstance(passage, Mapping):
                    continue
                row = dict(passage)
                row.setdefault("source_path", str((meta or {}).get("source_path") or "").strip())
                row.setdefault("source_name", str((meta or {}).get("source_name") or "").strip())
                row.setdefault("evidence_quote", str(passage.get("text") or "").strip())
                out.append(row)
            continue
        row = dict(hit)
        row.setdefault("source_path", str((meta or {}).get("source_path") or "").strip())
        row.setdefault("heading_path", str((meta or {}).get("heading_path") or "").strip())
        out.append(row)
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
    answer_hits: Sequence[Mapping[str, object]] | None = None,
    max_items: int = 3,
) -> list[dict[str, object]]:
    """Find upstream bibliography refs that should surface in ordinary answers.

    The detector only trusts refs that are explicitly attached to the current
    paper evidence sentence or clause, either as inline markers such as "[4]"
    or as ref_spans parsed from that same sentence/clause.
    """

    family = str(prompt_family or "").strip().lower()
    if family in _SKIP_FAMILIES:
        return []
    try:
        limit = max(1, min(4, int(max_items)))
    except Exception:
        limit = 3

    rows: list[tuple[float, dict[str, object]]] = []
    structured_refs_by_source: dict[str, dict[int, dict[str, object]]] = {}
    resolved_target_refs: set[int] = set()
    if family == "citation_lookup":
        for record in list(support_resolution or []):
            if not isinstance(record, Mapping):
                continue
            for raw_ref in list(record.get("ref_nums") or record.get("candidate_refs") or []):
                try:
                    ref_num = int(raw_ref)
                except Exception:
                    continue
                if ref_num > 0:
                    resolved_target_refs.add(ref_num)
    candidate_records = _iter_candidate_records(
        support_resolution=support_resolution,
        support_slots=support_slots,
        cards=cards,
        answer_hits=answer_hits,
    )
    if source_path and _UPSTREAM_INTENT_RE.search(str(prompt or "")):
        # Origin questions need the sentence where the current paper actually
        # cites the named upstream work. A bounded retrieval bundle may retain
        # a nearby DPR usage sentence but omit the preceding ``based on DPR
        # [26]`` sentence. Recover only source-verbatim inline-reference
        # contexts whose structured bibliography identity matches the prompt.
        try:
            targeted_reference_rows = load_paper_guide_reference_index(source_path)
        except Exception:
            targeted_reference_rows = []
        prompt_tokens = _tokens(prompt)
        prompt_upper = str(prompt or "").upper()
        prompt_labels = _candidate_labels_from_text(
            prompt=str(prompt or ""), answer="", max_labels=12
        )
        specific_prompt_labels = [
            label
            for label in prompt_labels
            if len(label) >= 4 and re.search(r"[-\d]", label)
        ]
        specifically_matched_refs = {
            int(ref_row.get("ref_num") or 0)
            for ref_row in list(targeted_reference_rows or [])
            if isinstance(ref_row, Mapping)
            and str(ref_row.get("ref_num") or "").strip().isdigit()
            and int(ref_row.get("ref_num") or 0) > 0
            and any(
                _label_matches_surface(
                    label,
                    str(ref_row.get("title") or ref_row.get("text") or ""),
                )
                for label in specific_prompt_labels
            )
        }
        targeted_refs: set[int] = set()
        for ref_row in list(targeted_reference_rows or []):
            if not isinstance(ref_row, Mapping):
                continue
            try:
                targeted_ref_num = int(ref_row.get("ref_num") or 0)
            except Exception:
                targeted_ref_num = 0
            ref_title = str(ref_row.get("title") or ref_row.get("text") or "").strip()
            title_tokens = _tokens(ref_title)
            title_words = [
                word
                for word in re.findall(r"[A-Za-z]+", ref_title)
                if word.lower() not in {"a", "an", "and", "for", "of", "the", "to", "with"}
            ]
            acronym_match = any(
                len(acronym) >= 3
                and re.search(
                    rf"(?<![A-Z0-9]){re.escape(acronym)}(?![A-Z0-9])",
                    prompt_upper,
                )
                for prefix_len in range(2, min(6, len(title_words)) + 1)
                for acronym in ["".join(word[0] for word in title_words[:prefix_len]).upper()]
            )
            if targeted_ref_num > 0 and specifically_matched_refs:
                if targeted_ref_num in specifically_matched_refs:
                    targeted_refs.add(targeted_ref_num)
            elif targeted_ref_num > 0 and (
                acronym_match or len(prompt_tokens & title_tokens) >= 2
            ):
                targeted_refs.add(targeted_ref_num)
        if targeted_refs:
            for ref_num, context in _explicit_ref_contexts_from_text(
                _source_body_text(source_path),
                max_contexts=160,
            ):
                if int(ref_num) not in targeted_refs:
                    continue
                candidate_records.append(
                    {
                        "source_path": source_path,
                        "evidence_quote": context,
                        "ref_num": int(ref_num),
                        "candidate_refs": [int(ref_num)],
                        "context_marker_verified": True,
                        "claim_type": "prior_work",
                    }
                )
            for ref_num, context in _author_year_ref_contexts_from_text(
                _source_body_text(source_path),
                reference_rows=[
                    row
                    for row in list(targeted_reference_rows or [])
                    if isinstance(row, Mapping)
                ],
                target_refs=targeted_refs,
                max_contexts=160,
            ):
                candidate_records.append(
                    {
                        "source_path": source_path,
                        "evidence_quote": context,
                        "ref_num": int(ref_num),
                        "candidate_refs": [int(ref_num)],
                        "context_marker_verified": True,
                        "author_year_marker_verified": True,
                        "claim_type": "prior_work",
                    }
                )
    for record in candidate_records:
        record_source = _source_path_from_record(record, fallback=source_path)
        if not record_source:
            continue
        text = _text_from_record(record)
        refs = _explicit_ref_nums_from_record(record, text=text)
        if (
            not refs
            and record.get("context_marker_verified") is True
            and record.get("author_year_marker_verified") is True
        ):
            for key in ("ref_num", "reference_number"):
                _append_ref_num(refs, record.get(key))
        refs = _prioritize_explicit_refs_for_prompt(prompt=prompt, text=text, refs=refs)
        if resolved_target_refs:
            refs = [ref_num for ref_num in refs if ref_num in resolved_target_refs]
        if not refs:
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
        source_key = _source_key(record_source)
        if source_key not in structured_refs_by_source:
            try:
                structured_rows = load_paper_guide_reference_index(record_source)
            except Exception:
                structured_rows = []
            structured_refs_by_source[source_key] = {
                int(row.get("ref_num") or 0): dict(row)
                for row in list(structured_rows or [])
                if isinstance(row, Mapping) and int(row.get("ref_num") or 0) > 0
            }
        structured_refs = structured_refs_by_source.get(source_key, {})
        for ref_num in refs[:3]:
            ref_row = dict(structured_refs.get(int(ref_num)) or {})
            ref_title = _compact_text(
                str(ref_row.get("title") or ref_row.get("text") or ""),
                max_len=180,
            )
            title_tokens = _tokens(ref_title)
            prompt_tokens = _tokens(prompt)
            title_overlap = len(prompt_tokens.intersection(title_tokens))
            title_words = [
                token
                for token in re.findall(r"[A-Za-z]+", ref_title)
                if token.lower() not in {"a", "an", "and", "for", "of", "the", "to", "with"}
            ]
            title_acronym = ""
            for prefix_len in range(2, min(6, len(title_words)) + 1):
                candidate_acronym = "".join(
                    word[0] for word in title_words[:prefix_len]
                ).upper()
                if re.search(
                    rf"(?<![A-Za-z0-9]){re.escape(candidate_acronym)}(?![A-Za-z0-9])",
                    str(prompt or ""),
                    flags=re.IGNORECASE,
                ):
                    title_acronym = candidate_acronym
                    break
            acronym_match = bool(title_acronym)
            opportunity_label = _label_for_opportunity(
                prompt=prompt,
                text=text,
                ref_num=int(ref_num),
            )
            matched_specific_label = next(
                (
                    label
                    for label in sorted(specific_prompt_labels, key=len, reverse=True)
                    if _label_matches_surface(label, ref_title)
                ),
                "",
            )
            if matched_specific_label:
                opportunity_label = matched_specific_label
            if acronym_match:
                opportunity_label = title_acronym
            rows.append(
                (
                    score
                    + min(12.0, 3.0 * float(title_overlap))
                    + (18.0 if acronym_match else 0.0)
                    - (0.05 * len(rows)),
                    {
                        "source_path": record_source,
                        "sid": sid,
                        "ref_num": int(ref_num),
                        "label": opportunity_label,
                        "heading_path": heading,
                        "evidence_quote": text,
                        "context_marker_verified": True,
                        "why_line": (
                            "The current paper cites this upstream work in the evidence used for the answer."
                        ),
                        "ref_title": ref_title,
                        "ref_authors": _compact_text(
                            str(ref_row.get("authors") or ""), max_len=160
                        ),
                        "ref_year": str(ref_row.get("year") or "").strip(),
                        "reference_style": str(ref_row.get("reference_style") or "").strip(),
                        "reference_source_page": int(
                            ref_row.get("source_page") or ref_row.get("page_start") or 0
                        ),
                        "ref_raw": _compact_text(
                            str(ref_row.get("text") or ""), max_len=360
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


def merge_reference_opportunities(
    *groups: Sequence[Mapping[str, object]] | None,
    max_items: int = 3,
) -> list[dict[str, object]]:
    """Merge opportunity groups in priority order without duplicating a ref."""

    try:
        limit = max(1, min(4, int(max_items)))
    except Exception:
        limit = 3
    merged: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for group in groups:
        for row in _normalized_opportunities(group, max_items=limit):
            key = (str(row.get("sid") or "").strip().lower(), int(row.get("ref_num") or 0))
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
            if len(merged) >= limit:
                return merged
    return merged


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
        "- For ordinary synthesis, concept, origin, prior-work, or method-background questions, place the exact cite_example inline next to the sentence that uses or explains that upstream work.",
        "- Do not dump these as a separate bibliography list unless the user asks for a reading list.",
        "- If the answer does not discuss the listed concept, do not force the citation.",
    ]
    for row in rows:
        sid = str(row.get("sid") or "").strip()
        ref_num = int(row.get("ref_num") or 0)
        label = _compact_text(str(row.get("label") or f"ref {ref_num}"), max_len=80)
        heading = _compact_text(str(row.get("heading_path") or ""), max_len=120)
        evidence = _compact_text(str(row.get("evidence_quote") or ""), max_len=180)
        ref_title = _compact_text(str(row.get("ref_title") or ""), max_len=140)
        ref_authors = _compact_text(str(row.get("ref_authors") or ""), max_len=100)
        parts = [f"label={label}", f"cite_example=[[CITE:{sid}:{ref_num}]]"]
        if ref_title:
            parts.append(f"reference_title={ref_title}")
        if ref_authors:
            parts.append(f"reference_authors={ref_authors}")
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


def _looks_like_reading_list_title_line(line: str) -> bool:
    text = _compact_text(line, max_len=260)
    if not text:
        return False
    if re.match(r"^\s*\*{0,2}(?:文献|论文|paper|reference)\*{0,2}\s*[:：]", text, flags=re.I):
        return True
    if not re.match(r"^\s*\d{1,2}\.\s+\*{0,2}[《\"“]", text):
        return False
    return not bool(re.search(r"(?:看什么|重点|理解|because|why|focus|read for)", text, flags=re.I))


def _line_score_for_opportunity(*, line: str, prompt: str, opp: Mapping[str, object]) -> float:
    plain = _compact_text(line, max_len=520)
    if len(plain) < 8 or _ANSWER_LINE_SKIP_RE.search(plain):
        return float("-inf")
    if _looks_like_reading_list_title_line(plain):
        return float("-inf")
    if _OPPORTUNITY_NOTE_RE.search(plain):
        return float("-inf")
    label = str(opp.get("label") or "").strip()
    evidence = str(opp.get("evidence_quote") or "").strip()
    ref_title = str(opp.get("ref_title") or "").strip()
    if not _line_has_grounded_opportunity_context(line=plain, prompt=prompt, opp=opp):
        return float("-inf")
    score = 0.0
    label_matches = _label_matches_surface(label, plain)
    if label_matches:
        score += 8.0
    label_tokens = _tokens(label)
    evidence_tokens = _tokens(evidence)
    ref_title_tokens = _tokens(ref_title)
    prompt_tokens = _tokens(prompt)
    line_tokens = _tokens(plain)
    meaningful_label = bool(label_tokens and not str(label or "").strip().lower().startswith("ref "))
    ref_title_matches = bool(
        ref_title and _loose_ascii_text(ref_title) in _loose_ascii_text(plain)
    )
    prompt_bound_fallback = ref_title_matches or _line_can_take_prompt_bound_opportunity(
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
    if ref_title_tokens:
        # For explicit origin/prior-work questions, prefer the answer sentence
        # that identifies the cited paper over a generic sentence that merely
        # mentions the short method acronym. This keeps the structured marker
        # on the claim that the System-B validator can independently verify.
        score += min(9.0, 2.5 * float(len(ref_title_tokens.intersection(line_tokens))))
    prompt_entities = {
        token
        for token in prompt_tokens
        if len(token) >= 5
    }
    if prompt_entities:
        score += min(5.0, 2.5 * float(len(prompt_entities.intersection(line_tokens))))
    prompt_author_surnames = {
        surname.lower()
        for surname in re.findall(
            r"(?i)\b([A-Z][A-Za-z-]{3,})\s+et\s+al\.?",
            str(prompt or ""),
        )
    }
    if prompt_author_surnames.intersection(line_tokens):
        score += 12.0
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


def _replace_exact_bare_ref_marker(line: str, *, ref_num: int, marker: str) -> tuple[str, bool]:
    text = str(line or "")
    if not text or int(ref_num or 0) <= 0 or not str(marker or "").strip() or marker in text:
        return text, False

    changed = False

    def _repl(m: re.Match[str]) -> str:
        nonlocal changed
        # Leave existing markdown links alone.
        if str(text[m.end() : m.end() + 1] or "") == "(":
            return str(m.group(0) or "")
        nums = [int(raw) for raw in re.findall(r"\d{1,4}", str(m.group(1) or ""))]
        if nums == [int(ref_num)]:
            changed = True
            return str(marker)
        return str(m.group(0) or "")

    return _INLINE_REF_RE.sub(_repl, text), changed


def inject_reference_opportunity_citations_inline(
    answer: str,
    *,
    prompt: str,
    opportunities: Sequence[Mapping[str, object]] | None,
    min_score: float = 6.0,
    max_injections: int = 3,
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
    try:
        injection_limit = max(1, min(3, int(max_injections)))
    except Exception:
        injection_limit = 3
    used_line_indexes: set[int] = set()
    for opp in rows:
        if len(injected_refs) >= injection_limit:
            break
        sid = str(opp.get("sid") or "").strip()
        ref_num = int(opp.get("ref_num") or 0)
        if not sid or ref_num <= 0 or (sid.lower(), ref_num) in existing:
            continue
        marker = _cite_marker_for_opportunity(opp)
        best_bare_idx = -1
        best_bare_score = float("-inf")
        best_idx = -1
        best_score = float("-inf")
        for idx, line in enumerate(lines):
            if idx in used_line_indexes:
                continue
            score = _line_score_for_opportunity(line=line, prompt=prompt, opp=opp)
            _, has_bare_marker = _replace_exact_bare_ref_marker(line, ref_num=ref_num, marker=marker)
            if has_bare_marker and score > best_bare_score:
                best_bare_idx = idx
                best_bare_score = score
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_bare_idx >= 0 and best_bare_score >= min(4.0, float(min_score)):
            replaced, changed = _replace_exact_bare_ref_marker(
                lines[best_bare_idx],
                ref_num=ref_num,
                marker=marker,
            )
            if changed:
                lines[best_bare_idx] = replaced
                injected_refs.append(ref_num)
                used_line_indexes.add(best_bare_idx)
                existing.add((sid.lower(), ref_num))
                continue
        if best_idx < 0 or best_score < float(min_score):
            continue
        lines[best_idx] = _insert_marker_before_terminal_punctuation(lines[best_idx], marker)
        injected_refs.append(ref_num)
        used_line_indexes.add(best_idx)
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

    if _UPSTREAM_INTENT_RE.search(str(prompt or "")):
        lines = text.splitlines()
        for row in rows:
            ref_title = _compact_text(str(row.get("ref_title") or ""), max_len=180)
            ref_surface = " ".join(
                str(row.get(key) or "").strip()
                for key in ("ref_authors", "ref_raw")
            )
            prompt_author_matches = re.findall(
                r"(?i)\b([A-Z][A-Za-z-]{3,})\s+et\s+al\.?",
                str(prompt or ""),
            )
            author_display = next(
                (
                    surname
                    for surname in prompt_author_matches
                    if re.search(rf"(?i)\b{re.escape(surname)}\b", ref_surface)
                ),
                "",
            )
            if not author_display:
                first_author = re.split(
                    r"\s*(?:,|;|\band\b)\s*",
                    str(row.get("ref_authors") or row.get("ref_raw") or ""),
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0]
                first_author = re.sub(
                    r"(?i)\s+et\s+al\.?$",
                    "",
                    first_author,
                ).strip()
                author_tokens = re.findall(
                    r"[A-Z][A-Za-z-]{2,}",
                    first_author,
                )
                if author_tokens:
                    author_display = author_tokens[-1]
            if not ref_title or not author_display:
                continue
            ref_year = str(row.get("ref_year") or "").strip()
            identity_present = bool(
                re.search(re.escape(ref_title), text, flags=re.IGNORECASE)
                and re.search(rf"(?i)\b{re.escape(author_display)}\b", text)
            )
            if identity_present:
                marker = _cite_marker_for_opportunity(row)
                identity_indexes = [
                    idx
                    for idx, line in enumerate(lines)
                    if re.search(re.escape(ref_title), line, flags=re.IGNORECASE)
                    and re.search(
                        rf"(?i)\b{re.escape(author_display)}\b",
                        line,
                    )
                ]
                marker_indexes = [
                    idx for idx, line in enumerate(lines) if marker in line
                ]
                if (
                    identity_indexes
                    and marker_indexes
                    and not any(marker in lines[idx] for idx in identity_indexes)
                ):
                    # A generated answer can contain the correct upstream
                    # identity and the correct structured marker, but attach
                    # that marker to a nearby implementation detail. Move the
                    # already-validated marker to the author/title claim so
                    # the visible System-B card audits the statement it proves.
                    for idx in marker_indexes:
                        lines[idx] = re.sub(
                            rf"\s*{re.escape(marker)}",
                            "",
                            lines[idx],
                        )
                    target_idx = max(
                        identity_indexes,
                        key=lambda idx: (
                            1 if re.search(r"(?i)upstream\s+paper", lines[idx]) else 0,
                            _line_score_for_opportunity(
                                line=lines[idx], prompt=prompt, opp=row
                            ),
                        ),
                    )
                    lines[target_idx] = _insert_marker_before_terminal_punctuation(
                        lines[target_idx],
                        marker,
                    )
                    text = "\n".join(lines).strip()
                continue
            candidate_indexes = [
                idx
                for idx, line in enumerate(lines)
                if _line_has_grounded_opportunity_context(
                    line=line,
                    prompt=prompt,
                    opp=row,
                )
                and (
                    _PRIOR_WORK_CUE_RE.search(line)
                    or re.search(r"(?i)upstream\s+paper|prior\s+work", line)
                )
            ]
            if not candidate_indexes:
                continue
            target_idx = max(
                candidate_indexes,
                key=lambda idx: (
                    (
                        2
                        if re.search(r"(?i)upstream\s+paper", lines[idx])
                        else 1
                        if re.search(r"(?i)prior\s+work", lines[idx])
                        else 0
                    ),
                    _line_score_for_opportunity(
                        line=lines[idx], prompt=prompt, opp=row
                    ),
                ),
            )
            prefix_match = re.match(
                r"^(?P<prefix>\s*(?:[-*]\s*)?(?:Upstream\s+paper|Prior\s+work)\s*:)\s*",
                lines[target_idx],
                flags=re.IGNORECASE,
            )
            identity = f"{ref_title}, by {author_display} et al."
            if re.fullmatch(r"(?:19|20)\d{2}", ref_year):
                identity += f" ({ref_year})"
            lines[target_idx] = (
                f"{prefix_match.group('prefix')} {identity}"
                if prefix_match
                else _insert_marker_before_terminal_punctuation(
                    lines[target_idx], f"({identity})"
                )
            )
            text = "\n".join(lines).strip()

    inline_text, inline_meta = inject_reference_opportunity_citations_inline(
        text,
        prompt=prompt,
        opportunities=rows,
        max_injections=3,
    )
    if inline_text != text:
        meta = dict(inline_meta)
        meta["tail_used"] = False
        return inline_text, meta

    if not _prompt_allows_reference_tail_note(prompt):
        return text, {
            "mode": str(inline_meta.get("mode") or "none"),
            "injected_refs": [],
            "tail_used": False,
            "tail_suppressed": True,
        }

    appended = append_reference_opportunity_note(text, prompt=prompt, opportunities=rows)
    if appended != text:
        tail_segment = appended[len(text) :] if appended.startswith(text) else appended
        tail_refs = [int(m.group(2) or 0) for m in _CITE_CANON_RE.finditer(tail_segment)]
        return appended, {
            "mode": "tail",
            "injected_refs": [],
            "tail_refs": [n for n in tail_refs if n > 0],
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
    if not _prompt_allows_reference_tail_note(prompt):
        return text
    opps = [dict(item) for item in list(opportunities or []) if isinstance(item, Mapping)]
    if not opps:
        return text
    existing = {
        (str(m.group(1) or "").strip().lower(), int(m.group(2) or 0))
        for m in _CITE_CANON_RE.finditer(text)
    }
    parts: list[str] = []
    seen_broad_tail_labels: set[str] = set()
    for opp in opps:
        sid = str(opp.get("sid") or "").strip()
        try:
            ref_num = int(opp.get("ref_num") or 0)
        except Exception:
            ref_num = 0
        if not sid or ref_num <= 0 or (sid.lower(), ref_num) in existing:
            continue
        raw_label = str(opp.get("label") or "").strip() or f"ref {ref_num}"
        if _is_broad_reference_label(raw_label):
            broad_key = _loose_ascii_text(raw_label) or raw_label.lower()
            if broad_key in seen_broad_tail_labels:
                continue
            seen_broad_tail_labels.add(broad_key)
        label = raw_label
        ref_title = str(opp.get("ref_title") or "").strip()
        if ref_title and (_is_broad_reference_label(label) or label.lower().startswith("ref ")):
            label = _compact_text(ref_title, max_len=90)
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
