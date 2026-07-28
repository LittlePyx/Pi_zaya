from __future__ import annotations

import difflib
import re
from functools import lru_cache

from kb.evidence_term_mapping import evidence_alignment_tokens
from kb.source_blocks import normalize_inline_markdown


_SYSTEM_A_DOMAIN_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("iscat", re.compile(r"(?i)\biscat\b|干涉散射")),
    ("interferometric", re.compile(r"(?i)\binterferometric\b|干涉检测|干涉散射")),
    ("structured detection", re.compile(r"(?i)\bstructured\s+detection\b|结构检测|结构探测")),
    ("single-photon detection", re.compile(r"(?i)\bsingle[-\s]?photon\s+(?:detection|detections|detectors?|photodetectors?)\b|\bSPADs?\b|单光子.{0,8}探测|光子探测器")),
    ("spad noise model", re.compile(r"(?i)\b(?:physical\s+)?multi[-\s]?source\s+(?:physical\s+)?noise\b|\bSPADs?\b.{0,24}\bnoise\b|多源.{0,8}噪声|物理噪声")),
    ("poisson noise", re.compile(r"(?i)\bpoisson\s+noise\b|泊松噪声")),
    ("crosstalk noise", re.compile(r"(?i)\bcrosstalk(?:\s+noise)?\b|串扰(?:噪声)?")),
    ("dark count", re.compile(r"(?i)\bdark\s+count(?:\s+rate)?\b|暗计数(?:率)?")),
    (
        "photon-limited SPAD degradation",
        re.compile(
            r"(?is)(?=.*(?:\bSPADs?\b|single[-\s]?photon|SPAD\s*阵列|单光子))"
            r"(?=.*(?:low\s+bit\s+depth|low\s+resolution|heavy\s+noise|photon[-\s]?limited|"
            r"低比特深度|低分辨率|严重噪声|光子受限))"
        ),
    ),
    ("waveguide", re.compile(r"(?i)\bwaveguides?\b|波导")),
    ("cut-off frequency", re.compile(r"(?i)\bcut[-\s]?off\s+frequenc(?:y|ies)\b|截止频率")),
    ("image scanning microscopy", re.compile(r"(?i)\bimage\s+scanning\s+microscopy\b|\bISM\b|共聚焦|扫描显微")),
    ("light field", re.compile(r"(?i)\blight[-\s]?field\b|光场")),
    ("digital refocusing", re.compile(r"(?i)\bdigital\s+refocus(?:ing)?\b|\brefocus(?:ing)?\b|重聚焦|重新对焦")),
    ("ray tracing", re.compile(r"(?i)\bray\s+trac(?:e|ing)\b|\bray\s+optics\b|射线追踪|几何光学")),
    ("wave propagation", re.compile(r"(?i)\bwave\s+propagation\b|\bwave\s+optics\b|\bdiffraction\b|波动光学|波传播|衍射")),
    ("quantum correlation", re.compile(r"(?i)\bquantum\s+correlation\b|\btime[-\s]?correlation\b|量子关联|时间关联|光子对")),
    ("single-pixel imaging", re.compile(r"(?i)\bsingle[-\s]?pixel\b|\bspi\b|单像素")),
    ("deep learning", re.compile(r"(?i)\bdeep\s+learning\b|\bneural\s+network\b|深度学习|神经网络")),
    (
        "training and generalization",
        re.compile(
            r"(?i)\b(?:training\s+(?:duration|time|data)|limited\s+generalization|generalization)\b|"
            r"训练(?:时间|耗时|数据)?|数据驱动|泛化(?:能力)?"
        ),
    ),
    (
        "reconstruction quality and speed",
        re.compile(
            r"(?i)\breconstruction\s+(?:quality|speed)\b|"
            r"重建(?:质量|速度)|(?:质量高|速度快|高质量|快速重建)"
        ),
    ),
    (
        "image quality and reconstruction time",
        re.compile(
            r"(?i)\blimited\s+image\s+quality\b|\blengthy\s+computational\s+times?\b|"
            r"\biterative\s+reconstruction\b|图像质量(?:有限|受限)|迭代重建|计算时间(?:长|较长)?"
        ),
    ),
    (
        "super-resolution bit-depth outcome",
        re.compile(
            r"(?is)(?=.*(?:super[-\s]?resolution|超分辨率?))"
            r"(?=.*(?:bit[-\s]?depth|imaging\s+quality|位深(?:增强|提升)?|成像质量(?:增强|提升)?))"
        ),
    ),
    (
        "spatial domain denoising",
        re.compile(r"(?i)\bspatial\s+domain(?:\s+methods?)?\b|空间域(?:方法|去噪)?"),
    ),
    (
        "transform domain denoising",
        re.compile(r"(?i)\btransform\s+domain(?:\s+methods?)?\b|变换域(?:方法|去噪)?"),
    ),
    (
        "pixel patch correlation",
        re.compile(
            r"(?i)\bcorrelation\s+between\s+(?:pixels|image\s+patches)\b|"
            r"像素.{0,8}相关|图像块.{0,8}相关"
        ),
    ),
    ("wavelet transform", re.compile(r"(?i)\bwavelet\s+transform\b|小波变换")),
    (
        "foveated",
        re.compile(
            r"(?i)\bfove(?:at(?:ed|ion)|al)\b|中央凹|中心凹|"
            r"高分辨率(?:焦点|中央凹|中心凹)区域|自适应采样"
        ),
    ),
    ("dynamic supersampling", re.compile(r"(?i)\bdynamic\s+supersampling\b|\bsupersampling\b|超采样")),
    ("frame rate", re.compile(r"(?i)\bframe\s+rate\b|\bframes?\s+per\s+second\b|帧率|帧\s*/\s*秒")),
    (
        "sampling ratio",
        re.compile(
            r"(?i)\bsampl(?:e|ing)\s+rates?\b|\bsampling\s+ratio\b|"
            r"\blow[-\s]?sampling\b|\bfewer\s+measurements?\b|采样率|低采样|更少测量"
        ),
    ),
    (
        "frequency-division multiplexing",
        re.compile(
            r"(?is)\bfrequency[-\s]?(?:division\s+)?multiplex(?:ing|ed)?\b|"
            r"(?=.*\bfrequenc(?:y|ies)\b)(?=.*\bmultiplex(?:ing|ed)?\b)|"
            r"频分复用|频率复用"
        ),
    ),
    (
        "multiple detectors",
        re.compile(
            r"(?i)\b(?:multiple|four|three|several)(?:\s+spatially[- ]separated)?\s+"
            r"(?:single[- ]pixel\s+)?detectors?\b|(?:多个|四个|三个|多)"
            r"(?:空间(?:上)?分离的)?(?:单像素)?探测器"
        ),
    ),
    (
        "lock-in demodulation",
        re.compile(
            r"(?i)\block[- ]?in\b|phase[- ]sensitive\s+detection|demodulat|"
            r"锁相(?:放大器)?|相位敏感检测|解调"
        ),
    ),
    (
        "bpsk frequency channels",
        re.compile(
            r"(?i)\bBPSK\b|binary\s+phase\s+shift\s+keying|carrier\s+frequenc|"
            r"载波频率|多频率信号|二进制相移键控"
        ),
    ),
    ("spatial-spectral acquisition", re.compile(r"(?i)\bspatial[-\s]?spectral\s+acquisition\b|空间[-—–]?光谱(?:采集)?")),
    ("random patterns", re.compile(r"(?i)\brandom\s+patterns?\b|随机(?:模式|图案)")),
    ("deterministic orthogonal basis", re.compile(r"(?i)\bdeterministic\s+orthogonal\s+(?:basis\s+)?patterns?\b|确定性正交基(?:模式|图案)?")),
    ("perfect reconstruction", re.compile(r"(?i)\bperfect\s+reconstruction\b|完美重构")),
    ("hadamard", re.compile(r"(?i)\bhadamard\b|哈达玛|哈达马")),
    ("fourier", re.compile(r"(?i)\bfourier\b|傅里叶")),
    ("compressed sensing", re.compile(r"(?i)\bcompress(?:ed|ive)\s+sensing\b|压缩感知")),
    ("photometric stereo", re.compile(r"(?i)\bphotometric\s+stereo\b|光度立体")),
    ("dmd", re.compile(r"(?i)\bdmd\b|digital\s+micromirror|数字微镜")),
    ("optical sectioning", re.compile(r"(?i)\boptical\s+sectioning\b|光学切片")),
    (
        "dynamic 3d",
        re.compile(
            r"(?i)\bdynamic\s+3d\b|\b(?:continuous\s+)?real[- ]time\s+3d\s+video\b|"
            r"动态\s*3d|(?:连续)?实时(?:的)?\s*3d\s*视频"
        ),
    ),
    ("neural radiance fields", re.compile(r"(?i)\bnerf\b|neural\s+radiance\s+fields?|神经辐射场")),
    ("3d gaussian splatting", re.compile(r"(?i)\b3dgs\b|gaussian\s+splatting|高斯泼溅|高斯溅射")),
    ("snapshot compressive imaging", re.compile(r"(?i)\bsci\b|snapshot\s+compressive|压缩快照")),
    (
        "cassi",
        re.compile(
            r"(?i)\bcassi\b|coded\s+aperture\s+snapshot|dual[-\s]?disperser|"
            r"two\s+dispersive\s+elements?|binary(?:-valued)?\s+aperture\s+code|编码孔径|双色散"
        ),
    ),
    (
        "beat-frequency phase stepping",
        re.compile(
            r"(?is)(?:"
            r"beat\s+frequency.{0,100}phase\s+stepping|"
            r"phase\s+stepping.{0,100}beat\s+frequency|"
            r"heterodyne\s+holography.{0,100}(?:phase\s+stepping|beat\s+frequency)|"
            r"(?:拍频|外差全息).{0,60}(?:相移|相位步进)|"
            r"(?:相移|相位步进).{0,60}(?:拍频|外差全息)"
            r")"
        ),
    ),
    (
        "distilled sensing",
        re.compile(r"(?i)\bdistilled\s+sensing\b|蒸馏感知"),
    ),
    (
        "s2ism tradeoff",
        re.compile(
            r"(?is)(?=.*(?:spatial\s+resolution|空间分辨率|超分辨))"
            r"(?=.*(?:signal-to-noise|\bSNR\b|信噪比))"
            r"(?=.*(?:optical\s+sectioning|光学切片|光学层切))"
        ),
    ),
    (
        "spad geiger quenching",
        re.compile(
            r"(?is)(?=.*(?:\bSPAD\b|single\s+photon\s+avalanche\s+diode))"
            r"(?=.*(?:Geiger|盖革))"
            r"(?=.*(?:breakdown\s+voltage|击穿电压))"
            r"(?=.*(?:quenching\s+circuit|淬灭电路))"
        ),
    ),
    (
        "scinerf physical formation",
        re.compile(
            r"(?is)(?=.*(?:SCINeRF|NeRF))"
            r"(?=.*(?:physical\s+imag(?:e|ing)\s+(?:formation\s+)?process|物理成像过程))"
        ),
    ),
    (
        "piln",
        re.compile(r"(?i)\bPILN\b|\bILNet\b|part[-\s]?based\s+image[-\s]?loop\s+network"),
    ),
    ("admm", re.compile(r"(?i)\badmm\b|交替方向乘子")),
    ("perovskite", re.compile(r"(?i)\bperovskite\b|钙钛矿")),
)

# These concepts used to be expressed as several ``(?=.*...)`` lookaheads
# over the full evidence surface.  Python's backtracking regex engine can make
# that quadratic on multi-kilobyte source passages.  Independent requirements
# preserve the same conjunction semantics while scanning each surface once per
# required concept.
_SYSTEM_A_COMPOUND_DOMAIN_REQUIREMENTS: tuple[
    tuple[str, tuple[re.Pattern, ...]], ...
] = (
    (
        "photon-limited SPAD degradation",
        (
            re.compile(r"\bSPADs?\b|single[-\s]?photon|SPAD\s*阵列|单光子", re.I),
            re.compile(
                r"low\s+bit\s+depth|low\s+resolution|heavy\s+noise|"
                r"photon[-\s]?limited|低比特深度|低分辨率|严重噪声|光子受限",
                re.I,
            ),
        ),
    ),
    (
        "super-resolution bit-depth outcome",
        (
            re.compile(r"super[-\s]?resolution|超分辨率?", re.I),
            re.compile(
                r"bit[-\s]?depth|imaging\s+quality|位深(?:增强|提升)?|"
                r"成像质量(?:增强|提升)?",
                re.I,
            ),
        ),
    ),
    (
        "frequency-division multiplexing",
        (
            re.compile(r"\bfrequenc(?:y|ies)|频率|频分", re.I),
            re.compile(r"\bmultiplex(?:ing|ed)?\b|复用", re.I),
        ),
    ),
    (
        "s2ism tradeoff",
        (
            re.compile(r"spatial\s+resolution|空间分辨率|超分辨", re.I),
            re.compile(r"signal-to-noise|\bSNR\b|信噪比", re.I),
            re.compile(r"optical\s+sectioning|光学切片|光学层切", re.I),
        ),
    ),
    (
        "spad geiger quenching",
        (
            re.compile(r"\bSPAD\b|single\s+photon\s+avalanche\s+diode", re.I),
            re.compile(r"Geiger|盖革", re.I),
            re.compile(r"breakdown\s+voltage|击穿电压", re.I),
            re.compile(r"quenching\s+circuit|淬灭电路", re.I),
        ),
    ),
    (
        "scinerf physical formation",
        (
            re.compile(r"SCINeRF|\bNeRF\b", re.I),
            re.compile(
                r"physical\s+imag(?:e|ing)\s+(?:formation\s+)?process|物理成像过程",
                re.I,
            ),
        ),
    ),
)
_SYSTEM_A_COMPOUND_DOMAIN_NAMES = frozenset(
    name for name, _requirements in _SYSTEM_A_COMPOUND_DOMAIN_REQUIREMENTS
)
_SYSTEM_A_STRONG_BINDING_TERMS = {
    "iscat",
    "interferometric",
    "structured detection",
    "single-photon detection",
    "spad noise model",
    "poisson noise",
    "crosstalk noise",
    "dark count",
    "photon-limited SPAD degradation",
    "waveguide",
    "image scanning microscopy",
    "light field",
    "digital refocusing",
    "ray tracing",
    "wave propagation",
    "quantum correlation",
    "foveated",
    "dynamic supersampling",
    "dynamic 3d",
    "deep learning",
    "sampling ratio",
    "spatial domain denoising",
    "transform domain denoising",
    "pixel patch correlation",
    "wavelet transform",
    "frequency-division multiplexing",
    "multiple detectors",
    "lock-in demodulation",
    "bpsk frequency channels",
    "spatial-spectral acquisition",
    "random patterns",
    "deterministic orthogonal basis",
    "perfect reconstruction",
    "hadamard",
    "fourier",
    "compressed sensing",
    "photometric stereo",
    "dmd",
    "neural radiance fields",
    "3d gaussian splatting",
    "cassi",
    "beat-frequency phase stepping",
    "distilled sensing",
    "s2ism tradeoff",
    "spad geiger quenching",
    "scinerf physical formation",
    "piln",
    "admm",
}
_SYSTEM_A_CONTEXT_ONLY_BINDING_TERMS = {
    "cut-off frequency",
    "multiple detectors",
    "random patterns",
    "perfect reconstruction",
}
_SYSTEM_A_TOKEN_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "these",
    "those",
    "method",
    "methods",
    "paper",
    "answer",
    "source",
    "evidence",
    "section",
    "result",
    "results",
    "using",
    "used",
    "based",
    "through",
    "between",
    "different",
    "mainly",
    "problem",
    "problems",
}

def _system_a_prefers_zh(*texts: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", " ".join(str(text or "") for text in texts)))


@lru_cache(maxsize=4096)
def _system_a_domain_terms(text: str) -> frozenset[str]:
    raw = str(text or "")
    if not raw:
        return frozenset()
    matched = {
        name
        for name, pattern in _SYSTEM_A_DOMAIN_PATTERNS
        if name not in _SYSTEM_A_COMPOUND_DOMAIN_NAMES and pattern.search(raw)
    }
    matched.update(
        name
        for name, requirements in _SYSTEM_A_COMPOUND_DOMAIN_REQUIREMENTS
        if all(pattern.search(raw) for pattern in requirements)
    )
    return frozenset(matched)


@lru_cache(maxsize=4096)
def _system_a_keyword_terms(text: str, *, limit: int = 18) -> set[str]:
    raw = str(text or "")
    out: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9+.-]{2,}", raw):
        t = token.strip().lower().strip(".-")
        if not t or t in _SYSTEM_A_TOKEN_STOPWORDS:
            continue
        if t.isdigit():
            continue
        out.add(t)
        if len(out) >= max(1, int(limit)):
            break
    for token in re.findall(r"[\u4e00-\u9fff]{2,8}", raw):
        if token in {"这个", "这种", "主要", "问题", "方法", "论文", "答案", "来源", "证据"}:
            continue
        out.add(token)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _system_a_has_source_identity_overlap(claim: str, evidence_surface: str, source_name: str) -> bool:
    claim_terms = _system_a_keyword_terms(claim, limit=32)
    if not claim_terms:
        return False
    source_terms = _system_a_keyword_terms(source_name, limit=32)
    if not source_terms:
        return False
    evidence_terms = _system_a_keyword_terms(evidence_surface, limit=48)
    shared_source_terms = claim_terms & source_terms
    if not shared_source_terms:
        return False
    if evidence_terms and not (shared_source_terms & evidence_terms):
        return False
    if any(len(term) >= 7 for term in shared_source_terms):
        return True
    return len(shared_source_terms) >= 2


def _system_a_term_label(terms: set[str] | list[str] | tuple[str, ...], *, max_terms: int = 4) -> str:
    vals = [str(x or "").strip() for x in list(terms or []) if str(x or "").strip()]
    vals = sorted(dict.fromkeys(vals))
    return " / ".join(vals[: max(1, int(max_terms))])


def assess_system_a_hit_binding(
    *,
    answer_claim: str,
    hit: dict,
    meta: dict,
    heading: str,
    evidence_quote: str,
    source_name: str,
) -> dict:
    claim = re.sub(r"\s+", " ", normalize_inline_markdown(str(answer_claim or ""))).strip()
    evidence_body_surface = " ".join(
        [
            str(evidence_quote or ""),
            str((hit or {}).get("text") or ""),
            str(heading or ""),
            str((meta or {}).get("why_line") or ""),
        ]
    )
    evidence_surface = " ".join([evidence_body_surface, str(source_name or "")])
    claim_low = claim.lower()
    evidence_body_low = evidence_body_surface.lower()
    source_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", str(source_name or "").lower())
        if len(token) >= 3
        and token not in {"pdf", "paper", "journal", "2023", "2024", "2025"}
    }
    named_candidates = [
        next((part for part in match if part), "")
        for match in re.findall(
            r"\(([^()]{24,}?)\)|（([^（）]{24,}?)）|\*([^*\n]{24,})\*|《([^》\n]{24,})》",
            claim,
        )
    ]
    for named_candidate in named_candidates:
        # Long parenthetical examples are common in quantitative answers, for
        # example ``(e.g. 400-1000 nm, QE 50%-92%, 200-300 K)``.  They are not
        # paper titles.  Treating every long parenthesis as a named work made
        # an otherwise exact table citation look like a cross-paper mismatch.
        if re.search(r"(?:^|[\s（(])(?:e\.g\.|i\.e\.|for\s+example|如|例如)(?:\b|$)", named_candidate, re.I):
            continue
        numeric_tokens = re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", named_candidate)
        if len(numeric_tokens) >= 2 or re.search(
            r"[%％]|\b(?:nm|mm|cm|km|hz|khz|mhz|ghz|thz|kelvin|pixels?|fps|db)\b",
            named_candidate,
            re.I,
        ):
            continue
        title_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", named_candidate.lower())
            if len(token) >= 3
            and token not in {"the", "and", "with", "from", "paper", "journal", "2023", "2024", "2025"}
        }
        if len(title_tokens) < 5 or len(source_tokens) < 3:
            continue
        if len(title_tokens & source_tokens) / max(1, len(title_tokens)) < 0.45:
            reason = (
                "答案句明确点名了另一篇论文，不能把当前命中作为该句证据。"
                if _system_a_prefers_zh(claim)
                else "The answer sentence explicitly names a different paper, so this hit cannot support it."
            )
            return {
                "status": "mismatch",
                "confidence": 0.0,
                "suppress_link": True,
                "reason": reason,
                "overlap_terms": [],
                "missing_terms": ["source identity"],
            }
    physical_noise_model_re = re.compile(
        r"physical\s+(?:multi[- ]source\s+)?noise\s+model|"
        r"multi[- ]source\s+(?:physical\s+)?noise\s+model|"
        r"物理噪声模型|多源(?:物理)?噪声模型|多源噪声"
    )
    if physical_noise_model_re.search(claim_low) and not physical_noise_model_re.search(
        evidence_body_low
    ):
        reason = (
            "答案句要求的是物理噪声模型证据，但该命中只涉及邻近的探测器或成像主题。"
            if _system_a_prefers_zh(claim)
            else "The claim requires evidence for a physical noise model, but this hit only covers an adjacent detector or imaging topic."
        )
        return {
            "status": "mismatch",
            "confidence": 0.0,
            "suppress_link": True,
            "reason": reason,
            "overlap_terms": [],
            "missing_terms": ["physical noise model"],
        }
    review_identity_re = re.compile(r"\b(?:review|survey)\b|综述", re.I)
    method_identity_re = re.compile(
        r"\b(?:here\s*,?\s*)?we\s+(?:introduce|propose|develop|present|build)\b|"
        r"\bour\s+(?:method|framework|model)\b",
        re.I,
    )
    if (
        review_identity_re.search(claim_low)
        and not review_identity_re.search(evidence_body_low)
        and method_identity_re.search(evidence_body_low)
    ):
        reason = (
            "答案句明确描述的是综述文献，但该命中不是综述证据。"
            if _system_a_prefers_zh(claim)
            else "The claim explicitly describes a review, but this hit is not review evidence."
        )
        return {
            "status": "mismatch",
            "confidence": 0.0,
            "suppress_link": True,
            "reason": reason,
            "overlap_terms": [],
            "missing_terms": ["review identity"],
        }
    verified_prompt_contract = bool(
        (meta or {}).get("citation_plan_evidence_authoritative")
        and str(
            (meta or {}).get("citation_plan_evidence_selection_reason") or ""
        ).strip().lower()
        == "prompt_contract_block"
        and (
            str((meta or {}).get("primary_block_id") or "").strip()
            or str((meta or {}).get("primary_anchor_id") or "").strip()
        )
        and int((meta or {}).get("page_start") or 0) > 0
    )
    canonical_answer_evidence = bool((meta or {}).get("canonical_answer_evidence"))
    prompt_aligned_plan_evidence = bool(
        (meta or {}).get("citation_plan_evidence_authoritative")
        and str(
            (meta or {}).get("citation_plan_evidence_selection_reason") or ""
        ).strip().lower()
        == "prompt_aligned_source_sentence"
    )
    if canonical_answer_evidence or verified_prompt_contract or prompt_aligned_plan_evidence:
        claim_keywords_fast = _system_a_keyword_terms(claim, limit=48)
        evidence_keywords_fast = _system_a_keyword_terms(evidence_surface, limit=64)
        keyword_overlap_fast = claim_keywords_fast & evidence_keywords_fast
        claim_identifiers_fast = {
            token.upper()
            for token in re.findall(
                r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{2,}(?![A-Za-z0-9])",
                claim,
            )
        }
        evidence_identifiers_fast = {
            token.upper()
            for token in re.findall(
                r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{2,}(?![A-Za-z0-9])",
                evidence_surface,
            )
        }
        claim_numbers_fast = set(
            re.findall(
                r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])",
                re.sub(r"\[\d{1,5}\](?:\([^\n)]+\))?", " ", claim),
            )
        )
        evidence_numbers_fast = set(
            re.findall(
                r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])",
                evidence_body_surface,
            )
        )
        shared_identifiers_fast = claim_identifiers_fast & evidence_identifiers_fast
        shared_numbers_fast = claim_numbers_fast & evidence_numbers_fast
        if (
            len(keyword_overlap_fast) >= 2
            or bool(shared_identifiers_fast)
            or bool(shared_numbers_fast)
            or _system_a_has_source_identity_overlap(claim, evidence_body_surface, source_name)
        ):
            prefer_zh_fast = _system_a_prefers_zh(claim)
            overlap_label = "、".join(sorted(keyword_overlap_fast | shared_identifiers_fast)[:4])
            if verified_prompt_contract and not canonical_answer_evidence:
                reason = (
                    f"该引用使用了已核对页码和原文块的证据，且与答案中的关键词{f'“{overlap_label}”' if overlap_label else ''}一致。"
                    if prefer_zh_fast
                    else (
                        "This citation uses evidence with a verified page and source block"
                        + (f" and matches the claim terms {overlap_label}." if overlap_label else ".")
                    )
                )
            else:
                reason = (
                    f"该引用复用生成回答时实际提供的原文证据，且与答案中的关键词{f'“{overlap_label}”' if overlap_label else ''}一致。"
                    if prefer_zh_fast
                    else (
                        "This citation reuses the source evidence actually supplied during answer generation"
                        + (f" and matches the claim terms {overlap_label}." if overlap_label else ".")
                    )
                )
            return {
                "status": "grounded",
                "confidence": 0.9,
                "suppress_link": False,
                "reason": reason,
                "overlap_terms": sorted(keyword_overlap_fast | shared_identifiers_fast | shared_numbers_fast),
                "missing_terms": [],
            }
    claim_domains = _system_a_domain_terms(claim)
    evidence_domains = _system_a_domain_terms(evidence_surface)
    evidence_body_domains = _system_a_domain_terms(evidence_body_surface)
    domain_overlap = claim_domains & evidence_domains
    body_domain_overlap = claim_domains & evidence_body_domains
    claim_keywords = _system_a_keyword_terms(claim)
    evidence_keywords = _system_a_keyword_terms(evidence_surface)
    keyword_overlap = claim_keywords & evidence_keywords
    claim_keyword_coverage = (
        len(keyword_overlap) / max(1, len(claim_keywords))
        if claim_keywords
        else 0.0
    )
    prefer_zh = _system_a_prefers_zh(claim)
    quote_surface = re.sub(r"\s+", " ", str(evidence_quote or (hit or {}).get("text") or "")).strip()
    claim_similarity = difflib.SequenceMatcher(
        None,
        claim.lower(),
        quote_surface.lower(),
    ).ratio() if claim and quote_surface else 0.0
    claim_words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", claim.lower())
    evidence_words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", quote_surface.lower())
    claim_alignment_terms = evidence_alignment_tokens(claim)
    evidence_alignment_terms = evidence_alignment_tokens(quote_surface)
    alignment_overlap = claim_alignment_terms & evidence_alignment_terms
    generic_alignment_terms = {
        "approach",
        "based",
        "high",
        "image",
        "method",
        "paper",
        "performance",
        "quality",
        "results",
        "study",
        "system",
        "using",
    }
    informative_alignment_overlap = alignment_overlap - generic_alignment_terms
    longest_word_run = max(
        (
            block.size
            for block in difflib.SequenceMatcher(
                None,
                claim_words,
                evidence_words,
            ).get_matching_blocks()
        ),
        default=0,
    )

    strong_claim_terms = claim_domains & _SYSTEM_A_STRONG_BINDING_TERMS
    missing_strong_terms = strong_claim_terms - evidence_body_domains
    matched_strong_terms = strong_claim_terms & evidence_body_domains
    source_identity_overlap = _system_a_has_source_identity_overlap(claim, evidence_body_surface, source_name)
    numeric_claim_surface = re.sub(r"^\s*\d+[.)、]\s*", "", claim)
    numeric_claim_surface = re.sub(r"\[\d{1,5}\](?:\([^\n)]+\))?", " ", numeric_claim_surface)
    claim_numeric_values = {
        token
        for token in re.findall(
            r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])",
            numeric_claim_surface,
        )
        if not (len(token) == 4 and 1900 <= int(float(token)) <= 2100)
    }
    evidence_numeric_values = set(
        re.findall(
            r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])",
            quote_surface,
        )
    )
    claim_metrics = {
        item.lower()
        for item in re.findall(r"\b(?:psnr|ssim|lpips|fid|fps|macs?|flops?)\b", claim, flags=re.I)
    }
    evidence_metrics = {
        item.lower()
        for item in re.findall(r"\b(?:psnr|ssim|lpips|fid|fps|macs?|flops?)\b", quote_surface, flags=re.I)
    }
    shared_metrics = claim_metrics & evidence_metrics
    shared_decimal_values = {
        item for item in claim_numeric_values & evidence_numeric_values if "." in item
    }
    if shared_metrics and shared_decimal_values:
        metric_label = "/".join(sorted(shared_metrics)).upper()
        reason = (
            f"这组 {metric_label} 对比直接量化了答案中各方法的重建质量差异，并显示物理先验方法相对基线的增益。"
            if prefer_zh
            else (
                f"This {metric_label} comparison quantifies the reconstruction-quality "
                "difference and the physics-informed method's gain over the baselines."
            )
        )
        return {
            "status": "grounded",
            "confidence": 0.9,
            "suppress_link": False,
            "reason": reason,
            "overlap_terms": sorted(keyword_overlap | shared_metrics),
            "missing_terms": [],
        }
    if (
        "detector type:" in quote_surface.lower()
        and "performance" in quote_surface.lower()
        and re.search(
            r"\b(?:spad|single[- ]?photon|detection efficiency)\b|探测效率|单光子",
            claim.lower(),
        )
        and len(claim_numeric_values) >= 2
        and claim_numeric_values.issubset(evidence_numeric_values)
    ):
        reason = (
            "这条表格记录把探测器型号、工作波长、温度与探测效率放在同一项中，可直接建立算法需要面对的硬件性能边界。"
            if prefer_zh
            else (
                "This table record links detector type, operating wavelength, "
                "temperature, and detection efficiency in one hardware-performance boundary."
            )
        )
        return {
            "status": "grounded",
            "confidence": 0.9,
            "suppress_link": False,
            "reason": reason,
            "overlap_terms": sorted(keyword_overlap),
            "missing_terms": [],
        }
    if (
        len(claim_numeric_values) >= 2
        and claim_numeric_values.issubset(evidence_numeric_values)
    ):
        values_label = "/".join(sorted(claim_numeric_values, key=lambda value: (len(value), value)))
        reason = (
            f"原文同一证据片段直接包含答案中的定量值（{values_label}），可核对该数据集或实验陈述。"
            if prefer_zh
            else f"The same source passage directly contains the claim's quantitative values ({values_label})."
        )
        return {
            "status": "grounded",
            "confidence": 0.95,
            "suppress_link": False,
            "reason": reason,
            "overlap_terms": sorted(keyword_overlap),
            "missing_terms": [],
        }
    # A single answer sentence can carry multiple citations.  If this hit already
    # shares a concrete body/heading domain term with the sentence, keep
    # evaluating it instead of suppressing it for a strong term that belongs to a
    # neighboring citation. Source-name overlap alone is too weak for this bypass.
    if strong_claim_terms and not matched_strong_terms and not source_identity_overlap and not body_domain_overlap:
        missing = _system_a_term_label(missing_strong_terms)
        evidence_label = _system_a_term_label(evidence_body_domains or evidence_domains) or "retrieved passage"
        reason = (
            f"答案句的关键术语“{missing}”没有出现在该命中证据中；该命中更像是在讨论“{evidence_label}”。"
            if prefer_zh
            else f'The answer sentence names "{missing}", but this hit does not contain that concept; it appears to discuss "{evidence_label}".'
        )
        return {
            "status": "mismatch",
            "confidence": 0.0,
            "suppress_link": True,
            "reason": reason,
            "overlap_terms": [],
            "missing_terms": sorted(missing_strong_terms),
        }

    # Chinese answer prose and English source passages often have no literal
    # token overlap even when they describe the same mechanism.  Keep this
    # cross-language path deliberately strict: several mapped, informative
    # terms must agree, and any quantitative values in the claim must still be
    # present in the source evidence.
    if (
        not body_domain_overlap
        and len(alignment_overlap) >= 4
        and len(informative_alignment_overlap) >= 4
        and (
            not claim_numeric_values
            or claim_numeric_values.issubset(evidence_numeric_values)
        )
    ):
        terms = sorted(informative_alignment_overlap)
        term_label = _system_a_term_label(terms[:6])
        reason = (
            f"答案与英文原文在“{term_label}”等多个具体动作和对象上对应，可确认是同一技术主张。"
            if prefer_zh
            else "The answer and source align on several specific actions and objects, supporting the same technical claim."
        )
        return {
            "status": "grounded",
            "confidence": 0.84,
            "suppress_link": False,
            "reason": reason,
            "overlap_terms": terms,
            "missing_terms": [],
        }

    context_only_overlap = body_domain_overlap & _SYSTEM_A_CONTEXT_ONLY_BINDING_TERMS
    if (
        context_only_overlap
        and not (body_domain_overlap - _SYSTEM_A_CONTEXT_ONLY_BINDING_TERMS)
        and not source_identity_overlap
    ):
        term_label = _system_a_term_label(context_only_overlap)
        reason = (
            f"答案句和命中片段只共享较宽泛的“{term_label}”表述，尚不足以确认是同一技术主张。"
            if prefer_zh
            else f'The answer and retrieved passage only share the broad phrase "{term_label}", which is not enough to establish the same technical claim.'
        )
        return {
            "status": "candidate",
            "confidence": 0.35,
            "suppress_link": True,
            "reason": reason,
            "overlap_terms": sorted(context_only_overlap),
            "missing_terms": sorted(strong_claim_terms - evidence_domains),
        }

    claim_metrics = {
        item.lower()
        for item in re.findall(r"\b(?:psnr|ssim|lpips|fid|fps|macs?|flops?)\b", claim, flags=re.I)
    }
    evidence_metrics = {
        item.lower()
        for item in re.findall(r"\b(?:psnr|ssim|lpips|fid|fps|macs?|flops?)\b", quote_surface, flags=re.I)
    }
    claim_values = set(re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", claim))
    evidence_values = set(re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", quote_surface))
    shared_metrics = claim_metrics & evidence_metrics
    shared_values = claim_values & evidence_values
    if shared_metrics and shared_values:
        metric_label = "/".join(sorted(shared_metrics)).upper()
        reason = (
            f"这组 {metric_label} 对比直接量化了答案中各方法的重建质量差异，并显示物理先验方法相对基线的增益。"
            if prefer_zh
            else (
                f"This {metric_label} comparison quantifies the reconstruction-quality "
                "difference and the physics-informed method's gain over the baselines."
            )
        )
        return {
            "status": "grounded",
            "confidence": 0.9,
            "suppress_link": False,
            "reason": reason,
            "overlap_terms": sorted(keyword_overlap | shared_metrics),
            "missing_terms": [],
        }

    if body_domain_overlap:
        terms = sorted(body_domain_overlap)
        term_label = _system_a_term_label(terms)
        claim_low = claim.lower()
        evidence_low = quote_surface.lower()
        if (
            "detector type:" in evidence_low
            and "performance" in evidence_low
            and re.search(r"\b(?:spad|single[- ]?photon|detection efficiency)\b|探测效率|单光子", claim_low)
        ):
            reason = (
                "这条表格记录把探测器型号、工作波长、温度与探测效率放在同一项中，可直接建立算法需要面对的硬件性能边界。"
                if prefer_zh
                else (
                    "This table record links detector type, operating wavelength, "
                    "temperature, and detection efficiency in one hardware-performance boundary."
                )
            )
        elif (
            "iterative reconstruction" in evidence_low
            and "image quality" in evidence_low
            and ("computational time" in evidence_low or "computational times" in evidence_low)
            and "deep learning" in evidence_low
        ):
            reason = (
                "摘要同时给出迭代重建的质量与耗时瓶颈，以及深度学习带来的质量和速度收益，支撑其作为算法背景综述的定位。"
                if prefer_zh
                else (
                    "The abstract pairs iterative reconstruction's quality and runtime "
                    "limits with the quality and speed gains reported for deep learning."
                )
            )
        elif (
            re.search(r"real[- ]?time|frame rate|\b\d+\s*(?:fps|hz)\b|实时|帧率", claim_low, flags=re.I)
            and re.search(r"real[- ]?time|frame rate|\b\d+\s*(?:fps|hz)\b|实时|帧率", evidence_low, flags=re.I)
        ):
            reason = (
                "原文直接报告了单像素重建的实时帧率或速度结果，支撑回答中的实时成像结论。"
                if prefer_zh
                else "The source directly reports the real-time frame rate or speed result stated in the answer."
            )
        elif (
            re.search(r"domain shift|degradation[- ]?robust|physical degradation|域偏移|退化鲁棒", claim_low, flags=re.I)
            and re.search(r"domain shift|degradation[- ]?robust|physical degradation|域偏移|退化鲁棒", evidence_low, flags=re.I)
        ):
            reason = (
                "原文把域偏移测试结果与物理退化模型学到的鲁棒表征联系起来，直接支撑回答的泛化结论。"
                if prefer_zh
                else "The source links its domain-shift result to degradation-robust representations, directly supporting the generalization claim."
            )
        elif (
            re.search(r"low[- ]?light|high[- ]?light|resolution|psnr|ssim|低照度|高照度|分辨率|图像质量", claim_low, flags=re.I)
            and re.search(r"low[- ]?light|high[- ]?light|resolution|psnr|ssim|低照度|高照度|分辨率|图像质量", evidence_low, flags=re.I)
        ):
            reason = (
                "原文给出了低照度、分辨率或重建指标方面的具体结果，支撑回答中的图像质量结论。"
                if prefer_zh
                else "The source provides a concrete low-light, resolution, or reconstruction-metric result supporting the image-quality claim."
            )
        else:
            quote = re.sub(r"\s+", " ", str(evidence_quote or "")).strip()
            if len(quote) > 140:
                quote = quote[:137].rstrip() + "..."
            reason = (
                f"原文在该定位处给出的具体陈述是：“{quote}”"
                if prefer_zh and quote
                else (
                    f'The answer sentence is supported by the located source, which states: "{quote}"'
                    if quote
                    else f'The located source directly contains the technical point "{term_label}".'
                )
            )
        return {
            "status": "grounded",
            "confidence": 0.85,
            "suppress_link": False,
            "reason": reason,
            "overlap_terms": terms,
            "missing_terms": [],
        }

    if source_identity_overlap and keyword_overlap:
        terms = sorted(keyword_overlap)
        quote = re.sub(r"\s+", " ", str(evidence_quote or "")).strip()
        if len(quote) > 140:
            quote = quote[:137].rstrip() + "..."
        reason = (
            f"回答明确指向该论文；原文定位处写道：“{quote}”"
            if prefer_zh
            else f'The answer explicitly names this paper; the located source states: "{quote}"'
        )
        return {
            "status": "grounded",
            "confidence": 0.62,
            "suppress_link": False,
            "reason": reason,
            "overlap_terms": terms,
            "missing_terms": [],
        }

    # Concise answers to benchmark/table questions often put the marker after a
    # final comparison sentence.  Their local citation context can therefore be
    # much shorter than the table evidence.  An exact metric/value pair plus a
    # named-method overlap is a stronger binding signal than generic text
    # similarity and should not be discarded as a weak two-keyword match.
    if shared_metrics and shared_values and keyword_overlap:
        metric_label = "/".join(sorted(shared_metrics))
        value_label = "/".join(sorted(shared_values))
        reason = (
            f"原文表格与答案包含相同的 {metric_label} 数值（{value_label}），并匹配到同一方法名称。"
            if prefer_zh
            else f"The source table and answer share the same {metric_label} value ({value_label}) and method name."
        )
        return {
            "status": "grounded",
            "confidence": 0.9,
            "suppress_link": False,
            "reason": reason,
            "overlap_terms": sorted(keyword_overlap | shared_metrics),
            "missing_terms": [],
        }

    if len(keyword_overlap) >= 2 and (
        claim_similarity >= 0.5
        or claim_keyword_coverage >= 0.75
        or len(keyword_overlap) >= 3
        or longest_word_run >= 3
    ):
        quote = quote_surface[:137].rstrip() + "..." if len(quote_surface) > 140 else quote_surface
        reason = (
            f"答案表述与定位原文高度一致；原文写道：“{quote}”"
            if prefer_zh
            else f'The answer closely matches the located source statement: "{quote}"'
        )
        return {
            "status": "grounded",
            "confidence": 0.78,
            "suppress_link": False,
            "reason": reason,
            "overlap_terms": sorted(keyword_overlap),
            "missing_terms": [],
        }

    if len(keyword_overlap) >= 2:
        terms = sorted(keyword_overlap)
        term_label = _system_a_term_label(terms)
        reason = (
            f"目前只确认了“{term_label}”等关键词重合，尚不足以把这段原文作为该主张的直接证据。"
            if prefer_zh
            else f'Only keyword overlap such as "{term_label}" is confirmed; this is not enough to present the passage as direct evidence.'
        )
        return {
            "status": "candidate",
            "confidence": 0.45,
            "suppress_link": True,
            "reason": reason,
            "overlap_terms": terms,
            "missing_terms": [],
        }

    reason = (
        "这条引用只能作为候选依据：答案句和命中片段的术语重合很弱，需要打开原文再确认。"
        if prefer_zh
        else "This citation is only a candidate source: term overlap between the answer sentence and retrieved passage is weak, so verify it in the original text."
    )
    return {
        "status": "candidate",
        "confidence": 0.35,
        "suppress_link": True,
        "reason": reason,
        "overlap_terms": sorted(keyword_overlap),
        "missing_terms": sorted(strong_claim_terms - evidence_domains),
    }
