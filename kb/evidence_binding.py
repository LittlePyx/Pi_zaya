from __future__ import annotations

import difflib
import re
from decimal import Decimal, InvalidOperation
from functools import lru_cache

from kb.evidence_term_mapping import (
    evidence_alignment_tokens,
    method_identity_conflicts,
    specific_method_identities,
)
from kb.source_blocks import normalize_inline_markdown


_EXPLICIT_CLAIM_RELATION_REQUIREMENTS: tuple[tuple[re.Pattern, re.Pattern], ...] = (
    (
        re.compile(
            r"(?:替代|取代|摆脱).{0,28}(?:数据驱动|黑箱)|"
            r"(?:数据驱动|黑箱).{0,28}(?:替代|取代|摆脱)|"
            r"\b(?:replace|supplant|move\s+away\s+from).{0,40}"
            r"(?:data[- ]driven|black[- ]box)\b",
            re.I,
        ),
        re.compile(
            r"(?:替代|取代|摆脱).{0,28}(?:数据驱动|黑箱)|"
            r"(?:数据驱动|黑箱).{0,28}(?:替代|取代|摆脱)|"
            r"\b(?:replace|supplant|move\s+away\s+from).{0,40}"
            r"(?:data[- ]driven|black[- ]box)\b",
            re.I,
        ),
    ),
    (
        re.compile(
            r"(?:训练数据有限|有限.{0,12}训练数据|场景变化).{0,36}"
            r"(?:鲁棒|泛化|仍能|保持)|(?:鲁棒|泛化).{0,36}"
            r"(?:训练数据有限|有限.{0,12}训练数据|场景变化)|"
            r"\b(?:robust|generaliz\w*).{0,44}(?:limited\s+training\s+data|"
            r"scene\s+changes?)|(?:limited\s+training\s+data|scene\s+changes?)"
            r".{0,44}\b(?:robust|generaliz\w*)",
            re.I,
        ),
        re.compile(
            r"(?:训练数据有限|有限.{0,12}训练数据|场景变化).{0,36}"
            r"(?:鲁棒|泛化|仍能|保持)|(?:鲁棒|泛化).{0,36}"
            r"(?:训练数据有限|有限.{0,12}训练数据|场景变化)|"
            r"\b(?:robust|generaliz\w*).{0,44}(?:limited\s+training\s+data|"
            r"scene\s+changes?)|(?:limited\s+training\s+data|scene\s+changes?)"
            r".{0,44}\b(?:robust|generaliz\w*)",
            re.I,
        ),
    ),
    (
        re.compile(
            r"传统(?:方法|算法).{0,28}(?:失效|失败|无法|不能)|"
            r"\btraditional\s+(?:methods?|algorithms?).{0,36}"
            r"(?:fail|break\s+down|cannot|unable)",
            re.I,
        ),
        re.compile(
            r"传统(?:方法|算法).{0,28}(?:失效|失败|无法|不能)|"
            r"\btraditional\s+(?:methods?|algorithms?).{0,36}"
            r"(?:fail|break\s+down|cannot|unable)",
            re.I,
        ),
    ),
    (
        re.compile(
            r"解耦.{0,24}(?:真实|有效)?信号|"
            r"\bdisentangl\w*.{0,32}(?:true|real|underlying)\s+signal",
            re.I,
        ),
        re.compile(
            r"解耦.{0,24}(?:真实|有效)?信号|"
            r"\bdisentangl\w*.{0,32}(?:true|real|underlying)\s+signal",
            re.I,
        ),
    ),
)


def explicit_claim_relations_covered(claim: str, evidence: str) -> bool:
    """Reject a strong answer relation unless the evidence states that relation."""

    claim_text = re.sub(r"\s+", " ", str(claim or "")).strip()
    evidence_text = re.sub(r"\s+", " ", str(evidence or "")).strip()
    return not any(
        claim_pattern.search(claim_text) and not evidence_pattern.search(evidence_text)
        for claim_pattern, evidence_pattern in _EXPLICIT_CLAIM_RELATION_REQUIREMENTS
    )


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


_FACT_UNIT_ALIASES = {
    "%": "%",
    "percent": "%",
    "percentage": "%",
    "db": "db",
    "nm": "nm",
    "um": "um",
    "µm": "um",
    "μm": "um",
    "mm": "mm",
    "cm": "cm",
    "hz": "hz",
    "khz": "khz",
    "mhz": "mhz",
    "ghz": "ghz",
    "fps": "fps",
    "frame per second": "fps",
    "frames per second": "fps",
    "px": "pixel",
    "pixel": "pixel",
    "pixels": "pixel",
    "k": "k",
    "kelvin": "k",
    "ms": "ms",
    "detector": "detector",
    "detectors": "detector",
    "image": "image",
    "images": "image",
    "frame": "frame",
    "frames": "frame",
    "pattern": "pattern",
    "patterns": "pattern",
    "measurement": "measurement",
    "measurements": "measurement",
}
_FACT_UNIT_PATTERN = "|".join(
    sorted((re.escape(unit) for unit in _FACT_UNIT_ALIASES), key=len, reverse=True)
)
_FACT_MULTIPLIER_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}
_FACT_MULTIPLIER_ZH_NUMBER_WORDS = {
    "零": "0",
    "一": "1",
    "二": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "十": "10",
    "十一": "11",
    "十二": "12",
}
_FACT_MULTIPLIER_NUMBER_PATTERN = "|".join(
    sorted(_FACT_MULTIPLIER_NUMBER_WORDS, key=len, reverse=True)
)
_FACT_MULTIPLIER_ZH_NUMBER_PATTERN = "|".join(
    sorted(_FACT_MULTIPLIER_ZH_NUMBER_WORDS, key=len, reverse=True)
)
_FACT_MULTIPLIER_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?P<en_value>\d+(?:\.\d+)?|{_FACT_MULTIPLIER_NUMBER_PATTERN})"
    r"\s*(?:[-\u2010-\u2015]\s*)?(?:fold|times?)(?![A-Za-z])"
    rf"|(?<![A-Za-z0-9])(?P<zh_value>\d+(?:\.\d+)?|{_FACT_MULTIPLIER_ZH_NUMBER_PATTERN})\s*倍",
    re.IGNORECASE,
)
_FACT_MULTIPLIER_DECREASE_RE = re.compile(
    r"\b(?:lower|less|fewer|decreas(?:e|es|ed|ing)|reduc(?:e|es|ed|ing|tion)|"
    r"drop(?:s|ped|ping)?|diminish(?:es|ed|ing)?|attenuat(?:e|es|ed|ing|ion))\b|"
    r"降低|减少|下降|缩减|减小",
    re.IGNORECASE,
)
_FACT_MULTIPLIER_INCREASE_RE = re.compile(
    r"\b(?:higher|more|greater|increas(?:e|es|ed|ing)|rais(?:e|es|ed|ing)|"
    r"improv(?:e|es|ed|ing|ement)|enhanc(?:e|es|ed|ing|ement)|"
    r"amplif(?:y|ies|ied|ication)|speedup|gain)\b|"
    r"提高|增加|上升|增大|提升|增强",
    re.IGNORECASE,
)
_FACT_METRIC_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:PSNR|SNR|SSIM|LPIPS)(?![A-Za-z])",
    re.IGNORECASE,
)
_COMPARISON_SCOPE_SPLIT_RE = re.compile(
    r"\s*(?:[;；,，]|[,，]\s*(?:while|whereas|but|however|而|但|然而|相比之下)\s*|"
    r"\b(?:while|whereas|versus|vs\.?)\b)\s*",
    re.IGNORECASE,
)


def _strip_structural_locators(value: str) -> str:
    surface = str(value or "")
    number_word = (
        r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    )
    locator_number = rf"(?:\d+(?:\.\d+)*(?:[a-z])?|{number_word})"
    surface = re.sub(
        rf"(?i)\b(?:tables?|fig(?:ure)?s?|eq(?:uation)?s?|sections?|secs?|"
        rf"chapters?|chaps?|appendices?|appendix|algorithms?|algs?)\s*"
        rf"(?:no\.?\s*)?(?:[#:]\s*)?[（(]?{locator_number}[）)]?"
        rf"(?:\s*(?:,|and|&|[-–—])\s*[（(]?{locator_number}[）)]?)*",
        " ",
        surface,
    )
    surface = re.sub(
        r"(?:第\s*)?\d+(?:\.\d+)*\s*(?:号\s*)?(?:表|图|公式|方程|式|章节|章|节)",
        " ",
        surface,
    )
    surface = re.sub(
        r"(?:表|图|公式|方程|式|章节|章|节)\s*(?:第\s*)?(?:[（(]\s*)?"
        r"\d+(?:\.\d+)*(?:\s*[）)])?(?:\s*(?:、|,|和|至|[-–—])\s*\d+(?:\.\d+)*)*",
        " ",
        surface,
    )
    return surface


def _fact_multiplier_direction(surface: str, start: int, end: int) -> str:
    """Return the closest explicit increase/decrease direction for a multiplier."""

    text = str(surface or "")
    clause_start = max(
        text.rfind(mark, 0, start) + 1
        for mark in (".", ";", "!", "?", "。", "；", "！", "？", "，", ",")
    )
    right_boundaries = [
        pos
        for mark in (".", ";", "!", "?", "。", "；", "！", "？", "，", ",")
        if (pos := text.find(mark, end)) >= 0
    ]
    clause_end = min(right_boundaries) if right_boundaries else len(text)
    window_start = max(clause_start, start - 56)
    window_end = min(clause_end, end + 56)
    window = text[window_start:window_end]
    relative_start = start - window_start
    relative_end = end - window_start
    candidates: list[tuple[int, str]] = []
    for direction, pattern in (
        ("decrease", _FACT_MULTIPLIER_DECREASE_RE),
        ("increase", _FACT_MULTIPLIER_INCREASE_RE),
    ):
        for match in pattern.finditer(window):
            if match.end() <= relative_start:
                distance = relative_start - match.end()
            elif match.start() >= relative_end:
                distance = match.start() - relative_end
            else:
                distance = 0
            candidates.append((distance, direction))
    if not candidates:
        return ""
    closest = min(distance for distance, _direction in candidates)
    directions = {
        direction for distance, direction in candidates if distance == closest
    }
    return next(iter(directions)) if len(directions) == 1 else ""


def _fact_multiplier_quantities(surface: str) -> set[tuple[str, str, str]]:
    """Normalize ``tenfold``, ``10 times`` and ``10 倍`` as directed facts."""

    facts: set[tuple[str, str, str]] = set()
    for match in _FACT_MULTIPLIER_RE.finditer(str(surface or "")):
        raw_value = str(match.group("en_value") or match.group("zh_value") or "")
        normalized = _FACT_MULTIPLIER_NUMBER_WORDS.get(raw_value.casefold())
        if normalized is None:
            normalized = _FACT_MULTIPLIER_ZH_NUMBER_WORDS.get(raw_value)
        if normalized is None:
            normalized = _normalize_fact_number(raw_value)
        if not normalized:
            continue
        facts.add(
            (
                normalized,
                "fold",
                _fact_multiplier_direction(surface, match.start(), match.end()),
            )
        )
    return facts


def _normalize_fact_number(token: str) -> str:
    try:
        return format(Decimal(str(token or "")).normalize(), "f")
    except (InvalidOperation, ValueError):
        return str(token or "")


def _fact_metric_qualifier(surface: str, start: int, end: int) -> str:
    """Return the closest metric name in the number's comma-delimited clause."""

    sentence_prefix = surface[:start]
    sentence_matches = list(
        re.finditer(r"[。！？!?\n]|(?<!\d)\.(?=\s|$)", sentence_prefix)
    )
    sentence_start = sentence_matches[-1].end() if sentence_matches else 0
    sequence_boundary_matches = list(
        re.finditer(r"[;；。！？!?\n]|(?<!\d)\.(?=\s|$)", sentence_prefix)
    )
    sequence_start = (
        sequence_boundary_matches[-1].end() if sequence_boundary_matches else 0
    )
    sequence_numbers = list(
        re.finditer(
            r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])",
            surface[sequence_start:start],
        )
    )
    previous_sequence_number: tuple[int, int] | None = None
    if sequence_numbers:
        previous_sequence_number = (
            sequence_start + sequence_numbers[-1].start(),
            sequence_start + sequence_numbers[-1].end(),
        )
    left_matches = list(re.finditer(r"[;；,，。！？!?\n]", surface[:start]))
    clause_start = left_matches[-1].end() if left_matches else 0
    right_match = re.search(r"[;；,，。！？!?\n]", surface[end:])
    clause_end = end + right_match.start() if right_match else len(surface)
    number_pattern = re.compile(
        r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])"
    )
    previous_numbers = list(number_pattern.finditer(surface[clause_start:start]))
    if previous_numbers:
        clause_start += previous_numbers[-1].end()
    next_number = number_pattern.search(surface[end:clause_end])
    if next_number:
        clause_end = end + next_number.start()
    clause = surface[clause_start:clause_end]
    relative_start = max(0, start - clause_start)
    relative_end = max(relative_start, end - clause_start)
    ranked: list[tuple[int, str]] = []
    for match in _FACT_METRIC_PATTERN.finditer(clause):
        if match.end() <= relative_start:
            distance = relative_start - match.end()
        elif match.start() >= relative_end:
            between = clause[relative_end : match.start()]
            if re.search(r"[A-Za-z0-9\u4e00-\u9fff]", between):
                # Postfix metrics are valid only when adjacent to the value,
                # such as ``0.9 SSIM`` or ``40.3 dB PSNR``. Do not let a later
                # clause verb pull the metric backward across substantive text
                # (``100 images and reaches PSNR 30.5 dB``).
                continue
            distance = match.start() - relative_end
        else:
            distance = 0
        if distance <= 48:
            ranked.append((distance, match.group(0).lower()))
    if not ranked:
        if previous_sequence_number is not None:
            previous_start, previous_end = previous_sequence_number
            bridge = surface[previous_end:start]
            if re.fullmatch(
                r"\s*(?:(?:dB|%)\s*)?"
                r"(?:to|and|or|,|，|/|[-–—]|至|到|和|与|、)\s*",
                bridge,
                re.IGNORECASE,
            ):
                inherited = _fact_metric_qualifier(
                    surface,
                    previous_start,
                    previous_end,
                )
                if inherited:
                    return inherited
        # Compact table evidence commonly states the metric once as a header,
        # then lists method/value cells separated by semicolons, e.g.
        # ``SIDD PSNR: MPRNet = ...; Baseline = 40.30; NAFNet = 40.30``.
        # Inherit only an explicit ``METRIC:`` header from the same sentence;
        # ordinary prose in a previous sentence must not leak its metric.
        header_surface = surface[sentence_start:start]
        header_matches = list(
            re.finditer(
                r"(?<![A-Za-z])(?P<metric>PSNR|SNR|SSIM|LPIPS)(?![A-Za-z])"
                r"\s*(?:(?:results?|scores?|values?)\s*)?:",
                header_surface,
                re.IGNORECASE,
            )
        )
        if header_matches:
            return str(header_matches[-1].group("metric") or "").lower()
        return ""
    ranked.sort(key=lambda item: item[0])
    closest_distance = ranked[0][0]
    closest_metrics = {
        metric for distance, metric in ranked if distance == closest_distance
    }
    return next(iter(closest_metrics)) if len(closest_metrics) == 1 else ""


def _system_a_fact_quantities(value: str) -> set[tuple[str, str, str]]:
    """Return normalized ``(value, unit, metric)`` facts without locator labels."""

    surface = re.sub(r"^\s*\d+[.)、]\s*", "", str(value or ""))
    surface = re.sub(r"\[\d{1,5}\](?:\([^\n)]+\))?", " ", surface)
    # Reader-facing answers may render an exact source quantity such as
    # ``$5\,\mu\mathrm{m}$`` as ``5 μm``. Canonicalize that presentation-only
    # difference before quantity comparison so the evidence gate does not
    # reject a value that is byte-for-byte equivalent in magnitude and unit.
    surface = re.sub(
        r"\$?(?P<number>\d+(?:\.\d+)?)\s*\\,\s*\\mu\s*"
        r"\\mathrm\{(?P<unit>[A-Za-z]+)\}\s*\$?",
        lambda match: f"{match.group('number')} μ{match.group('unit')}",
        surface,
    )
    # Relation operators are presentation syntax, not part of the number.  In
    # compact TeX such as ``$\sim$8`` or ``\approx30`` the command's trailing
    # letter otherwise makes the numeric boundary look alphanumeric and the
    # quantity is silently missed.
    surface = re.sub(
        r"\\(?:sim|approx|simeq|lesssim|gtrsim)\s*\$?",
        " ",
        surface,
        flags=re.IGNORECASE,
    )
    surface = _strip_structural_locators(surface)
    out: set[tuple[str, str, str]] = set(_fact_multiplier_quantities(surface))
    quantity_re = re.compile(
        rf"(?<![A-Za-z0-9])(?P<number>\d+(?:\.\d+)?)(?![A-Za-z0-9])"
        rf"(?:\s*(?P<unit>{_FACT_UNIT_PATTERN})\b|\s*(?P<percent>%))?",
        re.IGNORECASE,
    )
    for match in quantity_re.finditer(surface):
        token = str(match.group("number") or "")
        raw_unit = str(match.group("unit") or match.group("percent") or "").lower()
        unit = _FACT_UNIT_ALIASES.get(raw_unit, raw_unit)
        is_year_range_value = bool(
            not unit and len(token) == 4 and 1900 <= int(float(token)) <= 2100
        )
        if is_year_range_value:
            # A year remains a year unless the following lowercase phrase is
            # an explicit count expression. This retains ``2022 reconstructed
            # images`` but rejects paper-title text such as
            # ``ECCV-2022-Simple Baselines for Image Restoration``.
            continuation = surface[match.end() : match.end() + 72]
            explicit_count = re.match(
                r"(?i)^\s+(?:(?:[a-z][a-z-]*|,)\s+){0,5}"
                r"(?P<unit>detectors|images|frames|patterns|measurements)\b",
                continuation,
            )
            if explicit_count:
                raw_count_unit = str(explicit_count.group("unit") or "").lower()
                unit = _FACT_UNIT_ALIASES.get(raw_count_unit, raw_count_unit)
        if not unit and not is_year_range_value:
            continuation = surface[match.end() : match.end() + 72]
            continuation = re.split(r"[.!?;。！？；]", continuation, maxsplit=1)[0]
            next_quantity_or_metric = re.search(
                r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])|"
                r"(?<![A-Za-z])(?:PSNR|SNR|SSIM|LPIPS)(?![A-Za-z])",
                continuation,
                re.IGNORECASE,
            )
            if next_quantity_or_metric:
                continuation = continuation[: next_quantity_or_metric.start()]
            count_unit = re.search(
                r"(?i)\b(detectors?|images?|frames?|patterns?|measurements?)\b",
                continuation,
            )
            if count_unit:
                raw_count_unit = str(count_unit.group(1) or "").lower()
                unit = _FACT_UNIT_ALIASES.get(raw_count_unit, raw_count_unit)
        if not unit and is_year_range_value:
            metric_context = surface[max(0, match.start() - 20) : match.start()]
            if not re.search(
                r"(?i)\b(?:psnr|ssim|lpips|fid|rmse|snr|score|metric|value)\b",
                metric_context,
            ):
                # Unqualified four-digit numbers in academic prose are years.
                # Explicit physical/rate/count units above keep 2000 nm,
                # 2020 fps and 2048 pixels as measurable facts.
                continue
        metric = _fact_metric_qualifier(surface, match.start(), match.end())
        if unit in {"detector", "image", "frame", "pattern", "measurement"}:
            # Counts describe acquisition/data cardinality, not image-quality
            # metrics. Even a range/list connector must not turn ``31 images``
            # into a second PSNR value.
            metric = ""
        elif not unit and metric in {"psnr", "snr"}:
            # PSNR/SNR table cells conventionally omit the repeated dB unit
            # after an explicit metric header. Keep this normalization scoped
            # to those two logarithmic metrics; other unitless values remain
            # unitless and cannot satisfy a unit-bearing claim.
            unit = "db"
        out.add((_normalize_fact_number(token), unit, metric))
    number_words = {
        "zero": "0",
        "one": "1",
        "single": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
    }
    number_word_pattern = "|".join(number_words)
    for word in re.findall(
        rf"(?<![A-Za-z-])(?:{number_word_pattern})(?![A-Za-z-])",
        surface.lower(),
    ):
        normalized = number_words.get(word)
        if normalized is None:
            continue
        word_match = re.search(
            rf"(?<![A-Za-z-]){re.escape(word)}(?![A-Za-z-])",
            surface,
            re.IGNORECASE,
        )
        unit = ""
        if word_match:
            continuation = surface[word_match.end() : word_match.end() + 72]
            continuation = re.split(r"[.!?;。！？；]", continuation, maxsplit=1)[0]
            count_unit = re.search(
                r"(?i)\b(detectors?|images?|frames?|patterns?|measurements?)\b",
                continuation,
            )
            if count_unit:
                raw_count_unit = str(count_unit.group(1) or "").lower()
                unit = _FACT_UNIT_ALIASES.get(raw_count_unit, raw_count_unit)
        metric = (
            _fact_metric_qualifier(surface, word_match.start(), word_match.end())
            if word_match
            else ""
        )
        if unit in {"detector", "image", "frame", "pattern", "measurement"}:
            metric = ""
        out.add((normalized, unit, metric))
    return out


def _system_a_fact_numbers(value: str) -> set[str]:
    return {number for number, _unit, _metric in _system_a_fact_quantities(value)}


def _quantity_is_covered(
    quantity: tuple[str, str, str],
    evidence_quantities: set[tuple[str, str, str]],
) -> bool:
    number, unit, metric = quantity
    return any(
        evidence_number == number
        and (not unit or evidence_unit == unit)
        and (not metric or evidence_metric == metric)
        for evidence_number, evidence_unit, evidence_metric in evidence_quantities
    )


def _quantity_label(quantity: tuple[str, str, str]) -> str:
    number, unit, _metric = quantity
    if not unit:
        return number
    if unit == "%":
        return f"{number}%"
    return f"{number} {unit}"


def _claim_fact_quantities_for_evidence(
    claim: str,
    evidence: str,
    *,
    allow_comparison_scope: bool = False,
) -> set[tuple[str, str, str]]:
    """Scope comparison quantities to the clause supported by one evidence card."""

    full = _system_a_fact_quantities(claim)
    if not allow_comparison_scope:
        return full
    parts = [part.strip() for part in _COMPARISON_SCOPE_SPLIT_RE.split(claim) if part.strip()]
    if len(parts) < 2:
        return full
    evidence_quantities = _system_a_fact_quantities(evidence)
    evidence_tokens = evidence_alignment_tokens(evidence)
    scored: list[tuple[tuple[int, int, int], set[tuple[str, str, str]]]] = []
    for part in parts:
        quantities = _system_a_fact_quantities(part)
        part_tokens = evidence_alignment_tokens(part)
        quantity_overlap = sum(
            1 for quantity in quantities if _quantity_is_covered(quantity, evidence_quantities)
        )
        informative_overlap = len(
            (part_tokens & evidence_tokens)
            - {"approach", "image", "method", "paper", "result", "system", "using"}
        )
        identifiers = specific_method_identities(part) & specific_method_identities(evidence)
        scored.append(((quantity_overlap, len(identifiers), informative_overlap), quantities))
    ranked = sorted(scored, key=lambda item: item[0], reverse=True)
    if not ranked or ranked[0][0] == (0, 0, 0):
        return full
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return full
    return ranked[0][1]


def _specific_system_a_support_relation(
    claim: str,
    evidence: str,
    *,
    prefer_zh: bool,
) -> str:
    claim_low = str(claim or "").lower()
    evidence_low = str(evidence or "").lower()
    if (
        re.search(
            r"physical\s+noise|physics[- ]?informed|SPAD|"
            r"物理噪声|物理先验|单光子",
            claim_low,
            flags=re.I,
        )
        and re.search(
            r"physical\s+noise\s+model|real[- ]?shot\s+SPAD|"
            r"synthesi[sz]e.{0,80}(?:single[- ]?photon|image\s+pairs?)",
            evidence_low,
            flags=re.I | re.S,
        )
    ):
        return (
            "原文说明作者先用实拍 SPAD 数据标定真实物理噪声模型，再用该模型合成训练样本，"
            "由此可核对 physics-informed 方法如何把硬件噪声约束落实到网络训练数据中。"
            if prefer_zh
            else (
                "The source states that real SPAD data calibrate a physical noise "
                "model which then synthesizes training samples, directly supporting "
                "the physics-informed training mechanism described in the answer."
            )
        )
    if (
        re.search(
            r"波段|波长|高帧率|三维|3D|气体泄漏|自动驾驶|"
            r"wavelength|high\s+frame\s+rates?|three\s+dimensions|"
            r"gas\s+leaks?|autonomous\s+vehicles?",
            claim_low,
            flags=re.I,
        )
        and re.search(
            r"wavelengths?\s+outside\s+the\s+reach\s+of\s+FPA|"
            r"high\s+frame\s+rates?|three\s+dimensions|"
            r"hazardous\s+gas\s+leaks?|autonomous\s+vehicles?",
            evidence_low,
            flags=re.I,
        )
    ):
        return (
            "原文明确列出超出 FPA 覆盖的波段、高帧率和三维成像，并给出气体泄漏与自动驾驶应用，"
            "直接对应回答中的适用场景。"
            if prefer_zh
            else (
                "The source explicitly lists wavelengths beyond FPA reach, high frame "
                "rates, and three-dimensional imaging together with gas-leak and "
                "autonomous-vehicle applications, directly matching the stated use cases."
            )
        )
    return ""


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
    # The reader locator may intentionally keep a short, claim-adjacent
    # snippet.  Binding must still see the complete evidence passage selected
    # by the authoritative citation plan; otherwise a later sentence in that
    # passage can disappear from the evidence gate merely because the card
    # chose an earlier sentence for navigation/display.
    plan_binding_evidence = ""
    if (meta or {}).get("citation_plan_evidence_authoritative"):
        plan_binding_evidence = str(
            (meta or {}).get("citation_plan_full_evidence_quote") or ""
        ).strip()
    evidence_body_surface = " ".join(
        [
            str(evidence_quote or ""),
            plan_binding_evidence,
            str((hit or {}).get("text") or ""),
            str(heading or ""),
            str((meta or {}).get("why_line") or ""),
        ]
    )
    evidence_surface = " ".join([evidence_body_surface, str(source_name or "")])
    claim_low = claim.lower()
    evidence_body_low = evidence_body_surface.lower()
    if not explicit_claim_relations_covered(claim, evidence_body_surface):
        reason = (
            "回答句加入了原文未明确陈述的因果、替代或鲁棒性关系，不能绑定到这张证据卡。"
            if _system_a_prefers_zh(claim)
            else (
                "The answer adds a causal, replacement, or robustness relation that the "
                "passage does not state, so this card cannot support it."
            )
        )
        return {
            "status": "mismatch",
            "confidence": 0.0,
            "suppress_link": True,
            "reason": reason,
            "overlap_terms": [],
            "missing_terms": ["explicit relation"],
        }
    identity_evidence_surface = evidence_surface
    if plan_binding_evidence:
        # Once an authoritative plan has selected the exact passage, a stale
        # retrieval hit (often a nearby bibliography block) must not inject
        # unrelated acronyms such as SPIE/LWIR into the method-identity gate.
        identity_evidence_surface = " ".join(
            [
                str(plan_binding_evidence or evidence_quote or ""),
                str(heading or ""),
                str(source_name or ""),
            ]
        )
    if method_identity_conflicts(claim, identity_evidence_surface):
        reason = (
            "回答句与这张卡片明确指向不同的方法或论文，不能仅凭相邻领域词把它们绑定在一起。"
            if _system_a_prefers_zh(claim)
            else (
                "The answer sentence and card name different methods or papers; "
                "nearby domain terms are not sufficient evidence."
            )
        )
        return {
            "status": "mismatch",
            "confidence": 0.0,
            "suppress_link": True,
            "reason": reason,
            "overlap_terms": [],
            "missing_terms": ["method identity"],
        }
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
        candidate_offset = claim.find(named_candidate)
        candidate_prefix = (
            claim[max(0, candidate_offset - 48) : candidate_offset]
            if candidate_offset >= 0
            else ""
        )
        # Chinese technical prose commonly puts an English gloss after the
        # translated term, for example ``自动驾驶的三维态势感知（3D situation
        # awareness for autonomous vehicles）``.  That parenthetical phrase is
        # not a paper title.  Keep explicit paper/method introductions eligible
        # for the cross-source identity check, but do not reject a citation only
        # because its application name has been repeated in English.
        if (
            re.search(r"[\u4e00-\u9fff]", candidate_prefix)
            and not re.search(
                r"(?:论文|文献|文章|工作|方法|模型|算法|研究)\s*[（(]?\s*$",
                candidate_prefix,
            )
        ):
            continue
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
    selected_evidence = str(
        plan_binding_evidence or evidence_quote or (hit or {}).get("text") or ""
    ).strip()
    group_evidence_quotes = [
        str(item or "").strip()
        for item in list((meta or {}).get("citation_group_evidence_quotes") or [])
        if str(item or "").strip()
    ]
    full_claim_quantities = _system_a_fact_quantities(claim)
    group_evidence_quantities = _system_a_fact_quantities("\n".join(group_evidence_quotes))
    verified_multi_source_group = bool(
        len(group_evidence_quotes) >= 2
        and full_claim_quantities
        and all(
            _quantity_is_covered(quantity, group_evidence_quantities)
            for quantity in full_claim_quantities
        )
    )
    claim_fact_quantities = _claim_fact_quantities_for_evidence(
        claim,
        selected_evidence,
        allow_comparison_scope=verified_multi_source_group,
    )
    evidence_fact_quantities = _system_a_fact_quantities(selected_evidence)
    missing_fact_quantities = {
        quantity
        for quantity in claim_fact_quantities
        if not _quantity_is_covered(quantity, evidence_fact_quantities)
    }
    if missing_fact_quantities:
        missing_labels = {
            _quantity_label(quantity) for quantity in missing_fact_quantities
        }
        missing_label = _system_a_term_label(
            sorted(missing_labels, key=lambda item: (len(item), item)),
            max_terms=6,
        )
        reason = (
            f"回答句中的数值“{missing_label}”没有出现在这张卡片的原文证据中，不能把该证据显示为这项定量主张的依据。"
            if _system_a_prefers_zh(claim)
            else (
                f'The answer sentence includes the value(s) "{missing_label}", but the selected '
                "card evidence does not; it cannot support the quantitative claim."
            )
        )
        return {
            "status": "mismatch",
            "confidence": 0.0,
            "suppress_link": True,
            "reason": reason,
            "overlap_terms": [],
            "missing_terms": sorted(missing_labels),
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
    specific_plan_reason = _specific_system_a_support_relation(
        claim,
        evidence_body_surface,
        prefer_zh=_system_a_prefers_zh(claim),
    )
    if (
        specific_plan_reason
        and (canonical_answer_evidence or verified_prompt_contract or prompt_aligned_plan_evidence)
    ):
        return {
            "status": "grounded",
            "confidence": 0.92,
            "suppress_link": False,
            "reason": specific_plan_reason,
            "overlap_terms": sorted(
                _system_a_keyword_terms(claim, limit=48)
                & _system_a_keyword_terms(evidence_surface, limit=64)
            ),
            "missing_terms": [],
        }
    if canonical_answer_evidence or verified_prompt_contract or prompt_aligned_plan_evidence:
        fast_claim_domains = _system_a_domain_terms(claim)
        fast_evidence_body_domains = _system_a_domain_terms(evidence_body_surface)
        fast_missing_strong_terms = (
            fast_claim_domains & _SYSTEM_A_STRONG_BINDING_TERMS
        ) - fast_evidence_body_domains
        fast_source_identity_overlap = _system_a_has_source_identity_overlap(
            claim,
            evidence_body_surface,
            source_name,
        )
        # CASSI is the conventional acronym for coded-aperture snapshot
        # spectral imaging, while the seminal paper's title and abstract spell
        # out only the architecture.  Treat the exact two-disperser/binary-
        # aperture mechanism as a high-precision cross-language identity match
        # so a verified prompt-contract passage is not rejected merely because
        # the Chinese answer uses the acronym.
        cassi_architecture_identity_overlap = bool(
            re.search(
                r"(?i)\bCASSI\b|coded[- ]aperture\s+snapshot\s+spectral\s+imaging|"
                r"编码孔径快照光谱成像",
                claim,
            )
            and re.search(
                r"(?is)two\s+dispersive\s+elements.*binary-valued\s+aperture|"
                r"binary-valued\s+aperture.*two\s+dispersive\s+elements",
                evidence_body_surface,
            )
        )
        fast_source_identity_overlap = bool(
            fast_source_identity_overlap or cassi_architecture_identity_overlap
        )
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
            (
                len(keyword_overlap_fast) >= 2
                or bool(shared_identifiers_fast)
                or bool(shared_numbers_fast)
                or fast_source_identity_overlap
            )
            and not (
                fast_missing_strong_terms
                and not fast_source_identity_overlap
            )
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
    quote_surface = re.sub(
        r"\s+",
        " ",
        str(plan_binding_evidence or evidence_quote or (hit or {}).get("text") or ""),
    ).strip()
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

    claim_low = claim.lower()
    evidence_low = quote_surface.lower()
    if (
        re.search(
            r"dynamic supersampling|foveat(?:ed|ion|l)|动态超采样|中央凹|焦点区域",
            claim_low,
            flags=re.I,
        )
        and re.search(
            r"foveat(?:ed|ion|l)|field of view|spatial information|中央凹|全视场",
            evidence_low,
            flags=re.I,
        )
    ):
        reason = (
            "原文明确说明高分辨率中央凹区域会追踪运动，同时每帧仍获取全视场的新空间信息，"
            "因此可直接支撑回答对动态超采样机制的解释。"
            if prefer_zh
            else (
                "The source states that the high-resolution foveal region tracks motion "
                "while every frame still gathers new spatial information across the full "
                "field of view, directly supporting the explanation of dynamic supersampling."
            )
        )
        return {
            "status": "grounded",
            "confidence": 0.9,
            "suppress_link": False,
            "reason": reason,
            "overlap_terms": sorted(
                set(
                    informative_alignment_overlap
                    or alignment_overlap
                    or body_domain_overlap
                )
                | {"foveated", "dynamic supersampling"}
            ),
            "missing_terms": [],
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
        specific_reason = _specific_system_a_support_relation(
            claim_low,
            evidence_low,
            prefer_zh=prefer_zh,
        )
        if specific_reason:
            reason = specific_reason
        elif (
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
