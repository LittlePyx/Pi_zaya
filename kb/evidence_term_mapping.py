from __future__ import annotations

import re
from typing import Iterable


_ASCII_STOPWORDS = {
    "about", "after", "also", "and", "are", "based", "because", "between",
    "does", "from", "have", "into", "more", "paper", "results", "that",
    "their", "there", "these", "this", "through", "using", "which", "with",
}


_GENERIC_METHOD_IDENTITIES = {
    "2d",
    "3d",
    "3dgs",
    "ai",
    "ccd",
    "cnn",
    "cnr",
    "cvpr",
    "dl",
    "dmd",
    "dnn",
    "doi",
    "fps",
    "ieee",
    "lpips",
    "lsa",
    "natphoton",
    "nerf",
    "pdf",
    "pnsr",
    "psnr",
    "rgb",
    "rmse",
    "sci",
    "snr",
    "spad",
    "spi",
    "ssim",
}
_METHOD_IDENTITY_EQUIVALENTS = (
    frozenset({"piln", "ilnet"}),
)


def specific_method_identities(value: object) -> set[str]:
    """Extract distinctive mixed-case/acronym method or paper identifiers."""

    identities: set[str] = set()
    # Split on hyphens instead of treating an entire bibliographic filename
    # prefix (for example ``OE-2007-Single-shot``) as one mixed-case method
    # identifier. Distinctive model names such as SCIGS, SCINeRF and NAFNet
    # remain single alphanumeric tokens and are still protected.
    surface = str(value or "")
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9]{2,}\b", surface):
        if sum(1 for char in token if char.isupper()) < 2:
            continue
        normalized = token.lower().strip("-")
        if normalized and normalized not in _GENERIC_METHOD_IDENTITIES:
            identities.add(normalized)
    # Comparison tables commonly spell out the basis name while answer prose
    # uses HSI/FSI. Treat those as the same method identity only in explicit
    # single-pixel-imaging context; plain Fourier/Hadamard mathematics remains
    # too broad to act as a paper/method identifier.
    surface_low = surface.lower()
    single_pixel_context = bool(
        re.search(r"\bsingle[-\s]?pixel\s+imaging\b|\bspi\b", surface_low)
    )
    if single_pixel_context and "hadamard" in surface_low:
        identities.add("hsi")
    if single_pixel_context and "fourier" in surface_low:
        identities.add("fsi")
    if re.search(r"\binterferometric\s+image\s+scanning\s+microscopy\b", surface_low):
        identities.add("iism")
    return identities


def method_identity_conflicts(claim: object, evidence: object) -> bool:
    """Return true when both sides name disjoint distinctive methods/papers."""

    claim_ids = specific_method_identities(claim)
    evidence_ids = specific_method_identities(evidence)
    if not claim_ids or not evidence_ids:
        return False
    if claim_ids & evidence_ids:
        return False
    return not any(
        bool(claim_ids & equivalents) and bool(evidence_ids & equivalents)
        for equivalents in _METHOD_IDENTITY_EQUIVALENTS
    )


# Conservative aliases used only to align Chinese questions and summaries with
# English academic source passages. Values intentionally mirror terms commonly
# written verbatim in papers; they are never used to generate user-facing prose.
_CJK_EVIDENCE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"各类(?:单光子)?探测器|探测器类型", ("detector", "detectors")),
    (r"物理原理|工作原理|技术原理", ("physical", "principles")),
    (
        r"制造难度|制造挑战|制造工艺(?:复杂|困难)",
        ("complexity", "manufacturing", "challenges"),
    ),
    (r"适用场景|应用场景|推广应用", ("applications", "adoption")),
    (r"深度学习|神经网络", ("deep", "learning", "neural", "network")),
    (r"物理信息|物理先验|物理约束", ("physics", "informed", "physical")),
    (r"重建", ("reconstruction", "reconstruct", "reconstructed")),
    (r"图像质量|重建质量|画质|保真", ("image", "quality", "fidelity")),
    (r"成像效率", ("imaging", "efficiency")),
    (r"噪声鲁棒性|抗噪(?:能力|性能)?", ("noise", "robustness")),
    (r"制造(?:工艺)?复杂|复杂.{0,6}制造", ("complexity", "complex", "manufacturing", "craft")),
    (r"成本高|高成本", ("high", "cost")),
    (r"低温|特殊(?:工作)?条件", ("low", "temperature", "special", "conditions")),
    (r"重建速度|采集速度|速度|加速|更快", ("speed", "fast", "faster", "acquisition")),
    (r"泛化", ("generalization", "generalize")),
    (r"训练", ("training", "train")),
    (r"过拟合", ("overfitting", "overfit")),
    (r"可解释", ("interpretability", "interpretable")),
    (r"测量次数|测量", ("measurement", "measurements")),
    (r"采样率|采样比例|欠采样", ("sampling", "ratio", "undersampling")),
    (r"信噪比|信号噪声比", ("signal", "noise", "ratio", "snr")),
    (r"分辨率|超分辨", ("resolution", "super", "spatial")),
    (r"光学切片|光学层切", ("optical", "sectioning")),
    (r"厚样本", ("thick", "samples")),
    (r"探测器尺寸", ("detector", "size")),
    (r"探测器积分时间|积分时间", ("detector", "integration", "time")),
    (r"位置.{0,8}角度|角度.{0,8}位置", ("position", "angular", "information")),
    (r"重聚焦|重新对焦|离焦", ("digital", "refocusing", "focus")),
    (r"光线追迹|光线追踪", ("ray", "tracing")),
    (r"波传播|波动传播", ("wave", "propagation")),
    (r"两步|两个步骤", ("two", "steps")),
    (r"相位步进|相移", ("phase", "stepping")),
    (r"拍频|差频", ("beat", "frequency")),
    (r"外差全息|异频全息", ("heterodyne", "holography")),
    (r"频分复用|频率复用", ("frequency", "division", "multiplexed", "multiplexing")),
    (r"多方向照明|不同照明方向", ("multiple", "illumination", "directions")),
    (r"并行化|并行编码|并行", ("parallelize", "parallel", "encoding")),
    (r"载波|子载波", ("carrier", "subcarrier", "frequencies")),
    (r"单张压缩图|单幅压缩图|一次压缩观测", ("single", "compressed", "image")),
    (r"动态.{0,4}(?:3d|三维).{0,4}场景|动态场景", ("dynamic", "scenes", "3d")),
    (r"显式.{0,4}(?:3d|三维)|显式场景", ("explicit", "scene", "3d")),
    (r"变体|适配|改进|核心新意|3DGS\s*本身", ("variant", "adapt", "propose")),
    (r"整个视场|全视场|视场其余区域", ("entire", "field", "view")),
    (r"普通\s*zoom|简单变焦|局部放大", ("unlike", "simple", "zoom", "every", "frame")),
    (r"中央凹(?:区域)?|中央区|焦点区域|焦区|运动区域|高分辨率区", ("foveal", "region", "fovea")),
    (r"快速变化|快速记录|快变(?:特征|区域)?", ("rapidly", "records", "quickly", "changing", "features")),
    (
        r"连续多帧|连续帧|慢变化区域|缓慢(?:变化|演变)|慢变(?:区域)?|"
        r"多帧累积|跨(?:连续)?多帧|累积细节",
        ("accumulating", "detail", "consecutive", "frames", "slowly", "evolving"),
    ),
    (r"动态超采样", ("dynamic", "supersampling")),
    (r"顺序自适应|序贯自适应", ("sequential", "adaptive")),
    (r"压缩感知", ("compressed", "compressive", "sensing")),
    (r"支撑集恢复|稀疏支撑恢复|主要保证恢复", ("signal", "support", "recovery")),
    (r"蒸馏感知", ("distilled", "sensing")),
    (
        r"自监督|不依赖成对真值|无需成对真值|"
        r"无需真实图像标签|不需要.{0,18}成对.{0,18}(?:图像|训练)?标签",
        ("self", "supervised", "labels"),
    ),
    (
        r"(?:物理采集的?)?\s*(?:1\s*D|一维)\s*信号|"
        r"单像素探测器.{0,12}(?:采集|采样).{0,8}信号",
        ("1d", "signals", "collected", "single", "pixel", "detector"),
    ),
    (
        r"作为.{0,10}(?:监督信号|训练标签|标签)|"
        r"用于.{0,12}(?:自适应)?优化.{0,12}重建",
        ("used", "labels", "adaptively", "optimizing", "reconstructing"),
    ),
    (r"图像循环|图像闭环|image-loop", ("image", "loop", "iteration")),
    (
        r"循环回.{0,24}(?:网络)?输入|半成品.{0,24}输入",
        ("input", "subsequent", "iteration"),
    ),
    (r"原始随机信号", ("randomly", "signal")),
    (r"分块|基于部件|特征分块|part-based", ("part", "based", "finer", "grained", "details")),
    (r"入射照明功率|照明功率", ("incident", "illumination", "power")),
    (r"降低约?(?:10|十)倍|低十倍|十分之一", ("tenfold", "lower")),
    (r"光损伤|光毒性", ("photodamage", "phototoxicity")),
    (r"活细胞", ("live", "cells", "photodamage")),
    (r"波长|波段|非可见光|红外|太赫兹", ("wavelengths", "outside", "fpa", "technology")),
    (r"高帧率", ("high", "frame", "rates")),
    (r"光度立体", ("photometric", "stereo")),
    (
        r"(?:四个|四路|多个).{0,18}(?:空间(?:上)?分离.{0,12})?(?:单像素)?探测器",
        ("four", "spatially", "separated", "single", "pixel", "detectors"),
    ),
    (
        r"(?:约\s*)?8\s*帧(?:/|每)?秒|每秒\s*(?:约\s*)?8\s*帧",
        ("frames", "per", "second"),
    ),
    (r"危险气体泄漏|气体泄漏", ("hazardous", "gas", "leaks", "visualization")),
    (r"自动驾驶|三维态势感知|3D\s*态势感知", ("autonomous", "vehicles", "3d", "situation", "awareness")),
    (r"空间域", ("spatial", "domain")),
    (r"变换域", ("transform", "domain")),
    (r"暗计数", ("dark", "count")),
    (r"后脉冲", ("afterpulsing", "afterpulse")),
    (r"串扰", ("crosstalk",)),
    (r"死区时间", ("dead", "time")),
    (r"泊松", ("poisson",)),
    (
        r"(?:照明图案|投影端).{0,24}(?:散射|非理想聚焦|模糊)|非理想聚焦",
        ("illumination", "patterns", "scattering", "non", "ideal", "focus", "blur"),
    ),
    (r"空间下采样|图案分辨率有限", ("spatial", "downsampling", "limited", "resolution", "patterns")),
    (
        r"机械抖动|相对错位|乘性波动",
        ("mechanical", "jitters", "relative", "misalignment", "multiplicative", "fluctuations"),
    ),
    (
        r"探测路径.{0,24}(?:散射|模糊)|散射缺陷",
        ("detection", "path", "scattering", "imperfections", "further", "blur"),
    ),
    (r"光子散粒噪声|散粒噪声", ("photon", "shot", "noise", "poisson")),
    (r"电子噪声", ("electronic", "noise")),
    (
        r"整个场景.{0,16}(?:光强|积分)|全局测量",
        ("single", "pixel", "detector", "integrates", "light", "intensities", "entire", "scene", "readout"),
    ),
    (
        r"传播到整幅图像|扩散到整幅图像|全局污染",
        ("propagate", "spread", "entire", "image", "reconstruction"),
    ),
)


def evidence_alignment_tokens(
    value: object,
    *,
    extra_stopwords: Iterable[str] | None = None,
) -> set[str]:
    """Return English evidence tokens and deterministic Chinese aliases."""

    text = str(value or "")
    stopwords = set(_ASCII_STOPWORDS)
    stopwords.update(str(item or "").strip().lower() for item in (extra_stopwords or ()))
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if (len(token) >= 3 or token in {"2d", "3d"}) and token not in stopwords
    }
    for pattern, aliases in _CJK_EVIDENCE_HINTS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            tokens.update(aliases)
    # One converted HSI/FSI benchmark table carries the obvious OCR/header
    # transposition ``PNSR`` even though the surrounding prose defines PSNR.
    # Treat it as the canonical metric for evidence alignment only; the source
    # text itself remains untouched.
    if "pnsr" in tokens:
        tokens.add("psnr")
    return tokens


__all__ = [
    "evidence_alignment_tokens",
    "method_identity_conflicts",
    "specific_method_identities",
]
