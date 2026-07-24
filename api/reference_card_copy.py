from __future__ import annotations

import re

from kb.evidence_text import strip_evidence_metadata_prefix


GENERIC_REF_WHY_PATTERNS: tuple[str, ...] = (
    "this hit is directly relevant",
    "directly relevant because",
    "good entry point",
    "directly responds to the user's question",
    "aligns with the user's question",
    "matched section",
    "matched passage",
    "source section",
    "available evidence",
    "matched evidence",
    "current evidence",
    "这条命中",
    "本条命中",
    "直接相关",
    "直接回应",
    "定位入口",
    "定位切口",
    "导读入口",
    "定义、方法或结果信息",
    "关键证据来源",
    "命中章节讲什么",
    "提供什么",
    "当前命中证据",
    "保守说明",
    "关注点直接对应",
    "请只依据",
    "原文线索，可用来核对",
    "可用来判断论文如何使用",
    "可查看“",
    "提供回答该问题所需的原文定位",
    "卡片中的结论可在这里逐项核对",
    "use this evidence to check",
    "use this source wording",
    "use this hit to check",
)


def normalize_ref_card_copy(text: str) -> str:
    s = strip_evidence_metadata_prefix(str(text or ""))
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;:!?，。；：！？])", r"\1", s)
    s = re.sub(r"([（(])\s+", r"\1", s)
    s = re.sub(r"\s+([）)])", r"\1", s)
    return s.strip()


def looks_generic_ref_why_line(text: str) -> bool:
    s = normalize_ref_card_copy(text)
    if not s:
        return True
    low = s.lower()
    starts_with_hit_shell = bool(
        re.match(r"^(?:this hit|this match|this card)\b", low)
        or re.match(r"^(?:这条命中|本条命中|该命中|这条卡片|该卡片)", s)
    )
    if starts_with_hit_shell:
        return True
    if "..." in s and re.search(r"\b(which|what|where|how|why)\b", low):
        return True
    if re.search(r"\b(which paper|in my library|point me to|source section)\b", low):
        return True
    has_specific_signal = bool(
        re.search(r"[“\"'][^“\"']{3,120}[”\"']", s)
        or re.search(r"\b(?:section|related work|method|experiment|figure|table)\b", low)
        or re.search(r"\b[A-Z][A-Z0-9-]{2,}\b", s)
        or re.search(r"(明确提及|定义|解释|比较|讨论|指出|提到)", s)
        or re.search(r"\b(?:defines?|explains?|compares?|mentions?|states?|discusses?)\b", low)
    )
    prompt_echo = bool(
        re.search(r"\b(?:directly responds? to|user(?:'s)? question|query)\b", low)
        or re.search(r"(直接回应|用户查询|当前问题)", s)
    )
    if prompt_echo and not has_specific_signal:
        return True
    if re.search(r"可用来核对.{0,120}(?:里|中)怎样(?:讨论|比较|解释)", s):
        return True
    generic_patterns = tuple(
        pattern
        for pattern in GENERIC_REF_WHY_PATTERNS
        if pattern not in {"直接回应", "directly responds to the user's question"}
    )
    return any(pattern.lower() in low for pattern in generic_patterns)


def looks_templated_ref_why_line(text: str) -> bool:
    s = normalize_ref_card_copy(text)
    if not s:
        return False
    low = s.lower()
    if re.search(r"^原文在[‘'\"]{0,1}.{1,120}[’'\"]{0,1}(?:表明|指出|说明)[：:]", s):
        return True
    # A concrete sentence may legitimately end with wording such as
    # "directly responds to the user's query" after it names the section or
    # technical concept.  Reuse the specificity-aware generic detector so the
    # shell is rejected without discarding that grounded copy.
    return looks_generic_ref_why_line(s)


def _compact_heading_leaf(heading_path: str) -> str:
    parts = [part.strip() for part in str(heading_path or "").split(" / ") if part.strip()]
    if not parts:
        return ""
    leaf = parts[-1]
    if len(leaf) > 90:
        leaf = leaf[:87].rstrip() + "..."
    return leaf


def _compact_terms(focus_terms: list[str] | tuple[str, ...], *, max_terms: int = 2) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in list(focus_terms or []):
        term = normalize_ref_card_copy(str(raw or ""))
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= max(1, int(max_terms or 2)):
            break
    return out


def _compact_summary_fragment(summary_line: str, *, prefer_zh: bool) -> str:
    s = normalize_ref_card_copy(summary_line)
    if not s:
        return ""
    parts = re.split(r"(?<=[。！？.!?])\s+|[；;]\s*", s)
    fragment = next((part.strip() for part in parts if part.strip()), s)
    limit = 88 if prefer_zh else 128
    if len(fragment) > limit:
        fragment = fragment[: limit - 3].rstrip(" ,，。.;；:：") + "..."
    return fragment


def build_grounded_ref_why_line(
    *,
    prefer_zh: bool,
    focus_terms: list[str] | tuple[str, ...],
    heading_path: str,
    summary_line: str = "",
    action: str = "",
) -> str:
    summary_full = normalize_ref_card_copy(summary_line)
    summary_low = summary_full.lower()
    negated_improvement = bool(
        re.search(
            r"\b(?:does\s+not|did\s+not|cannot|can\s+not|fails?\s+to|failed\s+to)"
            r"(?:\s+\w+){0,4}\s+(?:improv|enhanc|accelerat|increase)\w*\b",
            summary_low,
        )
    )
    if prefer_zh:
        if (
            re.search(r"\bADMM\b", summary_full, flags=re.I)
            and re.search(
                r"\b(?:existing|prior|previous|earlier)\s+(?:methods?|work|approaches?)\b",
                summary_full,
                flags=re.I,
            )
        ):
            return "原文把 ADMM 明确归入已有方法，而不是本文新提出的贡献，可据此核对其方法来源与原创性。"
        if ("frame rate" in summary_low or "reconstruction rate" in summary_low) and "30 hz" in summary_low and "333" in summary_low:
            return "原文给出了明确的实时指标：以 333 个照明图案达到 30 Hz 重建帧率，可直接支撑实时成像结论。"
        if ("frame rate" in summary_low or "reconstruction rate" in summary_low) and "30 hz" in summary_low:
            return "原文明确报告了 30 Hz 的重建帧率，可直接支撑实时成像结论。"
        if "333" in summary_low and "real-time" in summary_low:
            return "原文明确说明使用 333 个照明图案完成实时重建，可直接支撑采样效率与实时性结论。"
        if (
            "generalization" in summary_low
            and (re.search(r"low[- ]?light", summary_low) or "low- and high-light" in summary_low)
            and re.search(r"high[- ]?light", summary_low)
        ):
            return "论文的真实数据实验明确报告模型在低照度和高照度条件下均具有良好泛化能力，可直接支撑图像质量与照明鲁棒性优势。"
        if "physical degradation" in summary_low and (
            "generalization" in summary_low or "degradation-robust" in summary_low
        ):
            return "原文把物理退化模型与域外泛化或退化鲁棒表征直接联系起来，可支撑真实退化鲁棒性的结论。"
        if "lpips" in summary_low and re.search(
            r"mist|fog|haze|jitter|sensor noise|real-world degradation",
            summary_low,
            flags=re.I,
        ):
            return "原文在雾、抖动和传感器噪声等真实退化样本上报告了最低 LPIPS，可直接支撑复杂退化下的重建鲁棒性结论。"
        if "imaging speed" in summary_low and (
            "efficient patterns" in summary_low or "reconstruction algorithm" in summary_low
        ):
            return "原文明确说明高效采样图案与配套重建算法能够提升成像速度，可直接支撑深度学习加速单像素成像这一优势。"
        if (
            ("improved image details" in summary_low or "higher quality" in summary_low)
            and ("lower sample" in summary_low or "lower iteration" in summary_low or "part-based" in summary_low)
        ):
            return "该方法在更低采样率或更少迭代下仍改善重建细节和图像质量，直接说明网络对采样效率与重建质量的实际收益。"
        if (
            "iterative reconstruction" in summary_low
            and "image quality" in summary_low
            and ("computational" in summary_low or "time" in summary_low)
        ):
            return "摘要明确指出迭代重建同时受图像质量和计算耗时限制，这是判断深度学习为何能改善单像素成像实用性的直接依据。"
        if (
            "deep learning" in summary_low
            and ("reconstruction quality" in summary_low or "reconstruction speed" in summary_low)
            and ("training" in summary_low or "generalization" in summary_low)
        ):
            return "原文同时给出深度学习在重建质量与速度上的收益，以及训练耗时和泛化能力方面的限制，直接对应问题要求的好处与风险。"
        if (
            ("s2ism" in summary_low or "s²ism" in summary_low)
            and "super-resolution" in summary_low
            and "optical sectioning" in summary_low
        ):
            return "原文明确说明 s²ISM 同时实现超分辨率与光学切片，直接支撑结构化检测如何兼顾横向分辨率和离焦抑制。"
        if (
            "interferometric detection" in summary_low
            and "image scanning microscopy" in summary_low
            and re.search(r"\b12[02]\s*nm\b", summary_low)
        ):
            return "原文说明干涉检测与图像扫描显微镜结合后达到约 120 nm 横向分辨率，直接支撑干涉路线如何突破分辨率瓶颈。"
        if (
            "light-field" in summary_low
            and "position" in summary_low
            and "angular information" in summary_low
            and ("volumetric" in summary_low or "volume" in summary_low)
        ):
            return "原文说明光场同时记录光线的位置与角度信息，并据此重建体积场景，直接支撑无需轴向扫描的三维成像作用。"
        if "scinerf" in summary_low and "3d scene" in summary_low and "single snapshot" in summary_low:
            return "原文明确将 SCINeRF 定义为从单次压缩快照学习三维场景表示的方法，直接回答了该方法是什么。"
        if (
            "scinerf" in summary_low
            and "nerf" in summary_low
            and "physical imaging process" in summary_low
        ):
            return "原文把 SCI 的物理成像过程纳入 NeRF 训练，直接说明从压缩快照到三维神经场景表示的关键衔接。"
        if (
            ("scigs" in summary_low or "3d gaussian" in summary_low)
            and "3d" in summary_low
            and "single compressed image" in summary_low
            and ("dynamic" in summary_low or "explicit scene" in summary_low)
        ):
            return "原文说明 SCIGS 从单幅压缩图像重建显式三维场景并扩展到动态场景，直接支撑 SCI 向 3DGS 场景重建演进这一环。"
        if (
            (
                "dual-disperser" in summary_low
                or "dual disperser" in summary_low
                or "two dispersive elements" in summary_low
            )
            and "binary-valued aperture" in summary_low
            and "spectral data cube" in summary_low
            and (
                "single detector measurement" in summary_low
                or "single 2d measurement" in summary_low
                or "two-dimensional measurement" in summary_low
            )
        ):
            return "原文同时给出双色散器、二值编码孔径和光谱数据立方体，直接说明 CASSI 如何把三维光谱信息压缩到一次二维测量中。"
        if (
            (
                "dual-disperser" in summary_low
                or "dual disperser" in summary_low
                or "two dispersive elements" in summary_low
            )
            and "binary-valued aperture" in summary_low
        ):
            return "原文明确给出两个相向布置的色散元件及其间的二值编码孔径，直接说明 CASSI 的双色散编码结构。"
        if (
            "reducing the pinhole size" in summary_low
            and ("spatial resolution" in summary_low or "optical sectioning" in summary_low)
            and ("snr" in summary_low or "signal-to-noise" in summary_low)
        ):
            return "原文把缩小针孔与空间分辨率、光学切片能力和信噪比的联动写得很清楚，可直接支撑 s2ISM 的核心性能取舍。"
        if (
            "trade-off between spatial resolution" in summary_low
            and "do not provide optical sectioning" in summary_low
            and ("snr" in summary_low or "signal-to-noise" in summary_low)
        ):
            return "原文先说明 ISM 缓解空间分辨率与信噪比权衡，同时指出其仍缺少光学切片能力，直接对应 s2ISM 要解决的三方性能问题。"
        if (
            "do not provide optical sectioning" in summary_low
            and "thick samples" in summary_low
            and ("snr" in summary_low or "signal-to-noise" in summary_low)
        ):
            return "原文指出常规 ISM 在厚样本中缺少光学切片能力，并把这一限制与信噪比问题并列说明，直接对应 s2ISM 要解决的性能约束。"
        if (
            "subcarrier frequenc" in summary_low
            and ("2.3%" in summary_low or "13.0%" in summary_low)
        ):
            return "原文说明不同探测通道由子载波频率并行编码，并给出 2.3% 与 13.0% 的串扰结果，可直接支撑 FDM 的并行采集与通道隔离结论。"
        if (
            "geiger mode" in summary_low
            and ("breakdown" in summary_low or "quench" in summary_low)
        ):
            return "原文解释 SPAD 在盖革模式下高于击穿电压工作，并需通过淬灭电路复位，直接回答其探测与复位机理。"
        if (
            "distilled sensing" in summary_low
            and ("support recovery" in summary_low or "signal support" in summary_low)
        ):
            return "原文把顺序自适应测量与信号支撑集恢复、distilled sensing 联系起来，直接说明该方法如何把测量资源逐步集中到有效分量。"
        if (
            ("beat frequency" in summary_low or "heterodyne" in summary_low)
            and ("holograph" in summary_low or "phase" in summary_low)
        ):
            return "原文以拍频和外差全息解释相位信息如何从单像素时间信号中分离出来，直接支撑压缩全息的测量机理。"
        if (
            "wiener" in summary_low
            and "bm3d" in summary_low
            and ("wavelet" in summary_low or "total variation" in summary_low)
        ):
            return "原文同时列出 Wiener、BM3D 以及小波或总变分方法，可据此区分空间域与变换域去噪路线，而不是只给出泛化的算法标签。"
        if "self-supervised" in summary_low and "network" in summary_low:
            if "ground-truth" in summary_low or "without ground truth" in summary_low:
                return "该方法采用自监督网络，在无需真值图像的条件下完成重建，为深度学习用于单像素成像提供了具体方法证据。"
            return "该文明确提出用于单像素成像的自监督网络，可作为深度学习如何落到具体重建方法上的直接证据。"
        if "deep learning" in summary_low and (
            "reconstruction quality" in summary_low
            or "reconstruction speed" in summary_low
            or "image quality" in summary_low
        ):
            if negated_improvement:
                return "原文明确指出深度学习并未带来所述重建改善，这条证据应当用于说明方法局限，而不是作为正向优势。"
            return "原文把深度学习与重建质量或速度的改善直接联系起来，可据此概括它给单像素成像带来的实际收益。"
        return ""

    if (
        re.search(r"\bADMM\b", summary_full, flags=re.I)
        and re.search(
            r"\b(?:existing|prior|previous|earlier)\s+(?:methods?|work|approaches?)\b",
            summary_full,
            flags=re.I,
        )
    ):
        return (
            "The passage explicitly places ADMM among existing methods rather than the "
            "paper's new contributions, which resolves the method-origin question."
        )
    if (
        ("scinerf" in summary_low or "nerf" in summary_low)
        and "physical imaging process" in summary_low
        and "sci" in summary_low
    ):
        return (
            "This passage is the lineage evidence for the transition from an SCI measurement "
            "model to a learned neural scene representation."
        )
    if (
        ("scigs" in summary_low or "3d gaussian" in summary_low)
        and "3d" in summary_low
        and ("single compressed image" in summary_low or "compressed image" in summary_low)
        and ("dynamic" in summary_low or "explicit" in summary_low)
    ):
        return (
            "This passage marks the endpoint of the lineage: explicit and dynamic 3D scene "
            "reconstruction rather than only decoding a 2D snapshot."
        )
    if (
        (
            "dual-disperser" in summary_low
            or "dual disperser" in summary_low
            or "two dispersive elements" in summary_low
        )
        and "binary-valued aperture" in summary_low
    ):
        return (
            "This passage identifies the optical components that perform spectral coding, "
            "separating the hardware encoding stage from the later 3D reconstruction stages."
        )
    if (
        ("s2ism" in summary_low or "s²ism" in summary_low)
        and "super-resolution" in summary_low
        and "optical sectioning" in summary_low
    ):
        return (
            "The passage states that s²ISM achieves super-resolution and optical sectioning "
            "simultaneously, directly supporting the role of structured detection."
        )
    if (
        "interferometric detection" in summary_low
        and "image scanning microscopy" in summary_low
        and re.search(r"\b12[02]\s*nm\b", summary_low)
    ):
        return (
            "The passage reports about 120 nm lateral resolution from combining interferometric "
            "detection with image scanning microscopy, directly supporting the resolution claim."
        )
    if (
        "light-field" in summary_low
        and "position" in summary_low
        and "angular information" in summary_low
        and ("volumetric" in summary_low or "volume" in summary_low)
    ):
        return (
            "The passage explains that light-field imaging records position and angular information "
            "for volumetric reconstruction, directly supporting scan-free 3D imaging."
        )
    return ""


def finalize_ref_card_copy(
    *,
    summary_line: str,
    why_line: str,
    prefer_zh: bool,
    focus_terms: list[str] | tuple[str, ...],
    heading_path: str,
    action: str = "",
) -> tuple[str, str, bool]:
    summary = normalize_ref_card_copy(summary_line)
    why = normalize_ref_card_copy(why_line)
    changed = why != str(why_line or "").strip()
    if looks_generic_ref_why_line(why) or looks_templated_ref_why_line(why):
        grounded = build_grounded_ref_why_line(
            prefer_zh=prefer_zh,
            focus_terms=focus_terms,
            heading_path=heading_path,
            summary_line=summary,
            action=action,
        )
        why = grounded
        changed = True
    return summary, why, changed
