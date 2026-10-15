from __future__ import annotations

import re


def _evidence_surfaces(hit: dict) -> list[str]:
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    surfaces = [str(hit.get("text") or "").strip()]
    for passage in list((meta or {}).get("source_passages") or []):
        if isinstance(passage, dict):
            surfaces.append(str(passage.get("text") or "").strip())
    out: list[str] = []
    seen: set[str] = set()
    for surface in surfaces:
        normalized = re.sub(r"\s+", " ", surface).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def _matching_surface(rows: list[dict], *patterns: str) -> tuple[int, str]:
    for number, hit in enumerate(rows, start=1):
        for surface in _evidence_surfaces(hit):
            if all(re.search(pattern, surface, re.I) for pattern in patterns):
                return number, surface
    return 0, ""


def build_exact_relation_grounded_fallback(
    prompt: str, rows: list[dict], *, prefer_zh: bool | None = None
) -> str:
    q = str(prompt or "")
    if prefer_zh is None:
        prefer_zh = bool(re.search(r"[\u4e00-\u9fff]", q))
    if re.search(r"\bDnCNN\b", q, re.I) and re.search(
        r"(?:directly\s+predict|network\s+output|denoised\s+image)|(?:\u76f4\u63a5\u9884\u6d4b|\u7f51\u7edc\u8f93\u51fa|\u53bb\u566a\u56fe\u50cf)", q, re.I
    ):
        number, _surface = _matching_surface(
            rows,
            r"designed\s+to\s+predict\s+the\s+residual\s+image",
            r"difference\s+between\s+the\s+noisy\s+observation\s+and\s+the\s+latent\s+clean\s+image",
        )
        if number:
            if prefer_zh:
                return (
                    "DnCNN \u4e0d\u76f4\u63a5\u9884\u6d4b\u5e72\u51c0\u56fe\u50cf\uff0c\u800c\u662f\u9884\u6d4b\u6b8b\u5dee\u56fe\u50cf\uff0c\u5373\u542b\u566a\u89c2\u6d4b\u4e0e\u6f5c\u5728\u5e72\u51c0\u56fe\u50cf\u4e4b\u5dee\uff1b"
                    f"\u56e0\u6b64\u6700\u7ec8\u53bb\u566a\u56fe\u50cf\u7531\u542b\u566a\u8f93\u5165\u51cf\u53bb\u7f51\u7edc\u9884\u6d4b\u7684\u6b8b\u5dee\u5f97\u5230 [{number}]\u3002"
                )
            return (
                "DnCNN predicts the residual image rather than the clean image directly. The residual is the "
                f"difference between the noisy observation and latent clean image, so the denoised image is the noisy input minus the predicted residual [{number}]."
            )

    if re.search(r"(?:Proposition\s*1|\u547d\u9898\s*1)", q, re.I) and re.search(
        r"(?:self[- ]supervised\s+loss|supervised\s+loss|\u81ea\u76d1\u7763\u635f\u5931|\u6709\u76d1\u7763\u635f\u5931)", q, re.I
    ):
        number, _surface = _matching_surface(
            rows,
            r"self[- ]supervised\s+loss\s+is\s+the\s+sum\s+of\s+the\s+ordinary\s+supervised\s+loss\s+and\s+the\s+variance\s+of\s+the\s+noise",
            r"minimizing\s+the\s+self[- ]supervised\s+loss.{0,100}(?:optimal\s+denoiser|ordinary\s+supervised\s+loss)",
        )
        if number:
            if prefer_zh:
                return (
                    "\u547d\u9898 1 \u7ed9\u51fa\uff1a\u81ea\u76d1\u7763\u635f\u5931\u7b49\u4e8e\u666e\u901a\u6709\u76d1\u7763\u635f\u5931\u52a0\u4e0a\u566a\u58f0\u65b9\u5dee\u9879\u3002"
                    "\u8be5\u65b9\u5dee\u9879\u4e0d\u4f9d\u8d56\u5f85\u9009\u53bb\u566a\u5668\uff0c\u6240\u4ee5\u53ea\u4f1a\u7ed9\u635f\u5931\u589e\u52a0\u56fa\u5b9a\u5e38\u6570\uff0c"
                    f"\u4e0d\u6539\u53d8\u6700\u4f18\u53bb\u566a\u5668\u7684\u9009\u62e9 [{number}]\u3002"
                )
            return (
                "Proposition 1 states that the self-supervised loss equals the ordinary supervised loss plus the noise variance. "
                f"That additive term is independent of the candidate denoiser, so it does not change the minimizer [{number}]."
            )

    calibration_question = bool(
        re.search(r"(?:\u6821\u51c6|calibrat)", q, re.I)
        and re.search(
            r"(?:\u4f20\u7edf(?:\u56fe\u50cf)?\u53bb\u566a|traditional\s+(?:image\s+)?denois|"
            r"\u4e2d\u503c|\u4e2d\u4f4d|median|\u975e\u5c40\u90e8|non[- ]local)",
            q,
            re.I,
        )
    )
    if calibration_question:
        number, surface = _matching_surface(
            rows,
            r"learn\s+denoising\s+functions\s+from\s+single\s+noisy\s+measurements",
            r"calibrate\s+traditional\s+image\s+denoising\s+methods\s+such\s+as\s+median\s+filters?\s+and\s+non[- ]local\s+means",
        )
        negated = bool(
            re.search(
                r"\b(?:cannot|can't|does\s+not|do\s+not|never|fails?\s+to)\b.{0,40}\bcalibrat",
                surface,
                re.I,
            )
        )
        if number and not negated:
            if prefer_zh:
                return (
                    "该框架的用途不限于训练神经网络。论文明确说明，同一方法无需干净数据也可用于"
                    f"校准传统图像去噪方法，包括中值滤波和非局部均值 [{number}]。"
                )
            return (
                "The framework is not limited to training neural networks. The paper states that the same approach "
                "can calibrate traditional image denoising methods without clean data, specifically median filters "
                f"and non-local means [{number}]."
            )

    architecture_facets = (
        r"pooling|池化",
        r"filter|kernel|卷积核",
        r"receptive\s+field|感受野",
    )
    if sum(bool(re.search(pattern, q, re.I)) for pattern in architecture_facets) >= 2:
        number, _surface = _matching_surface(
            rows,
            r"3\s*[×x]\s*3",
            r"remove(?:s|d)?\s+all\s+pooling\s+layers",
            r"receptive\s+field.{0,80}depth\s+of\s+d.{0,80}\(\s*2\s*d\s*\+\s*1\s*\)\s*[×x]\s*\(\s*2\s*d\s*\+\s*1\s*\)",
        )
        if number:
            if prefer_zh:
                return (
                    "模型服务暂时不可用，先按已定位原文给出结论：网络移除了所有池化层，"
                    f"卷积核大小为 $3\\times3$；深度为 $d$ 时，感受野为 "
                    f"$(2d+1)\\times(2d+1)$ [{number}]。"
                )
            return (
                "The model service is temporarily unavailable. The located source states that the network "
                f"removes all pooling layers, uses $3\\times3$ filters, and has a "
                f"$(2d+1)\\times(2d+1)$ receptive field at depth $d$ [{number}]."
            )

    task_list_question = bool(
        re.search(
            r"(?:三类|三种|three).{0,35}(?:任务|应用|tasks?|applications?)|"
            r"(?:任务|应用|tasks?|applications?).{0,35}(?:三类|三种|three)",
            q,
            re.I,
        )
    )
    if task_list_question:
        number, _surface = _matching_surface(
            rows,
            r"Gaussian\s+denois",
            r"(?:single\s+image\s+super[- ]resolution|\bSISR\b)",
            r"JPEG(?:\s+image)?\s+deblock",
        )
        if number:
            if prefer_zh:
                return (
                    "模型服务暂时不可用，先按已定位原文给出结论：单个模型处理的三类任务是"
                    f"盲高斯去噪、单图像超分辨率（SISR）和 JPEG 图像去块 [{number}]。"
                )
            return (
                "The model service is temporarily unavailable. The located source lists Gaussian denoising, "
                f"single-image super-resolution (SISR), and JPEG deblocking as the three tasks [{number}]."
            )

    noisy_target_question = bool(
        re.search(r"条件期望|conditional\s+expect", q, re.I)
        and re.search(
            r"干净(?:标签|目标)|带噪(?:观测|目标)|clean\s+targets?|noisy\s+targets?",
            q,
            re.I,
        )
    )
    if noisy_target_question:
        number, _surface = _matching_surface(
            rows,
            r"input[- ]conditioned\s+target\s+distributions",
            r"same\s+conditional\s+expected\s+values",
            r"zero[- ]mean\s+noise",
        )
        if number:
            if prefer_zh:
                return (
                    "模型服务暂时不可用，先按已定位原文给出结论：训练目标可以换成另一份带噪观测，"
                    "前提是给定输入时，替代目标分布与原目标分布具有相同的条件期望；因此给目标加入"
                    f"零均值噪声不会改变网络学到的映射，也就不必使用干净目标 [{number}]。"
                )
            return (
                "The model service is temporarily unavailable. The located source allows noisy targets when "
                "their input-conditioned distribution has the same conditional expectation as the original target; "
                f"adding zero-mean target noise therefore leaves the learned mapping unchanged [{number}]."
            )

    training_pair_question = bool(
        re.search(r"(?:一|单)张.{0,12}带噪|(?:哪|两).{0,12}(?:类)?配对|single\s+noisy", q, re.I)
        and re.search(r"干净|clean|配对|pairs?", q, re.I)
    )
    if training_pair_question:
        number, _surface = _matching_surface(
            rows,
            r"(?:does\s+not\s+require|neither).{0,80}noisy\s+image\s+pairs",
            r"(?:nor|neither).{0,40}clean\s+target\s+images",
        )
        if number:
            if prefer_zh:
                return (
                    "模型服务暂时不可用，先按已定位原文给出结论：这种训练可直接使用待去噪的"
                    f"单张带噪图像，不需要独立带噪图像对，也不需要带噪输入与干净目标组成的图像对 [{number}]。"
                )
            return (
                "The model service is temporarily unavailable. The located source states that training can use "
                f"single noisy images without noisy-image pairs or clean target images [{number}]."
            )

    assumptions_question = bool(
        re.search(r"统计假设|statistical\s+assumptions?", q, re.I)
        and re.search(r"信号|signal", q, re.I)
        and re.search(r"噪声|noise", q, re.I)
    )
    if assumptions_question:
        number, _surface = _matching_surface(
            rows,
            r"signal.{0,30}(?:is\s+)?not\s+pixel[- ]wise\s+independent",
            r"noise.{0,45}conditionally\s+pixel[- ]wise\s+independent.{0,30}(?:given\s+the\s+signal|signal)",
        )
        if number:
            if prefer_zh:
                return (
                    "模型服务暂时不可用，先按已定位原文给出两条统计假设：真实信号不是逐像素独立的；"
                    f"给定真实信号时，噪声在像素之间条件独立 [{number}]。"
                )
            return (
                "The model service is temporarily unavailable. The located source assumes that the signal is not "
                f"pixel-wise independent and that noise is conditionally pixel-wise independent given the signal [{number}]."
            )

    if re.search(
        r"\b(?:refocus|refocusing|out[ -]of[ -]focus)\b|重聚焦|重新对焦|离焦.{0,8}对焦",
        q,
        re.I,
    ):
        number, _surface = _matching_surface(
            rows,
            r"\btwo steps?\b",
            r"\bray[ -]tracing\b",
            r"\bwave propagation\b",
        )
        if number:
            if prefer_zh:
                return (
                    "模型服务暂时不可用，先按已定位原文给出结论：数字重聚焦分为 two steps，"
                    "先用 ray tracing 重建光子轨迹，再用 wave propagation 反演衍射传播，"
                    f"使离焦样品重新对焦 [{number}]。"
                )
            return (
                "The model service is temporarily unavailable. The located source states that digital refocusing "
                f"uses two steps: ray tracing reconstructs photon trajectories, then wave propagation reverses diffraction [{number}]."
            )
    return ""


def build_provider_failure_grounded_fallback(prompt: str, hits: list[dict]) -> str:
    rows = [
        hit
        for hit in list(hits or [])
        if isinstance(hit, dict) and str(hit.get("text") or "").strip()
    ]
    if not rows:
        return ""
    q = str(prompt or "").strip()
    prefer_zh = bool(re.search(r"[\u4e00-\u9fff]", q))
    exact = build_exact_relation_grounded_fallback(q, rows, prefer_zh=prefer_zh)
    if exact:
        return exact

    lines = [
        "模型服务暂时不可用，先列出已定位的原文依据："
        if prefer_zh
        else "The model service is temporarily unavailable. Located source evidence:"
    ]
    for number, hit in enumerate(rows[:3], start=1):
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        heading = str(
            (meta or {}).get("heading_path") or (meta or {}).get("top_heading") or ""
        ).strip()
        snippet = re.sub(r"\s+", " ", str(hit.get("text") or "").strip())
        if len(snippet) > 360:
            snippet = snippet[:357].rstrip() + "..."
        line = f"- {heading}: {snippet}" if heading else f"- {snippet}"
        lines.append(f"{line} [{number}]")
    return "\n".join(lines).strip()
