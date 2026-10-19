from __future__ import annotations

import re

from api.method_relation_copy import method_relation_copy


# These translations require the relation in the evidence body. Paper titles,
# generated summaries and retrieval scores must never trigger factual copy.
_RELATIONS = (
    (
        r"invariant\s+function.{0,100}minimizing.{0,40}satisfies",
        "这里讨论使目标最小化的不变函数，并引出其应满足的等式。",
        "优化对象限定为不变函数；这一段引出最优解的等式，具体公式及逐子集适用条件需连同后续源块核对。",
    ),
    (
        r"variance of the estimate is the average variance of the corruptions in the targets, divided by the number of training samples",
        "有限数据下，估计方差等于目标污染的平均方差除以训练样本数。",
        "该段明确给出分子、分母及除法关系，可核对估计方差如何随目标污染与样本量变化。",
    ),
    (
        r"self-supervised loss is the sum of the ordinary supervised loss and the variance of the noise",
        "自监督损失等于普通监督损失与噪声方差之和。",
        "该段给出两种损失之间的差值项；判断最优去噪器的选择依据时，还需连同命题的独立性与函数约束阅读。",
    ),
    (
        r"calibrate traditional image denoising methods such as median filters and non-local means",
        "该方法还可校准中值滤波与非局部均值等传统图像去噪算法。",
        "原文逐项列出可校准的传统算法，说明校准范围不限于神经网络；具体参数选择仍需查看相应实验。",
    ),
    (
        r"noise (?:to be|is) pixel-wise independent given the signal",
        "给定信号时，Noise2Void 假设噪声在像素之间逐像素相互独立。",
        "该段用逐像素条件独立解释邻域为何不携带目标像素的噪声信息，并据此说明盲点网络不能学习恒等映射。",
    ),
    (
        r"noise is independent conditional on the true signal",
        "噪声独立性以真实信号为条件，不能理解为允许任意相关噪声。",
        "这里直接限定了噪声独立性的条件，可用于核对自监督预测依赖的噪声假设；不能据此推断相关噪声下的性能。",
    ),
    (
        r"photographic noise removal, denoising of synthetic Monte Carlo images, and reconstruction of MRI scans from undersampled inputs",
        "恢复应用包括照片去噪、Monte Carlo 合成图像去噪与欠采样 MRI 重建。",
        "原文列出的数据类型覆盖摄影、模拟渲染与磁共振测量，可据此区分三类应用，不能概括成普通照片去噪。",
    ),
)

_COMPOUND_RELATIONS = (
    (
        (
            r"signal.{0,30}(?:is\s+)?not\s+pixel[- ]wise\s+independent",
            r"noise.{0,45}conditionally\s+pixel[- ]wise\s+independent.{0,30}(?:given\s+the\s+signal|signal)",
        ),
        "Noise2Void 假设真实信号不是逐像素独立的，而噪声在给定信号时逐像素条件独立。",
        "该段把信号相关性与噪声条件独立性并列为两条假设，可用于核对邻域预测为何保留信号信息却不能预测目标像素噪声。",
    ),
    (
        (
            r"(?:does\s+not\s+require|neither).{0,80}noisy\s+image\s+pairs",
            r"(?:nor|neither).{0,40}clean\s+target\s+im(?:[-‐‑]\s*)?ages",
        ),
        "Noise2Void 不需要独立带噪图像对，也不需要干净目标图像。",
        "该段直接界定训练数据需求：独立带噪图像对与干净目标图像这两类配对数据均不是前提；相对性能仍须核对比较段落。",
    ),
    (
        (
            r"3\s*[×x]\s*3",
            r"remove(?:s|d)?\s+all\s+pooling\s+layers",
            r"receptive\s+field.{0,80}depth\s+of\s+d.{0,80}\(\s*2\s*d\s*\+\s*1\s*\)\s*[×x]\s*\(\s*2\s*d\s*\+\s*1\s*\)",
        ),
        "DnCNN 使用 3×3 卷积核并移除全部池化层；深度为 d 时感受野为 (2d+1)×(2d+1)。",
        "该段同时给出卷积核、池化设置和感受野公式，可直接核对网络深度如何扩大上下文范围。",
    ),
    (
        (
            r"Gaussian\s+denois",
            r"(?:single\s+image\s+super[- ]resolution|\bSISR\b)",
            r"JPEG(?:\s+image)?\s+deblock",
        ),
        "单个 DnCNN 模型覆盖盲高斯去噪、单图像超分辨率和 JPEG 图像去块。",
        "原文逐项列出三类恢复任务，可核对单模型覆盖范围；这不表示三类退化具有相同噪声分布。",
    ),
    (
        (
            r"input[- ]conditioned\s+target\s+distributions",
            r"same\s+conditional\s+expected\s+values",
            r"zero[- ]mean\s+noise",
        ),
        "带噪训练目标必须在给定输入时保持与原目标相同的条件期望。",
        "该段明确把零均值目标噪声与条件期望不变联系起来，因此可解释何时能省去干净目标，而不能推广到有偏目标噪声。",
    ),
    (
        (
            r"designed\s+to\s+predict\s+the\s+residual\s+image",
            r"difference\s+between\s+the\s+noisy\s+observation\s+and\s+the\s+latent\s+clean\s+image",
        ),
        "DnCNN \u9884\u6d4b\u7684\u662f\u542b\u566a\u89c2\u6d4b\u4e0e\u6f5c\u5728\u5e72\u51c0\u56fe\u50cf\u4e4b\u95f4\u7684\u6b8b\u5dee\uff0c\u800c\u975e\u76f4\u63a5\u8f93\u51fa\u5e72\u51c0\u56fe\u50cf\u3002",
        "\u8be5\u6bb5\u660e\u786e\u5b9a\u4e49\u4e86\u7f51\u7edc\u8f93\u51fa\u7684\u7269\u7406\u542b\u4e49\uff0c\u53ef\u7528\u4e8e\u6838\u5bf9\u201c\u8f93\u51fa\u6b8b\u5dee\u3001\u518d\u4ece\u542b\u566a\u8f93\u5165\u4e2d\u6263\u9664\u201d\u7684\u53bb\u566a\u5173\u7cfb\u3002",
    ),
    (
        (r"independent,?\s+mean[- ]zero\s+noise\s+in\s+each\s+pixel",
         r"choose.{0,160}singletons"),
        "独立且均值为零的像素噪声允许选择每个坐标的单元素集作为分组。",
        "单元素分组的选择明确依赖逐像素独立和零均值两个条件，不能把这一例子推广到任意相关或有偏噪声。",
    ),
    (
        (
            r"expectation\s+of\s+the\s+corrupted\s+input\s+data\s+is\s+the\s+clean\s+target",
            r"dispose\s+of\s+the\s+clean\s+targets\s+entirely",
            r"as\s+long\s+as\s+we\s+are\s+able\s+to\s+observe\s+each\s+source\s+image\s+twice",
        ),
        "当含噪输入的期望等于干净目标时，可省去干净目标，但每张源图像仍须观测两次。",
        "该条件关系明确保留了重复观测的数据采集要求，核对训练数据时还须同时检查含噪输入的期望条件。",
    ),
)


def source_relation_copy(evidence: str) -> tuple[str, str] | None:
    surface = re.sub(r"\s+", " ", evidence).strip()
    method_copy = method_relation_copy(surface)
    if method_copy:
        return method_copy
    for index, (patterns, summary, why) in enumerate(_COMPOUND_RELATIONS):
        if not all(re.search(pattern, surface, re.I) for pattern in patterns):
            continue
        # The first two relations contain their own required negation:
        # signal is not independent; the method does not require paired data.
        if index in {0, 1}:
            return summary, why
        if not re.search(
            r"\b(?:does\s+not|do\s+not|never|cannot|fails?\s+to)\b|不能|并非|不等于",
            surface,
            re.I,
        ):
            return summary, why
    for pattern, summary, why in _RELATIONS:
        for sentence in re.split(r"[.!?。！？]", surface):
            if re.search(pattern, sentence, re.I) and not re.search(
                r"\b(?:not|never|cannot|fail\w*|unlike)\b|不能|并非|不等于", sentence, re.I
            ):
                return summary, why
    return None
