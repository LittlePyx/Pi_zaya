from __future__ import annotations

import re


_RULES = (
    (
        (r"Given the sum of two i\.\s*i\.\s*d\. noise samples, the best estimate of one of them is half of their sum",),
        "两份独立同分布噪声样本相加时，其中一份的最佳估计是两者之和的一半。",
        "该句限定两份噪声样本之间的独立同分布关系，并给出和的一半这一估计；不能推导每份样本内部各分量也独立。",
    ),
    (
        (r"original noisy image.{0,100}synthetic sample from the same noise distribution",),
        "原始带噪图像中的噪声与另外合成的噪声样本来自同一噪声分布。",
        "该句区分原始观测与合成样本，并明确二者所用噪声分布相同；不能把允许的分布条件仅归于合成样本。",
    ),
    (
        (r"Recall that [A-Z] and [A-Z] are i\.\s*i\.\s*d\.",),
        "原文明确给出所指两个随机量独立同分布的关系。",
        "该句同时给出独立性与分布相同两个条件，约束对象是句中的两个随机量；不能据此推定每个随机量内部的各分量也独立。",
    ),
    (
        (r"Assume the noise components are independent and of zero mean",),
        "原文把噪声分量独立且均值为零列为前提。",
        "独立性与零均值同时限定了所述关系的适用范围；当前摘录不能作为相关噪声或非零均值情形仍满足该关系的证据。",
    ),
    (
        (r"This paper shows that using dropout in both training and test is very effective.{0,180}only an input noisy image",
         r"main reason is that it can effectively reduce the variance of the prediction"),
        "作者在单张带噪图像训练场景中同时使用训练和测试阶段的 dropout，并将其作用解释为降低预测方差。",
        "该段明确指出 dropout 的两个使用阶段，并给出单图像训练与方差降低的适用语境，可用于核对测试阶段的操作。",
    ),
    (
        (r"Owing to model uncertainty introduced by dropout.{0,180}likely to have certain degree of statistical independence",
         r"the average of these predictions will reduce the variance of the result"),
        "dropout 引入模型不确定性，使多次预测可能具有一定统计独立性；对预测取平均用于降低结果方差。",
        "原文把模型不确定性、多次预测及平均后的方差降低相连；是否在测试时启用仍须结合测试方案原文核对。",
    ),
    (
        (r"We also allow spatially correlated noise models",
         r"we require the ability to sample from the noise distribution"),
        "方法允许空间相关噪声，但仍要求能够从噪声分布中采样。",
        "该段同时限定噪声相关性的范围和采样前提，可用于核对方法适用条件，不能推广为无需噪声模型。",
    ),
    (
        (r"Our method requires only a single noisy realization of each training example and a statistical model of the noise distribution",),
        "每个训练样本只需一份带噪观测，同时仍依赖噪声分布的统计模型。",
        "该段并列给出观测数量和噪声模型要求，可核对训练数据的放宽条件；单份观测不等于完全不需要噪声先验。",
    ),
    (
        (r"We can therefore recover an estimate of the clean image by doubling our network.s output and subtracting its input",),
        "将网络输出加倍，再减去网络输入，以恢复干净图像的估计。",
        "该段直接给出输出加倍、减去输入的校正方向；噪声同分布等适用条件仍须连同前文读取。",
    ),
    (
        (r"Blue bars denote the PSNR results of individual inferences, and red curves denote the cumulative average PSNR",),
        "蓝色柱形表示单次推理的 PSNR，红色曲线表示累计平均 PSNR。",
        "图注给出两种结果的展示方式，可用于核对柱形和曲线的统计口径；不能将单次推理与累计平均混为同一个量。",
    ),
    (
        (r"we train on the Color BSD68 dataset",
         r"Our reported values for Noise2Void should therefore be regarded as a lower bound on that method['’]s potential"),
        "摘录提及 Color BSD68 训练集，并将所报 Noise2Void 结果视为该方法潜在表现的下界。",
        "这里的下界限定的是 Noise2Void 的报告结果，不是性能上界；训练集对应哪个方法还应连同前文核对。",
    ),
    (
        (r"^(?:On the other hand,\s*)?we require the ability to sample from the noise distribution",),
        "作者明确要求能够从噪声分布中采样。",
        "这句把能够从噪声分布采样列为方法要求，可用于核对训练前提；不应据此推断分布未知时同样适用。",
    ),
)
_NEGATION = re.compile(r"\b(?:not|never|cannot|false\w*|incorrect\w*|untrue|fail\w*)\b|不能|并非|不等于", re.I)


def method_relation_copy(surface: str) -> tuple[str, str] | None:
    """Translate complete explicit clauses from evidence, retaining premises."""
    for patterns, summary, why in _RULES:
        valid = True
        for pattern in patterns:
            match = re.search(pattern, surface, re.I)
            if not match:
                valid = False
                break
            # Scope negation to this assertion, not an adjacent comparison
            # saying that another method does not have the same requirement.
            start = max(surface.rfind(c, 0, match.start()) for c in '.!?。！？') + 1
            if _NEGATION.search(surface[start:match.end()]):
                valid = False
                break
        if valid:
            return summary, why
    return None
