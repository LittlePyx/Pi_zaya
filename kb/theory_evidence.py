from __future__ import annotations

import re

from kb.source_equation_literals import source_equation_literals
from kb.waived_data_requirements import waived_data_requirements_covered


_ESTIMATE_VARIANCE = r"(?:估计(?:量|值)?(?:的)?方差|variance\s+of\s+the\s+estimate)"
_TARGET_VARIANCE = r"(?:目标.{0,16}方差|(?:average\s+)?variance\s+of\s+the\s+corruptions\s+in\s+the\s+targets)"
_SAMPLES = r"(?:训练)?样本(?:数|数量)|(?:number\s+of\s+)?training\s+samples"
_DIVIDE = r"(?:除以|除|÷|/|divided\s+by)"
_RECEPTIVE_FORMULA = (
    r"\(\s*2\s*d\s*\+\s*1\s*\)\s*(?:[×x]|\\times)\s*"
    r"\(\s*2\s*d\s*\+\s*1\s*\)"
)


def variance_relation_covered(claim: str, evidence: str) -> bool:
    """Check operand order for the finite-sample variance relation, not overlap.

    This is a narrow deterministic relation check, not general entailment.
    Unsupported algebraic rewrites remain conservative.
    """
    if not (re.search(_ESTIMATE_VARIANCE, claim, re.I) and re.search(_SAMPLES, claim, re.I)):
        return True
    if re.search(r"瓶颈|bottleneck", claim, re.I) and not re.search(r"瓶颈|bottleneck", evidence, re.I):
        return False
    if not re.search(r"等于|[=＝÷*/×]|除|乘|正比|反比|倒数|\bis\b|proportional|divided|multiplied", claim, re.I):
        return True
    forward = rf"{_ESTIMATE_VARIANCE}.{{0,35}}{_TARGET_VARIANCE}\s*[,，]?\s*{_DIVIDE}\s*(?:the\s+)?(?:{_SAMPLES})"
    inverse = rf"{_ESTIMATE_VARIANCE}.{{0,20}}(?:{_SAMPLES}).{{0,8}}(?:反比|倒数)"
    evidence_relation = re.search(forward, evidence, re.I)
    if not evidence_relation or re.search(r"\b(?:not|never)\b|不等于|并非", evidence_relation[0], re.I):
        return False
    if re.search(r"乘|×|\*|平方|立方|squared|cubed|multiplied|times|不等于|并非|not\s+equal", claim, re.I):
        return False
    return bool(re.search(forward, claim, re.I) or re.search(inverse, claim, re.I))


def architecture_relation_covered(claim: str, evidence: str) -> bool:
    """Require the exact source formula when a claim states DnCNN's field size."""
    if re.search(r"池化|pooling", claim, re.I):
        claim_removes = bool(
            re.search(r"(?:移除|去掉|不保留).{0,24}池化|remove(?:s|d)?\s+(?:all\s+)?pooling", claim, re.I)
        )
        claim_retains = bool(
            re.search(r"(?:保留|使用|含有).{0,24}池化|retain(?:s|ed)?\s+pooling|use(?:s|d)?\s+pooling", claim, re.I)
        )
        evidence_removes = bool(
            re.search(r"remove(?:s|d)?\s+(?:all\s+)?pooling|(?:移除|去掉).{0,24}池化", evidence, re.I)
        )
        evidence_retains = bool(
            re.search(r"retain(?:s|ed)?\s+pooling|use(?:s|d)?\s+pooling|(?:保留|使用).{0,24}池化", evidence, re.I)
        )
        if claim_removes and not evidence_removes:
            return False
        if claim_retains and not evidence_retains:
            return False
    if not re.search(r"感受野|receptive\s+field", claim, re.I):
        return True
    if not re.search(r"2\s*d.{0,12}[×x\\]?(?:times)?.{0,12}2\s*d", claim, re.I):
        return True
    return bool(
        re.search(_RECEPTIVE_FORMULA, claim, re.I)
        and re.search(_RECEPTIVE_FORMULA, evidence, re.I)
        and not re.search(r"\b(?:not|never)\b|不等于|并非", evidence, re.I)
    )


def noisy_target_relation_covered(claim: str, evidence: str) -> bool:
    """Preserve the direction of the noisy-target conditional-mean condition."""
    input_replacement = r"(?:替换|污染|替代|更换).{0,10}(?:训练)?输入|(?:replac(?:e[sd]?|ing)|corrupt(?:s|ing)?)\s+(?:the\s+)?(?:training\s+)?inputs?"
    if not (
        re.search(r"条件期望|\bconditional\s+expect|\bunconditional\s+expect", claim, re.I)
        and (re.search(r"目标|target", claim, re.I) or re.search(input_replacement, claim, re.I))
    ):
        return True
    # A number in a section heading cannot establish the loss function, and
    # an unrelated numeric fact cannot establish the mean of target noise.
    def loss_ids(text: str) -> set[str]:
        return set(re.findall(r"\bL\s*(\d+)\b", re.sub(r"[$_{}]", " ", text), re.I))

    if not loss_ids(claim) <= loss_ids(evidence):
        return False
    mean_pattern = r"(?:noise\s+with\s+mean|mean\s*(?:of|=)?|均值(?:为|等于|是|=))\s*(-?\d+(?:\.\d+)?)"
    means = {float(m) for m in re.findall(mean_pattern, evidence, re.I)}
    if re.search(r"zero[- ]mean|零均值", evidence, re.I):
        means.add(0.0)
    if not {float(m) for m in re.findall(mean_pattern, claim, re.I)} <= means:
        return False
    if re.search(input_replacement, claim, re.I):
        # Mentioning corrupted inputs does not move a target-replacement
        # theorem's condition onto those inputs.
        if re.search(r"input[- ]conditioned\s+target\s+distributions", evidence, re.I) or not re.search(input_replacement, evidence, re.I):
            return False
    if re.search(r"无条件期望|\bunconditional\s+expect", claim, re.I) and not re.search(r"无条件期望|\bunconditional\s+expect", evidence, re.I):
        return False
    if re.search(r"target[- ]conditioned\s+input\s+distributions", evidence, re.I):
        return False
    learned_en = r"(?:(?:the\s+)?(?:optimal\s+)?(?:network\s+)?parameters|(?:the\s+)?learned\s+(?:mapping|function)|what\s+the\s+network\s+learns)"
    learned_zh = r"(?:最优(?:网络)?参数|网络参数|最优解|(?:网络)?(?:学到|所学)(?:的)?(?:映射|内容|东西))"
    outcome_change = (
        rf"\bchang(?:e[sd]?|ing)\s+{learned_en}\b|\b{learned_en}\s+(?:(?:will|would)\s+)?chang(?:e[sd]?|ing)\b|"
        rf"改变(?:了)?{learned_zh}|{learned_zh}(?:会|将|发生)?改变"
    )
    changes = any(
        not re.search(r"(?:\b(?:not|without|cannot|never)|不(?:会)?|未|没有)\s*$", claim[:m.start()], re.I)
        for m in re.finditer(outcome_change, claim, re.I)
    )
    if changes and re.search(r"without\s+changing|unchanged|不改变|不变", evidence, re.I):
        return False
    claim_same = bool(re.search(r"相同|不变|等于|same|unchanged|equal", claim, re.I))
    claim_different = bool(re.search(r"不同|改变|different|changed?", claim, re.I))
    evidence_same = bool(
        re.search(r"same\s+conditional\s+expected\s+values|相同.{0,24}条件期望|条件期望.{0,24}相同", evidence, re.I)
    ) and not bool(re.search(r"not\s+(?:the\s+)?same\s+conditional|不相同.{0,12}条件期望|条件期望.{0,12}不相同", evidence, re.I))
    evidence_different = bool(
        re.search(r"different\s+conditional\s+expected\s+values|不同.{0,24}条件期望|条件期望.{0,24}不同", evidence, re.I)
    )
    if re.search(r"条件期望.{0,8}(?:不同|不相同|不一致|不等)|(?:不同|不相同|不一致).{0,8}条件期望|different\s+conditional\s+expect", claim, re.I):
        return evidence_different
    if claim_same:
        if not evidence_same:
            return False
        if re.search(r"零均值|zero[- ]mean", claim, re.I):
            return bool(re.search(r"zero[- ]mean|零均值", evidence, re.I))
        return True
    if claim_different:
        return evidence_different
    return True


def restoration_task_enumeration_covered(claim: str, evidence: str) -> bool:
    """Do not validate a three-task list from only two overlapping task names."""
    applications = (r"照片去噪|photographic\s+noise\s+removal", r"蒙特卡洛|Monte\s+Carlo", r"MRI|磁共振")
    if all(re.search(pattern, claim, re.I) for pattern in applications):
        return all(re.search(pattern, evidence, re.I) for pattern in applications) and (
            not re.search(r"欠采样|undersampled", claim, re.I)
            or bool(re.search(r"undersampled|欠采样", evidence, re.I))
        )
    task_prompt = bool(
        re.search(r"三(?:类|种|项).{0,20}任务|three.{0,20}tasks?", claim, re.I)
    )
    if not task_prompt or not re.search(r"高斯去噪|Gaussian\s+denois", claim, re.I):
        return True
    patterns = (
        r"高斯去噪|Gaussian\s+denois",
        r"超分辨|super[- ]resolution|\bSISR\b",
        r"JPEG.{0,16}(?:去块|deblock)",
    )
    named_count = sum(bool(re.search(pattern, claim, re.I)) for pattern in patterns)
    if named_count < 2:
        return True
    return all(re.search(pattern, claim, re.I) for pattern in patterns) and all(
        re.search(pattern, evidence, re.I) for pattern in patterns
    )


def training_pair_requirement_covered(claim: str, evidence: str) -> bool:
    """Require each explicitly waived data type within its own negation scope."""
    return waived_data_requirements_covered(claim, evidence)


def _noise_given_signal_variables(evidence: str) -> list[str]:
    return re.findall(
        r"\bnoise\s+(?:\$?[A-Za-z]\$?\s+)?(?:is\s+)?conditionally\s+"
        r"pixel[- ]wise\s+independent\s+given\s+(?:the\s+)?(?:true\s+)?signal\b"
        r"(?:\s*\$?([A-Za-z])\b\$?)?", evidence, re.I,
    )


def _signal_subject_surface(claim: str, evidence: str) -> str:
    variables = _noise_given_signal_variables(evidence)
    if not variables:
        return claim
    # The verified conditioning object is not the subject of independence.
    return re.sub(
        r"(?:在)?给定(?:真实)?信号(?:\s*\$?([A-Za-z])\b\$?)?\s*(?:的条件下|条件下|之后|后|时)",
        lambda m: " " if not m[1] or m[1] in variables else m[0], claim,
    )


def conditional_noise_domain_surface(claim: str, evidence: str) -> str:
    """Keep a verified negative noise qualifier out of positive-domain tags."""
    if (
        _signal_subject_surface(claim, evidence) != claim
        and re.search(r"噪声.{0,40}给定.{0,24}的条件下是逐像素(?:条件)?独立", claim)
        and not re.search(r"无条件|无论|unconditional|regardless", claim, re.I)
        and signal_noise_assumptions_covered(claim, evidence)
    ):
        return re.sub(r"噪声\s*(?:\$?[A-Za-z]\$?\s*)?在像素(?:间|之间)没有(?:条件)?相关性", " ", claim)
    return claim


def signal_noise_assumptions_covered(claim: str, evidence: str) -> bool:
    """Keep the signal and noise pixel-independence directions distinct."""
    evidence_signal_not = bool(
        re.search(r"signal.{0,30}(?:is\s+)?not\s+pixel[- ]wise\s+independent", evidence, re.I)
    )
    evidence_noise_conditional = bool(
        re.search(
            r"noise.{0,45}conditionally\s+pixel[- ]wise\s+independent.{0,30}(?:given\s+the\s+signal|signal)",
            evidence,
            re.I,
        )
    )
    signal_subject_surface = _signal_subject_surface(claim, evidence)
    if re.search(r"(?:真实)?信号.{0,24}逐像素.{0,8}独立|signal.{0,30}pixel[- ]wise\s+independent", signal_subject_surface, re.I):
        claim_signal_not = bool(
            re.search(r"(?:真实)?信号.{0,24}(?:不是|并非|不).{0,12}逐像素.{0,8}独立|signal.{0,30}not\s+pixel[- ]wise\s+independent", signal_subject_surface, re.I)
        )
        if claim_signal_not != evidence_signal_not:
            return False
    if re.search(
        r"噪声.{0,40}(?:逐像素.{0,8}(?:条件)?独立|像素之间.{0,8}条件独立)|"
        r"noise.{0,50}(?:conditionally\s+)?pixel[- ]wise\s+independent",
        claim,
        re.I,
    ):
        claim_noise_negated = bool(
            re.search(
                r"噪声.{0,24}(?:不是|并非|不).{0,12}(?:逐像素|像素之间)|"
                r"noise.{0,30}not.{0,16}independent",
                claim,
                re.I,
            )
        )
        if claim_noise_negated or not evidence_noise_conditional:
            return False
    return True


def measurement_conditions_covered(claim: str, evidence: str) -> bool:
    """Check explicit x/y unbiasedness conditions without filling source gaps.

    This is a literal relation guard, not an algebra solver. In particular,
    unconditional expectations cannot stand in for the conditional condition.
    """
    if re.search(r"unbiased|无偏", claim, re.I):
        expectation = r"E\s*\[\s*[xy]\s*(?:\|\s*[xy]\s*)?\]"
        relation = re.compile(
            expectation + r"\s*=\s*(?:" + expectation
            + r"|[+-]?\d*\s*[xy](?:\s*[-+*/^]\s*(?:\d+|[xy]|\{[^}]+\}))*)"
        )

        def equations(text: str) -> set[str] | None:
            text = re.sub(r"\\(?:mathbb|mathrm)\s*\{E\}", "E", text)
            text = text.replace(r"\mid", "|")
            matches = list(relation.finditer(text))
            starts = list(re.finditer(expectation + r"\s*=", text))
            if len(starts) != len(matches) or any(
                re.match(r"\s*[-+*/^=]", text[m.end():]) for m in matches
            ):
                return None
            return {re.sub(r"\s+", "", m[0]) for m in matches}

        claim_equations, source_equations = equations(claim), equations(evidence)
        if claim_equations is None or not claim_equations.issubset(source_equations or set()):
            return False
    if re.search(r"noise|噪声", claim, re.I) and re.search(
        r"unconditional(?:ly)?|无条件.{0,8}独立", claim, re.I
    ):
        return bool(re.search(r"unconditional(?:ly)?|无条件.{0,8}独立", evidence, re.I))
    return True


def normalized_equation(text: str) -> str:
    """Normalize presentation only; preserve operands, signs and exponents."""
    text = re.sub(r"\\(?:ldots|dots)\b", "...", text)
    text = re.sub(r"\\mid\b", "|", text)
    text = re.sub(r"\\ast\b", "*", text)
    text = re.sub(r"([_^])\s*\{\s*([A-Za-z0-9*])\s*\}", r"\1\2", text)
    text = re.sub(r"\\(?:mathbb|mathrm)\s*\{E\}", "E", text)
    text = text.replace(r"\Vert", "∥").replace(r"\|", "∥").replace("||", "∥")
    text = re.sub(r"\\(?:left|right|,|;|!)", "", text)
    text = re.sub(r"\\tag\{\d+\}\s*$", "", text.strip())
    text = re.sub(r"\^\{(\d+)\}", r"^\1", text)
    # The converter emits Unicode norm followed by a plain superscript digit.
    text = re.sub(r"∥\s*\^?\s*(\d+)", r"∥^\1", text)
    return re.sub(r"\s+", "", text).strip("$ .。,，;；")


def normalize_supported_variance_relation(
    answer: str,
    *,
    prompt: str,
    evidence: list[str],
) -> tuple[str, int]:
    """Spell out a supported Chinese divide sign for stable reading/auditing."""
    if not (re.search(r"方差|variance", prompt, re.I) and re.search(_SAMPLES, prompt, re.I)):
        return answer, 0
    canonical = "有限数据下，估计方差等于目标污染平均方差除以训练样本数。"
    if not any(variance_relation_covered(canonical, item) for item in evidence):
        return answer, 0
    pattern = re.compile(
        r"(?P<numerator>目标.{0,16}方差)\s*÷\s*(?P<denominator>(?:训练)?样本(?:数|数量))"
    )
    normalized, count = pattern.subn(r"\g<numerator>除以\g<denominator>", answer)
    return normalized, count


def protect_source_equations(answer: str, hits: list[dict]) -> tuple[str, dict[str, str]]:
    """Shield exact source equations from line-oriented prose editing.

    Only use the evidence actually selected for the citation. A nearby formula
    or a different source's formula cannot validate a changed equation.
    """
    equations: dict[str, str] = {}
    for number, hit in enumerate(hits, 1):
        meta = hit.get("meta") or {}
        text = str(meta.get("citation_plan_full_evidence_quote") or hit.get("text") or "")
        for literal in source_equation_literals(text):
            key = normalized_equation(literal)
            if "=" in key and len(key) >= 12:
                equations.setdefault(key, str(number))
    protected: dict[str, str] = {}

    def replace(match: re.Match) -> str:
        number = equations.get(normalized_equation(match[1]))
        if not number:
            return match[0]
        token = f"<!-- source_equation_{len(protected)} -->"
        protected[token] = "$$" + match[1] + "$$" + f"\n\n原文等式 / Source equation [{number}]."
        # Keep the opaque marker on its own line. Inline prose citation repair
        # can split the exclamation mark in an HTML comment and prevent restore.
        return f"\n\n{token}\n\n"

    # Finalization can run twice. Consume our previous external source note
    # together with the equation so a recheck cannot duplicate it.
    pattern = r"\$\$(.*?)\$\$(?:\s*原文等式 / Source equation(?:\s*\[\d+\])?\s*\.)*"
    return re.sub(pattern, replace, answer, flags=re.S), protected


def drop_conflicting_source_equations(answer: str, hits: list[dict]) -> tuple[str, list[str]]:
    """Veto a changed equality for the same source-defined expression.

    This runs after exact formulas have been protected. Ignoring alphabet style
    on the left only detects a potential conflict; it never grants support.
    Unrelated expressions remain subject to the ordinary claim audit.
    """
    def subject(equation: str) -> str:
        lhs, separator, _ = normalized_equation(equation).partition("=")
        if not separator:
            return ""
        return normalized_equation(re.sub(r"\\(?:mathcal|mathbf|mathrm)\{([^{}]+)\}", r"\1", lhs))

    subjects = set()
    equations = set()
    for hit in hits:
        text = str((hit.get("meta") or {}).get("citation_plan_full_evidence_quote") or hit.get("text") or "")
        for literal in source_equation_literals(text):
            subjects.add(subject(literal))
            equations.add(normalized_equation(literal))
    subjects.discard("")
    removed = []

    def replace(match):
        if normalized_equation(match[1]) in equations or subject(match[1]) not in subjects:
            return match[0]
        removed.append(match[0])
        return ""

    return re.sub(r"\$\$(.*?)\$\$", replace, answer, flags=re.S), removed
