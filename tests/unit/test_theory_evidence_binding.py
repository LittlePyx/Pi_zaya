from __future__ import annotations

import pytest

from kb.claim_evidence_runtime import _support_score, audit_and_repair_claim_evidence
from kb.claim_evidence_runtime import _mentioned_source_hit_indexes
from kb.evidence_binding import assess_system_a_hit_binding
from kb.theory_evidence import normalize_supported_variance_relation


VARIANCE = (
    "For finite data, the variance of the estimate is the average variance of the "
    "corruptions in the targets, divided by the number of training samples."
)
CLAIM = "有限数据下，估计量的方差 = 目标中噪声的平均方差 ÷ 训练样本数。"
EQUATION = r"\mathbb{E}\|f(x)-x\|^2=\mathbb{E}\|f(x)-y\|^2+\mathbb{E}\|x-y\|^2."
SOURCE_EQUATION = "$$\nE ∥ f(x) - x ∥ 2 = E ∥ f(x) - y ∥ 2 + E ∥ x - y ∥ 2 .\n$$"
CONDITIONS = (
    "Proposition 1. Suppose x is an unbiased estimator of y, i.e. E[x|y]=y, "
    "and the noise in each subset J is independent from the noise in its "
    "complement Jc, conditional on y. Let f be J-invariant."
)
CONDITION_CLAIM = (
    "Unbiasedness means E[x|y]=y, and noise in each subset is independent "
    "of its complement conditional on y."
)

PIXEL_ASSUMPTIONS = (
    "In this work we make two simple statistical assumptions: (i) the signal s "
    "is not pixel-wise independent, (ii) the noise n is conditionally pixel-wise "
    "independent given the signal s."
)


@pytest.mark.parametrize("symbol", ["s", "$s$"])
def test_given_signal_condition_is_not_a_signal_independence_assertion(symbol):
    claim = (
        f"噪声 n 在给定信号 {symbol} 的条件下是逐像素独立的"
        "（conditionally pixel-wise independent given s）——即噪声在像素间没有条件相关性。"
    )
    assert _support_score(claim, PIXEL_ASSUMPTIONS) >= 5
    assert not binding(claim, PIXEL_ASSUMPTIONS)["suppress_link"]
    answer, _ = audit_and_repair_claim_evidence(
        "两条假设：\n\n" + claim + " [1]", [hit(PIXEL_ASSUMPTIONS)],
        allowed_citation_numbers={1}, drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True, enforce_user_visible_binding=True,
    )
    assert f"给定信号 {symbol} 的条件下" in answer
    assert "[1]" in answer


def test_real_conditional_pixel_noise_paraphrase_keeps_the_condition():
    claim = '对噪声 $n$ 的假设：噪声在给定信号 $s$ 的条件下是逐像素条件独立的（conditionally pixel-wise independent given the signal $s$），即噪声在像素之间没有相关性。'
    assert not binding(claim, PIXEL_ASSUMPTIONS)['suppress_link']
    answer, _ = audit_and_repair_claim_evidence(claim+' [1]', [hit(PIXEL_ASSUMPTIONS)], allowed_citation_numbers={1}, drop_unsupported_high_risk_claims=True, enforce_user_visible_binding=True)
    assert '给定信号 $s$ 的条件下是逐像素条件独立' in answer and '[1]' in answer


@pytest.mark.parametrize('claim', [
    '噪声在像素之间没有相关性。',
    '噪声在给定信号 $t$ 的条件下是逐像素条件独立的，即噪声在像素之间没有相关性。',
    '噪声在给定信号 $s$ 的条件下是逐像素条件独立的，即噪声在像素之间存在相关性。',
    '噪声在给定信号 $s$ 的条件下是逐像素条件独立的，即噪声在像素之间无条件没有相关性。',
])
def test_noise_paraphrase_does_not_grant_standalone_or_changed_correlation(claim):
    assert binding(claim, PIXEL_ASSUMPTIONS)['suppress_link']


@pytest.mark.parametrize("claim, source", [
    ("噪声 n 在给定信号 t 的条件下是逐像素独立的。", PIXEL_ASSUMPTIONS),
    ("信号 s 是逐像素独立的。", PIXEL_ASSUMPTIONS),
    ("噪声 n 在给定信号 s 的条件下不是逐像素独立的。", PIXEL_ASSUMPTIONS),
    ("噪声 n 在给定信号 s 的条件下是逐像素独立的。", "The signal s is not pixel-wise independent."),
    ("噪声 n 在给定信号 s 的条件下是逐像素独立的。", "The noise n is pixel-wise independent."),
    ("噪声 n 在给定信号 s 的条件下是逐像素独立的。", PIXEL_ASSUMPTIONS.replace("given the signal s", "given the signal t")),
    ("噪声 n 在给定信号 s 的条件下是逐像素独立的。", PIXEL_ASSUMPTIONS.replace("noise n is conditionally", "noise n is not conditionally")),
    ("噪声 n 在给定信号 s 的条件下是逐像素独立的。", PIXEL_ASSUMPTIONS.replace("given the signal s", "given the signal t") + " An unrelated measurement is given the signal s."),
])
def test_given_signal_scope_cannot_borrow_an_opposite_or_different_condition(claim, source):
    assert _support_score(claim, source) == 0
    assert binding(claim, source)["suppress_link"]


@pytest.mark.parametrize("claim", [
    "噪声在像素间没有条件相关性。",
    "噪声 n 在给定信号 s 的条件下是逐像素独立的，即噪声在像素间存在条件相关性。",
    "噪声 n 在给定信号 s 的条件下是逐像素独立的，即噪声在像素间没有条件相关性，并利用像素之间的相关性去噪。",
])
def test_conditional_noise_qualifier_cannot_grant_positive_correlation_support(claim):
    assert binding(claim, PIXEL_ASSUMPTIONS)["suppress_link"]


def hit(text, name="Theory"):
    return {"text": text, "meta": {
        "source_name": name,
        "citation_plan_evidence_authoritative": True,
        "citation_plan_full_evidence_quote": text,
        "citation_plan_evidence_quotes": [text],
        "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
    }}


def binding(claim, text=VARIANCE, name="Theory"):
    h = hit(text, name)
    return assess_system_a_hit_binding(answer_claim=claim, hit=h, meta=h["meta"],
        heading="Theory", evidence_quote=text, source_name=name)


def test_chinese_variance_relation_survives_both_gates():
    assert _support_score(CLAIM, VARIANCE) >= 5
    assert not binding(CLAIM)["suppress_link"]
    answer, _ = audit_and_repair_claim_evidence(CLAIM + " [1]", [hit(VARIANCE)],
        allowed_citation_numbers={1}, drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True, enforce_user_visible_binding=True)
    assert "平均方差" in answer and "[1]" in answer


@pytest.mark.parametrize("claim", [
    "有限数据下，估计方差等于目标污染平均方差乘以训练样本数。",
    "有限数据下，估计方差等于训练样本数除以目标污染平均方差。",
    "有限数据下，估计方差与训练样本数成正比。",
    "有限数据下，估计方差等于目标污染平均方差除以训练样本数的平方。",
    "有限数据下，估计方差等于目标污染平均方差除以训练样本数，再乘以2。",
    "For finite data, the variance of the estimate is the average variance of the corruptions in the targets multiplied by the number of training samples.",
])
def test_variance_relation_rejects_reversed_or_changed_claim(claim):
    assert _support_score(claim, VARIANCE) == 0
    assert binding(claim)["suppress_link"]


def test_variance_requires_relation_in_evidence_not_a_bag_of_terms():
    assert binding(CLAIM, "We measure the variance of the targets and estimate the number of training samples.")["suppress_link"]
    assert binding("Noise2Noise：" + CLAIM, VARIANCE, "Noise2Self")["suppress_link"]


def test_named_source_does_not_include_secondary_discussion():
    assert _mentioned_source_hit_indexes("Noise2Noise 需要成对观测。", [
        hit("We observe each source twice.", "Noise2Noise"),
        hit("Noise2Noise uses pairs; we use single images.", "Noise2Void"),
    ]) == [1]


def test_abbreviated_method_identity_matches_only_its_source():
    claim = "N2S 使用给定真实信号下噪声条件独立的特征组相互预测。"
    evidence = "We use groups of features whose noise is independent conditional on the true signal to predict one another."
    assert not binding(claim, evidence, "Noise2Self")["suppress_link"]
    assert binding(claim, evidence, "Noise2Void")["suppress_link"]


def test_source_equation_survives_atomic_audit_with_external_citation():
    answer, _ = audit_and_repair_claim_evidence("原文等式：\n\n$$\n" + EQUATION + "\n$$\n",
        [hit(SOURCE_EQUATION)], allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True, drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True)
    assert EQUATION in answer
    assert "[1]" in answer.split("$$")[-1]
    assert "[1]" not in answer.split("$$")[1]
    again, _ = audit_and_repair_claim_evidence(answer, [hit(SOURCE_EQUATION)],
        allowed_citation_numbers={1}, drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True, enforce_user_visible_binding=True)
    assert again.count("Source equation") == 1


@pytest.mark.parametrize("mutation", [
    lambda eq: eq.replace(r"\mathcal{J}", "J"),
    lambda eq: eq.replace("J^{c}", "J"),
    lambda eq: eq.replace("=E", "=2E"),
    lambda eq: eq + r"\in\mathcal{J}",
])
def test_partition_equation_requires_exact_font_complement_coefficient_and_tail(mutation):
    from kb.theory_evidence import protect_source_equations

    equation = r"f^{\ast}_{\mathcal{J}}(x)_{J}=E[y_{J}|x_{J^{c}}]"
    source = "$$" + equation + "$$"
    assert protect_source_equations(source, [hit(source)])[1]
    assert not protect_source_equations("$$" + mutation(equation) + "$$", [hit(source)])[1]
    assert not protect_source_equations(source, [hit("A different source without this formula")])[1]
    changed = mutation(equation)
    answer, _ = audit_and_repair_claim_evidence(
        "$$" + changed + "$$", [hit(source)], allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True, drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True)
    assert changed not in answer


def test_equation_conflict_veto_preserves_exact_short_and_unrelated_expressions():
    from kb.theory_evidence import drop_conflicting_source_equations

    source = [hit("$$a=b$$")]
    assert drop_conflicting_source_equations("$$a=b$$", source) == ("$$a=b$$", [])
    assert drop_conflicting_source_equations("$$u=v$$", source) == ("$$u=v$$", [])
    assert drop_conflicting_source_equations("$$a=2b$$", source)[1] == ["$$a=2b$$"]


@pytest.mark.parametrize("presentation", [
    lambda eq: eq.replace("|", r"\mid "),
    lambda eq: eq.replace(r"^{\ast}", "^*"),
    lambda eq: eq.replace("_{J}", "_J").replace("^{c}", "^c"),
])
def test_partition_equation_allows_equivalent_tex_script_and_bar_presentation(presentation):
    equation = r"f^{\ast}_{\mathcal{J}}(x)_{J}=E[y_{J}|x_{J^{c}}]"
    displayed = presentation(equation)
    answer, _ = audit_and_repair_claim_evidence(
        "$$" + displayed + "$$", [hit("$$" + equation + "$$")], allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True, drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True)
    assert displayed in answer and "Source equation [1]" in answer


def test_exact_plaintext_inline_set_assignment_survives_audit():
    source = r"For independent, mean-zero noise we choose $\mathcal{J}$ = {{1}, . . . , {m}} to be the singletons."
    equation = r"\mathcal{J} = \{\{1\}, \ldots, \{m\}\}"
    answer, _ = audit_and_repair_claim_evidence(
        "$$" + equation + "$$", [hit(source)], allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True, enforce_user_visible_binding=True)
    assert equation in answer and "Source equation [1]" in answer


@pytest.mark.parametrize("source", [
    r"We choose $\mathcal{J}$ = {{1}, . . . , {n}} to be the singletons.",
    r"We choose $\mathcal{J}$ = {{1}, . . . , {m} to be the singletons.",
    r"We choose $\mathcal{J}$ = {an unrelated description, m}.",
    r"We choose $J$ = {{1}, . . . , {m}} to be the singletons.",
    r"$$\mathcal{J} = {{1}, . . . , {m}}$$",
])
def test_inline_set_assignment_cannot_infer_members_font_or_visible_braces(source):
    from kb.theory_evidence import protect_source_equations
    answer = r"$$\mathcal{J} = \{\{1\}, \ldots, \{m\}\}$$"
    assert not protect_source_equations(answer, [hit(source)])[1]


def test_variance_does_not_use_negated_relation_as_support():
    assert binding(CLAIM, VARIANCE.replace("estimate is", "estimate is not"))["suppress_link"]


def test_supported_chinese_divide_sign_is_spelled_out():
    answer, count = normalize_supported_variance_relation(
        "估计的方差 = 目标中污染的平均方差 ÷ 训练样本数 [1]。",
        prompt="估计方差、目标污染方差和样本数是什么关系？",
        evidence=[VARIANCE],
    )
    assert count == 1 and "除以训练样本数" in answer and "÷" not in answer


def test_divide_sign_is_unchanged_without_the_exact_source_relation():
    answer = "估计的方差 = 目标中污染的平均方差 ÷ 训练样本数 [1]。"
    assert normalize_supported_variance_relation(
        answer,
        prompt="估计方差、目标污染方差和样本数是什么关系？",
        evidence=["We measure variance and count training samples."],
    ) == (answer, 0)


@pytest.mark.parametrize("equation", [EQUATION.replace("+", "-"), EQUATION.replace("^2", "^3", 1)])
def test_changed_equation_is_not_preserved_by_word_overlap(equation):
    answer, _ = audit_and_repair_claim_evidence("$$\n" + equation + "\n$$\n", [hit(SOURCE_EQUATION)],
        allowed_citation_numbers={1}, drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True, enforce_user_visible_binding=True)
    assert equation not in answer


@pytest.mark.parametrize("punctuation", [",", ";", "，", "；"])
def test_source_equation_sentence_punctuation_survives_audit(punctuation):
    equation = EQUATION.rstrip(".") + punctuation
    answer, _ = audit_and_repair_claim_evidence(
        "$$\n" + equation + "\n$$", [hit(SOURCE_EQUATION)],
        allowed_citation_numbers={1}, drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True, enforce_user_visible_binding=True,
    )
    assert equation in answer and "Source equation [1]" in answer


@pytest.mark.parametrize("equation", [
    EQUATION.replace("+", "-").rstrip(".") + ",",
    EQUATION.replace("^2", "^3", 1).rstrip(".") + ",",
    EQUATION.replace("f(x)-y", "f(y)-x").rstrip(".") + ",",
])
def test_sentence_punctuation_cannot_protect_changed_equation(equation):
    answer, _ = audit_and_repair_claim_evidence(
        "$$\n" + equation + "\n$$", [hit(SOURCE_EQUATION)],
        allowed_citation_numbers={1}, drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True, enforce_user_visible_binding=True,
    )
    assert equation not in answer


def test_equation_requires_selected_evidence_and_allowed_citation():
    equation = EQUATION.rstrip(".") + ","
    for hits, allowed in [([hit("This source contains no equation.")], {1}),
                          ([hit(SOURCE_EQUATION), hit("Other source.")], {2})]:
        answer, _ = audit_and_repair_claim_evidence(
            "$$\n" + equation + "\n$$", hits,
            allowed_citation_numbers=allowed, drop_unsupported_unplanned_claims=True,
            drop_unsupported_high_risk_claims=True, enforce_user_visible_binding=True,
        )
        assert equation not in answer


def test_inline_display_equation_survives_sentence_citation_repair():
    source = "The expected loss is the same as that of " + SOURCE_EQUATION
    answer = "Source-grounded detail: " + " ".join(source.split()) + " [1]."
    for _ in range(2):
        answer, _ = audit_and_repair_claim_evidence(
            answer, [hit(source)], allowed_citation_numbers={1},
            drop_unsupported_unplanned_claims=True,
            drop_unsupported_high_risk_claims=True,
            enforce_user_visible_binding=True,
        )
        assert "source_equation_" not in answer
        assert "E ∥ f(x) - x ∥ 2" in answer
        assert answer.count("Source equation [1]") == 1


def test_equation_number_is_presentation_but_its_operands_are_not():
    from kb.theory_evidence import protect_source_equations

    source = SOURCE_EQUATION.replace(".\n$$", r". \tag{2}" + "\n$$")
    answer = "$$\n" + EQUATION.rstrip(".") + ",\n$$"
    protected, replacements = protect_source_equations(answer, [hit(source)])
    assert replacements and "source_equation_0" in protected
    assert not protect_source_equations(answer.replace("+", "-"), [hit(source)])[1]


def test_complete_measurement_conditions_keep_source_direction():
    assert _support_score(CONDITION_CLAIM, CONDITIONS) >= 5
    assert not binding(CONDITION_CLAIM, CONDITIONS, "Noise2Self")["suppress_link"]


@pytest.mark.parametrize("claim", [
    CONDITION_CLAIM.replace("E[x|y]=y", "E[y|x]=x"),
    CONDITION_CLAIM.replace("E[x|y]=y", "E[x|y]=2y"),
    CONDITION_CLAIM.replace("E[x|y]=y", "E[x|y]=y+1"),
    CONDITION_CLAIM.replace("E[x|y]=y", "E[x|y]=y/2"),
    CONDITION_CLAIM.replace("E[x|y]=y", "E[x|y]=y^2"),
    CONDITION_CLAIM.replace("E[x|y]=y", "E[x|y]=y+z"),
    CONDITION_CLAIM.replace("E[x|y]=y", "E[x]=E[y]"),
    CONDITION_CLAIM.replace("conditional on y", "unconditionally"),
])
def test_word_overlap_cannot_validate_changed_measurement_conditions(claim):
    assert _support_score(claim, CONDITIONS) == 0
    assert binding(claim, CONDITIONS, "Noise2Self")["suppress_link"]


def test_incomplete_source_cannot_supply_missing_formal_condition():
    partial = CONDITIONS.replace("E[x|y]=y, and the noise in each subset J is indepen", "")
    assert _support_score(CONDITION_CLAIM, partial) == 0
    assert binding(CONDITION_CLAIM, partial, "Noise2Self")["suppress_link"]
    assert binding("Noise2Self: " + CONDITION_CLAIM, CONDITIONS, "Noise2Void")["suppress_link"]


def test_dncnn_architecture_relation_requires_exact_formula_and_pooling_fact():
    evidence = (
        "We set the size of convolutional filters to 3 × 3 but remove all pooling layers. "
        "The receptive field of DnCNN with depth of d should be (2d+1)×(2d+1)."
    )
    supported = (
        "DnCNN 移除所有池化层，卷积核为 3×3；深度为 d 时，"
        "感受野为 (2d+1)×(2d+1)。"
    )
    assert _support_score(supported, evidence) >= 5
    assert not binding(supported, evidence, "DnCNN")["suppress_link"]
    for changed in (
        supported.replace("2d+1", "2d+3"),
        supported.replace("移除所有池化层", "保留池化层"),
    ):
        assert binding(changed, evidence, "DnCNN")["suppress_link"]


def test_noisy_target_condition_rejects_reversed_expectation_and_bias():
    evidence = (
        "Input-conditioned target distributions may be replaced by arbitrary distributions that have "
        "the same conditional expected values. Targets can be corrupted with zero-mean noise without "
        "changing what the network learns."
    )
    claim = "替代目标与原目标具有相同的条件期望；零均值噪声不改变网络学到的映射。"
    assert _support_score(claim, evidence) >= 5
    assert not binding(claim, evidence, "Noise2Noise")["suppress_link"]
    assert binding(
        "替代目标与原目标具有不同的条件期望；零均值噪声仍不改变网络学到的映射。",
        evidence,
        "Noise2Noise",
    )["suppress_link"]
    assert binding(
        claim,
        evidence.replace("same conditional expected values", "different conditional expected values"),
        "Noise2Noise",
    )["suppress_link"]


def test_three_task_list_cannot_replace_jpeg_with_an_unstated_task():
    evidence = (
        "We consider three specific tasks: blind Gaussian denoising, SISR, and JPEG deblocking."
    )
    supported = "三类任务是盲高斯去噪、单图像超分辨率和 JPEG 去块。"
    assert _support_score(supported, evidence) >= 5
    assert not binding(supported, evidence, "DnCNN")["suppress_link"]
    assert binding(
        "三类任务是盲高斯去噪、单图像超分辨率和 MRI 重建。",
        evidence,
        "DnCNN",
    )["suppress_link"]


def test_training_pair_requirement_needs_both_pair_types():
    evidence = "N2V does not require noisy image pairs, nor clean target images."
    claim = "N2V 不需要独立带噪图像对，也不需要干净目标图像。"
    assert _support_score(claim, evidence) >= 5
    assert not binding(claim, evidence, "Noise2Void")["suppress_link"]
    assert binding(
        claim,
        "N2V does not require noisy image pairs, but still uses clean target images.",
        "Noise2Void",
    )["suppress_link"]


def test_signal_and_noise_assumption_directions_are_not_interchangeable():
    evidence = (
        "The signal is not pixel-wise independent. The noise is conditionally pixel-wise "
        "independent given the signal."
    )
    claim = "真实信号不是逐像素独立的；给定真实信号时，噪声逐像素条件独立。"
    assert _support_score(claim, evidence) >= 5
    assert not binding(claim, evidence, "Noise2Void")["suppress_link"]
    assert binding(
        "真实信号逐像素独立；给定真实信号时，噪声不是逐像素条件独立。",
        evidence,
        "Noise2Void",
    )["suppress_link"]
