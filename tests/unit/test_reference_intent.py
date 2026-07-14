from __future__ import annotations

from api.reference_intent import (
    refs_prompt_section_intent,
    refs_prompt_topic_terms,
    refs_section_intent_heading_score,
    refs_section_intent_terms,
)


def test_related_intent_handles_natural_chinese_prior_work_question() -> None:
    prompt = "ADMM 是怎么来的？作者这里是不是借鉴了别人以前的想法？"

    assert refs_prompt_section_intent(prompt) == "related"
    assert "ADMM" in refs_prompt_topic_terms(prompt)
    assert "ADMM" in refs_section_intent_terms(prompt, "related")


def test_domain_term_is_not_in_related_base_terms_without_prompt_signal() -> None:
    terms = refs_section_intent_terms("作者之前是谁做的？", "related")

    assert "ADMM" not in terms
    assert "ADMM-Net" not in terms


def test_technical_phrase_extraction_is_generic_not_fixed_to_one_paper() -> None:
    terms = refs_prompt_topic_terms("我想理解 snapshot compressive imaging 的基本问题")

    assert "snapshot compressive imaging" in terms


def test_heading_scores_follow_intent_without_paper_specific_terms() -> None:
    related_prompt = "这个想法之前是谁做过？"
    method_prompt = "它的方法流程是怎么工作的？"
    exp_prompt = "实验结果能不能支撑结论？"

    assert refs_section_intent_heading_score(related_prompt, "2. Related Work") > refs_section_intent_heading_score(
        related_prompt,
        "3. Method",
    )
    assert refs_section_intent_heading_score(method_prompt, "3. Proposed Framework") > refs_section_intent_heading_score(
        method_prompt,
        "4. Experiments",
    )
    assert refs_section_intent_heading_score(exp_prompt, "4. Experimental Results") > refs_section_intent_heading_score(
        exp_prompt,
        "3. Method",
    )


def test_problem_intent_handles_plain_beginner_question() -> None:
    prompt = "这篇文章到底解决了什么问题？为什么要做这个研究？"

    assert refs_prompt_section_intent(prompt) == "problem"
    assert refs_section_intent_heading_score(prompt, "1. Introduction") > refs_section_intent_heading_score(
        prompt,
        "References",
    )


def test_problem_intent_handles_research_line_scope_question() -> None:
    prompt = "这篇 perovskite laser 和我的单像素成像主线关系大吗？值得一起读吗？"

    assert refs_prompt_section_intent(prompt) == "problem"
    assert refs_section_intent_heading_score(prompt, "Abstract") > 0


def test_experiment_intent_wins_when_user_asks_if_method_is_reliable() -> None:
    prompt = "这个方法靠谱吗？实验是不是有点少，结论能不能站得住？"

    assert refs_prompt_section_intent(prompt) == "experiments"


def test_method_intent_handles_reproduction_question() -> None:
    prompt = "我想复现这个方法，应该从哪个流程和关键步骤看起？"

    assert refs_prompt_section_intent(prompt) == "method"


def test_method_intent_handles_chinese_digital_refocusing_question() -> None:
    prompt = "这个量子关联光场显微镜怎么把离焦样品重新对焦？"

    assert refs_prompt_section_intent(prompt) == "method"
    assert refs_section_intent_heading_score(prompt, "A. Concept") > 0
    assert refs_section_intent_heading_score(prompt, "Digital Refocusing Procedure") > 0


def test_related_intent_handles_authorship_origin_question() -> None:
    prompt = "这个想法是作者自己发明的吗，还是借鉴了前人的路线？"

    assert refs_prompt_section_intent(prompt) == "related"
