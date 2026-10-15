from __future__ import annotations

import pytest

from kb.denoising_quality_boundary import (
    complete_denoising_quality_boundary,
    complete_n2n_corruption_model_boundary,
)


N2N = (
    "Under certain common circumstances, it is possible to learn without clean ones, "
    "at performance close or equal to training using clean exemplars."
)
N2V = (
    "N2V cannot be expected to outperform methods that have more information available during training. "
    "Still, denoising performance drops in moderation."
)


def _plan(*slots: tuple[str, str, int]) -> dict:
    return {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": f"C:/papers/{source}/{source}.en.md",
                "source_name": source,
                "evidence_quote": evidence,
                "candidate_hits": [number],
            }
            for source, evidence, number in slots
        ]
    }


def test_pair_completion_keeps_each_relation_on_its_own_source() -> None:
    result = complete_denoising_quality_boundary(
        "Partial answer.",
        prompt=(
            "Do these papers justify saying that removing the requirement for paired noisy observations "
            "necessarily improves denoising quality? Compare the statistical justification and the stated information trade-off."
        ),
        citation_plan=_plan(("noise2noise", N2N, 2), ("noise2void", N2V, 1)),
    )
    assert "Noise2Noise" in result and "close or equal" in result and "[2]" in result
    assert "Noise2Void" in result and "more information" in result and "[1]" in result
    assert "does not guarantee" in result


def test_single_completion_states_negative_information_boundary() -> None:
    result = complete_denoising_quality_boundary(
        "原回答遗漏了边界。",
        prompt="本文是否声称这种训练方式应当胜过拥有干净目标或独立噪声对的训练？作者怎样解释这个性能边界？",
        citation_plan=_plan(("noise2void", N2V, 3)),
    )
    assert "不能" in result and "信息" in result and "并不意味着" in result and "[3]" in result


def test_completion_rejects_reverse_incomplete_and_cross_source_relations() -> None:
    prompt = (
        "Do these papers justify saying that removing the requirement for paired noisy observations "
        "necessarily improves denoising quality? Compare the statistical justification and the stated information trade-off."
    )
    bad_n2v = "N2V is expected to outperform methods despite having less information during training."
    assert complete_denoising_quality_boundary(
        "Keep me.", prompt=prompt, citation_plan=_plan(("noise2noise", N2N, 1), ("noise2void", bad_n2v, 2))
    ) == "Keep me."
    assert complete_denoising_quality_boundary(
        "Keep me.", prompt=prompt, citation_plan=_plan(("noise2noise", N2N + " " + N2V, 1))
    ) == "Keep me."
    assert complete_denoising_quality_boundary(
        "Keep me.", prompt=prompt, citation_plan=_plan(("noise2noise", N2N, 1), ("other-paper", N2V, 2))
    ) == "Keep me."


def test_single_completion_requires_the_full_relation_on_one_surface() -> None:
    prompt = "本文是否声称这种训练方式应当胜过拥有干净目标或独立噪声对的训练？作者怎样解释这个性能边界？"
    incomplete = "N2V does not require noisy image pairs and uses less information."
    assert complete_denoising_quality_boundary(
        "保留原文。", prompt=prompt, citation_plan=_plan(("noise2void", incomplete, 1))
    ) == "保留原文。"


N2N_CORRUPTION = (
    "In stark contrast to our approach, they need a perfect, explicit computational model of the corruption; "
    "our requirement of the knowledge of the appropriate statistical summary (mean, median) is significantly less limiting."
)
N2N_CORRUPTION_PROMPT = (
    "Does this method require a likelihood or an explicit model of the corruption process? "
    "What statistical condition remains necessary?"
)
N2N_CORRUPTION_ANSWER = (
    "No, Noise2Noise does not require an explicit model of the corruption process [1]. "
    "It still requires the appropriate statistical summary of the corrupted-target distribution [1]."
)


def test_n2n_corruption_completion_names_the_directly_supported_summary() -> None:
    result = complete_n2n_corruption_model_boundary(
        N2N_CORRUPTION_ANSWER,
        prompt=N2N_CORRUPTION_PROMPT,
        citation_plan=_plan(("noise2noise", N2N_CORRUPTION, 2)),
    )
    assert "mean or median" in result
    assert "[2]" in result


def test_n2n_corruption_completion_keeps_an_already_complete_answer() -> None:
    answer = N2N_CORRUPTION_ANSWER + " The needed summary can be the mean."
    assert complete_n2n_corruption_model_boundary(
        answer,
        prompt=N2N_CORRUPTION_PROMPT,
        citation_plan=_plan(("noise2noise", N2N_CORRUPTION, 1)),
    ) == answer


def test_n2n_corruption_completion_rejects_reverse_incomplete_and_wrong_source_evidence() -> None:
    reverse = N2N_CORRUPTION.replace("they need", "our approach needs")
    incomplete = "They need a perfect, explicit computational model of the corruption; our approach is less limiting."
    for source, evidence in (
        ("noise2noise", reverse),
        ("noise2noise", incomplete),
        ("noise2self", N2N_CORRUPTION),
    ):
        assert complete_n2n_corruption_model_boundary(
            N2N_CORRUPTION_ANSWER,
            prompt=N2N_CORRUPTION_PROMPT,
            citation_plan=_plan((source, evidence, 1)),
        ) == N2N_CORRUPTION_ANSWER


def test_n2n_corruption_completion_does_not_join_relation_fragments_across_sources() -> None:
    assert complete_n2n_corruption_model_boundary(
        N2N_CORRUPTION_ANSWER,
        prompt=N2N_CORRUPTION_PROMPT,
        citation_plan=_plan(
            ("noise2noise", "They need a perfect, explicit computational model of the corruption.", 1),
            ("noise2self", "Our requirement is the appropriate statistical summary (mean, median).", 2),
        ),
    ) == N2N_CORRUPTION_ANSWER


@pytest.mark.parametrize(
    "path",
    ["C:/papers/noise2self/noise2self.en.md", "D:/other/noise2noise.en.md"],
)
def test_completion_rejects_stale_candidate_number_and_namesake_source(path: str) -> None:
    hits = [{"meta": {"source_path": path}}]
    assert complete_n2n_corruption_model_boundary(
        N2N_CORRUPTION_ANSWER,
        prompt=N2N_CORRUPTION_PROMPT,
        citation_plan=_plan(("noise2noise", N2N_CORRUPTION, 1)),
        answer_hits=hits,
    ) == N2N_CORRUPTION_ANSWER
    assert complete_denoising_quality_boundary(
        "Keep me.",
        prompt="Why should it outperform clean targets? Explain the performance boundary.",
        citation_plan=_plan(("noise2void", N2V, 1)),
        answer_hits=hits,
    ) == "Keep me."


def test_completion_remaps_stale_number_to_the_exact_source_path() -> None:
    hits = [
        {"meta": {"source_path": "D:/other/noise2noise.en.md"}},
        {"meta": {"source_path": r"c:\papers\noise2noise\noise2noise.en.md"}},
    ]
    result = complete_n2n_corruption_model_boundary(
        N2N_CORRUPTION_ANSWER,
        prompt=N2N_CORRUPTION_PROMPT,
        citation_plan=_plan(("noise2noise", N2N_CORRUPTION, 1)),
        answer_hits=hits,
    )
    assert result.endswith("corruption model [2].")


def test_completion_rejects_an_explicitly_empty_answer_source_list() -> None:
    assert complete_n2n_corruption_model_boundary(
        N2N_CORRUPTION_ANSWER,
        prompt=N2N_CORRUPTION_PROMPT,
        citation_plan=_plan(("noise2noise", N2N_CORRUPTION, 1)),
        answer_hits=[],
    ) == N2N_CORRUPTION_ANSWER


def test_parent_directory_name_cannot_relabel_a_different_paper() -> None:
    plan = _plan(("noise2self", N2N_CORRUPTION, 1))
    path = "C:/papers/noise2noise/noise2self.en.md"
    plan["slots"][0]["source_path"] = path
    assert complete_n2n_corruption_model_boundary(
        N2N_CORRUPTION_ANSWER,
        prompt=N2N_CORRUPTION_PROMPT,
        citation_plan=plan,
        answer_hits=[{"meta": {"source_path": path}}],
    ) == N2N_CORRUPTION_ANSWER
