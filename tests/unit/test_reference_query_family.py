from __future__ import annotations

from kb.reference_query_family import (
    extract_multi_paper_topic,
    extract_requested_paper_count,
    prompt_explicitly_requests_multi_paper_list,
    prompt_likely_multi_paper_synthesis,
    prompt_requests_answer_audit,
    prompt_requires_reference_focus_match,
)


def test_reading_route_counts_as_multi_paper_list_and_synthesis() -> None:
    prompt = "我刚开始看单像素成像，想先建立主线，应该先读哪几篇？每篇主要看什么？"

    assert prompt_explicitly_requests_multi_paper_list(prompt) is True
    assert prompt_likely_multi_paper_synthesis(prompt) is True


def test_lineage_and_method_map_count_as_multi_paper_synthesis() -> None:
    assert prompt_likely_multi_paper_synthesis("SCI 这条线是怎么从光谱成像走到 3D 场景重建的？")
    assert prompt_likely_multi_paper_synthesis("这些 structured detection、interferometric 方法分别解决什么麻烦？")
    assert prompt_likely_multi_paper_synthesis("探测器综述和 physics-informed deep learning 这篇应该怎么搭配读？")


def test_explicit_numeric_paper_request_extracts_count() -> None:
    prompt = "请只用库里最相关的 4 篇论文，按阅读顺序做一条路线。"

    assert extract_requested_paper_count(prompt) == 4
    assert prompt_explicitly_requests_multi_paper_list(prompt) is True


def test_natural_library_selection_phrase_extracts_requested_count() -> None:
    prompt = "请从库里选3篇最适合按顺序阅读的论文，每篇说明为什么。"

    assert extract_requested_paper_count(prompt) == 3
    assert prompt_explicitly_requests_multi_paper_list(prompt) is True


def test_natural_single_best_phrase_extracts_one_paper() -> None:
    prompt = "库里哪篇论文最直接？只给最直接的一篇，并说明依据。"

    assert extract_requested_paper_count(prompt) == 1
    assert prompt_explicitly_requests_multi_paper_list(prompt) is False


def test_flexible_english_count_phrase_extracts_one_paper() -> None:
    prompt = "Which paper is the direct match? Only give 1 paper."

    assert extract_requested_paper_count(prompt) == 1


def test_previous_answer_audit_is_not_a_multi_paper_list_request() -> None:
    prompt = (
        "审查上一条回答：是否严格只用了 4 篇？"
        "逐条核对论文标题与依据是否来自同一篇。不要重新生成阅读路线。"
    )

    assert prompt_requests_answer_audit(prompt) is True
    assert extract_requested_paper_count(prompt) is None
    assert prompt_explicitly_requests_multi_paper_list(prompt) is False


def test_exact_count_reading_route_does_not_require_missing_discussion_topic() -> None:
    prompt = "Please use only 4 papers for a reading route and cite each source."

    assert prompt_explicitly_requests_multi_paper_list(prompt) is True
    assert prompt_requires_reference_focus_match(prompt) is False
    assert prompt_requires_reference_focus_match("Which papers discuss SCI?") is True


def test_compare_and_locate_prompts_extract_explicit_focus_topics() -> None:
    compare_prompt = (
        "Which paper in my library directly compares Hadamard single-pixel imaging "
        "and Fourier single-pixel imaging?"
    )
    locate_prompt = "In the SCINeRF paper, where is ADMM discussed?"

    assert "Hadamard" in extract_multi_paper_topic(compare_prompt)
    assert prompt_requires_reference_focus_match(compare_prompt) is True
    assert extract_multi_paper_topic(locate_prompt) == "ADMM"
    assert prompt_requires_reference_focus_match(locate_prompt) is True
