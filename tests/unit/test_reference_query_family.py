from __future__ import annotations

from kb.reference_query_family import (
    prompt_explicitly_requests_multi_paper_list,
    prompt_likely_multi_paper_synthesis,
)


def test_reading_route_counts_as_multi_paper_list_and_synthesis() -> None:
    prompt = "我刚开始看单像素成像，想先建立主线，应该先读哪几篇？每篇主要看什么？"

    assert prompt_explicitly_requests_multi_paper_list(prompt) is True
    assert prompt_likely_multi_paper_synthesis(prompt) is True


def test_lineage_and_method_map_count_as_multi_paper_synthesis() -> None:
    assert prompt_likely_multi_paper_synthesis("SCI 这条线是怎么从光谱成像走到 3D 场景重建的？")
    assert prompt_likely_multi_paper_synthesis("这些 structured detection、interferometric 方法分别解决什么麻烦？")
    assert prompt_likely_multi_paper_synthesis("探测器综述和 physics-informed deep learning 这篇应该怎么搭配读？")
