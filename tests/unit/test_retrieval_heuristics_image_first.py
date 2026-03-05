from __future__ import annotations

from kb.retrieval_heuristics import _should_prioritize_attached_image


def test_image_first_for_generic_attached_image_prompt():
    assert _should_prioritize_attached_image("这张图写了啥")
    assert _should_prioritize_attached_image("帮我看下这张图是什么意思")
    assert _should_prioritize_attached_image("describe this image")


def test_not_image_first_when_user_explicitly_links_to_paper():
    assert not _should_prioritize_attached_image("这张图出自哪篇论文")
    assert not _should_prioritize_attached_image("which paper is this figure from")
