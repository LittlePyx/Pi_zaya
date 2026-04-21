from api.reference_ui import _prompt_likely_cross_paper_refs
from kb.generation_answer_finalize_runtime import (
    _extract_multi_paper_topic,
    _prompt_targets_sci_topic,
)
from kb.reference_query_family import (
    extract_multi_paper_topic,
    prompt_explicitly_requests_multi_paper_list,
    prompt_explicitly_requests_single_paper_pick,
    prompt_prefers_zh,
    prompt_reference_focus_action,
    prompt_requests_reference_compare,
    prompt_requests_reference_definition,
    prompt_requests_reference_discussion,
    prompt_requires_reference_focus_match,
    prompt_targets_sci_topic,
)


def test_prompt_explicitly_requests_multi_paper_list_detects_chinese_query():
    prompt = "\u6709\u54ea\u51e0\u7bc7\u6587\u7ae0\u63d0\u5230\u4e86SCI\uff08\u5355\u6b21\u66dd\u5149\u538b\u7f29\u6210\u50cf\uff09"

    assert prompt_explicitly_requests_multi_paper_list(prompt) is True
    assert prompt_explicitly_requests_single_paper_pick(prompt) is False
    assert prompt_prefers_zh(prompt) is True
    assert _prompt_likely_cross_paper_refs(prompt) is True


def test_prompt_explicitly_requests_single_paper_pick_detects_library_rank_query():
    prompt = "Which paper in my library most directly discusses ADMM? Please point me to the source section."

    assert prompt_explicitly_requests_single_paper_pick(prompt) is True
    assert prompt_explicitly_requests_multi_paper_list(prompt) is False


def test_extract_multi_paper_topic_reads_chinese_focus_term():
    prompt = "\u6709\u54ea\u51e0\u7bc7\u6587\u7ae0\u63d0\u5230\u4e86SCI\uff08\u5355\u6b21\u66dd\u5149\u538b\u7f29\u6210\u50cf\uff09"

    topic = extract_multi_paper_topic(prompt)

    assert "SCI" in topic
    assert "\u5355\u6b21\u66dd\u5149\u538b\u7f29\u6210\u50cf" in topic
    assert _extract_multi_paper_topic(prompt) == topic


def test_prompt_targets_sci_topic_detects_chinese_alias():
    prompt = "\u6709\u54ea\u51e0\u7bc7\u6587\u7ae0\u63d0\u5230\u4e86SCI\uff08\u5355\u6b21\u66dd\u5149\u538b\u7f29\u6210\u50cf\uff09"

    assert prompt_targets_sci_topic(prompt) is True
    assert _prompt_targets_sci_topic(prompt) is True


def test_prompt_reference_focus_helpers_classify_compare_define_and_discuss_generically():
    compare_prompt = "Which paper in my library directly compares method A and method B?"
    define_prompt = "Which paper in my library defines adaptive sampling?"
    discuss_prompt = "Which papers in my library discuss photon-efficient reconstruction?"

    assert prompt_requests_reference_compare(compare_prompt) is True
    assert prompt_reference_focus_action(compare_prompt) == "compare"

    assert prompt_requests_reference_definition(define_prompt) is True
    assert prompt_reference_focus_action(define_prompt) == "define"

    assert prompt_requests_reference_discussion(discuss_prompt) is True
    assert prompt_reference_focus_action(discuss_prompt) == "discuss"


def test_prompt_requires_reference_focus_match_stays_false_for_non_locate_prompt():
    assert prompt_requires_reference_focus_match("Summarize the main contribution of this paper.") is False
    assert prompt_reference_focus_action("Summarize the main contribution of this paper.") == ""


def test_prompt_requires_reference_focus_match_supports_chinese_locate_phrasing():
    prompt = "\u54ea\u51e0\u7bc7\u8bba\u6587\u5b9a\u4e49\u4e86\u81ea\u9002\u5e94\u91c7\u6837\uff1f"

    assert prompt_requests_reference_definition(prompt) is True
    assert prompt_requires_reference_focus_match(prompt) is True
    assert prompt_reference_focus_action(prompt) == "define"
