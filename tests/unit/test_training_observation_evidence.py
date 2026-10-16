from __future__ import annotations

import pytest

from kb.direct_question_evidence import direct_question_evidence_hits


PAIR_QUESTION = "比较 Noise2Noise 与 Noise2Void：没有干净目标时，它们分别还需要怎样的训练观测？为什么单张噪声图像场景更适合后者？"
PAIR_PREMISE = (
    "In many image restoration tasks, the expectation of the corrupted input data is the clean target "
    "that we seek to restore. Low-light photography is an example: a long, noise-free exposure is the "
    "average of short, independent, noisy exposures. The above findings indicate, that in these cases, "
    "we can dispose of the clean targets entirely, as long as we are able to observe each source image twice."
)
TARGET_CONDITION = (
    "A trivial property of L 2 minimization is that the estimate remains unchanged if we replace the targets "
    "with random numbers whose expectations match the targets. The optimal network parameters remain unchanged, "
    "if input-conditioned target distributions p ( y | x ) are replaced with arbitrary distributions that "
    "have the same conditional expected values. This implies that we can, in principle, corrupt the training "
    "targets of a neural network with zero-mean noise without changing what the network learns."
)
CAPTION = (
    "Comparison of various loss functions for training a Monte Carlo denoiser with noisy target images rendered "
    "at 8 samples per pixel. In this high-dynamic range setting, our custom relative loss is superior."
)
QUALITY_BOUNDARY = (
    "While it cannot be expected that our approach outperforms methods that have additional information "
    "available during training, we observe that the denoising performance of our results only drops "
    "moderately and is still outperforming BM3D."
)
QUALITY_QUESTION = "本文是否声称这种训练方式应当胜过拥有干净目标或独立噪声对的训练？作者怎样解释这个性能边界？"


def _source(tmp_path):
    path = tmp_path / "noise2noise.en.md"
    path.write_text(
        "<!-- kb_page: 2 -->\n" + PAIR_PREMISE + "\n\n" + TARGET_CONDITION
        + "\n<!-- kb_page: 6 -->\n" + CAPTION,
        encoding="utf8",
    )
    return path


def test_selected_source_scan_keeps_question_separate_from_expansion(tmp_path):
    from kb.task_runtime import _selected_research_context_evidence_hits
    from kb.direct_question_evidence import selected_source_evidence_prompt

    path = _source(tmp_path)
    expanded = [PAIR_QUESTION, "Noise2Noise Monte Carlo noisy target training loss data observations"]
    hits = _selected_research_context_evidence_hits(
        [{"kind": "reference", "title": "Noise2Noise", "sourcePath": str(path)}],
        prompt=selected_source_evidence_prompt(PAIR_QUESTION, expanded), max_hits=2, db_dir=tmp_path,
    )
    assert hits and all(h["meta"]["page_start"] == 2 for h in hits)
    assert "observe each source image twice" in hits[0]["text"]
    assert all(CAPTION not in h["text"] for h in hits)


def test_nonfacet_question_retains_translated_ranking_terms():
    from kb.direct_question_evidence import selected_source_evidence_prompt

    assert "multiplexing" in selected_source_evidence_prompt("解释本文的方法", ["解释本文的方法", "frequency multiplexing"])


@pytest.mark.parametrize("question", [
    PAIR_QUESTION,
    "Compare Noise2Noise and Noise2Void: what training observations are needed without clean targets, and why is the latter suited to single noisy images?",
])
def test_pair_requirement_uses_author_owned_observation_not_application_caption(tmp_path, question):
    hits = direct_question_evidence_hits(_source(tmp_path), question)
    assert hits and "observe each source image twice" in hits[0]["text"]
    assert hits[0]["meta"]["page_start"] == 2


def test_conditional_target_question_keeps_clean_target_premise_and_condition(tmp_path):
    hits = direct_question_evidence_hits(
        _source(tmp_path), "没有干净标签时，为什么可以把另一次带噪观测当训练目标？本文对条件期望提出了什么要求？",
    )
    assert len(hits) == 2
    assert any("same conditional expected values" in h["text"] for h in hits)
    assert any("expectation of the corrupted input data is the clean target" in h["text"] for h in hits)
    assert {h["meta"]["page_start"] for h in hits} == {2}


def test_pre_generation_plan_contains_both_verified_target_facets(tmp_path):
    from kb.direct_question_evidence import attach_direct_source_evidence
    from kb.citation_plan import build_citation_plan, build_citation_plan_prompt_block
    from kb.paper_guide_answer_selection import _merge_same_source_answer_hits

    path = _source(tmp_path)
    prompt = "没有干净标签时，为什么可以把另一次带噪观测当训练目标？本文对条件期望提出了什么要求？"
    hits = _merge_same_source_answer_hits(direct_question_evidence_hits(path, prompt))
    plan = attach_direct_source_evidence(build_citation_plan(prompt=prompt, answer_hits=hits), hits)
    block = build_citation_plan_prompt_block(plan)
    assert "same conditional expected values" in block
    assert "expectation of the corrupted input data is the clean target" in block
    assert all(s["page_start"] == 2 for s in plan["slots"])
    condition = next(s for s in plan["slots"] if "same conditional expected values" in s["evidence_quote"])
    premise = next(s for s in plan["slots"] if "observe each source image twice" in s["evidence_quote"])
    assert PAIR_PREMISE not in condition.get("source_evidence_quotes", [])
    assert TARGET_CONDITION not in premise.get("source_evidence_quotes", [])


@pytest.mark.parametrize("text", [
    CAPTION,
    "Previous work uses input and target pairs from two independent observations.",
    "We evaluate the PSNR of training on clean targets. The noisy data gives a different score.",
])
def test_single_observation_comparison_does_not_promote_related_or_application_text(tmp_path, text):
    path = tmp_path / "other.en.md"
    path.write_text("<!-- kb_page: 4 -->\n" + text, encoding="utf8")
    question = "What training observations are needed without clean targets for single noisy images?"
    assert not direct_question_evidence_hits(path, question)


def test_long_training_paragraph_keeps_both_requirements_in_its_actual_card(tmp_path):
    from kb.direct_question_evidence import attach_direct_source_evidence
    from kb.paper_guide_answer_selection import _merge_same_source_answer_hits

    prefix = (
        "This requires that two images capturing the same content with independent noises can be acquired. "
        "N2N training requires pairs of noisy images, which can only be acquired for quasi static scenes. "
        "Here we present N2V, our novel training scheme. Unlike N2N, N2V can also be applied to data for which "
        "neither noisy image pairs nor clean target images are available."
    )
    path = tmp_path / "noise2void.en.md"
    path.write_text("<!-- kb_page: 2 -->\n" + prefix + " We evaluate experiments. " * 90, encoding="utf8")
    direct = direct_question_evidence_hits(path, PAIR_QUESTION)
    hits = _merge_same_source_answer_hits(direct + [{"text": "Other evidence.", "meta": {"source_path": str(path)}}])
    plan = attach_direct_source_evidence({"slots": [{
        "preferred_system": "system_a", "source_path": str(path), "candidate_hits": [1],
        "evidence_quote": "A broad source preview.", "block_id": "old", "page_start": 1,
    }]}, hits)
    slot = plan["slots"][0]
    assert slot["block_id"] == direct[0]["meta"]["block_id"]
    assert slot["page_start"] == 2
    assert prefix in slot["evidence_quote"]
    assert "We evaluate" not in slot["evidence_quote"]
    assert slot["strict_locate"]


@pytest.mark.parametrize("key,value", [("anchor_id", "wrong"), ("page_start", 7), ("heading_path", "wrong"), ("source_path", "another/paper.md")])
def test_contained_excerpt_requires_its_own_complete_locator(tmp_path, key, value):
    from kb.direct_question_evidence import attach_direct_source_evidence
    from kb.source_blocks import load_source_blocks

    path = _source(tmp_path)
    block = load_source_blocks(path)[0]
    row = {**block, "text": PAIR_PREMISE, "source_path": str(path), "direct_question_evidence": True}
    row[key] = value
    hit = {"text": PAIR_PREMISE, "meta": {"source_path": str(path), "source_passages": [row]}}
    slot = {"preferred_system": "system_a", "source_path": str(path), "block_id": "old", "page_start": 1, "evidence_quote": "Old preview."}
    result = attach_direct_source_evidence({"slots": [slot]}, [hit])
    assert result["slots"][0]["block_id"] == "old"
    assert not result["slots"][0].get("direct_source_evidence")


def test_audit_overlay_does_not_give_two_passages_one_primary_locator(tmp_path):
    from kb.generation_answer_finalize_runtime import _claim_evidence_hits_with_citation_plan
    from kb.source_blocks import load_source_blocks

    path = _source(tmp_path)
    blocks = [b for b in load_source_blocks(path) if b["kind"] == "paragraph"][:2]
    slots = [{**b, "preferred_system": "system_a", "source_path": str(path), "candidate_hits": [1], "evidence_quote": b["raw_text"]} for b in blocks]
    hit = {"text": "Earlier broad hit.", "meta": {"source_path": str(path)}}
    result = _claim_evidence_hits_with_citation_plan([hit], {"slots": slots})[0]
    primary = result["meta"]["primary_evidence"]
    own = next(b for b in blocks if b["block_id"] == primary["block_id"])
    assert primary["snippet"] == own["raw_text"]
    assert result["meta"]["evidence_quote"] == own["raw_text"]
    assert all(b["raw_text"] in result["text"] for b in blocks)
    assert all(b["raw_text"] in result["meta"]["citation_plan_full_evidence_quote"] for b in blocks)


def test_cited_observation_claim_prefers_direct_excerpt_over_generic_theorem_context(tmp_path):
    from api.claim_passage_plan import prepare_claim_passage_plan, message_citation_plan
    from kb.source_blocks import load_source_blocks
    from kb.n2n_target_condition import target_condition_quote

    path = tmp_path / "noise2noise.en.md"
    path.write_text("# Noise2Noise: Learning Image Restoration without Clean Data\n<!-- kb_page: 2 -->\n"
                    + TARGET_CONDITION + "\n\n" + PAIR_PREMISE, encoding="utf8")
    blocks = [b for b in load_source_blocks(path) if b["kind"] == "paragraph"]
    slots = [{**b, "preferred_system": "system_a", "source_path": str(path), "source_name": "noise2noise", "candidate_hits": [1],
              "evidence_quote": target_condition_quote(b["raw_text"]) or b["raw_text"]} for b in blocks]
    claim = "因此，在许多图像复原任务中，带噪输入的期望就是我们要复原的干净目标，例如低光摄影中，长曝光无噪图像是多次短时独立带噪曝光的平均，于是可以完全不用干净目标，只要每个源图像能被观测两次 [1]。"
    rec = {"meta": {"canonical_hit_paths": [str(path)], "answer_quality": {"citation_plan": {"slots": slots}}}}
    result = prepare_claim_passage_plan(rec, {"hits": [{"meta": {"source_path": str(path)}}]}, claim)
    bound = message_citation_plan(result)["slots"]
    assert any("每个源图像" in c for c in bound[1].get("passage_bound_claims", []))
    assert not bound[0].get("passage_bound_claims")


@pytest.mark.parametrize("already_same_block", [False, True])
def test_performance_question_keeps_negative_information_boundary_instead_of_training_preamble(tmp_path, already_same_block):
    from kb.direct_question_evidence import attach_direct_source_evidence, selected_source_evidence_prompt
    from kb.paper_guide_answer_selection import _merge_same_source_answer_hits
    from kb.denoising_quality_boundary import complete_denoising_quality_boundary

    path = tmp_path / "noise2void.en.md"
    body = ("Here we present N2V, our novel training scheme for data for which neither noisy image pairs nor clean target images are available. "
            + "We evaluate datasets and observations. " * 40 + QUALITY_BOUNDARY)
    path.write_text("<!-- kb_page: 2 -->\n" + body, encoding="utf8")
    prompt = selected_source_evidence_prompt(QUALITY_QUESTION, [QUALITY_QUESTION, "noisy image pairs training data"])
    hits = direct_question_evidence_hits(path, prompt)
    assert hits and hits[0]["text"] == QUALITY_BOUNDARY
    bundled = _merge_same_source_answer_hits(hits)
    original = {**hits[0]["meta"]} if already_same_block else {}
    plan = attach_direct_source_evidence({"slots": [{**original, "preferred_system": "system_a", "source_path": str(path), "candidate_hits": [1], "evidence_quote": body[:body.index("We evaluate")] if already_same_block else "Old broad preview."}]}, bundled)
    assert plan["slots"][0]["evidence_quote"] == QUALITY_BOUNDARY
    result = complete_denoising_quality_boundary("原回答遗漏了信息边界。", prompt=QUALITY_QUESTION, citation_plan=plan, answer_hits=bundled)
    assert "不能" in result and "更多训练信息" in result and "并不意味着" in result


@pytest.mark.parametrize("change", [
    lambda s: s.replace("cannot be expected", "can be expected"),
    lambda s: s.replace("additional information", "the same information"),
    lambda s: s.replace("only drops moderately", "drops substantially"),
    lambda s: s.replace("our approach", "the previous approach"),
])
def test_performance_question_does_not_invent_a_missing_or_reversed_boundary(tmp_path, change):
    path = tmp_path / "noise2void.en.md"
    path.write_text("<!-- kb_page: 2 -->\n" + change(QUALITY_BOUNDARY), encoding="utf8")
    assert not direct_question_evidence_hits(path, QUALITY_QUESTION)


@pytest.mark.parametrize("unreliable", [False, True])
def test_application_rescue_uses_original_question_and_respects_excluded_pages(tmp_path, unreliable):
    import json
    from kb.direct_question_evidence import prioritize_direct_question_hits
    from kb.converter.quality_repair import conversion_quality_result_path
    from kb.source_blocks import load_source_blocks

    path = tmp_path / "noise2noise.en.md"
    abstract = "We show applications in photographic noise removal, denoising of synthetic Monte Carlo images, and reconstruction of MRI scans from undersampled inputs."
    discussion = "There are several situations in low-light photography and imaging. We discuss why collecting clean data is difficult."
    path.write_text("<!-- kb_page: 1 -->\n" + abstract + "\n<!-- kb_page: 8 -->\n" + discussion, encoding="utf8")
    if unreliable:
        report = conversion_quality_result_path(path)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"source_quality": {"evidence_unreliable_pages": [1]}}), encoding="utf8")
    block = load_source_blocks(path)[1]
    old = {"text": discussion, "meta": {**block, "source_path": str(path), "direct_question_evidence": True}}
    result = prioritize_direct_question_hits([old], [old], limit=2, prompt="本文展示了哪三种不同类型的数据恢复应用？请指出不只涉及普通照片去噪的证据。")
    if unreliable:
        assert all(h["meta"]["page_start"] != 1 for h in result)
    else:
        assert result[0]["meta"]["page_start"] == 1
        assert "undersampled" in result[0]["text"]


@pytest.mark.parametrize('same_block', [False, True])
def test_explicit_application_list_promotes_its_own_locator_over_existing_discussion(tmp_path, same_block):
    from kb.direct_question_evidence import attach_direct_source_evidence
    from kb.source_blocks import load_source_blocks
    from kb.paper_guide_answer_selection import _merge_same_source_answer_hits

    path = tmp_path / 'paper.en.md'
    direct = 'We show applications in photographic noise removal, denoising of synthetic Monte Carlo images, and reconstruction of MRI scans from undersampled inputs.'
    broad = 'There are several situations in low-light photography, physically-based image synthesis, and magnetic resonance imaging. Our demonstrations remove the need to collect clean data.'
    intro = 'We apply basic statistical reasoning to reconstruction without clean exemplars. arXiv:1803.04189v1 [cs.LG] 12 Mar 2018 '
    path.write_text('<!-- kb_page: 1 -->\n' + intro + direct + '\n<!-- kb_page: 8 -->\n' + broad, encoding='utf8')
    prompt = '本文展示了哪三种不同类型的数据恢复应用？'
    hits = direct_question_evidence_hits(path, prompt)
    old = load_source_blocks(path)[0 if same_block else 1]
    hits.append({'text': broad, 'meta': {**old, 'source_path': str(path), 'direct_question_evidence': True}})
    merged = _merge_same_source_answer_hits(hits)
    plan = attach_direct_source_evidence({'slots': [{**old, 'preferred_system': 'system_a', 'source_path': str(path), 'candidate_hits': [1], 'evidence_quote': old['raw_text']}]}, merged)
    assert plan['slots'][0]['page_start'] == 1
    assert plan['slots'][0]['evidence_quote'] == direct
    assert plan['slots'][0]['block_id'] == hits[0]['meta']['block_id']


@pytest.mark.parametrize('source', [
    'There are several situations in low-light photography, physically-based image synthesis, and magnetic resonance imaging.',
    'We show photographic noise removal and reconstruction of MRI scans from undersampled inputs.',
    'We show photographic noise removal, denoising of synthetic Monte Carlo images, and reconstruction of MRI scans.',
])
def test_application_enumeration_rejects_broad_or_incomplete_source(source):
    from kb.claim_evidence_runtime import _support_score
    from kb.evidence_binding import assess_system_a_hit_binding

    claim = '本文展示了三种应用：照片去噪、合成蒙特卡洛图像去噪、从欠采样输入重建MRI。'
    assert _support_score(claim, source) == 0
    assert assess_system_a_hit_binding(answer_claim=claim, hit={'text': source}, meta={}, heading='', evidence_quote=source, source_name='')['suppress_link']
