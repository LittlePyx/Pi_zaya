from __future__ import annotations

from kb.direct_question_evidence import direct_question_evidence_hits


def test_explicit_page_singleton_assignment_requires_its_actual_source_block(tmp_path):
    source = tmp_path / "sets.md"
    source.write_text(
        "<!-- kb_page: 2 -->\nFor independent, mean-zero noise in each pixel we choose "
        r"$\mathcal{J}$ = {{1}, . . . , {m}} to be the singletons."
        "\n<!-- kb_page: 8 -->\nWe choose a different set = {a,b} of singletons.", encoding="utf8")
    hits = direct_question_evidence_hits(source, "第2页选择单元素分组的等式是什么？")
    assert len(hits) == 1 and "mean-zero" in hits[0]["text"]
    assert hits[0]["meta"]["page_start"] == 2
    assert not direct_question_evidence_hits(source, "第3页选择单元素分组的等式是什么？")


def test_application_enumeration_does_not_require_literal_number(tmp_path):
    source = tmp_path / "applications.md"
    source.write_text("<!-- kb_page: 1 -->\nWe show applications in photographic noise removal, denoising of synthetic images, and reconstruction of undersampled measurements.\n\n## References\n\nWe show applications in unrelated imaging.", encoding="utf-8")
    hits = direct_question_evidence_hits(source, "本文展示哪三种数据恢复应用？")
    assert len(hits) == 1 and "undersampled" in hits[0]["text"]


def test_independence_uses_authors_assumption_not_previous_method(tmp_path):
    source = tmp_path / "independence.md"
    source.write_text("<!-- kb_page: 2 -->\nEarlier work acquires noise with independent realizations of the signal.\n\nWe assume the noise to be pixel-wise independent given the signal. The signal contains spatial correlations.", encoding="utf-8")
    hits = direct_question_evidence_hits(source, "分别指出论文中的独立性要求")
    assert len(hits) == 1 and hits[0]["text"].startswith("We assume")


def test_training_source_requirement_excludes_related_work(tmp_path):
    source = tmp_path / "training.md"
    source.write_text("<!-- kb_page: 2 -->\nPrevious work uses input and target pairs.\n\nWe can dispose of clean targets as long as we observe each source image twice.", encoding="utf-8")
    hits = direct_question_evidence_hits(source, "它还需要怎样的训练观测？")
    assert len(hits) == 1 and "twice" in hits[0]["text"]


def test_calibration_does_not_invent_missing_algorithms(tmp_path):
    source = tmp_path / "calibration.md"
    source.write_text("<!-- kb_page: 3 -->\nWe train a neural network on noisy observations.", encoding="utf-8")
    assert not direct_question_evidence_hits(source, "可以校准哪些传统去噪方法？")


def test_noise_interval_prefers_actual_values_to_prior_work(tmp_path):
    source = tmp_path / "unseen-method.md"
    source.write_text(
        "<!-- kb_page: 3 -->\n## Training\n\nEarlier denoisers estimate the noise level.\n\nWe train with noise levels in [7,47]. Testing within this range needs no estimate.\n",
        encoding="utf-8",
    )
    hits = direct_question_evidence_hits(
        source, "What training noise-level interval is used?"
    )
    assert "[7,47]" in hits[0]["text"]
    assert hits[0]["meta"]["page_start"] == 3
    assert hits[0]["meta"]["block_id"]


def test_statement_preserves_page_local_equation_and_explanation(tmp_path):
    source = tmp_path / "theory.md"
    source.write_text(
        "<!-- kb_page: 2 -->\n## Theory\n\nProposition 4. Assume independent noise.\n\n$$R(f) = Q(f) + V$$\n\nThe loss differs by the noise variance, independent of f.\n\n## Next\n\nUnrelated theory.\n<!-- kb_page: 9 -->\n## Appendix\n\nProposition 4. Another claim.\n",
        encoding="utf-8",
    )
    hits = direct_question_evidence_hits(source, "命题 4 中，损失相差什么？")
    assert "variance" in hits[0]["text"]
    assert any("R(f)" in hit["text"] for hit in hits)
    assert all(hit["meta"]["page_start"] == 2 for hit in hits)
    assert all("Unrelated" not in hit["text"] for hit in hits)


def test_missing_statement_does_not_substitute_a_different_number(tmp_path):
    source = tmp_path / "theory.md"
    source.write_text(
        "<!-- kb_page: 1 -->\nProposition 3. A valid but different claim.\n",
        encoding="utf-8",
    )
    assert direct_question_evidence_hits(source, "Explain Proposition 8.") == []


def test_statement_bundle_stops_before_another_statement(tmp_path):
    source = tmp_path / "theory.md"
    source.write_text(
        "<!-- kb_page: 2 -->\nProposition 4. Independent noise gives a constant loss gap.\n\nThe difference is the variance of the noise.\n\nProposition 5. A different estimator can have bias.\n\nIts error can increase.\n",
        encoding="utf-8",
    )
    hits = direct_question_evidence_hits(source, "Explain Proposition 4.")
    assert len(hits) == 2
    assert all("bias" not in h["text"] and "increase" not in h["text"] for h in hits)


def test_direct_equation_audit_requires_same_page_actual_blocks(tmp_path):
    from kb.direct_question_evidence import attach_direct_source_evidence
    from kb.paper_guide_answer_selection import _merge_same_source_answer_hits
    from kb.generation_answer_finalize_runtime import (
        _claim_evidence_hits_with_citation_plan,
    )

    source = tmp_path / "statement.md"
    source.write_text(
        "<!-- kb_page: 2 -->\n## Theory\n\nProposition 4. We assume independent noise in the measured data.\n\n$$R(f)=Q(f)+V$$\n\nThe loss differs by noise variance, independent of the function f.\n",
        encoding="utf-8",
    )
    hits = direct_question_evidence_hits(source, "Explain the loss in Proposition 4.")
    merged = _merge_same_source_answer_hits(hits)
    lead = hits[0]
    slot = {
        **lead["meta"],
        "evidence_quote": lead["text"],
        "preferred_system": "system_a",
        "candidate_hits": [1],
    }
    plan = {"slots": [slot]}
    attached = attach_direct_source_evidence(plan, merged)
    assert any("R(f)" in q for q in attached["slots"][0]["source_evidence_quotes"])
    assert attached["slots"][0]["block_id"] == slot["block_id"]
    assert attached["slots"][0]["evidence_quote"] == slot["evidence_quote"]
    audited = _claim_evidence_hits_with_citation_plan(merged, attached)
    assert "R(f)" in audited[0]["meta"]["citation_plan_full_evidence_quote"]
    assert "source_evidence_quotes" not in plan["slots"][0]
    broad_plan = {
        "slots": [
            {
                **slot,
                "block_id": "broad-preflight-block",
                "evidence_quote": "Old broad retrieval summary.",
                "page_start": 9,
            }
        ]
    }
    promoted = attach_direct_source_evidence(broad_plan, merged)["slots"][0]
    assert promoted["block_id"] == slot["block_id"]
    assert promoted["page_start"] == 2 and promoted["candidate_hits"] == [1]
    assert promoted["evidence_quote"] == slot["evidence_quote"]
    wrong_page = {"slots": [{**slot, "page_start": 7}]}
    assert (
        "source_evidence_quotes"
        not in attach_direct_source_evidence(wrong_page, merged)["slots"][0]
    )
    wrong_source = {"slots": [{**slot, "source_path": str(tmp_path / "other.md")}]}
    assert (
        "source_evidence_quotes"
        not in attach_direct_source_evidence(wrong_source, merged)["slots"][0]
    )
    source.write_text("<!-- kb_page: 2 -->\nChanged source.\n", encoding="utf-8")
    assert (
        "source_evidence_quotes"
        not in attach_direct_source_evidence(plan, merged)["slots"][0]
    )


def test_three_tasks_uses_authors_actual_statement(tmp_path):
    source = tmp_path / "method.md"
    source.write_text(
        "<!-- kb_page: 3 -->\nEarlier work addressed three image restoration tasks.\n\nWe train a single network for three tasks: denoising, scaling and artifact removal.\n",
        encoding="utf-8",
    )
    hits = direct_question_evidence_hits(
        source, "Which three tasks does the network handle?"
    )
    assert len(hits) == 1 and hits[0]["text"].startswith("We train")


def test_three_tasks_prefers_complete_named_enumeration(tmp_path):
    source = tmp_path / "method.md"
    source.write_text(
        "<!-- kb_page: 1 -->\nWe investigate several general image denoising tasks such as Gaussian denoising and single image...\n\n"
        "<!-- kb_page: 4 -->\nWe consider three specific tasks: blind Gaussian denoising, SISR, and JPEG deblocking.\n",
        encoding="utf-8",
    )
    hits = direct_question_evidence_hits(source, "单个模型处理哪三类任务？")
    assert len(hits) == 1
    assert "JPEG deblocking" in hits[0]["text"]
    assert hits[0]["meta"]["page_start"] == 4


def test_architecture_relation_prefers_block_with_complete_formula(tmp_path):
    source = tmp_path / "architecture.md"
    source.write_text(
        "<!-- kb_page: 3 -->\nWe use 3 × 3 filters and remove all pooling layers. The receptive field grows with depth d.\n\n"
        "We use 3 × 3 filters and remove all pooling layers. The receptive field at depth of d should be (2d+1)×(2d+1).\n",
        encoding="utf-8",
    )
    hits = direct_question_evidence_hits(
        source, "网络保留池化层吗？卷积核大小和深度 d 时的感受野是什么？"
    )
    assert len(hits) == 1
    assert "(2d+1)×(2d+1)" in hits[0]["text"]


def test_architecture_relation_does_not_promote_incomplete_formula(tmp_path):
    source = tmp_path / "architecture.md"
    source.write_text(
        "<!-- kb_page: 3 -->\nWe use 3 × 3 filters and remove all pooling layers. The receptive field grows with depth d.\n",
        encoding="utf-8",
    )
    assert not direct_question_evidence_hits(
        source, "网络保留池化层吗？卷积核大小和深度 d 时的感受野是什么？"
    )


def test_output_relation_requires_residual_direction(tmp_path):
    source = tmp_path / "dncnn.md"
    source.write_text(
        "<!-- kb_page: 2 -->\n\n"
        "DnCNN predicts an image output.\n\n"
        "Rather than directly outputting the denoised image $\\hat{x}$, the proposed DnCNN is designed "
        "to predict the residual image $\\hat{v}$, i.e., the difference between the noisy observation "
        "and the latent clean image.\n",
        encoding="utf-8",
    )
    hits = direct_question_evidence_hits(
        source, "DnCNN 直接预测干净图像吗？网络输出和去噪图像是什么关系？"
    )
    assert len(hits) == 1
    assert "difference between the noisy observation and the latent clean image" in hits[0]["text"]

    source.write_text(
        "<!-- kb_page: 2 -->\n\nDnCNN predicts the clean image; a residual is the difference "
        "between the latent clean image and noisy observation.\n",
        encoding="utf-8",
    )
    assert not direct_question_evidence_hits(
        source, "DnCNN 直接预测干净图像吗？网络输出和去噪图像是什么关系？"
    )


def test_noisy_target_relation_prefers_conditional_expectation(tmp_path):
    source = tmp_path / "targets.md"
    source.write_text(
        "<!-- kb_page: 2 -->\nNoisy targets are 2000 times faster to produce.\n\n"
        "Input-conditioned target distributions may be replaced by arbitrary distributions that have the same conditional expected values. "
        "Thus targets may be corrupted with zero-mean noise without changing what the network learns.\n",
        encoding="utf-8",
    )
    hits = direct_question_evidence_hits(
        source, "为什么能用带噪观测作目标？对条件期望有什么要求？"
    )
    assert len(hits) == 1
    assert "same conditional expected values" in hits[0]["text"]
    assert "2000" not in hits[0]["text"]


def test_noisy_target_relation_rejects_wrong_expectation_or_bias(tmp_path):
    source = tmp_path / "targets.md"
    source.write_text(
        "<!-- kb_page: 2 -->\nInput-conditioned target distributions have different conditional expected values and non-zero-mean noise.\n",
        encoding="utf-8",
    )
    assert not direct_question_evidence_hits(
        source, "为什么能用带噪观测作目标？对条件期望有什么要求？"
    )


def test_single_noisy_image_question_selects_both_waived_pair_types(tmp_path):
    source = tmp_path / "training-data.md"
    source.write_text(
        "<!-- kb_page: 1 -->\nEarlier work trains from noisy image pairs.\n\n"
        "We introduce N2V. It does not require noisy image pairs, nor clean target images. "
        "It trains directly on the noisy images to be denoised.\n",
        encoding="utf-8",
    )
    hits = direct_question_evidence_hits(
        source, "只有一张带噪图且没有干净参照时还能训练吗？免除哪两类配对数据？"
    )
    assert len(hits) == 1
    assert "noisy image pairs" in hits[0]["text"]
    assert "clean target images" in hits[0]["text"]


def test_single_noisy_image_question_rejects_only_one_waived_pair_type(tmp_path):
    source = tmp_path / "training-data.md"
    source.write_text(
        "<!-- kb_page: 1 -->\nOur method does not require noisy image pairs.\n",
        encoding="utf-8",
    )
    assert not direct_question_evidence_hits(
        source, "只有一张带噪图且没有干净参照时还能训练吗？免除哪两类配对数据？"
    )


def test_statistical_assumptions_selects_complete_signal_noise_relation(tmp_path):
    source = tmp_path / "assumptions.md"
    source.write_text(
        "<!-- kb_page: 2 -->\nPrevious methods use independent noise.\n\n"
        "We make two simple statistical assumptions: the signal is not pixel-wise independent, "
        "and the noise is conditionally pixel-wise independent given the signal.\n",
        encoding="utf-8",
    )
    hits = direct_question_evidence_hits(
        source, "论文对真实信号和噪声分别作了哪两条统计假设？"
    )
    assert len(hits) == 1
    assert "signal is not pixel-wise independent" in hits[0]["text"]
    assert "given the signal" in hits[0]["text"]


def test_direct_block_survives_summary_selection(tmp_path):
    from kb.direct_question_evidence import prioritize_direct_question_hits

    source = tmp_path / "method.md"
    source.write_text(
        "<!-- kb_page: 3 -->\nWe use a noise interval [7,47].\n", encoding="utf-8"
    )
    direct = direct_question_evidence_hits(source, "What noise interval is used?")
    selected = [
        {"text": "Broad training summary", "meta": {"source_path": str(source)}}
    ]
    assert prioritize_direct_question_hits(selected, direct, limit=1) == direct


def test_verified_direct_plan_survives_stale_reference_enrichment(
    tmp_path, monkeypatch
):
    import ui.refs_renderer as renderer
    from api.chat_render import enrich_messages_with_reference_render
    from kb.direct_question_evidence import attach_direct_source_evidence
    from kb.source_blocks import load_source_blocks

    monkeypatch.setattr(renderer, "_is_temp_source_path", lambda _path: False)
    source = tmp_path / "paper.md"
    direct = "We train one network for three tasks: denoising, super-resolution, and JPEG deblocking."
    source.write_text(
        f"<!-- kb_page: 2 -->\n## Applications\n\n{direct}\n\n<!-- kb_page: 9 -->\n## Architecture\n\nThe network uses three types of layers in its architecture.\n",
        encoding="utf-8",
    )
    blocks = [b for b in load_source_blocks(source) if b.get("kind") == "paragraph"]
    stale = blocks[-1]
    path = str(source)
    plan = {
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                **stale,
                "preferred_system": "system_a",
                "source_path": path,
                "evidence_quote": stale["raw_text"],
                "candidate_hits": [1],
            }
        ],
    }
    plan = attach_direct_source_evidence(
        plan, direct_question_evidence_hits(source, "What three tasks are handled?")
    )
    answer = {
        "id": 2,
        "role": "assistant",
        "content": direct[:-1] + " [1].",
        "meta": {
            "canonical_hit_paths": [path],
            "answer_quality": {
                "has_hits": True,
                "minimum_ok": True,
                "citation_plan": plan,
            },
        },
    }
    pack = {
        "hits": [
            {
                "text": stale["raw_text"],
                "meta": {**stale, "source_path": path},
                "ui_meta": {
                    "primary_evidence": {
                        **stale,
                        "source_path": path,
                        "snippet": stale["raw_text"],
                    }
                },
            }
        ]
    }
    result = enrich_messages_with_reference_render(
        [{"id": 1, "role": "user", "content": "What three tasks are handled?"}, answer],
        {1: pack},
        conv_id="direct-plan-enrichment",
    )[-1]
    assert result["cite_details"]
    assert result["cite_details"][0]["block_id"] == blocks[0]["block_id"]
    assert result["cite_details"][0]["page_start"] == 2
    assert "JPEG deblocking" in result["cite_details"][0]["evidence_quote"]


def test_multi_facet_refresh_retains_verified_direct_binding(tmp_path):
    from kb.direct_question_evidence import attach_direct_source_evidence
    from kb.generation_answer_finalize_runtime import (
        _citation_plan_with_late_evidence_cards,
    )

    source = tmp_path / "paper.md"
    text = "We train a single model for three tasks: denoising, scaling, and compression artifact removal."
    source.write_text(
        f"<!-- kb_page: 2 -->\n## Tasks\n\n{text}\n\nThe same network shares its parameters across all image restoration tasks.\n",
        encoding="utf-8",
    )
    hits = direct_question_evidence_hits(source, "What three tasks are handled?")
    primary = {**hits[0]["meta"], "snippet": text, "evidence_quote": text}
    plan = attach_direct_source_evidence(
        {
            "budget": {"system_a": 2, "system_b": 0},
            "slots": [
                {**primary, "preferred_system": "system_a", "candidate_hits": [1]}
            ],
        },
        hits,
    )
    from kb.source_blocks import load_source_blocks

    other = [b for b in load_source_blocks(source) if b.get("kind") == "paragraph"][-1]
    other_primary = {**other, "source_path": str(source), "snippet": other["raw_text"]}
    refreshed = _citation_plan_with_late_evidence_cards(
        plan,
        answer_hits=hits,
        evidence_cards=[
            {"source_path": str(source), "primary_evidence": primary},
            {"source_path": str(source), "primary_evidence": other_primary},
        ],
        prompt="分别说明三类任务",
    )
    assert refreshed["late_evidence_refresh"] is True
    assert refreshed["slots"][0]["evidence_selection_reason"] == "prompt_contract_block"
    assert refreshed["slots"][0]["strict_locate"] is True
    assert (
        refreshed["slots"][0]["direct_source_evidence"][0]["block_id"]
        == primary["block_id"]
    )
