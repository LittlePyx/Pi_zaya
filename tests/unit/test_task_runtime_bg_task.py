import api.reference_ui as reference_ui
import api.routers.library as library_router
import api.routers.references as references_router
from pathlib import Path

from kb.chat_store import ChatStore
from kb.task_runtime import (
    _apply_bound_source_hints,
    _augment_paper_guide_retrieval_prompt,
    _augment_prompt_with_source_hint,
    _build_bg_task,
    _build_precomputed_refs_render_payload,
    _maybe_append_library_figure_markdown,
    _build_paper_guide_direct_abstract_answer,
    _build_paper_guide_direct_citation_lookup_answer,
    _build_paper_guide_evidence_cards_block,
    _build_paper_guide_special_focus_block,
    _build_paper_guide_support_slots,
    _build_paper_guide_support_slots_block,
    _collect_paper_guide_candidate_refs_by_source,
    _drop_paper_guide_locate_only_line_citations,
    _extract_inline_reference_numbers,
    _extract_bound_paper_method_focus,
    _extract_bound_paper_figure_caption,
    _extract_caption_focus_fragment,
    _extract_caption_prompt_fragment,
    _extract_paper_guide_method_detail_excerpt,
    _inject_paper_guide_focus_citations,
    _inject_paper_guide_card_citations,
    _filter_history_for_multimodal_turn,
    _has_anchor_grounded_answer_hits,
    _inject_paper_guide_fallback_citations,
    _merge_paper_guide_deepread_context,
    _needs_conversational_source_hint,
    _paper_guide_has_requested_target_hits,
    _paper_guide_prompt_family,
    _paper_guide_requests_cross_paper_refs,
    _exclude_bound_source_hits_for_cross_paper_refs,
    _select_refs_async_rebuild_hits_raw,
    _should_allow_refs_async_enrich,
    _paper_guide_targeted_source_block_hits,
    _pick_recent_source_hint,
    _repair_paper_guide_focus_answer,
    _repair_paper_guide_focus_answer_generic,
    _inject_paper_guide_support_markers,
    _resolve_paper_guide_support_ref_num,
    _resolve_paper_guide_support_markers,
    _resolve_paper_guide_support_slot_block,
    _sanitize_paper_guide_answer_for_user,
    _select_paper_guide_deepread_extras,
    _select_paper_guide_answer_hits,
    _select_paper_guide_raw_target_hits,
    _stabilize_paper_guide_output_mode,
)
from tests._paper_guide_fixtures import build_paper_guide_runtime_fixture


def test_ultra_fast_does_not_force_no_llm():
    task = _build_bg_task(
        pdf_path=Path("a.pdf"),
        out_root=Path("out"),
        db_dir=Path("db"),
        no_llm=False,
        replace=False,
        speed_mode="ultra_fast",
    )
    assert task["speed_mode"] == "ultra_fast"
    assert task["no_llm"] is False


def test_no_llm_mode_respects_flag():
    task = _build_bg_task(
        pdf_path=Path("a.pdf"),
        out_root=Path("out"),
        db_dir=Path("db"),
        no_llm=True,
        replace=False,
        speed_mode="no_llm",
    )
    assert task["speed_mode"] == "no_llm"
    assert task["no_llm"] is True


def test_conversational_source_hint_detection_and_augmentation():
    assert _needs_conversational_source_hint("那这篇文章里的公式8写的是什么")
    assert not _needs_conversational_source_hint("NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf里的公式8写的是什么")
    assert (
        _augment_prompt_with_source_hint(
            "那这篇文章里的公式8写的是什么",
            "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf",
        )
        == "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf 那这篇文章里的公式8写的是什么"
    )


def test_apply_bound_source_hints_forces_bound_doc_context_on_short_followup():
    out = _apply_bound_source_hints(
        "how does it work",
        [
            "LSA-2026-Interferometric Image Scanning lateral resolution inside live cells",
            "LSA-2026-Interferometric Image Scanning lateral resolution inside live cells.pdf",
        ],
        limit=2,
    )
    assert out.endswith("how does it work")
    assert "LSA-2026-Interferometric Image Scanning lateral resolution inside live cells.pdf" in out
    assert "LSA-2026-Interferometric Image Scanning lateral resolution inside live cells" in out


def test_augment_prompt_with_source_hint_is_idempotent():
    q = "LSA-2026 paper how does it work"
    assert _augment_prompt_with_source_hint(q, "LSA-2026 paper") == q


def test_paper_guide_prompt_family_detects_broad_overview_compare_and_reproduce():
    assert _paper_guide_prompt_family("What problem does this paper solve and what are its core contributions?") == "overview"
    assert _paper_guide_prompt_family("Compared with open-pinhole confocal, what are the trade-offs?") == "compare"
    assert _paper_guide_prompt_family("If I want to reproduce the experiment, which hardware and acquisition parameters matter most?") == "reproduce"


def test_paper_guide_prompt_family_detects_citation_lookup_before_reproduce():
    assert (
        _paper_guide_prompt_family(
            "Which prior work is RVT attributed to in this paper, and what in-paper citation do they use when introducing it?"
        )
        == "citation_lookup"
    )
    assert (
        _paper_guide_prompt_family(
            "Which references are cited for Hadamard-basis examples in the acquisition strategy section, and where is that stated?"
        )
        == "citation_lookup"
    )


def test_augment_paper_guide_retrieval_prompt_avoids_reference_list_bias_for_citation_lookup():
    out = _augment_paper_guide_retrieval_prompt(
        "Which prior work is RVT attributed to in this paper, and what in-paper citation do they use when introducing it?",
        family="citation_lookup",
    )

    low = out.lower()
    assert "reference list" not in low
    assert "works cited" not in low
    assert "attributed to" in low


def test_augment_paper_guide_retrieval_prompt_keeps_reference_list_terms_for_explicit_request():
    out = _augment_paper_guide_retrieval_prompt(
        "From the reference list only, which entries correspond to the Richardson-Lucy method?",
        family="citation_lookup",
    )

    low = out.lower()
    assert "reference list" in low
    assert "works cited" in low


def test_sanitize_paper_guide_answer_for_user_keeps_structured_cites_for_citation_lookup():
    out = _sanitize_paper_guide_answer_for_user(
        "The paper uses [34] when introducing RVT [[CITE:s3583e628:1]].",
        has_hits=True,
        prompt="Which prior work is RVT attributed to in this paper, and what in-paper citation do they use when introducing it?",
    )
    assert "[[CITE:s3583e628:1]]" not in out
    assert "[34]" in out


def test_sanitize_paper_guide_answer_for_user_strips_structured_cites_for_negative_shell():
    out = _sanitize_paper_guide_answer_for_user(
        "The Discussion section does not mention commercial integration [[CITE:s3583e628:32]].",
        has_hits=True,
    )
    assert "[[CITE:" not in out
    assert "does not mention commercial integration" in out


def test_paper_guide_prompt_family_detects_chinese_overview_method_and_abstract_requests():
    assert _paper_guide_prompt_family("这篇文章讲了什么") == "overview"
    assert _paper_guide_prompt_family("这个方法具体介绍一下") == "method"
    assert _paper_guide_prompt_family("把摘要原文给出并翻译") == "abstract"


def test_paper_guide_prompt_family_detects_figure_walkthrough_requests():
    assert _paper_guide_prompt_family("Walk me through what Figure 1 demonstrates.") == "figure_walkthrough"
    assert _paper_guide_prompt_family("解释一下图1每个 panel 在说什么") == "figure_walkthrough"


def test_paper_guide_requests_cross_paper_refs_detects_external_library_queries():
    assert _paper_guide_requests_cross_paper_refs(
        "Besides this paper, what other papers in my library discuss Fourier single-pixel imaging?"
    )
    assert _paper_guide_requests_cross_paper_refs(
        "Which other papers in my library mention ADMM?"
    )
    assert not _paper_guide_requests_cross_paper_refs(
        "Where in this paper is Figure 2 discussed?"
    )


def test_should_allow_refs_async_enrich_keeps_cross_paper_paper_guide_queries_enabled():
    assert _should_allow_refs_async_enrich(
        refs_async_enabled=True,
        paper_guide_mode=True,
        refs_async_in_paper_guide=False,
        paper_guide_cross_paper_refs=True,
    )
    assert not _should_allow_refs_async_enrich(
        refs_async_enabled=True,
        paper_guide_mode=True,
        refs_async_in_paper_guide=False,
        paper_guide_cross_paper_refs=False,
    )


def test_select_refs_async_rebuild_hits_raw_prefers_unscoped_hits_for_cross_paper_refs():
    scoped = [{"id": "scoped"}]
    unscoped = [{"id": "external"}, {"id": "bound"}]

    assert _select_refs_async_rebuild_hits_raw(
        hits_raw=scoped,
        refs_unscoped_hits_raw=unscoped,
        paper_guide_cross_paper_refs=True,
    ) == unscoped
    assert _select_refs_async_rebuild_hits_raw(
        hits_raw=scoped,
        refs_unscoped_hits_raw=unscoped,
        paper_guide_cross_paper_refs=False,
    ) == scoped


def test_build_precomputed_refs_render_payload_uses_bounded_full_variant(monkeypatch):
    calls: dict[str, object] = {}

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: None)
    monkeypatch.setattr(library_router, "_md_dir", lambda: None)
    monkeypatch.setattr(references_router, "_refs_pack_render_signature", lambda **kwargs: "sig-precomputed")

    def fake_enrich_refs_payload(refs_by_user, **kwargs):
        calls["refs_by_user"] = refs_by_user
        calls["kwargs"] = dict(kwargs)
        return {7: {"hits": [{"ui_meta": {"summary_line": "bounded-full"}}]}}

    monkeypatch.setattr(reference_ui, "enrich_refs_payload", fake_enrich_refs_payload)

    payload, sig = _build_precomputed_refs_render_payload(
        user_msg_id=7,
        prompt="Which paper compares Hadamard and Fourier SPI?",
        prompt_sig="sig-7",
        hits=[
            {
                "text": "Figure 1 compares Hadamard and Fourier basis patterns.",
                "meta": {
                    "source_path": r"db\OE-2017\OE-2017.en.md",
                    "ref_pack_state": "ready",
                },
            }
        ],
        scores=[9.4],
        used_query="Hadamard Fourier single-pixel imaging compare",
        used_translation=False,
        guide_mode=False,
        guide_source_path="",
        guide_source_name="",
        library_db_path=None,
    )

    assert payload == {"hits": [{"ui_meta": {"summary_line": "bounded-full"}}]}
    assert sig == "sig-precomputed"
    assert (calls.get("refs_by_user") or {}).get(7, {}).get("hits")[0]["meta"]["ref_pack_state"] == "ready"
    kwargs = dict(calls.get("kwargs") or {})
    assert kwargs.get("render_variant") == "bounded_full"
    assert kwargs.get("allow_expensive_llm_for_ready") is False
    assert kwargs.get("allow_exact_locate") is True


def test_exclude_bound_source_hits_for_cross_paper_refs_drops_current_paper_before_grouping():
    hits = [
        {
            "meta": {
                "source_path": r"db\NatPhoton-2019-Principles and prospects for single-pixel imaging\NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md",
            }
        },
        {
            "meta": {
                "source_path": r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md",
            }
        },
    ]

    out = _exclude_bound_source_hits_for_cross_paper_refs(
        hits,
        bound_source_path=r"db\NatPhoton-2019-Principles and prospects for single-pixel imaging\NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md",
        bound_source_name="NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf",
    )

    assert len(out) == 1
    kept = str(((out[0].get("meta") or {}).get("source_path") or ""))
    assert "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging" in kept


def test_augment_paper_guide_retrieval_prompt_adds_section_bias_for_generic_question():
    out = _augment_paper_guide_retrieval_prompt(
        "What problem does this paper solve?",
        intent="reading",
        output_mode="reading_guide",
    )
    assert "abstract" in out.lower()
    assert "introduction" in out.lower()
    assert "results" in out.lower()


def test_augment_paper_guide_retrieval_prompt_uses_explicit_family_not_source_title_tokens():
    out = _augment_paper_guide_retrieval_prompt(
        "LSA-2026 Interferometric Image Scanning Microscopy for 120 nm lateral resolution inside live cells 把摘要原文给出并翻译",
        family="abstract",
        intent="reading",
        output_mode="reading_guide",
    )
    low = out.lower()
    assert "abstract" in low
    assert "contrast-to-noise ratio" not in low
    assert "open pinhole" not in low


def test_augment_paper_guide_retrieval_prompt_adds_figure_caption_bias():
    out = _augment_paper_guide_retrieval_prompt(
        "Walk me through what Figure 1 demonstrates.",
        family="figure_walkthrough",
        intent="reading",
        output_mode="reading_guide",
    )
    low = out.lower()
    assert "figure 1" in low
    assert "caption" in low
    assert "panel" in low


def test_augment_paper_guide_retrieval_prompt_adds_box_and_discussion_targets():
    out_box = _augment_paper_guide_retrieval_prompt(
        "From Box 1 only, what condition on M is given for reconstructing the image in the transform domain?",
        intent="reading",
        output_mode="reading_guide",
    )
    out_discussion = _augment_paper_guide_retrieval_prompt(
        "From the Discussion only, what future directions do the authors suggest?",
        intent="reading",
        output_mode="reading_guide",
    )
    assert "box 1" in out_box.lower()
    assert "discussion" in out_discussion.lower()


def test_stabilize_paper_guide_output_mode_prevents_generic_overview_from_drifting_critical():
    out = _stabilize_paper_guide_output_mode(
        "critical_review",
        prompt="What problem does this paper solve, and what are its core contributions?",
        intent="reading",
        explicit_hint="",
    )
    assert out == "reading_guide"


def test_pick_recent_source_hint_from_message_refs(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation()
    user1 = store.append_message(conv_id, "user", "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf这篇文章的第二张图是什么")
    store.append_message(conv_id, "assistant", "...")
    store.upsert_message_refs(
        user_msg_id=user1,
        conv_id=conv_id,
        prompt="NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf这篇文章的第二张图是什么",
        prompt_sig="sig1",
        hits=[
            {
                "text": "Fig. 2 | Evaluating regularized and non-regularized image reconstructions with compressive sensing",
                "meta": {
                    "source_path": str(
                        Path("db")
                        / "NatPhoton-2019-Principles and prospects for single-pixel imaging"
                        / "NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md"
                    ),
                },
            }
        ],
        scores=[9.0],
        used_query="NatPhoton-2019 ... figure 2",
        used_translation=False,
    )
    user2 = store.append_message(conv_id, "user", "那这篇文章里的公式8写的是什么")
    hint = _pick_recent_source_hint(conv_id=conv_id, user_msg_id=user2, chat_store=store)
    assert hint == "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf"


def test_has_anchor_grounded_answer_hits_detects_numbered_item_match():
    hits = [
        {
            "meta": {
                "anchor_target_kind": "equation",
                "anchor_target_number": 8,
                "anchor_match_score": 14.5,
            }
        }
    ]
    assert _has_anchor_grounded_answer_hits(hits) is True


def test_has_anchor_grounded_answer_hits_ignores_unmatched_items():
    hits = [
        {
            "meta": {
                "anchor_target_kind": "equation",
                "anchor_target_number": 8,
                "anchor_match_score": 0.0,
            }
        }
    ]
    assert _has_anchor_grounded_answer_hits(hits) is False


def test_filter_history_for_multimodal_turn_skips_old_image_roundtrip():
    history = [
        {
            "id": 1,
            "role": "user",
            "content": "[Image attachment x1]",
            "attachments": [{"mime": "image/png", "path": "a.png"}],
        },
        {
            "id": 2,
            "role": "assistant",
            "content": "上一张图里是一个流程图。",
        },
        {
            "id": 3,
            "role": "user",
            "content": "继续说说方法细节",
            "attachments": [],
        },
        {
            "id": 4,
            "role": "assistant",
            "content": "这里的方法用了 NeRF。",
        },
    ]

    out = _filter_history_for_multimodal_turn(
        history,
        cur_user_msg_id=9,
        cur_assistant_msg_id=10,
        has_current_images=True,
    )

    assert [int(item["id"]) for item in out] == [3, 4]


def test_filter_history_for_multimodal_turn_keeps_old_history_without_current_images():
    history = [
        {
            "id": 1,
            "role": "user",
            "content": "[Image attachment x1]",
            "attachments": [{"mime": "image/png", "path": "a.png"}],
        },
        {
            "id": 2,
            "role": "assistant",
            "content": "上一张图里是一个流程图。",
        },
    ]

    out = _filter_history_for_multimodal_turn(
        history,
        cur_user_msg_id=9,
        cur_assistant_msg_id=10,
        has_current_images=False,
    )

    assert [int(item["id"]) for item in out] == [1, 2]


def test_select_paper_guide_answer_hits_prefers_specific_sections_over_title_and_refs():
    src = r"db\demo\paper.en.md"
    hits = [
        {
            "score": 39.0,
            "text": "# Demo Paper\nAuthor A",
            "meta": {"source_path": src, "heading_path": "Demo Paper"},
        },
        {
            "score": 27.0,
            "text": "We developed a new interferometric microscope and modified the workflow.",
            "meta": {"source_path": src, "heading_path": "Abstract / Results / Principle of interferometric ISM (iISM)"},
        },
        {
            "score": 18.0,
            "text": "## References\n[1] Demo",
            "meta": {"source_path": src, "heading_path": "Abstract / References"},
        },
        {
            "score": 16.0,
            "text": "APR was performed using image registration based on phase correlation.",
            "meta": {"source_path": src, "heading_path": "Abstract / Materials and Methods / Adaptive pixel-reassignment (APR)"},
        },
    ]

    out = _select_paper_guide_answer_hits(
        grouped_docs=[],
        heading_hits=hits,
        prompt="文中的方法怎么做到的",
        top_n=3,
    )

    headings = [str((h.get("meta", {}) or {}).get("top_heading") or "") for h in out]
    assert "Results / Principle of interferometric ISM (iISM)" in headings
    assert "Materials and Methods / Adaptive pixel-reassignment (APR)" in headings
    assert all("References" not in h for h in headings)
    assert all("Demo Paper" != h for h in headings)


def test_select_paper_guide_answer_hits_prefers_overview_sections_for_generic_question():
    src = r"db\demo\paper.en.md"
    hits = [
        {
            "score": 15.0,
            "text": "We built the setup with a custom microscope and camera acquisition chain.",
            "meta": {"source_path": src, "heading_path": "Materials and Methods / Microscope setup"},
        },
        {
            "score": 14.0,
            "text": "Here, we introduce interferometric image scanning microscopy for label-free live-cell imaging.",
            "meta": {"source_path": src, "heading_path": "Abstract"},
        },
        {
            "score": 13.0,
            "text": "In this work, we propose a coherent ISM workflow and summarize the main contribution.",
            "meta": {"source_path": src, "heading_path": "Introduction"},
        },
    ]

    out = _select_paper_guide_answer_hits(
        grouped_docs=[],
        heading_hits=hits,
        prompt="What problem does this paper solve and what are the core contributions?",
        top_n=2,
    )

    headings = [str((h.get("meta", {}) or {}).get("top_heading") or (h.get("meta", {}) or {}).get("heading_path") or "") for h in out]
    assert headings[0] in {"Abstract", "Introduction"}
    assert any(h in {"Abstract", "Introduction"} for h in headings)
    assert any("Introduction" in h for h in headings)


def test_select_paper_guide_answer_hits_prefers_abstract_for_abstract_request():
    src = r"db\demo\paper.en.md"
    hits = [
        {
            "score": 16.0,
            "text": "Light microscopy remains indispensable in life sciences.",
            "meta": {"source_path": src, "heading_path": "Abstract"},
        },
        {
            "score": 18.0,
            "text": "Recent work has extended iSCAT for live-cell imaging.",
            "meta": {"source_path": src, "heading_path": "Introduction"},
        },
        {
            "score": 15.0,
            "text": "## References\n[1] Demo",
            "meta": {"source_path": src, "heading_path": "References"},
        },
    ]

    out = _select_paper_guide_answer_hits(
        grouped_docs=[],
        heading_hits=hits,
        prompt="把摘要原文给出并翻译",
        top_n=2,
    )

    headings = [str((h.get("meta", {}) or {}).get("top_heading") or (h.get("meta", {}) or {}).get("heading_path") or "") for h in out]
    assert headings[0] == "Abstract"
    assert all("References" not in h for h in headings)


def test_select_paper_guide_answer_hits_prefers_figure_caption_for_figure_walkthrough():
    src = r"db\demo\paper.en.md"
    hits = [
        {
            "score": 17.0,
            "text": "Recent work has extended iSCAT inside live cells.",
            "meta": {"source_path": src, "heading_path": "Introduction"},
        },
        {
            "score": 15.0,
            "text": "Figure 1. a, Scheme of the setup. d/e, Open and closed pinhole confocal iSCAT. f/g, APR and line profiles.",
            "meta": {"source_path": src, "heading_path": "Results / Figure 1"},
        },
        {
            "score": 16.0,
            "text": "APR was performed using image registration based on phase correlation.",
            "meta": {"source_path": src, "heading_path": "Materials and Methods / APR"},
        },
    ]

    out = _select_paper_guide_answer_hits(
        grouped_docs=[],
        heading_hits=hits,
        prompt="Walk me through what Figure 1 demonstrates.",
        top_n=2,
    )

    headings = [str((h.get("meta", {}) or {}).get("top_heading") or (h.get("meta", {}) or {}).get("heading_path") or "") for h in out]
    assert headings[0] == "Results / Figure 1"
    assert any("Figure 1" in h for h in headings)


def test_select_paper_guide_answer_hits_prefers_box_target_over_generic_sections():
    src = r"db\demo\paper.en.md"
    hits = [
        {
            "score": 18.0,
            "text": "## Introduction\nThis section motivates the problem.",
            "meta": {"source_path": src, "heading_path": "Introduction"},
        },
        {
            "score": 12.0,
            "text": "**[Box 1 - The maths behind single-pixel imaging]**\nWhen M >= O(K log(N/K)), the image can be reconstructed.",
            "meta": {
                "source_path": src,
                "heading_path": "Acquisition / Box 1",
                "paper_guide_targeted_block": True,
            },
        },
    ]

    out = _select_paper_guide_answer_hits(
        grouped_docs=[],
        heading_hits=hits,
        prompt="From Box 1 only, what condition on M is given for reconstructing the image in the transform domain?",
        top_n=1,
    )

    assert len(out) == 1
    assert "Box 1" in str(out[0].get("text") or "")


def test_select_paper_guide_answer_hits_keeps_multiple_target_blocks_from_same_heading():
    src = r"db\demo\paper.en.md"
    hits = [
        {
            "score": 18.0,
            "text": "**[Box 1 - The maths behind single-pixel imaging]**",
            "meta": {
                "source_path": src,
                "heading_path": "Acquisition and image reconstruction strategies",
                "block_id": "blk_box_header",
                "paper_guide_targeted_block": True,
            },
        },
        {
            "score": 22.0,
            "text": "It can be shown that when the number of sampling patterns used $M \\ge O(K \\log(N/K))$, the image in the transform domain can be reconstructed.",
            "meta": {
                "source_path": src,
                "heading_path": "Acquisition and image reconstruction strategies",
                "block_id": "blk_box_condition",
                "paper_guide_targeted_block": True,
            },
        },
    ]

    out = _select_paper_guide_answer_hits(
        grouped_docs=hits,
        heading_hits=hits,
        prompt="From Box 1 only, what condition on M is given for reconstructing the image in the transform domain?",
        top_n=2,
    )

    assert len(out) == 2
    assert any(str((hit.get('meta') or {}).get('block_id') or '') == 'blk_box_condition' for hit in out)


def test_select_paper_guide_answer_hits_prefers_acquisition_strategy_target():
    src = r"db\demo\paper.en.md"
    hits = [
        {
            "score": 18.0,
            "text": "## Introduction\nThis section motivates the problem.",
            "meta": {"source_path": src, "heading_path": "Introduction"},
        },
        {
            "score": 12.0,
            "text": "An alternative approach is to perform sampling using the Hadamard [64,65] basis.",
            "meta": {
                "source_path": src,
                "heading_path": "Acquisition and image reconstruction strategies",
                "paper_guide_targeted_block": True,
            },
        },
    ]

    out = _select_paper_guide_answer_hits(
        grouped_docs=[],
        heading_hits=hits,
        prompt="Which references are cited for Hadamard-basis examples in the acquisition strategy section, and where is that stated?",
        top_n=1,
    )

    assert len(out) == 1
    assert "Hadamard" in str(out[0].get("text") or "")


def test_select_paper_guide_answer_hits_prefers_intext_attribution_over_reference_list():
    src = r"db\demo\paper.en.md"
    hits = [
        {
            "score": 18.0,
            "text": "[33] Richardson, W. H. Bayesian-based iterative method of image restoration.",
            "meta": {
                "source_path": src,
                "heading_path": "References",
                "block_id": "blk_refs",
            },
        },
        {
            "score": 15.0,
            "text": (
                "We invert the ISM image-formation model using a maximum likelihood estimation technique "
                "akin to the Richardson-Lucy method [33,34]."
            ),
            "meta": {
                "source_path": src,
                "heading_path": "Abstract / Results",
                "block_id": "blk_intext",
            },
        },
    ]

    out = _select_paper_guide_answer_hits(
        grouped_docs=[],
        heading_hits=hits,
        prompt="Which references does the paper cite for the maximum-likelihood / Richardson-Lucy connection, and where is that stated exactly?",
        top_n=1,
    )

    assert len(out) == 1
    assert str((out[0].get("meta") or {}).get("block_id") or "") == "blk_intext"


def test_select_paper_guide_deepread_extras_skips_references_for_abstract_request():
    extras = [
        {
            "score": 99.0,
            "text": "## References\n[1] Demo reference.\n[2] Another reference.",
            "meta": {"heading_path": "References"},
        },
        {
            "score": 55.0,
            "text": "# Abstract\nHere we introduce interferometric image scanning microscopy for live-cell imaging.",
            "meta": {"heading_path": "Abstract"},
        },
        {
            "score": 54.0,
            "text": "## Introduction\nRecent work has extended iSCAT inside live cells.",
            "meta": {"heading_path": "Introduction"},
        },
    ]

    out = _select_paper_guide_deepread_extras(
        extras,
        prompt="把摘要原文给出并翻译",
        prompt_family="abstract",
        limit=1,
    )

    assert len(out) == 1
    assert out[0].startswith("# Abstract")


def test_select_paper_guide_deepread_extras_prefers_box_target():
    extras = [
        {
            "score": 65.0,
            "text": "## Introduction\nThis section motivates the problem.",
            "meta": {"heading_path": "Introduction"},
        },
        {
            "score": 30.0,
            "text": "**[Box 1 - The maths behind single-pixel imaging]**\nWhen the number of sampling patterns used M >= O(K log(N/K)), the image in the transform domain can be reconstructed.",
            "meta": {"heading_path": "Box 1"},
        },
        {
            "score": 50.0,
            "text": "## References\n[57] Candes and Tao.",
            "meta": {"heading_path": "References"},
        },
    ]

    out = _select_paper_guide_deepread_extras(
        extras,
        prompt="From Box 1 only, what condition on M is given for reconstructing the image in the transform domain?",
        prompt_family="",
        limit=1,
    )

    assert len(out) == 1
    assert "Box 1" in out[0]


def test_select_paper_guide_deepread_extras_prefers_discussion_target():
    extras = [
        {
            "score": 52.0,
            "text": "## Results\nThe method improves sectioning and SNR.",
            "meta": {"heading_path": "Results"},
        },
        {
            "score": 28.0,
            "text": "## Discussion\nThe approach could be extended to multicolour and live-cell experiments.",
            "meta": {"heading_path": "Discussion"},
        },
    ]

    out = _select_paper_guide_deepread_extras(
        extras,
        prompt="From the Discussion only, what future directions do the authors suggest?",
        prompt_family="overview",
        limit=1,
    )

    assert len(out) == 1
    assert out[0].startswith("## Discussion")


def test_paper_guide_has_requested_target_hits_detects_box_hit():
    hits = [
        {
            "text": "## Introduction\nThis section motivates the problem.",
            "meta": {"heading_path": "Introduction"},
        },
        {
            "text": "**[Box 1 - The maths behind single-pixel imaging]**\nWhen M >= O(K log(N/K)), the image can be reconstructed.",
            "meta": {"heading_path": "Box 1"},
        },
    ]
    assert _paper_guide_has_requested_target_hits(
        hits,
        prompt="From Box 1 only, what condition on M is given for reconstructing the image in the transform domain?",
    )


def test_paper_guide_has_requested_target_hits_detects_target_miss():
    hits = [
        {
            "text": "## Introduction\nThis section motivates the problem.",
            "meta": {"heading_path": "Introduction"},
        },
        {
            "text": "## Results\nThe method improves sectioning and SNR.",
            "meta": {"heading_path": "Results"},
        },
    ]
    assert not _paper_guide_has_requested_target_hits(
        hits,
        prompt="From Box 1 only, what condition on M is given for reconstructing the image in the transform domain?",
    )


def test_paper_guide_targeted_source_block_hits_extracts_box_block(tmp_path: Path):
    fixture = build_paper_guide_runtime_fixture(tmp_path)
    hits = _paper_guide_targeted_source_block_hits(
        bound_source_path=str(fixture["nat_source_path"]),
        prompt="From Box 1 only, what condition on M is given for reconstructing the image in the transform domain?",
        db_dir=Path(fixture["db_root"]),
        limit=4,
    )
    assert hits
    assert any("Box 1" in str(hit.get("text") or "") for hit in hits)
    assert any("M \\ge O(K \\log(N/K))" in str(hit.get("text") or "") for hit in hits)


def test_paper_guide_targeted_source_block_hits_extracts_hadamard_block(tmp_path: Path):
    fixture = build_paper_guide_runtime_fixture(tmp_path)
    hits = _paper_guide_targeted_source_block_hits(
        bound_source_path=str(fixture["nat_source_path"]),
        prompt="Which references are cited for Hadamard-basis examples in the acquisition strategy section, and where is that stated?",
        db_dir=Path(fixture["db_root"]),
        limit=3,
    )
    assert hits
    assert any("Hadamard" in str(hit.get("text") or "") and "64,65" in str(hit.get("text") or "") for hit in hits)


def test_select_paper_guide_raw_target_hits_prefers_box_condition_block_over_box_header():
    src = r"db\demo\paper.en.md"
    hits = [
        {
            "score": 20.0,
            "text": "**[Box 1 - The maths behind single-pixel imaging]**",
            "meta": {
                "source_path": src,
                "heading_path": "Acquisition / Box 1",
                "block_id": "blk_box_header",
                "paper_guide_targeted_block": True,
            },
        },
        {
            "score": 16.0,
            "text": "It can be shown that when the number of sampling patterns used $M \\ge O(K \\log(N/K))$, the image in the transform domain can be reconstructed.",
            "meta": {
                "source_path": src,
                "heading_path": "Acquisition / Box 1",
                "block_id": "blk_box_condition",
                "paper_guide_targeted_block": True,
            },
        },
    ]

    out = _select_paper_guide_raw_target_hits(
        hits_raw=hits,
        prompt="From Box 1 only, what condition on M is given for reconstructing the image in the transform domain?",
        top_n=1,
    )

    assert len(out) == 1
    assert str((out[0].get("meta") or {}).get("block_id") or "") == "blk_box_condition"


def test_select_paper_guide_raw_target_hits_supports_citation_lookup_without_explicit_section_target():
    src = r"db\demo\paper.en.md"
    hits = [
        {
            "score": 12.0,
            "text": "[33] Richardson, W. H. Bayesian-based iterative method of image restoration.",
            "meta": {
                "source_path": src,
                "heading_path": "References",
                "block_id": "blk_refs",
            },
        },
        {
            "score": 10.0,
            "text": "Specifically, we use the radial variance transform (RVT)[34], which converts an interferogram into an intensity-only map.",
            "meta": {
                "source_path": src,
                "heading_path": "Methods / RVT",
                "block_id": "blk_rvt_intro",
                "paper_guide_targeted_block": True,
            },
        },
    ]

    out = _select_paper_guide_raw_target_hits(
        hits_raw=hits,
        prompt="Which prior work is RVT attributed to in this paper, and what in-paper citation do they use when introducing it?",
        top_n=1,
    )

    assert len(out) == 1
    assert str((out[0].get("meta") or {}).get("block_id") or "") == "blk_rvt_intro"


def test_build_paper_guide_special_focus_block_for_citation_lookup_prefers_explicit_ref_span():
    block = _build_paper_guide_special_focus_block(
        [
            {
                "doc_idx": 1,
                "sid": "s1",
                "source_path": r"db\demo\paper.en.md",
                "heading": "Acquisition and image reconstruction strategies",
                "snippet": "An alternative approach is to perform sampling using the Hadamard$^{64,65}$ basis.",
                "deepread_texts": [],
            }
        ],
        prompt="Which references are cited for Hadamard-basis examples in the acquisition strategy section, and where is that stated?",
        prompt_family="citation_lookup",
        source_path=r"db\demo\paper.en.md",
        db_dir=Path("db"),
        answer_hits=[],
    )

    assert "citation focus" in block.lower()
    assert "Hadamard$^{64,65}$" in block


def test_build_paper_guide_special_focus_block_for_citation_lookup_prefers_intext_attribution_over_reference_list():
    block = _build_paper_guide_special_focus_block(
        [
            {
                "doc_idx": 1,
                "sid": "s1",
                "source_path": r"db\demo\paper.en.md",
                "heading": "References",
                "snippet": "[33] Richardson, W. H. Bayesian-based iterative method of image restoration.",
                "deepread_texts": [],
            },
            {
                "doc_idx": 2,
                "sid": "s2",
                "source_path": r"db\demo\paper.en.md",
                "heading": "Abstract / Results",
                "snippet": (
                    "We invert the model using a maximum likelihood estimation technique "
                    "akin to the Richardson-Lucy method [33,34]."
                ),
                "deepread_texts": [],
            },
        ],
        prompt="Which references does the paper cite for the maximum-likelihood / Richardson-Lucy connection, and where is that stated exactly?",
        prompt_family="citation_lookup",
        source_path=r"db\demo\paper.en.md",
        db_dir=Path("db"),
        answer_hits=[],
    )

    assert "citation focus" in block.lower()
    assert "akin to the Richardson-Lucy method [33,34]" in block


def test_build_paper_guide_special_focus_block_for_box_target_prefers_condition_span():
    block = _build_paper_guide_special_focus_block(
        [
            {
                "doc_idx": 1,
                "sid": "s1",
                "source_path": r"db\demo\paper.en.md",
                "heading": "Acquisition and image reconstruction strategies",
                "snippet": "It can be shown that when the number of sampling patterns used $M \\ge O(K \\log(N/K))$, the image in the transform domain can be reconstructed.",
                "deepread_texts": [],
            }
        ],
        prompt="From Box 1 only, what condition on M is given for reconstructing the image in the transform domain?",
        prompt_family="",
        source_path=r"db\demo\paper.en.md",
        db_dir=Path("db"),
        answer_hits=[],
    )

    assert "targeted focus" in block.lower()
    assert "M \\ge O(K \\log(N/K))" in block


def test_repair_paper_guide_focus_answer_generic_replaces_not_stated_for_citation_lookup():
    out = _repair_paper_guide_focus_answer_generic(
        "The retrieved context does not mention the cited references.",
        prompt="Which references are cited for Hadamard-basis examples in the acquisition strategy section, and where is that stated?",
        prompt_family="citation_lookup",
        special_focus_block=(
            "Paper-guide citation focus:\n"
            "- Focus snippet:\n"
            "An alternative approach is to perform sampling using the Hadamard$^{64,65}$ basis."
        ),
        source_path=r"db\demo\paper.en.md",
        db_dir=Path("db"),
    )

    assert "states this explicitly" in out
    assert "64,65" in out


def test_build_paper_guide_direct_citation_lookup_answer_for_rvt(tmp_path: Path):
    fixture = build_paper_guide_runtime_fixture(tmp_path)
    out = _build_paper_guide_direct_citation_lookup_answer(
        prompt="Which prior work is RVT attributed to in this paper, and what in-paper citation do they use when introducing it?",
        source_path=str(fixture["lsa_source_path"]),
        answer_hits=[
            {
                "text": (
                    "Specifically, we use the radial variance transform (RVT)[34], "
                    "which converts an interferogram into an intensity-only map."
                ),
                "meta": {"heading_path": "Results / APR"},
            }
        ],
        special_focus_block="",
        db_dir=Path(fixture["db_root"]),
    )

    assert "[34]" in out
    assert "Precision single-particle localization using radial variance transform" in out


def test_build_paper_guide_direct_citation_lookup_answer_for_richardson_lucy():
    out = _build_paper_guide_direct_citation_lookup_answer(
        prompt="Which references does the paper cite for the maximum-likelihood / Richardson-Lucy connection, and where is that stated exactly?",
        source_path=r"X:\NatPhoton-2025-Structured detection for...in laser scanning microscopy.pdf",
        answer_hits=[
            {
                "text": (
                    "We invert the ISM image formation model using a maximum likelihood estimation technique "
                    "akin to the Richardson-Lucy method [33,34]."
                ),
                "meta": {"heading_path": "Abstract"},
            }
        ],
        special_focus_block="",
        db_dir=Path("db"),
    )

    assert "[33]" in out and "[34]" in out
    assert "Richardson" in out and "Lucy" in out
    assert "akin to the Richardson-Lucy method [33,34]" in out


def test_build_paper_guide_direct_citation_lookup_answer_prefers_intext_hadamard_example():
    out = _build_paper_guide_direct_citation_lookup_answer(
        prompt="Which references are cited for Hadamard-basis examples in the acquisition strategy section, and where is that stated?",
        source_path=r"X:\NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf",
        answer_hits=[
            {
                "text": (
                    "An alternative approach is to perform sampling using a basis that is not necessarily incoherent "
                    "with the spatial properties of the image, for example by using the Hadamard$^{64,65}$ basis."
                ),
                "meta": {"heading_path": "Acquisition and image reconstruction strategies"},
            },
            {
                "text": "In one experiment, an evolutionary Hadamard basis scanning technique was used$^{16}$.",
                "meta": {"heading_path": "Acquisition and image reconstruction strategies"},
            },
        ],
        special_focus_block="",
        db_dir=Path("db"),
    )

    assert "[64]" in out and "[65]" in out
    assert "[16]" not in out
    assert "[19]" not in out and "[66]" not in out


def test_extract_inline_reference_numbers_ignores_s2ism_bracketed_digit():
    out = _extract_inline_reference_numbers(
        "To validate s [2]ISM, we compare it with standard reconstruction methods. "
        "Furthermore, dropping the detector dimension recovers the Richardson-Lucy equation [33,34].",
        max_candidates=6,
    )

    assert out == [33, 34]


def test_repair_paper_guide_focus_answer_generic_replaces_not_stated_for_box_target():
    out = _repair_paper_guide_focus_answer_generic(
        "The retrieved context does not state the condition on M.",
        prompt="From Box 1 only, what condition on M is given for reconstructing the image in the transform domain?",
        prompt_family="",
        special_focus_block=(
            "Paper-guide targeted focus:\n"
            "- Focus snippet:\n"
            "It can be shown that when the number of sampling patterns used $M \\ge O(K \\log(N/K))$, the image in the transform domain can be reconstructed."
        ),
        source_path=r"db\demo\paper.en.md",
        db_dir=Path("db"),
    )

    assert "states this explicitly" in out
    assert "M \\ge O(K \\log(N/K))" in out


def test_merge_paper_guide_deepread_context_replaces_title_shell_for_abstract():
    out = _merge_paper_guide_deepread_context(
        "DOC-1 [SID:s1] paper.en.md | Abstract\n# Demo Paper\nAuthor A",
        "# Abstract\nLight microscopy remains indispensable in life sciences.",
        prompt_family="abstract",
    )
    assert "Demo Paper" not in out
    assert "Light microscopy remains indispensable" in out


def test_merge_paper_guide_deepread_context_prepends_figure_caption_for_walkthrough():
    out = _merge_paper_guide_deepread_context(
        "DOC-1 [SID:s1] paper.en.md | Results / Figure 1\nShort result summary.",
        "Figure 1. d/e, Open and closed pinhole confocal iSCAT. f/g, APR and line profiles.",
        prompt_family="figure_walkthrough",
    )
    assert "Figure 1." in out
    assert "Short result summary." in out
    assert out.index("Figure 1.") < out.index("Short result summary.")


def test_merge_paper_guide_deepread_context_prepends_targeted_box_excerpt():
    out = _merge_paper_guide_deepread_context(
        "DOC-1 [SID:s1] paper.en.md | Introduction\nShort overview.",
        "**[Box 1 - The maths behind single-pixel imaging]**\nWhen M >= O(K log(N/K)), the image can be reconstructed in the transform domain.",
        prompt_family="overview",
        prompt="From Box 1 only, what condition on M is given for reconstructing the image in the transform domain?",
    )
    assert "Box 1" in out
    assert "Short overview." in out
    assert out.index("Box 1") < out.index("Short overview.")


def test_build_paper_guide_evidence_cards_block_prefers_abstract_excerpt_without_title_shell():
    block = _build_paper_guide_evidence_cards_block(
        [
            {
                "doc_idx": 1,
                "sid": "s50f9c165",
                "source_path": r"db\demo\paper.en.md",
                "heading": "Abstract",
                "snippet": "# Demo Paper\nAuthor A\n\n# Abstract\nLight microscopy remains indispensable in life sciences.\nWe introduce a coherent ISM workflow.",
                "deepread_texts": [],
            }
        ],
        prompt="把摘要原文给出并翻译",
        prompt_family="abstract",
    )
    assert "Paper-guide evidence cards" in block
    assert "Light microscopy remains indispensable" in block
    assert "Demo Paper" not in block
    assert "Author A" not in block


def test_build_paper_guide_evidence_cards_block_prefers_figure_caption_deepread_text():
    block = _build_paper_guide_evidence_cards_block(
        [
            {
                "doc_idx": 1,
                "sid": "s50f9c165",
                "source_path": r"db\demo\paper.en.md",
                "heading": "Results / Figure 1",
                "snippet": "Short result summary.",
                "candidate_refs": [8],
                "deepread_texts": [
                    "Figure 1. a, Scheme of the setup. d/e, Open and closed pinhole confocal iSCAT. f/g, APR and line profiles."
                ],
            }
        ],
        prompt="Walk me through what Figure 1 demonstrates.",
        prompt_family="figure_walkthrough",
    )
    assert "Describe only the figure/panels stated here" in block
    assert "Figure 1." in block
    assert "refs=8" in block
    assert "cite_example=[[CITE:s50f9c165:8]]" in block


def test_build_paper_guide_support_slots_extracts_method_ref_binding(tmp_path: Path):
    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "# Methods\n\n"
            "APR was performed using image registration based on phase correlation of the off-axis raw images "
            "with respect to the central one [35].\n"
        ),
        encoding="utf-8",
    )

    slots = _build_paper_guide_support_slots(
        [
            {
                "doc_idx": 1,
                "sid": "s50f9c165",
                "source_path": str(source_pdf),
                "heading": "Methods / APR",
                "snippet": (
                    "APR was performed using image registration based on phase correlation of the off-axis "
                    "raw images with respect to the central one [35]."
                ),
                "candidate_refs": [35],
                "deepread_texts": [],
            }
        ],
        prompt="Explain how APR is implemented in this method.",
        prompt_family="method",
        db_dir=tmp_path,
    )

    assert len(slots) == 1
    slot = slots[0]
    assert slot["claim_type"] == "method_detail"
    assert slot["cite_policy"] == "prefer_ref"
    assert slot["candidate_refs"] == [35]
    assert slot["support_example"] == "[[SUPPORT:DOC-1]]"
    assert slot["cite_example"] == "[[CITE:s50f9c165:35]]"
    assert str(slot.get("block_id") or "").strip()
    assert "phase correlation" in str(slot.get("locate_anchor") or "").lower()


def test_build_paper_guide_support_slots_prefers_matching_content_over_title_block(tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "Interferometric Image Scanning Microscopy for label-free imaging at 120 nm lateral resolution inside live cells\n\n"
            "## Results\n\n"
            "### iISM with adaptive pixel-reassignment (APR)\n\n"
            "a Matrix representing the scanned interference images of a single 60 nm polystyrene nanoparticle (PNP) "
            "with a 9 x 9 array detection. White dashed circle corresponds to 1 AU.\n"
        ),
        encoding="utf-8",
    )

    blocks = task_runtime.load_source_blocks(md_main)
    title_block = next(
        block for block in blocks
        if "120 nm lateral resolution inside live cells" in str(block.get("text") or "")
    )

    slots = _build_paper_guide_support_slots(
        [
            {
                "doc_idx": 1,
                "sid": "s50f9c165",
                "source_path": str(source_pdf),
                "heading": "Results / Figure 2",
                "snippet": (
                    "a Matrix representing the scanned interference images of a single 60 nm polystyrene "
                    "nanoparticle (PNP) with a 9 x 9 array detection. White dashed circle corresponds to 1 AU."
                ),
                "candidate_refs": [],
                "deepread_texts": [],
            }
        ],
        prompt="Walk through Figure 2 and explain the APR workflow.",
        prompt_family="figure_walkthrough",
        db_dir=tmp_path,
    )

    assert len(slots) == 1
    slot = slots[0]
    assert str(slot.get("block_id") or "") != str(title_block.get("block_id") or "")
    assert "Matrix representing the scanned interference images" in str(slot.get("locate_anchor") or "")


def test_build_paper_guide_support_slots_marks_method_detail_inside_figure_walkthrough(tmp_path: Path):
    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "## Results\n\n"
            "### Figure 2\n\n"
            "Figure 2. APR workflow.\n\n"
            "## Materials and Methods\n\n"
            "### Radial variance transform (RVT)\n\n"
            "RVT converts an interferogram into an intensity-only map that reflects the local degree of symmetry.\n"
        ),
        encoding="utf-8",
    )

    slots = _build_paper_guide_support_slots(
        [
            {
                "doc_idx": 1,
                "sid": "s50f9c165",
                "source_path": str(source_pdf),
                "heading": "Results / Figure 2",
                "snippet": "RVT converts an interferogram into an intensity-only map that reflects the local degree of symmetry.",
                "candidate_refs": [],
                "deepread_texts": [],
            }
        ],
        prompt="Walk through Figure 2 and explain what RVT does in the pipeline.",
        prompt_family="figure_walkthrough",
        db_dir=tmp_path,
    )

    assert len(slots) == 1
    slot = slots[0]
    assert slot["claim_type"] == "method_detail"
    assert slot["cite_policy"] == "prefer_ref"
    assert "Radial variance transform" in str(slot.get("heading_path") or "")


def test_build_paper_guide_support_slots_prefers_rvt_block_over_setup_in_figure_walkthrough(tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "Interferometric Image Scanning Microscopy for live cells\n\n"
            "## Results\n\n"
            "### Figure 2\n\n"
            "Figure 2. APR workflow.\n\n"
            "## Materials and Methods\n\n"
            "### Microscope setup\n\n"
            "The microscope setup uses two lasers and a polarizing beam splitter.\n\n"
            "### Radial variance transform (RVT)\n\n"
            "RVT converts an interferogram into an intensity-only map that reflects the local degree of symmetry.\n"
        ),
        encoding="utf-8",
    )

    blocks = task_runtime.load_source_blocks(md_main)
    setup_block = next(
        block for block in blocks
        if str(block.get("heading_path") or "").strip().lower().endswith("microscope setup")
    )

    slots = _build_paper_guide_support_slots(
        [
            {
                "doc_idx": 1,
                "sid": "s50f9c165",
                "source_path": str(source_pdf),
                "heading": "Results / Figure 2",
                "snippet": "RVT converts an interferogram into an intensity-only map that reflects the local degree of symmetry.",
                "candidate_refs": [],
                "deepread_texts": [],
            }
        ],
        prompt="Walk through Figure 2 and explain what RVT does in the pipeline.",
        prompt_family="figure_walkthrough",
        db_dir=tmp_path,
    )

    assert len(slots) == 1
    slot = slots[0]
    assert str(slot.get("block_id") or "") != str(setup_block.get("block_id") or "")
    assert "Radial variance transform" in str(slot.get("heading_path") or "")


def test_build_paper_guide_support_slots_uses_unique_markers_for_same_doc(tmp_path: Path):
    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "## Results\n\n"
            "Figure 2. APR workflow.\n\n"
            "## Methods\n\n"
            "APR was performed using image registration based on phase correlation.\n"
        ),
        encoding="utf-8",
    )

    slots = _build_paper_guide_support_slots(
        [
            {
                "doc_idx": 1,
                "sid": "s50f9c165",
                "source_path": str(source_pdf),
                "heading": "Results / Figure 2",
                "snippet": "Figure 2. APR workflow.",
                "candidate_refs": [],
                "deepread_texts": [],
            },
            {
                "doc_idx": 1,
                "sid": "s50f9c165",
                "source_path": str(source_pdf),
                "heading": "Methods / APR",
                "snippet": "APR was performed using image registration based on phase correlation [35].",
                "candidate_refs": [35],
                "deepread_texts": [],
            },
        ],
        prompt="Walk through Figure 2 and explain the APR workflow.",
        prompt_family="figure_walkthrough",
        db_dir=tmp_path,
    )

    assert len(slots) == 2
    assert slots[0]["support_example"] == "[[SUPPORT:DOC-1-S1]]"
    assert slots[1]["support_example"] == "[[SUPPORT:DOC-1-S2]]"


def test_build_paper_guide_support_slots_uses_phrase_candidate_for_reapply_detail(tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "## Results\n\n"
            "### Figure 2\n\n"
            "The workflow estimates shift vectors for the interferometric stack.\n\n"
            "For circularly polarized beams, the standard workflow is adapted to the interferometric case.\n"
            "In effect, the phase-correlation of the individual RVT images with the central pixel image defines the required shift.\n"
            "Finally, these RVT-APR shift vectors were applied back to the original iISM dataset, yielding reconstructions with enhanced spatial resolution.\n"
        ),
        encoding="utf-8",
    )

    blocks = task_runtime.load_source_blocks(md_main)
    broad_block = next(
        block for block in blocks
        if "estimates shift vectors" in str(block.get("text") or "").lower()
    )

    slots = _build_paper_guide_support_slots(
        [
            {
                "doc_idx": 1,
                "sid": "s50f9c165",
                "source_path": str(source_pdf),
                "heading": "Results / Figure 2",
                "snippet": "The resulting shift vectors were applied back to the original iISM dataset, yielding reconstructions with enhanced spatial resolution.",
                "candidate_refs": [],
                "deepread_texts": [],
            }
        ],
        prompt="Walk through Figure 2 and explain where the shift vectors are reapplied.",
        prompt_family="figure_walkthrough",
        db_dir=tmp_path,
    )

    assert len(slots) == 1
    slot = slots[0]
    assert str(slot.get("block_id") or "") != str(broad_block.get("block_id") or "")
    assert "applied back to the original iISM dataset" in str(slot.get("locate_anchor") or "")


def test_build_paper_guide_support_slots_targets_caption_panel_clauses_from_prompt(tmp_path: Path):
    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "## Results\n\n"
            "Figure 1. a Setup. d Open pinhole confocal iSCAT. e Closed pinhole confocal iSCAT. "
            "f Resulting iPSF from iISM after adaptive pixel-reassignment (APR). "
            "g Line profiles of the iPSF in the three configurations.\n"
        ),
        encoding="utf-8",
    )

    slots = _build_paper_guide_support_slots(
        [
            {
                "doc_idx": 1,
                "sid": "s50f9c165",
                "source_path": str(source_pdf),
                "heading": "Results / Figure 1",
                "snippet": (
                    "Figure 1. a Setup. d Open pinhole confocal iSCAT. e Closed pinhole confocal iSCAT. "
                    "f Resulting iPSF from iISM after adaptive pixel-reassignment (APR). "
                    "g Line profiles of the iPSF in the three configurations."
                ),
                "candidate_refs": [],
                "deepread_texts": [],
            }
        ],
        prompt="For Figure 1 panels (f) and (g), what exactly does each panel show? Point me to the exact supporting part of the paper.",
        prompt_family="figure_walkthrough",
        db_dir=tmp_path,
    )

    assert len(slots) == 1
    slot = slots[0]
    assert str(slot.get("claim_type") or "") == "figure_panel"
    assert "f Resulting iPSF from iISM after adaptive pixel-reassignment (APR)" in str(slot.get("snippet") or "")
    assert "g Line profiles of the iPSF in the three configurations" in str(slot.get("snippet") or "")


def test_resolve_paper_guide_support_slot_block_prefers_caption_for_multi_panel_probe(tmp_path: Path):
    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "## Results\n\n"
            "**Figure 1.** a Setup. d Open pinhole confocal iSCAT. e Closed pinhole confocal iSCAT. "
            "f Resulting iPSF from iISM after adaptive pixel-reassignment (APR), with same incident illumination power and number of detected photons. "
            "g Line profiles of the iPSF in the three configurations.\n\n"
            "Using iISM with our modified APR algorithm, we obtain a lateral resolution limit of 120 nm and report improved CNR in Fig. 1(g), red.\n"
        ),
        encoding="utf-8",
    )

    rec = _resolve_paper_guide_support_slot_block(
        source_path=str(source_pdf),
        snippet="f Resulting iPSF from iISM after adaptive pixel-reassignment (APR). g Line profiles of the iPSF in the three configurations.",
        heading="Results / Figure 1",
        prompt_family="figure_walkthrough",
        claim_type="figure_panel",
        db_dir=tmp_path,
    )

    assert "g Line profiles of the iPSF" in str(rec.get("locate_anchor") or "")
    assert "f Resulting iPSF from iISM after adaptive pixel-reassignment (APR)" in str(rec.get("block_text") or "")


def test_build_paper_guide_support_slots_block_includes_support_examples():
    block = _build_paper_guide_support_slots_block(
        [
            {
                "doc_idx": 1,
                "sid": "s50f9c165",
                "heading_path": "Methods / APR",
                "claim_type": "method_detail",
                "cite_policy": "prefer_ref",
                "support_example": "[[SUPPORT:DOC-1]]",
                "candidate_refs": [35],
                "cite_example": "[[CITE:s50f9c165:35]]",
                "snippet": "APR was performed using image registration based on phase correlation [35].",
                "locate_anchor": "APR was performed using image registration based on phase correlation.",
                "ref_spans": [{"text": "phase correlation [35]", "nums": [35]}],
            }
        ]
    )

    assert "Paper-guide support slots" in block
    assert "support_example=[[SUPPORT:DOC-1]]" in block
    assert "cite_example=[[CITE:s50f9c165:35]]" in block
    assert "ref_span:" in block


def test_resolve_paper_guide_support_markers_rewrites_to_structured_cite():
    answer, resolutions = _resolve_paper_guide_support_markers(
        "APR uses phase correlation to register the off-axis raw images [[SUPPORT:DOC-1]].",
        support_slots=[
            {
                "doc_idx": 1,
                "sid": "s50f9c165",
                "source_path": "demo.pdf",
                "block_id": "blk_apr",
                "anchor_id": "anc_apr",
                "heading_path": "Methods / APR",
                "locate_anchor": "APR uses phase correlation to register the off-axis raw images.",
                "claim_type": "method_detail",
                "cite_policy": "prefer_ref",
                "candidate_refs": [35],
                "ref_spans": [{"text": "phase correlation [35]", "nums": [35], "scope": "same_sentence"}],
            }
        ],
    )

    assert answer.endswith("[[CITE:s50f9c165:35]].")
    assert len(resolutions) == 1
    assert resolutions[0]["resolved_ref_num"] == 35
    assert resolutions[0]["citation_resolution_mode"] == "slot_ref_span"


def test_inject_paper_guide_support_markers_appends_method_marker():
    out = _inject_paper_guide_support_markers(
        "Evidence:\n- Implementation detail: APR uses phase correlation to register the off-axis raw images.",
        support_slots=[
            {
                "doc_idx": 1,
                "claim_type": "method_detail",
                "cite_policy": "prefer_ref",
                "support_example": "[[SUPPORT:DOC-1]]",
                "heading_path": "Methods / APR",
                "snippet": "APR was performed using image registration based on phase correlation of the off-axis raw images [35].",
                "locate_anchor": "APR was performed using image registration based on phase correlation.",
            }
        ],
        prompt_family="method",
    )

    assert "Implementation detail:" in out
    assert "[[SUPPORT:DOC-1]]" in out


def test_inject_paper_guide_support_markers_appends_compare_marker_for_numeric_tradeoff():
    out = _inject_paper_guide_support_markers(
        "Evidence:\n- Closed-pinhole: CNR = 10 and open-pinhole: CNR = 14 with lower NEC.",
        support_slots=[
            {
                "doc_idx": 1,
                "claim_type": "compare_result",
                "cite_policy": "allow_none",
                "support_example": "[[SUPPORT:DOC-1]]",
                "heading_path": "Results / Figure 1",
                "snippet": "Closed pinhole yields CNR 10 while open pinhole yields CNR 14 with lower NEC.",
                "locate_anchor": "Closed pinhole yields CNR 10 while open pinhole yields CNR 14 with lower NEC.",
            }
        ],
        prompt_family="compare",
    )

    assert "[[SUPPORT:DOC-1]]" in out


def test_inject_paper_guide_support_markers_skips_method_slot_for_broad_overview_claim():
    out = _inject_paper_guide_support_markers(
        "Conclusion:\nThis paper solves a label-free live-cell imaging problem and reports its main contributions.",
        support_slots=[
            {
                "doc_idx": 1,
                "claim_type": "method_detail",
                "cite_policy": "prefer_ref",
                "support_example": "[[SUPPORT:DOC-1]]",
                "heading_path": "Methods / APR",
                "snippet": "APR was performed using image registration based on phase correlation [35].",
                "locate_anchor": "APR was performed using image registration based on phase correlation.",
            }
        ],
        prompt_family="overview",
    )

    assert "[[SUPPORT:DOC-1]]" not in out


def test_inject_paper_guide_support_markers_skips_meta_line_with_overlap_tokens():
    out = _inject_paper_guide_support_markers(
        "Based on the retrieved context, APR is discussed together with phase correlation and registration.",
        support_slots=[
            {
                "doc_idx": 1,
                "claim_type": "method_detail",
                "cite_policy": "prefer_ref",
                "support_example": "[[SUPPORT:DOC-1]]",
                "heading_path": "Methods / APR",
                "snippet": "APR was performed using image registration based on phase correlation [35].",
                "locate_anchor": "APR was performed using image registration based on phase correlation.",
            }
        ],
        prompt_family="method",
    )

    assert "[[SUPPORT:DOC-1]]" not in out


def test_inject_paper_guide_support_markers_skips_broad_figure_summary_line():
    out = _inject_paper_guide_support_markers(
        "In summary, Figure 1 walks the reader through the hardware, iPSFs, and APR gains via a panel-by-panel walkthrough.",
        support_slots=[
            {
                "doc_idx": 1,
                "claim_type": "figure_panel",
                "cite_policy": "locate_only",
                "support_example": "[[SUPPORT:DOC-1]]",
                "heading_path": "Results / Figure 1",
                "snippet": "Figure 1. a Setup. d/e open and closed pinhole confocal iSCAT. f/g APR and line profiles.",
                "locate_anchor": "Figure 1. a Setup. d/e open and closed pinhole confocal iSCAT. f/g APR and line profiles.",
            }
        ],
        prompt_family="figure_walkthrough",
    )

    assert "[[SUPPORT:DOC-1]]" not in out


def test_inject_paper_guide_support_markers_figure_walkthrough_uses_more_than_three_slots():
    out = _inject_paper_guide_support_markers(
        "\n".join(
            [
                "Panel (a): hardware setup.",
                "Panels (b) and (c): polarization-dependent iPSF.",
                "Panels (d) and (e): open vs closed pinhole.",
                "Panel (f): APR result.",
                "Panel (g): line profiles.",
            ]
        ),
        support_slots=[
            {
                "doc_idx": 1,
                "claim_type": "figure_panel",
                "cite_policy": "locate_only",
                "support_example": f"[[SUPPORT:DOC-1-S{i+1}]]",
                "heading_path": "Results / Figure 1",
                "snippet": snippet,
                "locate_anchor": snippet,
            }
            for i, snippet in enumerate(
                [
                    "a Setup.",
                    "b/c polarization-dependent iPSF.",
                    "d/e open and closed pinhole confocal iSCAT.",
                    "f Resulting iPSF from iISM after APR.",
                    "g Line profiles of the iPSF in the three configurations.",
                ]
            )
        ],
        prompt_family="figure_walkthrough",
    )

    assert "[[SUPPORT:DOC-1-S4]]" in out
    assert "[[SUPPORT:DOC-1-S5]]" in out


def test_inject_paper_guide_support_markers_figure_walkthrough_prefers_panel_slot_over_method_slot():
    out = _inject_paper_guide_support_markers(
        "Panel (f) shows the APR result.",
        support_slots=[
            {
                "doc_idx": 1,
                "support_id": "DOC-1-S1",
                "claim_type": "method_detail",
                "cite_policy": "prefer_ref",
                "support_example": "[[SUPPORT:DOC-1-S1]]",
                "heading_path": "Methods / APR",
                "snippet": "APR was performed using image registration based on phase correlation [35].",
                "locate_anchor": "APR was performed using image registration based on phase correlation.",
            },
            {
                "doc_idx": 1,
                "support_id": "DOC-1-S2",
                "claim_type": "figure_panel",
                "cite_policy": "locate_only",
                "support_example": "[[SUPPORT:DOC-1-S2]]",
                "heading_path": "Results / Figure 1",
                "snippet": "Figure 1. f Resulting iPSF from iISM after adaptive pixel-reassignment (APR). g Line profiles of the iPSF in the three configurations.",
                "locate_anchor": "f Resulting iPSF from iISM after adaptive pixel-reassignment (APR).",
            },
        ],
        prompt_family="figure_walkthrough",
    )

    assert "[[SUPPORT:DOC-1-S2]]" in out
    assert "[[SUPPORT:DOC-1-S1]]" not in out


def test_resolve_paper_guide_support_markers_drops_locate_only_marker():
    answer, resolutions = _resolve_paper_guide_support_markers(
        "Figure 1d/e compares the open and closed pinhole conditions [[SUPPORT:DOC-1]].",
        support_slots=[
            {
                "doc_idx": 1,
                "sid": "s50f9c165",
                "source_path": "demo.pdf",
                "block_id": "blk_fig1",
                "anchor_id": "anc_fig1",
                "heading_path": "Results / Figure 1",
                "locate_anchor": "Figure 1d/e compares the open and closed pinhole conditions.",
                "claim_type": "figure_panel",
                "cite_policy": "locate_only",
                "candidate_refs": [8],
                "ref_spans": [{"text": "Figure 1d/e [8]", "nums": [8], "scope": "same_sentence"}],
            }
        ],
    )

    assert "[[SUPPORT:" not in answer
    assert "[[CITE:" not in answer
    assert answer.endswith("conditions.")
    assert len(resolutions) == 1
    assert resolutions[0]["resolved_ref_num"] == 0
    assert resolutions[0]["citation_resolution_mode"] == "locate_only"


def test_resolve_paper_guide_support_ref_num_prefers_local_ref_span_over_broad_candidate():
    ref_num, mode = _resolve_paper_guide_support_ref_num(
        {
            "cite_policy": "prefer_ref",
            "candidate_refs": [32, 34],
            "ref_spans": [
                {
                    "text": "Specifically, we use the radial variance transform (RVT)[34].",
                    "nums": [34],
                    "scope": "same_sentence",
                },
                {
                    "text": "This effect is analogous to super-concentration of light [32].",
                    "nums": [32],
                    "scope": "same_sentence",
                },
            ],
        },
        context_text="The RVT step is introduced explicitly in the method paragraph.",
    )

    assert ref_num == 34
    assert mode == "slot_ref_span"


def test_resolve_paper_guide_support_ref_num_drops_meta_line_citation():
    ref_num, mode = _resolve_paper_guide_support_ref_num(
        {
            "cite_policy": "prefer_ref",
            "candidate_refs": [35],
            "ref_spans": [],
        },
        context_text="The retrieved context does not explicitly state where the shift vectors are re-applied.",
    )

    assert ref_num is None
    assert mode == "missing_local_evidence"


def test_resolve_paper_guide_support_ref_num_prefers_explicit_context_ref():
    ref_num, mode = _resolve_paper_guide_support_ref_num(
        {
            "cite_policy": "prefer_ref",
            "candidate_refs": [1, 34],
            "ref_spans": [{"text": "RVT [34]", "nums": [34], "scope": "same_sentence"}],
        },
        context_text="The paper introduces RVT as [34].",
    )

    assert ref_num == 34
    assert mode == "context_explicit_ref"


def test_resolve_paper_guide_support_markers_records_segment_index_metadata():
    answer, resolutions = _resolve_paper_guide_support_markers(
        "APR uses phase correlation to register the off-axis raw images [[SUPPORT:DOC-1]].\n\n"
        "A separate follow-up paragraph stays untouched.",
        support_slots=[
            {
                "doc_idx": 1,
                "sid": "s50f9c165",
                "source_path": "demo.pdf",
                "block_id": "blk_apr",
                "anchor_id": "anc_apr",
                "heading_path": "Methods / APR",
                "locate_anchor": "APR uses phase correlation to register the off-axis raw images.",
                "claim_type": "method_detail",
                "cite_policy": "prefer_ref",
                "candidate_refs": [35],
                "ref_spans": [{"text": "phase correlation [35]", "nums": [35], "scope": "same_sentence"}],
            }
        ],
    )

    assert answer.endswith("A separate follow-up paragraph stays untouched.")
    assert len(resolutions) == 1
    assert resolutions[0]["segment_index"] == 0
    assert resolutions[0]["segment_kind"] == "paragraph"
    assert resolutions[0]["segment_snippet_key"]


def test_resolve_paper_guide_support_markers_picks_best_same_doc_slot_from_context():
    answer, resolutions = _resolve_paper_guide_support_markers(
        "APR uses phase correlation to register the off-axis raw images [[SUPPORT:DOC-1]].\n\n"
        "The resulting shift vectors are applied back to the original iISM dataset [[SUPPORT:DOC-1]].",
        support_slots=[
            {
                "doc_idx": 1,
                "support_id": "DOC-1-S1",
                "sid": "s50f9c165",
                "source_path": "demo.pdf",
                "block_id": "blk_apr",
                "anchor_id": "anc_apr",
                "heading_path": "Methods / APR",
                "locate_anchor": "APR was performed using image registration based on phase correlation.",
                "claim_type": "method_detail",
                "cite_policy": "prefer_ref",
                "candidate_refs": [35],
                "ref_spans": [{"text": "phase correlation [35]", "nums": [35], "scope": "same_sentence"}],
            },
            {
                "doc_idx": 1,
                "support_id": "DOC-1-S2",
                "sid": "s50f9c165",
                "source_path": "demo.pdf",
                "block_id": "blk_reapply",
                "anchor_id": "anc_reapply",
                "heading_path": "Results / Figure 2",
                "locate_anchor": "Finally, these RVT-APR shift vectors were applied back to the original iISM dataset.",
                "claim_type": "method_detail",
                "cite_policy": "locate_only",
                "candidate_refs": [],
                "ref_spans": [],
            },
        ],
    )

    assert "[[CITE:s50f9c165:35]]" in answer
    assert "original iISM dataset" in answer
    assert len(resolutions) == 2
    assert resolutions[0]["block_id"] == "blk_apr"
    assert resolutions[0]["resolved_ref_num"] == 35
    assert resolutions[1]["block_id"] == "blk_reapply"
    assert resolutions[1]["citation_resolution_mode"] == "locate_only"


def test_resolve_paper_guide_support_markers_rebinds_slot_block_from_line_context(tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "## Methods\n\n"
            "### Microscope setup\n\n"
            "The microscope setup uses two lasers and a beam splitter.\n\n"
            "## Results\n\n"
            "### Figure 2\n\n"
            "Finally, these RVT-APR shift vectors were applied back to the original iISM dataset, yielding reconstructions with enhanced spatial resolution.\n"
        ),
        encoding="utf-8",
    )

    blocks = task_runtime.load_source_blocks(md_main)
    setup_block = next(
        block for block in blocks
        if str(block.get("heading_path") or "").strip().lower().endswith("microscope setup")
    )

    answer, resolutions = _resolve_paper_guide_support_markers(
        "The resulting shift vectors are applied back to the original iISM dataset [[SUPPORT:DOC-1]].",
        support_slots=[
            {
                "doc_idx": 1,
                "sid": "s50f9c165",
                "source_path": str(source_pdf),
                "block_id": str(setup_block.get("block_id") or ""),
                "anchor_id": str(setup_block.get("anchor_id") or ""),
                "heading_path": str(setup_block.get("heading_path") or ""),
                "locate_anchor": str(setup_block.get("text") or ""),
                "claim_type": "method_detail",
                "cite_policy": "locate_only",
                "candidate_refs": [],
                "ref_spans": [],
            }
        ],
        prompt_family="figure_walkthrough",
        db_dir=tmp_path,
    )

    assert "original iISM dataset" in answer
    assert len(resolutions) == 1
    assert resolutions[0]["block_id"] != str(setup_block.get("block_id") or "")
    assert "Figure 2" in str(resolutions[0]["heading_path"] or "")
    assert "applied back to the original iISM dataset" in str(resolutions[0]["locate_anchor"] or "")


def test_build_paper_guide_support_slots_prefers_local_ref_span_for_candidate_order(monkeypatch):
    from kb import task_runtime

    monkeypatch.setattr(
        task_runtime,
        "_resolve_paper_guide_support_slot_block",
        lambda **_kwargs: {
            "block_id": "blk_rvt",
            "anchor_id": "anc_rvt",
            "heading_path": "Methods / RVT",
            "locate_anchor": "Specifically, we use the radial variance transform (RVT)[34].",
        },
    )

    slots = _build_paper_guide_support_slots(
        [
            {
                "doc_idx": 1,
                "sid": "s50f9c165",
                "source_path": "demo.pdf",
                "heading": "Methods / RVT",
                "candidate_refs": [32],
                "cue": "super-concentration of light [32]",
                "snippet": "Specifically, we use the radial variance transform (RVT)[34], which converts an interferogram into an intensity-only map.",
                "deepread_texts": [],
            }
        ],
        prompt="Which prior work do the authors cite for the RVT step?",
        prompt_family="method",
        db_dir=None,
    )

    assert len(slots) == 1
    assert slots[0]["candidate_refs"][0] == 34
    assert slots[0]["cite_example"] == "[[CITE:s50f9c165:34]]"


def test_repair_paper_guide_focus_answer_generic_overrides_not_stated_for_exact_method_prompt():
    out = _repair_paper_guide_focus_answer_generic(
        (
            "The retrieved context does not explicitly state where the shift vectors are re-applied to the original iISM dataset.\n\n"
            "Therefore, based on the available evidence, this detail is not stated."
        ),
        prompt="In the APR pipeline, where do the authors say the shift vectors are re-applied to the original iISM dataset? Point me to the exact supporting part of the paper.",
        prompt_family="method",
        special_focus_block=(
            "Paper-guide method focus:\n"
            "- Focus snippet:\n"
            "Finally, these RVT-APR shift vectors were applied back to the original iISM dataset, yielding reconstructions with enhanced spatial resolution."
        ),
        source_path="",
        db_dir=None,
    )

    assert "not stated" not in out.lower()
    assert "applied back to the original iism dataset" in out.lower()


def test_drop_paper_guide_locate_only_line_citations_strips_structured_cite():
    out = _drop_paper_guide_locate_only_line_citations(
        "Figure 2 overview [[CITE:s50f9c165:25]].\n\n"
        "Implementation detail: APR uses phase correlation [[CITE:s50f9c165:35]].",
        support_resolution=[
            {
                "line_index": 0,
                "cite_policy": "locate_only",
                "resolved_ref_num": 0,
            }
        ],
    )

    assert "[[CITE:s50f9c165:25]]" not in out
    assert "[[CITE:s50f9c165:35]]" in out


def test_build_paper_guide_direct_abstract_answer_uses_extracted_span_and_translation(monkeypatch):
    from kb import task_runtime

    class FakeLLM:
        def chat(self, *, messages, temperature, max_tokens):
            assert temperature == 0.0
            assert max_tokens >= 1000
            assert "Light microscopy remains indispensable" in str(messages[-1]["content"])
            return "这里给出忠实中文翻译。"

    monkeypatch.setattr(
        task_runtime,
        "_extract_bound_paper_abstract",
        lambda source_path, db_dir=None: "Light microscopy remains indispensable in life sciences.",
    )
    out = _build_paper_guide_direct_abstract_answer(
        prompt="Give the abstract text and translate it into Chinese.",
        source_path="demo.pdf",
        db_dir=Path("."),
        llm=FakeLLM(),
    )
    assert "Abstract text" in out
    assert "Light microscopy remains indispensable in life sciences." in out
    assert "这里给出忠实中文翻译。" in out


def test_build_paper_guide_direct_abstract_answer_uses_clean_chinese_titles(monkeypatch):
    from kb import task_runtime
    from kb.task_runtime import _build_paper_guide_direct_abstract_answer

    class FakeLLM:
        def chat(self, messages, temperature=0.0, max_tokens=0):
            assert "Light microscopy remains indispensable" in str(messages[-1]["content"])
            return "这里给出忠实中文翻译。"

    monkeypatch.setattr(
        task_runtime,
        "_extract_bound_paper_abstract",
        lambda source_path, db_dir=None: "Light microscopy remains indispensable in life sciences.",
    )
    out = _build_paper_guide_direct_abstract_answer(
        prompt="给出摘要原文并翻译成中文。",
        source_path="demo.pdf",
        db_dir=Path("."),
        llm=FakeLLM(),
    )
    assert "摘要原文" in out
    assert "中文翻译" in out
    assert "这里给出忠实中文翻译。" in out


def test_maybe_append_library_figure_markdown_uses_clean_heading(monkeypatch):
    from kb import task_runtime

    monkeypatch.setattr(task_runtime, "_requested_figure_number", lambda prompt, hits: 1)
    monkeypatch.setattr(
        task_runtime,
        "_build_doc_figure_card",
        lambda source_path, figure_num: {
            "source_name": "demo.pdf",
            "figure_num": figure_num,
            "url": "/api/references/asset?path=demo",
            "label": "Figure 1 caption",
        },
    )
    monkeypatch.setattr(task_runtime, "_score_figure_card_source_binding", lambda **kwargs: 1.0)

    out = _maybe_append_library_figure_markdown(
        "Figure 1 explanation.",
        prompt="Explain Figure 1.",
        answer_hits=[{"meta": {"source_path": "demo.pdf"}}],
    )
    assert "### Library Figure" in out
    assert "\ufffd" not in out


def test_extract_bound_paper_method_focus_prefers_methods_block_with_phase_correlation(tmp_path: Path):
    md = tmp_path / "paper.en.md"
    md.write_text(
        "# Results\n\n"
        "## iISM with adaptive pixel-reassignment (APR)\n\n"
        "APR is the key innovation that improves coherent reconstruction and CNR.\n\n"
        "# Materials and Methods\n\n"
        "## Adaptive pixel-reassignment (APR)\n\n"
        "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one. The resulting shift vectors were applied to the original stack.\n",
        encoding="utf-8",
    )
    out = _extract_bound_paper_method_focus(str(md), db_dir=tmp_path, focus_terms=["APR"])
    assert "phase correlation" in out
    assert "image registration" in out
    assert "improves coherent reconstruction and CNR" not in out


def test_extract_bound_paper_figure_caption_returns_matching_caption(tmp_path: Path):
    md = tmp_path / "paper.en.md"
    md.write_text(
        "# Results\n\n"
        "![Figure 1](./assets/page_3_fig_1.png)\n\n"
        "Figure 1. a, Setup. d/e, Open and closed pinhole confocal iSCAT. f/g, APR and line profiles.\n\n"
        "More text.\n",
        encoding="utf-8",
    )
    out = _extract_bound_paper_figure_caption(str(md), figure_num=1, db_dir=tmp_path)
    assert out.startswith("Figure 1.")
    assert "d/e, Open and closed pinhole confocal iSCAT" in out
    assert "f/g, APR and line profiles" in out


def test_extract_bound_paper_figure_caption_prefers_explicit_caption_over_body_mentions(tmp_path: Path):
    md = tmp_path / "paper.en.md"
    md.write_text(
        "# Results\n\n"
        "The custom microscope setup is shown in Fig. 1a.\n\n"
        "Figure 1. a, Setup. d/e, Open and closed pinhole confocal iSCAT. "
        "f Resulting iPSF from iISM after APR. g Line profiles of the iPSF in the three configurations.\n\n"
        "Figure 1d, e compare the open and closed pinhole configurations.\n",
        encoding="utf-8",
    )
    out = _extract_bound_paper_figure_caption(str(md), figure_num=1, db_dir=tmp_path)
    assert out.startswith("Figure 1.")
    assert "f Resulting iPSF from iISM after APR" in out
    assert "Line profiles of the iPSF in the three configurations" in out
    assert "custom microscope setup is shown in Fig. 1a" not in out


def test_extract_caption_focus_fragment_prefers_missing_panel_clause_pair():
    excerpt = (
        "Figure 1. a Setup. d Open pinhole confocal iSCAT. e Closed pinhole confocal iSCAT. "
        "f Resulting iPSF from iISM after adaptive pixel-reassignment (APR). "
        "g Line profiles of the iPSF in the three configurations."
    )
    answer_text = (
        "Figure 1 demonstrates the method. "
        "Panel (a) shows the setup. "
        "Panels (d) and (e) compare open and closed pinhole. "
        "Panel (g) shows the quantitative line-profile comparison."
    )
    out = _extract_caption_focus_fragment(excerpt, answer_text=answer_text)
    assert "f Resulting iPSF from iISM after adaptive pixel-reassignment (APR)" in out
    assert "g Line profiles of the iPSF in the three configurations" in out


def test_extract_caption_prompt_fragment_targets_requested_panels():
    excerpt = (
        "Figure 1. a Setup. d Open pinhole confocal iSCAT. e Closed pinhole confocal iSCAT. "
        "f Resulting iPSF from iISM after adaptive pixel-reassignment (APR). "
        "g Line profiles of the iPSF in the three configurations."
    )
    out = _extract_caption_prompt_fragment(
        excerpt,
        prompt="For Figure 1 panels (f) and (g), what exactly does each panel show?",
    )
    assert "f Resulting iPSF from iISM after adaptive pixel-reassignment (APR)" in out
    assert "g Line profiles of the iPSF in the three configurations" in out
    assert "d Open pinhole confocal iSCAT" not in out


def test_extract_paper_guide_method_detail_excerpt_prefers_applied_back_sentence():
    excerpt = (
        "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one, as detailed in [35]. "
        "Finally, these RVT-APR shift vectors were applied back to the original iISM dataset, yielding reconstructions with enhanced spatial resolution."
    )
    out = _extract_paper_guide_method_detail_excerpt(
        excerpt,
        focus_terms=["shift vectors", "applied back", "original iism dataset"],
    )
    assert "applied back to the original iISM dataset" in out


def test_extract_paper_guide_method_detail_excerpt_prefers_original_iism_stack_sentence():
    excerpt = (
        "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one, as detailed in [35]. "
        "Image registration of the RVT pinhole stack yielded shift vectors for each off-axis pinhole image relative to the central one. "
        "The obtained shift vectors were then applied to the original iISM pinhole stack, enabling pixel reassignment."
    )
    out = _extract_paper_guide_method_detail_excerpt(
        excerpt,
        focus_terms=["shift vectors", "original iism dataset", "applied back"],
    )
    assert "applied to the original iISM pinhole stack" in out


def test_build_paper_guide_special_focus_block_prefers_apr_implementation_span():
    out = _build_paper_guide_special_focus_block(
        [
            {
                "snippet": "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one.",
                "deepread_texts": [],
            }
        ],
        prompt="Explain how this method works, especially what APR does in the pipeline and why it matters.",
        prompt_family="method",
        source_path="demo.pdf",
        db_dir=Path("."),
        answer_hits=[],
    )
    assert "Paper-guide method focus" in out
    assert "phase correlation" in out


def test_build_paper_guide_special_focus_block_ignores_broad_intro_when_named_submodule_is_present():
    out = _build_paper_guide_special_focus_block(
        [
            {
                "snippet": "Recent work has extended iSCAT to image intracellular structures in live cells using a combination and detection scheme.",
                "deepread_texts": [],
            },
            {
                "snippet": "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one.",
                "deepread_texts": [],
            },
        ],
        prompt="Explain how this method works, especially what APR does in the pipeline and why it matters.",
        prompt_family="method",
        source_path="demo.pdf",
        db_dir=Path("."),
        answer_hits=[],
    )
    assert "phase correlation" in out
    assert "Recent work has extended iSCAT" not in out


def test_build_paper_guide_special_focus_block_prefers_strong_detail_cue_over_broad_apr_summary():
    out = _build_paper_guide_special_focus_block(
        [
            {
                "heading": "Results / Overview",
                "snippet": "APR is the key innovation that enables coherent reconstruction and improved CNR.",
                "deepread_texts": [],
            },
            {
                "heading": "Methods / APR",
                "snippet": "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one.",
                "deepread_texts": [],
            },
        ],
        prompt="Explain how this method works, especially what APR does in the pipeline and why it matters.",
        prompt_family="method",
        source_path="demo.pdf",
        db_dir=Path("."),
        answer_hits=[],
    )
    assert "phase correlation" in out
    assert "improved CNR" not in out


def test_build_paper_guide_special_focus_block_falls_back_to_bound_method_focus_when_hit_only_has_broad_summary(monkeypatch):
    from kb import task_runtime

    monkeypatch.setattr(
        task_runtime,
        "_extract_bound_paper_method_focus",
        lambda source_path, db_dir=None, focus_terms=None: "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one.",
    )
    out = _build_paper_guide_special_focus_block(
        [
            {
                "heading": "Results / Overview",
                "snippet": "APR is the key innovation that enables coherent reconstruction and improved CNR.",
                "deepread_texts": [],
            }
        ],
        prompt="Explain how this method works, especially what APR does in the pipeline and why it matters.",
        prompt_family="method",
        source_path="demo.pdf",
        db_dir=Path("."),
        answer_hits=[],
    )
    assert "phase correlation" in out
    assert "improved CNR" not in out


def test_build_paper_guide_special_focus_block_uses_figure_caption(monkeypatch):
    from kb import task_runtime

    monkeypatch.setattr(
        task_runtime,
        "_extract_bound_paper_figure_caption",
        lambda source_path, figure_num, db_dir=None: "Figure 1. a, Setup. d/e, Open and closed pinhole confocal iSCAT. f/g, APR and line profiles.",
    )
    out = _build_paper_guide_special_focus_block(
        [],
        prompt="Walk me through what Figure 1 demonstrates.",
        prompt_family="figure_walkthrough",
        source_path="demo.pdf",
        db_dir=Path("."),
        answer_hits=[],
    )
    assert "Paper-guide figure focus" in out
    assert "Requested figure: Figure 1" in out
    assert "f/g, APR and line profiles." in out


def test_build_paper_guide_special_focus_block_falls_back_to_bound_method_excerpt(tmp_path: Path):
    md = tmp_path / "paper.en.md"
    md.write_text(
        "# Materials and Methods\n\n"
        "## Adaptive pixel-reassignment (APR)\n\n"
        "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one.\n",
        encoding="utf-8",
    )
    out = _build_paper_guide_special_focus_block(
        [],
        prompt="Explain how this method works, especially what APR does in the pipeline and why it matters.",
        prompt_family="method",
        source_path=str(md),
        db_dir=tmp_path,
        answer_hits=[],
    )
    assert "Paper-guide method focus" in out
    assert "phase correlation" in out
    assert "image registration" in out


def test_repair_paper_guide_focus_answer_injects_missing_apr_implementation_detail():
    out = _repair_paper_guide_focus_answer(
        "Conclusion:\nAPR restores resolution.\n\nEvidence:\n- It is tailored to coherent detection.",
        prompt="Explain how this method works, especially what APR does in the pipeline and why it matters.",
        prompt_family="method",
        special_focus_block=(
            "Paper-guide method focus:\n"
            "- Focus snippet:\n"
            "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one."
        ),
    )
    assert "Implementation detail:" in out
    assert "phase correlation" in out
    assert "image registration" in out


def test_repair_paper_guide_focus_answer_generic_injects_method_detail_from_focus_excerpt():
    out = _repair_paper_guide_focus_answer_generic(
        "Conclusion:\nAPR restores resolution.\n\nEvidence:\n- It is tailored to coherent detection.",
        prompt="Explain how this method works, especially what APR does in the pipeline and why it matters.",
        prompt_family="method",
        special_focus_block=(
            "Paper-guide method focus:\n"
            "- Focus snippet:\n"
            "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one."
        ),
    )
    assert "Implementation detail:" in out
    assert "phase correlation" in out
    assert "image registration" in out


def test_repair_paper_guide_focus_answer_generic_prefers_bound_method_detail_over_broad_focus_excerpt(monkeypatch):
    from kb import task_runtime

    monkeypatch.setattr(
        task_runtime,
        "_extract_bound_paper_method_focus",
        lambda source_path, db_dir=None, focus_terms=None: (
            "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one, as detailed in [35]."
        ),
    )
    out = _repair_paper_guide_focus_answer_generic(
        "Conclusion:\nAPR restores resolution.\n\nEvidence:\n- APR is tailored to coherent detection.",
        prompt="Explain how this method works, especially what APR does in the pipeline and why it matters.",
        prompt_family="method",
        special_focus_block=(
            "Paper-guide method focus:\n"
            "- Focus snippet:\n"
            "APR is the key innovation that enables coherent reconstruction and improved CNR."
        ),
        source_path="demo.pdf",
        db_dir=Path("."),
    )
    assert "Implementation detail:" in out
    assert "phase correlation" in out
    assert "image registration" in out
    assert "improved CNR" not in out


def test_repair_paper_guide_focus_answer_adds_missing_figure_fg_anchor():
    out = _repair_paper_guide_focus_answer(
        "Conclusion: Figure 1 shows the trade-off between confocal iSCAT and iISM.\n\n"
        "Evidence:\n"
        "- Panel (g): line profiles quantify the width and CNR differences.",
        prompt="Walk me through what Figure 1 demonstrates.",
        prompt_family="figure_walkthrough",
        special_focus_block=(
            "Paper-guide figure focus:\n"
            "- Caption excerpt:\n"
            "Figure 1. d/e, Open and closed pinhole confocal iSCAT. f/g, APR and line profiles."
        ),
    )
    assert "Caption anchor: Panels (f) and (g) are the APR result and the corresponding line profiles." in out


def test_repair_paper_guide_focus_answer_generic_adds_caption_fragment_for_missing_panels():
    out = _repair_paper_guide_focus_answer_generic(
        "Conclusion: Figure 1 shows the trade-off between confocal iSCAT and iISM.\n\n"
        "Evidence:\n"
        "- Panel (g): line profiles quantify the width and CNR differences.",
        prompt="Walk me through what Figure 1 demonstrates.",
        prompt_family="figure_walkthrough",
        special_focus_block=(
            "Paper-guide figure focus:\n"
            "- Caption excerpt:\n"
            "Figure 1. d/e, Open and closed pinhole confocal iSCAT. f/g, APR and line profiles."
        ),
    )
    assert "Caption anchor: d/e, Open and closed pinhole confocal iSCAT." in out or "Caption anchor: f/g, APR and line profiles." in out


def test_inject_paper_guide_fallback_citations_adds_marker_for_cue_overlap():
    out = _inject_paper_guide_fallback_citations(
        "Conclusion: APR is the key method step.\n\n"
        "Evidence:\n"
        "- Uses phase correlation based image registration between each off-axis raw image and the central image.",
        cards=[
            {
                "sid": "s50f9c165",
                "candidate_refs": [24],
                "cue": "APR was performed using image registration based on phase correlation [24].",
            }
        ],
        prompt_family="method",
    )
    assert "[[CITE:s50f9c165:24]]" in out


def test_inject_paper_guide_card_citations_uses_heading_and_short_tokens():
    out = _inject_paper_guide_card_citations(
        "Conclusion: APR improves CNR.\n\nEvidence:\n- APR improves CNR in the reassigned image.",
        cards=[
            {
                "sid": "s50f9c165",
                "candidate_refs": [24],
                "heading": "Adaptive pixel-reassignment (APR)",
                "snippet": "APR improves CNR in the reassigned image.",
            }
        ],
        prompt_family="method",
    )
    assert "[[CITE:s50f9c165:24]]" in out


def test_inject_paper_guide_card_citations_uses_snippet_when_cue_is_empty():
    out = _inject_paper_guide_card_citations(
        "Conclusion: The method uses image registration.\n\nEvidence:\n- It uses phase correlation based image registration between detector images.",
        cards=[
            {
                "sid": "s50f9c165",
                "candidate_refs": [35],
                "heading": "Methods / Alignment",
                "snippet": "Phase correlation based image registration is applied between detector images before reassignment.",
            }
        ],
        prompt_family="method",
    )
    assert "[[CITE:s50f9c165:35]]" in out


def test_inject_paper_guide_card_citations_skips_when_answer_already_has_numeric_ref():
    out = _inject_paper_guide_card_citations(
        "The paper introduces RVT as [34].",
        cards=[
            {
                "sid": "s50f9c165",
                "candidate_refs": [1],
                "heading": "Methods / RVT",
                "snippet": "Specifically, we use the radial variance transform (RVT) [34].",
            }
        ],
        prompt_family="method",
    )
    assert out == "The paper introduces RVT as [34]."


def test_inject_paper_guide_card_citations_skips_compare_family():
    out = _inject_paper_guide_card_citations(
        "APR improves CNR relative to the confocal baseline.",
        cards=[
            {
                "sid": "s50f9c165",
                "candidate_refs": [24],
                "heading": "Results / Comparison",
                "snippet": "APR improves CNR relative to the confocal baseline [24].",
            }
        ],
        prompt_family="compare",
    )
    assert out == "APR improves CNR relative to the confocal baseline."


def test_collect_paper_guide_candidate_refs_by_source_merges_focus_excerpt_refs():
    out = _collect_paper_guide_candidate_refs_by_source(
        [
            {
                "source_path": r"db\doc\paper.en.md",
                "candidate_refs": [],
                "snippet": "APR aligns the interferometric stack.",
            }
        ],
        focus_source_path=r"db\doc\paper.en.md",
        special_focus_block=(
            "Paper-guide method focus:\n"
            "- Focus snippet:\n"
            "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one, as detailed in [35]."
        ),
    )
    assert out == {r"db\doc\paper.en.md": [35]}


def test_collect_paper_guide_candidate_refs_by_source_falls_back_to_bound_method_focus(monkeypatch):
    from kb import task_runtime

    monkeypatch.setattr(
        task_runtime,
        "_extract_bound_paper_method_focus",
        lambda source_path, db_dir=None, focus_terms=None: (
            "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one, as detailed in [35]."
        ),
    )
    out = _collect_paper_guide_candidate_refs_by_source(
        [
            {
                "source_path": r"db\doc\paper.en.md",
                "candidate_refs": [],
                "snippet": "APR improves coherent reconstruction.",
            }
        ],
        focus_source_path=r"db\doc\paper.en.md",
        special_focus_block=(
            "Paper-guide method focus:\n"
            "- Focus snippet:\n"
            "APR is the key innovation that improves CNR."
        ),
        prompt_family="method",
        prompt="Explain how this method works, especially what APR does in the pipeline and why it matters.",
        db_dir=Path("."),
    )
    assert out == {r"db\doc\paper.en.md": [35]}


def test_inject_paper_guide_focus_citations_rewrites_inline_focus_ref():
    out = _inject_paper_guide_focus_citations(
        "Conclusion:\nAPR improves alignment.\n\nEvidence:\n- Implementation detail: APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one, as detailed in [35].",
        special_focus_block=(
            "Paper-guide method focus:\n"
            "- Focus snippet:\n"
            "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one, as detailed in [35]."
        ),
        source_path=r"db\doc\paper.en.md",
        prompt_family="method",
    )
    assert "[35]" not in out
    assert "[[CITE:" in out
    assert "[[CITE:" in out.split("Implementation detail:", 1)[-1]


def test_inject_paper_guide_focus_citations_falls_back_to_bound_method_focus(monkeypatch):
    from kb import task_runtime

    monkeypatch.setattr(
        task_runtime,
        "_extract_bound_paper_method_focus",
        lambda source_path, db_dir=None, focus_terms=None: (
            "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one, as detailed in [35]."
        ),
    )
    out = _inject_paper_guide_focus_citations(
        "Conclusion:\nAPR improves alignment.\n\nEvidence:\n- Implementation detail: APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one, as detailed in [35].",
        special_focus_block=(
            "Paper-guide method focus:\n"
            "- Focus snippet:\n"
            "APR is the key innovation that improves CNR."
        ),
        source_path=r"db\doc\paper.en.md",
        prompt_family="method",
        prompt="Explain how this method works, especially what APR does in the pipeline and why it matters.",
        db_dir=Path("."),
    )
    assert "[35]" not in out
    assert "[[CITE:" in out


def test_inject_paper_guide_focus_citations_skips_citation_lookup_prompt():
    out = _inject_paper_guide_focus_citations(
        "The paper attributes RVT to prior work cited as [34].",
        special_focus_block=(
            "Paper-guide method focus:\n"
            "- Focus snippet:\n"
            "Specifically, we use the radial variance transform (RVT)[34], which converts an interferogram into an intensity-only map."
        ),
        source_path=r"db\\doc\\paper.en.md",
        prompt_family="method",
        prompt="Which prior work is RVT attributed to in this paper, and what in-paper citation do they use when introducing it?",
    )
    assert "[[CITE:" not in out
    assert "[34]" in out


def test_inject_paper_guide_focus_citations_skips_non_method_families():
    out = _inject_paper_guide_focus_citations(
        "The method achieves higher CNR and sectioning.",
        special_focus_block=(
            "Paper-guide compare focus:\n"
            "- Focus snippet:\n"
            "APR improves CNR and optical sectioning [24]."
        ),
        source_path=r"db\\doc\\paper.en.md",
        prompt_family="compare",
        prompt="Compare APR with the confocal baseline.",
    )
    assert out == "The method achieves higher CNR and sectioning."


def test_inject_paper_guide_fallback_citations_skips_when_overlap_is_weak():
    out = _inject_paper_guide_fallback_citations(
        "Conclusion: The paper uses interferometric detection.\n\nEvidence:\n- Live-cell imaging is demonstrated.",
        cards=[
            {
                "sid": "s50f9c165",
                "candidate_refs": [24],
                "cue": "APR was performed using image registration based on phase correlation [24].",
            }
        ],
        prompt_family="overview",
    )
    assert "[[CITE:" not in out


def test_sanitize_paper_guide_answer_for_user_strips_internal_policy_language():
    raw = (
        "未命中知识库片段。\n\n"
        "当前检索到的原文证据不足。\n\n"
        "根据规则第 2 条与第 3 条，不得编造未出现的原文内容。\n\n"
        "若您能提供以下任一内容，我可以继续。\n\n"
        "结论：文中明确提到使用 APR 来适配 iISM。"
    )
    out = _sanitize_paper_guide_answer_for_user(raw, has_hits=False)
    assert "根据规则第" not in out
    assert "不得编造" not in out
    assert "若您能提供以下任一内容" not in out
    assert "当前检索到的原文证据不足" in out
    assert "结论：文中明确提到使用 APR 来适配 iISM。" in out
