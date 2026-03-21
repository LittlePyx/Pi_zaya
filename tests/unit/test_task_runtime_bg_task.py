from pathlib import Path

from kb.chat_store import ChatStore
from kb.task_runtime import (
    _apply_bound_source_hints,
    _augment_paper_guide_retrieval_prompt,
    _augment_prompt_with_source_hint,
    _build_bg_task,
    _filter_history_for_multimodal_turn,
    _has_anchor_grounded_answer_hits,
    _needs_conversational_source_hint,
    _paper_guide_prompt_family,
    _pick_recent_source_hint,
    _sanitize_paper_guide_answer_for_user,
    _select_paper_guide_deepread_extras,
    _select_paper_guide_answer_hits,
    _stabilize_paper_guide_output_mode,
)


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


def test_paper_guide_prompt_family_detects_chinese_overview_method_and_abstract_requests():
    assert _paper_guide_prompt_family("这篇文章讲了什么") == "overview"
    assert _paper_guide_prompt_family("这个方法具体介绍一下") == "method"
    assert _paper_guide_prompt_family("把摘要原文给出并翻译") == "abstract"


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
