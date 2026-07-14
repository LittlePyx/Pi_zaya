from __future__ import annotations

from kb.task_runtime import (
    _align_async_refs_to_finalized_selected_sources,
    _align_refs_hits_to_source_order,
    _align_multi_paper_doc_list_contract_with_display_hits,
    _citation_plan_system_a_source_paths,
    _filter_reference_rows_to_allowed_sources,
    _merge_refs_display_docs_with_answer_hits,
    _selected_research_context_source_paths,
    _trim_multi_paper_answer_to_planned_sources,
)


def _hit(source_path: str, text: str) -> dict:
    return {"text": text, "meta": {"source_path": source_path, "ref_pack_state": "pending"}}


def test_multi_paper_plan_filters_unplanned_optional_recommendation_everywhere():
    planned = ["db/review.en.md", "db/compare.en.md", "db/deep-learning.en.md"]
    hits = [
        _hit("db/deep-learning.en.md", "deep learning"),
        _hit("db/compare.en.md", "comparison"),
        _hit("db/review.en.md", "review"),
        _hit("db/other.en.md", "other"),
        _hit("db/realtime-video.en.md", "realtime video"),
    ]
    answer = (
        "先读综述 [3]，再读方法对比 [2]，最后读深度学习 [1]。\n\n"
        "如果关注实时成像，重点看 [5] 中的三维视频方案。"
    )
    plan = {
        "slots": [
            {"preferred_system": "system_a", "source_path": path}
            for path in planned
        ]
    }

    plan_paths = _citation_plan_system_a_source_paths(plan)
    trimmed = _trim_multi_paper_answer_to_planned_sources(
        answer,
        answer_hits=hits,
        planned_source_paths=plan_paths,
    )
    filtered_hits = _filter_reference_rows_to_allowed_sources(hits, plan_paths)
    filtered_doc_list = _filter_reference_rows_to_allowed_sources(
        [
            {"source_path": path, "source_name": path.rsplit("/", 1)[-1]}
            for path in [*planned, "db/realtime-video.en.md"]
        ],
        plan_paths,
    )

    assert plan_paths == planned
    assert "[5]" not in trimmed
    assert "实时成像" not in trimmed
    assert all(marker in trimmed for marker in ("[1]", "[2]", "[3]"))
    assert [item["meta"]["source_path"] for item in filtered_hits] == [
        "db/deep-learning.en.md",
        "db/compare.en.md",
        "db/review.en.md",
    ]
    assert [item["source_path"] for item in filtered_doc_list] == planned


def test_refs_display_docs_prefer_answer_sources_and_deduplicate():
    merged = _merge_refs_display_docs_with_answer_hits(
        refs_seed_docs=[_hit("db/other.en.md", "other"), _hit("db/answer.en.md", "stale")],
        answer_hits=[_hit("db/answer.en.md", "answer"), _hit("db/second.en.md", "second")],
        limit=3,
    )

    assert [item["meta"]["source_path"] for item in merged] == [
        "db/answer.en.md",
        "db/second.en.md",
        "db/other.en.md",
    ]
    assert all(item["meta"]["ref_pack_state"] == "ready" for item in merged)
    assert [item["meta"].get("ref_display_reason", "") for item in merged] == [
        "answer_hit_top",
        "answer_hit_top",
        "",
    ]


def test_refs_display_docs_use_only_cited_answer_sources_when_answer_has_citations():
    merged = _merge_refs_display_docs_with_answer_hits(
        refs_seed_docs=[_hit("db/seed.en.md", "seed")],
        answer_hits=[
            _hit("db/first.en.md", "first"),
            _hit("db/second.en.md", "second"),
            _hit("db/third.en.md", "third"),
        ],
        limit=4,
        answer="Only [1] and [3] are cited.",
    )

    assert [item["meta"]["source_path"] for item in merged] == [
        "db/first.en.md",
        "db/third.en.md",
    ]
    assert [item["meta"]["ref_answer_citation_num"] for item in merged] == [1, 3]


def test_refs_display_docs_preserve_all_non_contiguous_sources_selected_by_answer():
    answer_hits = [_hit(f"db/paper-{idx}.en.md", f"paper {idx}") for idx in range(1, 7)]

    merged = _merge_refs_display_docs_with_answer_hits(
        refs_seed_docs=answer_hits[:4],
        answer_hits=answer_hits,
        limit=4,
        answer="Route uses [2], [6], [3], and [5].",
    )

    assert [item["meta"]["source_path"] for item in merged] == [
        "db/paper-2.en.md",
        "db/paper-6.en.md",
        "db/paper-3.en.md",
        "db/paper-5.en.md",
    ]
    assert [item["meta"]["ref_answer_citation_num"] for item in merged] == [2, 6, 3, 5]


def test_selected_context_comparison_keeps_only_cited_allowed_docs_in_answer_order():
    scigs = "db/ICIP-2025-SCIGS/SCIGS.en.md"
    scinerf = "db/CVPR-2024-SCINeRF/SCINeRF.en.md"
    denoising = "db/VCIBA-2019-Denoising/Denoising.en.md"
    refs_seed_docs = [
        {
            "text": "Rich SCIGS evidence",
            "meta": {
                "source_path": scigs,
                "doi": "10.1109/ICIP.2025.123",
                "ref_locs": [{"heading_path": "Abstract", "page_start": 1}],
            },
        },
        {
            "text": "Rich SCINeRF evidence",
            "meta": {
                "source_path": scinerf.replace("/", "\\"),
                "year": "2024",
                "ref_locs": [{"heading_path": "Abstract", "page_start": 1}],
            },
        },
        _hit(denoising, "Unrelated denoising review"),
    ]
    answer_hits = [
        _hit(scigs, "Selected basket SCIGS title"),
        _hit(scinerf, "Selected basket SCINeRF title"),
        _hit(scinerf, "SCINeRF abstract evidence"),
        _hit(scigs, "SCIGS dynamic 3D evidence"),
        _hit(denoising, "Unrelated evidence"),
    ]

    merged = _merge_refs_display_docs_with_answer_hits(
        refs_seed_docs=refs_seed_docs,
        answer_hits=answer_hits,
        limit=4,
        answer="SCIGS supports dynamic 3D [4]; SCINeRF uses NeRF [3]. Ignore [5].",
        allowed_source_paths=[scigs.replace("/", "\\"), scinerf],
    )

    assert [item["meta"]["source_path"].replace("\\", "/") for item in merged] == [
        scigs,
        scinerf,
    ]
    assert [item["meta"]["ref_answer_citation_num"] for item in merged] == [4, 3]
    assert merged[0]["meta"]["doi"] == "10.1109/ICIP.2025.123"
    assert merged[1]["meta"]["year"] == "2024"
    assert all(item["meta"]["ref_locs"][0]["heading_path"] == "Abstract" for item in merged)


def test_selected_context_matches_library_pdf_to_original_markdown_and_keeps_both_paths():
    library_pdf = r"F:\papers\SCIGS.pdf"
    source_md = "F:/markdown/SCIGS.en.md"
    extra_md = "F:/markdown/Denoising.en.md"
    selected_paths = _selected_research_context_source_paths(
        [
            {
                "libraryMatchPath": library_pdf,
                "libraryMatchStatus": "in_library",
                "sourcePath": source_md,
            }
        ]
    )

    assert selected_paths == [library_pdf, source_md]

    merged = _merge_refs_display_docs_with_answer_hits(
        refs_seed_docs=[
            {
                "text": "Rich library record",
                "meta": {"source_path": library_pdf, "doi": "10.1109/SCIGS.2025"},
            },
            _hit(extra_md, "Unrelated review"),
        ],
        answer_hits=[
            _hit(source_md, "SCIGS dynamic 3D evidence"),
            _hit(extra_md, "Denoising evidence"),
        ],
        limit=4,
        answer="SCIGS reconstructs dynamic 3D scenes [1]; ignore unrelated [2].",
        allowed_source_paths=selected_paths,
    )

    assert len(merged) == 1
    assert merged[0]["meta"]["source_path"] == library_pdf
    assert merged[0]["meta"]["doi"] == "10.1109/SCIGS.2025"
    assert merged[0]["meta"]["ref_answer_citation_num"] == 1


def test_selected_context_no_citation_fallback_never_reintroduces_extra_docs():
    library_pdf = "F:/papers/SCIGS.pdf"
    source_md = "F:/markdown/SCIGS.en.md"
    extra_md = "F:/markdown/Denoising.en.md"

    merged = _merge_refs_display_docs_with_answer_hits(
        refs_seed_docs=[
            _hit(extra_md, "Unrelated seed"),
            {"text": "Selected seed", "meta": {"source_path": library_pdf, "year": "2025"}},
        ],
        answer_hits=[
            _hit(extra_md, "Unrelated answer hit"),
            _hit(source_md, "Selected answer hit"),
        ],
        limit=4,
        answer="A plain answer without numeric citations.",
        allowed_source_paths=[library_pdf, source_md],
    )

    assert len(merged) == 1
    assert merged[0]["meta"]["source_path"] == library_pdf
    assert merged[0]["meta"]["year"] == "2025"


def test_async_refs_alignment_filters_and_reorders_without_losing_enriched_metadata():
    hits = [
        {"text": "extra", "meta": {"source_path": "db/extra.en.md", "doi": "extra"}},
        {"text": "second", "meta": {"source_path": r"db\second.en.md", "doi": "second"}},
        {"text": "first", "meta": {"source_path": "db/first.en.md", "doi": "first"}},
    ]

    aligned = _align_refs_hits_to_source_order(
        hits,
        [r"db\first.en.md", "db/second.en.md"],
    )

    assert [item["meta"]["doi"] for item in aligned] == ["first", "second"]


def test_selected_context_async_refs_wait_for_finalization_then_filter_pdf_md_aliases():
    enriched = [
        {"text": "extra", "meta": {"source_path": "F:/md/Extra.en.md", "doi": "extra"}},
        {"text": "scigs", "meta": {"source_path": "F:/md/SCIGS.en.md", "doi": "scigs"}},
    ]

    before_final, can_persist_before = _align_async_refs_to_finalized_selected_sources(
        enriched,
        {"refs_final_source_paths": ["F:/pdf/SCIGS.pdf"]},
        selected_context_guarded=True,
    )
    after_final, can_persist_after = _align_async_refs_to_finalized_selected_sources(
        enriched,
        {
            "refs_final_source_paths": ["F:/pdf/SCIGS.pdf"],
            "refs_final_source_paths_finalized": True,
        },
        selected_context_guarded=True,
    )

    assert before_final == []
    assert can_persist_before is False
    assert [item["meta"]["doi"] for item in after_final] == ["scigs"]
    assert can_persist_after is True


def test_selected_context_async_refs_treat_final_empty_set_as_authoritative():
    aligned, can_persist = _align_async_refs_to_finalized_selected_sources(
        [{"text": "extra", "meta": {"source_path": "F:/md/Extra.en.md"}}],
        {
            "refs_final_source_paths": [],
            "refs_final_source_paths_finalized": True,
        },
        selected_context_guarded=True,
    )

    assert aligned == []
    assert can_persist is False


def test_async_refs_without_selected_context_keep_previous_unfinalized_behavior():
    enriched = [
        {"text": "first", "meta": {"source_path": "db/first.en.md"}},
        {"text": "second", "meta": {"source_path": "db/second.en.md"}},
    ]

    aligned, can_persist = _align_async_refs_to_finalized_selected_sources(
        enriched,
        {},
        selected_context_guarded=False,
    )

    assert [item["meta"]["source_path"] for item in aligned] == [
        "db/first.en.md",
        "db/second.en.md",
    ]
    assert can_persist is True


def test_three_cited_papers_do_not_expand_back_to_six_doc_list_cards():
    answer_hits = [_hit(f"db/paper-{idx}.en.md", f"paper {idx} evidence") for idx in range(1, 7)]
    display_hits = _merge_refs_display_docs_with_answer_hits(
        refs_seed_docs=answer_hits,
        answer_hits=answer_hits,
        limit=6,
        answer="Read the overview [3], then the comparison [2], and finish with the review [1].",
    )
    doc_list = [
        {
            "source_path": f"db/paper-{idx}.en.md",
            "summary_line": f"Rich summary {idx}",
            "citation_num": idx,
        }
        for idx in range(1, 7)
    ]

    aligned = _align_multi_paper_doc_list_contract_with_display_hits(
        prompt="Which papers should I read first?",
        doc_list=doc_list,
        display_hits=display_hits,
        evidence_cards=[],
    )

    assert [item["source_path"] for item in aligned] == [
        "db/paper-3.en.md",
        "db/paper-2.en.md",
        "db/paper-1.en.md",
    ]
    assert [item["citation_num"] for item in aligned] == [3, 2, 1]
    assert [item["summary_line"] for item in aligned] == [
        "Rich summary 3",
        "Rich summary 2",
        "Rich summary 1",
    ]
