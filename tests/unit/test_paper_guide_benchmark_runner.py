from tools.manual_regression import paper_guide_benchmark as runner


def test_load_suite_cross_paper_v1_resolves_repo_local_sources():
    suite = runner.load_suite(suite_name="cross_paper_v1")

    assert suite["suite_id"] == "cross_paper_v1"
    paper_ids = [paper["id"] for paper in suite["papers"]]
    assert "lsa_2026" in paper_ids
    assert "natphoton_2019_spi" in paper_ids
    assert all(str(paper.get("source_path") or "").endswith(".en.md") for paper in suite["papers"])


def test_load_manifest_paper_guide_baseline_p0_v1():
    suite = runner.load_suite(
        manifest_path="tools/manual_regression/manifests/paper_guide_baseline_p0_v1.json"
    )

    assert suite["suite_id"] == "paper_guide_baseline_p0_v1"
    paper_ids = [paper["id"] for paper in suite["papers"]]
    assert len(paper_ids) >= 5
    assert "cvpr_2024_scinerf" in paper_ids
    assert "natcommun_2021_sph" in paper_ids
    assert all(str(paper.get("source_path") or "").endswith(".en.md") for paper in suite["papers"])


def test_evaluate_case_passes_expected_citation_and_locate_from_provenance():
    case = {
        "id": "CASE1",
        "title": "citation locate",
        "paper_id": "demo",
        "paper_title": "Demo",
        "prompt_family": "citation_lookup",
        "prompt": "Which citation?",
        "checks": {
            "answer": {"contains_any": ["RVT"]},
            "citation": {"required": True, "expected_ref_nums": [34], "allow_only_expected": True},
            "locate": {
                "required": True,
                "exact": True,
                "heading_includes_any": ["Methods / RVT"],
                "anchor_contains_any": ["RVT"],
            },
        },
    }
    answer = "The method uses RVT [34]."
    provenance = {
        "segments": [
            {
                "evidence_mode": "direct",
                "locate_policy": "required",
                "primary_block_id": "blk_rvt",
                "primary_heading_path": "Methods / RVT",
                "support_locate_anchor": "Specifically, we use the radial variance transform (RVT)[34].",
                "resolved_ref_num": 34,
                "support_slot_claim_type": "method_detail",
            }
        ]
    }
    message = {
        "cite_details": [{"num": 34, "anchor": "kb-cite-demo-34"}],
        "rendered_body": "The method uses RVT [34](#kb-cite-demo-34).",
    }

    out = runner._evaluate_case(
        case,
        suite_defaults={},
        answer_text=answer,
        answer_quality={"minimum_ok": True},
        render_cache={},
        provenance=provenance,
        message=message,
    )

    assert out["status"] == "PASS"
    assert out["gate_results"]["citation"]["matched_ref_nums"] == [34]
    assert out["gate_results"]["locate"]["matched_segment_count"] == 1


def test_evaluate_case_section_target_exclusive_fails_when_visible_segment_leaks_other_section():
    case = {
        "id": "CASE2",
        "title": "discussion only",
        "paper_id": "demo",
        "paper_title": "Demo",
        "prompt_family": "discussion_only",
        "prompt": "From discussion only",
        "checks": {
            "section_target": {
                "required": True,
                "exclusive": True,
                "heading_includes_any": ["Discussion"],
                "heading_excludes_any": ["Results", "Methods"],
            }
        },
    }
    provenance = {
        "segments": [
            {
                "evidence_mode": "direct",
                "locate_policy": "required",
                "primary_block_id": "blk_disc",
                "primary_heading_path": "Discussion",
                "anchor_text": "Future work includes SPAD arrays.",
            },
            {
                "evidence_mode": "direct",
                "locate_policy": "required",
                "primary_block_id": "blk_results",
                "primary_heading_path": "Results / Figure 1",
                "anchor_text": "APR improves CNR.",
            },
        ]
    }

    out = runner._evaluate_case(
        case,
        suite_defaults={},
        answer_text="Future work includes SPAD arrays.",
        answer_quality={},
        render_cache={},
        provenance=provenance,
        message={},
    )

    assert out["status"] == "FAIL"
    assert out["gate_results"]["section_target"]["status"] == "FAIL"
    assert "section_target_has_non_target_segment" in out["gate_results"]["section_target"]["reasons"]


def test_evaluate_case_figure_panel_requires_matching_panel_and_claim_type():
    case = {
        "id": "CASE3",
        "title": "figure panel",
        "paper_id": "demo",
        "paper_title": "Demo",
        "prompt_family": "figure_walkthrough",
        "prompt": "Figure panel f",
        "checks": {
            "figure_panel": {
                "required": True,
                "panel_letters": ["f"],
                "heading_includes_any": ["Figure 3"],
                "anchor_contains_any": ["methane imaging using SPC"],
                "claim_types": ["figure_panel"],
            }
        },
    }
    provenance = {
        "segments": [
            {
                "evidence_mode": "direct",
                "locate_policy": "required",
                "primary_block_id": "blk_fig",
                "primary_heading_path": "Results / Figure 3",
                "support_locate_anchor": "f methane imaging using SPC.",
                "support_slot_claim_type": "figure_panel",
            }
        ]
    }

    out = runner._evaluate_case(
        case,
        suite_defaults={},
        answer_text="Figure 3 panel (f) corresponds to methane imaging using SPC.",
        answer_quality={},
        render_cache={},
        provenance=provenance,
        message={},
    )

    assert out["status"] == "PASS"
    assert out["gate_results"]["figure_panel"]["matched_segment_count"] == 1


def test_evaluate_case_fails_when_raw_support_marker_leaks():
    case = {
        "id": "CASE4",
        "title": "support marker leak",
        "paper_id": "demo",
        "paper_title": "Demo",
        "prompt_family": "method",
        "prompt": "Explain method",
        "checks": {},
    }

    out = runner._evaluate_case(
        case,
        suite_defaults={"structured_markers": {"forbid_raw_support_markers": True}},
        answer_text="APR detail [[SUPPORT:DOC-1]].",
        answer_quality={},
        render_cache={},
        provenance={},
        message={},
    )

    assert out["status"] == "FAIL"
    assert out["gate_results"]["structured_markers"]["status"] == "FAIL"
    assert "raw_support_marker_count=1" in out["gate_results"]["structured_markers"]["reasons"]
