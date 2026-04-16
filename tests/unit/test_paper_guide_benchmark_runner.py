from tools.manual_regression import paper_guide_benchmark as runner


def test_load_suite_cross_paper_v1_resolves_repo_local_sources():
    suite = runner.load_suite(suite_name="cross_paper_v1")

    assert suite["suite_id"] == "cross_paper_v1"
    paper_ids = [paper["id"] for paper in suite["papers"]]
    assert "lsa_2026" in paper_ids
    assert "natphoton_2019_spi" in paper_ids
    assert all(str(paper.get("source_path") or "").endswith(".en.md") for paper in suite["papers"])
    case_ids = {
        case["id"]
        for paper in suite["papers"]
        for case in paper.get("cases") or []
    }
    assert "LSA_BEGINNER_RVT_APR_ROLE" in case_ids
    assert "NP2019_BEGINNER_OVERVIEW_APPLICATIONS" in case_ids
    assert "LSA_DISCUSSION_ONLY" in case_ids
    assert "NP2019_STRENGTH_LIMITS_TRADEOFF" in case_ids


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
    thresholds = suite.get("scorecard_thresholds")
    assert isinstance(thresholds, dict)
    assert thresholds["locate.first_click_rate"]["min"] == 0.9
    assert thresholds["citation.ref_num_accuracy"]["min"] == 0.95
    case_ids = {
        case["id"]
        for paper in suite["papers"]
        for case in paper.get("cases") or []
    }
    assert "LSA_BEGINNER_RVT_APR_ROLE" in case_ids
    assert "NP2019_BEGINNER_OVERVIEW_APPLICATIONS" in case_ids
    assert "LSA_DISCUSSION_ONLY" in case_ids
    assert "NC2023_BEGINNER_OVERVIEW_PIPELINE" in case_ids
    assert "NC2023_DISCUSSION_LIMITATION_FPM" in case_ids
    assert "NC2023_DISCUSSION_TIME_GATING" in case_ids
    assert "NP2019_STRENGTH_LIMITS_TRADEOFF" in case_ids


def test_load_manifest_paper_guide_smoke_general_v1_includes_beginner_overview_cases():
    suite = runner.load_suite(
        manifest_path="tools/manual_regression/manifests/paper_guide_smoke_general_v1.json"
    )

    case_ids = {
        case["id"]
        for paper in suite["papers"]
        for case in paper.get("cases") or []
    }
    assert "GEN_LSA_BEGINNER_RVT_APR_ROLE_EN" in case_ids
    assert "GEN_NP2019_BEGINNER_OVERVIEW_APPLICATIONS_EN" in case_ids
    assert "GEN_LSA_DISCUSSION_ONLY_EN" in case_ids
    assert "GEN_NP2019_STRENGTH_LIMITS_TRADEOFF_EN" in case_ids


def test_load_manifest_paper_guide_smoke_natcomm2023_v1_includes_discussion_case():
    suite = runner.load_suite(
        manifest_path="tools/manual_regression/manifests/paper_guide_smoke_natcomm2023_v1.json"
    )

    case_ids = {
        case["id"]
        for paper in suite["papers"]
        for case in paper.get("cases") or []
    }
    assert "NC2023_BEGINNER_OVERVIEW_PIPELINE_EN" in case_ids
    assert "NC2023_DISCUSSION_LIMITATION_FPM_EN" in case_ids
    assert "NC2023_DISCUSSION_TIME_GATING_EN" in case_ids


def test_core_smoke_manifests_keep_scorecard_thresholds():
    manifest_paths = [
        "tools/manual_regression/manifests/paper_guide_baseline_p0_v1.json",
        "tools/manual_regression/manifests/paper_guide_smoke_scinerf2024_v1.json",
        "tools/manual_regression/manifests/paper_guide_smoke_general_v1.json",
        "tools/manual_regression/manifests/paper_guide_smoke_natcomm2023_v1.json",
    ]

    for path in manifest_paths:
        suite = runner.load_suite(manifest_path=path)
        thresholds = suite.get("scorecard_thresholds")
        assert isinstance(thresholds, dict), path
        assert thresholds["overall_pass_rate"]["min"] == 1.0, path
        assert thresholds["locate.first_click_rate"]["min"] == 0.9, path
        assert thresholds["locate.heading_or_none_rate"]["max"] == 0.05, path
        assert thresholds["citation.ref_num_accuracy"]["min"] == 0.95, path
        assert thresholds["structured_markers.clean_rate"]["min"] == 1.0, path
        assert thresholds["quality.minimum_ok_rate"]["min"] == 1.0, path


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
        "cite_details": [
            {
                "num": 34,
                "anchor": "kb-cite-demo-34",
                "title": "Radial Variance Transform",
                "cite_fmt": "RVT foundational paper.",
            }
        ],
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
    assert out["gate_results"]["locate"]["render_target_checked"] is False


def test_evaluate_case_fails_when_render_packet_target_mismatches_visible_jump_target():
    case = {
        "id": "CASE1B",
        "title": "citation locate render target mismatch",
        "paper_id": "demo",
        "paper_title": "Demo",
        "prompt_family": "citation_lookup",
        "prompt": "Which ADMM citation?",
        "checks": {
            "citation": {"required": True, "expected_ref_nums": [4], "allow_only_expected": True},
            "locate": {
                "required": True,
                "exact": True,
                "heading_includes_any": ["Related Work"],
                "anchor_contains_any": ["ADMM", "alternating direction method of multipliers"],
            },
        },
    }
    provenance = {
        "segments": [
            {
                "segment_id": "seg-2",
                "evidence_mode": "direct",
                "locate_policy": "required",
                "primary_block_id": "blk_admm",
                "primary_anchor_id": "a-admm",
                "primary_heading_path": "Related Work / Snapshot Compressive Imaging",
                "support_locate_anchor": "most of the existing methods employ alternating direction method of multipliers (ADMM) [4],",
                "resolved_ref_num": 4,
                "support_slot_claim_type": "prior_work",
                "hit_level": "exact",
            }
        ]
    }
    message = {
        "meta": {
            "paper_guide_contracts": {
                "render_packet": {
                    "cite_details": [{"num": 4, "title": "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers"}],
                    "locate_target": {
                        "segmentId": "seg-1",
                        "headingPath": "Method / Wrong Section",
                        "snippet": "A generic method sentence unrelated to the citation prompt.",
                        "anchorText": "A generic method sentence unrelated to the citation prompt.",
                        "blockId": "blk_wrong",
                        "anchorId": "a-wrong",
                        "anchorKind": "sentence",
                    },
                    "reader_open": {
                        "sourcePath": "demo.md",
                        "headingPath": "Method / Wrong Section",
                        "snippet": "A generic method sentence unrelated to the citation prompt.",
                        "blockId": "blk_wrong",
                        "anchorId": "a-wrong",
                        "strictLocate": True,
                    },
                }
            }
        }
    }

    out = runner._evaluate_case(
        case,
        suite_defaults={},
        answer_text="The paper cites [4].",
        answer_quality={"minimum_ok": True},
        render_cache={},
        provenance=provenance,
        message=message,
    )

    assert out["status"] == "FAIL"
    assert out["gate_results"]["locate"]["status"] == "FAIL"
    assert "locate_render_target_mismatch" in out["gate_results"]["locate"]["reasons"]


def test_evaluate_case_citation_requires_reasonable_reference_detail_text():
    case = {
        "id": "CASE1C",
        "title": "citation detail quality",
        "paper_id": "demo",
        "paper_title": "Demo",
        "prompt_family": "citation_lookup",
        "prompt": "Which citation is Duarte?",
        "checks": {
            "citation": {
                "required": True,
                "expected_ref_nums": [4],
                "allow_only_expected": True,
                "expected_reference_text_contains_any": ["compressed sensing", "duarte"],
            }
        },
    }
    provenance = {
        "segments": [
            {
                "evidence_mode": "direct",
                "locate_policy": "required",
                "resolved_ref_num": 4,
            }
        ]
    }
    message = {
        "meta": {
            "paper_guide_contracts": {
                "render_packet": {
                    "cite_details": [
                        {
                            "num": 4,
                            "title": "Attention Is All You Need",
                            "authors": "Vaswani A",
                            "cite_fmt": "Vaswani A. Attention Is All You Need.",
                        }
                    ]
                }
            }
        }
    }

    out = runner._evaluate_case(
        case,
        suite_defaults={},
        answer_text="The paper cites [4].",
        answer_quality={"minimum_ok": True},
        render_cache={},
        provenance=provenance,
        message=message,
    )

    assert out["status"] == "FAIL"
    assert out["gate_results"]["citation"]["status"] == "FAIL"
    assert "expected_reference_text_missing_any" in out["gate_results"]["citation"]["reasons"]


def test_evaluate_case_accepts_render_packet_target_when_visible_jump_is_correct_even_if_internal_match_is_sparse():
    case = {
        "id": "CASE1D",
        "title": "overview render target fallback",
        "paper_id": "demo",
        "paper_title": "Demo",
        "prompt_family": "overview",
        "prompt": "Why is SPI useful and where is that stated?",
        "checks": {
            "locate": {
                "required": True,
                "exact": True,
                "heading_includes_any": ["Applications"],
                "anchor_contains_any": ["remote sensing", "quantum state tomography", "infrared"],
            }
        },
    }
    provenance = {
        "segments": [
            {
                "segment_id": "seg-shell",
                "evidence_mode": "direct",
                "locate_policy": "required",
                "primary_block_id": "blk-shell",
                "primary_anchor_id": "a-shell",
                "primary_heading_path": "Overview",
                "support_locate_anchor": "A generic summary sentence.",
                "support_slot_claim_type": "own_result",
                "hit_level": "exact",
            }
        ]
    }
    message = {
        "meta": {
            "paper_guide_contracts": {
                "render_packet": {
                    "locate_target": {
                        "segmentId": "seg-app",
                        "headingPath": "Applications / Why SPI matters",
                        "snippet": "SPI is useful for infrared imaging, remote sensing, and quantum state tomography.",
                        "anchorText": "SPI is useful for infrared imaging, remote sensing, and quantum state tomography.",
                        "blockId": "blk-app",
                        "anchorId": "a-app",
                        "anchorKind": "sentence",
                    },
                    "reader_open": {
                        "sourcePath": "demo.md",
                        "headingPath": "Applications / Why SPI matters",
                        "snippet": "SPI is useful for infrared imaging, remote sensing, and quantum state tomography.",
                        "blockId": "blk-app",
                        "anchorId": "a-app",
                        "strictLocate": True,
                    },
                }
            }
        }
    }

    out = runner._evaluate_case(
        case,
        suite_defaults={},
        answer_text="SPI is useful for infrared imaging, remote sensing, and quantum state tomography.",
        answer_quality={"minimum_ok": True},
        render_cache={},
        provenance=provenance,
        message=message,
    )

    assert out["status"] == "PASS"
    assert out["gate_results"]["locate"]["status"] == "PASS"
    assert out["gate_results"]["locate"]["render_target_ok"] is True


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


def test_build_summary_payload_emits_scorecard_and_family_breakdown():
    results = [
        {
            "id": "CITE1",
            "status": "PASS",
            "prompt_family": "citation_lookup",
            "primary_hit_level": "exact",
            "visible_direct_segment_count": 1,
            "raw_structured_cite_count": 0,
            "hit_level_counts": {"exact": 1, "block": 0, "heading": 0, "none": 0},
            "gate_results": {
                "citation": {"status": "PASS", "reasons": []},
                "locate": {"status": "PASS", "reasons": []},
                "structured_markers": {"status": "PASS", "reasons": []},
                "quality": {"status": "PASS", "reasons": []},
            },
        },
        {
            "id": "METHOD1",
            "status": "FAIL",
            "prompt_family": "method",
            "primary_hit_level": "heading",
            "visible_direct_segment_count": 1,
            "raw_structured_cite_count": 1,
            "hit_level_counts": {"exact": 0, "block": 0, "heading": 1, "none": 0},
            "gate_results": {
                "locate": {"status": "FAIL", "reasons": ["locate_missing_matching_segment"]},
                "structured_markers": {"status": "FAIL", "reasons": ["raw_structured_cite_count=1"]},
                "quality": {"status": "PASS", "reasons": []},
            },
        },
        {
            "id": "OV1",
            "status": "PASS",
            "prompt_family": "overview",
            "primary_hit_level": "none",
            "visible_direct_segment_count": 0,
            "raw_structured_cite_count": 0,
            "hit_level_counts": {"exact": 0, "block": 0, "heading": 0, "none": 0},
            "gate_results": {
                "structured_markers": {"status": "PASS", "reasons": []},
                "quality": {"status": "PASS", "reasons": []},
            },
        },
    ]

    summary = runner._build_summary_payload(
        base_url="http://127.0.0.1:8016",
        suite={"suite_id": "demo_suite", "title": "Demo Suite"},
        results=results,
        started_at="2026-04-09T12:00:00",
    )

    assert summary["counts"] == {"total": 3, "pass": 2, "fail": 1}
    assert summary["overall_status"] == "FAIL"
    assert summary["scorecard"]["locate"]["cases"] == 2
    assert summary["scorecard"]["locate"]["first_click_ok"] == 1
    assert summary["scorecard"]["locate"]["heading_or_none"] == 1
    assert summary["scorecard"]["citation"]["cases"] == 1
    assert summary["scorecard"]["citation"]["ref_num_accuracy"] == 1.0
    assert summary["scorecard"]["structured_markers"]["raw_cite_leak_cases"] == 1
    assert summary["by_family"]["citation_lookup"]["citation_pass"] == 1
    assert summary["by_family"]["method"]["locate_cases"] == 1
    assert summary["by_family"]["overview"]["cases_with_direct"] == 0


def test_build_summary_payload_fails_when_scorecard_thresholds_fail():
    results = [
        {
            "id": "CASE1",
            "status": "PASS",
            "prompt_family": "method",
            "primary_hit_level": "heading",
            "visible_direct_segment_count": 1,
            "raw_structured_cite_count": 0,
            "hit_level_counts": {"exact": 0, "block": 0, "heading": 1, "none": 0},
            "gate_results": {
                "locate": {"status": "PASS", "reasons": []},
                "structured_markers": {"status": "PASS", "reasons": []},
                "quality": {"status": "PASS", "reasons": []},
            },
        }
    ]

    summary = runner._build_summary_payload(
        base_url="http://127.0.0.1:8016",
        suite={
            "suite_id": "threshold_demo",
            "title": "Threshold Demo",
            "scorecard_thresholds": {
                "locate.first_click_rate": {"min": 0.90},
                "locate.heading_or_none_rate": {"max": 0.05},
            },
        },
        results=results,
        started_at="2026-04-09T12:00:00",
    )

    assert summary["thresholds"]["enabled"] is True
    assert summary["thresholds"]["status"] == "FAIL"
    assert summary["overall_status"] == "FAIL"
    failed_metrics = {
        str(item.get("metric") or "")
        for item in summary["thresholds"]["checks"]
        if not bool(item.get("ok"))
    }
    assert "locate.first_click_rate" in failed_metrics
    assert "locate.heading_or_none_rate" in failed_metrics
