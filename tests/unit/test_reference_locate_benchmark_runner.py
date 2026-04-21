from tools.manual_regression import reference_locate_benchmark as runner


def test_load_suite_reference_locate_quality_v1_resolves_bound_sources():
    suite = runner.load_suite("tools/manual_regression/manifests/reference_locate_quality_v1.json")

    assert suite["suite_id"] == "reference_locate_quality_v1"
    case_ids = {case["id"] for case in suite["cases"]}
    assert case_ids == {
        "NORMAL_DYNAMIC_SUPERSAMPLING_DEFINE",
        "NORMAL_HADAMARD_FOURIER_COMPARE",
        "NORMAL_ADMM_NEGATIVE",
        "GUIDE_OTHER_PAPERS_FOURIER",
        "GUIDE_OTHER_PAPERS_ADMM_NEGATIVE",
    }
    guide_cases = [case for case in suite["cases"] if str(case.get("mode") or "") == "paper_guide"]
    assert all(str(case.get("bound_source_path") or "").endswith(".en.md") for case in guide_cases)


def test_evaluate_case_passes_precise_top_hit_quality_checks():
    case = {
        "id": "demo",
        "checks": {
            "hits": {"min": 1, "max": 1},
            "top_hit": {
                "source_contains_any": ["SciAdv-2017", "dynamic supersampling"],
                "summary_contains_any": ["dynamic supersampling"],
                "why_contains_any": ["dynamic supersampling"],
                "summary_basis_contains_any": ["LLM", "matched section"],
                "why_basis_contains_any": ["LLM", "matched section"],
            },
        },
    }
    refs_pack = {
        "hits": [
            {
                "meta": {"source_path": r"db\SciAdv-2017\SciAdv-2017.en.md"},
                "ui_meta": {
                    "display_name": "SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.pdf",
                    "summary_line": "The paper explicitly defines dynamic supersampling for adaptive single-pixel imaging.",
                    "why_line": "This hit is directly relevant because the section defines dynamic supersampling itself.",
                    "summary_basis": "LLM-distilled from matched section evidence",
                    "why_basis": "LLM-grounded relevance from matched section evidence",
                },
            }
        ]
    }

    out = runner._evaluate_case(case, refs_pack=refs_pack)

    assert out["status"] == "PASS"
    assert out["gate_results"]["hits"]["count"] == 1
    assert out["gate_results"]["top_hit"]["status"] == "PASS"


def test_evaluate_case_accepts_localized_deterministic_basis_for_matched_section_gate():
    case = {
        "id": "demo-localized",
        "checks": {
            "hits": {"min": 1, "max": 1},
            "top_hit": {
                "source_contains_any": ["OE-2017", "Fourier single-pixel imaging"],
                "summary_contains_any": ["Hadamard", "Fourier"],
                "why_contains_any": ["Hadamard", "Fourier"],
                "summary_basis_contains_any": ["LLM", "matched section"],
                "why_basis_contains_any": ["LLM", "matched section"],
            },
        },
    }
    refs_pack = {
        "hits": [
            {
                "meta": {"source_path": r"db\OE-2017\OE-2017.en.md"},
                "ui_meta": {
                    "display_name": "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
                    "summary_line": "Hadamard single-pixel imaging and Fourier single-pixel imaging are compared in Figure 1.",
                    "why_line": "这条命中直接比较了 Hadamard 和 Fourier single-pixel imaging。",
                    "summary_basis": "基于命中章节/定位证据",
                    "why_basis": "基于命中章节和关键词对齐的规则化说明",
                },
            }
        ]
    }

    out = runner._evaluate_case(case, refs_pack=refs_pack)

    assert out["status"] == "PASS"
    assert out["gate_results"]["top_hit"]["status"] == "PASS"


def test_evaluate_case_fails_when_top_hit_is_wrong_source():
    case = {
        "id": "demo-wrong",
        "checks": {
            "hits": {"min": 1, "max": 1},
            "top_hit": {
                "source_contains_any": ["SciAdv-2017"],
                "source_not_contains_any": ["SCIGS"],
            },
        },
    }
    refs_pack = {
        "hits": [
            {
                "meta": {"source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md"},
                "ui_meta": {
                    "display_name": "ICIP-2025-SCIGS.pdf",
                    "summary_line": "A different paper.",
                    "why_line": "Not the expected one.",
                    "summary_basis": "Condensed from matched section evidence",
                    "why_basis": "Conservative relevance note from the available evidence",
                },
            }
        ]
    }

    out = runner._evaluate_case(case, refs_pack=refs_pack)

    assert out["status"] == "FAIL"
    assert "top_hit_source_mismatch" in out["failures"]
    assert "top_hit_source_forbidden" in out["failures"]


def test_evaluate_case_passes_negative_no_hit_and_guide_filter():
    case = {
        "id": "negative",
        "checks": {
            "hits": {"min": 0, "max": 0},
            "guide_filter": {"active": True, "hidden_self_source": True},
        },
    }
    refs_pack = {
        "hits": [],
        "guide_filter": {
            "active": True,
            "hidden_self_source": True,
            "filtered_hit_count": 1,
        },
    }

    out = runner._evaluate_case(case, refs_pack=refs_pack)

    assert out["status"] == "PASS"
    assert out["gate_results"]["guide_filter"]["status"] == "PASS"


def test_evaluate_case_passes_pipeline_debug_gate_when_raw_and_focus_hits_exist():
    case = {
        "id": "pipeline-pass",
        "checks": {
            "hits": {"min": 1, "max": 1},
            "pipeline_debug": {
                "require_raw_hits": True,
                "require_post_focus_hits": True,
            },
        },
    }
    refs_pack = {
        "hits": [{"ui_meta": {"display_name": "OE-2017.pdf"}}],
        "pipeline_debug": {
            "raw_hit_count": 3,
            "post_score_gate_hit_count": 2,
            "post_focus_filter_hit_count": 1,
            "post_llm_filter_hit_count": 1,
            "final_hit_count": 1,
        },
    }

    out = runner._evaluate_case(case, refs_pack=refs_pack)

    assert out["status"] == "PASS"
    assert out["gate_results"]["pipeline_debug"]["status"] == "PASS"


def test_evaluate_case_fails_pipeline_debug_gate_when_retrieval_is_empty_before_filters():
    case = {
        "id": "pipeline-fail",
        "checks": {
            "hits": {"min": 1, "max": 1},
            "pipeline_debug": {
                "require_raw_hits": True,
                "require_post_focus_hits": True,
            },
        },
    }
    refs_pack = {
        "hits": [],
        "pipeline_debug": {
            "raw_hit_count": 0,
            "post_score_gate_hit_count": 0,
            "post_focus_filter_hit_count": 0,
            "post_llm_filter_hit_count": 0,
            "final_hit_count": 0,
        },
    }

    out = runner._evaluate_case(case, refs_pack=refs_pack)

    assert out["status"] == "FAIL"
    assert "retrieval_empty_before_ui_filters" in out["failures"]
    assert "all_hits_removed_by_focus_filter" in out["failures"]


def test_evaluate_case_passes_evidence_identity_gate_when_surfaces_align():
    case = {
        "id": "identity-pass",
        "checks": {
            "hits": {"min": 1, "max": 1},
            "evidence_identity": {
                "require_pack_primary": True,
                "require_hit_reader_sync": True,
                "require_assistant_primary": True,
            },
        },
    }
    refs_pack = {
        "primary_evidence": {
            "source_name": "OE-2017.pdf",
            "block_id": "blk_22",
            "anchor_id": "a_22",
            "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
            "snippet": "Section 2.2 compares Hadamard and Fourier basis patterns.",
        },
        "hits": [
            {
                "meta": {"source_path": r"db\OE-2017\OE-2017.en.md"},
                "ui_meta": {
                    "primary_evidence": {
                        "source_name": "OE-2017.pdf",
                        "block_id": "blk_22",
                        "anchor_id": "a_22",
                        "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                    },
                    "reader_open": {
                        "sourcePath": r"db\OE-2017\OE-2017.en.md",
                        "blockId": "blk_22",
                        "anchorId": "a_22",
                        "headingPath": "2. Comparison of theory / 2.2 Basis patterns generation",
                        "primaryEvidence": {
                            "source_name": "OE-2017.pdf",
                            "block_id": "blk_22",
                            "anchor_id": "a_22",
                            "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                        },
                    },
                },
            }
        ],
    }
    assistant_message = {
        "id": 9,
        "role": "assistant",
        "meta": {
            "paper_guide_contracts": {
                "render_packet": {
                    "primary_evidence": {
                        "source_name": "OE-2017.pdf",
                        "block_id": "blk_22",
                        "anchor_id": "a_22",
                        "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                        "snippet": "Section 2.2 compares Hadamard and Fourier basis patterns.",
                    }
                }
            }
        },
    }

    out = runner._evaluate_case(case, refs_pack=refs_pack, assistant_message=assistant_message)

    assert out["status"] == "PASS"
    assert out["gate_results"]["evidence_identity"]["status"] == "PASS"
    assert out["gate_results"]["evidence_identity"]["assistant_primary"]["block_id"] == "blk_22"


def test_evaluate_case_fails_evidence_identity_gate_when_assistant_heading_drifts():
    case = {
        "id": "identity-fail",
        "checks": {
            "hits": {"min": 1, "max": 1},
            "evidence_identity": {
                "require_pack_primary": True,
                "require_hit_reader_sync": True,
                "require_assistant_primary": True,
            },
        },
    }
    refs_pack = {
        "primary_evidence": {
            "source_name": "OE-2017.pdf",
            "block_id": "blk_22",
            "anchor_id": "a_22",
            "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
        },
        "hits": [
            {
                "ui_meta": {
                    "reader_open": {
                        "primaryEvidence": {
                            "source_name": "OE-2017.pdf",
                            "block_id": "blk_22",
                            "anchor_id": "a_22",
                            "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                        }
                    }
                }
            }
        ],
    }
    assistant_message = {
        "id": 10,
        "role": "assistant",
        "meta": {
            "paper_guide_contracts": {
                "render_packet": {
                    "primary_evidence": {
                        "source_name": "OE-2017.pdf",
                        "block_id": "blk_24",
                        "anchor_id": "a_24",
                        "heading_path": "2. Comparison of theory / 2.4 Efficiency",
                    }
                }
            }
        },
    }

    out = runner._evaluate_case(case, refs_pack=refs_pack, assistant_message=assistant_message)

    assert out["status"] == "FAIL"
    assert out["gate_results"]["evidence_identity"]["status"] == "FAIL"
    assert any(str(reason).startswith("pack_assistant_primary_mismatch:") for reason in out["failures"])


def test_evaluate_case_skips_missing_assistant_primary_when_answer_is_model_failure():
    case = {
        "id": "identity-skip-model-failure",
        "checks": {
            "hits": {"min": 1, "max": 1},
            "evidence_identity": {
                "require_pack_primary": True,
                "require_hit_reader_sync": True,
                "require_assistant_primary": True,
            },
        },
    }
    refs_pack = {
        "primary_evidence": {
            "source_name": "OE-2017.pdf",
            "block_id": "blk_22",
            "anchor_id": "a_22",
            "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
        },
        "hits": [
            {
                "ui_meta": {
                    "reader_open": {
                        "primaryEvidence": {
                            "source_name": "OE-2017.pdf",
                            "block_id": "blk_22",
                            "anchor_id": "a_22",
                            "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                        }
                    }
                }
            }
        ],
    }
    assistant_message = {
        "id": 12,
        "role": "assistant",
        "content": "（调用模型失败：Connection error.）\n\n请检查 DEEPSEEK_API_KEY / BASE_URL / MODEL 是否正确。",
        "meta": {},
    }

    out = runner._evaluate_case(case, refs_pack=refs_pack, assistant_message=assistant_message)

    assert out["status"] == "PASS"
    assert out["gate_results"]["evidence_identity"]["status"] == "PASS"
    assert out["gate_results"]["evidence_identity"]["assistant_model_failure"] is True
    assert out["gate_results"]["evidence_identity"]["assistant_primary"] == {}


def test_evaluate_case_passes_consistency_metrics_when_heading_and_primary_surfaces_align():
    case = {
        "id": "consistency-pass",
        "checks": {
            "hits": {"min": 1, "max": 1},
            "consistency_metrics": {
                "require_summary_heading_consistent": True,
                "require_heading_reader_open_consistent": True,
                "require_fast_to_full_primary_block_same": True,
            },
        },
    }
    refs_pack = {
        "render_status": "full",
        "primary_evidence": {
            "source_name": "OE-2017.pdf",
            "block_id": "blk_22",
            "anchor_id": "a_22",
            "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
        },
        "hits": [
            {
                "ui_meta": {
                    "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                    "summary_line": "Section 2.2 directly compares Hadamard and Fourier basis patterns.",
                    "primary_evidence": {
                        "source_name": "OE-2017.pdf",
                        "block_id": "blk_22",
                        "anchor_id": "a_22",
                        "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                    },
                    "reader_open": {
                        "headingPath": "2. Comparison of theory / 2.2 Basis patterns generation",
                        "primaryEvidence": {
                            "source_name": "OE-2017.pdf",
                            "block_id": "blk_22",
                            "anchor_id": "a_22",
                            "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                        },
                    },
                }
            }
        ],
    }
    first_refs_pack = {
        "render_status": "fast",
        "primary_evidence": {
            "source_name": "OE-2017.pdf",
            "block_id": "blk_22",
            "anchor_id": "a_22",
            "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
        },
    }

    out = runner._evaluate_case(
        case,
        refs_pack=refs_pack,
        first_refs_pack=first_refs_pack,
        final_refs_pack=refs_pack,
    )

    assert out["status"] == "PASS"
    gate = out["gate_results"]["consistency_metrics"]
    assert gate["status"] == "PASS"
    assert gate["summary_heading_consistent"] is True
    assert gate["heading_reader_open_consistent"] is True
    assert gate["fast_to_full_primary_block_same"] is True


def test_evaluate_case_fails_consistency_metrics_when_heading_and_fast_full_drift():
    case = {
        "id": "consistency-fail",
        "checks": {
            "hits": {"min": 1, "max": 1},
            "consistency_metrics": {
                "require_summary_heading_consistent": True,
                "require_heading_reader_open_consistent": True,
                "require_fast_to_full_primary_block_same": True,
            },
        },
    }
    refs_pack = {
        "render_status": "full",
        "primary_evidence": {
            "source_name": "OE-2017.pdf",
            "block_id": "blk_24",
            "anchor_id": "a_24",
            "heading_path": "2. Comparison of theory / 2.4 Efficiency",
        },
        "hits": [
            {
                "ui_meta": {
                    "heading_path": "2. Comparison of theory / 2.4 Efficiency",
                    "summary_line": "Section 2.2 directly compares Hadamard and Fourier basis patterns.",
                    "primary_evidence": {
                        "source_name": "OE-2017.pdf",
                        "block_id": "blk_22",
                        "anchor_id": "a_22",
                        "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                    },
                    "reader_open": {
                        "headingPath": "2. Comparison of theory / 2.2 Basis patterns generation",
                        "primaryEvidence": {
                            "source_name": "OE-2017.pdf",
                            "block_id": "blk_22",
                            "anchor_id": "a_22",
                            "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                        },
                    },
                }
            }
        ],
    }
    first_refs_pack = {
        "render_status": "fast",
        "primary_evidence": {
            "source_name": "OE-2017.pdf",
            "block_id": "blk_22",
            "anchor_id": "a_22",
            "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
        },
    }

    out = runner._evaluate_case(
        case,
        refs_pack=refs_pack,
        first_refs_pack=first_refs_pack,
        final_refs_pack=refs_pack,
    )

    assert out["status"] == "FAIL"
    gate = out["gate_results"]["consistency_metrics"]
    assert gate["status"] == "FAIL"
    assert gate["summary_heading_consistent"] is False
    assert gate["heading_reader_open_consistent"] is False
    assert gate["fast_to_full_primary_block_same"] is False
    assert "summary_heading_inconsistent" in out["failures"]
    assert "heading_reader_open_inconsistent" in out["failures"]
    assert "fast_to_full_primary_block_drift" in out["failures"]


def test_assistant_primary_evidence_prefers_contract_snapshot_over_render_packet_surface():
    message = {
        "id": 21,
        "role": "assistant",
        "meta": {
            "paper_guide_contracts": {
                "primary_evidence": {
                    "source_name": "OE-2017.pdf",
                    "block_id": "blk_22",
                    "anchor_id": "a_22",
                    "heading_path": "2. Comparison / 2.2 Basis patterns generation",
                },
                "render_packet": {
                    "primary_evidence": {
                        "source_name": "NatPhoton-2019.pdf",
                        "heading_path": "Abstract / Acquisition and image reconstruction strategies.",
                    }
                },
            }
        },
    }

    out = runner._assistant_primary_evidence(message)

    assert out["source_name"] == "OE-2017.pdf"
    assert out["block_id"] == "blk_22"
    assert out["heading_path"] == "2. Comparison / 2.2 Basis patterns generation"
