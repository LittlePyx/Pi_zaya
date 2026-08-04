from __future__ import annotations

from tools.research_qa import run_research_qa_eval as eval_mod
from tools.research_qa.run_research_qa_eval import (
    ResearchQaFixture,
    _answer_matches_locale,
    _assistant_message_by_id,
    _build_report,
    _case_requires_full_refs_wait,
    _claim_evidence_contract_failures,
    _detail_matches_source_page,
    _generation_should_wait_for_full_refs,
    _latency_budget_checks,
    _missing_term_groups,
    _refs_payload_is_converged_for_case,
    _refs_payload_is_full,
    _refs_payload_is_terminal_for_case,
    _ref_hit_locale_failures,
    _timing_summary,
    evaluate_retrieval_coverage,
    evaluate_replay_rows,
    load_fixture,
    load_replay,
    select_fixture_cases,
    source_path_for_doc,
    summarize_suite_coverage,
    validate_case,
    validate_fixture_contracts,
    validate_fixture_sources,
)


def test_locale_gate_checks_answer_and_reference_card_setting():
    assert _answer_matches_locale("这是一个有依据的中文回答，说明了方法和限制。", "zh")
    assert _answer_matches_locale(
        "This grounded English answer explains the method and its practical limitations.",
        "en",
    )
    assert not _answer_matches_locale("这不是英文回答。", "en")
    assert _ref_hit_locale_failures(
        [{"ui_meta": {"render_locale": "en"}}, {"ui_meta": {"render_locale": "zh"}}],
        "en",
    ) == [{"index": 2, "expected": "en", "actual": "zh"}]


def test_retrieval_coverage_reports_required_document_ranks(monkeypatch, tmp_path):
    docs = [
        {"id": "alpha", "dir": "alpha", "file": "alpha"},
        {"id": "beta", "dir": "beta", "file": "beta"},
    ]
    fixture = ResearchQaFixture(
        db_root=str(tmp_path),
        docs=docs,
        cases=[
            {
                "id": "compare-alpha-beta",
                "question": "Compare the alpha mechanism with the beta mechanism.",
                "expected": {"requiredRefDocIds": ["alpha", "beta"]},
            }
        ],
        forbidden_phrases=[],
    )
    alpha_path = tmp_path / "alpha" / "alpha.en.md"
    beta_path = tmp_path / "beta" / "beta.en.md"
    alpha_path.parent.mkdir(parents=True)
    beta_path.parent.mkdir(parents=True)
    alpha_path.write_text("# Alpha\n\nAlpha mechanism evidence.", encoding="utf-8")
    beta_path.write_text("# Beta\n\nBeta mechanism evidence.", encoding="utf-8")
    monkeypatch.setattr(
        eval_mod,
        "load_all_chunks",
        lambda _root: [{"id": "placeholder", "text": "placeholder", "meta": {}}],
    )
    monkeypatch.setattr(
        eval_mod,
        "_search_hits_with_fallback",
        lambda *_args, **_kwargs: ([{"id": "hit"}], [1.0], _args[0], False, [_args[0]]),
    )
    monkeypatch.setattr(
        eval_mod,
        "_group_hits_by_doc_for_refs",
        lambda *_args, **_kwargs: [
            {"meta": {"source_path": str(alpha_path)}},
            {"meta": {"source_path": str(beta_path)}},
        ],
    )

    summary = evaluate_retrieval_coverage(fixture, db_root=tmp_path, top_k=2)

    assert summary["ok"] is True, summary
    assert summary["passed"] == 1
    assert summary["cases"][0]["required_ranks"] == {"alpha": 1, "beta": 2}


def test_research_qa_latency_budgets_track_answer_and_async_cards_separately():
    checks = _latency_budget_checks(
        {
            "maxFirstAnswerMs": 10000,
            "maxAnswerCompleteMs": 10000,
            "maxCardsCompleteMs": 30000,
        },
        {
            "first_answer_ms": 4200,
            "answer_complete_ms": 8700,
            "cards_complete_ms": 31500,
        },
    )

    assert [item["ok"] for item in checks] == [True, True, False]
    assert checks[-1]["name"] == "latency_cards_complete_ms"


def test_inline_citation_contract_requires_repeated_claim_level_grounding():
    contracts = [
        {
            "id": "s2ism-inline",
            "termGroups": [
                ["s2ISM", "structured detection"],
                ["super-resolution", "optical sectioning"],
            ],
            "minCitedUnits": 2,
        }
    ]
    answer = (
        "s2ISM structured detection provides super-resolution [1](#kb-cite-1).\n"
        "The detailed body says s2ISM improves optical sectioning without a marker."
    )

    failures = eval_mod._inline_citation_contract_failures(answer, contracts)

    assert failures[0]["id"] == "s2ism-inline"
    assert failures[0]["matching_unit_count"] == 2
    assert failures[0]["cited_unit_count"] == 1


def test_inline_citation_contract_accepts_reused_marker_in_detailed_body():
    contracts = [
        {
            "id": "s2ism-inline",
            "termGroups": [
                ["s2ISM", "structured detection"],
                ["super-resolution", "optical sectioning"],
            ],
            "minCitedUnits": 2,
        }
    ]
    answer = (
        "s2ISM structured detection provides super-resolution [1](#kb-cite-1).\n"
        "The detailed body says s2ISM improves optical sectioning [1](#kb-cite-1)."
    )

    assert eval_mod._inline_citation_contract_failures(answer, contracts) == []


def test_quality_term_matching_treats_s2ism_superscript_as_same_method() -> None:
    assert eval_mod._contains_term("s²ISM structured detection", "s2ISM")
    assert eval_mod._contains_term("s2ISM structured detection", "s²ISM")
    assert eval_mod._contains_term("s₂ISM structured detection", "s2ISM")


def test_refs_payload_full_state_rejects_fast_or_pending_cards():
    assert _refs_payload_is_full({"9": {"payload_mode": "full", "render_status": "full"}}, user_msg_id=9)
    assert not _refs_payload_is_full({"9": {"payload_mode": "fast", "render_status": "fast"}}, user_msg_id=9)
    assert not _refs_payload_is_full({"9": {"payload_mode": "pending", "pending": True}}, user_msg_id=9)


def test_refs_payload_convergence_waits_for_answer_aligned_card_copy() -> None:
    generic = {
        "9": {
            "payload_mode": "full",
            "render_status": "full",
            "hits": [
                {
                    "ui_meta": {
                        "display_name": "Paper.pdf",
                        "summary_line": "A grounded summary long enough for the card.",
                        "why_line": "",
                    }
                }
            ],
        }
    }
    aligned = {
        "9": {
            **generic["9"],
            "hits": [
                {
                    "ui_meta": {
                        "display_name": "Paper.pdf",
                        "summary_line": "A grounded summary long enough for the card.",
                        "why_line": "This passage establishes the exact role used by the answer.",
                    }
                }
            ],
        }
    }

    assert not _refs_payload_is_converged_for_case(
        generic,
        user_msg_id=9,
        expected={"requireRefsReady": True},
    )
    assert _refs_payload_is_converged_for_case(
        aligned,
        user_msg_id=9,
        expected={"requireRefsReady": True},
    )


def test_refs_payload_terminal_distinguishes_finished_bad_copy_from_pending_work() -> None:
    terminal_bad = {
        "9": {
            "payload_mode": "full",
            "render_status": "full",
            "polish_status": "heuristic",
            "hits": [{"ui_meta": {"polish_status": "heuristic", "why_line": ""}}],
        }
    }
    pending = {
        "9": {
            "payload_mode": "full",
            "render_status": "full",
            "polish_status": "pending",
            "hits": [{"ui_meta": {"polish_status": "pending"}}],
        }
    }

    assert _refs_payload_is_terminal_for_case(terminal_bad, user_msg_id=9)
    assert not _refs_payload_is_terminal_for_case(pending, user_msg_id=9)


def test_timing_summary_reports_user_visible_percentiles() -> None:
    timing = _timing_summary(
        [
            {"first_answer_ms": 100, "answer_complete_ms": 500, "latency_ms": 900},
            {"first_answer_ms": 300, "answer_complete_ms": 700, "cards_complete_ms": 800, "latency_ms": 1100},
            {"first_answer_ms": 200, "answer_complete_ms": 600, "cards_complete_ms": 1000, "latency_ms": 1000},
        ]
    )

    assert timing["first_answer_ms"] == {"count": 3, "p50": 200.0, "p95": 290.0, "max": 300.0}
    assert timing["cards_complete_ms"] == {"count": 2, "p50": 900.0, "p95": 990.0, "max": 1000.0}


def test_quality_contract_waits_for_full_reference_cards() -> None:
    assert _case_requires_full_refs_wait({"requireRefsReady": True})
    assert _case_requires_full_refs_wait({"requirePolishStatus": True})
    assert _case_requires_full_refs_wait({"requireCitationShelfQuality": True})
    assert _case_requires_full_refs_wait({"requireCitationCardQuality": True})
    assert _case_requires_full_refs_wait({"maxCardsCompleteMs": 30000})
    assert not _case_requires_full_refs_wait({"minRefHits": 3})


def test_failed_generation_does_not_wait_for_full_reference_cards() -> None:
    expected = {"maxCardsCompleteMs": 30000}

    assert _generation_should_wait_for_full_refs(
        {"done": True, "status": "done"},
        expected,
    )
    assert not _generation_should_wait_for_full_refs(
        {"done": True, "status": "error", "error": "not_found"},
        expected,
    )


def _case_by_id(fixture, case_id: str):
    for case in fixture.cases:
        if case.get("id") == case_id:
            return case
    raise AssertionError(f"missing case {case_id}")


def test_assistant_message_by_id_prefers_refetched_converged_message():
    messages = [
        {"id": 10, "role": "assistant", "content": "old"},
        {"id": 11, "role": "assistant", "content": "target", "meta": {"ready": True}},
        {"id": 12, "role": "assistant", "content": "newer but unrelated"},
    ]

    msg = _assistant_message_by_id({"messages": messages}, 11)

    assert msg["content"] == "target"
    assert msg["meta"]["ready"] is True


def test_research_qa_fixture_loads_shared_docs_and_cases():
    fixture = load_fixture()

    assert len(fixture.docs) == 22
    assert len(fixture.cases) == 50
    case_ids = {str(item.get("id") or "") for item in fixture.cases}
    assert {
        "spi-roadmap-beginner",
        "cassi-to-3d-sci-lineage",
        "microscopy-methods-map",
        "single-photon-reading-pair",
        "piln-dl-spi-position",
        "simple-baselines-sidd-best",
    }.issubset(case_ids)
    assert source_path_for_doc(fixture, "scinerf").endswith(
        "CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image/"
        "CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.en.md"
    )
    assert source_path_for_doc(fixture, "simple-baselines").endswith(
        "ECCV-2022-Simple Baselines for Image Restoration/"
        "ECCV-2022-Simple Baselines for Image Restoration.en.md"
    )
    assert sum(1 for case in fixture.cases if case.get("sourceGrounded")) == 35
    assert len(fixture.splits["baseline_real_v1"]) == 12
    assert len(fixture.splits["holdout_v1"]) == 23
    assert len(fixture.splits["blind_holdout_v2"]) == 15
    assert set(fixture.splits["baseline_real_v1"]).isdisjoint(fixture.splits["holdout_v1"])
    assert set(fixture.splits["baseline_real_v1"]).isdisjoint(
        fixture.splits["blind_holdout_v2"]
    )
    assert set(fixture.splits["holdout_v1"]).isdisjoint(
        fixture.splits["blind_holdout_v2"]
    )
    assert (
        set(fixture.splits["baseline_real_v1"])
        | set(fixture.splits["holdout_v1"])
        | set(fixture.splits["blind_holdout_v2"])
    ) == case_ids


def test_research_qa_fixture_split_selection_is_stable_and_intersectable():
    fixture = load_fixture()

    holdout = select_fixture_cases(fixture, split_names=["holdout_v1"])
    one_case = select_fixture_cases(
        fixture,
        split_names=["holdout_v1"],
        case_ids={"fdm-vs-3d-video-parallelism"},
    )

    assert [case["id"] for case in holdout] == fixture.splits["holdout_v1"]
    assert [case["id"] for case in one_case] == ["fdm-vs-3d-video-parallelism"]


def test_full_library_acceptance_suite_is_bounded_and_covers_the_corpus():
    fixture = load_fixture()

    cases = select_fixture_cases(fixture, suite_names=["full_library_acceptance_v1"])
    one_case = select_fixture_cases(
        fixture,
        suite_names=["live_smoke_v1"],
        case_ids={"scinerf-forward-model-equation"},
    )
    coverage = summarize_suite_coverage(fixture, "full_library_acceptance_v1")

    assert len(cases) == 29
    assert [case["id"] for case in one_case] == ["scinerf-forward-model-equation"]
    assert coverage["case_count"] == 29
    assert coverage["doc_count"] == len(fixture.docs) == 22
    assert coverage["locales"] == ["en", "zh"]
    assert {
        "cross_paper_synthesis",
        "table_lookup",
        "formula_reasoning",
        "figure_architecture",
        "negative_or_insufficient_evidence",
        "system_a",
        "system_b",
        "answer_reference_alignment",
    }.issubset(coverage["coverage"])


def test_fixture_validation_rejects_split_overlap_and_unassigned_cases():
    fixture = ResearchQaFixture(
        db_root="db",
        docs=[{"id": "paper"}],
        cases=[
            {
                "id": "alpha",
                "question": "What is alpha?",
                "acceptance": ["one", "two"],
                "docIds": ["paper"],
                "expected": {
                    "requiredAnswerTerms": ["alpha"],
                    "requiredRefDocIds": ["paper"],
                    "requiredCitationDocIds": ["paper"],
                },
            },
            {
                "id": "beta",
                "question": "What is beta?",
                "acceptance": ["one", "two"],
                "docIds": ["paper"],
                "expected": {
                    "requiredAnswerTerms": ["beta"],
                    "requiredRefDocIds": ["paper"],
                    "requiredCitationDocIds": ["paper"],
                },
            },
        ],
        forbidden_phrases=[],
        splits={"baseline": ["alpha"], "holdout": ["alpha"]},
    )

    errors = validate_fixture_contracts(fixture)

    assert "case alpha appears in multiple splits: baseline, holdout" in errors
    assert "fixture cases missing a split: beta" in errors


def test_fixture_validation_rejects_incomplete_full_library_suite():
    fixture = ResearchQaFixture(
        db_root="db",
        docs=[{"id": "alpha"}, {"id": "beta"}],
        cases=[
            {
                "id": "alpha-question",
                "question": "What is alpha?",
                "docIds": ["alpha"],
                "acceptance": ["one", "two"],
                "expected": {},
            }
        ],
        forbidden_phrases=[],
        suites={
            "full_library_acceptance_v1": {
                "caseIds": ["alpha-question"],
                "requireAllDocs": True,
                "requiredLocales": ["zh", "en"],
                "requiredCoverage": ["formula_reasoning"],
                "coverage": {},
            }
        },
    )

    errors = validate_fixture_contracts(fixture)

    assert "suite full_library_acceptance_v1 does not cover all docs: beta" in errors
    assert "suite full_library_acceptance_v1 missing locales: zh" in errors
    assert "suite full_library_acceptance_v1 missing required coverage: formula_reasoning" in errors


def test_source_grounded_contracts_match_page_marked_markdown(tmp_path):
    source_dir = tmp_path / "paper"
    source_dir.mkdir()
    (source_dir / "paper.en.md").write_text(
        "<!-- kb_page: 1 -->\n## Abstract\nalpha evidence\n"
        "<!-- kb_page: 2 -->\n## Method\nbeta evidence\n",
        encoding="utf-8",
    )
    fixture = ResearchQaFixture(
        db_root=str(tmp_path),
        docs=[{"id": "paper", "dir": "paper", "file": "paper"}],
        cases=[
            {
                "id": "grounded",
                "sourceGrounded": True,
                "expected": {
                    "claimEvidenceContracts": [
                        {"id": "claim", "docId": "paper", "sourcePage": 2, "evidenceTerms": ["beta"]}
                    ],
                    "requiredLocateContracts": [
                        {"id": "locate", "docId": "paper", "sourcePage": 2, "evidenceTerms": ["evidence"]}
                    ],
                },
            }
        ],
        forbidden_phrases=[],
    )

    assert validate_fixture_sources(fixture, db_root=tmp_path) == []

    fixture.cases[0]["expected"]["claimEvidenceContracts"][0]["evidenceTerms"] = ["alpha"]
    errors = validate_fixture_sources(fixture, db_root=tmp_path)
    assert len(errors) == 1
    assert "page 2 missing evidence terms: alpha" in errors[0]


def test_source_grounded_contract_ignores_inline_math_delimiters(tmp_path):
    source_dir = tmp_path / "paper"
    source_dir.mkdir()
    (source_dir / "paper.en.md").write_text(
        "<!-- kb_page: 2 -->\nEach pixel is modulated on $p$ frequencies simultaneously.\n",
        encoding="utf-8",
    )
    fixture = ResearchQaFixture(
        db_root=str(tmp_path),
        docs=[{"id": "paper", "dir": "paper", "file": "paper"}],
        cases=[
            {
                "id": "grounded-math",
                "sourceGrounded": True,
                "expected": {
                    "claimEvidenceContracts": [
                        {
                            "id": "claim",
                            "docId": "paper",
                            "sourcePage": 2,
                            "evidenceTerms": ["p frequencies simultaneously"],
                        }
                    ],
                    "requiredLocateContracts": [],
                },
            }
        ],
        forbidden_phrases=[],
    )

    assert validate_fixture_sources(fixture, db_root=tmp_path) == []


def test_source_page_contract_rejects_missing_or_wrong_citation_page():
    assert _detail_matches_source_page({"page_start": 2}, 2)
    assert _detail_matches_source_page({"page_start": 2, "page_end": 4}, 3)
    assert not _detail_matches_source_page({"page_start": 3}, 2)
    assert not _detail_matches_source_page({"location_label": "Method"}, 2)


def test_multilingual_term_groups_accept_one_reviewed_surface_per_claim():
    groups = [["Poisson", "泊松"], ["crosstalk", "串扰"], ["dark count rate", "暗计数"]]

    assert _missing_term_groups("模型包含泊松噪声、串扰和暗计数。", groups) == []
    assert _missing_term_groups("模型只包含泊松噪声。", groups) == groups[1:]


def test_design_layer_alias_accepts_natural_chinese_negation():
    groups = [["不同层面", "不是同一层面", "different design layers"]]

    assert _missing_term_groups("两种方法并非同一层面的采样策略。", groups) == []


def test_term_groups_ignore_spacing_between_numbers_and_cjk_units():
    groups = [["four", "4 个", "四个"], ["8 frames per second", "8 帧/秒", "8 fps"]]

    assert _missing_term_groups("系统使用4个探测器，速度约为8帧/秒。", groups) == []


def test_term_groups_accept_negated_same_layer_paraphrase():
    groups = [["不同层面", "不是同一层面", "different design layers"]]

    assert _missing_term_groups("两者不属于同一层面的采样策略。", groups) == []


def test_claim_contract_can_validate_split_claims_against_full_response():
    fixture = load_fixture()
    case = _case_by_id(fixture, "three-d-video-real-time-mechanism")
    contract = dict((case.get("expected") or {})["claimEvidenceContracts"][0])
    source_path = source_path_for_doc(fixture, "3d-sp-video")
    answer = "系统使用四个探测器，在 64 \\times 64 分辨率下约为 8 帧/秒。"
    details = [
        {
            "citation_route": "system_a",
            "source_path": source_path,
            "page_start": 2,
            "page_end": 2,
            "answer_claim": "该方法支持动态场景重建。",
            "evidence_quote": (
                "The system uses four spatially-separated detectors and reconstructs video at "
                "8 frames per second for 64 \\times 64 pixels."
            ),
        }
    ]

    assert _claim_evidence_contract_failures(fixture, details, [contract], answer=answer) == []


def test_claim_contract_requires_terms_in_visible_card_evidence_not_hidden_raw_metadata():
    fixture = load_fixture()
    contract = {
        "id": "visible-compound-evidence",
        "docId": "qclfm",
        "route": "system_a",
        "evidenceTerms": ["ray tracing", "wave propagation"],
    }
    detail = {
        "citation_route": "system_a",
        "source_path": source_path_for_doc(fixture, "qclfm"),
        "evidence_quote": "First, photon trajectories are reconstructed through ray tracing.",
        "card_evidence": "First, photon trajectories are reconstructed through ray tracing.",
        "raw": (
            "First, photon trajectories are reconstructed through ray tracing. "
            "Second, diffraction is reversed through wave propagation."
        ),
        "support_relation": "ray tracing is followed by wave propagation",
    }

    failures = _claim_evidence_contract_failures(fixture, [detail], [contract])
    assert [item["id"] for item in failures] == ["visible-compound-evidence"]

    detail["card_evidence"] = str(detail["raw"])
    detail["evidence_quote"] = str(detail["raw"])
    assert _claim_evidence_contract_failures(fixture, [detail], [contract]) == []


def test_fdm_claim_contract_uses_source_bound_full_plan_evidence() -> None:
    fixture = load_fixture()
    case = _case_by_id(fixture, "fdm-vs-3d-video-parallelism")
    contract = next(
        item
        for item in (case.get("expected") or {}).get("claimEvidenceContracts") or []
        if item.get("id") == "fdm-parallel-encoding-layer"
    )
    source_path = source_path_for_doc(fixture, "fdm-metamaterials")
    compact = (
        "Each pixel of the SLM is modulated with either 0 or \\pi phase on p frequencies "
        "simultaneously. The modulated light is multiplexed into a single-pixel detector. "
        "The signal is then demodulated by a number (p) of LIAs."
    )
    plan_evidence = (
        "The mask values are encoded in the phase of intensity modulation, and thus we require "
        "phase-sensitive detection, in this case provided by a lock-in amplifier (LIA). "
        f"{compact}"
    )
    result = {
        "assistant_message": {
            "content": "Frequency-division methods parallelize the modulation and encoding layer.",
            "meta": {
                "paper_guide_contracts": {
                    "render_packet": {
                        "cite_details": [
                            {
                                "citation_route": "system_a",
                                "source_path": source_path,
                                "source_name": "FDM.pdf",
                                "heading_path": "B. Encoding",
                                "page_start": 2,
                                "page_end": 2,
                                "evidence_quote": compact,
                            }
                        ]
                    }
                },
                "answer_quality": {
                    "citation_plan": {
                        "slots": [
                            {
                                "preferred_system": "system_a",
                                "source_path": source_path,
                                "heading_path": "B. Encoding",
                                "page_start": 2,
                                "page_end": 2,
                                "evidence_quote": plan_evidence,
                            }
                        ]
                    }
                },
            },
        }
    }

    details = eval_mod._citation_details(result)

    assert details[0]["citation_plan_evidence_quote"] == plan_evidence
    assert _claim_evidence_contract_failures(
        fixture,
        details,
        [contract],
        answer="Frequency-division methods parallelize the modulation and encoding layer.",
    ) == []

    result["assistant_message"]["meta"]["answer_quality"]["citation_plan"]["slots"][0][
        "source_path"
    ] = source_path_for_doc(fixture, "3d-sp-video")
    unmatched = eval_mod._citation_details(result)
    assert "citation_plan_evidence_quote" not in unmatched[0]


def test_claim_level_citation_gate_rejects_extra_uncited_mechanism_claim():
    fixture = load_fixture()
    case = {
        "id": "claim-level-citation-gate",
        "question": "What does the method do?",
        "expected": {
            "requireClaimLevelCitations": True,
            "maxUncitedHighRiskClaims": 0,
        },
    }
    result = {
        "answer": (
            "The paper models SPAD crosstalk and dark count rate [1].\n\n"
            "It also improves acquisition speed through a new sampling policy."
        ),
        "status": "done",
        "done": True,
        "refs_payload": {},
        "render_cache": {},
    }

    quality = validate_case(case, fixture, result)
    failed_names = {item["name"] for item in quality["failures"]}

    assert "answer_high_risk_claims_are_cited" in failed_names


def test_claim_level_citation_gate_accepts_cited_mechanism_claims():
    fixture = load_fixture()
    case = {
        "id": "claim-level-citation-gate-pass",
        "question": "What does the method do?",
        "expected": {
            "requireClaimLevelCitations": True,
            "maxUncitedHighRiskClaims": 0,
        },
    }
    result = {
        "answer": (
            "The paper models SPAD crosstalk and dark count rate [1].\n\n"
            "It improves reconstruction quality through the physical noise model [1]."
        ),
        "status": "done",
        "done": True,
        "refs_payload": {},
        "render_cache": {},
    }

    quality = validate_case(case, fixture, result)
    failed_names = {item["name"] for item in quality["failures"]}

    assert "answer_high_risk_claims_are_cited" not in failed_names


def test_research_qa_fixture_enforces_system_b_trace_policy():
    fixture = load_fixture()
    policy_case_ids = {
        "scinerf-admm-origin",
        "cassi-to-3d-sci-lineage",
    }

    for case_id in policy_case_ids:
        expected = _case_by_id(fixture, case_id).get("expected") or {}
        assert expected.get("requireSystemB") or int(expected.get("minSystemBCount") or 0) >= 1
        assert expected.get("requireSystemBTraceComplete") is True
        assert expected.get("forbidSystemBAnswerContextOnly") is True
        assert expected.get("maxSystemBNeedsReviewCount") == 0
        assert expected.get("maxSystemBAnswerContextOnlyCount") == 0
        assert expected.get("maxSystemBReferenceIndexFallbackCount") == 0
        assert expected.get("minSystemBCompleteRate") == 1.0

    scinerf_expected = _case_by_id(fixture, "scinerf-admm-origin").get("expected") or {}
    assert scinerf_expected.get("maxSystemBCount") == 1

    lineage_expected = _case_by_id(fixture, "cassi-to-3d-sci-lineage").get("expected") or {}
    assert lineage_expected.get("requiredSystemBDocIds") == ["scigs"]
    assert "Theory, Algorithms, and Applications" in lineage_expected.get("requiredSystemBTerms", [])

    roadmap_expected = _case_by_id(fixture, "spi-roadmap-beginner").get("expected") or {}
    assert int(roadmap_expected.get("minSystemBCount") or 0) == 0
    assert roadmap_expected.get("maxSystemBCount") == 0
    assert roadmap_expected.get("requireSystemBTraceComplete") is not True
    assert roadmap_expected.get("allowedRefDocIds") == ["spi-prospects", "dl-spi-review", "hsi-fsi"]
    assert roadmap_expected.get("allowedCitationDocIds") == ["spi-prospects", "dl-spi-review", "hsi-fsi"]
    assert roadmap_expected.get("maxRefHits") == 3
    assert roadmap_expected.get("maxRefDocCount") == 3
    assert roadmap_expected.get("maxCitationDocCount") == 3
    assert roadmap_expected.get("maxCitationCount") == 3

    ordinary_case_ids = {
        "spi-roadmap-beginner",
        "microscopy-methods-map",
        "single-photon-reading-pair",
        "piln-dl-spi-position",
    }
    for case_id in ordinary_case_ids:
        expected = _case_by_id(fixture, case_id).get("expected") or {}
        assert int(expected.get("minSystemBCount") or 0) == 0
        assert expected.get("maxSystemBCount") == 0
        assert expected.get("requireSystemBTraceComplete") is not True


def test_research_qa_fixture_cases_have_acceptance_contracts():
    fixture = load_fixture()
    assert fixture.cases
    doc_ids = {str(item.get("id") or "") for item in fixture.docs}

    for case in fixture.cases:
        case_id = str(case.get("id") or "")
        question = str(case.get("question") or "").strip()
        expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
        acceptance = case.get("acceptance") if isinstance(case.get("acceptance"), list) else []
        case_doc_ids = [str(item) for item in case.get("docIds") or [] if str(item or "").strip()]

        assert case_id, "case id is required"
        assert question, f"{case_id} must include a natural research question"
        assert len(acceptance) >= 2, f"{case_id} must describe user-facing acceptance criteria"
        assert expected, f"{case_id} must include machine-checkable expectations"
        assert case_doc_ids, f"{case_id} must bind at least one library document"
        assert not (set(case_doc_ids) - doc_ids), f"{case_id} references unknown docs"
        assert (
            expected.get("requiredAnswerTerms") or expected.get("requiredAnswerTermGroups")
        ), f"{case_id} must define answer terms or answer term groups"
        assert expected.get("requiredRefDocIds"), f"{case_id} must define required refs docs"
        assert expected.get("requiredCitationDocIds"), f"{case_id} must define required citation docs"


def test_research_qa_fixture_covers_six_real_user_quality_focuses():
    fixture = load_fixture()

    assert validate_fixture_contracts(fixture) == []
    focused = {
        str(case.get("evaluationFocus") or ""): case
        for case in fixture.cases
        if str(case.get("evaluationFocus") or "").strip()
    }
    assert set(focused) == {
        "paper_summary",
        "method_detail",
        "method_comparison",
        "multi_paper_synthesis",
        "upstream_reference",
        "scope_boundary",
    }
    for focus, case in focused.items():
        expected = case.get("expected") or {}
        assert expected.get("allowedRefDocIds"), focus
        assert expected.get("claimEvidenceContracts"), focus
        assert expected.get("requiredRouteCounts") is not None, focus
        assert expected.get("requiredLocateContracts"), focus


def test_reviewed_real_paper_replay_passes_every_focused_quality_contract():
    fixture = load_fixture()

    summary = evaluate_replay_rows(fixture, load_replay())

    assert summary["ok"], summary["errors"]
    assert summary["total"] == 6
    assert summary["passed"] == 6
    assert summary["failed"] == 0


def test_admm_live_style_hallucination_is_rejected_even_when_keywords_and_system_b_exist():
    fixture = load_fixture()
    case = _case_by_id(fixture, "scinerf-admm-origin")
    scinerf_path = source_path_for_doc(fixture, "scinerf")
    scigs_path = source_path_for_doc(fixture, "scigs")
    result = {
        "status": "done",
        "done": True,
        "user_msg_id": 901,
        "assistant_message": {
            "role": "assistant",
            "content": (
                "ADMM 不是 SCINeRF 的新东西，而是已有方法。"
                "作者是在利用 ADMM 作为解决某个子问题的工具。"
                "ADMM 很可能被用作一个优化策略。"
            ),
            "cite_details": [
                {
                    "num": 4,
                    "anchor": "scinerf-r4",
                    "source_path": scinerf_path,
                    "source_name": "SCINeRF",
                    "is_inpaper": True,
                    "title": "Distributed Optimization and Statistical Learning via ADMM",
                    "authors": "Stephen Boyd; Neal Parikh",
                    "venue": "Foundations and Trends in Machine Learning",
                    "year": "2011",
                    "raw": "Boyd et al. Distributed Optimization and Statistical Learning via ADMM, 2011.",
                    "heading_path": "2. Related Work / Snapshot Compressive Imaging",
                    "location_label": "Related Work / ref 4",
                    "answer_claim": "ADMM 不是 SCINeRF 原创，而是已有优化方法。",
                    "citation_context": "Most existing methods employ ADMM [4].",
                    "citation_context_source": "source_markdown",
                    "upstream_work_role": "This is upstream ADMM background.",
                    "user_question_relation": "It predates SCINeRF.",
                    "system_b_trace_complete": True,
                    "system_b_trace_score": 0.9,
                    "system_b_trace_source": "source_markdown",
                    "routing_reason": "structured_cite",
                }
            ],
        },
        "refs_payload": {
            "901": {
                "display_state": "ready",
                "hits": [
                    {
                        "text": "Most existing SCI methods employ ADMM.",
                        "meta": {"source_path": scinerf_path, "ref_pack_state": "ready"},
                        "ui_meta": {
                            "display_name": "SCINeRF",
                            "source_path": scinerf_path,
                            "summary_line": "Related Work describes ADMM as prior SCI reconstruction background.",
                            "why_line": "This evidence answers whether ADMM is a new SCINeRF contribution.",
                        },
                    },
                    {
                        "text": "SCIGS reconstructs dynamic scenes with 3D Gaussian splatting.",
                        "meta": {"source_path": scigs_path, "ref_pack_state": "ready"},
                        "ui_meta": {
                            "display_name": "SCIGS",
                            "source_path": scigs_path,
                            "summary_line": "SCIGS is an unrelated extra retrieval result for this focused question.",
                            "why_line": "It should not have been included in a current-paper ADMM origin answer.",
                        },
                    },
                ],
            }
        },
    }

    quality = validate_case(case, fixture, result)
    failed_names = {item["name"] for item in quality["failures"]}

    assert quality["ok"] is False
    assert "answer_avoids_forbidden_claims" in failed_names
    assert "refs_avoid_unexpected_docs" in failed_names
    assert "citations_match_required_routes" in failed_names
    assert "claims_have_matching_evidence" in failed_names
    assert "citations_have_expected_locators" in failed_names


def test_research_qa_fixture_real_regression_cases_require_card_quality_gates():
    fixture = load_fixture()
    strict_case_ids = {
        "spi-roadmap-beginner",
        "cassi-to-3d-sci-lineage",
        "microscopy-methods-map",
        "single-photon-reading-pair",
        "piln-dl-spi-position",
    }

    for case_id in strict_case_ids:
        expected = _case_by_id(fixture, case_id).get("expected") or {}
        assert expected.get("requireRefsReady") is True
        assert expected.get("requirePolishStatus") is True
        assert expected.get("requireCitationShelfQuality") is True
        assert int(expected.get("minCitationShelfMetadataReadyCount") or 0) >= 1
        assert int(expected.get("minCitationShelfExportReadyCount") or 0) >= 1
        assert int(expected.get("minCitationShelfDoiCount") or 0) >= 1
        assert int(expected.get("minCitationShelfSourceClickCount") or 0) >= 1
        assert expected.get("maxCitationShelfMetadataReviewCount") == 0
        assert "full" in expected.get("allowedRefPolishStatuses", [])
        assert "heuristic" in expected.get("allowedRefPolishStatuses", [])
        assert int(expected.get("minRefHits") or 0) >= 2
        assert int(expected.get("minCitationCount") or 0) >= 2
        assert int(expected.get("minCitationDocCount") or 0) >= 1


def test_research_qa_fixture_has_claim_level_grounding_gate_on_core_cases():
    fixture = load_fixture()
    strict_case_ids = {
        "single-photon-pidl",
        "cassi-to-3d-sci-lineage",
        "microscopy-methods-map",
        "single-photon-reading-pair",
        "piln-dl-spi-position",
        "spi-prospects-when-use-single-pixel",
        "s2ism-three-way-tradeoff",
        "pidl-real-spad-noise",
    }

    for case_id in strict_case_ids:
        expected = _case_by_id(fixture, case_id).get("expected") or {}
        assert expected.get("requireClaimLevelCitations") is True
        assert expected.get("maxUncitedHighRiskClaims") == 0


def test_validate_case_accepts_grounded_system_b_answer():
    fixture = load_fixture()
    fixture_case = _case_by_id(fixture, "scinerf-admm-origin")
    case = {
        **fixture_case,
        "expected": {
            "requiredAnswerTerms": ["ADMM", "不是", "已有"],
            "requiredRefDocIds": ["scinerf"],
            "requiredCitationDocIds": ["scinerf"],
            "requireSystemB": True,
            "requiredSystemBTerms": ["ADMM", "Distributed optimization"],
            "requireSystemBTraceComplete": True,
        },
    }
    scinerf_path = source_path_for_doc(fixture, "scinerf")
    answer = "ADMM 不是 SCINeRF 作者新提出的模块，而是已有的优化框架；论文只是把它放在相关工作里说明来源。"
    result = {
        "status": "done",
        "done": True,
        "user_msg_id": 101,
        "assistant_message": {
            "role": "assistant",
            "content": answer,
            "cite_details": [
                {
                    "num": 4,
                    "anchor": "admm-r4",
                    "source_path": scinerf_path,
                    "source_name": "SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
                    "is_inpaper": True,
                    "title": "Distributed optimization and statistical learning via the alternating direction method of multipliers",
                    "raw": "Distributed optimization and statistical learning via ADMM.",
                    "heading_path": "SCINeRF / 2. Related Work / Snapshot Compressive Imaging",
                    "location_label": "SCINeRF / 2. Related Work / Snapshot Compressive Imaging",
                    "answer_claim": "ADMM is existing optimization background rather than a new SCINeRF contribution.",
                    "citation_context": "Most existing snapshot compressive imaging methods employ ADMM-based optimization.",
                    "citation_context_source": "source_markdown",
                    "upstream_work_role": "This upstream paper provides the ADMM optimization framework used as prior work.",
                    "user_question_relation": "It shows ADMM is existing background rather than a new SCINeRF contribution.",
                    "system_b_trace_complete": True,
                    "system_b_trace_score": 0.86,
                    "system_b_trace_source": "source_markdown",
                }
            ],
        },
        "refs_payload": {
            "101": {
                "hits": [
                    {
                        "text": "SCINeRF discusses ADMM in Related Work.",
                        "ui_meta": {
                            "source_path": scinerf_path,
                            "display_name": "SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
                            "summary_line": "这段 Related Work 把 ADMM 放在已有优化框架脉络里，而不是当作本文创新点。",
                            "why_line": "用户问 ADMM 的来源，这张卡定位到论文解释该方法背景和引用来源的位置。",
                        },
                    }
                ]
            }
        },
    }

    quality = validate_case(case, fixture, result)

    assert quality["ok"] is True
    assert quality["system_b_count"] == 1


def test_validate_case_rejects_system_b_for_multi_doc_ordinary_question():
    fixture = load_fixture()
    case = _case_by_id(fixture, "spi-roadmap-beginner")
    spi_path = source_path_for_doc(fixture, "spi-prospects")
    dl_path = source_path_for_doc(fixture, "dl-spi-review")
    hsi_path = source_path_for_doc(fixture, "hsi-fsi")
    answer = (
        "A good single-pixel imaging route is: first read the SPI principles review, "
        "then read Hadamard/Fourier coding choices, and finally read the deep learning SPI review "
        "for upstream single-pixel imaging background [R18](#spi-r18)."
    )
    result = {
        "status": "done",
        "done": True,
        "user_msg_id": 301,
        "assistant_message": {
            "role": "assistant",
            "content": answer,
            "cite_details": [
                {
                    "num": 1,
                    "anchor": "spi-a1",
                    "source_path": spi_path,
                    "source_name": "Principles and prospects for single-pixel imaging",
                    "heading_path": "Abstract / Principles",
                    "evidence_quote": "Single-pixel imaging measures correlations between a scene and projected patterns.",
                    "answer_claim": "Start from the principles review to build the measurement model.",
                    "support_relation": "The evidence explains the core SPI measurement principle.",
                },
                {
                    "num": 2,
                    "anchor": "spi-a2",
                    "source_path": dl_path,
                    "source_name": "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
                    "heading_path": "Deep learning SPI / Overview",
                    "evidence_quote": "Deep learning methods improve reconstruction quality but face data and generalization risks.",
                    "answer_claim": "The deep learning review should be read after the foundations.",
                    "support_relation": "The evidence gives both benefits and limitations for DL-SPI.",
                },
                {
                    "num": 3,
                    "anchor": "spi-a3",
                    "source_path": hsi_path,
                    "source_name": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
                    "heading_path": "Experiment design / Coding choice",
                    "evidence_quote": "Hadamard and Fourier single-pixel imaging differ in measurement efficiency and noise behavior.",
                    "answer_claim": "Coding choice is the practical bridge from principles to experiments.",
                    "support_relation": "The evidence explains why the roadmap should include coding strategies.",
                },
                {
                    "num": 18,
                    "anchor": "spi-r18",
                    "is_inpaper": True,
                    "source_path": dl_path,
                    "source_name": "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
                    "title": "Principles and prospects for single-pixel imaging",
                    "authors": "Edgar M P; Gibson G M; Padgett M J",
                    "venue": "Nature Photonics",
                    "year": "2019",
                    "doi": "10.1038/s41566-018-0300-7",
                    "raw": "Edgar M P, Gibson G M, Padgett M J. Principles and prospects for single-pixel imaging. Nature Photonics, 2019. doi:10.1038/s41566-018-0300-7.",
                    "heading_path": "Deep learning SPI review / References",
                    "location_label": "Deep learning SPI review / References",
                    "answer_claim": "The reading route should include upstream single-pixel imaging foundations.",
                    "citation_context": "The review cites earlier single-pixel imaging foundations when introducing the field.",
                    "citation_context_source": "source_markdown",
                    "upstream_work_role": "This upstream review is the foundation source for the SPI reading route.",
                    "user_question_relation": "It gives the user a concrete upstream source to open after the roadmap answer.",
                    "system_b_trace_complete": True,
                    "system_b_trace_score": 0.82,
                    "system_b_trace_source": "source_markdown",
                },
            ],
        },
        "refs_payload": {
            "301": {
                "display_state": "ready",
                "hits": [
                    {
                        "text": "single-pixel imaging principles",
                        "meta": {"source_path": spi_path, "ref_pack_state": "ready"},
                        "ui_meta": {
                            "source_path": spi_path,
                            "display_name": "Principles and prospects for single-pixel imaging",
                            "summary_line": "This card establishes the core single-pixel imaging measurement route.",
                            "why_line": "The user asks for a reading roadmap, so this evidence should be the starting point.",
                            "polish_status": "full",
                        },
                    },
                    {
                        "text": "deep learning SPI review",
                        "meta": {"source_path": dl_path, "ref_pack_state": "ready"},
                        "ui_meta": {
                            "source_path": dl_path,
                            "display_name": "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
                            "summary_line": "This card explains where deep learning expands SPI and where it remains risky.",
                            "why_line": "It lets the answer include both benefits and limitations rather than a hype summary.",
                            "polish_status": "heuristic",
                        },
                    },
                    {
                        "text": "Hadamard versus Fourier SPI",
                        "meta": {"source_path": hsi_path, "ref_pack_state": "ready"},
                        "ui_meta": {
                            "source_path": hsi_path,
                            "display_name": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
                            "summary_line": "This card turns the route into a coding and measurement-budget choice.",
                            "why_line": "It is the practical bridge from principles to experiment design.",
                            "polish_status": "full",
                        },
                    },
                ],
            }
        },
    }

    quality = validate_case(case, fixture, result)

    assert quality["ok"] is False
    assert quality["ref_hit_count"] == 3
    assert quality["system_b_count"] == 1
    assert "citations_max_count" in {item["name"] for item in quality["failures"]}
    assert "system_b_max_count" in {item["name"] for item in quality["failures"]}


def test_citation_details_deduplicates_message_and_render_packet_cards():
    detail = {
        "num": 4,
        "citation_route": "system_b",
        "is_inpaper": True,
        "source_path": "scinerf.md",
        "title": "Distributed Optimization via ADMM",
        "doi": "10.1561/2200000016",
    }
    result = {
        "assistant_message": {
            "content": "ADMM is prior work [[CITE:s1234abcd:4]].",
            "cite_details": [detail],
            "meta": {
                "paper_guide_contracts": {
                    "render_packet": {"cite_details": [dict(detail)]},
                }
            },
        }
    }

    details = eval_mod._citation_details(result)

    assert details == [detail]


def test_citation_details_preserves_distinct_occurrence_cards_from_same_paper():
    benefit = {
        "num": 1,
        "anchor": "kb-cite-benefit-1",
        "citation_route": "system_a",
        "source_path": "dl-spi.md",
        "title": "DL SPI review",
        "doi": "10.1000/dl-spi",
        "heading_path": "Abstract",
        "evidence_quote": "Exceptional reconstruction quality and fast reconstruction speed.",
        "occurrence_specific": True,
    }
    risk = {
        **benefit,
        "anchor": "kb-cite-risk-1",
        "heading_path": "4. Strategy and Advantages",
        "evidence_quote": "Prolonged training duration and limited generalization.",
    }
    result = {
        "assistant_message": {
            "content": "Benefit [1]. Risk [1].",
            "cite_details": [benefit, risk],
            "meta": {
                "paper_guide_contracts": {
                    "render_packet": {"cite_details": [dict(benefit), dict(risk)]},
                }
            },
        }
    }

    details = eval_mod._citation_details(result)

    assert [detail["anchor"] for detail in details] == [
        "kb-cite-benefit-1",
        "kb-cite-risk-1",
    ]


def test_validate_case_rejects_unready_unpolished_refs_without_forcing_ordinary_system_b():
    fixture = load_fixture()
    case = _case_by_id(fixture, "spi-roadmap-beginner")
    spi_path = source_path_for_doc(fixture, "spi-prospects")
    result = {
        "status": "done",
        "done": True,
        "user_msg_id": 302,
        "assistant_message": {
            "role": "assistant",
            "content": "Read a single-pixel imaging review first, then maybe deep learning and Hadamard later.",
            "cite_details": [
                {"source_path": spi_path, "source_name": "Principles and prospects for single-pixel imaging"}
            ],
        },
        "refs_payload": {
            "302": {
                "display_state": "pending",
                "hits": [
                    {
                        "text": "single-pixel imaging principles",
                        "meta": {"source_path": spi_path, "ref_pack_state": "pending"},
                        "ui_meta": {
                            "source_path": spi_path,
                            "display_name": "Principles and prospects for single-pixel imaging",
                            "summary_line": "This card is long enough to avoid the short-copy failure.",
                            "why_line": "This explanation is also long enough, but it has no polish status.",
                        },
                    }
                ],
            }
        },
    }

    quality = validate_case(case, fixture, result)
    failed_names = {item["name"] for item in quality["failures"]}

    assert quality["ok"] is False
    assert "refs_min_hit_count" in failed_names
    assert "refs_ready" in failed_names
    assert "refs_card_polish_status" in failed_names
    assert "system_b_min_count" not in failed_names


def test_validate_case_counts_unlinked_system_b_candidates_against_budget():
    fixture = load_fixture()
    case = _case_by_id(fixture, "spi-roadmap-beginner")
    spi_path = source_path_for_doc(fixture, "spi-prospects")
    answer = "Read single-pixel imaging foundations before deep learning and Hadamard coding."
    result = {
        "status": "done",
        "done": True,
        "user_msg_id": 303,
        "assistant_message": {
            "role": "assistant",
            "content": answer,
            "meta": {
                "paper_guide_contracts": {
                    "render_packet": {
                        "rendered_body": answer,
                        "cite_details": [],
                        "unlinked_reference_candidates": [
                            {
                                "title": "Real-time imaging of methane gas leaks using a single-pixel camera",
                                "cite_detail": {
                                    "is_inpaper": True,
                                    "citation_route": "system_b",
                                    "source_path": spi_path,
                                },
                            }
                        ],
                    }
                }
            },
        },
        "refs_payload": {},
    }

    quality = validate_case(case, fixture, result)
    system_b_check = next(
        item for item in quality["checks"] if item["name"] == "system_b_max_count"
    )

    assert system_b_check["ok"] is False
    assert system_b_check["detail"]["citations"] == 0
    assert system_b_check["detail"]["unlinked_candidates"] == 1


def test_validate_case_rejects_template_answer_and_missing_system_b():
    fixture = load_fixture()
    case = _case_by_id(fixture, "scinerf-admm-origin")
    scinerf_path = source_path_for_doc(fixture, "scinerf")
    result = {
        "status": "done",
        "done": True,
        "user_msg_id": 102,
        "assistant_message": {
            "role": "assistant",
            "content": "The paper cites [4] for this point. This is stated in SCINeRF / Related Work.",
            "cite_details": [
                {
                    "source_path": scinerf_path,
                    "source_name": "SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
                    "is_inpaper": False,
                }
            ],
        },
        "refs_payload": {
            "102": {
                "hits": [
                    {
                        "text": "SCINeRF Related Work mentions ADMM.",
                        "ui_meta": {
                            "source_path": scinerf_path,
                            "display_name": "SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
                            "summary_line": "直接定义或解释了 ADMM。",
                            "why_line": "适合作为定位入口。",
                        },
                    }
                ]
            }
        },
    }

    quality = validate_case(case, fixture, result)
    failed_names = {item["name"] for item in quality["failures"]}

    assert quality["ok"] is False
    assert "answer_no_template_phrase" in failed_names
    assert "system_b_present" in failed_names
    assert "refs_card_copy_quality" in failed_names


def test_validate_case_prefers_render_packet_over_raw_content_for_citation_quality():
    fixture = load_fixture()
    fixture_case = _case_by_id(fixture, "scinerf-admm-origin")
    case = {
        **fixture_case,
        "expected": {
            "requiredAnswerTerms": ["ADMM", "不是", "已有"],
            "requiredRefDocIds": ["scinerf"],
            "requiredCitationDocIds": ["scinerf"],
            "requireSystemB": True,
            "requiredSystemBTerms": ["ADMM", "Distributed optimization"],
            "requireSystemBTraceComplete": True,
        },
    }
    scinerf_path = source_path_for_doc(fixture, "scinerf")
    result = {
        "status": "done",
        "done": True,
        "user_msg_id": 106,
        "assistant_message": {
            "role": "assistant",
            "content": "ADMM 不是新东西，而是已有优化框架 [[CITE:s12345678:4]]。",
            "meta": {
                "paper_guide_contracts": {
                    "render_packet": {
                        "rendered_body": "ADMM 不是新东西，而是已有优化框架 [R4](#admm-r4)。",
                        "cite_details": [
                            {
                                "num": 4,
                                "anchor": "admm-r4",
                                "source_path": scinerf_path,
                                "source_name": "SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
                                "is_inpaper": True,
                                "title": "Distributed optimization and statistical learning via the alternating direction method of multipliers",
                                "raw": "Distributed optimization and statistical learning via ADMM.",
                                "heading_path": "SCINeRF / 2. Related Work / Snapshot Compressive Imaging",
                                "location_label": "SCINeRF / 2. Related Work / Snapshot Compressive Imaging",
                                "answer_claim": "ADMM is existing optimization background rather than a new SCINeRF contribution.",
                                "citation_context": "Most existing snapshot compressive imaging methods employ ADMM-based optimization.",
                                "citation_context_source": "source_markdown",
                                "upstream_work_role": "This upstream paper provides the ADMM optimization framework used as prior work.",
                                "user_question_relation": "It shows ADMM is existing background rather than a new SCINeRF contribution.",
                                "system_b_trace_complete": True,
                                "system_b_trace_score": 0.86,
                                "system_b_trace_source": "source_markdown",
                            }
                        ],
                    }
                }
            },
        },
        "refs_payload": {
            "106": {
                "hits": [
                    {
                        "text": "SCINeRF discusses ADMM in Related Work.",
                        "ui_meta": {
                            "source_path": scinerf_path,
                            "display_name": "SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
                            "summary_line": "Related Work explains ADMM as an existing optimization method.",
                            "why_line": "The user asks whether ADMM is new, and this card points to the prior-work context.",
                        },
                    }
                ]
            }
        },
    }

    quality = validate_case(case, fixture, result)
    failed_names = {item["name"] for item in quality["failures"]}

    assert "citation_card_quality" not in failed_names
    assert quality["citation_quality"]["count"] == 1
    assert quality["system_b_audit"]["trace_complete_count"] == 1


def test_validate_case_rejects_unresolved_user_visible_citation_markers() -> None:
    fixture = load_fixture()
    case = {"id": "synthetic-unresolved-citation", "expected": {}}
    result = {
        "status": "done",
        "done": True,
        "assistant_message": {
            "role": "assistant",
            "content": "Raw answer [[CITE:s12345678:5]].",
            "meta": {
                "paper_guide_contracts": {
                    "render_packet": {
                        "rendered_body": "A supported claim [1](#cite-1), but this marker is unresolved [5].",
                    }
                }
            },
        },
    }

    quality = validate_case(case, fixture, result)
    failed_names = {item["name"] for item in quality["failures"]}

    assert quality["ok"] is False
    assert "answer_no_unresolved_citation_markers" in failed_names


def test_validate_case_flags_system_b_audit_policy_failures():
    fixture = load_fixture()
    dl_path = source_path_for_doc(fixture, "dl-spi-review")
    case = {
        "id": "audit-policy",
        "expected": {
            "minSystemBCount": 1,
            "requireSystemBTraceComplete": True,
            "forbidSystemBAnswerContextOnly": True,
            "forbidSystemBReferenceIndexFallback": True,
        },
    }
    result = {
        "status": "done",
        "done": True,
        "assistant_message": {
            "role": "assistant",
            "content": "This answer points to an upstream SPI reference [R18](#spi-r18).",
            "cite_details": [
                {
                    "num": 18,
                    "anchor": "spi-r18",
                    "is_inpaper": True,
                    "source_path": dl_path,
                    "source_name": "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
                    "title": "Principles and prospects for single-pixel imaging",
                    "raw": "Principles and prospects for single-pixel imaging.",
                    "answer_claim": "This answer points to an upstream SPI reference.",
                    "citation_context": "This answer points to an upstream SPI reference.",
                    "citation_context_source": "answer_context",
                    "location_label": "Deep learning SPI review / References",
                    "upstream_work_role": "This upstream paper is a bibliography source.",
                    "system_b_trace_complete": False,
                    "system_b_trace_score": 0.31,
                    "system_b_trace_source": "answer_context",
                    "system_b_trace_flags": ["answer_context_only"],
                    "routing_reason": "reference_index_fallback",
                }
            ],
        },
    }

    quality = validate_case(case, fixture, result)
    failed_names = {item["name"] for item in quality["failures"]}

    assert quality["ok"] is False
    assert "system_b_audit" in failed_names
    assert quality["system_b_audit"]["answer_context_only_count"] == 1
    assert quality["system_b_audit"]["reference_index_fallback_count"] == 1


def test_validate_case_flags_citation_shelf_quality_failures():
    fixture = load_fixture()
    case = {
        "id": "shelf-policy",
        "expected": {
            "requireCitationShelfQuality": True,
            "minCitationShelfQualityCount": 1,
        },
    }
    result = {
        "status": "done",
        "done": True,
        "assistant_message": {
            "role": "assistant",
            "content": "A weak citation [1](#bad-cite).",
            "cite_details": [
                {
                    "num": 1,
                    "anchor": "bad-cite",
                    "source_path": "demo.en.md",
                    "title": "INTRODUCTION",
                    "summary_line": "No summary available",
                }
            ],
        },
    }

    quality = validate_case(case, fixture, result)
    failed_names = {item["name"] for item in quality["failures"]}

    assert quality["ok"] is False
    assert "citation_shelf_quality" in failed_names
    assert quality["citation_shelf_quality"]["count"] == 1
    assert any(
        item["name"] == "shelf_template_phrase_visible"
        for item in quality["citation_shelf_quality"]["failures"]
    )


def test_validate_case_enforces_citation_shelf_metadata_contract():
    fixture = load_fixture()
    case = {
        "id": "shelf-metadata-policy",
        "expected": {
            "requireCitationShelfQuality": True,
            "minCitationShelfQualityCount": 1,
            "minCitationShelfMetadataReadyCount": 1,
            "minCitationShelfExportReadyCount": 1,
            "minCitationShelfDoiCount": 1,
            "minCitationShelfSourceClickCount": 1,
            "maxCitationShelfMetadataReviewCount": 0,
        },
    }
    result = {
        "status": "done",
        "done": True,
        "assistant_message": {
            "role": "assistant",
            "content": "A citation [1](#bad-cite).",
            "cite_details": [
                {
                    "num": 1,
                    "anchor": "bad-cite",
                    "is_inpaper": True,
                    "source_path": "demo.en.md",
                    "title": "Distributed optimization and statistical learning via the alternating direction method of multipliers",
                    "authors": "Stephen Boyd; Neal Parikh; Eric Chu",
                    "venue": "Foundations and Trends in Machine Learning",
                    "year": "2011",
                    "raw": "[1] Boyd et al. Distributed optimization and statistical learning via the alternating direction method of multipliers. doi:10.1561/2200000016",
                    "summary_line": "This source identifies ADMM as an upstream optimization method used by existing snapshot-compressive imaging work.",
                }
            ],
        },
    }

    quality = validate_case(case, fixture, result)
    failed_names = {item["name"] for item in quality["failures"]}

    assert quality["ok"] is False
    assert "citation_shelf_quality" in failed_names
    assert quality["citation_shelf_quality"]["doi_count"] == 1
    assert quality["citation_shelf_quality"]["metadata_ready_count"] == 1
    assert quality["citation_shelf_quality"]["export_ready_count"] == 1
    assert any(
        item["name"] == "shelf_doi_not_promoted"
        for item in quality["citation_shelf_quality"]["failures"]
    )
    check = next(item for item in quality["checks"] if item["name"] == "citation_shelf_quality")
    assert "shelf_1_shelf_doi_not_promoted" in check["detail"]
    assert "shelf_quality_count:0<1" in check["detail"]


def test_validate_case_enforces_citation_shelf_export_readiness():
    fixture = load_fixture()
    case = {
        "id": "shelf-export-policy",
        "expected": {
            "requireCitationShelfQuality": True,
            "minCitationShelfQualityCount": 1,
            "minCitationShelfMetadataReadyCount": 1,
            "minCitationShelfExportReadyCount": 1,
            "minCitationShelfDoiCount": 1,
            "minCitationShelfSourceClickCount": 1,
        },
    }
    result = {
        "status": "done",
        "done": True,
        "assistant_message": {
            "role": "assistant",
            "content": "A citation [1](#weak-export).",
            "cite_details": [
                {
                    "num": 1,
                    "anchor": "weak-export",
                    "is_inpaper": True,
                    "source_path": "demo.en.md",
                    "title": "Single-shot compressive spectral imaging",
                    "raw": "Gehm et al. Single-shot compressive spectral imaging.",
                    "summary_line": "This upstream paper identifies a single-shot compressive spectral imaging baseline for later SCI work.",
                    "summary_quality": {"ok": True, "status": "grounded", "export_ready": True},
                }
            ],
        },
    }

    quality = validate_case(case, fixture, result)
    check = next(item for item in quality["checks"] if item["name"] == "citation_shelf_quality")
    failure_names = {item["name"] for item in quality["citation_shelf_quality"]["failures"]}

    assert quality["ok"] is False
    assert quality["citation_shelf_quality"]["metadata_ready_count"] == 0
    assert quality["citation_shelf_quality"]["export_ready_count"] == 0
    assert quality["citation_shelf_quality"]["summary_export_ready_count"] == 1
    assert "shelf_export_missing_doi" in failure_names
    assert "shelf_export_missing_authors" in failure_names
    assert "shelf_export_ready_count:0<1" in check["detail"]
    assert "shelf_doi_count:0<1" in check["detail"]


def test_research_qa_report_surfaces_system_b_audit(tmp_path):
    report = _build_report(
        [
            {
                "id": "case-a",
                "quality": {
                    "ok": True,
                    "ref_card_quality": {
                        "ok": False,
                        "count": 2,
                        "ok_count": 1,
                        "min_score": 0.8,
                        "failures": [
                            {
                                "index": 2,
                                "name": "ref_card_template_phrase_visible",
                                "field": "summary_line",
                                "detail": "This hit is directly relevant",
                            }
                        ],
                        "warnings": [],
                    },
                    "system_b_audit": {
                        "system_b_total": 2,
                        "trace_complete_count": 1,
                        "needs_review_count": 1,
                        "answer_context_only_count": 1,
                        "reference_index_fallback_count": 1,
                    },
                    "citation_shelf_quality": {
                        "ok": False,
                        "count": 2,
                        "ok_count": 1,
                        "metadata_ready_count": 1,
                        "export_ready_count": 1,
                        "summary_export_ready_count": 1,
                        "doi_count": 1,
                        "source_clickable_count": 1,
                        "min_score": 0.76,
                        "failures": [
                            {
                                "index": 2,
                                "name": "shelf_summary_too_short",
                                "field": "summary",
                                "detail": "short",
                            }
                        ],
                        "warnings": [],
                    },
                },
            }
        ],
        fixture_path=tmp_path / "fixture.json",
        base_url="http://127.0.0.1:8000",
        output_dir=tmp_path,
    )

    assert "## System B Audit" in report
    assert "`case-a`: total=2, complete=1, review=1, answer_context_only=1, fallback=1" in report
    assert "## Ref Card Quality" in report
    assert "`case-a`: ok=False, cards=2, ok_cards=1, failures=1, warnings=0, min_score=0.800" in report
    assert "card 2: ref_card_template_phrase_visible (summary_line) - This hit is directly relevant" in report
    assert "## Citation Shelf Quality" in report
    assert "`case-a`: ok=False, items=2, ok_items=1, metadata_ready=1, export_ready=1, summary_export_ready=1, doi=1, source_clickable=1, failures=1, warnings=0, min_score=0.760" in report
    assert "item 2: shelf_summary_too_short (summary) - short" in report


def test_validate_case_accepts_common_zh_en_synonyms():
    fixture = load_fixture()
    case = _case_by_id(fixture, "single-photon-pidl")
    source_path = source_path_for_doc(fixture, "pidl-single-photon")
    result = {
        "status": "done",
        "done": True,
        "user_msg_id": 103,
        "assistant_message": {
            "role": "assistant",
            "content": "physics-informed deep learning 用 SPAD 物理模型处理多种噪声，并改善单光子成像质量 [1]。",
            "cite_details": [
                {
                    "num": 1,
                    "source_path": source_path,
                    "source_name": "High-resolution single-photon imaging with physics-informed deep learning",
                    "citation_route": "system_a",
                    "is_inpaper": False,
                    "heading_path": "Introduction",
                    "location_label": "Introduction / p. 3",
                    "page_start": 3,
                    "page_end": 3,
                    "answer_claim": "physics-informed deep learning 用 SPAD 物理模型处理多种噪声。",
                    "evidence_quote": "The multi-source physical noise model of SPAD arrays describes real noise.",
                }
            ],
        },
        "refs_payload": {
            "103": {
                "hits": [
                    {
                        "text": "SPAD physical noise model.",
                        "ui_meta": {
                            "source_path": source_path,
                            "display_name": "High-resolution single-photon imaging with physics-informed deep learning",
                            "summary_line": "摘要说明 physics-informed deep learning 如何建模 SPAD 的多种物理噪声。",
                            "why_line": "用户问这种方法帮了什么，这张卡定位到噪声模型和训练数据的证据。",
                        },
                    }
                ]
            }
        },
    }

    quality = validate_case(case, fixture, result)

    assert quality["ok"] is True


def test_validate_case_accepts_scientific_translation_synonyms():
    fixture = load_fixture()
    case = _case_by_id(fixture, "s2ism-thick-samples")
    source_path = source_path_for_doc(fixture, "s2ism")
    result = {
        "status": "done",
        "done": True,
        "user_msg_id": 104,
        "assistant_message": {
            "role": "assistant",
            "content": "s2ISM 的权衡是空间分辨率、信噪比和光学层切能力；厚样本会带来离焦背景。",
            "cite_details": [
                {
                    "source_path": source_path,
                    "source_name": "Structured detection for high-SNR image scanning microscopy in thick samples",
                    "is_inpaper": False,
                }
            ],
        },
        "refs_payload": {
            "104": {
                "hits": [
                    {
                        "text": "Structured detection improves sectioning and SNR.",
                        "ui_meta": {
                            "source_path": source_path,
                            "display_name": "Structured detection for high-SNR image scanning microscopy in thick samples",
                            "summary_line": "摘要说明 s2ISM 处理厚样本时的分辨率、信噪比和层切权衡。",
                            "why_line": "用户问 trade-off，这张卡定位到论文直接解释厚样本成像瓶颈的证据。",
                        },
                    }
                ]
            }
        },
    }

    quality = validate_case(case, fixture, result)

    assert quality["ok"] is True


def test_validate_case_accepts_super_resolution_as_resolution_alias():
    fixture = load_fixture()
    case = _case_by_id(fixture, "s2ism-thick-samples")
    source_path = source_path_for_doc(fixture, "s2ism")
    result = {
        "status": "done",
        "done": True,
        "user_msg_id": 104,
        "assistant_message": {
            "role": "assistant",
            "content": "s2ISM 同时讨论超分辨、SNR 和 optical sectioning；厚样本会带来离焦背景。",
            "cite_details": [
                {
                    "source_path": source_path,
                    "source_name": "Structured detection for high-SNR image scanning microscopy in thick samples",
                    "is_inpaper": False,
                }
            ],
        },
        "refs_payload": {
            "104": {
                "hits": [
                    {
                        "text": "Structured detection improves sectioning and SNR.",
                        "ui_meta": {
                            "source_path": source_path,
                            "display_name": "Structured detection for high-SNR image scanning microscopy in thick samples",
                            "summary_line": "s2ISM explains the trade-off between super-resolution, SNR and sectioning.",
                            "why_line": "The user asks about thick samples, and this card points to that trade-off.",
                        },
                    }
                ]
            }
        },
    }

    quality = validate_case(case, fixture, result)

    assert quality["ok"] is True


def test_validate_case_accepts_foveated_chinese_paraphrase():
    fixture = load_fixture()
    case = _case_by_id(fixture, "foveated-dynamic-supersampling")
    source_path = source_path_for_doc(fixture, "foveated-spi")
    result = {
        "status": "done",
        "done": True,
        "user_msg_id": 105,
        "assistant_message": {
            "role": "assistant",
            "content": "dynamic supersampling 会围绕重点区域做自适应采样，并通过 supersampling 提升细节。",
            "cite_details": [
                {
                    "source_path": source_path,
                    "source_name": "Adaptive foveated single-pixel imaging with dynamic supersampling",
                    "is_inpaper": False,
                }
            ],
        },
        "refs_payload": {
            "105": {
                "hits": [
                    {
                        "text": "Adaptive foveated single-pixel imaging with dynamic supersampling.",
                        "ui_meta": {
                            "source_path": source_path,
                            "display_name": "Adaptive foveated single-pixel imaging with dynamic supersampling",
                            "summary_line": "论文说明重点区域如何使用更高采样密度。",
                            "why_line": "用户问 dynamic supersampling 的直觉，这张卡定位到自适应采样机制。",
                        },
                    }
                ]
            }
        },
    }

    quality = validate_case(case, fixture, result)

    assert quality["ok"] is True


def test_validate_case_checks_primary_evidence_terms():
    fixture = load_fixture()
    source_path = source_path_for_doc(fixture, "qclfm")
    case = {
        "id": "primary-evidence-smoke",
        "expected": {
            "requiredPrimaryEvidenceTerms": ["Digital Refocusing", "ray tracing", "wave propagation"],
            "forbiddenPrimaryEvidenceTerms": ["Conventional light-field microscope designs"],
        },
    }
    result = {
        "status": "done",
        "done": True,
        "user_msg_id": 201,
        "assistant_message": {
            "role": "assistant",
            "content": "The method uses ray tracing and then wave propagation to refocus the sample.",
        },
        "refs_payload": {
            "201": {
                "primary_evidence": {
                    "source_path": source_path,
                    "heading_path": "Digital Refocusing Procedure",
                    "snippet": "The trajectory of the photons is determined through ray tracing; the second step applies wave propagation.",
                },
                "hits": [
                    {
                        "ui_meta": {
                            "source_path": source_path,
                            "display_name": "Quantum correlation light-field microscope",
                            "summary_line": "Digital refocusing is described with ray tracing and wave propagation.",
                            "why_line": (
                                "Ray tracing maps photon paths before wave propagation "
                                "performs digital refocusing."
                            ),
                        }
                    }
                ],
            }
        },
    }

    quality = validate_case(case, fixture, result)

    assert quality["ok"] is True


def test_validate_case_rejects_misaligned_primary_evidence():
    fixture = load_fixture()
    source_path = source_path_for_doc(fixture, "qclfm")
    case = {
        "id": "primary-evidence-smoke",
        "expected": {
            "requiredPrimaryEvidenceTerms": ["Digital Refocusing", "ray tracing", "wave propagation"],
            "forbiddenPrimaryEvidenceTerms": ["Conventional light-field microscope designs"],
        },
    }
    result = {
        "status": "done",
        "done": True,
        "user_msg_id": 202,
        "assistant_message": {
            "role": "assistant",
            "content": "The method uses ray tracing and then wave propagation to refocus the sample.",
        },
        "refs_payload": {
            "202": {
                "primary_evidence": {
                    "source_path": source_path,
                    "heading_path": "I. INTRODUCTION",
                    "snippet": "Conventional light-field microscope designs typically make use of a microlens array.",
                },
                "hits": [
                    {
                        "ui_meta": {
                            "source_path": source_path,
                            "display_name": "Quantum correlation light-field microscope",
                            "summary_line": "The introduction describes conventional light-field microscopes.",
                            "why_line": "This card points to broad background rather than the refocusing procedure.",
                        }
                    }
                ],
            }
        },
    }

    quality = validate_case(case, fixture, result)
    failed_names = {item["name"] for item in quality["failures"]}

    assert quality["ok"] is False
    assert "primary_evidence_contains_required_terms" in failed_names
    assert "primary_evidence_avoids_forbidden_terms" in failed_names
