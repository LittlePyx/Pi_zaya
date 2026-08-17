from pathlib import Path
from types import SimpleNamespace

import kb.converter.benchmark as benchmark_module
from kb.converter.benchmark import (
    BenchmarkProfile,
    build_llm_config,
    build_arg_parser,
    discover_pdf_paths,
    parse_converter_log_metrics,
    parse_profile_spec,
    profile_env_overrides,
    run_benchmark_suite,
    summarize_runs_by_case,
    summarize_runs_by_profile,
)


def test_parse_profile_spec_parses_worker_and_stage_fields():
    profile = parse_profile_spec(
        "name=vision-fast,speed_mode=normal,llm_page_workers=4,llm_workers=2,max_inflight=3,vision_dpi=240,vision_compress=2,stage_timings=1,adaptive_page_budgets=1,text_local_fastpath=1"
    )

    assert profile == BenchmarkProfile(
        name="vision-fast",
        speed_mode="normal",
        no_llm_workers=None,
        llm_page_workers=4,
        llm_workers=2,
        max_inflight=3,
        vision_dpi=240,
        vision_compress=2,
        stage_timings=True,
        adaptive_page_budgets=True,
        text_local_fastpath=True,
    )


def test_parse_profile_spec_builds_default_name_when_missing():
    profile = parse_profile_spec("speed=no_llm,no_llm_workers=3")

    assert profile.name == "no_llm-no3"
    assert profile.speed_mode == "no_llm"
    assert profile.no_llm_workers == 3


def test_discover_pdf_paths_collects_files_and_dirs(tmp_path):
    top_pdf = tmp_path / "b.pdf"
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_pdf = nested_dir / "a.pdf"
    ignored = nested_dir / "note.txt"
    top_pdf.write_bytes(b"%PDF-1.7")
    nested_pdf.write_bytes(b"%PDF-1.7")
    ignored.write_text("not a pdf", encoding="utf-8")

    out = discover_pdf_paths([top_pdf, tmp_path], recursive=True)

    assert out == [nested_pdf.resolve(), top_pdf.resolve()]


def test_profile_env_overrides_clears_unset_values():
    profile = BenchmarkProfile(name="baseline", speed_mode="no_llm", no_llm_workers=2, stage_timings=False)

    env = profile_env_overrides(profile)

    assert env["KB_PDF_NO_LLM_PAGE_WORKERS"] == "2"
    assert env["KB_PDF_LLM_PAGE_WORKERS"] is None
    assert env["KB_LLM_MAX_INFLIGHT"] is None
    assert env["KB_PDF_STAGE_TIMINGS"] == "0"
    assert env["KB_PDF_VISION_ADAPTIVE_PAGE_BUDGETS"] == "0"
    assert env["KB_PDF_VISION_TEXT_LOCAL_FASTPATH"] == "0"


def test_build_llm_config_uses_dedicated_vision_provider(monkeypatch):
    monkeypatch.setattr(
        benchmark_module,
        "load_settings",
        lambda: SimpleNamespace(
            api_key="text-key",
            base_url="https://text.invalid/v1",
            model="text-only-model",
            vision_api_key="vision-key",
            vision_base_url="https://vision.invalid/v1",
            vision_model="vision-model",
            timeout_s=42.0,
            max_retries=1,
        ),
    )

    config = build_llm_config(required=True)

    assert config is not None
    assert config.api_key == "vision-key"
    assert config.base_url == "https://vision.invalid/v1"
    assert config.model == "vision-model"


def test_summarize_runs_groups_by_case_and_profile():
    runs = [
        {
            "profile": "no-llm-2",
            "pdf_name": "paper_a.pdf",
            "pdf_path": str(Path("paper_a.pdf")),
            "ok": True,
            "elapsed_s": 2.0,
            "output_md_chars": 100,
            "page_total_avg_s": 0.8,
            "vision_step6_avg_s": 0.0,
            "references_pages": 0,
            "empty_retry_count": 0,
            "math_retry_count": 0,
            "fallback_count": 0,
        },
        {
            "profile": "no-llm-2",
            "pdf_name": "paper_a.pdf",
            "pdf_path": str(Path("paper_a.pdf")),
            "ok": False,
            "elapsed_s": 4.0,
            "output_md_chars": 80,
            "page_total_avg_s": 1.2,
            "vision_step6_avg_s": 0.0,
            "references_pages": 0,
            "empty_retry_count": 0,
            "math_retry_count": 0,
            "fallback_count": 0,
        },
        {
            "profile": "normal-1",
            "pdf_name": "paper_b.pdf",
            "pdf_path": str(Path("paper_b.pdf")),
            "ok": True,
            "elapsed_s": 5.0,
            "output_md_chars": 200,
            "page_total_avg_s": 4.0,
            "vision_step6_avg_s": 3.2,
            "references_pages": 1,
            "empty_retry_count": 2,
            "math_retry_count": 1,
            "fallback_count": 1,
        },
    ]

    by_case = summarize_runs_by_case(runs)
    by_profile = summarize_runs_by_profile(runs)

    assert by_case == [
        {
            "profile": "no-llm-2",
            "pdf_name": "paper_a.pdf",
            "pdf_path": "paper_a.pdf",
            "runs": 2,
            "ok_runs": 1,
            "fail_runs": 1,
            "avg_elapsed_s": 3.0,
            "min_elapsed_s": 2.0,
            "max_elapsed_s": 4.0,
            "avg_output_md_chars": 90.0,
            "avg_page_total_s": 1.0,
            "avg_vision_step6_s": 0.0,
            "avg_references_pages": 0.0,
            "avg_empty_retries": 0.0,
            "avg_math_retries": 0.0,
            "avg_fallbacks": 0.0,
            "avg_text_local_attempts": 0.0,
            "avg_text_local_accepted": 0.0,
            "avg_text_local_fallbacks": 0.0,
            "avg_vision_calls_avoided": 0.0,
        },
        {
            "profile": "normal-1",
            "pdf_name": "paper_b.pdf",
            "pdf_path": "paper_b.pdf",
            "runs": 1,
            "ok_runs": 1,
            "fail_runs": 0,
            "avg_elapsed_s": 5.0,
            "min_elapsed_s": 5.0,
            "max_elapsed_s": 5.0,
            "avg_output_md_chars": 200.0,
            "avg_page_total_s": 4.0,
            "avg_vision_step6_s": 3.2,
            "avg_references_pages": 1.0,
            "avg_empty_retries": 2.0,
            "avg_math_retries": 1.0,
            "avg_fallbacks": 1.0,
            "avg_text_local_attempts": 0.0,
            "avg_text_local_accepted": 0.0,
            "avg_text_local_fallbacks": 0.0,
            "avg_vision_calls_avoided": 0.0,
        },
    ]
    assert by_profile == [
        {
            "profile": "no-llm-2",
            "runs": 2,
            "pdfs": 1,
            "ok_runs": 1,
            "fail_runs": 1,
            "avg_elapsed_s": 3.0,
            "min_elapsed_s": 2.0,
            "max_elapsed_s": 4.0,
            "avg_page_total_s": 1.0,
            "avg_vision_step6_s": 0.0,
            "avg_references_pages": 0.0,
            "avg_empty_retries": 0.0,
            "avg_math_retries": 0.0,
            "avg_fallbacks": 0.0,
            "avg_text_local_attempts": 0.0,
            "avg_text_local_accepted": 0.0,
            "avg_text_local_fallbacks": 0.0,
            "avg_vision_calls_avoided": 0.0,
        },
        {
            "profile": "normal-1",
            "runs": 1,
            "pdfs": 1,
            "ok_runs": 1,
            "fail_runs": 0,
            "avg_elapsed_s": 5.0,
            "min_elapsed_s": 5.0,
            "max_elapsed_s": 5.0,
            "avg_page_total_s": 4.0,
            "avg_vision_step6_s": 3.2,
            "avg_references_pages": 1.0,
            "avg_empty_retries": 2.0,
            "avg_math_retries": 1.0,
            "avg_fallbacks": 1.0,
            "avg_text_local_attempts": 0.0,
            "avg_text_local_accepted": 0.0,
            "avg_text_local_fallbacks": 0.0,
            "avg_vision_calls_avoided": 0.0,
        },
    ]


def test_parse_converter_log_metrics_extracts_stage_timings_and_retry_counts(tmp_path):
    log_path = tmp_path / "converter.log"
    log_path.write_text(
        "\n".join(
            [
                "  [Page 1] Step 1 (refs check): 0.05s, references=0",
                "  [Page 1] Step 4 (page render): 1.25s, bytes=123456",
                "  [Page 1] Step 6 (vision convert): 12.00s, chars=1000",
                "[VISION_DIRECT][BUDGET] page 1: class=text_dense_body enabled=1 dpi=200 base_dpi=220 max_tokens=2560 density=dense text_chars=5000 formulas=0 images=0 visuals=0 profile=adaptive_text_dense_body",
                "  [Page 1] TOTAL: 13.50s",
                "  [Page 2] Step 1 (refs check): 0.06s, references=1",
                "  [Page 2] Step 6 (vision convert): 6.00s, chars=500",
                "[VISION_DIRECT][BUDGET] page 2: class=references enabled=1 dpi=220 base_dpi=220 max_tokens=default density=dense text_chars=4300 formulas=0 images=0 visuals=0 profile=adaptive_references",
                "  [Page 2] TOTAL: 7.00s",
                "[VISION_DIRECT][REFS] page 2: column mode enabled (2 crops, dpi=240)",
                "[VISION_DIRECT][REFS] page 2 crop 1/2 done (3.5s, 123 chars)",
                "[VISION_DIRECT][REFS] page 2 crop 2/2 done (4.5s, 234 chars)",
                "[VISION_DIRECT][LAYOUT] page 1: structured crop mode enabled (3 crops, dpi=220)",
                "[VISION_DIRECT][LAYOUT] page 1 crop 1/3 (left) done (2.0s, 120 chars)",
                "[VISION_DIRECT] VL empty on page 3, retry 1/2",
                "[VISION_DIRECT] fragmented math detected on page 1, retrying with strict formula hint",
                "[VISION_DIRECT] VL returned empty for page 3, falling back to extraction pipeline",
                "[VISION_DIRECT] fragmented math persists on page 4, using extraction fallback",
            ]
        ),
        encoding="utf-8",
    )

    metrics, page_metrics = parse_converter_log_metrics(log_path)

    assert metrics == {
        "page_metric_count": 2,
        "page_timed_pages": 2,
        "page_total_avg_s": 10.25,
        "page_total_p50_s": 10.25,
        "page_total_p90_s": 12.85,
        "vision_step6_pages": 2,
        "vision_step6_avg_s": 9.0,
        "vision_step6_p50_s": 9.0,
        "vision_step6_p90_s": 11.4,
        "references_pages": 1,
        "refs_column_pages": 1,
        "refs_crop_call_count": 2,
        "refs_crop_avg_s": 4.0,
        "layout_crop_pages": 1,
        "layout_crop_call_count": 1,
        "layout_crop_avg_s": 2.0,
        "empty_retry_count": 1,
        "math_retry_count": 1,
        "fallback_count": 2,
        "page_class_counts": {"text_dense_body": 1, "references": 1},
        "adaptive_budget_pages": 2,
        "reduced_dpi_pages": 1,
        "token_capped_pages": 1,
        "text_local_attempt_pages": 0,
        "text_local_accepted_pages": 0,
        "text_local_fallback_pages": 0,
        "vision_calls_avoided": 0,
    }
    assert page_metrics == [
        {
            "page": 1,
            "is_references_page": 0,
            "uses_refs_column_mode": 0,
            "uses_layout_crop_mode": 1,
            "step_1_refs_check_s": 0.05,
            "step_4_page_render_s": 1.25,
            "step_6_vision_convert_s": 12.0,
            "page_class": "text_dense_body",
            "adaptive_budget_enabled": 1,
            "render_dpi": 200,
            "base_dpi": 220,
            "max_tokens_override": 2560,
            "text_density": "dense",
            "text_chars": 5000,
            "formula_candidate_count": 0,
            "image_count": 0,
            "visual_rect_count": 0,
            "render_profile": "adaptive_text_dense_body",
            "page_total_s": 13.5,
        },
        {
            "page": 2,
            "is_references_page": 1,
            "uses_refs_column_mode": 1,
            "uses_layout_crop_mode": 0,
            "step_1_refs_check_s": 0.06,
            "step_6_vision_convert_s": 6.0,
            "page_class": "references",
            "adaptive_budget_enabled": 1,
            "render_dpi": 220,
            "base_dpi": 220,
            "max_tokens_override": None,
            "text_density": "dense",
            "text_chars": 4300,
            "formula_candidate_count": 0,
            "image_count": 0,
            "visual_rect_count": 0,
            "render_profile": "adaptive_references",
            "page_total_s": 7.0,
        },
    ]


def test_parse_converter_log_metrics_counts_local_reference_fastpath_class(tmp_path):
    log_path = tmp_path / "converter.log"
    log_path.write_text(
        "\n".join(
            [
                "  [Page 7] Step 1 (refs check): 0.01s, references=1",
                "  [Page 7] Step 4 (page render): 0.00s, skipped=refs-local-fastpath",
                "  [Page 7] TOTAL: 1.25s",
            ]
        ),
        encoding="utf-8",
    )

    metrics, page_metrics = parse_converter_log_metrics(log_path)

    assert metrics["page_class_counts"] == {"references": 1}
    assert page_metrics[0]["page_class"] == "references"


def test_parse_converter_log_metrics_extracts_text_local_fastpath(tmp_path):
    log_path = tmp_path / "converter.log"
    log_path.write_text(
        "\n".join(
            [
                "  [Page 6] Step 1 (refs check): 0.01s, references=0",
                "  [Page 6] Step 6 (local text convert): 0.42s, text_local=1 chars=4200",
                "[VISION_DIRECT][TEXT_LOCAL] page 6: accepted=1 reason=accepted source_chars=4300 output_chars=4200 source_tokens=700 output_tokens=690 coverage=0.9800 bigram_coverage=0.9500 order_ratio=0.9200 length_ratio=0.9857 headings=3/3 elapsed=0.4200",
                "[VISION_DIRECT][BUDGET] page 6: class=text_dense_body enabled=0 dpi=0 base_dpi=220 max_tokens=default density=dense text_chars=4300 formulas=0 images=0 visuals=0 profile=text_local_fastpath",
                "  [Page 6] TOTAL: 0.50s",
            ]
        ),
        encoding="utf-8",
    )

    metrics, page_metrics = parse_converter_log_metrics(log_path)

    assert metrics["text_local_attempt_pages"] == 1
    assert metrics["text_local_accepted_pages"] == 1
    assert metrics["text_local_fallback_pages"] == 0
    assert metrics["vision_calls_avoided"] == 1
    assert metrics["vision_step6_pages"] == 0
    assert page_metrics[0]["text_local_reason"] == "accepted"
    assert page_metrics[0]["text_local_coverage"] == 0.98
    assert page_metrics[0]["text_local_headings_promoted"] == 3
    assert page_metrics[0]["text_local_heading_candidates"] == 3
    assert page_metrics[0]["render_profile"] == "text_local_fastpath"


def test_build_arg_parser_allows_listing_default_profiles_without_inputs():
    args = build_arg_parser().parse_args(["--list-default-profiles"])

    assert args.list_default_profiles is True
    assert args.inputs == []


def test_build_arg_parser_parses_warm_cache_flag():
    args = build_arg_parser().parse_args(["paper.pdf", "--warm-cache", "--paired-profiles"])

    assert args.inputs == ["paper.pdf"]
    assert args.warm_cache is True
    assert args.paired_profiles is True


def test_run_benchmark_suite_pairs_profiles_and_reverses_even_repeats(tmp_path, monkeypatch):
    seen = []

    def _fake_run(*, pdf_path, profile, repeat_index, **kwargs):
        seen.append((pdf_path.name, repeat_index, profile.name))
        return (
            {
                "ok": True,
                "elapsed_s": 1.0,
                "output_md_chars": 100,
                "log_path": "converter.log",
                "profile": profile.name,
                "pdf_name": pdf_path.name,
                "pdf_path": str(pdf_path),
                "error": "",
            },
            [],
        )

    monkeypatch.setattr(benchmark_module, "run_single_benchmark", _fake_run)
    profiles = [
        BenchmarkProfile(name="control", speed_mode="normal"),
        BenchmarkProfile(name="candidate", speed_mode="normal", text_local_fastpath=True),
    ]

    payload = run_benchmark_suite(
        pdf_paths=[Path("paper-a.pdf"), Path("paper-b.pdf")],
        profiles=profiles,
        out_root=tmp_path,
        repeat=2,
        paired_profiles=True,
    )

    assert seen == [
        ("paper-a.pdf", 1, "control"),
        ("paper-a.pdf", 1, "candidate"),
        ("paper-a.pdf", 2, "candidate"),
        ("paper-a.pdf", 2, "control"),
        ("paper-b.pdf", 1, "control"),
        ("paper-b.pdf", 1, "candidate"),
        ("paper-b.pdf", 2, "candidate"),
        ("paper-b.pdf", 2, "control"),
    ]
    assert payload["paired_profiles"] is True
