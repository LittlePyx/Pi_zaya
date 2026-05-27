# Converter Quality Eval Runbook

Updated: 2026-05-27

## Purpose

The converter quality eval protects the upstream Markdown that feeds retrieval, answer grounding, citation cards, reader-open targets, and the literature basket. It checks existing converted Markdown for research-paper structure instead of asking an LLM to judge quality.

The shared manifest is:

```text
tools/manual_regression/manifests/converter_markdown_quality_v1.json
```

## Lightweight CI Check

CI runs:

```bash
python tools/converter_quality/run_converter_quality_eval.py --dry-run
```

This only loads the manifest and lists planned cases. It does not require the full local `db/` corpus to exist in CI.

## Local Structural Eval

Run this before and after changes to PDF conversion, Markdown post-processing, figure extraction, reference extraction, chunking, retrieval quality, citation routing, reader-open targeting, or literature-basket source summaries:

```bash
python tools/converter_quality/run_converter_quality_eval.py --fail-on-quality
```

For one case:

```bash
python tools/converter_quality/run_converter_quality_eval.py --case-id scinerf_structured_cvpr --fail-on-quality
```

## Outputs

Default output directory:

```text
test_results/converter_quality_eval/<timestamp>/
```

Files:

1. `summary.json`: suite totals and status.
2. `raw_results.json`: per-case metrics and failures.
3. `report.md`: human-readable failure list and metric table.

## Go/No-Go

Use `go` only when:

1. `overall_status == PASS`.
2. `missing_image_count == 0` for all strict figure cases.
3. `unclosed_display_math_block_count == 0`.
4. `analyzer_error_count == 0`.
5. Reference counts, body citation counts, figures, captions, tables, and formulas stay above manifest thresholds.

Use `no-go` when any strict case fails. A fluent answer, good-looking citation card, or polished literature-basket summary cannot compensate for broken upstream Markdown.

## When Extending The Manifest

Add cases that represent real paper shapes the product depends on:

1. CV/ML papers with figures, tables, equations, and long reference lists.
2. Optics or physics papers with formulas and dense citation markers.
3. Review papers with long bibliographies.
4. Papers with known conversion risks: frontmatter, figure captions, tables, supplemental references, or page markers.

Each case should include stable thresholds and at least one title/topic `must_contain_text` guard so a path mix-up cannot pass by accident.

## Visual PDF Checks

This eval is structural and CI-friendly. For risky layout changes, pair it with visual inspection of rendered PDF pages and generated Markdown assets: page screenshots, figure crops, and image existence checks. Keep those heavier checks local or manual unless the fixture assets are committed.
