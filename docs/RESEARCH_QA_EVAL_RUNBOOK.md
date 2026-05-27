# Research QA Eval Runbook

Updated: 2026-05-26

## Purpose

The research QA eval protects the real researcher-facing workflow:

1. Natural research questions over the local paper library.
2. Answer grounding through required source documents.
3. System A citations for current-paper evidence.
4. System B citations for in-paper upstream references.
5. Reference locator card quality, including summary, relevance, polish state, and reader-open evidence.
6. Citation shelf quality, so saved literature keeps a useful title, source/export identity, summary, and clean visible copy.

The shared fixture is `web/src/testing/researchQaData.json`.

## Lightweight CI Check

CI runs the fixture smoke check:

```bash
python tools/research_qa/run_research_qa_eval.py --dry-run
```

This does not call the API or an LLM. It verifies that the real-question fixture loads, lists the covered documents and cases, and is backed by unit tests that enforce case contracts.

Use this for every PR.

## Live Eval

Run the live eval before and after substantial changes to retrieval, answer generation, citation rendering, citation-card copy, reference indexing, or reader-open targeting.

Start the API locally:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Then run:

```bash
python tools/research_qa/run_research_qa_eval.py --base-url http://127.0.0.1:8000 --fail-on-quality
```

For a faster spot check:

```bash
python tools/research_qa/run_research_qa_eval.py --base-url http://127.0.0.1:8000 --limit 3 --fail-on-quality
```

For one case:

```bash
python tools/research_qa/run_research_qa_eval.py --base-url http://127.0.0.1:8000 --case-id scinerf-admm-origin --fail-on-quality
```

## Outputs

Default output directory:

```text
test_results/research_qa_eval/<timestamp>/
```

Files:

1. `raw_results.jsonl`: full per-case payloads and quality checks.
2. `summary.json`: total, passed, failed, base URL, fixture path, output path.
3. `report.md`: human-readable report with failures, reference-card quality, citation-shelf quality, and System B audit.

## Go/No-Go

Use `go` only when:

1. `summary.failed == 0`.
2. No `refs_card_copy_quality` failures.
3. No `citation_card_quality` failures.
4. No `citation_shelf_quality` failures for strict reading-list or cross-paper cases.
5. System B audit has no `needs_review`, `answer_context_only`, or `reference_index_fallback` failures for strict cases.
6. Any changed UI path still passes the Playwright research QA replay when frontend behavior is involved.

Use `no-go` when any strict case fails, even if the answer text looks plausible. A plausible answer with weak cards or untraceable citations is still a product regression.

## Common Failure Buckets

1. `refs_include_required_docs`: retrieval/card selection missed a required source.
2. `citations_include_required_docs`: answer citations no longer bind to the expected source.
3. `refs_card_copy_quality`: locator card copy is short, templated, duplicated, raw Markdown, or has broken evidence.
4. `citation_card_quality`: popover/shelf citation payload is missing evidence, locator, click anchor, or System B trace fields.
5. `citation_shelf_quality`: a saved shelf item has a weak title, missing source/export identity, placeholder summary, raw Markdown, or templated visible copy.
6. `system_b_audit`: in-paper upstream citation is present but not trace-complete.

## When Extending The Fixture

Every new case should include:

1. A natural question, not a feature test phrased as implementation language.
2. At least two user-facing `acceptance` statements.
3. `expected.requiredAnswerTerms`.
4. `expected.requiredRefDocIds`.
5. `expected.requiredCitationDocIds`.
6. Stricter gates such as `requireRefsReady`, `requirePolishStatus`, `requireCitationShelfQuality`, `minRefHits`, `minCitationCount`, and System B trace checks when the case is meant to protect citation, card, or reading-list quality.
