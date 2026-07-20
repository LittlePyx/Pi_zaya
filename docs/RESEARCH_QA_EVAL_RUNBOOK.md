# Research QA Eval Runbook

Updated: 2026-07-20

## Purpose

The research QA eval protects the real researcher-facing workflow:

1. Natural research questions over the local paper library.
2. Answer grounding through required source documents.
3. System A citations for current-paper evidence.
4. System B citations for in-paper upstream references.
5. Reference locator card quality, including summary, relevance, polish state, and reader-open evidence.
6. Citation shelf quality, so saved literature keeps a useful title, source/export identity, summary, and clean visible copy.

The shared fixture is `web/src/testing/researchQaData.json`. It contains 30
natural research questions. Fifteen are source-grounded cases whose claim and
reader-locator contracts are pinned to a page in the current Markdown corpus.

Six focused user journeys also have a human-reviewed deterministic replay in
`docs/research_qa_grounded_replay_v1.jsonl`. The replay uses real paper identities
and reviewed source excerpts for paper summaries, method details, comparisons,
multi-paper synthesis, upstream-reference reasoning, and scope-boundary decisions.

## Lightweight CI Check

CI runs the fixture smoke check:

```bash
python tools/research_qa/run_research_qa_eval.py --dry-run
python tools/research_qa/run_research_qa_eval.py --replay docs/research_qa_grounded_replay_v1.jsonl --fail-on-quality
```

These commands do not call the API or an LLM. The first validates fixture coverage;
the second sends reviewed answers and evidence payloads through the same validator
used by live runs. The replay rejects unexpected source documents, unsupported
claim/evidence bindings, wrong citation routes, and incorrect reader locators.

When the local paper corpus is available, also verify that every reviewed
source page still contains the expected evidence after conversion or repair:

```bash
python tools/research_qa/run_research_qa_eval.py --validate-sources --db-root db
```

This check reads page-marked Markdown only; it does not call the API or a model.
It intentionally remains a local gate because `db/` is runtime data and is not
committed to CI.

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
7. Every focused journey passes `claims_have_matching_evidence`,
   `refs_avoid_unexpected_docs`, `citations_match_required_routes`, and
   `citations_have_expected_locators`.
8. Changed conversion output passes `--validate-sources`, and each
   source-grounded live answer cites the reviewed `sourcePage` rather than only
   the right paper or section.

Use `no-go` when any strict case fails, even if the answer text looks plausible. A plausible answer with weak cards or untraceable citations is still a product regression.

## Common Failure Buckets

1. `refs_include_required_docs`: retrieval/card selection missed a required source.
2. `citations_include_required_docs`: answer citations no longer bind to the expected source.
3. `refs_card_copy_quality`: locator card copy is short, templated, duplicated, raw Markdown, or has broken evidence.
4. `citation_card_quality`: popover/shelf citation payload is missing evidence, locator, click anchor, or System B trace fields.
5. `citation_shelf_quality`: a saved shelf item has a weak title, missing source/export identity, placeholder summary, raw Markdown, or templated visible copy.
6. `system_b_audit`: in-paper upstream citation is present but not trace-complete.
7. `refs_avoid_unexpected_docs`: a focused question retrieved papers outside its reviewed source scope.
8. `claims_have_matching_evidence`: answer wording exists, but no citation payload binds that claim to the reviewed evidence terms.
9. `citations_match_required_routes`: required System A current-paper evidence or System B upstream evidence is missing.
10. `citations_have_expected_locators`: a citation points to the right paper but the wrong section or evidence span.
11. Source validation errors: the reviewed evidence terms moved, disappeared,
    or became corrupted on the pinned page after conversion/index changes.

## When Extending The Fixture

Every new case should include:

1. A natural question, not a feature test phrased as implementation language.
2. At least two user-facing `acceptance` statements.
3. `expected.requiredAnswerTerms`.
4. `expected.requiredRefDocIds`.
5. `expected.requiredCitationDocIds`.
6. Stricter gates such as `requireRefsReady`, `requirePolishStatus`, `requireCitationShelfQuality`, `minRefHits`, `minCitationCount`, and System B trace checks when the case is meant to protect citation, card, or reading-list quality.
7. For `sourceGrounded: true`, every claim and locate contract must include a
   `sourcePage` and page-local `evidenceTerms` verified against the current
   Markdown with `--validate-sources`.
