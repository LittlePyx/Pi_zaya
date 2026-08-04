# Research QA Eval Runbook

Updated: 2026-08-05

## Purpose

The research QA eval protects the real researcher-facing workflow:

1. Natural research questions over the local paper library.
2. Answer grounding through required source documents.
3. System A citations for current-paper evidence.
4. System B citations for in-paper upstream references.
5. Reference locator card quality, including summary, relevance, polish state, and reader-open evidence.
6. Citation shelf quality, so saved literature keeps a useful title, source/export identity, summary, and clean visible copy.

The shared fixture is `web/src/testing/researchQaData.json`. It contains 56
natural research questions. Forty-one are source-grounded cases whose claim and
reader-locator contracts are pinned to a page in the current Markdown corpus.

The overlapping `full_library_acceptance_v1` suite is the release baseline. It
reuses 29 of those questions to cover every configured paper and both UI
locales, plus cross-paper synthesis, table lookup, formula reasoning, figure
architecture, exact location, negative evidence, System A/System B routing, and
answer/reference alignment. `live_smoke_v1` is the five-question paid-model
spot check. These suites select existing cases; they do not duplicate the
underlying regression definitions.

Six focused user journeys also have a human-reviewed deterministic replay in
`docs/research_qa_grounded_replay_v1.jsonl`. The replay uses real paper identities
and reviewed source excerpts for paper summaries, method details, comparisons,
multi-paper synthesis, upstream-reference reasoning, and scope-boundary decisions.

## Lightweight CI Check

CI runs the fixture smoke check:

```bash
python tools/research_qa/run_research_qa_eval.py --dry-run
python tools/research_qa/run_research_qa_eval.py --suite full_library_acceptance_v1 --dry-run
python tools/research_qa/run_research_qa_eval.py --replay docs/research_qa_grounded_replay_v1.jsonl --fail-on-quality
```

These commands do not call the API or an LLM. The first two validate fixture and
suite coverage; the replay sends reviewed answers and evidence payloads through
the same validator used by live runs. The replay rejects unexpected source documents, unsupported
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
python tools/research_qa/run_research_qa_eval.py --base-url http://127.0.0.1:8000 --suite full_library_acceptance_v1 --fail-on-quality
```

For the faster five-question real-model spot check:

```bash
python tools/research_qa/run_research_qa_eval.py --base-url http://127.0.0.1:8000 --suite live_smoke_v1 --fail-on-quality
```

For the 15-question source-grounded blind regression set:

```bash
python tools/research_qa/run_research_qa_eval.py --base-url http://127.0.0.1:8000 --split blind_holdout_v2 --fail-on-quality
```

For one case:

```bash
python tools/research_qa/run_research_qa_eval.py --base-url http://127.0.0.1:8000 --case-id scinerf-admm-origin --fail-on-quality
```

## 2026-08-04 Latency And Quality Baseline

Latency is measured at four user-visible milestones. Do not compare only the
provider request: citation planning, answer finalization, and evidence-card
completion are part of the product latency.

The 15-question `blind_holdout_v2` run improved without changing its reviewed
answer, document, citation-route, or locator contracts:

| Milestone | Baseline p50 / p95 / max | Optimized p50 / p95 / max |
|---|---:|---:|
| First visible answer | 2940 / 3910 / 3923 ms | 1561 / 2994 / 3004 ms |
| Answer complete | 7564 / 12243 / 13277 ms | 2371 / 6618 / 6989 ms |
| Evidence cards complete | 11680 / 17436 / 19973 ms | 4632 / 8222 / 10184 ms |
| End-to-end evaluation | 12492 / 18393 / 20492 ms | 5772 / 10561 / 11075 ms |

The final release evidence for this baseline is:

1. Blind live QA: 15/15 passed in
   `test_results/research_qa_blind/final_regression/20260804_185023`.
2. Full-library live QA: 29/29 passed in
   `test_results/research_qa_full_library/final_live_pass/20260804_184619`;
   end-to-end p50/p95/max was 6357/12061/14563 ms.
3. Paid-model smoke: 5/5 passed in
   `test_results/research_qa_live_smoke/final/20260804_184007`.
4. Deterministic full-library retrieval: 29/29 passed in
   `test_results/research_qa_full_library/final_retrieval/20260804_185016`.
5. Source validation: 35/35 source-grounded contracts passed, and the reviewed
   replay passed 6/6.

The optimized path may bypass free-form model generation only when the complete
System A citation plan contains source-verbatim evidence for every requested
facet. It must keep the same answer terms, document identities, citation routes,
and reader locators. Incomplete plans continue through normal model generation;
they must never be made faster by omitting requested evidence or suppressing a
quality failure.

## 2026-08-05 Tail-Latency Acceptance

The follow-up optimization measures the answer path in phases: first visible
answer, answer completion, evidence-card completion, UI readiness, and final
quality validation. During generation the browser no longer polls for evidence
cards that cannot yet be final. At the terminal state it hydrates the final
message and evidence cards in parallel. The reference endpoint returns a cheap
pending snapshot for an active generation and reuses authoritative cached
snapshots after completion. These scheduling changes do not remove retrieval,
source checks, card polishing, or claim/evidence validation.

For the comparable 15-question `blind_holdout_v2` run, the serialized baseline
and accepted implementation were:

| Milestone | Serialized baseline p50 / p95 / max | Accepted p50 / p95 / max |
|---|---:|---:|
| First visible answer | 1951 / 3601 / 3701 ms | 1875 / 3702 / 3879 ms |
| Answer complete | 2711 / 6263 / 7000 ms | 2661 / 7671 / 10598 ms |
| Evidence cards complete | 4062 / 9289 / 11665 ms | 4879 / 9665 / 14011 ms |
| UI/evaluation ready | 6746 / 12525 / 13158 ms | 5382 / 9665 / 14011 ms |

The UI/evaluation-ready p50 improved by 20.2% and p95 by 22.8%. The maximum
regressed by 6.5%, and the answer/card p95 values also regressed because one
QCLFM real-model answer took 10598 ms. Keep those values visible: the accepted
change removes avoidable client/server serialization, but does not claim to
eliminate provider/model variance. The accepted phased result is in
`test_results/research_qa_blind/tail_latency_acceptance_v2_final_pass/20260805_015708`;
the serialized baseline is in
`test_results/research_qa_blind/next_card_baseline/20260804_221632`.

Quality and coverage gates after the final implementation:

1. Full-library live QA: 29/29 passed in
   `test_results/research_qa_blind/full_library_acceptance_final/20260805_023427`.
2. Paid-model smoke: 5/5 passed in
   `test_results/research_qa_blind/live_smoke_final_pass/20260805_022037`.
3. New long-tail blind cases: 6/6 passed in
   `test_results/research_qa_blind/blind_holdout_v3_acceptance/20260805_012040`.
4. Source validation: 41/41 passed, deterministic full-library retrieval passed
   29/29, and the reviewed replay passed 6/6.
5. The complete backend suite passed 4292 tests with 43 skips. Frontend
   production build, ESLint, and the terminal-reference Playwright regression
   also passed.

The final 29-question run reported first-visible p50/p95/max of
2476/3840/4926 ms and UI-ready p50/p95/max of 7465/12747/14092 ms. It is a
broader suite than `blind_holdout_v2`, so use it as a quality/coverage release
gate rather than as a direct latency comparison with the 15-question baseline.

## Outputs

Default output directory:

```text
test_results/research_qa_eval/<timestamp>/
```

Files:

1. `raw_results.jsonl`: full per-case payloads and quality checks.
2. `summary.json`: total, passed, failed, base URL, fixture path, output path.
3. `report.md`: human-readable report with failures, reference-card quality, citation-shelf quality, and System B audit.

The runner temporarily applies each case's Chinese or English UI/card locale
through the same settings API used by the product, then restores the user's
original locale preferences in a `finally` block. The resulting diagnostics
remain in `test_results`; they are never inserted into the visible answer.

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
