# Research Gap Queue Runbook

Updated: 2026-08-10

## Purpose

The project research gap queue turns already-audited evidence problems into a
durable research worklist. It aggregates gaps across evidence matrices,
matrix-backed research briefs, paired comparison audits, and source-change
monitoring without asking a model to invent a task or infer a missing fact.

The queue is intentionally downstream of the existing evidence contracts. It
does not relax matrix verification, treat an unconfirmed passage as accepted
evidence, rewrite a brief, or hide a source-change event. It helps a researcher
see what is missing, understand the impact, search the affected paper for a
strict same-source repair, find other local papers that may expand coverage,
and keep each decision human-reviewed.

## Gap Sources

Each scan reads current persisted project artifacts and emits a stable
`gap_key` for these structured signals:

- `missing_cell`: the matrix audit kept a field explicitly empty;
- `unsupported_cell`: a populated value failed the same-source evidence
  contract;
- `comparison_not_comparable`: a saved paired comparison did not satisfy the
  complete task, dataset, protocol, metric, target, value, and evidence
  contract;
- `matrix_needs_review`: matrix-level source coverage or support checks failed;
- `brief_stale`: the brief's saved matrix lineage is no longer current or
  verifiable;
- `source_change`: the evidence-change inbox reports a current source event.

No free-form LLM classification participates in this aggregation. Matrix row,
field, brief, comparison, source, reason, and revision identifiers remain in the
persisted payload so the UI can show exactly why an item exists.

## Priority And Impact

Priority is deterministic. Each kind has a conservative base score. The queue
then adds bounded increments for affected briefs, affected brief citations, and
affected paired comparisons. Scores at least 85 are high priority, scores from
60 through 84 are medium, and lower scores are low. The ordering does not alter
evidence status; it only puts source changes and unsupported claims ahead of
ordinary missing coverage when their downstream impact is larger.

The impact calculation follows matrix-to-brief lineage and same-source citation
identity. It reports counts and affected brief records but does not claim that
every downstream sentence is wrong. A stale or changed dependency remains
visible until its source workflow resolves it.

## Candidate Evidence Workflow

Candidate search is available for cell-level coverage and comparison gaps. It:

1. creates a deterministic query from the matrix objective, field facet, row
   label, and saved comparison terms;
2. searches only the local indexed corpus with BM25;
3. excludes every source already present in that matrix;
4. returns at most one exact indexed passage per other source, including page,
   heading, block, and anchor locators where available;
5. rejects reference/bibliography sections and never emits arbitrary zero-match
   chunks. A strict two-meaningful-term lexical fallback is used only for very
   small repetitive corpora where BM25 has no positive IDF result.

The UI labels all results as candidates. A researcher can open the exact source
passage, then explicitly confirm it. Confirmation recomputes the candidate on
the server, adds the exact quote and stable locator to the project literature
basket, and marks the gap `in_progress`. This first confirmation does not update
a matrix cell or brief. A separately confirmed source-expansion workflow can
then add that other paper as its own matrix row; it can never fill the original
paper's row.

## Comparison Candidate Workflow

A verified matrix can also discover comparison candidates directly from its
own active rows. This is separate from cross-paper gap search: it pairs only
already selected matrix sources, accepts only structured metric tables, and
requires exact normalized task and dataset plus a controlled metric and matching
unit. Each side carries its own quote and reader locator. The candidate preloads
the complete paired contract but does not create a comparison or resolve a gap.

The researcher reviews both source excerpts and explicitly confirms every
semantic mapping flagged by the server. On submission, the API checks the matrix
revision and fresh indexes, regenerates the candidate, rejects extra or missing
confirmations, and invokes the existing strict paired audit. Only the stored
audit revision triggers a fresh gap scan. If it is not comparable, the resulting
`comparison_not_comparable` gap remains visible. If the matrix revision changes,
affected briefs are reported as update-ready and still use their existing
claim-by-claim incremental review; no brief text is silently rewritten.

## Cross-Source Matrix Expansion Workflow

A confirmed cross-paper candidate exposes an extractive new-row preview. This
second stage:

1. requires the gap revision to equal the current matrix revision and the
   candidate ID/source to equal the persisted basket-confirmation action;
2. verifies the candidate source file against the committed index SHA and
   recomputes the candidate server-side;
3. rejects sources already present in the matrix and matrices at the eight-source
   contract limit;
4. verifies that the discovery quote still occurs in the exact indexed chunk
   with a page, heading, block, or anchor locator;
5. runs the existing field-specific extractors only on that candidate paper and
   previews all grounded values plus every honestly missing field;
6. requires another explicit confirmation before writing anything;
7. appends one independent row and its same-source evidence while preserving
   every existing row and evidence record byte-for-byte;
8. recomputes matrix quality and comparison warnings, reaudits every saved
   comparison, refreshes the source-watch baseline, and rescans gaps and brief
   lineage.

Adding another paper does not resolve a missing fact in the original paper. The
original `gap_key` therefore stays active when that row/field remains empty.
New-row missing fields also remain visible. Existing comparisons are reaudited,
but a comparative claim involving the newly added row still requires the
normal user-specified paired comparison audit. Affected briefs remain unchanged
until their incremental update decisions are reviewed.

## Same-Source Cell Repair Workflow

`missing_cell` and `unsupported_cell` items expose a separate repair action.
This path never searches another paper to fill the affected row. It:

1. requires the gap's saved matrix revision to equal the current revision;
2. verifies that the row's source file and committed index SHA still match;
3. searches and then scans only chunks whose resolved source identity equals
   the row's source paper, including protection against same-filename files in
   other directories;
4. applies the existing field-specific method, experiment, metric, result, and
   limitation guards, excluding references, captions, positive statements
   about absent limitations, and passages already assigned to another field;
5. returns only exact indexed sentences with a page, heading, block, or anchor
   locator;
6. recomputes the candidate on confirmation and rejects stale or missing
   candidates with HTTP 409;
7. writes the exact sentence and evidence identity into a new matrix revision,
   then reruns the matrix quality contract and every saved comparison involving
   the repaired row.

The repair is marked `grounded` only because the stored value is the exact
same-source evidence sentence; arbitrary manual edits still use the existing
`needs_review` path. A gap becomes `resolved` only when a fresh structured scan
no longer emits that row/field identity. Matrix changes can create a
`brief_stale` item. When lineage reports `matrix_updated`, the UI opens the
affected brief directly in the existing incremental update workflow, where
each proposed block remains separately accepted or rejected.

`source_change` gaps cannot be deferred from this queue; they must be handled in
the source-change inbox. Other gaps may be deferred and remain outside the
active view for the same stable gap identity. An active or in-progress gap that
disappears on a later scan becomes `resolved`; if the same structured problem
then returns, it reopens.

## API Surface

- `GET /api/projects/{project_id}/research-gaps`
- `POST /api/projects/{project_id}/research-gaps/scan`
- `POST /api/projects/{project_id}/research-gaps/{gap_id}/ignore`
- `GET /api/projects/{project_id}/research-gaps/{gap_id}/candidates`
- `POST /api/projects/{project_id}/research-gaps/{gap_id}/candidates/{candidate_id}/confirm`
- `GET /api/projects/{project_id}/research-gaps/{gap_id}/candidates/{candidate_id}/expansion`
- `POST /api/projects/{project_id}/research-gaps/{gap_id}/candidates/{candidate_id}/expansion/apply`
- `GET /api/projects/{project_id}/research-gaps/{gap_id}/repairs`
- `POST /api/projects/{project_id}/research-gaps/{gap_id}/repairs/{repair_id}/apply`
- `GET /api/projects/{project_id}/evidence-matrices/{matrix_id}/comparison-candidates`
- `POST /api/projects/{project_id}/evidence-matrices/{matrix_id}/comparison-candidates/{candidate_id}/audit`

List status is one of `active`, `open`, `in_progress`, `ignored`, or `resolved`.
Project ownership is checked on every mutation. Candidate confirmation fails
with HTTP 409 if a fresh server-side search no longer returns the candidate.

## Release Checks

Run the focused contract, persistence, API, and browser checks:

```bash
python -m pytest -q tests/unit/test_research_gap.py tests/unit/test_research_gap_repair.py tests/unit/test_research_gap_expansion.py tests/unit/test_chat_store_research_gaps.py tests/sanity/test_research_gaps_api.py
python tools/evidence_matrix/run_research_gap_eval.py --db-root db
python tools/evidence_matrix/run_research_gap_repair_eval.py --db-root db
python tools/evidence_matrix/run_research_gap_expansion_eval.py --db-root db
cd web
npm run lint
npm run build
node scripts/playwright-with-port.mjs research-gap-workspace.spec.ts
```

Then run the complete backend suite and Playwright smoke suite. The unchanged
release gates in `docs/RESEARCH_QA_EVAL_RUNBOOK.md` remain mandatory: 29-question
full-library QA, five-question paid-model smoke, deterministic retrieval, source
validation, reviewed replay, and evidence/citation accuracy. A faster or emptier
queue is not a substitute for any of those gates.

## 2026-08-10 Real-Corpus Acceptance

The queue replayed the five previously reviewed real evidence matrices from
`test_results/evidence_matrix/20260806_012452/deterministic_report.json`. All
five cases passed. The queue reproduced all 9/9 reviewed missing cells exactly,
without adding unsupported gap identities. It searched sources outside each
matrix and returned 17 candidate passages; 17/17 matched the recorded indexed
chunk from the same source and carried a reader locator.

The accepted report is
`test_results/research_gap/20260810_135217/report.json`. Deterministic gap
aggregation had a 0.606 ms median and 1.344 ms maximum. Searching every gap in
a case had a 268.763 ms median and 1,340.134 ms maximum. Candidate latency is
separate from ordinary chat retrieval and is paid only when the researcher
opens a gap and asks for local candidates.

The unchanged product gates also passed after the implementation:

1. Full-library live QA: 29/29 in
   `test_results/research_qa_research_gap_full_library/20260810_134330`.
2. Paid-model smoke: 5/5 in
   `test_results/research_qa_research_gap_live_smoke/20260810_134250`.
3. Deterministic full-library retrieval: 29/29; source validation: 41/41;
   reviewed replay: 6/6; paired comparison audit: 5/5 with zero false
   comparisons; evidence-change replay: 5/5.
4. Backend suite: 4,348 passed with 43 configuration-dependent skips. Frontend
   smoke: 122 passed with two private-auth-gate-only skips; all 109 core
   citation/library tests and all four public-surface tests passed. ESLint and
   the production build passed.

The final 29-question run reported first-visible p50/p95/max of
3,000/5,802/7,630 ms, answer-complete p50/p95/max of
5,517/11,774/12,859 ms, and final validation p50/p95/max of
8,690/14,470/17,249 ms. The queue does not modify the ordinary answer path, so
these values remain a visible real-model/provider baseline rather than a speed
claim for this feature.

## 2026-08-10 Same-Source Repair Acceptance

The repair replay created one holdout per reviewed real matrix by removing a
previously grounded cell while preserving its original paper and indexed
corpus. All 5/5 cases recovered the exact original value from the same source
chunk with a reader locator, applied it as a non-manual grounded cell, and
finished with zero unsupported cells. Searching the nine pre-existing honest
missing cells did not mutate the matrices: all 9/9 structured missing
identities remained present until an explicit repair application.

The accepted report is
`test_results/research_gap_repair/20260810_152124/report.json`. Same-source deep
search had a 91.209 ms median and 135.090 ms maximum. Applying the selected
repair and rerunning affected matrix/comparison audits had an 8.869 ms median
and 11.395 ms maximum. These timings are local deterministic measurements; the
path is explicit and does not alter ordinary chat retrieval or generation.

The unchanged release contract also passed after the final code: full-library
live QA 29/29, paid-model smoke 5/5, deterministic retrieval 29/29, source
validation 41/41, reviewed replay 6/6, backend 4,353 passed with 43 skips,
frontend smoke 123 passed with two private-auth-gate-only skips, core E2E
109/109, and public-surface E2E 4/4. Exact result paths and phase timings are in
`docs/RESEARCH_QA_EVAL_RUNBOOK.md`.

## 2026-08-10 Cross-Source Expansion Acceptance

The expansion replay exercised five real structured gaps from the reviewed
matrix corpus. Each selected candidate came from a source outside the current
matrix, matched its exact indexed chunk, and carried a reader locator. All 5/5
previews produced a separately grounded row. All 5/5 applications preserved
the complete old row and evidence prefixes, kept new evidence bound to the new
paper, introduced zero unsupported cells, and left the original paper's gap
identity visible.

The accepted report is
`test_results/research_gap_expansion/20260810_155105/report.json`. Extractive
preview p50/max was 77.227/97.896 ms; application plus saved-comparison
reaudit p50/max was 10.246/11.062 ms. These timings exclude the explicit user
review pauses and do not alter ordinary chat generation.

The final unchanged product gates passed: full-library live QA 29/29, paid-model
smoke 5/5, deterministic retrieval 29/29, source validation 41/41, reviewed
replay 6/6, backend 4,355 passed with 43 skips, frontend smoke 124 passed with
two private-auth-gate-only skips, core E2E 109/109, and public-surface E2E 4/4.
The 29-question provider run retained a visible long tail rather than hiding it;
exact report paths and phase timings are recorded in
`docs/RESEARCH_QA_EVAL_RUNBOOK.md`.
