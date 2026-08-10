# Research Gap Queue Runbook

Updated: 2026-08-10

## Purpose

The project research gap queue turns already-audited evidence problems into a
durable research worklist. It aggregates gaps across evidence matrices,
matrix-backed research briefs, paired comparison audits, and source-change
monitoring without asking a model to invent a task or infer a missing fact.

The queue is intentionally downstream of the existing evidence contracts. It
does not relax matrix verification, fill empty cells, treat a retrieved passage
as accepted evidence, rewrite a brief, or hide a source-change event. It only
helps a researcher see what is missing, understand the impact, find other local
papers that may contain relevant evidence, and record a human-selected
candidate in the project literature basket.

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
basket, and marks the gap `in_progress`. It does not update a matrix cell or
brief. The researcher must use the existing matrix generation, edit, comparison
audit, or brief update workflow to produce a newly verified artifact.

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

List status is one of `active`, `open`, `in_progress`, `ignored`, or `resolved`.
Project ownership is checked on every mutation. Candidate confirmation fails
with HTTP 409 if a fresh server-side search no longer returns the candidate.

## Release Checks

Run the focused contract, persistence, API, and browser checks:

```bash
python -m pytest -q tests/unit/test_research_gap.py tests/unit/test_chat_store_research_gaps.py tests/sanity/test_research_gaps_api.py
python tools/evidence_matrix/run_research_gap_eval.py --db-root db
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
