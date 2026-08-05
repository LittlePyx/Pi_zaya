# Evidence Matrix Runbook

Updated: 2026-08-06

## Purpose

The project evidence matrix turns selected local papers into a durable,
reviewable comparison before synthesis. Each paper has cells for method,
dataset or experiment, metric, key result, and limitation, plus manual notes.
The matrix is project-scoped, versioned, searchable, editable, and exportable as
Markdown, CSV, or XLSX.

This workflow is intentionally conservative. It must not fill a cell by
inference, copy evidence from another paper, broaden retrieval beyond the
selected literature basket, or label unequal metrics and experimental
conditions as a confirmed conflict. If no suitable local passage is found, the
cell stays empty.

## User Workflow

1. Put the conversation in a project and add up to eight locally matched papers
   to its literature basket.
2. Open **Evidence matrix**, enter a title and research objective, then generate.
3. Review the status banner, incomplete-cell filter, and comparison-boundary
   notices.
4. Open the Evidence tab to inspect the exact supporting sentence and jump to
   its page, heading, block, or anchor in the reader.
5. Add notes or edit cells. Notes survive source refreshes. A factual edit is
   marked `needs_review` until the matrix is regenerated from source evidence.
6. Inspect or restore immutable revisions, export the current revision, or use a
   verified matrix to create a research brief.

## Evidence Contract

A matrix is `verified` only when all of these checks pass:

- every selected local source has exactly one active row;
- every selected source contributes at least one supported factual cell;
- every populated factual cell names one or more evidence records from the same
  source and its value is contained in the recorded evidence quote;
- no populated cell is a manual override;
- no row or evidence record comes from outside the selected basket.

Missing cells are explicit and are warnings, not invented facts. They do not
invalidate an otherwise verified matrix. `completeness` describes extraction
coverage; it is not evidence quality. `confirmed_conflicts` remains empty unless
a future verifier can establish comparable conditions and metrics. The current
implementation emits only informational boundaries when experimental
conditions or metric surfaces differ.

The generator performs source-balanced retrieval: it builds one local BM25
index per selected paper, queries all five field facets within that paper, and
extracts the best qualifying source sentence. The full corpus is loaded for
index access, but a paper's row can only use chunks matched to that paper's
recorded `sourcePath`.

## Version and Concurrency Rules

Every update, refresh, and restore uses `expected_revision`. A stale write
returns HTTP 409 rather than overwriting newer work. List endpoints are
lightweight; the workspace loads full rows and evidence for the selected
matrix. Each successful mutation records the previous state as an immutable
revision. Deletion first creates an automatic snapshot and stops if backup
creation fails.

## API Surface

- `GET/POST /api/projects/{project_id}/evidence-matrices`
- `POST /api/projects/{project_id}/evidence-matrices/generate`
- `GET/PATCH/DELETE /api/evidence-matrices/{matrix_id}`
- `GET /api/evidence-matrices/{matrix_id}/revisions`
- `GET /api/evidence-matrices/{matrix_id}/revisions/{revision}`
- `POST /api/evidence-matrices/{matrix_id}/restore`
- `GET /api/evidence-matrices/{matrix_id}/export?format=markdown|csv|xlsx`

Refreshing requires `matrix_id` and `expected_revision`. Generation rejects an
unknown basket key or any selected item without its own local Markdown full
text. `PATCH` accepts only row IDs, notes, and the five factual cell fields; it
cannot replace source identity or evidence locators.

## Release Checks

Run the focused backend and UI checks:

```bash
python -m pytest -q tests/unit/test_evidence_matrix.py tests/unit/test_chat_store_evidence_matrices.py tests/sanity/test_evidence_matrices_api.py tests/unit/test_research_brief.py tests/sanity/test_research_briefs_api.py
cd web
npm run lint
npm run build
npx playwright test tests/e2e/evidence-matrix-workspace.spec.ts tests/e2e/research-brief-workspace.spec.ts
```

For real-data acceptance, use multiple papers from the actual indexed corpus
and record build and audit time separately. Inspect every populated cell for
same-source exact-quote support and a usable reader locator. Generate five
matrix-backed research briefs covering different paper pairs and require all
five to pass source coverage, citation resolution, claim support, and
out-of-scope evidence checks.

Then run the unchanged gates in `docs/RESEARCH_QA_EVAL_RUNBOOK.md`: 29-question
live full-library QA, five-question paid-model smoke, 29-question deterministic
retrieval, source validation, reviewed replay, backend suite, frontend lint and
build, and Playwright smoke. Any evidence, route, locator, coverage, or citation
failure is a release no-go.

## 2026-08-06 Acceptance

Five real two-paper matrices covered dynamic 3D reconstruction, basis selection,
foveation versus real-time differential acquisition, image restoration, and
microscopy depth signals. They populated 41 of 50 factual cells; the other nine
remained explicitly empty. All 41 populated cells matched an exact sentence in
the same selected source and all 41 had a reader locator. Warm two-paper matrix
builds ranged from 979 to 1,083 ms; the separate evidence audit ranged from 5.3
to 8.5 ms. The report is in
`test_results/evidence_matrix/20260806_012452/deterministic_report.json`.

The same five matrices generated real-model research briefs with
`deepseek-v4-flash`. All five passed the final audit: 36/36 claims supported,
both selected sources represented in every brief, no unexpected source, and no
unresolved citation. One passed as model synthesis; four were visibly marked
source-balanced extractive fallbacks after synthesis did not pass the support
or source-coverage gate. The report is in the same directory as
`live_brief_report.json`.

The unchanged product gates also passed: 29/29 live full-library QA, 5/5 paid
smoke, 29/29 deterministic retrieval, 41/41 source validation, 6/6 reviewed
replay, 4,317 backend tests with 43 skips, and 119/119 executed Playwright smoke
tests with two configuration-dependent skips. ESLint and the production build
passed.

## 2026-08-06 Matrix-Brief Tail-Latency Acceptance

The original five real matrix briefs were safe but slow: the accepted baseline
in `test_results/evidence_matrix/20260806_012452/live_brief_report.json` had a
median of 8,135 ms and maximum of 8,295 ms. Only one brief passed as direct
model synthesis; four spent time on a second whole-answer model rewrite and
then used a source-balanced extractive fallback. A detailed basis-selection
diagnostic measured 5,302 ms for initial synthesis plus 2,944 ms for the second
rewrite, or 8,511 ms before fallback.

The optimized path keeps the evidence contract and replaces the second
whole-answer rewrite with these stages:

1. Build up to eight source-balanced claim-plan items from grounded matrix
   cells, prioritizing method, experiment, key result, metric, and limitation.
2. Generate exactly one source-attributed sentence per plan item, with one
   citation marker and no plan-external claims.
3. Audit every visible citation, source year, contrast clause, claim count, and
   selected-source coverage.
4. If needed, preserve supported model sentences, remove only failed or
   out-of-contract sentences, and supplement only a missing selected source
   from its verified matrix evidence. The UI displays the preserved, removed,
   and supplemented counts. If that result still fails, use the existing
   explicit extractive fallback.

The final real-model report is
`test_results/evidence_matrix_brief_latency/20260806_022537/live_report.json`.
All five briefs passed the strict final audit with 35/35 supported claims, no
source gap, no unexpected source, and no unresolved citation. Four passed on
the first model synthesis; the microscopy brief removed one unsupported
contrast clause in a 5.6 ms targeted repair while preserving seven supported
model claims. No brief used extractive fallback.

| Measure | Prior accepted baseline | Final accepted run |
|---|---:|---:|
| Total median | 8,135 ms | 3,963 ms |
| Total maximum | 8,295 ms | 4,065 ms |
| Direct model synthesis | 1/5 | 4/5 |
| Targeted sentence repair | 0/5 | 1/5 |
| Extractive fallback | 4/5 | 0/5 |
| Final evidence audit | 5/5 | 5/5 |

The final run's claim-plan stage stayed below 1 ms, initial citation audit was
below 3 ms, and the only targeted repair took 5.6 ms. The latency reduction
comes from removing the failed second model call, not from lowering retrieval,
source coverage, citation, or support gates.

Unchanged product gates also passed: 29/29 live full-library QA, 5/5 paid-model
smoke, 29/29 deterministic retrieval, 41/41 source validation, 6/6 reviewed
replay, and 4,321 backend tests with 43 configuration-dependent skips. Frontend
ESLint, production build, 119/119 executed smoke tests with two skips, 109/109
core tests, and 4/4 public-surface tests passed.
