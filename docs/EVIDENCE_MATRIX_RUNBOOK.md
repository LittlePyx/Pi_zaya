# Evidence Matrix Runbook

Updated: 2026-08-10

## Purpose

> Temporary release-surface status (2026-08-19): the evidence-matrix
> workspace and its matrix-dependent research brief, research gap, and project
> status entry points are hidden from ordinary builds. Existing APIs, stored
> records, exports, and regressions remain intact. Internal testing must set
> `VITE_ENABLE_EVIDENCE_MATRIX_WORKSPACE=1`. Do not restore the ordinary-user
> entry until the revised multi-source synthesis, field relevance, Markdown
> cleaning, and usability gates have passed.

The project evidence matrix turns selected local papers into a durable,
reviewable comparison before synthesis. Each paper has cells for method,
dataset or experiment, metric, key result, and limitation, plus manual notes.
The matrix is project-scoped, versioned, searchable, editable, and exportable as
Markdown, CSV, or XLSX.

Verified matrices also support explicit paired comparison audits. A researcher
names the task, dataset, evaluation protocol, metric, compared target, and
reported value for each source. The verifier locates those phrases in each
paper's own local full text and produces a ranking, replication agreement, or
reporting conflict only when the complete contract passes.

Before entering that contract manually, a researcher can run a high-precision
candidate scan over structured metric-table chunks. A candidate only preloads
reviewable values and exact source excerpts. It is not a comparison conclusion:
every semantic mapping still requires explicit confirmation, and the server
recomputes the candidate before invoking the same strict paired verifier.

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
7. Open **Comparisons** and run **Find comparison candidates**. Inspect the two
   independently located structured-table excerpts, all 12 prefilled contract
   values, and the exact task/dataset/protocol/metric labels. Confirm each
   required semantic mapping; no audit action is enabled before confirmation.
8. Run the candidate audit, or select two sources and enter the contract
   manually. The server recomputes candidate evidence and stores only the
   strict verifier's comparable/not-comparable result. Inspect both paired
   excerpts and their reader locators.
9. Use **Scan evidence changes** or review the automatic scan when the workspace
   opens. Inspect the affected rows, fields, comparison audits, briefs, and
   citations before acknowledging metadata-only changes or confirming an
   affected-source refresh.

## Evidence Change Inbox

Each generated matrix stores an exact SHA-256 full-text fingerprint and a
separate bibliographic-metadata fingerprint for every selected source. The
project scan compares that immutable generation baseline with the current
literature basket and current files. It persists and deduplicates five visible
event types:

- `source_added`, `source_removed`, and `source_content_changed` can be applied;
- `source_unavailable` stays visible until the source is repaired or removed
  from the basket;
- `source_metadata_changed` is explicitly metadata-only and can be
  acknowledged without rebuilding evidence.

The scan keeps the matrix's existing sources stable when a project basket is
larger than the eight-source matrix contract. Newly added basket sources fill
only remaining matrix slots, so basket ordering cannot create a false removal.

No event changes a matrix automatically. Applying confirmed events uses
optimistic concurrency, rebuilds only rows and evidence for the affected
sources, preserves unaffected rows and manual notes, recomputes comparison
boundaries, and re-audits only saved comparisons that touch an affected row.
It then runs the complete matrix evidence contract and creates exactly one new
immutable revision. Matrix-backed briefs remain historical snapshots and show
`matrix_updated`; their existing incremental-update review is the only path to
a new brief revision.

A changed or newly added full text must have a current chunk index before it
can be applied. The API returns HTTP 409 when the file hash and indexed hash do
not match, rather than rebuilding a row from stale chunks. An unavailable
source is likewise blocked until repaired or removed. The UI reports these
states instead of hiding them or treating fewer sources as a successful speed
optimization.

Legacy matrices have no historical file fingerprint. Their first scan creates
a baseline from the matrix's recorded sources and current files, so it can
immediately detect basket additions/removals but does not claim to know whether
a file changed before monitoring began.

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
coverage; it is not evidence quality. Unreviewed cell differences remain only
informational boundaries.

A comparison audit is `verified` only when both sources independently contain
the stated task, dataset, protocol, and controlled metric, and each source has
one same-source passage that jointly contains its dataset, metric, target, and
numeric result. Units must match and the metric direction must be known. Task,
dataset, protocol, and replication-target aliases can be accepted only through
an explicit user-confirmed mapping; metrics use a controlled alias registry and
cannot be manually equated. A failed contract is stored as `not_comparable`
with concrete reasons and never emits a preferred source. A
`confirmed_conflict` is limited to unequal repeated reports of the same
user-confirmed target under the complete matched contract; its wording
explicitly avoids claiming a broader scientific contradiction.

Comparison candidate discovery adds no weaker path around this contract. It
accepts only active verified-matrix rows and `table_metric` structured chunks;
requires an exact normalized task and dataset on both sides; uses only the
controlled metric alias registry and matching result units; and requires an
exact quote plus page, heading, block, or anchor locator for each source. It
prefills four dimensions, two targets, and two results for both papers (12
values total). Protocol differences are exposed as a required human mapping,
not silently normalized. Metric equivalence and cross-dataset ranking cannot be
confirmed manually. Saved audits suppress duplicate candidates for the same row
pair, dataset, and metric.

Candidate audit is an optimistic-concurrency mutation. The server reloads the
current verified matrix, rejects any stale source index, regenerates the
candidate from the current corpus, accepts only the mappings the candidate
explicitly marked for review, and then calls the unchanged strict paired audit.
The resulting matrix revision updates comparison quality, rescans project gaps,
and reports affected brief lineage. A `not_comparable` result remains stored and
visible; it is never converted into a ranking to make the workflow look faster.

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
- `POST /api/evidence-matrices/{matrix_id}/comparison-audits`
- `DELETE /api/evidence-matrices/{matrix_id}/comparison-audits/{comparison_id}`
- `GET /api/projects/{project_id}/evidence-matrices/{matrix_id}/comparison-candidates`
- `POST /api/projects/{project_id}/evidence-matrices/{matrix_id}/comparison-candidates/{candidate_id}/audit`
- `GET /api/projects/{project_id}/evidence-changes`
- `POST /api/projects/{project_id}/evidence-changes/scan`
- `POST /api/projects/{project_id}/evidence-changes/{event_id}/ignore`
- `POST /api/evidence-matrices/{matrix_id}/evidence-changes/apply`
- `GET /api/evidence-matrices/{matrix_id}/export?format=markdown|csv|xlsx`

Refreshing requires `matrix_id` and `expected_revision`. Generation rejects an
unknown basket key or any selected item without its own local Markdown full
text. `PATCH` accepts only row IDs, notes, and the five factual cell fields; it
cannot replace source identity or evidence locators.

## Release Checks

Run the focused backend and UI checks:

```bash
python -m pytest -q tests/unit/test_evidence_matrix.py tests/unit/test_evidence_comparison_candidates.py tests/unit/test_evidence_watch.py tests/unit/test_chat_store_evidence_matrices.py tests/sanity/test_evidence_matrices_api.py tests/sanity/test_evidence_comparison_candidates_api.py tests/unit/test_research_brief.py tests/sanity/test_research_briefs_api.py
python tools/evidence_matrix/run_comparison_eval.py --dry-run
python tools/evidence_matrix/run_comparison_eval.py --db-root db
python tools/evidence_matrix/run_comparison_candidate_eval.py --dry-run
python tools/evidence_matrix/run_comparison_candidate_eval.py --db-root db
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

For paired comparisons, the reviewed fixture is
`docs/evidence_comparison_eval_v1.json`. Require all comparable and
not-comparable cases to match their reviewed outcome, zero false comparisons,
same-source exact evidence, and a reader locator on every recorded excerpt.
Candidate discovery adds
`docs/evidence_comparison_candidate_eval_v1.json`. Require every reviewed pair
to be discovered and then pass the strict audit, every discovered candidate to
have exact same-source locators and all 12 prefilled values, and zero
cross-dataset or uncontrolled-metric candidates.

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

## 2026-08-07 Comparison Audit And Matrix-Build Latency Acceptance

The paired comparison verifier was evaluated on five reviewed cases drawn from
the real SCIGS and SCINeRF corpus: one LPIPS ranking, one repeated-report
agreement, and three intentional no-comparison cases covering a dataset
mismatch, a metric mismatch, and a fabricated result. The final report is
`test_results/evidence_comparison/20260807_012534/report.json`. All 5/5 matched
the reviewed outcome, every recorded excerpt was exact same-source evidence
with a reader locator, and there were zero false comparisons. Corpus loading
took 35.3 ms; audit median/max were 67.7/69.9 ms, with a 103.1 ms cold median
estimate. Failed contracts retained their explicit reasons and emitted neither
a preferred side nor a conflict claim.

The same five real two-paper matrices used by the prior acceptance retained all
41 supported factual cells. Before the path-matching optimization, matrix build
median was 1,053.6 ms and evidence-audit median was 7.7 ms. The final five
builds were 193.6, 188.5, 131.0, 166.3, and 167.1 ms: median 167.1 ms, maximum
193.6 ms, and an 84.1% lower median. The separate evidence audit remained 7.6
ms median. The change normalizes each selected source identity once instead of
resolving the same path for every chunk; it does not reduce the corpus, query
facets, selected sources, populated cells, or evidence checks.

The product workflow now persists each audit with matrix revision history,
supports optimistic-concurrency upsert/delete, re-audits saved comparisons on
refresh, exposes paired evidence and phase timings in the React workspace, and
includes verified source-specific observations in downstream research briefs.
It never inserts a cross-source winner sentence into a brief.

The unchanged release gates passed in the final state: 29/29 live full-library
QA in
`test_results/research_qa_evidence_comparison_final_full_library_fixed/20260807_012052`,
5/5 paid-model smoke, 29/29 deterministic retrieval, 41/41 source validation,
and 6/6 reviewed replay. Frontend ESLint, the production build, 119/119 executed
Playwright smoke tests with two configuration-dependent skips, 109/109 core
tests, and 4/4 public-surface tests also passed. The backend suite completed
with 4,327 passed and 43 configuration-dependent skips.

## 2026-08-08 Evidence Change Inbox Acceptance

The real-corpus change replay used the accepted two-paper dynamic-3D matrix and
a third indexed paper. Full-text change, metadata-only change, source addition,
source removal, and source unavailability all matched their reviewed event and
actionability contract (5/5). The changed real row exposed four affected
fields, one bound brief, and one affected citation; a removed real row exposed
all five populated fields in that source. Detection median/max were 0.562/0.680
ms after the exact file snapshots had been captured. The report is
`test_results/evidence_change_watch/20260808_133414/report.json`.

Persistence tests proved that repeated scans deduplicate the same event,
acknowledgement survives later scans, and project deletion cascades baselines
and events. The API and browser tests proved that applying a newly added source
creates one matrix revision, rebuilds only that source, and preserves an
unaffected reviewed row byte-for-byte. A changed source with a stale index is
rejected instead of using old chunks. Metadata-only and unavailable states stay
visible and are not silently treated as a successful refresh.

The unchanged release gates passed in the final state: 29/29 live full-library
QA at
`test_results/research_qa_evidence_watch_full_library_release/20260808_131635`,
5/5 paid-model smoke, 29/29 deterministic retrieval, 41/41 source validation,
6/6 reviewed replay, and 5/5 comparison audit with zero false comparisons. The
backend suite completed with 4,344 passed and 43 configuration-dependent skips.
Frontend ESLint and the production build passed; Playwright passed 121 executed
smoke tests with two configuration-dependent skips, 109/109 core tests, and
4/4 public-surface tests.

Two preliminary full-library live runs each had one different stochastic
failure: one answer failed the exact denoising claim/locator binding and one
extra English background card was suppressed in the Chinese locale. Both cases
passed immediate focused reruns and the independent final 29-question run.
These failures remain recorded in their reports; they were not hidden by
weakening a validator or reducing the source set.

## 2026-08-10 Comparison Candidate Acceptance

The candidate workflow was evaluated against five human-reviewed SCIGS/SCINeRF
table results spanning Airplants, Cozy2room, Factory, Hotdog, and Tanabata and
the LPIPS, PSNR, and SSIM metrics. The accepted report is
`test_results/evidence_comparison_candidates/20260810_192837/report.json`.
All 5/5 reviewed pairs were discovered and then passed the unchanged strict
paired audit. The scan found 18 total candidates; all 18 had exact same-source
table evidence, reader locators, controlled metrics, and all 12 contract values.
There were zero failed candidate contracts, zero evidence/locator failures,
zero incomplete prefills, zero cross-dataset candidates, and zero
uncontrolled-metric candidates. The one protocol difference on each reviewed
pair remained an explicit human confirmation.

Loading the 2,233-chunk corpus took 39.549 ms in the accepted run. Candidate
scanning took 69.066 ms; strict audit median/max were 68.982/71.758 ms. These
are local deterministic timings and exclude human review. The feature does not
alter the ordinary retrieval or generation path and does not claim to measure
saved human time.

The unchanged release gates passed after implementation: paid-model smoke 5/5
at `test_results/research_qa_eval/20260810_185202`, final full-library live QA
29/29 at `test_results/research_qa_eval/20260810_191501`, deterministic retrieval
29/29, source validation 41/41, reviewed replay 6/6, reviewed Agent replay 5/5,
paired comparison audit 5/5 with zero false comparisons, and converter quality
13/13. The backend suite passed 4,358 tests with 43 configuration-dependent
skips. Frontend smoke passed 125 tests with two private-auth-gate-only skips;
core E2E passed 109/109 and public-surface E2E passed 4/4. ESLint and the
production build passed.

One preliminary 29-question live run was 28/29 because an extra background
reference card was locale-suppressed and therefore failed the existing card-copy
contract. The answer and primary evidence were valid; the affected case then
passed two focused reruns and the independent final 29/29 run. The failing
report remains at `test_results/research_qa_eval/20260810_185236`; no source,
card, evidence, or citation gate was changed. An initial full backend run also
hit one pre-existing Windows scheduler boundary in a synthetic deadline test;
the exact test passed five consecutive reruns and the final full suite passed.
