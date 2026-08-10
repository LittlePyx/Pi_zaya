# Project Research Status Runbook

Updated: 2026-08-10

## Purpose

The project research status center answers one operational question: given the
current project sources and audited artifacts, what should the researcher do
next? It combines source freshness, matrix verification, explicit evidence
gaps, comparison coverage, and research-brief lineage into one deterministic
status contract and exactly one primary action.

This is navigation over existing evidence contracts, not a new model judgment.
It does not summarize paper quality, infer whether a topic is important, accept
a comparison, fill a missing fact, or rewrite a brief. Every recommendation is
traceable to persisted structured state or a fresh comparison-candidate scan.

## Fixed Quality-First Order

The backend uses this order and stops at the first applicable condition:

1. refresh changed or stale-index source evidence;
2. repair unsupported evidence or a matrix that has not passed verification;
3. fill explicit missing evidence cells;
4. resolve saved comparisons that failed the strict comparability contract;
5. review evidence-bound comparison candidates with human confirmation;
6. update a stale or lineage-blocked research brief;
7. add at least two usable local full-text sources;
8. create or verify an evidence matrix;
9. refresh comparison-candidate coverage if it has not been measured;
10. create or verify a matrix-backed research brief;
11. open the current verified brief for export.

A changed source therefore cannot be hidden by a polished brief, and a pending
comparison cannot be skipped merely to reach an export action. `ready` requires
fresh sources, no active gaps, complete candidate-scan coverage, no pending
candidates, at least one verified matrix, and a verified brief whose lineage is
`current` or `current_equivalent`.

## Measured Coverage

Opening or refreshing the center runs the normal evidence-change and research-
gap scan, then scans every verified matrix with at least two rows for structured
comparison candidates. The corpus chunks are loaded once and reused across all
eligible matrices. Matrices with stale indexed sources are counted separately
and route to the higher-priority source-refresh action.

The response exposes:

- eligible, scanned, and stale-skipped matrix counts;
- total pending candidates, examined row pairs, and structured observations;
- artifact-load, gap-scan, comparison-scan, assembly, and total elapsed time;
- every stage's counts and status;
- the exact matrix, brief, gap count, or candidate count behind the primary
  action.

The fast `GET` endpoint is a persisted snapshot and intentionally reports the
comparison scan as incomplete. It cannot claim the project is ready. The UI
uses the `POST .../refresh` endpoint on open so the visible recommendation is
based on current measured coverage.

## UI Workflow

Open a project's actions menu in the left sidebar and choose **Research
status**. The center shows one primary action plus the five source, matrix,
evidence, comparison, and brief stages. The primary button routes directly to:

- the affected matrix and its source-change inbox;
- the project research-gap queue;
- the affected matrix's **Comparisons** tab;
- the affected or new research brief;
- or the project literature basket, creating a project conversation first when
  no suitable conversation exists.

Opening a comparison candidate does not accept it. The matrix workspace still
requires exact paired evidence, any server-marked semantic confirmations, a
fresh server-side recomputation, and the unchanged strict comparison audit.

## API Surface

- `GET /api/projects/{project_id}/research-status`
- `POST /api/projects/{project_id}/research-status/refresh`

Both return contract version 1 with `readiness`, `stages`, `gap_counts`, one
`recommended_action`, `comparison_scan`, and `phase_timings_ms`. Project
ownership/existence is checked before loading any artifact.

## Release Checks

Run the focused deterministic and real-paper checks:

```bash
python -m pytest -q tests/unit/test_project_status.py tests/sanity/test_project_research_status_api.py
python tools/evidence_matrix/run_project_status_eval.py --dry-run
python tools/evidence_matrix/run_project_status_eval.py --db-root db
cd web
npm run lint
npm run build
node scripts/playwright-with-port.mjs project-action-center.spec.ts
```

Then run the full backend and frontend suites. The unchanged release gates in
`docs/RESEARCH_QA_EVAL_RUNBOOK.md` remain mandatory: 29-question full-library
QA, 5/5 paid-model smoke, deterministic 29-question retrieval, 41 source
contracts, reviewed answer replay, and strict comparison evidence accuracy.
The status center must never gain speed by dropping a matrix from coverage,
marking an unresolved state ready, or replacing exact evidence checks with a
model opinion.

## 2026-08-10 Real-Paper Baseline

`docs/project_research_status_eval_v1.json` reuses the reviewed SCIGS/SCINeRF
sources and indexed corpus from the comparison-candidate gate. It represents
five successive project states: changed source, unsupported evidence, pending
comparison candidates, stale brief, and current export-ready brief.

The final report at
`test_results/project_research_status/20260810_210830/report.json` passed 5/5.
The real corpus contained 2,233 chunks; the scan found 18 pending comparison
candidates in 82.002 ms, and every candidate retained exact same-source evidence
plus a reader locator. Building the project recommendation took a 2.662 ms
median and 2.856 ms maximum. These measurements separate corpus loading and
candidate discovery from the inexpensive deterministic status assembly. The
earlier baseline remains at
`test_results/project_research_status/20260810_195918/report.json` so the final
release result does not replace or hide the first measurement.
