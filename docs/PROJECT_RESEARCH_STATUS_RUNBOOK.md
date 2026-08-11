# Project Research Status Runbook

Updated: 2026-08-11

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

After the primary action opens a child workspace, the chat composer keeps a
compact project-journey control. **Continue with the next action** refreshes the
measured project status and opens the next required stage; **End workflow**
clears the journey explicitly. This state is carried in the URL, so closing a
matrix or brief does not strand the researcher or silently skip a stage.

After a comparison candidate is accepted, its review card is removed and the
remaining-candidate count is updated. An accepted card cannot remain visible as
an apparently actionable control that would only produce a duplicate-request
error.

The comparison workspace presents one candidate at a time and groups the queue
by paper pair, task, and dataset. Metrics remain separate candidates, with
previous/next controls and left/right keyboard navigation. A reviewed semantic
mapping is reused only when matrix, row pair, task, dataset, dimension, and both
normalized dimension values are identical. Reuse never crosses a dataset or
paper pair. Every candidate still sends its own confirmation list to the server,
which recomputes it from the current index, checks the current matrix revision,
and runs the complete paired evidence audit before saving a new revision.

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
python tools/evidence_matrix/run_project_research_journey_eval.py --dry-run
python tools/evidence_matrix/run_project_research_journey_eval.py --out-root test_results/project_research_journey
cd web
npm run lint
npm run build
node scripts/playwright-with-port.mjs project-action-center.spec.ts evidence-matrix-workspace.spec.ts
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

## 2026-08-11 Continuous Real-Project Journey Acceptance

`docs/project_research_journey_eval_v1.json` defines a reviewed three-paper
project using SCIGS, SCINeRF, and the indexed 3D single-pixel-video paper. The
evaluator drives only public project APIs through the complete operational
sequence: add sources, create a matrix, resolve or explicitly defer reviewed
gaps, audit every discovered comparison candidate, create a matrix-backed brief,
and export the verified current brief. It verifies exact indexed evidence,
same-source identity, reader locators, source coverage, matrix and brief audits,
lineage, and the exact six-action recommendation sequence.

The first complete run remains at
`test_results/project_research_journey_baseline/20260811_143756/report.json`. It
passed 13/18 checks but correctly stopped before ready/export because dense
comparison observations exhausted the 20-hit brief budget and excluded the
third source. The brief covered only 2/3 papers, its bibliography contained two
sources, and the audit marked it `needs_review`. This was not retried away or
accepted as a timing success.

The fix reserves one ordinary grounded matrix cell for every active row before
filling the remaining brief hit budget with verified comparison observations.
It retains the comparison evidence and the 20-hit budget while guaranteeing
that a dense source pair cannot starve another selected paper. The exact-code
run at
`test_results/project_research_journey_source_balance/20260811_144024/report.json`
passed 18/18: all three sources and bibliography entries were represented, the
13-evidence matrix and 20-evidence brief were verified, all 18 comparison
candidates were audited, the brief was current, and the status reached ready
and export. Total wall time was 16,899.798 ms; brief generation took 5,303.581
ms; all comparison audits took 5,754.490 ms; status refresh median was 218.205
ms.

The fixture permits exactly two explicit deferrals: the reviewed unavailable
limitation cells for SCIGS and the 3D-video source. They remain visible as honest
source limitations and are never filled with invented evidence. No evidence,
source, locator, comparison, matrix-audit, brief-audit, or readiness gate was
removed to obtain the final pass.

The final release rerun against the complete working tree and the CI-portable
fixture check is
`test_results/project_research_journey_final_release/20260811_150754/report.json`.
It again passed 18/18 with 13 matrix evidence records, 18/18 verified
comparisons, 20 brief evidence records, all three sources in the bibliography,
current lineage, and a verified export. Total wall time was 16,102.208 ms;
brief generation took 4,598.957 ms; all comparison audits took 5,744.330 ms;
status refresh median/max were 223.543/237.132 ms. The companion five-state
status report at
`test_results/project_research_status_journey_release/20260811_150304/report.json`
passed 5/5 with 2.357/2.567 ms status-build median/max.

## 2026-08-11 Grouped Comparison Review Acceptance

The three-paper fixture now also pins the human review shape: all 18 candidates
must remain discoverable, they must form six paper/task/dataset groups, and the
18 required protocol confirmations may collapse only to six exact contextual
signatures. This saves 12 repeated confirmations (66.7%) while retaining 18
separate server recomputations and strict audits.

The final report at
`test_results/project_comparison_review_workbench/20260811_213123/report.json`
passed 20/20. The verified matrix retained 13 evidence records; all 18
comparisons passed exact two-source evidence and locator checks; the current
brief retained 20 evidence records and all three bibliography sources. Total
wall time was 21,245.084 ms, comparison audits totaled 7,340.202 ms, and brief
generation took 5,834.247 ms. These times are higher than the preceding
16,102.208 ms project run and remain visible as real local/model variance. The
feature reduces human repetition, not server evidence work, source coverage, or
quality thresholds.

The final UI regression exercised the grouped workbench, exact in-group reuse,
cross-dataset isolation, keyboard navigation, and an independent audit request
for every accepted candidate. The evidence-matrix file passed 4/4, the complete
smoke set passed 127/127 applicable tests, and the unchanged full release gates
remained 29/29 live QA, 5/5 paid-model smoke, 29/29 deterministic retrieval,
41/41 source validation, and both comparison acceptance sets at 5/5.
