# Research QA Eval Runbook

Updated: 2026-08-12

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

## 2026-08-05 Research-Brief Acceptance

The project research-brief workflow reuses the Research Agent verifier under a
strict basket scope. It does not accept a brief as `verified` unless every
audited claim is supported, every selected source contributes evidence, every
visible citation resolves, and no evidence comes from outside the selected
literature basket. A cited sentence whose cited snippet does not support the
claim now fails the generation gate as well as the final audit.

Real acceptance used five two-paper research objectives spanning dynamic 3D
reconstruction, single-pixel parallelism, basis selection versus foveation,
denoising taxonomy versus a restoration baseline, and QCLFM versus iISM depth
signals. All five passed the final evidence audit. Four passed as direct model
synthesis; the cross-language QCLFM case was explicitly labeled as a safe,
source-balanced extractive fallback after Chinese synthesis could not pass the
English-evidence overlap check. Its fallback still verified 4/4 claims and
covered both selected sources.

The unchanged product QA gates then passed:

1. Full-library live QA: 29/29 in
   `test_results/research_qa_final_full_library/20260805_110231`.
2. Paid-model smoke: 5/5 in
   `test_results/research_qa_final_live_smoke/20260805_110202`.
3. Deterministic full-library retrieval: 29/29 in
   `test_results/research_qa_final_retrieval/20260805_110146`.
4. Source validation: 41/41; reviewed replay: 6/6.
5. Backend suite: 4309 passed with 43 configuration-dependent skips; frontend
   ESLint, production build, and the 118-test Playwright smoke suite passed.

The final 29-question run reported first-visible p50/p95/max of
2504/3871/5021 ms, answer-complete p50/p95/max of 4117/9123/11326 ms, and
UI-ready p50/p95/max of 7259/11174/12372 ms. This keeps 29/29 coverage while
reducing the previous accepted UI-ready maximum from 14092 ms to 12372 ms.

## 2026-08-06 Evidence-Matrix Acceptance

Adding the persistent project evidence matrix did not change the ordinary chat
generation path. The unchanged gates passed after the implementation:

1. Full-library live QA: 29/29 in
   `test_results/research_qa_matrix_final_full_library/20260806_012819`.
2. Paid-model smoke: 5/5 in
   `test_results/research_qa_matrix_final_live_smoke/20260806_012749`.
3. Deterministic full-library retrieval: 29/29 in
   `test_results/research_qa_matrix_final_retrieval/20260806_012651`.
4. Source validation: 41/41; reviewed replay: 6/6.
5. Backend suite: 4,317 passed with 43 configuration-dependent skips; frontend
   ESLint, production build, and the 121-test Playwright smoke suite passed 119
   executed tests with two configuration-dependent skips.

The final 29-question run reported first-visible p50/p95/max of
2618/4170/5764 ms, answer-complete p50/p95/max of 4153/11339/13597 ms, and
UI-ready p50/p95/max of 7672/13531/14681 ms. Quality and coverage stayed 29/29,
but the real-model p95 and maximum are higher than the 2026-08-05 run. No
ordinary QA generation code changed in this feature, so treat this as visible
provider/model long-tail variance rather than claiming a latency improvement.
The evidence-matrix-specific acceptance, including its phase timings and 5/5
matrix-backed brief audit, is documented in `docs/EVIDENCE_MATRIX_RUNBOOK.md`.

## 2026-08-06 Matrix-Brief Synthesis Tail-Latency Acceptance

The follow-up optimization is isolated to verified evidence-matrix briefs. The
ordinary research QA path keeps its existing full-answer repair and fallback
behavior. Matrix briefs instead build a source-balanced eight-item claim plan,
ask the model for exactly one sentence per plan item, audit every sentence, and
preserve only fully supported claims. A matrix claim now fails if any visible
citation does not match, a publication year belongs to another cited source, a
contrast clause is unsupported, or the answer exceeds the eight-claim brief
contract. Missing source coverage is supplemented from the verified matrix;
otherwise the existing source-balanced extractive fallback remains available.
Any targeted repair is disclosed in the saved brief and UI.

The prior five-matrix live report had a median/max of 8,135/8,295 ms, with one
direct model synthesis and four extractive fallbacks. A phased diagnostic on
the basis-selection case measured 5,302 ms for the first model answer and
2,944 ms for the old second whole-answer rewrite. The accepted run in
`test_results/evidence_matrix_brief_latency/20260806_022537/live_report.json`
had a median/max of 3,963/4,065 ms. All five briefs passed: four were direct
model synthesis, one preserved seven supported model claims after a 5.6 ms
targeted repair, and none used extractive fallback. The final audit supported
35/35 claims, represented every selected source, resolved every citation, and
used no unexpected source.

The unchanged product gates passed after the final implementation:

1. Full-library live QA: 29/29 in
   `test_results/research_qa_matrix_brief_latency_final_full_library/20260806_020615`.
2. Paid-model smoke: 5/5 in
   `test_results/research_qa_matrix_brief_latency_final_live_smoke/20260806_021111`.
3. Deterministic full-library retrieval: 29/29 in
   `test_results/research_qa_matrix_brief_latency_final_retrieval/20260806_021158`.
4. Source validation: 41/41; reviewed replay: 6/6.
5. Backend suite: 4,321 passed with 43 configuration-dependent skips; frontend
   ESLint, two production builds, 119/119 executed smoke tests with two skips,
   109/109 core tests, and 4/4 public-surface tests passed.

The final 29-question run reported first-visible p50/p95/max of
3,461/4,891/6,514 ms, answer-complete p50/p95/max of
5,342/12,611/15,240 ms, and UI-ready p50/p95/max of
9,398/15,703/16,576 ms. These ordinary-QA tail values are higher than the prior
matrix acceptance even though its generation path did not change; keep them
visible as provider/model variance rather than claiming an ordinary-QA latency
improvement from the matrix-brief optimization.

## 2026-08-07 Comparison-Audit Release Gate

The first two full-library runs after adding paired matrix comparison audits
were 28/29. Both exposed the same real defect in
`foveated-dynamic-supersampling`: the answer and required foveated citation were
correct, but an extra NatPhoton card stayed empty. Investigation showed a more
important upstream issue than card copy: the binder had treated the paper title
"single-pixel imaging" as if it were support in a selected Hadamard-subset
passage. The fix removes a repeated document title from the positive
body-evidence surface and requires authoritative fast bindings to have passage-
level terms, identifiers, numbers, mapped concepts, or domain overlap. A direct
`transformer network` alias preserves a valid deep-learning training citation
without relying on the title. The quality gate was not weakened.

The focused rerun then kept only the supported foveated source and passed. The
final full-library run passed 29/29 at
`test_results/research_qa_evidence_comparison_final_full_library_fixed/20260807_012052`.
All answer terms, required documents, citation routes, locators, reference-card
copy, citation-shelf checks, and System B audits passed. First-visible
p50/p95/max were 2,517/4,177/4,495 ms; answer-complete were
4,037/13,467/14,793 ms; evidence-card complete were 6,738/15,321/16,346 ms;
and UI-ready were 6,738/16,323/17,043 ms. The 5/5 paid-model smoke run passed at
`test_results/research_qa_evidence_comparison_final_live_smoke/20260807_005509`,
and unchanged deterministic retrieval passed 29/29 at
`test_results/research_qa_evidence_comparison_final_retrieval/20260807_005600`.
Source validation remained 41/41 and reviewed replay remained 6/6.
The backend suite completed with 4,327 passed and 43 configuration-dependent
skips; frontend ESLint, production build, 119/119 executed Playwright smoke
tests with two skips, 109/109 core tests, and 4/4 public-surface tests passed.

The comparison-specific reviewed report is
`test_results/evidence_comparison/20260807_012534/report.json`: 5/5 outcomes,
zero false comparisons, exact same-source evidence, and a locator for every
recorded excerpt. These gates demonstrate that the feature neither narrows the
29-question library search nor substitutes polished copy for accurate evidence.

## 2026-08-08 Research-Brief Lineage Acceptance

Research briefs now distinguish an immutable historical verification from
freshness against the latest evidence-matrix revision. The change does not alter
chat retrieval, answer generation, source selection, citation binding, or the
evidence-matrix quality contract. A valid older snapshot stays exportable with
an explicit historical lineage marker; export is blocked only when the source
matrix, recorded revision, fingerprint integrity, or revision order cannot be
verified. Updating a stale brief remains bound to the same verified matrix and
reruns the complete brief generation and evidence audit.

The real lineage replay at
`test_results/research_brief_lineage/20260808_022355/report.json` passed 5/5
reviewed matrices. Each case removed one actually cited grounded cell as an
honest gap while the unchanged matrix contract remained verified. All five
briefs reported `matrix_updated`, exactly one changed field, and affected
citation `[1]`; evidence-equivalent revision changes reported
`current_equivalent`, and missing source matrices blocked export with
`matrix_missing`. The cases completed in 14.23-25.78 ms. Local detailed lineage
enrichment measured a 6.61 ms median and 7.10 ms p95, while raw store access
remained a 3.19 ms median.

The unchanged product gates passed after the final implementation:

1. Full-library live QA: 29/29 in
   `test_results/research_qa_lineage_final_full_library_pass/20260808_025212`.
2. Paid-model smoke: 5/5 in
   `test_results/research_qa_lineage_final_live_smoke/20260808_023315`.
3. Deterministic full-library retrieval: 29/29 in
   `test_results/research_qa_lineage_final_retrieval/20260808_022408`.
4. Source validation: 41/41; reviewed replay: 6/6; reviewed comparison audit:
   5/5 with zero false comparisons.
5. Backend suite: 4,331 passed with 43 configuration-dependent skips; frontend
   ESLint, production build, 120/120 executed Playwright smoke tests with two
   skips, 109/109 core tests, and 4/4 public-surface tests passed.

The final 29-question run reported first-visible p50/p95/max of
3,138/4,718/6,330 ms, answer-complete p50/p95/max of
5,049/11,927/14,572 ms, and UI-ready p50/p95/max of
8,215/14,849/16,072 ms. The first full-library attempt exposed a real CASSI
boundary: a model sentence was rebound from the verified Abstract fact to a
same-paper experimental passage and consequently lost its clickable citation.
The finalizer now inserts the exact planned Abstract statement when the model
mentions the two opposing dispersers and binary aperture without citing that
claim. Unsupported optical-path detail remains subject to the existing removal
gate. The focused CASSI rerun and the subsequent complete 29-question run both
passed; no lineage status hides a quality failure or narrows retrieval to
improve speed.

## 2026-08-08 Reviewable Incremental Brief-Update Acceptance

Stale matrix-backed research briefs now update through a persisted review
plan rather than replacing the whole document. The plan is bound to the brief
revision and content hash plus the source and target matrix revisions. It
proposes changes only for citation-bearing spans affected by the matrix diff,
shows old and proposed Markdown, and requires an explicit accept or keep-old
decision for every item. Applying a plan creates an immutable brief revision
and reruns the complete citation, source, and evidence audit over the merged
document. Rejected items remain visible as `needs_review`; direct whole-brief
replacement of a stale matrix-backed brief is rejected by the API.

The deterministic replay at
`test_results/research_brief_incremental_update/20260808_095434/report.json`
passed 5/5 reviewed matrices. Every unchanged span was preserved byte for
byte, the minimum unaffected-content preservation ratio was 80.94%, all five
merged briefs remained verified, and there were no unresolved citations. The
deterministic plan median/max was 1.01/1.02 ms. The paid live-model replay at
`test_results/research_brief_incremental_update/20260808_095853/report.json`
also passed 5/5. Four cases used focused model synthesis; one candidate failed
the unchanged citation verifier and safely used source-extractive fallback.
All five final full-document audits passed. Live plan median/max was
1,550.51/2,000.26 ms versus 3,962.71/4,064.58 ms for the recorded legacy
whole-regeneration baseline, which also did not preserve unaffected manual
content.

The unchanged product gates passed after the final implementation:

1. Full-library live QA: 29/29 in
   `test_results/research_qa_incremental_update_full_library/20260808_100146`.
2. Paid-model smoke: 5/5 in
   `test_results/research_qa_incremental_update_live_smoke/20260808_100055`.
3. Deterministic full-library retrieval: 29/29 in
   `test_results/research_qa_eval/20260808_100042`.
4. Source validation: 41/41; reviewed replay: 6/6; reviewed comparison audit:
   5/5 with zero false comparisons. The final lineage replay remained 5/5 at
   `test_results/research_brief_lineage/20260808_102647/report.json`.
5. Backend suite: 4,336 passed with 43 configuration-dependent skips; frontend
   ESLint, production build, 120/120 executed Playwright smoke tests with two
   skips, 109/109 core tests, and 4/4 public-surface tests passed.

The final 29-question run reported first-visible p50/p95/max of
3,248/5,338/6,070 ms, answer-complete p50/p95/max of
5,546/11,188/14,562 ms, evidence-card complete p50/p95/max of
7,837/12,988/15,888 ms, and UI-ready p50/p95/max of
8,706/14,312/15,888 ms. These gates retain the complete 29-question retrieval
coverage, the 5/5 live answer-quality contract, and exact evidence validation;
latency is reduced by limiting generation scope, not by dropping sources,
citations, audits, or user-visible review states.

## 2026-08-08 Evidence Change Inbox Acceptance

Adding persisted evidence-source fingerprints and the project change inbox did
not change the ordinary chat retrieval, answer-generation, citation, or card
contracts. The final release run passed all 29 full-library cases at
`test_results/research_qa_evidence_watch_full_library_release/20260808_131635`.
First-visible p50/p95/max were 2,823/4,147/5,282 ms; answer-complete were
4,536/11,658/16,062 ms; evidence-card complete were
6,760/12,354/17,409 ms; and UI-ready were 7,431/14,720/17,409 ms.

The five-question real-model smoke passed 5/5 at
`test_results/research_qa_evidence_watch_live_smoke/20260808_130506` with
UI-ready p50/p95/max of 4,026/5,428/5,436 ms. Deterministic full-library
retrieval passed 29/29, source validation passed 41/41, reviewed replay passed
6/6, and the paired comparison audit passed 5/5 with zero false comparisons.

Two earlier complete live attempts each produced one different non-repeating
failure (`denoising-classical-map` claim/locator binding, then a locale-
suppressed extra background card in `foveated-dynamic-supersampling`). Both
passed focused reruns and the final complete suite. They remain visible as
model/card long-tail variance; no validator, citation requirement, source
coverage rule, or timeout was weakened to obtain the final pass.

## 2026-08-10 Research Gap Queue Release Gate

The project research gap queue reads existing matrix audits, comparison
boundaries, brief lineage, and source-change events. It does not modify ordinary
chat retrieval, generation, citation binding, reference-card rendering, or the
quality validator. Candidate search is an explicit project action and excludes
the current matrix's sources; confirming a candidate only adds its exact local
passage and locator to the literature basket.

The unchanged product gates passed after implementation:

1. Full-library live QA: 29/29 in
   `test_results/research_qa_research_gap_full_library/20260810_134330`.
2. Paid-model smoke: 5/5 in
   `test_results/research_qa_research_gap_live_smoke/20260810_134250`.
3. Deterministic full-library retrieval: 29/29; source validation: 41/41;
   reviewed replay: 6/6; paired comparison audit: 5/5 with zero false
   comparisons; evidence-change replay: 5/5.
4. The five reviewed real evidence matrices reproduced all 9/9 missing cells;
   all 17 retrieved candidate passages were exact same-source indexed excerpts
   with locators. See `docs/RESEARCH_GAP_RUNBOOK.md`.
5. Backend suite: 4,348 passed with 43 configuration-dependent skips. Frontend
   smoke: 122 passed with two private-auth-gate-only skips; all 109 core
   citation/library tests and all four public-surface tests passed. ESLint and
   the production build passed.

The final 29-question run reported first-visible p50/p95/max of
3,000/5,802/7,630 ms, answer-complete p50/p95/max of
5,517/11,774/12,859 ms, evidence-card p50/p95/max of
8,089/13,013/14,067 ms, and final validation p50/p95/max of
8,690/14,470/17,249 ms. These values stay visible as ordinary real-model and
provider variance; no evidence or coverage gate was weakened to obtain them.

## 2026-08-10 Same-Source Gap Repair Gate

The same-source repair path is an explicit project action outside ordinary chat
retrieval and generation. It accepts only exact indexed sentences from the
affected matrix row's freshly indexed source paper, recomputes the candidate on
apply, creates a new matrix revision, and reruns matrix and affected comparison
audits. Other-paper discovery remains a separate literature-basket workflow and
cannot fill the original row.

The five reviewed real matrices passed 5/5 repair holdouts in
`test_results/research_gap_repair/20260810_152124/report.json`. Every recovered
cell exactly matched its original same-source indexed value and carried a reader
locator. The nine honest pre-existing missing identities remained 9/9 after
search-only replay. Same-source search measured 91.209/135.090 ms median/max;
apply plus re-audit measured 8.869/11.395 ms median/max. These checks supplement,
and do not replace, the unchanged 29-question retrieval, 5/5 live QA, source,
citation, and evidence validation gates.

The final product gates after all implementation and evidence-binding fixes were:

1. Full-library live QA: 29/29 in
   `test_results/research_qa_gap_repair_full_library_release/20260810_151311`.
2. Paid-model smoke: 5/5 in
   `test_results/research_qa_gap_repair_live_smoke_release_v2/20260810_151232`.
3. Deterministic full-library retrieval: 29/29; source validation: 41/41;
   reviewed replay: 6/6; paired comparison audit: 5/5 with zero false
   comparisons; same-source repair replay: 5/5 while preserving all 9/9 honest
   missing identities.
4. Backend suite: 4,353 passed with 43 configuration-dependent skips. Frontend
   smoke: 123 passed with two private-auth-gate-only skips; all 109 core
   citation/library tests and all four public-surface tests passed. ESLint and
   the production build passed.

The final 29-question run reported first-visible p50/p95/max of
2,825/5,672/7,321 ms, answer-complete p50/p95/max of
5,016/10,522/12,232 ms, evidence-card p50/p95/max of
7,666/12,370/13,044 ms, and final validation p50/p95/max of
8,508/13,687/14,463 ms. The final five-question smoke reported UI-ready
p50/p95/max of 4,232/6,565/6,823 ms.

Earlier complete runs remain under `test_results`: one 29-question run failed a
PIDL answer whose retrieval-boundary sentence was rebound to an unplanned hit,
and one five-question run failed because SCINeRF's compact evidence card kept
the variable definitions and differentiability sentence but omitted the
intervening synthesis sentence. The release fixes align the claim audit with
the visible citation-plan allowlist and preserve the exact, page-4 compound
source excerpt within the card budget. No source, citation, coverage, or quality
threshold was relaxed.

## 2026-08-10 Cross-Source Gap Expansion Gate

Cross-paper discovery now has two separate human decisions. The first confirms
an exact candidate into the project literature basket. The second reviews a
full extractive row preview before adding that paper as a new matrix source.
The workflow recomputes the candidate and index freshness, rejects current
matrix sources, preserves all old rows/evidence, reaudits saved comparisons,
refreshes source monitoring, and marks bound briefs stale. It never fills the
original paper's missing cell, and a new-row comparison still requires the
normal paired comparison audit.

Five real structured gaps passed 5/5 in
`test_results/research_gap_expansion/20260810_155105/report.json`. All five
candidates were outside the matrix and exact-locator-bound; all old rows and
evidence were preserved; all new evidence was exact and same-source; and all
five original gap identities remained visible. Preview p50/max was
77.227/97.896 ms and apply-plus-reaudit p50/max was 10.246/11.062 ms.

The final product gates after this implementation were:

1. Full-library live QA: 29/29 in
   `test_results/research_qa_gap_expansion_full_library_release/20260810_160203`.
2. Paid-model smoke: 5/5 in
   `test_results/research_qa_gap_expansion_live_smoke_release/20260810_160129`.
3. Deterministic retrieval: 29/29; source validation: 41/41; reviewed replay:
   6/6; reviewed Agent replay: 5/5; comparison audit: 5/5; converter quality:
   13/13; cross-source expansion: 5/5.
4. Backend suite: 4,355 passed with 43 configuration-dependent skips. Frontend
   smoke: 124 passed with two private-auth-gate-only skips; core citation/library
   E2E: 109/109; public-surface E2E: 4/4. ESLint and the production build passed.

The 29-question run reported first-visible p50/p95/max of
2,521/11,481/12,669 ms, answer-complete p50/p95/max of
4,425/20,200/38,476 ms, evidence-card p50/p95/max of
6,664/22,562/39,579 ms, and final validation p50/p95/max of
7,231/22,962/40,550 ms. One provider stream crossed the visible-output deadline,
but the affected case completed through the existing quality path and all 29
strict cases passed; the tail remains visible rather than being removed from the
report. The five-question smoke reported UI-ready p50/p95/max of
4,017/6,524/6,885 ms. No retrieval, source, citation, locator, or evidence-card
gate was reduced.

## 2026-08-10 Evidence-Bound Comparison Candidate Gate

Verified evidence matrices can now discover paired comparison inputs from
structured metric-table chunks. The discovery stage requires exact normalized
task and dataset, controlled metric identity, matching units, and an exact
same-source locator on both sides. It only preloads a contract. A researcher
must confirm every flagged semantic mapping, after which the server recomputes
the candidate and runs the existing strict paired audit before creating a
matrix revision, rescanning gaps, or refreshing brief lineage.

The reviewed candidate report is
`test_results/evidence_comparison_candidates/20260810_192837/report.json`. Five
reviewed real comparisons passed 5/5. The scan discovered 18 candidates with
zero strict-audit, evidence/locator, or incomplete-prefill failures, zero
cross-dataset candidates, zero uncontrolled metrics, and 12 prefilled contract
values per candidate. Candidate scan time was 69.066 ms; strict-audit median/max
were 68.982/71.758 ms. The original reviewed
paired audit independently remained 5/5 with zero false comparisons.

The complete unchanged product gates were:

1. Paid-model smoke: 5/5 in
   `test_results/research_qa_eval/20260810_185202`.
2. Final full-library live QA: 29/29 in
   `test_results/research_qa_eval/20260810_191501`.
3. Deterministic retrieval: 29/29; source validation: 41/41; grounded replay:
   6/6; reviewed Agent replay: 5/5; converter quality: 13/13.
4. Backend: 4,358 passed and 43 configuration-dependent skips. Frontend smoke:
   125 passed and two private-auth-gate-only skips; core E2E 109/109;
   public-surface E2E 4/4. ESLint and production build passed.

The accepted 29-question run reported first-visible p50/p95/max of
2,482/3,918/5,335 ms, answer-complete p50/p95/max of
3,870/9,535/12,359 ms, evidence-card p50/p95/max of
6,684/10,459/13,593 ms, and final validation p50/p95/max of
7,803/11,409/14,737 ms. The five-question smoke reported final validation
p50/p95/max of 4,144/8,415/9,174 ms.

A preliminary full-library run was 28/29 because one additional background
reference card was correctly rejected after locale suppression left it without
summary and relevance copy. That case passed two focused reruns, followed by the
independent final 29/29 run. The original failure remains in
`test_results/research_qa_eval/20260810_185236`; the validator, source set, and
evidence/card thresholds were not changed.

## 2026-08-10 Project Research Status and Citation-Binding Release Gate

The project research status center measures source freshness, matrix
verification, explicit gaps, complete comparison-candidate coverage, and brief
lineage, then returns exactly one deterministic next action. Its fixed priority
order prevents a polished brief or export action from hiding changed sources,
unsupported evidence, unresolved gaps, or pending comparisons. Opening the
center performs the current gap and comparison scans and exposes their coverage
and phase timings; it does not accept evidence, confirm a semantic mapping, or
generate a conclusion.

The real-paper status report at
`test_results/project_research_status/20260810_210830/report.json` passed all
5/5 project states. It loaded 2,233 real indexed chunks, found all 18 pending
SCIGS/SCINeRF comparison candidates, retained exact same-source evidence and
reader locators, and built the deterministic recommendation in 2.662/2.856 ms
median/max. The companion candidate report at
`test_results/evidence_comparison_candidates/20260810_210831/report.json` also
passed 5/5 with zero evidence, prefill, contract, cross-dataset, or uncontrolled-
metric failures. Its complete scan took 84.217 ms and strict re-audit took
104.494/107.158 ms median/max.

The first complete live release attempt was 28/29 at
`test_results/research_qa_project_status_full_library/20260810_202439`. The
`denoising-classical-map` case returned the right paper and answer shape but
bound the two-child spatial-domain detail as if it satisfied the complete
spatial/transform taxonomy claim, then retained a weaker stale card locator.
The release fix ranks taxonomy evidence by complete claim alignment, recognizes
the same leaf heading across article-title variants, and permits a prompt-
aligned page locator to replace weaker evidence only when claim overlap
improves. The focused real-model retest passed with the exact page-2 taxonomy at
`test_results/research_qa_project_status_denoising_fix_v2/20260810_205400`.

The final unchanged release gates were:

1. Full-library live QA: 29/29 in
   `test_results/research_qa_project_status_full_library_release/20260810_210412`.
2. Paid-model smoke: 5/5 in
   `test_results/research_qa_project_status_live_smoke_release/20260810_210839`.
3. Deterministic full-library retrieval: 29/29 in
   `test_results/research_qa_eval/20260810_214001`; source validation: 41/41;
   grounded replay: 6/6; reviewed Agent replay: 5/5; converter quality: 13/13.
4. Backend: 4,372 passed and 43 configuration-dependent skips. Frontend smoke:
   126 passed with two private-auth-gate-only skips; core E2E: 109/109; public-
   surface E2E: 4/4. Ruff, ESLint, and the production build passed.

The accepted 29-question run reported first-visible p50/p95/max of
2,917/4,861/4,946 ms, answer-complete p50/p95/max of
4,610/10,557/12,714 ms, evidence-card p50/p95/max of
7,436/12,872/13,848 ms, UI-ready p50/p95/max of
7,436/13,637/14,393 ms, and final-validation p50/p95/max of
8,810/13,963/14,950 ms. Compared with the initial 28/29 run, every recorded
maximum fell, including final validation from 16,985 to 14,950 ms; p95 values
were not uniformly lower and remain visible as real provider variance. No
timeout, evidence term, locator, source scope, card threshold, or case was
removed to obtain the pass.

An additional non-gating diagnostic deliberately ran all 56 fixture questions
at
`test_results/research_qa_project_status_full_library_fixed/20260810_205431`.
It passed 49/56 and exposed seven long-tail backlog cases:
`iism-live-cell-benefit`, `qclfm-correlation-resolution`,
`scinerf-motion-assumption-en`, `perovskite-dual-cavity-threshold`,
`sph-heterodyne-sampling-conditions`, `three-d-video-daq-budget`, and
`sequential-cs-two-stage-elimination`. These are answer-term, citation-card
copy, and high-risk claim-binding defects outside the formal 29-question
release suite; they are recorded rather than reclassified, hidden, or used to
weaken the release gate. The expanded run's final-validation p50/p95/max was
8,440/15,271/20,114 ms and defines the next long-tail quality backlog.

## 2026-08-11 Full 56-Question Long-Tail Acceptance

The recorded seven-case backlog is now closed without removing a case, source,
answer term, evidence term, citation route, page locator, or reference-card
quality check. The fixes keep answer generation and evidence validation strict:

1. Citation-plan evidence selection preserves the complete requested source
   sentence or table boundary instead of binding a nearby, partially matching
   passage.
2. Claim/evidence finalization restores only prompt-requested facts that are
   present in the selected source evidence, including the Sequential Compressed
   Sensing method name and its second-stage `k log n` measurement count.
3. Answer-citation reference cards use the final grounded render packet when a
   stored card wins the background race before final citation details are
   durable. Empty guide/relevance copy remains non-terminal.
4. A complete public card clears stale cache `pending` state only after it has a
   source path, grounded evidence excerpt, and localized guide. Citation-card
   evidence and its reader locator are updated atomically, so an Abstract quote
   cannot inherit a Method page or anchor from another same-paper occurrence.

The exact-code final run passed all 56/56 questions and completed all 56/56
evidence-card snapshots in
`test_results/research_qa_long_tail_full_56_release_verified/20260811_031858`:

| Milestone | p50 / p95 / max |
|---|---:|
| First visible answer | 3,122 / 4,550 / 6,347 ms |
| Answer complete | 4,721 / 10,805 / 13,152 ms |
| Evidence cards complete | 7,747 / 12,727 / 14,982 ms |
| End-to-end UI ready | 8,454 / 13,409 / 14,982 ms |
| Validation snapshot complete | 8,818 / 14,842 / 17,365 ms |
| Full evaluator wall time | 9,052 / 14,873 / 17,397 ms |

Compared with the 49/56 backlog-defining run, final-validation p95 fell from
15,271 to 14,842 ms (2.8%) and the maximum fell from 20,114 to 17,365 ms
(13.7%); p50 increased from 8,440 to 8,818 ms (4.5%) and remains visible as
real model/provider variance. More importantly, all seven previously failing
quality cases now pass, and every card reaches a quality-checked terminal
state.

Two failed attempts remain intentionally visible. The run at
`test_results/research_qa_long_tail_full_56_final_release/20260811_025258`
passed 55/56 and exposed a mixed SCIGS Abstract-evidence/Method-locator card.
The next run at
`test_results/research_qa_long_tail_full_56_final_green/20260811_030623`
passed 56/56 but exposed one stale cached `pending` bit, producing a 55,917 ms
validation maximum. Neither failure was retried away or accepted as green; both
causes were fixed and regression-tested before the final 56/56 run.

The unchanged release gates also passed:

1. Standalone full-library live QA: 29/29 in
   `test_results/research_qa_full_library_acceptance_release/20260811_023724`;
   standalone paid-model smoke: 5/5 in
   `test_results/research_qa_live_smoke_release/20260811_024158`. Every case in
   both suites also ran inside the later exact-code 56/56 final run.
2. Deterministic full-library retrieval: 29/29 in
   `test_results/research_qa_retrieval_only_release/20260811_024255`; source
   grounding: 41/41; grounded replay: 6/6; reviewed Agent replay: 5/5.
3. Backend unit tests: 4,126 passed with 41 configuration-dependent skips;
   backend sanity: 262 passed with two skips; the visible Agent contract passed
   5/5; Ruff passed.
4. Frontend ESLint and the production build passed. Playwright smoke passed 126
   tests with two private-auth-only skips; core citation/library E2E passed
   109/109; ordinary-user public-surface isolation passed 4/4.

The final acceptance does not shorten the evidence-card wait timeout or treat a
pending card as ready. It removes false pending states only after the same
source, evidence, localization, and locator contracts have already passed.

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

## 2026-08-11 Continuous Project-Journey Acceptance

The project research-status follow-up now validates a real three-paper journey,
not only isolated status snapshots. The first full run at
`test_results/project_research_journey_baseline/20260811_143756/report.json`
passed 13/18 and exposed a genuine source-balance defect: dense verified
comparison observations consumed the matrix-brief hit budget before the third
selected paper could contribute evidence. The brief audit remained
`needs_review`, so the workflow correctly did not reach ready/export.

The accepted run at
`test_results/project_research_journey_source_balance/20260811_144024/report.json`
passed 18/18 after reserving one ordinary grounded matrix cell per active source
row. It retained all 18 strictly audited comparison candidates, produced a
20-evidence verified brief with all three bibliography sources, preserved two
reviewed unavailable limitation cells as explicit deferrals, reached the exact
six-action sequence, and exported only after current verified lineage. The
16,899.798 ms total, 5,303.581 ms brief stage, 5,754.490 ms comparison-audit
total, and 218.205 ms status-refresh median remain visible.

This gate supplements the unchanged 56/56 long-tail acceptance, 29/29 formal
full-library suite, 5/5 paid-model smoke, 29/29 deterministic retrieval, 41/41
source validation, grounded and Agent replays, and strict comparison evidence
checks. It does not lower any answer, evidence, citation, locator, source-
coverage, audit, readiness, or export requirement.

The complete exact-code release gates then passed:

1. Full-library live QA: 29/29 in
   `test_results/research_qa_project_journey_full_library_release/20260811_145735`;
   paid-model smoke: 5/5 in
   `test_results/research_qa_project_journey_live_smoke_release/20260811_150139`.
2. Deterministic retrieval: 29/29 in
   `test_results/research_qa_project_journey_retrieval_release/20260811_150225`;
   source validation: 41/41; paired comparison audit: 5/5 with zero false
   comparisons; candidate audit: 5/5 with 18 discoveries and zero evidence or
   contract failures.
3. The final continuous project journey passed 18/18 at
   `test_results/project_research_journey_final_release/20260811_150754/report.json`;
   the independent status snapshots passed 5/5.
4. Backend unit tests passed 4,127 with 41 configuration-dependent skips;
   sanity passed 262 with two skips. Frontend smoke passed 126 with two private-
   auth-only skips; core citation/library E2E passed 109/109; public-surface E2E
   passed 4/4. Ruff, ESLint, and the production build passed.

The final 29-question run reported first-visible p50/p95/max of
2,868/4,757/5,952 ms, answer-complete p50/p95/max of
3,969/10,447/13,156 ms, evidence-card complete p50/p95/max of
8,226/11,519/14,674 ms, and final-validation p50/p95/max of
9,157/11,911/15,706 ms. The five-question smoke's final-validation p50/p95/max
was 3,935/9,421/10,436 ms. These real-model tails remain visible; no timeout,
source, evidence, locator, card, or quality threshold was reduced.

## 2026-08-11 Grouped Comparison Review Workbench

The comparison-candidate UI now groups the real 18-item SCIGS/SCINeRF queue into
six dataset groups and shows one fully evidenced metric candidate at a time.
Confirmation reuse is deliberately narrower than text equality alone: matrix,
paper rows, task, dataset, dimension, and both dimension values must match. A
confirmation therefore applies to the three metrics within one dataset group,
but never to another dataset, task, or paper pair.

The exact-code project report at
`test_results/project_comparison_review_workbench/20260811_213123/report.json`
passed 20/20. It reduced 18 repeated protocol-confirmation actions to six exact
signatures while still performing and passing 18/18 independent server-side
candidate recomputations and strict paired audits. Matrix evidence remained 13,
brief evidence remained 20, bibliography coverage remained 3/3, and the brief
remained verified and current. The 21,245.084 ms total and 7,340.202 ms audit
total are retained as measured variance; no answer, retrieval, source, evidence,
locator, comparison, brief, readiness, or export gate was weakened.

The complete release rerun against the final worktree also passed. Full-library
live QA was 29/29 at
`test_results/research_qa_comparison_review_workbench_full_library_release/20260811_214828`,
with first-visible p50/p95/max of 2,804/4,758/5,617 ms and final-validation
p50/p95/max of 8,074/12,172/14,994 ms. The independent paid-model smoke was 5/5
at
`test_results/research_qa_comparison_review_workbench_live_smoke_release/20260811_215235`,
with final-validation p50/p95/max of 4,074/9,370/10,420 ms. Deterministic
retrieval remained 29/29, source validation remained 41/41, paired comparisons
passed 5/5 with zero false comparisons, and candidate acceptance passed 5/5
with all 18 candidates discovered and zero contract, evidence, prefill,
cross-dataset, or uncontrolled-metric failures.

Backend unit and sanity suites passed 4,127 and 262 tests respectively, with 41
and two configuration-dependent skips. Frontend smoke passed all 127 applicable
tests with two private-auth-only skips; core citation/library regressions passed
109/109; public-surface isolation passed 4/4. Ruff, the Agent 5/5 contract,
reviewed replays, converter 13/13, ESLint, and the production build all passed.

## 2026-08-12 Historical Same-Corpus A/B Acceptance

The strict historical evaluator in `tools/version_ab/run_version_ab_eval.py`
compares the 2026-07-11 baseline (`e11096db`) with the exact product candidate
(`0c47135d`). Its checked contract is `docs/version_ab_eval_v1.json`: 29 full-
library questions, five paid-model smoke questions, a 45-second per-case total
deadline, and three complete project research journeys. Unsupported historical
capabilities, timeouts, missing coverage, and quality failures count as failures;
they are never skipped.

Both services used the same active, read-only corpus root and separate writable
chat/library databases. The strong corpus fingerprint covered 1,472 files and
568,628,457 bytes, with aggregate SHA-256
`5c679c7fd920fbe4d698d37118643dc8d3fb027548a54d080ced93b478e9e8a5`.
The `docs.json` and `references_index.json` identities also matched exactly.

The accepted report is
`F:\research-papers\2026\Jan\else\kb_chat_ab_runtime\version_ab_fair\20260812_002419\report.json`.
The first aggregate incorrectly treated the valid integer zero in
`failed: 0` as missing; the tested report rebuild recomputed the aggregate from
the unchanged per-suite JSON/JSONL and project reports. It now reports a complete
comparison, release-ready candidate, QA pass delta +34, and project pass delta
+3.

| Version | Full library | Live smoke | Project journeys | Full UI p50 / p95 / max | Smoke UI p50 / p95 / max |
|---|---:|---:|---:|---:|---:|
| 2026-07-11 baseline | 0/29 | 0/5 | 0/3 unsupported | 45,056 / 45,967 / 48,322 ms | 45,084 / 45,507 / 45,606 ms |
| 2026-08-12 candidate | 29/29 | 5/5 | 3/3, each 20/20 | 8,285 / 13,324 / 15,974 ms | 3,970 / 7,673 / 8,235 ms |

Candidate first-visible p50/p95 was 2,740/4,523 ms for the 29-question suite
and 2,066/2,873 ms for smoke. The baseline full suite produced no usable first-
answer samples; all 29 cases hit the bounded runner-error path. In baseline
smoke, three cases timed out and two completed answers still failed answer,
evidence, citation-card, route, or locator contracts. The old version also lacks
the evidence-matrix, research-gap, and research-brief journey APIs, so all three
historical project journeys remain explicitly unsupported rather than hidden.

The old service was terminated after its smoke suite because historical cancel
requests could leave provider calls running for minutes. The candidate's first
four full-library cases began during that conservative handoff and all passed in
1.16-11.38 seconds; retaining them can only make the candidate latency result
worse, not manufacture an improvement. The remaining candidate cases ran after
the old port was closed. No answer terms, source coverage, retrieval top-k,
evidence binding, citation route, locator, card-quality, project audit, or export
gate was reduced.

CI validates the immutable A/B contract without calling a model:

```bash
python tools/version_ab/run_version_ab_eval.py --dry-run
```

For a live comparison, start both exact revisions against the same corpus and
run the evaluator with explicit repositories, URLs, and source roots. If only an
aggregate bug is corrected after the expensive run, recompute solely from the
saved immutable artifacts:

```bash
python tools/version_ab/run_version_ab_eval.py --rebuild-report <report.json>
```

## 2026-08-12 Answer-to-Citation Tail-Latency Follow-up

This follow-up measured the current answer, citation-card, UI-ready, and final-
validation milestones separately before changing code. The baseline repeated ten
long-tail cases three times under
`F:\research-papers\2026\Jan\else\kb_chat_ab_runtime\tail_latency_current\baseline_repeat_20260812`.
It passed 29/30 cases. The retained failure was the second-round foveated dynamic
supersampling answer: its second card's summary and relevance copy were too short
and the terminal reference state was still pending. It was not retried, removed,
or reclassified.

Profiling showed that the dominant controllable post-answer work was not
retrieval or evidence selection. The final reference read was rerunning whole-
paper answer-alignment and canonical-evidence scans even though generation had
already persisted an answer-bound citation plan and canonical hit evidence. On
representative cases, `finish_overlay_refresh` alone cost 3.2-5.9 seconds. A
fresh profiled render took 4.668 seconds, including 2.239 seconds in canonical
citation augmentation and 1.613 seconds in answer-alignment. The change reuses
only source-identity-checked, answer-ordinal-checked persisted evidence and exact
block/anchor plans. Legacy, incomplete, mismatched, and unbound payloads retain
the existing scan and repair path.

The same four representative cases then passed 12/12 repeated runs under
`F:\research-papers\2026\Jan\else\kb_chat_ab_runtime\tail_latency_current\optimized_repeat_20260812`.
Median answer-to-card time changed as follows:

| Case | Baseline | Optimized | Change |
|---|---:|---:|---:|
| Hadamard/Fourier basis choice | 7,326 ms | 1,034 ms | -86% |
| Classical denoising map | 3,336 ms | 855 ms | -74% |
| QCLFM refocus | 2,366 ms | 2,293 ms | -3% |
| SCINeRF forward equation | 6,328 ms | 2,753 ms | -56% |

The final exact-code 29-question release is
`F:\research-papers\2026\Jan\else\kb_chat_ab_runtime\tail_latency_current\full_library_release\20260812_020222`.
It passed 29/29 with first-visible p50/p95/max of
2,776/4,819/5,684 ms, UI-ready p50/p95/max of
6,520/12,650/16,526 ms, and final-validation p50/p95/max of
6,520/12,650/17,612 ms. Answer-to-card p50/p95/max was
1,169/2,872/3,781 ms. Compared with the accepted same-corpus candidate above,
UI-ready p50 improved 21.3% and p95 improved 5.1%; max increased 3.5% and is
reported rather than hidden. The max case was PILN: its measured LLM answer
stage took 10.644 seconds, so the remaining extreme is provider/model output,
not the removed citation scan. Answer length, evidence count, source coverage,
and quality gates were not reduced to chase that extreme.

The independent five-question paid-model smoke passed 5/5 at
`F:\research-papers\2026\Jan\else\kb_chat_ab_runtime\tail_latency_current\live_smoke_release\20260812_020817`,
with UI-ready p50/p95/max of 3,830/5,312/5,343 ms and final-validation
p50/p95/max of 3,830/5,820/5,939 ms. Three complete project journeys under
`F:\research-papers\2026\Jan\else\kb_chat_ab_runtime\tail_latency_current\project_journeys_release`
each passed 20/20. Deterministic retrieval remained 29/29, source validation
remained 41/41, and the reviewed grounded replay remained 6/6.

CI-equivalent local validation passed 4,141 backend unit tests with 41
configuration-dependent skips, 262 sanity tests with two skips, the visible
Agent contract 5/5, Ruff, ESLint, the production build, all quality fixtures,
frontend smoke 127 applicable tests with two private-auth skips, core citation
and library regressions 109/109, and public-surface isolation 4/4.
