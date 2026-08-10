# Research QA Eval Runbook

Updated: 2026-08-10

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
