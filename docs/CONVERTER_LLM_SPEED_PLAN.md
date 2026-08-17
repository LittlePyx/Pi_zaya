# PDF LLM Converter Speed Plan

## 1. Purpose

This document defines the next execution phase for the `PDF -> Markdown`
converter.

The scope is intentionally narrow:

- focus only on the `LLM / vision_direct` path
- optimize first-run throughput on real PDFs
- keep conversion quality stable enough that there is no systematic regression

This is not a general converter roadmap.

Out of scope for this phase:

- `no_llm` path optimization
- refactoring for its own sake
- warm-cache wins that do not improve first conversion latency
- large rendering-layer changes

## 2. Current State

The codebase is already in a good enough shape to shift the priority from
refactoring to performance work.

What is already in place:

- `pipeline.py` has been reduced substantially and page-level flows have been
  split into helper modules
- `vision_direct` now has a thin orchestration layer and shared page helpers
- both `no_llm` and `vision_direct` reuse worker-local `fitz` documents
- benchmark tooling exists and can emit `JSON + CSV + per-run output dirs`
- `KB_PDF_STAGE_TIMINGS=1` already exposes page-stage timings
- `LLMWorker` already limits request concurrency with `max_inflight`
- page OCR warm-cache already exists, but it only helps repeated runs

What is already known:

- the dominant first-run bottleneck is `Step 6 (vision convert)`
- `assets`, `page render`, and metadata masking are no longer the main issue
- adding more page workers alone will not solve single-page VL cost
- the current `normal` profile is conservative and stable, which makes it a
  good candidate for page-adaptive cost control rather than a global quality cut

## 3. Primary Goal

The primary goal is simple:

- reduce first-run real-PDF wall time for the `normal` LLM path without causing
  a clear quality downgrade

This breaks down into three sub-goals:

1. reduce single-page `vision convert` cost
2. reduce avoidable second calls and retries
3. only then revisit higher concurrency ceilings

## 4. Non-Goals

This phase does not aim to:

- keep shrinking `pipeline.py` before performance work
- optimize `ultra_fast` first
- replace real first-run measurements with warm-cache measurements
- globally lower `dpi` or `max_tokens` without page-level safeguards
- keep adding parallelism when the request itself is still too heavy

## 5. Baseline Understanding

The current `LLM / vision_direct` path is roughly:

1. references-page detection
2. metadata region collection and masking
3. figure and visual-asset extraction
4. full-page render to PNG
5. hint and optional formula-overlay preparation
6. `call_llm_page_to_markdown`
7. page post-processing, guardrails, and fallback logic

Based on current stage timings and real-library runs:

- `Step 6` dominates page latency
- references pages can trigger dedicated crop-based OCR
- empty-output retry and math-quality retry can cause extra full-page VL calls
- `layout crop mode` is currently off by default and is not the main baseline
  bottleneck

This means the optimization strategy must shift from:

- more preprocessing tweaks

to:

- lighter per-page VL requests
- fewer extra VL requests

## 6. Benchmark Policy

All performance claims in this phase must be based on first-run conversion of
real PDFs from the library.

### 6.1 Library Root

Library PDF root:

- `F:\research-papers\research-paper-pyx`

### 6.2 Fixed Sample Set

Use a fixed set of representative papers so that every optimization is measured
against the same corpus.

Recommended sample set:

1. `Psychological Review-1954-Some informational aspects of visual perception.pdf`
   Reason: older, text-heavy, stable layout
2. `Nature-2025-Electrically driven lasing from a dual-cavity perovskite device.pdf`
   Reason: modern layout, figure-heavy
3. `LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.pdf`
   Reason: long review, formula-heavy, references-heavy
4. `Optics & Laser Technology-2024-Part-based image-loop network for single-pixel imaging.pdf`
   Reason: mid-length mixed-content paper
5. `OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture.pdf`
   Reason: older paper with different layout patterns

### 6.3 Profile Policy

Do not spend full-suite time on repeated `normal-1` runs.

Use three benchmark roles:

- `normal-control`
  Use only on one long paper to keep a serial control sample.
  Config:
  `speed_mode=normal,llm_page_workers=1,max_inflight=1`
- `normal-max`
  Main throughput profile.
  Config:
  `speed_mode=normal,llm_page_workers=N,max_inflight=N`
- `normal-mid`
  Optional midpoint when throughput behavior needs one more reference point.

`N` must not be treated as "whatever the code allows".
`N` should mean:

- the highest stable concurrency level for the current provider, account, and
  network conditions

Stability means:

- no timeout cascade
- no broad empty-page failures
- no clear explosion of `429` or provider-side errors

Initial sweep plan:

- test `4 / 8 / 12 / 16`
- lock the highest stable value as `normal-max`
- until the sweep is refreshed, treat `8` as the temporary observation point

### 6.4 Measurement Rules

Each optimization round must report:

- total wall time
- per-PDF elapsed time
- page-level `Step 6` latency distribution
- references-page count and average references latency
- retry count
- fallback count
- output markdown size
- failed page count

Do not accept a single "it got faster" number without breakdown.

## 7. Quality Gate Policy

Quality evaluation cannot rely on exact-output equality alone because the VL
model is not fully deterministic.

Use three gates instead.

### 7.1 Hard Gate

Any of the following is unacceptable:

- missing pages
- major dropped sections
- broken reading order
- missing tables as a structural pattern
- badly broken references pages
- systematic loss of image links or captions
- clearly worse fragmented-math behavior

### 7.2 Soft Gate

The following are acceptable if they stay minor and isolated:

- token-level OCR differences
- punctuation differences
- whitespace differences
- minor reference-line wording variations

### 7.3 Manual Review Checklist

After every major optimization, manually inspect at least:

1. one long review paper
2. one figure-heavy paper
3. one paper with a long references section

Review dimensions:

- headings and hierarchy
- table structure
- figure-caption order
- math completeness
- references line splitting
- duplicated or missing paragraphs

## 8. Guiding Principle

The optimization order must be:

1. reduce per-page cost
2. reduce avoidable extra calls
3. only then revisit more aggressive concurrency

Do not invert this order.

Why:

- the dominant bottleneck is single-page VL latency
- page-level parallelism already exists
- more workers improve throughput but do not make a heavy page request cheaper
- provider-side latency and timeout behavior gets amplified by aggressive
  concurrency

## 9. Workstream A: Page-Adaptive Cost Control

This is the highest-value next step.

### 9.1 Goal

Different page types should use different request budgets instead of sending
every page through the same effective VL cost envelope.

### 9.2 Target Page Classes

Start with four lightweight classes:

1. `references`
2. `text_dense_body`
3. `figure_or_visual_heavy`
4. `formula_sensitive`

Classification must use only cheap local signals that already exist.

Candidate signals:

- references-page detection
- visual-rect count
- extracted figure count
- formula-candidate count
- text density
- metadata-rect pattern

### 9.3 Planned Changes

Add a lightweight page-type classifier and use it to choose:

- effective render `dpi`
- effective `max_tokens`
- whether to keep the full page hint
- whether some guardrails stay active

Initial strategy:

- `references`
  - lower token cap
  - prioritize a lighter references-specific OCR path
- `text_dense_body`
  - lower `dpi`
  - shorter hint
  - lower token cap
- `figure_or_visual_heavy`
  - keep current `normal` quality settings
- `formula_sensitive`
  - keep current `dpi`
  - keep a safer token budget

### 9.4 Constraints

Do not globally lower the `normal` defaults first.

Requirements:

- unknown page types must fall back to current behavior
- the adaptive policy must be feature-flagged
- benchmark logs must expose page class and chosen request budget

### 9.5 Expected Files

- `kb/converter/page_vision_direct_page.py`
- `kb/converter/llm_worker.py`
- `kb/converter/pipeline_vision_direct.py`
- `tests/unit/test_page_vision_direct_page.py`
- `tests/unit/test_pipeline_vision_direct.py`

## 10. Workstream B: Reduce Avoidable Second Calls

This is the second priority.

### 10.1 Problem

The main sources of extra VL calls are:

1. empty-output retry
2. math-quality retry
3. references crop-based OCR

These mechanisms help quality, but if their trigger conditions are too broad,
they increase wall time significantly.

### 10.2 Empty Retry Plan

Tighten empty-output retry into a risk-driven policy.

Policy direction:

- references pages should avoid broad full-page empty retries
- explicit timeout, unsupported-vision, and provider-error paths should not
  trigger more empty retries
- retries should be limited to recoverable, non-references cases

Goal:

- fewer extra full-page VL calls

### 10.3 Math Retry Plan

Do not remove the math-quality gate, but make it more selective.

Policy direction:

- only aggressive on `formula_sensitive` pages
- avoid retry on text-dense pages with no formula evidence
- avoid retry on references pages
- raise the confidence threshold for fragmented-math detection

### 10.4 References Path Plan

The references column mode is valuable, but it should become adaptive.

Target behavior:

- single-column references pages: one request
- true two-column references pages: two crops
- short references pages: lighter path first

### 10.5 Expected Files

- `kb/converter/page_vision_guardrails.py`
- `kb/converter/reference_page_vl.py`
- `kb/converter/pipeline.py`
- `tests/unit/test_pipeline_vision_direct.py`
- `tests/unit/test_page_vision_direct_page.py`

## 11. Workstream C: Prompt And Payload Slimming

This is the third priority and should be treated as low-risk, cumulative gain.

### 11.1 Problem

Current page requests can include:

- a relatively long system prompt
- a relatively long user prompt
- page hint
- figure mapping hint
- references-specific instructions

Not every page needs the same prompt payload.

### 11.2 Plan

Split prompt construction into:

- a stable baseline contract
- small page-type-specific additions

Policy direction:

- text-heavy pages get a shorter prompt
- pages without images do not receive figure mapping hints
- pages without formulas do not receive formula-related guidance
- references pages use a shorter references-specific contract

### 11.3 Guardrail

Prompt slimming must not change the output contract:

- output is still markdown
- headings, tables, figures, formulas, and captions remain supported
- references pages still avoid math wrappers and code fences

### 11.4 Expected Files

- `kb/converter/llm_worker.py`
- `kb/converter/page_vision_direct_page.py`
- `tests/unit/test_pipeline_vision_direct.py`

## 12. Workstream D: Structured Benchmark And Reporting

Without better structured benchmark output, the team will keep optimizing by
intuition.

### 12.1 Goal

Turn the existing benchmark tool into a real performance decision tool.

### 12.2 Required Additions

Add structured reporting for:

- aggregated page-stage timing
- per-PDF page latency `p50 / p90`
- references-page average latency
- retry count
- fallback count
- page-class distribution

### 12.3 Required A/B Output

After each major optimization, produce:

- baseline vs optimized speed summary
- output-file path mapping for the compared PDFs
- a quality-diff report template or generated report

### 12.4 Expected Files

- `kb/converter/benchmark.py`
- `tools/benchmark_converter.py`
- `tests/unit/test_converter_benchmark.py`

## 13. Recommended Implementation Order

Use the following fixed order.

### Phase 1

- extend benchmark output with structured page timing aggregation
- run the `4 / 8 / 12 / 16` concurrency sweep
- lock the current stable `normal-max`

### Phase 2

- land the most conservative version of adaptive `dpi / token` policy
- measure first-run gains on the fixed real-library sample set

### Phase 3

- tighten empty-output retry
- tighten math-retry triggers
- validate quality gates again

### Phase 4

- make the references path adaptive between one-column and two-column handling
- reduce references request cost further

### Phase 5

- slim prompts and hints
- rerun the full benchmark set

### Phase 6

- only after the first five phases flatten out should more aggressive provider
  saturation tests or even higher concurrency be considered

## 14. Success Criteria

This phase is considered successful only if all of the following hold:

1. first-run `normal-max` wall time improves clearly on the fixed real-library
   sample set
2. page-level `Step 6` average latency improves clearly
3. timeout, empty, and fallback counts do not rise
4. unit tests and regression tests stay green
5. manual review finds no new systematic quality regressions

Suggested first target:

- total wall time improvement of at least `25%`
- average `Step 6` latency improvement of at least `20%`
- severe quality regressions equal to `0`

The definition of severe regressions is the `Hard Gate` section above.

## 15. Testing Policy For Every Change

Every optimization slice must run:

1. `py_compile`
2. relevant unit tests
3. real-library benchmark runs
4. at least one long-paper output inspection

Minimum recurring test set:

- `tests/unit/test_pipeline_vision_direct.py`
- `tests/unit/test_page_vision_direct_page.py`
- `tests/unit/test_converter_pipeline.py`
- `tests/unit/test_pipeline_math_guardrails.py`
- `tests/unit/test_converter_benchmark.py`

If the change touches references behavior:

- rerun references-related regression coverage

If the change touches prompt, retry, or page typing:

- add comparison-oriented unit tests

## 16. Immediate Next Task

The next step should not be another large refactor and should not be another
round of simply raising worker counts.

The immediate next execution step should be:

1. add structured page-level timing aggregation to benchmark results
2. run the `4 / 8 / 12 / 16` sweep and lock `normal-max`
3. implement the first conservative version of page-adaptive `dpi / token`

Only after those three steps are done should the next speed decision be made.

## 17. Decision Rules

If priorities conflict, decide using these rules:

1. a change that reduces first-run VL request cost is more valuable than a
   warm-cache-only win
2. a change that removes a second full-page call is more valuable than a
   preprocessing micro-optimization
3. page-local cost control is preferred over global quality reduction
4. "feels faster" without benchmark evidence is not a merge criterion
5. if speed and quality conflict, preserve quality first and then seek a more
   selective page-level strategy

## 18. Summary

The key conclusion for this phase is clear:

- refactoring has already created enough room for performance work
- the main bottleneck is single-page `vision convert`
- the highest-value next speed gains will come from adaptive per-page VL cost
  control, not from unlimited worker growth
- the correct order is:
  - improve benchmark structure
  - land adaptive page budgets
  - tighten retry and references guardrails
  - revisit concurrency only after request cost comes down

All future `LLM / vision_direct` optimization should follow this plan and use
first-run real-library benchmark results as the main decision source.

## 19. Cross-Document Concurrency Plan

This section defines how multi-PDF throughput should be improved without
destroying provider stability.

### 19.1 Why This Needs A Separate Plan

The current code already supports page-level parallelism inside one document,
but the library/background path still behaves like a single-file queue.

If multi-PDF parallelism is added naively, each converter instance may assume
it owns the full `KB_LLM_MAX_INFLIGHT` budget. That would multiply provider
pressure and create timeout cascades.

Therefore cross-document concurrency must be unlocked in this order:

1. make `LLM inflight` a process-level shared budget
2. measure stable total inflight under one-document and two-document load
3. only then allow limited multi-PDF parallel execution

### 19.2 Execution Phases

#### Phase X1

- replace per-worker / per-converter inflight gating with a process-level
  shared limiter
- keep visible behavior unchanged for single-document conversions
- add unit coverage proving two worker instances share the same limit

#### Phase X2

- extend benchmark coverage to compare:
  - one PDF at `N` inflight
  - two PDFs sharing the same total inflight budget
- report:
  - total wall time
  - per-document wall time
  - timeout/error count
  - fallback count

#### Phase X3

- change the background conversion runner from one active task to a small active
  set
- start with a hard cap of `2` active PDFs
- default to OFF behind a flag until stable

#### Phase X4

- only after the shared limiter and two-document tests are stable:
  - consider a default-on multi-PDF mode
  - consider larger active-PDF caps

### 19.3 Practical Upper-Bound Rules

For this phase, do not treat larger numbers as automatically better.

Working rules:

- total process-level inflight is the real budget
- `page_workers` must not exceed shared inflight by default
- test `8 / 12 / 16`, but only keep a higher ceiling if timeout/fallback
  behavior stays stable
- first multi-PDF rollout should target throughput, not maximum aggression

### 19.4 Immediate Implementation Step

Start now with Phase X1:

1. land the process-level shared inflight limiter
2. verify with unit tests
3. keep real-PDF validation to one document per run until the shared limiter is
   proven stable

## 20. 2026-08-17 Conservative Page-Budget Slice

The first class-aware budget slice is implemented behind
`KB_PDF_VISION_ADAPTIVE_PAGE_BUDGETS=0`. The default remains disabled until a
fixed first-run real-PDF A/B passes the complete conversion and downstream
source-grounding gates.

The local classifier uses only signals already available during conversion and
assigns one of:

1. `references`;
2. `text_dense_body`;
3. `figure_or_visual_heavy`;
4. `formula_sensitive`;
5. `unknown`.

Risk signals take precedence. First pages, textless/scan-like pages, pages whose
formula signal cannot be computed, references pages, formula-sensitive pages,
and visual pages retain the current render budget. Only an ordinary body page
with a successfully checked empty formula-candidate set may use the lower plain-
page DPI. Existing token caps remain restricted to normal-mode body pages with
no figure or formula payload. When the adaptive flag is enabled, a source-
verified text-only page may keep a layout-order hint and still receive the text
budget; with the flag disabled, the legacy page-hint restriction is unchanged.
An explicit `KB_PDF_VISION_DPI` continues to override adaptive rendering.

With `stage_timings=1`, every full-page VL conversion emits a structured
`[VISION_DIRECT][BUDGET]` record. The benchmark parser persists page class,
chosen/base DPI, token override, density, formula/image/visual counts, render
profile, per-run class distribution, adaptive-page count, reduced-DPI count,
and token-capped count.

Use paired profiles on the same first-run PDFs:

```powershell
python tools\benchmark_converter.py <pdf-or-fixed-sample-dir> `
  --profile "name=normal-control,speed_mode=normal,llm_page_workers=8,max_inflight=8,stage_timings=1,adaptive_page_budgets=0" `
  --profile "name=normal-adaptive,speed_mode=normal,llm_page_workers=8,max_inflight=8,stage_timings=1,adaptive_page_budgets=1" `
  --out-dir tmp\benchmarks\adaptive_page_budget_ab
```

This implementation is not itself a performance acceptance. Do not enable it
by default or claim a latency improvement until the fixed real-library sample
meets the Section 14 targets, the hard/manual conversion checks pass, source
validation remains green, and the unchanged Research QA gates pass on the exact
candidate code.

### 20.1 Real-PDF Decision Record

The final paired run used the fixed 18-page Nature perovskite paper and the
exact candidate code. Both profiles used eight page workers, shared inflight
eight, stage timings, a cold page cache, and the configured `qwen3-vl-plus`
vision provider. Results are stored under
`tmp/benchmarks/adaptive_page_budget_nature_ab_20260817`.

| Profile | Wall time | Output chars | Reduced-DPI pages | Token-capped pages | Empty/math retries | Fallbacks |
|---|---:|---:|---:|---:|---:|---:|
| control | 127.64 s | 52,189 | 7 | 3 | 0 / 0 | 0 |
| adaptive | 134.98 s | 54,219 | 5 | 5 | 0 / 0 | 0 |

The adaptive profile was 5.7% slower, so this slice is a performance **no-go**
and remains disabled by default. Both outputs retained all 18 page markers.
The adaptive output contained five display-math blocks versus none in the
control output, consistent with its deliberate decision to restore 220 DPI on
two formula-sensitive pages that the legacy heuristic rendered at 200 DPI.
That is a useful quality signal, but it does not satisfy the speed target and
is not enough to change the default from this single stochastic provider run.

Two earlier diagnostic pairs (a seven-page visual/formula-heavy paper and an
11-page formula-heavy paper) also had no eligible adaptive speed work or were
run before the final benchmark-fairness patch; they are not acceptance runs.
Their mixed timing direction reinforces that single provider timings must not
be presented as a latency win.

Keep the classifier, feature flag, structured metrics, and benchmark plumbing
for the next experiment. The next speed candidate should avoid a VL call on a
strictly source-verified text-only page and validate the local/layout-aware
result, rather than micro-adjusting DPI and maximum output tokens on a VL call
that still has to run.

## 21. 2026-08-17 Verified Text-Only Local Fast Path

The next slice is implemented behind
`KB_PDF_VISION_TEXT_LOCAL_FASTPATH=0`. It is independent of adaptive page
budgets, so a paired benchmark can change only whether a verified text page
bypasses the full-page vision call.

A page is considered only in normal mode and only when all of the following are
true:

1. it is not the first page or a references page;
2. the source-backed formula scan ran successfully and found no formula;
3. figure/image extraction found no visual payload;
4. the PDF text layer contains at least 900 normalized characters;
5. the source dictionary has no table-risk signal; and
6. the converter exposes the no-remote-call local page pipeline.

The local result is accepted only when it contains at least 84% of source word
occurrences, 68% of adjacent source-word pairs, 62% sequence similarity, and a
normalized length ratio between 0.72 and 1.35. Incomplete markers, replacement
characters, control characters, insufficient source/output tokens, or a
missing source-backed heading reject the result. Rejected output is discarded
and conversion continues through the unchanged VL path.

Bold/semibold source prefixes are promoted to Markdown headings before
validation. Repeated header/footer strings from the converter's existing
global noise scan are excluded, preventing journal mastheads from becoming
required document headings.

Each attempted page emits `[VISION_DIRECT][TEXT_LOCAL]` with the decision,
reason, source/output sizes, word and bigram coverage, order and length ratios,
heading promotion counts, and local elapsed time. Benchmark run metrics include
attempted, accepted, rejected, and avoided-VL-call counts; these are also
aggregated into the case/profile CSV summaries.

Use a paired profile with adaptive budgets held off:

```powershell
python tools\benchmark_converter.py <fixed-pdf> `
  --profile "name=normal-control,speed_mode=normal,llm_page_workers=8,max_inflight=8,stage_timings=1,adaptive_page_budgets=0,text_local_fastpath=0" `
  --profile "name=normal-text-local,speed_mode=normal,llm_page_workers=8,max_inflight=8,stage_timings=1,adaptive_page_budgets=0,text_local_fastpath=1" `
  --out-dir tmp\benchmarks\text_local_fastpath_ab
```

### 21.1 Real-PDF Results and Rollout Decision

The final candidate was exercised on two fixed real-library papers with cold
page caches, eight page workers, shared inflight eight, and the configured
`qwen3-vl-plus` provider.

| Paper | Pages | Control | Final candidate | Attempted / accepted / rejected | VL calls avoided | Decision |
|---|---:|---:|---:|---:|---:|---|
| Nature perovskite | 18 | 132.73 s | 71.72 s | 5 / 2 / 3 | 2 | 46.0% faster; multi-wave win |
| NatPhoton SPI review | 8 | 128.84 s | 130.43 s | 2 / 1 / 1 | 1 | within provider noise; one VL batch remains |

Nature's accepted pages had word coverage 0.861/0.964, bigram coverage
0.851/0.934, and order ratios 0.911/0.940. Its three short pages were rejected
without running the local pipeline. The final output retained all 18 page
markers, had no empty/math retry or fallback, and its generated quality report
contained no warnings.

For NatPhoton, page 3 was accepted with 0.974 word coverage, 0.961 bigram
coverage, and 0.957 order ratio after a repeated journal masthead was correctly
removed from heading requirements. Page 7 was rejected at 0.751 word coverage
and returned to VL. All eight page markers remained present. Compared with the
paired control's page 3, the final local page had slightly higher source-word
and bigram coverage; total wall time did not improve because seven VL pages
still fit in one concurrent batch.

This slice passes functional acceptance: an accepted page avoids its VL call,
unsafe/weak pages fall back, and a document with enough pages to remove a VL
batch materially improves. Keep the flag opt-in for controlled rollout rather
than changing the production default from two papers. The next rollout gate is
a repeated multi-paper corpus run that reports both median wall time and
accepted-page source metrics; no threshold should be relaxed to increase the
acceptance count.

## 22. 2026-08-17 Repeated 10-Paper Text-Local Rollout Gate

The controlled rollout gate used 10 fixed real-library papers (103 source PDF
pages), three cold-cache repeats per profile, and paired control/candidate runs.
Profile order was reversed on the second repeat to reduce order bias. The 60
valid conversions are split across these raw result roots because the first
long-running parent process was interrupted after the first seven papers:

- `tmp/benchmarks/text_local_fastpath_rollout_v1_20260817` (first seven papers,
  42 complete runs selected from per-run `benchmark_run.json` files);
- `tmp/benchmarks/text_local_fastpath_rollout_resume_v1_20260817` (last three
  papers, 18 complete runs and normal suite CSV/JSON output).

One extra complete control and one partial candidate from the interrupted
eighth-paper pair were excluded; the last three papers were rerun as a complete
paired block. No completed result was overwritten. All primary comparisons use
the median of the three within-repeat percentage changes, not the difference
between separately computed profile medians; this matters when provider latency
changes sharply between adjacent runs.

| Paper | Pages | Median attempts / accepts | Control median | Candidate median | Median paired improvement |
|---|---:|---:|---:|---:|---:|
| SCINeRF | 11 | 1 / 1 | 58.91 s | 54.74 s | +4.20% |
| 3D single-pixel video | 8 | 0 / 0 | 58.81 s | 58.72 s | +0.16% |
| NatCommun compressive holography | 12 | 2 / 2 | 85.71 s | 144.87 s | +4.90% |
| NatPhoton SPI review | 8 | 2 / 1 | 128.82 s | 130.43 s | -0.93% |
| Nature perovskite | 18 | 5 / 2 | 132.51 s | 59.10 s | +54.97% |
| Optica metamaterial SPI | 6 | 2 / 1 | 126.78 s | 61.97 s | +51.15% |
| Optics & Laser Technology image-loop | 7 | 0 / 0 | 55.54 s | 58.89 s | -12.89% |
| Psychological Review visual perception | 11 | 0 / 0 | 48.98 s | 46.28 s | +3.04% |
| SciAdv adaptive foveated SPI | 10 | 0 / 0 | 161.22 s | 161.33 s | +0.15% |
| Visual Computing SPI review | 12 | 0 / 0 | 95.81 s | 101.27 s | -6.45% |

Provider latency had large step changes in several adjacent runs. For example,
the three NatCommun paired improvements were +4.90%, -70.61%, and +5.16%; the
paired median remains meaningful while separately computed profile medians are
misleading. Optica likewise had two approximately 50% wins and one tied long-
tail run. The schedule therefore reports both values but makes the paired
median authoritative.

Across the 30 candidate runs, 36 pages entered the strict local check, 21 were
accepted, 15 returned to VL, and 21 VL calls were avoided. The 21 accepted
instances represented seven unique PDF pages. Their minimum validation scores
were 0.8608 word-occurrence recall, 0.8512 bigram recall, and 0.9113 order
similarity; length ratios stayed between 0.8828 and 1.0658. No acceptance
threshold was relaxed.

### 22.1 Acceptance Decision

| Gate | Target | Result | Decision |
|---|---:|---:|---|
| Papers with median accepted pages >= 2 | median improvement >= 20% | +29.94% (2 papers) | pass |
| Full corpus, all paired repeats | median improvement >= 10% | +0.16% | **fail** |
| Paper-level median | no slowdown worse than 10% | worst -12.89% | **fail** |
| Accepted-page validation | unchanged strict thresholds | all accepted instances passed | pass |
| Conversion/source quality | no candidate-only regression | candidate-only issues observed | **fail** |

All 60 conversions completed and every output retained its exact PDF page-marker
count. Across both profiles, missing-image, table-break, mojibake, and missing-
source-page counts were zero. However, NatCommun candidate repeats 2 and 3 had
display-math/analyzer issues that their paired controls did not. The source
corruption diagnosis was on page 11, while the deterministic local accepts were
pages 2 and 4, so the accepted local content was not the directly damaged page;
the conservative no-regression gate still fails because changing the request
set can indirectly change stochastic VL outputs.

The audit also found that SCINeRF page 11 was a heading-less continuation of
numbered references. The existing reference detector classified it as ordinary
text, and the pre-tightening candidate accepted it with 0.9938 word/bigram
coverage. The final code adds a conservative continuation guard: at least three
strictly increasing line-start `[n]` reference entries reject the body fast
path. A post-guard cold run is stored under
`tmp/benchmarks/text_local_fastpath_reference_guard_20260817`; it rejected page
11 with `references_continuation`, kept all 11 page markers, reported no quality
issues, and avoided no VL call.

The rollout decision is therefore **no-go for default enablement**.
`KB_PDF_VISION_TEXT_LOCAL_FASTPATH` remains `0` by default and remains available
only as an explicit opt-in experiment. The post-gate reference guard can only
reduce optimized coverage, so it cannot invalidate the conservative no-go
decision. Do not increase acceptance by weakening content or risk thresholds.

### 22.2 Exact-Code Release Gates

After the post-gate continuation guard, the complete unchanged release suite
passed on the exact working tree:

- Ruff: pass;
- backend unit: 4,424 passed, 41 skipped;
- backend sanity: 262 passed, 2 skipped;
- Agent answer runtime: 5 passed;
- frontend ESLint and production build: pass;
- Research QA fixture/full-library dry runs: 56 and 29 cases;
- source grounding: 41/41;
- grounded replay: 6/6;
- Agent golden validation: 10 cases, zero schema errors;
- reviewed Agent replay: 5/5;
- converter quality: 13/13 at
  `test_results/converter_quality_eval/20260817_180105`;
- version A/B fixture dry run: 34 QA cases and three project journeys loaded;
- paired comparison audit: 5/5 with zero false comparisons;
- comparison-candidate audit: 5/5, 18 discoveries, zero contract/evidence/
  prefill failures;
- project research status: 5/5 at
  `test_results/project_research_status/20260817_180107`;
- continuous project journey: 20/20 at
  `test_results/project_research_journey/20260817_180127`.

These green release gates verify that the opt-in implementation and the new
benchmark pairing mode do not regress the production-default path. They do not
override the failed rollout thresholds and are not evidence for enabling the
feature by default.

## 23. 2026-08-17 Two-Document Product-Path Throughput Gate

The next optimization targeted queue throughput without reducing conversion
quality or raising the provider request budget. An implementation audit found
that Phase X1 was already present on the baseline: `LLMWorker` uses a class-level
shared inflight gate, every converter model request passes through `_llm_create`,
and its `finally` path releases the slot. The Web/background product path also
passes the active-document count into `run_pdf_to_md`, which divides the fixed
global inflight budget between child converter processes.

This pass therefore hardened and measured the existing design instead of adding
a second limiter:

- an exception-release regression proves that a failed provider call cannot
  strand a shared slot and block another worker;
- background concurrency is bounded to 1--4 active documents;
- `tools/benchmark_converter_throughput.py` exercises the real
  `kb.pdf_tools.run_pdf_to_md` subprocess path with exactly two PDFs;
- serial and parallel modes use the same global inflight budget, workers, model
  settings, and experimental feature flags;
- odd repeats run serial first and even repeats run parallel first;
- every experiment checkpoints document, page, paired, and quality metrics;
- harness-owned case directories use short stable IDs so long paper titles do
  not exceed legacy Windows path limits.

The fixed acceptance run is stored at
`tmp/tp_ab_0817a/throughput_results.json`. It used the 18-page Nature perovskite
laser paper and the 8-page Nature Photonics single-pixel imaging paper, three
paired repeats, `KB_LLM_MAX_INFLIGHT=8`, four page workers, three LLM workers,
and both experimental page-budget and text-local fast paths disabled.

| Repeat | Serial elapsed | Parallel elapsed | Throughput improvement |
|---:|---:|---:|---:|
| 1 | 256.36 s | 138.96 s | 45.79% |
| 2 | 245.90 s | 132.07 s | 46.29% |
| 3 | 242.25 s | 131.11 s | 45.88% |

The paired median throughput improvement was **45.88%**, above the 25% gate.
The Nature paper's serial/parallel p95 was 123.17/127.20 seconds (+3.27%); the
Nature Photonics paper's was 133.35/138.25 seconds (+3.67%). The worst
per-document p95 slowdown was therefore **3.67%**, below the 15% gate. All 12
document conversions completed, parallel mode added zero timeouts, empty/math
retries, or fallbacks, all page-marker counts were exact, and all six paired
quality comparisons had zero critical regression.

### 23.1 Acceptance Decision

The two-document product-path gate is a **go**. Background conversion now
defaults to `KB_BG_CONVERT_MAX_ACTIVE=2`; operators can set it to `1` when the
lowest single-document latency is more important than queue throughput. The
global provider budget remains fixed and is divided across active documents.
Adaptive page budgets and the text-local fast path remain separately disabled
by default; this decision does not override their earlier no-go results.

### 23.2 Exact-Code Release Gates

After enabling the accepted default on the exact working tree, the complete
unchanged release suite passed:

- Ruff: pass;
- backend unit: 4,428 passed, 41 skipped;
- backend sanity: 262 passed, 2 skipped;
- Agent answer runtime: 5 passed;
- frontend ESLint and production build: pass;
- Research QA fixture/full-library dry runs: 56 and 29 cases;
- source grounding: 41/41;
- grounded replay: 6/6;
- Agent golden validation: 10 cases, zero schema errors;
- reviewed Agent replay: 5/5;
- converter quality: 13/13 at
  `test_results/converter_quality_eval/20260817_192721`;
- version A/B fixture dry run: 34 QA cases and three project journeys loaded;
- paired comparison audit: 5/5 with zero false comparisons;
- comparison-candidate audit: 5/5, 18 discoveries, zero contract/evidence/
  prefill failures;
- project research status: 5/5 at
  `test_results/project_research_status/20260817_192723`;
- continuous project journey: 20/20 at
  `test_results/project_research_journey/20260817_192744`.

## 24. 2026-08-17 Per-Document Cancellation for Concurrent Conversion

After the default two-document rollout, the previous global-only cancellation
control was too broad: stopping one problematic conversion also requested that
every active conversion stop. Cancellation is now task-scoped while the legacy
global endpoint behavior remains available for explicit queue-wide shutdown.

The queue state owns the cancellation decision under its existing lock:

- cancelling a queued task removes only that task and adjusts the queue total;
- cancelling an active task marks only its task record as `cancelling`;
- each worker checks the cancellation bit for its own task ID;
- late page-progress, running-page, and conversion-stage callbacks cannot
  overwrite a task that has entered `cancelling`;
- a missing or already-finished task ID is an idempotent no-op;
- the existing cancel endpoint accepts an optional `task_id`; omitting it keeps
  the backward-compatible cancel-all behavior;
- library file payloads expose task IDs for queued and active conversions, so
  each busy row can offer **Cancel conversion**, while the process-level control
  is now labeled **Stop all**.

The isolation contract is covered at both state and product levels. Deterministic
two-active-task tests cancel the first task, reject its late progress callbacks,
complete it as cancelled, and complete the sibling normally. A separate queued
test proves that removing one queued task leaves the other queued task intact.
The browser test starts two concurrent papers, cancels the first row with its
task ID, observes that row enter `cancelling`, and verifies that the sibling
remains converting and independently cancellable.

### 24.1 Acceptance Decision

Per-document cancellation is accepted for the default concurrent conversion
path. It changes queue control only: the surviving task continues through the
unchanged conversion, quality, and indexing path, and the fixed global provider
budget remains unchanged. Queue-wide cancellation remains available as an
explicit **Stop all** action.

### 24.2 Exact-Code Release Gates

After the task-scoped cancellation implementation, the complete unchanged
release suite passed on the exact working tree:

- Ruff and `git diff --check`: pass;
- backend unit: 4,430 passed, 41 skipped;
- backend sanity: 263 passed, 2 skipped;
- Agent answer runtime: 5 passed;
- frontend ESLint and production build: pass;
- targeted concurrent-cancellation Playwright test: 1 passed;
- Research QA fixture/full-library dry runs: 56 and 29 cases;
- source grounding: 41/41;
- grounded replay: 6/6;
- Agent golden validation: 10 cases, zero schema errors;
- reviewed Agent replay: 5/5;
- converter quality: 13/13 at
  `test_results/converter_quality_eval/20260817_194917`;
- version A/B fixture dry run: 34 QA cases and three project journeys loaded;
- paired comparison audit: 5/5 with zero false comparisons;
- comparison-candidate audit: 5/5, 18 discoveries, zero contract/evidence/
  prefill failures at
  `test_results/evidence_comparison_candidates/20260817_194918`;
- project research status: 5/5 at
  `test_results/project_research_status/20260817_194917`;
- continuous project journey: 20/20 at
  `test_results/project_research_journey/20260817_194939`.

## 25. 2026-08-17 Per-Document Terminal Results and Precise Retry

Concurrent conversion made the former single `last` message insufficient: the
last worker to finish overwrote the result of its sibling, and an idle library
row could not distinguish success, cancellation, conversion failure, a quality
block, or an indexing failure. Conversion control now has a bounded,
task-addressable terminal-result contract.

The queue records up to 50 recent task results under the same lock that owns
queued and active tasks. Each result includes task/document identity, operation,
outcome, safe detail, retry action, original speed mode, page counts, start/end
timestamps, and duration. The public outcomes are:

- `success`;
- `cancelled`;
- `conversion_failed`;
- `quality_blocked`;
- `index_failed`.

Active tasks write their result exactly once when removed from `active_tasks`.
Cancelling a queued task also writes a cancellation result even though no worker
started. Duplicate external result IDs replace their earlier record, and the
history is capped to avoid unbounded process memory. A missing Markdown output
is now classified as conversion failure, while a missing or failed ingest step
is classified as index failure instead of being reported as a successful run.

The library files and conversion-status APIs expose compact recent results, and
each idle file row displays its own newest result with finish time, duration,
and detail tooltip. Cancelled, conversion-failed, and quality-blocked rows offer
**Retry conversion** using the original speed mode. Index-failed rows offer
**Retry index**; the new management-protected single-file endpoint runs
incremental ingest plus structured-index rebuilding for only the existing
Markdown target and does not reconvert the PDF. The index retry writes its own
success or failure terminal result, closing the UI loop.

### 25.1 Acceptance Decision and Boundary

The per-document terminal-result and precise-retry contract is accepted for the
current local product path. State, API, and browser tests cover out-of-order
sibling completion, every terminal outcome, queued cancellation, bounded and
deduplicated history, result-to-file mapping, index-retry success/failure, and a
browser proof that retrying an index does not invoke PDF conversion.

The recent-result journal is intentionally process-local in this slice. It
solves concurrent result ambiguity during a running app session, but it is not
a durable job ledger and does not restore queued/active work after a backend
restart. Durable task recovery remains a separate release-engineering gate for
a downloadable end-user application.

### 25.2 Exact-Code Release Gates

After the terminal-result and precise-retry implementation, the complete
unchanged release suite passed on the exact working tree:

- Ruff and `git diff --check`: pass;
- backend unit: 4,432 passed, 41 skipped;
- backend sanity: 266 passed, 2 skipped;
- Agent answer runtime: 5 passed;
- frontend ESLint and production build: pass;
- targeted concurrent cancellation and terminal-result/index-retry Playwright
  tests: 2 passed;
- Research QA fixture/full-library dry runs: 56 and 29 cases;
- source grounding: 41/41;
- grounded replay: 6/6;
- Agent golden validation: 10 cases, zero schema errors;
- reviewed Agent replay: 5/5;
- converter quality: 13/13 at
  `test_results/converter_quality_eval/20260817_202018`;
- version A/B fixture dry run: 34 QA cases and three project journeys loaded;
- paired comparison audit: 5/5 with zero false comparisons;
- comparison-candidate audit: 5/5, 18 discoveries, zero contract/evidence/
  prefill failures at
  `test_results/evidence_comparison_candidates/20260817_202019`;
- project research status: 5/5 at
  `test_results/project_research_status/20260817_202018`;
- continuous project journey: 20/20 at
  `test_results/project_research_journey/20260817_202044`.
