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
