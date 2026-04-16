# Paper Guide Execution Plan (2026-04-10)

## Current State

Paper Guide is no longer in the "just finish the refactor skeleton" phase.
It has moved into a quality-driven iteration phase with four meaningful guardrail layers:

- package architecture: `router / skills / grounder`
- backend regression: benchmark + replay pool
- frontend regression: reader-level Playwright coverage
- live validation: smoke + baseline runs against the running service

Recent progress confirms that the refactor produced real gains:

- live benchmark suites previously reached all-green baseline and smoke runs
- section-targeted prompts now locate more reliably
- beginner overview / role explanation prompts improved materially
- replay coverage now includes more real captured cases instead of only synthetic fixtures
- converter frontmatter recovery now has a real-PDF end-to-end regression suite, not only postprocess-only checks

### Converter Note

The converter now has two distinct guardrail layers for first-page structure:

- implicit section recovery from existing markdown heads
- real-PDF frontmatter recovery through end-to-end conversion

Current status:

- `normal` / vision-LLM frontmatter regression is now green on the current sample set
- `no_llm` still lags on the same real PDFs, so first-page recovery quality is currently strongest on the vision path

## Architectural Assessment

### What is now healthy

- `kb/paper_guide/router.py` owns intent resolution and package dispatch
- `kb/paper_guide/skills.py` owns exact-support orchestration and now also starts to own broad deterministic skills
- `kb/paper_guide/grounder.py` owns support-marker grounding, locate targets, and reader-open assembly
- quality guardrails exist at benchmark, replay, and reader-E2E levels

### What is still heavy

- `kb/task_runtime.py` still contains too much Paper Guide orchestration glue
- `kb/paper_guide_answer_post_runtime.py` still owns many resolver implementations
- broad question handling has improved, but only part of it is skill-first today

### Most important next move

Continue shifting "broad but structured" Paper Guide behavior into package skills, especially:

1. beginner overview / component-role explanation
2. section-targeted discussion / limitations / future-work
3. section-targeted overview / abstract / box-only guidance

This gives better long-term leverage than more ad-hoc heuristics inside `task_runtime`.

## Working Loop

Every iteration should follow this order:

1. pick one prompt family or failure family
2. add or tighten benchmark / replay / E2E coverage first
3. implement the smallest behavioral change that fixes the gap
4. run unit tests plus at least one live benchmark or E2E slice
5. write back the result into docs or regression assets

This keeps the project grounded in measurable quality instead of architecture-only progress.

## Next Phase Plan

### Phase A. Broad Skills First

Goal:
Move broad deterministic Paper Guide behavior out of inline runtime conditionals and into reusable package skills.

Tasks:

1. introduce package skills for:
   - component-role overview
   - section-targeted discussion / strength-limits
   - abstract direct answer
   - box-only targeted excerpt
2. route direct-answer override through package skill dispatch
3. keep behavior stable while reducing duplication in `paper_guide_direct_answer_runtime.py`
4. add unit coverage for the new skill layer and router dispatch

Exit criteria:

- the new broad skills are covered by package-level tests
- direct-answer behavior remains stable
- broad deterministic logic is no longer duplicated inline

### Phase B. Expand Real Captured Replay Coverage

Goal:
Make replay pool `v1` closer to real user reading behavior.

Priority families:

1. `figure_walkthrough`
2. `citation_lookup`
3. `method`
4. `abstract`
5. `box_only`

Tasks:

1. promote a small number of strong captured cases into curated sources
2. keep "best per signature" behavior so the pool does not fill with duplicates
3. maintain family coverage assertions in unit tests

Exit criteria:

- replay pool includes stable captured coverage across the families above
- curator output backlog shrinks for the highest-value families

### Phase C. Thin the Runtime Glue

Goal:
Reduce Paper Guide logic living in `task_runtime.py` and legacy flat modules.

Tasks:

1. continue moving deterministic orchestration into package modules
2. keep `task_runtime.py` focused on top-level request flow
3. treat `paper_guide_answer_post_runtime.py` as an implementation source, not the long-term public API

Exit criteria:

- new Paper Guide behavior lands in package modules first
- `task_runtime.py` stops gaining new Paper Guide family-specific branches

### Phase D. Keep Live Quality Honest

Goal:
Avoid drifting into architecture progress without real user-visible gains.

Tasks:

1. keep smoke and baseline suites running after meaningful behavior changes
2. keep reader regression aligned with backend locate behavior
3. use replay and live benchmark summaries as the main truth source

Exit criteria:

- runtime changes still preserve live locate quality
- new long-tail prompt families enter both backend and frontend guardrails

## This Iteration

This round should do the following:

1. formalize broad deterministic skills in `kb/paper_guide/skills.py`
2. add package router dispatch for those broad skills
3. route `paper_guide_direct_answer_runtime.py` through that dispatch
4. add unit tests for broad-skill behavior
5. recheck live benchmark slices that cover `abstract` and `box_only`
6. continue the next replay-pool promotion step only after the refactor is verified

## Not Priority Right Now

- another large contract rewrite
- agent-style orchestration
- aggressive cleanup of every legacy import
- speculative retrieval tuning without a failing case to anchor it

The current highest-value work is turning already-working behavior into stable, testable, skill-first behavior.
