# Reference Locate Execution Plan 2026-04-21

## Scope

This plan continues the post-handoff cleanup after the multi-paper SCI fixes.

The next work is deliberately narrower than the previous window:

1. remove the current `ADMM` false-positive leak for single-paper direct lookup prompts
2. unify the evidence surface used by the top answer and the refs cards
3. keep the code path readable and shared instead of adding more one-off fallback branches

## Current State

What is already materially better:

- multi-paper authoritative `doc_list` is the right source of truth for paper identity
- SCI multi-paper pending and full refs can stay on the same paper set
- SCI refs card copy has recovered from the shallow adapter regression and now reads much more naturally

What still needs work:

- the benchmark negative case `Which paper in my library most directly discusses ADMM?` still leaks one weakly related card
- top answer and refs cards can still use different `heading_path` or `primary_evidence` for the same paper
- there are still a few places where answer-side and refs-side directness logic drift apart

## Root Cause Reading

### A. Directness is still weaker than term match for some single-paper prompts

For prompts like:

- `Which paper in my library most directly discusses X?`
- `Which paper ... defines X?`
- `Which paper ... compares A and B?`

the current refs pipeline still treats a surviving exact-term mention as good enough in some cases.

That is not strict enough.

For this class of prompts, we need:

- exact focus matching
- and a minimum directness threshold
- and rejection of incidental/background mentions when the paper is not actually centered on the requested concept

### B. `doc_list` authority is still only partial

`doc_list` is already authoritative for paper identity, but not yet fully authoritative for the final displayed evidence surface.

Current effect:

- answer-side output may keep one `heading_path`
- refs card rendering may later select a different `heading_path` or summary candidate

That creates a softer version of the old inconsistency bug:

- same paper
- different locate section
- different evidence framing

### C. The next step should reduce branching, not add more rescue branches

The main risk now is code drift:

- one fix in `generation_answer_finalize_runtime`
- another in `reference_ui`
- another in router pending handling

If each layer gets its own local exception, the system will become brittle again.

So the next implementation must prefer:

- shared helpers
- explicit scoring contracts
- narrower fallback triggers
- fewer silent overrides

## Implementation Principles

### Principle 1: one shared notion of “direct match”

Answer-side and refs-side code should rely on the same concept of directness whenever possible.

That means:

- avoid duplicating prompt-shape heuristics in multiple modules
- centralize directness signals into helper functions
- make the negative gate deterministic before any optional LLM stage

### Principle 2: evidence identity and display identity should not diverge

Once we have an authoritative `primary_evidence` for a doc-list item, downstream rendering should not casually rebind the card to a different section unless there is a strong quality reason and the evidence surface still matches the same authoritative target.

### Principle 3: fewer fallbacks, stronger predicates

Do not add broad “if this fails, try three more rescue paths” logic.

Instead:

- tighten candidate acceptance
- reject title-like or incidental summary candidates earlier
- only invoke rescue paths when the current candidate is clearly unusable

### Principle 4: tests should lock behavior, not implementation details

We should keep a small high-signal regression set:

- SCI multi-paper Chinese query
- dynamic supersampling define
- Hadamard vs Fourier compare
- ADMM negative
- one additional library-wide multi-paper query such as single-photon imaging

## Planned Changes

### Step 1. Add a deterministic directness gate for strict single-paper prompts

Target files:

- `api/reference_ui.py`
- tests in `tests/unit/test_reference_ui_score_calibration.py`

Plan:

- identify prompts that ask for the most direct single-paper match
- score hits not just by term match, but also by whether the matched section actually defines, explains, compares, or centers on the requested concept
- suppress hits that are only background mentions for these prompt shapes

Expected outcome:

- `ADMM` negative query returns zero refs cards
- strong positive definition/compare queries still pass

### Step 2. Let answer and refs consume the same authoritative evidence surface

Target files:

- `kb/generation_answer_finalize_runtime.py`
- `kb/task_runtime.py`
- `api/reference_ui.py`

Plan:

- make `doc_list_contract` carry the final authoritative evidence surface more consistently
- preserve `primary_evidence`, `heading_path`, and summary seed that answer-side formatting should use
- keep refs card rendering aligned with the same authoritative evidence unless a strictly better equivalent surface is chosen

Expected outcome:

- same paper in top answer and refs card uses the same locate section most of the time
- benchmark evidence-identity checks become more meaningful

### Step 3. Remove contradictory fallback behavior where possible

Target files:

- mostly `api/reference_ui.py`

Plan:

- review prompt-aligned summary rescue logic
- trim paths that overwrite an already-acceptable authoritative summary with a weaker block rescue
- avoid adding more “just in case” fallbacks unless a concrete regression needs them

Expected outcome:

- clearer card-building path
- easier debugging
- less “random walk” behavior

## Validation Plan

### Unit

- extend existing ADMM negative tests so incidental ADMM mentions are rejected
- keep existing SCI doc-list tests green
- add evidence-alignment assertions where answer-side and refs-side authoritative evidence should agree

### Smoke

Run a small, bounded set only:

- `NORMAL_DYNAMIC_SUPERSAMPLING_DEFINE`
- `NORMAL_HADAMARD_FOURIER_COMPARE`
- `NORMAL_ADMM_NEGATIVE`
- unicode-safe SCI multi-paper in-process check
- one additional multi-paper semantic case such as `single-photon imaging`

## Acceptance Criteria For This Window

- `ADMM` negative no longer leaks a refs card
- SCI query still keeps the authoritative 3-paper set
- SCI pending/full behavior does not regress
- answer and refs are closer to using the same `heading_path` and `primary_evidence`
- no new broad, conflicting, or low-signal fallback branches are introduced

## Progress Notes 2026-04-21

What is now concretely improved:

- multi-paper `doc_list` now normalizes weak seed surfaces from raw snippet content before answer formatting
- weak `answer_hit_top` card primaries no longer overwrite a better seed surface just because they arrived later
- SCI multi-paper answer-side `doc_list` now lands on more natural sections such as `Abstract` instead of mismatched headings like `2. Related Work`
- SCI top answer and final refs now consistently stay on the same 3-paper set: `SCINeRF`, `SCIGS`, and the `OE-2007` predecessor
- answer-side multi-paper `doc_list` can now be rebuilt from the already-rendered refs payload, so `summary_line` and `heading_path` are no longer locked to the shallow seed surface
- pending seed filtering now reuses the same multi-paper `doc_list` filter used by the final answer path instead of always padding broad prompts to 3 docs
- pending refs focus filtering now also applies compare-specific scoring on raw hits, so single-paper compare prompts stop leaking broad “mention only” papers before full refs arrive

Important implementation choice:

- a broader attempt to force deep seed rebuild for all multi-paper prompts was tested and then rejected
- reason: it regressed the stable SCI path and introduced worse paper selection/section quality on other prompts
- decision: keep the narrow `doc_list` surface-normalization fix, and defer any earlier-seed rebuild change until there is a much stricter trigger

Latest in-process checks after the above changes:

- `有哪几篇文章提到了SCI（单次曝光压缩成像）`
  - pending refs and full refs now keep the same 3-paper identity set throughout: `SCINeRF`, `OE-2007`, `SCIGS`
  - top answer uses the same 3 papers
- `Which papers in my library discuss single-photon imaging?`
  - top answer `doc_list` now matches refs on the better `Frontiers-2024 / 5.1 Optical imaging` section instead of the earlier weak `5.3 Quantum communication` seed
- `Which papers in my library discuss Fourier single-pixel imaging?`
  - pending refs no longer pad with extra broad-review papers and now stay aligned with the single final `OE-2017` paper from the start
- `Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?`
  - pending refs no longer leak the broad `NatPhoton-2019` review card before full refs land
  - pending and full now stay on the same `OE-2017` compare paper identity throughout

Residual mismatch still worth watching:

- some cases can still have small section-level drift between top answer and final refs for the same paper even when paper identity is already aligned
- current observed example: the SCI `OE-2007` predecessor can still appear as `Abstract` in the answer-side `doc_list` while the final refs card chooses a more specific internal section
- another smaller remaining example is `single-photon imaging`, where pending refs can still show an earlier weaker section for the same `Frontiers-2024` paper before full refs upgrades it to `5.1 Optical imaging`
- this is much smaller than the original wrong-paper/pending drift issue, but if we continue polishing, the next narrow step should be to make answer-side `doc_list` consume the exact final card surface whenever the refs renderer upgrades the same paper

Latest narrow improvement after the above note:

- the raw/full non-authoritative `reference_ui` path could still keep a weak focus-prefixed fallback such as `single-photon imaging: ABSTRACT ...`, which prevented source-block rescue from upgrading the same `Frontiers-2024` paper to `5.1 Optical imaging`
- this is now tightened in `api/reference_ui.py` by:
  - treating `focus term: ABSTRACT/Introduction/...` shells as low-quality ref summaries
  - and letting focus-prefixed fallback summaries trigger source-block rescue even if they are not otherwise empty
- result: the direct real `Frontiers-2024` hit for `Which papers in my library discuss single-photon imaging?` now re-renders as:
  - heading: `5 Application / 5.1 Optical imaging`
  - summary: a concrete optical-imaging sentence instead of the earlier `single-photon imaging: ABSTRACT ...` shell
- another direct ready-path cleanup landed for compare prompts:
  - the non-authoritative ready render could still let synthesized `summary/why` inflate compare directness for papers that merely mentioned `Hadamard` and `Fourier`
  - compare filtering now:
    - requires a higher minimum directness score
    - and caps ready compare score by the corresponding raw evidence score when raw evidence exists
  - result: the direct real prompt `Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?` now keeps only `OE-2017`

Validation run for this increment:

- targeted `reference_ui` regression tests for:
  - SCI authoritative doc-list behavior
  - single-photon pending/full block rescue
  - compare pending filtering
  - compare ready filtering
  - ADMM negative filtering
- targeted `generation_answer_finalize_runtime` regressions for:
  - SCI filtering
  - Fourier filtering
  - dynamic supersampling focus filtering
  - single-photon vs `NatPhoton` false-positive rejection
- targeted `task_runtime` regressions for:
  - SCI pending seed stabilization
  - Fourier pending no-padding
  - rendered-payload to `doc_list` surface sync

Important environment note from bounded live replay:

- live `/api/generate` answer replay in this workspace currently fails before the final answer step with:
  - provider: `Qwen`
  - model: `qwen3-vl-plus`
  - error: `Connection error`
- retrieval and refs rendering still run locally, so the latest verification used:
  - direct real retrieval/render replay for refs
  - plus deterministic/unit regression coverage for answer/doc-list behavior

Bounded smoke observations from real retrieval/render replay:

- `Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?`
  - final refs now keeps only `OE-2017`
- `Which papers in my library discuss Fourier single-pixel imaging?`
  - answer-side/filtering still keeps only `OE-2017`
  - remaining polish gap: the copied summary sentence is still too fragmentary (`Fourier single-pixel imaging: of Fourier coefficients.`)
- `Which paper in my library most directly discusses ADMM? Please point me to the source section.`
  - final refs stays empty, which is the intended negative behavior for this library
- `Which paper in my library most directly defines dynamic supersampling?`
  - final refs stays on `SciAdv-2017`
- `Which papers in my library discuss single-photon imaging?`
  - after rendered-payload sync, answer-side `doc_list` and refs both land on:
    - `NatCommun-2023`
    - `Frontiers-2024 / 5.1 Optical imaging`

Latest narrow cleanup after the above bounded replay:

- the next maintainability risk was not another wrong-paper case, but the same prompt intent being inferred separately in:
  - `reference_ui`
  - pending refs router copy
  - multi-paper finalize filtering
- that shape invited future regressions because `compare / define / discuss / single-vs-multi` prompt families could slowly drift apart even when the current examples still passed
- this is now tightened by introducing shared generic prompt-family helpers in `kb/reference_query_family.py` for:
  - compare intent
  - definition intent
  - general reference-focus intent
  - reference-focus action selection
- the follow-on changes deliberately avoid adding any library-specific paper names or topic literals to the new helpers; they only encode generic query-shape language such as:
  - compare
  - define
  - discuss / mention
  - where / point me / locate
- current consumers now reuse the same shared intent layer in:
  - `api/reference_ui.py`
  - `api/routers/references.py`
  - `kb/generation_answer_finalize_runtime.py`

Validation run for this cleanup:

- targeted `reference_query_family` unit coverage for:
  - generic compare/define/discuss prompts
  - non-locate prompts staying out of the strict-focus path
  - Chinese definition phrasing
- plus the existing high-signal regressions for:
  - pending compare filtering
  - ADMM negative filtering
  - SCI authoritative doc-list behavior
  - rendered-payload to `doc_list` surface sync
