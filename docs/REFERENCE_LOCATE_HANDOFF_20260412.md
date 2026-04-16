# Reference Locate Handoff (2026-04-12)

## 1. Current Goal

The current priority is **reference locate quality**, not a broad architecture rewrite.

Success means:

1. do not miss obviously relevant papers
2. do not show extra misleading papers
3. only show cards whose locate target is genuinely relevant
4. keep the card summary / guide line precise
5. keep the "why relevant" line grounded and useful
6. keep Paper Guide external-paper refs clear and conservative

## 2. What Was Improved In This Round

### A. Wrong library cards are much less likely to leak through

Main work happened in:

- `api/reference_ui.py`

Key changes:

- strict prompt-focus filtering no longer keeps a single generic hit just because only one row remains
- negative / strict prompts are now much more conservative
- misleading cards such as "ADMM" -> unrelated SCINeRF card are now suppressed instead of shown

Most important evidence:

- live benchmark case `NORMAL_ADMM_NEGATIVE` is now `PASS`
- result file:
  - `tmp/reference_locate_benchmark/20260412_181448/summary.json`

What this means in product terms:

- the system is closer to "show nothing rather than show a wrong paper"

### B. Card content is no longer treated as a generic "summary" in all cases

Main work happened in:

- `api/reference_ui.py`
- `web/src/components/refs/RefsPanel.tsx`

Current behavior:

- if a paper has a reliable abstract, the card prefers an **LLM-refined abstract summary**
- if there is no abstract but there is a reliable body hit, the card is treated as **Guide**
- if only metadata is available, the card degrades to **Meta**
- UI exposes the basis so the card is more honest

This prevents fake summaries for papers that do not actually have a usable abstract.

### C. "Why relevant" is now closer to grounded explanation instead of template stitching

Main work happened in:

- `api/reference_ui.py`

Current behavior:

- top hits prefer a grounded LLM relevance explanation
- the LLM input is constrained by:
  - user prompt
  - paper title
  - heading / locate context
  - hit snippets
  - current summary / guide line
- if the LLM output is weak or unavailable, the code falls back to deterministic prompt-aligned explanation

Goal of this change:

- the "relevant" section should answer "why this paper is related to my question", not just repeat prompt words

### D. Pending refs path is lighter

Main work happened in:

- `api/reference_ui.py`
- `api/routers/references.py`
- `kb/retrieval_engine.py`
- `kb/chat_store.py`

Key changes:

- pending refs no longer repeatedly run expensive LLM polish
- pending refs skip heavy exact locate work
- pending refs skip citation prefetch
- `/api/references/conversation/{conv_id}` now has a short TTL route cache
- refs read path now uses a short read timeout and can fall back to cached payload
- docwise refs-pack retry timeout now respects settings instead of using hidden hardcoded long values

Why this mattered:

- the user observed repeated retry / timeout behavior when only testing the reference locate feature
- part of that pain came from pending-path repeated heavy work rather than one genuinely long request

## 3. Concrete User-Facing Improvement

### Most meaningful improvement

Before:

- negative or strict prompts could still show a plausible-looking but wrong paper card

Now:

- these cases are much more likely to produce no card instead of a misleading one

This is a **quality-of-trust** improvement, not just a cosmetic improvement.

### Practical interpretation

The system is now better at:

- not overclaiming
- not pretending a paper is relevant when the evidence is weak
- separating:
  - abstract-based summary
  - section-grounded guide
  - metadata-only fallback

## 4. Quantified Validation

### Unit / integration regression

The relevant backend test runs in this window reached:

- `231 passed`

This included:

- `tests/unit/test_reference_ui_score_calibration.py`
- `tests/unit/test_retrieval_engine_refs_pack.py`
- `tests/unit/test_references_router_cache.py`
- `tests/unit/test_reference_locate_benchmark_tool.py`
- `tests/unit/test_task_runtime_bg_task.py`
- `tests/unit/test_paper_guide_contracts.py`
- `tests/unit/test_chat_render_reference_notes.py`

### Frontend E2E

Relevant Playwright runs reached:

- `11 passed`

Main files:

- `web/tests/e2e/refs-panel-regression.spec.ts`
- `web/tests/e2e/message-list-locate-primary.spec.ts`

### Live benchmark evidence

Confirmed pass:

- `NORMAL_ADMM_NEGATIVE`

Evidence file:

- `tmp/reference_locate_benchmark/20260412_181448/summary.json`

## 5. Main Files Changed

These are the most important files for the next session:

- `api/reference_ui.py`
- `api/routers/references.py`
- `kb/retrieval_engine.py`
- `kb/chat_store.py`
- `tools/manual_regression/reference_locate_benchmark.py`
- `tests/unit/test_reference_ui_score_calibration.py`
- `tests/unit/test_retrieval_engine_refs_pack.py`
- `tests/unit/test_references_router_cache.py`
- `tests/unit/test_reference_locate_benchmark_tool.py`

## 6. What Is Still Not Fully Solved

There is still one important live issue left:

- some **positive** reference-locate cases still hit an occasional `/api/references/conversation/{conv_id}` GET timeout on a local single-worker dev server

Most visible remaining cases:

- `NORMAL_DYNAMIC_SUPERSAMPLING_DEFINE`
- `NORMAL_HADAMARD_FOURIER_COMPARE`
- Paper Guide positive cross-paper prompts such as:
  - "Besides this paper, what other papers ..."

Important nuance:

- the filtering / relevance logic is much better now
- the remaining bottleneck looks more like **live pending/read-path stability** than "wrong quality logic"

## 7. Recommended Next Steps

### Priority 1. Finish live positive-case stability

Focus on:

- why `/api/references/conversation/{conv_id}` still sometimes times out under single-worker local serving
- whether pending payload generation still does too much work
- whether route cache is being bypassed more than expected

### Priority 2. Re-run the live reference locate benchmark

After stabilizing the route:

- re-run the positive normal-chat cases
- re-run Paper Guide external-paper cases
- confirm that the system is now both:
  - conservative on negative prompts
  - complete enough on positive prompts

### Priority 3. Keep improving card quality only after stability is fixed

Once live stability is good enough:

- keep improving summary precision
- keep improving grounded "why relevant"
- add more prompt families:
  - define
  - compare
  - discuss
  - besides this paper

## 8. Suggested Commands For The Next Session

Backend regressions:

```powershell
python -m pytest -q tests/unit/test_reference_ui_score_calibration.py tests/unit/test_retrieval_engine_refs_pack.py tests/unit/test_references_router_cache.py tests/unit/test_reference_locate_benchmark_tool.py tests/unit/test_task_runtime_bg_task.py tests/unit/test_paper_guide_contracts.py tests/unit/test_chat_render_reference_notes.py
```

Frontend E2E:

```powershell
cd web
npm run test:e2e -- tests/e2e/refs-panel-regression.spec.ts tests/e2e/message-list-locate-primary.spec.ts
```

Targeted live benchmark:

```powershell
python tools/manual_regression/run_reference_locate_benchmark.py --base-url http://127.0.0.1:8016 --timeout-s 120 --case-id NORMAL_DYNAMIC_SUPERSAMPLING_DEFINE
python tools/manual_regression/run_reference_locate_benchmark.py --base-url http://127.0.0.1:8016 --timeout-s 120 --case-id NORMAL_HADAMARD_FOURIER_COMPARE
```

## 9. Bottom-Line Assessment

These changes were worth doing.

Why:

- they reduced misleading reference cards
- they made the card content more honest and grounded
- they exposed the next bottleneck much more clearly

This round was not just refactoring. It produced real user-facing quality improvement, especially on the most trust-damaging failure mode: **showing a wrong paper as if it were relevant**.
