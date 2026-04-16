# Reference Locate Architecture Execution Plan (2026-04-14)

## 1. Goal

Deliver a `reference locate` implementation that is:

1. faster on first read
2. stable under local single-worker serving
3. precise about which paper / section / block was matched
4. clear about whether the result is `pending`, `fast`, `full`, or `failed`
5. consistent between:
   - main answer text
   - refs panel
   - reader open target
   - citation behavior
   - language selection

This plan is written as an executable engineering document with concrete code touchpoints, test commands, verification steps, and acceptance criteria.

## 2. Current Observations

### Product-Level Symptoms

1. The same hit can appear correct in one surface and wrong in another.
   - Example pattern:
     - summary sounds like `2.2`
     - card heading shows `2.4`
     - reader open lands near `2.1`
2. Normal locate answers can still leak raw or implied in-paper citation behavior that should belong to `citation lookup`, not `reference locate`.
3. Clean fresh processes already show better quality than the long-running dev reload server, which means some failures are state / cache / persistence problems rather than pure retrieval quality problems.
4. Fast payloads can already be good enough for first paint, but the full payload persistence path is not yet reliably observable.
5. Language behavior is still too distributed across prompt inference, UI inference, and cache reuse.

### Code-Level Causes

The same evidence identity is currently inferred multiple times across the stack:

- card evidence selection:
  - [api/reference_ui.py](/f:/research-papers/2026/Jan/else/kb_chat/api/reference_ui.py:2896)
  - [api/reference_ui.py](/f:/research-papers/2026/Jan/else/kb_chat/api/reference_ui.py:3034)
  - [api/reference_ui.py](/f:/research-papers/2026/Jan/else/kb_chat/api/reference_ui.py:4136)
- route cache / warm / fallback:
  - [api/routers/references.py](/f:/research-papers/2026/Jan/else/kb_chat/api/routers/references.py:373)
  - [api/routers/references.py](/f:/research-papers/2026/Jan/else/kb_chat/api/routers/references.py:435)
  - [api/routers/references.py](/f:/research-papers/2026/Jan/else/kb_chat/api/routers/references.py:538)
- async generation + persistence:
  - [kb/task_runtime.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/task_runtime.py:477)
  - [kb/task_runtime.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/task_runtime.py:2233)
- raw answer render / citation linking:
  - [api/chat_render.py](/f:/research-papers/2026/Jan/else/kb_chat/api/chat_render.py:264)
  - [api/chat_render.py](/f:/research-papers/2026/Jan/else/kb_chat/api/chat_render.py:855)
  - [api/chat_render.py](/f:/research-papers/2026/Jan/else/kb_chat/api/chat_render.py:958)
- frontend polling / suppression / rendering:
  - [web/src/stores/chatStore.ts](/f:/research-papers/2026/Jan/else/kb_chat/web/src/stores/chatStore.ts:212)
  - [web/src/stores/chatStore.ts](/f:/research-papers/2026/Jan/else/kb_chat/web/src/stores/chatStore.ts:337)
  - [web/src/stores/chatStore.ts](/f:/research-papers/2026/Jan/else/kb_chat/web/src/stores/chatStore.ts:407)
  - [web/src/components/refs/RefsPanel.tsx](/f:/research-papers/2026/Jan/else/kb_chat/web/src/components/refs/RefsPanel.tsx:211)
  - [web/src/components/refs/RefsPanel.tsx](/f:/research-papers/2026/Jan/else/kb_chat/web/src/components/refs/RefsPanel.tsx:230)

## 3. Scope

### In Scope

- `normal` and `paper_guide` reference locate quality
- fast / full payload consistency
- rendered refs persistence
- evidence identity consistency
- separation of `reference locate` vs `citation lookup`
- locale consistency for refs panel cards
- benchmark + regression + live verification workflow

### Out of Scope

- large retrieval model changes
- replacing the underlying chat model
- redesigning the entire conversation UI
- rebuilding paper guide prompt families unrelated to refs behavior

## 4. Target Architecture

### 4.1 Layered Responsibilities

| Layer | Responsibility | Main Files | Output |
| --- | --- | --- | --- |
| Retrieval / block identity | Produce doc-level hits plus stable block candidates | [kb/source_blocks.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/source_blocks.py:14), [kb/source_blocks.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/source_blocks.py:788) | evidence atoms |
| Primary evidence resolver | Pick one primary block and ranked alternatives once | [api/reference_ui.py](/f:/research-papers/2026/Jan/else/kb_chat/api/reference_ui.py:2896) | `primary_evidence` |
| Render state machine | Build `pending/fast/full/failed` payloads from the same evidence identity | [api/reference_ui.py](/f:/research-papers/2026/Jan/else/kb_chat/api/reference_ui.py:4136), [kb/task_runtime.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/task_runtime.py:2233) | typed refs payload |
| Delivery / caching | Read stored payloads, warm only when needed, never re-decide semantics | [api/routers/references.py](/f:/research-papers/2026/Jan/else/kb_chat/api/routers/references.py:538) | API response |
| Answer rendering | Strip or link citations according to explicit mode | [api/chat_render.py](/f:/research-papers/2026/Jan/else/kb_chat/api/chat_render.py:264), [api/chat_render.py](/f:/research-papers/2026/Jan/else/kb_chat/api/chat_render.py:958) | rendered answer text |
| Frontend display | Render state, never reclassify backend semantics | [web/src/stores/chatStore.ts](/f:/research-papers/2026/Jan/else/kb_chat/web/src/stores/chatStore.ts:212), [web/src/components/refs/RefsPanel.tsx](/f:/research-papers/2026/Jan/else/kb_chat/web/src/components/refs/RefsPanel.tsx:230) | visible panel |

### 4.2 Required New Contract

Introduce a structured `primary_evidence` object per displayed hit.

```json
{
  "doc_id": "optional-doc-id",
  "source_path": "db/.../paper.en.md",
  "block_id": "p_00042",
  "anchor_id": "hd_00008",
  "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
  "snippet": "The matched sentence or compacted block text",
  "anchor_kind": "section",
  "anchor_number": 2,
  "selection_reason": "prompt_aligned",
  "score": 9.47,
  "alternatives": [
    {
      "block_id": "p_00039",
      "heading_path": "2. Comparison of theory / 2.4 Efficiency",
      "reason": "secondary_candidate"
    }
  ]
}
```

### 4.3 Required Render Status Contract

Persist explicit render status in `message_refs`.

Minimum fields:

- `render_status`: `pending | fast | full | failed`
- `render_error`: short machine-readable code
- `render_error_detail`: optional debug string
- `render_built_at`
- `render_attempts`
- `evidence_sig`
- `locale`

This is needed because the current `rendered_payload_json` + `rendered_payload_sig` pair in [kb/chat_store.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/chat_store.py:735) and [kb/chat_store.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/chat_store.py:829) is not enough to explain why full persistence did or did not happen.

## 5. Execution Plan

## Phase A. Primary Evidence Contract

### Objective

Choose one primary evidence block once and reuse it everywhere.

### Files

- [api/reference_ui.py](/f:/research-papers/2026/Jan/else/kb_chat/api/reference_ui.py:2896)
- [kb/source_blocks.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/source_blocks.py:788)
- [api/reference_ui.py](/f:/research-papers/2026/Jan/else/kb_chat/api/reference_ui.py:3034)

### Tasks

1. Extract a dedicated `select_primary_evidence()` helper from the current `_select_primary_ref_evidence()` flow.
2. Return structured identity, not just polished strings.
3. Make `build_hit_ui_meta()` consume `primary_evidence` rather than rebuilding heading/snippet identity independently.
4. Carry `primary_evidence.block_id`, `anchor_id`, `heading_path`, and `selection_reason` into:
   - `ui_meta`
   - `reader_open`
   - persisted rendered payload

### Tests To Add Or Tighten

- `tests/unit/test_reference_ui_score_calibration.py`
  - primary evidence stays on `2.2` when `2.4` is only a weaker fallback
  - `summary_heading_consistent`
  - `heading_reader_open_consistent`
- `tests/unit/test_retrieval_engine_refs_pack.py`
  - block identity survives doc grouping for the top hit

### Verification

1. Run unit tests.
2. Replay `NORMAL_HADAMARD_FOURIER_COMPARE`.
3. Inspect returned payload:
   - `ui_meta.primary_evidence_source`
   - `ui_meta.primary_evidence_heading_path`
   - `reader_open.headingPath`
   - `reader_open.blockId`

### Acceptance Criteria

- `NORMAL_HADAMARD_FOURIER_COMPARE` returns top hit `OE-2017...`
- top-hit `heading_path` and `reader_open.headingPath` are identical
- `blockId` is present for the top hit
- summary text, heading path, and reader target all point to the same evidence block

## Phase A+. Shared Primary Evidence Reuse In Main Answer

### Objective

Stop letting the main answer re-pick evidence independently from the refs card.

This phase turns `primary_evidence` from a refs-only payload field into a shared answer contract input. The main answer should consume the same structured evidence identity that the refs card already resolved.

### Files

- [kb/paper_guide_context_runtime.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/paper_guide_context_runtime.py:26)
- [kb/generation_answer_finalize_runtime.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/generation_answer_finalize_runtime.py:681)
- [kb/answer_contract.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/answer_contract.py:997)

### Tasks

1. Add `primary_evidence` to paper-guide evidence cards and `paper_guide_contracts_seed`.
2. In `_finalize_generation_answer()`, resolve one shared `primary_evidence` from:
   - explicit seed first
   - else top evidence card
3. Pass that shared `primary_evidence` into `_apply_answer_contract_v1()`.
4. Make answer-contract fallback builders prefer explicit `primary_evidence` over re-scoring `answer_hits`.
5. Keep `answer_hits` as fallback only when shared `primary_evidence` is absent or incomplete.

### Tests To Add Or Tighten

- `tests/unit/test_paper_guide_context_runtime.py`
  - evidence cards include `primary_evidence`
  - contracts seed includes top `primary_evidence`
- `tests/unit/test_generation_answer_finalize_runtime.py`
  - finalize passes shared `primary_evidence` into answer contract
- `tests/unit/test_task_runtime_answer_contract.py`
  - explicit `primary_evidence` wins over weaker `answer_hits`
  - conclusion/evidence bridge use the same heading/snippet identity

### Verification

1. Run the three unit suites above.
2. Replay one mixed-language locate case and one compare case.
3. Inspect:
   - main answer `Conclusion`
   - main answer `Evidence`
   - refs card `ui_meta.primary_evidence`
   - `reader_open.primaryEvidence`
4. Confirm the answer text now names the same section/snippet that the refs card resolved.

### Acceptance Criteria

- main answer `Conclusion` and `Evidence` reuse the same `heading_path` as refs `primary_evidence`
- answer fallback no longer drifts to a different section when `primary_evidence` is present
- `paper_guide_contracts_seed.primary_evidence` is populated for top-hit paper-guide cases
- contract fallback still works when `primary_evidence` is absent

### Progress Update 2026-04-15

- Implemented shared `primary_evidence` generation in evidence cards and `paper_guide_contracts_seed`
- `_finalize_generation_answer()` now passes shared `primary_evidence` into `answer_contract`
- `answer_contract` fallback now prefers explicit `primary_evidence` over weaker `answer_hits`
- shared `primary_evidence` is now also propagated into provenance payloads and render packets so replay/debug views can compare main answer, refs card, and render identity against the same evidence object
- Unit coverage added across:
  - `tests/unit/test_paper_guide_context_runtime.py`
  - `tests/unit/test_generation_answer_finalize_runtime.py`
  - `tests/unit/test_task_runtime_answer_contract.py`
  - `tests/unit/test_task_runtime_provenance.py`
  - `tests/unit/test_generation_state_runtime.py`
  - `tests/unit/test_chat_render_reference_notes.py`

## Phase B. Render State Machine And Persistence Observability

### Objective

Make `pending`, `fast`, `full`, and `failed` explicit and observable.

### Files

- [kb/task_runtime.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/task_runtime.py:2233)
- [kb/task_runtime.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/task_runtime.py:477)
- [kb/chat_store.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/chat_store.py:735)
- [kb/chat_store.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/chat_store.py:858)
- [api/routers/references.py](/f:/research-papers/2026/Jan/else/kb_chat/api/routers/references.py:538)

### Tasks

1. Extend `message_refs` persistence with render status metadata.
2. In `_bg_enrich_refs()`, write:
   - render start
   - render success
   - render failure
3. Stop swallowing full-render failures silently.
4. Make `GET /api/references/conversation/{conv_id}` read status first, then decide:
   - return stored `full`
   - else return `fast`
   - else return `pending`
   - if failed, return `fast` with explicit `enrichment_pending=false` and failure metadata
5. Keep route cache as read-through cache only; do not let it become the source of truth.

### Tests To Add Or Tighten

- `tests/unit/test_chat_store_rendered_refs.py`
  - render status round-trip
  - failure metadata round-trip
- `tests/unit/test_references_router_cache.py`
  - stored full payload wins over fast cache
  - failed full render does not lock the route into endless pending
- `tests/unit/test_task_runtime_bg_task.py`
  - async render failure records status and error code

### Verification

1. Start a fresh backend process.
2. Run one positive benchmark case.
3. Poll:
   - `/api/references/conversation/{conv_id}`
   - `message_refs` row in SQLite
4. Confirm status transitions:
   - `pending` -> `fast` -> `full`
   - or `pending` -> `fast` -> `failed`

### Acceptance Criteria

- full render status is visible in storage
- full render success writes non-empty `rendered_payload_json`
- full render failure writes a non-empty `render_error`
- route no longer stays indefinitely in `fast` without recorded reason

## Phase C. Split Reference Locate From Citation Lookup

### Objective

Normal locate answers must not produce in-paper citation behavior.

### Files

- [api/chat_render.py](/f:/research-papers/2026/Jan/else/kb_chat/api/chat_render.py:264)
- [api/chat_render.py](/f:/research-papers/2026/Jan/else/kb_chat/api/chat_render.py:855)
- [api/chat_render.py](/f:/research-papers/2026/Jan/else/kb_chat/api/chat_render.py:958)
- generation code in [kb/task_runtime.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/task_runtime.py:2233)

### Tasks

1. Introduce explicit answer mode:
   - `locate_only`
   - `citation_lookup`
2. In `locate_only` mode:
   - do not emit `[[CITE:...]]`
   - strip any residual raw cite tokens before storing final assistant content
3. Keep in-paper citation linking only for:
   - explicit citation lookup
   - structured citation outputs meant for popup rendering

### Tests To Add Or Tighten

- `tests/unit/test_chat_render_reference_notes.py`
  - normal locate answer does not retain `[[CITE:...]]`
  - citation lookup still links correctly
- `tests/unit/test_task_runtime_bg_task.py`
  - generated normal locate content is persisted without cite tokens

### Verification

1. Ask:
   - `Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?`
2. Inspect stored assistant `content` and `rendered_content`.

### Acceptance Criteria

- normal locate answer raw `content` contains no `[[CITE:`
- rendered answer contains no clickable `[2]`-style in-paper cite derived from the matched paper
- citation lookup flows remain unchanged

## Phase D. Locale Contract And UI Simplification

### Objective

Make refs card language consistent with user/UI preference and stop frontend semantic reclassification.

### Files

- [api/reference_ui.py](/f:/research-papers/2026/Jan/else/kb_chat/api/reference_ui.py:3034)
- [api/routers/references.py](/f:/research-papers/2026/Jan/else/kb_chat/api/routers/references.py:41)
- [web/src/stores/chatStore.ts](/f:/research-papers/2026/Jan/else/kb_chat/web/src/stores/chatStore.ts:212)
- [web/src/components/refs/RefsPanel.tsx](/f:/research-papers/2026/Jan/else/kb_chat/web/src/components/refs/RefsPanel.tsx:211)

### Tasks

1. Persist locale with rendered payload and render status.
2. Ensure locale is an input to render generation, not a best-effort guess.
3. Reduce frontend suppression logic so backend returns:
   - `display_state`
   - `suppression_reason`
   - `guide_filter`
4. Keep frontend focused on rendering and polling only.

### Tests To Add Or Tighten

- `tests/unit/test_reference_ui_score_calibration.py`
  - zh preference yields zh card copy
  - en preference yields en card copy
- `web/tests/e2e/refs-panel-regression.spec.ts`
  - card copy language follows preference
  - panel state is preserved through polling updates

### Verification

1. Set UI locale to zh.
2. Ask an English prompt.
3. Confirm refs card stays zh.
4. Repeat with en.

### Acceptance Criteria

- card copy uses configured preferred locale, not whichever text happened to dominate the prompt
- cache invalidation respects locale changes
- frontend no longer needs to suppress obviously wrong cards by re-guessing semantics in common paths

## Phase E. Benchmark And Visual Verification Harness

### Objective

Make the feature continuously verifiable, not just manually arguable.

### Files

- `tools/manual_regression/reference_locate_benchmark.py`
- `tools/manual_regression/manifests/reference_locate_quality_v1.json`
- `web/tests/e2e/refs-panel-regression.spec.ts`
- `web/tests/e2e/message-list-locate-primary.spec.ts`

### Tasks

1. Extend benchmark output to record:
   - `summary_heading_consistent`
   - `heading_reader_open_consistent`
   - `fast_to_full_primary_block_same`
2. Add at least one fresh-process benchmark run to the runbook.
3. Add a visual verification checklist inspired by the `pdf` skill:
   - do not trust text-only payloads for final acceptance
   - verify the actual reader landing and highlight

### Cases That Must Stay In The Suite

- `NORMAL_DYNAMIC_SUPERSAMPLING_DEFINE`
- `NORMAL_HADAMARD_FOURIER_COMPARE`
- `NORMAL_ADMM_NEGATIVE`
- `GUIDE_OTHER_PAPERS_FOURIER`
- `GUIDE_OTHER_PAPERS_ADMM_NEGATIVE`

### Acceptance Criteria

- benchmark suite records consistency metrics
- positive cases pass with correct paper identity and consistent locate target
- negative cases produce zero misleading cards

### Progress Update 2026-04-16

- `reference_locate_benchmark.py` now fetches assistant messages after refs settle, so case evaluation can inspect assistant-side evidence surfaces in addition to refs payloads
- the benchmark manifest now includes `evidence_identity` gates for positive cases
- positive cases currently verify:
  - pack-level `primary_evidence` exists
  - top-hit `reader_open.primaryEvidence` stays aligned with pack-level `primary_evidence`
  - paper-guide positive case also requires assistant-side `primary_evidence` alignment
- non-paper-guide positive flows now emit a minimal assistant-side contract snapshot with `primary_evidence` and `render_packet.primary_evidence`, so normal locate cases can also participate in assistant/refs identity checks
- unit coverage added in `tests/unit/test_reference_locate_benchmark_runner.py` for both aligned and drifting evidence identities
- definition-style locate prompts now suppress synthetic `focus_term: ...` summary candidates when the underlying sentence only loosely matches a partial token such as `dynamic`
- definition-style why-lines now require the explicit focus term itself, not just a nearby section heading, before they are treated as specific enough
- latest fresh in-process benchmark run: `tmp/reference_locate_benchmark/20260416_113329/summary.json` -> `overall_status=PASS`, `fail_count=0`, `case_count=5`
- positive normal-locate answers now generate evidence-aware `Next Steps` from the same locate target, so definition and comparison prompts point users back to the matched section/source instead of falling back to generic reading templates
- positive normal-locate `Conclusion / Evidence` now also reuse the same locate target more explicitly: paper-identity prompts are rewritten to name the matched paper, and evidence fallback uses relation-specific phrasing such as `directly defines` / `directly compares` instead of generic `states`
- mixed-language positive locate answers now align `Conclusion` and `Evidence` on the same `source + heading + relation`: Chinese bridge conclusions name the matched paper explicitly, while Chinese evidence keeps the grounded original snippet instead of repeating a generic summary sentence
- refs payloads now include `pipeline_debug` counters (`raw_hit_count`, `post_score_gate_hit_count`, `post_focus_filter_hit_count`, `post_llm_filter_hit_count`, `final_hit_count`), and the benchmark suite uses them to distinguish `retrieval_empty_before_ui_filters` from later-stage suppression

## 6. Test Plan

### Backend Regression

```powershell
python -m pytest -q `
  tests/unit/test_reference_ui_score_calibration.py `
  tests/unit/test_retrieval_engine_refs_pack.py `
  tests/unit/test_references_router_cache.py `
  tests/unit/test_task_runtime_bg_task.py `
  tests/unit/test_chat_render_reference_notes.py `
  tests/unit/test_chat_store_rendered_refs.py
```

### Frontend E2E

```powershell
cd web
npm run test:e2e -- tests/e2e/refs-panel-regression.spec.ts tests/e2e/message-list-locate-primary.spec.ts
```

### Live Benchmark

Run against a fresh single-process backend first, then against the normal dev server.

```powershell
python tools/manual_regression/run_reference_locate_benchmark.py --base-url http://127.0.0.1:8018 --timeout-s 180 --case-id NORMAL_DYNAMIC_SUPERSAMPLING_DEFINE
python tools/manual_regression/run_reference_locate_benchmark.py --base-url http://127.0.0.1:8018 --timeout-s 180 --case-id NORMAL_HADAMARD_FOURIER_COMPARE
python tools/manual_regression/run_reference_locate_benchmark.py --base-url http://127.0.0.1:8018 --timeout-s 180 --case-id NORMAL_ADMM_NEGATIVE
python tools/manual_regression/run_reference_locate_benchmark.py --base-url http://127.0.0.1:8018 --timeout-s 180 --case-id GUIDE_OTHER_PAPERS_FOURIER
python tools/manual_regression/run_reference_locate_benchmark.py --base-url http://127.0.0.1:8018 --timeout-s 180 --case-id GUIDE_OTHER_PAPERS_ADMM_NEGATIVE
```

### Manual Visual Verification

For the top hit in each positive case:

1. open the refs card
2. click `定位`
3. verify that the reader lands on the expected section heading
4. verify that the highlighted snippet belongs to the same section as the card summary
5. verify that no wrong `[2]` / `[4]` in-paper citation is attached to the matched paper title

## 7. Global Acceptance Standard

The feature is accepted only if all of the following are true:

1. `NORMAL_HADAMARD_FOURIER_COMPARE`
   - top hit is the OE comparison paper
   - top hit summary, heading, and reader target all point to the same section
2. `NORMAL_DYNAMIC_SUPERSAMPLING_DEFINE`
   - only the defining paper is shown
   - no extra misleading paper is shown
3. `NORMAL_ADMM_NEGATIVE`
   - zero misleading cards are shown
4. `GUIDE_OTHER_PAPERS_FOURIER`
   - bound paper is excluded
   - external Fourier paper is shown
5. `GUIDE_OTHER_PAPERS_ADMM_NEGATIVE`
   - zero misleading external cards are shown
   - guide filter state is still visible
6. normal locate answers produce no raw or clickable in-paper citation tokens
7. refs card language follows configured preference consistently
8. full render persistence is observable and debuggable
9. clean fresh-process benchmark results are reproducible across two independent runs

## 8. Rollout Order

1. Phase A
2. Phase B
3. Phase C
4. Phase D
5. Phase E

Reason:

- A and B solve the core correctness and observability problem.
- C removes the most trust-damaging UI bug.
- D stabilizes presentation semantics.
- E makes the whole system easier to keep healthy.

## 9. Stop Conditions

Pause rollout and fix immediately if any of the following happens:

1. top-hit paper identity regresses on any existing positive benchmark case
2. negative cases start showing misleading cards again
3. full render persistence silently stops without recorded failure state
4. normal locate answers start emitting raw cite tokens again
5. locale changes fail to invalidate cached refs payloads

## 10. Deliverables

The implementation is complete only when all of these are present:

1. code changes for Phases A-E
2. updated benchmark outputs
3. passing backend regression suite
4. passing frontend E2E suite
5. at least one fresh-process live benchmark report saved under `tmp/reference_locate_benchmark/`
6. brief implementation handoff note summarizing:
   - what changed
   - what passed
   - what remains risky

## 11. Suggested Implementation Notes

1. Prefer evolving the current code incrementally instead of rewriting the whole feature in one pass.
2. Keep fast path cheap, but never let it choose a different evidence identity from full path unless explicitly recorded.
3. When adding new persistence columns or metadata, default to forward-compatible reads.
4. Treat user-visible trust as the main optimization target:
   - better to show no card than a wrong card
   - better to show `fast` honestly than to pretend it is `full`
