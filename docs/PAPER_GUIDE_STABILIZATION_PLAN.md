# Paper Guide Stabilization Plan

## 1. Purpose

This document defines the next execution phase for the paper-guide feature after the recent locate and reader UX work.

It is intentionally separate from:

- `docs/PAPER_GUIDE_LOCATE_RECOVERY_PLAN.md`
- `docs/PAPER_GUIDE_LOCATE_C1_HARDENING_PLAN.md`
- `docs/PAPER_GUIDE_EVIDENCE_REBUILD_PLAN.md`
- `docs/PAPER_GUIDE_P2_DEEP_READING_PLAN.md`

Those documents now contain a mix of roadmap, audit notes, and landed changes. This plan is the clean execution baseline for the next phase.

## 2. Current Status

The paper-guide feature is already useful:

- bound-paper scope works
- strict locate is much tighter than before
- reader split-pane is usable on desktop
- Ask from selection works
- session-scoped sentence highlight works

But the system is not yet in a good long-term state.

The main problems are:

1. The frontend reader is overloaded.
2. Locate behavior is split across backend provenance, message rendering, and reader-side fallback logic.
3. The interaction layer still lacks browser-level regression coverage.
4. Reader highlight state still contains compatibility fields that should not become permanent.
5. Deep-reading workflows are still weaker than the locate workflow.

## 3. Success Criteria

The next phase is successful only if all of the following become true:

1. Structured locate has one clear contract from backend to frontend.
2. The reader no longer re-invents structured target selection when the backend already resolved it.
3. The reader code is split into smaller units with clear ownership.
4. Browser regressions around locate, selection, highlight, and resize are covered by automated tests.
5. The feature supports actual deep reading, not only point-and-jump evidence lookup.

## 4. Non-Goals

This phase does not aim to:

1. redesign the backend retrieval stack from scratch
2. change the answer contract again unless needed by locate correctness
3. ship large new visual redesigns for the chat page
4. optimize bundle size as a primary goal

## 5. Phase Plan

### P0. Locate Contract And Guardrails

Goal:
Make locate behavior deterministic and testable.

Why first:
Without a single contract, every later UI or UX improvement can reintroduce drift.

Work items:

1. Define three explicit objects and use them as the shared language across backend and frontend:
   - `LocateTarget`
   - `ClaimGroup`
   - `ReaderSelectionRange`
2. Make structured locate authoritative.
   - If a structured `LocateTarget` exists, `MessageList` must not re-rank or reinterpret it.
   - Reader may only do constrained fallback when the target cannot be resolved in DOM.
3. Normalize locate payload shape from `MessageList -> ChatPage -> PaperGuideReaderDrawer`.
4. Reduce compatibility layers in `ReaderSessionHighlight`.
   - Keep the range-based path as the long-term primary path.
   - Mark legacy fallback fields as temporary compatibility only.
5. Add frontend/browser regression coverage for the critical paper-guide flows.

Primary files:

- `kb/paper_guide_provenance.py`
- `kb/task_runtime.py`
- `web/src/components/chat/MessageList.tsx`
- `web/src/components/chat/PaperGuideReaderDrawer.tsx`
- `web/src/pages/ChatPage.tsx`

Acceptance:

1. A structured locate click reaches the same target on every repeated click.
2. Reader does not silently replace a structured target with a front-end guessed target.
3. The following flows have browser regression coverage:
   - strict locate
   - alternative candidate failover
   - Ask from selection
   - same-paragraph highlight
   - cross-paragraph highlight
   - split-pane resize

Exit condition:
P0 is done only when locate correctness is defined by contract, not by heuristic agreement between layers.

### P1. Reader Decomposition

Goal:
Split the reader into maintainable units with a stable shell/panel model.

Why second:
The current reader file is too large to safely extend. New UX work will keep creating regressions if the structure stays the same.

Work items:

1. Extract a pure panel body component from the current reader shell.
2. Split logic into focused hooks/modules:
   - `useReaderDocument`
   - `useReaderLocateEngine`
   - `useReaderSelectionHighlights`
   - `useReaderSelectionActions`
3. Keep the shell layer thin:
   - inline pane shell
   - drawer shell
   - future standalone reader page shell
4. Reduce `MarkdownRenderer` coupling.
   - move reader-only locate/highlight helpers out of the generic renderer where practical
   - keep the renderer focused on rendering, not reader orchestration
5. Add a small local state model for reader UI state:
   - active candidate
   - candidate panel open/closed
   - selection bubble state
   - locate status

Primary files:

- `web/src/components/chat/PaperGuideReaderDrawer.tsx`
- `web/src/components/chat/MarkdownRenderer.tsx`
- `web/src/pages/ChatPage.tsx`

Acceptance:

1. `PaperGuideReaderDrawer.tsx` becomes a shell/composition file instead of the entire reader system.
2. Locate logic, selection logic, and shell layout are in separate units.
3. Reader behavior remains the same across inline and drawer presentations.

Exit condition:
P1 is done when a new reader interaction can be changed in one focused module without touching unrelated locate code.

### P2. Deep Reading Product Layer

Goal:
Turn paper-guide from an evidence lookup feature into a real reading workflow.

Why third:
Once locate correctness and reader structure are stable, the next value is reading efficiency.

Work items:

1. Add a section outline / table-of-contents panel inside the reader.
2. Add a compact "My Highlights" view for the current session.
3. Add next/previous evidence navigation inside the reader.
4. Improve long-document navigation affordances:
   - section jump
   - reveal current section
   - better locate status phrasing when degraded
5. Evaluate whether a standalone reader route should be added after split-pane stabilizes.

Primary files:

- `web/src/components/chat/PaperGuideReaderDrawer.tsx`
- `web/src/pages/ChatPage.tsx`
- new reader subcomponents as needed

Acceptance:

1. A user can navigate long papers without repeatedly relying on chat locate entrances.
2. A user can revisit their highlights inside the current conversation.
3. Evidence review across a long answer no longer requires repeated manual scroll hunting.

Exit condition:
P2 is done when paper-guide supports both "answer verification" and "focused reading" comfortably.

Detailed execution note:
See `docs/PAPER_GUIDE_P2_DEEP_READING_PLAN.md` for the concrete slice plan and implementation order.

## 6. Recommended Execution Slices

### Slice A

Scope:
P0 contract cleanup only.

Tasks:

1. Define the stable locate payload types.
2. Remove frontend re-ranking on structured targets.
3. Add browser regression harness for the six critical flows.

Why this slice first:
It reduces the highest regression risk before more product work lands.

### Slice B

Scope:
P1 reader split.

Tasks:

1. extract reader panel
2. extract locate hook
3. extract selection/highlight hook
4. keep shell behavior identical

Why this slice second:
It shrinks the blast radius of every later change.

### Slice C

Scope:
P2 deep-reading upgrades.

Tasks:

1. section outline
2. highlights list
3. next/prev evidence controls

Why this slice third:
It builds product value on top of a stable technical base.

## 7. Risks

1. The reader currently relies on both DOM identity and text-range identity.
   - During P0 and P1, these paths must not drift apart.
2. `MarkdownRenderer` is shared by chat and reader.
   - Reader-only behavior must not regress normal chat rendering.
3. Browser text-range behavior can vary.
   - Highlight tests must run in a real browser environment, not only unit tests.

## 8. Testing Strategy

Required:

1. backend unit tests for provenance and locate target contract
2. frontend browser tests for reader interactions
3. `npm run build` on every slice

Recommended browser fixtures:

1. a paragraph quote target
2. a blockquote target
3. a display equation target
4. a figure target
5. same-paragraph highlight
6. cross-paragraph highlight

## 9. Immediate Next Task

Start with `Slice A`.

Concrete first implementation target:

1. define the stable locate payload shape
2. stop `MessageList` from re-ranking structured locate targets
3. add browser regression tests for:
   - strict locate
   - selection Ask
   - sentence highlight
   - cross-paragraph highlight

If this is stable, move to the reader split in `Slice B`.
