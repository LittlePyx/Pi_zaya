# Reference Locate Improvement Plan (2026-04-11)

## Goal

Improve "reference locate" quality across both normal chat and Paper Guide so that:

1. the jump target shown to users is actually correct
2. the visible locate cut is grounded and reasonable
3. citation / source metadata shown next to the locate is trustworthy
4. hidden or internal locate targets do not leak back into the UI
5. cross-paper hits in Paper Guide are presented clearly instead of disappearing silently

This plan treats user-visible locate behavior as the main truth source, not internal provenance alone.

## Current Problems

### Normal Chat Refs

- ordinary refs currently build `reader_open` mostly from `headingPath + snippet + anchorKind/anchorNumber`
- exact identity fields such as `blockId`, `anchorId`, `strictLocate`, and `locateTarget` are not reliably carried through
- this means the locate button can look plausible while still opening the reader heuristically

### Paper Guide Render Packet Fallback

- structured provenance already hides entries with `locatePolicy = hidden`
- but `render_packet` fallback can still recreate a clickable locate entry
- this risks exposing internal or low-confidence locate targets that should stay hidden

### Paper Guide Cross-Paper Refs UX

- guide mode already filters the currently guided paper out of external refs
- when only the self-source was filtered and no other paper remains, the refs panel can disappear
- users then cannot tell whether nothing matched, or whether only the bound paper was removed

### Test Coverage Gap

- some tests validate payload shape but not final reader landing behavior
- acceptance needs to be defined in terms of user-visible jump correctness

## Principles

1. Prefer no exact jump over a wrong exact jump.
2. Preserve exact identity end-to-end once it has been resolved.
3. Use one shared locate contract wherever possible.
4. Make hidden locate a hard UI constraint, not a best-effort hint.
5. Add guardrails before broad rollout.

## Target Architecture

### Shared Contract

Use the same core locate fields across normal refs and Paper Guide:

- `blockId`
- `anchorId`
- `relatedBlockIds`
- `strictLocate`
- `locateTarget`

This keeps the reader-opening path consistent and testable.

### Layer Responsibilities

- `reference_ui.py`
  - choose source hit presentation
  - resolve exact block candidates when confidence is good enough
  - emit the full `reader_open` contract
- `RefsPanel.tsx`
  - preserve the backend locate contract
  - never downgrade an exact locate payload into a heuristic-only payload
- `MessageList.tsx`
  - enforce surface policy consistently
  - never re-surface hidden locate through fallback
- `reader`
  - consume exact locate identity when available
  - fall back to heuristic locate only when exact identity is absent

## Implementation Plan

### Phase 1. Contract Hardening

Goal:
Upgrade normal-chat refs from heuristic-only locate to exact-capable locate when source blocks can be resolved safely.

Tasks:

1. in `api/reference_ui.py`
   - resolve source markdown path reliably
   - load source blocks
   - match primary and secondary locate seeds to source blocks
   - emit `blockId`, `anchorId`, `strictLocate`, `relatedBlockIds`, and `locateTarget`
2. in frontend reader-open helpers
   - preserve all exact locate fields instead of dropping them
3. in `RefsPanel`
   - pass through the richer payload unchanged
4. keep fallback behavior when exact resolution is unavailable

Exit criteria:

- ordinary refs can produce exact locate payloads on regression fixtures
- frontend preserves those exact fields when the locate button is clicked
- no existing refs behavior regresses when exact resolution is absent

### Phase 2. Hidden Locate Consistency

Goal:
Ensure hidden/internal locate entries never become user-clickable through fallback.

Tasks:

1. make `render_packet` fallback respect `locatePolicy = hidden`
2. also respect `locateSurfacePolicy = hidden`
3. add a reader-level regression scenario that confirms no locate chip appears

Exit criteria:

- hidden locate regression produces zero visible locate chips
- render-packet fallback still works for valid visible locate targets

### Phase 3. Guide External-Refs UX

Goal:
Make Paper Guide explain what happened when self-source refs are filtered out.

Tasks:

1. treat `guide_filter.hidden_self_source` as a renderable refs state
2. show a small explanatory note when:
   - the guide source was filtered out
   - no external papers remain
3. keep the panel hidden only when there is truly nothing to show or explain

Exit criteria:

- users can distinguish "no external hit" from "bound paper filtered"

### Phase 3.5. LLM-Assisted Cross-Paper Hit Disambiguation

Goal:
Use a small, gated LLM step only where ordinary score-based ranking is most likely to surface the wrong top paper.

Tasks:

1. keep deterministic score filtering as the default
2. sort ready refs by calibrated UI score before display
3. only when top hits are close or low-confidence:
   - ask the LLM to rerank the top `3-4` paper hits
   - provide prompt, source name, heading, summary, why-line, snippet, and citation hints
4. preserve deterministic fallback whenever:
   - LLM is disabled
   - credentials are unavailable
   - the current ranking is already clearly separated

Exit criteria:

- ambiguous cross-paper prompts can promote the more directly relevant paper to top-1
- strong deterministic winners keep their existing order
- the new LLM step remains bounded and optional, not a wholesale rewrite

### Phase 4. Shared Skill / Architecture Follow-up

Goal:
Move cross-paper reference locate into a more explicit skill-first design.

Candidate skills:

- `CrossPaperReferenceLocateSkill`
- `NegativeEvidenceSkill`
- `ReferenceCiteDetailSkill`

Tasks:

1. define the skill contract
2. let skills choose evidence families
3. let shared grounder own exact landing and snippet selection
4. unify normal refs and Paper Guide external refs around the same locate contract

Exit criteria:

- cross-paper locate behavior stops being scattered across UI/meta assembly code

## How To Test

### Unit Tests

Backend:

- `tests/unit/test_reference_ui_score_calibration.py`
  - exact locate fields appear when source blocks can be matched
  - fallback still works when exact match is unavailable
- `tests/unit/test_chat_render_reference_notes.py`
  - render-packet compatibility remains stable

Frontend / reader regression:

- `web/tests/e2e/refs-panel-regression.spec.ts`
  - exact fields survive click-through from refs panel
- `web/tests/e2e/message-list-locate-primary.spec.ts`
  - hidden locate does not render a chip
  - visible render-packet locate still renders correctly

### Live / Manual Checks

Run these against the active frontend and backend:

1. normal chat targeted locate
   - prompt example:
     - `In the SCINeRF paper, where is ADMM discussed? Please point me to the source section.`
2. Paper Guide external refs
   - prompt example:
     - `Besides this paper, what other papers in my library discuss ADMM? Please point me to those sources too.`
3. Paper Guide hidden/negative locate sanity
   - prompt example:
     - `In this paper, where is ADMM mentioned? Please point me to the exact source location.`

Check:

- whether the UI shows a locate button only when it should
- whether the reader opens to the intended block
- whether the snippet/highlight shown to the user is a sensible cut

## Acceptance Metrics

### Correctness

- normal refs regression fixture:
  - `100%` preservation of backend `blockId/anchorId/strictLocate/locateTarget`
- hidden locate regression:
  - `0` visible locate chips
- render-packet visible locate regression:
  - `100%` visible chips remain functional

### User-Visible Quality

- guide self-source filtered state:
  - explanatory note rendered `100%` of the time in the regression scenario
- locate cut quality:
  - top-level snippet/highlight must come from either:
    - the matched block text
    - a snippet with strong overlap to the matched block

### Live Behavior

For the curated manual prompts above:

- no knowingly wrong exact jump
- no hidden locate leakage
- no silent disappearance when guide self-source filtering occurred

## This Iteration

This round should implement:

1. Phase 1 contract hardening
2. Phase 2 hidden locate consistency
3. the smallest Phase 3 UX note for guide self-source filtering
4. a bounded Phase 3.5 LLM reranker for ambiguous cross-paper refs only

It should also add or tighten:

1. backend unit coverage for exact locate payload generation
2. refs-panel E2E coverage for exact payload preservation
3. message-list E2E coverage for hidden locate suppression

## Not Doing In This Iteration

- a large retrieval-ranking rewrite
- a brand-new citation scoring system
- full skill migration for cross-paper refs
- aggressive runtime refactors unrelated to user-visible locate correctness

The immediate priority is to make the current locate experience safer, more precise, and easier to trust.
