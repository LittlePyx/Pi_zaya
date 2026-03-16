# Paper Guide P2 Deep Reading Plan

## 1. Goal

P2 is the phase that turns paper-guide from:

- "I can jump to evidence"

into:

- "I can actually read this paper efficiently inside the product"

The target is not a visual redesign. The target is a better reading workflow.

## 2. Product Outcomes

P2 is successful only if the reader supports all three workflows well:

1. Verify a claim from an answer.
2. Continue reading around that claim without losing context.
3. Revisit personal highlights and move across evidence without manual scroll hunting.

## 3. Scope

### In scope

1. Section outline / table of contents inside the reader.
2. Current-session highlights list.
3. Next / previous evidence navigation.
4. Better current-section awareness and degraded-locate phrasing.

### Out of scope

1. Persistent highlight sync to backend.
2. Collaborative annotations.
3. A full academic PDF reader replacement.
4. A new retrieval/provenance backend redesign.

## 4. Constraints

P2 should build on the current stabilized base:

1. Structured locate remains authoritative.
2. Session highlights stay conversation-scoped and frontend-only.
3. Reader inline and drawer shells must keep the same behavior.
4. New reading UI should live in reader subcomponents, not go back into the shell file.

## 5. Architecture Direction

P2 should be implemented with small reader-focused modules:

1. `useReaderOutline`
   - derives the heading tree
   - tracks active section
   - exposes jump-to-section actions
2. `ReaderOutlinePanel`
   - compact section list
   - active section indicator
   - reveal current section action
3. `useReaderSessionHighlightsIndex`
   - converts current session highlights into panel-ready items
   - resolves section label / preview text
4. `ReaderHighlightsPanel`
   - list of user highlights
   - jump and remove actions
5. `useReaderEvidenceNavigator`
   - tracks current evidence list
   - exposes next/prev evidence actions
6. `ReaderEvidenceNav`
   - small reader chrome control

The reader shell should only compose these modules.

## 6. Execution Order

### Slice C1. Section Outline

Goal:
Add a compact in-reader table of contents.

Implementation:

1. Derive outline entries from `readerBlocks` first, not by scraping headings from rendered DOM.
2. Each outline item should include:
   - `id`
   - `headingPath`
   - `label`
   - `depth`
   - `blockId`
   - `anchorId`
3. Track active section by scroll position in the reader content area.
4. Clicking an outline item should use the same target resolution path as locate, not a separate scroll heuristic.
5. Keep the outline collapsed by default on narrow widths and visible by default on desktop split view.

Acceptance:

1. Long papers can be navigated by section without using answer-side locate chips.
2. The active section updates while scrolling.
3. Section jump lands consistently on the intended heading block.

Primary files:

- `web/src/components/chat/reader/useReaderOutline.ts`
- `web/src/components/chat/reader/ReaderOutlinePanel.tsx`
- `web/src/components/chat/reader/PaperGuideReaderPanel.tsx`

### Slice C2. Session Highlights Workspace

Goal:
Make current-session highlights navigable and useful.

Implementation:

1. Build a highlight index from `sessionHighlights` plus current reader structure.
2. Each list item should include:
   - highlight id
   - short preview text
   - section label
   - range offsets
   - jump action
   - remove action
3. Clicking a highlight jumps to the exact saved range and briefly re-emphasizes it.
4. Removing a highlight from the panel updates both the list and the rendered highlight layer.
5. Keep the storage model session-only; do not add persistence in P2.

Acceptance:

1. A user can revisit highlights without remembering where they were made.
2. Highlight removal is possible from both the selection bubble and the highlights panel.
3. Cross-paragraph highlights appear as one logical list item, not fragmented by blocks.

Primary files:

- `web/src/components/chat/reader/useReaderSessionHighlightsIndex.ts`
- `web/src/components/chat/reader/ReaderHighlightsPanel.tsx`
- `web/src/components/chat/reader/useReaderSessionHighlightLayer.ts`

### Slice C3. Evidence Navigation

Goal:
Let users move through the evidence set for the current answer or locate request.

Implementation:

1. Normalize an evidence list model:
   - `id`
   - `label`
   - `blockId`
   - `anchorId`
   - `anchorKind`
   - `anchorNumber`
   - `headingPath`
2. Prefer provenance-provided order when available.
3. If only current locate alternatives exist, use them as the initial limited evidence list.
4. Add compact `Prev` / `Next` controls to reader chrome.
5. When the evidence changes, keep the same locate contract:
   - switch active evidence item
   - resolve target
   - update locate hint
   - scroll once

Acceptance:

1. Reviewing a long answer with multiple evidence nodes no longer requires repeated manual clicking from chat.
2. Evidence order is stable across repeated navigation.
3. Reader candidate failover and evidence navigation do not conflict.

Primary files:

- `web/src/components/chat/reader/useReaderEvidenceNavigator.ts`
- `web/src/components/chat/reader/ReaderEvidenceNav.tsx`
- `web/src/components/chat/MessageList.tsx`
- `web/src/components/chat/PaperGuideReaderDrawer.tsx`

### Slice C4. Reading Polish

Goal:
Reduce friction in long-form reading after the core panels exist.

Implementation:

1. Add `Reveal current section`.
2. Improve locate hint copy for degraded states.
3. Add a minimal "section / highlights" mode switch inside the reader side panel area.
4. Ensure keyboard focus order is sane for outline, highlight list, and evidence nav.

Acceptance:

1. The reader feels like one coherent reading workspace, not three unrelated widgets.
2. Degraded locate states communicate what happened clearly.

## 7. Data Contract Notes

P2 does not require a major backend change, but two small contract improvements are worth planning:

1. Provide ordered evidence groups per answer segment when available.
2. Expose stable heading metadata that the outline can trust without DOM guessing.

These are incremental enhancements, not blockers for starting C1/C2.

## 8. Testing Plan

Add browser coverage for:

1. outline jump lands on the selected section
2. active section updates on scroll
3. highlight panel jump lands on the saved highlight
4. removing a highlight from the panel clears the rendered highlight
5. next/prev evidence walks a stable ordered list
6. evidence nav still honors strict locate constraints

## 9. Recommended Build Sequence

Build P2 in this order:

1. `C1` outline
2. `C2` highlights panel
3. `C3` evidence navigation
4. `C4` polish

Why this order:

1. Outline gives immediate deep-reading value with low contract risk.
2. Highlights panel builds directly on the session highlight model that already exists.
3. Evidence navigation depends more on finalizing the evidence list model.
4. Polish should happen only after the interaction pieces are real.

## 10. Immediate Next Task

If we start P2 now, the first concrete task should be:

1. create `useReaderOutline`
2. render a minimal `ReaderOutlinePanel`
3. wire section jump + active section tracking

That is the highest-value, lowest-risk entry point into deep reading.
