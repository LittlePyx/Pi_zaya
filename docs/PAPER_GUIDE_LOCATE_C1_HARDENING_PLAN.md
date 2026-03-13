# Paper Guide Locate C1 Hardening Plan

## 1. Current Findings

This round's failures are not caused by a single UI bug. There are two separate structural problems.

### 1.1 Must-locate segments are still missing strict identity

I inspected recent real messages in `chat.sqlite3`:

- `1380`
- `1378`
- `1382`

Observed facts:

1. `provenance.segments` exists and contains `must_locate=true` segments.
2. For those same segments, these fields are empty:
   - `block_id`
   - `anchor_id`
   - `block_ids`
   - and in practice also the newer strict fields are not present in stored data
3. `provenance.block_map` does contain candidate blocks.

Examples:

- Message `1380`
  - three `blockquote_claim` segments are `must_locate=true`
  - but the segments do not carry direct block identity
  - `block_map` contains `blk_29cad7662df5_00050`
  - that block points to:
    - `4. Experiments / 4.1. Experimental Setup`
    - the paragraph that contains:
      - `We noticed that due to the lack of high-quality details...`
      - `So when feeding the reconstructed images into NeRF...`
      - `...making the comparison unfair to our SCINeRF...`

- Message `1382`
  - `seg_003`
  - `claim_type = formula_claim`
  - `must_locate = true`
  - `anchor_kind = equation`
  - `equation_number = 1`
  - but the segment still has no direct block identity
  - `block_map` contains:
    - `blk_29cad7662df5_00025`
    - `anchor_id = eq_00001`
    - `kind = equation`
    - heading path:
      - `3. Method / 3.1. Background on NeRF`

Conclusion:

- The backend is still storing a weak provenance shape for the messages you are currently testing.
- The frontend therefore cannot do strict block locate for those segments.
- It is forced onto a weaker snippet/heading fallback path.

### 1.2 Reader equation anchoring is still structurally weak

I tested the actual markdown -> rehype-katex tree for the SCINeRF paper.

Observed facts:

1. Visible `.katex-display` nodes produced by `rehype-katex` do not retain source `position`.
2. `createReaderBlockResolver()` currently depends on `node.position.start.line/end.line` to bind `data-kb-block-id`.
3. That means equation nodes cannot be reliably matched by the same line-range strategy used for paragraph/blockquote nodes.
4. We already removed paragraph ownership of equation anchors, but that only removes one conflict. It does not solve the missing-position problem.

Conclusion:

- Formula locate is still not reliable because visible formula DOM nodes are not strongly bound to `ReaderDocBlock` identity.

### 1.3 Why first click still often fails

The first-click problem now has a narrower cause than before:

1. The new `reader` retry loop helps only if a valid target identity already exists.
2. For the messages above, many must-locate entries still do not carry strict identity.
3. So the first click is often resolving through weak fallback ranking instead of exact block targeting.

Conclusion:

- First-click instability is now mainly a data-contract problem, not only a timing problem.

## 2. Root Cause Summary

Current failures come from two different layers:

1. `must_locate` segments are allowed to reach the UI without strict identity.
2. Equation anchors in the reader are still not bound from a reliable visible-node strategy.

This is why you are seeing both:

1. quote/blockquote entries that need multiple clicks
2. formula entries that open the paper but do not land on the formula

## 3. What Must Change Next

The next stage should not be more threshold tuning.

It should be a hardening phase with two main tracks:

1. `strict provenance contract hardening`
2. `equation DOM anchor hardening`

## 4. Phase C1: Strict Provenance Contract Hardening

### 4.1 New invariant

If a segment is `must_locate=true`, then it must also satisfy:

1. `primary_block_id` is present
2. `evidence_block_ids` is non-empty
3. `anchor_kind` is present
4. `anchor_text` or `evidence_quote` is present

If any of the above is missing:

1. the segment must be downgraded from `must_locate`
2. the frontend must not render a strict locate entrance for it

### 4.2 Add explicit schema version

Add:

1. `provenance_schema_version`
2. `strict_identity_ready`

Rules:

1. `strict_identity_ready=true` only when all `must_locate` segments satisfy the hard identity invariant.
2. The chat UI may only use strict locate rendering when `strict_identity_ready=true`.
3. Otherwise, the UI should either:
   - hide the strict entrance
   - or show a disabled/debug state, but never pretend it is exact

### 4.3 Add persistence-side assertion

At provenance write time:

1. count `must_locate` segments
2. count how many of them have strict identity
3. store debug counters:
   - `must_locate_count`
   - `strict_identity_count`
   - `strict_identity_ready`
   - `identity_missing_reasons`

This turns the current silent failure into a measurable condition.

### 4.4 Add regression tests

Required tests:

1. `blockquote_claim` with `must_locate=true` must persist `primary_block_id`
2. `formula_claim` with `must_locate=true` must persist `primary_block_id`
3. if a segment lacks strict identity, it must not remain `must_locate=true`

## 5. Phase C2: Equation DOM Anchor Hardening

### 5.1 Current issue

Visible equation nodes do not have stable source `position`.

Therefore:

1. line-range matching is not sufficient
2. allocator fallback is too weak for exact formula locate

### 5.2 Short-term implementation

Add a reader-side visible equation binder:

1. after markdown render
2. collect visible `.katex-display` nodes in DOM order
3. collect `readerBlocks` where `kind = equation` in document order
4. bind `data-kb-block-id / data-kb-anchor-id / data-kb-anchor-kind=equation` at runtime by ordered matching
5. when `equation_number` exists, prefer exact number match before order-only fallback

This is the fastest path to a reliable visible equation target.

### 5.3 Medium-term implementation

Replace the runtime binder with a markdown/rehype plugin path:

1. annotate `math` nodes before `rehype-katex`
2. preserve block identity into rendered display-math DOM

This is cleaner, but slower to land.

### 5.4 Formula success definition

Formula locate is considered successful only when:

1. the reader scrolls to the visible formula block
2. that visible formula block has block-level highlight/focus

It is not enough to match hidden KaTeX annotation text.

## 6. Phase C3: First-Click Contract

### 6.1 Rule

First click must always produce visible feedback.

For quote/blockquote:

1. exact quote highlight if possible
2. otherwise exact block focus

For formula:

1. exact visible equation block focus
2. optional token-level formula highlight later

### 6.2 Rule for weak provenance

If the locate payload is weak:

1. do not silently act like an exact locate
2. do not open reader with exact-language hints
3. surface a clear degraded state instead

The product must distinguish:

1. `exact`
2. `block`
3. `weak`

## 7. Phase C4: Frontend Fallback Containment

Current weak fallback path in `MessageList` is still necessary for old data, but it should not be used for new strict claims.

Rules:

1. If `must_locate=true` and `strict_identity_ready=false`, do not render the strict inline entrance.
2. If `must_locate=true` and strict identity exists, do not fall back to generic snippet ranking.
3. Generic snippet ranking remains only for:
   - legacy messages
   - non-strict optional evidence
4. `strict_identity_ready=false` must not suppress the normal non-strict locate entrance for legacy / degraded data.

## 8. Recommended Execution Order

### Step 1

Fix provenance contract first.

Target:

1. new messages must store strict identity for `must_locate` segments
2. weak must-locate data must stop reaching the UI as if it were exact

### Step 2

Implement reader-side equation DOM binder.

Target:

1. formula clicks land on visible formula blocks
2. no more hidden-annotation pseudo-success

### Step 3

Add first-click result state.

Target:

1. every locate action reports `exact / block / weak`
2. future debugging stops depending on screenshot guessing

### Step 4

Only after the above, continue refining chat-side inline entrance placement.

## 9. Acceptance Criteria

### 9.1 Provenance

For a newly generated must-locate message:

1. every must-locate segment has `primary_block_id`
2. every must-locate segment has non-empty `evidence_block_ids`
3. formula must-locate segments also have `anchor_kind=equation`

### 9.2 Quote / blockquote

1. first click lands on the exact source block
2. first click shows visible block or phrase highlight

### 9.3 Formula

1. first click lands on the visible target equation block
2. no jump that only opens the document without moving to the formula
3. if the answer includes formula explanation text extracted from the paper, that explanation unit also has a visible locate entrance
4. formula and explanation may share one grouped locate target

## 10. Immediate Next Coding Task

The next implementation task should be:

1. backend provenance hardening for `must_locate`
2. reader visible-equation binder

Not:

1. more threshold tuning
2. more generic snippet ranking rules
3. more UI micro-adjustments

## 11. Progress Update (2026-03-11)

### 11.1 Completed: strict provenance contract hardening

Implemented:

1. `must_locate=true` segments are now required to carry:
   - `primary_block_id`
   - non-empty `evidence_block_ids`
   - `anchor_kind`
   - `anchor_text` or `evidence_quote`
2. segments that fail this invariant are downgraded before persistence
3. provenance root now stores:
   - `provenance_schema_version = 3`
   - `strict_identity_ready`
   - `must_locate_candidate_count`
   - `must_locate_count`
   - `strict_identity_count`
   - `identity_missing_reasons`
   - `identity_missing_segments`

Verified:

1. `pytest tests/unit/test_task_runtime_provenance.py -q`
2. regression coverage now includes:
   - `blockquote_claim` strict identity persistence
   - `formula_claim` strict identity persistence
   - downgrade of invalid `must_locate` segments

### 11.2 Completed: reader visible-equation binder

Implemented:

1. reader-side runtime binding for visible `.katex-display`
2. equation blocks are matched in document order, with equation-number preference when available
3. visible equation DOM now receives:
   - `data-kb-block-id`
   - `data-kb-anchor-id`
   - `data-kb-anchor-kind=equation`
   - `data-kb-anchor-number`
4. direct strict locate now prefers the visible equation node before generic DOM lookup

Verified:

1. `npm run build` in `web/`

### 11.21 Progress landed (2026-03-11, live-browser formula locate replay for message 1416)

This phase used a real headless browser replay against the latest SCINeRF formula turn (`1416`) instead of relying only on static reasoning.

Runtime findings:

1. provenance was still correct, so the failure was not a backend identity issue
2. in the live browser path, reader display math could be statically mis-bound to a neighboring paragraph block
3. locate then updated React state too early, which could invalidate the just-resolved DOM target before the visual scroll/highlight completed

Implemented:

1. `web/src/components/chat/MarkdownRenderer.tsx`
   - reader display math no longer uses static line-based anchor assignment
   - equations are left to the runtime visible-anchor binder only
2. `web/src/components/chat/PaperGuideReaderDrawer.tsx`
   - each locate run now re-binds visible equation anchors against the current DOM before direct-target resolution
   - locate hint updates are deferred so they do not preempt the actual scroll

Verified in browser replay:

1. latest real message `1416`
2. reader `scrollTop` changed from `0` to `4596`
3. the drawer landed on `3.1 Background on NeRF` with equation `(1)` visible

### 11.22 Progress landed (2026-03-11, sticky highlight persistence after locate)

After 11.21, one remaining runtime gap was that the reader could land on the correct equation region but lose the visible highlight after subsequent rerenders.

Implemented:

1. `web/src/components/chat/PaperGuideReaderDrawer.tsx`
   - stores the last successful locate result as sticky highlight metadata
   - re-applies block highlight after reader rerenders
   - equations now get a fallback re-resolution path using equation number + formula text overlap, instead of relying only on block id
2. `web/src/styles/index.css`
   - strengthened `.kb-reader-focus`
   - added persistent styling for `.kb-reader-inline-hit`

Verified in browser replay:

1. latest real message `1416`
2. after waiting 5.5 seconds post-locate:
   - `scrollTop = 4596`
   - `focusCount = 1`
3. highlight now stays visible after the jump

### 11.20 Progress landed (2026-03-11, explicit reader scroll + equation-note mojibake cleanup)

This phase closed the two remaining issues observed on the latest real SCINeRF formula turn (`1414`):

1. the equation-source note under display math was still produced from a mojibake legacy string path, so the user saw a garbled filename line even though the model output itself was clean
2. reader locate still had a browser/runtime risk where `scrollIntoView` alone could leave the drawer open without an actual container scroll

Implemented:

1. `ui/refs_renderer.py`
   - equation source notes now emit a clean canonical line:
     `（式(n) 对应命中的库内文献：filename.pdf）`
   - removed the legacy broken `Open/Page` note producer
2. `api/chat_render.py`
   - added pdf-label sanitation for equation-source notes
   - legacy/mojibake note lines are normalized back to the real `*.pdf` basename during render
3. `web/src/components/chat/PaperGuideReaderDrawer.tsx`
   - equation focus prefers `.katex-display`
   - locate success now explicitly scrolls the reader container by computed `scrollTop`, then keeps native `scrollIntoView` as a secondary assist
   - cleaned remaining user-visible mojibake strings inside the reader drawer

Verified:

1. `pytest tests/unit/test_chat_render_reference_notes.py -q`
2. `pytest tests/unit/test_task_runtime_provenance.py -q`
3. `npm run build` in `web/`

### 11.21 Progress landed (2026-03-11, live-browser formula locate replay for message 1416)

This phase used a real headless browser replay against the latest SCINeRF formula turn (`1416`) instead of relying on static code inspection.

Runtime findings:

1. provenance was still correct, so the failure was not a backend identity issue
2. in the live browser path, reader display math nodes were being statically mis-bound to neighboring paragraph blocks
3. locate then updated React state (`locateHint`) before scroll finished, which could invalidate the just-resolved DOM target and leave the drawer at the document start

Implemented:

1. `web/src/components/chat/MarkdownRenderer.tsx`
   - reader display math no longer uses static line-based anchor assignment
   - equations are left for the runtime visible-anchor binder only
2. `web/src/components/chat/PaperGuideReaderDrawer.tsx`
   - each locate run now re-binds visible equation anchors against the current DOM before direct-target resolution
   - locate hint state updates are deferred so they do not preempt the actual scroll

Verified in browser replay:

1. latest real message `1416`
2. reader `scrollTop` changed from `0` to `4596`
3. the drawer landed on `3.1 Background on NeRF` with equation `(1)` visible

### 11.21 Hotfix landed (2026-03-11, equation binder no longer clears native equation anchors)

Investigation on the newest real message `1412` narrowed the failure further:

1. provenance is intact for both:
   - `blk_29cad7662df5_00025 / eq_00001`
   - `blk_29cad7662df5_00026 / p_00015`
2. SSR rendering of the same SCINeRF markdown with `MarkdownRenderer variant=reader` also confirmed those `data-kb-*` anchors exist in the DOM before reader runtime logic runs
3. the actual runtime regression was inside `bindVisibleEquationAnchors()`:
   - old cleanup removed attributes from every `[data-kb-anchor-kind=\"equation\"]`
   - that also erased the native markdown-rendered equation wrapper anchors
   - during the rebinding window, strict direct locate temporarily had no equation target to resolve

Implemented:

1. visible-equation cleanup now only removes binder-owned nodes marked with `data-kb-visible-equation-bound=\"1\"`
2. native reader equation anchors are preserved
3. direct locate can still hit the original equation wrapper even if `.katex-display` rebinding lags

Verified:

1. `npm run build` in `web/`
2. SSR check on the SCINeRF reader render still shows the target `data-kb-block-id / data-kb-anchor-id` in the output DOM

### 11.20 Hotfix landed (2026-03-11, strict-to-fuzzy locate fallback)

After 11.19, a remaining usability issue was still reported: some locate clicks opened the reader but did not jump/highlight.

Implemented:

1. strict locate still attempts exact identity matching first (with retry window)
2. when strict exact resolution expires, it now degrades to fuzzy locate instead of hard-stopping
3. when fuzzy still cannot resolve a target, a final readable-block fallback is used so the reader always scrolls somewhere meaningful

Impact:

1. prevents “open-only, no jump” dead-end behavior
2. keeps strict-exact as primary path but restores user-visible motion/feedback in degraded cases
3. allows subsequent iterations to improve precision without regressing baseline usability

Verified:

1. `npm run build` in `web/`

### 11.3 Remaining C1 follow-up

Still pending from the broader hardening track:

1. explicit locate result state: `exact / block / weak`
2. additional fallback containment around weak legacy payloads
3. render-coverage contract for every direct evidence sentence / formula bundle
4. only after the above, more inline placement refinement

### 11.4 Follow-up fix landed (2026-03-11)

A regression appeared after the first C1 pass: locate entrances disappeared entirely for structured messages whose `strict_identity_ready=false`.

Fixed rule:

1. strict-ready messages keep strict-only behavior
2. not-ready / degraded structured messages hide strict entrances
3. but they still retain normal fallback locate entrances instead of returning no candidate at all
4. locate-button dedupe must not collapse different evidence snippets merely because they map to the same `block_id`

### 11.5 Follow-up fix landed (2026-03-11, same-block dedupe)

Observed regression:

1. one paragraph block may legitimately support multiple quote / blockquote claims
2. the previous chat-side dedupe key used only `block_id` / `anchor_id`
3. that suppressed later entrances in the same message even when the user-visible evidence snippets were different

Fixed rule:

1. dedupe now uses `target block/anchor + rendered snippet`
2. same block may therefore expose multiple entrances when the evidence snippets are distinct
3. exact duplicate snippet entrances are still suppressed

### 11.6 Follow-up fix landed (2026-03-11, direct-evidence priority over optional cap)

Observed regression:

1. legacy / degraded messages still use the non-strict fallback locate path
2. that path had a global inline entrance cap of `5`
3. the cap was checked before direct provenance matching
4. so earlier fuzzy / summary snippets could consume the full budget
5. later direct evidence quotes in the same message then lost their entrance even though the provenance already contained exact direct block targets

Real repro:

1. message `1394` contains three obvious English evidence quotes in the `依据` section
2. those quotes already map to direct provenance on `blk_29cad7662df5_00085`
3. once optional locate candidates compete earlier in the message, the direct evidence lines can be suppressed by the shared cap

Fixed rule:

1. direct provenance-backed locate entrances are now resolved before the optional cap check
2. the `5`-button cap now applies only to optional fuzzy entrances
3. direct evidence quotes / formulas no longer lose their entrance just because earlier summary bullets already used the optional budget

Verified:

1. local SSR replay of message `1394`
2. with competing optional locate candidates present, the rendered inline entrance count is `8`
3. the three English evidence quotes in `依据` still retain their locate entrances

### 11.7 Next phase: required-coverage contract

The current C1 work hardened strict identity and restored missing entrances, but it still does not fully encode your newer product goal:

1. every direct sentence extracted from the paper must expose a locate entrance
2. every extracted formula must expose a locate entrance
3. every extracted formula-explanation sentence must expose a locate entrance
4. formula and explanation may share one grouped locate target

This creates a new phase boundary:

1. C1 solved `can strict locate exist reliably`
2. the next phase must solve `does every required evidence unit actually render an entrance`

#### 11.7.1 Contract changes needed next

Add a render-oriented policy layer on top of `must_locate`:

1. `locate_policy = required | optional | hidden`
2. `required` means:
   - never suppressed by optional caps
   - never downgraded by generic snippet competition
3. add `claim_group_id`
4. add `claim_group_kind`
   - `formula_bundle`
   - `quote_bundle`
5. add `equation_explanation_claim` for formula-adjacent explanation text

#### 11.7.2 Rendering rules needed next

Chat-side rendering should split into two lanes:

1. `required` entrances:
   - render all of them
   - dedupe only exact duplicates on the same visible unit
2. `optional` entrances:
   - remain capped
   - remain best-effort only

For formula bundles:

1. the equation block must have an entrance
2. the explanation sentence must also have an entrance if it is displayed separately
3. both may open the same grouped target in the reader

#### 11.7.3 Reader rules needed next

Grouped formula locate should become explicit:

1. focus the visible equation block first
2. then highlight the adjacent explanation block/sentence if available
3. clicking the explanation sentence must not fall back to a generic paragraph target

#### 11.7.4 Test gap and next test plan

Current tests are still weighted toward provenance persistence. They do not yet prove render coverage.

Add next:

1. backend unit test for `equation_explanation_claim`
2. backend unit test for stable `claim_group_id` on `formula_bundle`
3. frontend SSR regression:
   - real message `1394` must render entrances for all three evidence quotes
4. formula replay regression:
   - real formula message such as `1382` must render an entrance for the formula block
   - if the answer also includes formula explanation text, the explanation unit must also expose an entrance

#### 11.7.5 Acceptance for the next phase

Required-coverage acceptance should be strict:

1. direct quote coverage = 100%
2. direct blockquote coverage = 100%
3. direct formula coverage = 100%
4. direct formula-explanation coverage = 100%
5. optional entrances may still be capped, but they must never suppress required entrances

### 11.8 Progress landed (2026-03-11, required/group contract phase 1)

Implemented in this phase:

1. provenance schema advanced to `3`
2. segments now persist:
   - `locate_policy`
   - `claim_group_id`
   - `claim_group_kind`
3. formula-adjacent explanation text can now be upgraded to `equation_explanation_claim`
4. `formula_claim` and `equation_explanation_claim` share one `formula_bundle`
5. strict structured rendering now understands `equation_explanation_claim` as a sentence-level entrance that may still open the grouped formula target

Verified:

1. backend regression:
   - `pytest tests/unit/test_task_runtime_provenance.py -q`
2. new coverage in unit tests:
   - `equation_explanation_claim` becomes `required`
   - stable shared `claim_group_id` for the `formula_bundle`
3. frontend compile verification:
   - `npm run build` in `web/`

Still pending:

1. automated frontend render replay for real message `1382`
2. reader-side grouped highlight that visibly marks both formula and adjacent explanation region

### 11.9 Progress landed (2026-03-11, summary-claim rerank fix for wrong locate target)

New real-message repro:

1. latest paper-guide message `1398`
2. the first innovation summary bullet was incorrectly mapped to:
   - `blk_707c81a1c1ff_00028`
   - `As the rendering primitive for 3DGS, a 3D Gaussian is defined as:`
3. this was a provenance-selection bug, not a reader scrolling bug

Root cause:

1. long `answer_hits` contribution text was entering the pool as one coarse snippet
2. summary-claim segment matching then ran on Chinese summary text against English source blocks
3. true contribution list items could fall below the raw `match_source_blocks` floor
4. definition-like method blocks survived because of token overlap on `3DGS / Gaussian`
5. an additional generic-heading post-check could still clear a valid contribution block even after rerank

Fix landed:

1. pool building now expands long hit text into bullet / sentence-level snippet hints
2. summary/result claims now use a lower raw score floor before semantic rerank
3. semantic rerank now:
   - boosts contribution/result blocks such as `first`, `introducing`, `experiments demonstrate`
   - penalizes definition-like blocks such as `defined as`, `denotes`, `parameterized as`
4. the generic-heading guard now allows summary claims through when semantic evidence is strong enough

Verified:

1. real replay of message `1398`
2. first innovation bullet now maps to:
   - `blk_707c81a1c1ff_00021`
   - `The proposed SCIGS is the first to recover explicit 3D representations ...`
3. dedicated backend regression added:
   - innovation-summary Chinese claim must prefer the contribution list item over the `defined as` method block
4. `pytest tests/unit/test_task_runtime_provenance.py -q`

### 11.10 Plan adjustment (2026-03-11, inline-formula entrance + first-click locate)

Latest real-message replay `1402` shows that the next gap is no longer generic strict identity. It is a narrower product gap:

1. the answer can contain a provenance-backed formula, but the inline formula token itself still has no entrance
2. formula locate can still require repeated clicks because reader readiness and equation binding race each other
3. generic knowledge / non-retrieved formula text can still be upgraded into `formula_claim required`

#### 11.10.1 Rendering changes needed next

`MarkdownRenderer` needs a new coverage pass for inline math:

1. add an `inline_math` locate token class
2. detect:
   - `$...$` inline math
   - KaTeX inline math spans
   - explicit equation references such as `公式(1)` / `Eq. (1)`
3. if provenance marks the formula as `required`, the formula token itself must expose an entrance
4. paragraph-level fallback is no longer sufficient for long inline formulas

#### 11.10.2 Reader lifecycle changes needed next

The current retry loop should be replaced by an explicit readiness chain:

1. drawer opened
2. markdown mounted
3. reader blocks loaded
4. visible equation anchors bound
5. strict locate executed
6. highlight acknowledged

Implementation target:

1. pass a fresh `locateNonce / requestId` for every click
2. do not allow strict formula locate to fail before equation binding has settled
3. accept only first-click success for formula locate

#### 11.10.3 Backend guard needed next

Required formula entrances must exclude generic or external knowledge text:

1. segments explicitly marked as generic knowledge / non-retrieved content should downgrade to `hidden` or `optional`
2. `formula_claim required` must keep aligned:
   - `primary_block_id`
   - `evidence_block_ids`
   - `equation_number` when present
   - `anchor_text` or `evidence_quote`
3. token overlap on variable names alone must not create a required formula locate

#### 11.10.4 Output-shape guidance to encode in the plan

Long formulas should prefer display-math rendering, but only as a support measure:

1. if the answer is explaining `公式(n)` or the formula is visually long, render it as display math by default
2. keep the explanation sentence on its own line / sentence when possible
3. keep short variable expressions inline
4. do not use display-math output as a substitute for inline-formula entrance support

#### 11.10.5 Next verification set

Add a real-message regression set centered on `1402`:

1. render replay:
   - the long inline formula in the answer must expose its own entrance
2. interaction replay:
   - one click lands on visible `eq_00001`
   - no repeated clicks needed to obtain block highlight
3. backend replay:
   - the generic-knowledge segment in `1402` must not persist as `formula_claim required`
4. output replay:
   - long-formula answers prefer display math plus a separated explanation sentence

### 11.11 Progress landed (2026-03-11, inline-formula entrance coverage phase 1)

This phase only implemented the first rendering-side slice from `11.10.1`.

Implemented:

1. `MarkdownRenderer` now has an `inline_math` locate token class
2. raw `$...$` inline formulas are detected as formula-level entrance candidates
3. visible KaTeX inline-math nodes are also bound as entrance targets
4. `equation_ref` and `inline_math` entrances now flow through strict rendering as `equation`-typed locate units
5. when a paragraph or list item contains a formula token entrance, chat rendering no longer has to fall back to a paragraph-level entrance for that formula

Verified:

1. `npm run build` in `web/`
2. local temporary SSR replay using the real long-formula snippet from message `1402`
   - the inline formula node emits a `kb-md-locate-inline-btn`
   - the surrounding list item no longer degrades to a whole-line fallback entrance for that formula

Still pending from `11.10`:

1. reader readiness / first-click locate lifecycle hardening
2. backend downgrade for generic-knowledge / non-retrieved formula segments
3. output-shape guidance that prefers display math for long formulas

### 11.12 Progress landed (2026-03-11, reader first-click locate phase 1)

This phase implemented the first lifecycle hardening slice from `11.10.2`.

Implemented:

1. `ChatPage` now assigns a fresh `locateRequestId` for every reader-open action
2. `PaperGuideReaderDrawer` now treats each click as a distinct locate request instead of reusing the previous effect state
3. formula locate waits for visible equation-anchor binding readiness before running strict locate
4. a new locate request clears the previous focus/highlight state before the next locate run
5. the drawer now exposes a short binding-state hint while equation anchors are still being attached

Verified:

1. `npm run build` in `web/`
2. request-id propagation is now explicit in the reader payload path:
   - `ChatPage -> ReaderOpenPayload -> PaperGuideReaderDrawer`

Still pending from `11.10`:

1. backend downgrade for generic-knowledge / non-retrieved formula segments
2. output-shape guidance that prefers display math for long formulas

### 11.13 Progress landed (2026-03-11, generic/non-source formula downgrade)

This phase implemented the backend guard from `11.10.3` and the corresponding hidden-policy consumption on the chat side.

Implemented:

1. `kb/task_runtime.py` now detects explicit non-source segments such as:
   - `通用知识`
   - `非检索片段内容`
   - `未出现在本文检索片段中`
   - `generic knowledge / non-retrieved`
2. those segments no longer upgrade into strict formula locate
3. if a legacy path still labels them as `formula_claim`, coverage hardening now forces:
   - `must_locate = false`
   - `locate_policy = hidden`
   - no `formula_bundle`
4. `MessageList` now respects `locate_policy = hidden` in both:
   - structured locate-entry building
   - provenance direct fallback candidate collection

Verified:

1. `pytest tests/unit/test_task_runtime_provenance.py -q`
   - now `17 passed`
2. `npm run build` in `web/`
3. replay of the old stored segment 8 from real message `1402`
   - previous state: `formula_claim`, `must_locate=true`
   - new contract output: `must_locate=false`, `locate_policy=hidden`

Still pending from `11.10`:

1. output-shape guidance that prefers display math for long formulas

### 11.14 Progress landed (2026-03-11, long-formula output-shape guidance)

The remaining generation-side guidance from `11.10.4` is now encoded in the paper-guide system prompt.

Implemented:

1. the base formula-format rule now explicitly says:
   - if the answer is explaining `公式(n)` / `Eq. (n)`, prefer display math
   - if the formula contains structures such as `\\frac / \\sum / \\int / \\mathcal / \\mathbf`, prefer display math
   - keep long formulas on their own line instead of embedding them inside prose
   - keep `where` / variable-definition explanation text on a separate sentence or line when possible
2. the paper-guide grounding rule now also says:
   - retrieved long equations and numbered equations should render as display math
   - do not compress a long equation plus its explanation into one mixed prose sentence

Verified:

1. `pytest tests/unit/test_task_runtime_provenance.py -q`

Residual risk:

1. this is prompt guidance, not a hard renderer transform
2. real-message manual replay is still needed for long-formula answers such as `1402` and formula-bundle answers such as `1382`
### 11.15 Progress landed (2026-03-11, equation-anchor retrieval no longer confuses References `[1]` with `Eq. (1)`)

The latest real failure sample `1405 -> 1406` showed a retrieval-layer bug, not a missing-formula problem:

1. the SCINeRF markdown already contains:
   - `C(\\mathbf{r}) = ... \\tag{1}`
   - the following `where t_n and t_f ...` explanation
2. the old equation-anchor rule still treated square-bracketed citation indices like `[1]` as valid equation-number matches
3. that allowed `References [1]` to be promoted as an anchor-focused snippet for an `Eq. (1)` question
4. generation then incorrectly concluded that only headings / references were retrieved

Implemented:

1. `kb/retrieval_engine.py`
   - equation anchor matching no longer accepts `[1]` as an equation-number hit
   - only true formula forms remain valid:
     - `Eq. 1`
     - `Equation (1)`
     - `公式(1)`
     - `式(1)`
     - `\\tag{1}`
2. the same file now filters reference-like snippets from answer packs when the question is not asking about references:
   - snippets under `References / Bibliography`
   - multi-item citation-list chunks such as `[1] ... [5] ...`
3. retrieval regression coverage was added in:
   - `tests/unit/test_retrieval_engine_doc_anchor.py`
   - it asserts that a doc containing both `Eq. (1)` and `References [1]` must still surface the formula body plus the `where` clause as the answer-facing snippet

Verified:

1. `pytest tests/unit/test_retrieval_engine_doc_anchor.py -q`
2. `pytest tests/unit/test_task_runtime_provenance.py -q`
3. equivalent replay on the real SCINeRF markdown now returns:
   - primary snippet = `\\tag{1}` formula body
   - answer-facing snippets = equation body + where explanation + the method paragraph
   - no reference-list snippet in the answer pack

Acceptance impact:

1. numbered formula questions must retrieve the actual equation block, not citation indices
2. answer context must prioritize `formula body + where clause`
3. reference snippets may only enter the answer pack when the user explicitly asks about citations / references

### 11.16 Progress landed (2026-03-11, inline-formula entrances disabled and formula-bundle entrances collapsed)

The newest real answer `1408` exposed a different quality problem:

1. too many entrances were shown on inline variable-sized formulas such as `$C(r)$`, `$t_n,t_f$`, `$T(t)$`
2. many of those entrances still pointed to the same visible `Eq. (1)` block
3. provenance already showed the structural cause:
   - multiple required segments (`seg_004 / seg_008 / seg_016`) shared the same `formula_bundle:blk_29cad7662df5_00025`

Implemented:

1. `web/src/components/chat/MarkdownRenderer.tsx`
   - disabled raw `$...$` inline-math locate tokens
   - disabled locate buttons on visible inline KaTeX nodes
   - chat-side entrances now remain focused on:
     - numbered equation references
     - display/block formulas
     - quote / blockquote / figure evidence units
2. `web/src/components/chat/MessageList.tsx`
   - structured provenance locate entries now collapse duplicate entries within the same `formula_bundle`
   - the representative entry prefers a true `formula_claim` over explanation-only siblings
   - strict inline rendering now allows only one entrance per render slot / formula bundle
3. `kb/task_runtime.py`
   - added a non-source scope propagation rule
   - once a `Supplementary note (generic knowledge / non-retrieved content)` marker appears, adjacent follow-on formula segments inside that scope are also forced to `hidden`
   - this prevents generic discrete NeRF formulas from leaking back into required locate

Verified:

1. `pytest tests/unit/test_task_runtime_provenance.py -q`
   - now `18 passed`
2. `pytest tests/unit/test_retrieval_engine_doc_anchor.py -q`
   - `5 passed`
3. `npm run build` in `web/`
4. dev services restarted successfully, backend health check returns `200`

Acceptance impact:

1. inline variable-sized formulas should no longer expose standalone entrances
2. a single formula bundle should expose one representative entrance in one rendered answer view
3. generic/non-source supplementary formulas should remain outside strict locate

### 11.17 Progress landed (2026-03-11, paper-guide no longer forces structured output)

The latest regression report was about answer shape rather than locate correctness: even with structured output disabled in user prefs, `paper_guide` still re-enabled the `Conclusion / Evidence / Limits / Next Steps` template.

Implemented:

1. `kb/task_runtime.py`
   - added `_answer_contract_enabled(task)` so `paper_guide_mode` now respects the incoming `answer_contract_v1` flag instead of overriding it
   - extracted `_build_paper_guide_grounding_rules(answer_contract_v1=...)`
   - when contract is disabled, paper-guide prompt guidance keeps formula/evidence grounding but drops:
     - `3-4 sections`
     - `Evidence / Limits` fixed-section wording
2. the same file now gates `_enhance_kb_miss_fallback(...)` on `contract_enabled`
   - contract enabled: fallback may still append structured `Next Steps`
   - contract disabled: no-hit answers keep their natural shape instead of being post-processed back into a template
3. regression coverage was added in `tests/unit/test_task_runtime_answer_contract.py`
   - paper-guide no longer forces `answer_contract_v1=true`
   - paper-guide grounding rules differ correctly between structured and non-structured mode
   - kb-miss fallback does not inject `Next Steps` when contract is disabled

Verified:

1. `pytest tests/unit/test_task_runtime_answer_contract.py -q`
   - `25 passed`
2. `pytest tests/unit/test_task_runtime_provenance.py -q`
   - `18 passed`

Acceptance impact:

1. `paper_guide` still requires evidence-grounded answers
2. fixed structured sections are now opt-in via `answer_contract_v1`
3. disabling structured output in user prefs must remain effective in paper-guide mode

### 11.18 Progress landed (2026-03-11, first-click reader locate hardening)

The next user-visible quality gap was not wrong provenance but unstable reader execution timing: some entrances required multiple clicks before the drawer finally scrolled to the highlighted block.

Implemented:

1. `web/src/components/chat/PaperGuideReaderDrawer.tsx`
   - added `drawerReady`, sourced from `Drawer.afterOpenChange`
   - bind / locate now wait for the drawer to be fully open instead of racing the opening animation
2. equation binding flow now uses:
   - deadline-based retries instead of a tiny fixed RAF budget
   - `MutationObserver` on the reader subtree
   - a short stability check before marking `equationBindingReady`
3. locate flow now uses:
   - deadline-based retries
   - `MutationObserver` on anchor/class changes
   - double-RAF scrolling after a successful focus/highlight resolution
   - success cleanup that stops retry/observer loops once a locate is acknowledged

Verified:

1. `npm run build` in `web/`

Acceptance impact:

1. strict locate should normally succeed on the first click
2. equation locate should not require repeated clicks just to wait for visible-anchor binding
3. scrolling/highlighting should happen after drawer-open completion, not during the transition race

### 11.19 Hotfix landed (2026-03-11, locate-opened-only regression)

A regression was reported right after 11.18: clicking locate would only open the reader file without jumping/highlighting.

Root cause and hotfix:

1. `Drawer.afterOpenChange` was used as a hard gate for locate execution.
2. In environments where that callback is delayed/missed, `drawerReady` stayed false and locate effects never ran.
3. Added a fallback readiness timer (`open` + 240ms) so locate can still execute even if `afterOpenChange` is not emitted.
4. Added explicit `finishLocate()` on terminal strict-failure branches to stop observer/retry loops cleanly.

Verified:

1. `npm run build` in `web/`
### 11.20 In progress: `1418` figure jump + quote-entrance hardening

Real browser replay was run against message `1418`.

Findings:

1. Stored provenance bound the `figure_claim` to method paragraph `blk_29cad7662df5_00022`, not the actual figure/caption area.
2. The first displayed direct quote had no entrance because strict rendering matched against the whole labeled blockquote instead of the quoted sentence itself.
3. The second displayed quote still has no entrance because no direct provenance segment was persisted for that quote.

Implemented:

1. `kb/task_runtime.py`
   - direct `figure_claim` segments are rebound toward figure/caption blocks during coverage hardening
   - added regression test for figure-block rebinding
2. `api/chat_render.py`
   - existing messages now re-run provenance hardening on fetch so old messages can pick up figure rebinding
3. `web/src/components/chat/MarkdownRenderer.tsx`
   - blockquote locate prefers the actual quoted span
4. `web/src/components/chat/MessageList.tsx`
   - structured render segmentation now merges consecutive blockquote lines
5. `web/src/styles/index.css`
   - figure focus styling now makes the visible image wrapper highlightable

Verified:

1. `pytest tests/unit/test_task_runtime_provenance.py -q`
2. `pytest tests/unit/test_chat_render_reference_notes.py -q`
3. `npm run build` in `web/`
4. Browser replay confirms the first quote in `1418` now renders a locate entrance.

Open items:

1. The second displayed quote in `1418` still lacks an entrance because provenance coverage is missing, not because of frontend rendering.
2. Old-message `Figure 1` mentions still need a stricter figure-entry fallback when slot alignment fails even though required figure provenance exists.

### 11.21 Landed: excerpted direct-quote rebinding for `1418` and old-message full-block normalization

This phase closed the remaining backend gap behind the wrong/missing quote entrances in message `1418`.

Root cause:

1. The second displayed blockquote already had a stored direct segment (`seg_004`), but it was bound to the wrong block:
   - wrong target: `blk_29cad7662df5_00037`
   - actual source block: conclusion block `blk_29cad7662df5_00085`
2. Old-message normalization only hardened against the stored `block_map`.
   - If the true target block was not already present in that map, fetch-time normalization could not recover it.

Implemented:

1. `kb/task_runtime.py`
   - added excerpt-aware rebinding for `quote_claim / blockquote_claim`
   - handles truncated direct quotes with `[...]` / `...`
   - rewrites strict identity fields to the true source block when a stronger quote match is found
2. `api/chat_render.py`
   - fetch-time provenance enrichment now loads full source blocks from `md_path`
   - hardening runs against the full block lookup, not only the stored `block_map`
   - any newly referenced blocks are merged back into `block_map` for frontend strict locate

Verified:

1. `pytest tests/unit/test_task_runtime_provenance.py -q`
2. `pytest tests/unit/test_chat_render_reference_notes.py -q`
3. Real-message local replay for `1418` now shows:
   - `seg_004 -> blk_29cad7662df5_00085`
   - `block_map` contains `blk_29cad7662df5_00085`

Impact:

1. Existing messages can now recover the correct source block on refresh, without regeneration.
2. The second displayed quote in `1418` is no longer blocked by missing backend provenance identity.

### 11.22 Landed: strict inline locate no longer mutates dedupe state during render (`1420`)

This phase fixed the latest real-message regression where quote/figure entries were resolved correctly but disappeared from the committed DOM.

Root cause:

1. `MessageList.tsx` used `shownStructuredInlineKeys` inside `canLocateSnippet()`.
2. Browser replay against real message `1420` showed:
   - `resolveStructuredInlineResolution()` already returned the correct entries for `Figure 1` and all three direct blockquotes
   - first render pass marked them as shown
   - later render/commit pass hit `blocked-shown`, so the final DOM lost the buttons
3. This was a render-time side effect bug, not a provenance-contract gap.

Implemented:

1. `web/src/components/chat/MessageList.tsx`
   - removed strict inline dedupe side effects from render-time `canLocateSnippet()`
   - strict eligibility is now a pure check
   - `blockquote_claim / quote_claim` can only attach to blockquote containers
   - `figure_claim` can only attach to explicit `Figure n / 图 n` refs, not caption/file-name fallbacks

Verified:

1. `npm run build` in `web/`
2. `pytest tests/unit/test_task_runtime_provenance.py -q`
3. `pytest tests/unit/test_chat_render_reference_notes.py -q`
4. Real browser replay on `1420` confirms:
   - all three direct blockquotes now render entrances
   - the latest `Figure 1` text renders a figure entrance
   - clicking each latest quote lands on the correct reader block, with inline hit text present for the quoted sentence
   - clicking the latest `Figure 1` lands on figure block `blk_29cad7662df5_00003 / fg_00001`
### 11.23 Landed: formula bundle entrances now honor `primary / secondary / hidden`

This phase implements the generalized formula-entry policy required by the latest real-message failures.

Implemented:

1. `kb/task_runtime.py`
   - schema upgraded to `4`
   - formula segments now emit:
     - `formula_origin = source | explanation | derived`
     - `locate_surface_policy = primary | secondary | hidden`
     - `related_block_ids`
   - only one formula surface per `formula_bundle` remains visible as the primary strict entrance
   - duplicate or model-rewritten formulas in the same bundle are downgraded to hidden
   - explanation claims now keep their own prose block as the primary target when that prose block exists
2. `web/src/components/chat/MessageList.tsx`
   - strict structured entries now consume the new formula bundle fields
   - hidden / derived formula surfaces no longer render locate buttons
3. `web/src/components/chat/PaperGuideReaderDrawer.tsx`
   - locate payload now carries `relatedBlockIds`
   - reader keeps related bundle blocks highlighted as secondary context after landing
4. `web/src/styles/index.css`
   - added a secondary reader highlight style

Verified:

1. `pytest tests/unit/test_task_runtime_provenance.py -q`
2. `pytest tests/unit/test_chat_render_reference_notes.py -q`
3. `npm run build` in `web/`
4. offline replay confirms:
   - `1416`: only the source Eq. (1) surface stays visible
   - `1420`: synthesized formula surfaces are downgraded instead of surfacing misleading entrances

Open follow-up:

1. browser-level replay on refreshed live messages
2. reducing `llm_refined` calls by pushing formula / figure / direct-quote mapping onto deterministic bundle-level resolution

### 11.24 Landed: duplicate quote bundles collapsed and formula entrances limited to display math

This phase addresses the latest live sample where repeated entrances still appeared even after formula-bundle hardening.

Implemented:

1. `kb/task_runtime.py`
   - duplicate direct quote/blockquote claims inside the same `quote_bundle` are now collapsed
   - only one identical quote surface remains `required + primary`
2. `web/src/components/chat/MessageList.tsx`
   - `formula_claim` can no longer bind to paragraph/list-item render slots
   - strict equation locate no longer uses generic fallback for unmatched display-math blocks
3. `web/src/components/chat/MarkdownRenderer.tsx`
   - removed inline `equation_ref` locate tokens
   - formula entrances are now display-math-only

Verified:

1. `pytest tests/unit/test_task_runtime_provenance.py -q`
2. `pytest tests/unit/test_chat_render_reference_notes.py -q`
3. `npm run build` in `web/`
4. offline replay on real message `1424` confirms:
   - one visible Eq. (1) entrance
   - one visible `where ...` quote-bundle entrance
   - the corrected `T(t)` formula no longer reuses the Eq. (1) entrance

### 11.25 Landed: inline-formula source targets for display-math answer surfaces

This phase fixes the remaining formula-target mismatch where the answer renders a display-math block, but the source paper only exposes the corresponding formula inline inside the explanation paragraph.

Implemented:

1. `kb/paper_guide_provenance.py`
   - added inline-formula span extraction and matching helpers
   - direct `formula_claim` entries that fail true display-equation grounding now attempt a nearby inline-formula rebind before being downgraded
   - successful rebinding emits:
     - `claim_type = inline_formula_claim`
     - `anchor_kind = inline_formula`
     - `formula_origin = explanation`
     - `locate_surface_policy = secondary`
   - the explanation block becomes the strict primary target, while the original equation block is preserved in `related_block_ids`
2. `web/src/components/chat/MessageList.tsx`
   - strict structured locate now accepts `inline_formula_claim`
   - display-math answer blocks may bind to an inline-formula source target without restoring global inline-formula buttons
3. `web/src/components/chat/PaperGuideReaderDrawer.tsx`
   - after landing on the explanation block, reader now focuses the best matching inline `.katex` node inside that block
   - the inline formula becomes the scroll focus while the parent explanation block remains highlighted

Verified:

1. `pytest tests/unit/test_task_runtime_provenance.py -q`
2. `pytest tests/unit/test_chat_render_reference_notes.py -q`
3. `pytest tests/unit -q`
4. `npm run build` in `web/`
5. regression coverage now includes:
   - display-math answer `T(t)=exp(...)`
   - source target rebinding to the inline `exp(...)` formula in the explanation paragraph rather than Eq. (1)

### 11.26 Landed: exact inline-formula rescue for legacy/live regressions

This follow-up hardens the first inline-formula implementation after a live sample showed both over-binding and under-binding:

1. legacy inline formulas stored as `formula_claim` still fell back to `Eq.(1)`
2. corrected / derived display formulas were sometimes rescued too broadly onto the source typo inline formula

Implemented:

1. `kb/paper_guide_provenance.py`
   - inline rescue now prefers the segment's own raw formula surface before any legacy stored `anchor_text/evidence_quote`
   - formula normalization strips only formatting-only TeX wrappers (`\left`, `\right`, `\mathbf`, spacing commands, braces)
   - rescue to `inline_formula_claim` now requires normalized equality/containment of the inline formula itself, not loose approximate similarity
   - legacy `formula_claim` segments that actually carry only an inline formula surface are converted during read-time hardening
   - duplicate inline-formula surfaces inside the same `formula_bundle` are collapsed to one visible secondary entrance

Verified:

1. `pytest tests/unit/test_task_runtime_provenance.py -q`
2. live replay through `api/chat_render._enrich_provenance_segments_for_display(...)`
   - message `1430` now keeps:
     - one `inline_formula_claim` for the original typo formula in the explanation paragraph
     - corrected ODE / derived formulas hidden instead of being rebound onto that typo source

### 11.27 Landed: temporary entrance-surface reduction to direct evidence only

This phase narrows the visible locate surfaces after live answers showed many low-value entrances on explanatory paragraphs and list items.

Implemented:

1. `web/src/components/chat/MarkdownRenderer.tsx`
   - removed inline `figure_ref` locate tokens from plain answer text
   - removed generic paragraph/list-item locate-button fallback
   - heading-like quoted titles such as `“Implementation Details”` are ignored by inline quote tokenization
   - kept only:
     - direct quote entrances
     - display-math entrances
     - image entrances
   - inline quote tokens are now surfaced as `blockquote`-kind strict evidence rather than paragraph/list-item surfaces
2. `web/src/components/chat/MessageList.tsx`
   - strict structured locate now only accepts `blockquote / equation / figure` render targets
   - paragraph/list-item fallback surfaces are therefore blocked even if structured resolution can otherwise find a segment

Verified:

1. `npm run build` in `web/`
2. code-path audit confirms:
   - explanatory paragraphs render without locate buttons
   - list items render without locate buttons
   - direct quotes, display equations, and images still render entrances

### 11.28 Landed: quote single-surface rendering + source-only display-math entrances

This follow-up closes two remaining UI-surface regressions seen in live answers:

1. a direct blockquote could show two entrances at once:
   - one on the quoted text itself
   - one on the outer blockquote container
2. generated/rewritten display-math blocks could still surface an equation entrance even when they were not the same formula surface as the source paper

Implemented:

1. `web/src/components/chat/MarkdownRenderer.tsx`
   - blockquote wrappers now suppress their outer locate button if the rendered children already contain a quote entrance
   - this keeps one visible entrance per direct quote block instead of two
2. `web/src/components/chat/MessageList.tsx`
   - strict equation entrances now require:
     - `claim_type = formula_claim`
     - `formula_origin = source`
     - `locate_surface_policy = primary`
   - explanation-level or rewritten/generated display-math surfaces therefore stop rendering locate entrances

Verified:

1. `npm run build` in `web/`
2. data audit on live quote message `1420` confirms each shown direct quote still maps to one `blockquote_claim`
3. code-path audit confirms:
   - quote duplication came from render-layer double exposure, not duplicate provenance segments
   - rewritten display-math surfaces are now filtered at the strict render gate

### 11.29 Landed: reference locate summary/related fallback restore

This phase restores `摘要 / 相关` cards in reference locate after live users reported that the hit cards were present but both text areas were blank.

Root cause:

1. answer-time refs storage now frequently persists snippet evidence only:
   - `ref_show_snippets`
   - `ref_overview_snippets`
   without an LLM `ref_pack`
2. `api/reference_ui.py` only exposed `summary_line / why_line` when `ref_pack_state == ready`
3. as a result, `build_hit_ui_meta()` produced empty strings for both fields even though snippet evidence and citation metadata already existed

Implemented:

1. `api/reference_ui.py`
   - added `summary_line` fallback synthesis when `ref_pack` is missing
   - fallback order:
     - `ref_show_snippets`
     - citation metadata `summary_line`
     - `ref_overview_snippets`
   - `why_line` now always falls back to the existing location-aware `_fallback_why_line_ui(...)` when pack output is absent

Verified:

1. `pytest tests/unit/test_reference_ui_score_calibration.py -q`
2. `pytest tests/unit/test_reference_metadata_guards.py -q`
3. live API replay on `/api/references/conversation/{conv_id}` now returns non-empty:
   - `ui_meta.summary_line`
   - `ui_meta.why_line`

### 11.30 Landed: paper-guide refs suppress bound source

This phase suppresses the currently bound guide paper from `reference locate` cards in `paper_guide` mode, so the panel only surfaces external references.

Root cause:

1. refs conversation payload enrichment previously did not know:
   - the conversation mode
   - the active `bound_source_path`
2. bound-paper hits were therefore enriched and rendered like ordinary external refs
3. this duplicated the already-open guide document and added low-value cards

Implemented:

1. `api/routers/references.py`
   - now loads conversation metadata and passes guide-source context into ref enrichment
2. `api/reference_ui.py`
   - added bound-source suppression in `enrich_refs_payload(...)`
   - emits a small `guide_filter` marker when self-source hits were intentionally hidden
3. `web/src/components/refs/RefsPanel.tsx`
   - shows a compact muted note instead of a full refs card when only the bound paper was suppressed
4. `web/src/components/chat/MessageList.tsx`
   - keeps refs rows renderable for that compact note state

Verified:

1. `pytest tests/unit/test_reference_ui_score_calibration.py -q`
2. `npm run build` in `web/`

### 11.31 Landed: self-ref suppression verified live; blockquote double-entry fixed

This follow-up closes two live regressions reported after the first paper-guide self-reference suppression change.

Observed:

1. paper-guide conversations still appeared to render the bound paper inside `reference locate`
2. some direct-quote blockquotes still showed two locate entrances for a single quote

Root cause:

1. backend suppression code had landed, but the running backend process was still serving the old payload shape without `guide_filter`
2. `MarkdownRenderer.tsx` blockquote suppression relied on already-rendered child buttons
   - ReactMarkdown parent blockquote nodes see unresolved child elements first
   - therefore the parent could not observe the future inline quote button and added a second outer button

Implemented:

1. restarted frontend/backend and re-verified the live SCINeRF guide conversation payload
2. `web/src/components/chat/MarkdownRenderer.tsx`
   - blockquote now suppresses its outer locate button whenever the raw blockquote text already contains a direct-quote inline locate candidate

Verified:

1. live `/api/references/conversation/{conv_id}` replay now shows:
   - `hits = 0`
   - `guide_filter.hidden_self_source = true`
2. `pytest tests/unit/test_reference_ui_score_calibration.py -q`
3. `npm run build` in `web/`

### 11.32 Landed: quote/figure entrance surface correction

This phase corrects three live entrance-surface regressions:

1. short quoted Chinese emphasis phrases in normal explanatory paragraphs were still surfacing unrelated locate buttons
2. quote blockquotes could expose entrances unstably because inline-vs-outer blockquote strategies were competing
3. explicit `Figure 1 / Fig. 1 / 图1` references no longer surfaced figure entrances after text figure refs had been disabled too broadly

Implemented:

1. `web/src/components/chat/MarkdownRenderer.tsx`
   - blockquote subtrees no longer render nested inline quote entrances
   - each quote block now keeps a single outer blockquote entrance
2. `web/src/components/chat/MarkdownRenderer.tsx`
   - explicit text figure-ref tokens were re-enabled for:
     - `Figure n`
     - `Fig. n`
     - `图 n`
3. `web/src/components/chat/MarkdownRenderer.tsx`
   - generic quote-token filtering now rejects short Chinese emphasis-style phrases that are not sentence-like direct evidence

Verified:

1. `npm run build` in `web/`

### 11.33 Review-driven next phase: surface ownership cleanup and inline-tail entrance design

This stage records the follow-up plan after reviewing the current end-to-end `paper_guide` locate implementation across:

- `kb/paper_guide_provenance.py`
- `kb/task_runtime.py`
- `web/src/components/chat/MarkdownRenderer.tsx`
- `web/src/components/chat/MessageList.tsx`
- `web/src/components/chat/PaperGuideReaderDrawer.tsx`
- `api/reference_ui.py`

Confirmed design issues:

1. frontend locate surface exposure is still split across two decision layers
   - `MarkdownRenderer.tsx` infers quote/figure entrances from rendered text
   - `MessageList.tsx` separately applies strict provenance/surface-policy gates
   - this makes regressions likely even when provenance is correct
2. provenance extraction is no longer co-located with `task_runtime`, but the module boundary is still incomplete
   - `paper_guide_provenance.py` still imports `task_runtime`
   - `task_runtime.py` still re-exports provenance helpers through wrappers
3. paper-guide self-reference suppression now works in the API, but the frontend still treats the suppressed-self state as a renderable refs panel
4. the current locate button visual treatment is functionally inline, but still reads like a detached mini-action rather than a tail annotation
5. mojibake remains in some backend Chinese heuristic constants and should be treated as correctness debt

Planned next work:

1. D1: unify locate surface ownership
   - backend provenance remains the only source that decides whether a visible entrance exists
   - visible entrance kinds stay restricted to:
     - `blockquote`
     - `equation` (display/source only)
     - `figure`
   - `MessageList.tsx` becomes the sole surface-policy consumer
   - `MarkdownRenderer.tsx` becomes a pure slot renderer and stops making quote/figure eligibility decisions
2. D2: finish provenance module decoupling
   - move shared regex/constants/path helpers into a small lower-level module
   - remove `paper_guide_provenance -> task_runtime` dependency
   - keep `task_runtime` focused on orchestration
3. D3: finalize paper-guide refs product behavior
   - default behavior should render nothing when only the bound guide paper was suppressed
   - any muted suppression notice should be an explicit product choice, not the baseline behavior
4. D4: inline-tail entrance UI polish
   - quote entrance sits at sentence tail, ideally after citation markers
   - display equation entrance sits on the same visual line as the equation number / right edge
   - figure entrance sits at caption tail, or image corner when no caption exists
   - replace the generic floating circle feel with a compact superscript-style tail chip
5. D5: cleanup/perf follow-up
   - fix mojibake Chinese heuristics
   - reduce duplicate resolve work after surface unification
   - evaluate frontend chunk splitting only after behavior is stable

Acceptance criteria:

1. one rendered surface maps to at most one visible entrance
2. no visible entrance exists without strict provenance backing
3. `MarkdownRenderer.tsx` no longer owns quote/figure business eligibility
4. paper-guide refs never render the bound paper as a normal card
5. entrances stay visually attached to sentence/formula/image tails instead of creating detached rows

### 11.34 Landed: D1 first slice narrows quote surfaces and hides self-suppressed refs panels

This first D1 landing narrows frontend visible surfaces without changing the provenance schema again.

Implemented:

1. `web/src/components/chat/MarkdownRenderer.tsx`
   - added `inlineLocateTokenPolicy`
   - `paper_guide` messages now disable inline quote-token entrances in ordinary paragraphs/list items
   - explicit text figure refs remain enabled
   - blockquote snippet selection still keeps its internal quote-preference helper
2. `web/src/components/chat/MessageList.tsx`
   - passes `inlineLocateTokenPolicy = { quote: false, figure_ref: true }` for guide messages
   - quote visibility therefore comes from `blockquote` surfaces instead of paragraph-level quote token guesses
3. `web/src/components/refs/RefsPanel.tsx`
   - no longer renders a default muted notice when the only suppressed hits belong to the currently bound guide paper
4. `web/src/components/chat/MessageList.tsx`
   - refs rows are not considered renderable solely because `hidden_self_source == true`

Why this slice:

1. it reduces the most common frontend regressions before the larger surface-ownership refactor
2. it makes paper-guide quote exposure closer to the intended visible-surface model
3. it aligns the default UX with the product rule that the currently bound guide paper should not keep a visible refs panel alive

Verified:

1. `npm run build` in `web/`

### 11.35 Landed: D1 second slice disables text-token entrances in paper-guide

This follow-up D1 slice makes the guide frontend stricter by removing paragraph/list-item inferred entrances entirely.

Implemented:

1. `web/src/components/chat/MarkdownRenderer.tsx`
   - added `inlineTextLocateEnabled`
   - inline quote/figure token decoration inside ordinary text nodes can now be disabled explicitly
2. `web/src/components/chat/MessageList.tsx`
   - guide messages now pass `inlineTextLocateEnabled = false`
   - visible entrances therefore come only from structural surfaces:
     - `blockquote`
     - display `equation`
     - actual `img` figure surfaces

Behavioral consequence:

1. ordinary text figure refs are intentionally removed in this tightening step
2. any future figure-text entrance should be restored only via explicit surface slots owned by `MessageList`, not by generic token scanning in `MarkdownRenderer`

Verified:

1. `npm run build` in `web/`

### 11.36 Landed: D1 third slice makes structural surface visibility explicit

This D1 follow-up removes one more implicit renderer default by introducing explicit structural surface policy.

Implemented:

1. `web/src/components/chat/MarkdownRenderer.tsx`
   - added `locateSurfacePolicy`
   - normalized visibility now exists for:
     - `paragraph`
     - `list_item`
     - `blockquote`
     - `equation`
     - `figure`
   - `renderLocateButton(...)` short-circuits when the current surface kind is disabled
2. `web/src/components/chat/MessageList.tsx`
   - guide messages now pass explicit structural visibility:
     - `paragraph: false`
     - `list_item: false`
     - `blockquote: true`
     - `equation: true`
     - `figure: true`

Impact:

1. guide-mode structural visibility is now caller-controlled rather than inherited from generic renderer defaults
2. this makes the next D1 step smaller because `MarkdownRenderer.tsx` no longer assumes all supported structural nodes are eligible to expose entrances

Verified:

1. `npm run build` in `web/`

### 11.37 Landed: D2 first slice extracts provenance shared primitives

This D2 slice removes the highest-risk backend reverse dependency in the paper-guide provenance stack.

Problem before this landing:

1. `kb/paper_guide_provenance.py` imported `kb.task_runtime as tr`
2. `kb/task_runtime.py` also lazy-imported `kb.paper_guide_provenance`
3. the provenance split was therefore file-level only, not a true dependency boundary

Implemented:

1. added `kb/paper_guide_shared.py`
   - exports provenance-facing low-level primitives:
     - `DeepSeekChat`
     - source-block matching helpers
     - md path resolver
     - provenance regex/constants
2. `kb/paper_guide_provenance.py`
   - no longer imports `kb.task_runtime`
   - now imports shared primitives directly from `kb.paper_guide_shared`
3. `kb/task_runtime.py`
   - still keeps compatibility wrappers, but is no longer the provenance module's dependency source

Impact:

1. provenance extraction no longer depends on the orchestration module that also lazy-loads it
2. future provenance changes now have a smaller dependency blast radius
3. this sets up the next D2 step where helper façade logic can be reduced inside `task_runtime.py`

Verified:

1. `pytest tests/unit/test_task_runtime_provenance.py -q`
2. `pytest tests/unit/test_chat_render_reference_notes.py -q`
3. `pytest tests/unit -q`

### 11.38 Landed: D2 second slice routes task_runtime runtime values through shared primitives

This D2 follow-up keeps the refactor safe by rebinding runtime values before removing the old literal blocks.

Implemented:

1. `kb/task_runtime.py`
   - imports shared paper-guide primitives with `_PG_...` aliases
   - rebinds runtime-facing provenance constants to the shared values:
     - figure/equation regexes
     - heading/summary/claim heuristics
     - non-source/equation-explanation helpers
     - quote ellipsis handling
2. the old literal blocks still exist temporarily, but they are no longer the runtime source of truth

Impact:

1. `task_runtime.py` and `paper_guide_provenance.py` now execute against the same shared primitive set
2. this removes the remaining live split-brain risk before the final duplicate literal deletion step

Verified:

1. `pytest tests/unit/test_task_runtime_provenance.py -q`
2. `pytest tests/unit -q`

### 11.41 Landed: D1 visual polish attaches locate entrances to quote/equation tails

This frontend-only slice improves the surface presentation of paper-guide locate entrances without widening eligibility.

Implemented:

1. `web/src/components/chat/MarkdownRenderer.tsx`
   - attaches blockquote locate buttons into the last quote line via a tail-append helper
   - renders display-math locate buttons inside an equation-tail shell instead of appending them as loose block-level trailing nodes
2. `web/src/styles/index.css`
   - adds dedicated quote-tail / equation-tail layout classes so entrances visually follow the sentence or formula end
   - switches the visual language from the heavier round quote button to a lighter superscript-style tail label:
     - `原`
     - `式`
     - `图`
   - fixes tail placement so trailing whitespace and citation-chip boundaries do not force the entrance onto a new line
   - increases quote-badge visibility after live feedback:
     - quote label becomes `原文`
     - contrast/border weight are raised so the tail entrance does not look absent

Impact:

1. quote entrances now look like inline tail annotations
2. display-equation entrances now stay visually attached to the formula block
3. the UI is closer to the intended paper-guide reading flow and avoids the “extra line control” look

Verified:

1. `npm run build` in `web/`

### 11.41 Landed: D3 first slice extracts answer-contract helpers from task_runtime

This D3 slice starts the next backend refactor phase by separating answer-contract logic from the task runtime orchestration file.

Implemented:

1. added `kb/answer_contract.py`
   - owns:
     - kb-miss notice parsing/reconciliation helpers
     - answer intent/depth detection
     - section/structure detection
     - answer quality probe logic
     - paper-guide grounding rules
     - default next-step generation
     - answer-contract repair and kb-miss enhancement logic
2. `kb/task_runtime.py`
   - now imports these helpers as static aliases
   - removes the old in-file answer-contract implementation block
   - preserves existing helper names for compatibility with tests and callers

Impact:

1. `task_runtime.py` is materially smaller and more orchestration-focused
2. answer-contract behavior is now isolated in a module with clearer unit-test scope
3. this prepares the next refactor step where answer-quality telemetry can be reviewed separately from generation runtime

Observed result:

1. `kb/task_runtime.py` line count is now `2664`

Verified:

1. `pytest tests/unit/test_task_runtime_answer_contract.py -q`
2. `pytest tests/unit -q`

### 11.42 Landed: D3 second slice extracts answer-quality telemetry from task_runtime

This D3 follow-up moves answer-quality telemetry out of the generation runtime orchestration file.

Implemented:

1. added `kb/answer_quality.py`
   - owns:
     - `_gen_record_answer_quality(...)`
     - `_gen_answer_quality_summary(...)`
   - continues to use shared `runtime_state` storage for `GEN_QUALITY_EVENTS`
2. `kb/task_runtime.py`
   - now imports these helpers as static aliases
   - removes the old in-file telemetry implementation block
   - preserves existing helper names and call sites

Impact:

1. `task_runtime.py` is smaller and more orchestration-focused
2. answer-quality telemetry now has a clearer unit-test and ownership boundary
3. keeps the refactor pattern consistent with provenance and answer-contract extraction

Observed result:

1. `kb/task_runtime.py` line count is now `2458`

Verified:

1. `pytest tests/unit/test_task_runtime_answer_contract.py -q`
2. `pytest tests/unit -q`

### 11.40 Landed: D2 fourth slice replaces dynamic provenance wrappers with static task_runtime aliases

This D2 slice removes the remaining dynamic provenance forwarding layer inside `kb/task_runtime.py`.

Implemented:

1. `kb/task_runtime.py`
   - imports paper-guide provenance helpers directly from `kb.paper_guide_provenance`
   - removes `_paper_guide_provenance_runtime()` and the large wrapper block that only forwarded helper calls into the provenance module
   - removes the leftover local `_NON_SOURCE_SEGMENT_HINTS` duplicate block
2. task-runtime-facing helper names are still preserved for compatibility:
   - callers still resolve provenance helpers through `task_runtime`
   - but those symbols are now static aliases instead of runtime dispatch wrappers

Impact:

1. `task_runtime.py` is materially smaller and less noisy
2. the provenance/orchestration boundary is clearer:
   - provenance logic lives in `kb.paper_guide_provenance`
   - orchestration remains in `kb.task_runtime`
3. removes another class of drift risk where wrapper stubs could lag behind provenance implementation changes

Observed result:

1. `kb/task_runtime.py` line count is now `3157`

Verified:

1. `pytest tests/unit/test_task_runtime_provenance.py -q`
2. `pytest tests/unit -q`

### 11.39 Landed: D2 third slice removes duplicated provenance literals from task_runtime

This D2 cleanup step completes the literal deduplication started in 11.38.

Implemented:

1. `kb/task_runtime.py`
   - removed the legacy local literal definitions for:
     - figure/equation tokenization regexes
     - heading/summary/claim heuristic regexes
     - quote-heading / figure-claim regexes
     - equation-explanation and quote-ellipsis regexes
   - retained only the shared-runtime rebinds that point to `kb.paper_guide_shared`
2. backend provenance runtime now has a single primitive source:
   - shared module owns the values
   - `task_runtime.py` only consumes them

Impact:

1. removes the last live "shared + duplicated local copy" ambiguity in the paper-guide provenance runtime
2. makes later heuristic edits lower-risk because there is now one backend source of truth
3. prepares the next phase where `task_runtime.py` can keep shrinking as a provenance façade

Verified:

1. `pytest tests/unit/test_task_runtime_provenance.py -q`
2. `pytest tests/unit -q`
