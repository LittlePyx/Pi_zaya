# Reference Locate Handoff 2026-04-20

## 1. Current Task

This handoff focuses on the new regression introduced while fixing multi-paper reference-locate consistency:

- the **top answer** and the **refs panel** are now much closer to using the same paper set
- but the **reference cards themselves became much worse**
- and the **pending window** for multi-paper queries can still show the wrong interim papers before the final full payload is returned

The user specifically complained that:

1. the card copy in `参考定位` became low quality, generic, and less useful
2. the card content sometimes looks obviously worse than before even when the final paper set is correct
3. some conversations still briefly show the wrong papers in the refs list before settling

## 2. What Was Fixed Already

These parts are now materially better than before:

- multi-paper SCI query detection was tightened, including Chinese prompt handling
- answer-side `doc_list` for `有哪几篇文章提到了SCI（单次曝光压缩成像）` now filters down to the expected 3 papers
- authoritative full refs payload can now persist from `doc_list`
- refs conversation cache signature now includes `render_status` and `rendered_payload_sig`, so old fast payloads no longer stay stuck forever

Relevant code:

- [kb/generation_answer_finalize_runtime.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/generation_answer_finalize_runtime.py:375)
- [kb/generation_answer_finalize_runtime.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/generation_answer_finalize_runtime.py:474)
- [kb/task_runtime.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/task_runtime.py:596)
- [api/routers/references.py](/f:/research-papers/2026/Jan/else/kb_chat/api/routers/references.py:96)

## 3. Latest Verified State

### 3.1 Good news

For the SCI multi-paper query, the **final authoritative state** now does converge to the expected 3-paper set:

- `ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image.pdf`
- `CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf`
- `OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture.pdf`

This was verified from:

- assistant-side `paper_guide_contracts.doc_list`
- raw stored `message_refs.rendered_payload`
- repeated `/api/references/conversation/{conv_id}` reads after full payload becomes ready

### 3.2 Still bad

Even after the consistency fix, card quality regressed for a structural reason:

- the multi-paper refs cards are now rendered by [api/reference_ui.py](/f:/research-papers/2026/Jan/else/kb_chat/api/reference_ui.py:3101) `build_doc_list_refs_payload`
- this path uses a **thin contract-to-card adapter**
- it does **not** go through the richer card-building and copy-polish stack used by normal locate cards

So the cards are now:

- more consistent in identity
- but worse in wording, specificity, and usefulness

## 4. Root Cause Analysis

### Root cause A: identity was unified, but display quality was downgraded

The current multi-paper fix made refs cards consume the answer-side `doc_list_contract`, which solved the “answer says 3 papers, refs panel shows different papers” problem.

But the rendering path is now too shallow:

- `doc_list_contract.summary_line` is created upstream by [kb/generation_answer_finalize_runtime.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/generation_answer_finalize_runtime.py:299) `_single_line_summary`
- `doc_list_contract` itself is built in [kb/generation_answer_finalize_runtime.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/generation_answer_finalize_runtime.py:474) `_build_multi_paper_doc_list_contract`
- cards then reuse that string directly in [api/reference_ui.py](/f:/research-papers/2026/Jan/else/kb_chat/api/reference_ui.py:3101) `build_doc_list_refs_payload`

This means the refs cards are now showing:

- answer-oriented compressed snippets
- not card-oriented evidence summaries

That is the main reason the cards feel much worse.

### Root cause B: multi-paper refs cards bypass the mature copy stack

Normal reference-locate cards still benefit from the richer flow:

- [api/reference_ui.py](/f:/research-papers/2026/Jan/else/kb_chat/api/reference_ui.py:3380) `build_hit_ui_meta`
- [api/reference_ui.py](/f:/research-papers/2026/Jan/else/kb_chat/api/reference_ui.py:1992) `_llm_ground_ref_why_line`
- [api/reference_ui.py](/f:/research-papers/2026/Jan/else/kb_chat/api/reference_ui.py:2238) `_maybe_polish_refs_card_copy`

The multi-paper authoritative path does **not** use that stack. It uses:

- a generic deterministic `why_line` from [api/reference_ui.py](/f:/research-papers/2026/Jan/else/kb_chat/api/reference_ui.py:3085) `_doc_list_ref_why_line`
- a raw compacted `summary_line`

So the current card quality regression is not accidental. It is a direct side effect of bypassing the higher-quality card generation path.

### Root cause C: pending preview still comes from the old raw-hit path

Early refs responses still use:

- [api/routers/references.py](/f:/research-papers/2026/Jan/else/kb_chat/api/routers/references.py:281) `_build_pending_conversation_refs_payload`

That path still seeds from raw retrieval hits, not from authoritative filtered `doc_list`.

So in the early pending window, the user can still briefly see:

- wrong papers
- weaker card text
- old-style provisional copy

Even when the final full payload is correct.

## 5. Most Appropriate Next Fix

Do **not** revert the shared `doc_list` authority. That part is the right direction.

The right next step is:

### 5.1 Split identity authority from display synthesis

Keep `doc_list_contract` as the **single source of truth for which papers belong in the list**.

But do **not** use it directly as the final UI copy.

Instead:

1. `doc_list_contract` keeps:
   - source identity
   - heading identity
   - primary evidence identity
   - rank/order

2. A new multi-paper card renderer should take each authoritative doc item and feed it back through the mature card pipeline, ideally a new helper such as:
   - `build_doc_list_hit_ui_meta(...)`
   - or reuse `build_hit_ui_meta(...)` with an authoritative override

3. That renderer should produce:
   - better `summary_line`
   - better `why_line`
   - correct locale
   - correct reader open target

In short:

- `doc_list` decides **which papers**
- `reference_ui` still decides **how each card should read**

### 5.2 Pending preview should reuse authoritative doc_list when available

If assistant-side `paper_guide_contracts.doc_list` already exists, then pending refs should not continue to display raw retrieval candidates.

Instead, the pending path should prefer:

- authoritative doc identities from `doc_list`
- downgraded provisional display state
- but still the same paper set

This likely means extending [api/routers/references.py](/f:/research-papers/2026/Jan/else/kb_chat/api/routers/references.py:281) `_build_pending_conversation_refs_payload`
or adding a new pending path that can render from stored `doc_list_contract`.

### 5.3 Introduce a multi-paper card quality contract

Right now `doc_list_contract` is overloaded. It tries to be both:

- answer-side list item contract
- refs-card display contract

That is too much coupling.

Recommended:

- keep `doc_list_contract` as identity/navigation contract
- optionally add `doc_card_contract` or `doc_list_display` later if needed

## 6. Recommended Implementation Order

1. Add an authoritative multi-paper card renderer in `api/reference_ui.py`
   - input: one `doc_list` item
   - output: high-quality `ui_meta`
   - requirement: reuse as much of the mature `build_hit_ui_meta` / polish stack as possible

2. Replace the shallow copy assembly in [api/reference_ui.py](/f:/research-papers/2026/Jan/else/kb_chat/api/reference_ui.py:3101)
   - keep the same final paper set
   - improve the card text

3. Add a pending-from-doc-list path in [api/routers/references.py](/f:/research-papers/2026/Jan/else/kb_chat/api/routers/references.py:281)
   - if doc list exists, pending refs should show the same papers as final refs

4. Add tests for:
   - paper set consistency
   - card copy quality
   - pending-to-full paper set stability

## 7. Concrete Acceptance Criteria

For the SCI multi-paper query:

### Identity

- top answer list and refs cards show the same paper set
- final refs panel shows exactly the 3 expected SCI papers
- no NatPhoton / NatCommun broad single-pixel papers appear in final refs

### Card quality

- `summary_line` is not just a raw intro shell or broad title echo
- `why_line` is not generic “kept as one of the direct matches...”
- Chinese locale cards use normal Chinese labels and reasons
- each card explains why that paper matches the user’s query, not just that it was retained

### Pending behavior

- once authoritative `doc_list` exists, pending refs should not show a materially different paper set
- a user should not see “wrong 2 papers first, correct 3 papers later” for the same query

## 8. Suggested Tests To Add

### Backend unit

- `doc_list_contract` item rendered into refs card should produce non-generic `summary_line`
- `doc_list_contract` item rendered into refs card should produce non-generic `why_line`
- when `doc_list` exists, pending refs should use the same paper set as full refs

### Live/manual

Use:

`有哪几篇文章提到了SCI（单次曝光压缩成像）`

Verify:

1. top answer lists 3 papers
2. refs panel lists the same 3 papers
3. card text is specific and readable
4. no broad unrelated single-pixel review paper appears
5. pending state does not transiently swap in the wrong papers

## 9. Current Best Reading Of The Situation

The system is currently in this state:

- **identity consistency** is much better than before
- **card copy quality** is worse than before
- **pending stability** is still not good enough for multi-paper queries

So the next task is **not** another retrieval/ranking fix first.

The next highest-value task is:

**restore card quality while preserving the new authoritative paper-set contract**

That is the most appropriate continuation point for the next window.
