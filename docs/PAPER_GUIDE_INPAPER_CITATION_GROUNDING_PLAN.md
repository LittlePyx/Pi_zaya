# Paper Guide In-Paper Citation Grounding Plan

Status: `phase-1 landed -> phase-2 design updated`
Last updated: `2026-03-23`

## 1. Problem Statement

The paper-guide feature already supports:

- bound-paper retrieval
- structured citation markers like `[[CITE:<sid>:<ref_num>]]`
- reference-index lookup
- clickable in-paper citation rendering

But the current citation grounding is still too weak.

Observed failure mode:

- the answer repeatedly shows the same in-paper reference such as `[1]`
- the repeated `[1]` often comes from model-side number guessing, not from correct claim-to-reference grounding
- once the model outputs `[[CITE:sid:1]]`, the runtime usually accepts it if reference `1` exists in the bound paper

This is not a rendering bug. It is a grounding and validation bug.

## 2. Root Cause

The current pipeline has two main weaknesses.

### 2.1 Retrieval Block Number And Reference Number Are Easy To Confuse

The answer-generation context currently contains headers like:

```text
[1] [SID:s1234abcd] paper.en.md | section
```

The leading `[1]` is a retrieval-block index, but it looks exactly like a paper reference number.

This creates a strong prompt-side bias toward generating `[[CITE:sid:1]]`.

### 2.2 Structured Citation Validation Only Checks Existence

`_validate_structured_citations()` currently verifies:

- whether the cited `sid` can be mapped to a source
- whether `ref_num` exists in that source's reference index

It does not adequately verify:

- whether the surrounding claim actually matches that reference
- whether the selected `ref_num` is consistent with the evidence block
- whether multiple repeated `ref_num=1` markers are suspicious

As a result, a wrong `ref_num` can survive end-to-end if it merely exists in the source paper.

## 3. Goals

This plan aims to make paper-guide in-paper citations:

1. less likely to default to `ref 1`
2. constrained by evidence-local candidate references where possible
3. repaired or dropped when the chosen `ref_num` conflicts with local context
4. testable with deterministic unit coverage

## 4. Non-Goals

This phase does not try to:

1. redesign the full retrieval stack
2. introduce a second mandatory LLM pass for every answer
3. fully solve citation grounding for general multi-document chat mode
4. change the frontend citation UI contract

## 5. Implementation Strategy

Use a phased approach.

### Phase 1

Scope:

- remove the prompt-side ambiguity that biases toward `ref 1`
- add deterministic post-generation grounding constraints for paper-guide mode
- keep the blast radius limited to the backend generation/validation path

Changes:

1. Replace retrieval context headers from `[1] [SID:...]` to `DOC-1 [SID:...]`.
2. Explicitly tell the model that `DOC-k` is a retrieval-block id, not a paper reference number.
3. Extract candidate reference numbers from the bound paper's retrieved evidence blocks.
4. During citation validation, prefer candidate refs over arbitrary existing refs.
5. When local answer context strongly suggests another reference, rewrite to that ref.
6. When the chosen `ref_num` cannot be justified, drop it instead of silently keeping it.

### Phase 2

Scope:

- strengthen local claim-to-reference matching
- move citation grounding earlier into answer generation
- keep locate and cite aligned to the same evidence slot
- reuse the same grounding logic in more than one backend layer

Planned work:

1. Introduce a support-slot contract so the model binds each paper-grounded claim to `DOC-k`, instead of directly guessing `ref_num`.
2. Extend evidence cards into slot-level metadata with `block_id`, `locate_anchor`, `claim_type`, `ref_spans`, and `candidate_refs`.
3. Resolve `[[SUPPORT:DOC-k]]` into `[[CITE:<sid>:<ref_num>]]` or locate-only provenance in runtime.
4. Store support-slot metadata in answer provenance so locate and cite use the same source block.
5. Add suspicious-citation analytics and support-slot coverage to answer-quality summaries.

### Phase 3

Scope:

- optional targeted citation-only repair pass

Trigger only when:

1. one answer collapses to the same ref number repeatedly
2. local author-year evidence conflicts with the selected ref
3. deterministic grounding cannot confidently repair the answer

## 6. Phase 1 Detailed Design

### 6.1 Context Header De-Ambiguation

Current:

```text
[1] [SID:s1234abcd] source | heading
```

New:

```text
DOC-1 [SID:s1234abcd] source | heading
```

Prompt rule to add:

- `DOC-1`, `DOC-2`, etc. are retrieval-block labels only
- they are not in-paper citation numbers

Expected benefit:

- reduce the model's tendency to copy retrieval-block index `1` into `[[CITE:sid:1]]`

### 6.2 Candidate Reference Extraction

For paper-guide mode only:

1. inspect retrieved bound-paper evidence blocks
2. extract explicit in-paper numeric reference mentions such as `[24]`, `[11-13]`
3. build a small candidate-ref set per source

Rules:

1. expand short numeric ranges
2. dedupe while preserving order
3. keep the set bounded

Expected benefit:

- if the retrieved evidence only mentions refs `24` and `25`, the validator should not casually preserve `ref 1`

### 6.3 Deterministic Citation Repair

When validating `[[CITE:sid:n]]` in paper-guide mode:

1. resolve the current ref metadata
2. inspect local answer context near the citation token
3. use DOI / author-year hints when available
4. if the current `n` conflicts with local context, search for a better candidate ref
5. if no justified ref can be found, drop the citation token

Priority order:

1. DOI match
2. author + year match
3. candidate-ref membership from retrieved evidence
4. fallback to old behavior only when no stronger grounding signal exists

### 6.4 Safety Bias

In paper-guide mode, when grounding is uncertain:

- prefer dropping a suspicious in-paper citation
- do not preserve a likely-wrong `ref_num` only because it exists in the paper

This is the correct tradeoff for a reading-guidance product.

## 7. Phase 2 Detailed Design

### 7.1 Design Principle

The key change in Phase 2 is:

- the model should not be asked to guess in-paper reference numbers directly
- the model should instead bind each paper-grounded claim to a support slot such as `DOC-2`
- runtime should resolve that support slot into the final citation or locate-only anchor

This is a better fit for paper-guide because:

1. choosing the correct local evidence block is easier than choosing the correct `ref_num`
2. cite and locate can be derived from the same slot, reducing cross-block drift
3. runtime can safely suppress citations for claims that should only expose locate/jump support

### 7.2 Support Slot Contract

Each paper-guide evidence card should be upgraded into a support slot with explicit grounding metadata.

Suggested slot shape:

```text
DOC-2
- sid=s3583e628
- source_path=...
- block_id=blk_...
- heading=Materials and Methods / Adaptive pixel-reassignment (APR)
- snippet=APR was performed using image registration based on phase correlation...
- locate_anchor=APR was performed using image registration...
- claim_type=method_detail
- cite_policy=prefer_ref
- ref_spans:
  - text=as detailed in [35]
  - nums=35
  - scope=same_sentence
- candidate_refs=35
- cite_example=[[CITE:s3583e628:35]]
```

Required fields:

1. `DOC-k`
2. `sid`
3. `block_id`
4. `heading`
5. `snippet`
6. `locate_anchor`
7. `claim_type`
8. `cite_policy`

Optional but strongly recommended fields:

1. `ref_spans`
2. `candidate_refs`
3. `cite_example`
4. `same_paragraph_refs`
5. `deepread_texts`

Recommended `claim_type` values:

1. `prior_work`
2. `method_detail`
3. `borrowed_tool`
4. `compare_result`
5. `figure_panel`
6. `own_result`
7. `abstract_quote`

Recommended `cite_policy` values:

1. `prefer_ref`: try hard to resolve a real in-paper citation
2. `allow_none`: use cite if strongly supported, otherwise keep locate only
3. `locate_only`: do not force an in-paper citation, keep provenance/jump only

Default mapping:

1. `prior_work` -> `prefer_ref`
2. `borrowed_tool` -> `prefer_ref`
3. `method_detail` -> `prefer_ref`
4. `compare_result` -> `allow_none`
5. `figure_panel` -> `locate_only`
6. `own_result` -> `locate_only`
7. `abstract_quote` -> `locate_only`

### 7.3 Generation Prompt Contract

The main answer prompt should stop asking the model to emit direct `[[CITE:<sid>:<ref_num>]]` as the primary grounding path.

Instead, the prompt should provide:

1. `Paper-guide support slots`
2. a rule that every paper-grounded claim or bullet must end with `[[SUPPORT:DOC-k]]`
3. a rule that `DOC-k` is a slot id, not a paper reference number
4. a rule that the model should not output raw numeric citations like `[35]`

Example answer skeleton:

```text
Conclusion:
- APR enables coherent reassignment for iISM [[SUPPORT:DOC-2]]

Evidence:
- Implementation detail: APR was performed using image registration based on phase correlation [[SUPPORT:DOC-2]]
- The paper reports improved CNR after APR [[SUPPORT:DOC-3]]
```

Prompt rules to add:

1. Use `[[SUPPORT:DOC-k]]` for paper-grounded claims.
2. Reuse the same `DOC-k` for a claim and its supporting locate anchor.
3. If the slot policy is `locate_only`, still use `[[SUPPORT:DOC-k]]`; runtime will decide whether to emit a citation.
4. If a needed detail is absent from all support slots, say it is not stated.

Rollout note:

- during migration, runtime should still accept legacy direct `[[CITE:...]]` output for backward compatibility
- the prompt should prefer support markers over direct cite markers

### 7.4 Runtime Resolution Algorithm

Once generation finishes, runtime should resolve support markers before final rendering.

Recommended resolution flow:

1. Parse the answer into segments and collect `[[SUPPORT:DOC-k]]` markers.
2. Map each marker to its support slot.
3. Copy locate metadata from the slot into provenance for that segment.
4. Decide whether the slot should emit a citation based on `cite_policy`.
5. If citation is allowed, choose `ref_num` with the following order:
   1. exact `ref_spans` in the same sentence
   2. same-paragraph refs stored on the slot
   3. `candidate_refs`
   4. DOI hint from nearby answer text
   5. author-year hint from nearby answer text
6. If no justified `ref_num` is found:
   1. keep the segment text
   2. keep locate provenance
   3. drop the in-paper citation
7. Replace `[[SUPPORT:DOC-k]]` with:
   1. `[[CITE:<sid>:<ref_num>]]` when resolved
   2. empty string when cite is not justified
8. Run `_validate_structured_citations()` as a compatibility and safety layer, not as the primary citation-binding layer.

Important safety rule:

- Phase 2 should still prefer a missing cite over a wrong cite

### 7.5 Runtime Integration Points

Implementation should be split into small, reviewable steps.

Step 1. Build support slots:

- extend the existing paper-guide evidence-card builder in `kb/task_runtime.py`
- compute `claim_type`, `cite_policy`, `ref_spans`, and `locate_anchor`
- move reusable ref-span extraction helpers into `kb/inpaper_citation_grounding.py` if the logic grows

Step 2. Prompt support-slot generation:

- inject `Paper-guide support slots` into the main answer prompt
- update `kb/answer_contract.py` rules so the model uses `[[SUPPORT:DOC-k]]`
- keep existing `cite_example` fields during migration, but mark them as examples rather than the primary output format

Step 3. Resolve support markers:

- add a runtime resolver in `kb/task_runtime.py`
- convert `[[SUPPORT:DOC-k]]` into final cite or locate-only behavior before rendering
- keep the existing cite validator as a fallback and repair layer

Step 4. Persist provenance:

- update `kb/paper_guide_provenance.py` so each output segment stores:
  1. `support_doc_k`
  2. `support_block_id`
  3. `support_locate_anchor`
  4. `support_claim_type`
  5. `support_ref_candidates`
  6. `resolved_ref_num`
  7. `citation_resolution_mode`

Step 5. Add analytics:

- record support-slot coverage in answer-quality summaries
- record suspicious cases such as:
  1. repeated same-ref collapse
  2. `prefer_ref` slots that resolved to no cite
  3. `locate_only` slots that incorrectly emitted cites

### 7.6 Backward Compatibility And Rollout

Phase 2 should be staged.

Stage 1:

- runtime supports both `[[SUPPORT:DOC-k]]` and legacy `[[CITE:...]]`
- tests prove no regression in current cite rendering

Stage 2:

- prompt prefers support markers
- direct cite injection becomes a fallback path only

Stage 3:

- support-slot provenance becomes the primary source for cite/locate analytics and jump generation

Frontend UI can remain backward compatible during rollout, but locate/jump presentation should be tightened so exact locate is not visually conflated with fallback locate.

### 7.7 Locate Entry Redesign

The current paper-guide locate UX still merges three different behaviors under the same "jump to source" mental model:

1. exact evidence locate via `block_id + anchor_id + anchor_text`
2. figure asset or caption jump
3. nearby/fuzzy fallback when the exact block cannot be resolved

This is the main reason users still perceive the locate入口 as "wrong" even when provenance exists.

The redesign principle is:

1. exact locate must be exact
2. figure asset jump must be presented as figure/caption support, not as exact text locate
3. fallback locate must never masquerade as exact locate

Desired user-visible layers:

1. `定位原文证据`
   - only for strict-identity, exact structured provenance
   - requires `primary_block_id`, `anchor_kind`, and anchor text/evidence quote
2. `查看图注/原图`
   - figure asset or caption jump
   - valid even when no exact paragraph-level block exists
3. `附近证据`
   - optional secondary surface for nearby/fuzzy fallback
   - clearly labeled as approximate, not exact

### 7.8 Locate Entry Implementation Slice 1

The first implementation slice should be intentionally narrow.

Goal:

- stop showing exact-locate UI for structured fallback matches

Scope:

1. keep exact structured locate buttons enabled
2. keep figure asset markdown / image jumps unchanged
3. disable inline exact-locate buttons when the underlying match is fallback-only
4. keep fallback information available internally for later `附近证据` design, but do not surface it as exact locate

Primary code path:

1. `web/src/components/chat/MessageList.tsx`
2. `web/src/components/chat/reader/useReaderLocateEngine.ts`

Expected effect:

1. fewer visible locate buttons in summary-style answers
2. fewer wrong jumps caused by fallback entries opening under the exact-locate label
3. no regression for strict figure/method/quote exact locate cases

Out of scope for slice 1:

1. redesigning the Reader drawer layout
2. adding a new `附近证据` button in the same PR
3. changing citation rendering

### 7.9 Focused Fix Package After Locate/Cite Audit

The focused audit on `2026-03-23` showed that the next highest-value fixes are no longer generic locate UI changes.

They are:

1. wrong in-paper cite resolution on method/prior-work questions
2. method exact-locate questions answering `not stated` despite explicit evidence
3. figure-panel questions binding to nearby result paragraphs instead of the caption clause itself

These fixes should be implemented in this order.

#### 7.9.1 Fix A: Cite Resolver Must Prefer Local Sentence Evidence

Observed failure:

- RVT method questions can resolve to `ref 32` even though the local method sentence explicitly points to `ref 34`
- method lines with no local reference can inherit `ref 35` from a nearby sentence and render a misleading cite

Implementation:

1. In support-slot construction, order `candidate_refs` by local `ref_spans` first, then broader card-level candidates.
2. Derive `cite_example` from the strongest local sentence-level ref, not from the first broad candidate.
3. In `_resolve_paper_guide_support_ref_num(...)`, rank evidence in this order:
   1. same-sentence `ref_spans` that best match the current answer line
   2. other local `ref_spans`
   3. slot `candidate_refs`
   4. context DOI / author-year hint fallback
4. If the answer line is an explicit `not stated` / missing-evidence statement, do not emit a cite from that line.

Expected effect:

1. RVT questions resolve to `ref 34` instead of `ref 32`
2. `not stated` method lines stop leaking unrelated cites such as `ref 35`

#### 7.9.2 Fix B: Method Exact-Locate Questions Must Override `not stated`

Observed failure:

- the system answers `not stated` for APR shift-vector re-application even when the Methods section explicitly says the vectors are applied back to the original iISM dataset

Implementation:

1. Expand method-focus term extraction so prompt cues such as `shift vectors`, `applied back`, and `original iISM dataset` influence method-slot selection.
2. In method-focus excerpt ranking, strongly reward fragments that match these exact prompt cues.
3. In `_repair_paper_guide_focus_answer_generic(...)`, if:
   1. the question is an exact-support / where-in-the-pipeline style method question
   2. a strong method detail excerpt is available
   3. the answer says `not stated`
   then replace the `not stated` shell with a direct evidence-grounded sentence built from that excerpt.

Expected effect:

1. exact method questions answer from the explicit method sentence instead of from a generic missing-evidence shell
2. locate provenance and visible answer become aligned

#### 7.9.3 Fix C: Figure Panel Questions Must Prefer Caption-Panel Slots

Observed failure:

- Figure 1 panel `(f)/(g)` questions can bind to nearby APR result paragraphs rather than the figure caption clause itself

Implementation:

1. In figure-walkthrough support-marker injection, when the answer line explicitly names panels, strongly prefer `figure_panel` slots over `method_detail` or `compare_result` slots.
2. Use panel-letter overlap as a scoring signal during support-slot selection.
3. When support resolution for a segment comes from a `figure_panel` slot, prefer the slot's caption-derived `locate_anchor` as the segment anchor text.
4. Keep broader result paragraphs as secondary context, not as the primary exact locate binding for explicit panel questions.

Expected effect:

1. `(d)/(e)` and `(f)/(g)` panel questions bind to the caption clause first
2. exact locate for figure-panel answers matches what the user actually asked for

### 7.10 `task_runtime.py` Split Track

`kb/task_runtime.py` is now large enough that feature work and debugging are fighting each other.

See also:

1. `docs/TASK_RUNTIME_SPLIT_ARCHITECTURE_PLAN.md` for the broader runtime/module architecture rationale and migration slices beyond paper-guide citation grounding

Current problem:

1. paper-guide prompt shaping, evidence selection, cite injection, provenance handoff, and sanitize/repair logic all live in the same file
2. small grounding fixes require touching distant sections with shared local helpers
3. it is hard to tell which helpers are paper-guide-only versus generally reusable

Refactor goal:

1. keep the top-level task orchestration in `kb/task_runtime.py`
2. move paper-guide-specific logic into focused modules with explicit contracts
3. leave call sites shallow enough that future debugging starts from one orchestration function plus one domain module

Recommended target split:

1. `kb/paper_guide_prompting.py`
   - prompt-family detection
   - retrieval prompt augmentation
   - deep-read extra selection
   - evidence-card block rendering
   - support-slot block rendering
2. `kb/paper_guide_focus.py`
   - method focus-term extraction
   - special-focus excerpt extraction
   - method/figure focus repair helpers
   - direct abstract answer helpers
3. `kb/paper_guide_grounding_runtime.py`
   - support-slot construction
   - support-marker injection
   - support-marker resolution
   - cite candidate ordering
   - paper-guide-specific cite fallback/drop logic
4. `kb/paper_guide_postprocess.py`
   - answer sanitization
   - user-facing cleanup
   - negative-shell / citation-lookup suppression rules
5. `kb/paper_guide_shared.py`
   - only stable regex/constants/shared normalizers should live here

What should remain in `kb/task_runtime.py`:

1. background task orchestration
2. retrieval / rerank / model call wiring
3. high-level answer pipeline sequencing
4. provenance/storage handoff

Recommended extraction order:

1. `paper_guide_postprocess.py`
   - lowest risk because functions are mostly pure string transforms
   - easy to regression-test
2. `paper_guide_focus.py`
   - second lowest risk because helpers are mostly local text selection
3. `paper_guide_prompting.py`
   - groups prompt-family and support-block builders
4. `paper_guide_grounding_runtime.py`
   - highest risk; move only after the above three are stable

Refactor rules:

1. each extraction PR should keep imports one-way
2. extracted modules should not import `task_runtime`
3. new modules should expose small public entrypoints, not dozens of cross-calling private helpers
4. shared regex/constants should be moved only when used by more than one extracted module

Acceptance for the split track:

1. `kb/task_runtime.py` no longer owns paper-guide-only string cleanup helpers
2. paper-guide focus/grounding logic can be unit-tested without importing the entire task runtime stack
3. paper-guide regressions remain green after each extraction slice
4. new paper-guide fixes touch at most one extracted module plus one orchestration call site

Current implementation status:

1. Slice 1 started on `2026-03-23`
2. `kb/paper_guide_postprocess.py` now owns paper-guide answer sanitization and cite-token cleanup helpers
3. `kb/task_runtime.py` imports those helpers instead of defining them inline
4. Slice 2 started on `2026-03-23`
5. `kb/paper_guide_prompting.py` now owns prompt-family detection, target parsing, and retrieval-prompt augmentation helpers
6. `kb/task_runtime.py` imports those prompting helpers instead of defining them inline
7. Slice 3 started on `2026-03-23`
8. `kb/paper_guide_focus.py` now owns abstract/method/caption focus helpers and special-focus excerpt extraction
9. `kb/paper_guide_focus.py` now also owns the direct abstract-answer builder and focus-answer repair implementation
10. `kb/paper_guide_focus.py` now also owns `special_focus_block` construction
11. `kb/task_runtime.py` keeps thin wrappers for compatibility but no longer owns the focus bodies
12. Slice 4 started on `2026-03-23`
13. `kb/paper_guide_grounding_runtime.py` now owns cue/ref primitives, support-slot block matching, the extracted support-slot builder path, support-slot prompt-block rendering, support-marker injection/resolution, and support-slot ref-resolution policy
14. `kb/task_runtime.py` now imports those grounding helpers and keeps only wrapper call sites for compatibility
15. `kb/paper_guide_prompting.py` now also owns deep-read context merging, evidence-card block rendering, citation-grounding block rendering, and requested-figure fallback helpers
16. `kb/paper_guide_shared.py` now owns shared prompt-field / prompt-snippet trimming and abstract-excerpt extraction used across prompting and grounding
17. `kb/paper_guide_retrieval_runtime.py` now also owns the higher-risk retrieval branch: `targeted_box_excerpt_hits`, `targeted_source_block_hits`, and `fallback_deepread_hits`
18. `kb/paper_guide_retrieval_runtime.py` now also owns the raw-target / citation-lookup retrieval glue: `select_paper_guide_raw_target_hits`, `paper_guide_citation_lookup_fragments`, `extract_paper_guide_local_citation_lookup_refs`, `paper_guide_citation_lookup_query_tokens`, `paper_guide_citation_lookup_signal_score`, and `build_paper_guide_direct_citation_lookup_answer`
19. `kb/paper_guide_citation_surfacing.py` now owns post-generation paper-guide cite surfacing: `collect_paper_guide_candidate_refs_by_source`, `inject_paper_guide_fallback_citations`, `inject_paper_guide_focus_citations`, `inject_paper_guide_card_citations`, `drop_paper_guide_locate_only_line_citations`, and `promote_paper_guide_numeric_reference_citations`
20. `kb/task_runtime.py` now imports those prompting/shared/retrieval/surfacing helpers instead of defining them inline
21. Slice 5 started on `2026-03-24`
22. `kb/paper_guide_answer_selection.py` now owns paper-guide output-mode stabilization, focused-heading cleanup, answer-hit scoring, answer-hit selection, generic answer-hit fallback assembly, and anchor-grounded answer-hit detection
23. `kb/task_runtime.py` now imports those answer-selection helpers and keeps wrapper call sites for compatibility
24. Slice 6 started on `2026-03-24`
25. `kb/paper_guide_context_runtime.py` now owns paper-guide context-record construction, deep-read context merge, and prompt-block preparation for answer generation
26. `kb.paper_guide_shared.py` now also owns `_cite_source_id` and `_source_name_from_md_path`
27. `kb/task_runtime.py` now imports those context/shared helpers and keeps only wrapper call sites plus top-level sequencing
28. Slice 7 started on `2026-03-24`
29. `kb/paper_guide_message_builder.py` now owns generation-time system/user prompt assembly and returns prompt-bundle state including `paper_guide_contract_enabled`
30. `kb/task_runtime.py` now imports that message-builder helper and keeps only wrapper call sites plus multimodal history/image wiring
31. Slice 8 started on `2026-03-24`
32. `kb/generation_message_runtime.py` now owns multimodal history filtering, user-content image payload assembly, and final generation-message list assembly
33. `kb/paper_guide_direct_answer_runtime.py` now owns direct abstract / citation-lookup override routing
34. `kb/task_runtime.py` now imports those generation/direct-answer helpers and keeps only wrappers plus actual LLM streaming lifecycle
35. Slice 9 started on `2026-03-24`
36. `kb/paper_guide_answer_post_runtime.py` now owns focus repair, support-marker resolution, cite surfacing, and paper-guide sanitization orchestration
37. `kb/task_runtime.py` now imports that answer-post helper and keeps only wrappers plus final generic validation/quality/store glue
38. Slice 10 started on `2026-03-24`
39. `kb/library_figure_runtime.py` now owns library-figure asset discovery, scoring, and appendix rendering
40. `kb/task_runtime.py` now imports that helper and keeps only thin compatibility wrappers for figure appendix helpers
41. Slice 11 started on `2026-03-24`
42. `kb/generation_answer_finalize_runtime.py` now owns final answer normalization, contract/kb-miss sequencing, paper-guide answer-post sequencing, citation-validation call sequencing, and answer-quality probe sequencing
43. `kb/task_runtime.py` now imports that finalize helper and keeps only the validator implementation plus storage/provenance/status glue
44. Slice 12 started on `2026-03-24`
45. `kb/generation_citation_validation_runtime.py` now owns source-ref lookup and final structured-citation validation / rewrite / drop logic, including paper-guide candidate-ref and support-slot aware grounding
46. `kb/task_runtime.py` now imports that citation-validation helper and keeps only thin wrappers plus storage/provenance/status glue
47. Slice 13 started on `2026-03-24`
48. `kb/generation_state_runtime.py` now owns live-assistant/task-state helpers plus answer/provenance persistence and async-refine helpers
49. `kb/task_runtime.py` now imports that state helper module and keeps only thin wrappers plus the top-level worker / streaming lifecycle
50. Next extraction target should be wrapper-surface reduction or the remaining high-level worker/orchestration glue

Measured size after Slice 13:

1. `kb/task_runtime.py`: `3081` lines / `119618` chars
2. `kb/paper_guide_postprocess.py`: `138` lines / `4592` chars
3. `kb/paper_guide_prompting.py`: `678` lines / `25092` chars
4. `kb/paper_guide_focus.py`: `1170` lines / `48232` chars
5. `kb/paper_guide_grounding_runtime.py`: `1415` lines / `55620` chars
6. `kb/paper_guide_shared.py`: `298` lines / `10454` chars
7. `kb/paper_guide_retrieval_runtime.py`: `802` lines / `29477` chars
8. `kb/paper_guide_citation_surfacing.py`: `476` lines / `16415` chars
9. `kb/paper_guide_answer_selection.py`: `416` lines / `15422` chars
10. `kb/paper_guide_context_runtime.py`: `288` lines / `11093` chars
11. `kb/paper_guide_message_builder.py`: `146` lines / `7688` chars
12. `kb/generation_message_runtime.py`: `81` lines / `2805` chars
13. `kb/paper_guide_direct_answer_runtime.py`: `63` lines / `1883` chars
14. `kb/paper_guide_answer_post_runtime.py`: `116` lines / `3706` chars
15. `kb/library_figure_runtime.py`: `231` lines / `7657` chars
16. `kb/generation_answer_finalize_runtime.py`: `111` lines / `3928` chars
17. `kb/generation_citation_validation_runtime.py`: `503` lines / `19202` chars
18. `kb/generation_state_runtime.py`: `238` lines / `7478` chars
19. The largest remaining paper-guide complexity now sits in wrapper compatibility surfaces and high-level worker/orchestration glue

### 7.11 Cross-Paper Audit After Slice 1

Audit set (`2026-03-23`):

1. `NatPhoton-2025-Structured detection for simultaneous super-resolution and optical sectioning in laser scanning microscopy`
2. `NatPhoton-2019-Principles and prospects for single-pixel imaging`

Questions exercised:

1. overview / contribution
2. citation lookup
3. explicit figure-panel exact support
4. comparison question
5. box-only / section-only exact support

Observed wins:

1. figure-level locate works across both papers when the caption is explicitly present
2. citation-lookup answers can now preserve literal numeric refs like `[33]`, `[34]`
3. `task_runtime.py` no longer owns the paper-guide postprocess branch

Observed failures that still need product work:

1. raw paper-guide answers can still transiently contain unresolved structured cite markers such as `[[CITE:<sid>:1]]`, even when the final rendered cite list is empty
2. citation-lookup answers can still anchor to the wrong block when the exact statement lives in intro/results text but the answer falls back to the reference list
3. figure-panel exact locate still binds at caption-block granularity, not panel-clause granularity
4. `Box 1 only` / `section-only` prompts can still miss clearly present content if retrieval or deep-read does not force that section
5. review-style questions with inline refs in prose still under-surface visible citations

Next implementation order after this audit:

1. eliminate unresolved structured-cite leakage from raw paper-guide answers
2. add section-targeted retrieval for `Box`, `Methods`, `Discussion`, and explicit figure targets
3. upgrade figure-panel locate from caption-block binding to panel-clause binding
4. extract `paper_guide_focus.py` from `kb/task_runtime.py`

## 8. Files To Change

Phase 1 primary files:

- `kb/task_runtime.py`
- `tests/unit/test_task_runtime_answer_contract.py`

Phase 1 optional/helper files:

- `kb/inpaper_citation_grounding.py`

Phase 2 primary files:

- `kb/task_runtime.py`
- `kb/answer_contract.py`
- `kb/paper_guide_provenance.py`
- `web/src/components/chat/MessageList.tsx`
- `tests/unit/test_task_runtime_bg_task.py`
- `tests/unit/test_task_runtime_answer_contract.py`

Phase 2 optional/helper files:

- `kb/inpaper_citation_grounding.py`
- `tools/manual_regression/run_paper_guide_lsa_regression.py`
- `tests/unit/test_paper_guide_regression_runner.py`
- `web/src/components/chat/PaperGuideReaderDrawer.tsx`
- `web/src/components/chat/reader/useReaderLocateEngine.ts`

## 9. Test Plan

### 9.1 Unit Tests

Add or update tests for:

1. SID-header sanitization still removes internal header markers after `DOC-k` migration
2. structured citations still rewrite to locked source when that is correct
3. paper-guide validation rewrites a wrong `ref_num` to the evidence-supported one
4. paper-guide validation drops a suspicious `ref_num` when there is no justified match
5. non-paper-guide behavior remains backward compatible
6. support-slot builder extracts `ref_spans` and `candidate_refs` from the same evidence block
7. support-slot resolver prefers same-sentence refs over broader candidate refs
8. `locate_only` slots do not emit forced in-paper citations
9. `prefer_ref` slots emit structured cites when a justified ref exists
10. weakly grounded support slots keep text but drop suspicious cites
11. provenance stores `support_doc_k`, `support_block_id`, and `resolved_ref_num`
12. legacy direct `[[CITE:...]]` output remains supported during migration
13. strict paper-guide locate does not expose exact-locate UI for fallback-only structured matches
14. support-slot candidate refs prefer local `ref_spans` over broader card-level candidate refs
15. explicit `not stated` answer lines do not emit paper-guide cites
16. exact method-support questions override `not stated` when a strong local method sentence exists
17. explicit figure-panel lines prefer `figure_panel` support markers over nearby method/result slots

### 9.2 Integration And Manual Checks

Run at least one real paper-guide session and verify:

1. the model is no longer exposed to retrieval-block headers that look like `[1]`
2. answers that previously collapsed to repeated `[1]` no longer do so by default
3. if the answer contains a wrong structured cite, it is either repaired or removed
4. clickable citation chips still render normally
5. `[[SUPPORT:DOC-k]]` markers never leak to the final user-visible answer
6. method/prior-work questions produce cites from the same slot used for locate provenance
7. abstract and figure-walkthrough questions can keep locate/jump support without being forced to invent in-paper cites
8. exact locate buttons only appear when the jump resolves through strict structured provenance, not through fuzzy fallback
9. figure image/caption jumps still remain available when exact text locate is intentionally withheld

Recommended manual prompt set:

1. overview question about the paper's main contribution
2. method question that names a specific sub-module
3. comparison question with reported numeric trade-offs
4. abstract text + translation request
5. figure walkthrough request
6. prior-work/tool question that should resolve to an explicit in-paper reference
7. exact figure-panel request such as `Figure 1 (d)/(e)` mapping
8. method-step locate request such as APR shift-vector re-application
9. `Discussion only` request that should prefer no locate over a wrong locate
10. RVT prior-work question where the correct cite should be `[34]`, not a discussion-only fallback cite
11. APR shift-vector question where the answer must not say `not stated` if the Methods section says `applied back to the original iISM dataset`

### 9.3 Regression Scope

Must re-run:

1. targeted answer-contract tests
2. targeted reference-metadata / citation-render tests if touched
3. paper-guide regression runner
4. any provenance tests if support-slot metadata is added
5. frontend build after any locate-entry UI change
6. focused locate/cite regression prompts that distinguish exact locate from asset-only jump
7. focused audit prompts for:
   1. wrong cite on RVT prior-work
   2. false `not stated` on APR shift-vector re-application
   3. caption-first binding on Figure 1 `(f)/(g)`

Recommended command set:

```bash
pytest -q tests/unit/test_task_runtime_bg_task.py -k "paper_guide or support_slot"
pytest -q tests/unit/test_task_runtime_answer_contract.py
pytest -q tests/unit/test_paper_guide_regression_runner.py
pytest -q tests/unit/test_task_runtime_inpaper_citation_grounding.py
pytest -q tests/unit/test_reference_metadata_guards.py -k "numeric_citation or structured_cite"
python tools/manual_regression/run_paper_guide_lsa_regression.py
npm run build
```

Recommended focused locate/cite eval cases:

1. exact figure-panel mapping: `Figure 1 panels (d)/(e)`
2. exact figure-panel mapping: `Figure 1 panels (f)/(g)`
3. exact method-step locate: APR shift-vector re-application
4. prior-work + method paragraph: RVT reference and local usage
5. section-only guard: `From the Discussion section only`
6. figure summary guard: overall Figure 1 explanation with exact-support-only requirement
7. cite resolver guard: RVT method/prior-work should ground to `ref 34` when the local sentence says `RVT[34]`
8. method override guard: APR `applied back to the original iISM dataset`
9. panel binding guard: explicit `Figure 1 (f)/(g)` question should prefer caption support over a nearby APR summary paragraph

## 10. Acceptance Criteria

Phase 1 is accepted only if all of the following are true.

### 10.1 Phase 1 Functional

1. retrieval context headers no longer use `[k] [SID:...]`
2. paper-guide citation validation is no longer pure existence-checking
3. wrong repeated `ref 1` citations can be rewritten or dropped deterministically

### 10.2 Phase 1 Test

1. all new unit tests pass
2. existing targeted citation tests still pass
3. no regression in locked-source citation rewriting

### 10.3 Phase 1 Product-Level

For a representative paper-guide answer:

1. `[1]` does not appear repeatedly just because the first retrieval block was labeled `1`
2. when answer-local evidence points to another reference, the final citation reflects that
3. when grounding is weak, the system avoids confidently wrong in-paper references

### 10.4 Phase 2 Functional

1. the main answer prompt contains support slots, not only flat candidate-ref hints
2. the model can bind claim text to `DOC-k` without directly outputting `ref_num`
3. runtime resolves support slots into structured cites when slot-local ref evidence exists
4. runtime suppresses cites for `locate_only` slots when a cite is not warranted
5. cite and locate for the same answer segment come from the same support slot / block provenance
6. strict paper-guide locate no longer opens fallback matches under the exact-locate surface
7. local same-sentence refs override broad candidate-ref hints when they conflict
8. exact method-support prompts no longer return `not stated` when a bound method sentence explicitly answers the question
9. explicit figure-panel prompts bind to caption-panel support before broader result paragraphs

### 10.5 Phase 2 Test

1. all new support-slot unit tests pass
2. existing Phase 1 citation tests still pass
3. paper-guide regression runner passes on the benchmark set
4. at least one method/prior-work case in manual regression shows non-zero `structured_cite_count`
5. no final answer leaks raw `[[SUPPORT:DOC-k]]` markers
6. focused locate regression confirms fallback-only cases either show no exact locate button or remain figure-asset-only
7. focused cite regression confirms RVT-style questions resolve to the local cited ref instead of a discussion-era fallback ref
8. focused method regression confirms APR shift-vector re-application answers no longer return `not stated`
9. focused figure regression confirms `(f)/(g)` panel prompts prefer caption-derived panel support

### 10.6 Phase 2 Product-Level

For a representative paper-guide answer set:

1. method or prior-work questions cite the correct in-paper reference from the same local slot that justified the claim
2. abstract and figure-walkthrough answers prefer locate/jump support over invented citations
3. repeated same-ref collapse no longer happens merely because one ref exists in the paper
4. when grounding is weak, the system returns a readable answer with locate support and no suspicious cite
5. exact locate surfaces no longer send the user to a nearby-but-wrong block just because fuzzy fallback found something similar
6. figure-summary answers may expose fewer locate buttons, but every visible exact locate button should correspond to a strict provenance target
7. RVT-style prior-work answers cite the local method reference, not a semantically related but wrong discussion reference
8. APR exact-support questions answer from the explicit method sentence when that sentence exists
9. figure-panel answers expose primary locate bindings that correspond to the requested panel clause, not merely to a nearby explanatory paragraph

## 11. Definition Of Done

This work is done when:

1. the plan is committed into the repo
2. Phase 1 backend changes are landed
3. Phase 2 support-slot design is landed in runtime and provenance
4. Phase 1 and Phase 2 tests are added and passing
5. the final implementation note records:
   - what changed
   - what was tested
   - what risks remain

## 12. Risks And Tradeoffs

### Risk 1

Some citations may now be dropped instead of preserved.

Tradeoff:

- this is acceptable
- a dropped citation is safer than a confidently wrong in-paper reference

### Risk 2

Candidate extraction from retrieved evidence is incomplete.

Tradeoff:

- Phase 1 keeps a bounded fallback path for cases with weak evidence
- Phase 2 can improve candidate recall later

### Risk 3

The validator may still miss some author-year patterns.

Tradeoff:

- the initial implementation should optimize for correctness and bounded complexity
- broader pattern coverage can be added after real-case observation

### Risk 4

Support markers may leak into the final answer if resolution fails or is skipped.

Tradeoff:

- this must be treated as a hard failure in tests
- the resolver should always strip unresolved `[[SUPPORT:DOC-k]]` markers before render

### Risk 5

Some answer types may end up with fewer visible citations after moving to slot-based grounding.

Tradeoff:

- this is acceptable if locate/jump support remains correct
- Phase 2 should optimize for correct cite-or-no-cite decisions, not for raw citation count inflation
