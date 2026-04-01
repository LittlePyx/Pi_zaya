# Paper Guide Pre-Implementation Blueprint

Updated: 2026-03-31
Status: proposed
Scope: `kb_chat` paper-guide feature before the next code-change cycle

## 1. Purpose

This document is the execution baseline before the next round of paper-guide implementation work.

It does not replace the existing plan documents. It consolidates them into one practical blueprint that answers:

1. What product are we actually trying to build?
2. What is already good in the current codebase?
3. What is still broken in the current behavior?
4. What architecture decisions must be fixed before more patching?
5. What should be implemented first, and how will we verify it?

Related documents:

- `docs/PAPER_GUIDE_QUESTION_COVERAGE_PLAN.md`
- `docs/PAPER_GUIDE_PROVENANCE_PLAN.md`
- `docs/PAPER_GUIDE_STABILIZATION_PLAN.md`
- `docs/PAPER_GUIDE_EVIDENCE_REBUILD_PLAN.md`
- `docs/PAPER_GUIDE_INPAPER_CITATION_GROUNDING_PLAN.md`
- `docs/PAPER_GUIDE_LOCATE_C1_HARDENING_PLAN.md`
- `docs/PAPER_GUIDE_P2_DEEP_READING_PLAN.md`

## 2. Product Definition

### 2.1 Core Product Statement

Paper Guide is not a generic RAG chat mode.

It is a bound-paper reading assistant that must support:

1. answering reading questions about one paper
2. grounding key claims in explicit evidence
3. opening the reader at the correct evidence location
4. distinguishing direct support from synthesis and from missing information

The desired user feeling is:

1. "It can explain the paper."
2. "It can show me exactly where that came from."
3. "If it says the paper did not state something, I can trust that it really checked."

### 2.2 Primary User Jobs

The feature should support both beginner and expert workflows.

Beginner jobs:

1. understand what the paper is about
2. find the right section to read next
3. ask for plain-language explanations of methods, figures, and equations

Expert jobs:

1. verify exact training or experiment details
2. find the exact sentence, paragraph, equation, or figure panel behind a claim
3. trace citations inside the paper
4. collect reproduction details without manually scanning the whole document

### 2.3 Supported Question Families

The target feature set should explicitly support these families:

1. `doc_map`
2. `overview`
3. `method_trace`
4. `equation_explain`
5. `figure_walkthrough`
6. `citation_lookup`
7. `reproduce_checklist`
8. `compare_limits`

### 2.4 Non-Goals

This phase should not try to do the following:

1. redesign the whole chat product
2. build a multi-paper literature review agent
3. solve open-domain retrieval quality for the whole knowledge base
4. rewrite the reader rendering stack from scratch
5. promise perfect reference extraction from every converted PDF before the mainline is stabilized

## 3. Current Reality

### 3.1 What Is Already Strong

The current codebase already has several good foundations:

1. bound-paper scope exists in the retrieval runtime
2. source-block-based reader data exists
3. provenance is already a backend concept, not only a frontend guess
4. locate contracts and strict locate ideas already exist
5. paper-guide prompt families and some deterministic rescue paths already exist

This means the next phase should be a consolidation effort, not a restart.

### 3.2 Current Failure Modes

The current user-visible failures cluster into four groups.

#### A. False Misses

Observed behavior:

1. the answer says the paper did not mention something
2. the relevant text actually exists in the paper
3. the system failed during coarse retrieval or weak fallback, not because the paper lacks the fact

Likely reasons:

1. coarse retrieval is still lexical-first
2. deep-read fallback is also token-overlap-heavy
3. rescue logic is too conservative when some weak hits already exist
4. answer-hit selection can drop rescued evidence before answer generation

#### B. Citation Grounding Failures

Observed behavior:

1. the system sometimes fails to identify the correct in-paper citation support
2. broken or incomplete reference extraction can contaminate citation lookup
3. body citation support and reference entry support are still too coupled

Likely reasons:

1. reference extraction quality is not stable across converted papers
2. citation questions still depend too much on general retrieval
3. body-side citation evidence is not always treated as the primary truth source

#### C. Locate Drift

Observed behavior:

1. the answer is roughly correct
2. the citation looks roughly plausible
3. the reader opens at the wrong paragraph, section, or a guessed fallback location

Likely reasons:

1. locate resolution is split across backend provenance, message rendering, and reader fallback
2. structured locate is not yet fully authoritative
3. frontend still re-ranks or reconstructs target candidates

#### D. Deep Reading Weakness

Observed behavior:

1. the system is better at point lookup than at guided reading
2. exact questions and summary questions still share too much of the same retrieval and answer path
3. method, figure, equation, and reproduction questions are not yet consistently skill-shaped

### 3.3 Concrete Example: SCINeRF

SCINeRF remains a useful diagnostic paper.

Current lessons:

1. re-conversion alone did not fix its broken reference section
2. this confirms that reference extraction is an independent asset problem
3. body evidence and locate quality must not depend on the reference tail being perfect

This paper should remain in the regression set as a known hard case.

## 4. Product-Level Decisions To Lock Before Coding

The next implementation cycle should treat the following decisions as fixed.

### 4.1 One Authoritative Mainline

Paper Guide should follow one mainline:

`question -> router/skill -> bound-paper retrieval -> rescue -> support resolution -> answer segments -> provenance -> locate`

This implies:

1. answer generation does not invent its own evidence source late in the pipeline
2. provenance is produced from the same support decisions that shaped the answer
3. locate is produced from the same evidence object, not guessed again in the frontend

### 4.2 Evidence Semantics Must Be Explicit

Every important answer segment should be classified as one of:

1. `direct`
2. `synthesis`
3. `not_stated`

Rules:

1. `direct` can offer precise locate actions
2. `synthesis` may offer grouped evidence, but should not pretend to be one exact sentence
3. `not_stated` can only be emitted after the full no-hit gate has been passed

### 4.3 False Misses Need A Stronger Gate

The system must not return "not found" or "not stated" after only coarse retrieval failure.

Required gate:

1. coarse retrieval
2. bound-paper rescue
3. skill-specific exact resolver, when the question family supports it
4. only then may the system answer `not_stated` or `paper_not_found_support`

### 4.4 Citation Lookup Must Prefer Body Evidence

For `citation_lookup`, the primary target is the sentence in the body that uses `[n]`.

Reference entry lookup is secondary.

If the body-side support is clear but the reference entry is unreliable:

1. answer the body-side support confidently
2. mark the reference entry as unavailable or low-confidence
3. do not fabricate or overconfidently map the wrong entry

### 4.5 Locate Must Be Evidence Navigation, Not Search

Reader locate should be based on structured targets, not document-wide fuzzy searching.

The locate contract should be:

1. `primary_block_id`
2. optional `anchor_id`
3. optional `selection_range`
4. optional `related_block_ids`
5. optional `alternative_targets`
6. `exactness`

Rules:

1. if a structured target exists, frontend does not re-rank it
2. fallback may only search within constrained alternatives
3. exact block miss must not degrade into whole-document fuzzy searching in strict paths

### 4.6 Asset Separation Must Become Explicit

Paper Guide should stop treating the final `.en.md` as the only truth source.

The long-term asset split should be:

1. `SourceBlockGraph`
2. `ReferenceCatalog`
3. `FigureEquationIndex`

Phase implication:

1. body evidence should already move toward `SourceBlockGraph` primacy
2. reference failures should be isolated to citation and bibliography-specific flows

## 5. Architecture Blueprint

### 5.1 Online Runtime Layers

#### Layer 1. Router

Input:

1. user question
2. conversation mode
3. bound paper metadata
4. optional UI intent, such as locate-only or ask-from-selection

Output:

1. question family
2. evidence strictness
3. whether exact resolver should be forced
4. whether locate-only support is allowed

#### Layer 2. Coarse Retrieval

Responsibilities:

1. gather broad candidate blocks within the bound paper
2. preserve original query signal
3. preserve translated query signal
4. preserve family-specific expansion without letting it overwrite the literal ask

Decision:

Do not replace the original query with the translated query as the only effective ask for paper-guide exact questions.

#### Layer 3. Bound-Paper Rescue

Responsibilities:

1. perform full source-block rescue inside the bound paper
2. run even when weak same-paper hits already exist, if those hits do not satisfy the question
3. promote rescued evidence into the answer path rather than leaving it as optional metadata

This layer is the main protection against false misses.

#### Layer 4. Skill-Specific Exact Resolver

Applies especially to:

1. exact method detail
2. exact citation support
3. equation support
4. figure caption or panel support
5. reproduction facts

Responsibilities:

1. scan the bound paper with family-specific cues
2. select evidence at block level
3. return support with confidence and reason codes

#### Layer 5. Support Resolution

Responsibilities:

1. consolidate raw hits into authoritative support slots
2. assign evidence mode
3. assign primary and related blocks
4. generate locate targets from support, not from answer text

#### Layer 6. Answer Segmentation

Responsibilities:

1. produce answer segments rather than one opaque paragraph
2. attach support metadata per segment
3. avoid over-claiming precision when support is synthesis-only

#### Layer 7. Reader Navigation

Responsibilities:

1. receive one structured locate target
2. resolve block and range in the reader DOM
3. surface constrained alternatives only when the primary target cannot be resolved

### 5.2 Offline Asset Layers

#### `SourceBlockGraph`

Stores:

1. stable block ids
2. heading hierarchy
3. page references
4. block types, such as paragraph, heading, figure, caption, equation, table
5. neighbor relationships

#### `ReferenceCatalog`

Stores:

1. cleaned bibliography entries
2. citation numbers or labels
3. extraction confidence
4. optional link back to source pages or blocks

#### `FigureEquationIndex`

Stores:

1. figure numbers and panels
2. captions
3. equation numbers
4. local explanatory neighbors, such as `where` sentences

## 6. Implementation Phases

### P0. Retrieval Truthfulness And No-Hit Gate

Goal:
Stop false misses from breaking user trust.

Required outcomes:

1. bound-paper questions do not return `not_stated` until rescue is exhausted
2. original query signal is preserved alongside translated and expanded forms
3. rescued hits can survive into answer generation

Primary modules:

1. `kb/retrieval_engine.py`
2. `kb/paper_guide_retrieval_runtime.py`
3. `kb/task_runtime.py`
4. `kb/paper_guide_answer_selection.py`

Acceptance:

1. representative exact questions on bound papers no longer fail only because coarse retrieval missed
2. "paper contains answer but model says no" cases drop materially in the regression set

### P1. Support Resolution And Authoritative Locate

Goal:
Make answer, provenance, and locate come from one source of truth.

Required outcomes:

1. support resolution decides the authoritative evidence block
2. locate target is emitted from backend support resolution
3. frontend stops re-ranking structured locate

Primary modules:

1. `kb/paper_guide_answer_post_runtime.py`
2. `kb/paper_guide_provenance.py`
3. `kb/generation_answer_finalize_runtime.py`
4. `api/chat_render.py`
5. `web/src/components/chat/MessageList.tsx`
6. `web/src/components/chat/reader/useReaderLocateEngine.ts`

Acceptance:

1. repeated clicks open the same reader target
2. exact evidence locate does not silently drift to a guessed paragraph
3. strict locate failures are explicit rather than hidden behind whole-document fuzzy fallback

### P2. Skill-First Deep Reading

Goal:
Make reading guidance feel deliberate instead of generic.

Required outcomes:

1. question families route to different evidence and answer behavior
2. method, figure, equation, citation, and reproduction questions have specialized paths
3. answer style matches the actual question type

Primary modules:

1. `kb/paper_guide_prompting.py`
2. `kb/paper_guide_direct_answer_runtime.py`
3. `kb/task_runtime.py`
4. family-specific helper modules as needed

Acceptance:

1. exact fact questions stop degrading into vague summaries
2. overview questions remain concise without losing evidence links
3. citation questions prioritize body evidence

### P3. Asset Separation For Reference Stability

Goal:
Decouple body evidence quality from bibliography extraction quality.

Required outcomes:

1. reference extraction failures do not break normal body evidence locate
2. citation lookup can degrade gracefully when bibliography parsing is weak
3. future converter work has a clear contract for downstream paper-guide consumers

Primary modules:

1. converter and indexing pipeline modules
2. `kb/reference_index.py`
3. reader/reference presentation surfaces

Acceptance:

1. broken reference tails do not poison non-reference question families
2. citation lookup transparently reports low-confidence reference entries

## 7. Default Decisions By Question Family

### 7.1 `doc_map`

1. primary evidence is heading-level
2. default locate goes to section heading, not sentence-level range
3. summary is allowed, but section pointers should stay concrete

### 7.2 `overview`

1. synthesis is allowed
2. key contributions should still map to direct evidence blocks where possible
3. locate defaults to the strongest supporting block, not the most generic section

### 7.3 `method_trace`

1. exact implementation questions should force rescue plus exact resolver
2. if the method spans multiple blocks, one primary block plus related blocks is preferred
3. reproduction-style facts inside methods are `direct` only if explicitly stated

### 7.4 `equation_explain`

1. equation block is primary
2. `where` text and variable explanations are related blocks
3. do not answer equation questions from a generic nearby summary paragraph when the equation exists

### 7.5 `figure_walkthrough`

1. caption block is primary by default
2. panel-specific questions should preserve panel hints
3. nearby body discussion may be related evidence, not the default primary locate

### 7.6 `citation_lookup`

1. body citation sentence is primary
2. bibliography entry is secondary
3. if bibliography mapping is uncertain, say so explicitly

### 7.7 `reproduce_checklist`

1. use a structured answer layout
2. every filled item should be `direct`
3. missing fields should become `not_stated`, not guesses

### 7.8 `compare_limits`

1. synthesis is allowed
2. every comparison point should still cite at least one direct block
3. limitations should prefer explicit author-stated text over inferred weaknesses

## 8. Acceptance Criteria

The feature should not be considered stable until all items below are satisfied.

### 8.1 Retrieval Truthfulness

1. a bound-paper exact question is not marked as unsupported if a valid evidence block exists in the paper
2. fallback search is judged by evidence adequacy, not just by whether some same-paper hits were found

### 8.2 Evidence Integrity

1. direct claims are supported by direct evidence blocks
2. synthesis claims are labeled or treated as synthesis
3. missing information claims are emitted only after the no-hit gate is exhausted

### 8.3 Locate Correctness

1. repeated clicks resolve to the same target
2. structured locate is not front-end reinterpreted
3. strict locate does not degrade into whole-document guessing

### 8.4 Citation Correctness

1. in-paper citation questions can locate the body-side citation sentence
2. broken bibliography extraction does not force a wrong answer
3. low-confidence bibliography mapping is surfaced honestly

### 8.5 Deep Reading Utility

1. overview answers help users orient themselves in the paper
2. method and reproduction answers can point to exact details
3. figure and equation answers open the reader at the right local context

## 9. Regression Set Blueprint

Before implementation starts, the team should treat the regression set as a product asset, not a side effect.

### 9.1 Paper Selection Rules

Use at least:

1. one paper with broken or noisy references, such as SCINeRF
2. one paper with dense equations
3. one paper with figure-heavy explanation
4. one paper with many experimental details
5. one paper with long related-work or citation-heavy sections

### 9.2 Required Question Buckets Per Paper

For each regression paper, cover:

1. one overview question
2. one section-navigation question
3. one exact method detail question
4. one equation or figure question when applicable
5. one citation question
6. one reproduction detail question
7. one "not stated" control question

### 9.3 Regression Output To Save

For each run, save:

1. question
2. answer text
3. evidence mode
4. primary block id
5. related blocks
6. locate target
7. whether the reader landed correctly
8. whether the bibliography mapping was trusted or degraded

## 10. Pre-Code Checklist

The following checklist should be completed or explicitly accepted before code changes begin.

### 10.1 Product Checklist

1. confirm that Paper Guide is scoped to bound-paper reading first
2. confirm that false-miss reduction is the top priority
3. confirm that authoritative locate is more important than broader UX polish in the next phase

### 10.2 Architecture Checklist

1. lock the mainline from retrieval to locate
2. lock `direct / synthesis / not_stated` as the evidence semantics
3. lock body-first citation grounding
4. lock block-first locate contract

### 10.3 Evaluation Checklist

1. define the regression paper set
2. define the representative question set
3. capture current baseline failures before modifications
4. agree on pass or fail criteria per question family

### 10.4 Implementation Checklist

1. start with P0, not with UI polish
2. keep frontend locate logic from growing while backend contract is being tightened
3. avoid mixing reference-pipeline fixes with core false-miss fixes unless required

## 11. File Map For The Next Coding Cycle

Use this as the default ownership map for implementation planning.

Retrieval and no-hit gate:

1. `kb/retrieval_engine.py`
2. `kb/paper_guide_retrieval_runtime.py`
3. `kb/retrieval_heuristics.py`
4. `kb/task_runtime.py`
5. `kb/paper_guide_answer_selection.py`

Support, provenance, and answer finalization:

1. `kb/paper_guide_answer_post_runtime.py`
2. `kb/paper_guide_provenance.py`
3. `kb/generation_answer_finalize_runtime.py`
4. `api/chat_render.py`

Reader and locate:

1. `web/src/components/chat/MessageList.tsx`
2. `web/src/components/chat/PaperGuideReaderDrawer.tsx`
3. `web/src/components/chat/reader/readerTypes.ts`
4. `web/src/components/chat/reader/useReaderLocateEngine.ts`
5. `api/routers/references.py`

Reference pipeline and bibliography behavior:

1. `kb/reference_index.py`
2. converter and indexing modules
3. any reader-side reference presentation surfaces

## 12. Definition Of Done For This Blueprint

This blueprint is considered ready to execute if the team accepts the following statements:

1. the product target is now clear
2. the current gaps are now explicit
3. the next coding cycle will prioritize false misses and authoritative locate first
4. the regression set and acceptance criteria will be prepared before large code changes
5. further patching will be judged against this blueprint instead of ad hoc behavior fixes

## 13. Recommended Immediate Next Step

Before code changes, complete one short baseline capture pass:

1. pick the regression papers
2. freeze the question list
3. run the current system and save failures by family
4. start implementation from P0 only after the baseline is written down

That baseline pass is the last missing preparation step before coding.
