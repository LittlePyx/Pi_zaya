# Task Runtime Split Architecture Plan

Status: `slice-1 landed -> slice-11 landed`
Last updated: `2026-03-24`

## 1. Executive Summary

The current backend is not built on LangChain, LlamaIndex, or a similar RAG orchestration framework.

It is a custom runtime built on:

1. FastAPI + React for interface and transport
2. custom retrieval / deep-read / rerank logic
3. custom paper-guide grounding, citation, provenance, and locate logic

That direction is still correct for this product.

The main problem is not framework choice. The main problem is that too much product logic currently lives inside `kb/task_runtime.py`.

## 2. What This System Is

Current architecture, in practice:

1. `Interface layer`
   - FastAPI routers
   - React chat / reader UI
2. `Runtime orchestration layer`
   - request shaping
   - retrieval wiring
   - model-call sequencing
   - storage / streaming handoff
3. `Retrieval layer`
   - BM25 retrieval
   - deep-read expansion
   - heading / target heuristics
4. `Grounding layer`
   - paper-guide focus selection
   - support slots
   - structured cite validation / repair
   - provenance binding
5. `Presentation layer`
   - rendered citations
   - locate entry exposure
   - figure/library jump surfaces

The codebase already contains these layers conceptually, but the implementation boundary is weak.

## 3. Why Not LangChain Or LlamaIndex

Using a general framework as the production runtime is not recommended right now.

Reasons:

1. the hard product logic here is not generic RAG orchestration
2. paper-guide requires tight control over:
   - bound-paper retrieval
   - section / figure / box targeting
   - in-paper citation grounding
   - support-slot resolution
   - provenance-backed locate entry generation
3. moving the main flow into LangChain or LlamaIndex would likely keep the same custom logic, but spread it across callbacks, node graphs, or framework adapters
4. that would increase indirection before reducing complexity

Recommended position:

1. keep the production runtime custom
2. do not migrate the main paper-guide path to LangChain or LlamaIndex
3. only consider external frameworks for isolated prototypes, not the core answer path

## 4. Real Problem

`kb/task_runtime.py` is acting as a god-orchestrator.

Current issues:

1. paper-guide prompt shaping, targeting, grounding, cite repair, sanitization, and orchestration are mixed together
2. small product fixes require editing distant helper clusters in the same file
3. unit tests for pure paper-guide behavior often import the full runtime module
4. reasoning about regression blast radius is harder than it should be

This is a maintainability problem, not a missing-framework problem.

## 5. Target Internal Architecture

Keep one thin orchestration file and move paper-guide behavior into explicit domain modules.

Target layout:

1. `kb/task_runtime.py`
   - high-level orchestration only
   - retrieval / model / storage sequencing
   - top-level feature gates
2. `kb/paper_guide_prompting.py`
   - prompt-family detection
   - section / figure / box target parsing
   - retrieval prompt augmentation
   - prompt-local heading hints
3. `kb/paper_guide_focus.py`
   - method / figure focus-term extraction
   - direct abstract answer helpers
   - focus excerpt selection and repair
4. `kb/paper_guide_grounding_runtime.py`
   - support-slot construction
   - support-marker injection / resolution
   - cite candidate ordering
   - paper-guide cite fallback / drop rules
5. `kb/paper_guide_postprocess.py`
   - answer sanitization
   - user-facing cleanup
   - negative-shell cleanup
   - structured-token cleanup
6. `kb/paper_guide_provenance.py`
   - provenance binding and locate identity logic
   - should stay separate from prompt shaping and cite surfacing

## 6. Dependency Rules

The split only helps if dependency direction stays clean.

Rules:

1. `task_runtime.py` may import extracted paper-guide modules
2. extracted paper-guide modules must not import `task_runtime.py`
3. shared regex/constants should move only when they are truly shared
4. do not create a new god-module by dumping everything into one `paper_guide_utils.py`
5. prefer a few small public entrypoints over many cross-calling helpers

## 7. Migration Strategy

Use small vertical slices.

### Slice 1

Scope:

1. move paper-guide answer sanitization into `kb/paper_guide_postprocess.py`

Status:

1. landed

### Slice 2

Scope:

1. move prompt-family detection
2. move section / figure / box target parsing
3. move retrieval prompt augmentation
4. keep runtime call sites unchanged other than imports

Status:

1. landed

### Slice 3

Scope:

1. move focus-term extraction
2. move focus excerpt selection
3. move direct abstract-answer helpers

Risk:

1. medium
2. touches more retrieval / answer-shaping branches

Status:

1. partially landed
2. `kb/paper_guide_focus.py` now owns:
    - prompt/snippet trimming helpers
    - abstract excerpt extraction
    - direct abstract-answer builder
    - bound-paper abstract loading helper
    - bound method-focus and figure-caption extraction helpers
    - method detail excerpt/scoring helpers
    - caption fragment helpers
    - special-focus excerpt extraction
    - special-focus block construction
    - focus-answer repair flows
3. `kb/task_runtime.py` still owns:
   - support-slot / grounding runtime

### Slice 4

Scope:

1. move support-slot and cite-resolution logic into `kb/paper_guide_grounding_runtime.py`

Risk:

1. highest
2. must happen after slices 1 to 3 stabilize

Status:

1. landed
2. `kb/paper_guide_grounding_runtime.py` now owns:
   - cue-token extraction
   - inline reference-spec parsing
   - support-slot claim typing / cite policy
   - support-slot block matching
   - support-slot construction
   - support-slot prompt-block rendering
   - support-slot selection by line context
   - support-marker injection
   - support-marker resolution
   - support-slot ref-resolution policy
3. `kb/task_runtime.py` still owns:
   - thin compatibility wrappers and orchestration call sites only

### Slice 5

Scope:

1. move paper-guide answer-hit scoring
2. move heading-focus cleanup for ranked hits
3. move output-mode stabilization
4. move answer-hit selection

Status:

1. landed
2. `kb/paper_guide_answer_selection.py` now owns:
   - output-mode stabilization for paper-guide families
   - focused heading extraction for ranked hits
   - answer-hit scoring
   - answer-hit selection
   - generic answer-hit fallback assembly
   - anchor-grounded answer-hit detection
3. `kb/task_runtime.py` now keeps thin wrappers only

### Slice 6

Scope:

1. move paper-guide context-record construction
2. move deep-read context merge orchestration
3. move prompt-block preparation for paper-guide answer generation
4. move source-id / source-name primitives to shared helpers

Status:

1. landed
2. `kb/paper_guide_context_runtime.py` now owns:
   - answer-hit to context-record / evidence-card assembly
   - deep-read merge for paper-guide context records
   - prompt-block preparation for evidence cards / support slots / special focus / citation grounding
3. `kb/paper_guide_shared.py` now also owns source-id and source-name helpers used across runtime slices
4. `kb/task_runtime.py` now keeps thin wrappers and top-level sequencing only

### Slice 7

Scope:

1. move system-prompt assembly for answer generation
2. move user-prompt assembly for answer generation
3. keep multimodal history/image payload wiring in `task_runtime.py`

Status:

1. landed
2. `kb/paper_guide_message_builder.py` now owns:
   - system prompt assembly for paper-guide and non-paper-guide generation
   - user prompt assembly for retrieved context / evidence blocks / image note
   - contract-enabled gating returned as prompt-bundle state
3. `kb/task_runtime.py` now keeps only wrapper call sites plus message/history wiring

### Slice 8

Scope:

1. move multimodal user-content assembly
2. move filtered generation-message assembly
3. move paper-guide direct-answer override routing

Status:

1. landed
2. `kb/generation_message_runtime.py` now owns:
   - multimodal history filtering
   - image payload to multimodal user-content conversion
   - final `messages` list assembly
3. `kb/paper_guide_direct_answer_runtime.py` now owns:
   - direct abstract-answer override routing
   - direct citation-lookup override routing
4. `kb/task_runtime.py` now keeps only wrappers and the actual LLM call / streaming lifecycle

### Slice 9

Scope:

1. move paper-guide answer postprocess orchestration
2. keep final citation validation / answer-quality probing in `task_runtime.py`

Status:

1. landed
2. `kb/paper_guide_answer_post_runtime.py` now owns:
   - focus-repair orchestration
   - support-marker / support-resolution orchestration
   - cite surfacing orchestration
   - paper-guide answer sanitization orchestration
3. `kb/task_runtime.py` now keeps wrappers plus the final generic validation / quality / persistence path

### Slice 10

Scope:

1. move library-figure asset discovery helpers
2. move library-figure appendix rendering

Status:

1. landed
2. `kb/library_figure_runtime.py` now owns:
   - doc-image path resolution
   - figure-asset collection and cache
   - figure-card scoring / rendering
   - library-figure markdown appendix generation
3. `kb/task_runtime.py` keeps thin wrappers so existing monkeypatch-style tests still work

### Slice 11

Scope:

1. move generation-answer finalize orchestration
2. keep citation-validator implementation in `task_runtime.py`

Status:

1. landed
2. `kb/generation_answer_finalize_runtime.py` now owns:
   - partial-answer normalization / cleanup sequencing
   - contract / kb-miss / paper-guide postprocess sequencing
   - final citation-validation call sequencing
   - answer-quality probe sequencing
3. `kb/task_runtime.py` now keeps the validator implementation plus storage / provenance / status updates

### Slice 12

Scope:

1. move citation-validator implementation and source-ref lookup into a dedicated runtime
2. keep wrapper names stable in `kb/task_runtime.py` so existing monkeypatch-style tests keep working

Status:

1. landed
2. `kb/generation_citation_validation_runtime.py` now owns:
   - source-ref lookup from reference-index docs
   - final structured-citation validation / rewrite / drop logic
   - paper-guide candidate-ref / support-slot aware citation grounding
3. `kb/task_runtime.py` now keeps thin wrappers plus storage / provenance / status updates

### Slice 13

Scope:

1. move live-assistant/task-state helpers plus answer/provenance persistence into a dedicated runtime
2. keep wrapper names stable in `kb/task_runtime.py` so existing provenance tests can keep patching the wrapper surface

Status:

1. landed
2. `kb/generation_state_runtime.py` now owns:
   - live-assistant text helpers
   - generation task state lookup / patch / cancel helpers
   - answer / partial-answer persistence helpers
   - provenance store / async-refine helpers
3. `kb/task_runtime.py` now keeps thin wrappers plus the top-level worker / streaming lifecycle

## 8. Testing Strategy

Every slice should be verified in three layers.

### 8.1 Pure Unit Tests

Goal:

1. extracted module behavior can be tested without importing the full runtime

Examples:

1. prompt-family detection
2. retrieval prompt augmentation
3. section / figure / box target parsing
4. sanitization cleanup

### 8.2 Existing Runtime Regression Tests

Goal:

1. the orchestration call sites still behave the same after extraction

Examples:

1. `tests/unit/test_task_runtime_bg_task.py`
2. `tests/unit/test_task_runtime_answer_contract.py`
3. `tests/unit/test_task_runtime_provenance.py`

### 8.3 Focused Real Smoke Tests

Goal:

1. verify no user-visible regression in paper-guide

Recommended prompt families:

1. overview
2. abstract
3. method exact support
4. figure walkthrough
5. citation lookup

## 9. Acceptance Criteria

An extraction slice is acceptable only if all of the following are true:

1. no user-visible paper-guide regression is introduced
2. `task_runtime.py` loses real ownership, not just dead-code copies
3. the extracted module is importable and testable on its own
4. the extracted module does not import `task_runtime.py`
5. at least one focused regression suite still passes after the move

## 10. Current Status Snapshot

As of `2026-03-24`:

1. the production architecture remains custom, not LangChain/LlamaIndex
2. `kb/paper_guide_postprocess.py` already owns sanitization helpers
3. `kb/paper_guide_prompting.py` now owns prompt-family detection, target parsing, retrieval prompt augmentation, deep-read context merging, evidence-card block rendering, citation-grounding block rendering, and requested-figure fallback helpers
4. `kb/paper_guide_focus.py` now owns the focus/helper slice plus direct-abstract and repair implementation
5. `kb/paper_guide_grounding_runtime.py` now owns the support-slot builder slice, support-marker flow, and grounding primitives
6. `kb/paper_guide_shared.py` now owns shared prompt-snippet trimming and abstract-excerpt extraction used by prompting/grounding
7. `kb/paper_guide_retrieval_runtime.py` now also owns the citation-lookup retrieval glue: raw-target hit selection, citation-lookup fragments/local-ref extraction, citation-lookup scoring/query-token helpers, and direct citation-lookup answer assembly
8. `kb/paper_guide_citation_surfacing.py` now owns post-generation paper-guide cite surfacing: candidate-ref collection, fallback/focus/card citation injection, locate-only cite stripping, and numeric-reference promotion
9. `kb/paper_guide_answer_selection.py` now owns output-mode stabilization, focused heading cleanup, answer-hit scoring, and answer-hit selection
10. `kb/paper_guide_context_runtime.py` now owns paper-guide context-record building, deep-read context merge, and prompt-block preparation for answer generation
11. `kb/paper_guide_message_builder.py` now owns system/user prompt assembly for generation
12. `kb/generation_message_runtime.py` now owns multimodal message payload assembly
13. `kb/paper_guide_direct_answer_runtime.py` now owns direct-answer override routing
14. `kb/paper_guide_answer_post_runtime.py` now owns paper-guide answer postprocess orchestration
15. `kb/library_figure_runtime.py` now owns library-figure appendix generation
16. `kb/generation_answer_finalize_runtime.py` now owns generation finalize sequencing
17. `kb/generation_citation_validation_runtime.py` now owns source-ref lookup and structured-citation validation
18. `kb/generation_state_runtime.py` now owns live-assistant/task-state helpers and answer/provenance persistence
19. `kb/task_runtime.py` is down to `3081` lines / `119618` chars
20. the next preferred extraction target is wrapper-surface reduction or the remaining high-level worker/orchestration glue

## 11. Immediate Next Steps

1. reduce wrapper surface in `kb/task_runtime.py` where extracted modules already own the real implementation
2. focus the next paper-guide slice on higher-level orchestration glue rather than more low-level helper extraction
3. keep adding module-level tests before each extraction slice
4. re-measure file size / line count after each landed slice
