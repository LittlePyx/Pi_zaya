# Paper Guide In-Paper Citation Grounding Plan

Status: `proposed -> phase-1 in progress`  
Last updated: `2026-03-20`

## 1. Problem Statement

The paper-guide feature already supports:

- bound-paper retrieval
- structured citation markers like `[[CITE:<sid>:<ref_num>]]`
- reference-index lookup
- clickable in-paper citation rendering

But the current citation grounding is still too weak.

Observed failure mode:

- the answer repeatedly shows the same in-paper reference such as `【1】`
- the repeated `【1】` often comes from model-side number guessing, not from correct claim-to-reference grounding
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
- reuse the same grounding logic in more than one backend layer

Possible work:

1. Move shared grounding logic into a dedicated reusable module if Phase 1 grows further.
2. Pass candidate reference summaries into the main answer prompt in a more structured way.
3. Add suspicious-citation analytics to answer-quality summaries.

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

## 7. Files To Change

Phase 1 primary files:

- `kb/task_runtime.py`
- `tests/unit/test_task_runtime_answer_contract.py`

Phase 1 optional/helper files:

- `kb/inpaper_citation_grounding.py`

No frontend file is required for correctness in Phase 1.

## 8. Test Plan

### 8.1 Unit Tests

Add or update tests for:

1. SID-header sanitization still removes internal header markers after `DOC-k` migration
2. structured citations still rewrite to locked source when that is correct
3. paper-guide validation rewrites a wrong `ref_num` to the evidence-supported one
4. paper-guide validation drops a suspicious `ref_num` when there is no justified match
5. non-paper-guide behavior remains backward compatible

### 8.2 Manual Checks

Run at least one real paper-guide session and verify:

1. the model is no longer exposed to retrieval-block headers that look like `[1]`
2. answers that previously collapsed to repeated `【1】` no longer do so by default
3. if the answer contains a wrong structured cite, it is either repaired or removed
4. clickable citation chips still render normally

### 8.3 Regression Scope

Must re-run:

1. targeted answer-contract tests
2. targeted reference-metadata / citation-render tests if touched

Recommended command set:

```bash
pytest -q tests/unit/test_task_runtime_answer_contract.py
pytest -q tests/unit/test_reference_metadata_guards.py -k "numeric_citation or structured_cite"
```

## 9. Acceptance Criteria

Phase 1 is accepted only if all of the following are true.

### 9.1 Functional

1. retrieval context headers no longer use `[k] [SID:...]`
2. paper-guide citation validation is no longer pure existence-checking
3. wrong repeated `ref 1` citations can be rewritten or dropped deterministically

### 9.2 Test

1. all new unit tests pass
2. existing targeted citation tests still pass
3. no regression in locked-source citation rewriting

### 9.3 Product-Level

For a representative paper-guide answer:

1. `【1】` does not appear repeatedly just because the first retrieval block was labeled `1`
2. when answer-local evidence points to another reference, the final citation reflects that
3. when grounding is weak, the system avoids confidently wrong in-paper references

## 10. Definition Of Done

This work is done when:

1. the plan is committed into the repo
2. Phase 1 backend changes are landed
3. Phase 1 tests are added and passing
4. the final implementation note records:
   - what changed
   - what was tested
   - what risks remain

## 11. Risks And Tradeoffs

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
