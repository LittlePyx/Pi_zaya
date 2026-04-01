# Paper Guide Pre-Code Baseline Checklist

Updated: 2026-03-31
Status: draft
Scope: baseline capture before the next paper-guide implementation cycle

Use this checklist together with:

- `docs/PAPER_GUIDE_PREIMPLEMENTATION_BLUEPRINT.md`
- `docs/ANSWER_MANUAL_REGRESSION_CHECKLIST.md`

## 1. Goal

Before code changes begin, capture the current paper-guide baseline in a way that makes later improvements measurable.

This checklist is complete only when:

1. the regression paper set is fixed
2. the question set is fixed
3. current failures are recorded by family
4. success criteria are fixed before implementation starts

## 2. Regression Paper Set

Select at least five papers.

Required coverage:

1. one paper with broken or noisy references
2. one paper with dense equations
3. one paper with figure-heavy explanation
4. one paper with many reproduction details
5. one paper with citation-heavy related work

Suggested seed set:

1. `SCINeRF` for broken references and hard locate cases
2. one equation-heavy paper
3. one figure-heavy paper
4. one experiment-heavy paper
5. one citation-heavy paper

Record:

1. paper title
2. source path
3. why it is in the set
4. which question families it must cover

## 3. Required Question Families Per Paper

For each paper, define at least these questions:

1. one `overview` question
2. one `doc_map` or section-navigation question
3. one `method_trace` exact-detail question
4. one `equation_explain` or `figure_walkthrough` question when applicable
5. one `citation_lookup` question
6. one `reproduce_checklist` question
7. one `not_stated` control question

Rules:

1. every question should be phrased as a real user question
2. at least one exact-detail question should use non-literal wording
3. at least one question should require the system to find a body citation sentence

## 4. Record Format Per Case

For each question, record the following.

### 4.1 Input

1. paper id
2. question family
3. user question
4. whether the answer should be `direct`, `synthesis`, or `not_stated`

### 4.2 Observed Output

1. answer summary
2. did the system answer the question
3. did it incorrectly say the paper did not mention the fact
4. did it give citation support
5. did it open the reader
6. did the reader land on the correct block
7. did the answer look overconfident

### 4.3 Evidence Record

1. primary block id if present
2. related block ids if present
3. citation number if applicable
4. reference entry quality if applicable
5. final evidence classification

### 4.4 Failure Tags

Use one or more of:

1. `false_miss`
2. `wrong_block`
3. `wrong_section`
4. `wrong_citation`
5. `citation_body_missing`
6. `reference_tail_broken`
7. `answer_too_vague`
8. `unsupported_but_claimed`
9. `correct_answer_wrong_locate`
10. `not_stated_correct`

## 5. Minimum Success Bar For The Next Coding Cycle

Lock the following goals before implementation begins.

### 5.1 P0 Success Bar

1. false misses materially decrease on bound-paper exact questions
2. rescued evidence survives into answer generation
3. the system no longer exits early just because weak same-paper hits exist

### 5.2 P1 Success Bar

1. structured locate resolves to the same block on repeated clicks
2. frontend no longer re-ranks authoritative locate targets
3. strict locate failure is explicit when exact evidence cannot be resolved

### 5.3 P2 Success Bar

1. method, figure, equation, citation, and reproduction questions no longer feel like one generic path
2. exact questions get exact evidence more consistently
3. overview questions remain concise and readable

## 6. Baseline Run Procedure

1. bind the conversation to one paper
2. ask the fixed question list
3. save raw answer output
4. open the reader for every direct-evidence answer
5. record whether locate is correct
6. record whether the paper really contains the asked fact
7. tag the failure family

If the system says the paper did not mention something:

1. manually verify whether the paper actually contains it
2. if yes, tag the case as `false_miss`

## 7. Exit Condition Before Coding

Do not start implementation until:

1. the regression set is written down
2. the question set is frozen
3. the current baseline failures are captured
4. the team agrees that P0 starts with false-miss reduction, not UI polish

## 8. Suggested Storage

Store baseline records in one of the following:

1. a dedicated markdown log under `docs/`
2. a JSON or JSONL fixture under `tests/` or `tools/manual_regression/`
3. both, if human review and machine replay are both needed

The important part is not the format. The important part is that the baseline is frozen before implementation starts.
