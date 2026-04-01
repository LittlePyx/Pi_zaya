# Paper Guide Converter Reconversion QA Checklist

Updated: 2026-04-01
Status: active
Scope: required validation steps whenever converter logic or converter-fed runtime contracts change

## 1. Purpose

This checklist exists to prevent a common failure mode:

1. a converter change appears to help Paper Guide
2. the new converted artifacts are not fully rechecked against the original PDF
3. hidden regressions appear in images, equations, references, or text completeness
4. runtime logic is then tuned on top of degraded assets

Use this checklist whenever changes touch:

1. `kb/converter/*`
2. `kb/source_blocks.py`
3. `kb/reference_index.py`
4. shared figure/equation/reference identity contracts
5. runtime code that depends on newly produced converter fields

## 2. Required Inputs

Before validation begins, record:

1. changed files
2. touched papers
3. original source PDF path for each paper
4. output markdown path or db directory for each paper
5. relevant paper-guide manifest or focused test cases

## 3. Reconversion Rules

For every touched representative paper:

1. reconvert from the original PDF after the code change
2. regenerate any dependent structured assets
3. ensure downstream tests use the new artifacts
4. clear or bypass stale caches when necessary

Validation is incomplete if it only uses:

1. old `.en.md` files
2. old `assets/` sidecars
3. old source-block caches
4. fixture-only examples

## 4. PDF-To-Artifact Review Checklist

Compare the reconverted artifacts directly against the PDF.

Review all of the following:

1. document structure
   - section order is preserved
   - headings are not dropped or merged
   - section boundaries still make sense
2. images and figures
   - figures appear in the right place
   - captions are paired with the correct figure
   - figure numbering matches the PDF
   - multi-figure pages do not swap figure identities
3. equations
   - equations are present where expected
   - numbering is preserved when visible in the PDF
   - nearby variable definitions or explanatory text are not lost
4. references
   - the tail list is present
   - numbering is continuous when the PDF is continuous
   - entries are not obviously truncated or merged
5. in-body citations
   - citation markers remain in the body text
   - citation-rich paragraphs are not broken apart incorrectly
6.正文文本完整性
   - paragraphs are not silently cut off
   - lines are not dropped between pages or columns
   - unrelated content is not fused into one block
7. markdown usability
   - output is still readable to a human
   - added metadata or repairs do not make the markdown messy

## 5. Runtime-On-New-Artifacts Review Checklist

Run Paper Guide against the reconverted paper, not old artifacts.

Check at least:

1. one figure question
2. one equation question if applicable
3. one citation lookup question
4. one exact method or reproduce detail question
5. one locate-sensitive question

Record for each:

1. whether the answer is correct
2. whether the evidence is grounded
3. whether the locate target is correct
4. whether the result depends on the new converter fields

## 6. Blockers

Stop and fix before proceeding if any of the following appear:

1. a figure or caption is missing or mismatched
2. an equation or its local explanation disappears
3. references are more truncated or less continuous than before
4. citation markers are lost from relevant body text
5. main-body text is truncated or materially incomplete
6. locate now depends on degraded block structure
7. Paper Guide only passes because runtime is compensating for obviously worse assets

## 7. Required Output Of Each Validation Cycle

Leave behind a short written record containing:

1. source PDF paths used
2. reconverted papers
3. reviewed output paths
4. converter quality findings
5. Paper Guide findings on the new artifacts
6. pass/fail decision
7. known residual risks

## 8. Practical Rule

If a converter change is real, it must survive this sequence:

1. change code
2. find the actual PDF
3. reconvert the paper
4. compare output back to the PDF
5. rerun Paper Guide on the new artifacts

If that full loop has not happened, the change is not fully validated.

