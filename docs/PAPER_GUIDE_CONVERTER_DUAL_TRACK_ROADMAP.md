# Paper Guide + Converter Dual-Track Roadmap

Updated: 2026-04-01
Status: proposed
Scope: coordinated improvement plan for `kb_chat` paper-guide quality and PDF conversion quality

## 1. Core Statement

Paper Guide and the converter must be treated as one product chain, not two separate systems.

The chain is:

`PDF -> structured paper assets -> retrieval/support -> answer/provenance -> reader locate`

If upstream paper assets are unstable, downstream answering and locate quality drift.
If downstream support and locate logic are weak, good conversion assets do not turn into a trustworthy reading experience.

The goal is therefore not:

1. improve Paper Guide while tolerating converter regressions
2. improve converter output while leaving Paper Guide behavior unstable

The goal is:

1. improve both sides together
2. define one shared contract between them
3. reject changes that help one side while materially harming the other

## 2. North-Star Product Outcome

After this roadmap is executed, the user experience should feel like this:

1. the system can explain one bound paper clearly
2. it can answer exact questions even when wording differs from the paper text
3. it can show the supporting paragraph, figure, equation, or citation sentence
4. clicking locate opens the right evidence region reliably
5. if it says the paper did not state something, that result is trustworthy
6. converter output is at least as readable and complete as before, and structurally more useful

This means success must be judged on both:

1. converted artifact quality
2. reading-guide runtime quality

## 3. Non-Negotiable Rules

The following rules should govern all future implementation work.

### 3.1 No single-sided wins

Do not accept a change that improves Paper Guide benchmarks while degrading conversion quality in figures, references, headings, block structure, or markdown readability.

Do not accept a change that improves conversion throughput or artifact formatting while breaking answer grounding, locate stability, or question coverage.

### 3.2 Shared contract first

Paper Guide must consume structured paper assets produced by conversion, not repeatedly rediscover structure from markdown text alone.

### 3.3 One support decision, multiple surfaces

Answer text, citation chips, and locate targets must come from the same support/provenance decision.
They must not independently guess the source again.

### 3.4 Bound-paper questions require bound-paper rescue

In paper-guide mode, coarse retrieval miss is not enough to return `not stated`.
The system must first attempt:

1. coarse retrieval
2. bound-paper rescue
3. skill-specific exact resolver

Only after all three fail may it conclude the paper did not state the fact.

### 3.5 Converter output must remain user-usable

Structural enrichment is not allowed to make converted markdown meaningfully less readable, less complete, or more brittle for human reading.

## 4. Shared Architecture Contract

The converter and Paper Guide should meet at three shared asset layers.

### 4.1 SourceBlockGraph

Primary structured text asset for reading, grounding, and locate.

Minimum fields:

1. `block_id`
2. `kind`
3. `text`
4. `page`
5. `heading_path`
6. `anchor_id`
7. `bbox` when available
8. `neighbor_block_ids`

### 4.2 FigureEquationIndex

Structured identity layer for figures, captions, panels, equations, and nearby explanation blocks.

Minimum figure fields:

1. `figure_id`
2. `paper_figure_number`
3. `figure_ident`
4. `asset_name_raw`
5. `asset_name_alias`
6. `figure_block_id`
7. `caption_block_id`
8. `binding_confidence`
9. `binding_source`

Minimum equation fields:

1. `equation_id`
2. `equation_number`
3. `equation_block_id`
4. `explanation_block_ids`
5. `binding_confidence`

### 4.3 ReferenceCatalog

Structured reference asset, independent from the main body markdown tail.

Minimum fields:

1. `reference_number`
2. `reference_text`
3. `reference_entry_id`
4. `parse_confidence`
5. `source_pages`
6. `tail_continuity_status`

## 5. Dual-Track Workstreams

The roadmap should always be executed on two tracks plus one shared contract layer.

### 5.1 Track A: Converter Quality

Primary responsibility:

1. produce stable structured assets
2. preserve or improve human-readable markdown output
3. prevent figure/reference/block regressions

Current code areas:

1. `kb/converter/page_text_blocks.py`
2. `kb/converter/page_figure_metadata.py`
3. `kb/converter/page_image_markdown.py`
4. `kb/converter/page_local_pipeline.py`
5. `kb/converter/pipeline_render_markdown.py`
6. `kb/converter/layout_analysis.py`

### 5.2 Shared Middle Layer: Asset Normalization

Primary responsibility:

1. convert converter output into authoritative runtime assets
2. keep identity stable across reading, retrieval, and locate

Current code areas:

1. `kb/source_blocks.py`
2. `kb/reference_index.py`
3. future `FigureEquationIndex` and `ReferenceCatalog` helpers

### 5.3 Track B: Paper Guide Runtime

Primary responsibility:

1. route questions by skill
2. retrieve and rescue support evidence
3. generate grounded answers
4. emit authoritative provenance and locate targets

Current code areas:

1. `kb/paper_guide_prompting.py`
2. `kb/paper_guide_retrieval_runtime.py`
3. `kb/paper_guide_answer_selection.py`
4. `kb/paper_guide_answer_post_runtime.py`
5. `kb/paper_guide_direct_answer_runtime.py`
6. `kb/paper_guide_provenance.py`
7. `kb/generation_answer_finalize_runtime.py`

### 5.4 Track C: Reader and UI Consumption

Primary responsibility:

1. honor structured locate decisions
2. distinguish exact locate from figure jump and nearby fallback
3. avoid frontend-side re-guessing when authoritative targets exist

Current code areas:

1. `web/src/components/chat/MessageList.tsx`
2. `web/src/components/chat/reader/useReaderLocateEngine.ts`
3. `web/src/components/chat/reader/readerTypes.ts`

## 6. What Must Improve Together

The following paired goals should be treated as coupled requirements.

### 6.1 Figure quality + figure jump

Converter side:

1. figure numbering and caption binding must become more stable
2. final markdown should include correct image-caption pairs
3. image aliases and figure indices should remain consistent

Paper Guide side:

1. figure questions should bind to the correct figure identity
2. locate should distinguish caption jump from exact text locate
3. panel questions should prefer panel-aware evidence when available

### 6.2 Reference quality + citation grounding

Converter side:

1. reference extraction should be independent from fragile markdown tail parsing
2. broken numbering and truncated entries should be surfaced as quality signals

Paper Guide side:

1. citation lookup should first ground to the body citation sentence
2. reference tail lookup should be secondary and confidence-aware
3. when tail extraction is weak, the answer must degrade honestly instead of hallucinating

### 6.3 Block quality + exact locate

Converter side:

1. heading and paragraph segmentation should remain stable
2. section boundaries should not regress
3. block identity should persist across reconversion

Paper Guide side:

1. exact questions should resolve to the smallest useful evidence block
2. locate should highlight within that block when possible
3. fallback should stay within claim-group alternatives instead of global fuzzy drift

### 6.4 Markdown readability + structured assets

Converter side:

1. structural metadata should not make markdown output noisier or harder to read
2. injected captions, figure aliases, and references should remain clean for humans

Paper Guide side:

1. runtime should prefer structured assets first
2. markdown should remain a readable fallback, not the only truth source

## 7. Phased Roadmap

### Phase 0: Guardrails And Baselines

Objective:

Create dual gates so future work cannot accidentally improve one side while degrading the other.

Converter tasks:

1. freeze a converter regression paper set
2. record current markdown quality, figure coverage, reference quality, and block quality
3. capture known hard papers such as SCINeRF

Paper Guide tasks:

1. freeze a paper-guide regression manifest per question family
2. record current false misses, locate errors, citation errors, and unsupported claims

Deliverables:

1. paper set manifest
2. question set manifest
3. converter baseline report
4. paper-guide baseline report

Exit gate:

No major code change proceeds without both baselines being reproducible.

### Phase 1: Converter Identity Hardening

Objective:

Make conversion emit stable identities that runtime can trust.

Converter tasks:

1. complete figure identity hardening
2. add or harden equation identity and nearby explanation binding
3. split reference extraction into a structured `ReferenceCatalog`
4. enrich block metadata without degrading markdown output
5. ensure no-LLM and LLM paths share the same postprocess contract

Paper Guide tasks:

1. update runtime to consume figure/equation/reference identity directly
2. stop rediscovering these identities from plain markdown when structured identity exists

Exit gate:

1. converter suites improve or hold on figure/reference/block metrics
2. paper-guide figure/equation/citation suites improve or hold on grounding metrics

### Phase 2: Retrieval And False-Miss Elimination

Objective:

Reduce cases where the paper contains the fact but the system says it does not.

Converter tasks:

1. ensure block text and section metadata remain suitable for bound-paper scanning
2. expose enough local context for exact resolvers to inspect nearby blocks

Paper Guide tasks:

1. keep original query, translated query, and family augmentation in parallel rather than replacement-only
2. strengthen bound-paper rescue even when weak same-paper hits already exist
3. promote rescued exact evidence into answer hits so it survives later selection
4. expand skill-specific exact resolvers for method, reproduce, equation, figure, and citation

Exit gate:

1. false-miss rate materially decreases
2. converter readability and artifact integrity do not regress

### Phase 3: Authoritative Provenance And Locate

Objective:

Turn locate into evidence navigation instead of post-hoc search.

Converter tasks:

1. preserve stable block and anchor identity across reconversion
2. improve figure/equation/reference block linking where gaps remain

Paper Guide tasks:

1. emit one authoritative `primary_block_id` per answer segment when direct evidence exists
2. attach `related_block_ids`, `jump_surface`, and range-level locate metadata
3. classify surfaces as `exact_text`, `figure_caption`, `figure_asset`, or `nearby_fallback`

Reader/UI tasks:

1. stop overriding authoritative locate with broad fuzzy search
2. keep fallback limited to structured alternatives when the primary target fails
3. surface approximate jumps honestly

Exit gate:

1. locate accuracy improves on real click-through checks
2. converter-produced anchor structure remains stable after reconversion

### Phase 4: Skill Coverage And Reading Workflows

Objective:

Make Paper Guide feel like a real reading assistant, not a patched chat mode.

Converter tasks:

1. preserve strong support for sections, tables, figure captions, equation neighborhoods, and references
2. continue improving structured asset completeness

Paper Guide tasks:

1. harden these skill families:
   - `doc_map`
   - `overview`
   - `method_trace`
   - `equation_explain`
   - `figure_walkthrough`
   - `citation_lookup`
   - `reproduce_checklist`
   - `compare_limits`
2. improve answer-style consistency for `direct`, `synthesis`, and `not_stated`

Exit gate:

All core families pass their regression suites with no meaningful converter regression.

## 8. Metrics That Must Be Watched Together

### 8.1 Converter Metrics

Track at least:

1. figure-caption binding rate
2. figure alias/index coverage
3. missing-figure-or-caption rate in final markdown
4. equation block and explanation binding coverage
5. reference continuity and truncation rate
6. broken asset link rate
7. heading and section segmentation quality
8. markdown readability spot-check quality

### 8.2 Paper Guide Metrics

Track at least:

1. false-miss rate on bound-paper exact questions
2. answer correctness by family
3. direct-support grounding rate
4. exact locate accuracy
5. figure/panel jump accuracy
6. citation body-sentence grounding accuracy
7. honest `not_stated` rate
8. unsupported-but-claimed rate

### 8.3 Cross-System Metrics

These are the most important because they show whether both sides are improving together.

Track at least:

1. figure question answered correctly and jumped correctly
2. equation question answered correctly and jumped correctly
3. citation question grounded to body sentence and, when possible, tail entry
4. method detail found despite non-literal wording
5. reconverted paper still passes the same reading-guide cases

## 9. Release Gates

No paper-guide-focused change should be called successful unless:

1. targeted paper-guide suite passes
2. converter smoke suite passes
3. no new converter regressions appear in the touched paper set

No converter-focused change should be called successful unless:

1. targeted converter suite passes
2. paper-guide smoke suite passes on affected question families
3. locate, grounding, or false-miss behavior does not regress on touched papers

For risky changes, require:

1. one focused suite
2. one full baseline suite
3. one reconversion of at least one known hard paper

## 10. Recommended Working Method For Future Changes

Every implementation cycle should follow this rhythm:

1. declare which track is primary for the cycle
2. declare which other track is at risk of regression
3. identify the shared contract being touched
4. run focused tests on the primary track
5. run smoke tests on the secondary track
6. reconvert at least one representative paper when converter contracts change
7. replay at least one real paper-guide manifest on that reconverted paper
8. record both wins and regressions before moving on

This should prevent the common failure mode:

1. patch runtime heuristics to hide bad assets
2. later improve converter output
3. accidentally invalidate the heuristic layer
4. rediscover the same class of bug again

## 11. Mandatory Converter-Change Validation Loop

Whenever a change touches conversion logic or any shared asset contract consumed by runtime, the following loop is required.

### 11.1 Reconvert with the real source PDF

Do not validate converter changes only with cached output, fixture markdown, or previously converted assets.

Required steps:

1. locate the original source PDF for each touched regression paper
2. reconvert the paper from the PDF after the code change
3. use the newly converted output in all relevant downstream checks
4. do not rely on stale `.md`, `.en.md`, figure sidecars, or old source-block caches

### 11.2 Compare output against the PDF, not just against prior markdown

For each reconverted paper, visually and structurally compare the new output against the original PDF.

Required review areas:

1. overall document structure and section order
2. image placement, figure numbering, and caption pairing
3. equation presence, numbering, and nearby explanation text
4. reference list continuity, truncation, and numbering quality
5. in-body citations and whether citation markers survive correctly
6. main-body paragraph completeness
7. heading quality and section boundary correctness
8. obvious text truncation, dropped lines, or merged unrelated content

### 11.3 Run runtime checks on the reconverted paper

After reconversion, the same paper must be exercised through Paper Guide using the new artifacts.

Required checks:

1. replay at least one existing manifest on the reconverted paper
2. prioritize figure, equation, citation, method-detail, and locate cases
3. confirm that runtime is reading the new converted assets rather than old cached ones

### 11.4 Stop if conversion quality regresses

If the reconverted paper shows new truncation, missing figures, broken references, misplaced captions, or obvious structure damage, treat that as a blocker.

Do not accept a runtime improvement that depends on degraded converted assets.

### 11.5 Record findings explicitly

Each converter-touching cycle should leave behind:

1. the source PDF path used for reconversion
2. the output directory or markdown path reviewed
3. the papers rechecked
4. the observed converter quality findings
5. the observed Paper Guide behavior on the new artifacts

## 12. Immediate Next-Step Recommendation

The next implementation wave should explicitly target both:

1. converter-side equation/reference identity hardening
2. runtime-side authoritative locate and citation grounding hardening

Recommended order:

1. finish asset-layer identity for equation and reference objects
2. make provenance depend on those identities
3. tighten frontend consumption of authoritative locate
4. then broaden family coverage and UX polish

## 13. Final Execution Principle

The long-term win is not "make the chat answer a little smarter."

The long-term win is:

1. convert papers into stable, trustworthy reading assets
2. make Paper Guide reason over those assets
3. make reader navigation open exactly those assets

That is the only path that reliably improves:

1. conversion quality
2. answer quality
3. locate quality
4. user trust
