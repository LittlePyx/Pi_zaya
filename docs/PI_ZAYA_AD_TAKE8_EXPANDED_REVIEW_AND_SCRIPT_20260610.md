# π-zaya Take8 Expanded Review And Script

Date: 2026-06-10

Production implementation:

- Use `docs/PI_ZAYA_AD_TAKE8_PRODUCTION_SCRIPT_20260610.md` as the executable shooting and editing script.
- This review document explains what needs to change and why.
- The production script explains exactly how to record it: shot order, click sequence, narration, subtitles, overlay placement, retake priority, and acceptance checks.

Reviewed sources:

- `docs/ad_assets/pi_zaya_20260609/pi_zaya_ad_take7_15min_1280x800_clean_v3_pi_greek_20260610.mp4`
- `docs/ad_assets/pi_zaya_20260609/pi_zaya_ad_take7_directors_cut_9min_1280x800_v4_pi_greek_20260610.mp4`
- `docs/ad_assets/pi_zaya_20260609/pi_zaya_ad_take7_short_60s_1280x800_v4_pi_greek_20260610.mp4`

## Professional Verdict

The current videos are strong product demos, but not yet the strongest possible product advertisement.

What works:

- The core story is understandable: ask the library, inspect references, verify in Reader, collect papers, export references.
- The `1280x800` canvas solves the earlier right-side clipping problem.
- The `π-zaya` brand spelling is now correct.
- The single-page Reader is correctly shown instead of the side-drawer Reader.
- The literature basket and export surface communicate real product value.

What still weakens the ad:

- The current long master does not fully show the ingestion/conversion/quality-repair pipeline, which is one of the project's more defensible technical advantages.
- The Reader section still contains the old recorded evidence strip with LaTeX/Markdown residue. The recording script has been fixed, but the video pixels remain old.
- The dark bottom subtitle sometimes covers the chat composer, send button, or lower shelf details. The script has been fixed for future recordings, but v3/v4 still show this in several places.
- The story ends at "export references", but a stronger research workflow should end with a visible writing-ready output: selected sources, exported citations, and a short related-work outline or note.
- Some jumps are causal but not explicit enough: the basket grows, but the viewer should see more count changes and source origins.

## Feature Details Worth Adding

### 1. PDF Intake And Quality Lifecycle

Add a 90-120 second segment near the start:

- drag/drop or upload PDF,
- status: uploaded, converting, ingesting, ready,
- background high-quality conversion,
- retry/cancel quality job,
- duplicate or renamed PDF state if available.

Why it matters:

This proves π-zaya is not just a chat UI. It is a PDF-to-knowledge-base pipeline.

### 2. Conversion Quality Center And Repair

Add a close-up of:

- source quality score,
- pending/repair recommendation,
- "修复转换" or repair queue,
- before/after quality result if available,
- Reader locate repair or source auto-repair warning.

Why it matters:

This is a moat feature. Many tools can chat with PDFs; fewer can notice conversion damage, plan repair, preserve page markers, and keep Reader evidence reliable.

### 3. Query Scope Switching

Show the same research task through three scopes:

- 全库: discover candidate papers.
- 本文: verify one selected paper.
- 文献篮: synthesize only selected sources.

Why it matters:

The scope control is a workflow concept, not just a UI toggle. It tells viewers π-zaya can move from discovery to verification to writing.

### 4. Answer Quality And Evidence Confidence

Add a short shot of answer-quality diagnostics if visually available:

- structure completeness,
- evidence coverage,
- failed/weak evidence indicators,
- "证据匹配置信度偏低" or similar warning when appropriate.

Why it matters:

This strengthens trust. The product should not look like it always pretends to know.

### 5. Reference Card Depth

Add 30-45 seconds of reference/citation cards:

- DOI,
- IF/JCR,
- GB/T and BibTeX copy,
- evidence focus,
- original evidence,
- external metadata warning,
- Crossref/OpenAlex/Semantic Scholar summary status.

Why it matters:

The current video shows citation cards, but not enough of the "why this reference is reliable or risky" layer.

### 6. Reader Advanced Interaction

Add a richer Reader sequence:

- evidence tab,
- chapter outline,
- highlights count,
- previous/next evidence,
- select text,
- highlight selection,
- ask about selected text,
- add selected text to basket,
- mark evidence useful / needs check,
- candidate fallback or manual alternate when exact locate is not available,
- image zoom only if it is clean and useful.

Why it matters:

Current Reader footage proves "open original text". A stronger ad proves "read, annotate, question, and collect while staying anchored to evidence."

### 7. Literature Basket Advanced Workflow

Add:

- visible basket count change after each add,
- origins: answer citation, Reader citation, Reader selection, reference table,
- tag filter and group by tag,
- batch tagging,
- metadata repair/preflight before export,
- snapshot save and restore,
- remove one item and recover from snapshot.

Why it matters:

The basket becomes a research staging area, not just a list.

### 8. Library Organization And Reference Sync

Add a library segment:

- search by title/tag/category/note,
- category edit,
- tag edit,
- batch classification,
- reference sync running/finished,
- metadata editor for one paper.

Why it matters:

It shows π-zaya can manage a living literature library, not only answer questions from one corpus snapshot.

### 9. Data Protection And Diagnostics

Optional but valuable for a serious product demo:

- API/model connection status,
- automatic backup,
- manual backup,
- backup verification,
- restore dry run,
- diagnostics package export,
- restore audit timeline.

Why it matters:

This makes the product feel operationally mature. Keep it short: 30-45 seconds.

### 10. Writing-Ready Output

Add a final "what the researcher gets" scene:

- switch scope to 文献篮,
- ask: `根据文献篮中已选文献，给我一个 related work 小节提纲，并列出每一点对应的引用。`
- show a short outline,
- show exported BibTeX/GB/T or Markdown references.

Why it matters:

The story should not end at "I collected references"; it should end at "I am ready to write."

## Occlusion And Framing Issues

Must fix in take8:

- Bottom subtitle must not cover the chat composer or send button.
- Bottom subtitle must not cover shelf export controls.
- Reader subtitle should not cover the paragraph currently being discussed.
- Top-left chapter card should not cover the first line of answer text; place it above content only during transitions, then hide it faster.
- Top-right flow pill sometimes competes with right panel title. Keep it narrower or make it fade after chapter transition.
- Cursor should not land on brand text, citation title, or the exact line being highlighted.
- Reader evidence strip must use clean natural-language text. Avoid raw LaTeX/Markdown.

Already acceptable:

- Right literature basket is not clipped at `1280x800`.
- Main content is readable.
- Brand cards are clean and use `π-zaya`.
- No black frames were detected in v3/v4.

## Recommended Take8 Length

If length is not a concern, make take8 about `22-25 minutes`.

This should be positioned as a full product walkthrough ad, not a short social ad. The length is justified if every section answers: "What does this help a researcher do?"

## Take8 Expanded Storyline

### 00:00-00:08 Brand

`π-zaya`

Subtitle: `从 PDF 文献库提问，到可追溯研究证据。`

### 00:08-01:20 Research Problem

Show the task:

`我要写 single-photon imaging 噪声建模的 related work，需要找到代表性做法并核对原文证据。`

### 01:20-03:30 PDF Intake And Library Readiness

Show upload / converting / ingesting / ready / high-quality background conversion.

### 03:30-05:20 Quality Center And Repair

Show conversion quality, pending item, repair recommendation, repair queue/result.

### 05:20-06:40 Library Organization

Show categories, tags, search, batch classification, metadata edit.

### 06:40-08:40 Full-Library Question

Ask:

`单光子成像为什么需要专门建模噪声？有哪些代表性做法？`

Show: type, scope 全库, send, retrieval/generation, answer.

### 08:40-10:20 Evidence And Reference Cards

Show reference positioning, DOI, IF/JCR, evidence focus, GB/T/BibTeX.

Add 1-2 references to basket and show count changes.

### 10:20-12:20 Scope Switch

Switch:

- 全库: discovery,
- 本文: verify,
- 文献篮: synthesize selected papers.

Use one short question per scope.

### 12:20-15:40 Single-Page Reader Deep Use

Show:

- clean evidence strip,
- sections,
- evidence tab,
- select text,
- highlight,
- ask selection,
- add selection to basket,
- useful/check feedback,
- previous/next evidence.

### 15:40-18:40 Literature Basket

Show:

- multiple origins,
- details expanded,
- tags,
- group/filter by tag,
- select 3 papers,
- snapshot save and restore.

### 18:40-20:20 Export And Metadata Preflight

Show:

- selected/current/all scopes,
- BibTeX/RIS/Markdown/GB/T/CSV,
- metadata preflight/autofill if visible.

### 20:20-21:40 Writing-Ready Output

Ask in 文献篮 scope:

`根据已选文献，生成 related work 小节提纲，并标注每一点对应引用。`

Show concise outline.

### 21:40-22:40 Trust And Operations

Quick montage:

- auto backup,
- diagnostics export,
- restore dry run/audit if visually available.

### 22:40-23:00 Ending Card

`π-zaya`

`让你的 PDF 文献库真正可问、可查、可引用。`

## Final Recommendation

Current best assets:

- Complete master: v3, `15:03`.
- Public long cut: v4 director's cut, `9:10`.
- Short cut: v4 short, `58s`.

Best next move:

Record take8 as a longer `22-25 min` expanded walkthrough. Do not merely add more time; add the missing product details that prove π-zaya is a complete research system: ingestion, quality repair, scope switching, Reader annotation, basket metadata repair, snapshot restore, and writing-ready output.
