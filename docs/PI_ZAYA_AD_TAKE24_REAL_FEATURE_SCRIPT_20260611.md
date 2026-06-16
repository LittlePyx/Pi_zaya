# Pi-zaya Take24 Real Feature Ad Script

Date: 2026-06-11

This rewrite is based on a fresh review of the React/FastAPI product code and the take23 review sheet. The core change is simple: the ad must prove the product, not just show a polished interface wall.

## Audit Findings

1. The previous take used too many page-wall compositions for proof shots. They look technical, but citation text, evidence text, Reader location, and basket details are too small to verify.
2. The current recorder seeds demo citation cards from repeated `summaryLine` / `whyLine` fields in `record_take13_ad.cjs`. That makes the card look templated even when the UI is capable of better evidence cards.
3. The evidence jump was not visually proven. The viewer saw a Reader page, but not the exact highlighted original evidence in a large enough crop.
4. Conversion quality was implied through Library / Quality Center, but the actual converted result was not shown strongly enough. The ad needs to show the converted Reader/Markdown output itself.
5. Reader copy should stop talking about figure captions as the main value. The stronger value is: converted content is readable, answerable, source-locatable, and reusable in a literature workflow.
6. Quality Center should not be a separate segment in this cut. Library taxonomy is more useful for the current ad: automatic category/tag suggestions, manual edits, and filterable views.
7. Literature basket shots need to show why it is more than a saved list: article summary, source/provenance, tags, open-source action, and "use for next turn" context.

## Priority Feature Story

The ad should emphasize these features in this order:

1. **Conversion result**
   Show a PDF transformed into readable, searchable, locatable knowledge: Library converted row, then Reader with clean title/abstract/body, page markers, equations/figures/tables where available, and reference links.

2. **Research agent, not search box**
   Ask a new task. Show the agent parsing, retrieving, reading, and generating, not just a finished answer.

3. **Answer evidence card**
   Click the strongest source number, preferably `#4` if that is the good card from the user's reference image. The card must show distinct answer claim, original evidence, and location/DOI metadata. Do not use a card where evidence/support/why are all the same seeded sentence.

4. **Jump to original evidence**
   Click "open evidence" or a provenance locate chip. The shot must stop on the Reader with the actual matched paragraph/block visible and highlighted. Required visible UI: `reader-evidence-focus`, `reader-locate-mode`, `reader-locate-resolution` or `reader-locate-status`, and the highlighted source text.

5. **Reference-location card**
   Show what papers were referenced and why. For upstream references, rely on the UI that actually exists: title/authors/venue/DOI, article overview, current-paper usage/takeaway, original reference entry when present, and add/open actions. Do not claim the hidden System-B location/support rows are visible unless the UI is changed.

6. **Literature basket as working context**
   Add a reference, open the shelf, expand one item, show summary quality/source, source trail, tags, and open-source action. Select several papers and click "use for next turn"; the next shot must show the context pack above the input.

7. **DOI / official source**
   Keep this short: click DOI, land on the official page, and hold on "Download PDF" or equivalent. This proves sources are real.

8. **Library taxonomy**
   Show automatic category/tag suggestions, category/tag filters, the row `分类` button, and the metadata drawer where the user can manually edit category, reading status, tags, and notes.

## Take24 Structure

Target length: 95-110 seconds.

## Visual Effect Grammar

The effects should make the product feel cinematic, but every proof shot must remain readable. Use effects as camera language:

1. **Logo kinetic open**
   Rotating zoom logo, then a snap zoom through the logo into the product UI. Keep it under 3 seconds so the ad starts with impact, not decoration.

2. **Page-wall parallax**
   Use a multi-page wall only as a compressed overview. Let tiles move at slightly different speeds, with the main tile 1.4x larger than the others. The large tile should be the evidence card or Reader highlight, not a generic dashboard.

3. **Macro proof zoom**
   For citation cards, zoom from full answer -> source number -> popover. Hold the final crop steady. Add a subtle ring or glow around `原始证据` / evidence, then fade the ring away so text is readable.

4. **Cinematic scroll**
   For Reader conversion, use slow vertical scroll with easing, like a camera passing over a paper. Pause on title/body and again on a structured block. Avoid fast scroll during narration.

5. **Evidence jump snap**
   When jumping to original evidence, use a quick whip/blur transition from the citation card into Reader, then immediately stabilize on the highlighted text. This makes the jump feel intentional and proves the location.

6. **Split-context reveal**
   For the basket segment, show answer/card on the left and shelf on the right for one beat, then push in to the expanded shelf item. This explains "collect from evidence" without drawing a flowchart.

7. **Context pack lift**
   After clicking "use for next turn", use a small upward slide/glow on the context pack above the input. This is the one place where a small effect can make the state change obvious.

8. **Domain-neutral montage**
   If showing "not limited to single-photon imaging", use a quick 4-tile swap of document types/library tags, not a static slogan. Keep it brief.

Effect limits:
- No heavy blur over evidence text.
- No tilted boxes on proof shots.
- No decorative flowchart arrows.
- Do not animate while the viewer needs to read the citation card, Reader evidence, or shelf summary.

## Product-Ad Layering

The final cut should use two layers:

1. **Recording layer**
   Real UI actions only: click, type, filter, open, locate, collect, use for next turn. The recorder now adds a small product strip and a spotlight box so the viewer knows which capability is active without turning the screen into a flowchart.

2. **Post-production layer**
   Page-wall overview, snap zooms, cinematic scroll, macro proof zoom, and context-pack lift. These effects are for transitions and focus; all proof text still needs steady readable holds.

Detailed effect spec:
`docs/ad_assets/pi_zaya_20260609/take24_product_ad_effects.json`

Timing rule:
If TTS has to speed up beyond about `1.08x`, rewrite the line or extend the block. Do not force a fast voice over a short visual hold.

### 0.0-9.0s - Logo And Product Montage

Visual:
- Rotating zoom Pi-zaya logo for about 2.4 seconds.
- Snap into a fast but ordered product montage: answer with source numbers, source #4 card, Reader highlighted evidence, Library taxonomy, DOI page, and literature basket.
- Use a page wall only here; the largest tile should be a readable evidence card or Reader highlight.

On-screen text:
- `PDF library -> research agent`
- `Ask / Trace / Collect / Continue`

Voice:
> Pi-zaya 是 PDF 文献库的 research agent。它把论文转成带定位点的 Markdown，让回答能追到原文证据。

### 9.0-21.0s - Library Taxonomy And Manual Edit

Visual:
- Focus on `文献分类与标签`.
- Show file rows with category/tag pills, `已转换`, `Q100`, `可用于问答`, pages/refs/fig/math.
- Click a category pill and a tag pill to show list filtering.
- Click the row `分类` button to open the metadata drawer.
- Hold on manual category/status/tag/note fields and system suggestions.

Voice:
> 上传论文后，Library 会自动推荐分类和标签；需要调整时，点“分类”就能手动修改。列表里还保留转换状态、页数、参考文献和可问答状态。

### 21.0-32.0s - Reader Conversion Result

Visual:
- Open Reader from a converted paper.
- Show converted Markdown content, not a screenshot preview.
- Use eased scroll with two readable pauses: first on readable body/structure, then on a structured block such as formula/reference content if available.

Voice:
> Reader 打开的不是截图，而是转换后的 Markdown。正文、公式、章节都能检索，并保留定位点；回答可以跳回原文。

### 32.0-44.0s - New Question, Real Agent Process

Visual:
- Chat input types a new task:
  `我在写 single-photon imaging 的 introduction。请从文献库里找噪声机制、方法对比和 related work 证据，并保留可核查引用。`
- Show compact agent status: read library, retrieve evidence, draft with sources.

Voice:
> 接下来提一个写作任务。agent 会读转换后的文献库，找噪声机制、方法对比和 related work 线索，并保留引用。

### 44.0-52.0s - Answer With Source Numbers

Visual:
- Generated answer appears with source chips.
- Push in to the source numbers and hold on source `#4` before clicking.

Voice:
> 答案生成时，来源编号跟着结论一起出现。每个判断都可以继续往回追。

### 52.0-63.0s - Source #4 Evidence Card

Visual:
- Macro proof zoom: full answer -> `#4` -> popover.
- Hold on claim, original evidence, relevance/support, source metadata, and add-to-basket action.

Voice:
> 点开第四个来源。结论、原文证据、相关性和论文信息在同一张卡里；哪些文章被用到、能不能引用，打开就能判断。

Capture rule:
- Reject the shot if the card repeats one generic summary in claim/evidence/why.

### 63.0-74.0s - Jump To Original Evidence

Visual:
- From the card, click open evidence.
- Reader opens at the matched passage, not the document top.
- Crop includes the highlighted block and locate status/resolution.

Voice:
> 从卡片继续打开原文，Reader 直接落到对应段落，高亮那一段证据，旁边还有命中状态。回答说到哪里，我就能核到哪里。

### 74.0-79.0s - DOI / Official Source

Visual:
- Open DOI or official paper page.
- Hold cursor/ring near Download PDF or equivalent official source action.

Voice:
> 需要原文时，DOI 回到官方论文页面。光标停在 Download PDF，来源能核到出版方。

### 79.0-92.0s - Literature Basket Becomes Context

Visual:
- Add the chosen reference.
- Open shelf and expand one item.
- Show article summary, source trail, tags, and open-source entry.
- Select several papers, click "use for next turn", and hold on the context pack above input.

Voice:
> 有用的文献收进文献篮。点开能看概要、来源轨迹和原文入口；选中几篇，就能带进下一轮上下文。它不是收藏夹，而是研究上下文。

### 92.0-101.0s - Follow-Up Writing

Visual:
- Context pack remains visible.
- The follow-up related-work answer begins with citations.

Voice:
> 下一轮写 related work 时，agent 就沿着这些已经核对过的来源继续展开。

### 101.0-107.0s - Close

Visual:
- Proven product tiles settle behind the logo.
- No abstract diagram; use the already-shown Library, Reader, citation, evidence, and basket surfaces.

On-screen text:
- `Ask. Trace. Collect. Continue.`
- `Pi-zaya - PDF library research agent`

Voice:
> Pi-zaya。让文献库能提问、能定位、能收录、能继续写。

## Recording Requirements

- Do not use the old seeded demo card as the hero evidence card unless claim, evidence, and why are distinct and grounded.
- Prefer real backend-generated citation card data for the `#4` evidence shot.
- If a fixture is used, it must include distinct `answerClaim`, `evidenceQuote`, `whyLine`, `headingPath`, and `reader_open`.
- Evidence jump must be verified visually: the highlighted original passage has to be inside the crop.
- Page-wall shots are allowed only as intro/outro/transition. All proof shots must be close-up.
- Do not include a separate Quality Center segment in this cut.
- Reader figure-caption copy should be removed from voiceover. Reader is about conversion quality, knowledge-base answering, and source location.

## Copy Tone

Use first-person demo language:
- Good: `我点开第四个来源`, `我让它只基于这几篇继续写`, `我直接回到原文核查`.
- Avoid: `系统可以`, `它能够`, `该功能用于`.

## Script Fix Already Applied

`record_take13_ad.cjs` now supports `focus_citation_index`, `focus_ref_index`, or `focus_display_num` on a block, so take24 can intentionally click the strongest citation, for example:

```json
{
  "id": "reference_cards",
  "duration": 12,
  "focus_display_num": 4
}
```

It also includes a new `evidence_jump` block that opens the selected citation card and then clicks the card's primary evidence action, so the recording can prove the jump from answer -> evidence card -> Reader original passage.

Recorder-ready draft:
- `docs/ad_assets/pi_zaya_20260609/take24_real_feature_blocks.json`
- Package with `docs/ad_assets/pi_zaya_20260609/take24_voice_overrides.json`; it explicitly mirrors the final Take24 copy, so cached manifest text or old Take13 overrides cannot replace it.
