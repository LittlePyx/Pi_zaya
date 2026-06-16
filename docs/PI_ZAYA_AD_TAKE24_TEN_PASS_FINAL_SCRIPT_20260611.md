# Pi-zaya Take24 Ten-Pass Final Ad Script

Date: 2026-06-11

This version responds to the latest review: the voiceover must stop sounding like production notes. The narrator should not say "the shot must" or "what we need to see"; those belong in the storyboard. The spoken copy now describes a real research workflow: I add PDFs, ask a new question, inspect evidence, jump to the source, collect papers, and continue writing.

## Product Evidence Used

- FastAPI + React product entry only: `run_new.ps1`, React `http://127.0.0.1:5173/`, backend `http://127.0.0.1:8000/`.
- Library page currently shows 21 converted papers, `Q100` chips, `可用于问答`, page/ref/fig/math counts, auto-suggested categories/tags, category/tag filters, and a manual metadata editor opened from each row's `分类` button.
- Reader supports real evidence focus, locate badges, exact/fallback status, highlighted source blocks, outline, evidence navigation, and source-level sessions.
- Citation popover supports answer claim, original evidence, support relation, compact metadata, DOI, source opening, and shelf actions.
- Literature basket supports summaries, quality/source trail, tags, source opening, selected batch actions, and "use as next-turn context".
- Recorder now supports `focus_display_num`/`focus_citation_index` for the #4 source and an `evidence_jump` block that opens Reader from a citation card.

## Voiceover Rule

Good voiceover:

- Sounds like a person demonstrating a workflow.
- Names the value the viewer gets, not the camera instruction.
- Uses short, concrete verbs: add, ask, open, check, collect, continue.
- Avoids "系统会", "这里要看", "镜头必须", "好的卡片应该", "展示一下".
- Does not spend a separate segment on Quality Center.
- Does not make figure captions the Reader selling point.
- Keeps "agent" explicit: the product is not only search, it reads a knowledge base and carries evidence forward.

## Ten Review Passes

### Pass 1 - Product Focus

1. Problem: Earlier scripts still felt like a generic UI tour.
   Fix: Make the story "PDF library becomes a research agent".
2. Problem: The opening page was too quiet.
   Fix: Start with rotating zoom logo, then fast page montage.
3. Problem: The first real proof came too late.
   Fix: Put conversion result and Reader before long chat output.
4. Problem: "Search" and "agent" were not visually separated.
   Fix: Ask a multi-part writing task, not a keyword query.
5. Problem: The Library shot was overused.
   Fix: Use Library for the strongest visible metadata story: automatic categories/tags, editable fields, filters, and ask-ready status.
6. Problem: Quality Center stole too much attention.
   Fix: Remove it from this ad. The Library taxonomy/status area carries the trust signal more naturally.
7. Problem: Reader copy talked about captions.
   Fix: Reframe Reader as readable, searchable, locatable converted content.
8. Problem: DOI page felt detached.
   Fix: Use it as quick external-source proof after evidence inspection.
9. Problem: Basket looked like a saved list.
   Fix: Show it becoming next-turn context.
10. Problem: Domain scope was only spoken.
    Fix: Show a short multi-page wall with domain-neutral labels.

### Pass 2 - Visual Proof

1. Problem: Page walls made proof text too small.
   Fix: Keep page walls only for overview; use macro zoom for evidence.
2. Problem: Citation chips were visible but not meaningful.
   Fix: Click #4 and hold on its card.
3. Problem: Prior evidence card looked templated.
   Fix: Seed #4 with distinct claim, evidence quote, and relevance line.
4. Problem: Evidence jump previously opened a document without proving location.
   Fix: Require Reader evidence focus, locate status, and highlighted paragraph.
5. Problem: Answer generation was too static.
   Fix: Show agent activity overlay while the answer is being built.
6. Problem: Conversion result was implied by row chips.
   Fix: Open Reader and show converted text/structure.
7. Problem: The left history sidebar distracted from product value.
   Fix: Collapse sidebar or crop it out for all hero shots.
8. Problem: DOI click did not point to the useful action.
   Fix: Hold cursor/ring near official download or source action.
9. Problem: Basket context transfer was easy to miss.
   Fix: Hold on the context pack above the input after "use for next turn".
10. Problem: Reader status could be hidden by overlay CSS.
    Fix: Stop hiding `reader-locate-status`; keep locate proof visible.

### Pass 3 - Voiceover

1. Problem: Some lines were production notes: "镜头必须".
   Fix: Move requirements to storyboard; rewrite spoken line as user benefit.
2. Problem: "这里要看" sounded instructional.
   Fix: Say what the user learns from the action.
3. Problem: "好的卡片应该" judged the UI instead of selling it.
   Fix: Say "我能直接判断它是否支撑当前结论".
4. Problem: Too many "它会/可以".
   Fix: Use first-person workflow: "我打开", "我点", "我把".
5. Problem: Some lines were too long for TTS.
   Fix: Split into shorter sentences with one idea each.
6. Problem: Reader paragraph overexplained secondary details.
   Fix: Mention conversion quality and source location only.
7. Problem: Quality Center line sounded like a feature checklist.
   Fix: Replace it with Library category/tag copy and manual-edit proof.
8. Problem: "research agent" was introduced once and then faded.
   Fix: Repeat agent concept at question, answer, and context transfer.
9. Problem: Closing was generic.
   Fix: Close with the four actual actions: Ask, Trace, Collect, Continue.
10. Problem: English subtitles could overpromise.
    Fix: Keep English short and aligned with visible proof.

### Pass 4 - Story Rhythm

1. Problem: Earlier cuts lingered on static pages.
   Fix: Use motion only during transitions, then stable proof holds.
2. Problem: Opening montage risked becoming random UI flashes.
   Fix: Order flashes by workflow: convert, ask, evidence, source, basket.
3. Problem: Chat answer appeared before the question mattered.
   Fix: Type a new multi-part task on screen.
4. Problem: Viewer needed a reason to care about #4.
   Fix: Make #4 the method-comparison source in the narration.
5. Problem: Evidence jump and DOI were too similar.
   Fix: Evidence jump proves exact passage; DOI proves external origin.
6. Problem: Basket had too many controls.
   Fix: Show add, expand, select, use-context only.
7. Problem: Export was less central than context.
   Fix: Do not spend a main segment on export.
8. Problem: The ad felt too long when every feature was described equally.
   Fix: Prioritize evidence chain; compress secondary surfaces.
9. Problem: Reader scroll could become dull.
   Fix: Use eased cinematic scroll with two readable pauses.
10. Problem: Closing page wall previously reset to generic marketing.
    Fix: Use tiles from already-proven shots.

### Pass 5 - Feature Coverage

1. Problem: "Conversion result" was under-demonstrated.
   Fix: Show Library row then Reader content.
2. Problem: "Knowledge base answer" needed a fresh query.
   Fix: Ask a new writing/research planning question.
3. Problem: "Agent process" needed visible progress.
   Fix: Use parsing/retrieval/reading/drafting overlay.
4. Problem: "Evidence card" needed original evidence.
   Fix: Use #4 with claim + quote + relevance + location.
5. Problem: "Source jump" needed exact proof.
   Fix: Land on highlighted Reader paragraph with locate badge.
6. Problem: "Reference relevance" was not obvious.
   Fix: Narrate why the paper is a comparison source.
7. Problem: "Article summary" in basket was not foregrounded.
   Fix: Expand one shelf card and hold on summary/source trail.
8. Problem: "Open original" from basket was not shown.
   Fix: Keep open-source/source-trail visible in the basket shot.
9. Problem: "Next round context" was only said.
   Fix: Show context pack above input.
10. Problem: "Not limited to one domain" needed tact.
    Fix: Put domain scope in on-screen text, not a long spoken claim.

### Pass 6 - Cinematic Style

1. Problem: Prior page wall resembled a slide deck.
   Fix: Use layered UI tiles with parallax, not flowchart boxes.
2. Problem: Too many tilted outlines weakened seriousness.
   Fix: Use clean crops, shadow, and depth.
3. Problem: Effects sometimes fought readability.
   Fix: Freeze all proof text during narration.
4. Problem: Logo open was too plain.
   Fix: Rotating scale-in logo with a snap-through transition.
5. Problem: Evidence zoom lacked intent.
   Fix: Full answer -> source number -> card -> original evidence.
6. Problem: Reader jump needed a cinematic but clear transition.
   Fix: Use a quick snap/blur, then immediate stable close-up.
7. Problem: Context transfer was invisible.
   Fix: Add a small lift/glow on the context pack.
8. Problem: Domain montage risked looking like PPT.
   Fix: Use product pages as moving tiles with text/icon overlays.
9. Problem: UI scale varied too much.
   Fix: Use 2x PNG capture and crop proof surfaces.
10. Problem: Closing needed polish but not abstraction.
    Fix: Return to the same proven screens behind the logo.

### Pass 7 - Macro Copy

1. Problem: The old headline "可问可查" was too common.
   Fix: Pair it with "Collect" and "Continue" for workflow distinction.
2. Problem: "不限领域" sounded like a claim without proof.
   Fix: Use "single-photon is only today's demo" as restrained copy.
3. Problem: Too many nouns.
   Fix: Use action words: import, ask, trace, collect, continue.
4. Problem: "文献库" sounded passive.
   Fix: Say it becomes a research agent.
5. Problem: Reader value was vague.
   Fix: "转换后的内容能被阅读、检索、定位、引用."
6. Problem: Evidence value was vague.
   Fix: "结论不是孤立文字，它带着可回到原文的位置."
7. Problem: Basket value was vague.
   Fix: "收录不是收藏，是下一轮上下文."
8. Problem: Quality value was too detailed.
   Fix: "先知道哪些材料能放心问."
9. Problem: DOI value was overexplained.
   Fix: "来源能回到论文官方页面."
10. Problem: Ending was too soft.
    Fix: End with "Ask. Trace. Collect. Continue."

### Pass 8 - Micro Copy

1. Problem: "我点开第四个来源" alone was abrupt.
   Fix: "第四个来源，是方法对比的关键."
2. Problem: "模板解释" was negative and awkward in voiceover.
   Fix: "我能直接判断它是否支撑当前结论."
3. Problem: "回答里的判断和原文证据怎么对应" was clunky.
   Fix: "结论、原文、相关性放在同一张卡里."
4. Problem: "再点定位" was too mechanical.
   Fix: "从这张卡继续打开原文."
5. Problem: "匹配得有多准" was abstract.
   Fix: "Reader 标出命中状态，并高亮那一段."
6. Problem: "生成内容的过程" needed natural phrasing.
   Fix: "我先让它拆任务，再看它读哪些材料."
7. Problem: "质量中心" sounded administrative.
   Fix: Speak about automatic category/tag recommendations and editable metadata instead.
8. Problem: "文献篮" needed workflow language.
   Fix: "把有用的来源放进下一轮."
9. Problem: "官方页" could be too dry.
   Fix: "需要原文时，可以一路回到论文页面."
10. Problem: Closing line had too many adjectives.
    Fix: Keep one sentence, then four action words.

### Pass 9 - Recording Reality

1. Problem: Unsupported storyboard IDs would not record.
   Fix: Use recorder-supported IDs only.
2. Problem: #4 focus could accidentally open #1.
   Fix: Use `focus_display_num: 4`.
3. Problem: Evidence locate could land on wrong block.
   Fix: Use corrected NatCommun block `blk_b787f761c5d5_00072` for clean Reader proof.
4. Problem: Smart quote mismatch broke exact phrase locate.
   Fix: Use the original `array’s` text in Reader session snippet.
5. Problem: Reader locate status was hidden.
   Fix: Remove CSS hiding `reader-locate-status`.
6. Problem: Opening montage still said "图片证据".
   Fix: Rename stage to "Reader 转换效果".
7. Problem: Library quality could show one recommendation and confuse viewers.
   Fix: Narrate status as "which sources are ready", not "everything perfect".
8. Problem: Browser history sidebar polluted the frame.
   Fix: Continue using `collapseLeftSidebar`.
9. Problem: Right dock could crowd citation cards.
   Fix: Collapse it during evidence card shots.
10. Problem: DOI publisher pages are unpredictable.
    Fix: Keep DOI segment short and accept fallback official Nature page.

### Pass 10 - Final Cut Decisions

1. Problem: Too many features dilute the agent message.
   Fix: Build around one chain: convert -> ask -> cite -> locate -> collect -> continue.
2. Problem: The ad needs both speed and proof.
   Fix: Fast montage at the start, slow holds at evidence moments.
3. Problem: Reader conversion still needs a strong line.
   Fix: "Reader 不是截图预览，是转换后的可检索文本入口."
4. Problem: Agent claim needs visible process.
   Fix: Type a new question and show progress before final answer.
5. Problem: Source card needs a natural line.
   Fix: "第四个来源，是方法对比的关键."
6. Problem: Locate line needs natural proof.
   Fix: "Reader 直接落到对应段落，高亮证据，旁边还有命中状态."
7. Problem: Basket line needs to avoid "收藏夹".
   Fix: "收录不是结束，是把来源带进下一轮."
8. Problem: Quality Center should not get another deep segment.
   Fix: Do not include it in Take24; show classification, tags, conversion status, page/reference counts, and ask-ready state instead.
9. Problem: Domain scope should not sound like empty marketing.
   Fix: Put it in an overlay: "Single-photon is today's demo. Replace the PDFs, replace the field."
10. Problem: Final script must be recorder-ready.
    Fix: Create `take24_10pass_final_blocks.json` and mirror it into `take24_real_feature_blocks.json`.

## Final Storyboard And Voiceover

Target length: about 105-108 seconds.

### 0.0-9.0s - Logo And Whole-System Montage

Visual:
- Rotating zoom Pi-zaya logo.
- Snap through logo into fast product montage: Library converted rows, Reader conversion, fresh question, source #4 card, Reader highlight, basket, DOI page.
- Page wall is allowed here only as an overview, with the evidence card/Reader highlight as the largest tile.

On-screen text:
- `PDF library -> research agent`
- `Ask / Trace / Collect / Continue`

Voice:
> Pi-zaya 是 PDF 文献库的 research agent。它把论文转成带定位点的 Markdown，让回答能追到原文证据。

### 9.0-21.0s - Library Categories And Tags

Visual:
- Library page focused on `文献分类与标签`.
- Crop on rows that show `已转换`, `Q100`, `可用于问答`, category/tags, pages/refs/fig/math counts.
- Click a row category pill and a tag pill to show the list filtering.
- Click the row `分类` button to open the metadata drawer.
- Hold on manual fields: category, reading status, tags, note; also show system suggestions and accept/ignore actions.
- Briefly switch to `分类` and `标签` browse views.

Voice:
> 上传论文后，Library 会自动推荐分类和标签；需要调整时，点“分类”就能手动修改。列表里还保留转换状态、页数、参考文献和可问答状态。

### 21.0-32.0s - Reader Conversion Result

Visual:
- Open Reader from a converted paper.
- Slow cinematic scroll over converted title, abstract/body, equations/structured content.
- No caption-focused copy.

Voice:
> Reader 打开的不是截图，而是转换后的 Markdown。正文、公式、章节都能检索，并保留定位点；回答可以跳回原文。

### 32.0-44.0s - Ask A New Agent Task

Visual:
- Chat input types a fresh question:
  `我在写 single-photon imaging 的 introduction。请从文献库里找噪声机制、方法对比和 related work 证据，并保留可核查引用。`
- Agent overlay: parsing task -> retrieving papers -> reading evidence -> drafting.

Voice:
> 接下来提一个写作任务。agent 会读转换后的文献库，找噪声机制、方法对比和 related work 线索，并保留引用。

### 44.0-52.0s - Answer With Source Numbers

Visual:
- Generated answer appears with source chips.
- Hold on source numbers and short answer paragraphs.

Voice:
> 答案生成时，来源编号跟着结论一起出现。每个判断都可以继续往回追。

### 52.0-63.0s - Source #4 Evidence Card

Visual:
- Macro zoom: answer -> #4 -> citation popover.
- Hold on claim, original evidence, support/relevance, source metadata.

Voice:
> 点开第四个来源。结论、原文证据、相关性和论文信息在同一张卡里；哪些文章被用到、能不能引用，打开就能判断。

Storyboard requirement:
- Do not speak this line; use it only for review: reject the shot if claim/evidence/relevance repeat the same template sentence.

### 63.0-74.0s - Jump To Original Evidence

Visual:
- Click source/open evidence.
- Reader opens to the matched paragraph.
- Visible: evidence focus, locate resolution/status, highlighted paragraph.

Voice:
> 从卡片继续打开原文，Reader 直接落到对应段落，高亮那一段证据，旁边还有命中状态。回答说到哪里，我就能核到哪里。

Storyboard requirement:
- The camera must land on the highlighted original text, not only on the document top.

### 74.0-79.0s - DOI / Official Source

Visual:
- DOI or official source page.
- Cursor/ring near official paper/download action.

Voice:
> 需要原文时，DOI 回到官方论文页面。光标停在 Download PDF，来源能核到出版方。

### 79.0-92.0s - Literature Basket

Visual:
- Add useful source to basket.
- Open basket, expand one item.
- Show title, summary, quality/source trail, tags, open-source action.

Voice:
> 有用的文献收进文献篮。点开能看概要、来源轨迹和原文入口；选中几篇，就能带进下一轮上下文。它不是收藏夹，而是研究上下文。

### 92.0-101.0s - Use As Next-Turn Context

Visual:
- Select several basket items.
- Click "use for next turn".
- Hold on context pack above input.

Voice:
> 选中几篇，放进下一轮上下文。下一次让 agent 写 related work，它就围绕这些已经核对过的来源继续展开。

### 101.0-107.0s - Closing

Visual:
- Proven tiles settle behind logo.
- Logo locks in.
- Optional small overlay: `Single-photon imaging is today's demo. Replace the PDFs, replace the field.`

On-screen text:
- `Ask. Trace. Collect. Continue.`

Voice:
> Pi-zaya。让文献库能提问、能定位、能收录、能继续写。

## Final Segment Blocks

Use `docs/ad_assets/pi_zaya_20260609/take24_10pass_final_blocks.json`.

The same voiceover has also been mirrored into `docs/ad_assets/pi_zaya_20260609/take24_real_feature_blocks.json` so the old unsuitable lines are not the default Take24 draft anymore.

Use `docs/ad_assets/pi_zaya_20260609/take24_voice_overrides.json` when packaging this take. It explicitly mirrors the final Take24 voiceover so cached manifest text or old Take13 overrides cannot replace the new copy.

## Recording Notes

- Use `focus_display_num: 4` for opening, answer, evidence card, evidence jump, and DOI shots.
- Keep the left sidebar collapsed for Chat shots.
- Keep Reader locate status visible.
- For the evidence jump, the final hold must include `reader-evidence-focus`, `reader-locate-resolution` or `reader-locate-status`, and highlighted Reader content.
- Use effects between proof shots, not over proof text.
- No flowchart-style arrows. Use multi-page layouts, text/icon overlays, parallax, macro zoom, and stable close-ups.
- Recorder-level product strip and spotlight helpers are now used to orient attention during Library, Reader, citation, evidence, DOI, basket, and writing shots.
- Product-ad effect spec: `docs/ad_assets/pi_zaya_20260609/take24_product_ad_effects.json`.
- Voice timing is controlled by `produce_take13_final.py`: review `take13_voice_timing_report.json`, and use `--strict-timing` to fail builds where a block needs obvious tempo forcing.
