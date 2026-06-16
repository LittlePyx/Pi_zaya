# Pi-zaya Take29 Redesign Plan

Date: 2026-06-12

## Current Diagnosis

Take28 is feature-complete, but the presentation is wrong in several places.

1. The magnifier/callout style is too noisy. Too many rectangles, lines, zoom boxes, and English labels make the video look like an internal review sheet instead of a polished product ad.
2. The pacing is still too fast for first-time viewers. The narration says a feature, then the video moves on before the user has understood the UI.
3. The voice is still recognizably synthetic. Qwen-TTS is cleaner than Edge, but long declarative product copy still sounds like TTS.
4. Reader uses the wrong visual emphasis. Formula-heavy and numeric-table-heavy frames make the Reader segment look like math rendering, not like "Markdown conversion + evidence workflow".
5. Reader missing features: highlight, follow-up questioning, and reference-card opening from Reader references need to be explained.
6. Reference cards need a better example. They should show journal, year, impact factor, one-line summary, DOI/source, and why the paper matters.
7. English overlay labels should be removed or translated. Chinese customers should not need to parse English UI captions.
8. The cover should not be a collage. It needs one strong hook and one strong product proof.

## Take29 Direction

Build Take29 as a slower, cleaner "product proof" version, around 125-140 seconds.

Do not add more random zoom boxes. Use three visual rules:

1. Full page first, then a slow camera move.
2. Only one highlighted product area at a time.
3. Use native UI state, cursor, and subtitle narration instead of extra explanatory labels whenever possible.

## Proposed Storyboard

### 1. Opening, 8-10 s

- Rotating Pi-zaya logo.
- Clean Chinese title: `PDF 文献库 research agent`.
- Quick page montage, but no floating English labels.
- Voice: "Pi-zaya 把一批 PDF 变成能提问、能定位证据、能继续写作的 research agent。"

### 2. Library, 18-22 s

- Show upload/conversion list.
- Hold long enough on: converted, pages, refs, fig, math, ask-ready.
- Show automatic category/tag recommendations.
- Open manual edit drawer.
- Show editing category/tag/reading status.
- Click a tag/category and show list narrowing.

### 3. Reader Conversion, 18-22 s

- Use a text-rich Markdown page, not the numeric table / formula-heavy frame.
- Show that PDF became Markdown text.
- Show search/highlight.
- Show that highlighted content remains readable as original evidence.
- Mention: "这不是截图预览，是转换后的 Markdown，可以检索，可以被 agent 引用。"

### 4. Reader References, 14-18 s

- Scroll to references or click a reference/citation inside Reader.
- Open a reference card directly from Reader.
- The card should show: title, journal, year, IF/JCR, DOI, one-line summary.
- Voice: "打开 Reader 里的参考文献，不只是看到编号，而是能看到这篇文献具体是什么。"

### 5. Ask Agent, 16-20 s

- Ask a new writing task.
- Let generation breathe; show output appearing, not just final answer.
- Keep source numbers visible beside claims.
- Voice should be more conversational: "我现在让它帮我找 related work 证据。它会读库里的原文，不只做关键词匹配。"

### 6. Source Card + Evidence Jump, 22-26 s

- Click source #4.
- Show the card: claim, evidence, relevance, paper identity.
- Click locate.
- Camera lands in Reader on highlighted original paragraph.
- Hold 2-3 seconds on highlight.
- Voice: "这里最关键：这句话为什么成立，原文证据在哪里，都能回到同一个 Reader 里核。"

### 7. Follow-Up Question, 12-16 s

- Use the highlighted evidence or selected papers as context.
- Ask a follow-up: "基于这几篇，写一段 related work 的对比。"
- Show that the next answer keeps using the same evidence path.

### 8. DOI + Basket, 14-18 s

- DOI opens publisher page.
- Literature basket shows selected papers.
- Open one basket item or show summary/source trail.
- Use selected papers for next round.

### 9. Closing, 5-7 s

- Clean logo.
- Chinese line: `可问、可定位、可收录、可继续写。`

## Narration Rules

Rewrite all narration as live-demo speech, not product brochure speech.

- Use `我点开`, `这里能看到`, `我继续追问`, `这几篇进入下一轮`.
- Avoid long abstract nouns like `工作流化`, `能力链路`, `质量背书`.
- Do not say "系统可以" too often.
- Keep each line under about 18-24 Chinese characters before a natural pause.
- Add silence after important UI moments; do not fill every second with voice.

## Voice Plan

Qwen-TTS alone may not reach fully human quality. Better options:

1. Best: record a real human voice track.
2. Next best: use Alibaba voice cloning/custom voice with a clean 20-40 second sample.
3. If still using Qwen-TTS: synthesize by phrase groups, not whole paragraphs; insert natural pauses; keep no `atempo` over 1.03.
4. Try at least three voices and choose by ear, not by model name.

## Reader Visual Requirements

Avoid:

- Pure numeric-table frames.
- Formula-only frames.
- Random boxed zooms that hide the page context.

Use:

- Text-rich Markdown sections.
- Visible search/highlight state.
- A Reader reference/citation card with journal/year/IF/summary.
- A highlighted paragraph that later matches the answer's source.

## Cover Direction

Use v2 cover direction, not the first collage:

- One big hook: `答案不是只给你看，还能直达原文证据`.
- One main Reader evidence screenshot.
- Two product proof elements: source card + literature basket.
- No dense screenshot pile, no random magnifier boxes.

Generated v2 covers:

- `docs/ad_assets/pi_zaya_20260609/take29_cover_20260612/pi_zaya_cover_take29_v2_1920x1080.jpg`
- `docs/ad_assets/pi_zaya_20260609/take29_cover_20260612/pi_zaya_cover_take29_v2_1080x1920.jpg`

## Implementation Plan

1. Remove most compositor callout rectangles and English overlay labels.
2. Re-record or re-cut Reader with a better text-rich Markdown page.
3. Add Reader reference-card demonstration.
4. Add follow-up question segment.
5. Rewrite narration for slower live-demo delivery.
6. Test 2-3 TTS voices or switch to cloned/real voice.
7. Build Take29 at 125-140 seconds.
8. Generate horizontal and vertical covers from the v2 design.
